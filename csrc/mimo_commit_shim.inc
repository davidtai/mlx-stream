// MiMo K6 (`decode_commits: "layer"`, src/mimo_v2/decode_barrier.zig): the Metal device's two command-buffer commit
// limits, swapped for the duration of one decode-lane call's commits and restored before the call returns.
//
// MLX cuts a command buffer after any primitive once the open buffer holds more than `max_ops_per_buffer_` dispatches
// or more than `max_mb_per_buffer_` MB of distinct inputs (`CommandEncoder::needs_commit`, device.cpp). Both are private
// members of `metal::Device`, set once from the architecture and MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER (a
// process-wide setting, which would also move every prompt-pass cut). The decode lane raises them only around its own
// async_evals, so each of those evals commits exactly once (its epilogue's finalize), and every other eval of the
// process keeps MLX's cuts. Inference thread only, like every MLX call: `needs_commit` reads them on the encoding
// thread, and nothing else does (MLX 0.32.3 @ 64ea011cb).
//
// The members are reached through an explicit instantiation's member pointers ([temp.explicit]: access checking does
// not apply to the names in an explicit instantiation), compiled from the staged headers the rest of this file uses.
// `mimo_cb_selfcheck` proves, once at construction on the GPU stream, that the member written here is the one libmlx's
// compiled `needs_commit` reads (an empty buffer commits iff the ops limit is negative); a mismatch refuses the route.
// Included by mlx_alloc_shim.cpp (the host compiles a fixed list of csrc files: no host change).
#include <climits>
#include <tuple>

#include "mlx/backend/metal/device.h"
#include "mlx/c/private/stream.h"

namespace mimo_commit {

template <typename Tag, typename Tag::type M>
struct Expose {
  friend typename Tag::type member(Tag) { return M; }
};
struct MaxOps {
  using type = int mx::metal::Device::*;
  friend type member(MaxOps);
};
struct MaxMb {
  using type = int mx::metal::Device::*;
  friend type member(MaxMb);
};
template struct Expose<MaxOps, &mx::metal::Device::max_ops_per_buffer_>;
template struct Expose<MaxMb, &mx::metal::Device::max_mb_per_buffer_>;

thread_local std::string g_error;

}  // namespace mimo_commit

extern "C" {

const char* mimo_cb_last_error(void) {
  return mimo_commit::g_error.c_str();
}

// The device's limits become (ops, mb); the previous ones are written to *prev_ops / *prev_mb. 0, or -1.
int mimo_cb_swap_limits(int ops, int mb, int* prev_ops, int* prev_mb) {
  try {
    auto& d = mx::metal::device(mx::Device::gpu);
    int& o = d.*member(mimo_commit::MaxOps{});
    int& m = d.*member(mimo_commit::MaxMb{});
    *prev_ops = o;
    *prev_mb = m;
    o = ops;
    m = mb;
    return 0;
  } catch (const std::exception& e) {
    mimo_commit::g_error = e.what();
    return -1;
  } catch (...) {
    mimo_commit::g_error = "mimo_cb_swap_limits failed";
    return -1;
  }
}

// The limits through MLX's own public getter (`Device::get_max_ops_mb_per_buffer`). 0, or -1.
int mimo_cb_limits(int* ops, int* mb) {
  try {
    auto [o, m] = mx::metal::device(mx::Device::gpu).get_max_ops_mb_per_buffer();
    *ops = o;
    *mb = m;
    return 0;
  } catch (const std::exception& e) {
    mimo_commit::g_error = e.what();
    return -1;
  } catch (...) {
    mimo_commit::g_error = "mimo_cb_limits failed";
    return -1;
  }
}

// GPU stream only (construction, on the inference thread, after the trunk's first evals made the stream's encoder):
// asks libmlx's compiled `needs_commit` with the limits written through the member pointers: (INT_MAX, INT_MAX) never
// commits and (-1, INT_MAX) always does, whatever the open buffer holds (its dispatch count is >= 0 and its input MB
// far below INT_MAX), so no synchronize and no GPU work is needed; the limits are restored either way. 0 = proven,
// 1 = a CPU stream (nothing to prove: CPU evals use no command buffer), -2 = the member written is not the one libmlx
// reads (a header / libmlx mismatch), -1 = an MLX error (e.g. no encoder for the stream on this thread).
int mimo_cb_selfcheck(mlx_stream s) {
  try {
    const mx::Stream& st = mlx_stream_get_(s);
    if (st.device.type != mx::Device::DeviceType::gpu) return 1;
    auto& d = mx::metal::device(st.device);
    auto& enc = mx::metal::get_command_encoder(st);
    int& o = d.*member(mimo_commit::MaxOps{});
    int& m = d.*member(mimo_commit::MaxMb{});
    const int o0 = o;
    const int m0 = m;
    auto [go, gm] = d.get_max_ops_mb_per_buffer();
    if (go != o0 || gm != m0) {
      mimo_commit::g_error = "the public getter disagrees with the member pointers";
      return -2;
    }
    o = INT_MAX;
    m = INT_MAX;
    const bool raised = enc.needs_commit();
    o = -1;
    const bool negative = enc.needs_commit();
    o = o0;
    m = m0;
    if (raised || !negative) {
      mimo_commit::g_error = "libmlx's needs_commit does not read the member written";
      return -2;
    }
    return 0;
  } catch (const std::exception& e) {
    mimo_commit::g_error = e.what();
    return -1;
  } catch (...) {
    mimo_commit::g_error = "mimo_cb_selfcheck failed";
    return -1;
  }
}

}  // extern "C"
