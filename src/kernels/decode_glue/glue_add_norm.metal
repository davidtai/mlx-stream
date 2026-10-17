// glue_add_norm: a decode row's sublayer boundary in one dispatch: the sublayer's output y [1, D] normed (w1), the
// depth-scaled add with the residual r (graph.zig DepthAdd: bf16(f32(normed) * a) + bf16(f32(r) * b)) -> res [1, D],
// then res normed (w2) -> out [1, D]: the next sublayer's input. Each norm is MLX's: rms_looped where D > 4,096 (1,024
// threads, thread t holding the 4 columns at 4 t of every 4,096-column block), else rms_single_row (32 x ceil(D / 128)
// threads, one block); `acc += x * x` in order, simd_sum, the 32-slot fold over the simdgroups, precise::rsqrt(acc / D +
// eps), `w * T(x * inv)`. A thread's columns of res are the ones its second norm reads: no exchange. w1, w2 [D] bf16,
// consts [8] f32 (0: a, 1: b, 2: eps). One threadgroup of TPG threads.
constexpr int NB = (D + 4 * TPG - 1) / (4 * TPG);
const int lid = int(thread_position_in_threadgroup.x);
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
threadgroup float local_sums[32];
threadgroup float local_inv[1];
float acc = 0;
for (int b = 0; b < NB; ++b) {
  const int c0 = b * 4 * TPG + 4 * lid;
  if (c0 + 4 <= D) {
    for (int i = 0; i < 4; ++i) {
      float xi = y[c0 + i];
      acc += xi * xi;
    }
  }
}
acc = simd_sum(acc);
if (sg == 0) {
  local_sums[lane] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
  local_sums[sg] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  acc = simd_sum(local_sums[lane]);
  if (lane == 0) {
    local_inv[0] = metal::precise::rsqrt(acc / D + consts[2]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv1 = local_inv[0];
bfloat16_t rv[NB][4];
float acc2 = 0;
for (int b = 0; b < NB; ++b) {
  const int c0 = b * 4 * TPG + 4 * lid;
  if (c0 + 4 <= D) {
    for (int i = 0; i < 4; ++i) {
      const bfloat16_t n1 = w1[c0 + i] * static_cast<bfloat16_t>(y[c0 + i] * inv1);
      const bfloat16_t p1 = static_cast<bfloat16_t>(static_cast<float>(n1) * consts[0]);
      const bfloat16_t p2 = static_cast<bfloat16_t>(static_cast<float>(r[c0 + i]) * consts[1]);
      const bfloat16_t s = p1 + p2;
      res[c0 + i] = s;
      rv[b][i] = s;
      float xi = s;
      acc2 += xi * xi;
    }
  }
}
// The second norm's fold reuses the slots: every thread has read the first's inverse.
threadgroup_barrier(mem_flags::mem_threadgroup);
acc2 = simd_sum(acc2);
if (sg == 0) {
  local_sums[lane] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
  local_sums[sg] = acc2;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  acc2 = simd_sum(local_sums[lane]);
  if (lane == 0) {
    local_inv[0] = metal::precise::rsqrt(acc2 / D + consts[2]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv2 = local_inv[0];
for (int b = 0; b < NB; ++b) {
  const int c0 = b * 4 * TPG + 4 * lid;
  if (c0 + 4 <= D) {
    for (int i = 0; i < 4; ++i) {
      out[c0 + i] = w2[c0 + i] * static_cast<bfloat16_t>(rv[b][i] * inv2);
    }
  }
}
