// The sampler texts' header (beam glue.zig `sampleRow`): the pieces of MLX 64ea011cb's kernels that a sampled row's
// token depends on, each computed as MLX computes it, so a text that spreads them over many threadgroups keeps every
// value. A row of V f32 values is read in MLX's blocks of 4,096 (1,024 threads x N_READS 4: softmax_looped's and
// contiguous_scan's one threadgroup): thread t of a block holds its values [4 t, 4 t + 4), simdgroup s its threads
// [32 s, 32 s + 32).
//   - softmax_looped's fold (softmax.h, its softmax_exp is fast::exp): each thread folds its 4 values of every block in
//     block order (the running max from -FLT_MAX, the normalizer rescaled by fast::exp(prevmax - max), then
//     fast::exp(v - max) added in order; past the row the values are -inf), then simd_max, the rescale and simd_sum in
//     its simdgroup; then over the 32 simdgroups the max, each normalizer rescaled to it, simd_sum, and 1 / the sum. A
//     value is fast::exp(v - max) * that.
//   - contiguous_scan's inclusive sum (scan.h, CumSum): in a block each thread's running sum of its 4 values, the
//     exclusive prefix of the threads' totals in the simdgroup (simd_prefix_exclusive_sum), the exclusive prefix of the
//     simdgroups' totals (simdgroup 0), then each output ((v + prefix) + simdgroup prefix) + thread prefix, where prefix
//     is the previous block's last output; past the row the values are 0.
#include <metal_simdgroup>

constant constexpr int GLUE_SM_BLOCK = 4096;

// The row a stage reads at j < V: MODE 0 the sorted negated row's negation (the tempered logits, descending: the
// filter's softmax input); MODE 1 the tempered logit where it is at or above the kept set's least (`thr`), else -inf (the
// draw's softmax input).
METAL_FUNC float glue_sm_value(int mode, int j, const device float* lt, const device float* sorted, float thr) {
  if (mode == 0) {
    return -sorted[j];
  }
  const float v = lt[j];
  return (v >= thr) ? v : -metal::numeric_limits<float>::infinity();
}

// softmax_looped's fold of thread t over the row of V values, then its simdgroup's simd_max, rescale and simd_sum: the
// simdgroup's (max, normalizer). The loads of GLUE_SM_BATCH blocks go out before their fold (the fold's order and
// arithmetic are MLX's: only the loads move ahead, so a thread waits on memory once a batch, not once a block).
constant constexpr int GLUE_SM_BATCH = 8;

METAL_FUNC float2 glue_sm_fold(int mode, int V, uint t, const device float* lt, const device float* sorted, float thr) {
  const int nb = (V + GLUE_SM_BLOCK - 1) / GLUE_SM_BLOCK;
  float prevmax;
  float maxval = -metal::numeric_limits<float>::max();
  float normalizer = 0;
  for (int r0 = 0; r0 < nb; r0 += GLUE_SM_BATCH) {
    float vals[GLUE_SM_BATCH][4];
    for (int b = 0; b < GLUE_SM_BATCH; b++) {
      const int offset = (r0 + b) * GLUE_SM_BLOCK + int(t) * 4;
      for (int i = 0; i < 4; i++) {
        const int j = offset + i;
        vals[b][i] = (j < V) ? glue_sm_value(mode, j, lt, sorted, thr) : -metal::numeric_limits<float>::infinity();
      }
    }
    for (int b = 0; b < GLUE_SM_BATCH && r0 + b < nb; b++) {
      prevmax = maxval;
      for (int i = 0; i < 4; i++) {
        maxval = (maxval < vals[b][i]) ? vals[b][i] : maxval;
      }
      normalizer *= metal::fast::exp(prevmax - maxval);
      for (int i = 0; i < 4; i++) {
        normalizer += metal::fast::exp(vals[b][i] - maxval);
      }
    }
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= metal::fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);
  return float2(maxval, normalizer);
}

// softmax_looped's last stage over the 32 simdgroups' (max, normalizer) in `part` (lane s reads simdgroup s's): the
// row's max and 1 / its normalizer, in every lane of the calling simdgroup.
METAL_FUNC float2 glue_sm_stat(const device float* part, uint lane) {
  const float mx = part[2 * lane];
  const float maxval = simd_max(mx);
  const float normalizer = simd_sum(part[2 * lane + 1] * metal::fast::exp(mx - maxval));
  return float2(maxval, 1 / normalizer);
}

// The softmax's value at j (MLX's output: fast::exp(v - max) * (1 / normalizer)); 0 past the row (the scan's init).
METAL_FUNC float glue_sm_prob(int mode, int V, int j, const device float* lt, const device float* sorted, float thr, float2 stat) {
  return (j < V) ? metal::fast::exp(glue_sm_value(mode, j, lt, sorted, thr) - stat.x) * stat.y : 0.0f;
}

// contiguous_scan's pieces of block `r` for thread `lid` (1,024 threads): its running sums `v` of its 4 values, its
// thread prefix (returned) and its simdgroup's prefix (`sgp`), the block's prefix not yet added. `sums` is the
// threadgroup's 32 floats.
METAL_FUNC float glue_scan_block(thread float* v, thread float& sgp, threadgroup float* sums, uint lane, uint sg) {
  for (int i = 1; i < 4; i++) {
    v[i] = v[i] + v[i - 1];
  }
  const float prev_thread = simd_prefix_exclusive_sum(v[3]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 31) {
    sums[sg] = prev_thread + v[3];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    sums[lane] = simd_prefix_exclusive_sum(sums[lane]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sgp = sums[sg];
  return prev_thread;
}
