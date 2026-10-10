// K2 decode glue (mimo/k2-glue). Every expression is the one the stock op runs (MLX 64ea011cb, -fno-fast-math metallib;
// the JIT's safe math mode differs only in the default fp32 functions, so transcendentals are named metal::precise).
// MLX's rms_single_row at D = 4096 (1024 threads x RMS_N_READS = 4: `acc += x * x`, simd_sum, a 32-slot fold, simd_sum,
// precise::rsqrt(acc / D + eps), `T(x * inv)`) is played by one 256-thread threadgroup in four rounds: in round r
// simdgroup s is MLX's simdgroup 8r + s and lane l MLX's thread 32 (8r + s) + l, so every simd_sum reduces MLX's 32
// values in MLX's lanes. A lane's 16 elements c = 1024 r + 4 (32 s + l) + i are also MLX gemv's v_coeff columns of its
// blockN = 1024 iteration r (gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4): the router GEMV reads them from registers.
#include <metal_simdgroup>

constant constexpr int GLUE_D = 4096;
constant constexpr float GLUE_EPS = 1e-6f;

// The row's 16 lane elements (ADD: MLX's binary Add of the two bf16 rows) and its inverse RMS, the same value in every
// simdgroup (each folds the 32 partials itself).
template <bool ADD, typename T>
METAL_FUNC float glue_row_inv(const device T* a, const device T* b, thread T (&hrow)[16], uint sg, uint lane, threadgroup float* sums) {
  for (int r = 0; r < 4; ++r) {
    const int c0 = 1024 * r + 4 * (32 * int(sg) + int(lane));
    float acc = 0;
    for (int i = 0; i < 4; ++i) {
      T h;
      if constexpr (ADD) {
        h = a[c0 + i] + b[c0 + i];
      } else {
        h = a[c0 + i];
      }
      hrow[4 * r + i] = h;
      const float x = h;
      acc += x * x;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      sums[8 * r + int(sg)] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = simd_sum(sums[lane]);
  return metal::precise::rsqrt(total / GLUE_D + GLUE_EPS);
}

// rms_single_row's `T(x * inv)` (its unit weight's product is exact), then the learned weight: MLX's binary Multiply.
template <typename T>
METAL_FUNC T glue_normed(T h, float inv, T w) {
  const float x = h;
  return static_cast<T>(x * inv) * w;
}
