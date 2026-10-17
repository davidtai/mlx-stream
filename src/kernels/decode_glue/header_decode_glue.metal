// The decode glue texts over a noaux_tc router (beam glue.zig): every expression is the stock op chain's at MLX
// 64ea011cb (its -fno-fast-math metallib; the JIT's safe math mode differs only in the default fp32 functions, so the
// transcendentals are named metal::precise). MiMo's K2 glue texts (src/arch/mimo_v2/kernels/mimo_glue_router1.metal,
// mimo_glue_top8.metal) with the geometry as template parameters and DeepSeek-V3's group stage.
#include <metal_simdgroup>

// MLX's Sigmoid on f32 (unary_ops.h).
METAL_FUNC float glue_sigmoid(float x) {
  const float y = 1 / (1 + metal::precise::exp(metal::precise::abs(x)));
  return (x < 0) ? y : 1 - y;
}

// MLX's sort order (sort.h LessThan): NaN last.
METAL_FUNC bool glue_lt(float a, float b) {
  const bool an = metal::isnan(a);
  const bool bn = metal::isnan(b);
  if (an | bn) {
    return (!an) & bn;
  }
  return a < b;
}

// Key `v` (index `e`)'s place in MLX's block merge sort of the `n` keys `ks` (stable: its odd-even thread sort and its
// merges take the left key on a tie): every key before it in the order and every equal key at a lower index.
METAL_FUNC int glue_rank(const threadgroup float* ks, int n, float v, int e) {
  int rank = 0;
  for (int j = 0; j < n; ++j) {
    const float u = ks[j];
    rank += (glue_lt(u, v) || (j < e && !glue_lt(v, u))) ? 1 : 0;
  }
  return rank;
}
