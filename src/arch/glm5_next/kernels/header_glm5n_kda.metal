// GLM-5.3-Flash's KDA decode step (glm5_next kda_decode.zig, `decode_kda: fused`): the host's oMLX KDA decode kernel
// (mlx-serve src/kernels/kda_decode.metal at glm5next/host a8f6c6b8, ported from oMLX (jundot/omlx)
// omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py, oMLX 0.7.0, Apache-2.0; mlx-serve's NOTICE) with its helpers
// (mlx-serve src/kernels/glm5_qmv_header.metal `glm_sigmoid`, `glm_sigmoid_precise`), BF16-gate variant.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline T glm_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// MLX's Sigmoid as its precompiled kernels evaluate it: the release metallib is built with -fno-fast-math, so
// metal::exp is the precise exp there, while runtime-compiled kernels get the default one.
template <typename T>
inline T glm_sigmoid_precise(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
