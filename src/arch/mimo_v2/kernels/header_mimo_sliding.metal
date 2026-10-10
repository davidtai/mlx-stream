// K3 sliding layer (mimo/k2-glue). The q/k/v projections at M <= 8 rows, each output row in the exact accumulation of
// the MLX kernel the kernels route runs at that M (MLX 64ea011cb gemv.h, matmul.cpp): M = 1 `gemv` (GEMVKernel, SN = 32,
// TN = 4: lane l takes columns 128 i + 4 l + t, i = 0..31, `result += inter * v_coeff`, halving shuffles 16 .. 1);
// M = 2..5 `gemv_wide` k_lanes 32 and M = 6..8 `gemv_wide` k_lanes 16 (GemvWide: vec4 chunks base + i k_lanes + lane,
// `acc += dot(wf[i], x)` over 8 chunks, `result += acc`, halving shuffles k_lanes / 2 .. 1). The stock rope and value
// scale are the trunk's op chain element for element (module.partialRope, module.scaleValues).
#include <metal_simdgroup>

constant constexpr int SL_D = 4096;
constant constexpr int SL_NV4 = SL_D / 4;

// The weight fetch (4 consecutive BF16 weights of row `w` from column `col`, as F32): the one function a resident tier
// replaces (the fp8 tier's hook: dequantize its block here, values identical to what its stock GEMV multiplies).
template <typename T>
METAL_FUNC float4 sliding_w4(const device T* w, int col) {
  return float4(*((const device vec<T, 4>*)(w + col)));
}

template <typename T>
METAL_FUNC float4 sliding_x4(const device T* x, int col) {
  return float4(*((const device vec<T, 4>*)(x + col)));
}
