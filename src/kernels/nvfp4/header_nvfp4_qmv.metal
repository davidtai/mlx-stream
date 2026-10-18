// NVFP4 decode experts in MLX's GEMV order (the qmv texts): each pair's row is MLX's gather_qmv_fast in nvfp4 mode with
// the slot's global scale, bit for bit (mlx/backend/metal/kernels/fp_quantized.h, MLX 64ea011cb: fp_qmv_fast_impl, qdot,
// dequantize_scale, fp4_e2m1, fp8_e4m3). Per 512-value chunk a lane holds 16 activations (one E4M3 group) and, per row,
// the group's 16 codes (one uint2). The chunk adds e4m3 * accum to the row's sum; accum is the four 4-code partial sums
// of x * e2m1 in MLX's order (each product exact: a bf16 activation times an E2M1 value). After the chunks: simd_sum,
// times g / 2688, rounded to T. This header follows header_glm5n_nvfp4.metal (the family's helpers: the slot ids, the
// global scale's line, the E4M3 decode, the activation load, the f32 sigmoid, the union vote).

// MLX's fp4_e2m1 -> float16: the code's low three bits as half bits, times 2^14 (exact); bit 3 the sign (code 8: -0).
inline half nvq_e2m1(uint c) {
  half converted = as_type<half>(ushort((c & 7u) << 9));
  converted *= 16384.0;
  return (c & 8u) ? -converted : converted;
}
// A lane's 16 codes (uint2: codes 0..7 in .x, 8..15 in .y, element j in bits 4 * (j % 8)) as their E2M1 values, in
// code order.
inline void nvq_decode16(uint2 w, thread half* d) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    d[j] = nvq_e2m1(w.x >> (4 * j));
    d[8 + j] = nvq_e2m1(w.y >> (4 * j));
  }
}
// MLX's qdot<float, 16, 4>: accum += (x0 d0 + x1 d1 + x2 d2 + x3 d3) for each four codes, then the group's scale times
// accum.
inline float nvq_dot16(thread const half* d, thread const float* x, float scale) {
  float accum = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    accum += (x[4 * i] * float(d[4 * i]) + x[4 * i + 1] * float(d[4 * i + 1]) + x[4 * i + 2] * float(d[4 * i + 2]) +
              x[4 * i + 3] * float(d[4 * i + 3]));
  }
  return scale * accum;
}
