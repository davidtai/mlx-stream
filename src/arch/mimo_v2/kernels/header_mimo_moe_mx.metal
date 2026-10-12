// MiMo MXFP4 g32 slot-bank decode MoE (design-decode-fusion 2c). Donor: mlx-serve origin/mimo-v2 @ 7a9e392f src/moe_fp4.zig
// E2M1_HEADER ++ MX_SCALE ++ DOT16 (bytes as passed to MLX there, renamed mimo_*), plus MLX's f32 sigmoid expression
// (mlx/backend/metal/kernels/unary_ops.h `Sigmoid`, MLX 64ea011c) for MiMo's F32 SiLU contract.
// e2m1 decodes as value x 2^-14, E8M0 as value x 2^-8; one x 2^22 per row restores both (exact in f32).
inline float mimo_e2m1(uint c) {
  return float(as_type<half>(ushort(((c & 0x7u) << 9) | ((c & 0x8u) << 12))));
}
inline float mimo_fp4_scale(uint b) {
  return as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
}
inline float mimo_fp4_dot16(uint2 w, thread const float* x) {
  float a = 0.0f, b = 0.0f;
  for (int j = 0; j < 8; j += 2) {
    a += x[j] * mimo_e2m1((w.x >> (4 * j)) & 0xFu) + x[j + 1] * mimo_e2m1((w.x >> (4 * j + 4)) & 0xFu);
    b += x[8 + j] * mimo_e2m1((w.y >> (4 * j)) & 0xFu) + x[9 + j] * mimo_e2m1((w.y >> (4 * j + 4)) & 0xFu);
  }
  return a + b;
}
inline float mimo_sigmoid_f32(float x) {
  float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// A packed slot id: bank << 24 | row (deepseek_v41_experts.zig gateUpWaveBanked); bank 0 base, 1 ext, 2 transient.
inline uint mimo_slot_bank(uint id) {
  return id >> 24;
}
inline size_t mimo_slot_row(uint id) {
  return size_t(id & 0xFFFFFFu);
}
