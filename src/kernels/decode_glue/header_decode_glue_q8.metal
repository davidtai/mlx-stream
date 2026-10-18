// The q8 GEMV's header (beam glue.zig `q8Gemv`): MLX 64ea011cb's affine 8-bit qmv helpers (quantized.h `load_vector`
// and `qdot` at bits 8, U = float), line for line: the x values summed in order and kept as floats; the dot of 8 codes
// with them in order, then `scale * accum + sum * bias`. The codes come as two 32-bit words (little-endian: byte i of
// the 8 is (word >> 8 (i % 4)) & 0xff), the same values MLX reads as bytes. Contraction as MLX's: the default.
METAL_FUNC float glue_q8_load_vector(const device bfloat16_t* x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += x[i];
    x_thread[i] = x[i];
  }
  return sum;
}

METAL_FUNC float glue_q8_qdot(uint lo, uint hi, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  for (int i = 0; i < 8; i++) {
    const uint wi = ((i < 4 ? lo : hi) >> (8 * (i & 3))) & 0xffu;
    accum += x_thread[i] * float(wi);
  }
  return scale * accum + sum * bias;
}
