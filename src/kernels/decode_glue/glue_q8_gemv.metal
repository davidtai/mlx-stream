// glue_q8_gemv: a decode row's product with a q8 trunk projection (MLX's affine 8 bits, group 64), MLX's
// affine_qmv_fast row for row (quantized.h qmv_fast_impl at bits 8: values_per_thread 8, block_size 256): lane l of a
// row's simdgroup holds the 8 values [256 b + 8 l, 256 b + 8 l + 8) of every block b; for each block in order the lane
// adds `qdot(codes, x, scale, bias, sum)` to its row's partial (header_decode_glue_q8.metal); then simd_sum and bf16
// once. Each output's arithmetic is that, whatever ROWS (rows a simdgroup: MLX's 4), SPT (simdgroups a threadgroup:
// MLX's 2) and PF (the next block's codes loaded before this block's products) are, so every output is the stock
// launch's bit for bit. x [1, K] bf16, w [N, K / 4] u32, scales, biases [N, K / 64] bf16 -> y [1, N] bf16.
constexpr int BLOCK = 256;
constexpr int KG = K / 64;
constexpr int NB = K / BLOCK;
const uint lane = thread_index_in_simdgroup;
const int out_row = (int(threadgroup_position_in_grid.y) * SPT + int(simdgroup_index_in_threadgroup)) * ROWS;
const device uint32_t* ws = w + (size_t(out_row) * K + lane * 8) / 4;
const device bfloat16_t* sl = scales + size_t(out_row) * KG + lane / 8;
const device bfloat16_t* bl = biases + size_t(out_row) * KG + lane / 8;
const device bfloat16_t* xs = x + lane * 8;
float result[ROWS];
uint lo[ROWS];
uint hi[ROWS];
for (int r = 0; r < ROWS; ++r) {
  result[r] = 0;
  lo[r] = ws[size_t(r) * (K / 4)];
  hi[r] = ws[size_t(r) * (K / 4) + 1];
}
for (int b = 0; b < NB; ++b) {
  uint nlo[ROWS];
  uint nhi[ROWS];
  if (PF != 0 && b + 1 < NB) {
    for (int r = 0; r < ROWS; ++r) {
      nlo[r] = ws[size_t(r) * (K / 4) + (b + 1) * (BLOCK / 4)];
      nhi[r] = ws[size_t(r) * (K / 4) + (b + 1) * (BLOCK / 4) + 1];
    }
  }
  float x_thread[8];
  const float sum = glue_q8_load_vector(xs + b * BLOCK, x_thread);
  for (int r = 0; r < ROWS; ++r) {
    const float s = sl[r * KG + b * (BLOCK / 64)];
    const float bias = bl[r * KG + b * (BLOCK / 64)];
    result[r] += glue_q8_qdot(lo[r], hi[r], x_thread, s, bias, sum);
  }
  if (b + 1 < NB) {
    for (int r = 0; r < ROWS; ++r) {
      lo[r] = PF != 0 ? nlo[r] : ws[size_t(r) * (K / 4) + (b + 1) * (BLOCK / 4)];
      hi[r] = PF != 0 ? nhi[r] : ws[size_t(r) * (K / 4) + (b + 1) * (BLOCK / 4) + 1];
    }
  }
}
for (int r = 0; r < ROWS; ++r) {
  result[r] = simd_sum(result[r]);
  if (lane == 0) {
    y[out_row + r] = static_cast<bfloat16_t>(result[r]);
  }
}
