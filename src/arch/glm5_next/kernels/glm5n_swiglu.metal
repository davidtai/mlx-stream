// glm5n_swiglu (GLM-5.3-Flash's decode MLP activation, glm5_next swiglu.zig, `decode_swiglu: fused`): the clamped
// SwiGLU between a dense MLP's or the shared expert's gate / up GEMVs and its down GEMV, silu(min(gate, lim)) *
// clip(up, -lim, lim) (mlx-serve's `clampedSwiGLU`, the SDK's `swiglu_clamped`), in one dispatch where the op chain runs
// five (clip, minimum, the compiled silu, mul). The GEMVs stay outside the text: the weight format's seam. In the
// pattern of MiMo's k1 gate/up epilogue (mlx-stream src/arch/mimo_v2/kernels/mimo_moe_k1_gateup.metal): f32 arithmetic,
// one rounding to T. gate / up / out [rows, I] row-contiguous; consts [8] f32 (consts[0]: the limit). One thread per
// four elements; the grid is exactly rows * I / 4 (I % 1024 == 0).
const uint i = thread_position_in_grid.x * 4;
const float lim = consts[0];
for (uint j = 0; j < 4; ++j) {
  const float gt = metal::min(float(gate[i + j]), lim);
  const float u = metal::clamp(float(up[i + j]), -lim, lim);
  out[i + j] = static_cast<T>(gt / (1.0f + metal::exp(-gt)) * u);
}
