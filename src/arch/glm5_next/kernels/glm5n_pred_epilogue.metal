// glm5n_pred_epilogue (`fuse_predictor`): the lookahead predictor's epilogue (experts.zig `scoresBf16`: MLX's astype to f32,
// its sigmoid and its add of the correction bias, over the router product's BF16 logits [M, E]) in one dispatch, element for
// element the three stock dispatches' arithmetic: the exact bf16 -> f32 widening; MLX's Sigmoid on f32 (mlx/backend/metal/
// kernels/unary_ops.h: y = 1 / (1 + precise::exp(|x|)), then x < 0 ? y : 1 - y); MLX's f32 Add with the bias broadcast over
// the rows. One thread per element (E threads a threadgroup, one threadgroup a row).
const uint i = thread_position_in_grid.x;
const float x = static_cast<float>(logits[i]);
const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
const float s = (x < 0) ? y : 1 - y;
scores[i] = s + correction[i % E];
