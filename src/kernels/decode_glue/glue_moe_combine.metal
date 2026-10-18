// glue_moe_combine: a decode-lane call's routed combine and its routed scale in one launch (experts.zig `forward`'s
// decode lane: the scheduler's join, route.zig `combineStock`, then graph.zig ScaleBy), with the join folded in: each
// routed row read in place from the call's concatenated outputs `rows` [R, D] bf16 through `order` u32 [M * K] (each
// routed position's row: deepseek_v41/experts.zig `joinOrder`) instead of from the join's take(rows, order) copy. The
// f32 score-weighted sum over the K routed rows in router order (each product f32(row) * w added to the 0 init, then
// k = 0..K-1 in order: header_decode_glue_combine.metal), bf16 once; then the routed scale bf16(f32(.) * consts[0]).
// w [M, K] f32, consts [8] f32 (0: the routed scale) -> y [M, D] bf16. One thread per (column, row). `order` and `w`
// take their address space from MLX's signature (a launch with fewer than 8 of them passes them as constants).
const uint h = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const auto o = order + size_t(t) * K;
const auto wt = w + size_t(t) * K;
float acc = static_cast<float>(rows[size_t(o[0]) * D + h]) * wt[0] + 0.0f;
for (int k = 1; k < K; ++k) {
  const float p = static_cast<float>(rows[size_t(o[k]) * D + h]) * wt[k] + 0.0f;
  acc = p + acc;
}
const bfloat16_t routed = static_cast<bfloat16_t>(acc);
y[size_t(t) * D + h] = static_cast<bfloat16_t>(static_cast<float>(routed) * consts[0]);
