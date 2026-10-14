// glm5n_moe_combine_idx (`fuse_join`): glm5n_moe_combine with the scheduler's join folded in: the routed rows read in place
// from the call's concatenated outputs `rows` [R, H] bf16 through `order` u32 [n * TOPK] (each routed position's row, the
// join's index: deepseek_v41/experts.zig `joinOrder`) instead of from the join's take(rows, order) copy. Every other line is
// glm5n_moe_combine's: the f32 score-weighted sum over the TOPK routed rows in router order (each product added to the 0
// init, then k = 0..TOPK-1 in order; bf16 once), then the bf16 add of the shared expert's row. w [n, TOPK] f32, shared
// [n, H] bf16 -> y [n, H] bf16. One thread per (column, token). No contraction into FMA (header_glm5n_combine.metal).
const uint h = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const device auto* o = order + size_t(t) * TOPK;
const device float* wt = w + size_t(t) * TOPK;
float acc = float(rows[size_t(o[0]) * H + h]) * wt[0] + 0.0f;
for (int k = 1; k < TOPK; ++k) {
  const float p = float(rows[size_t(o[k]) * H + h]) * wt[k] + 0.0f;
  acc = p + acc;
}
const T routed = static_cast<T>(acc);
y[size_t(t) * H + h] = static_cast<T>(static_cast<float>(routed) + static_cast<float>(shared[size_t(t) * H + h]));
