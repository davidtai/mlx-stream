// glm5n_moe_combine: the decode lane's routed combine and the shared expert's add (experts.zig `combineStock`, then
// graph.zig's `add(routed, shared)`): MiMo's mimo_moe_combine text (the f32 score-weighted sum over the TOPK routed rows in
// router order as MLX's col_reduce_small folds it: each product added to the 0 init, then k = 0..TOPK-1 in order; bf16
// once), then MLX's bf16 binary add of the shared expert's row. rows [n * TOPK, H] bf16, w [n, TOPK] f32, shared [n, H]
// bf16 -> y [n, H] bf16. One thread per (column, token). No contraction into FMA (header_glm5n_combine.metal).
const uint h = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const device auto* r = rows + size_t(t) * TOPK * H + h;
const device float* wt = w + size_t(t) * TOPK;
float acc = float(r[0]) * wt[0] + 0.0f;
for (int k = 1; k < TOPK; ++k) {
  const float p = float(r[size_t(k) * H]) * wt[k] + 0.0f;
  acc = p + acc;
}
const T routed = static_cast<T>(acc);
y[size_t(t) * H + h] = static_cast<T>(static_cast<float>(routed) + static_cast<float>(shared[size_t(t) * H + h]));
