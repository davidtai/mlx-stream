// glm5n_fp8_gemv_pair (`fuse_gate_up`): an MLP's gate and up FP8 projections of the same rows x [M, K] in one launch, both
// [N, K] on their one plan (R, KS, RG: the stock launch's threadgroup shape): gate's threadgroups, then up's, each its own
// output [M, N]. Every output is glm5n_fp8_gemv_rows's (mimo_fp8_gemv.metal's code): the stock launch's sum bit for bit.
const int sgi = int(simdgroup_index_in_threadgroup);
const int lane = int(thread_index_in_simdgroup);
const int tg = int(threadgroup_position_in_grid.y);
constexpr int GP = N / (R * RG);
threadgroup float part[KS > 1 ? RG * KS * M * R : 1];
if (tg < GP) {
  glm5n_fp8_gemv_rows<M, N, K, R, KS, RG>(x, w_gate, s_gate, y_gate, tg, sgi, lane, part);
} else {
  glm5n_fp8_gemv_rows<M, N, K, R, KS, RG>(x, w_up, s_up, y_up, tg - GP, sgi, lane, part);
}
