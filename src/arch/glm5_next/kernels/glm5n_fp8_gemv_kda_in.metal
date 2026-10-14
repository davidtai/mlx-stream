// glm5n_fp8_gemv_kda_in (`fuse_kda_inputs`): a KDA layer's six FP8 input projections of the same rows x [M, K] in one
// launch: q, k, v [NB, K] (the q / k / v sites' plan: RB rows a simdgroup, KSB K partitions) then f_a, g_a [NA, K] and b
// [NC, K] (their plan: RS, KSS), each segment its own block of threadgroups in that order and its own output [M, N]. One
// threadgroup shape (TG simdgroups) for every segment: RGB = TG / KSB row groups in a q / k / v threadgroup, RGS = TG / KSS
// in the others. Every output is glm5n_fp8_gemv_rows's (mimo_fp8_gemv.metal's code) at its site's KS: the stock launch's
// sum bit for bit (R and RG only pick which rows a simdgroup owns).
const int sgi = int(simdgroup_index_in_threadgroup);
const int lane = int(thread_index_in_simdgroup);
int tg = int(threadgroup_position_in_grid.y);
constexpr int GB = NB / (RB * RGB);
constexpr int GA = NA / (RS * RGS);
constexpr int GC = NC / (RS * RGS);
constexpr int PB = KSB > 1 ? RGB * KSB * M * RB : 1;
constexpr int PS = KSS > 1 ? RGS * KSS * M * RS : 1;
static_assert(RGB * KSB == RGS * KSS, "one threadgroup shape for every segment");
threadgroup float part[PB > PS ? PB : PS];
if (tg < GB) {
  glm5n_fp8_gemv_rows<M, NB, K, RB, KSB, RGB>(x, w_q, s_q, y_q, tg, sgi, lane, part);
  return;
}
tg -= GB;
if (tg < GB) {
  glm5n_fp8_gemv_rows<M, NB, K, RB, KSB, RGB>(x, w_k, s_k, y_k, tg, sgi, lane, part);
  return;
}
tg -= GB;
if (tg < GB) {
  glm5n_fp8_gemv_rows<M, NB, K, RB, KSB, RGB>(x, w_v, s_v, y_v, tg, sgi, lane, part);
  return;
}
tg -= GB;
if (tg < GA) {
  glm5n_fp8_gemv_rows<M, NA, K, RS, KSS, RGS>(x, w_fa, s_fa, y_fa, tg, sgi, lane, part);
  return;
}
tg -= GA;
if (tg < GA) {
  glm5n_fp8_gemv_rows<M, NA, K, RS, KSS, RGS>(x, w_ga, s_ga, y_ga, tg, sgi, lane, part);
  return;
}
tg -= GA;
if (tg < GC) {
  glm5n_fp8_gemv_rows<M, NC, K, RS, KSS, RGS>(x, w_b, s_b, y_b, tg, sgi, lane, part);
}
