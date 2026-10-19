// glue_router_select: the router's tail after its product (DeepSeek-V3's noaux_tc, beam graph.zig `route`): MLX's
// Sigmoid, the correction add; with G > 1 groups of EG = E / G experts, each group scored by the sum of its top two
// choice scores (MLX's topk: the last two of its stable sort, or the whole group when it holds two; summed as
// row_reduce_small sums, `total = vals[j] + total` from 0), the TG best groups kept (argpartition of the negated scores:
// the first TG of its stable sort), the other groups' choice scores -inf; then the negation, argpartition's first K (MLX
// runs it as its stable block merge sort: LessThan with NaN last, ties to the lower index), take_along_axis of the
// sigmoid scores, the row sum (`total = vals[j] + total` in order), + 1e-20, the divide (norm_topk_prob), the product
// with consts[0] (the weights' scale; 1 where the model scales the experts' sum instead). logits [M, E] f32, bias [E]
// f32, consts [8] f32 -> ids [M, K] (the scheduler's index type), weights [M, K] f32. One E-thread threadgroup per row.
const int e = int(thread_position_in_threadgroup.x);
const size_t row = threadgroup_position_in_grid.y;
threadgroup float vs[E];
threadgroup float top2[2 * G];
threadgroup float gs[G];
threadgroup float picked[K];
const float s = glue_sigmoid(logits[row * E + e]);
const float c = s + bias[e];
float v = -c;
if (G > 1) {
  constexpr int EG = E / G;
  const int gi = e / EG;
  const int ge = e - gi * EG;
  vs[e] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int in_group = (EG == 2) ? ge : glue_rank(vs + gi * EG, EG, c, ge);
  if (in_group >= EG - 2) {
    top2[2 * gi + (in_group - (EG - 2))] = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e < G) {
    float total = 0;
    total = top2[2 * e] + total;
    total = top2[2 * e + 1] + total;
    gs[e] = -total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool kept = glue_rank(gs, G, gs[gi], gi) < TG;
  v = kept ? -c : INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
vs[e] = v;
threadgroup_barrier(mem_flags::mem_threadgroup);
const int rank = glue_rank(vs, E, v, e);
if (rank < K) {
  ids[row * K + rank] = e;
  picked[rank] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e < K) {
  float total = 0;
  for (int j = 0; j < K; ++j) {
    total = picked[j] + total;
  }
  const float den = total + 1e-20f;
  const float q = picked[e] / den;
  weights[row * K + e] = q * consts[0];
}
