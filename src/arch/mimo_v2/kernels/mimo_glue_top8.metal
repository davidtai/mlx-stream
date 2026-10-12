// mimo_glue_top8: the router's tail after its product (module.route): MLX's Sigmoid (metallib -fno-fast-math: abs and
// exp are the precise ones), the correction add, the negation,
// argpartition's first 8 (MLX runs it as its stable block merge sort: LessThan with NaN last, ties to the lower index),
// take_along_axis of the sigmoid scores, the row sum (row_reduce_small: `total = vals[j] + total` in order), + 1e-20 and
// the divide. logits [M, 256], bias [256] -> ids [M, 8] (U32), scores [M, 8]. One 256-thread threadgroup per row.
const int e = int(thread_position_in_threadgroup.x);
const size_t row = threadgroup_position_in_grid.y;
threadgroup float vs[256];
threadgroup float picked[8];
const float x = logits[row * 256 + e];
const float y = 1 / (1 + metal::precise::exp(metal::precise::abs(x)));
const float s = (x < 0) ? y : 1 - y;
const float v = -(s + bias[e]);
vs[e] = v;
threadgroup_barrier(mem_flags::mem_threadgroup);
// A stable sort puts e after every key strictly less and every equal key at a lower index.
const bool vn = metal::isnan(v);
int rank = 0;
for (int j = 0; j < 256; ++j) {
  const float u = vs[j];
  const bool un = metal::isnan(u);
  const bool u_lt_v = (un | vn) ? ((!un) & vn) : (u < v);
  const bool v_lt_u = (un | vn) ? ((!vn) & un) : (v < u);
  rank += (u_lt_v || (j < e && !v_lt_u)) ? 1 : 0;
}
if (rank < 8) {
  ids[row * 8 + rank] = uint(e);
  picked[rank] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e < 8) {
  float total = 0;
  for (int j = 0; j < 8; ++j) {
    total = picked[j] + total;
  }
  const float den = total + 1e-20f;
  scores[row * 8 + e] = picked[e] / den;
}
