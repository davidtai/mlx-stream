// glue_sample_scan: contiguous_scan's inclusive sum (header_decode_glue_sample.metal) of a sampled row's MODE softmax
// (its (max, 1 / normalizer) from glue_sample_fold's part) over NB = ceil(V / 4096) threadgroups of 1,024 threads:
// threadgroup r is MLX's loop iteration r.
//   STAGE 0: each block's pieces before its prefix: its last thread's running sum, simdgroup prefix and thread prefix
//            (tri_out[3 r .. 3 r + 2]), and the same three of the row's last value (tri_out[3 NB ..]).
//   STAGE 1: each block's prefix from the blocks before it (prefix_{r+1} = ((A_r + prefix_r) + B_r) + C_r: MLX's carry,
//            the block's last output), its outputs ((v + prefix) + simdgroup prefix) + thread prefix, and the count of
//            its values the stage keeps -> cnt_out[r]: MODE 0 the filter's `cum - p < top_p` (consts[1]); MODE 1 the
//            draw's `cum < u * total` (consts[2] the token's uniform, total the row's last output).
// lt, sorted [1, V] f32, cnt [NB] int32 (MODE 1: the kept counts), part [64] f32, tri [3 NB + 3] f32 (STAGE 1),
// consts [8] f32 -> tri_out [3 NB + 3] f32 (STAGE 0), cnt_out [NB] int32 (STAGE 1), dbg [1, V] f32 (STAGE 1 with DBG 1,
// the device test's: every scan output; a served launch's dbg is one unwritten value).
constexpr int NB = (V + GLUE_SM_BLOCK - 1) / GLUE_SM_BLOCK;
const uint r = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
threadgroup float sums[32];
threadgroup float2 stat_tg;
threadgroup int counts[32];
float thr = 0;
if (MODE == 1) {
  int count = 0;
  for (int q = 0; q < NB; q++) {
    count += cnt[q];
  }
  thr = -sorted[count - 1];
}
if (sg == 0) {
  const float2 st = glue_sm_stat(part, lane);
  if (lane == 0) {
    stat_tg = st;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float2 stat = stat_tg;
const int offset = int(r) * GLUE_SM_BLOCK + int(lid) * 4;
float p[4];
float v[4];
for (int i = 0; i < 4; i++) {
  p[i] = glue_sm_prob(MODE, V, offset + i, lt, sorted, thr, stat);
  v[i] = p[i];
}
float sgp;
const float pt = glue_scan_block(v, sgp, sums, lane, sg);
if (STAGE == 0) {
  if (lid == 1023) {
    tri_out[3 * r] = v[3];
    tri_out[3 * r + 1] = sgp;
    tri_out[3 * r + 2] = pt;
  }
  const int e = V - 1 - (NB - 1) * GLUE_SM_BLOCK;
  if (int(r) == NB - 1 && int(lid) == e / 4) {
    tri_out[3 * NB] = v[e % 4];
    tri_out[3 * NB + 1] = sgp;
    tri_out[3 * NB + 2] = pt;
  }
} else {
  float prefix = 0;
  float last = 0;
  for (int q = 0; q < NB - 1; q++) {
    if (q == int(r)) {
      last = prefix;
    }
    prefix = ((tri[3 * q] + prefix) + tri[3 * q + 1]) + tri[3 * q + 2];
  }
  // `prefix` now carries into the last block; `last` is this block's (the last block's own is `prefix`).
  const float own = (int(r) == NB - 1) ? prefix : last;
  float bound = consts[1];
  if (MODE == 1) {
    const float total = ((tri[3 * NB] + prefix) + tri[3 * NB + 1]) + tri[3 * NB + 2];
    bound = consts[2] * total;
  }
  int c = 0;
  for (int i = 0; i < 4; i++) {
    const float cum = ((v[i] + own) + sgp) + pt;
    if (DBG != 0 && offset + i < V) {
      dbg[offset + i] = cum;
    }
    const bool keep = (MODE == 0) ? ((cum - p[i]) < bound) : (cum < bound);
    c += (offset + i < V && keep) ? 1 : 0;
  }
  c = simd_sum(c);
  if (lane == 0) {
    counts[sg] = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const int block_count = simd_sum(counts[lane]);
    if (lane == 0) {
      cnt_out[r] = block_count;
    }
  }
}
