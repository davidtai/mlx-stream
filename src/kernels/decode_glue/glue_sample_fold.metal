// glue_sample_fold: softmax_looped's fold (header_decode_glue_sample.metal) of a sampled row's MODE row over 1,024
// threads in 32 threadgroups of one simdgroup (MLX's simdgroup s is threadgroup s here): each simdgroup's (max,
// normalizer) -> part [64] f32. MODE 0 reads the sorted row (`sorted`: the ascending sort of -lt); MODE 1 the tempered
// logits `lt` kept at or above the kept set's least, the sorted row's entry count - 1 (count: the kept blocks' sum, cnt
// [NB] int32 from glue_sample_scan MODE 0 STAGE 1).
const uint t = thread_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
float thr = 0;
if (MODE == 1) {
  int count = 0;
  for (int q = 0; q < (V + GLUE_SM_BLOCK - 1) / GLUE_SM_BLOCK; q++) {
    count += cnt[q];
  }
  thr = -sorted[count - 1];
}
const float2 st = glue_sm_fold(MODE, V, t, lt, sorted, thr);
if (lane == 0) {
  part[2 * (t / 32)] = st.x;
  part[2 * (t / 32) + 1] = st.y;
}
