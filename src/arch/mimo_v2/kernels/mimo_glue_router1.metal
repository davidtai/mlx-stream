// mimo_glue_router1 (M = 1): the residual add, the post-attention norm and the router's F32 product in one dispatch.
// a, b [1, 4096], w [4096], gate [256, 4096] (the router's BF16 weight; its F32 copy holds the same values) ->
// h, n [1, 4096] (threadgroup 0 writes them) and logits [1, 256]. 64 threadgroups of 256 threads: threadgroup g is MLX
// gemv's threadgroup g (rows 4g .. 4g + 3, BN = 8 simdgroups x 32 lanes x TN = 4, blockN = 1024), each re-deriving
// the normed row in the lane order of its own columns.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int g = int(threadgroup_position_in_grid.y);
threadgroup float sums[32];
threadgroup float part[8 * 8];
T hrow[16];
const float inv = glue_row_inv<true>(a, b, hrow, sg, lane, sums);
float xv[16];
for (int r = 0; r < 4; ++r) {
  for (int i = 0; i < 4; ++i) {
    const int c = 1024 * r + 4 * (32 * int(sg) + int(lane)) + i;
    const T m = glue_normed(hrow[4 * r + i], inv, w[c]);
    xv[4 * r + i] = m;
    if (g == 0) {
      h[c] = hrow[4 * r + i];
      n[c] = m;
    }
  }
}
// GEMVKernel::run: per row tm, `result[tm] += inter[tn] * v_coeff[tn]` over the 4 blocks and 4 columns in order, the
// halving shuffles 16 .. 1, then simdgroup 0 adds simdgroups 1 .. 7's results in order.
float result[4] = {0, 0, 0, 0};
for (int r = 0; r < 4; ++r) {
  const int bn = 1024 * r + 4 * (32 * int(sg) + int(lane));
  for (int tm = 0; tm < 4; ++tm) {
    const device T* wr = gate + size_t(4 * g + tm) * GLUE_D + bn;
    float inter[4];
    for (int tn = 0; tn < 4; ++tn) {
      inter[tn] = wr[tn];
    }
    for (int tn = 0; tn < 4; ++tn) {
      result[tm] += inter[tn] * xv[4 * r + tn];
    }
  }
}
for (int tm = 0; tm < 4; ++tm) {
  for (ushort o = 16; o >= 1; o >>= 1) {
    result[tm] += simd_shuffle_down(result[tm], o);
  }
}
if (lane == 0) {
  for (int tm = 0; tm < 4; ++tm) {
    part[int(sg) * 8 + tm] = result[tm];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0 && lane == 0) {
  for (int s2 = 1; s2 < 8; ++s2) {
    for (int tm = 0; tm < 4; ++tm) {
      result[tm] += part[s2 * 8 + tm];
    }
  }
  for (int tm = 0; tm < 4; ++tm) {
    logits[4 * g + tm] = result[tm];
  }
}
