// mimo_glue_norm: layer 0's input norm (no residual add). x [M, 4096] -> n [M, 4096]. One 256-thread threadgroup per row.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const size_t row = threadgroup_position_in_grid.y;
threadgroup float sums[32];
T hrow[16];
const device T* xr = x + row * GLUE_D;
const float inv = glue_row_inv<false>(xr, xr, hrow, sg, lane, sums);
device T* nr = n + row * GLUE_D;
for (int r = 0; r < 4; ++r) {
  for (int i = 0; i < 4; ++i) {
    const int c = 1024 * r + 4 * (32 * int(sg) + int(lane)) + i;
    nr[c] = glue_normed(hrow[4 * r + i], inv, w[c]);
  }
}
