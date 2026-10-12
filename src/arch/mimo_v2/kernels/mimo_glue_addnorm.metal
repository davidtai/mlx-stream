// mimo_glue_addnorm: the residual add and the next norm (a layer boundary: the previous MLP output into the residual and
// this layer's input norm; or the attention output and the post-attention norm). a, b [M, 4096] -> h = a + b, n.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const size_t row = threadgroup_position_in_grid.y;
threadgroup float sums[32];
T hrow[16];
const float inv = glue_row_inv<true>(a + row * GLUE_D, b + row * GLUE_D, hrow, sg, lane, sums);
device T* hr = h + row * GLUE_D;
device T* nr = n + row * GLUE_D;
for (int r = 0; r < 4; ++r) {
  for (int i = 0; i < 4; ++i) {
    const int c = 1024 * r + 4 * (32 * int(sg) + int(lane)) + i;
    hr[c] = hrow[4 * r + i];
    nr[c] = glue_normed(hrow[4 * r + i], inv, w[c]);
  }
}
