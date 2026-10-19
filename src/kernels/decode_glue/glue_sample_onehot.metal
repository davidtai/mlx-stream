// glue_sample_onehot: the plugin's row for its drawn token (sdk_ext/sampler.zig `chosen`: 0 at the token, -inf
// elsewhere), the token the count of the row's scan outputs under u * total (glue_sample_scan MODE 1 STAGE 1's per-block
// counts, summed: acceptance.zig `drawRows`). cnt [NB] int32 -> out [1, V] f32. One thread a value.
const uint i = thread_position_in_grid.x;
if (int(i) < V) {
  int tok = 0;
  for (int q = 0; q < (V + GLUE_SM_BLOCK - 1) / GLUE_SM_BLOCK; q++) {
    tok += cnt[q];
  }
  out[i] = (int(i) == tok) ? 0.0f : -metal::numeric_limits<float>::infinity();
}
