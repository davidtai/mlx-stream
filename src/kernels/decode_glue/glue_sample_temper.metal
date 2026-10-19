// glue_sample_temper: a sampled row's tempered logits (acceptance.zig `filteredRows`: f32(logits) * (1 / T), MLX's
// vsn_Multiply) and their negation (the sort's key: the row in descending order is the ascending sort of -lt, negated).
// x [1, V] f32, consts [8] f32 (0: 1 / T) -> lt, nlt [1, V] f32. One thread a value.
const uint i = thread_position_in_grid.x;
if (int(i) < V) {
  const float v = x[i] * consts[0];
  lt[i] = v;
  nlt[i] = -v;
}
