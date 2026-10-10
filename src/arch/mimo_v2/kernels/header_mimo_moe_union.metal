// M4 lever 1 (mimo/m4-attribution): the union texts' helpers, appended to header_mimo_moe_mx.metal. A slot's 16 FP4
// codes decoded once (as mimo_e2m1's half, exact) and dotted with every member row in mimo_fp4_dot16's order: per row
// the same products, the same two partial sums, the same fold.
inline void mimo_fp4_decode16(uint2 w, thread half* o) {
  for (int j = 0; j < 8; ++j) {
    const uint cx = (w.x >> (4 * j)) & 0xFu;
    const uint cy = (w.y >> (4 * j)) & 0xFu;
    o[j] = as_type<half>(ushort(((cx & 0x7u) << 9) | ((cx & 0x8u) << 12)));
    o[8 + j] = as_type<half>(ushort(((cy & 0x7u) << 9) | ((cy & 0x8u) << 12)));
  }
}
inline float mimo_fp4_dot16h(thread const half* w, thread const float* x) {
  float a = 0.0f, b = 0.0f;
  for (int j = 0; j < 8; j += 2) {
    a += x[j] * float(w[j]) + x[j + 1] * float(w[j + 1]);
    b += x[8 + j] * float(w[8 + j]) + x[9 + j] * float(w[9 + j]);
  }
  return a + b;
}
// The pairs a wave names one slot with: pair e leads when its rank among them (pair order) is a multiple of MAXM, and
// then takes itself and the next MAXM - 1 of them; every other pair's threadgroup has nothing to do. Returns the
// members (pair indices, pair order) and their count (0: not a leader).
template <int MAXM>
inline int mimo_union_members(const device uint32_t* ids, uint np, uint e, thread uint* mem) {
  const uint id = ids[e];
  uint rank = 0u;
  for (uint q = 0u; q < e; ++q) rank += ids[q] == id ? 1u : 0u;
  if (rank % uint(MAXM) != 0u) return 0;
  int nm = 0;
  for (uint q = e; q < np && nm < MAXM; ++q)
    if (ids[q] == id) mem[nm++] = q;
  return nm;
}
template <int MAXM>
inline int mimo_union_members(const constant uint32_t* ids, uint np, uint e, thread uint* mem) {
  const uint id = ids[e];
  uint rank = 0u;
  for (uint q = 0u; q < e; ++q) rank += ids[q] == id ? 1u : 0u;
  if (rank % uint(MAXM) != 0u) return 0;
  int nm = 0;
  for (uint q = e; q < np && nm < MAXM; ++q)
    if (ids[q] == id) mem[nm++] = q;
  return nm;
}
