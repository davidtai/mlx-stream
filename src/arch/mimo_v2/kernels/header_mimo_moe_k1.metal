// K1 (mimo/k1-moe-bw): the decode expert kernel family's helpers, appended to header_mimo_moe_mx.metal. Per member row
// the banked texts' arithmetic exactly (same lane -> code mapping, products, partial sums, fold, simd_sum, epilogue);
// what changes is the work around it: the FP4 decode, the union membership, and when loads are issued.
// Decode: one 32-bit op handles two codes. Half j of word w's decode holds codes j (lo) and j + 4 (hi), each as the bits
// mimo_e2m1 builds ((c & 7) << 9 | (c & 8) << 12), so float() of either half is mimo_e2m1(c) bit for bit. 19 integer
// ops per 8 codes (mimo_e2m1: extract + 5 per code).
inline void mimo_k1_decode8(uint w, thread half2* h) {
  h[0] = as_type<half2>(((w << 9) & 0x0E000E00u) | ((w << 12) & 0x80008000u));
  h[1] = as_type<half2>(((w << 5) & 0x0E000E00u) | ((w << 8) & 0x80008000u));
  h[2] = as_type<half2>(((w << 1) & 0x0E000E00u) | ((w << 4) & 0x80008000u));
  h[3] = as_type<half2>(((w >> 3) & 0x0E000E00u) | (w & 0x80008000u));
}
// A lane's 16 codes (uint2: codes 0..7 in .x, 8..15 in .y) decoded: h[0..3] from .x, h[4..7] from .y.
inline void mimo_k1_decode16(uint2 w, thread half2* h) {
  mimo_k1_decode8(w.x, h);
  mimo_k1_decode8(w.y, h + 4);
}
// mimo_fp4_dot16 on the decoded codes: code i of .x is h[i % 4].x (i < 4) or h[i % 4].y (i >= 4), likewise .y in h[4..7];
// a folds codes 0..7, b codes 8..15, each `+= x[j] * v_j + x[j + 1] * v_{j+1}` for j = 0, 2, 4, 6, then a + b.
inline float mimo_k1_dot16(thread const half2* h, thread const float* x) {
  float a = 0.0f, b = 0.0f;
  a += x[0] * float(h[0].x) + x[1] * float(h[1].x);
  b += x[8] * float(h[4].x) + x[9] * float(h[5].x);
  a += x[2] * float(h[2].x) + x[3] * float(h[3].x);
  b += x[10] * float(h[6].x) + x[11] * float(h[7].x);
  a += x[4] * float(h[0].y) + x[5] * float(h[1].y);
  b += x[12] * float(h[4].y) + x[13] * float(h[5].y);
  a += x[6] * float(h[2].y) + x[7] * float(h[3].y);
  b += x[14] * float(h[6].y) + x[15] * float(h[7].y);
  return a + b;
}
// FMA == 1 (rounding-class alternative, timed by the m4attr family, never routed): the lane's 16 products in one f32 FMA
// chain in code order (p = x0 v0, then p = fma(x_j, v_j, p)); the caller folds the block scale once per lane block,
// acc = fma(p, scale, acc). 17 FP ops per 16 codes instead of the banked order's 24 + 2. A row's order is fixed whatever
// the wave (M-invariant); vs the banked order rounding-class (each FMA rounds once).
inline float mimo_k1_dot16_fma(thread const half2* h, thread const float* x) {
  float p = x[0] * float(h[0].x);
  p = fma(x[1], float(h[1].x), p);
  p = fma(x[2], float(h[2].x), p);
  p = fma(x[3], float(h[3].x), p);
  p = fma(x[4], float(h[0].y), p);
  p = fma(x[5], float(h[1].y), p);
  p = fma(x[6], float(h[2].y), p);
  p = fma(x[7], float(h[3].y), p);
  p = fma(x[8], float(h[4].x), p);
  p = fma(x[9], float(h[5].x), p);
  p = fma(x[10], float(h[6].x), p);
  p = fma(x[11], float(h[7].x), p);
  p = fma(x[12], float(h[4].y), p);
  p = fma(x[13], float(h[5].y), p);
  p = fma(x[14], float(h[6].y), p);
  p = fma(x[15], float(h[7].y), p);
  return p;
}
// A lane's 16 activation values of one 512-value chunk (two 16-byte loads), each float(T) as the banked texts convert it.
template <typename T>
inline void mimo_k1_x16(const device T* xt, thread float* xf) {
  const device uint4* v = (const device uint4*)xt;
  const uint4 p = v[0];
  const uint4 q = v[1];
  const uint w[8] = {p.x, p.y, p.z, p.w, q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const vec<T, 2> t = as_type<vec<T, 2>>(w[i]);
    xf[2 * i] = float(t.x);
    xf[2 * i + 1] = float(t.y);
  }
}
// The union rule of mimo_union_members (header_mimo_moe_union.metal) by simdgroup vote (a wave holds <= 64 pairs): pair
// e leads when its rank among the pairs naming its slot is a multiple of MAXM, and then takes itself and the next
// MAXM - 1 of them in pair order. Returns the members and their count (0: not a leader). MAXM == 1: every pair leads
// itself alone (the rule's own answer), no vote. Uniform across the simdgroup (every lane votes on the same ids).
template <int MAXM, typename I>
inline int mimo_k1_members(I ids, uint np, uint e, uint lane, thread uint* mem) {
  if (MAXM == 1) {
    mem[0] = e;
    return 1;
  }
  const uint id = ids[e];
  const uint q1 = lane + 32u;
  const bool v0 = lane < np && ids[min(lane, np - 1u)] == id;
  const bool v1 = q1 < np && ids[min(q1, np - 1u)] == id;
  uint lo = uint(static_cast<simd_vote::vote_t>(simd_ballot(v0)));
  uint hi = uint(static_cast<simd_vote::vote_t>(simd_ballot(v1)));
  const uint rank = e < 32u ? popcount(lo & ((1u << e) - 1u)) : popcount(lo) + popcount(hi & ((1u << (e - 32u)) - 1u));
  if (rank % uint(MAXM) != 0u) return 0;
  if (e < 32u) {
    lo &= ~((1u << e) - 1u);
  } else {
    lo = 0u;
    hi &= ~((1u << (e - 32u)) - 1u);
  }
  int nm = 0;
#pragma unroll
  for (int k = 0; k < MAXM; ++k) {
    if (lo != 0u) {
      mem[nm++] = ctz(lo);
      lo &= lo - 1u;
    } else if (hi != 0u) {
      mem[nm++] = 32u + ctz(hi);
      hi &= hi - 1u;
    }
  }
  return nm;
}
