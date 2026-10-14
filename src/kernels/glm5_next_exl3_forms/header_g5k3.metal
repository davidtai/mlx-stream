// header_g5k3 (mlx-stream glm5_next exl3 forms): the helpers of the fused EXL3 K=3 decode forms at GLM-5.3-Flash's
// expert shape. Every helper keeps the arithmetic and its order of DeepSeek-V4.1's banked mul1h chain
// (src/kernels/exl3/dsv41_exl3_b3_*.metal, instantiated at GLM's shape in src/kernels/glm5_next_exl3/): only where a
// value lives (register, threadgroup memory, device memory) and which thread computes it change.

// The f32 scale of the 128-point rotation (1 / sqrt(128)), the b3 texts' SCALE.
constant constexpr uint G5K3_SCALE_BITS = 1035273459u;

// The lane's constants of the stock GEMV: the two words of a tile that hold its eight codewords, the funnel shift, and
// each weight m's input row dr[m] within the 16-row k-tile.
struct g5k3_lane {
  uint wa;
  uint wb;
  uint lp;
  uint dr[8];
};

inline g5k3_lane g5k3_lane_of(uint lane) {
  g5k3_lane L;
  L.wa = 2u * ((3u * lane + 2u) >> 2);
  L.wb = (L.wa + 46u) % 48u;
  L.lp = 8u * ((lane + 1u) & 3u);
  for (uint m = 0; m < 8u; ++m) {
    uint q = m & 3u;
    L.dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q;
  }
  return L;
}

// The lane's eight weights of one 16 x 16 tile, mul1 (the stock GEMV's decode, line for line).
inline void g5k3_mul1_w8(const device short* tile, thread const g5k3_lane& L, thread float* w) {
  const device uint* tile32 = reinterpret_cast<const device uint*>(tile);
  uint rd8 = tile32[L.wa >> 1u];
  uint rd9 = tile32[L.wb >> 1u];
  uint lp = L.lp;
  uint v0 = (lp == 0u) ? rd8 : ((rd8 >> lp) | (rd9 << (32u - lp)));
  uint v1 = (lp < 16u) ? ((rd8 >> (lp + 16u)) | (rd9 << (16u - lp))) : (rd9 >> (lp - 16u));
  for (uint m = 0; m < 6u; ++m) {
    uint window = (v0 >> (3u * m)) & 0xffffu;
    uint rr = window * 0x83DCD12Du;
    uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
    half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
    w[m] = float(fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
  }
  for (uint m = 6u; m < 8u; ++m) {
    uint window = (v1 >> (3u * m - 16u)) & 0xffffu;
    uint rr = window * 0x83DCD12Du;
    uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
    half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
    w[m] = float(fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
  }
}

// The stock GEMV's column sums without its threadgroup array: there, red[c][l] = 0 + acc[4] + .. + acc[7] at c = l >> 2
// and 0 + acc[0] + .. + acc[3] at c = (l >> 2) + 8 (zero elsewhere), and column c's sum adds red[c][0..31] from 0 in lane
// order. Only lanes 4 (c & 7) .. 4 (c & 7) + 3 hold a nonzero term of column c, and a running sum that starts at +0 is
// never -0, so adding the other lanes' zeros changes no bit: column c = 0 + the four lanes' terms in lane order. Lane c
// (c < 16) returns column c's sum; the other lanes return a value nobody stores.
inline float g5k3_colsum(thread const float* acc, uint lane) {
  float hi = 0.0f;
  hi += acc[4];
  hi += acc[5];
  hi += acc[6];
  hi += acc[7];
  float lo = 0.0f;
  lo += acc[0];
  lo += acc[1];
  lo += acc[2];
  lo += acc[3];
  const uint c = lane & 15u;
  const uint b = (c & 7u) * 4u;
  float s = 0.0f;
  for (uint k = 0; k < 4u; ++k) {
    const float h = simd_shuffle(hi, ushort(b + k));
    const float l = simd_shuffle(lo, ushort(b + k));
    s += (c < 8u) ? h : l;
  }
  return s;
}

// The b3 texts' 128-point butterflies over B (64 threads, thread t = tid), one stage per barrier. Every thread of the
// threadgroup calls it (the barriers); threads with `on` false only wait.
inline void g5k3_fwht128(threadgroup float* B, uint tid, bool on) {
  for (uint h = 1u; h < 128u; h <<= 1u) {
    if (on) {
      uint k = tid & (h - 1u);
      uint j = ((tid - k) << 1u) + k;
      float a = B[j];
      float b = B[j + h];
      B[j] = a + b;
      B[j + h] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The NVFP4 k1 texts' union rule (header_glm5n_nvfp4.metal glm5n_members, instantiated here unchanged): pair e leads
// when its rank among the wave's pairs naming its slot is a multiple of MAXM, and then takes itself and the next MAXM - 1
// of them in pair order. Returns the member count (0: not a leader). A wave holds <= 64 pairs.
template <int MAXM, typename I>
inline int g5k3_members(I ids, uint np, uint e, uint lane, thread uint* mem) {
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

// ── The w forms (g5k3_guw, g5k3_dnw, g5k3_guwp, g5k3_dnwp): the GEMV of gu_one / down_pair per row, with g5k3_pair's
// decode words on every path, the lane's x as two float2 loads per k-tile, and g5k3_colsum (no threadgroup memory). ──

// g5k3_pair's decode words (g5k3_pair.metal:47-50): a 16-bit window -> the weight. Equal to guone's words for every
// window (256 q = rr & 0xFF00FF00, so rr - 255 q = (rr & 0x00FF00FF) + q): one integer op fewer than guone's.
inline float g5k3_wpair(uint window) {
  uint rr = window * 0x83DCD12Du;
  uint q = (rr >> 8u) & 0x00FF00FFu;
  uint z = (rr - q * 255u) * 0x00010001u + 0x64000000u;
  return float(fma(as_type<half2>(z).y, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));
}

// The lane's two funnel-shifted words of one tile: bits 0..31 (x) and 16..47 (y) of (rd9:rd8) >> lp (the stock GEMV's
// v0 / v1, line for line).
inline uint2 g5k3w_v(const device short* tile, uint wa, uint wb, uint lp) {
  const device uint* tile32 = reinterpret_cast<const device uint*>(tile);
  uint rd8 = tile32[wa >> 1u];
  uint rd9 = tile32[wb >> 1u];
  uint v0 = (lp == 0u) ? rd8 : ((rd8 >> lp) | (rd9 << (32u - lp)));
  uint v1 = (lp < 16u) ? ((rd8 >> (lp + 16u)) | (rd9 << (16u - lp))) : (rd9 >> (lp - 16u));
  return uint2(v0, v1);
}

// Weight m's input x[dr[m]] (dr = 2k + (9, 8, 1, 0)[m & 3], k = lane & 3) from the lane's two float2 loads of the k-tile:
// xa = (x[2k], x[2k + 1]), xb = (x[2k + 8], x[2k + 9]). m is a literal after the unroll: the select folds.
inline float g5k3w_x(uint m, float2 xa, float2 xb) {
  const uint q = m & 3u;
  return q == 0u ? xb.y : (q == 1u ? xb.x : (q == 2u ? xa.y : xa.x));
}

// One tile's 8 weights of the lane into one row's fma chains (the stock GEMV's order: m = 0..7, fma(x, w, acc)).
inline void g5k3w_tile1(uint2 v, float2 xa, float2 xb, thread float* acc) {
  for (uint m = 0; m < 6u; ++m) {
    float w = g5k3_wpair((v.x >> (3u * m)) & 0xffffu);
    acc[m] = fma(g5k3w_x(m, xa, xb), w, acc[m]);
  }
  for (uint m = 6u; m < 8u; ++m) {
    float w = g5k3_wpair((v.y >> (3u * m - 16u)) & 0xffffu);
    acc[m] = fma(g5k3w_x(m, xa, xb), w, acc[m]);
  }
}

// The pair rule's leader: one decode per weight for two rows' chains (row: xa / xb, mate: ya / yb).
inline void g5k3w_tile2(uint2 v, float2 xa, float2 xb, float2 ya, float2 yb, thread float* acc, thread float* accb) {
  for (uint m = 0; m < 6u; ++m) {
    float w = g5k3_wpair((v.x >> (3u * m)) & 0xffffu);
    acc[m] = fma(g5k3w_x(m, xa, xb), w, acc[m]);
    accb[m] = fma(g5k3w_x(m, ya, yb), w, accb[m]);
  }
  for (uint m = 6u; m < 8u; ++m) {
    float w = g5k3_wpair((v.y >> (3u * m - 16u)) & 0xffffu);
    acc[m] = fma(g5k3w_x(m, xa, xb), w, acc[m]);
    accb[m] = fma(g5k3w_x(m, ya, yb), w, accb[m]);
  }
}

// The pair rule (g5k3_pair.metal:8-19): rows of one slot pair up in row order; returns the row's rank among the rows
// naming its slot, the count, and the next such row (the mate; the row itself when none). Threadgroup-uniform. `ids`
// is `constant` below 8 pairs (MLX's small inputs), `device` above.
template <typename I>
inline uint3 g5k3w_rank(I ids, uint nrows, uint row) {
  uint slot = ids[row];
  uint rank = 0u;
  uint cnt = 0u;
  uint mate = row;
  for (uint j = 0; j < nrows; ++j) {
    uint same = (ids[j] == slot) ? 1u : 0u;
    rank += (j < row) ? same : 0u;
    cnt += same;
    mate = (mate == row && j > row && same == 1u) ? j : mate;
  }
  return uint3(rank, cnt, mate);
}
