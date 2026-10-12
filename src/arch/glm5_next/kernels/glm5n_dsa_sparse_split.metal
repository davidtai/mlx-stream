// glm5n_dsa_sparse_split (GLM-5.3-Flash's DSA decode, glm5_next dsa_decode.zig, `decode_dsa: fused`): the sparse latent
// attention of S <= 4 decode-lane query rows over the latent rows their indexer selected, the split half of mlx-serve's
// msv_dec_attn split/merge (MiMo's dec_attn_split.metal / dec_attn_merge.metal, mlx-stream src/arch/mimo_v2/kernels):
// each slice of the selection keeps an online softmax in exp2 units (the scaled score times log2 e) and writes its
// unnormalised output, running max and sum; MiMo's dec_attn_merge text merges the slices. Here the attention is MLA's
// (one latent row [R] is every head's key and value), its keys gathered by index (sel: -1 marks an unused slot, never
// attended) and every head of a group reads the same tile of rows.
// q [H, S, R] (q_abs [1, H, S, R]: the queries folded through each head's key half), lat [T, R] (the latent lane),
// sel [S, W] int32, consts [8] f32 (consts[0]: the attention scale) -> pacc [H * S, NS, R], pm / pl [H * S, NS] f32.
// Grid: x = NS slices x 32 * HG threads, y = H / HG head groups, z = S rows; simdgroup g owns head (group * HG + g), lane
// l the dims [l * R / 32, (l + 1) * R / 32).
constexpr int DPL = R / 32;
const int j = int(threadgroup_position_in_grid.x);
const int h = int(threadgroup_position_in_grid.y) * HG + int(simdgroup_index_in_threadgroup);
const int qs = int(threadgroup_position_in_grid.z);
const int S = int(threadgroups_per_grid.z);
const int NS = int(threadgroups_per_grid.x);
const int lane = int(thread_index_in_simdgroup);
const int tix = int(thread_index_in_threadgroup);
const int W = sel_shape[1];
const int chunk = ((W + NS - 1) / NS + BK - 1) / BK * BK;
const int k0 = j * chunk;
const int k1 = metal::min(W, k0 + chunk);
const float sl2 = consts[0] * 1.44269504088896340736f;
const device T* qr = q + (size_t(h) * S + qs) * R + lane * DPL;
float qv[DPL];
float acc[DPL];
for (int d = 0; d < DPL; ++d) {
  qv[d] = float(qr[d]) * sl2;
  acc[d] = 0.0f;
}
float m = -3.0e38f;
float l = 0.0f;
threadgroup uchar tgm[BK * R * sizeof(T)] __attribute__((aligned(16)));
threadgroup T* ks = (threadgroup T*)tgm;
threadgroup int kid[BK];
const device int* sr = sel + size_t(qs) * W;
for (int c0 = k0; c0 < k1; c0 += BK) {
  const int rows = metal::min(BK, k1 - c0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (R / 8); i += 32 * HG) {
    const int r = i / (R / 8);
    const int c8 = i % (R / 8);
    const int src = r < rows ? sr[c0 + r] : -1;
    *((threadgroup vec<T, 8>*)(ks + r * R + c8 * 8)) = src >= 0 ? *((const device vec<T, 8>*)(lat + size_t(src) * R + c8 * 8)) : vec<T, 8>(0);
  }
  if (tix < BK) {
    kid[tix] = tix < rows ? sr[c0 + tix] : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < rows; ++r) {
    if (kid[r] < 0) {
      continue;
    }
    const threadgroup T* kr = ks + r * R + lane * DPL;
    float s = 0.0f;
    for (int d = 0; d < DPL; ++d) {
      s = fma(qv[d], float(kr[d]), s);
    }
    s = simd_sum(s);
    const float mn = metal::max(m, s);
    const float fac = metal::exp2(m - mn);
    const float p = metal::exp2(s - mn);
    l = l * fac + p;
    for (int d = 0; d < DPL; ++d) {
      acc[d] = fma(p, float(kr[d]), acc[d] * fac);
    }
    m = mn;
  }
}
const size_t pb = (size_t(h) * S + qs) * NS + j;
for (int d = 0; d < DPL; ++d) {
  pacc[pb * R + lane * DPL + d] = acc[d];
}
if (lane == 0) {
  pm[pb] = m;
  pl[pb] = l;
}
