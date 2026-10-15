// glm5n_dsa_sparse_split_lean (`dsa_split_form: lean`, glm5_next dsa_decode.zig): glm5n_dsa_sparse_split.metal's split with
// the same floating-point operations in the same order for every output (pacc, pm, pl bit for bit), arranged for fewer
// instructions and more independent work:
//   - each tile's keys in groups of GK: the GK scores (each the same per-lane fma chain and simd_sum) first, then the
//     online-softmax update of each key of the group in key order (the update reads only its own key's score);
//   - a key whose factor exp2(m - mn) is exactly 1 adds p * k to the accumulator directly (acc * 1.0f == acc: the same
//     fma operands); every other key runs the split's update as written;
//   - HG = 16 heads a threadgroup (one simdgroup each) share each gathered tile (the split's keys, slices and per-head
//     work unchanged: only how many simdgroups read a tile).
// q [H, S, R], lat [T, R], sel [S, W] int32, consts [8] f32 (consts[0]: the attention scale) -> pacc [H * S, NS, R],
// pm / pl [H * S, NS] f32. Grid: x = NS slices x 32 * HG threads, y = H / HG head groups, z = S rows.
constexpr int DPL = R / 32;
constexpr int GK = 4;
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
  for (int r0 = 0; r0 < rows; r0 += GK) {
    float sc[GK];
    for (int u = 0; u < GK; ++u) {
      const threadgroup T* kr = ks + (r0 + u) * R + lane * DPL;
      float s = 0.0f;
      for (int d = 0; d < DPL; ++d) {
        s = fma(qv[d], float(kr[d]), s);
      }
      sc[u] = simd_sum(s);
    }
    for (int u = 0; u < GK; ++u) {
      const int r = r0 + u;
      if (r >= rows || kid[r] < 0) {
        continue;
      }
      const threadgroup T* kr = ks + r * R + lane * DPL;
      const float s = sc[u];
      const float mn = metal::max(m, s);
      const float fac = metal::exp2(m - mn);
      const float p = metal::exp2(s - mn);
      l = l * fac + p;
      if (fac == 1.0f) {
        for (int d = 0; d < DPL; ++d) {
          acc[d] = fma(p, float(kr[d]), acc[d]);
        }
      } else {
        for (int d = 0; d < DPL; ++d) {
          acc[d] = fma(p, float(kr[d]), acc[d] * fac);
        }
      }
      m = mn;
    }
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
