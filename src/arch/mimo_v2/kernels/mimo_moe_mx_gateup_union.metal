const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
uint mem[MAXM];
const int nm = mimo_union_members<MAXM>(ids, uint(ids_shape[0]), e, mem);
if (nm == 0) return;
const uint id = ids[e];
const uint b = mimo_slot_bank(id);
const device uint32_t* wg = b == 0u ? g0 : (b == 1u ? g1 : g2);
const device uint8_t* sg_ = b == 0u ? gs0 : (b == 1u ? gs1 : gs2);
const device uint32_t* wu = b == 0u ? u0 : (b == 1u ? u1 : u2);
const device uint8_t* su_ = b == 0u ? us0 : (b == 1u ? us1 : us2);
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
const size_t row = mimo_slot_row(id) * N + row0;
const device uint2* gw = (const device uint2*)wg + row * KW2 + lane;
const device uint2* uw = (const device uint2*)wu + row * KW2 + lane;
float ag[MAXM][ROWS];
float au[MAXM][ROWS];
for (int m = 0; m < MAXM; ++m)
  for (int r = 0; r < ROWS; ++r) {
    ag[m][r] = 0.0f;
    au[m][r] = 0.0f;
  }
for (int k = 0; k < K; k += 512) {
  const int kk = k + int(lane) * 16;
  const int gi = kk / GS;
  for (int r = 0; r < ROWS; ++r) {
    const size_t si = (row + r) * KG + gi;
    const float sgr = mimo_fp4_scale(uint(sg_[si]));
    const float sur = mimo_fp4_scale(uint(su_[si]));
    half wgr[16];
    half wur[16];
    mimo_fp4_decode16(gw[r * KW2 + k / 16], wgr);
    mimo_fp4_decode16(uw[r * KW2 + k / 16], wur);
    for (int m = 0; m < MAXM; ++m) {
      if (m < nm) {
        const device auto* xt = x + size_t(mem[m]) * K;
        float xr[16];
        for (int i = 0; i < 16; ++i) xr[i] = float(xt[kk + i]);
        ag[m][r] += mimo_fp4_dot16h(wgr, xr) * sgr;
        au[m][r] += mimo_fp4_dot16h(wur, xr) * sur;
      }
    }
  }
}
for (int m = 0; m < MAXM; ++m) {
  if (m < nm) {
    for (int r = 0; r < ROWS; ++r) {
      const float g = simd_sum(ag[m][r]) * 4194304.0f;
      const float u = simd_sum(au[m][r]) * 4194304.0f;
      if (lane == 0) {
        const float gf = float(T(g));
        y[size_t(mem[m]) * N + row0 + r] = T(gf * mimo_sigmoid_f32(gf)) * T(u);
      }
    }
  }
}
