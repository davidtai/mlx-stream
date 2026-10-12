const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
uint mem[MAXM];
const int nm = mimo_union_members<MAXM>(ids, uint(ids_shape[0]), e, mem);
if (nm == 0) return;
const uint id = ids[e];
const uint b = mimo_slot_bank(id);
const device uint32_t* wd = b == 0u ? d0 : (b == 1u ? d1 : d2);
const device uint8_t* sd = b == 0u ? ds0 : (b == 1u ? ds1 : ds2);
constexpr int KW2 = I / 16;
constexpr int KG = I / GS;
const size_t row = mimo_slot_row(id) * H + row0;
const device uint2* dw = (const device uint2*)wd + row * KW2 + lane;
float acc[MAXM][ROWS];
for (int m = 0; m < MAXM; ++m)
  for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.0f;
for (int k = 0; k < I; k += 512) {
  const int kk = k + int(lane) * 16;
  const int gi = kk / GS;
  for (int r = 0; r < ROWS; ++r) {
    const float sdr = mimo_fp4_scale(uint(sd[(row + r) * KG + gi]));
    half wdr[16];
    mimo_fp4_decode16(dw[r * KW2 + k / 16], wdr);
    for (int m = 0; m < MAXM; ++m) {
      if (m < nm) {
        const device auto* at = act + size_t(mem[m]) * I;
        float xr[16];
        for (int i = 0; i < 16; ++i) xr[i] = float(at[kk + i]);
        acc[m][r] += mimo_fp4_dot16h(wdr, xr) * sdr;
      }
    }
  }
}
for (int m = 0; m < MAXM; ++m) {
  if (m < nm) {
    for (int r = 0; r < ROWS; ++r) {
      const float v = simd_sum(acc[m][r]) * 4194304.0f;
      if (lane == 0) y[size_t(mem[m]) * H + row0 + r] = T(v);
    }
  }
}
