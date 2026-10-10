const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
uint mem[MAXM];
const int nm = mimo_k1_members<MAXM>(ids, threads_per_grid.z, e, lane, mem);
if (nm == 0) return;
const uint id = ids[e];
const uint b = mimo_slot_bank(id);
const device uint32_t* wd = b == 0u ? d0 : (b == 1u ? d1 : d2);
const device uint8_t* sd = b == 0u ? ds0 : (b == 1u ? ds1 : ds2);
constexpr int KW2 = I / 16;
constexpr int KG = I / GS;
constexpr int NC = I / 512;
constexpr int CW2 = 512 / 16;
constexpr int CG = 512 / GS;
const size_t row = mimo_slot_row(id) * H + row0;
const device uint2* dw = (const device uint2*)wd + row * KW2 + lane;
const device uint8_t* dsc = sd + row * KG + (lane * 16) / GS;
float acc[MAXM][ROWS];
#pragma unroll
for (int m = 0; m < MAXM; ++m)
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.0f;
uint2 cd[ROWS];
uint cs[ROWS];
#pragma unroll
for (int r = 0; r < ROWS; ++r) {
  cd[r] = dw[r * KW2];
  cs[r] = uint(dsc[r * KG]);
}
for (int c = 0; c < NC; ++c) {
  uint2 nd[ROWS];
  uint ns[ROWS];
  if (c + 1 < NC) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      nd[r] = dw[r * KW2 + (c + 1) * CW2];
      ns[r] = uint(dsc[r * KG + (c + 1) * CG]);
    }
  }
  const int kk = c * 512 + int(lane) * 16;
  if (MAXM == 1) {
    float xf[16];
    mimo_k1_x16(act + size_t(mem[0]) * I + kk, xf);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      half2 hd[8];
      mimo_k1_decode16(cd[r], hd);
      if (FMA == 0) {
        acc[0][r] += mimo_k1_dot16(hd, xf) * mimo_fp4_scale(cs[r]);
      } else {
        acc[0][r] = fma(mimo_k1_dot16_fma(hd, xf), mimo_fp4_scale(cs[r]), acc[0][r]);
      }
    }
  } else {
    half2 hd[ROWS][8];
    float sdr[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      mimo_k1_decode16(cd[r], hd[r]);
      sdr[r] = mimo_fp4_scale(cs[r]);
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m < nm) {
        float xf[16];
        mimo_k1_x16(act + size_t(mem[m]) * I + kk, xf);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          if (FMA == 0) {
            acc[m][r] += mimo_k1_dot16(hd[r], xf) * sdr[r];
          } else {
            acc[m][r] = fma(mimo_k1_dot16_fma(hd[r], xf), sdr[r], acc[m][r]);
          }
        }
      }
    }
  }
  if (c + 1 < NC) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      cd[r] = nd[r];
      cs[r] = ns[r];
    }
  }
}
#pragma unroll
for (int m = 0; m < MAXM; ++m) {
  if (m < nm) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float v = simd_sum(acc[m][r]) * 4194304.0f;
      if (lane == 0) y[size_t(mem[m]) * H + row0 + r] = T(v);
    }
  }
}
