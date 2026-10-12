const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
uint mem[MAXM];
const int nm = mimo_k1_members<MAXM>(ids, threads_per_grid.z, e, lane, mem);
if (nm == 0) return;
const uint id = ids[e];
const uint b = mimo_slot_bank(id);
const device uint32_t* wg = b == 0u ? g0 : (b == 1u ? g1 : g2);
const device uint8_t* sg_ = b == 0u ? gs0 : (b == 1u ? gs1 : gs2);
const device uint32_t* wu = b == 0u ? u0 : (b == 1u ? u1 : u2);
const device uint8_t* su_ = b == 0u ? us0 : (b == 1u ? us1 : us2);
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
constexpr int NC = K / 512;
constexpr int CW2 = 512 / 16;
constexpr int CG = 512 / GS;
const size_t row = mimo_slot_row(id) * N + row0;
const device uint2* gw = (const device uint2*)wg + row * KW2 + lane;
const device uint2* uw = (const device uint2*)wu + row * KW2 + lane;
const device uint8_t* gsc = sg_ + row * KG + (lane * 16) / GS;
const device uint8_t* usc = su_ + row * KG + (lane * 16) / GS;
float ag[MAXM][ROWS];
float au[MAXM][ROWS];
#pragma unroll
for (int m = 0; m < MAXM; ++m)
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    ag[m][r] = 0.0f;
    au[m][r] = 0.0f;
  }
uint2 cg[ROWS];
uint2 cu[ROWS];
uint cs[ROWS];
uint ct[ROWS];
#pragma unroll
for (int r = 0; r < ROWS; ++r) {
  cg[r] = gw[r * KW2];
  cu[r] = uw[r * KW2];
  cs[r] = uint(gsc[r * KG]);
  ct[r] = uint(usc[r * KG]);
}
for (int c = 0; c < NC; ++c) {
  uint2 ng[ROWS];
  uint2 nu[ROWS];
  uint ns[ROWS];
  uint nt[ROWS];
  if (c + 1 < NC) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      ng[r] = gw[r * KW2 + (c + 1) * CW2];
      nu[r] = uw[r * KW2 + (c + 1) * CW2];
      ns[r] = uint(gsc[r * KG + (c + 1) * CG]);
      nt[r] = uint(usc[r * KG + (c + 1) * CG]);
    }
  }
  const int kk = c * 512 + int(lane) * 16;
  if (MAXM == 1) {
    float xf[16];
    mimo_k1_x16(x + size_t(mem[0]) * K + kk, xf);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      half2 hg[8];
      half2 hu[8];
      mimo_k1_decode16(cg[r], hg);
      mimo_k1_decode16(cu[r], hu);
      if (FMA == 0) {
        ag[0][r] += mimo_k1_dot16(hg, xf) * mimo_fp4_scale(cs[r]);
        au[0][r] += mimo_k1_dot16(hu, xf) * mimo_fp4_scale(ct[r]);
      } else {
        ag[0][r] = fma(mimo_k1_dot16_fma(hg, xf), mimo_fp4_scale(cs[r]), ag[0][r]);
        au[0][r] = fma(mimo_k1_dot16_fma(hu, xf), mimo_fp4_scale(ct[r]), au[0][r]);
      }
    }
  } else {
    half2 hg[ROWS][8];
    half2 hu[ROWS][8];
    float sgr[ROWS];
    float sur[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      mimo_k1_decode16(cg[r], hg[r]);
      mimo_k1_decode16(cu[r], hu[r]);
      sgr[r] = mimo_fp4_scale(cs[r]);
      sur[r] = mimo_fp4_scale(ct[r]);
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m < nm) {
        float xf[16];
        mimo_k1_x16(x + size_t(mem[m]) * K + kk, xf);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          if (FMA == 0) {
            ag[m][r] += mimo_k1_dot16(hg[r], xf) * sgr[r];
            au[m][r] += mimo_k1_dot16(hu[r], xf) * sur[r];
          } else {
            ag[m][r] = fma(mimo_k1_dot16_fma(hg[r], xf), sgr[r], ag[m][r]);
            au[m][r] = fma(mimo_k1_dot16_fma(hu[r], xf), sur[r], au[m][r]);
          }
        }
      }
    }
  }
  if (c + 1 < NC) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      cg[r] = ng[r];
      cu[r] = nu[r];
      cs[r] = ns[r];
      ct[r] = nt[r];
    }
  }
}
#pragma unroll
for (int m = 0; m < MAXM; ++m) {
  if (m < nm) {
    #pragma unroll
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
