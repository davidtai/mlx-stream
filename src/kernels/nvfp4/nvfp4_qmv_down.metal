const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
uint mem[MAXM];
const int nm = glm5n_members<MAXM>(ids, threads_per_grid.z, e, lane, mem);
if (nm == 0) return;
const uint id = ids[e];
const uint b = glm5n_slot_bank(id);
const size_t slot = glm5n_slot_row(id);
const device uint32_t* wd = b == 0u ? down_w0 : (b == 1u ? down_w1 : down_w2);
const device uint8_t* sd = b == 0u ? down_s0 : (b == 1u ? down_s1 : down_s2);
// fp_qmv_fast_impl's `*global_scale / (F8E4M3_MAX * F4E2M1_MAX)`, read by value from the slot's own line (glm5n_g_at)
const float dinv = (b == 0u ? float(down_g0[glm5n_g_at(slot)]) : (b == 1u ? float(down_g1[glm5n_g_at(slot)]) : float(down_g2[glm5n_g_at(slot)]))) / 2688.0f;
constexpr int KW2 = I / 16;
constexpr int KG = I / GS;
constexpr int NC = I / 512;
constexpr int CW2 = 512 / 16;
constexpr int CG = 512 / GS;
const size_t row = slot * H + row0;
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
    glm5n_x16(act + size_t(mem[0]) * I + kk, xf);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      half dd[16];
      nvq_decode16(cd[r], dd);
      acc[0][r] += nvq_dot16(dd, xf, glm5n_e4m3(cs[r]));
    }
  } else {
    half dd[ROWS][16];
    float sdr[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      nvq_decode16(cd[r], dd[r]);
      sdr[r] = glm5n_e4m3(cs[r]);
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m < nm) {
        float xf[16];
        glm5n_x16(act + size_t(mem[m]) * I + kk, xf);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[m][r] += nvq_dot16(dd[r], xf, sdr[r]);
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
// The stock route's down row: simd_sum, times g / 2688, rounded to T.
#pragma unroll
for (int m = 0; m < MAXM; ++m) {
  if (m < nm) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float v = simd_sum(acc[m][r]);
      if (lane == 0) y[size_t(mem[m]) * H + row0 + r] = T(v * dinv);
    }
  }
}
