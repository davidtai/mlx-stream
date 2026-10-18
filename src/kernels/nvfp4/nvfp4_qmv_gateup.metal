const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
uint mem[MAXM];
const int nm = glm5n_members<MAXM>(ids, threads_per_grid.z, e, lane, mem);
if (nm == 0) return;
const uint id = ids[e];
const uint b = glm5n_slot_bank(id);
const size_t slot = glm5n_slot_row(id);
const device uint32_t* wg = b == 0u ? gate_w0 : (b == 1u ? gate_w1 : gate_w2);
const device uint8_t* sg_ = b == 0u ? gate_s0 : (b == 1u ? gate_s1 : gate_s2);
const device uint32_t* wu = b == 0u ? up_w0 : (b == 1u ? up_w1 : up_w2);
const device uint8_t* su_ = b == 0u ? up_s0 : (b == 1u ? up_s1 : up_s2);
// fp_qmv_fast_impl's `*global_scale / (F8E4M3_MAX * F4E2M1_MAX)`, read by value from the slot's own line (glm5n_g_at)
const float ginv = (b == 0u ? float(gate_g0[glm5n_g_at(slot)]) : (b == 1u ? float(gate_g1[glm5n_g_at(slot)]) : float(gate_g2[glm5n_g_at(slot)]))) / 2688.0f;
const float uinv = (b == 0u ? float(up_g0[glm5n_g_at(slot)]) : (b == 1u ? float(up_g1[glm5n_g_at(slot)]) : float(up_g2[glm5n_g_at(slot)]))) / 2688.0f;
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
constexpr int NC = K / 512;
constexpr int CW2 = 512 / 16;
constexpr int CG = 512 / GS;
const size_t row = slot * N + row0;
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
    glm5n_x16(x + size_t(mem[0]) * K + kk, xf);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      half dg[16];
      half du[16];
      nvq_decode16(cg[r], dg);
      nvq_decode16(cu[r], du);
      ag[0][r] += nvq_dot16(dg, xf, glm5n_e4m3(cs[r]));
      au[0][r] += nvq_dot16(du, xf, glm5n_e4m3(ct[r]));
    }
  } else {
    half dg[ROWS][16];
    half du[ROWS][16];
    float sgr[ROWS];
    float sur[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      nvq_decode16(cg[r], dg[r]);
      nvq_decode16(cu[r], du[r]);
      sgr[r] = glm5n_e4m3(cs[r]);
      sur[r] = glm5n_e4m3(ct[r]);
    }
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m < nm) {
        float xf[16];
        glm5n_x16(x + size_t(mem[m]) * K + kk, xf);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          ag[m][r] += nvq_dot16(dg[r], xf, sgr[r]);
          au[m][r] += nvq_dot16(du[r], xf, sur[r]);
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
// The stock route's gate and up rows (simd_sum, times g / 2688, rounded to T), then its f32 SwiGLU
// (sdk_ext.quant.Activation.swiglu_f32: silu(f32(gate)) * f32(up), MLX's f32 sigmoid, rounded to T once).
#pragma unroll
for (int m = 0; m < MAXM; ++m) {
  if (m < nm) {
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float g = simd_sum(ag[m][r]);
      const float u = simd_sum(au[m][r]);
      if (lane == 0) {
        const float gf = float(T(g * ginv));
        const float uf = float(T(u * uinv));
        y[size_t(mem[m]) * N + row0 + r] = T((gf * glm5n_sigmoid_f32(gf)) * uf);
      }
    }
  }
}
