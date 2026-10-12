const uint lane = thread_index_in_simdgroup;
const int row0 = int(thread_position_in_grid.y) * ROWS;
const uint e = thread_position_in_grid.z;
const uint id = ids[e];
const uint b = mimo_slot_bank(id);
const device uint32_t* wg = b == 0u ? g0 : (b == 1u ? g1 : g2);
const device uint8_t* sg_ = b == 0u ? gs0 : (b == 1u ? gs1 : gs2);
const device uint32_t* wu = b == 0u ? u0 : (b == 1u ? u1 : u2);
const device uint8_t* su_ = b == 0u ? us0 : (b == 1u ? us1 : us2);
const device auto* xt = x + size_t(e) * K;
constexpr int KW2 = K / 16;
constexpr int KG = K / GS;
const size_t row = mimo_slot_row(id) * N + row0;
const device uint2* gw = (const device uint2*)wg + row * KW2 + lane;
const device uint2* uw = (const device uint2*)wu + row * KW2 + lane;
float ag[ROWS] = {0.0f};
float au[ROWS] = {0.0f};
for (int k = 0; k < K; k += 512) {
  const int kk = k + int(lane) * 16;
  float xr[16];
  for (int i = 0; i < 16; ++i) xr[i] = float(xt[kk + i]);
  const int gi = kk / GS;
  for (int r = 0; r < ROWS; ++r) {
    const size_t si = (row + r) * KG + gi;
    ag[r] += mimo_fp4_dot16(gw[r * KW2 + k / 16], xr) * mimo_fp4_scale(uint(sg_[si]));
    au[r] += mimo_fp4_dot16(uw[r * KW2 + k / 16], xr) * mimo_fp4_scale(uint(su_[si]));
  }
}
for (int r = 0; r < ROWS; ++r) {
  const float g = simd_sum(ag[r]) * 4194304.0f;
  const float u = simd_sum(au[r]) * 4194304.0f;
  if (lane == 0) {
    const float gf = float(T(g));
    y[size_t(e) * N + row0 + r] = T(gf * mimo_sigmoid_f32(gf)) * T(u);
  }
}
