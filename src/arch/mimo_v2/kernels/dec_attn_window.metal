// mimo_dec_attn_window: msv_dec_attn_split (mlx-serve 7a9e392 src/dec_attn.zig SPLIT_SOURCE)
// for MiMo's sliding layers. The ring view holds at most W - 1 + S keys, so one
// split covers it: no partials, no merge dispatch. Changes from the donor text:
// (1) k0 = 0, k1 = N (one split); (2) a lower key bound, the row at view index
// lim attends (lim - W, lim]; (3) the per-head sink, one softmax column with no
// value row, folded once after the key-slice merge (never per slice); (4) the
// normalised row is written as T to out [1, Hq, S, BDV]. MMA, staging, online
// softmax and the slice merge are the donor's lines unchanged.
constexpr int RG = G * S / 8;
constexpr int LDK = BDK + 8;
constexpr int LDV = BDV + 8;
constexpr int NT = 32 * RG * KS;
const int h = int(threadgroup_position_in_grid.y);
const ushort lane = ushort(thread_index_in_simdgroup);
const int sg = int(simdgroup_index_in_threadgroup);
const int rg = sg % RG;
const int ksl = sg / RG;
const int tix = int(thread_index_in_threadgroup);
const int N = k_shape[2];
const short2 sc = msv_coord(lane);
const short sn = sc.x;
const short sm = sc.y;
const int rho = rg * 8 + sm;
const int hq = h * G + rho / S;
const int qs = rho % S;
const int lim = N - S + qs;
const int lo = lim - (W - 1);
const float sl2 = scl[0] * 1.44269504088896340736f;
const device T* Qr = q + (long)hq * q_strides[1] + (long)qs * q_strides[2];
float2 Qf[BDK / 8];
for (int dd = 0; dd < BDK / 8; ++dd) Qf[dd] = float2(float(Qr[dd * 8 + sn]), float(Qr[dd * 8 + sn + 1])) * sl2;
const device T* Kb = k + (long)h * k_strides[1];
const device T* Vb = v + (long)h * v_strides[1];
const long ks = k_strides[2];
const long vs = v_strides[2];
float2 O[BDV / 8];
for (int i = 0; i < BDV / 8; ++i) O[i] = float2(0.0f);
float m = -3.0e38f;
float l = 0.0f;
constexpr int TILE_BYTES = BK * (LDK + LDV) * int(sizeof(T));
constexpr int MERGE_BYTES = (KS - 1) * RG * 8 * BDV * 4;
threadgroup uchar tgm[TILE_BYTES > MERGE_BYTES ? TILE_BYTES : MERGE_BYTES] __attribute__((aligned(16)));
threadgroup T* Ks = (threadgroup T*)tgm;
threadgroup T* Vs = Ks + BK * LDK;
for (int c0 = 0; c0 < N; c0 += BK) {
  const int rows = metal::min(BK, N - c0);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BDK / 8); i += NT) {
    const int r = i / (BDK / 8);
    const int c8 = i % (BDK / 8);
    *((threadgroup vec<T, 8>*)(Ks + r * LDK + c8 * 8)) = r < rows ? *((const device vec<T, 8>*)(Kb + (long)(c0 + r) * ks + c8 * 8)) : vec<T, 8>(0);
  }
  for (int i = tix; i < BK * (BDV / 8); i += NT) {
    const int r = i / (BDV / 8);
    const int c8 = i % (BDV / 8);
    *((threadgroup vec<T, 8>*)(Vs + r * LDV + c8 * 8)) = r < rows ? *((const device vec<T, 8>*)(Vb + (long)(c0 + r) * vs + c8 * 8)) : vec<T, 8>(0);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int kt = ksl; kt < BK / 8; kt += KS) {
    const int kr = kt * 8;
    float2 P = float2(0.0f);
    for (int dd = 0; dd < BDK / 8; ++dd) {
      const threadgroup T* kp = Ks + (kr + sn) * LDK + dd * 8 + sm;
      msv_mma(P, Qf[dd], float2(float(kp[0]), float(kp[LDK])));
    }
    const int kx = c0 + kr + sn;
    P.x = kr + sn < rows && kx <= lim && kx >= lo ? P.x : -INFINITY;
    P.y = kr + sn + 1 < rows && kx + 1 <= lim && kx + 1 >= lo ? P.y : -INFINITY;
    const float mn = metal::max(m, msv_row_max(P));
    const float fac = metal::exp2(m - mn);
    P = metal::exp2(P - mn);
    l = l * fac + msv_row_sum(P);
    m = mn;
    for (int i = 0; i < BDV / 8; ++i) {
      O[i] *= fac;
      const threadgroup T* vp = Vs + (kr + sm) * LDV + i * 8 + sn;
      msv_mma(O[i], P, float2(float(vp[0]), float(vp[1])));
    }
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
threadgroup float* so = (threadgroup float*)tgm;
threadgroup float sml[RG * KS * 8 * 2];
const int slot = (ksl * RG + rg) * 8 + sm;
if (sn == 0) { sml[slot * 2] = m; sml[slot * 2 + 1] = l; }
if (ksl > 0) {
  for (int i = 0; i < BDV / 8; ++i) {
    so[(slot - RG * 8) * BDV + i * 8 + sn] = O[i].x;
    so[(slot - RG * 8) * BDV + i * 8 + sn + 1] = O[i].y;
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (ksl == 0) {
  float M = m;
  for (int t = 1; t < KS; ++t) M = metal::max(M, sml[((t * RG + rg) * 8 + sm) * 2]);
  const float f0 = metal::exp2(m - M);
  float L = l * f0;
  for (int i = 0; i < BDV / 8; ++i) O[i] *= f0;
  for (int t = 1; t < KS; ++t) {
    const int st = (t * RG + rg) * 8 + sm;
    const float ft = metal::exp2(sml[st * 2] - M);
    L += sml[st * 2 + 1] * ft;
    for (int i = 0; i < BDV / 8; ++i) O[i] += ft * float2(so[(st - RG * 8) * BDV + i * 8 + sn], so[(st - RG * 8) * BDV + i * 8 + sn + 1]);
  }
  // The sink: one logit (unscaled, as MLX SDPA takes it) with no value row, once.
  const float sk = float(sinks[(long)hq * sinks_strides[0]]) * 1.44269504088896340736f;
  const float Mx = metal::max(M, sk);
  const float e = metal::exp2(M - Mx);
  const float Z = L * e + metal::exp2(sk - Mx);
  const long ob = ((long)hq * S + qs) * BDV;
  for (int i = 0; i < BDV / 8; ++i) {
    out[ob + i * 8 + sn] = T(e * O[i].x / Z);
    out[ob + i * 8 + sn + 1] = T(e * O[i].y / Z);
  }
}
