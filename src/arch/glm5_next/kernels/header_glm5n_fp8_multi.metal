// glm5n_fp8_multi (glm5next/exl3-k3-launchfusion, 2026-10-14): MiMo's FP8 GEMV text (src/arch/mimo_v2/kernels/
// mimo_fp8_gemv.metal, sha256 69c5f783f59aa9a347375ecd6ab327420ec7f4665b0ef07db29d7e6583dfa33b) as a function of one output
// segment, so one launch serves several projections of the same rows (glm5n_fp8_gemv_kda_in, glm5n_fp8_gemv_pair). Its
// body is the text's code line for line but four lines: the simdgroup and lane indices and the threadgroup memory come
// from the kernel (a threadgroup array is declared at kernel scope), and the segment's threadgroup index `tg` stands for
// threadgroup_position_in_grid.y in n0 (fusion.zig's CPU test checks the lines against the pinned text). So every output
// of a segment is the text's at the segment's (KS, R, RG): per lane an f32 FMA chain of x * (code x scale) over the K
// partition's 256-column steps then the 8 codes, the xor butterfly 16 .. 1, the KS partitions summed in order, one BF16
// rounding; fixed by KS and the text, independent of M, R, RG and of the segment's place in the launch. Needs
// header_fp8res.metal (fp8r_dq4, fp8r_bf_lo / fp8r_bf_hi) before it.
template <int M, int N, int K, int R, int KS, int RG>
static inline void glm5n_fp8_gemv_rows(
    const device bfloat16_t* x,
    const device uint8_t* w,
    const device float* scales,
    device bfloat16_t* y,
    const int tg,
    const int sgi,
    const int lane,
    threadgroup float* part) {
    constexpr int KV = 8;
    constexpr int STEP = 32 * KV;
    constexpr int KP = K / KS;
    constexpr int SC = K / 128;
    static_assert(K % (KS * STEP) == 0, "a K partition is whole 256-column steps");
    static_assert(N % (R * RG) == 0, "N tiles by R * RG rows");
    static_assert(64 % R == 0, "a simdgroup's rows share one 64-row scale row");
    static_assert(M >= 1 && M <= 8, "decode rows");
    const int ks = sgi % KS;
    const int rg = sgi / KS;
    const int n0 = (tg * RG + rg) * R;
    const int k0 = ks * KP + KV * lane;
    const device uint8_t* wb = w + size_t(n0) * size_t(K) + size_t(k0);
    const device float* sb = scales + size_t(n0 / 64) * size_t(SC) + size_t(k0 / 128);
    const device bfloat16_t* xb = x + k0;
    float acc[M][R];
    for (int m = 0; m < M; m++) {
      for (int r = 0; r < R; r++) {
        acc[m][r] = 0.0f;
      }
    }
    for (int st = 0; st < KP / STEP; st++) {
      const float s256 = sb[st * (STEP / 128)] * 256.0f;
      float wv[R][KV];
      for (int r = 0; r < R; r++) {
        const uint2 q = *(const device uint2*)(wb + size_t(r) * size_t(K) + size_t(st * STEP));
        fp8r_dq4(q.x, s256, &wv[r][0]);
        fp8r_dq4(q.y, s256, &wv[r][4]);
      }
      for (int m = 0; m < M; m++) {
        const uint4 q = *(const device uint4*)(xb + size_t(m) * size_t(K) + size_t(st * STEP));
        float xv[KV];
        xv[0] = fp8r_bf_lo(q.x);
        xv[1] = fp8r_bf_hi(q.x);
        xv[2] = fp8r_bf_lo(q.y);
        xv[3] = fp8r_bf_hi(q.y);
        xv[4] = fp8r_bf_lo(q.z);
        xv[5] = fp8r_bf_hi(q.z);
        xv[6] = fp8r_bf_lo(q.w);
        xv[7] = fp8r_bf_hi(q.w);
        for (int r = 0; r < R; r++) {
          for (int i = 0; i < KV; i++) {
            acc[m][r] = fma(xv[i], wv[r][i], acc[m][r]);
          }
        }
      }
    }
    for (int m = 0; m < M; m++) {
      for (int r = 0; r < R; r++) {
        float v = acc[m][r];
        v += simd_shuffle_xor(v, ushort(16));
        v += simd_shuffle_xor(v, ushort(8));
        v += simd_shuffle_xor(v, ushort(4));
        v += simd_shuffle_xor(v, ushort(2));
        v += simd_shuffle_xor(v, ushort(1));
        acc[m][r] = v;
      }
    }
    if constexpr (KS == 1) {
      if (lane == 0) {
        for (int m = 0; m < M; m++) {
          for (int r = 0; r < R; r++) {
            y[size_t(m) * size_t(N) + size_t(n0 + r)] = static_cast<bfloat16_t>(acc[m][r]);
          }
        }
      }
    } else {
      if (lane == 0) {
        for (int m = 0; m < M; m++) {
          for (int r = 0; r < R; r++) {
            part[((rg * KS + ks) * M + m) * R + r] = acc[m][r];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks == 0 && lane == 0) {
        for (int m = 0; m < M; m++) {
          for (int r = 0; r < R; r++) {
            float t = part[((rg * KS) * M + m) * R + r];
            for (int p = 1; p < KS; p++) {
              t += part[((rg * KS + p) * M + m) * R + r];
            }
            y[size_t(m) * size_t(N) + size_t(n0 + r)] = static_cast<bfloat16_t>(t);
          }
        }
      }
    }
}
