    // mimo_fp8_gemm_nax (fp8res): y[m, n] = sum_kb (s[n / 64, kb] * 256) * (sum_{k in block kb} x[m, k] * (e4m3(w[n, k]) / 256))
    // for any M rows (the prompt path, M > 8), BF16 x / y. Threadgroup tile BM x BN, WM x WN simdgroups of SM x SN
    // (SN = 32: one 64-row scale row per simdgroup), MLX's swizzle (SWZ), 32-wide k steps.
    // Accumulation order of every output (m, n), independent of M and of the row's tile position:
    //   per 128-column block kb (ascending): P = the tensor unit's f32 sum of the block's exact products x * code / 256
    //   (4 k steps x 2 fragments, MMA mode RELAXED); D = fma(P, scale * 256, D); one rounding to BF16 at the store.
    using namespace omlx_nax;
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;
    static_assert(SN == 32 && SM % 16 == 0 && N % BN == 0 && K % 128 == 0, "fp8res GEMM geometry");
    const int M = x_shape[0];
    const int tiles_m = (M + BM - 1) / BM;
    const int tid_y = (int(threadgroup_position_in_grid.y) << SWZ) + (int(threadgroup_position_in_grid.x) & ((1 << SWZ) - 1));
    const int tid_x = int(threadgroup_position_in_grid.x) >> SWZ;
    if (tid_x >= N / BN || tid_y >= tiles_m) {
      return;
    }
    const int sg = int(simdgroup_index_in_threadgroup);
    const int row0 = tid_y * BM + SM * (sg / WN);
    const int col0 = tid_x * BN + SN * (sg % WN);
    const int rows = min(int(SM), M - row0);
    const device bfloat16_t* A = x + size_t(row0) * size_t(K);
    const device uint8_t* B = w + size_t(col0) * size_t(K);
    const device float* S = scales + size_t(col0 / 64) * size_t(K / 128);
    NAXTile<float, TM, TN> D;
    D.clear();
    for (int kb = 0; kb < K / 128; kb++) {
      threadgroup_barrier(mem_flags::mem_none);
      if (rows <= 0) {
        continue;
      }
      NAXTile<float, TM, TN> P;
      P.clear();
      for (int kk = 0; kk < 128; kk += SK) {
        NAXTile<bfloat16_t, TM, TK> At;
        NAXTile<bfloat16_t, TN, TK> Bt;
        if (rows == SM) {
          At.load(A + kb * 128 + kk, K);
        } else {
          At.load_safe(A + kb * 128 + kk, K, short2(SK, rows));
        }
        fp8r_nax::load_codes(Bt, B + kb * 128 + kk, K);
        fp8r_nax::matmad<RELAXED>(P, At, Bt);
      }
      const float s256 = S[kb] * 256.0f;
      thread float* d = D.elems();
      thread const float* p = P.elems();
      for (short i = 0; i < NAXTile<float, TM, TN>::kElemsPerTile; i++) {
        d[i] = fma(p[i], s256, d[i]);
      }
    }
    if (rows <= 0) {
      return;
    }
    device bfloat16_t* Y = y + size_t(row0) * size_t(N) + size_t(col0);
    if (rows == SM) {
      D.store(Y, N);
    } else {
      D.store_safe(Y, N, short2(SN, rows));
    }
