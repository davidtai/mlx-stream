    // mimo_fp8_gemm_tg (fp8res, a microbench candidate; not installed): MLX's fp-quantized NAX matmul structure
    // (fp_quantized_nax.h `fp_qmm_t_impl`: the threadgroup's weight tile decoded once into threadgroup memory, every
    // simdgroup's B fragments read from it) with the release's e4m3 codes decoded to bfloat code / 256 (exact) and each
    // 128-column block's f32 sum promoted by its scale x 256, as mimo_fp8_gemm_nax. The decode runs once per threadgroup
    // (1 / BM per output) instead of once per simdgroup (1 / SM).
    // Accumulation order of every output: per 128-column block (ascending), the tensor unit's f32 sum of the exact
    // products over the block's two BK = 64 tiles, then fma(block, scale x 256, acc); one BF16 rounding at the store.
    using namespace omlx_nax;
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short BK = 64;
    constexpr short BKP = BK + 8;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;
    constexpr int THREADS = WM * WN * 32;
    constexpr int WORDS = BN * BK / 4;
    static_assert(SN == 32 && SM % 16 == 0 && N % BN == 0 && K % 128 == 0 && WORDS % THREADS == 0, "fp8res TG GEMM geometry");
    threadgroup bfloat16_t Ws[BN * BKP];
    const int M = x_shape[0];
    const int tiles_m = (M + BM - 1) / BM;
    const int tid_y = (int(threadgroup_position_in_grid.y) << SWZ) + (int(threadgroup_position_in_grid.x) & ((1 << SWZ) - 1));
    const int tid_x = int(threadgroup_position_in_grid.x) >> SWZ;
    if (tid_x >= N / BN || tid_y >= tiles_m) {
      return;
    }
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lid = int(thread_index_in_threadgroup);
    const int tn = SN * (sg % WN);
    const int row0 = tid_y * BM + SM * (sg / WN);
    const int col0 = tid_x * BN + tn;
    const int rows = min(int(SM), M - row0);
    const device bfloat16_t* A = x + size_t(row0) * size_t(K);
    const device uint8_t* Wt = w + size_t(tid_x * BN) * size_t(K);
    const device float* S = scales + size_t(col0 / 64) * size_t(K / 128);
    NAXTile<float, TM, TN> D;
    D.clear();
    NAXTile<float, TM, TN> P;
    P.clear();
    for (int k0 = 0; k0 < K; k0 += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = lid; i < WORDS; i += THREADS) {
        const int r = i / (BK / 4);
        const int c = (i % (BK / 4)) * 4;
        bfloat16_t v[4];
        fp8r_bf4(*(const device uint*)(Wt + size_t(r) * size_t(K) + size_t(k0 + c)), v);
        for (short j = 0; j < 4; j++) {
          Ws[r * BKP + c + j] = v[j];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (rows > 0) {
        for (int kk = 0; kk < BK; kk += SK) {
          NAXTile<bfloat16_t, TM, TK> At;
          NAXTile<bfloat16_t, TN, TK> Bt;
          if (rows == SM) {
            At.load(A + k0 + kk, K);
          } else {
            At.load_safe(A + k0 + kk, K, short2(SK, rows));
          }
          Bt.template load<bfloat16_t, BKP, 1>(Ws + tn * BKP + kk);
          fp8r_nax::matmad<RELAXED>(P, At, Bt);
        }
        if ((k0 % 128) == 128 - BK) {
          const float s256 = S[k0 / 128] * 256.0f;
          thread float* d = D.elems();
          thread float* p = P.elems();
          for (short i = 0; i < NAXTile<float, TM, TN>::kElemsPerTile; i++) {
            d[i] = fma(p[i], s256, d[i]);
            p[i] = 0.0f;
          }
        }
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
