
    // GLM-5.3 EXL3 prompt, stage 2: the gate and up GEMMs of a wave's experts at a 128-row M tile (8 simdgroups, WM 4
    // x WN 2, each a 32 x 32 share of the 128 x 64 tile), each BK = 64 stage's trellis tiles decoded into threadgroup
    // memory by the 256 threads (one 16-weight column each), then the NAX MMA. A threadgroup is (expert j of the wave
    // table, M tile, op = 2 mini + projection, N tile): the table gives each expert's slot, first row, rows and first
    // threadgroup. a: [rows, 4, HID] f16 per projection (the prepared inputs, rows grouped by expert); z: [rows, 4,
    // MINI] f32.
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WN = 2;
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int LDB = BK + 8;
    constexpr int KT = BK / 16;
    constexpr int NT = BN / 16;
    constexpr int NWT = 8 * K;
    constexpr int K16 = HID / 16;
    constexpr int N16 = MINI / 16;
    constexpr int NTILES = MINI / BN;
    constexpr int S = HID / BK;
    constexpr int LDA = 4 * HID;
    constexpr int LDZ = 4 * MINI;
    threadgroup uint4 smem[BN * LDB * 2 / 16];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(smem);

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = tid & 31u;
    const int g = int(threadgroup_position_in_grid.x);
    int j = 0;
    STEEL_PRAGMA_UNROLL
    for (int q = 1; q < 16; ++q) {
        j = tbl[48 + q] <= g ? q : j;
    }
    const int slot = tbl[j];
    const int row0 = tbl[16 + j];
    const int rows = tbl[32 + j];
    const int local = g - tbl[48 + j];
    const int tiles_m = (rows + BM - 1) / BM;
    const int nt_all = local / tiles_m;
    const int m0 = (local - nt_all * tiles_m) * BM;
    const int op = nt_all / NTILES;
    const int n0 = (nt_all - op * NTILES) * BN;
    const int r = op >> 1;
    const int proj = op & 1;
    const int tm = SM * int(sg / WN);
    const short tn = short(SN * int(sg % WN));
    const int sgp_sm = min(SM, max(0, min(BM, rows - m0) - tm));
    // the loader: job = trellis tile t (kt, nt) of the stage, column half h = sg & 1, column c7 = lane & 7 -> B^T row
    // 16 nt + 8 h + c7, k slots 16 kt .. 16 kt + 15
    const uint h = sg & 1u;
    const uint c7 = lane & 7u;
    const int t = int(lane >> 3u) + 4 * int(sg >> 1u);
    const int kt = t / NT;
    const int nt = t - kt * NT;
    const int dst_off = (nt * 16 + 8 * int(h) + int(c7)) * LDB + kt * 16;
    const int src_off = (kt * N16 + nt) * NWT;
    uint o[5];
    if (h != 0u) {
        g53_col<K, 1>::offsets(c7, o);
    } else {
        g53_col<K, 0>::offsets(c7, o);
    }
    const device ushort* code = proj == 0 ? cg : cu;
    const device uint* cd = reinterpret_cast<const device uint*>(code)
        + (size_t(slot) * 4u + size_t(r)) * size_t(K16 * N16) * size_t(NWT) + size_t(n0 / 16) * size_t(NWT);
    const device half* xa = (proj == 0 ? ag : au) + (size_t(row0) + size_t(m0 + tm)) * size_t(LDA) + size_t(r) * size_t(HID);
    const bool full = sgp_sm == SM;
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    for (int s = 0; s < S; ++s) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device uint* tile = cd + size_t(s) * size_t(KT * N16 * NWT) + src_off;
        if (h != 0u) {
            g53_col<K, 1>::template decode<half>(tile, o, Bs + dst_off);
        } else {
            g53_col<K, 0>::template decode<half>(tile, o, Bs + dst_off);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgp_sm > 0) {
            STEEL_PRAGMA_NO_UNROLL
            for (int i = 0; i < 2; ++i) {
                NAXTile<half, TM, TK> Atile;
                NAXTile<half, TN, TK> Btile;
                volatile int compiler_barrier;
                if (full) {
                    Atile.load(xa + s * BK + i * SK, LDA);
                } else {
                    Atile.load_safe(xa + s * BK + i * SK, LDA, short2(SK, sgp_sm));
                }
                Btile.template load<half, LDB, 1>(Bs + tn * LDB + i * SK);
                tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
            }
        }
    }
    if (sgp_sm > 0) {
        device float* zp = (proj == 0 ? zg : zu) + (size_t(row0) + size_t(m0 + tm)) * size_t(LDZ) + size_t(r) * size_t(MINI) + n0 + tn;
        if (full) {
            Dtile.store(zp, LDZ);
        } else {
            Dtile.store_safe(zp, LDZ, short2(SN, sgp_sm));
        }
    }
