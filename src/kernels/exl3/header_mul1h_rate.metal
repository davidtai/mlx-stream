
// A circular 16-bit window ending at (position + 1) K; words retain EXL3's bit order.
template <int K>
METAL_FUNC uint rate_window(const device uint* tile, uint position) {
    constexpr uint words = 8u * K;
    const uint end = (position + 1u) * K;
    const uint wi = (end - 1u) / 32u;
    const uint used = (end - 1u) % 32u + 1u;
    const uint lo = tile[wi % words];
    if (used >= 16u) return (lo >> (32u - used)) & 0xffffu;
    const uint hi = tile[(wi + words - 1u) % words];
    return ((lo >> (32u - used)) | (hi << used)) & 0xffffu;
}

template <int K, int H>
METAL_FUNC void rate_dec_job(const device uint* tile, uint c7, threadgroup half* dst) {
    if constexpr (K == 3) {
        dig_dec_job<half, H>(tile, (3u * c7 + 23u) % 24u,
            (3u * c7 + 24u) % 24u, (3u * c7 + 25u) % 24u,
            (3u * c7 + 26u) % 24u, dst);
    } else {
        STEEL_PRAGMA_UNROLL
        for (uint r = 0; r < 16u; ++r) {
            const uint lane = 4u * c7 + (r & 7u) / 2u;
            const uint p = 8u * lane + (r & 1u) + (r >= 8u ? 2u : 0u) + 4u * H;
            dst[r] = dig_cb(rate_window<K>(tile, p));
        }
    }
}

// ---- decode lane windows (compact/rate GEMV) ----
// Decode GEMV lane `lane` owns positions p = 8 lane + 7 - m (m = 0..7) of every 16 x 16 tile.  Window m ends at stream
// bit E - m K with E = 8 K (lane + 1), so the lane's 8 windows are 16-bit slices of the 7 K + 16 stream bits ending at
// E (30 / 37 / 44 / 51 bits for K = 2 / 3 / 4 / 5): 2 u32 words, 3 for K = 5.  With a = the word holding stream bit
// E - 1, b and c the two words before it (circular over the 8 K tile words; only lane 0 wraps) and r = 31 - (E - 1) % 32
// the bits of a past E, v = (c:b:a) >> r has stream bit E - 1 - i at bit i and window m = (v >> m K) & 0xffff, the
// value rate_window<K>(tile, 8 lane + 7 - m) gathers with one or two loads per window.
struct gemv_lane_plan {
    uint a;     // word holding stream bit E - 1
    uint b;     // the word before a (circular)
    uint c;     // the word before b (read for K = 5 only)
    uint r;     // right shift putting stream bit E - 1 at bit 0: 0/16 (K2), 0/8/16/24 (K3, K5), 0 (K4)
};

// depends on lane and K only: computed once, outside the tk loop
template <int K>
METAL_FUNC gemv_lane_plan gemv_lane_plan_of(uint lane) {
    constexpr uint words = 8u * K;
    const uint e1 = 8u * K * (lane + 1u) - 1u;
    gemv_lane_plan pl;
    pl.a = e1 >> 5u;
    pl.b = (pl.a + words - 1u) % words;
    pl.c = (pl.a + words - 2u) % words;
    pl.r = 31u - (e1 & 31u);
    return pl;
}

// bits r .. r + 31 of (hi:lo), r < 32: a 32-bit funnel shift (Apple GPUs emulate ulong shifts with several 32-bit ops)
METAL_FUNC uint gemv_funnel(uint lo, uint hi, uint r) {
    return (r == 0u) ? lo : ((lo >> r) | (hi << (32u - r)));
}

// the lane's 8 windows of one tile, win[m] = rate_window<K>(tile, 8 lane + 7 - m)
template <int K>
METAL_FUNC void gemv_lane_windows(const device uint* tile, gemv_lane_plan pl, thread uint* win);

template <>
METAL_FUNC void gemv_lane_windows<2>(const device uint* tile, gemv_lane_plan pl, thread uint* win) {
    // 30 bits: all eight windows (offsets 2m, ending at bit 29 at most) sit in bits 0..31 of v
    const uint v0 = gemv_funnel(tile[pl.a], tile[pl.b], pl.r);
    for (uint m = 0; m < 8u; ++m) win[m] = (v0 >> (2u * m)) & 0xffffu;
}

template <>
METAL_FUNC void gemv_lane_windows<3>(const device uint* tile, gemv_lane_plan pl, thread uint* win) {
    // the K3 kernels' extraction (rd8 = word a, rd9 = word b, lp = r): windows 0..5 end at bit 31 at most, 6..7 need
    // bits 18..40; lp is one of (0, 8, 16, 24)
    const uint rd8 = tile[pl.a];
    const uint rd9 = tile[pl.b];
    const uint lp = pl.r;
    const uint v0 = (lp == 0u) ? rd8 : ((rd8 >> lp) | (rd9 << (32u - lp)));      // bits 0..31 of v
    const uint v1 = (lp < 16u) ? ((rd8 >> (lp + 16u)) | (rd9 << (16u - lp)))     // bits 16..47 of v
                               : (rd9 >> (lp - 16u));
    for (uint m = 0; m < 6u; ++m) win[m] = (v0 >> (3u * m)) & 0xffffu;
    for (uint m = 6u; m < 8u; ++m) win[m] = (v1 >> (3u * m - 16u)) & 0xffffu;
}

template <>
METAL_FUNC void gemv_lane_windows<4>(const device uint* tile, gemv_lane_plan pl, thread uint* win) {
    // E = 32 (lane + 1) is word aligned (r = 0): v = (b:a), windows 0..4 in a, 5..7 straddle into b
    const uint v0 = tile[pl.a];                                       // bits 0..31 of v
    const uint v1 = (v0 >> 16u) | (tile[pl.b] << 16u);                // bits 16..47 of v
    for (uint m = 0; m < 5u; ++m) win[m] = (v0 >> (4u * m)) & 0xffffu;
    for (uint m = 5u; m < 8u; ++m) win[m] = (v1 >> (4u * m - 16u)) & 0xffffu;
}

template <>
METAL_FUNC void gemv_lane_windows<5>(const device uint* tile, gemv_lane_plan pl, thread uint* win) {
    // 51 bits: windows 0..3 (offsets 0..15) in bits 0..31 of v, 4..6 (20..30) in bits 16..47, 7 (35) in bits 32..63
    const uint wb = tile[pl.b];
    const uint v0 = gemv_funnel(tile[pl.a], wb, pl.r);                // bits 0..31 of v
    const uint v2 = gemv_funnel(wb, tile[pl.c], pl.r);                // bits 32..63 of v
    const uint v1 = (v0 >> 16u) | (v2 << 16u);                        // bits 16..47 of v
    for (uint m = 0; m < 4u; ++m) win[m] = (v0 >> (5u * m)) & 0xffffu;
    for (uint m = 4u; m < 7u; ++m) win[m] = (v1 >> (5u * m - 16u)) & 0xffffu;
    win[7] = (v2 >> 3u) & 0xffffu;
}

// The decode GEMV k loop at a compile-time rate K: TILES k-tiles (tile (tk, tn) at base + (tk COLS + tn) 16 K shorts),
// acc[m] += x[tk 16 + dr[m]] * w_m in tk order, exactly as the K3 kernels accumulate.  Callers with a per-expert K
// dispatch to one instance per threadgroup, so each instance keeps only its own plan and words live.
template <int K, uint TILES, uint COLS, uint UNROLL>
METAL_FUNC void gemv_rate_loop(const device short* base, const device float* xp, uint tn, uint lane,
                               thread const uint* dr, thread float* acc) {
    const gemv_lane_plan pl = gemv_lane_plan_of<K>(lane);
    #pragma clang loop unroll_count(UNROLL)
    for (uint tk = 0; tk < TILES; ++tk) {
        const device short* tile = base + (tk * COLS + tn) * (16u * K);
        uint win[8];
        gemv_lane_windows<K>(reinterpret_cast<const device uint*>(tile), pl, win);
        for (uint m = 0; m < 8u; ++m) {
            const float w = float(dig_cb(win[m]));
            acc[m] = metal::fma(xp[tk * 16u + dr[m]], w, acc[m]);
        }
    }
}

// Each member keeps its own accumulator order; ROWS is selected once outside this loop.
template <int K, uint TILES, uint COLS, uint UNROLL, uint ROWS>
METAL_FUNC void gemv_rate_group_loop(const device short* base, const device float* xh,
                                    thread const uint* members, uint tn, uint lane,
                                    thread const uint* dr, thread float* acc) {
    const gemv_lane_plan pl = gemv_lane_plan_of<K>(lane);
    #pragma clang loop unroll_count(UNROLL)
    for (uint tk = 0; tk < TILES; ++tk) {
        const device short* tile = base + (tk * COLS + tn) * (16u * K);
        uint win[8];
        gemv_lane_windows<K>(reinterpret_cast<const device uint*>(tile), pl, win);
        for (uint m = 0; m < 8u; ++m) {
            const float w = float(dig_cb(win[m]));
            #pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS; ++r) {
                acc[r * 8u + m] = metal::fma(xh[members[r] * (TILES * 16u) + tk * 16u + dr[m]], w, acc[r * 8u + m]);
            }
        }
    }
}
