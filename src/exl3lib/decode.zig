//! The decode chain (rows <= DECODE_ROWS_MAX), form 1: two launches per MoE layer, any admitted rate, codebook, window
//! and 128-aligned geometry, every constant a template int of one Metal text per kernel.
//!
//! Launch 1, `exl3lib_gu`, one 128-thread threadgroup per (128-wide intermediate block, slot, row): the four simdgroups
//! rotate the row's input for the slot's expert (f16 round, suh, H128 per block as a 5-shuffle + 2-register butterfly,
//! 1/sqrt(128), f16 round: `format.prepareInput`) into threadgroup memory once, then split the input tiles four ways and
//! stream the gate and up trellis tiles of the block's 8 output tiles through the lane plan (`lane.zig`), decode in
//! registers and accumulate f32; the partial sums meet in threadgroup memory, simdgroup 0 rounds to f16 (`innerGemv`),
//! rotates the block's 128 outputs (H128), scales by svh and rounds (`finishOutput`), and writes h = silu(gate) * up as f32.
//!
//! Launch 2, `exl3lib_down`, one threadgroup per (128-wide hidden block, row): for each of the row's slots in order it
//! rotates that slot's h (f16 round, down suh, H128, f16 round), streams the down tiles of its 8 output tiles, reduces,
//! rounds, rotates, scales by svh, rounds, and accumulates score_k * y_k in registers; one f32 write per output at the end.
//! No atomics, no cross-threadgroup traffic: the sum over slots is in slot order, so the bytes are deterministic.
//!
//! Rounding points are Sushi's host reference's (`ref.zig`); the reductions run in a different order, so parity is a
//! tolerance, not bit identity (the GPU test states it).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("../sdk_ext/kernel_set.zig");
const format = @import("format.zig");
const lane = @import("lane.zig");

pub const Geometry = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rate: format.Rate,
    dec: format.Decode,
    /// timing probes only (never a served kernel): 1 skips the codebook hash, 2 skips the funnel too
    probe: u8 = 0,

    pub fn admitted(g: Geometry) bool {
        return g.hidden % 128 == 0 and g.inter % 128 == 0 and g.hidden > 0 and g.inter > 0 and g.topk > 0 and g.topk <= 32;
    }
};

/// The Metal body shared by both kernels after the helpers: the lane plan's tile walk. `exl3lib_tile_fma` decodes a
/// lane's eight weights of one tile and accumulates them against the rotated input held in threadgroup memory (f16).
const metal_common =
    \\// lane L's eight weights of a tile: code positions 8L + m, m < 4 at column L / 4, m >= 4 at column L / 4 + 8,
    \\// rows 2 (L & 3) + {0, 1, 8, 9}[m & 3] (quantize.py's tensor-core layout)
    \\inline uint exl3lib_row_of(uint L, uint m) { const uint q = m & 3u; return 2u * (L & 3u) + ((q & 1u) ? 1u : 0u) + ((q & 2u) ? 8u : 0u); }
    \\
;

/// Launch 1's text.
pub const text_gu =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int PRB = int(PROBE);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the intermediate
    \\    const uint k = threadgroup_position_in_grid.y;        // slot
    \\    const uint row = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = slots[row * TOPKC + k];
    \\    threadgroup half xg[HIDC];
    \\    threadgroup half xu[HIDC];
    \\    threadgroup float red_g[4][128];
    \\    threadgroup float red_u[4][128];
    \\    // 1. the rotated inputs for gate and up (their suh differ): simdgroup sg owns blocks sg, sg + 4, ...
    \\    for (uint b = sg; b < HIDC / 128u; b += 4u) {
    \\        float vg[4]; float vu[4];
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = b * 128u + L + 32u * j;
    \\            const float xr = float(half(x[row * HIDC + i]));
    \\            vg[j] = xr * float(gate_suh[e * HIDC + i]);
    \\            vu[j] = xr * float(up_suh[e * HIDC + i]);
    \\        }
    \\        for (uint s = 0; s < 5u; ++s) {
    \\            const uint bit = 1u << s;
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
    \\                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
    \\                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
    \\                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
    \\            }
    \\        }
    \\        {   // stages 5 and 6 pair the registers (positions L + 32 j: bit 5 is j & 1, bit 6 is j & 2)
    \\            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
    \\            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
    \\            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
    \\            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
    \\        }
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = b * 128u + L + 32u * j;
    \\            xg[i] = half(vg[j] * HAD_SCALE);
    \\            xu[i] = half(vu[j] * HAD_SCALE);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles, gate and up
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = exl3lib_row_of(L, m);
    \\    float ag[8][2]; float au[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { ag[t][0] = 0.0f; ag[t][1] = 0.0f; au[t][0] = 0.0f; au[t][1] = 0.0f; }
    \\    const uint tk_per = IN_TILES / 4u;
    \\    const device uint* gbase = (const device uint*)(gate_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device uint* ubase = (const device uint*)(up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\        float xr_g[4]; float xr_u[4];
    \\        for (uint m = 0; m < 4u; ++m) { xr_g[m] = float(xg[tk * 16u + rows_of[m]]); xr_u[m] = float(xu[tk * 16u + rows_of[m]]); }
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const uint tn = blk * 8u + t;
    \\            const ulong toff = (ulong(tk) * OUT_TILES + tn) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords wg = exl3_int_words<N>(gbase + toff, base, ia);
    \\                const IntWords wu = exl3_int_words<N>(ubase + toff, base, ia);
    \\                uint cg[8], cu[8];
    \\                cg[0] = exl3_int_cw<N, WMASK, 0>(wg, ib); cg[1] = exl3_int_cw<N, WMASK, 1>(wg, ib); cg[2] = exl3_int_cw<N, WMASK, 2>(wg, ib); cg[3] = exl3_int_cw<N, WMASK, 3>(wg, ib);
    \\                cg[4] = exl3_int_cw<N, WMASK, 4>(wg, ib); cg[5] = exl3_int_cw<N, WMASK, 5>(wg, ib); cg[6] = exl3_int_cw<N, WMASK, 6>(wg, ib); cg[7] = exl3_int_cw<N, WMASK, 7>(wg, ib);
    \\                cu[0] = exl3_int_cw<N, WMASK, 0>(wu, ib); cu[1] = exl3_int_cw<N, WMASK, 1>(wu, ib); cu[2] = exl3_int_cw<N, WMASK, 2>(wu, ib); cu[3] = exl3_int_cw<N, WMASK, 3>(wu, ib);
    \\                cu[4] = exl3_int_cw<N, WMASK, 4>(wu, ib); cu[5] = exl3_int_cw<N, WMASK, 5>(wu, ib); cu[6] = exl3_int_cw<N, WMASK, 6>(wu, ib); cu[7] = exl3_int_cw<N, WMASK, 7>(wu, ib);
    \\                for (uint m = 0; m < 8u; ++m) {
    \\                    const float wgv = (PRB == 0) ? float(exl3_decode<CB>(cg[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cg[m] & 0x3fffu))) : float(as_type<half>(ushort((wg.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                    const float wuv = (PRB == 0) ? float(exl3_decode<CB>(cu[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cu[m] & 0x3fffu))) : float(as_type<half>(ushort((wu.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                    const uint c = (m >> 2);
    \\                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
    \\                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
    \\                }
    \\            } else {
    \\                const LaneWords wg = exl3_lane_words<N>(gbase + toff, base);
    \\                const LaneWords wu = exl3_lane_words<N>(ubase + toff, base);
    \\                for (uint m = 0; m < 8u; ++m) {
    \\                    const float wgv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wg, off[m])));
    \\                    const float wuv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(wu, off[m])));
    \\                    const uint c = (m >> 2);
    \\                    ag[t][c] = fma(xr_g[m & 3u], wgv, ag[t][c]);
    \\                    au[t][c] = fma(xr_u[m & 3u], wuv, au[t][c]);
    \\                }
    \\            }
    \\        }
    \\    }
    \\    // 3. the column sums: lanes 4c..4c+3 hold column c (and c + 8); fold them, then the four simdgroups
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) {
    \\        ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(1)); ag[t][c] += simd_shuffle_xor(ag[t][c], ushort(2));
    \\        au[t][c] += simd_shuffle_xor(au[t][c], ushort(1)); au[t][c] += simd_shuffle_xor(au[t][c], ushort(2));
    \\    }
    \\    if ((L & 3u) == 0u) {
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            red_g[sg][t * 16u + (L >> 2)] = ag[t][0]; red_g[sg][t * 16u + (L >> 2) + 8u] = ag[t][1];
    \\            red_u[sg][t * 16u + (L >> 2)] = au[t][0]; red_u[sg][t * 16u + (L >> 2) + 8u] = au[t][1];
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 4. simdgroup 0: the block's 128 outputs (positions L + 32 j): f16 round, H128, svh, f16 round, SwiGLU
    \\    if (sg == 0u) {
    \\        float vg[4]; float vu[4];
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint i = L + 32u * j;
    \\            vg[j] = float(half(((red_g[0][i] + red_g[1][i]) + red_g[2][i]) + red_g[3][i]));
    \\            vu[j] = float(half(((red_u[0][i] + red_u[1][i]) + red_u[2][i]) + red_u[3][i]));
    \\        }
    \\        for (uint s = 0; s < 5u; ++s) {
    \\            const uint bit = 1u << s;
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const float pg = simd_shuffle_xor(vg[j], ushort(bit));
    \\                const float pu = simd_shuffle_xor(vu[j], ushort(bit));
    \\                vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg);
    \\                vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu);
    \\            }
    \\        }
    \\        {
    \\            float tg0 = vg[0] + vg[1], tg1 = vg[0] - vg[1], tg2 = vg[2] + vg[3], tg3 = vg[2] - vg[3];
    \\            float tu0 = vu[0] + vu[1], tu1 = vu[0] - vu[1], tu2 = vu[2] + vu[3], tu3 = vu[2] - vu[3];
    \\            vg[0] = tg0 + tg2; vg[1] = tg1 + tg3; vg[2] = tg0 - tg2; vg[3] = tg1 - tg3;
    \\            vu[0] = tu0 + tu2; vu[1] = tu1 + tu3; vu[2] = tu0 - tu2; vu[3] = tu1 - tu3;
    \\        }
    \\        for (uint j = 0; j < 4u; ++j) {
    \\            const uint o = blk * 128u + L + 32u * j;
    \\            const float g = float(half(vg[j] * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\            const float u = float(half(vu[j] * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\            h[(row * TOPKC + k) * INTC + o] = (g / (1.0f + metal::exp(-g))) * u;
    \\        }
    \\    }
    \\
;

/// Launch 2's text.
pub const text_down =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr int PRB = int(PROBE);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;      // 128-wide block of the hidden output
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    threadgroup half hd[INTC];
    \\    threadgroup float red[4][128];
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = exl3lib_row_of(L, m);
    \\    const uint tk_per = IN_TILES / 4u;
    \\    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};              // simdgroup 0's outputs L + 32 j, summed over the slots
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint e = slots[row * TOPKC + k];
    \\        const float score = scores[row * TOPKC + k];
    \\        // 1. this slot's rotated h: f16 round, down suh, H128, 1/sqrt(128), f16 round
    \\        for (uint b = sg; b < INTC / 128u; b += 4u) {
    \\            float v[4];
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const uint i = b * 128u + L + 32u * j;
    \\                v[j] = float(half(h[(row * TOPKC + k) * INTC + i])) * float(down_suh[e * INTC + i]);
    \\            }
    \\            for (uint s = 0; s < 5u; ++s) {
    \\                const uint bit = 1u << s;
    \\                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
    \\            }
    \\            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\            for (uint j = 0; j < 4u; ++j) hd[b * 128u + L + 32u * j] = half(v[j] * HAD_SCALE);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // 2. the GEMV over this simdgroup's quarter of the input tiles, the block's 8 output tiles
    \\        float a[8][2];
    \\        for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\        const device uint* dbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\        for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\            float xr[4];
    \\            for (uint m = 0; m < 4u; ++m) xr[m] = float(hd[tk * 16u + rows_of[m]]);
    \\            for (uint t = 0; t < 8u; ++t) {
    \\                const uint tn = blk * 8u + t;
    \\                if (INTK) {
    \\                    const IntWords w = exl3_int_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base, ia);
    \\                    uint cw[8];
    \\                    cw[0] = exl3_int_cw<N, WMASK, 0>(w, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(w, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(w, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(w, ib);
    \\                    cw[4] = exl3_int_cw<N, WMASK, 4>(w, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(w, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(w, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(w, ib);
    \\                    for (uint m = 0; m < 8u; ++m) {
    \\                        const float wv = (PRB == 0) ? float(exl3_decode<CB>(cw[m])) : ((PRB == 1) ? float(as_type<half>(ushort(cw[m] & 0x3fffu))) : float(as_type<half>(ushort((w.h[0] >> (m * 2u)) & 0x3fffu))));
    \\                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
    \\                    }
    \\                } else {
    \\                    const LaneWords w = exl3_lane_words<N>(dbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW), base);
    \\                    for (uint m = 0; m < 8u; ++m) {
    \\                        const float wv = float(exl3_decode<CB>(exl3_lane_cw<WMASK>(w, off[m])));
    \\                        a[t][m >> 2] = fma(xr[m & 3u], wv, a[t][m >> 2]);
    \\                    }
    \\                }
    \\            }
    \\        }
    \\        for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\        if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // 3. simdgroup 0: f16 round, H128, svh, f16 round, score-weighted accumulate
    \\        if (sg == 0u) {
    \\            float v[4];
    \\            for (uint j = 0; j < 4u; ++j) { const uint i = L + 32u * j; v[j] = float(half(((red[0][i] + red[1][i]) + red[2][i]) + red[3][i])); }
    \\            for (uint s = 0; s < 5u; ++s) {
    \\                const uint bit = 1u << s;
    \\                for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); }
    \\            }
    \\            { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\            for (uint j = 0; j < 4u; ++j) {
    \\                const uint o = blk * 128u + L + 32u * j;
    \\                acc[j] = fma(score, float(half(v[j] * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc[j]);
    \\            }
    \\        }
    \\        // the next slot's hd and red overwrite what every simdgroup just read
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (sg == 0u) for (uint j = 0; j < 4u; ++j) y[row * HIDC + blk * 128u + L + 32u * j] = acc[j];
    \\
;

pub const header: [:0]const u8 = lane.metal_helpers ++ metal_common;

const gu_inputs = [_][*:0]const u8{ "x", "slots", "gate_t", "gate_suh", "gate_svh", "up_t", "up_suh", "up_svh" };
const gu_outputs = [_][*:0]const u8{"h"};
const down_inputs = [_][*:0]const u8{ "h", "slots", "scores", "down_t", "down_suh", "down_svh" };
const down_outputs = [_][*:0]const u8{"y"};

/// The two kernels of one geometry (compiled once per (rate, codebook, window, hidden, inter, topk)); configs per row count.
pub const Kernels = struct {
    geo: Geometry,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    cfg_gu: [max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    cfg_down: [max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),

    pub const max_rows: usize = 16;

    pub fn init(geo: Geometry) !Kernels {
        if (!geo.admitted()) return error.BadExl3Shape;
        const gu = ks.newKernel("exl3lib_gu", &gu_inputs, &gu_outputs, text_gu, header, true) orelse return error.KernelCreateFailed;
        const down = ks.newKernel("exl3lib_down", &down_inputs, &down_outputs, text_down, header, true) orelse return error.KernelCreateFailed;
        return .{ .geo = geo, .gu = gu, .down = down };
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.cfg_gu) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        for (&self.cfg_down) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        _ = mlx.mlx_fast_metal_kernel_free(self.gu);
        _ = mlx.mlx_fast_metal_kernel_free(self.down);
    }

    fn template(self: *const Kernels, buf: *[7]ks.TemplateArg) []const ks.TemplateArg {
        buf.* = .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.geo.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.geo.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.geo.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = @intCast(self.geo.hidden) } },
            .{ .name = "INTER", .value = .{ .int = @intCast(self.geo.inter) } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(self.geo.topk) } },
            .{ .name = "PROBE", .value = .{ .int = self.geo.probe } },
        };
        return buf;
    }

    fn configs(self: *Kernels, rows: usize) !struct { gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config } {
        if (rows == 0 or rows > max_rows) return error.BadExl3Shape;
        if (self.cfg_gu[rows] == null) {
            var tbuf: [7]ks.TemplateArg = undefined;
            const t = self.template(&tbuf);
            const g = self.geo;
            const r: c_int = @intCast(rows);
            const h_shape = [_]c_int{ r * @as(c_int, @intCast(g.topk)), @intCast(g.inter) };
            self.cfg_gu[rows] = try ks.buildConfig(.{
                .grid = .{ (g.inter / 128) * 128, g.topk, @intCast(rows) },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{&h_shape},
                .out_dtypes = &.{.float32},
                .template = t,
            });
            const y_shape = [_]c_int{ r, @intCast(g.hidden) };
            self.cfg_down[rows] = try ks.buildConfig(.{
                .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{&y_shape},
                .out_dtypes = &.{.float32},
                .template = t,
            });
        }
        return .{ .gu = self.cfg_gu[rows].?, .down = self.cfg_down[rows].? };
    }

    pub const Bank = struct {
        gate_t: mlx.mlx_array,
        gate_suh: mlx.mlx_array,
        gate_svh: mlx.mlx_array,
        up_t: mlx.mlx_array,
        up_suh: mlx.mlx_array,
        up_svh: mlx.mlx_array,
        down_t: mlx.mlx_array,
        down_suh: mlx.mlx_array,
        down_svh: mlx.mlx_array,
    };

    /// Launch 1: h f32 [rows * topk, inter] from x f32 [rows, hidden] and slots u32 [rows * topk].
    pub fn gateUp(self: *Kernels, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Bank, rows: usize) !mlx.mlx_array {
        const c = try self.configs(rows);
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.gu, c.gu, &.{ x, slots, bank.gate_t, bank.gate_suh, bank.gate_svh, bank.up_t, bank.up_suh, bank.up_svh }, &outs, s);
        return outs[0];
    }

    /// Launch 2: y f32 [rows, hidden] from h, slots and scores f32 [rows * topk].
    pub fn downReduce(self: *Kernels, s: mlx.mlx_stream, h: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Bank, rows: usize) !mlx.mlx_array {
        const c = try self.configs(rows);
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.down, c.down, &.{ h, slots, scores, bank.down_t, bank.down_suh, bank.down_svh }, &outs, s);
        return outs[0];
    }

    /// The layer: both launches; the caller frees h and y.
    pub fn moe(self: *Kernels, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Bank, rows: usize) !struct { h: mlx.mlx_array, y: mlx.mlx_array } {
        const h = try self.gateUp(s, x, slots, bank, rows);
        errdefer _ = mlx.mlx_array_free(h);
        const y = try self.downReduce(s, h, slots, scores, bank, rows);
        return .{ .h = h, .y = y };
    }
};

test "exl3lib decode texts: every template constant the texts read is one the launch passes" {
    const t = std.testing;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "TOPK", "PROBE" }) |name| {
        try t.expect(std.mem.indexOf(u8, header ++ text_gu, name) != null);
        try t.expect(std.mem.indexOf(u8, header ++ text_down, name) != null);
    }
    // every input the signatures name is read by its text, and the outputs are written
    for (gu_inputs) |n| try t.expect(std.mem.indexOf(u8, text_gu, std.mem.span(n)) != null);
    for (down_inputs) |n| try t.expect(std.mem.indexOf(u8, text_down, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_gu, "h[(row * TOPKC + k) * INTC + o] =") != null);
    try t.expect(std.mem.indexOf(u8, text_down, "y[row * HIDC + blk * 128u + L + 32u * j] = acc[j]") != null);
}

test "exl3lib decode geometry admission: 128-aligned widths, top-k within the reduce bank" {
    const t = std.testing;
    const ok: Geometry = .{ .hidden = 2560, .inter = 640, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 };
    try t.expect(ok.admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 600, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 640, .topk = 33, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
}
