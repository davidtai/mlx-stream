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

/// Launch 1 of form 1b: `text_gu` with the input rotation done once per (row, slot) by `unfused.text_prep`.
pub const text_gu4 =
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
    \\    threadgroup float red_g[4][128];
    \\    threadgroup float red_u[4][128];
    \\    const device half* xg = xgp + ulong(row * TOPKC + k) * ulong(HIDC);
    \\    const device half* xu = xup + ulong(row * TOPKC + k) * ulong(HIDC);
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

/// Launch 2 of form 1b: `text_down` with each slot's h rotation done once by `unfused.text_prep` (over inter).
pub const text_down4 =
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
    \\        const device half* hd = hdp + ulong(row * TOPKC + k) * ulong(INTC);
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

/// The GEMV core every form-3 kernel inlines: lane L of a simdgroup accumulates its 8 tiles x 2 columns over input
/// tiles [tk0, tk1) of expert `wbase` (the stacked bank's expert base, in uint words), x in lane order.
const metal_gemv_core =
    \\template <uint N, uint WMASK, int CB, uint OUT_TILES, uint BT>
    \\inline void exl3lib_gemv_core(const device uint* wbase, const device half* xrow, uint tk0, uint tk1, uint blk, uint L, thread float (&a)[BT][2]) {
    \\  constexpr uint NW = N / 2u;
    \\  // the general fast path (every even N): one select, one shift, one mask per weight
    \\  const uint qb = exl3_gen_qb<N>(L);
    \\  const uint ga = exl3_gen_a<N>(L);
    \\  const uint lp = exl3_gen_lp<N>(L);
    \\  for (uint tk = tk0; tk < tk1; ++tk) {
    \\    const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    \\    const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    \\    for (uint t = 0; t < BT; ++t) {
    \\      const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BT + t) * ulong(NW);
    \\      const IntWords ww = exl3_int_words<N>(tile, qb, ga);
    \\      uint cw[8];
    \\      cw[0] = exl3_gen_cw<N, WMASK, 0>(ww, lp); cw[1] = exl3_gen_cw<N, WMASK, 1>(ww, lp); cw[2] = exl3_gen_cw<N, WMASK, 2>(ww, lp); cw[3] = exl3_gen_cw<N, WMASK, 3>(ww, lp);
    \\      cw[4] = exl3_gen_cw<N, WMASK, 4>(ww, lp); cw[5] = exl3_gen_cw<N, WMASK, 5>(ww, lp); cw[6] = exl3_gen_cw<N, WMASK, 6>(ww, lp); cw[7] = exl3_gen_cw<N, WMASK, 7>(ww, lp);
    \\      for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    \\    }
    \\  }
    \\}
    \\// the simdgroup's BT x 2 accumulators folded over the quad lanes into red[sg][16 BT] (output = tile * 16 + column)
    \\template <uint BT>
    \\inline void exl3lib_fold(thread float (&a)[BT][2], threadgroup float* red_sg, uint L) {
    \\  for (uint t = 0; t < BT; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\  if ((L & 3u) == 0u) for (uint t = 0; t < BT; ++t) { red_sg[t * 16u + (L >> 2)] = a[t][0]; red_sg[t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\}
    \\// H128 of 128 values held one per thread (tid): 5 shuffle stages in the simdgroup, 2 exchange stages through xch
    \\inline float exl3lib_h128(float v, threadgroup float* xch, uint tid, uint L) {
    \\  for (uint st = 0; st < 5u; ++st) { const uint bit = 1u << st; const float p = simd_shuffle_xor(v, ushort(bit)); v = (L & bit) ? (p - v) : (v + p); }
    \\  for (uint st = 5; st < 7u; ++st) {
    \\    const uint bit = 1u << st;
    \\    xch[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const float p = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    v = (tid & bit) ? (p - v) : (v + p);
    \\  }
    \\  return v;
    \\}
    \\
;

pub const header: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core;

const gu_inputs = [_][*:0]const u8{ "x", "slots", "gate_t", "gate_suh", "gate_svh", "up_t", "up_suh", "up_svh" };
const gu_outputs = [_][*:0]const u8{"h"};
const down_inputs = [_][*:0]const u8{ "h", "slots", "scores", "down_t", "down_suh", "down_svh" };
const down_outputs = [_][*:0]const u8{"y"};
const gu4_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "gate_svh", "up_t", "up_svh" };
const down4_inputs = [_][*:0]const u8{ "hdp", "slots", "scores", "down_t", "down_svh" };

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

/// Form 1b: four launches per layer. The two prepares are `unfused.text_prep` (one simdgroup per 128-block and row-slot:
/// each rotation computed once), then `text_gu4` (GEMV + finish + SwiGLU) and `text_down4` (GEMV + finish + reduce).
pub const Kernels4 = struct {
    geo: Geometry,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, prep_i: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    pub fn init(geo: Geometry) !Kernels4 {
        if (!geo.admitted()) return error.BadExl3Shape;
        const unf = @import("unfused.zig");
        return .{
            .geo = geo,
            .prep = ks.newKernel("exl3lib_prep", &prep_in, &prep_out, unf.text_prep, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("exl3lib_gu4", &gu4_inputs, &gu_outputs, text_gu4, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("exl3lib_down4", &down4_inputs, &down_outputs, text_down4, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels4) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.prep_i, cc.gu, cc.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gu, self.down }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn cfgs(self: *Kernels4, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const t = [_]ks.TemplateArg{
            .{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "INTER", .value = .{ .int = I } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
            .{ .name = "PROBE", .value = .{ .int = 0 } },
        };
        const c: Cfg = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } } } }),
            .prep_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = I } }, .{ .name = "TOPK", .value = .{ .int = 1 } } } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, g.topk, @intCast(rows) }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &t }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &t }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The four launches; the caller frees y.
    pub fn moe(self: *Kernels4, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer _ = mlx.mlx_array_free(xg);
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer _ = mlx.mlx_array_free(xu);
        const h = try one(self.gu, c.gu, &.{ xg, xu, slots, bank.gate_t, bank.gate_svh, bank.up_t, bank.up_svh }, s);
        defer _ = mlx.mlx_array_free(h);
        const hd = try one(self.prep, c.prep_i, &.{ h, slots, bank.down_suh }, s);
        defer _ = mlx.mlx_array_free(hd);
        return one(self.down, c.down, &.{ hd, slots, scores, bank.down_t, bank.down_svh }, s);
    }
};

/// Form 2, launch 1: one 128-thread threadgroup per (128-wide intermediate block, row-slot); simdgroup sg handles
/// projection sg >> 1 (gate, up) over input half sg & 1. Each lane holds one projection's 8 tiles x 2 columns and loads
/// its quad's four x rows per tile as one half4 (lane-ordered xg / xu from the prepare). The two halves meet in
/// threadgroup memory; all 128 threads then round, rotate the block's outputs (5 shuffle stages + 2 threadgroup
/// exchanges), scale by svh, round and write h = silu(gate) * up.
pub const text_gu2 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = slots[rk];
    \\    const uint proj = sg >> 1u;          // 0 gate, 1 up
    \\    const uint khalf = sg & 1u;
    \\    threadgroup float red[4][128];       // [sg][output]
    \\    threadgroup float xch[128];          // the rotation's cross-simdgroup exchange
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    float a[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device half* xrow = (proj == 0u ? xgp : xup) + ulong(rk) * ulong(HIDC) + 4u * (L & 3u);
    \\    const uint tk_per = IN_TILES / 2u;
    \\    for (uint tk = khalf * tk_per; tk < (khalf + 1u) * tk_per; ++tk) {
    \\        const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    \\        const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * 8u + t) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords ww = exl3_int_words<N>(tile, base, ia);
    \\                uint cw[8];
    \\                cw[0] = exl3_int_cw<N, WMASK, 0>(ww, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(ww, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(ww, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(ww, ib);
    \\                cw[4] = exl3_int_cw<N, WMASK, 4>(ww, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(ww, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(ww, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(ww, ib);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    \\            } else {
    \\                const LaneWords ww = exl3_lane_words<N>(tile, base);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(exl3_lane_cw<WMASK>(ww, off[m]))), a[t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\    if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // every thread owns output o = tid for the finish: gate (red 0 + 1) and up (red 2 + 3), f16 round, H128
    \\    float g = float(half(red[0][tid] + red[1][tid]));
    \\    float u = float(half(red[2][tid] + red[3][tid]));
    \\    for (uint st = 0; st < 5u; ++st) {
    \\        const uint bit = 1u << st;
    \\        const float pg = simd_shuffle_xor(g, ushort(bit)); const float pu = simd_shuffle_xor(u, ushort(bit));
    \\        g = (L & bit) ? (pg - g) : (g + pg); u = (L & bit) ? (pu - u) : (u + pu);
    \\    }
    \\    // stages 5 and 6 cross simdgroups (bit 5 = sg & 1, bit 6 = sg >> 1) through threadgroup memory
    \\    for (uint st = 5; st < 7u; ++st) {
    \\        const uint bit = 1u << st;
    \\        xch[tid] = g; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float pg = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        g = (tid & bit) ? (pg - g) : (g + pg);
    \\        xch[tid] = u; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float pu = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        u = (tid & bit) ? (pu - u) : (u + pu);
    \\    }
    \\    const uint o = blk * 128u + tid;
    \\    const float gv = float(half(g * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\    const float uv = float(half(u * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\    h[ulong(rk) * ulong(INTC) + o] = (gv / (1.0f + metal::exp(-gv))) * uv;
    \\
;

/// Form 2, launch 2: one threadgroup per (128-wide hidden block, row-slot); the four simdgroups split the input tiles;
/// all 128 threads finish (round, rotate, svh, round) and write yk f32 [RK, hidden] scaled by the slot's score, for the
/// reduce launch (slot-order sums: deterministic bytes).
pub const text_down2 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = slots[rk];
    \\    threadgroup float red[4][128];
    \\    threadgroup float xch[128];
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    float a[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const device uint* wbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device half* xrow = hdp + ulong(rk) * ulong(INTC) + 4u * (L & 3u);
    \\    const uint tk_per = IN_TILES / 4u;
    \\    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\        const half4 x4 = *reinterpret_cast<const device half4*>(xrow + tk * 16u);
    \\        const float xr[4] = { float(x4.x), float(x4.y), float(x4.z), float(x4.w) };
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * 8u + t) * ulong(NW);
    \\            if (INTK) {
    \\                const IntWords ww = exl3_int_words<N>(tile, base, ia);
    \\                uint cw[8];
    \\                cw[0] = exl3_int_cw<N, WMASK, 0>(ww, ib); cw[1] = exl3_int_cw<N, WMASK, 1>(ww, ib); cw[2] = exl3_int_cw<N, WMASK, 2>(ww, ib); cw[3] = exl3_int_cw<N, WMASK, 3>(ww, ib);
    \\                cw[4] = exl3_int_cw<N, WMASK, 4>(ww, ib); cw[5] = exl3_int_cw<N, WMASK, 5>(ww, ib); cw[6] = exl3_int_cw<N, WMASK, 6>(ww, ib); cw[7] = exl3_int_cw<N, WMASK, 7>(ww, ib);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(cw[m])), a[t][m >> 2]);
    \\            } else {
    \\                const LaneWords ww = exl3_lane_words<N>(tile, base);
    \\                for (uint m = 0; m < 8u; ++m) a[t][m >> 2] = fma(xr[m & 3u], float(exl3_decode<CB>(exl3_lane_cw<WMASK>(ww, off[m]))), a[t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint t = 0; t < 8u; ++t) for (uint c = 0; c < 2u; ++c) { a[t][c] += simd_shuffle_xor(a[t][c], ushort(1)); a[t][c] += simd_shuffle_xor(a[t][c], ushort(2)); }
    \\    if ((L & 3u) == 0u) for (uint t = 0; t < 8u; ++t) { red[sg][t * 16u + (L >> 2)] = a[t][0]; red[sg][t * 16u + (L >> 2) + 8u] = a[t][1]; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float v = float(half(((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid]));
    \\    for (uint st = 0; st < 5u; ++st) { const uint bit = 1u << st; const float p = simd_shuffle_xor(v, ushort(bit)); v = (L & bit) ? (p - v) : (v + p); }
    \\    for (uint st = 5; st < 7u; ++st) {
    \\        const uint bit = 1u << st;
    \\        xch[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const float p = xch[tid ^ bit]; threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        v = (tid & bit) ? (p - v) : (v + p);
    \\    }
    \\    const uint o = blk * 128u + tid;
    \\    yk[ulong(rk) * ulong(HIDC) + o] = scores[rk] * float(half(v * HAD_SCALE * float(down_svh[e * HIDC + o])));
    \\
;

/// reduce over the slots of a row (the scores already applied): y f32 [rows, hidden] = sum_k yk[r k], slot order
pub const text_reduce2 =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    const uint d = thread_position_in_grid.x;
    \\    const uint r = thread_position_in_grid.y;
    \\    float acc = 0.0f;
    \\    for (uint k = 0; k < TOPKC; ++k) acc += yk[(r * TOPKC + k) * DIMC + d];
    \\    y[r * DIMC + d] = acc;
    \\
;

const gu2_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "gate_svh", "up_t", "up_svh" };
const down2_inputs = [_][*:0]const u8{ "hdp", "slots", "scores", "down_t", "down_svh" };
const down2_outputs = [_][*:0]const u8{"yk"};
const reduce2_inputs = [_][*:0]const u8{"yk"};

/// Form 2: five launches per layer (prepare gate, prepare up, gate/up GEMV+finish+SwiGLU, prepare h, down GEMV+finish,
/// reduce), every threadgroup one (block, row-slot), lane-ordered prepared inputs.
pub const Kernels2 = struct {
    geo: Geometry,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    reduce: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, prep_i: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, reduce: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    pub fn init(geo: Geometry) !Kernels2 {
        if (!geo.admitted()) return error.BadExl3Shape;
        const unf = @import("unfused.zig");
        return .{
            .geo = geo,
            .prep = ks.newKernel("exl3lib_prep", &prep_in, &prep_out, unf.text_prep, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("exl3lib_gu2", &gu2_inputs, &gu_outputs, text_gu2, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("exl3lib_down2", &down2_inputs, &down2_outputs, text_down2, header, true) orelse return error.KernelCreateFailed,
            .reduce = ks.newKernel("exl3lib_reduce2", &reduce2_inputs, &down_outputs, text_reduce2, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels2) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.prep_i, cc.gu, cc.down, cc.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gu, self.down, self.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn cfgs(self: *Kernels2, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const t = [_]ks.TemplateArg{
            .{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = H } },
            .{ .name = "INTER", .value = .{ .int = I } },
        };
        const c: Cfg = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } } } }),
            .prep_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = I } }, .{ .name = "TOPK", .value = .{ .int = 1 } } } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &t }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float32}, .template = &t }),
            .reduce = try ks.buildConfig(.{ .grid = .{ g.hidden, @intCast(rows), 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } } } }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The five launches; the caller frees y.
    pub fn moe(self: *Kernels2, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer _ = mlx.mlx_array_free(xg);
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer _ = mlx.mlx_array_free(xu);
        const h = try one(self.gu, c.gu, &.{ xg, xu, slots, bank.gate_t, bank.gate_svh, bank.up_t, bank.up_svh }, s);
        defer _ = mlx.mlx_array_free(h);
        const hd = try one(self.prep, c.prep_i, &.{ h, slots, bank.down_suh }, s);
        defer _ = mlx.mlx_array_free(hd);
        const yk = try one(self.down, c.down, &.{ hd, slots, scores, bank.down_t, bank.down_svh }, s);
        defer _ = mlx.mlx_array_free(yk);
        return one(self.reduce, c.reduce, &.{yk}, s);
    }
};

/// Form 3 (split-K planes). Launch 2: one threadgroup per (128-wide intermediate block, row-slot, K slice of KS); the
/// four simdgroups are projection x half of the slice; the slice's partial sums go to the planes zg / zu f32
/// [KS, RK, inter] (written once each; no atomics).
pub const text_gu3 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint BTC = uint(BT);     // output tiles per threadgroup: 2 BT f32 accumulators per lane
    \\    constexpr uint IN_TILES = HIDC / 16u;
    \\    constexpr uint OUT_TILES = INTC / 16u;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint ks = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = slots[rk];
    \\    const uint proj = sg >> 1u;
    \\    const uint khalf = sg & 1u;
    \\    threadgroup float red[4][16u * BTC];
    \\    float a[BTC][2];
    \\    for (uint t = 0; t < BTC; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const device uint* wbase = (const device uint*)(proj == 0u ? gate_t : up_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const device half* xrow = (proj == 0u ? xgp : xup) + ulong(rk) * ulong(HIDC) + 4u * (L & 3u);
    \\    const uint per = IN_TILES / (KSC * 2u);
    \\    const uint tk0 = (ks * 2u + khalf) * per;
    \\    exl3lib_gemv_core<N, WMASK, CB, OUT_TILES, BTC>(wbase, xrow, tk0, tk0 + per, blk, L, a);
    \\    exl3lib_fold<BTC>(a, red[sg], L);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 16u * BTC) {
    \\        const ulong o = (ulong(ks) * ulong(RKC) + rk) * ulong(INTC) + blk * (16u * BTC) + tid;
    \\        zg[o] = red[0][tid] + red[1][tid];
    \\        zu[o] = red[2][tid] + red[3][tid];
    \\    }
    \\
;

/// Form 3, launch 3: per (128-wide intermediate block, row-slot): sum the planes, f16 round, H128, svh, f16 round,
/// SwiGLU, then the down's input rotation (f16 round, down suh, H128, f16 round) into hd in lane order.
pub const text_fin3 =
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = slots[rk];
    \\    threadgroup float xch[128];
    \\    const uint o = blk * 128u + tid;
    \\    float g = 0.0f, u = 0.0f;
    \\    for (uint ks = 0; ks < KSC; ++ks) { const ulong i = (ulong(ks) * ulong(RKC) + rk) * ulong(INTC) + o; g += zg[i]; u += zu[i]; }
    \\    g = exl3lib_h128(float(half(g)), xch, tid, L);
    \\    u = exl3lib_h128(float(half(u)), xch, tid, L);
    \\    const float gv = float(half(g * HAD_SCALE * float(gate_svh[e * INTC + o])));
    \\    const float uv = float(half(u * HAD_SCALE * float(up_svh[e * INTC + o])));
    \\    const float hv = (gv / (1.0f + metal::exp(-gv))) * uv;
    \\    // the down's prepare: f16 round, down suh, H128, f16 round, stored in lane order within the 16-wide tile
    \\    float d = float(half(hv)) * float(down_suh[e * INTC + o]);
    \\    d = exl3lib_h128(d, xch, tid, L);
    \\    const uint r = o & 15u;
    \\    const uint q = (r & 7u) >> 1u;
    \\    const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\    hd[ulong(rk) * ulong(INTC) + (o & ~15u) + 4u * q + m] = half(d * HAD_SCALE);
    \\
;

/// Form 3, launch 4: the down GEMV as split-K planes zd f32 [KS2, RK, hidden].
pub const text_down3 =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint INTC = uint(INTER);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint BTC = uint(BT);     // output tiles per threadgroup: 2 BT f32 accumulators per lane
    \\    constexpr uint IN_TILES = INTC / 16u;
    \\    constexpr uint OUT_TILES = HIDC / 16u;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint ks = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = slots[rk];
    \\    threadgroup float red[4][16u * BTC];
    \\    float a[BTC][2];
    \\    for (uint t = 0; t < BTC; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const device uint* wbase = (const device uint*)(down_t) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(N / 2u);
    \\    const device half* xrow = hdp + ulong(rk) * ulong(INTC) + 4u * (L & 3u);
    \\    const uint per = IN_TILES / (KSC * 4u);
    \\    const uint tk0 = (ks * 4u + sg) * per;
    \\    exl3lib_gemv_core<N, WMASK, CB, OUT_TILES, BTC>(wbase, xrow, tk0, tk0 + per, blk, L, a);
    \\    exl3lib_fold<BTC>(a, red[sg], L);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid < 16u * BTC) zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + blk * (16u * BTC) + tid] = ((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid];
    \\
;

/// Form 3, launch 5: per (128-wide hidden block, row): for each slot in order sum the planes, f16 round, H128, svh,
/// f16 round, times the score; y f32 [rows, hidden].
pub const text_dfin3 =
    \\    constexpr uint HIDC = uint(HID);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float xch[128];
    \\    const uint o = blk * 128u + tid;
    \\    float acc = 0.0f;
    \\    for (uint k = 0; k < TOPKC; ++k) {
    \\        const uint rk = row * TOPKC + k;
    \\        const uint e = slots[rk];
    \\        float v = 0.0f;
    \\        for (uint ks = 0; ks < KSC; ++ks) v += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
    \\        v = exl3lib_h128(float(half(v)), xch, tid, L);
    \\        acc = fma(scores[rk], float(half(v * HAD_SCALE * float(down_svh[e * HIDC + o]))), acc);
    \\    }
    \\    y[ulong(row) * ulong(HIDC) + o] = acc;
    \\
;

const gu3_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "up_t" };
const gu3_outputs = [_][*:0]const u8{ "zg", "zu" };
const fin3_inputs = [_][*:0]const u8{ "zg", "zu", "slots", "gate_svh", "up_svh", "down_suh" };
const fin3_outputs = [_][*:0]const u8{"hd"};
const down3_inputs = [_][*:0]const u8{ "hdp", "slots", "down_t" };
const down3_outputs = [_][*:0]const u8{"zd"};
const dfin3_inputs = [_][*:0]const u8{ "zd", "slots", "scores", "down_svh" };

/// Form 3: five launches (prepare gate, prepare up, split-K gate/up planes, finish + SwiGLU + down prepare, split-K
/// down planes, finish + reduce) -> six with the two prepares; KS slices fill the GPU at one row.
pub const Kernels3 = struct {
    geo: Geometry,
    ks_gu: u32,
    ks_dn: u32,
    /// output tiles per GEMV threadgroup (8, 4, 2 or 1): 2 bt f32 accumulators per lane, 128 / (16 bt) x more threadgroups
    bt: u32,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    dfin: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    /// K slices: gate/up over hidden/16 tiles in KS x 2 parts, down over inter/16 tiles in KS2 x 4 parts; each part
    /// at least 4 tiles.
    pub fn init(geo: Geometry, bt: u32, ks_max: u32) !Kernels3 {
        if (!geo.admitted() or (bt != 8 and bt != 4 and bt != 2 and bt != 1) or (ks_max != 1 and ks_max != 2 and ks_max != 4 and ks_max != 8)) return error.BadExl3Shape;
        var ks_gu: u32 = ks_max;
        while (ks_gu > 1 and (geo.hidden / 16) % (ks_gu * 2) != 0) ks_gu /= 2;
        var ks_dn: u32 = ks_max;
        while (ks_dn > 1 and ((geo.inter / 16) % (ks_dn * 4) != 0 or (geo.inter / 16) / (ks_dn * 4) < 2)) ks_dn /= 2;
        const unf = @import("unfused.zig");
        return .{
            .geo = geo,
            .ks_gu = ks_gu,
            .ks_dn = ks_dn,
            .bt = bt,
            .prep = ks.newKernel("exl3lib_prep", &prep_in, &prep_out, unf.text_prep, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("exl3lib_gu3", &gu3_inputs, &gu3_outputs, text_gu3, header, true) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("exl3lib_fin3", &fin3_inputs, &fin3_outputs, text_fin3, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("exl3lib_down3", &down3_inputs, &down3_outputs, text_down3, header, true) orelse return error.KernelCreateFailed,
            .dfin = ks.newKernel("exl3lib_dfin3", &dfin3_inputs, &down_outputs, text_dfin3, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels3) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gu, self.fin, self.down, self.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn cfgs(self: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = ks.TemplateArg{ .name = "HID", .value = .{ .int = H } };
        const inter_t = ks.TemplateArg{ .name = "INTER", .value = .{ .int = I } };
        const rkt = ks.TemplateArg{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks_dn_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_dn) } };
        const bt_t = ks.TemplateArg{ .name = "BT", .value = .{ .int = @intCast(self.bt) } };
        const bw: u32 = 16 * self.bt;
        const c: Cfg = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, topk } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / bw) * 128, rk, self.ks_gu }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I } }, .out_dtypes = &.{ .float32, .float32 }, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, bt_t } }),
            .fin = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ inter_t, rkt, ks_gu_t } }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / bw) * 128, rk, self.ks_dn }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.ks_dn), rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, bt_t } }),
            .dfin = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &.{ hid, topk, rkt, ks_dn_t } }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The six launches; the caller frees y.
    pub fn moe(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer _ = mlx.mlx_array_free(xg);
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer _ = mlx.mlx_array_free(xu);
        var planes: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(self.gu, c.gu, &.{ xg, xu, slots, bank.gate_t, bank.up_t }, &planes, s);
        defer _ = mlx.mlx_array_free(planes[0]);
        defer _ = mlx.mlx_array_free(planes[1]);
        const hd = try one(self.fin, c.fin, &.{ planes[0], planes[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh }, s);
        defer _ = mlx.mlx_array_free(hd);
        const zd = try one(self.down, c.down, &.{ hd, slots, bank.down_t }, s);
        defer _ = mlx.mlx_array_free(zd);
        return one(self.dfin, c.dfin, &.{ zd, slots, scores, bank.down_svh }, s);
    }
};
