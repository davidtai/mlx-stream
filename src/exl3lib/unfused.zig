//! The unfused decode chain (the benchmark's `lib_steps` arm): the same math as `decode.zig`'s two launches, as the
//! separately launched steps a host would compose from GEMMs: per projection a prepare (f16 round, suh, H128, f16 round),
//! a raw trellis GEMV (f32 inner sums) and a finish (f16 round, H128, svh, f16 round); SwiGLU; the top-k weighted reduce.
//! Eleven launches per layer. Every kernel reads the same lane plan and decode helpers as the fused chain, so the
//! difference between the two arms is fusion alone (launch count, intermediate traffic), not the reader.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("../sdk_ext/kernel_set.zig");
const format = @import("format.zig");
const lane = @import("lane.zig");
const decode = @import("decode.zig");

/// prepare: xp half [RK, DIM] = H128(f16(x[row]) * suh[slot]) per 128-block, f16; one simdgroup per (block, row-slot)
pub const text_prep =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = slots[rk];
    \\    const uint row = rk / TOPKC;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(x[row * DIMC + i])) * float(suh[e * DIMC + i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) xp[rk * DIMC + b * 128u + L + 32u * j] = half(v[j] * HAD_SCALE);
    \\
;

/// raw GEMV: z f32 [RK, OUT] = xp[rk] . W[slot] (inner sums, unrounded); one 128-thread threadgroup per (128-wide out
/// block, row-slot), the four simdgroups split the input tiles, lanes read their x rows from device memory
pub const text_gemv =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint IN_TILES = INC / 16u;
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    constexpr bool INTK = (N % 16u) == 0u;
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    const uint e = slots[rk];
    \\    threadgroup float red[4][128];
    \\    const uint base = INTK ? exl3_int_qb<N>(L) : exl3_lane_base<N>(L);
    \\    const uint ia = INTK ? exl3_int_a<N>(L) : 0u;
    \\    const uint ib = INTK ? exl3_int_b<N>(L) : 0u;
    \\    uint off[8];
    \\    for (uint m = 0; m < 8u; ++m) off[m] = INTK ? 0u : exl3_lane_off<N>(L, m, base);
    \\    uint rows_of[4];
    \\    for (uint m = 0; m < 4u; ++m) rows_of[m] = exl3lib_row_of(L, m);
    \\    float a[8][2];
    \\    for (uint t = 0; t < 8u; ++t) { a[t][0] = 0.0f; a[t][1] = 0.0f; }
    \\    const uint tk_per = IN_TILES / 4u;
    \\    const device uint* wbase = (const device uint*)(w) + ulong(e) * ulong(IN_TILES) * ulong(OUT_TILES) * ulong(NW);
    \\    const device half* xrow = xp + ulong(rk) * ulong(INC);
    \\    for (uint tk = sg * tk_per; tk < (sg + 1u) * tk_per; ++tk) {
    \\        float xr[4];
    \\        for (uint m = 0; m < 4u; ++m) xr[m] = float(xrow[tk * 16u + rows_of[m]]);
    \\        for (uint t = 0; t < 8u; ++t) {
    \\            const uint tn = blk * 8u + t;
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + tn) * ulong(NW);
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
    \\    z[ulong(rk) * ulong(OUTC) + blk * 128u + tid] = ((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid];
    \\
;

/// finish: y f32 [RK, DIM] = f16(H128(f16(z)) * svh[slot]) per 128-block; one simdgroup per (block, row-slot)
pub const text_finish =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint rk = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint e = slots[rk];
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(half(z[rk * DIMC + b * 128u + L + 32u * j]));
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[rk * DIMC + o] = float(half(v[j] * HAD_SCALE * float(svh[e * DIMC + o]))); }
    \\
;

/// SwiGLU: h = silu(g) * u, elementwise over [RK, DIM]
pub const text_swiglu =
    \\    const uint i = thread_position_in_grid.x;
    \\    const float gv = g[i];
    \\    h[i] = (gv / (1.0f + metal::exp(-gv))) * u[i];
    \\
;

/// reduce: y f32 [R, DIM] = sum_k scores[r k] * yk[r k], in slot order
pub const text_reduce =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint TOPKC = uint(TOPK);
    \\    const uint d = thread_position_in_grid.x;
    \\    const uint r = thread_position_in_grid.y;
    \\    float acc = 0.0f;
    \\    for (uint k = 0; k < TOPKC; ++k) acc = fma(scores[r * TOPKC + k], yk[(r * TOPKC + k) * DIMC + d], acc);
    \\    y[r * DIMC + d] = acc;
    \\
;

pub const header: [:0]const u8 = decode.header;

const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
const prep_out = [_][*:0]const u8{"xp"};
const gemv_in = [_][*:0]const u8{ "xp", "slots", "w" };
const gemv_out = [_][*:0]const u8{"z"};
const finish_in = [_][*:0]const u8{ "z", "slots", "svh" };
const finish_out = [_][*:0]const u8{"y"};
const swiglu_in = [_][*:0]const u8{ "g", "u" };
const swiglu_out = [_][*:0]const u8{"h"};
const reduce_in = [_][*:0]const u8{ "yk", "scores" };
const reduce_out = [_][*:0]const u8{"y"};

/// The step kernels of one geometry and their configs per row count.
pub const Steps = struct {
    geo: decode.Geometry,
    prep: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,
    finish: mlx.mlx_fast_metal_kernel,
    swiglu: mlx.mlx_fast_metal_kernel,
    reduce: mlx.mlx_fast_metal_kernel,
    cfg: [decode.Kernels.max_rows + 1]?Cfgs = @splat(null),

    const Cfgs = struct {
        prep_h: mlx.mlx_fast_metal_kernel_config, // prepare over hidden (gate, up)
        prep_i: mlx.mlx_fast_metal_kernel_config, // prepare over inter (down)
        gemv_gu: mlx.mlx_fast_metal_kernel_config,
        gemv_d: mlx.mlx_fast_metal_kernel_config,
        fin_i: mlx.mlx_fast_metal_kernel_config,
        fin_h: mlx.mlx_fast_metal_kernel_config,
        swiglu: mlx.mlx_fast_metal_kernel_config,
        reduce: mlx.mlx_fast_metal_kernel_config,
    };

    pub fn init(geo: decode.Geometry) !Steps {
        if (!geo.admitted()) return error.BadExl3Shape;
        return .{
            .geo = geo,
            .prep = ks.newKernel("exl3lib_prep", &prep_in, &prep_out, text_prep, header, true) orelse return error.KernelCreateFailed,
            .gemv = ks.newKernel("exl3lib_gemv", &gemv_in, &gemv_out, text_gemv, header, true) orelse return error.KernelCreateFailed,
            .finish = ks.newKernel("exl3lib_finish", &finish_in, &finish_out, text_finish, header, true) orelse return error.KernelCreateFailed,
            .swiglu = ks.newKernel("exl3lib_swiglu", &swiglu_in, &swiglu_out, text_swiglu, header, true) orelse return error.KernelCreateFailed,
            .reduce = ks.newKernel("exl3lib_reduce", &reduce_in, &reduce_out, text_reduce, header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Steps) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.prep_i, cc.gemv_gu, cc.gemv_d, cc.fin_i, cc.fin_h, cc.swiglu, cc.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gemv, self.finish, self.swiglu, self.reduce }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn cfgs(self: *Steps, rows: usize) !Cfgs {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const dim_h = ks.TemplateArg{ .name = "DIM", .value = .{ .int = H } };
        const dim_i = ks.TemplateArg{ .name = "DIM", .value = .{ .int = I } };
        const c: Cfgs = .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float16}, .template = &.{ dim_h, topk } }),
            .prep_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float16}, .template = &.{ dim_i, ks.TemplateArg{ .name = "TOPK", .value = .{ .int = 1 } } } }),
            .gemv_gu = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, .{ .name = "IN", .value = .{ .int = H } }, .{ .name = "OUT", .value = .{ .int = I } } } }),
            .gemv_d = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, .{ .name = "IN", .value = .{ .int = I } }, .{ .name = "OUT", .value = .{ .int = H } } } }),
            .fin_i = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &.{dim_i} }),
            .fin_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{dim_h} }),
            .swiglu = try ks.buildConfig(.{ .grid = .{ rk * g.inter, 1, 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ rkc, I }}, .out_dtypes = &.{.float32}, .template = &.{} }),
            .reduce = try ks.buildConfig(.{ .grid = .{ g.hidden, @intCast(rows), 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{.float32}, .template = &.{ dim_h, topk } }),
        };
        self.cfg[rows] = c;
        return c;
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    fn drop(a: mlx.mlx_array) void {
        _ = mlx.mlx_array_free(a);
    }

    /// The eleven launches; the caller frees y.
    pub fn moe(self: *Steps, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: decode.Kernels.Bank, rows: usize) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        // gate
        const xg = try one(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh }, s);
        defer drop(xg);
        const zg = try one(self.gemv, c.gemv_gu, &.{ xg, slots, bank.gate_t }, s);
        defer drop(zg);
        const yg = try one(self.finish, c.fin_i, &.{ zg, slots, bank.gate_svh }, s);
        defer drop(yg);
        // up
        const xu = try one(self.prep, c.prep_h, &.{ x, slots, bank.up_suh }, s);
        defer drop(xu);
        const zu = try one(self.gemv, c.gemv_gu, &.{ xu, slots, bank.up_t }, s);
        defer drop(zu);
        const yu = try one(self.finish, c.fin_i, &.{ zu, slots, bank.up_svh }, s);
        defer drop(yu);
        // SwiGLU, then down over h as [RK, inter] rows (prepare with TOPK 1: row-slot i is its own x row)
        const h = try one(self.swiglu, c.swiglu, &.{ yg, yu }, s);
        defer drop(h);
        const hd = try one(self.prep, c.prep_i, &.{ h, slots, bank.down_suh }, s);
        defer drop(hd);
        const zd = try one(self.gemv, c.gemv_d, &.{ hd, slots, bank.down_t }, s);
        defer drop(zd);
        const yk = try one(self.finish, c.fin_h, &.{ zd, slots, bank.down_svh }, s);
        defer drop(yk);
        return one(self.reduce, c.reduce, &.{ yk, scores }, s);
    }
};

test "exl3lib unfused texts: the inputs each signature names are read, the outputs written" {
    const t = std.testing;
    for (prep_in) |n| try t.expect(std.mem.indexOf(u8, text_prep, std.mem.span(n)) != null);
    for (gemv_in) |n| try t.expect(std.mem.indexOf(u8, text_gemv, std.mem.span(n)) != null);
    for (finish_in) |n| try t.expect(std.mem.indexOf(u8, text_finish, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_prep, "xp[rk * DIMC") != null);
    try t.expect(std.mem.indexOf(u8, text_gemv, "z[ulong(rk) * ulong(OUTC)") != null);
    try t.expect(std.mem.indexOf(u8, text_reduce, "y[r * DIMC + d] = acc") != null);
}
