//! A dense EXL3 linear (one trellis, no experts): y [rows, out] = finish(innerGemv(prepare(x))) with exllamav3's rounding
//! points, for the trunk of a fully EXL3 model (attention / GDN projections, shared expert, lm_head, MTP). Decode rows
//! (<= DECODE_ROWS_MAX) run the row-slot kernels of `unfused.zig` with a single expert and TOPK 1: prepare (f16 round, suh,
//! H128), the trellis GEMV, finish (H128, svh, f16 round). Wider calls dequantize the trellis to an f16 W_hat [in, out] once,
//! run MLX's matmul on the prepared rows (f32 accumulation, f16 output = the finish's first rounding) and finish. Output in
//! x's dtype. Held to `format.project` per row.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("../sdk_ext/kernel_set.zig");
const format = @import("format.zig");
const decode = @import("decode.zig");
const unfused = @import("unfused.zig");

/// prepare for a dense linear: xp half [rows, IN] = H128(f16(x[row]) * suh) per 128-block; LANE_ORDER 1 writes the GEMV's
/// lane order within each 16-block, 0 the natural order the matmul path needs; one simdgroup per (block, row); x any float dtype
pub const text_prep =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint r = i & 15u;
    \\        const uint q = (r & 7u) >> 1u;
    \\        const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\        const uint dst = (LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i;
    \\        xp[row * DIMC + dst] = half(v[j] * HAD_SCALE);
    \\    }
    \\
;

/// finish for a dense linear: y OUT_T [rows, OUT] = f16(H128(f16(z)) * svh) per 128-block; z f32 (decode) or f16 (prefill)
pub const text_finish =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[row * DIMC + b * 128u + L + 32u * j])));
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;

/// dequantize: W half [IN, OUT] from the trellis [IN/16, OUT/16, n]; one simdgroup per tile, lane L decodes its 8 weights:
/// weight m of lane L is row 2 (L & 3) + {0, 1, 8, 9}[m & 3], column (L >> 2) + 8 (m >> 2) of the tile (the GEMV's layout)
pub const text_dequant =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    const uint tile_id = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    if (tile_id >= (INC / 16u) * OUT_TILES) return;
    \\    const uint tk = tile_id / OUT_TILES;
    \\    const uint tn = tile_id - tk * OUT_TILES;
    \\    const device uint* tile = (const device uint*)(w) + ulong(tile_id) * ulong(NW);
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    const IntWords ww = exl3_int_words<N>(tile, qb, ga);
    \\    uint cw[8];
    \\    cw[0] = exl3_gen_cw<N, WMASK, 0>(ww, lp); cw[1] = exl3_gen_cw<N, WMASK, 1>(ww, lp); cw[2] = exl3_gen_cw<N, WMASK, 2>(ww, lp); cw[3] = exl3_gen_cw<N, WMASK, 3>(ww, lp);
    \\    cw[4] = exl3_gen_cw<N, WMASK, 4>(ww, lp); cw[5] = exl3_gen_cw<N, WMASK, 5>(ww, lp); cw[6] = exl3_gen_cw<N, WMASK, 6>(ww, lp); cw[7] = exl3_gen_cw<N, WMASK, 7>(ww, lp);
    \\    for (uint m = 0; m < 8u; ++m) {
    \\        const uint r = exl3lib_row_of(L, m & 3u);
    \\        const uint c = (L >> 2u) + 8u * (m >> 2u);
    \\        W[ulong(tk * 16u + r) * ulong(OUTC) + tn * 16u + c] = exl3_decode<CB>(cw[m]);
    \\    }
    \\
;

const prep_in = [_][*:0]const u8{ "x", "suh" };
const prep_out = [_][*:0]const u8{"xp"};
const gemv_in = [_][*:0]const u8{ "xp", "slots", "w" };
const gemv_out = [_][*:0]const u8{"z"};
const finish_in = [_][*:0]const u8{ "z", "svh" };
const finish_out = [_][*:0]const u8{"y"};
const dequant_in = [_][*:0]const u8{"w"};
const dequant_out = [_][*:0]const u8{"W"};

pub const Shape = struct { in: u32, out: u32, rate: format.Rate, dec: format.Decode, out_dtype: mlx.mlx_dtype };

/// The kernels of one dense linear shape; configs per row count (decode) plus the prefill set.
pub const Dense = struct {
    shape: Shape,
    prep: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,
    finish: mlx.mlx_fast_metal_kernel,
    dequant: mlx.mlx_fast_metal_kernel,
    cfg: [decode.Kernels.max_rows + 1]?Cfg = @splat(null),
    dequant_cfg: ?mlx.mlx_fast_metal_kernel_config = null,

    const Cfg = struct { prep: mlx.mlx_fast_metal_kernel_config, gemv: mlx.mlx_fast_metal_kernel_config, finish: mlx.mlx_fast_metal_kernel_config };

    pub fn init(shape: Shape) !Dense {
        if (shape.in % 128 != 0 or shape.out % 128 != 0 or (shape.in / 16) % 4 != 0 or shape.in == 0 or shape.out == 0) return error.BadExl3Shape;
        return .{
            .shape = shape,
            .prep = ks.newKernel("exl3lib_dense_prep", &prep_in, &prep_out, text_prep, decode.header, true) orelse return error.KernelCreateFailed,
            .gemv = ks.newKernel("exl3lib_dense_gemv", &gemv_in, &gemv_out, unfused.text_gemv, decode.header, true) orelse return error.KernelCreateFailed,
            .finish = ks.newKernel("exl3lib_dense_finish", &finish_in, &finish_out, text_finish, decode.header, true) orelse return error.KernelCreateFailed,
            .dequant = ks.newKernel("exl3lib_dense_dequant", &dequant_in, &dequant_out, text_dequant, decode.header, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Dense) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep, cc.gemv, cc.finish }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        if (self.dequant_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        inline for (.{ self.prep, self.gemv, self.finish, self.dequant }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn templ(self: *const Dense) [3]ks.TemplateArg {
        return .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.shape.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.shape.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.shape.dec.window.bits()) } },
        };
    }

    /// prepare and finish configs for `rows` rows (any count); the GEMV config only up to max_rows
    fn cfgs(self: *Dense, rows: usize) !Cfg {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const c = try self.buildCfg(rows, true);
        self.cfg[rows] = c;
        return c;
    }

    fn buildCfg(self: *const Dense, rows: usize, with_gemv: bool) !Cfg {
        const r: c_int = @intCast(rows);
        const IN: c_int = @intCast(self.shape.in);
        const OUT: c_int = @intCast(self.shape.out);
        const t = self.templ();
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } };
        return .{
            .prep = try ks.buildConfig(.{ .grid = .{ (self.shape.in / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, IN }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = IN } }, .{ .name = "LANE_ORDER", .value = .{ .int = if (with_gemv) 1 else 0 } } } }),
            .gemv = if (with_gemv) try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{.float32}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = IN } }, .{ .name = "OUT", .value = .{ .int = OUT } } } }) else undefined,
            .finish = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{self.shape.out_dtype}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, out_t } }),
        };
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// W half [in, out] decoded from the trellis (the prefill operand); the caller frees it.
    pub fn dequantize(self: *Dense, s: mlx.mlx_stream, trellis: mlx.mlx_array) !mlx.mlx_array {
        if (self.dequant_cfg == null) {
            const tiles = (self.shape.in / 16) * (self.shape.out / 16);
            const t = self.templ();
            self.dequant_cfg = try ks.buildConfig(.{ .grid = .{ ((tiles + 3) / 4) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.shape.in), @intCast(self.shape.out) }}, .out_dtypes = &.{.float16}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = @intCast(self.shape.in) } }, .{ .name = "OUT", .value = .{ .int = @intCast(self.shape.out) } } } });
        }
        return one(self.dequant, self.dequant_cfg.?, &.{trellis}, s);
    }

    /// y [rows, out] in `shape.out_dtype` for x [rows, in] (any float dtype); the caller frees y.
    pub fn linear(self: *Dense, s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        if (rows <= decode.Kernels.max_rows) {
            const c = try self.cfgs(rows);
            const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
            defer _ = mlx.mlx_array_free(xp);
            // every row-slot is row r of expert 0: a zero slot per row
            var slots = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(slots);
            try mlx.check(mlx.mlx_zeros(&slots, &[_]c_int{@intCast(rows)}, 1, .uint32, s));
            const z = try one(self.gemv, c.gemv, &.{ xp, slots, trellis }, s);
            defer _ = mlx.mlx_array_free(z);
            return one(self.finish, c.finish, &.{ z, svh }, s);
        }
        // prefill: prepared rows x dequantized W in f16 (f32 accumulation inside MLX's GEMM, f16 output = the finish's
        // first rounding), then finish
        const c = try self.buildCfg(rows, false);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
        const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
        defer _ = mlx.mlx_array_free(xp);
        const W = try self.dequantize(s, trellis);
        defer _ = mlx.mlx_array_free(W);
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_matmul(&inner, xp, W, s));
        return one(self.finish, c.finish, &.{ inner, svh }, s);
    }
};

test "exl3lib dense texts: inputs read, outputs written, the dequant layout names the GEMV's row/column rule" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, text_prep, "xp[row * DIMC") != null);
    try t.expect(std.mem.indexOf(u8, text_finish, "y[row * DIMC + o] = OUT_T(") != null);
    try t.expect(std.mem.indexOf(u8, text_dequant, "exl3lib_row_of(L, m & 3u)") != null);
    try t.expect(std.mem.indexOf(u8, text_dequant, "(L >> 2u) + 8u * (m >> 2u)") != null);
}
