//! The NVFP4 projection format for MiMo's expert math (`mimo_v2/math.zig` `MathOf`), shared by every NVFP4-streamed arch
//! (glm5_next, beam): the NVFP4 bank's slot arrays (nvfp4_bank.zig: E2M1 words [slots, out, in / 8] u32, E4M3 group
//! scales [slots, out, in / 16] u8, one f32 global scale per slot on its own 128-byte line [>= slots, 32]) and MLX's
//! gather_qmm over them in nvfp4 mode (group 16, bits 4) with each slot's global scale. mlx-c's `mlx_gather_qmm` has no
//! argument for the global scale (and its signature stays as it is), so the gathers go through the plugin's shim
//! `glm5_gather_qmm_nvfp4` (csrc/glm5_gather_qmm.inc, compiled into the plugin's C++ shim with the rest of csrc): MLX's
//! own C++ `mx::gather_qmm(..., "nvfp4", global_scale, sorted)`, so the scale applies exactly where MLX's kernels fold it
//! into each group scale (`scale * (g / 2688)`, then the code), with no output scaling and no extra rounding. Every
//! gather route MLX picks (gather_qmv / gather_qmv_fast for the decode rows, the sorted NAX GEMM for the prompt tiles)
//! has its nvfp4 global-scale (`_hgs`) instances at the pinned MLX, at any geometry.
//!
//! An arch takes `FormatOf(.{ ... })`: with its fused decode texts (`Fused`, geometry-checked at their install:
//! glm5_next's `decode_kernels.zig`) or without (`Fused = null`: the stock gathers on every route, beam's).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../arch/deepseek_v41/ops.zig");
const bank = @import("nvfp4_bank.zig");

const T = mlx.mlx_array;

extern "c" fn glm5_gather_qmm_nvfp4(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, scales: mlx.mlx_array, global_scale: mlx.mlx_array, rhs_indices: mlx.mlx_array, sorted: bool, s: mlx.mlx_stream) c_int;
extern "c" fn dsv41_alloc_last_error() [*:0]const u8;

pub const Params = struct {
    /// The decode lane's fused texts over the bank (`MathOf`'s `F.Fused`), or null: the stock gathers.
    Fused: ?type = null,
    /// A prompt call's tokens at most (the arch's prompt calls); the math's row scratch holds their routed rows.
    max_call_tokens: u32 = 16_384,
};

/// `MathOf`'s format (`mimo_v2/math.zig` `Mxfp4` is MiMo's).
pub fn FormatOf(comptime params: Params) type {
    return struct {
        pub const has_fused = params.Fused != null;
        pub const Fused = if (params.Fused) |F| F else void;
        pub const max_call_tokens: u32 = params.max_call_tokens;

        pub fn Projection(comptime A: type) type {
            return bank.Arrays(A);
        }

        /// MLX's gather_qmm in nvfp4 mode with the slots' global scales: x [.., M, in] against the slots `rhs` names.
        fn gatherQmm(g: *ops.MlxOps, x: T, p: Projection(T), rhs: T, sorted: bool) !T {
            const gs = try perExpertScale(g, p);
            var res = mlx.mlx_array_new();
            if (glm5_gather_qmm_nvfp4(&res, x, p.w, p.scales, gs, rhs, sorted, g.s) != 0) {
                _ = mlx.mlx_array_free(res);
                sdk.log.err("nvfp4: gather_qmm nvfp4: {s}\n", .{std.mem.span(dsv41_alloc_last_error())});
                return error.MlxError;
            }
            return g.adopt(res);
        }

        /// A prompt tile's sorted gather: x [rows, in], rhs u32 [rows] sorted by slot -> [rows, out] (MiMo's
        /// `GatherQmm.gatherMatmul` shapes: each row its own [1, in] matrix).
        pub fn sortedGather(g: *ops.MlxOps, x: T, p: Projection(T), rhs: T) !T {
            const sx = g.shapeOf(x);
            const rows = sx.d[0];
            const y = try gatherQmm(g, try g.reshape(x, &.{ rows, 1, sx.d[1] }), p, rhs, true);
            const sy = g.shapeOf(y);
            return g.reshape(y, &.{ rows, sy.d[sy.n - 1] });
        }

        /// The decode lane's gather (and `matrix`'s one bank row): x [n, 1, in] (or [1, count, in]) against `ids`, unsorted.
        pub fn gather(g: *ops.MlxOps, x: T, p: Projection(T), ids: T) !T {
            return gatherQmm(g, x, p, ids, false);
        }

        pub const perExpertScale = perExpertScaleOf;

        /// Bank rows [lo, hi) of one projection: leading-axis slices of the row-contiguous slot arrays (shared-buffer
        /// views; the global scales' lines whole).
        pub fn window(g: *ops.MlxOps, p: Projection(T), lo: u32, hi: u32) !Projection(T) {
            const ws = g.shapeOf(p.w);
            const ss = g.shapeOf(p.scales);
            const l: c_int = @intCast(lo);
            const h: c_int = @intCast(hi);
            return .{
                .w = try g.slice(p.w, &.{ l, 0, 0 }, &.{ h, ws.d[1], ws.d[2] }, &.{ 1, 1, 1 }),
                .scales = try g.slice(p.scales, &.{ l, 0, 0 }, &.{ h, ss.d[1], ss.d[2] }, &.{ 1, 1, 1 }),
                .global_scale = try g.slice(p.global_scale, &.{ l, 0 }, &.{ h, bank.scale_line_floats }, &.{ 1, 1 }),
            };
        }

        /// The event gates' aliases: a projection's three arrays (its global scale lands in the span its weights do).
        pub fn EventProjection(comptime A: type) type {
            return struct {
                pub const count: usize = 3;
                pub fn inputs(p: Projection(A)) [count]A {
                    return .{ p.w, p.scales, p.global_scale };
                }
                pub fn fromWaited(out: [count]A) Projection(A) {
                    return .{ .w = out[0], .scales = out[1], .global_scale = out[2] };
                }
            };
        }
    };
}

/// MLX's per-expert global scale of `p`: column 0 (each slot's word) of the first `slots` rows of the line-isolated
/// array [>= slots, 32] (nvfp4_bank.zig `scale_line_bytes`), f32 [slots], `slots` the words' leading dimension as MLX's
/// gather_qmm requires. A strided view: MLX's GatherQMM copies it row-contiguous on the GPU inside the op (after any
/// event wait the arrays came through), the same words, so the arithmetic is the packed array's.
pub fn perExpertScaleOf(g: *ops.MlxOps, p: bank.Arrays(T)) !T {
    const n = g.shapeOf(p.w).d[0];
    return g.reshape(try g.slice(p.global_scale, &.{ 0, 0 }, &.{ n, 1 }, &.{ 1, 1 }), &.{n});
}

/// A projection's slot arrays as the gathers take them: [slots, out, in / 8] u32, [slots, out, in / 16] u8, and the
/// line-isolated global scales [>= slots, 32] f32 of at least a page (nvfp4_bank.zig `scale_line_bytes`). Construction
/// only (each routed layer's banks, once); `arch` names the log line. Refused as `Nvfp4SlotArrays`.
pub fn checkArrays(comptime arch: []const u8, g: *ops.MlxOps, which: []const u8, w: bank.Arrays(T), in: u32, out: u32) !void {
    const sw = g.shapeOf(w.w);
    const ss = g.shapeOf(w.scales);
    const sg = g.shapeOf(w.global_scale);
    const outc: c_int = @intCast(out);
    if (sw.n != 3 or sw.d[1] != outc or sw.d[2] != @as(c_int, @intCast(in / 8)) or g.dtypeOf(w.w) != .uint32 or
        ss.n != 3 or ss.d[0] != sw.d[0] or ss.d[1] != outc or ss.d[2] != @as(c_int, @intCast(in / bank.group_size)) or g.dtypeOf(w.scales) != .uint8 or
        !scaleShapeOk(sg.slice(), sw.d[0]) or g.dtypeOf(w.global_scale) != .float32)
    {
        sdk.log.err(arch ++ ": {s} slot arrays are {any} {any} {any}, not the nvfp4 bank's [slots, {d}, {d}] u32, [slots, {d}, {d}] u8, [>= slots, {d}] f32 (>= {d} B)\n", .{ which, sw.slice(), ss.slice(), sg.slice(), out, in / 8, out, in / bank.group_size, bank.scale_line_floats, bank.scale_array_min_bytes });
        return error.Nvfp4SlotArrays;
    }
}

/// A global-scale array's shape for `slots` slots: [rows, 32] f32, rows >= slots, at least `scale_array_min_bytes`.
pub fn scaleShapeOk(shape: []const c_int, slots: c_int) bool {
    if (shape.len != 2 or shape[1] != bank.scale_line_floats or shape[0] < slots) return false;
    return @as(u64, @intCast(shape[0])) * bank.scale_line_bytes >= bank.scale_array_min_bytes;
}

const testing = std.testing;

test "nvfp4 format: the event aliases are the projection's three arrays, in order; no fused texts without a Fused type" {
    const F = FormatOf(.{});
    try testing.expect(!F.has_fused);
    try testing.expectEqual(@as(u32, 16_384), F.max_call_tokens);
    const E = F.EventProjection(u32);
    const p: F.Projection(u32) = .{ .w = 7, .scales = 11, .global_scale = 13 };
    try testing.expectEqual(@as(usize, 3), E.count);
    try testing.expectEqualSlices(u32, &.{ 7, 11, 13 }, &E.inputs(p));
    const back = E.fromWaited(.{ 17, 19, 23 });
    try testing.expectEqual(@as(u32, 17), back.w);
    try testing.expectEqual(@as(u32, 19), back.scales);
    try testing.expectEqual(@as(u32, 23), back.global_scale);
}

test "nvfp4 format: a global-scale array is line-isolated: [>= slots, 32] f32 of a page at least, never packed [slots]" {
    try testing.expect(scaleShapeOk(&.{ 32, 32 }, 3));
    try testing.expect(scaleShapeOk(&.{ 131, 32 }, 131));
    try testing.expect(scaleShapeOk(&.{ 288, 32 }, 288));
    // Packed (the old layout), fewer rows than slots, another line, under a page.
    try testing.expect(!scaleShapeOk(&.{131}, 131));
    try testing.expect(!scaleShapeOk(&.{ 130, 32 }, 131));
    try testing.expect(!scaleShapeOk(&.{ 131, 1 }, 131));
    try testing.expect(!scaleShapeOk(&.{ 3, 32 }, 3));
}
