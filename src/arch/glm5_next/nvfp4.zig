//! glm5_next's routed-expert projection format for MiMo's expert math (`mimo_v2/math.zig` `MathOf`): the NVFP4 bank's
//! slot arrays (bank.zig: E2M1 words [slots, out, in / 8] u32, E4M3 group scales [slots, out, in / 16] u8, one f32
//! global scale per slot [slots]) and MLX's gather_qmm over them in nvfp4 mode (group 16, bits 4) with each slot's global
//! scale. mlx-c's `mlx_gather_qmm` has no argument for the global scale (and its signature stays as it is), so the gathers
//! go through the plugin's shim `glm5_gather_qmm_nvfp4` (csrc/glm5_gather_qmm.inc, compiled into the plugin's C++ shim
//! with the rest of csrc): MLX's own C++ `mx::gather_qmm(..., "nvfp4", global_scale, sorted)`, so the scale applies
//! exactly where MLX's kernels fold it into each group scale (`scale * (g / 2688)`, then the code), with no output
//! scaling and no extra rounding. Every gather route MLX picks (gather_qmv / gather_qmv_fast for the decode rows, the
//! sorted NAX GEMM for the prompt tiles) has its nvfp4 global-scale (`_hgs`) instances at the pinned MLX.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const bank = @import("bank.zig");

const T = mlx.mlx_array;

extern "c" fn glm5_gather_qmm_nvfp4(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, scales: mlx.mlx_array, global_scale: mlx.mlx_array, rhs_indices: mlx.mlx_array, sorted: bool, s: mlx.mlx_stream) c_int;
extern "c" fn dsv41_alloc_last_error() [*:0]const u8;

/// `MathOf`'s format (`mimo_v2/math.zig` `Mxfp4` is MiMo's).
pub const Format = struct {
    /// The decode lane's fused texts: none yet (the stock gathers; the fused NVFP4 set is a later construction route).
    pub const has_fused = false;
    /// A prompt call's tokens at most (the module's calls: `graph.prompt_call_rows`); the math's row scratch holds
    /// their routed rows.
    pub const max_call_tokens: u32 = 16_384;

    pub fn Projection(comptime A: type) type {
        return bank.Arrays(A);
    }

    /// MLX's gather_qmm in nvfp4 mode with the slots' global scales: x [.., M, in] against the slots `rhs` names.
    fn gatherQmm(g: *ops.MlxOps, x: T, p: Projection(T), rhs: T, sorted: bool) !T {
        var res = mlx.mlx_array_new();
        if (glm5_gather_qmm_nvfp4(&res, x, p.w, p.scales, p.global_scale, rhs, sorted, g.s) != 0) {
            _ = mlx.mlx_array_free(res);
            sdk.log.err("glm5_next: gather_qmm nvfp4: {s}\n", .{std.mem.span(dsv41_alloc_last_error())});
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

    /// Bank rows [lo, hi) of one projection: leading-axis slices of the row-contiguous slot arrays (shared-buffer views).
    pub fn window(g: *ops.MlxOps, p: Projection(T), lo: u32, hi: u32) !Projection(T) {
        const ws = g.shapeOf(p.w);
        const ss = g.shapeOf(p.scales);
        const l: c_int = @intCast(lo);
        const h: c_int = @intCast(hi);
        return .{
            .w = try g.slice(p.w, &.{ l, 0, 0 }, &.{ h, ws.d[1], ws.d[2] }, &.{ 1, 1, 1 }),
            .scales = try g.slice(p.scales, &.{ l, 0, 0 }, &.{ h, ss.d[1], ss.d[2] }, &.{ 1, 1, 1 }),
            .global_scale = try g.slice(p.global_scale, &.{l}, &.{h}, &.{1}),
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

/// A projection's slot arrays as the gathers take them: [slots, out, in / 8] u32, [slots, out, in / 16] u8, [slots] f32.
/// Construction only (each routed layer's banks, once).
pub fn checkArrays(g: *ops.MlxOps, which: []const u8, w: bank.Arrays(T), in: u32, out: u32) !void {
    const sw = g.shapeOf(w.w);
    const ss = g.shapeOf(w.scales);
    const sg = g.shapeOf(w.global_scale);
    const outc: c_int = @intCast(out);
    if (sw.n != 3 or sw.d[1] != outc or sw.d[2] != @as(c_int, @intCast(in / 8)) or g.dtypeOf(w.w) != .uint32 or
        ss.n != 3 or ss.d[0] != sw.d[0] or ss.d[1] != outc or ss.d[2] != @as(c_int, @intCast(in / bank.group_size)) or g.dtypeOf(w.scales) != .uint8 or
        sg.n != 1 or sg.d[0] != sw.d[0] or g.dtypeOf(w.global_scale) != .float32)
    {
        sdk.log.err("glm5_next: {s} slot arrays are {any} {any} {any}, not the nvfp4 bank's [slots, {d}, {d}] u32, [slots, {d}, {d}] u8, [slots] f32\n", .{ which, sw.slice(), ss.slice(), sg.slice(), out, in / 8, out, in / bank.group_size });
        return error.Glm5NextSlotArrays;
    }
}

const testing = std.testing;

test "glm5_next nvfp4 format: the event aliases are the projection's three arrays, in order" {
    const E = Format.EventProjection(u32);
    const p: Format.Projection(u32) = .{ .w = 7, .scales = 11, .global_scale = 13 };
    try testing.expectEqual(@as(usize, 3), E.count);
    try testing.expectEqualSlices(u32, &.{ 7, 11, 13 }, &E.inputs(p));
    const back = E.fromWaited(.{ 17, 19, 23 });
    try testing.expectEqual(@as(u32, 17), back.w);
    try testing.expectEqual(@as(u32, 19), back.scales);
    try testing.expectEqual(@as(u32, 23), back.global_scale);
}
