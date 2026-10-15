//! The window's parity test of the fused KDA decode text (kda_decode.zig): GLM5NEXT_KDA_DECODE_GPU=1 (else skipped) and
//! _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child). One synthetic KDA layer at GLM-5.3-Flash's shapes
//! (random BF16 projections, conv and norm weights, the f32 gate tables as the bind builds them), each step against the
//! op chain it replaces (`kda.forward`):
//!   - from a zero state, a call of R = 1..4 rows: y and the next state within 2e-2 relative RMS of the op chain's
//!     (rounding-class: the conv output rounded to bf16 before the SiLU, the L2 norms' epsilon, the gate stages' order);
//!   - the state handed both ways: the op chain's state into the text and the text's into the op chain, a further row each,
//!     within the same bound (the two share the module's state layout);
//!   - a call of 4 rows bit for bit four 1-row calls carrying the state (the text's rows run in order).
//! Static bound (declared before any run): q / k / v / o 4 x 64 MiB BF16, the gate stages and conv < 8 MiB, their host
//! copies freed after each transfer, states 2 x 4 MiB per step: < 0.4 GB, plus the text's JIT; declared 0.8 GB.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const kda = @import("kda.zig");
const kda_decode = @import("kda_decode.zig");
const graph = @import("graph.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops_be.MlxOps;

const parity_bound_bytes: u64 = 800_000_000;

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn randBf16(g: *G, a: std.mem.Allocator, r: std.Random, shape: []const c_int, sd: f32, mean: f32) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try a.alloc(u16, n);
    defer a.free(v);
    for (v) |*x| x.* = bf16Bits(mean + sd * r.floatNorm(f32));
    const arr = g.keep(try g.hostArray(std.mem.sliceAsBytes(v), shape, .bfloat16));
    try g.evalAll(&.{arr});
    return arr;
}

fn hostF32(g: *G, a: std.mem.Allocator, x: T) ![]f32 {
    const f = try g.astype(x, .float32);
    try g.evalAll(&.{f});
    const out = try a.alloc(f32, mlx.mlx_array_size(f));
    _ = try g.hostF32(f, out);
    return out;
}

fn relRms(got: []const f32, want: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, want) |x, t| {
        num += (@as(f64, x) - t) * (@as(f64, x) - t);
        den += @as(f64, t) * t;
    }
    return @sqrt(num / @max(den, 1e-30));
}

fn compare(g: *G, a: std.mem.Allocator, label: []const u8, got: T, want: T, bound: f64) !void {
    const gv = try hostF32(g, a, got);
    defer a.free(gv);
    const wv = try hostF32(g, a, want);
    defer a.free(wv);
    const rel = relRms(gv, wv);
    std.debug.print("GLM5N_KDA_DECODE_PARITY {s} rel_rms={e}\n", .{ label, rel });
    try testing.expect(rel <= bound);
}

test "glm5next kda decode GPU: the fused text against the op chain (rows 1..4, the state handed both ways); 4 rows bit for bit four 1-row calls" {
    _ = std.c.getenv("GLM5NEXT_KDA_DECODE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_KDA_DECODE_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: glm.Config = .{};
    var prng = std.Random.DefaultPrng.init(0x6b6461);
    const r = prng.random();
    const h: c_int = @intCast(c.hidden_size);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim = hv * dk;
    const sd = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
    var kept: std.ArrayList(T) = .empty;
    defer {
        for (kept.items) |x| g.release(x);
        kept.deinit(a);
    }
    const W = struct {
        fn mk(gg: *G, al: std.mem.Allocator, rr: std.Random, list: *std.ArrayList(T), shape: []const c_int, d: f32, m: f32) !T {
            const x = try randBf16(gg, al, rr, shape, d, m);
            try list.append(al, x);
            return x;
        }
    };
    const conv_raw = try W.mk(&g, a, r, &kept, &.{ 3 * dim, 1, @intCast(c.kda_conv) }, 0.5, 0.0);
    const a_log = try W.mk(&g, a, r, &kept, &.{hv}, 0.5, 0.0);
    const dt_raw = try W.mk(&g, a, r, &kept, &.{dim}, 0.5, 0.0);
    // The f32 tables as the bind builds them (graph.zig `aExp`, `f32Copy`, `convLayout`).
    const a_exp = blk: {
        const f = try g.astype(a_log, .float32);
        const e = try g.exp(try g.reshape(try g.broadcastTo(try g.reshape(f, &.{ hv, 1 }), &.{ hv, dk }), &.{dim}));
        try g.evalAll(&.{e});
        break :blk g.keep(e);
    };
    try kept.append(a, a_exp);
    // A_log as its own F32 table [hv] (graph.zig `aLogTable`: the per-core prompt route's input; the decode routes read
    // a_exp).
    const a_log32 = blk: {
        const f = try g.astype(a_log, .float32);
        try g.evalAll(&.{f});
        break :blk g.keep(f);
    };
    try kept.append(a, a_log32);
    const dt_bias = try graph.f32Copy(s, dt_raw, false);
    defer _ = mlx.mlx_array_free(dt_bias);
    const conv_t = try g.transposeAxes(conv_raw, &.{ 0, 2, 1 });
    const conv_w = try graph.f32Copy(s, conv_t, false);
    defer _ = mlx.mlx_array_free(conv_w);
    const w: kda.Weights = .{
        .q = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dim, h }, sd, 0.0) },
        .k = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dim, h }, sd, 0.0) },
        .v = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dim, h }, sd, 0.0) },
        .f_a = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dk, h }, sd, 0.0) },
        .f_b = try W.mk(&g, a, r, &kept, &.{ dim, dk }, 0.1, 0.0),
        .g_a = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dk, h }, sd, 0.0) },
        .g_b = try W.mk(&g, a, r, &kept, &.{ dim, dk }, 0.1, 0.0),
        .b = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ hv, h }, sd, 0.0) },
        .a_log = a_log32,
        .a_exp = a_exp,
        .dt_bias = dt_bias,
        .conv_w = conv_w,
        .conv_raw = conv_raw,
        .o_norm = try W.mk(&g, a, r, &kept, &.{dk}, 0.2, 1.0),
        .o = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ h, dim }, 1.0 / @sqrt(@as(f32, @floatFromInt(dim))), 0.0) },
    };
    var diag: @import("sdk").Diag = .{};
    // The op chain's recurrence: its kernels and launches as the module installs them.
    try kda.install(&c);
    var k = kda_decode.install(a, s, &c, &.{&w}, &diag) catch |e| {
        std.debug.print("glm5next kda decode install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer k.deinit();

    for (1..kda_decode.max_rows + 1) |m| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const x = try randBf16(&g, a, r, &.{ 1, @intCast(m), h }, 1.0, 0.0);
        defer g.release(x);
        var s_ops: kda.State = .{};
        defer s_ops.reset(&g);
        var s_txt: kda.State = .{};
        defer s_txt.reset(&g);
        const y_ops = try kda.forward(&g, &c, &w, x, &s_ops);
        const y_txt = try kda_decode.step(&g, &k, &c, &w, x, &s_txt);
        var lb: [64]u8 = undefined;
        try compare(&g, a, try std.fmt.bufPrint(&lb, "rows={d} y", .{m}), y_txt, y_ops, 2e-2);
        try compare(&g, a, try std.fmt.bufPrint(&lb, "rows={d} state", .{m}), s_txt.rec.?, s_ops.rec.?, 2e-2);
        try compare(&g, a, try std.fmt.bufPrint(&lb, "rows={d} conv", .{m}), s_txt.conv.?, s_ops.conv.?, 0);
        // The state handed both ways: a further row each, from the other route's state.
        const x2 = try randBf16(&g, a, r, &.{ 1, 1, h }, 1.0, 0.0);
        defer g.release(x2);
        var cross_txt: kda.State = .{};
        defer cross_txt.reset(&g);
        cross_txt.set(&g, s_ops.conv.?, s_ops.rec.?);
        var cross_ops: kda.State = .{};
        defer cross_ops.reset(&g);
        cross_ops.set(&g, s_ops.conv.?, s_ops.rec.?);
        const y2_txt = try kda_decode.step(&g, &k, &c, &w, x2, &cross_txt);
        const y2_ops = try kda.forward(&g, &c, &w, x2, &cross_ops);
        try compare(&g, a, try std.fmt.bufPrint(&lb, "rows={d} handoff ops->text y", .{m}), y2_txt, y2_ops, 2e-2);
        var back_ops: kda.State = .{};
        defer back_ops.reset(&g);
        back_ops.set(&g, s_txt.conv.?, s_txt.rec.?);
        const y3_ops = try kda.forward(&g, &c, &w, x2, &back_ops);
        try compare(&g, a, try std.fmt.bufPrint(&lb, "rows={d} handoff text->ops y", .{m}), y3_ops, y2_ops, 2e-2);
        // Row order: R rows at once against R one-row calls carrying the state, bit for bit.
        if (m > 1) {
            var seq: kda.State = .{};
            defer seq.reset(&g);
            var ys: [kda_decode.max_rows]T = undefined;
            for (0..m) |i| {
                const ii: c_int = @intCast(i);
                const xi = try g.slice(x, &.{ 0, ii, 0 }, &.{ 1, ii + 1, h }, &.{ 1, 1, 1 });
                ys[i] = try kda_decode.step(&g, &k, &c, &w, xi, &seq);
            }
            const joined = try g.concat(ys[0..m], 1);
            const jv = try hostF32(&g, a, joined);
            defer a.free(jv);
            const tv = try hostF32(&g, a, y_txt);
            defer a.free(tv);
            for (jv, tv) |p, q| try testing.expectEqual(@as(u32, @bitCast(q)), @as(u32, @bitCast(p)));
        }
    }
}
