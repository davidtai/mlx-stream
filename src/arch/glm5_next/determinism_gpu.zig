//! The window's determinism check of the fused decode routes: GLM5NEXT_DETERMINISM_GPU=1 (else skipped) and
//! _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child). NEEDS A WINDOW: it launches every fused text on
//! the GPU. Each route runs `runs` times on the same inputs, each run a fresh graph evaluated alone; every output of every
//! run must equal run 0's bit for bit (f32 views: a bf16 or int32 value widens to f32 exactly). Covered:
//!   - decode_glue: hcPre on a stream, its deferred variant (the cross-threadgroup reduction and the tail), materialize,
//!     the router + top-8 texts, the combine with the shared add;
//!   - decode_kda: the KDA text over one layer's random tables from the zero state (y and both states);
//!   - decode_dsa: the split / merge texts at S = 1 and 4 over a 4,096-row lane;
//!   - decode_swiglu: the activation text at both widths.
//! Not covered here (no fixture outside the served stream): decode_moe hitpass (the device hit wave against the host's
//! route over arriving reads) and decode_predict (MLX ops; it picks reads only): those take the served two-run check.
//! Static bound (declared before any run): tables 0.1 GB (the router 2.4 MB, fn 6.3 MB, KDA projections 34 MB), lanes and
//! outputs < 0.1 GB; declared 0.3 GB.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const ops = @import("ops.zig");
const glue = @import("glue.zig");
const kda = @import("kda.zig");
const kda_decode = @import("kda_decode.zig");
const dsa_decode = @import("dsa_decode.zig");
const swiglu = @import("swiglu.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops_be.MlxOps;

const bound_bytes: u64 = 300_000_000;
const runs: usize = 64;

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn rand(g: *G, a: std.mem.Allocator, r: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, sd: f32, mean: f32) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const x = switch (dt) {
        .bfloat16 => blk: {
            const v = try a.alloc(u16, n);
            defer a.free(v);
            for (v) |*e| e.* = bf16Bits(mean + sd * r.floatNorm(f32));
            break :blk try g.hostArray(std.mem.sliceAsBytes(v), shape, .bfloat16);
        },
        .float32 => blk: {
            const v = try a.alloc(f32, n);
            defer a.free(v);
            for (v) |*e| e.* = mean + sd * r.floatNorm(f32);
            break :blk try g.hostArray(std.mem.sliceAsBytes(v), shape, .float32);
        },
        else => return error.Unsupported,
    };
    const k = g.keep(x);
    try g.evalAll(&.{k});
    return k;
}

/// The outputs of one run as f32 bits (owned by `a`).
fn bits(g: *G, a: std.mem.Allocator, outs: []const T) ![]u32 {
    var total: usize = 0;
    for (outs) |o| total += mlx.mlx_array_size(o);
    const all = try a.alloc(u32, total);
    errdefer a.free(all);
    var at: usize = 0;
    for (outs) |o| {
        const f = try g.astype(o, .float32);
        try g.evalAll(&.{f});
        const n = mlx.mlx_array_size(f);
        const buf = try a.alloc(f32, n);
        defer a.free(buf);
        _ = try g.hostF32(f, buf);
        for (buf, 0..) |v, i| all[at + i] = @bitCast(v);
        at += n;
    }
    return all;
}

/// `runs` runs of `body` (a fresh graph each, its outputs evaluated), each compared with run 0: the first differing
/// element printed with the route's name.
fn expectRunsEqual(g: *G, a: std.mem.Allocator, name: []const u8, ctx: anytype, comptime body: fn (@TypeOf(ctx), *G, *[8]T) anyerror!usize) !void {
    var first: ?[]u32 = null;
    defer if (first) |f| a.free(f);
    for (0..runs) |run| {
        const m = g.mark();
        defer g.resetTo(m);
        var outs: [8]T = undefined;
        const n = try body(ctx, g, &outs);
        try g.evalAll(outs[0..n]);
        const b = try bits(g, a, outs[0..n]);
        if (first) |f| {
            defer a.free(b);
            if (!std.mem.eql(u32, f, b)) {
                const i = std.mem.indexOfDiff(u32, f, b).?;
                std.debug.print("GLM5N_DETERMINISM route={s} run={d} differs at element {d}: {x} vs {x}\n", .{ name, run, i, f[i], b[i] });
                return error.Glm5NextNonDeterministic;
            }
        } else first = b;
    }
    std.debug.print("GLM5N_DETERMINISM route={s} runs={d} bit-equal\n", .{ name, runs });
}

const GlueCtx = struct { k: *const glue.Kernels, hidden: c_int, stream: T, y: T, fn_rows: T, scale: T, base: T, norm_w: T, router_w: T, bias: T, routed: T, shared: T };

fn glueRun(c: GlueCtx, g: *G, outs: *[8]T) anyerror!usize {
    const p0 = try glue.pre(g, c.k, c.stream, null, c.fn_rows, c.scale, c.base, c.norm_w);
    const df: ops.Deferred = .{ .y = c.y, .post = p0.post, .comb = p0.comb, .h = c.stream };
    const p1 = try glue.pre(g, c.k, null, &df, c.fn_rows, c.scale, c.base, c.norm_w);
    const mat = try g.adopt(try ops.materialize(g.s, &c.k.hc, &df));
    const x = try g.reshape(p1.normed, &.{ 1, c.hidden });
    const r = try glue.route(g, c.k, c.router_w, c.bias, x, 1);
    const y = try glue.combineShared(g, c.k, c.routed, r.weights, c.shared, 1);
    outs[0..8].* = .{ p0.normed, p0.post, p0.comb, p1.normed, p1.h.?, mat, r.ids, y };
    return 8;
}

const KdaCtx = struct { k: *const kda_decode.Kernels, c: *const glm.Config, w: *const kda.Weights, x: T };

fn kdaRun(c: KdaCtx, g: *G, outs: *[8]T) anyerror!usize {
    var st: kda.State = .{};
    const y = try kda_decode.step(g, c.k, c.c, c.w, c.x, &st);
    outs[0] = y;
    outs[1] = st.conv.?;
    outs[2] = st.rec.?;
    return 3;
}

const DsaCtx = struct { k: *const dsa_decode.Kernels, q: T, lat: T, sel: T, rows: u32 };

fn dsaRun(c: DsaCtx, g: *G, outs: *[8]T) anyerror!usize {
    outs[0] = try dsa_decode.attend(g, c.k, c.q, c.lat, c.sel, c.rows);
    return 1;
}

const SwigluCtx = struct { k: *const swiglu.Kernels, w: swiglu.Width, gate: T, up: T };

fn swigluRun(c: SwigluCtx, g: *G, outs: *[8]T) anyerror!usize {
    outs[0] = try swiglu.activate(g, c.k, c.w, c.gate, c.up, 1);
    return 1;
}

test "glm5next determinism GPU (NEEDS A WINDOW): every fused decode text, run 64 times on the same inputs, is bit-equal to its first run" {
    _ = std.c.getenv("GLM5NEXT_DETERMINISM_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_DETERMINISM bound_bytes={d}\n", .{bound_bytes});
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: glm.Config = .{};
    var prng = std.Random.DefaultPrng.init(0x5eed_6c6d);
    const r = prng.random();
    const h: c_int = @intCast(c.hidden_size);
    const hc: c_int = @intCast(c.hc_mult);
    const mix: c_int = (2 + hc) * hc;
    const e: c_int = @intCast(c.n_routed_experts);
    var diag: sdk.Diag = .{};

    // decode_glue.
    {
        const fn_rows = try rand(&g, a, r, &.{ mix, hc * h }, .float32, 0.01, 0);
        const scale = try rand(&g, a, r, &.{3}, .float32, 0.1, 1);
        const base = try rand(&g, a, r, &.{mix}, .float32, 0.1, 0);
        const norm_w = try rand(&g, a, r, &.{h}, .bfloat16, 0.1, 1);
        const router_w = try rand(&g, a, r, &.{ e, h }, .bfloat16, 0.02, 0);
        const bias = try rand(&g, a, r, &.{e}, .float32, 0.01, 0);
        try ops.install(glue.hcGeometry(&c, norm_w));
        var k = glue.install(a, s, &c, .{ .fn_rows = fn_rows, .scale = scale, .base = base, .norm_w = norm_w, .router_w = router_w, .router_bias = bias }, &diag) catch |err| {
            std.debug.print("glue install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        const ctx: GlueCtx = .{
            .k = &k,
            .hidden = h,
            .stream = try rand(&g, a, r, &.{ 1, 1, hc, h }, .bfloat16, 1, 0),
            .y = try rand(&g, a, r, &.{ 1, 1, h }, .bfloat16, 1, 0),
            .fn_rows = fn_rows,
            .scale = scale,
            .base = base,
            .norm_w = norm_w,
            .router_w = router_w,
            .bias = bias,
            .routed = try rand(&g, a, r, &.{ 1, 8, h }, .bfloat16, 1, 0),
            .shared = try rand(&g, a, r, &.{ 1, h }, .bfloat16, 1, 0),
        };
        try expectRunsEqual(&g, a, "decode_glue", ctx, glueRun);
    }

    // decode_kda.
    {
        const hv: c_int = @intCast(c.kda_heads);
        const dk: c_int = @intCast(c.kda_head_dim);
        const dim = hv * dk;
        const sd: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        const w: kda.Weights = .{
            .q = try rand(&g, a, r, &.{ dim, h }, .bfloat16, sd, 0),
            .k = try rand(&g, a, r, &.{ dim, h }, .bfloat16, sd, 0),
            .v = try rand(&g, a, r, &.{ dim, h }, .bfloat16, sd, 0),
            .f_a = try rand(&g, a, r, &.{ dk, h }, .bfloat16, sd, 0),
            .f_b = try rand(&g, a, r, &.{ dim, dk }, .bfloat16, 0.1, 0),
            .g_a = try rand(&g, a, r, &.{ dk, h }, .bfloat16, sd, 0),
            .g_b = try rand(&g, a, r, &.{ dim, dk }, .bfloat16, 0.1, 0),
            .b = try rand(&g, a, r, &.{ hv, h }, .bfloat16, sd, 0),
            .a_log = try rand(&g, a, r, &.{hv}, .float32, 0.5, 0),
            .a_exp = try rand(&g, a, r, &.{dim}, .float32, 0.1, 1),
            .dt_bias = try rand(&g, a, r, &.{dim}, .float32, 0.1, 0),
            .conv_w = try rand(&g, a, r, &.{ 3 * dim, @intCast(c.kda_conv), 1 }, .float32, 0.5, 0),
            .conv_raw = try rand(&g, a, r, &.{ 3 * dim, 1, @intCast(c.kda_conv) }, .bfloat16, 0.5, 0),
            .o_norm = try rand(&g, a, r, &.{dk}, .bfloat16, 0.2, 1),
            .o = try rand(&g, a, r, &.{ h, dim }, .bfloat16, sd, 0),
        };
        try kda.install(&c);
        var k = kda_decode.install(a, s, &c, &.{&w}, &diag) catch |err| {
            std.debug.print("kda install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        try expectRunsEqual(&g, a, "decode_kda", KdaCtx{ .k = &k, .c = &c, .w = &w, .x = try rand(&g, a, r, &.{ 1, 1, h }, .bfloat16, 1, 0) }, kdaRun);
    }

    // decode_dsa at S = 1 and 4.
    {
        var k = dsa_decode.install(a, s, &c, &diag) catch |err| {
            std.debug.print("dsa install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        const lanes: c_int = 4096;
        const width: c_int = 2051;
        const lat = try rand(&g, a, r, &.{ 1, 1, lanes, @intCast(dsa_decode.latent) }, .bfloat16, 1, 0);
        for ([_]u32{ 1, 4 }) |rows| {
            const sel_h = try a.alloc(i32, rows * @as(usize, @intCast(width)));
            defer a.free(sel_h);
            for (sel_h, 0..) |*x, i| {
                const j: i32 = @intCast(i % @as(usize, @intCast(width)));
                x.* = if (j < 2048) j * 2 else if (j < 2050) 4090 + j - 2048 else -1;
            }
            const sel = g.keep(try g.hostArray(std.mem.sliceAsBytes(sel_h), &.{ 1, @intCast(rows), width }, .int32));
            const q = try rand(&g, a, r, &.{ 1, @intCast(c.n_heads), @intCast(rows), @intCast(dsa_decode.latent) }, .bfloat16, 1, 0);
            var nb: [32]u8 = undefined;
            try expectRunsEqual(&g, a, try std.fmt.bufPrint(&nb, "decode_dsa S={d}", .{rows}), DsaCtx{ .k = &k, .q = q, .lat = lat, .sel = sel, .rows = rows }, dsaRun);
        }
    }

    // decode_swiglu at both widths.
    {
        var k = swiglu.install(a, s, &c, c.intermediate_size, c.moe_intermediate_size, &diag) catch |err| {
            std.debug.print("swiglu install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        for ([_]struct { swiglu.Width, u32 }{ .{ .dense, c.intermediate_size }, .{ .shared, c.moe_intermediate_size } }) |wv| {
            const gate = try rand(&g, a, r, &.{ 1, @intCast(wv[1]) }, .bfloat16, 4, 0);
            const up = try rand(&g, a, r, &.{ 1, @intCast(wv[1]) }, .bfloat16, 4, 0);
            try expectRunsEqual(&g, a, "decode_swiglu", SwigluCtx{ .k = &k, .w = wv[0], .gate = gate, .up = up }, swigluRun);
        }
    }
}
