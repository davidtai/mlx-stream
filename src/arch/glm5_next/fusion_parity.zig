//! The window's parity test of the trunk launch fusion (fusion.zig), in the pattern of `fp8_parity.zig`:
//! GLM5NEXT_FUSION_GPU=1 (else skipped) and _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child). Never run
//! outside a window. Synthetic FP8 sites at Flash's shapes (MiMo's `Fixture`: codes over every non-NaN e4m3 value, scales
//! in the release's range, rows ~ N(0, 1.5^2)), installed through `fusion.install` on the GPU stream (the plan, the
//! texts, the launches, the compile check), then at every width M = 1 .. 8:
//!   - fuse_kda_inputs: a KDA layer's six input projections (q / k / v [8192, 4096], f_a / g_a [128, 4096], b [64, 4096])
//!     by the fused launch against the stock route's six launches, every output bit for bit;
//!   - fuse_gate_up: the dense MLP's and the shared expert's gate / up ([12288, 4096], [2048, 4096]) by the fused launch
//!     against the stock route's two, every output bit for bit;
//!   - fuse_predictor: the lookahead predictor over a random BF16 router [288, 4096] and correction (experts.zig
//!     `scoresBf16Fused` against `scoresBf16`: the same product, the epilogue one launch against MLX's three), every score
//!     bit for bit;
//!   - fuse_join: a routed call's end over random joined rows [8 M + 5, 4096] bf16, a random join order, weights and shared
//!     rows (fusion.zig `combineIndexed` against the join's take then glue.zig's combine text), every output bit for bit.
//! One line per switch and width `GLM5NEXT_FUSION_PARITY shape=<switch>_m<M> PASS|FAIL bit_diffs=<n> outputs=<n>` and one
//! `GLM5NEXT_FUSION_PARITY overall PASS|FAIL total_bit_diffs=<n>`.
//! Static bound (declared before any run): the sites' codes on the host and their MLX copies (2 x (3 x 33.6 + 0.8 + 2 x
//! 50.3 + 2 x 8.4) MB = 437 MB), their scales (< 1 MB), rows and outputs (< 2 MB), plus the JIT (two texts and MiMo's
//! GEMV): declared 0.6 GB.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const glm = @import("config.zig");
const fp8 = @import("fp8.zig");
const kda = @import("kda.zig");
const graph = @import("graph.zig");
const fusion = @import("fusion.zig");
const experts = @import("experts.zig");
const glue = @import("glue.zig");
const mtl = @import("../../sdk_ext/metal_launch.zig");
const mimo_fp8 = @import("../mimo_v2/fp8_linear.zig");
const mimo_math = @import("../mimo_v2/math.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = fusion.G;

pub const parity_bound_bytes: u64 = 600_000_000;

test "glm5_next fusion GPU: the fused launches (fuse_kda_inputs, fuse_gate_up, fuse_predictor, fuse_join) against the stock launches at M = 1 .. 8, every output bit for bit" {
    _ = std.c.getenv("GLM5NEXT_FUSION_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5NEXT_FUSION_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const route = try fp8.Route.install(a);
    defer route.destroy(a);
    const c: glm.Config = .{};
    const h = c.hidden_size;
    var fixtures: std.ArrayList(mimo_fp8.Fixture) = .empty;
    defer {
        for (fixtures.items) |f| f.deinit(a);
        fixtures.deinit(a);
    }
    const Site = struct {
        fn make(gg: *G, al: std.mem.Allocator, list: *std.ArrayList(mimo_fp8.Fixture), r: *fp8.Route, n: u32, k: u32, seed: u64) !fp8.Linear {
            const f = try mimo_fp8.Fixture.init(al, n, k, fusion.max_rows, seed);
            try list.append(al, f);
            const w = gg.keep(try gg.hostArray(f.codes, &.{ @intCast(n), @intCast(k) }, .uint8));
            const sc = gg.keep(try gg.hostArray(std.mem.sliceAsBytes(f.scales), &fp8.scaleShape(n, k), .float32));
            try gg.evalAll(&.{ w, sc });
            return .{ .fp8 = try fp8.bind(r, w, sc, n, k) };
        }
    };
    var w: kda.Weights = undefined;
    w.q = try Site.make(&g, a, &fixtures, route, c.kdaDim(), h, 0x6c66_0001);
    w.k = try Site.make(&g, a, &fixtures, route, c.kdaDim(), h, 0x6c66_0002);
    w.v = try Site.make(&g, a, &fixtures, route, c.kdaDim(), h, 0x6c66_0003);
    w.f_a = try Site.make(&g, a, &fixtures, route, c.kda_head_dim, h, 0x6c66_0004);
    w.g_a = try Site.make(&g, a, &fixtures, route, c.kda_head_dim, h, 0x6c66_0005);
    w.b = try Site.make(&g, a, &fixtures, route, c.kda_heads, h, 0x6c66_0006);
    const none: fp8.Linear = .{ .bf16 = .{ .ctx = null } };
    const dense: graph.Mlp = .{ .gate = try Site.make(&g, a, &fixtures, route, c.intermediate_size, h, 0x6c66_0007), .up = try Site.make(&g, a, &fixtures, route, c.intermediate_size, h, 0x6c66_0008), .down = none };
    const shared: graph.Mlp = .{ .gate = try Site.make(&g, a, &fixtures, route, c.moe_intermediate_size, h, 0x6c66_0009), .up = try Site.make(&g, a, &fixtures, route, c.moe_intermediate_size, h, 0x6c66_000a), .down = none };
    const layers = [_]graph.Layer{
        .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = w }, .dense = dense, .slot = 0 },
        .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = w }, .shared = shared, .slot = 1 },
    };
    var diag: sdk.Diag = .{};
    fusion.install(a, s, &c, &layers, .{ .kda_inputs = true, .gate_up = true, .predictor = true, .join = true }, &diag) catch |e| {
        std.debug.print("GLM5NEXT_FUSION_PARITY install refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
        return e;
    };
    defer fusion.install(a, s, &c, &layers, .{}, &diag) catch {};
    try testing.expect(fusion.kdaInputsEngaged() and fusion.gateUpEngaged() and fusion.predInstalled() and fusion.joinInstalled());
    // The stock combine (glue.zig's text and launch: the indexed combine's launch less its order input).
    const stock_combine = blk: {
        const ins = [_][*:0]const u8{ "rows", "w", "shared" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        break :blk mtl.mlx_fast_metal_kernel_new("glm5n_moe_combine", iv, ov, glue.texts.combine, glue.texts.header_combine, true, false);
    };
    defer mtl.mlx_fast_metal_kernel_free(stock_combine);
    // The predictor's router (BF16 [E, hidden], N(0, 1/64) weights as bf16) and correction (f32 [E], N(0, 0.1)).
    const e: c_int = @intCast(c.n_routed_experts);
    var rng: std.Random.DefaultPrng = .init(0x6c66_0b0b);
    const rnd = rng.random();
    const rbits = try a.alloc(u16, @as(usize, @intCast(e)) * h);
    defer a.free(rbits);
    for (rbits) |*b| b.* = mimo_fp8.f32ToBf16(rnd.floatNorm(f32) / 8.0);
    var cvals: [512]f32 = undefined;
    for (cvals[0..@intCast(e)]) |*v| v.* = 0.1 * rnd.floatNorm(f32);
    const router = g.keep(try g.hostArray(std.mem.sliceAsBytes(rbits), &.{ e, @intCast(h) }, .bfloat16));
    const corr = g.keep(try g.hostArray(std.mem.sliceAsBytes(cvals[0..@intCast(e)]), &.{e}, .float32));
    try g.evalAll(&.{ router, corr });
    const pr: mimo_math.Predictor = .{};
    var total: usize = 0;
    var failed_lines: usize = 0;
    // x rows: the first fixture's (every width a prefix of its 8 rows).
    const xs = fixtures.items[0].x;
    for (1..fusion.max_rows + 1) |m| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const x3 = try g.hostArray(std.mem.sliceAsBytes(xs[0 .. m * h]), &.{ 1, @intCast(m), @intCast(h) }, .bfloat16);
        const x2 = try g.hostArray(std.mem.sliceAsBytes(xs[0 .. m * h]), &.{ @intCast(m), @intCast(h) }, .bfloat16);
        // The six, both routes (`testUse` switches the installed routes; the fused launch is the install's).
        fusion.testUse(false);
        const stock = try fusion.kdaInputs(&g, &w, x3);
        fusion.testUse(true);
        const fused = try fusion.kdaInputs(&g, &w, x3);
        const s6 = [_]T{ stock.q, stock.k, stock.v, stock.f_a, stock.g_a, stock.b };
        const f6 = [_]T{ fused.q, fused.k, fused.v, fused.f_a, fused.g_a, fused.b };
        var pairs: [4][2]T = undefined;
        for ([_]struct { graph.Mlp, @import("swiglu.zig").Width }{ .{ dense, .dense }, .{ shared, .shared } }, 0..) |mw, i| {
            fusion.testUse(false);
            const sg = try fusion.gateUp(&g, mw[0].gate, mw[0].up, x2, mw[1], @intCast(m));
            fusion.testUse(true);
            const fg = try fusion.gateUp(&g, mw[0].gate, mw[0].up, x2, mw[1], @intCast(m));
            pairs[2 * i] = .{ sg.gate, fg.gate };
            pairs[2 * i + 1] = .{ sg.up, fg.up };
        }
        var all: [12 + 8]T = undefined;
        @memcpy(all[0..6], &s6);
        @memcpy(all[6..12], &f6);
        for (pairs, 0..) |p, i| {
            all[12 + 2 * i] = p[0];
            all[12 + 2 * i + 1] = p[1];
        }
        try g.evalAll(&all);
        var diffs: usize = 0;
        var outputs: usize = 0;
        for (s6, f6) |x, y| {
            const r = try bitDiffs(&g, x, y);
            diffs += r.diffs;
            outputs += r.n;
        }
        std.debug.print("GLM5NEXT_FUSION_PARITY shape=kda_inputs_m{d} {s} bit_diffs={d} outputs={d}\n", .{ m, if (diffs == 0) "PASS" else "FAIL", diffs, outputs });
        total += diffs;
        if (diffs != 0) failed_lines += 1;
        var pd: usize = 0;
        var pn: usize = 0;
        for (pairs) |p| {
            const r = try bitDiffs(&g, p[0], p[1]);
            pd += r.diffs;
            pn += r.n;
        }
        std.debug.print("GLM5NEXT_FUSION_PARITY shape=gate_up_m{d} {s} bit_diffs={d} outputs={d}\n", .{ m, if (pd == 0) "PASS" else "FAIL", pd, pn });
        total += pd;
        if (pd != 0) failed_lines += 1;
        // The predictor: f32 scores, compared bit for bit.
        const ps = try experts.scoresBf16(&pr, &g, x2, router, corr);
        const pf = try experts.scoresBf16Fused(&pr, &g, x2, router, corr);
        try g.evalAll(&.{ ps, pf });
        const n_s = mlx.mlx_array_size(ps);
        const vs = mlx.mlx_array_data_float32(ps) orelse return error.MlxNoData;
        const vf = mlx.mlx_array_data_float32(pf) orelse return error.MlxNoData;
        var sd: usize = 0;
        for (0..n_s) |i| sd += @intFromBool(@as(u32, @bitCast(vs[i])) != @as(u32, @bitCast(vf[i])));
        std.debug.print("GLM5NEXT_FUSION_PARITY shape=predictor_m{d} {s} bit_diffs={d} outputs={d}\n", .{ m, if (sd == 0) "PASS" else "FAIL", sd, n_s });
        total += sd;
        if (sd != 0) failed_lines += 1;
        // The join: joined rows [8 M + 5, hidden] (the first fixture's x rows repeated, as bf16), each routed position's row
        // drawn at random (a device wave's and a miss part's rows), the weights and the shared rows random.
        {
            const tk: usize = glue.top_k;
            const n_ids = tk * m;
            const r_rows = n_ids + 5;
            const jbits = try a.alloc(u16, r_rows * h);
            defer a.free(jbits);
            for (jbits, 0..) |*b, i| b.* = xs[i % xs.len];
            var ord: [8 * fusion.max_rows]u32 = undefined;
            for (ord[0..n_ids]) |*o| o.* = rnd.uintLessThan(u32, @intCast(r_rows));
            var wv: [8 * fusion.max_rows]f32 = undefined;
            for (wv[0..n_ids]) |*v| v.* = @abs(rnd.floatNorm(f32)) / 4.0;
            const joined = try g.hostArray(std.mem.sliceAsBytes(jbits), &.{ @intCast(r_rows), @intCast(h) }, .bfloat16);
            const order = try g.hostArray(std.mem.sliceAsBytes(ord[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            const wts = try g.hostArray(std.mem.sliceAsBytes(wv[0..n_ids]), &.{ @intCast(m), @intCast(tk) }, .float32);
            const shared_rows = try g.hostArray(std.mem.sliceAsBytes(xs[0 .. m * h]), &.{ @intCast(m), @intCast(h) }, .bfloat16);
            const flat = try g.reshape(try g.take(joined, order, 0), &.{ @intCast(n_ids), @intCast(h) });
            const cfg = try fusion.combineIdxConfig(@intCast(m));
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            const sin = mlx.mlx_vector_array_new_data(&[_]T{ flat, wts, shared_rows }, 3);
            defer _ = mlx.mlx_vector_array_free(sin);
            var so = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(so);
            try mlx.check(mtl.mlx_fast_metal_kernel_apply(&so, stock_combine, sin, cfg, g.s));
            var sy_raw = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&sy_raw, so, 0));
            const sy = try g.adopt(sy_raw);
            const fy = try fusion.combineIndexed(&g, joined, order, wts, shared_rows, @intCast(m));
            try g.evalAll(&.{ sy, fy });
            const r = try bitDiffs(&g, sy, fy);
            std.debug.print("GLM5NEXT_FUSION_PARITY shape=join_m{d} {s} bit_diffs={d} outputs={d}\n", .{ m, if (r.diffs == 0) "PASS" else "FAIL", r.diffs, r.n });
            total += r.diffs;
            if (r.diffs != 0) failed_lines += 1;
        }
    }
    fusion.testUse(false);
    std.debug.print("GLM5NEXT_FUSION_PARITY overall {s} total_bit_diffs={d}\n", .{ if (total == 0 and failed_lines == 0) "PASS" else "FAIL", total });
    try testing.expectEqual(@as(usize, 0), total);
}

/// Bit differences between two evaluated bf16 arrays of the same shape (widened to f32, exact: one f32 per bf16 bit
/// pattern).
fn bitDiffs(g: *G, x: T, y: T) !struct { diffs: usize, n: usize } {
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(y) != .bfloat16 or !g.shapeOf(x).eql(g.shapeOf(y))) return error.TestUnexpectedResult;
    const fx = try g.astype(x, .float32);
    const fy = try g.astype(y, .float32);
    try g.evalAll(&.{ fx, fy });
    const n = mlx.mlx_array_size(fx);
    const px = mlx.mlx_array_data_float32(fx) orelse return error.MlxNoData;
    const py = mlx.mlx_array_data_float32(fy) orelse return error.MlxNoData;
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(@as(u32, @bitCast(px[i])) != @as(u32, @bitCast(py[i])));
    return .{ .diffs = d, .n = n };
}
