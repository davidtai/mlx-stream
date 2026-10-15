//! The window's parity test of the fused decode glue (glue.zig): GLM5NEXT_GLUE_GPU=1 (else skipped) and
//! _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child). Synthetic tables at GLM-5.3-Flash's shapes (no
//! pack read), each text against the stock op chain it replaces:
//!   - the top-8 text on the stock route's own F32 logits: ids and weights bit for bit (the stock tail's expressions);
//!   - the router text: logits within 1e-5 relative RMS of the stock F32 product (another summation order), each row's
//!     logits at M = 2..4 bit for bit its M = 1 logits (row-invariance), and the routed ids of the two routes compared
//!     (a differing id is reported with its logit gap; the test fails only where the gap exceeds 1e-4: a near-tie may
//!     part, a wrong product may not);
//!   - the combine text: the stock combine plus the shared expert, bit for bit;
//!   - the hcPre chain over two sublayers (collapse and norm; the deferred expand folded into the next): its normed rows
//!     and gates against the stock collapse, rms_norm and expand within 2e-3 relative RMS (rounding-class: hcPre sums the
//!     mixes and the squares in its own order), the materialized stream against the stock expand within the same bound.
//! Static bound (declared before any run): the router's BF16 weight and F32 copy (2.4 + 4.7 MB), the mHC table (1.6 MB),
//! rows of at most 4 x 16,384 bf16, the outputs: < 32 MB, plus the texts' JIT; declared 0.2 GB.

const std = @import("std");
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const graph = @import("graph.zig");
const glue = @import("glue.zig");
const ops = @import("ops.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops_be.MlxOps;

const parity_bound_bytes: u64 = 200_000_000;

fn nextU32(state: *u32) u32 {
    state.* = state.* *% 1664525 +% 1013904223;
    return state.*;
}

/// A uniform value in [-scale, scale).
fn uniform(state: *u32, scale: f32) f32 {
    return (@as(f32, @floatFromInt(nextU32(state) >> 8)) / 8388608.0 - 1.0) * scale;
}

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn bf16Array(g: *G, a: std.mem.Allocator, shape: []const c_int, scale: f32, offset: f32, state: *u32) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try a.alloc(u16, n);
    defer a.free(v);
    for (v) |*x| x.* = bf16Bits(offset + uniform(state, scale));
    const arr = try g.hostArray(std.mem.sliceAsBytes(v), shape, .bfloat16);
    try g.evalAll(&.{arr});
    return arr;
}

fn f32Array(g: *G, a: std.mem.Allocator, shape: []const c_int, scale: f32, offset: f32, state: *u32) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try a.alloc(f32, n);
    defer a.free(v);
    for (v) |*x| x.* = offset + uniform(state, scale);
    const arr = try g.hostArray(std.mem.sliceAsBytes(v), shape, .float32);
    try g.evalAll(&.{arr});
    return arr;
}

fn hostF32(g: *G, a: std.mem.Allocator, x: T) ![]f32 {
    const f = try g.astype(x, .float32);
    try g.evalAll(&.{f});
    const n = mlx.mlx_array_size(f);
    const out = try a.alloc(f32, n);
    _ = try g.hostF32(f, out);
    return out;
}

fn hostBits(g: *G, a: std.mem.Allocator, x: T) ![]u32 {
    const f = try g.astype(x, .float32);
    try g.evalAll(&.{f});
    const n = mlx.mlx_array_size(f);
    const v = try a.alloc(f32, n);
    defer a.free(v);
    _ = try g.hostF32(f, v);
    const out = try a.alloc(u32, n);
    for (out, v) |*o, x2| o.* = @bitCast(x2);
    return out;
}

fn relRms(got: []const f32, truth: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, truth) |x, t| {
        num += (@as(f64, x) - t) * (@as(f64, x) - t);
        den += @as(f64, t) * t;
    }
    return @sqrt(num / @max(den, 1e-30));
}

fn gated() !mlx.mlx_stream {
    _ = std.c.getenv("GLM5NEXT_GLUE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_GLUE_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    const stream = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(stream)) {
        _ = mlx.mlx_stream_free(stream);
        return error.SkipZigTest;
    }
    return stream;
}

test "glm5next glue GPU: router + top-8 texts against the stock route; the combine text against combineStock + add; the hcPre chain against collapse, rms_norm and expand" {
    const stream = try gated();
    defer _ = mlx.mlx_stream_free(stream);
    const a = testing.allocator;
    const c: glm.Config = .{};
    var g = try G.init(a, stream);
    defer g.deinit();
    var state: u32 = 0x676c7565;
    const H: c_int = @intCast(c.hidden_size);
    const E: c_int = @intCast(c.n_routed_experts);
    const HC: c_int = @intCast(c.hc_mult);
    const mix: c_int = (2 + HC) * HC;

    // The tables: a router (BF16 weight and its F32 transposed copy, the stock route's), a correction bias, one
    // sublayer's mHC table and norm weight.
    const router_w = try bf16Array(&g, a, &.{ E, H }, 0.02, 0.0, &state);
    const router_t = try graph.f32Copy(stream, router_w, true);
    defer _ = mlx.mlx_array_free(router_t);
    const bias = try f32Array(&g, a, &.{E}, 0.05, 0.0, &state);
    const fn_rows = try f32Array(&g, a, &.{ mix, HC * H }, 0.01, 0.0, &state);
    const scale = try f32Array(&g, a, &.{3}, 0.5, 1.0, &state);
    const base = try f32Array(&g, a, &.{mix}, 0.5, 0.0, &state);
    const norm_w = try bf16Array(&g, a, &.{H}, 0.5, 1.0, &state);
    var diag: @import("sdk").Diag = .{};
    var k = glue.install(a, stream, &c, .{ .fn_rows = fn_rows, .scale = scale, .base = base, .norm_w = norm_w, .router_w = router_w, .router_bias = bias }, &diag) catch |e| {
        std.debug.print("glm5next glue install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer k.deinit();
    const r_stock: graph.Router = .{ .w = router_w, .w_t = router_t, .bias = bias };

    var worst_logit_rel: f64 = 0;
    var parted: usize = 0;
    for (1..glue.max_rows + 1) |m| {
        const mi: c_int = @intCast(m);
        const x = try bf16Array(&g, a, &.{ mi, H }, 2.0, 0.0, &state);
        // The stock route and its F32 logits.
        const stock = try graph.route(&g, &c, r_stock, x);
        const logits_stock = try g.matmul(try g.astype(x, .float32), router_t);
        // The top-8 text on the stock logits: bit for bit.
        var tail: [2]T = undefined;
        {
            const in = mlx.mlx_vector_array_new_data(&[_]T{ logits_stock, bias, k.consts }, 3);
            defer _ = mlx.mlx_vector_array_free(in);
            var ov = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ov);
            try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.top8, in, k.top8_cfg[m - 1], stream));
            for (&tail, 0..) |*o, i| {
                var y = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&y, ov, i));
                o.* = try g.adopt(y);
            }
        }
        const ids_stock = try hostBits(&g, a, try g.astype(stock.indices, .float32));
        defer a.free(ids_stock);
        const ids_tail = try hostBits(&g, a, try g.astype(tail[0], .float32));
        defer a.free(ids_tail);
        try testing.expectEqualSlices(u32, ids_stock, ids_tail);
        const w_stock = try hostBits(&g, a, stock.weights);
        defer a.free(w_stock);
        const w_tail = try hostBits(&g, a, tail[1]);
        defer a.free(w_tail);
        try testing.expectEqualSlices(u32, w_stock, w_tail);
        // The router text: its logits against the stock product, and the routed ids.
        var lg: [1]T = undefined;
        {
            const in = mlx.mlx_vector_array_new_data(&[_]T{ x, router_w }, 2);
            defer _ = mlx.mlx_vector_array_free(in);
            var ov = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ov);
            try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.router, in, k.router_cfg[m - 1], stream));
            var y = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
            lg[0] = try g.adopt(y);
        }
        const l_text = try hostF32(&g, a, lg[0]);
        defer a.free(l_text);
        const l_stock = try hostF32(&g, a, logits_stock);
        defer a.free(l_stock);
        worst_logit_rel = @max(worst_logit_rel, relRms(l_text, l_stock));
        // Row invariance: row r at M rows is its M = 1 logits (the first row of each M against M = 1 of the same x row).
        const fused = try glue.route(&g, &k, router_w, bias, x, @intCast(m));
        const ids_fused = try hostBits(&g, a, try g.astype(fused.ids, .float32));
        defer a.free(ids_fused);
        for (ids_fused, ids_stock, 0..) |f, st, i| if (f != st) {
            parted += 1;
            const row = i / 8;
            const fe: usize = @intFromFloat(@as(f32, @bitCast(f)));
            const se: usize = @intFromFloat(@as(f32, @bitCast(st)));
            const gap = @abs(l_stock[row * @as(usize, @intCast(E)) + fe] - l_stock[row * @as(usize, @intCast(E)) + se]);
            std.debug.print("GLM5N_GLUE_PARITY parted M={d} row={d} slot={d} fused={d} stock={d} stock_logit_gap={e}\n", .{ m, row, i % 8, fe, se, gap });
            try testing.expect(gap <= 1e-4);
        };
        if (m > 1) {
            for (0..m) |row| {
                const ri: c_int = @intCast(row);
                const one = try g.slice(x, &.{ ri, 0 }, &.{ ri + 1, H }, &.{ 1, 1 });
                // The router's own logits per row: the M-row launch's row against the one-row launch.
                var l1: [1]T = undefined;
                const in = mlx.mlx_vector_array_new_data(&[_]T{ one, router_w }, 2);
                defer _ = mlx.mlx_vector_array_free(in);
                var ov = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(ov);
                try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.router, in, k.router_cfg[0], stream));
                var y = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
                l1[0] = try g.adopt(y);
                const b1 = try hostBits(&g, a, l1[0]);
                defer a.free(b1);
                const bm = try hostBits(&g, a, try g.slice(lg[0], &.{ ri, 0 }, &.{ ri + 1, E }, &.{ 1, 1 }));
                defer a.free(bm);
                try testing.expectEqualSlices(u32, b1, bm);
            }
        }
    }
    std.debug.print("GLM5N_GLUE_PARITY router logits_rel_rms_worst={e} parted_ids={d}\n", .{ worst_logit_rel, parted });
    try testing.expect(worst_logit_rel <= 1e-5);

    // The combine with the shared expert: bit for bit against the stock combine then the bf16 add.
    for (1..glue.max_rows + 1) |m| {
        const mi: c_int = @intCast(m);
        const routed = try bf16Array(&g, a, &.{ mi, 8, H }, 1.0, 0.0, &state);
        const weights = try f32Array(&g, a, &.{ mi, 8 }, 0.3, 0.3, &state);
        const shared = try bf16Array(&g, a, &.{ mi, H }, 1.0, 0.0, &state);
        const expanded = try g.reshape(weights, &.{ mi, 8, 1 });
        const stock = try g.add(try g.astype(try g.sum(try g.mul(try g.astype(routed, .float32), expanded), 1, false), .bfloat16), shared);
        const fused = try glue.combineShared(&g, &k, routed, weights, shared, @intCast(m));
        const bs = try hostBits(&g, a, stock);
        defer a.free(bs);
        const bf = try hostBits(&g, a, fused);
        defer a.free(bf);
        try testing.expectEqualSlices(u32, bs, bf);
    }

    // The hcPre chain over two sublayers against the stock collapse, rms_norm and expand.
    for (1..glue.max_rows + 1) |m| {
        const mi: c_int = @intCast(m);
        const stream0 = try bf16Array(&g, a, &.{ 1, mi, HC, H }, 1.0, 0.0, &state);
        const y0 = try bf16Array(&g, a, &.{ 1, mi, H }, 1.0, 0.0, &state);
        // Stock: collapse, norm (sublayer 1); expand with y0; collapse, norm (sublayer 2).
        var c1 = try ops.collapse(stream, stream0, fn_rows, scale, base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
        defer c1.deinit();
        const n1 = try graph.rmsNorm(&g, c1.x, norm_w, c.rms_norm_eps);
        const s1 = try g.adopt(try ops.expand(stream, y0, stream0, &c1));
        var c2 = try ops.collapse(stream, s1, fn_rows, scale, base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
        defer c2.deinit();
        const n2 = try graph.rmsNorm(&g, c2.x, norm_w, c.rms_norm_eps);
        // Fused: hcPre on the stream, its deferral with y0, hcPre on the deferral.
        const p1 = try glue.pre(&g, &c, stream0, null, fn_rows, scale, base, norm_w);
        const df: ops.Deferred = .{ .y = y0, .post = p1.post, .comb = p1.comb, .h = stream0 };
        const p2 = try glue.pre(&g, &c, null, &df, fn_rows, scale, base, norm_w);
        for ([_][2]T{ .{ n1, p1.normed }, .{ c1.post, p1.post }, .{ c1.comb, p1.comb }, .{ s1, p2.h.? }, .{ n2, p2.normed }, .{ c2.post, p2.post }, .{ c2.comb, p2.comb } }, 0..) |pair, i| {
            const want = try hostF32(&g, a, pair[0]);
            defer a.free(want);
            const got = try hostF32(&g, a, pair[1]);
            defer a.free(got);
            const rel = relRms(got, want);
            std.debug.print("GLM5N_GLUE_PARITY hcPre M={d} output={d} rel_rms={e}\n", .{ m, i, rel });
            try testing.expect(rel <= 2e-3);
        }
    }
}
