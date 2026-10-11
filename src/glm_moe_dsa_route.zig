//! GLM-5.3's routing after the router's sigmoid, in one launch: the correction bias added, each row's top-k experts
//! as `argpartition(-(sig + bias), k - 1)[..., :k]` gives them, their sigmoid scores, normalized (`norm_topk_prob`) and
//! scaled. MLX's chain is seven launches in a row (add, negate, sort, gather, sum, divide, multiply) on the way to the
//! routing barrier; the kernel makes the same values at each step. MLX partitions by a stable merge sort of the whole
//! row (`LessThan`: a NaN after every number), so an expert's position is the count of experts before it in that order;
//! the sum is `row_reduce_small`'s, one thread adding the k scores in order from 0; the divide and the multiply are
//! each one rounding, as MLX's kernels do them.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("deepseek_v41_ops.zig");

const G = ops.MlxOps;
const T = G.T;

/// One threadgroup of `E` threads per row (one per expert); `E`, `K` and `NORM` template arguments, the scale in
/// `params[0]`.
const source: [:0]const u8 =
    \\    const uint e = thread_position_in_threadgroup.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    threadgroup float keyv[E];
    \\    threadgroup int ids[K];
    \\    threadgroup float picked[K];
    \\    const float s = sig[ulong(row) * uint(E) + e];
    \\    const float v = -(s + bias[e]);
    \\    keyv[e] = v;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // The expert's position in the stable ascending order of -(sig + bias): those before it, and its ties of a
    \\    // lower index.
    \\    const bool vn = metal::isnan(v);
    \\    uint pos = 0;
    \\    for (uint j = 0; j < uint(E); ++j) {
    \\        const float u = keyv[j];
    \\        const bool un = metal::isnan(u);
    \\        const bool u_lt = (un | vn) ? (!un & vn) : (u < v);
    \\        const bool v_lt = (un | vn) ? (!vn & un) : (v < u);
    \\        pos += (u_lt || (!v_lt && j < e)) ? 1u : 0u;
    \\    }
    \\    if (pos < uint(K)) { ids[pos] = int(e); picked[pos] = s; }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (e < uint(K)) {
    \\        float w = picked[e];
    \\        if (NORM) {
    \\            float total = 0.0f;
    \\            for (uint j = 0; j < uint(K); ++j) total = picked[j] + total;
    \\            w = w / total;
    \\        }
    \\        weights[ulong(row) * uint(K) + e] = w * params[0];
    \\        indices[ulong(row) * uint(K) + e] = ids[e];
    \\    }
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The chosen experts `[n, k]` (int32) and their weights `[n, k]` (f32).
pub const Top = struct { indices: T, weights: T };

/// Whether the kernel serves `n_experts` experts and a top `k`: one thread per expert in a threadgroup.
pub fn serves(n_experts: u32, k: u32) bool {
    return n_experts <= 1024 and k > 0 and k <= n_experts;
}

/// The top `k` expert ids `[n, k]` (int32) and their weights `[n, k]` (f32) of `sig` (`[n, n_experts]` f32, the
/// router's sigmoid) by `sig + bias` (`[n_experts]` f32): `glm_moe_dsa_graph.route`'s chain after its sigmoid, the
/// same bits (`serves` the shape; a GPU stream).
pub fn top(g: *G, sig: T, bias: T, k: u32, norm: bool, scale: f32) !Top {
    const sh = g.shapeOf(sig);
    const n = sh.dim(0);
    const ne = sh.dim(1);
    std.debug.assert(sh.n == 2 and serves(@intCast(ne), k) and g.dtypeOf(sig) == .float32 and g.dtypeOf(bias) == .float32);
    const kern = kernel orelse blk: {
        const ins = [_][*:0]const u8{ "sig", "bias", "params" };
        const outs = [_][*:0]const u8{ "indices", "weights" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = mlx.mlx_fast_metal_kernel_new("glm_moe_dsa_route_top", iv, ov, source, "", true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        kernel = kn;
        break :blk kn;
    };
    // Eight slots: an input of fewer is bound in the constant address space.
    const pv = [8]f32{ scale, 0, 0, 0, 0, 0, 0, 0 };
    const params = try g.hostArray(std.mem.sliceAsBytes(&pv), &.{8}, .float32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ n, @intCast(k) }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ n, @intCast(k) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ne, n, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ne, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "E", ne));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(k)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NORM", norm));
    const inputs = [_]mlx.mlx_array{ sig, bias, params };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var outs: [2]T = undefined;
    for (&outs, 0..) |*o, i| {
        var a = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&a, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        o.* = try g.adopt(a);
    }
    return .{ .indices = outs[0], .weights = outs[1] };
}

/// MLX's chain after the sigmoid (`glm_moe_dsa_graph.route`'s on a CPU stream, and the kernel's reference).
pub fn chain(g: *G, sig: T, bias: T, k: u32, norm: bool, scale: f32) !Top {
    const kk: c_int = @intCast(k);
    const sc = try g.add(sig, bias);
    const inds = try lastSlice(g, try g.argpartition(try g.neg(sc), kk - 1, -1), 0, kk);
    var w = try g.takeAlongAxis(sig, inds, -1);
    if (norm) w = try g.div(w, try g.sum(w, -1, true));
    w = try g.mul(w, try g.scalar(scale, .float32));
    return .{ .indices = try g.astype(inds, .int32), .weights = w };
}

fn lastSlice(g: *G, x: T, lo: c_int, hi: c_int) !T {
    const s = g.shapeOf(x);
    return g.slice(x, &.{ 0, lo }, &.{ s.dim(0), hi }, &.{ 1, 1 });
}

const testing = std.testing;

test "glm route: the top-k kernel's ids and weights are MLX's chain's bits after the sigmoid, ties, a NaN, 1 to 64 rows, 256 and 16 experts" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(23);
    const rand = prng.random();
    // (rows, experts, k, norm, kind): logits N(0, 2) through MLX's sigmoid with a small bias (GLM-5.3's); the scores on
    // a coarse grid (ties at and inside the top-k); a NaN in some rows; an unnormalized arm.
    const cases = [_]struct { n: u32, e: u32, k: u32, norm: bool, kind: u8 }{
        .{ .n = 1, .e = 256, .k = 8, .norm = true, .kind = 0 },
        .{ .n = 4, .e = 256, .k = 8, .norm = true, .kind = 0 },
        .{ .n = 5, .e = 256, .k = 8, .norm = true, .kind = 1 },
        .{ .n = 64, .e = 256, .k = 8, .norm = true, .kind = 0 },
        .{ .n = 37, .e = 256, .k = 8, .norm = true, .kind = 2 },
        .{ .n = 6, .e = 16, .k = 8, .norm = true, .kind = 1 },
        .{ .n = 4, .e = 256, .k = 8, .norm = false, .kind = 0 },
    };
    var checked: usize = 0;
    for (cases) |cs| {
        const m = g.mark();
        defer g.resetTo(m);
        const lv = try a.alloc(f32, cs.n * cs.e);
        defer a.free(lv);
        for (lv, 0..) |*x, i| x.* = switch (cs.kind) {
            1 => @floor(rand.floatNorm(f32) * 2) / 2,
            2 => if (i % 97 == 5) std.math.nan(f32) else rand.floatNorm(f32) * 2,
            else => rand.floatNorm(f32) * 2,
        };
        const bv = try a.alloc(f32, cs.e);
        defer a.free(bv);
        for (bv) |*x| x.* = if (cs.kind == 1) 0 else (rand.float(f32) - 0.5) * 0.1;
        const sig = try g.sigmoid(try g.hostArray(std.mem.sliceAsBytes(lv), &.{ @intCast(cs.n), @intCast(cs.e) }, .float32));
        const bias = try g.hostArray(std.mem.sliceAsBytes(bv), &.{@intCast(cs.e)}, .float32);
        const got = try top(&g, sig, bias, cs.k, cs.norm, 2.5);
        const want = try chain(&g, sig, bias, cs.k, cs.norm, 2.5);
        const len = cs.n * cs.k;
        const iv = try a.alloc(u32, 2 * len);
        defer a.free(iv);
        _ = try g.hostU32(try g.astype(got.indices, .uint32), iv[0..len]);
        _ = try g.hostU32(try g.astype(want.indices, .uint32), iv[len..]);
        try testing.expectEqualSlices(u32, iv[len..], iv[0..len]);
        const wv = try a.alloc(f32, 2 * len);
        defer a.free(wv);
        _ = try g.hostF32(got.weights, wv[0..len]);
        _ = try g.hostF32(want.weights, wv[len..]);
        try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(wv[len..])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(wv[0..len])));
        checked += len;
    }
    std.debug.print("glm route: {d} ids and weights equal MLX's chain's bits\n", .{checked});
}
