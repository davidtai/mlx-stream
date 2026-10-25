//! Standalone GPU sparse selection and selected-block attention for B=1,L=1.
//! Ordinary Metal SIMD kernels cover M2; no NAX, tensor instructions or Steel
//! headers are required. Callers retain the existing projection/norm/RoPE path.
//! The attention kernels are adapted from oMLX (Apache-2.0); see their headers.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const mtl = @import("../../sdk_ext/metal_launch.zig");
const T = ops.T;
const Kernel = mlx.mlx_fast_metal_kernel;
const block_size = 128;
pub const Template = struct { name: [*:0]const u8, value: c_int };
pub const Output = struct { shape: []const c_int, dtype: mlx.mlx_dtype };
const Kernels = struct { pool: Kernel, topk: Kernel, partials: Kernel, combine: Kernel };
var kernels: ?Kernels = null;

pub fn supported(dim: u32, block: u32, heads: u32, kv_heads: u32) bool {
    return dim == 128 and block == block_size and kv_heads > 0 and heads >= kv_heads and heads % kv_heads == 0 and heads / kv_heads <= 16;
}

pub fn available() !void {
    var yes = false;
    try mlx.check(mlx.mlx_metal_is_available(&yes));
    if (!yes) return error.MiniMaxSparseMetalUnavailable;
}

pub fn make(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !Kernel {
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new(name, iv, ov, source, "#include <metal_simdgroup>\n", false, false);
    if (k.ctx == null) return error.MiniMaxSparseKernel;
    return k;
}

fn getKernels() !Kernels {
    if (kernels) |k| return k;
    const pool = try make("minimax_sparse_pool", &.{ "scores", "meta" }, &.{"pooled"}, @embedFile("kernels/minimax_sparse_pool.metal"));
    errdefer mtl.mlx_fast_metal_kernel_free(pool);
    const topk = try make("minimax_sparse_topk", &.{ "pooled", "meta" }, &.{"selected"}, @embedFile("kernels/minimax_sparse_topk.metal"));
    errdefer mtl.mlx_fast_metal_kernel_free(topk);
    const partials = try make("minimax_sparse_partials", &.{ "q", "k", "v", "block_ids", "qpos", "pad", "meta", "scale", "vk" }, &.{ "part_o", "part_m", "part_l" }, @embedFile("kernels/minimax_sparse_partials.metal"));
    errdefer mtl.mlx_fast_metal_kernel_free(partials);
    const combine = try make("minimax_sparse_combine", &.{ "part_o", "part_m", "part_l" }, &.{"out"}, @embedFile("kernels/minimax_sparse_combine.metal"));
    const out: Kernels = .{ .pool = pool, .topk = topk, .partials = partials, .combine = combine };
    kernels = out;
    return out;
}

pub fn contiguous(g: *ops.G, x: T) !T {
    var y = mlx.mlx_array_new();
    mlx.check(mlx.mlx_contiguous(&y, x, false, g.s)) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

pub fn launch(g: *ops.G, kernel: Kernel, inputs: []const T, outputs: []const Output, grid: [3]c_int, threadgroup: [3]c_int, dtype: mlx.mlx_dtype, template: []const Template) ![3]T {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outputs) |out| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, out.shape.ptr, out.shape.len, out.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threadgroup[0], threadgroup[1], threadgroup[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype));
    for (template) |arg| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, arg.name, arg.value));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, g.s));
    var result: [3]T = @splat(.{});
    for (outputs, 0..) |_, i| {
        var value = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&value, ov, i));
        result[i] = try g.adopt(value);
    }
    return result;
}

pub fn topkPooled(g: *ops.G, pooled: T, heads: u32, tokens: u32, position: u32, topk: u32) !T {
    if (heads == 0 or tokens == 0 or position >= tokens or topk == 0 or topk > 16) return error.MiniMaxSparseShape;
    const blocks: c_int = @intCast((tokens - 1) / block_size + 1);
    if (!std.mem.eql(c_int, g.shapeOf(pooled).slice(), &.{ @intCast(heads), blocks }) or g.dtypeOf(pooled) != .float32) return error.MiniMaxSparseShape;
    const meta_values: [8]i32 = .{ @intCast(tokens), blocks, @intCast(position), 0, 0, 0, 0, 0 };
    const meta = try g.hostArray(std.mem.sliceAsBytes(&meta_values), &.{8}, .int32);
    const args = [_]Template{ .{ .name = "BLOCK", .value = block_size }, .{ .name = "TOPK", .value = @intCast(topk) } };
    return (try launch(g, (try getKernels()).topk, &.{ pooled, meta }, &.{.{ .shape = &.{ 1, @intCast(heads), 1, @intCast(topk) }, .dtype = .int32 }}, .{ @intCast(heads * 128), 1, 1 }, .{ 128, 1, 1 }, .float32, &args))[0];
}

pub fn combineParts(g: *ops.G, parts: *const [3]T, shape: []const c_int, kv: c_int, topk: c_int, dtype: mlx.mlx_dtype) !T {
    const h = shape[1];
    const args = [_]Template{ .{ .name = "H_KV", .value = kv }, .{ .name = "G", .value = @divExact(h, kv) }, .{ .name = "L", .value = 1 }, .{ .name = "D", .value = 128 }, .{ .name = "TOPK", .value = topk } };
    return (try launch(g, (try getKernels()).combine, parts, &.{.{ .shape = shape, .dtype = dtype }}, .{ 128, h, 1 }, .{ 128, 1, 1 }, dtype, &args))[0];
}

/// F32 scores [1,index_heads,1,tokens] from the unchanged index matmul produce
/// I32 block IDs [1,index_heads,1,topk]. Keys after position and partial-block
/// padding never participate. Ties choose the lower block ID, matching reference.zig.
/// Invalid slots are -1. No scores or masks are copied to the CPU.
pub fn select(g: *ops.G, scores: T, heads: u32, tokens: u32, position: u32, topk: u32, local: u32, initial: u32) !T {
    if (heads == 0 or tokens == 0 or position >= tokens or topk == 0 or topk > 16 or tokens > std.math.maxInt(c_int)) return error.MiniMaxSparseShape;
    const sh = g.shapeOf(scores);
    if (!std.mem.eql(c_int, sh.slice(), &.{ 1, @intCast(heads), 1, @intCast(tokens) }) or g.dtypeOf(scores) != .float32) return error.MiniMaxSparseShape;
    try available();
    const k = try getKernels();
    const blocks: c_int = @intCast((tokens - 1) / block_size + 1);
    const meta_values: [8]i32 = .{ @intCast(tokens), blocks, @intCast(position), 0, 0, 0, 0, 0 };
    const meta = try g.hostArray(std.mem.sliceAsBytes(&meta_values), &.{8}, .int32);
    const args = [_]Template{ .{ .name = "BLOCK", .value = block_size }, .{ .name = "INIT_BLOCKS", .value = @intCast(initial) }, .{ .name = "LOCAL_BLOCKS", .value = @intCast(local) }, .{ .name = "TOPK", .value = @intCast(topk) } };
    const pooled = (try launch(g, k.pool, &.{ try contiguous(g, scores), meta }, &.{.{ .shape = &.{ @intCast(heads), blocks }, .dtype = .float32 }}, .{ blocks * block_size, @intCast(heads), 1 }, .{ block_size, 1, 1 }, .float32, &args))[0];
    return (try launch(g, k.topk, &.{ pooled, meta }, &.{.{ .shape = &.{ 1, @intCast(heads), 1, @intCast(topk) }, .dtype = .int32 }}, .{ @intCast(heads * 128), 1, 1 }, .{ 128, 1, 1 }, .float32, &args))[0];
}

/// q [1,H,1,128], k/v [1,KV,S,128], selected [1,KV,1,K]. Scratch is F32
/// partial numerator/max/denominator for each selected block. Invalid entries,
/// the incomplete tail and future keys contribute zero. Reduction/exp order
/// differs from dense SDPA: compare numerically, not bitwise, before enabling.
/// K/V use all four explicit strides; capacity slices remain views. Q and block
/// IDs are made contiguous because their kernel offsets are packed.
pub fn attention(g: *ops.G, q: T, k: T, v: T, selected: T, position: u32, scale: f32) !T {
    const qs = g.shapeOf(q);
    const ks = g.shapeOf(k);
    const vs = g.shapeOf(v);
    const bs = g.shapeOf(selected);
    if (qs.n != 4 or ks.n != 4 or bs.n != 4 or qs.d[0] != 1 or ks.d[0] != 1 or qs.d[2] != 1 or bs.d[0] != 1 or bs.d[2] != 1 or ks.d[2] < 1 or !std.mem.eql(c_int, ks.slice(), vs.slice())) return error.MiniMaxSparseShape;
    const h = qs.d[1];
    const kv = ks.d[1];
    const topk = bs.d[3];
    if (h < 1 or kv < 1 or !supported(@intCast(qs.d[3]), block_size, @intCast(h), @intCast(kv)) or ks.d[3] != qs.d[3] or bs.d[1] != kv or topk < 1 or topk > 16 or position >= @as(u32, @intCast(ks.d[2])) or !std.math.isFinite(scale) or scale <= 0) return error.MiniMaxSparseShape;
    const dt = g.dtypeOf(q);
    if ((dt != .bfloat16 and dt != .float16 and dt != .float32) or g.dtypeOf(k) != dt or g.dtypeOf(v) != dt or g.dtypeOf(selected) != .int32) return error.MiniMaxSparseShape;
    try available();
    const kernels_ = try getKernels();
    const meta_values: [8]i32 = .{ ks.d[2], 0, 0, 0, 0, 0, 0, 0 };
    const positions: [8]i32 = @splat(@intCast(position));
    const scales: [8]f32 = @splat(scale);
    const meta = try g.hostArray(std.mem.sliceAsBytes(&meta_values), &.{8}, .int32);
    const qpos = try g.hostArray(std.mem.sliceAsBytes(&positions), &.{8}, .int32);
    const sm = try g.hostArray(std.mem.sliceAsBytes(&scales), &.{8}, .float32);
    const args = [_]Template{ .{ .name = "H_KV", .value = kv }, .{ .name = "G", .value = @divExact(h, kv) }, .{ .name = "L", .value = 1 }, .{ .name = "D", .value = 128 }, .{ .name = "BLOCK", .value = block_size }, .{ .name = "TOPK", .value = topk }, .{ .name = "HAS_MASK", .value = 0 }, .{ .name = "HAS_PAD", .value = 0 } };
    const parts = try launch(g, kernels_.partials, &.{ try contiguous(g, q), k, v, try contiguous(g, selected), qpos, qpos, meta, sm, qpos }, &.{ .{ .shape = &.{ h * topk, 128 }, .dtype = .float32 }, .{ .shape = &.{h * topk}, .dtype = .float32 }, .{ .shape = &.{h * topk}, .dtype = .float32 } }, .{ topk * block_size, kv, 1 }, .{ block_size, 1, 1 }, dt, &args);
    return (try launch(g, kernels_.combine, &parts, &.{.{ .shape = qs.slice(), .dtype = dt }}, .{ 128, h, 1 }, .{ 128, 1, 1 }, dt, &args))[0];
}

test "MiniMax sparse decode supports portable M2 geometry and refuses wider layouts" {
    try std.testing.expect(supported(128, 128, 64, 4));
    try std.testing.expect(supported(128, 128, 4, 2));
    try std.testing.expect(!supported(128, 128, 64, 0));
    try std.testing.expect(!supported(64, 128, 64, 4));
    try std.testing.expect(!supported(128, 64, 64, 4));
    try std.testing.expect(!supported(128, 128, 65, 4));
    try std.testing.expect(!supported(128, 128, 128, 4));
}

test "MiniMax GPU sparse decode matches independent block selection and dense mathematical attention" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ref = @import("reference.zig");
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    const tokens = 513;
    const position = 500;
    const heads = 64;
    const kv_heads = 4;
    const dim = 128;
    const q_values = try a.alloc(f32, heads * dim);
    defer a.free(q_values);
    const k_values = try a.alloc(f32, kv_heads * tokens * dim);
    defer a.free(k_values);
    const v_values = try a.alloc(f32, k_values.len);
    defer a.free(v_values);
    // Binary fractions are exactly BF16-representable, so the CPU reference
    // consumes precisely the same inputs under both tested device dtypes.
    for (q_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
    for (0..kv_heads) |h| for (0..tokens) |t| for (0..dim) |d| {
        const i = (h * tokens + t) * dim + d;
        k_values[i] = @as(f32, @floatFromInt((@as(i32, @intCast(t % 13)) - 6) * (@as(i32, @intCast(d % 3)) - 1))) / 16;
        v_values[i] = @as(f32, @floatFromInt((@as(i32, @intCast(h * 3 + t % 19)) - 9) * (@as(i32, @intCast(d % 7)) - 3))) / 32;
    };
    var scores: [kv_heads * tokens]f32 = undefined;
    for (0..kv_heads) |h| for (0..tokens) |t| {
        const b = t / block_size;
        scores[h * tokens + t] = if (t > position) 10000 else if (b == h % 3) 9 else 0;
    };
    const expected_mask = try a.alloc(bool, scores.len);
    defer a.free(expected_mask);
    const actual_mask = try a.alloc(bool, scores.len);
    defer a.free(actual_mask);
    var pooled: [kv_heads * 5]f32 = undefined;
    var reference_out: [heads * dim]f32 = undefined;
    var reference_scratch: [tokens]f32 = undefined;
    const scale = 1 / @sqrt(@as(f32, dim));
    for ([_]u32{ 2, 16 }) |topk| {
        const mark = g.mark();
        defer g.resetTo(mark);
        try ref.selectSparseBlocks(&scores, kv_heads, tokens, position, block_size, topk, 1, 0, expected_mask, &pooled);
        const score_array = try g.hostArray(std.mem.sliceAsBytes(&scores), &.{ 1, kv_heads, 1, tokens }, .float32);
        const ids = try select(&g, score_array, kv_heads, tokens, position, topk, 1, 0);
        var selected_host: [kv_heads * 16]f32 = undefined;
        _ = try g.hostF32(try g.astype(ids, .float32), selected_host[0 .. kv_heads * topk]);
        @memset(actual_mask, false);
        for (0..kv_heads) |h| for (0..topk) |rank| {
            const value = selected_host[h * topk + rank];
            if (value == -1) continue;
            try std.testing.expect(value >= 0 and value == @floor(value) and value < 5);
            const b: usize = @intFromFloat(value);
            @memset(actual_mask[h * tokens + b * block_size .. h * tokens + @min((b + 1) * block_size, position + 1)], true);
        };
        try std.testing.expectEqualSlices(bool, expected_mask, actual_mask);
        try ref.groupedAttention(q_values, k_values, v_values, heads, kv_heads, kv_heads, tokens, dim, expected_mask, &reference_out, &reference_scratch);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
            const q = try g.astype(try g.hostArray(std.mem.sliceAsBytes(q_values), &.{ 1, heads, 1, dim }, .float32), dtype);
            const k = try g.astype(try g.hostArray(std.mem.sliceAsBytes(k_values), &.{ 1, kv_heads, tokens, dim }, .float32), dtype);
            const v = try g.astype(try g.hostArray(std.mem.sliceAsBytes(v_values), &.{ 1, kv_heads, tokens, dim }, .float32), dtype);
            // Keep padding between tokens so K/V cannot accidentally pass a
            // packed-offset kernel. Capacity-cache slices also retain gaps.
            const pad = try g.zeros(&.{ 1, kv_heads, tokens, 8 }, dtype);
            const padded_k = try g.concat(&.{ k, pad }, 3);
            const padded_v = try g.concat(&.{ v, pad }, 3);
            const strided_k = try g.slice(padded_k, &.{ 0, 0, 0, 0 }, &.{ 1, kv_heads, tokens, dim }, &.{ 1, 1, 1, 1 });
            const strided_v = try g.slice(padded_v, &.{ 0, 0, 0, 0 }, &.{ 1, kv_heads, tokens, dim }, &.{ 1, 1, 1, 1 });
            const out = try attention(&g, q, strided_k, strided_v, ids, position, scale);
            var actual: [heads * dim]f32 = undefined;
            _ = try g.hostF32(try g.astype(out, .float32), &actual);
            const tolerance: f32 = if (dtype == .float32) 2e-5 else 4e-3;
            for (actual, reference_out) |got, want| {
                try std.testing.expect(std.math.isFinite(got));
                try std.testing.expectApproxEqAbs(want, got, tolerance);
            }
        }
    }
    // All ties outside the forced local block retain the scalar's lower ID.
    @memset(&scores, 0);
    const tied = try g.hostArray(std.mem.sliceAsBytes(&scores), &.{ 1, kv_heads, 1, tokens }, .float32);
    const tied_ids = try select(&g, tied, kv_heads, tokens, position, 2, 1, 0);
    var tied_host: [kv_heads * 2]f32 = undefined;
    _ = try g.hostF32(try g.astype(tied_ids, .float32), &tied_host);
    for (0..kv_heads) |h| try std.testing.expectEqualSlices(f32, &.{ 3, 0 }, tied_host[h * 2 ..][0..2]);
}
