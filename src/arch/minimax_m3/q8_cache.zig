//! Opt-in symmetric Q8 cache operations. Only new rows are quantized; index
//! scores and selected-block attention restore values in registers on M2 SIMD.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const sparse = @import("sparse_decode.zig");
const T = ops.T;
const Kernel = mlx.mlx_fast_metal_kernel;
const Template = sparse.Template;
extern fn mlx_array_data_int8(mlx.mlx_array) ?[*]const i8;
pub const Projection = struct { codes: T, scales: T };
const Kernels = struct { quantize: Kernel, pool: Kernel, partials: Kernel };
var kernels: ?Kernels = null;

fn getKernels() !Kernels {
    if (kernels) |k| return k;
    const quant = try sparse.make("minimax_q8_quantize", &.{"x"}, &.{ "codes", "scales" }, @embedFile("kernels/minimax_q8_quantize.metal"));
    errdefer @import("../../sdk_ext/metal_launch.zig").mlx_fast_metal_kernel_free(quant);
    const pool = try sparse.make("minimax_q8_index_pool", &.{ "q", "k", "scales", "meta" }, &.{"pooled"}, @embedFile("kernels/minimax_q8_index_pool.metal"));
    errdefer @import("../../sdk_ext/metal_launch.zig").mlx_fast_metal_kernel_free(pool);
    const partials = try sparse.make("minimax_q8_partials", &.{ "q", "k", "v", "block_ids", "qpos", "pad", "meta", "scale", "vk", "kscale", "vscale" }, &.{ "part_o", "part_m", "part_l" }, @embedFile("kernels/minimax_q8_partials.metal"));
    const k: Kernels = .{ .quantize = quant, .pool = pool, .partials = partials };
    kernels = k;
    return k;
}

pub fn quantize(g: *ops.G, x: T) !Projection {
    const shape = g.shapeOf(x);
    const dt = g.dtypeOf(x);
    if (shape.n != 4 or shape.d[0] != 1 or shape.d[1] < 1 or shape.d[2] < 1 or shape.d[3] != 128 or (dt != .bfloat16 and dt != .float16 and dt != .float32)) return error.MiniMaxQ8Shape;
    try sparse.available();
    const groups = shape.d[1] * shape.d[2] * 4;
    const out = try sparse.launch(g, (try getKernels()).quantize, &.{try sparse.contiguous(g, x)}, &.{ .{ .shape = shape.slice(), .dtype = .int8 }, .{ .shape = &.{ 1, shape.d[1], shape.d[2], 4 }, .dtype = .float32 } }, .{ groups * 32, 1, 1 }, .{ 32, 1, 1 }, dt, &.{});
    return .{ .codes = out[0], .scales = out[1] };
}

fn validate(g: *ops.G, p: Projection) !@import("../deepseek_v41/ops.zig").Shape {
    const shape = g.shapeOf(p.codes);
    const sc = g.shapeOf(p.scales);
    if (shape.n != 4 or shape.d[0] != 1 or shape.d[1] < 1 or shape.d[2] < 1 or shape.d[3] != 128 or !std.mem.eql(c_int, sc.slice(), &.{ 1, shape.d[1], shape.d[2], 4 }) or g.dtypeOf(p.codes) != .int8 or g.dtypeOf(p.scales) != .float32) return error.MiniMaxQ8Shape;
    return shape;
}

/// The query stays at its model dtype. Pooling visits compressed index keys
/// directly and emits only one F32 score per head/block.
pub fn select(g: *ops.G, query: T, keys: Projection, position: u32, topk: u32, local: u32, initial: u32) !T {
    const shape = try validate(g, keys);
    const qs = g.shapeOf(query);
    if (shape.d[1] != 1 or qs.n != 4 or qs.d[0] != 1 or qs.d[1] < 1 or qs.d[2] != 1 or qs.d[3] != 128 or position >= @as(u32, @intCast(shape.d[2])) or topk == 0 or topk > 16) return error.MiniMaxQ8Shape;
    const dt = g.dtypeOf(query);
    if (dt != .float32 and dt != .float16 and dt != .bfloat16) return error.MiniMaxQ8Shape;
    try sparse.available();
    const blocks = @divTrunc(shape.d[2] + 127, 128);
    const meta_values: [8]i32 = .{ shape.d[2], blocks, @intCast(position), 0, 0, 0, 0, 0 };
    const meta = try g.hostArray(std.mem.sliceAsBytes(&meta_values), &.{8}, .int32);
    const args = [_]Template{ .{ .name = "LOCAL_BLOCKS", .value = @intCast(local) }, .{ .name = "INIT_BLOCKS", .value = @intCast(initial) } };
    const pooled = (try sparse.launch(g, (try getKernels()).pool, &.{ try sparse.contiguous(g, query), keys.codes, keys.scales, meta }, &.{.{ .shape = &.{ qs.d[1], blocks }, .dtype = .float32 }}, .{ blocks * 128, qs.d[1], 1 }, .{ 128, 1, 1 }, g.dtypeOf(query), &args))[0];
    return sparse.topkPooled(g, pooled, @intCast(qs.d[1]), @intCast(shape.d[2]), position, topk);
}

pub fn allBlocks(g: *ops.G, heads: u32, tokens: u32, position: u32, topk: u32) !T {
    if (heads == 0 or tokens == 0 or tokens > std.math.maxInt(c_int) or position >= tokens or topk == 0 or topk > 16) return error.MiniMaxQ8Shape;
    const blocks = (tokens + 127) / 128;
    return sparse.topkPooled(g, try g.zeros(&.{ @intCast(heads), @intCast(blocks) }, .float32), heads, tokens, position, topk);
}

pub fn attention(g: *ops.G, q: T, k: Projection, v: Projection, selected: T, position: u32, scale: f32) !T {
    const qs = g.shapeOf(q);
    const ks = try validate(g, k);
    const vs = try validate(g, v);
    const bs = g.shapeOf(selected);
    if (qs.n != 4 or qs.d[0] != 1 or qs.d[2] != 1 or qs.d[1] < 1 or !std.mem.eql(c_int, ks.slice(), vs.slice()) or bs.n != 4 or bs.d[0] != 1 or bs.d[1] != ks.d[1] or bs.d[2] != 1 or bs.d[3] < 1 or bs.d[3] > 16 or g.dtypeOf(selected) != .int32 or position >= @as(u32, @intCast(ks.d[2])) or !sparse.supported(@intCast(qs.d[3]), 128, @intCast(qs.d[1]), @intCast(ks.d[1])) or !std.math.isFinite(scale) or scale <= 0) return error.MiniMaxQ8Shape;
    const dt = g.dtypeOf(q);
    if (dt != .float32 and dt != .float16 and dt != .bfloat16) return error.MiniMaxQ8Shape;
    try sparse.available();
    const h = qs.d[1];
    const kv = ks.d[1];
    const topk = bs.d[3];
    const meta_values: [8]i32 = .{ ks.d[2], 0, 0, 0, 0, 0, 0, 0 };
    const positions: [8]i32 = @splat(@intCast(position));
    const scales: [8]f32 = @splat(scale);
    const meta = try g.hostArray(std.mem.sliceAsBytes(&meta_values), &.{8}, .int32);
    const qpos = try g.hostArray(std.mem.sliceAsBytes(&positions), &.{8}, .int32);
    const sm = try g.hostArray(std.mem.sliceAsBytes(&scales), &.{8}, .float32);
    const args = [_]Template{ .{ .name = "H_KV", .value = kv }, .{ .name = "G", .value = @divExact(h, kv) }, .{ .name = "L", .value = 1 }, .{ .name = "D", .value = 128 }, .{ .name = "BLOCK", .value = 128 }, .{ .name = "TOPK", .value = topk }, .{ .name = "HAS_MASK", .value = 0 }, .{ .name = "HAS_PAD", .value = 0 } };
    const parts = try sparse.launch(g, (try getKernels()).partials, &.{ try sparse.contiguous(g, q), k.codes, v.codes, try sparse.contiguous(g, selected), qpos, qpos, meta, sm, qpos, k.scales, v.scales }, &.{ .{ .shape = &.{ h * topk, 128 }, .dtype = .float32 }, .{ .shape = &.{h * topk}, .dtype = .float32 }, .{ .shape = &.{h * topk}, .dtype = .float32 } }, .{ topk * 128, kv, 1 }, .{ 128, 1, 1 }, g.dtypeOf(q), &args);
    return sparse.combineParts(g, &parts, qs.slice(), kv, topk, g.dtypeOf(q));
}

test {
    _ = @import("q8_reference.zig");
}

// The CPU oracle quantizes and restores synthetic inputs independently. The
// device consumes only compressed, strided capacity views throughout attention.
test "MiniMax GPU q8 cache matches scalar quantization selection and attention across growth" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const qr = @import("q8_reference.zig");
    const ref = @import("reference.zig");
    const Cache = @import("cache.zig").Cache;
    const a = std.testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try ops.G.init(a, stream);
    defer g.deinit();
    {
        var ties: [128]f32 = @splat(0);
        ties[0] = 127;
        ties[1] = -127;
        ties[2] = 63.5;
        ties[3] = -62.5;
        const qp = try quantize(&g, try g.hostArray(std.mem.sliceAsBytes(&ties), &.{ 1, 1, 1, 128 }, .float32));
        var got: [128]f32 = undefined;
        var got_scales: [4]f32 = undefined;
        _ = try g.hostF32(try g.astype(qp.codes, .float32), &got);
        _ = try g.hostF32(qp.scales, &got_scales);
        try std.testing.expectEqualSlices(f32, &.{ 127, -127, 64, -62 }, got[0..4]);
        try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &got_scales);
    }
    const tokens = 513;
    const position = 500;
    const heads = 4;
    const kv = 2;
    const dim = 128;
    var query: [heads * dim]f32 = undefined;
    for (&query, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
    const kval = try a.alloc(f32, kv * tokens * dim);
    defer a.free(kval);
    const vval = try a.alloc(f32, kval.len);
    defer a.free(vval);
    const ival = try a.alloc(f32, tokens * dim);
    defer a.free(ival);
    for (0..kv) |h| for (0..tokens) |t| for (0..dim) |d| {
        const i = (h * tokens + t) * dim + d;
        kval[i] = @as(f32, @floatFromInt(@as(i32, @intCast((t * 3 + d + h) % 127)) - 63)) / 32;
        vval[i] = @as(f32, @floatFromInt(@as(i32, @intCast((t + d * 5 + h) % 63)) - 31)) / 16;
    };
    for (0..tokens) |t| for (0..dim) |d| {
        ival[t * dim + d] = if (d < kv) @floatFromInt((t / 128 + d) % 3 + 1) else 0;
    };
    const kcodes = try a.alloc(i8, kval.len);
    defer a.free(kcodes);
    const vcodes = try a.alloc(i8, vval.len);
    defer a.free(vcodes);
    const icodes = try a.alloc(i8, ival.len);
    defer a.free(icodes);
    const ks = try a.alloc(f32, kval.len / 32);
    defer a.free(ks);
    const vs = try a.alloc(f32, vval.len / 32);
    defer a.free(vs);
    const is = try a.alloc(f32, ival.len / 32);
    defer a.free(is);
    try qr.quantize(kval, kcodes, ks);
    try qr.quantize(vval, vcodes, vs);
    try qr.quantize(ival, icodes, is);
    try qr.dequantize(kcodes, ks, kval);
    try qr.dequantize(vcodes, vs, vval);
    try qr.dequantize(icodes, is, ival);
    // Recreate the original exactly representable model rows for device input.
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
        const tolerance: f32 = if (dtype == .float32) 2e-5 else 4e-3;
        var cache: Cache = .{ .limit = tokens, .bits = 8 };
        defer cache.clear(&g);
        var pointers: [6]usize = @splat(0);
        var previous_capacity: u32 = 0;
        for ([_]u32{ 1, 7, 249, 256 }) |count| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const rows = try a.alloc(f32, kv * count * dim);
            defer a.free(rows);
            const values = try a.alloc(f32, rows.len);
            defer a.free(values);
            const index = try a.alloc(f32, count * dim);
            defer a.free(index);
            for (0..kv) |h| for (0..count) |t| for (0..dim) |d| {
                const at = cache.len + t;
                rows[(h * count + t) * dim + d] = @as(f32, @floatFromInt(@as(i32, @intCast((at * 3 + d + h) % 127)) - 63)) / 32;
                values[(h * count + t) * dim + d] = @as(f32, @floatFromInt(@as(i32, @intCast((at + d * 5 + h) % 63)) - 31)) / 16;
            };
            for (0..count) |t| for (0..dim) |d| {
                index[t * dim + d] = if (d < kv) @floatFromInt(((cache.len + t) / 128 + d) % 3 + 1) else 0;
            };
            const k = try g.astype(try g.hostArray(std.mem.sliceAsBytes(rows), &.{ 1, kv, @intCast(count), dim }, .float32), dtype);
            const v = try g.astype(try g.hostArray(std.mem.sliceAsBytes(values), &.{ 1, kv, @intCast(count), dim }, .float32), dtype);
            const ik = try g.astype(try g.hostArray(std.mem.sliceAsBytes(index), &.{ 1, 1, @intCast(count), dim }, .float32), dtype);
            _ = try cache.append(&g, k, v, ik);
            const current: [6]usize = .{
                @intFromPtr(mlx_array_data_int8(cache.k.?).?),
                @intFromPtr(mlx_array_data_int8(cache.v.?).?),
                @intFromPtr(mlx_array_data_int8(cache.index_k.?).?),
                @intFromPtr(mlx.mlx_array_data_float32(cache.k_scales.?).?),
                @intFromPtr(mlx.mlx_array_data_float32(cache.v_scales.?).?),
                @intFromPtr(mlx.mlx_array_data_float32(cache.index_scales.?).?),
            };
            if (cache.capacity == previous_capacity) try std.testing.expectEqualSlices(usize, &pointers, &current);
            pointers = current;
            previous_capacity = cache.capacity;
        }
        try std.testing.expectEqual(@as(u32, 768), cache.capacity);
        const mark = g.mark();
        defer g.resetTo(mark);
        const view = try cache.view(&g, tokens);
        const got_codes = try a.alloc(f32, kcodes.len);
        defer a.free(got_codes);
        _ = try g.hostF32(try g.astype(view.k, .float32), got_codes);
        for (got_codes, kcodes) |actual, want| try std.testing.expectEqual(@as(f32, @floatFromInt(want)), actual);
        const got_scales = try a.alloc(f32, ks.len);
        defer a.free(got_scales);
        _ = try g.hostF32(view.k_scales.?, got_scales);
        for (got_scales, ks) |actual, want| try std.testing.expectApproxEqAbs(want, actual, 1e-7);
        var iq: [kv * dim]f32 = @splat(0);
        for (0..kv) |h| iq[h * dim + h] = 1;
        const index_q = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&iq), &.{ 1, kv, 1, dim }, .float32), dtype);
        var scores: [kv * tokens]f32 = undefined;
        for (0..kv) |h| for (0..tokens) |t| {
            scores[h * tokens + t] = ival[t * dim + h];
        };
        var expected_mask: [kv * tokens]bool = undefined;
        var pooled: [kv * 5]f32 = undefined;
        var actual_mask: [kv * tokens]bool = undefined;
        var scratch: [tokens]f32 = undefined;
        var expected: [heads * dim]f32 = undefined;
        for ([_]u32{ 2, 16 }) |topk| {
            const selected = try select(&g, index_q, .{ .codes = view.index_k, .scales = view.index_scales.? }, position, topk, 1, 0);
            try ref.selectSparseBlocks(&scores, kv, tokens, position, 128, topk, 1, 0, &expected_mask, &pooled);
            var ids: [kv * 16]f32 = undefined;
            _ = try g.hostF32(try g.astype(selected, .float32), ids[0 .. kv * topk]);
            @memset(&actual_mask, false);
            for (0..kv) |h| for (0..topk) |rank| {
                if (ids[h * topk + rank] < 0) continue;
                const block: usize = @intFromFloat(ids[h * topk + rank]);
                try std.testing.expect(block < 4);
                @memset(actual_mask[h * tokens + block * 128 .. h * tokens + @min((block + 1) * 128, position + 1)], true);
            };
            try std.testing.expectEqualSlices(bool, &expected_mask, &actual_mask);
            try ref.groupedAttention(&query, kval, vval, heads, kv, kv, tokens, dim, &expected_mask, &expected, &scratch);
            const q = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&query), &.{ 1, heads, 1, dim }, .float32), dtype);
            const out = try attention(&g, q, .{ .codes = view.k, .scales = view.k_scales.? }, .{ .codes = view.v, .scales = view.v_scales.? }, selected, position, 1 / @sqrt(@as(f32, dim)));
            var actual: [heads * dim]f32 = undefined;
            _ = try g.hostF32(try g.astype(out, .float32), &actual);
            for (actual, expected) |got, want| try std.testing.expectApproxEqAbs(want, got, tolerance);
        }
        const short = try cache.view(&g, 5);
        const all = try allBlocks(&g, kv, 5, 4, 16);
        const zero_q = try g.zeros(&.{ 1, heads, 1, dim }, dtype);
        const mean = try attention(&g, zero_q, .{ .codes = short.k, .scales = short.k_scales.? }, .{ .codes = short.v, .scales = short.v_scales.? }, all, 4, 1);
        var got_mean: [heads * dim]f32 = undefined;
        _ = try g.hostF32(try g.astype(mean, .float32), &got_mean);
        for (0..heads) |h| for (0..dim) |d| {
            var want: f32 = 0;
            for (0..5) |t| want += vval[((h / 2) * tokens + t) * dim + d] / 5;
            try std.testing.expectApproxEqAbs(want, got_mean[h * dim + d], tolerance);
        };
        cache.clear(&g);
        try std.testing.expectEqual(@as(u8, 8), cache.bits);
        try std.testing.expect(cache.k == null and cache.k_scales == null and cache.index_scales == null);
    }
}
