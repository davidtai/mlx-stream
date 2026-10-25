//! Serial text attention with per-index-head sparse block masks. Index keys are
//! retained even before sparsity activates, so later queries can select old blocks.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const C = @import("config.zig").Config;
const reference = @import("reference.zig");
pub const Weights = struct { q: ops.Linear, k: ops.Linear, v: ops.Linear, o: ops.Linear, q_norm: ops.T, k_norm: ops.T, index_q: ops.Linear, index_k: ops.Linear, index_q_norm: ops.T, index_k_norm: ops.T };
pub const Cache = @import("cache.zig").Cache;
const sparse = @import("sparse_decode.zig");
var sparse_engaged = false;
var q8_engaged = false;
fn heads(g: *ops.G, x: ops.T, n: u32, d: u32) !ops.T {
    return g.reshape(x, &.{ 1, @intCast(n), 1, @intCast(d) });
}

pub fn forward(g: *ops.G, a: std.mem.Allocator, c: *const C, w: Weights, cache: *Cache, x: ops.T, position: u32) !ops.T {
    return forwardChunk(g, a, c, w, cache, x, position);
}

/// Layer-major prompt chunks retain one-row projection/norm/RoPE arithmetic.
/// Only the newly projected rows are concatenated. Capacity storage is appended
/// once, then each query reads its own causal prefix without copying the cache.
pub fn forwardChunk(g: *ops.G, a: std.mem.Allocator, c: *const C, w: Weights, cache: *Cache, x: ops.T, position: u32) !ops.T {
    const sx = g.shapeOf(x);
    if (sx.n != 2 or sx.d[0] < 1 or sx.d[0] > 8 or sx.d[1] != c.hidden_size or position != cache.len) return error.MiniMaxAttentionChunk;
    const n: usize = @intCast(sx.d[0]);
    var q: [8]ops.T = undefined;
    var k: [8]ops.T = undefined;
    var v: [8]ops.T = undefined;
    var ik: [8]ops.T = undefined;
    var iq: [8]?ops.T = @splat(null);
    for (0..n) |i| {
        const row: c_int = @intCast(i);
        const xr = try g.slice(x, &.{ row, 0 }, &.{ row + 1, sx.d[1] }, &.{ 1, 1 });
        const pos = position + @as(u32, @intCast(i));
        q[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.q), c.n_heads, c.head_dim), w.q_norm, c), c.rotary_dim, c, pos);
        k[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.k), c.n_kv_heads, c.head_dim), w.k_norm, c), c.rotary_dim, c, pos);
        v[i] = try heads(g, try ops.linear(g, xr, w.v), c.n_kv_heads, c.head_dim);
        ik[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.index_k), 1, c.index_head_dim), w.index_k_norm, c), @min(c.rotary_dim, c.index_head_dim), c, pos);
        if (pos / c.index_block_size + 1 > c.index_topk_blocks)
            iq[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.index_q), c.index_n_heads, c.index_head_dim), w.index_q_norm, c), @min(c.rotary_dim, c.index_head_dim), c, pos);
    }
    _ = try cache.append(g, if (n == 1) k[0] else try g.concat(k[0..n], 2), if (n == 1) v[0] else try g.concat(v[0..n], 2), if (n == 1) ik[0] else try g.concat(ik[0..n], 2));
    var outputs: [8]ops.T = undefined;
    for (0..n) |i| {
        const pos = position + @as(u32, @intCast(i));
        const view = try cache.view(g, pos + 1);
        const attended = try attend(g, a, c, q[i], iq[i], view, pos);
        try g.evalAll(&.{attended});
        outputs[i] = try ops.linear(g, try g.reshape(attended, &.{ 1, @intCast(c.n_heads * c.head_dim) }), w.o);
    }
    return if (n == 1) outputs[0] else g.concat(outputs[0..n], 0);
}

fn attend(g: *ops.G, a: std.mem.Allocator, c: *const C, q: ops.T, iq: ?ops.T, view: Cache.View, position: u32) !ops.T {
    var mask: ops.T = .{};
    const tokens = position + 1;
    const blocks = (tokens + c.index_block_size - 1) / c.index_block_size;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    if (view.bits == 8) {
        const q8 = @import("q8_cache.zig");
        if (c.index_n_heads != c.n_kv_heads or !sparse.supported(c.head_dim, c.index_block_size, c.n_heads, c.n_kv_heads)) return error.MiniMaxQ8Shape;
        const selected = if (blocks > c.index_topk_blocks)
            try q8.select(g, iq.?, .{ .codes = view.index_k, .scales = view.index_scales.? }, position, c.index_topk_blocks, c.index_local_blocks, c.index_init_blocks)
        else
            try q8.allBlocks(g, c.n_kv_heads, tokens, position, c.index_topk_blocks);
        const out = try q8.attention(g, q, .{ .codes = view.k, .scales = view.k_scales.? }, .{ .codes = view.v, .scales = view.v_scales.? }, selected, position, scale);
        if (!q8_engaged) {
            sdk.log.info("minimax_m3: Q8 KV/index cache engaged (symmetric group32, portable SIMD)\n", .{});
            q8_engaged = true;
        }
        return out;
    }
    if (blocks > c.index_topk_blocks) {
        const scores = try g.matmul(try g.astype(iq.?, .float32), try g.transposeAxes(try g.astype(view.index_k, .float32), &.{ 0, 1, 3, 2 }));
        if (c.index_n_heads == c.n_kv_heads and c.index_topk_blocks <= 16 and sparse.supported(c.head_dim, c.index_block_size, c.n_heads, c.n_kv_heads)) {
            const selected = try sparse.select(g, scores, c.index_n_heads, tokens, position, c.index_topk_blocks, c.index_local_blocks, c.index_init_blocks);
            const out = try sparse.attention(g, q, view.k, view.v, selected, position, scale);
            if (!sparse_engaged) {
                sdk.log.info("minimax_m3: GPU selected-block attention engaged (portable SIMD)\n", .{});
                sparse_engaged = true;
            }
            return out;
        }
        // The small host oracle geometries retain the general masked route.
        const host_scores = try a.alloc(f32, @as(usize, c.index_n_heads) * tokens);
        defer a.free(host_scores);
        _ = try g.hostF32(scores, host_scores);
        const selected = try a.alloc(bool, host_scores.len);
        defer a.free(selected);
        const scratch = try a.alloc(f32, @as(usize, c.index_n_heads) * blocks);
        defer a.free(scratch);
        try reference.selectSparseBlocks(host_scores, c.index_n_heads, tokens, position, c.index_block_size, c.index_topk_blocks, c.index_local_blocks, c.index_init_blocks, selected, scratch);
        const expanded = try a.alloc(bool, @as(usize, c.n_heads) * tokens);
        defer a.free(expanded);
        for (0..c.n_heads) |h| {
            const ih = h / (c.n_heads / c.index_n_heads);
            @memcpy(expanded[h * tokens ..][0..tokens], selected[ih * tokens ..][0..tokens]);
        }
        mask = try g.hostArray(std.mem.sliceAsBytes(expanded), &.{ 1, @intCast(c.n_heads), 1, @intCast(tokens) }, .bool_);
    }
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, view.k, view.v, scale, if (mask.ctx != null) "array" else "", mask, .{}, false, g.s)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}

// Opt-in local device fixture: verifies the actual graph crosses the block
// budget, drops old blocks, retains index keys and widens each KV group correctly.
test "MiniMax GPU attention crosses dense-to-sparse boundary with retained index keys" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    var c: C = .{};
    c.hidden_size = 4;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 2;
    c.rotary_dim = 2;
    c.index_n_heads = 2;
    c.index_head_dim = 2;
    c.index_block_size = 2;
    c.index_topk_blocks = 1;
    const identity = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const output = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
    const w: Weights = .{
        .q = .{ .weight = try g.zeros(&.{ 8, 4 }, .float32) },
        .k = .{ .weight = try g.zeros(&.{ 4, 4 }, .float32) },
        .v = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&identity), &.{ 4, 4 }, .float32) },
        .o = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&output), &.{ 4, 8 }, .float32) },
        .q_norm = try g.zeros(&.{2}, .float32),
        .k_norm = try g.zeros(&.{2}, .float32),
        .index_q = .{ .weight = try g.zeros(&.{ 4, 4 }, .float32) },
        .index_k = .{ .weight = try g.zeros(&.{ 2, 4 }, .float32) },
        .index_q_norm = try g.zeros(&.{2}, .float32),
        .index_k_norm = try g.zeros(&.{2}, .float32),
    };
    var cache: Cache = .{};
    defer cache.clear(&g);
    const rows = [_][4]f32{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 10, 11, 12 } };
    const wants = [_][4]f32{ .{ 1, 2, 3, 4 }, .{ 3, 4, 5, 6 }, .{ 9, 10, 11, 12 } };
    for (rows, wants, 0..) |row, want, position| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try g.hostArray(std.mem.sliceAsBytes(&row), &.{ 1, 4 }, .float32);
        const y = try forward(&g, a, &c, w, &cache, x, @intCast(position));
        var actual: [4]f32 = undefined;
        _ = try g.hostF32(y, &actual);
        for (actual, want) |got, expected| try std.testing.expectApproxEqAbs(expected, got, 1e-5);
        try std.testing.expectEqual(@as(u32, @intCast(position + 1)), cache.len);
        const retained = try cache.view(&g, cache.len);
        try std.testing.expectEqual(@as(c_int, @intCast(position + 1)), g.shapeOf(retained.index_k).d[2]);
    }
}

test "MiniMax GPU chunk attention matches serial queries across cache growth and sparse block boundaries" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    var c: C = .{};
    c.hidden_size = 4;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 128;
    c.rotary_dim = 64;
    c.index_n_heads = 2;
    c.index_head_dim = 128;
    c.index_block_size = 128;
    c.index_topk_blocks = 1;
    var v_weights: [256 * 4]f32 = @splat(0);
    for (0..256) |i| v_weights[i * 4 + i % 4] = 1;
    var o_weights: [4 * 512]f32 = @splat(0);
    for (0..4) |i| o_weights[i * 512 + i] = 1;
    const w: Weights = .{
        .q = .{ .weight = try g.zeros(&.{ 512, 4 }, .float32) },
        .k = .{ .weight = try g.zeros(&.{ 256, 4 }, .float32) },
        .v = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&v_weights), &.{ 256, 4 }, .float32) },
        .o = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&o_weights), &.{ 4, 512 }, .float32) },
        .q_norm = try g.zeros(&.{128}, .float32),
        .k_norm = try g.zeros(&.{128}, .float32),
        .index_q = .{ .weight = try g.zeros(&.{ 256, 4 }, .float32) },
        .index_k = .{ .weight = try g.zeros(&.{ 128, 4 }, .float32) },
        .index_q_norm = try g.zeros(&.{128}, .float32),
        .index_k_norm = try g.zeros(&.{128}, .float32),
    };
    var serial: Cache = .{ .limit = 300 };
    defer serial.clear(&g);
    var chunked: Cache = .{ .limit = 300 };
    defer chunked.clear(&g);
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        const history = try a.alloc(f32, 2 * 250 * 128);
        defer a.free(history);
        for (0..2) |h| for (0..250) |t| for (0..128) |d| {
            history[(h * 250 + t) * 128 + d] = @as(f32, @floatFromInt(t + d % 4)) / 256;
        };
        const k = try g.zeros(&.{ 1, 2, 250, 128 }, .float32);
        const v = try g.hostArray(std.mem.sliceAsBytes(history), &.{ 1, 2, 250, 128 }, .float32);
        const index = try g.zeros(&.{ 1, 1, 250, 128 }, .float32);
        _ = try serial.append(&g, k, v, index);
        _ = try chunked.append(&g, k, v, index);
    }
    var rows: [8 * 4]f32 = undefined;
    for (0..8) |t| for (0..4) |d| {
        rows[t * 4 + d] = @as(f32, @floatFromInt(t * 4 + d + 10)) / 8;
    };
    var expected: [8 * 4]f32 = undefined;
    for (0..8) |i| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const row = rows[i * 4 ..][0..4];
        const x = try g.hostArray(std.mem.sliceAsBytes(row), &.{ 1, 4 }, .float32);
        const y = try forward(&g, a, &c, w, &serial, x, @intCast(250 + i));
        _ = try g.hostF32(y, expected[i * 4 ..][0..4]);
    }
    const input = try g.hostArray(std.mem.sliceAsBytes(&rows), &.{ 8, 4 }, .float32);
    const out = try forwardChunk(&g, a, &c, w, &chunked, input, 250);
    var actual: [8 * 4]f32 = undefined;
    _ = try g.hostF32(out, &actual);
    for (actual, expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    try std.testing.expectEqual(@as(u32, 258), chunked.len);
    try std.testing.expectEqual(serial.len, chunked.len);
    try std.testing.expectEqual(serial.capacity, chunked.capacity);
    // The new block begins at token256: query256 must see exactly its own
    // value, despite all eight rows having landed before attention started.
    for (0..4) |d| try std.testing.expectApproxEqAbs(rows[6 * 4 + d], actual[6 * 4 + d], 1e-6);
}
