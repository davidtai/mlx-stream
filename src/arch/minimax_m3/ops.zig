//! MiniMax text graph primitives over the shared MLX backend.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
pub const G = @import("../deepseek_v41/ops.zig").MlxOps;
pub const T = mlx.mlx_array;
const C = @import("config.zig").Config;
const layout = @import("resident_layout.zig");
pub const Linear = LinearOf(T);
pub fn LinearOf(comptime Tensor: type) type {
    return struct {
        weight: Tensor,
        scales: ?Tensor = null,
        biases: ?Tensor = null,

        pub fn validate(self: @This(), g: anytype, format: layout.Format, outputs: c_int, inputs: c_int) !void {
            _ = try self.parameters(g, format, outputs, inputs);
        }
        pub fn parameters(self: @This(), g: anytype, format: layout.Format, outputs: c_int, inputs: c_int) !layout.Params {
            const weight_shape = g.shapeOf(self.weight);
            const weight: layout.Tensor = .{ .dtype = dtype(g.dtypeOf(self.weight)), .shape = weight_shape.slice() };
            if (self.scales) |scales| {
                const scale_shape = g.shapeOf(scales);
                const scale: layout.Tensor = .{ .dtype = dtype(g.dtypeOf(scales)), .shape = scale_shape.slice() };
                if (self.biases) |biases| {
                    const bias_shape = g.shapeOf(biases);
                    return layout.parameters(format, weight, scale, .{ .dtype = dtype(g.dtypeOf(biases)), .shape = bias_shape.slice() }, outputs, inputs);
                }
                return layout.parameters(format, weight, scale, null, outputs, inputs);
            }
            if (self.biases != null) return error.MiniMaxProjectionScalesMissing;
            return layout.parameters(format, weight, null, null, outputs, inputs);
        }
    };
}
fn dtype(value: mlx.mlx_dtype) layout.Dtype {
    return switch (value) {
        .bfloat16 => .bf16,
        .float32 => .f32,
        .uint32 => .u32,
        .uint8 => .u8,
        else => .other,
    };
}
pub fn linear(g: anytype, x: @TypeOf(g.*).T, projection: anytype) !@TypeOf(g.*).T {
    if (comptime @TypeOf(projection) == @TypeOf(g.*).T) {
        return g.matmul(x, try g.transpose(projection));
    } else {
        const w: LinearOf(@TypeOf(g.*).T) = projection;
        if (w.scales) |scales| {
            if (w.biases) |biases| {
                const params = try w.parameters(g, .mixed, g.shapeOf(w.weight).dim(-2), g.shapeOf(x).dim(-1));
                return g.quantizedMatmul(x, w.weight, scales, biases, true, params.bits, params.group_size, .affine);
            }
            return g.qmm(x, w.weight, scales, .mxfp8);
        }
        return g.matmul(x, try g.transpose(w.weight));
    }
}
pub fn embeddingRows(g: anytype, embedding: LinearOf(@TypeOf(g.*).T), ids: @TypeOf(g.*).T, hidden: u32) !@TypeOf(g.*).T {
    const weights = try g.take(embedding.weight, ids, 0);
    const scales = embedding.scales orelse return weights;
    const selected_scales = try g.take(scales, ids, 0);
    const params = try embedding.parameters(g, .mixed, g.shapeOf(embedding.weight).dim(-2), @intCast(hidden));
    const biases = if (embedding.biases) |b| try g.take(b, ids, 0) else null;
    return g.dequantizeWith(weights, selected_scales, biases, params.bits, params.group_size, if (params.kind == .affine) .affine else .mxfp8);
}
pub fn norm(g: *G, x: T, w: T, c: *const C) !T {
    var out = mlx.mlx_array_new();
    const weight = try g.add(w, try g.scalar(1, g.dtypeOf(w)));
    mlx.check(mlx.mlx_fast_rms_norm(&out, x, weight, c.rms_norm_eps, g.s)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}
pub fn rope(g: *G, x: T, dims: u32, c: *const C, position: u32) !T {
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_fast_rope(&out, x, @intCast(dims), false, .some(c.rope_theta), 1, @intCast(position), .{}, g.s)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}
/// SwiGLUOAI clips gate only above, clips up on both sides, and includes beta.
pub fn activate(g: *G, gate: T, up: T, c: *const C) !T {
    const dt = g.dtypeOf(gate);
    const limit = try g.scalar(c.swiglu_limit, dt);
    const gc = try g.minimum(gate, limit);
    const uc = try g.minimum(try g.maximum(up, try g.scalar(-c.swiglu_limit, dt)), limit);
    const sig = try g.sigmoid(try g.mul(gc, try g.scalar(c.swiglu_alpha, dt)));
    return g.mul(try g.mul(gc, sig), try g.add(uc, try g.scalar(c.swiglu_beta, dt)));
}
pub const Mlp = struct {
    gate: Linear,
    up: Linear,
    down: Linear,
    pub fn apply(m: Mlp, g: *G, x: T, c: *const C) !T {
        return linear(g, try activate(g, try linear(g, x, m.gate), try linear(g, x, m.up), c), m.down);
    }
    pub fn applyRows(m: Mlp, g: *G, x: T, c: *const C) !T {
        const shape = g.shapeOf(x);
        if (shape.n != 2 or shape.d[0] < 1 or shape.d[0] > 8) return error.MiniMaxChunkShape;
        var rows: [8]T = undefined;
        for (0..@intCast(shape.d[0])) |i| rows[i] = try m.apply(g, try row(g, x, @intCast(i)), c);
        return g.concat(rows[0..@intCast(shape.d[0])], 0);
    }
};

pub fn row(g: *G, x: T, index: c_int) !T {
    const shape = g.shapeOf(x);
    return g.slice(x, &.{ index, 0 }, &.{ index + 1, shape.d[1] }, &.{ 1, 1 });
}
pub fn normRows(g: *G, x: T, weight: T, c: *const C) !T {
    const shape = g.shapeOf(x);
    if (shape.n != 2 or shape.d[0] < 1 or shape.d[0] > 8) return error.MiniMaxChunkShape;
    var rows: [8]T = undefined;
    for (0..@intCast(shape.d[0])) |i| rows[i] = try norm(g, try row(g, x, @intCast(i)), weight, c);
    return g.concat(rows[0..@intCast(shape.d[0])], 0);
}

test "MiniMax MXFP8 linear trace keeps packed weights without dequantization" {
    const Trace = @import("../deepseek_v41/ops.zig").TraceOps;
    var g = Trace.init(std.testing.allocator);
    defer g.deinit();
    const input = try g.input(&.{ 1, 512 }, .bfloat16);
    const weight = try g.input(&.{ 512, 128 }, .uint32);
    const scales = try g.input(&.{ 512, 16 }, .uint8);
    const projection: LinearOf(Trace.T) = .{ .weight = weight, .scales = scales };
    try projection.validate(&g, .mxfp8, 512, 512);
    const before = g.nodes.items.len;
    const output = try linear(&g, input, projection);
    try std.testing.expectEqual(before + 1, g.nodes.items.len);
    try std.testing.expectEqual(@import("../deepseek_v41/ops.zig").Op.qmm, g.nodes.items[output].op);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, g.dtypeOf(output));
    try std.testing.expectEqualSlices(c_int, &.{ 1, 512 }, g.shapeOf(output).slice());
    const dense = try g.input(&.{ 512, 512 }, .bfloat16);
    const dense_projection: LinearOf(Trace.T) = .{ .weight = dense };
    try dense_projection.validate(&g, .bf16, 512, 512);
    const dense_output = try linear(&g, input, dense_projection);
    try std.testing.expectEqual(@import("../deepseek_v41/ops.zig").Op.matmul, g.nodes.items[dense_output].op);
}

test "MiniMax mixed affine trace uses native matmul and expands only selected embedding rows" {
    const Trace = @import("../deepseek_v41/ops.zig").TraceOps;
    var g = Trace.init(std.testing.allocator);
    defer g.deinit();
    const input = try g.input(&.{ 1, 512 }, .bfloat16);
    for ([_]c_int{ 3, 4, 5, 6, 8 }) |bits| {
        const projection: LinearOf(Trace.T) = .{ .weight = try g.input(&.{ 16, bits * 16 }, .uint32), .scales = try g.input(&.{ 16, 8 }, .bfloat16), .biases = try g.input(&.{ 16, 8 }, .bfloat16) };
        try projection.validate(&g, .mixed, 16, 512);
        const before = g.nodes.items.len;
        const output = try linear(&g, input, projection);
        try std.testing.expectEqual(before + 1, g.nodes.items.len);
        try std.testing.expectEqual(@import("../deepseek_v41/ops.zig").Op.qmm, g.nodes.items[output].op);
    }
    const embedding: LinearOf(Trace.T) = .{ .weight = try g.input(&.{ 200064, 128 }, .uint32), .scales = try g.input(&.{ 200064, 8 }, .bfloat16), .biases = try g.input(&.{ 200064, 8 }, .bfloat16) };
    const ids = try g.input(&.{3}, .uint32);
    const rows = try embeddingRows(&g, embedding, ids, 512);
    try std.testing.expectEqualSlices(c_int, &.{ 3, 512 }, g.shapeOf(rows).slice());
    try std.testing.expectEqual(@import("../deepseek_v41/ops.zig").Op.dequantize, g.nodes.items[rows].op);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, g.dtypeOf(rows));
}

test "MiniMax GPU MXFP8 resident linear preserves source codes and bounded BF16 parity" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(std.testing.allocator, stream);
    defer g.deinit();
    // MXFP8 qmv_fast requires K%256 and N%8; 512 also matches the native
    // resident geometry's dispatch family. Heap storage bounds the test stack.
    const n = 512;
    const words = try std.testing.allocator.alloc(u32, n * n / 4);
    defer std.testing.allocator.free(words);
    const scales = try std.testing.allocator.alloc(u8, n * n / 32);
    defer std.testing.allocator.free(scales);
    const dense_values = try std.testing.allocator.alloc(f32, n * n);
    defer std.testing.allocator.free(dense_values);
    const codes = [_]u8{ 0x21, 0x2b, 0x35, 0x3d, 0x42, 0x49, 0x51, 0x59 };
    const magnitudes = [_]f32{ 0.140625, 0.34375, 0.8125, 1.625, 2.5, 4.5, 9, 18 };
    const exponents = [_]u8{ 124, 125, 126, 127, 128, 129 };
    const multipliers = [_]f32{ 0.125, 0.25, 0.5, 1, 2, 4 };
    var state: u32 = 0x120843;
    @memset(words, 0);
    for (0..n) |out| {
        for (0..n / 32) |block| {
            state = state *% 1664525 +% 1013904223;
            const si = state % exponents.len;
            scales[out * (n / 32) + block] = exponents[si];
            for (0..32) |within| {
                state = state *% 1664525 +% 1013904223;
                const ci = (state >> 8) % codes.len;
                const negative = state & 0x80000000 != 0;
                const code = codes[ci] | (if (negative) @as(u8, 0x80) else 0);
                const k = block * 32 + within;
                words[out * (n / 4) + k / 4] |= @as(u32, code) << @as(u5, @intCast((k % 4) * 8));
                dense_values[out * n + k] = magnitudes[ci] * multipliers[si] * (if (negative) @as(f32, -1) else 1);
            }
        }
    }
    var input_values: [n]f32 = undefined;
    for (&input_values) |*value| {
        state = state *% 1664525 +% 1013904223;
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((state >> 8) % 33)) - 16)) / 256;
    }
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input_values), &.{ 1, n }, .float32), .bfloat16);
    const quantized: Linear = .{
        .weight = try g.hostArray(std.mem.sliceAsBytes(words), &.{ n, n / 4 }, .uint32),
        .scales = try g.hostArray(std.mem.sliceAsBytes(scales), &.{ n, n / 32 }, .uint8),
    };
    const dense: Linear = .{ .weight = try g.astype(try g.hostArray(std.mem.sliceAsBytes(dense_values), &.{ n, n }, .float32), .bfloat16) };
    try quantized.validate(&g, .mxfp8, n, n);
    const y = try linear(&g, x, quantized);
    const reference = try linear(&g, x, dense);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, g.dtypeOf(y));
    var actual: [n]f32 = undefined;
    var expected: [n]f32 = undefined;
    _ = try g.hostF32(try g.astype(y, .float32), &actual);
    _ = try g.hostF32(try g.astype(reference, .float32), &expected);
    for (actual, expected, 0..) |got, want, out| {
        try std.testing.expect(std.math.isFinite(got));
        try std.testing.expect(@abs(got - want) <= (1 + @abs(want)) / 128);
        // Independent scalar F32 dot checks the code/exponent interpretation;
        // this fixture's binary products sum exactly before BF16 output rounding.
        var scalar: f32 = 0;
        for (input_values, 0..) |value, k| scalar += value * dense_values[out * n + k];
        try std.testing.expect(@abs(got - scalar) <= (1 + @abs(scalar)) / 128);
    }
}
