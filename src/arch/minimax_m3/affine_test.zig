const std = @import("std");
const ops = @import("ops.zig");
const experts = @import("experts.zig");
const sdk = @import("sdk");
const C = @import("config.zig").Config;

test "MiniMax OQ3 prepared config selects affine banks and mixed residents" {
    const architecture = @import("root.zig");
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const text = try std.mem.replaceOwned(u8, arena.allocator(), @embedFile("config_fixture.json"), "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"residents_format\":\"mixed\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-affine-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},");
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", text);
    var diag: sdk.Diag = .{};
    const config = try architecture.parse(a, &peek, &diag);
    defer architecture.freeConfig(a, config);
    try std.testing.expectEqual(@import("bank.zig").Kind.affine, config.expert_format);
    try std.testing.expectEqual(@import("resident_layout.zig").Format.mixed, config.residents_format);
    try std.testing.expectEqual(sdk.Priority.native, architecture.claims(&peek).?);
}

const Fixture = struct {
    allocator: std.mem.Allocator,
    dense: []f32,
    weight: ops.T,
    scales: ops.T,
    biases: ops.T,
    outputs: usize,
    inputs: usize,
    bits: u32,
    fn make(g: *ops.G, count: usize, outputs: usize, inputs: usize, bits: u32, seed: usize) !Fixture {
        const packed_cols = inputs * bits / 32;
        const words = try g.gpa.alloc(u32, count * outputs * packed_cols);
        defer g.gpa.free(words);
        @memset(words, 0);
        const scales = try g.gpa.alloc(f32, count * outputs * inputs / 64);
        defer g.gpa.free(scales);
        const biases = try g.gpa.alloc(f32, scales.len);
        defer g.gpa.free(biases);
        const dense = try g.gpa.alloc(f32, count * outputs * inputs);
        errdefer g.gpa.free(dense);
        const levels = @as(u32, 1) << @as(u5, @intCast(bits));
        for (0..count * outputs) |row| for (0..inputs / 64) |block| {
            const scale = @as(f32, @floatFromInt(@as(u32, 1) << @as(u5, @intCast((row + block + seed) % 4)))) / 64;
            const bias = -@as(f32, @floatFromInt(levels / 2)) * scale;
            scales[row * (inputs / 64) + block] = scale;
            biases[row * (inputs / 64) + block] = bias;
            for (0..64) |within| {
                const col = block * 64 + within;
                const code: u32 = @intCast((row * 5 + col * 3 + seed) % levels);
                const bit = col * bits;
                const shift: u5 = @intCast(bit % 32);
                words[row * packed_cols + bit / 32] |= code << shift;
                if (@as(u32, shift) + bits > 32) words[row * packed_cols + bit / 32 + 1] |= code >> @as(u5, @intCast(32 - @as(u32, shift)));
                dense[row * inputs + col] = @as(f32, @floatFromInt(code)) * scale + bias;
            }
        };
        return .{ .allocator = g.gpa, .dense = dense, .outputs = outputs, .inputs = inputs, .bits = bits, .weight = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(count), @intCast(outputs), @intCast(packed_cols) }, .uint32), .scales = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scales), &.{ @intCast(count), @intCast(outputs), @intCast(inputs / 64) }, .float32), .bfloat16), .biases = try g.astype(try g.hostArray(std.mem.sliceAsBytes(biases), &.{ @intCast(count), @intCast(outputs), @intCast(inputs / 64) }, .float32), .bfloat16) };
    }
    fn deinit(self: Fixture) void {
        self.allocator.free(self.dense);
    }
    fn linear(self: Fixture, g: *ops.G) !ops.Linear {
        return .{ .weight = try g.reshape(self.weight, &.{ @intCast(self.outputs), @intCast(self.inputs * self.bits / 32) }), .scales = try g.reshape(self.scales, &.{ @intCast(self.outputs), @intCast(self.inputs / 64) }), .biases = try g.reshape(self.biases, &.{ @intCast(self.outputs), @intCast(self.inputs / 64) }) };
    }
    fn arrays(self: Fixture) @import("affine_bank.zig").Arrays {
        return .{ .w = self.weight, .scales = self.scales, .biases = self.biases };
    }
    fn denseLinear(self: Fixture, g: *ops.G, row: usize, x: ops.T) !ops.T {
        const values = self.dense[row * self.outputs * self.inputs ..][0 .. self.outputs * self.inputs];
        const w = try g.astype(try g.hostArray(std.mem.sliceAsBytes(values), &.{ @intCast(self.outputs), @intCast(self.inputs) }, .float32), .bfloat16);
        return ops.linear(g, x, w);
    }
    fn quantizedLinear(self: Fixture, g: *ops.G, row: usize, x: ops.T) !ops.T {
        const index = [_]u32{@intCast(row)};
        const id = try g.hostArray(std.mem.sliceAsBytes(&index), &.{1}, .uint32);
        const projection: ops.Linear = .{
            .weight = try g.reshape(try g.take(self.weight, id, 0), &.{ @intCast(self.outputs), @intCast(self.inputs * self.bits / 32) }),
            .scales = try g.reshape(try g.take(self.scales, id, 0), &.{ @intCast(self.outputs), @intCast(self.inputs / 64) }),
            .biases = try g.reshape(try g.take(self.biases, id, 0), &.{ @intCast(self.outputs), @intCast(self.inputs / 64) }),
        };
        return ops.linear(g, x, projection);
    }
};

fn gpuEnabled() !void {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const locked = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(locked), "1")) return error.GpuLockRequired;
}

test "MiniMax GPU affine resident matmuls and selected-row embeddings preserve original packing" {
    try gpuEnabled();
    const stream = sdk.mlx.mlx_default_gpu_stream_new();
    defer _ = sdk.mlx.mlx_stream_free(stream);
    var g = try ops.G.init(std.testing.allocator, stream);
    defer g.deinit();
    const inputs = 512;
    var input: [inputs]f32 = undefined;
    for (&input, 0..) |*value, col| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(col % 9)) - 4)) / 64;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input), &.{ 1, inputs }, .float32), .bfloat16);
    for ([_]u32{ 3, 4, 5, 6, 8 }) |bits| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const fixture = try Fixture.make(&g, 1, 32, inputs, bits, bits);
        defer fixture.deinit();
        const projection = try fixture.linear(&g);
        try projection.validate(&g, .mixed, 32, inputs);
        const result = try ops.linear(&g, x, projection);
        var expected: [32]f32 = @splat(0);
        for (&expected, 0..) |*value, row| for (input, 0..) |scalar, col| {
            value.* += scalar * fixture.dense[row * inputs + col];
        };
        const rounded = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&expected), &.{ 1, 32 }, .float32), .bfloat16);
        var want: [32]f32 = undefined;
        var got: [32]f32 = undefined;
        _ = try g.hostF32(try g.astype(rounded, .float32), &want);
        _ = try g.hostF32(try g.astype(result, .float32), &got);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want), std.mem.sliceAsBytes(&got));
    }
    const embedding = try Fixture.make(&g, 1, 16, inputs, 8, 3);
    defer embedding.deinit();
    const ids = [_]u32{ 7, 2, 9, 1 };
    const indices = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{4}, .uint32);
    const result = try ops.embeddingRows(&g, try embedding.linear(&g), indices, inputs);
    var actual: [4 * inputs]f32 = undefined;
    _ = try g.hostF32(try g.astype(result, .float32), &actual);
    for (ids, 0..) |id, row| try std.testing.expectEqualSlices(f32, embedding.dense[id * inputs ..][0..inputs], actual[row * inputs ..][0..inputs]);
}

fn roundBf16(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    return @bitCast((bits +% (0x7fff + ((bits >> 16) & 1))) & 0xffff0000);
}

test "MiniMax affine oracle distinguishes BF16 bias sums from real sums" {
    try std.testing.expectEqual(@as(f32, 1), roundBf16(1 + @as(f32, 1) / 256));
    try std.testing.expectEqual(@as(f32, 1.015625), roundBf16(1 + @as(f32, 3) / 256));
    var sum: f32 = 1;
    for (0..7) |_| sum = roundBf16(sum + @as(f32, 1) / 256);
    try std.testing.expectEqual(@as(f32, 1), sum);
    try std.testing.expectEqual(@as(f64, 7) / 256, @as(f64, 1) + @as(f64, 7) / 256 - sum);
}

// A decoded F64 dot is independent of MLX's packed kernel. Its affine bias
// correction uses BF16 eight-term sums in quantized.h::load_vector<T,float,...,3>.
// Account for that signed correction before bounding F32 arithmetic (gamma_n)
// and the final BF16 rounding. An output-relative dense tolerance cannot bound
// cancellation in the factored bias term.
fn expectIndependentProjection(g: *ops.G, fixture: Fixture, expert: usize, x: ops.T, y: ops.T) !void {
    const a = g.gpa;
    const input = try a.alloc(f32, fixture.inputs);
    defer a.free(input);
    const actual = try a.alloc(f32, fixture.outputs);
    defer a.free(actual);
    _ = try g.hostF32(try g.astype(x, .float32), input);
    _ = try g.hostF32(try g.astype(y, .float32), actual);
    const unit_f32: f64 = 1.0 / 16777216.0;
    const operations: f64 = @floatFromInt(2 * fixture.inputs + 64);
    const gamma = operations * unit_f32 / (1 - operations * unit_f32);
    const unit_bf16: f64 = 1.0 / 256.0;
    for (actual, 0..) |value, row| {
        const weights = fixture.dense[(expert * fixture.outputs + row) * fixture.inputs ..][0..fixture.inputs];
        var real_dot: f64 = 0;
        var correction: f64 = 0;
        var factor_mass: f64 = 0;
        for (0..fixture.inputs / 64) |group| {
            // Positive scales and the fixture's full code range make the
            // minimum decoded weight exactly the independently decoded bias.
            var bias: f64 = std.math.inf(f64);
            for (weights[group * 64 ..][0..64]) |w| bias = @min(bias, w);
            for (0..8) |chunk| {
                const begin = group * 64 + chunk * 8;
                var exact_sum: f64 = 0;
                var rounded_sum = input[begin];
                for (0..8) |j| {
                    const i = begin + j;
                    real_dot += @as(f64, input[i]) * weights[i];
                    factor_mass += @abs(@as(f64, input[i]) * (@as(f64, weights[i]) - bias));
                    exact_sum += input[i];
                    if (j > 0) rounded_sum = roundBf16(rounded_sum + input[i]);
                }
                correction += bias * (@as(f64, rounded_sum) - exact_sum);
                factor_mass += @abs(bias * rounded_sum);
            }
        }
        const corrected = real_dot + correction;
        const f32_bound = gamma * factor_mass;
        const bound = f32_bound + unit_bf16 / (1 - unit_bf16) * (@abs(corrected) + f32_bound);
        const difference = @abs(@as(f64, value) - corrected);
        if (!std.math.isFinite(value) or difference > bound) {
            std.debug.print("AFFINE_ORACLE expert={d} output={d} actual={d:.12} decoded_f64={d:.12} bf16_bias_correction={d:.12} corrected={d:.12} error={d:.12} bound={d:.12} factor_mass={d:.12}\n", .{ expert, row, value, real_dot, correction, corrected, difference, bound, factor_mass });
        }
        try std.testing.expect(std.math.isFinite(value) and difference <= bound);
    }
}

test "MiniMax GPU affine streamed tiles preserve packed rows and projection error bounds" {
    try gpuEnabled();
    const stream = sdk.mlx.mlx_default_gpu_stream_new();
    defer _ = sdk.mlx.mlx_stream_free(stream);
    var g = try ops.G.init(std.testing.allocator, stream);
    defer g.deinit();
    const hidden = 512;
    var fixtures: [2][3]Fixture = undefined;
    var initialized: usize = 0;
    defer for (0..initialized) |index| fixtures[index / 3][index % 3].deinit();
    for (0..2) |bank| for (0..3) |projection| {
        fixtures[bank][projection] = try Fixture.make(&g, 4, hidden, hidden, 3, bank * 7 + projection * 11);
        initialized += 1;
    };
    const banks: experts.Affine.SlotBanks = .{
        .base = .{ .gate = fixtures[0][0].arrays(), .up = fixtures[0][1].arrays(), .down = fixtures[0][2].arrays() },
        .transient = .{ .gate = fixtures[1][0].arrays(), .up = fixtures[1][1].arrays(), .down = fixtures[1][2].arrays() },
    };
    const c: C = .{ .hidden_size = hidden, .intermediate_size = hidden };
    for ([_]usize{ 8, 3 }) |tokens| for ([_][4]u16{ .{ 0, 1, 2, 3 }, .{ 4, 1, 7, 2 } }) |base_ids| {
        const mark = g.mark();
        defer g.resetTo(mark);
        var input: [8 * hidden]f32 = undefined;
        for (input[0 .. tokens * hidden], 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 64;
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(input[0 .. tokens * hidden]), &.{ @intCast(tokens), hidden }, .float32), .bfloat16);
        var ids: [32]u16 = undefined;
        var weights: [32]f32 = undefined;
        var refs: [32]experts.S.SlotRef = undefined;
        var expected_rows: [8]ops.T = undefined;
        for (0..tokens) |token| {
            var sum: ?ops.T = null;
            for ([_]f32{ 0.25, 1, 0.5, 0.25 }, 0..) |weight, route| {
                const at = token * 4 + route;
                const id = base_ids[(route + token) % 4];
                ids[at] = id;
                weights[at] = weight;
                refs[at] = .{ .bank = if (id < 4) .base else .transient, .row = id % 4 };
                const bank: usize = if (id < 4) 0 else 1;
                const row = try ops.row(&g, x, @intCast(token));
                const gate = try fixtures[bank][0].quantizedLinear(&g, id % 4, row);
                const up = try fixtures[bank][1].quantizedLinear(&g, id % 4, row);
                const activation = try ops.activate(&g, gate, up, &c);
                const down = try fixtures[bank][2].quantizedLinear(&g, id % 4, activation);
                try expectIndependentProjection(&g, fixtures[bank][0], id % 4, row, gate);
                try expectIndependentProjection(&g, fixtures[bank][1], id % 4, row, up);
                try expectIndependentProjection(&g, fixtures[bank][2], id % 4, activation, down);
                const y = try g.mul(down, try g.scalar(weight, .bfloat16));
                sum = if (sum) |previous| try g.add(previous, y) else y;
            }
            expected_rows[token] = sum.?;
        }
        const expected = try g.concat(expected_rows[0..tokens], 0);
        const actual = try experts.Affine.applyRows(&g, &c, x, ids[0 .. tokens * 4], weights[0 .. tokens * 4], refs[0 .. tokens * 4], banks, null);
        var want: [8 * hidden]f32 = undefined;
        var got: [8 * hidden]f32 = undefined;
        _ = try g.hostF32(try g.astype(expected, .float32), want[0 .. tokens * hidden]);
        _ = try g.hostF32(try g.astype(actual, .float32), got[0 .. tokens * hidden]);
        for (got[0 .. tokens * hidden]) |value| try std.testing.expect(std.math.isFinite(value));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want[0 .. tokens * hidden]), std.mem.sliceAsBytes(got[0 .. tokens * hidden]));
    };
}
