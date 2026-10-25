//! Device oracle for the production layer driver, with small synthetic resident/expert banks.
const std = @import("std");
const sdk = @import("sdk");
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const experts = @import("experts.zig");
const layer = @import("chunk_layer.zig");
const C = @import("config.zig").Config;
const h = 512;
const tokens = 8;
const layers = 2;
const start = 127;

const Fixture = struct {
    fn matrix(g: *ops.G, outputs: usize, inputs: usize, quantized: bool) !ops.Linear {
        if (quantized) {
            const words = try g.gpa.alloc(u32, outputs * inputs / 4);
            defer g.gpa.free(words);
            @memset(words, 0);
            for (0..outputs) |out| words[out * (inputs / 4) + (out % inputs) / 4] = @as(u32, 0x20) << @as(u5, @intCast((out % 4) * 8));
            const scales = try g.gpa.alloc(u8, outputs * inputs / 32);
            defer g.gpa.free(scales);
            @memset(scales, 127);
            return .{ .weight = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(outputs), @intCast(inputs / 4) }, .uint32), .scales = try g.hostArray(scales, &.{ @intCast(outputs), @intCast(inputs / 32) }, .uint8) };
        }
        const values = try g.gpa.alloc(f32, outputs * inputs);
        defer g.gpa.free(values);
        @memset(values, 0);
        for (0..outputs) |out| values[out * inputs + out % inputs] = 0.125;
        return .{ .weight = try g.astype(try g.hostArray(std.mem.sliceAsBytes(values), &.{ @intCast(outputs), @intCast(inputs) }, .float32), .bfloat16) };
    }
    fn projection(g: *ops.G, offset: usize) !@import("bank.zig").Arrays(ops.T) {
        const words = try g.gpa.alloc(u32, 4 * h * h / 8);
        defer g.gpa.free(words);
        @memset(words, 0);
        for (0..4) |expert| for (0..h) |out| {
            words[(expert * h + out) * (h / 8) + out / 8] = @as(u32, 2 + @as(u32, @intCast(expert % 3))) << @as(u5, @intCast((out % 8) * 4));
        };
        const scales = try g.gpa.alloc(u8, 4 * h * h / 16);
        defer g.gpa.free(scales);
        @memset(scales, 0x38);
        var globals: [32 * 32]f32 = @splat(-1000);
        const values = [_]f32{ 2688, 1344, 672, 5376 };
        for (0..4) |row| globals[row * 32] = values[(row + offset) % 4];
        return .{ .w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ 4, h, h / 8 }, .uint32), .scales = try g.hostArray(scales, &.{ 4, h, h / 16 }, .uint8), .global_scale = try g.hostArray(std.mem.sliceAsBytes(&globals), &.{ 32, 32 }, .float32) };
    }
    fn modelLayer(g: *ops.G, quantized: bool, tied: bool) !layer.Layer {
        var router_values: [8 * h]f32 = @splat(0);
        if (!tied) for (0..8) |expert| {
            router_values[expert * h + expert] = 0.125;
        };
        const zero_norm = try g.zeros(&.{h}, .bfloat16);
        const head_norm = try g.zeros(&.{128}, .bfloat16);
        return .{
            .input_norm = zero_norm,
            .post_norm = zero_norm,
            .attn = .{ .q = try matrix(g, h, h, quantized), .k = try matrix(g, 128, h, quantized), .v = try matrix(g, 128, h, quantized), .o = try matrix(g, h, h, quantized), .q_norm = head_norm, .k_norm = head_norm, .index_q = try matrix(g, 128, h, quantized), .index_k = try matrix(g, 128, h, quantized), .index_q_norm = head_norm, .index_k_norm = head_norm },
            .mlp = .{ .gate = try matrix(g, h, h, quantized), .up = try matrix(g, h, h, quantized), .down = try matrix(g, h, h, quantized) },
            .router = try g.hostArray(std.mem.sliceAsBytes(&router_values), &.{ 8, h }, .float32),
            .correction = try g.zeros(&.{8}, .float32),
        };
    }
};

const Source = struct {
    banks: experts.Banks,
    values: []f32,
    ids: [layers * tokens * 4]u16 = @splat(std.math.maxInt(u16)),
    weights: [layers * tokens * 4]f32 = @splat(0),
    fn refs(ids: []const u16, out: *[32]experts.S.SlotRef) []const experts.S.SlotRef {
        for (ids, out[0..ids.len]) |id, *ref| ref.* = .{ .bank = if (id < 4) .base else .transient, .row = id % 4 };
        return out[0..ids.len];
    }
    pub fn apply(self: *Source, g: *ops.G, c: *const C, _: u32, x: ops.T, ids: []const u16, weights: []const f32, bias: ?ops.T) !ops.T {
        var slots: [32]experts.S.SlotRef = undefined;
        const y = try experts.applyBankOne(g, c, x, ids, weights, refs(ids, &slots), self.banks, bias);
        try g.evalAll(&.{y});
        return y;
    }
    pub fn applyChunk(self: *Source, g: *ops.G, c: *const C, _: u32, x: ops.T, ids: []const u16, weights: []const f32, bias: ?ops.T) !ops.T {
        var slots: [32]experts.S.SlotRef = undefined;
        const y = try experts.applyBankRows(g, c, x, ids, weights, refs(ids, &slots), self.banks, bias);
        try g.evalAll(&.{y});
        return y;
    }
    pub fn observeRoutes(self: *Source, index: usize, position: u32, routes: layer.Routes) !void {
        const at = index * tokens * 4 + (position - start) * 4;
        @memcpy(self.ids[at..][0..routes.n], routes.ids[0..routes.n]);
        @memcpy(self.weights[at..][0..routes.n], routes.weights[0..routes.n]);
    }
    pub fn observeLayer(self: *Source, g: *ops.G, index: usize, position: u32, value: ops.T) !void {
        const at = (index * tokens + position - start) * h;
        const n: usize = @intCast(g.shapeOf(value).d[0]);
        _ = try g.hostF32(try g.astype(value, .float32), self.values[at..][0 .. n * h]);
    }
};

test "MiniMax GPU chunk layers match serial routing hidden states final logits and cache" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const locked = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(locked), "1")) return error.GpuLockRequired;
    const stream = sdk.mlx.mlx_default_gpu_stream_new();
    defer _ = sdk.mlx.mlx_stream_free(stream);
    var g = try ops.G.init(std.testing.allocator, stream);
    defer g.deinit();
    for ([_]bool{ false, true }) |quantized| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const base = try Fixture.projection(&g, 0);
        const transient = try Fixture.projection(&g, 2);
        const banks: experts.Banks = .{ .base = .{ .gate = base, .up = base, .down = base }, .transient = .{ .gate = transient, .up = transient, .down = transient } };
        const model = [_]layer.Layer{ try Fixture.modelLayer(&g, quantized, false), try Fixture.modelLayer(&g, quantized, true) };
        var serial_cache: [layers]attention.Cache = @splat(.{ .limit = 256 });
        defer for (&serial_cache) |*cache| cache.clear(&g);
        var chunk_cache: [layers]attention.Cache = @splat(.{ .limit = 256 });
        defer for (&chunk_cache) |*cache| cache.clear(&g);
        const prefix_values = try std.testing.allocator.alloc(f32, start * 128);
        defer std.testing.allocator.free(prefix_values);
        for (prefix_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 64;
        const prefix = try g.astype(try g.hostArray(std.mem.sliceAsBytes(prefix_values), &.{ 1, 1, start, 128 }, .float32), .bfloat16);
        const seed_mark = g.mark();
        for (&serial_cache, &chunk_cache) |*one, *many| {
            _ = try one.append(&g, prefix, prefix, prefix);
            _ = try many.append(&g, prefix, prefix, prefix);
        }
        g.resetTo(seed_mark);
        var input_values: [tokens * h]f32 = undefined;
        for (&input_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
        const input = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input_values), &.{ tokens, h }, .float32), .bfloat16);
        const serial_values = try std.testing.allocator.alloc(f32, layers * tokens * h);
        defer std.testing.allocator.free(serial_values);
        const chunk_values = try std.testing.allocator.alloc(f32, layers * tokens * h);
        defer std.testing.allocator.free(chunk_values);
        @memset(serial_values, std.math.nan(f32));
        @memset(chunk_values, std.math.nan(f32));
        var one: Source = .{ .banks = banks, .values = serial_values };
        var many: Source = .{ .banks = banks, .values = chunk_values };
        var serial_last: ?ops.T = null;
        defer if (serial_last) |value| g.release(value);
        for (0..tokens) |i| {
            const wave = g.mark();
            if (serial_last) |value| g.release(value);
            serial_last = null;
            serial_last = try layer.forwardLayers(&g, std.testing.allocator, &.{ .hidden_size = h, .intermediate_size = h, .dense_intermediate_size = h, .shared_intermediate_size = h, .n_layers = layers, .n_heads = 4, .n_kv_heads = 1, .head_dim = 128, .n_routed_experts = 8, .index_n_heads = 1, .index_head_dim = 128, .index_block_size = 128, .index_topk_blocks = 1 }, &model, &serial_cache, try ops.row(&g, input, @intCast(i)), @intCast(start + i), &one, false);
            g.resetTo(wave);
        }
        const output = try layer.forwardLayers(&g, std.testing.allocator, &.{ .hidden_size = h, .intermediate_size = h, .dense_intermediate_size = h, .shared_intermediate_size = h, .n_layers = layers, .n_heads = 4, .n_kv_heads = 1, .head_dim = 128, .n_routed_experts = 8, .index_n_heads = 1, .index_head_dim = 128, .index_block_size = 128, .index_topk_blocks = 1 }, &model, &chunk_cache, input, start, &many, true);
        defer g.release(output);
        for (one.ids) |id| try std.testing.expect(id < 8);
        for (serial_values) |value| try std.testing.expect(std.math.isFinite(value));
        try std.testing.expectEqualSlices(u16, &one.ids, &many.ids);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&one.weights), std.mem.sliceAsBytes(&many.weights));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(serial_values), std.mem.sliceAsBytes(chunk_values));
        const head = try Fixture.matrix(&g, 16, h, false);
        const norm = try g.zeros(&.{h}, .bfloat16);
        const c: C = .{};
        const expected_logits = try ops.linear(&g, try ops.norm(&g, serial_last.?, norm, &c), head);
        const actual_logits = try ops.linear(&g, try ops.norm(&g, try ops.row(&g, output, tokens - 1), norm, &c), head);
        var expected: [16]f32 = undefined;
        var actual: [16]f32 = undefined;
        _ = try g.hostF32(try g.astype(expected_logits, .float32), &expected);
        _ = try g.hostF32(try g.astype(actual_logits, .float32), &actual);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected), std.mem.sliceAsBytes(&actual));
        for (&serial_cache, &chunk_cache) |*sc, *cc| {
            try std.testing.expectEqual(@as(u32, start + tokens), sc.len);
            try std.testing.expectEqual(sc.len, cc.len);
            const sv = try sc.view(&g, sc.len);
            const cv = try cc.view(&g, cc.len);
            for ([_]ops.T{ sv.k, sv.v, sv.index_k }, [_]ops.T{ cv.k, cv.v, cv.index_k }) |svalue, cvalue| {
                const n: usize = @intCast(sdk.mlx.mlx_array_size(svalue));
                const sdata = try std.testing.allocator.alloc(f32, n);
                defer std.testing.allocator.free(sdata);
                const cdata = try std.testing.allocator.alloc(f32, n);
                defer std.testing.allocator.free(cdata);
                _ = try g.hostF32(try g.astype(svalue, .float32), sdata);
                _ = try g.hostF32(try g.astype(cvalue, .float32), cdata);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(sdata), std.mem.sliceAsBytes(cdata));
            }
        }
    }
}
