//! Bounded MiniMax demand reads through the SDK stream. Every route is evaluated
//! before its slots are unpinned; NVFP4 gathers retain the shared global-scale math.
const std = @import("std");
const sdk = @import("sdk");
const bank = @import("bank.zig");
const ops = @import("ops.zig");
const C = @import("config.zig").Config;
const nvfp4 = @import("../glm5_next/nvfp4.zig").Format;
const Grouping = @import("gather_groups.zig").Grouping;
const GroupingCapacity = @import("gather_groups.zig").GroupingCapacity;
pub const chunk_tokens = 8;
pub const chunk_routes = 4 * chunk_tokens;
pub fn Backend(comptime B: type, comptime F: type) type {
    return struct {
        pub const StreamType = @import("../../sdk_ext.zig").expert.stream.StreamOf(B.SDK, false);
        pub const SourceType = struct {
            a: std.mem.Allocator,
            owner: B.Bank,
            view: B.SDKBank,
            stream: *StreamType.Stream,
            rows: []u32,
            pub fn init(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: sdk.mlx.mlx_stream, rows: u32) !*SourceType {
                if (rows < 4 or rows > 128) return error.MiniMaxExpertRows;
                const self = try a.create(SourceType);
                errdefer a.destroy(self);
                self.a = a;
                self.owner = try B.Bank.open(a, io, dir);
                errdefer self.owner.deinit();
                try B.requireFlash(&self.owner.geometry);
                self.view = try B.SDKBank.init(a, &self.owner);
                errdefer self.view.deinit();
                const row_counts = try a.alloc(u32, self.owner.geometry.num_moe_layers);
                errdefer a.free(row_counts);
                self.rows = row_counts;
                @memset(row_counts, rows);
                self.stream = try StreamType.Stream.init(a, &self.view, .{
                    .rows = row_counts,
                    .max_route_ids = chunk_routes,
                    .decode_route_ids = 4,
                    .transient_rows = chunk_routes,
                    .records_per_part = 4,
                    .pool = .{ .workers = 2, .tickets = 64 },
                    .staging_from_bank = true,
                    .slot_memory = .{ .mlx = s },
                });
                return self;
            }
            pub fn deinit(self: *SourceType) void {
                const a = self.a;
                self.stream.deinit();
                self.view.deinit();
                self.owner.deinit();
                a.free(self.rows);
                a.destroy(self);
            }
            pub fn beginPrompt(self: *SourceType) !void {
                if (self.stream.phase == .decode) {
                    _ = try self.stream.shrink(self.rows, .keep);
                } else try self.stream.forgetPrompt();
                self.stream.resetPromptCounts();
            }
            pub fn beginDecode(self: *SourceType) !void {
                if (self.stream.phase == .prefill) try self.stream.grow(self.rows);
            }
            /// x [1,H], selected ids and normalized weights in the same top-four order.
            pub fn apply(self: *SourceType, g: *ops.G, c: *const C, layer: u32, x: ops.T, ids: []const u16, weights: []const f32, down_bias: ?ops.T) !ops.T {
                if (ids.len != 4 or weights.len != 4) return error.MiniMaxRouteWidth;
                const rel = layer - self.owner.geometry.first_moe_layer;
                if (self.stream.phase == .prefill) try self.stream.seedPrefill(rel, ids);
                const route = try self.stream.route(rel, ids, &.{});
                var released = false;
                errdefer {
                    // Drain any submitted consumer before a later route can reuse its slots.
                    _ = sdk.mlx.mlx_synchronize(g.s);
                    if (!released) self.stream.release(route);
                    self.stream.flush() catch {};
                }
                for (0..route.n_parts) |part| try self.stream.waitDown(route, @intCast(part));
                var refs: [StreamType.max_route_ids]StreamType.SlotRef = undefined;
                _ = self.stream.refsOf(route, &refs);
                const arrays: SlotBanks = .{ .base = self.stream.bankArrays(rel, .base), .ext = self.stream.bankArrays(rel, .ext), .transient = self.stream.bankArrays(rel, .transient) };
                const result = try applyOne(g, c, x, ids, weights, refs[0..ids.len], arrays, down_bias);
                try g.evalAll(&.{result});
                self.stream.release(route);
                released = true;
                try self.stream.flush();
                return result;
            }
            pub fn applyChunk(self: *SourceType, g: *ops.G, c: *const C, layer: u32, x: ops.T, ids: []const u16, weights: []const f32, down_bias: ?ops.T) !ops.T {
                if (ids.len == 0 or ids.len > chunk_routes or ids.len % 4 != 0 or weights.len != ids.len) return error.MiniMaxRouteWidth;
                if (self.stream.phase != .prefill) return error.MiniMaxChunkPhase;
                const rel = layer - self.owner.geometry.first_moe_layer;
                try self.stream.seedPrefill(rel, ids);
                const route = try self.stream.route(rel, ids, &.{});
                var released = false;
                errdefer {
                    _ = sdk.mlx.mlx_synchronize(g.s);
                    if (!released) self.stream.release(route);
                    self.stream.flush() catch {};
                }
                for (0..route.n_parts) |part| try self.stream.waitDown(route, @intCast(part));
                var refs: [StreamType.max_route_ids]StreamType.SlotRef = undefined;
                _ = self.stream.refsOf(route, &refs);
                const arrays: SlotBanks = .{ .base = self.stream.bankArrays(rel, .base), .ext = self.stream.bankArrays(rel, .ext), .transient = self.stream.bankArrays(rel, .transient) };
                const result = try applyRows(g, c, x, ids, weights, refs[0..ids.len], arrays, down_bias);
                try g.evalAll(&.{result});
                self.stream.release(route);
                released = true;
                try self.stream.flush();
                return result;
            }
        };

        pub const SlotBanks = struct {
            base: ?B.BankArrays = null,
            ext: ?B.BankArrays = null,
            transient: ?B.BankArrays = null,
            fn get(self: SlotBanks, kind: StreamType.BankKind) !B.BankArrays {
                return (switch (kind) {
                    .base => self.base,
                    .ext => self.ext,
                    .transient => self.transient,
                }) orelse error.MiniMaxSlotBankMissing;
            }
        };

        pub fn applyOne(g: *ops.G, c: *const C, x: ops.T, ids: []const u16, weights: []const f32, refs: []const StreamType.SlotRef, banks: SlotBanks, down_bias: ?ops.T) !ops.T {
            const outputs = try grouped(g, c, x, refs, banks);
            return combine(g, ids, weights, &outputs, down_bias);
        }

        pub fn applyRows(g: *ops.G, c: *const C, x: ops.T, ids: []const u16, weights: []const f32, refs: []const StreamType.SlotRef, banks: SlotBanks, down_bias: ?ops.T) !ops.T {
            const groups = try GroupingCapacity(StreamType.SlotRef, chunk_routes).init(refs);
            const rows = ids.len / 4;
            if (ids.len != refs.len or weights.len != ids.len or rows == 0 or rows > chunk_tokens) return error.MiniMaxRouteWidth;
            if (!std.mem.eql(c_int, g.shapeOf(x).slice(), &.{ @intCast(rows), @intCast(c.hidden_size) })) return error.MiniMaxChunkShape;
            var outputs: [chunk_routes]ops.T = undefined;
            for (groups.groups[0..groups.n]) |group| {
                const arrays = try banks.get(group.bank);
                var tokens: [chunk_routes]u32 = undefined;
                for (group.positions[0..group.n], tokens[0..group.n]) |position, *token| token.* = @intCast(position / 4);
                const token_ids = try g.hostArray(std.mem.sliceAsBytes(tokens[0..group.n]), &.{@intCast(group.n)}, .uint32);
                const input = try g.reshape(try g.take(x, token_ids, 0), &.{ @intCast(group.n), 1, @intCast(c.hidden_size) });
                const slot_ids = try g.hostArray(std.mem.sliceAsBytes(group.rows[0..group.n]), &.{@intCast(group.n)}, .uint32);
                const gate = try F.gather(g, input, arrays.gate, slot_ids);
                const up = try F.gather(g, input, arrays.up, slot_ids);
                const down = try F.gather(g, try ops.activate(g, gate, up, c), arrays.down, slot_ids);
                for (group.positions[0..group.n], 0..) |position, local| {
                    const row: c_int = @intCast(local);
                    outputs[position] = try g.reshape(try g.slice(down, &.{ row, 0, 0 }, &.{ row + 1, 1, @intCast(c.hidden_size) }, &.{ 1, 1, 1 }), &.{ 1, @intCast(c.hidden_size) });
                }
            }
            var mixed: [chunk_tokens]ops.T = undefined;
            for (0..rows) |token| mixed[token] = try combine(g, ids[token * 4 ..][0..4], weights[token * 4 ..][0..4], outputs[token * 4 ..][0..4], down_bias);
            return g.concat(mixed[0..rows], 0);
        }

        fn grouped(g: *ops.G, c: *const C, x: ops.T, refs: []const StreamType.SlotRef, banks: SlotBanks) ![4]ops.T {
            const groups = try Grouping(StreamType.SlotRef).init(refs);
            const input = try g.reshape(x, &.{ 1, 1, @intCast(c.hidden_size) });
            var outputs: [4]ops.T = undefined;
            for (groups.groups[0..groups.n]) |group| {
                const arrays = try banks.get(group.bank);
                const idx = try g.hostArray(std.mem.sliceAsBytes(group.rows[0..group.n]), &.{@intCast(group.n)}, .uint32);
                const gate = try F.gather(g, input, arrays.gate, idx);
                const up = try F.gather(g, input, arrays.up, idx);
                // The expert axis stays a batch dimension: [group, 1, intermediate].
                // The down gather pairs batch i with indices[i], rather than cross-products.
                const down = try F.gather(g, try ops.activate(g, gate, up, c), arrays.down, idx);
                for (group.positions[0..group.n], 0..) |position, local| {
                    const row: c_int = @intCast(local);
                    outputs[position] = try g.reshape(try g.slice(down, &.{ row, 0, 0 }, &.{ row + 1, 1, @intCast(c.hidden_size) }, &.{ 1, 1, 1 }), &.{ 1, @intCast(c.hidden_size) });
                }
            }
            return outputs;
        }

        fn combine(g: *ops.G, ids: []const u16, weights: []const f32, outputs: *const [4]ops.T, down_bias: ?ops.T) !ops.T {
            var sum: ?ops.T = null;
            // Grouping changes only the gathers. Bias, BF16 weighting and left-associated
            // additions retain the original router order, including cancellation rounding.
            for (ids, weights, outputs) |id, weight, output| {
                var y = output;
                if (down_bias) |bias| {
                    const eid = [_]u32{id};
                    const ei = try g.hostArray(std.mem.sliceAsBytes(&eid), &.{1}, .uint32);
                    y = try g.add(y, try g.take(bias, ei, 0));
                }
                y = try g.mul(y, try g.scalar(weight, g.dtypeOf(y)));
                sum = if (sum) |prev| try g.add(prev, y) else y;
            }
            return sum.?;
        }
    };
}

const NV = Backend(bank, nvfp4);
pub const Affine = Backend(@import("affine_bank.zig"), @import("affine_format.zig"));
pub const S = NV.StreamType;
pub const Banks = NV.SlotBanks;
pub const applyBankOne = NV.applyOne;
pub const applyBankRows = NV.applyRows;
const groupedOutputs = NV.grouped;
const mixOutputs = NV.combine;

pub const Source = struct {
    a: std.mem.Allocator,
    rows: []u32,
    backend: union(bank.Kind) { nvfp4: *NV.SourceType, affine: *Affine.SourceType },
    pub fn init(a: std.mem.Allocator, io: std.Io, dir: []const u8, stream: sdk.mlx.mlx_stream, rows: u32, kind: bank.Kind) !*Source {
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.a = a;
        switch (kind) {
            .nvfp4 => {
                const source = try NV.SourceType.init(a, io, dir, stream, rows);
                self.backend = .{ .nvfp4 = source };
                self.rows = source.rows;
            },
            .affine => {
                const source = try Affine.SourceType.init(a, io, dir, stream, rows);
                self.backend = .{ .affine = source };
                self.rows = source.rows;
            },
        }
        return self;
    }
    pub fn deinit(self: *Source) void {
        const a = self.a;
        switch (self.backend) {
            inline else => |source| source.deinit(),
        }
        a.destroy(self);
    }
    pub fn beginPrompt(self: *Source) !void {
        switch (self.backend) {
            inline else => |source| try source.beginPrompt(),
        }
    }
    pub fn beginDecode(self: *Source) !void {
        switch (self.backend) {
            inline else => |source| try source.beginDecode(),
        }
    }
    pub fn apply(self: *Source, g: *ops.G, c: *const C, layer: u32, x: ops.T, ids: []const u16, weights: []const f32, bias: ?ops.T) !ops.T {
        return switch (self.backend) {
            inline else => |source| source.apply(g, c, layer, x, ids, weights, bias),
        };
    }
    pub fn applyChunk(self: *Source, g: *ops.G, c: *const C, layer: u32, x: ops.T, ids: []const u16, weights: []const f32, bias: ?ops.T) !ops.T {
        return switch (self.backend) {
            inline else => |source| source.applyChunk(g, c, layer, x, ids, weights, bias),
        };
    }
    pub fn stats(self: *Source) @import("../../sdk_ext.zig").expert.Stats {
        return switch (self.backend) {
            inline else => |source| source.stream.stats(),
        };
    }
    pub fn countRoutes(self: *Source) void {
        switch (self.backend) {
            inline else => |source| source.stream.countRoutes(),
        }
    }
    pub fn addUsage(self: *Source, counts: []u64) void {
        switch (self.backend) {
            inline else => |source| source.stream.addUsage(counts),
        }
    }
    pub fn seedResidents(self: *Source, context: anytype, comptime order: fn (@TypeOf(context), u32) []const u16) !u64 {
        return switch (self.backend) {
            inline else => |source| source.stream.seedResidents(context, order),
        };
    }
};

test {
    _ = @import("gather_groups.zig");
}

// Exercise the runtime's gather directly, including the padded per-expert global
// scale rows. Every scalar product is a binary fraction, so BF16 results are exact.
test "MiniMax GPU NVFP4 gather selects packed expert rows and their global scales" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const mlx = sdk.mlx;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();

    const inputs = 64;
    const outputs = 32;
    const codes = [_]u4{ 1, 2, 4, 6 }; // E2M1: 0.5, 1, 2, 4.
    const magnitudes = [_]f32{ 0.5, 1, 2, 4 };
    const scale_bytes = [_]u8{ 0x38, 0x40, 0x30, 0x48 }; // E4M3: 1, 2, 0.5, 4.
    const scale_values = [_]f32{ 1, 2, 0.5, 4 };
    var words: [2 * outputs * (inputs / 8)]u32 = undefined;
    var scales: [2 * outputs * (inputs / 16)]u8 = undefined;
    for (0..2) |expert| {
        for (0..outputs) |out| {
            const code: u32 = codes[out % codes.len] | (if (expert == 1) @as(u4, 8) else @as(u4, 0));
            @memset(words[(expert * outputs + out) * (inputs / 8) ..][0 .. inputs / 8], code * 0x11111111);
            @memcpy(scales[(expert * outputs + out) * (inputs / 16) ..][0 .. inputs / 16], &scale_bytes);
        }
    }
    // The stream isolates each scalar on its own 128-byte line, with a 4 KiB
    // minimum backing. Only column zero of expert rows zero and one is consumed.
    var global_scales: [32 * 32]f32 = @splat(0);
    global_scales[0] = 2688;
    global_scales[32] = 672;
    const projection: bank.Arrays(ops.T) = .{
        .w = try g.hostArray(std.mem.sliceAsBytes(&words), &.{ 2, outputs, inputs / 8 }, .uint32),
        .scales = try g.hostArray(std.mem.sliceAsBytes(&scales), &.{ 2, outputs, inputs / 16 }, .uint8),
        .global_scale = try g.hostArray(std.mem.sliceAsBytes(&global_scales), &.{ 32, 32 }, .float32),
    };
    // Swapped bank order and unequal inputs distinguish the index mapping from
    // both ignored global scales and a global scale taken from the wrong row.
    const cases = [_]struct { expert: u32, input: f32, multiplier: f32 }{
        .{ .expert = 1, .input = 1, .multiplier = -0.25 },
        .{ .expert = 0, .input = 3, .multiplier = 1 },
    };
    for (cases) |case| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const input_values: [inputs]f32 = @splat(case.input);
        const ids = [_]u32{case.expert};
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input_values), &.{ 1, 1, inputs }, .float32), .bfloat16);
        const idx = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{1}, .uint32);
        const y = try nvfp4.gather(&g, x, projection, idx);
        try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, g.dtypeOf(y));
        try std.testing.expectEqualSlices(c_int, &.{ 1, 1, outputs }, g.shapeOf(y).slice());
        var actual: [outputs]f32 = undefined;
        _ = try g.hostF32(try g.astype(y, .float32), &actual);
        for (actual, 0..) |got, out| {
            var expected: f32 = 0;
            for (0..inputs) |k| expected += case.input * (magnitudes[out % magnitudes.len] * (scale_values[k / 16] * case.multiplier));
            try std.testing.expectEqual(expected, got);
        }
    }
}

test "MiniMax GPU grouped NVFP4 gathers match scalar experts and original BF16 add order" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const mlx = sdk.mlx;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(std.testing.allocator, s);
    defer g.deinit();
    const Fixture = struct {
        fn projection(comptime h: usize, graph: *ops.G, bank_index: usize, projection_index: usize) !bank.Arrays(ops.T) {
            const words = try graph.gpa.alloc(u32, 4 * h * (h / 8));
            defer graph.gpa.free(words);
            const scales = try graph.gpa.alloc(u8, 4 * h * (h / 16));
            defer graph.gpa.free(scales);
            var globals: [32 * 32]f32 = @splat(-12345);
            const code_values = [_]u32{ 1, 2, 3, 4 }; // E2M1 .5, 1, 1.5, 2.
            const scale_values = [_]u8{ 0x30, 0x38, 0x40, 0x48 }; // E4M3 .5, 1, 2, 4.
            const global_values = [_]f32{ 2688, 1344, 672, 5376 };
            for (0..4) |row| {
                globals[row * 32] = global_values[(bank_index * 2 + row) % global_values.len] / @as(f32, @floatFromInt(@as(u32, 1) << @as(u5, @intCast(projection_index))));
                for (0..h) |out| {
                    const sign: u32 = if ((bank_index + row + projection_index) % 2 == 0) 0 else 8;
                    const code = if (projection_index == 2 and out == 0) 0 else code_values[(out + row + projection_index) % code_values.len] | sign;
                    @memset(words[(row * h + out) * (h / 8) ..][0 .. h / 8], code * 0x11111111);
                    for (0..h / 16) |block| scales[(row * h + out) * (h / 16) + block] = scale_values[(block + bank_index + row) % scale_values.len];
                }
            }
            return .{
                .w = try graph.hostArray(std.mem.sliceAsBytes(words), &.{ 4, h, h / 8 }, .uint32),
                .scales = try graph.hostArray(std.mem.sliceAsBytes(scales), &.{ 4, h, h / 16 }, .uint8),
                .global_scale = try graph.hostArray(std.mem.sliceAsBytes(&globals), &.{ 32, 32 }, .float32),
            };
        }
        // The pre-grouping implementation is deliberately independent of the
        // production grouping and mixing helpers; it is only this test's oracle.
        fn raw(graph: *ops.G, c: *const C, x: ops.T, ref: S.SlotRef, banks: Banks) !ops.T {
            const arrays = try banks.get(ref.bank);
            const rows = [_]u32{ref.row};
            const idx = try graph.hostArray(std.mem.sliceAsBytes(&rows), &.{1}, .uint32);
            const input = try graph.reshape(x, &.{ 1, 1, @intCast(c.hidden_size) });
            const gate = try nvfp4.gather(graph, input, arrays.gate, idx);
            const up = try nvfp4.gather(graph, input, arrays.up, idx);
            return graph.reshape(try nvfp4.gather(graph, try ops.activate(graph, gate, up, c), arrays.down, idx), &.{ 1, @intCast(c.hidden_size) });
        }
        fn scalar(graph: *ops.G, c: *const C, x: ops.T, ids: []const u16, weights: []const f32, refs: []const S.SlotRef, banks: Banks, bias: ?ops.T) !ops.T {
            var sum: ?ops.T = null;
            for (ids, weights, refs) |id, weight, ref| {
                var y = try raw(graph, c, x, ref, banks);
                if (bias) |b| {
                    const eid = [_]u32{id};
                    const ei = try graph.hostArray(std.mem.sliceAsBytes(&eid), &.{1}, .uint32);
                    y = try graph.add(y, try graph.take(b, ei, 0));
                }
                y = try graph.mul(y, try graph.scalar(weight, graph.dtypeOf(y)));
                sum = if (sum) |previous| try graph.add(previous, y) else y;
            }
            return sum.?;
        }
    };
    // 64 exercises qmv; 512 meets bits4's 8*2*32 K alignment and N%8,
    // selecting the same qmv_fast family as H6144/I3072 in the released model.
    inline for (.{ 64, 512 }) |h| {
        const dimension_mark = g.mark();
        defer g.resetTo(dimension_mark);
        const banks: Banks = .{
            .base = .{ .gate = try Fixture.projection(h, &g, 0, 0), .up = try Fixture.projection(h, &g, 0, 1), .down = try Fixture.projection(h, &g, 0, 2) },
            .transient = .{ .gate = try Fixture.projection(h, &g, 1, 0), .up = try Fixture.projection(h, &g, 1, 1), .down = try Fixture.projection(h, &g, 1, 2) },
        };
        const ref_cases = [_][4]S.SlotRef{
            .{ .{ .bank = .base, .row = 3 }, .{ .bank = .base, .row = 1 }, .{ .bank = .base, .row = 0 }, .{ .bank = .base, .row = 2 } },
            .{ .{ .bank = .transient, .row = 1 }, .{ .bank = .base, .row = 0 }, .{ .bank = .transient, .row = 0 }, .{ .bank = .base, .row = 1 } },
        };
        const ids = [_]u16{ 7, 2, 9, 1 };
        const weight_cases = [_][4]f32{ .{ 0.5, 0.5, 0.5, 0.5 }, .{ 0.5, 0.25, 1, 0.25 } };
        const input_values: [h]f32 = @splat(1 / @as(f32, @floatFromInt(h)));
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input_values), &.{ 1, h }, .float32), .bfloat16);
        var bias_values: [10 * h]f32 = undefined;
        for (&bias_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i / h)) * 0.125;
        // Down channel zero is zero: these biases produce weighted terms
        // [256, 1, -256, 1]. Left association yields 1; grouping bank sums yields 2.
        for (ids, [_]f32{ 512, 2, -512, 2 }) |id, value| bias_values[id * h] = value;
        const bias = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&bias_values), &.{ 10, h }, .float32), .bfloat16);
        const c: C = .{ .hidden_size = h, .intermediate_size = h };
        for (ref_cases) |refs| for ([_]bool{ false, true }) |with_bias| for (weight_cases, 0..) |weights, weight_case| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const down_bias: ?ops.T = if (with_bias) bias else null;
            const expected = try Fixture.scalar(&g, &c, x, &ids, &weights, &refs, banks, down_bias);
            const grouped = try groupedOutputs(&g, &c, x, &refs, banks);
            if (!with_bias and weight_case == 0) for (grouped, refs) |output, ref| {
                const scalar_output = try Fixture.raw(&g, &c, x, ref, banks);
                try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, g.dtypeOf(output));
                var scalar_values: [h]f32 = undefined;
                var grouped_values: [h]f32 = undefined;
                _ = try g.hostF32(try g.astype(scalar_output, .float32), &scalar_values);
                _ = try g.hostF32(try g.astype(output, .float32), &grouped_values);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&scalar_values), std.mem.sliceAsBytes(&grouped_values));
            };
            const actual = try mixOutputs(&g, &ids, &weights, &grouped, down_bias);
            try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, g.dtypeOf(actual));
            try std.testing.expectEqualSlices(c_int, &.{ 1, h }, g.shapeOf(actual).slice());
            var expected_values: [h]f32 = undefined;
            var actual_values: [h]f32 = undefined;
            _ = try g.hostF32(try g.astype(expected, .float32), &expected_values);
            _ = try g.hostF32(try g.astype(actual, .float32), &actual_values);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_values), std.mem.sliceAsBytes(&actual_values));
            if (with_bias and weight_case == 0) try std.testing.expectEqual(@as(f32, 1), actual_values[0]);
        };
        // Full 8x4 tiles reach B32 in one bank, and B16 in each mixed bank.
        // Repeated experts across tokens must still consume each token's input.
        for ([_]usize{ 8, 3 }) |tokens| for (ref_cases) |refs| for ([_]bool{ false, true }) |with_bias| {
            const mark = g.mark();
            defer g.resetTo(mark);
            var values: [8 * h]f32 = undefined;
            for (0..tokens) |token| @memset(values[token * h ..][0..h], @as(f32, @floatFromInt(token + 1)) / @as(f32, @floatFromInt(h)));
            const inputs = try g.astype(try g.hostArray(std.mem.sliceAsBytes(values[0 .. tokens * h]), &.{ @intCast(tokens), h }, .float32), .bfloat16);
            var all_ids: [32]u16 = undefined;
            var all_refs: [32]S.SlotRef = undefined;
            var all_weights: [32]f32 = undefined;
            var baseline: [8]ops.T = undefined;
            const down_bias: ?ops.T = if (with_bias) bias else null;
            for (0..tokens) |token| {
                for (0..4) |route| {
                    const source = (route + token) % 4;
                    all_ids[token * 4 + route] = ids[source];
                    all_refs[token * 4 + route] = refs[source];
                    all_weights[token * 4 + route] = weight_cases[token % 2][route];
                }
                baseline[token] = try applyBankOne(&g, &c, try ops.row(&g, inputs, @intCast(token)), all_ids[token * 4 ..][0..4], all_weights[token * 4 ..][0..4], all_refs[token * 4 ..][0..4], banks, down_bias);
            }
            const expected = try g.concat(baseline[0..tokens], 0);
            const actual = try applyBankRows(&g, &c, inputs, all_ids[0 .. tokens * 4], all_weights[0 .. tokens * 4], all_refs[0 .. tokens * 4], banks, down_bias);
            var expected_values: [8 * h]f32 = undefined;
            var actual_values: [8 * h]f32 = undefined;
            _ = try g.hostF32(try g.astype(expected, .float32), expected_values[0 .. tokens * h]);
            _ = try g.hostF32(try g.astype(actual, .float32), actual_values[0 .. tokens * h]);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected_values[0 .. tokens * h]), std.mem.sliceAsBytes(actual_values[0 .. tokens * h]));
        };
    }
}
