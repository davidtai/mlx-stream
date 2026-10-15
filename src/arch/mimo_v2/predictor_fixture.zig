//! Small host contracts for MiMo's borrowed lookahead installation.
const std = @import("std");
const mlx = @import("sdk").mlx;
const source_mod = @import("source.zig");
const stream_mod = @import("stream.zig");
const bank_mod = @import("bank.zig");
const math = @import("math.zig").Math;
const expert = @import("../../sdk_ext.zig").expert;
const ops = @import("../deepseek_v41/ops.zig");

test "mimo lookahead: F32 promotion precedes borrowed transposed sigmoid correction" {
    if (comptime @hasDecl(math, "predictorScores")) {
        var g: ScalarOps = .{};
        const x: ScalarOps.T = .{ .rows = 1, .cols = 2, .dtype = .bfloat16, .values = [_]f32{ 0.5, -0.25 } ++ @as([14]f32, @splat(0)) };
        const router: ScalarOps.T = .{ .rows = 2, .cols = 8, .dtype = .float32, .values = .{ -2, -1, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8 } };
        const bias: ScalarOps.T = .{ .rows = 1, .cols = 8, .dtype = .float32, .values = [_]f32{ 0, 0, 0, 0, 0, 0, 0.5, 1 } ++ @as([8]f32, @splat(0)) };
        const actual = try math.predictorScores(&g, x, router, bias);
        try std.testing.expectEqual(@as(u8, 4), g.stage);
        for (actual.values[0..8], 0..) |value, i| {
            const z: f32 = x.values[0] * router.values[i] + x.values[1] * router.values[8 + i];
            try std.testing.expectApproxEqAbs(1 / (1 + @exp(-z)) + bias.values[i], value, 0.000001);
        }
        var selector = try expert.lookahead.SelectorOf(8).init(std.testing.allocator, 8, 8, std.math.inf(f32), 2);
        defer selector.deinit(std.testing.allocator);
        var policy = try expert.policy.LayerPolicy.init(std.testing.allocator, 8, 1);
        defer policy.deinit(std.testing.allocator);
        var selected: [2]u16 = undefined;
        try std.testing.expectEqualSlices(u16, &.{ 7, 6 }, selector.select(actual.values[0..8], &policy, &selected));
    } else return error.MimoLookaheadPredictorMissing;
}

test "mimo lookahead: installation validates explicit borrowed absolute gates once" {
    if (comptime @hasDecl(source_mod, "Gate") and @hasDecl(source_mod, "validateGatesFor")) {
        const geometry = try bank_mod.Geometry.init(64, 64, 64, 1, 2);
        const borrowed: source_mod.Gate = .{ .w = .{ .ctx = @ptrFromInt(1) }, .bias = .{ .ctx = @ptrFromInt(2) } };
        const gates = [_]source_mod.Gate{ .{ .w = .{}, .bias = .{} }, borrowed, borrowed };
        var g: GateOps = .{};
        try source_mod.validateGatesFor(&g, geometry, &gates, .f32);
        try std.testing.expectEqual(@as(usize, 4), g.shapes);
        try std.testing.expectError(error.MimoPredictorGateTable, source_mod.validateGatesFor(&g, geometry, gates[0..2], .f32));
        g.bad_shape = true;
        try std.testing.expectError(error.MimoPredictorGateShape, source_mod.validateGatesFor(&g, geometry, &gates, .f32));
        g.bad_shape = false;
        g.bad_dtype = true;
        try std.testing.expectError(error.MimoPredictorGateDtype, source_mod.validateGatesFor(&g, geometry, &gates, .f32));
    } else return error.MimoLookaheadGateValidationMissing;
}

test "mimo lookahead: reader options are explicit and prediction follows genuine phase" {
    const options = stream_mod.eventOptions(&.{1}, .host, .{ .id = 7, .object = 123 });
    const lookahead = options.lookahead orelse return error.MimoLookaheadOptionsMissing;
    try std.testing.expectEqual(@as(u32, 8), lookahead.k);
    try std.testing.expectEqual(std.math.inf(f32), lookahead.tau);
    try std.testing.expectEqual(@as(u32, 2), lookahead.budget);
    try std.testing.expectEqual(@as(u32, 4), lookahead.chunks);
    try std.testing.expectEqual(@as(u32, 0), lookahead.idle_busy);
    try std.testing.expect(lookahead.preread);
    if (comptime @hasDecl(stream_mod.StreamSource, "decoding")) {
        var stream: stream_mod.Stream = undefined;
        stream.phase = .prefill;
        var source = stream_mod.StreamSource.init(&stream);
        try std.testing.expect(!source.decoding());
        stream.phase = .decode;
        try std.testing.expect(source.decoding());
        stream.phase = .prefill;
        try std.testing.expect(!source.decoding());
    } else return error.MimoLookaheadPhaseMissing;
}

test "mimo lookahead: bill charges page-rounded speculation and predictor resources" {
    if (comptime @hasField(source_mod.MemoryBill, "spec_staging_bytes") and @hasField(source_mod.MemoryBill, "spec_runtime_bytes") and
        @hasField(source_mod.MemoryBill, "predictor_metadata_bytes") and @hasField(source_mod.MemoryBill, "predictor_workspace_bytes") and
        @hasField(source_mod.MemoryBill, "predictor_runtime_bytes"))
    {
        const geometry = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
        const rows: [47]u32 = @splat(124);
        const bill = try source_mod.memoryBill(geometry, &rows);
        try std.testing.expectEqual(4 * expert.io.slotBytes(geometry.record_bytes, std.heap.pageSize()), bill.spec_staging_bytes);
        try std.testing.expectEqual(@as(u64, 53_608_448), 4 * expert.io.slotBytes(geometry.record_bytes, 16_384));
        try std.testing.expect(bill.spec_runtime_bytes >= 2 * (1 << 20));
        try std.testing.expect(bill.predictor_metadata_bytes >= 32 + 48 * @sizeOf(source_mod.Gate));
        try std.testing.expect(bill.predictor_workspace_bytes >= 4 * (4096 + 3 * 256));
        try std.testing.expect(bill.predictor_runtime_bytes > 0);
        try std.testing.expectEqual(bill.persistent_bytes + bill.transient_bytes + bill.staging_bytes + bill.policy_bytes +
            bill.reader_metadata_bytes + bill.manifest_bytes + bill.workspace_bytes + bill.event_runtime_bytes + bill.event_graph_bytes +
            bill.spec_staging_bytes + bill.spec_runtime_bytes + bill.predictor_metadata_bytes + bill.predictor_workspace_bytes + bill.predictor_runtime_bytes, bill.total_bytes);
        const fixture = try bank_mod.Geometry.init(4096, 2048, 64, 1, 2);
        const native = try source_mod.memoryBillForTokens(fixture, &.{ 1, 1 }, 7);
        // Same fixture controls: 13,374,720 B graphs/outputs + 1,048,832 B
        // synthetic borrowed router/correction. One 64-row oracle is reused.
        const bound = native.total_bytes + 64 * fixture.record_bytes + 14_423_552 + ((256 + 512 + 128) << 20);
        try std.testing.expect(bound < 4 << 30);
        std.debug.print("MIMO_LOOKAHEAD_FIXTURE_CPU bound={d} ceiling={d} source={d} oracle={d} controls={d}\n", .{ bound, @as(u64, 4 << 30), native.total_bytes, 64 * fixture.record_bytes, @as(u64, 14_423_552) });
    } else return error.MimoLookaheadBillMissing;
}

// No MLX handles, device or kernels. The call sequence is part of the
// arithmetic contract; omitting promotion or transposing again fails here.
const ScalarOps = struct {
    pub const T = struct { rows: usize, cols: usize, dtype: mlx.mlx_dtype, values: [16]f32 };
    stage: u8 = 0,
    pub fn astype(g: *ScalarOps, x: T, dtype: mlx.mlx_dtype) !T {
        if (g.stage != 0 or dtype != .float32) return error.PredictorOrder;
        g.stage = 1;
        var result = x;
        result.dtype = dtype;
        return result;
    }
    pub fn matmul(g: *ScalarOps, x: T, w: T) !T {
        if (g.stage != 1 or x.dtype != .float32 or w.dtype != .float32 or x.cols != w.rows) return error.PredictorOrder;
        g.stage = 2;
        var result: T = .{ .rows = x.rows, .cols = w.cols, .dtype = .float32, .values = @splat(0) };
        for (0..result.cols) |col| for (0..x.cols) |k| {
            result.values[col] += x.values[k] * w.values[k * w.cols + col];
        };
        return result;
    }
    pub fn sigmoid(g: *ScalarOps, x: T) !T {
        if (g.stage != 2) return error.PredictorOrder;
        g.stage = 3;
        var result = x;
        for (result.values[0..x.cols]) |*v| v.* = 1 / (1 + @exp(-v.*));
        return result;
    }
    pub fn add(g: *ScalarOps, x: T, bias: T) !T {
        if (g.stage != 3 or bias.dtype != .float32) return error.PredictorOrder;
        g.stage = 4;
        var result = x;
        for (result.values[0..x.cols], bias.values[0..x.cols]) |*v, b| v.* += b;
        return result;
    }
};

test "mimo lookahead X19: the gate table is checked for the installed predictor route: F32 [hidden, experts] (f32), the BF16 weight as stored [experts, hidden] (bf16); a mismatch refuses by name" {
    const geometry = try bank_mod.Geometry.init(128, 64, 32, 1, 2);
    var g: RouteGateOps = .{};
    const empty: source_mod.Gate = .{ .w = .{}, .bias = .{} };
    const copy: source_mod.Gate = .{ .w = .{ .ctx = @ptrFromInt(1) }, .bias = .{ .ctx = @ptrFromInt(3) } };
    const stored: source_mod.Gate = .{ .w = .{ .ctx = @ptrFromInt(2) }, .bias = .{ .ctx = @ptrFromInt(3) } };
    const flipped: source_mod.Gate = .{ .w = .{ .ctx = @ptrFromInt(4) }, .bias = .{ .ctx = @ptrFromInt(3) } };
    try source_mod.validateGatesFor(&g, geometry, &.{ empty, copy, copy }, .f32);
    try source_mod.validateGatesFor(&g, geometry, &.{ empty, stored, stored }, .bf16);
    try std.testing.expectError(error.MimoPredictorGateDtype, source_mod.validateGatesFor(&g, geometry, &.{ empty, stored, stored }, .f32));
    try std.testing.expectError(error.MimoPredictorGateDtype, source_mod.validateGatesFor(&g, geometry, &.{ empty, copy, copy }, .bf16));
    try std.testing.expectError(error.MimoPredictorGateShape, source_mod.validateGatesFor(&g, geometry, &.{ empty, flipped, flipped }, .bf16));
    try std.testing.expectError(error.MimoPredictRouter, source_mod.validateGatesFor(&g, geometry, &.{ empty, copy, copy }, .invalid));
    // The scheduler reaches the installed route only through the math's pub decl.
    try std.testing.expect(@hasDecl(math, "predictorScoresBound"));
}

/// Arrays by handle: 1 the F32 copy [128, 32], 2 the stored BF16 weight [32, 128], 4 a BF16 weight in the copy's
/// orientation, anything else the F32 correction [32].
const RouteGateOps = struct {
    pub fn shapeOf(_: *RouteGateOps, x: mlx.mlx_array) ops.Shape {
        return switch (@intFromPtr(x.ctx.?)) {
            1, 4 => .{ .n = 2, .d = .{ 128, 32, 0, 0, 0, 0, 0, 0 } },
            2 => .{ .n = 2, .d = .{ 32, 128, 0, 0, 0, 0, 0, 0 } },
            else => .{ .n = 1, .d = .{ 32, 0, 0, 0, 0, 0, 0, 0 } },
        };
    }
    pub fn dtypeOf(_: *RouteGateOps, x: mlx.mlx_array) mlx.mlx_dtype {
        return switch (@intFromPtr(x.ctx.?)) {
            2, 4 => .bfloat16,
            else => .float32,
        };
    }
};

const GateOps = struct {
    bad_shape: bool = false,
    bad_dtype: bool = false,
    shapes: usize = 0,
    pub fn shapeOf(g: *GateOps, x: mlx.mlx_array) ops.Shape {
        g.shapes += 1;
        return if (@intFromPtr(x.ctx.?) == 1)
            .{ .n = 2, .d = .{ if (g.bad_shape) 32 else 64, 64, 0, 0, 0, 0, 0, 0 } }
        else
            .{ .n = 1, .d = .{ 64, 0, 0, 0, 0, 0, 0, 0 } };
    }
    pub fn dtypeOf(g: *GateOps, _: mlx.mlx_array) mlx.mlx_dtype {
        return if (g.bad_dtype) .bfloat16 else .float32;
    }
};
