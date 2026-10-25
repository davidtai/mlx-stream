const std = @import("std");

pub const Format = enum {
    bf16,
    mxfp8,
    mixed,
    pub fn parse(value: std.json.Value) !Format {
        if (value == .string) {
            if (std.mem.eql(u8, value.string, "bf16")) return .bf16;
            if (std.mem.eql(u8, value.string, "mxfp8")) return .mxfp8;
            if (std.mem.eql(u8, value.string, "mixed")) return .mixed;
        }
        return error.MiniMaxResidentFormatUnsupported;
    }
};
pub const Dtype = enum { bf16, f32, u32, u8, other };
pub const Tensor = struct { dtype: Dtype, shape: []const c_int };
pub const Kind = enum { dense, mxfp8, affine };
pub const Params = struct { kind: Kind, bits: u32 = 16, group_size: u32 = 0 };

pub fn validate(format: Format, weight: Tensor, scales: ?Tensor, outputs: c_int, inputs: c_int) !void {
    _ = try parameters(format, weight, scales, null, outputs, inputs);
}
pub fn parameters(format: Format, weight: Tensor, scales: ?Tensor, biases: ?Tensor, outputs: c_int, inputs: c_int) !Params {
    if (weight.dtype == .bf16 and scales == null and biases == null) {
        if (!std.mem.eql(c_int, weight.shape, &.{ outputs, inputs })) return error.MiniMaxResidentShape;
        return .{ .kind = .dense };
    }
    if (format == .bf16) return error.MiniMaxResidentFormatMismatch;
    if (weight.dtype != .u32) return error.MiniMaxProjectionDtype;
    const scale = scales orelse return error.MiniMaxProjectionScalesMissing;
    if (format == .mixed and scale.dtype == .bf16) {
        const bias = biases orelse return error.MiniMaxProjectionBiasesMissing;
        if (bias.dtype != .bf16) return error.MiniMaxProjectionDtype;
        if (inputs <= 0 or outputs <= 0 or weight.shape.len != 2 or scale.shape.len != 2 or weight.shape[0] != outputs or scale.shape[0] != outputs or scale.shape[1] <= 0 or !std.mem.eql(c_int, bias.shape, scale.shape)) return error.MiniMaxResidentShape;
        const packed_bits: i64 = @as(i64, weight.shape[1]) * 32;
        if (@mod(packed_bits, inputs) != 0 or @mod(inputs, scale.shape[1]) != 0) return error.MiniMaxResidentShape;
        const bits = @divExact(packed_bits, inputs);
        const group = @divExact(inputs, scale.shape[1]);
        if ((bits != 3 and bits != 4 and bits != 5 and bits != 6 and bits != 8) or group != 64) return error.MiniMaxProjectionQuantizationUnsupported;
        return .{ .kind = .affine, .bits = @intCast(bits), .group_size = @intCast(group) };
    }
    if (scale.dtype != .u8 or biases != null) return error.MiniMaxProjectionDtype;
    if (inputs <= 0 or @mod(inputs, 32) != 0 or outputs <= 0 or
        !std.mem.eql(c_int, weight.shape, &.{ outputs, @divExact(inputs, 4) }) or
        !std.mem.eql(c_int, scale.shape, &.{ outputs, @divExact(inputs, 32) })) return error.MiniMaxResidentShape;
    return .{ .kind = .mxfp8, .bits = 8, .group_size = 32 };
}

test "MiniMax residents format explicitly admits MXFP8 and preserves BF16 default layout" {
    try std.testing.expectEqual(Format.bf16, try Format.parse(.{ .string = "bf16" }));
    try std.testing.expectEqual(Format.mxfp8, try Format.parse(.{ .string = "mxfp8" }));
    try std.testing.expectError(error.MiniMaxResidentFormatUnsupported, Format.parse(.{ .string = "fp8" }));
    try std.testing.expectError(error.MiniMaxResidentFormatUnsupported, Format.parse(.null));
    try validate(.bf16, .{ .dtype = .bf16, .shape = &.{ 32, 64 } }, null, 32, 64);
    try validate(.mxfp8, .{ .dtype = .bf16, .shape = &.{ 32, 64 } }, null, 32, 64);
}

test "MiniMax MXFP8 residents require exact packed shape and unsigned exponent scales" {
    const weight: Tensor = .{ .dtype = .u32, .shape = &.{ 32, 16 } };
    const scales: Tensor = .{ .dtype = .u8, .shape = &.{ 32, 2 } };
    try validate(.mxfp8, weight, scales, 32, 64);
    try std.testing.expectError(error.MiniMaxResidentFormatMismatch, validate(.bf16, weight, scales, 32, 64));
    try std.testing.expectError(error.MiniMaxProjectionScalesMissing, validate(.mxfp8, weight, null, 32, 64));
    try std.testing.expectError(error.MiniMaxResidentShape, validate(.mxfp8, weight, .{ .dtype = .u8, .shape = &.{ 32, 4 } }, 32, 64));
    try std.testing.expectError(error.MiniMaxResidentShape, validate(.mxfp8, .{ .dtype = .u32, .shape = &.{ 32, 64 } }, scales, 32, 64));
    try std.testing.expectError(error.MiniMaxProjectionDtype, validate(.mxfp8, weight, .{ .dtype = .f32, .shape = &.{ 32, 2 } }, 32, 64));
    try std.testing.expectError(error.MiniMaxProjectionDtype, validate(.mxfp8, .{ .dtype = .bf16, .shape = &.{ 32, 64 } }, scales, 32, 64));
}

test "MiniMax mixed residents infer every source affine width and require matching bias grids" {
    for ([_]c_int{ 3, 4, 5, 6, 8 }) |bits| {
        const q = try parameters(.mixed, .{ .dtype = .u32, .shape = &.{ 32, bits * 2 } }, .{ .dtype = .bf16, .shape = &.{ 32, 1 } }, .{ .dtype = .bf16, .shape = &.{ 32, 1 } }, 32, 64);
        try std.testing.expectEqual(Kind.affine, q.kind);
        try std.testing.expectEqual(@as(u32, @intCast(bits)), q.bits);
        try std.testing.expectEqual(@as(u32, 64), q.group_size);
    }
    const w: Tensor = .{ .dtype = .u32, .shape = &.{ 32, 6 } };
    const s: Tensor = .{ .dtype = .bf16, .shape = &.{ 32, 1 } };
    try std.testing.expectError(error.MiniMaxProjectionBiasesMissing, parameters(.mixed, w, s, null, 32, 64));
    try std.testing.expectError(error.MiniMaxResidentShape, parameters(.mixed, w, s, .{ .dtype = .bf16, .shape = &.{ 32, 2 } }, 32, 64));
}
