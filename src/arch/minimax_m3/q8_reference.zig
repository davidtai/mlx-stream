const std = @import("std");
pub const group_size = 32;

pub fn vectorBytes(dim: u32) !u64 {
    if (dim == 0 or dim % group_size != 0) return error.MiniMaxQ8Shape;
    return @as(u64, dim) + @as(u64, dim / group_size) * 4;
}

pub fn cacheBytes(layers: u32, kv_heads: u32, dim: u32, index_dim: u32, tokens: u64) !u64 {
    return @as(u64, layers) * tokens * (2 * @as(u64, kv_heads) * try vectorBytes(dim) + try vectorBytes(index_dim));
}

pub fn quantize(values: []const f32, codes: []i8, scales: []f32) !void {
    if (values.len == 0 or values.len % group_size != 0 or codes.len != values.len or scales.len != values.len / group_size) return error.MiniMaxQ8Shape;
    for (scales, 0..) |*scale, group| {
        var max: f32 = 0;
        for (values[group * group_size ..][0..group_size]) |value| {
            if (!std.math.isFinite(value)) return error.MiniMaxQ8NonFinite;
            max = @max(max, @abs(value));
        }
        scale.* = if (max > 0) @max(max / 127, 1e-30) else 1;
        for (values[group * group_size ..][0..group_size], codes[group * group_size ..][0..group_size]) |value, *code| {
            const scaled = std.math.clamp(value / scale.*, -127, 127);
            const floor = @floor(scaled);
            const low: i32 = @intFromFloat(floor);
            const fraction = scaled - floor;
            const rounded = if (fraction < 0.5 or (fraction == 0.5 and @mod(low, 2) == 0)) low else low + 1;
            code.* = @intCast(rounded);
        }
    }
}

pub fn dequantize(codes: []const i8, scales: []const f32, out: []f32) !void {
    if (codes.len == 0 or codes.len % group_size != 0 or out.len != codes.len or scales.len != codes.len / group_size) return error.MiniMaxQ8Shape;
    for (codes, out, 0..) |code, *value, i| value.* = @as(f32, @floatFromInt(code)) * scales[i / group_size];
}

test "MiniMax q8 cache accounts codes and F32 group scales" {
    try std.testing.expectEqual(@as(u64, 144), try vectorBytes(128));
    try std.testing.expectEqual(@as(u64, 10_192_158_720), try cacheBytes(60, 4, 128, 128, 131072));
    try std.testing.expectError(error.MiniMaxQ8Shape, vectorBytes(127));
}

test "MiniMax q8 symmetric groups preserve zeros endpoints and round ties to even" {
    var values: [64]f32 = @splat(0);
    values[0] = 127;
    values[1] = -127;
    values[2] = 63.5;
    values[3] = -62.5;
    values[32] = 0.25;
    var codes: [64]i8 = undefined;
    var scales: [2]f32 = undefined;
    try quantize(&values, &codes, &scales);
    try std.testing.expectEqualSlices(i8, &.{ 127, -127, 64, -62 }, codes[0..4]);
    try std.testing.expectEqual(@as(i8, 127), codes[32]);
    var decoded: [64]f32 = undefined;
    try dequantize(&codes, &scales, &decoded);
    for (decoded, values, 0..) |got, want, i| try std.testing.expect(@abs(got - want) <= scales[i / 32] * 0.50001);
    @memset(&values, 0);
    try quantize(&values, &codes, &scales);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, &scales);
    for (codes) |code| try std.testing.expectEqual(@as(i8, 0), code);
}
