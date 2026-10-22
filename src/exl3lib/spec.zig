//! The pack-facing contract, hermetic (no MLX): `expert_quant` parsing, the admission every bank passes before upload,
//! and the row threshold between the decode chain and the prefill GEMM. The names, error names and refusals are
//! Sushi's `sushi_exl3` module's (lib/sushi/src/exl3/root.zig, MIT) so a host switches engines by import alone.
const std = @import("std");
pub const format = @import("format.zig");

pub const Spec = struct {
    rate: format.Rate,
    codebook: format.Codebook,
    window: format.Window = .w16,

    pub fn decode(self: Spec) format.Decode {
        return .{ .codebook = self.codebook, .window = self.window };
    }
};

/// `k` is a rate, integer or fractional: admitted only when 16k is an even whole number of halfwords per tile.
fn rateFromConfigK(k_v: std.json.Value) ?format.Rate {
    const scaled: f64 = switch (k_v) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |f| f * 16.0,
        else => return null,
    };
    const rounded = @round(scaled);
    if (@abs(scaled - rounded) > 1e-6) return null;
    if (rounded < 0 or rounded > 1024) return null;
    return format.kFromPackedDim(@intFromFloat(rounded));
}

/// `window` is the codeword width the pack's search hashed. Absent means 16; a width this build cannot decode is
/// refused, not rounded: the same bitstream decodes to different weights at each width.
fn windowFromConfig(v: ?std.json.Value) ?format.Window {
    const raw = v orelse return format.Window.w16;
    if (raw != .integer) return null;
    return format.Window.fromBits(raw.integer);
}

pub fn parseExpertQuant(obj: std.json.ObjectMap) !Spec {
    const block = obj.get("expert_quant") orelse return error.ExpertLayoutUnsupported;
    if (block != .object) return error.ExpertLayoutUnsupported;
    const fmt_v = block.object.get("format") orelse return error.ExpertLayoutUnsupported;
    if (fmt_v != .string or !std.mem.eql(u8, fmt_v.string, "exl3")) return error.ExpertLayoutUnsupported;
    const k_v = block.object.get("k") orelse return error.ExpertLayoutUnsupported;
    const rate = rateFromConfigK(k_v) orelse return error.ExpertLayoutUnsupported;
    const cb_v = block.object.get("codebook") orelse return error.ExpertLayoutUnsupported;
    if (cb_v != .string) return error.ExpertLayoutUnsupported;
    const codebook = format.Codebook.fromName(cb_v.string) orelse return error.ExpertLayoutUnsupported;
    const window = windowFromConfig(block.object.get("window")) orelse return error.Exl3WindowUnsupported;
    return .{ .rate = rate, .codebook = codebook, .window = window };
}

pub fn specFromConfigJson(allocator: std.mem.Allocator, raw: []const u8) !Spec {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    return parseExpertQuant(parsed.value.object);
}

/// Rows at or below this take the decode chain (GEMV forms); wider calls take the prefill GEMM.
pub const DECODE_ROWS_MAX: usize = 16;
/// The widest top-k the decode chain's reduction holds.
pub const REDUCE_MAX_TOPK: c_int = 32;

pub fn usesPrefillArm(rows: usize) bool {
    return rows > DECODE_ROWS_MAX;
}

/// The rows of an activation [..., D]: every dim but the last.
pub fn rowsOfShape(shape: []const c_int) usize {
    if (shape.len < 2) return 1;
    var rows: usize = 1;
    for (shape[0 .. shape.len - 1]) |d| rows *= @intCast(@max(d, 0));
    return rows;
}

pub fn admitTopK(topk: u32) !void {
    if (topk > REDUCE_MAX_TOPK) return error.Exl3TopKExceedsReduceBank;
}

/// A stacked trellis [E, in/16, out/16, n] at or under the config's rate.
pub fn trellisAdmitted(shape: []const c_int, experts: u32, in_dim: u32, out_dim: u32, rate: format.Rate) bool {
    if (shape.len != 4) return false;
    for (shape) |d| if (d <= 0) return false;
    if (in_dim % 16 != 0 or out_dim % 16 != 0) return false;
    const want = [3]u32{ experts, in_dim / 16, out_dim / 16 };
    for (want, 0..) |w, i| if (w != @as(u32, @intCast(shape[i]))) return false;
    const packed_rate = format.kFromPackedDim(@intCast(shape[3])) orelse return false;
    return packed_rate.n <= rate.n;
}

const t = std.testing;

test "exl3lib expert_quant: the window is 16 unless the config names one this build decodes" {
    const base = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\"";
    try t.expectEqual(format.Window.w16, (try specFromConfigJson(t.allocator, base ++ "}}")).window);
    try t.expectEqual(format.Window.w16, (try specFromConfigJson(t.allocator, base ++ ",\"window\":16}}")).window);
    try t.expectEqual(format.Window.w12, (try specFromConfigJson(t.allocator, base ++ ",\"window\":12}}")).window);
    try t.expectEqual(format.Window.w8, (try specFromConfigJson(t.allocator, base ++ ",\"window\":8}}")).window);
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":17}}"));
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":7}}"));
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":\"12\"}}"));
}

test "exl3lib expert_quant: integer and fractional K1..K8 under both codebooks; the rest refused by name" {
    const cases = [_]struct { text: []const u8, n: u32 }{
        .{ .text = "1", .n = 16 },    .{ .text = "2", .n = 32 }, .{ .text = "2.25", .n = 36 }, .{ .text = "2.5", .n = 40 },
        .{ .text = "2.75", .n = 44 }, .{ .text = "3", .n = 48 }, .{ .text = "3.5", .n = 56 },  .{ .text = "4", .n = 64 },
        .{ .text = "5", .n = 80 },    .{ .text = "6", .n = 96 }, .{ .text = "6.5", .n = 104 }, .{ .text = "7", .n = 112 },
        .{ .text = "8", .n = 128 },
    };
    for (cases) |c| {
        for ([_]format.Codebook{ .mul1, .mcg }) |cb| {
            var buf: [96]u8 = undefined;
            const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"{s}\"}}}}", .{ c.text, @tagName(cb) });
            const spec = try specFromConfigJson(t.allocator, raw);
            try t.expectEqual(c.n, spec.rate.n);
            try t.expectEqual(cb, spec.codebook);
            try t.expectEqual(cb, spec.decode().codebook);
        }
    }
    // 2.3 is not a multiple of 1/16; 2.0625 is an odd halfword count; 0.875 and 8.125 are off the range
    for ([_][]const u8{ "2.3", "8.125", "9", "0.875", "2.0625" }) |bad| {
        var buf: [96]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"mul1\"}}}}", .{bad});
        try t.expectError(error.ExpertLayoutUnsupported, specFromConfigJson(t.allocator, raw));
    }
    try t.expectError(error.ExpertLayoutUnsupported, specFromConfigJson(t.allocator,
        \\{"expert_quant":{"format":"exl3","k":4,"codebook":"mul2"}}
    ));
    try t.expectError(error.ExpertLayoutUnsupported, specFromConfigJson(t.allocator, "{}"));
    try t.expectError(error.ExpertLayoutUnsupported, specFromConfigJson(t.allocator,
        \\{"expert_quant":{"format":"affine","k":4,"codebook":"mul1"}}
    ));
}

test "exl3lib top-k above the reduce bank is a named refusal; rows pick the arm" {
    try admitTopK(10);
    try admitTopK(32);
    try t.expectError(error.Exl3TopKExceedsReduceBank, admitTopK(33));
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(DECODE_ROWS_MAX));
    try t.expect(usesPrefillArm(DECODE_ROWS_MAX + 1));
    try t.expectEqual(@as(usize, 6), rowsOfShape(&.{ 2, 3, 2560 }));
    try t.expectEqual(@as(usize, 1), rowsOfShape(&.{2560}));
}

test "exl3lib trellisAdmitted: Qwen3.8-Flash-Next's K5 bank, at or under the ceiling, geometry refusals" {
    const k5 = format.Rate.fromK(5);
    // gate/up [512, 2560/16, 640/16, 80], down [512, 40, 160, 80]
    try t.expect(trellisAdmitted(&.{ 512, 160, 40, 80 }, 512, 2560, 640, k5));
    try t.expect(trellisAdmitted(&.{ 512, 40, 160, 80 }, 512, 640, 2560, k5));
    try t.expect(trellisAdmitted(&.{ 512, 160, 40, 64 }, 512, 2560, 640, k5)); // a K4 shard under a K5 ceiling: billed at the ceiling
    try t.expect(!trellisAdmitted(&.{ 512, 160, 40, 96 }, 512, 2560, 640, k5)); // K6 over a K5 ceiling
    try t.expect(!trellisAdmitted(&.{ 512, 160, 40, 81 }, 512, 2560, 640, k5)); // an odd halfword count
    try t.expect(!trellisAdmitted(&.{ 512, 160, 40 }, 512, 2560, 640, k5));
    try t.expect(!trellisAdmitted(&.{ 256, 160, 40, 80 }, 512, 2560, 640, k5));
    try t.expect(!trellisAdmitted(&.{ 512, 160, 40, 80 }, 512, 2568, 640, k5));
    // DeepSeek-V4.1 K2 (5120 x 2304) and a K8 ceiling
    try t.expect(trellisAdmitted(&.{ 384, 320, 144, 32 }, 384, 5120, 2304, format.Rate.fromK(2)));
    try t.expect(trellisAdmitted(&.{ 384, 320, 144, 128 }, 384, 5120, 2304, format.Rate.fromK(8)));
}
