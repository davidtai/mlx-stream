//! The settings helpers a streamed arch reads its `expert_streaming.settings` object with (beam; glm5_next and MiMo keep
//! their own readers for now): a key named an enum's value, a bool key, a positive row count, the unknown-key check and
//! the per-layer rows split. A helper marks a bad value; the arch refuses it by its own name in its `validate`.

const std = @import("std");

pub const LayerRows = @import("expert/layer_rows.zig").LayerRows;
pub const layerRows = @import("expert/layer_rows.zig").layerRows;

/// `obj[key]` as one of `E`'s names into `dst` (left as it is when absent); a value that is not one of them sets `bad`.
/// Returns whether the key was named.
pub fn route(comptime E: type, obj: std.json.ObjectMap, key: []const u8, dst: *E, bad: *bool) bool {
    const v = obj.get(key) orelse return false;
    const r: ?E = if (v == .string) std.meta.stringToEnum(E, v.string) else null;
    if (r) |x| dst.* = x else bad.* = true;
    return true;
}

/// `obj[key]` as a bool into `dst`; another type sets `bad`. Returns whether the key was named.
pub fn boolKey(obj: std.json.ObjectMap, key: []const u8, dst: *bool, bad: *bool) bool {
    const v = obj.get(key) orelse return false;
    if (v == .bool) dst.* = v.bool else bad.* = true;
    return true;
}

/// `obj[key]` as a positive row count into `dst` (null when absent); anything else sets `bad`.
pub fn rows(obj: std.json.ObjectMap, key: []const u8, dst: *?u32, bad: *bool) void {
    const v = obj.get(key) orelse return;
    if (v == .integer and v.integer > 0 and v.integer <= std.math.maxInt(u32)) dst.* = @intCast(v.integer) else bad.* = true;
}

/// The first key of `o` that is not in `keys` (null: every key is a setting).
pub fn unknownSetting(o: std.json.ObjectMap, keys: []const []const u8) ?[]const u8 {
    var it = o.iterator();
    while (it.next()) |kv| {
        for (keys) |k| {
            if (std.mem.eql(u8, k, kv.key_ptr.*)) break;
        } else return kv.key_ptr.*;
    }
    return null;
}

const testing = std.testing;

test "sdk_ext settings: an enum route, a bool, rows and the unknown key, each marked bad by value" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"r\":\"b\",\"q\":\"z\",\"f\":true,\"g\":1,\"n\":12,\"m\":0}", .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const E = enum { a, b };
    var e: E = .a;
    var bad = false;
    try testing.expect(route(E, o, "r", &e, &bad) and e == .b and !bad);
    try testing.expect(route(E, o, "q", &e, &bad) and bad);
    try testing.expect(!route(E, o, "absent", &e, &bad));
    var f = false;
    bad = false;
    try testing.expect(boolKey(o, "f", &f, &bad) and f and !bad);
    try testing.expect(boolKey(o, "g", &f, &bad) and bad);
    var n: ?u32 = null;
    bad = false;
    rows(o, "n", &n, &bad);
    try testing.expectEqual(@as(?u32, 12), n);
    rows(o, "m", &n, &bad);
    try testing.expect(bad);
    try testing.expectEqualStrings("q", unknownSetting(o, &.{ "r", "f", "g", "n", "m" }).?);
    try testing.expect(unknownSetting(o, &.{ "r", "q", "f", "g", "n", "m" }) == null);
}
