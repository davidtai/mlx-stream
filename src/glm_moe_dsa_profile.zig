//! A pack's slot profile (`slot-profile.json` beside the bank's manifest): per bank layer a weight on its share of the
//! fill's units, from a recorded decode's misses (`scripts/glm_moe_dsa_slot_profile.py`). The bank layers a decode
//! misses most hold more expert slots at every fill, the ones it misses least fewer. A pack without a profile fills
//! every bank layer at the share its bank gives it.
//!
//!   {"format": "glm-slot-profile-v1", "experts": [<each bank layer's experts>], "weights": [<each one's weight>]}

const std = @import("std");
const glm = @import("glm_moe_dsa.zig");
const Diag = glm.Diag;

pub const file_name = "slot-profile.json";
pub const format = "glm-slot-profile-v1";
/// The largest profile read (150 bank layers' experts and weights fit in a few KiB).
const max_bytes = 1 << 20;

pub const Error = error{ SlotProfileInvalid, SlotProfileMismatch };

/// The weights a profile set: their bank layers and their spread (the bank's line).
pub const Summary = struct { layers: usize, min: f64, max: f64 };

const Json = struct { format: []const u8, experts: []const u32, weights: []const f64 };

fn refuse(diag: ?*Diag, err: Error, comptime fmt: []const u8, args: anytype) Error {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// `dir`'s profile onto `slots`: each bank layer's share times its weight; null when the pack has none. Refused by
/// name: a file that is not the format or whose weights are not positive and finite (`SlotProfileInvalid`), one whose
/// bank layers or their experts are not the bank's (`SlotProfileMismatch`).
pub fn apply(a: std.mem.Allocator, io: std.Io, dir: []const u8, slots: []glm.LayerSlots, diag: ?*Diag) !?Summary {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const path = try std.fmt.allocPrint(aa, "{s}/" ++ file_name, .{dir});
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, aa, .limited(max_bytes)) catch |e| switch (e) {
        error.FileNotFound => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(diag, error.SlotProfileInvalid, file_name ++ ": {s}", .{@errorName(e)}),
    };
    const p = std.json.parseFromSliceLeaky(Json, aa, text, .{}) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(diag, error.SlotProfileInvalid, file_name ++ ": {s}", .{@errorName(e)}),
    };
    if (!std.mem.eql(u8, p.format, format)) return refuse(diag, error.SlotProfileInvalid, file_name ++ ": format \"{s}\", want \"" ++ format ++ "\"", .{p.format});
    if (p.experts.len != slots.len or p.weights.len != slots.len)
        return refuse(diag, error.SlotProfileMismatch, file_name ++ ": {d} experts and {d} weights for the bank's {d} bank layers", .{ p.experts.len, p.weights.len, slots.len });
    var s: Summary = .{ .layers = slots.len, .min = std.math.inf(f64), .max = 0 };
    for (slots, p.experts, p.weights, 0..) |sl, e, w, l| {
        const experts = sl.n_records / @max(sl.rows_per_unit, 1);
        if (e != experts) return refuse(diag, error.SlotProfileMismatch, file_name ++ ": bank layer {d} has {d} experts, the profile {d}", .{ l, experts, e });
        if (!std.math.isFinite(w) or w <= 0) return refuse(diag, error.SlotProfileInvalid, file_name ++ ": bank layer {d}'s weight {d} is not positive and finite", .{ l, w });
        s.min = @min(s.min, w);
        s.max = @max(s.max, w);
    }
    for (slots, p.weights) |*sl, w| sl.share *= w;
    return s;
}

const testing = std.testing;

fn withProfile(tmp: *std.testing.TmpDir, text: []const u8, buf: []u8) ![]const u8 {
    try tmp.dir.writeFile(testing.io, .{ .sub_path = file_name, .data = text });
    return buf[0..try tmp.dir.realPath(testing.io, buf)];
}

test "glm slot profile: each bank layer's share times its weight; none without the file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var slots = [_]glm.LayerSlots{ .{ .record_bytes = 10, .n_records = 148, .share = 148.0 / 128.0 }, .{ .record_bytes = 20, .n_records = 108, .share = 108.0 / 128.0 } };
    const root = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    try testing.expectEqual(@as(?Summary, null), try apply(testing.allocator, testing.io, root, &slots, null));
    try testing.expectEqual(@as(f64, 148.0 / 128.0), slots[0].share);
    const dir = try withProfile(&tmp, "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 108], \"weights\": [1.25, 0.5]}", &buf);
    const s = (try apply(testing.allocator, testing.io, dir, &slots, null)).?;
    try testing.expectEqual(Summary{ .layers = 2, .min = 0.5, .max = 1.25 }, s);
    try testing.expectEqual(@as(f64, 1.25 * 148.0 / 128.0), slots[0].share);
    try testing.expectEqual(@as(f64, 0.5 * 108.0 / 128.0), slots[1].share);
    // Whole-number weights parse as weights.
    slots[0].share = 1;
    _ = try apply(testing.allocator, testing.io, try withProfile(&tmp, "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 108], \"weights\": [2, 1]}", &buf), &slots, null);
    try testing.expectEqual(@as(f64, 2), slots[0].share);
}

test "glm slot profile: a profile that is not the format, or not the bank's, refuses by name" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const Case = struct { text: []const u8, err: Error, why: []const u8 };
    const cases = [_]Case{
        .{ .text = "{\"format\": \"glm-slot-profile-v2\", \"experts\": [148, 108], \"weights\": [1, 1]}", .err = error.SlotProfileInvalid, .why = "format" },
        .{ .text = "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 108], \"weights\": [1, 0]}", .err = error.SlotProfileInvalid, .why = "not positive" },
        .{ .text = "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 108], \"weights\": [1, -2]}", .err = error.SlotProfileInvalid, .why = "not positive" },
        .{ .text = "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 108], \"weights\": [1, 1], \"note\": \"x\"}", .err = error.SlotProfileInvalid, .why = "UnknownField" },
        .{ .text = "[1, 2]", .err = error.SlotProfileInvalid, .why = "UnexpectedToken" },
        .{ .text = "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148], \"weights\": [1]}", .err = error.SlotProfileMismatch, .why = "bank layers" },
        .{ .text = "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 108, 148], \"weights\": [1, 1]}", .err = error.SlotProfileMismatch, .why = "bank layers" },
        .{ .text = "{\"format\": \"glm-slot-profile-v1\", \"experts\": [148, 100], \"weights\": [1, 1]}", .err = error.SlotProfileMismatch, .why = "bank layer 1 has 108 experts, the profile 100" },
    };
    for (cases) |cs| {
        var slots = [_]glm.LayerSlots{ .{ .record_bytes = 10, .n_records = 148 }, .{ .record_bytes = 20, .n_records = 108 } };
        var diag: Diag = .{};
        try testing.expectError(cs.err, apply(testing.allocator, testing.io, try withProfile(&tmp, cs.text, &buf), &slots, &diag));
        if (std.mem.indexOf(u8, diag.message(), cs.why) == null) {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ cs.why, diag.message() });
            return error.TestUnexpectedResult;
        }
        // A refused profile leaves the shares as the bank gave them.
        try testing.expectEqual(@as(f64, 1), slots[0].share);
    }
}
