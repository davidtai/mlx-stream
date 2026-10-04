//! The cell's host heap at three marks, for PROFILE builds only (`dsv41_decode_timers.enabled`; in every other build
//! `Sample` is void and nothing is read): libc malloc's statistics over all zones (`malloc_zone_statistics(NULL)`) and
//! zone by zone (`malloc_get_all_zones`, each named; `default` marks the zone `malloc_default_zone()` is, or fronts
//! under the same name: the one the cell's c_allocator calls), so the decode's freed-but-cached heap
//! (`size_allocated - size_in_use`, what the host-relief route could return) is read from inside the process, per zone.
//! `HOST_HEAP {...}` and the receipt's `host_heap`.
const std = @import("std");
const dt = @import("dsv41_decode_timers.zig");

pub const enabled = dt.enabled;

/// `malloc_statistics_t` (<malloc/malloc.h>).
pub const Stats = extern struct { blocks_in_use: c_uint = 0, size_in_use: usize = 0, max_size_in_use: usize = 0, size_allocated: usize = 0 };

pub const max_zones = 16;
pub const Zone = struct { name: [48]u8 = @splat(0), name_len: u8 = 0, default: bool = false, stats: Stats = .{} };
/// One mark: every zone's sum, and each zone (at most `max_zones`; `more_zones` counts the rest).
pub const Mark = struct { total: Stats = .{}, n: u8 = 0, more_zones: u32 = 0, zones: [max_zones]Zone = @splat(.{}) };

pub const Sample = if (enabled) Mark else void;

pub const marks = [_][]const u8{ "construction", "phase_change", "decode_end" };

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *Stats) void;
extern "c" fn malloc_get_all_zones(task: c_uint, reader: ?*anyopaque, addresses: *?[*]usize, count: *c_uint) c_int;
extern "c" fn malloc_get_zone_name(zone: *anyopaque) ?[*:0]const u8;
extern "c" fn malloc_default_zone() *anyopaque;
extern "c" var mach_task_self_: c_uint;

pub fn sample() Sample {
    if (comptime !enabled) return {};
    var m: Mark = .{};
    malloc_zone_statistics(null, &m.total);
    var addrs: ?[*]usize = null;
    var count: c_uint = 0;
    if (malloc_get_all_zones(mach_task_self_, null, &addrs, &count) != 0) return m;
    const def = @intFromPtr(malloc_default_zone());
    for (addrs.?[0..count]) |za| {
        if (m.n == max_zones) {
            m.more_zones += 1;
            continue;
        }
        const z: *anyopaque = @ptrFromInt(za);
        const zr = &m.zones[m.n];
        malloc_zone_statistics(z, &zr.stats);
        const nm = if (malloc_get_zone_name(z)) |p| std.mem.span(p) else "";
        zr.name_len = @intCast(@min(nm.len, zr.name.len));
        @memcpy(zr.name[0..zr.name_len], nm[0..zr.name_len]);
        zr.default = za == def;
        m.n += 1;
    }
    // malloc_default_zone() can be a front for a registered zone (same name, outside the list): mark that zone
    for (m.zones[0..m.n]) |z| if (z.default) return m;
    const dn = if (malloc_get_zone_name(@ptrFromInt(def))) |p| std.mem.span(p) else "";
    for (m.zones[0..m.n]) |*z| if (std.mem.eql(u8, z.name[0..z.name_len], dn)) {
        z.default = true;
        return m;
    };
    return m;
}

fn writeStats(w: *std.Io.Writer, s: Stats) !void {
    try w.print("\"blocks_in_use\": {d}, \"size_in_use\": {d}, \"size_allocated\": {d}, \"max_size_in_use\": {d}, \"cached\": {d}", .{ s.blocks_in_use, s.size_in_use, s.size_allocated, s.max_size_in_use, s.size_allocated -| s.size_in_use });
}

fn writeSamples(w: *std.Io.Writer, ms: *const [marks.len]Mark) !void {
    try w.writeAll("{\"total\": \"malloc_zone_statistics NULL (every zone)\"");
    for (marks, ms) |name, m| {
        try w.print(", \"{s}\": {{", .{name});
        try writeStats(w, m.total);
        try w.writeAll(", \"zones\": [");
        for (m.zones[0..m.n], 0..) |z, i| {
            try w.print("{s}{{\"name\": \"{s}\", \"default\": {}, ", .{ if (i == 0) "" else ", ", z.name[0..z.name_len], z.default });
            try writeStats(w, z.stats);
            try w.writeAll("}");
        }
        try w.print("], \"more_zones\": {d}}}", .{m.more_zones});
    }
    try w.writeAll("}");
}

/// `HOST_HEAP {...}`: per mark every zone's sum (`cached` = allocated - in use, bytes) and each zone's own.
pub fn line(buf: []u8, ms: *const [marks.len]Mark) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.writeAll("HOST_HEAP ") catch return buf[0..0];
    writeSamples(&w, ms) catch return buf[0..0];
    return w.buffered();
}

/// The receipt's `host_heap` (the line's object).
pub fn writeJson(w: *std.Io.Writer, ms: *const [marks.len]Mark) !void {
    try writeSamples(w, ms);
}

test "dsv41 host heap: a default build reads nothing (Sample is void)" {
    if (enabled) return error.SkipZigTest;
    try std.testing.expect(Sample == void);
    try std.testing.expect(@TypeOf(sample()) == void);
}

test "dsv41 host heap (profile builds): the line's shape over a hand mark; a live sample names every zone, the default among them" {
    if (comptime !enabled) return error.SkipZigTest;
    var z: Zone = .{ .default = true, .stats = .{ .blocks_in_use = 3, .size_in_use = 300, .size_allocated = 1300, .max_size_in_use = 900 } };
    @memcpy(z.name[0..17], "DefaultMallocZone");
    z.name_len = 17;
    var ms: [marks.len]Mark = @splat(.{});
    ms[2] = .{ .total = z.stats, .n = 1 };
    ms[2].zones[0] = z;
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqualStrings("HOST_HEAP {\"total\": \"malloc_zone_statistics NULL (every zone)\", \"construction\": {\"blocks_in_use\": 0, \"size_in_use\": 0, \"size_allocated\": 0, \"max_size_in_use\": 0, \"cached\": 0, \"zones\": [], \"more_zones\": 0}, \"phase_change\": {\"blocks_in_use\": 0, \"size_in_use\": 0, \"size_allocated\": 0, \"max_size_in_use\": 0, \"cached\": 0, \"zones\": [], \"more_zones\": 0}, \"decode_end\": {\"blocks_in_use\": 3, \"size_in_use\": 300, \"size_allocated\": 1300, \"max_size_in_use\": 900, \"cached\": 1000, \"zones\": [{\"name\": \"DefaultMallocZone\", \"default\": true, \"blocks_in_use\": 3, \"size_in_use\": 300, \"size_allocated\": 1300, \"max_size_in_use\": 900, \"cached\": 1000}], \"more_zones\": 0}}", line(&buf, &ms));
    const live = sample();
    try std.testing.expect(live.total.blocks_in_use > 0 and live.total.size_allocated >= live.total.size_in_use);
    try std.testing.expect(live.n > 0);
    var defaults: u32 = 0;
    var in_use: u64 = 0;
    for (live.zones[0..live.n]) |zz| {
        defaults += @intFromBool(zz.default);
        in_use += zz.stats.size_in_use;
    }
    try std.testing.expectEqual(@as(u32, 1), defaults);
    try std.testing.expect(in_use > 0);
    // the line over three live samples (the host dry run of its format)
    var lb: [8192]u8 = undefined;
    const three = [_]Mark{ live, sample(), sample() };
    std.debug.print("NATIVE {s}\n", .{line(&lb, &three)});
}
