//! The DSpark draft's routed expert ids, per stage and cycle, for PROFILE builds only (`dsv41_decode_timers.enabled`;
//! in every other build `StageIds` is void and each entry below compiles to nothing). Active in a decode-profile run
//! (`install`): each stage's routed ids (before a subset head's lut) are kept across the stage's wave, read in place
//! after the draft block's own eval (no extra sync), and folded into static storage; printed after the decode
//! (`DRAFT_ROUTE_HIST` per stage, `DRAFT_ROUTE_LRU`) and written whole into the receipt (`draftStreamJson`).
const std = @import("std");
const dt = @import("dsv41_decode_timers.zig");

pub const enabled = dt.enabled;

/// Set by `install` (a decode-profile run), cleared by `uninstall`.
pub var active: bool = false;

/// Cycles <= max_tokens - 1 (the cell's default cap is 1,024 ids: the primary + 1,023).
pub const max_cycles = 1024;
pub const max_stages = 4;
/// `Resident.routed` refuses a call of 64 routed ids or more (the unsorted gather), so a stage holds at most 63.
pub const max_ids = 63;
pub const max_experts = 512;
/// The geometric bound of the id storage (u16 ids + a u8 count per cycle and stage).
pub const storage_bytes = max_cycles * max_stages * (max_ids * @sizeOf(u16) + 1);

/// The draft head's geometry as installed.
pub const Geometry = struct { n_stages: u32, n_experts: u32, top_k: u32, block: u32 };

var ids: [max_cycles][max_stages][max_ids]u16 = undefined;
var n_ids: [max_cycles][max_stages]u8 = undefined;
pub var cycles: u32 = 0;
pub var geo: Geometry = .{ .n_stages = 0, .n_experts = 0, .top_k = 0, .block = 0 };

/// The hot-set capacities the measurements prices (over all stages' experts) and its cold cache.
pub const lru_capacities = [_]u32{ 96, 128, 201, 256 };
pub const cold_slots = 45;

/// A draft block's kept stage ids (`Draft.stage_ids`): void outside profile builds.
pub fn StageIds(comptime T: type) type {
    return if (enabled) [max_stages]?T else void;
}

pub fn noIds(comptime T: type) StageIds(T) {
    return if (enabled) @splat(null) else {};
}

/// Refuses a geometry the storage does not hold (before the decode, not inside it).
pub fn install(g: Geometry, max_tokens: u32) !void {
    if (comptime !enabled) return;
    if (g.n_stages == 0 or g.n_stages > max_stages) return error.DraftRouteStages;
    if (g.n_experts == 0 or g.n_experts > max_experts) return error.DraftRouteExperts;
    if (g.block * g.top_k == 0 or g.block * g.top_k > max_ids) return error.DraftRouteIds;
    if (max_tokens == 0 or max_tokens - 1 > max_cycles) return error.DraftRouteCycles;
    geo = g;
    cycles = 0;
    active = true;
}

pub fn uninstall() void {
    if (comptime !enabled) return;
    active = false;
}

/// One cycle's stages' ids (each a stage's row-major `[rows, top_k]`).
pub fn record(stage_ids: []const []const u16) void {
    if (comptime !enabled) return;
    std.debug.assert(cycles < max_cycles and stage_ids.len <= max_stages);
    for (0..max_stages) |s| {
        const src: []const u16 = if (s < stage_ids.len) stage_ids[s] else &.{};
        std.debug.assert(src.len <= max_ids);
        @memcpy(ids[cycles][s][0..src.len], src);
        n_ids[cycles][s] = @intCast(src.len);
    }
    cycles += 1;
}

/// The draft block's kept stage ids, read in place (the block's eval realised them) and recorded as one cycle,
/// then released. Called after the draft's own eval.
pub fn take(comptime G: type, g: *G, kept: *StageIds(G.T)) !void {
    if (comptime !enabled) return;
    var buf: [max_stages][max_ids]u16 = undefined;
    var sl: [max_stages][]const u16 = undefined;
    var n: usize = 0;
    defer drop(G, g, kept);
    for (kept, 0..) |k, s| {
        const x = k orelse continue;
        const len: usize = @intCast(g.shapeOf(x).numel());
        if (len > max_ids) return error.DraftRouteIds;
        sl[s] = try g.hostIds(x, buf[s][0..len]);
        n = s + 1;
    }
    if (n == 0) return;
    record(sl[0..n]);
}

/// Releases a block's kept stage ids without recording them (a dropped block).
pub fn drop(comptime G: type, g: *G, kept: *StageIds(G.T)) void {
    if (comptime !enabled) return;
    for (kept) |*k| if (k.*) |x| {
        g.release(x);
        k.* = null;
    };
}

/// The stage `s` per-expert counts over the recorded cycles (`out[0..n_experts]`); returns the routed total.
pub fn counts(s: usize, out: []u32) u64 {
    @memset(out, 0);
    var total: u64 = 0;
    for (0..cycles) |c| {
        for (ids[c][s][0..n_ids[c][s]]) |e| {
            if (e < out.len) out[e] += 1;
            total += 1;
        }
    }
    return total;
}

/// An online LRU of `cap` experts over all stages' experts (global id = stage x n_experts + id): each cycle's distinct
/// ids are its working set; one not resident is a miss (read into a cold slot), then every id of the set is made
/// most recent, evicting the least recent resident outside the set.
pub const Lru = struct { cap: u32, misses: u64 = 0, max_misses_cycle: u32 = 0, cycles_over_cold: u32 = 0 };

pub const Totals = struct { touched: u64 = 0, distinct_ever: u32 = 0, max_set: u32 = 0 };

pub fn replay(lrus: []Lru) Totals {
    var tot: Totals = .{};
    const n_global = geo.n_stages * geo.n_experts;
    var ever: std.StaticBitSet(max_stages * max_experts) = .empty;
    for (lrus) |*l| {
        var last: [max_stages * max_experts]u32 = @splat(0);
        var res: std.StaticBitSet(max_stages * max_experts) = .empty;
        var n_res: u32 = 0;
        l.misses = 0;
        l.max_misses_cycle = 0;
        l.cycles_over_cold = 0;
        for (0..cycles) |c| {
            const stamp: u32 = @intCast(c + 1);
            var set: std.StaticBitSet(max_stages * max_experts) = .empty;
            for (0..geo.n_stages) |s| {
                for (ids[c][s][0..n_ids[c][s]]) |e| set.set(s * geo.n_experts + e);
            }
            var miss: u32 = 0;
            var it = set.iterator(.{});
            while (it.next()) |x| {
                if (!res.isSet(x)) miss += 1;
                last[x] = stamp;
            }
            it = set.iterator(.{});
            while (it.next()) |x| {
                if (res.isSet(x)) continue;
                if (n_res == l.cap) {
                    // the least recent resident outside this cycle's set (the set fits: cap >= the set)
                    var victim: ?usize = null;
                    var rit = res.iterator(.{});
                    while (rit.next()) |r| {
                        if (last[r] == stamp) continue;
                        if (victim == null or last[r] < last[victim.?]) victim = r;
                    }
                    res.unset(victim orelse break);
                    n_res -= 1;
                }
                res.set(x);
                n_res += 1;
            }
            l.misses += miss;
            l.max_misses_cycle = @max(l.max_misses_cycle, miss);
            l.cycles_over_cold += @intFromBool(miss > cold_slots);
            if (l == &lrus[0]) {
                tot.touched += set.count();
                tot.max_set = @max(tot.max_set, @as(u32, @intCast(set.count())));
                ever.setUnion(set);
            }
        }
    }
    tot.distinct_ever = @intCast(ever.count());
    std.debug.assert(n_global <= max_stages * max_experts);
    return tot;
}

/// `DRAFT_ROUTE_HIST {...}` for stage `s`: its per-expert counts, routed total and distinct experts.
pub fn histLine(buf: []u8, s: usize) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    writeHist(&w, s) catch return buf[0..0];
    return w.buffered();
}

fn writeHist(w: *std.Io.Writer, s: usize) !void {
    var cs: [max_experts]u32 = undefined;
    const total = counts(s, cs[0..geo.n_experts]);
    var distinct: u32 = 0;
    for (cs[0..geo.n_experts]) |x| distinct += @intFromBool(x > 0);
    var max_cycle: u32 = 0;
    for (0..cycles) |c| max_cycle = @max(max_cycle, n_ids[c][s]);
    try w.print("DRAFT_ROUTE_HIST {{\"stage\": {d}, \"cycles\": {d}, \"experts\": {d}, \"top_k\": {d}, \"block\": {d}, \"routed\": {d}, \"distinct\": {d}, \"max_ids_per_cycle\": {d}, \"counts\": [", .{ s, cycles, geo.n_experts, geo.top_k, geo.block, total, distinct, max_cycle });
    for (cs[0..geo.n_experts], 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", x });
    try w.writeAll("]}");
}

/// `DRAFT_ROUTE_LRU {...}`: the online LRU replay at each hot-set capacity (all stages' experts), against the cold
/// cache's slots.
pub fn lruLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    writeLru(&w) catch return buf[0..0];
    return w.buffered();
}

fn writeLru(w: *std.Io.Writer) !void {
    var lrus: [lru_capacities.len]Lru = undefined;
    for (&lrus, lru_capacities) |*l, cap| l.* = .{ .cap = cap };
    const tot = replay(&lrus);
    const n: f64 = @floatFromInt(@max(cycles, 1));
    try w.print("DRAFT_ROUTE_LRU {{\"cycles\": {d}, \"stages\": {d}, \"experts\": {d}, \"touched\": {d}, \"touched_per_cycle\": {d:.2}, \"max_set\": {d}, \"distinct_ever\": {d}, \"cold_slots\": {d}, \"lru\": [", .{ cycles, geo.n_stages, geo.n_stages * geo.n_experts, tot.touched, @as(f64, @floatFromInt(tot.touched)) / n, tot.max_set, tot.distinct_ever, cold_slots });
    for (lrus, 0..) |l, i| {
        const p: f64 = @as(f64, @floatFromInt(l.misses)) / @as(f64, @floatFromInt(@max(tot.touched, 1)));
        try w.print("{s}{{\"h\": {d}, \"misses\": {d}, \"misses_per_cycle\": {d:.3}, \"p_cold\": {d:.4}, \"max_misses_cycle\": {d}, \"cycles_over_cold\": {d}}}", .{ if (i == 0) "" else ", ", l.cap, l.misses, @as(f64, @floatFromInt(l.misses)) / n, p, l.max_misses_cycle, l.cycles_over_cold });
    }
    try w.writeAll("]}");
}

/// The receipt's `draft_route_stream`: per cycle, each stage's ids in routed order (row-major `[rows, top_k]`).
pub fn writeStreamJson(w: *std.Io.Writer) !void {
    try w.print("{{\"stages\": {d}, \"experts_per_stage\": {d}, \"top_k\": {d}, \"block\": {d}, \"global_id\": \"stage * experts_per_stage + id\", \"cycles\": [", .{ geo.n_stages, geo.n_experts, geo.top_k, geo.block });
    for (0..cycles) |c| {
        try w.writeAll(if (c == 0) "[" else ", [");
        for (0..geo.n_stages) |s| {
            try w.writeAll(if (s == 0) "[" else ", [");
            for (ids[c][s][0..n_ids[c][s]], 0..) |e, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", e });
            try w.writeAll("]");
        }
        try w.writeAll("]");
    }
    try w.writeAll("]}");
}

test "dsv41 draft routes: a default build keeps no ids (StageIds void, install a no-op); the bound is geometric" {
    try std.testing.expectEqual(@as(usize, 1024 * 4 * (63 * 2 + 1)), storage_bytes);
    if (enabled) return error.SkipZigTest;
    try std.testing.expect(StageIds(u32) == void);
    try install(.{ .n_stages = 3, .n_experts = 128, .top_k = 3, .block = 5 }, 1024);
    try std.testing.expect(!active);
}

test "dsv41 draft routes (profile builds): hand-computed counts, distinct experts and the LRU replay; install refuses what the storage cannot hold" {
    if (comptime !enabled) return error.SkipZigTest;
    defer uninstall();
    try std.testing.expectError(error.DraftRouteCycles, install(.{ .n_stages = 3, .n_experts = 128, .top_k = 3, .block = 5 }, 1026));
    try std.testing.expectError(error.DraftRouteIds, install(.{ .n_stages = 3, .n_experts = 128, .top_k = 8, .block = 8 }, 1024));
    try std.testing.expectError(error.DraftRouteStages, install(.{ .n_stages = 5, .n_experts = 128, .top_k = 3, .block = 5 }, 1024));
    try install(.{ .n_stages = 2, .n_experts = 4, .top_k = 2, .block = 1 }, 8);
    try std.testing.expect(active);
    // three cycles; global ids are stage x 4 + id
    record(&.{ &.{ 0, 1 }, &.{ 2, 3 } }); // {0, 1, 6, 7}
    record(&.{ &.{ 1, 2 }, &.{ 3, 3 } }); // {1, 2, 7}
    record(&.{ &.{ 0, 3 }, &.{ 0, 2 } }); // {0, 3, 4, 6}
    var cs: [4]u32 = undefined;
    try std.testing.expectEqual(@as(u64, 6), counts(0, &cs));
    try std.testing.expectEqualSlices(u32, &.{ 2, 2, 1, 1 }, &cs);
    try std.testing.expectEqual(@as(u64, 6), counts(1, &cs));
    try std.testing.expectEqualSlices(u32, &.{ 1, 0, 2, 3 }, &cs);
    // LRU of 4: c1 misses 4 {0,1,6,7}; c2 misses 1 (2), evicts 0 (stamp 1, outside {1,2,7}; 6 also stamp 1, 0 found first)
    // -> {1,2,6,7}; c3 misses 0, 3, 4 (6 resident) = 3. LRU of 8 holds all: first touches only, 4 + 1 + 2 = 7.
    var lrus = [_]Lru{ .{ .cap = 4 }, .{ .cap = 8 } };
    const tot = replay(&lrus);
    try std.testing.expectEqual(@as(u64, 4 + 3 + 4), tot.touched);
    try std.testing.expectEqual(@as(u32, 7), tot.distinct_ever);
    try std.testing.expectEqual(@as(u64, 4 + 1 + 3), lrus[0].misses);
    try std.testing.expectEqual(@as(u64, 7), lrus[1].misses);
    try std.testing.expectEqual(@as(u32, 4), lrus[0].max_misses_cycle);
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqualStrings("DRAFT_ROUTE_HIST {\"stage\": 1, \"cycles\": 3, \"experts\": 4, \"top_k\": 2, \"block\": 1, \"routed\": 6, \"distinct\": 3, \"max_ids_per_cycle\": 2, \"counts\": [1, 0, 2, 3]}", histLine(&buf, 1));
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    try writeStreamJson(&aw.writer);
    try std.testing.expectEqualStrings("{\"stages\": 2, \"experts_per_stage\": 4, \"top_k\": 2, \"block\": 1, \"global_id\": \"stage * experts_per_stage + id\", \"cycles\": [[[0,1], [2,3]], [[1,2], [3,3]], [[0,3], [0,2]]]}", aw.written());
}
