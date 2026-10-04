//! sdk_ext.expert.slot_cache's tests (host rows or no memory; never MLX), in the unit tests beside the read pool they drive.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const xsc = sdk_ext.expert.slot_cache;
const expert_io = sdk_ext.expert.io;
const expert_policy = sdk_ext.expert.policy;
const Pair = expert_io.Records(2, 1);
const Component = xsc.Component;
const Geometry = xsc.Geometry;
const Memory = xsc.Memory;
const Cache = xsc.Cache;
const Loc = xsc.Loc;
const Error = xsc.Error;
const alloc_page_bytes = xsc.alloc_page_bytes;
const max_components = xsc.max_components;

// ── Tests (host: host rows or no memory; never MLX) ──

const testing = std.testing;

/// A file of `n_groups x n_experts` records, each part at a shuffled offset, bytes a function of the offset.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    path: [:0]u8,
    offsets: []u64,

    fn init(a: std.mem.Allocator, n_groups: usize, n_experts: usize, comps: []const Component, seed_: u64) !Fixture {
        const n_parts = n_groups * n_experts * comps.len;
        const order = try a.alloc(usize, n_parts);
        defer a.free(order);
        for (order, 0..) |*o, i| o.* = i;
        var prng = std.Random.DefaultPrng.init(seed_);
        prng.random().shuffle(usize, order);
        const offsets = try a.alloc(u64, n_parts);
        errdefer a.free(offsets);
        var at: u64 = 123; // an unaligned header
        for (order) |i| {
            offsets[i] = at;
            at += comps[i % comps.len].bytes;
        }
        const image = try a.alloc(u8, @intCast(at + 77));
        errdefer a.free(image);
        for (image, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard.safetensors", .data = image });
        var root: [512]u8 = undefined;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/shard.safetensors", .{root[0..try tmp.dir.realPath(std.testing.io, &root)]}, 0);
        return .{ .tmp = tmp, .image = image, .path = path, .offsets = offsets };
    }

    fn deinit(f: *Fixture, a: std.mem.Allocator) void {
        a.free(f.offsets);
        a.free(f.image);
        a.free(f.path);
        f.tmp.cleanup();
    }

    fn part(f: *const Fixture, comps: []const Component, n_experts: usize, group: usize, expert: usize, k: usize) []const u8 {
        const off = f.offsets[(group * n_experts + expert) * comps.len + k];
        return f.image[off..][0..comps[k].bytes];
    }
};

const test_comps = [_]Component{
    .{ .bytes = 3 * 4096 + 17, .shape = &.{ 3, 4113 }, .dtype = .uint8 },
    .{ .bytes = 513, .shape = &.{513}, .dtype = .uint8 },
    .{ .bytes = 4096, .shape = &.{ 2, 512 }, .dtype = .uint32 },
    .{ .bytes = 99, .shape = &.{99}, .dtype = .uint8 },
};

test "dsv41 slot cache: every routed id's slot holds its record's bytes through evictions, the default bank's input row for row" {
    const a = testing.allocator;
    const n_groups = 2;
    const n_experts = 12;
    var fx = try Fixture.init(a, n_groups, n_experts, &test_comps, 7);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 3, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 256 });
    defer pool.stop();
    const geom: Geometry = .{ .n_experts = n_experts, .components = &test_comps, .capacity = &.{ 2, 3 }, .transient = 6 };
    const cache = try Cache.init(a, geom, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..n_groups) |s| for (0..n_experts) |e| for (0..test_comps.len) |k| {
        cache.setLoc(s, e, k, .{ .file = f, .offset = fx.offsets[(s * n_experts + e) * test_comps.len + k] });
    };
    try cache.checkLocs();
    // The construction seed: the first ids into the persistent slots.
    try testing.expectEqual(@as(u32, 2), try cache.seed(0, &.{ 0, 1 }));
    try testing.expectEqual(@as(u32, 3), try cache.seed(1, &.{ 0, 1, 2 }));
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    var evictions: u64 = 0;
    for (0..60) |round| {
        const s = round % n_groups;
        var ids: [6]u16 = undefined; // 2 rows x top-3, repeats across rows allowed
        for (&ids, 0..) |*d, i| {
            d.* = rnd.uintLessThan(u16, n_experts);
            if (i % 3 != 0) while (d.* == ids[i - 1] or (i % 3 == 2 and d.* == ids[i - 2])) {
                d.* = rnd.uintLessThan(u16, n_experts);
            };
        }
        var slots: [6]u32 = undefined;
        const ev0 = cache.stats.expert_cache_evictions;
        try cache.route(s, &ids, &slots);
        evictions += cache.stats.expert_cache_evictions - ev0;
        // The default route's kernel reads bank[id]; this route's reads bank[slot]: the same bytes, every part.
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| {
            try testing.expectEqualSlices(u8, fx.part(&test_comps, n_experts, s, e, k), cache.row(s, k, slot));
        };
    }
    try testing.expect(evictions > 0);
    const st = cache.stats;
    try testing.expectEqual(@as(u64, 60), st.route_calls);
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    var per_load: u64 = 0;
    for (test_comps) |c| per_load += c.bytes;
    try testing.expectEqual((5 + st.expert_cache_misses) * per_load, st.expert_bytes_read);
}

test "dsv41 slot cache: geometry and placement refused by name; a failed read forgets its loads" {
    const a = testing.allocator;
    try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{3}, .transient = 6 }, .none, null));
    try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 8, .components = test_comps[0..3], .capacity = &.{1}, .transient = 2 }, .none, null));
    try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{1}, .transient = 2 }, .host, null));
    var fx = try Fixture.init(a, 1, 4, &test_comps, 3);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 64 });
    defer pool.stop();
    const cache = try Cache.init(a, .{ .n_experts = 4, .components = &test_comps, .capacity = &.{1}, .transient = 3 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    try testing.expectError(error.CacheLocation, cache.checkLocs());
    for (0..4) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    cache.setLoc(0, 3, 1, .{ .file = f, .offset = fx.image.len });
    try testing.expectError(error.CacheLocation, cache.checkLocs());
    // A part past the end of the file reads short: the call fails by name and its persistent load is forgotten.
    var slots: [1]u32 = undefined;
    try testing.expectError(error.ReadFailed, cache.route(0, &.{3}, &slots));
    try testing.expect(cache.slotOf(0, 3) == null);
}

test "dsv41 slot cache: no memory plans only (the trace backend), with the policy's slots and the bill's bytes" {
    const a = testing.allocator;
    const comps = [_]Component{
        .{ .bytes = 5_898_240, .shape = &.{ 2304, 640 }, .dtype = .uint32 },
        .{ .bytes = 368_640, .shape = &.{ 2304, 160 }, .dtype = .uint8 },
    };
    const cache = try Cache.init(a, .{ .n_experts = 128, .components = &comps, .capacity = &.{ 43, 42 }, .transient = 15 }, .none, null);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 3), try cache.seed(0, &.{ 0, 1, 2 }));
    var slots: [3]u32 = undefined;
    try cache.route(0, &.{ 1, 9, 2 }, &slots);
    try testing.expectEqual(@as(u32, 1), slots[0]);
    try testing.expectEqual(@as(u32, 2), slots[2]);
    try testing.expectEqual(@as(u64, 0), cache.stats.expert_bytes_read);
    // 58 + 57 rows: weights a page multiple, scales rounded up to the page per array.
    try testing.expectEqual(@as(u64, 115 * 5_898_240 + std.mem.alignForward(u64, 58 * 368_640, 16384) + std.mem.alignForward(u64, 57 * 368_640, 16384)), cache.geom.billBytes());
}

test "dsv41 slot cache: one group shared by consecutive routes reuses a row the previous route served as early as the policy allows; every route's rows are its records" {
    // Three "stages" of 12 experts in one group (global id = stage x 12 + e), 2 hot + 6 transient rows: each route is a
    // stage's 2 rows x top-3; stage s + 1 routes right after stage s, the next block's stage 0 right after stage 2.
    const a = testing.allocator;
    const n_stages = 3;
    const per = 12;
    var fx = try Fixture.init(a, 1, n_stages * per, &test_comps, 41);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 3, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 256 });
    defer pool.stop();
    const cache = try Cache.init(a, .{ .n_experts = n_stages * per, .components = &test_comps, .capacity = &.{2}, .transient = 6 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..n_stages * per) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    _ = try cache.seed(0, &.{ 0, 12 });
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var prev: [6]u32 = undefined;
    var prev_ids: [6]u16 = undefined;
    var have_prev = false;
    var reuse_in_block: u32 = 0;
    var reuse_across: u32 = 0;
    for (0..20) |blk| for (0..n_stages) |st| {
        var ids: [6]u16 = undefined;
        for (&ids, 0..) |*d, i| {
            while (true) {
                d.* = @intCast(st * per + rnd.uintLessThan(u16, per));
                if (std.mem.indexOfScalar(u16, ids[i - i % 3 .. i], d.*) == null) break;
            }
        }
        var slots: [6]u32 = undefined;
        try cache.route(0, &ids, &slots);
        // The kernel's input at this route's evaluation (before the next route plans): its records' bytes, row for row.
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| {
            try testing.expectEqualSlices(u8, fx.part(&test_comps, n_stages * per, 0, e, k), cache.row(0, k, slot));
        };
        // A row the previous route read now holds another record: the earliest reuse there is.
        if (have_prev) for (slots, ids) |slot, e| for (prev, prev_ids) |ps, pe| if (slot == ps and e != pe) {
            if (st == 0) reuse_across += 1 else reuse_in_block += 1;
        };
        _ = blk;
        prev = slots;
        prev_ids = ids;
        have_prev = true;
    };
    try testing.expect(reuse_in_block > 0 and reuse_across > 0);
}

test "dsv41 slot cache: a read it cannot see land latches the cache: no row of that call is planned again, even after the late write" {
    const a = testing.allocator;
    var fx = try Fixture.init(a, 1, 8, &test_comps, 19);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 64, .aux_tickets = 32 });
    defer pool.stop();
    defer expert_io.clearFaults();
    const cache = try Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{2}, .transient = 3 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..8) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    cache.wait_ns = 30 * std.time.ns_per_ms;
    // Expert 5's second pair (its down span) sleeps 300 ms before its read: one ticket of a multi-ticket call is late.
    const page = std.heap.pageSize();
    const late = fx.offsets[5 * test_comps.len + 3];
    expert_io.injectFault(late / page * page, 5, 300 * std.time.ns_per_ms);
    var slots: [3]u32 = undefined;
    try testing.expectError(error.Timeout, cache.route(0, &.{ 4, 5, 6 }, &slots));
    try testing.expect(cache.failed);
    // Refused by name from here on, before and after the late ticket publishes.
    try testing.expectError(error.CacheFailed, cache.route(0, &.{ 4, 5, 6 }, &slots));
    try testing.expectError(error.CacheFailed, cache.seed(0, &.{1}));
    std.Io.sleep(testing.io, .fromMilliseconds(400), .awake) catch {};
    try testing.expectError(error.CacheFailed, cache.route(0, &.{1}, slots[0..1]));
}

test "dsv41 slot cache: on a pool with an aux ring every read takes the aux tickets; the demand ring is untouched" {
    const a = testing.allocator;
    var fx = try Fixture.init(a, 1, 8, &test_comps, 23);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 96, .aux_tickets = 32 });
    defer pool.stop();
    try testing.expectEqual(@as(u32, 64), pool.demand_tickets);
    try testing.expectEqual(@as(u32, 64), pool.aux_first);
    const cache = try Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{2}, .transient = 6 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..8) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    var slots: [6]u32 = undefined;
    // 6 loads x 2 pairs = 24 tickets per route, the aux ring 32: the ring wraps inside [64, 96) across routes.
    for (0..5) |r| {
        const ids = [_]u16{ @intCast(r % 8), @intCast((r + 1) % 8), @intCast((r + 2) % 8), @intCast((r + 3) % 8), @intCast((r + 4) % 8), @intCast((r + 5) % 8) };
        try cache.route(0, &ids, &slots);
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| try testing.expectEqualSlices(u8, fx.part(&test_comps, 8, 0, e, k), cache.row(0, k, slot));
        try testing.expect(pool.next_aux >= pool.aux_first and pool.next_aux <= pool.aux_end);
    }
    try testing.expectEqual(@as(u32, 0), pool.next_ticket);
    // The stream's demand submits stay below the aux ring.
    var buf: [8192]u8 = undefined;
    const dests: Pair.Rows = .{ @intFromPtr(&buf), @intFromPtr(&buf) + 4096 };
    const lens: Pair.Rows = .{ 100, 100 };
    for (0..40) |_| {
        const t = try Pair.submit(pool, cache.files.items[0], &.{0}, &.{200}, &.{dests}, &lens);
        try testing.expect(t + 2 <= pool.demand_tickets);
        try pool.wait(t, 2, 5 * std.time.ns_per_s);
    }
}

test "dsv41 slot cache: the LRU policy keeps every routed id's rows its record's bytes through evictions, every row residency" {
    const a = testing.allocator;
    var fx = try Fixture.init(a, 1, 12, &test_comps, 29);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 3, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 256 });
    defer pool.stop();
    const cache = try Cache.init(a, .{ .n_experts = 12, .policy = .lru, .components = &test_comps, .capacity = &.{2}, .transient = 6 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..12) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    var prng = std.Random.DefaultPrng.init(31);
    const rnd = prng.random();
    for (0..40) |_| {
        var ids: [6]u16 = undefined;
        for (&ids, 0..) |*d, i| while (true) {
            d.* = rnd.uintLessThan(u16, 12);
            if (std.mem.indexOfScalar(u16, ids[i - i % 3 .. i], d.*) == null) break;
        };
        var slots: [6]u32 = undefined;
        try cache.route(0, &ids, &slots);
        for (ids, slots) |e, slot| for (0..test_comps.len) |k| try testing.expectEqualSlices(u8, fx.part(&test_comps, 12, 0, e, k), cache.row(0, k, slot));
    }
    try testing.expect(cache.stats.expert_cache_evictions > 0);
    // Every load is persistent: the 8 rows (2 + 6) all hold residency.
    try testing.expectEqual(@as(u64, 0), cache.stats.transient_loads);
    try testing.expectEqual(@as(usize, 8), cache.policies[0].residents().len);
}

fn cacheInitDeinit(a: std.mem.Allocator, policy: xsc.PolicyKind) !void {
    const cache = try Cache.init(a, .{ .n_experts = 12, .policy = policy, .components = &test_comps, .capacity = &.{ 2, 3 }, .transient = 6 }, .none, null);
    cache.deinit();
}

test "dsv41 slot cache: the geometry's every bound and a missing file are refused by name" {
    const a = testing.allocator;
    var ten: [10]Component = undefined;
    for (&ten) |*c| c.* = test_comps[0];
    const Bad = struct { comps: []const Component, cap: []const u32, transient: u32 };
    for ([_]Bad{
        .{ .comps = &test_comps, .cap = &.{}, .transient = 2 }, // no group
        .{ .comps = &.{}, .cap = &.{1}, .transient = 2 }, // no component
        .{ .comps = &ten, .cap = &.{1}, .transient = 2 }, // past max_components
        .{ .comps = &test_comps, .cap = &.{1}, .transient = 0 }, // no transient row
        .{ .comps = &test_comps, .cap = &.{1}, .transient = expert_policy.max_route_ids + 1 },
    }) |b| try testing.expectError(error.CacheGeometry, Cache.init(a, .{ .n_experts = 64, .components = b.comps, .capacity = b.cap, .transient = b.transient }, .none, null));
    const cache = try Cache.init(a, .{ .n_experts = 8, .components = &test_comps, .capacity = &.{1}, .transient = 2 }, .none, null);
    defer cache.deinit();
    try testing.expectError(error.NotFound, cache.openFile("/nonexistent/cov-d/shard.safetensors"));
    // The trace backend (no memory) seeds and routes without reading.
    try testing.expectEqual(@as(u32, 1), try cache.seed(0, &.{ 5, 6 }));
    var slots: [2]u32 = undefined;
    try cache.route(0, &.{ 5, 7 }, &slots);
    try testing.expectEqual(@as(?u32, 0), cache.slotOf(0, 5));
    try testing.expectEqual(@as(u64, 0), cache.stats.expert_bytes_read);
}

test "dsv41 slot cache: a seed whose read fails forgets what it admitted; the route after it reads again" {
    const a = testing.allocator;
    var fx = try Fixture.init(a, 1, 6, &test_comps, 9);
    defer fx.deinit(a);
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .staging_bytes = 4 * std.heap.pageSize(), .tickets = 64 });
    defer pool.stop();
    const cache = try Cache.init(a, .{ .n_experts = 6, .components = &test_comps, .capacity = &.{2}, .transient = 4 }, .host, pool);
    defer cache.deinit();
    const f = try cache.openFile(fx.path);
    for (0..6) |e| for (0..test_comps.len) |k| cache.setLoc(0, e, k, .{ .file = f, .offset = fx.offsets[e * test_comps.len + k] });
    try cache.checkLocs();
    // The file is emptied under the open descriptor: every read finds its end at once (short).
    try testing.expectEqual(@as(c_int, 0), c_truncate(fx.path.ptr, 0));
    try testing.expectError(error.ReadFailed, cache.seed(0, &.{ 1, 2 }));
    try testing.expectEqual(@as(?u32, null), cache.slotOf(0, 1));
    try testing.expectEqual(@as(?u32, null), cache.slotOf(0, 2));
    // A failed read is not a latched cache (it saw every job land): with the file whole again the same ids read.
    try tmpRewrite(&fx);
    try testing.expectEqual(@as(u32, 2), try cache.seed(0, &.{ 1, 2 }));
    for ([_]u16{ 1, 2 }) |e| for (0..test_comps.len) |k| {
        try testing.expectEqualSlices(u8, fx.part(&test_comps, 6, 0, e, k), cache.row(0, k, cache.slotOf(0, e).?));
    };
    // forgetAll: nothing resident until read again.
    try cache.forgetAll();
    try testing.expectEqual(@as(?u32, null), cache.slotOf(0, 1));
}

extern "c" fn truncate(path: [*:0]const u8, len: i64) c_int;
const c_truncate = truncate;

fn tmpRewrite(fx: *Fixture) !void {
    const fd = std.c.open(fx.path.ptr, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var done: usize = 0;
    while (done < fx.image.len) {
        const n = std.c.pwrite(fd, fx.image[done..].ptr, fx.image.len - done, @intCast(done));
        if (n <= 0) return error.WriteFailed;
        done += @intCast(n);
    }
}

test "dsv41 slot cache: every allocation failure of the cache's construction unwinds as OutOfMemory (both planners)" {
    try std.testing.checkAllAllocationFailures(testing.allocator, cacheInitDeinit, .{.shipped});
    try std.testing.checkAllAllocationFailures(testing.allocator, cacheInitDeinit, .{.lru});
}
