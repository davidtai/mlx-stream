//! The reader's tests (`sdk_ext.expert.io`), kept in the package's test root under their names: the C pool and its
//! fault injection compile into the test binary, never into the SDK's own tests.

const std = @import("std");
const expert_bank = @import("expert_bank.zig");
const io = @import("sdk_ext.zig").expert.io;
const c = io.test_abi.abi;
const Pool = io.Pool;
const Status = io.Status;
const Spec = io.Spec;
const Warm = io.Warm;
const Sched = io.Sched;
const max_items = io.max_items;
const max_spec = io.max_spec;
const max_pre = io.max_pre;
const max_gate_tickets = io.max_gate_tickets;
const res_w = io.test_abi.status_words;
const spec_state_w = io.spec_state_w;
const pre_state_w = io.pre_state_w;
const abi_version = io.abi_version;
const slotBytes = io.slotBytes;
const chunkBytes = io.chunkBytes;
const injectFault = io.injectFault;
const injectFaults = io.injectFaults;
const clearFaults = io.clearFaults;
const n_components = expert_bank.n_components;
const gu_components = expert_bank.gu_components;
/// EXL3's record topology (the bank of record's).
const R96 = io.Records(n_components, gu_components);

const testing = std.testing;

/// A temp file whose byte at offset o is a function of o, plus its image.
const PatternFile = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    ufd: io.UncachedFd,

    fn init(len: usize) !PatternFile {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try testing.allocator.alloc(u8, len);
        errdefer testing.allocator.free(image);
        expert_bank.fillPattern(image, 11);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "pattern.bin", .data = image });
        var root: [512]u8 = undefined;
        var pbuf: [600]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/pattern.bin", .{root[0..try tmp.dir.realPath(std.testing.io, &root)]}, 0);
        const ufd = try io.openUncached(path.ptr, null);
        return .{ .tmp = tmp, .image = image, .ufd = ufd };
    }

    fn deinit(self: *PatternFile) void {
        self.ufd.close();
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }
};

/// Nine destination rows per record, `lens[c]` bytes each, in one buffer.
const Dests = struct {
    buf: []u8,
    rows: [max_items][n_components]u64,
    at: [max_items][n_components]usize,

    fn init(n: usize, lens: *const [n_components]u64) !Dests {
        var d: Dests = .{ .buf = undefined, .rows = undefined, .at = undefined };
        var total: usize = 0;
        for (lens.*) |l| total += l;
        d.buf = try testing.allocator.alloc(u8, total * n);
        @memset(d.buf, 0xAA);
        var off: usize = 0;
        for (0..n) |i| for (lens.*, 0..) |l, k| {
            d.at[i][k] = off;
            d.rows[i][k] = @intFromPtr(d.buf.ptr) + off;
            off += l;
        };
        return d;
    }

    fn part(self: *const Dests, i: usize, k: usize, lens: *const [n_components]u64) []const u8 {
        return self.buf[self.at[i][k]..][0..lens[k]];
    }

    /// Record i's nine rows hold the file's bytes at `gu` (six parts) and `down` (three).
    fn expectRecord(self: *const Dests, i: usize, image: []const u8, gu: u64, down: u64, lens: *const [n_components]u64) !void {
        var at = gu;
        for (0..n_components) |k| {
            if (k == gu_components) at = down;
            try testing.expectEqualSlices(u8, image[at..][0..lens[k]], self.part(i, k, lens));
            at += lens[k];
        }
    }
};

test "dsv41 io: pool scatters a synthetic file" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page + 321);
    defer f.deinit();
    // One-page staging: every range needs several aligned reads.
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    // Unequal, page-unaligned parts; gate/up parts contiguous, down parts contiguous.
    const lens = [n_components]u64{ page + 3, 17, 4099, 1, 777, 2 * page + 5, 1000, page - 1, 33 };
    var gu_len: u64 = 0;
    for (lens[0..gu_components]) |l| gu_len += l;
    for ([_]usize{ 1, 3, 8 }, 0..) |n, round| {
        var d = try Dests.init(n, &lens);
        defer testing.allocator.free(d.buf);
        var gu: [max_items]u64 = undefined;
        var down: [max_items]u64 = undefined;
        for (0..n) |i| {
            gu[i] = 7 + round * 13 + i * (5 * page + 11);
            down[i] = 40 * page + 3 + i * (2 * page + 7);
        }
        const seq0 = c.q3ld_seq();
        const first = try R96.submit(pool, f.ufd, gu[0..n], down[0..n], d.rows[0..n], &lens);
        const count: u32 = @intCast(2 * n);
        try pool.wait(first, count, 10 * std.time.ns_per_s);
        for (0..n) |i| {
            try d.expectRecord(i, f.image, gu[i], down[i], &lens);
            const r_gu = pool.result(first + @as(u32, @intCast(i)));
            try testing.expectEqual(Status.ok, r_gu.status);
            try testing.expectEqual(@as(i64, @intCast(gu_len)), r_gu.payload);
            try testing.expect(r_gu.preadv_calls >= @as(i64, @intCast(gu_len / page)));
            try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(n + i))).status);
        }
        // One worker runs the job in ticket order: every gate/up before any down.
        var order: [2 * max_items]u32 = undefined;
        const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
        try testing.expectEqual(@as(usize, count), got.len);
        for (got, 0..) |t, k| try testing.expectEqual(first + @as(u32, @intCast(k)), t);
    }
}

test "dsv41 io: injected faults map to status words" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(40 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    defer clearFaults();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    // Every range starts 500 bytes into its own page, so its first read skips 500.
    const Case = struct { code: i64, arg: i64, hit: enum { gu1, down0 }, want: [4]Status, err: i64 = 0, hit_calls: i64 = 1 };
    const cases = [_]Case{
        // 1 EINTR: retried inside the call, uncounted.
        .{ .code = 1, .arg = 0, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok } },
        // 2 EIO: that range fails with errno, the rest of the job is skipped.
        .{ .code = 2, .arg = 0, .hit = .gu1, .want = .{ .ok, .os_error, .skipped, .skipped }, .err = @intFromEnum(std.posix.E.IO) },
        // 3 zero return: a short read.
        .{ .code = 3, .arg = 0, .hit = .down0, .want = .{ .ok, .ok, .short, .skipped } },
        // 4 truncated past the skip: 200 bytes land, the range continues.
        .{ .code = 4, .arg = 700, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok }, .hit_calls = 2 },
        // 4 truncated inside the skip: no byte past it, the range ends short (never a retry of the same request).
        .{ .code = 4, .arg = 300, .hit = .gu1, .want = .{ .ok, .short, .skipped, .skipped } },
    };
    for (cases, 0..) |cs, round| {
        // Each range on its own page, 500 bytes in, so a rule's aligned offset names one range.
        const base = round * 8 * page;
        const gu = [2]u64{ base + 500, base + 2 * page + 500 };
        const down = [2]u64{ base + 4 * page + 500, base + 6 * page + 500 };
        const hit_off: u64 = switch (cs.hit) {
            .gu1 => base + 2 * page,
            .down0 => base + 4 * page,
        };
        injectFault(hit_off, cs.code, cs.arg);
        var d = try Dests.init(2, &lens);
        defer testing.allocator.free(d.buf);
        const first = try R96.submit(pool, f.ufd, &gu, &down, d.rows[0..2], &lens);
        try pool.wait(first, 4, 10 * std.time.ns_per_s);
        for (cs.want, 0..) |w, k| {
            const r = pool.result(first + @as(u32, @intCast(k)));
            testing.expectEqual(w, r.status) catch |e| {
                std.debug.print("case {d} range {d}\n", .{ round, k });
                return e;
            };
            if (w == .os_error) try testing.expectEqual(cs.err, r.errno);
            if (w == .ok) try testing.expectEqual(@as(i64, if (k < 2) 600 else 150), r.payload);
            const hit_k: usize = if (cs.hit == .gu1) 1 else 2;
            if (k == hit_k and w == .ok) try testing.expectEqual(cs.hit_calls, r.preadv_calls);
        }
        if (cs.want[1] == .ok) try testing.expectEqualSlices(u8, f.image[gu[1]..][0..100], d.part(1, 0, &lens));
    }
}

test "dsv41 io: stop joins and restarts" {
    try testing.expectEqual(@as(i32, abi_version), c.q3ld_abi());
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 });
    try testing.expectError(error.PoolUnavailable, Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 }));
    pool.stop();
    pool = try Pool.start(testing.allocator, .{ .workers = 4, .staging_bytes = 9 << 20, .tickets = 16 });
    pool.stop();
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .staging_bytes = std.heap.pageSize() + 1 }));
    const bad_spec: Spec = .{ .threads = 1, .slots = max_spec + 1, .record_bytes = 100, .chunk_bytes = std.heap.pageSize() };
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .spec = bad_spec }));
}

test "dsv41 io: A0 (a): warm jobs start only with demand idle, after a demand job submitted later, and land a demand read's bytes" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    // One worker: the order is the queue rule's. Warm tickets 16..31 above demand's 0..15.
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
    defer pool.stop();
    defer clearFaults();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    // The first demand job holds the worker 80 ms (its first range sleeps), so everything below queues behind it.
    injectFault(0, 5, 80 * std.time.ns_per_ms);
    var d = try Dests.init(4, &lens);
    defer testing.allocator.free(d.buf);
    const seq0 = c.q3ld_seq();
    const j0 = try R96.submit(pool, f.ufd, &.{500}, &.{700}, d.rows[0..1], &lens);
    const w0 = try R96.submitWarm(pool, f.ufd, &.{2 * page + 500}, &.{2 * page + 700}, d.rows[1..2], &lens);
    const w1 = try R96.submitWarm(pool, f.ufd, &.{4 * page + 500}, &.{4 * page + 700}, d.rows[2..3], &lens);
    const j1 = try R96.submit(pool, f.ufd, &.{6 * page + 500}, &.{6 * page + 700}, d.rows[3..4], &lens);
    try testing.expect(w0 >= 16 and w1 == w0 + 2 and j1 == j0 + 2 and j1 < 16);
    try pool.wait(j0, 2, 10 * std.time.ns_per_s);
    try pool.wait(j1, 2, 10 * std.time.ns_per_s);
    try pool.wait(w0, 4, 10 * std.time.ns_per_s);
    // Completion order: the held job, the demand job submitted after the warm ones, then the warm jobs.
    var order: [8]u32 = undefined;
    const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
    try testing.expectEqualSlices(u32, &.{ j0, j0 + 1, j1, j1 + 1, w0, w0 + 1, w1, w1 + 1 }, got);
    for ([_]u64{ 500, 2 * page + 500, 4 * page + 500, 6 * page + 500 }, 0..) |gu, i| try d.expectRecord(i, f.image, gu, gu + 200, &lens);
    try testing.expectEqual(@as(i64, 2), pool.counter(.warm_submitted));
    try testing.expectEqual(@as(i64, 2), pool.counter(.warm_started));
    try testing.expectEqual(@as(i64, 0), pool.counter(.warm_cancelled));
    try testing.expectEqual(@as(i64, 0), pool.counter(.warm_max_busy_at_start));
}

test "dsv41 io: A0 (a): a warm cancel publishes the queued jobs skipped at once and lets a started one land" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
    defer pool.stop();
    defer clearFaults();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    var d = try Dests.init(3, &lens);
    defer testing.allocator.free(d.buf);
    // The started one: its first range sleeps 80 ms, so the two after it stay queued (one worker).
    injectFault(0, 5, 80 * std.time.ns_per_ms);
    const w0 = try R96.submitWarm(pool, f.ufd, &.{500}, &.{700}, d.rows[0..1], &lens);
    const w1 = try R96.submitWarm(pool, f.ufd, &.{2 * page + 500}, &.{2 * page + 700}, d.rows[1..2], &lens);
    const w2 = try R96.submitWarm(pool, f.ufd, &.{4 * page + 500}, &.{4 * page + 700}, d.rows[2..3], &lens);
    while (pool.counter(.warm_started) == 0) std.Thread.yield() catch {};
    try testing.expectEqual(@as(u32, 4), pool.cancelWarm(w0, w2 + 2 - w0));
    try pool.wait(w1, 4, std.time.ns_per_s);
    for (0..4) |k| try testing.expectEqual(Status.skipped, pool.result(w1 + @as(u32, @intCast(k))).status);
    // The cancelled rows were never written; the started one lands.
    try testing.expect(std.mem.allEqual(u8, d.part(1, 0, &lens), 0xAA) and std.mem.allEqual(u8, d.part(2, 8, &lens), 0xAA));
    try pool.wait(w0, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(w0).status);
    try testing.expectEqual(Status.ok, pool.result(w0 + 1).status);
    try d.expectRecord(0, f.image, 500, 700, &lens);
    try testing.expectEqual(@as(i64, 2), pool.counter(.warm_cancelled));
    try testing.expectEqual(@as(u32, 0), pool.cancelWarm(w0, 2));
}

test "dsv41 io: A0 (a): demand wraps below the warm tickets; a pool stopping with warm jobs queued returns" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    var d = try Dests.init(2, &lens);
    defer testing.allocator.free(d.buf);
    {
        var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
        defer pool.stop();
        // Demand's 16 tickets: eight 1-record jobs, then the ninth wraps to 0, never into the warm tickets.
        for (0..9) |k| {
            const t = try R96.submit(pool, f.ufd, &.{500}, &.{700}, d.rows[0..1], &lens);
            try testing.expectEqual(@as(u32, @intCast((2 * k) % 16)), t);
            try pool.wait(t, 2, 10 * std.time.ns_per_s);
        }
        const w = try R96.submitWarm(pool, f.ufd, &.{500}, &.{700}, d.rows[1..2], &lens);
        try testing.expectEqual(@as(u32, 16), w);
        try pool.wait(w, 2, 10 * std.time.ns_per_s);
    }
    {
        var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 16, .busy_max = 1 } });
        defer clearFaults();
        injectFault(0, 5, 50 * std.time.ns_per_ms);
        _ = try R96.submit(pool, f.ufd, &.{500}, &.{700}, d.rows[0..1], &lens);
        _ = try R96.submitWarm(pool, f.ufd, &.{2 * page + 500}, &.{2 * page + 700}, d.rows[1..2], &lens);
        pool.stop();
    }
    // Refusals at construction: odd or oversized warm tickets, a busy limit of 0 or past the workers.
    for ([_]Warm{ .{ .tickets = 3, .busy_max = 1 }, .{ .tickets = 32, .busy_max = 1 }, .{ .tickets = 8, .busy_max = 0 }, .{ .tickets = 8, .busy_max = 3 } }) |bad|
        try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 32, .warm = bad }));
    // Without the class, a warm submit is refused by name.
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32 });
    defer pool.stop();
    try testing.expectError(error.InvalidJob, R96.submitWarm(pool, f.ufd, &.{500}, &.{700}, d.rows[0..1], &lens));
}

test "dsv41 io: speculative slots and chunks follow the lane's sizes" {
    // q3_lookahead4_candidate.slot_bytes / chunk_bytes on the 3.0 bank's 13,315,584-byte record, 16 KiB pages.
    try testing.expectEqual(@as(u64, 13_352_960), slotBytes(13_315_584, 16384));
    try testing.expectEqual(@as(u64, 3_342_336), chunkBytes(4, 13_315_584, 16384));
    try testing.expectEqual(@as(u64, 13_352_960), chunkBytes(1, 13_315_584, 16384));
    try testing.expectEqual(@as(u64, 49_152), slotBytes(2880, 16384));
    try testing.expectEqual(@as(u64, 16_384), chunkBytes(4, 2880, 16384));
}

// Speculative class. One record = a gate/up range then a down range, back to back.
const spec_lens = [n_components]u64{ 3000, 100, 200, 3000, 100, 200, 3000, 200, 100 };
const spec_gu_len = 6600;
const spec_rec_len = 9900;

fn specPool(workers: u32, slots: u32) !*Pool {
    const page = std.heap.pageSize();
    return Pool.start(testing.allocator, .{ .workers = workers, .staging_bytes = 4 * page, .tickets = 128, .spec = .{
        .threads = 1,
        .slots = slots,
        .record_bytes = spec_rec_len,
        .chunk_bytes = page,
    } });
}

const SpecSlot = struct { state: i64, tag: i64, base: i64, landed: i64, claimed: i64 };

fn specSlots(out: *[max_spec]SpecSlot) []SpecSlot {
    var raw: [spec_state_w * max_spec]i64 = undefined;
    const n: usize = @intCast(c.q3ld_spec_state(&raw));
    for (out[0..n], 0..) |*s, i| {
        const w = raw[i * spec_state_w ..][0..spec_state_w];
        s.* = .{ .state = w[0], .tag = w[1], .base = w[2], .landed = w[3], .claimed = w[6] };
    }
    return out[0..n];
}

/// Polls until `pred` holds (10 s).
fn waitFor(ctx: anytype, comptime pred: fn (@TypeOf(ctx)) bool) !void {
    var t: u32 = 0;
    while (!pred(ctx)) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn landedAt(base: i64) bool {
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == base and s.state == 3) return true;
    return false;
}

test "dsv41 io: a landed speculative record serves the demand read with no preadv" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(32 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    const base: u64 = 3 * page + 100;
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 1, &.{@intCast(base)}, spec_rec_len));
    // A second issue of a live record only refreshes it.
    try testing.expectEqual(@as(u32, 0), try pool.specStep(f.ufd, 1, &.{@intCast(base)}, spec_rec_len));
    try waitFor(@as(i64, @intCast(base)), landedAt);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    for (0..2) |k| {
        const r = pool.result(first + @as(u32, @intCast(k)));
        try testing.expectEqual(Status.ok, r.status);
        try testing.expectEqual(@as(i64, 0), r.preadv_calls);
    }
    try testing.expectEqual(@as(i64, 1), pool.counter(.claimed));
    try testing.expectEqual(@as(i64, 2), pool.counter(.adopt_ranges));
    try testing.expectEqual(@as(i64, spec_rec_len), pool.counter(.adopt_bytes));
    try testing.expectEqual(@as(i64, 1), pool.counter(.refreshed));
    try testing.expect(pool.counter(.spec_bytes) >= spec_rec_len);
}

test "dsv41 io: a queued speculative record is cancelled by the demand read; settle drops the unclaimed" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(2, 4);
    defer pool.stop();
    defer clearFaults();
    // Two slow demand jobs keep both workers busy, so no unclaimed record may start.
    const slow = [2]u64{ 20 * page, 30 * page };
    injectFaults(&.{ @intCast(slow[0]), @intCast(slow[1]) }, &.{ 5, 5 }, &.{ 300 * std.time.ns_per_ms, 300 * std.time.ns_per_ms });
    var busy = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(busy.buf);
    const j0 = try R96.submit(pool, f.ufd, slow[0..1], &.{slow[0] + spec_gu_len}, busy.rows[0..1], &spec_lens);
    const j1 = try R96.submit(pool, f.ufd, slow[1..2], &.{slow[1] + spec_gu_len}, busy.rows[1..2], &spec_lens);
    const a: u64 = 2 * page + 7;
    const b: u64 = 6 * page + 9;
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.ufd, 1, &.{ @intCast(a), @intCast(b) }, spec_rec_len));
    // The demand read of `a` cancels its queued record and reads it itself.
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{a}, &.{a + spec_gu_len}, d.rows[0..1], &spec_lens);
    try testing.expectEqual(@as(i64, 1), pool.counter(.cancelled_by_demand));
    // The next layer call's settle drops `b`, never claimed.
    _ = try pool.specStep(f.ufd, 2, &.{}, spec_rec_len);
    try testing.expectEqual(@as(i64, 1), pool.counter(.expired));
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| try testing.expectEqual(@as(i64, 0), s.state);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try pool.wait(j0, 2, 10 * std.time.ns_per_s);
    try pool.wait(j1, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, a, a + spec_gu_len, &spec_lens);
    try testing.expect(pool.result(first).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 0), pool.counter(.adopt_ranges));
    try testing.expectEqual(@as(i64, 0), pool.counter(.started));
}

fn preCount(p: *Pool) bool {
    return p.counter(.pre_started) >= 4;
}

fn preTwo(p: *Pool) bool {
    return p.counter(.pre_started) >= 2;
}

fn preIdle(p: *Pool) bool {
    _ = p;
    var raw: [pre_state_w * max_pre]i64 = undefined;
    _ = c.q3ld_pre_state(&raw);
    for (0..max_pre) |i| if (raw[i * pre_state_w] != 0) return false;
    return true;
}

test "dsv41 io: pre-read ranges bind to the demand submit; unbound ones expire at the settle" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(4, 2);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    // Bound: two records pre-read (4 ranges, each worker holds one at its first copy), then demanded.
    const recs = [2]u64{ 5 * page + 11, 9 * page + 13 };
    try testing.expectEqual(@as(u32, 4), try R96.preRead(pool, f.ufd, 1, &.{ @intCast(recs[0]), @intCast(recs[1]) }, &spec_lens));
    try waitFor(pool, preCount);
    var d = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &recs, &.{ recs[0] + spec_gu_len, recs[1] + spec_gu_len }, d.rows[0..2], &spec_lens);
    try pool.wait(first, 4, 10 * std.time.ns_per_s);
    for (0..2) |i| try d.expectRecord(i, f.image, recs[i], recs[i] + spec_gu_len, &spec_lens);
    for (0..4) |k| try testing.expect(pool.result(first + @as(u32, @intCast(k))).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_bound));
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_served));
    try testing.expectEqual(@as(i64, 0), pool.counter(.pre_cancelled));
    _ = try pool.specStep(f.ufd, 1, &.{}, spec_rec_len);
    // Unbound: pre-read, never demanded; the call's settle expires both ranges and frees their workers.
    const lone: u64 = 20 * page + 5;
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 2, &.{@intCast(lone)}, &spec_lens));
    try waitFor(pool, preCount6);
    _ = try pool.specStep(f.ufd, 2, &.{}, spec_rec_len);
    try waitFor(pool, preIdle);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_served));
}

fn preCount6(p: *Pool) bool {
    return p.counter(.pre_started) >= 6;
}

// Event gates on a host word. Gate/up spans cross a page, so a down range's
// first preadv never shares an aligned offset with its record's gate/up read.
const ev_lens = [n_components]u64{ 8000, 100, 200, 8000, 100, 200, 3000, 200, 100 };
const ev_gu_len = 16600;

const EvRig = struct {
    f: PatternFile,
    pool: *Pool,
    word: *i64,
    dests: std.ArrayList(Dests) = .empty,
    value: u64 = 0,

    fn init(timeout_ms: i64, workers: u32) !EvRig {
        const page = std.heap.pageSize();
        var f = try PatternFile.init(96 * page);
        errdefer f.deinit();
        const word = try testing.allocator.create(i64);
        errdefer testing.allocator.destroy(word);
        word.* = 0;
        const pool = try Pool.start(testing.allocator, .{ .workers = workers, .staging_bytes = 4 * page, .tickets = 512 });
        errdefer pool.stop();
        try pool.armEvent(.host, @intFromPtr(word), timeout_ms * std.time.ns_per_ms, 0);
        return .{ .f = f, .pool = pool, .word = word };
    }

    fn deinit(self: *EvRig) void {
        // The pool writes the word until it stops.
        self.pool.stop();
        for (self.dests.items) |d| testing.allocator.free(d.buf);
        self.dests.deinit(testing.allocator);
        testing.allocator.destroy(self.word);
        self.f.deinit();
    }

    fn base(r: u64) u64 {
        return 2 * std.heap.pageSize() + r * 3 * std.heap.pageSize() + 17 * r;
    }

    /// One job of records `recs`; returns its first ticket.
    fn job(self: *EvRig, recs: []const u64) !u32 {
        var d = try Dests.init(recs.len, &ev_lens);
        errdefer testing.allocator.free(d.buf);
        var gu: [max_items]u64 = undefined;
        var down: [max_items]u64 = undefined;
        for (recs, 0..) |r, i| {
            gu[i] = base(r);
            down[i] = base(r) + ev_gu_len;
        }
        const first = try R96.submit(self.pool, self.f.ufd, gu[0..recs.len], down[0..recs.len], d.rows[0..recs.len], &ev_lens);
        try self.dests.append(testing.allocator, d);
        return first;
    }

    /// Gates over ticket groups, values continuing from the last one.
    fn register(self: *EvRig, groups: []const []const i64) !void {
        var values: [16]u64 = undefined;
        var counts: [16]i32 = undefined;
        var tickets: [256]i64 = undefined;
        var k: usize = 0;
        for (groups, 0..) |g, i| {
            values[i] = self.value + 1 + i;
            counts[i] = @intCast(g.len);
            @memcpy(tickets[k..][0..g.len], g);
            k += g.len;
        }
        try self.pool.registerGates(values[0..groups.len], counts[0..groups.len], tickets[0..k]);
        self.value += groups.len;
    }

    fn waitWord(self: *EvRig, value: u64, timeout_ms: u32) !void {
        var t: u32 = 0;
        while (@as(u64, @intCast(@atomicLoad(i64, self.word, .acquire))) < value) : (t += 1) {
            if (t > timeout_ms * 2) return error.Timeout;
            std.Io.sleep(std.testing.io, .fromMicroseconds(500), .awake) catch {};
        }
    }
};

test "dsv41 io: event gates hand the satisfied prefix to the event in order" {
    var rig = try EvRig.init(10_000, 2);
    defer rig.deinit();
    var log: [4 * 64]i64 = undefined;
    _ = c.q3ld_test_ev_log(&log, 64);
    defer _ = c.q3ld_test_ev_log(null, -1);
    c.q3ld_test_delay(11, 2 * std.time.ns_per_ms);
    defer c.q3ld_test_delay(0, 0);
    // Three jobs; gate 1 = every gate/up ticket, then one gate per job's down tickets.
    var firsts: [3]u32 = undefined;
    for (&firsts, 0..) |*fst, j| {
        const r: u64 = 2 * j;
        fst.* = try rig.job(&.{ r, r + 1 });
    }
    var gu: [6]i64 = undefined;
    var downs: [3][2]i64 = undefined;
    for (firsts, 0..) |fst, j| {
        gu[2 * j] = fst;
        gu[2 * j + 1] = fst + 1;
        downs[j] = .{ fst + 2, fst + 3 };
    }
    const groups = [_][]const i64{ &gu, &downs[0], &downs[1], &downs[2] };
    try rig.register(&groups);
    try rig.waitWord(4, 10_000);
    for (firsts) |fst| try rig.pool.wait(fst, 4, 10 * std.time.ns_per_s);
    // Every signal value is new and higher, and every gate at or below it had all its tickets published before the call.
    const n: usize = @intCast(c.q3ld_test_ev_log(null, 0));
    try testing.expect(n >= 1);
    var prev: i64 = 0;
    for (0..n) |i| {
        const e = log[4 * i ..][0..4];
        try testing.expect(e[0] > prev);
        prev = e[0];
        for (groups, 1..) |g, v| {
            if (@as(i64, @intCast(v)) > e[0]) break;
            for (g) |t| {
                const r = rig.pool.result(@intCast(t));
                try testing.expect(r.status != .pending and r.t_end_ns <= e[1]);
            }
        }
    }
    try testing.expectEqual(@as(i64, 4), prev);
    for (rig.dests.items, 0..) |d, j| for (0..2) |i| {
        const r: u64 = 2 * j + i;
        try d.expectRecord(i, rig.f.image, EvRig.base(r), EvRig.base(r) + ev_gu_len, &ev_lens);
    };
    try testing.expectEqual(@as(i64, 4), rig.pool.counter(.ev_gates));
    try testing.expectEqual(@as(i64, 0), rig.pool.counter(.ev_wd_forced));
    // Gates whose tickets already landed are handed over inside the registration.
    const late = try rig.job(&.{7});
    try rig.pool.wait(late, 2, 10 * std.time.ns_per_s);
    try rig.register(&.{ &.{late}, &.{late + 1} });
    try testing.expectEqual(@as(i64, 6), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 2), rig.pool.counter(.ev_immediate));
}

test "dsv41 io: a failed read is terminal for its gate" {
    var rig = try EvRig.init(10_000, 2);
    defer rig.deinit();
    defer clearFaults();
    const page = std.heap.pageSize();
    injectFault(EvRig.base(4) / page * page, 2, 0);
    const f0 = try rig.job(&.{ 4, 5 });
    const f1 = try rig.job(&.{6});
    try rig.register(&.{ &.{ f0, f0 + 1, f1 }, &.{ f0 + 2, f0 + 3 }, &.{f1 + 1} });
    try rig.waitWord(3, 10_000);
    try rig.pool.wait(f0, 4, 10 * std.time.ns_per_s);
    try rig.pool.wait(f1, 2, 10 * std.time.ns_per_s);
    const want = [4]Status{ .os_error, .skipped, .skipped, .skipped };
    for (want, 0..) |w, k| try testing.expectEqual(w, rig.pool.result(f0 + @as(u32, @intCast(k))).status);
    try testing.expectEqual(@as(i64, 0), rig.pool.counter(.ev_wd_forced));
}

test "dsv41 io: the watchdog forces a gate whose bytes never land; stop and the host release the rest" {
    var rig = try EvRig.init(100, 2);
    defer rig.deinit();
    defer clearFaults();
    const page = std.heap.pageSize();
    // Record 8's down range is held 600 ms: its gate is forced after ~100 ms, record 9's still waits for its bytes.
    injectFault((EvRig.base(8) + ev_gu_len) / page * page, 5, 600 * std.time.ns_per_ms);
    const f0 = try rig.job(&.{8});
    const f1 = try rig.job(&.{9});
    const t_reg = c.q3ld_monotonic_ns();
    try rig.register(&.{ &.{ f0, f1 }, &.{f0 + 1}, &.{f1 + 1} });
    try rig.waitWord(2, 5_000);
    const forced_ms = @divTrunc(c.q3ld_monotonic_ns() - t_reg, std.time.ns_per_ms);
    try testing.expectEqual(Status.pending, rig.pool.result(f0 + 1).status);
    try testing.expect(forced_ms >= 90 and forced_ms < 450);
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_wd_forced));
    try testing.expectEqual(@as(i64, 2), rig.pool.counter(.ev_wd_last_value));
    try rig.waitWord(3, 5_000);
    try rig.pool.wait(f0, 2, 10 * std.time.ns_per_s);
    // Host release: a gate over a ticket that never publishes is forced by value.
    const phantom: u32 = 500;
    rig.pool.res[phantom * res_w] = @intFromEnum(Status.pending);
    try rig.register(&.{&.{phantom}});
    rig.pool.releaseGates(4);
    try testing.expectEqual(@as(i64, 4), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_host_released));
    // Stop: a live gate is released so no GPU wait outlives the pool.
    const phantom2: u32 = 501;
    rig.pool.res[phantom2 * res_w] = @intFromEnum(Status.pending);
    try rig.register(&.{&.{phantom2}});
    try testing.expectEqual(@as(i64, 4), @atomicLoad(i64, rig.word, .acquire));
    _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
    _ = c.q3ld_stop();
    try testing.expectEqual(@as(i64, 5), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_stop_released));
    rig.pool.res[phantom * res_w] = 0;
    rig.pool.res[phantom2 * res_w] = 0;
}

test "dsv41 io: event-gate refusals" {
    var rig = try EvRig.init(60_000, 1);
    defer rig.deinit();
    try testing.expectError(error.EventRefused, rig.pool.armEvent(.host, @intFromPtr(rig.word), std.time.ns_per_s, 0));
    const f = try rig.job(&.{11});
    try rig.pool.wait(f, 2, 10 * std.time.ns_per_s);
    try rig.register(&.{&.{f}});
    // Not above the last value; not increasing; out-of-range and repeated tickets; too many tickets.
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{1}, &.{0}, &.{}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{ 4, 3 }, &.{ 0, 0 }, &.{}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{1}, &.{512}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{1}, &.{-1}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{2}, &.{ f + 1, f + 1 }));
    var many: [max_gate_tickets + 1]i64 = undefined;
    for (&many, 0..) |*t, i| t.* = @intCast(i);
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{max_gate_tickets + 1}, &many));
    // Nothing was registered by a refused call.
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_calls));
}

test "dsv41 io: the event class refuses a pool without it and bad arguments" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    try testing.expectError(error.GateRefused, pool.registerGates(&.{1}, &.{0}, &.{}));
    var word: i64 = 0;
    try testing.expectError(error.EventRefused, pool.armEvent(.host, @intFromPtr(&word), 999_999, 0));
    try testing.expectError(error.EventRefused, pool.armEvent(.host, 0, std.time.ns_per_s, 0));
    // The pre-read class needs the speculative class.
    try testing.expectError(error.PreReadRefused, R96.armPreRead(pool, &spec_lens));
}

/// The pool threads as the kernel sees them (tests): each thread's QoS class and name.
const ThreadProbe = struct {
    const mach_port_t = u32;
    extern "c" var mach_task_self_: mach_port_t;
    extern "c" fn task_threads(task: mach_port_t, list: *[*]mach_port_t, count: *u32) c_int;
    extern "c" fn pthread_from_mach_thread_np(port: mach_port_t) ?std.c.pthread_t;
    extern "c" fn pthread_get_qos_class_np(t: std.c.pthread_t, qos: *c_uint, rel: *c_int) c_int;
    extern "c" fn pthread_getname_np(t: std.c.pthread_t, name: [*]u8, len: usize) c_int;
    extern "c" fn pthread_self() std.c.pthread_t;

    const user_interactive: c_uint = 0x21;
    const utility: c_uint = 0x11;

    const Seen = struct { demand: u32 = 0, demand_ui: u32 = 0, spec: u32 = 0, spec_utility: u32 = 0, spec_inherited: u32 = 0, watchdog: u32 = 0, watchdog_ui: u32 = 0, named: u32 = 0 };

    fn qosOf(t: std.c.pthread_t) c_uint {
        var q: c_uint = 0;
        var rel: c_int = 0;
        _ = pthread_get_qos_class_np(t, &q, &rel);
        return q;
    }

    /// Every thread named q3ld-* with its QoS; printed one line each (QOSPROBE).
    fn scan(label: []const u8) Seen {
        var list: [*]mach_port_t = undefined;
        var n: u32 = 0;
        var s: Seen = .{};
        if (task_threads(mach_task_self_, &list, &n) != 0) return s;
        for (list[0..n]) |port| {
            const t = pthread_from_mach_thread_np(port) orelse continue;
            var name: [64]u8 = @splat(0);
            _ = pthread_getname_np(t, &name, name.len);
            const nm = std.mem.sliceTo(&name, 0);
            if (!std.mem.startsWith(u8, nm, "q3ld-")) continue;
            const q = qosOf(t);
            std.debug.print("QOSPROBE {s}: \"{s}\" qos 0x{x}\n", .{ label, nm, q });
            s.named += 1;
            if (std.mem.startsWith(u8, nm, "q3ld-demand-")) {
                s.demand += 1;
                s.demand_ui += @intFromBool(q == user_interactive);
            } else if (std.mem.startsWith(u8, nm, "q3ld-spec-")) {
                s.spec += 1;
                s.spec_utility += @intFromBool(q == utility);
                s.spec_inherited += @intFromBool(q == qosOf(pthread_self()));
            } else if (std.mem.eql(u8, nm, "q3ld-watchdog")) {
                s.watchdog += 1;
                s.watchdog_ui += @intFromBool(q == user_interactive);
            }
        }
        return s;
    }
};

test "dsv41 io: the reader scheduling sets each pool thread's QoS and name at start; off leaves them unnamed, inherited" {
    // The bank sweep (DSV41_TEST_READER_SCHED) runs every pool at one value: this test reads all three itself.
    if (std.c.getenv("DSV41_TEST_READER_SCHED") != null) return error.SkipZigTest;
    const page = std.heap.pageSize();
    for ([_]Sched{ .{}, .{ .qos = true }, .{ .qos = true, .spin = true }, .{ .demand_first = true }, .{ .qos = true, .spin = true, .demand_first = true }, .{ .qos_demand = true } }) |sched| {
        var nb: [Sched.name_len]u8 = undefined;
        const label = sched.name(&nb);
        var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64, .sched = sched, .spec = .{ .threads = 1, .slots = 1, .record_bytes = page, .chunk_bytes = page } });
        defer pool.stop();
        var word: i64 align(8) = 0;
        try pool.armEvent(.host, @intFromPtr(&word), std.time.ns_per_s, 0);
        // Each thread sets its class and name as its first act: poll until all four show (at most 1 s).
        var seen: ThreadProbe.Seen = .{};
        var tries: u32 = 0;
        while (tries < 200) : (tries += 1) {
            seen = ThreadProbe.scan(label);
            if (!(sched.qos or sched.qos_demand) or seen.named == 4) break;
            std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
        }
        std.debug.print("QOSPROBE {s}: self qos 0x{x}; demand {d} (UI {d}), spec {d} (UTILITY {d}), watchdog {d} (UI {d})\n", .{ label, ThreadProbe.qosOf(ThreadProbe.pthread_self()), seen.demand, seen.demand_ui, seen.spec, seen.spec_utility, seen.watchdog, seen.watchdog_ui });
        if (sched.qos_demand) {
            // qosdemand: demand + watchdog raised, the speculative worker at the creating thread's class (no UTILITY)
            try testing.expectEqual(@as(u32, 2), seen.demand_ui);
            try testing.expectEqual(@as(u32, 0), seen.spec_utility);
            try testing.expectEqual(@as(u32, 1), seen.spec_inherited);
            try testing.expectEqual(@as(u32, 1), seen.watchdog_ui);
            try testing.expectEqual(@as(u32, 4), seen.named);
        } else if (!sched.qos) {
            try testing.expectEqual(@as(u32, 0), seen.named);
        } else {
            try testing.expectEqual(@as(u32, 2), seen.demand_ui);
            try testing.expectEqual(@as(u32, 1), seen.spec_utility);
            try testing.expectEqual(@as(u32, 1), seen.watchdog_ui);
            try testing.expectEqual(@as(u32, 4), seen.named);
        }
    }
}

test "dsv41 io: keepwarm: the spinner runs only while switched on; a pool without it refuses the switch" {
    var nb: [Sched.name_len]u8 = undefined;
    try testing.expectEqual(@as(i32, 32), (Sched.parse("keepwarm").?).bits());
    try testing.expectEqualStrings("qos,demandfirst,keepwarm", (Sched.parse("keepwarm,demandfirst,qos").?).name(&nb));
    for ([_][]const u8{ "qos,spin,keepwarm", "keepwarm,keepwarm", "keepwarm0", "keepwarm1001", "keepwarmx" }) |bad| try testing.expect(Sched.parse(bad) == null);
    try testing.expectEqualStrings("qos,demandfirst,keepwarm1000", (Sched.parse("qos,demandfirst,keepwarm1000").?).name(&nb));
    try testing.expectEqual(@as(u16, 25), (Sched.parse("keepwarm25").?).keep_warm_us);
    try testing.expect((Sched.parse("keepwarm25p").?).keep_warm_prefill and !(Sched.parse("keepwarm25").?).keep_warm_prefill);
    try testing.expectEqualStrings("qos,demandfirst,keepwarm1000p", (Sched.parse("qos,demandfirst,keepwarm1000p").?).name(&nb));
    try testing.expectEqualStrings("keepwarmp", (Sched.parse("keepwarmp").?).name(&nb));
    for ([_][]const u8{ "keepwarm25pp", "keepwarmp25" }) |bad| try testing.expect(Sched.parse(bad) == null);
    const page = std.heap.pageSize();
    {
        var plain = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16 });
        defer plain.stop();
        try testing.expectError(error.PoolUnavailable, plain.keepWarm(true));
    }
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16, .sched = .{ .keep_warm = true } });
    defer pool.stop();
    const s0 = pool.keepWarmSpins();
    std.Io.sleep(testing.io, .fromMilliseconds(10), .awake) catch {};
    try testing.expectEqual(s0, pool.keepWarmSpins());
    try pool.keepWarm(true);
    std.Io.sleep(testing.io, .fromMilliseconds(20), .awake) catch {};
    try pool.keepWarm(false);
    const s1 = pool.keepWarmSpins();
    try testing.expect(s1 > s0 + 1000);
    std.Io.sleep(testing.io, .fromMilliseconds(10), .awake) catch {};
    // At most the iteration in flight at the switch.
    try testing.expect(pool.keepWarmSpins() <= s1 + 1);
}

test "dsv41 io: startui: the pool's starting thread becomes USER_INTERACTIVE at start; without it the thread keeps its class" {
    var nb: [Sched.name_len]u8 = undefined;
    try testing.expectEqualStrings("qos,startui,keepwarm25", (Sched.parse("keepwarm25,startui,qos").?).name(&nb));
    try testing.expectEqual(@as(i32, 64 | 32 | 1), (Sched.parse("qos,startui,keepwarm25").?).bits());
    try testing.expectEqualStrings("qosdemand,demandfirst,startui,keepwarm1000p", (Sched.parse("keepwarm1000p,startui,demandfirst,qosdemand").?).name(&nb));
    const Run = struct {
        sched: Sched,
        before: c_uint = 0,
        after: c_uint = 0,
        err: ?anyerror = null,
        fn go(r: *@This()) void {
            r.before = ThreadProbe.qosOf(ThreadProbe.pthread_self());
            var pool = Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = std.heap.pageSize(), .tickets = 16, .sched = r.sched }) catch |e| {
                r.err = e;
                return;
            };
            r.after = ThreadProbe.qosOf(ThreadProbe.pthread_self());
            pool.stop();
        }
    };
    var on: Run = .{ .sched = .{ .start_ui = true } };
    const t1 = try std.Thread.spawn(.{}, Run.go, .{&on});
    t1.join();
    if (on.err) |e| return e;
    try testing.expect(on.before != ThreadProbe.user_interactive);
    try testing.expectEqual(ThreadProbe.user_interactive, on.after);
    var off: Run = .{ .sched = .{ .keep_warm = true } };
    const t2 = try std.Thread.spawn(.{}, Run.go, .{&off});
    t2.join();
    if (off.err) |e| return e;
    try testing.expectEqual(off.before, off.after);
}

test "dsv41 io: keepwarm<us>: the thread sleeps that long per loop instead of spinning" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16, .sched = .{ .keep_warm = true, .keep_warm_us = 100 } });
    defer pool.stop();
    const s0 = pool.keepWarmSpins();
    try pool.keepWarm(true);
    std.Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
    try pool.keepWarm(false);
    const loops = pool.keepWarmSpins() - s0;
    // 50 ms of 100 us sleeps: some loops, and far fewer than a yield loop's millions.
    try testing.expect(loops > 20 and loops < 1000);
}

test "dsv41 io: the reader scheduling list: off or qos, spin, demandfirst (spin only with qos), and the pool refuses spin alone" {
    var nb: [Sched.name_len]u8 = undefined;
    try testing.expectEqualStrings("off", (Sched.parse("off").?).name(&nb));
    try testing.expectEqualStrings("qos,spin,demandfirst", (Sched.parse("demandfirst,spin,qos").?).name(&nb));
    try testing.expectEqual(@as(i32, 5), (Sched.parse("qos,demandfirst").?).bits());
    for ([_][]const u8{ "spin", "qos,qos", "qos,fast", "", "QOS" }) |bad| try testing.expect(Sched.parse(bad) == null);
    // qosdemand: its own bit, named after qos, refused with qos and with spin; the C pool refuses the pair as well
    try testing.expectEqualStrings("qosdemand", (Sched.parse("qosdemand").?).name(&nb));
    try testing.expectEqual(@as(i32, 8), (Sched.parse("qosdemand").?).bits());
    try testing.expectEqualStrings("qosdemand,demandfirst", (Sched.parse("demandfirst,qosdemand").?).name(&nb));
    try testing.expectEqual(@as(i32, 12), (Sched.parse("qosdemand,demandfirst").?).bits());
    for ([_][]const u8{ "qos,qosdemand", "qosdemand,spin", "qosdemand,qosdemand", "qosdemand," }) |bad| try testing.expect(Sched.parse(bad) == null);
    try testing.expectError(error.PoolUnavailable, Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = std.heap.pageSize(), .tickets = 16, .sched = .{ .qos = true, .qos_demand = true } }));
    try testing.expectError(error.PoolUnavailable, Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = std.heap.pageSize(), .tickets = 16, .sched = .{ .spin = true } }));
}

test "dsv41 io: demand first: no unclaimed speculative chunk starts while a demand job executes; the record lands after it" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * page, .tickets = 128, .sched = .{ .demand_first = true }, .spec = .{ .threads = 1, .slots = 2, .record_bytes = spec_rec_len, .chunk_bytes = page } });
    defer pool.stop();
    defer clearFaults();
    // The demand job's first read sleeps 80 ms (a demand job executing), during which a record is queued for speculation.
    const base: u64 = 2 * page;
    injectFault(base / page * page, 5, 80 * std.time.ns_per_ms);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    std.Io.sleep(testing.io, .fromMilliseconds(10), .awake) catch {};
    const spec_base: u64 = 30 * page + 7;
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 1, &.{@intCast(spec_base)}, spec_rec_len));
    std.Io.sleep(testing.io, .fromMilliseconds(30), .awake) catch {};
    // Still executing: nothing speculative started.
    try testing.expectEqual(@as(i64, 0), pool.counter(.spec_chunks));
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    try waitFor(@as(i64, @intCast(spec_base)), landedAt);
    try testing.expectEqual(@as(i64, 0), pool.counter(.max_busy_at_start));
}

test "dsv41 io: qos: a claimed speculative record's worker runs at the demand class while claimed" {
    if (std.c.getenv("DSV41_TEST_READER_SCHED") != null) return error.SkipZigTest;
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    // A record of three page chunks (the demand ranges inside its first page) whose second and third chunks each sleep
    // 800 ms: claimed during the second, so the worker promotes itself at the third chunk's boundary. Both checks poll
    // for the state (a loaded box schedules the UTILITY worker late) and end on the chunk count, not on a sleep.
    const rec = 3 * page;
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * page, .tickets = 128, .sched = .{ .qos = true }, .spec = .{ .threads = 1, .slots = 2, .record_bytes = rec, .chunk_bytes = page } });
    defer pool.stop();
    defer clearFaults();
    const base: u64 = 20 * page;
    injectFaults(&.{ @intCast(base + page), @intCast(base + 2 * page) }, &.{ 5, 5 }, &.{ 800 * std.time.ns_per_ms, 800 * std.time.ns_per_ms });
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 1, &.{@intCast(base)}, rec));
    const SpecQos = struct {
        fn of() ?c_uint {
            var list: [*]ThreadProbe.mach_port_t = undefined;
            var n: u32 = 0;
            if (ThreadProbe.task_threads(ThreadProbe.mach_task_self_, &list, &n) != 0) return null;
            for (list[0..n]) |port| {
                const t = ThreadProbe.pthread_from_mach_thread_np(port) orelse continue;
                var name: [64]u8 = @splat(0);
                _ = ThreadProbe.pthread_getname_np(t, &name, name.len);
                if (std.mem.eql(u8, std.mem.sliceTo(&name, 0), "q3ld-spec-0")) return ThreadProbe.qosOf(t);
            }
            return null;
        }
        fn is(want: c_uint) bool {
            return of() == want;
        }
    };
    // Unclaimed: the worker has named itself and set UTILITY (its first chunk is unclaimed; the claim is not yet made).
    try waitFor(ThreadProbe.utility, SpecQos.is);
    try testing.expectEqual(@as(i64, 0), pool.counter(.claimed));
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    if (pool.counter(.spec_chunks) >= 2) {
        // The claim landed after the second chunk ended (the test thread stalled > 800 ms): no boundary left to promote at.
        std.debug.print("QOSPROBE skipped: the claim came after the second chunk (box loaded)\n", .{});
        try pool.wait(first, 2, 10 * std.time.ns_per_s);
        return error.SkipZigTest;
    }
    // Claimed: the worker reaches USER_INTERACTIVE before its third (claimed) chunk ends.
    var seen: ?c_uint = null;
    var t: u32 = 0;
    while (pool.counter(.spec_chunks) < 3) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        seen = SpecQos.of();
        if (seen == ThreadProbe.user_interactive) break;
        std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    std.debug.print("QOSPROBE claimed: \"q3ld-spec-0\" qos 0x{x}\n", .{seen orelse 0});
    try testing.expectEqual(@as(?c_uint, ThreadProbe.user_interactive), seen);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    try testing.expectEqual(@as(i64, 1), pool.counter(.claimed));
}

// ── The reader's C ABI at its edges. Deterministic by construction: a worker is held at a pre-range's bind point (a
// pre-read nobody demands), a ticket is pending through its status word, a deadline of 0 is already past. ──

/// The C pool's entry points the Pool does not wrap: the stock and horizon-N steps, the stock pre-read, the snapshots.
const q3raw = struct {
    extern fn q3ld_spec_step(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64) i32;
    extern fn q3ld_spec_stepn(fd: i32, file_size: i64, cur: i64, nh: i32, nval: [*]const i32, val: [*]const i64, nissue: [*]const i32, iss: [*]const i64) i32;
    extern fn q3ld_pre_read(fd: i32, file_size: i64, tag: i64, n: i32, bases: [*]const i64) i32;
    extern fn q3ld_ev_state(out: *[10]i64) i32;
    extern fn q3ld_test_events(buf: ?[*]i64, cap: i64) i64;
    extern fn q3ld_test_spec_delay(max_ns: i64) void;
    extern fn q3ld_spec_idle_busy() i32;
    extern fn q3ld_max_h() i32;
};

/// A known reader defect's test fails until the defect is fixed, so it runs only with DSV41_COV_KNOWN_BUGS=1.
fn knownBug(comptime what: []const u8) !void {
    if (std.c.getenv("DSV41_COV_KNOWN_BUGS") != null) return;
    std.debug.print("KNOWN BUG (skipped; DSV41_COV_KNOWN_BUGS=1 runs it): " ++ what ++ "\n", .{});
    return error.SkipZigTest;
}

/// One record's job on the raw ABI at ticket `first`: gate/up parts `gl` at `gu`, down parts `dl` at `down`, back to
/// back into `dst`. The Pool's published marks of its two tickets are cleared first (Pool.wait reads them).
fn rawSubmit(pool: *Pool, fd: i32, size: i64, deadline: i64, gu: u64, down: u64, gl: []const i64, dl: []const i64, dst: []u8, first: u32) c_int {
    var lens: [12]i64 = undefined;
    var row: [12]u64 = undefined;
    var at: usize = 0;
    var k: usize = 0;
    for ([_][]const i64{ gl, dl }) |part| for (part) |l| {
        lens[k] = l;
        row[k] = @intFromPtr(dst.ptr) + at;
        at += @intCast(l);
        k += 1;
    };
    const offs = [2]i64{ @intCast(gu), @intCast(down) };
    const rows = [1][*]const u64{&row};
    @memset(pool.published[first..][0..2], false);
    return c.q3ld_submit(fd, size, deadline, 1, @intCast(gl.len), @intCast(dl.len), &offs, &rows, &lens, first);
}

fn counterAt(p: *const Pool, i: usize) i64 {
    return @atomicLoad(i64, &p.counters[i], .monotonic);
}

fn preStarted1(p: *Pool) bool {
    return p.counter(.pre_started) >= 1;
}

fn preStarted2(p: *Pool) bool {
    return p.counter(.pre_started) >= 2;
}

/// A started-and-stopped raw pool: the C state's configuration flag cleared, whatever an earlier test left.
fn resetPoolState() !void {
    const page = std.heap.pageSize();
    const mem = try std.heap.page_allocator.alloc(u8, page);
    defer std.heap.page_allocator.free(mem);
    var res_arr: [16 * res_w]i64 = @splat(0);
    var log_arr: [16]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const staging = [1]u64{@intFromPtr(mem.ptr)};
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, null));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_start(1, &staging, @intCast(page), @intCast(page), &res_arr, 16, &log_arr, 16, &gauge));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_stop());
}

test "dsv41 io cov: with no pool running every entry point refuses, and the snapshots read the stopped state" {
    const lens = [2]i64{ 10, 10 };
    const offs = [2]i64{ 0, 100 };
    const row = [2]u64{ 0, 0 };
    const rows = [1][*]const u64{&row};
    const bases = [1]i64{0};
    const one = [1]i32{1};
    const vals = [1]u64{1};
    const tix = [1]i64{0};
    var word: i64 = 0;
    try testing.expectEqual(@as(c_int, -1), c.q3ld_stop());
    try testing.expectEqual(@as(c_int, 0), c.q3ld_quiesce(std.time.ns_per_ms));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_submit(3, 1000, -1, 1, 1, 1, &offs, &rows, &lens, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_submit_warm(3, 1000, 1, 1, 1, &offs, &rows, &lens, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_warm_config(1));
    try testing.expectEqual(@as(i64, -1), c.q3ld_warm_cancel(0, 2));
    try testing.expectEqual(@as(i32, -1), c.q3ld_spec_step_len(3, 1000, 0, 1, &bases, 10));
    try testing.expectEqual(@as(i32, -1), q3raw.q3ld_spec_step(3, 1000, 0, 1, &bases));
    for ([_]i32{ -1, 1, 5 }) |nh| try testing.expectEqual(@as(i32, -1), q3raw.q3ld_spec_stepn(3, 1000, 0, nh, &one, &bases, &one, &bases));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(1, 1, &lens));
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(3, 1000, 1, 1, &bases, &lens));
    try testing.expectEqual(@as(i32, -1), q3raw.q3ld_pre_read(3, 1000, 1, 1, &bases));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, @intFromPtr(&word), std.time.ns_per_s, 0));
    for ([_]i32{ 0, 1, io.max_gates + 1 }) |n| try testing.expectEqual(@as(i32, -1), c.q3ld_ev_gates(n, &vals, &one, &tix));
    try testing.expectEqual(@as(i32, -1), c.q3ld_ev_release(1));
    var ev: [10]i64 = undefined;
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_ev_state(&ev));
    try testing.expectEqual(@as(i64, 0), ev[0]);
    try testing.expectEqual(@as(i64, 0), ev[8]);
    var pre: [pre_state_w * max_pre]i64 = undefined;
    try testing.expectEqual(@as(i32, 0), c.q3ld_pre_state(&pre));
    var sp: [spec_state_w * max_spec]i64 = undefined;
    try testing.expectEqual(@as(i32, 0), c.q3ld_spec_state(&sp));
    // A stop restores the stock idle rule; the horizon count is the header's.
    try testing.expectEqual(@as(i32, 1), q3raw.q3ld_spec_idle_busy());
    try testing.expectEqual(@as(i32, 4), q3raw.q3ld_max_h());
    try testing.expectEqual(@as(i64, 0), word);
}

test "dsv41 io cov: the stopped pool's configuration refuses bad arguments; the start refuses an unconfigured pool and misaligned buffers" {
    try resetPoolState();
    const page = std.heap.pageSize();
    const pg: i64 = @intCast(page);
    var counters: [io.counters_n]i64 = @splat(0);
    const mem = try std.heap.page_allocator.alloc(u8, 2 * page);
    defer std.heap.page_allocator.free(mem);
    const p0: u64 = @intFromPtr(mem.ptr);
    const bufs = [1]u64{p0};
    const Bad = struct { t: i32 = 1, n: i32 = 1, slot: i64, rec: i64 = 100, chunk: i64, ctr: bool = true };
    for ([_]Bad{
        .{ .t = -1, .slot = pg, .chunk = pg },
        .{ .t = io.max_spec_threads + 1, .slot = pg, .chunk = pg },
        .{ .n = -1, .slot = pg, .chunk = pg },
        .{ .n = max_spec + 1, .slot = pg, .chunk = pg },
        .{ .n = 0, .slot = pg, .chunk = pg },
        .{ .rec = 0, .slot = pg, .chunk = pg },
        .{ .rec = pg + 1, .slot = pg, .chunk = pg },
        .{ .slot = pg, .chunk = 0 },
        .{ .slot = pg, .chunk = pg, .ctr = false },
    }) |b| try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_config(b.t, &bufs, b.n, b.slot, b.rec, b.chunk, if (b.ctr) &counters else null));
    for ([_]i32{ -1, 2 }) |v| try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_streams(v));
    // Out of range, spin without qos, qos with qosdemand.
    for ([_]i32{ -1, 16, 2, 9 }) |m| try testing.expectEqual(@as(c_int, -1), c.q3ld_sched_config(m));

    var res_arr: [16 * res_w]i64 = @splat(0);
    var log_arr: [16]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const staging = [1]u64{p0};
    // No q3ld_spec_config since the last stop.
    try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    // Configured without the speculative class (counters optional): each start argument out of range.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, null));
    for ([_][3]i64{ .{ 0, pg, pg }, .{ io.max_workers + 1, pg, pg }, .{ 1, pg, 0 }, .{ 1, pg + 1, pg } }) |a|
        try testing.expectEqual(@as(c_int, -1), c.q3ld_start(@intCast(a[0]), &staging, a[1], a[2], &res_arr, 16, &log_arr, 16, &gauge));
    // A speculative slot off its page, a slot size or a chunk that is not a page multiple.
    const odd = [1]u64{p0 + 8};
    for ([_]struct { b: *const [1]u64, slot: i64, chunk: i64 }{ .{ .b = &odd, .slot = pg, .chunk = pg }, .{ .b = &bufs, .slot = pg + 8, .chunk = pg }, .{ .b = &bufs, .slot = pg, .chunk = 100 } }) |s| {
        try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(1, s.b, 1, s.slot, 100, s.chunk, &counters));
        try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    }
    // A running pool refuses every configuration call and a second start; its stop returns it to unconfigured.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_streams(0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_sched_config(0));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_stop());
    try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
}

test "dsv41 io cov: a running pool refuses a malformed job, a pending ticket and the warm class's bad arguments by their codes" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(8 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 32, .warm = .{ .tickets = 8, .busy_max = 1 } });
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var dst: [20]u8 = @splat(0xAA);
    const lens = [2]i64{ 10, 10 };
    const row = [2]u64{ @intFromPtr(&dst), @intFromPtr(&dst) + 10 };
    const rows = [1][*]const u64{&row};
    const offs = [2]i64{ 0, 100 };
    const Shape = struct { n: i32 = 1, ngu: i32 = 1, ndown: i32 = 1, first: i64 = 0 };
    for ([_]Shape{ .{ .n = 0 }, .{ .n = max_items + 1 }, .{ .ngu = 0 }, .{ .ngu = 7 }, .{ .ndown = 0 }, .{ .ndown = 7 }, .{ .first = -1 }, .{ .first = 31 } }) |s| {
        try testing.expectEqual(@as(c_int, -1), c.q3ld_submit(fd, size, -1, s.n, s.ngu, s.ndown, &offs, &rows, &lens, s.first));
        try testing.expectEqual(@as(c_int, -1), c.q3ld_submit_warm(fd, size, s.n, s.ngu, s.ndown, &offs, &rows, &lens, s.first));
    }
    // A ticket still pending (its status word) refuses both rings' submits, raw and through Records.
    pool.res[5 * res_w] = @intFromEnum(Status.pending);
    try testing.expectEqual(@as(c_int, -2), rawSubmit(pool, fd, size, -1, 0, 100, lens[0..1], lens[1..2], &dst, 4));
    try testing.expectEqual(@as(c_int, -2), c.q3ld_submit_warm(fd, size, 1, 1, 1, &offs, &rows, &lens, 4));
    pool.res[5 * res_w] = 0;
    const small = [n_components]u64{ 10, 10, 10, 10, 10, 10, 10, 10, 10 };
    var d = try Dests.init(1, &small);
    defer testing.allocator.free(d.buf);
    pool.res[0] = @intFromEnum(Status.pending);
    try testing.expectError(error.TicketsBusy, R96.submit(pool, f.ufd, &.{0}, &.{1000}, d.rows[0..1], &small));
    pool.res[0] = 0;
    pool.res[24 * res_w] = @intFromEnum(Status.pending);
    try testing.expectError(error.TicketsBusy, R96.submitWarm(pool, f.ufd, &.{0}, &.{1000}, d.rows[0..1], &small));
    pool.res[24 * res_w] = 0;
    // Shapes Records refuses before the pool: no record, offsets that do not match the rows, no aux ring.
    try testing.expectError(error.InvalidJob, R96.submit(pool, f.ufd, &.{}, &.{}, d.rows[0..0], &small));
    try testing.expectError(error.InvalidJob, R96.submit(pool, f.ufd, &.{ 0, 1 }, &.{1000}, d.rows[0..1], &small));
    try testing.expectError(error.InvalidJob, R96.submitAux(pool, f.ufd, &.{0}, &.{1000}, d.rows[0..1], &small));
    try testing.expectEqual(@as(u32, 0), pool.auxTickets());
    // The warm class: a busy limit outside 0..workers, a negative cancel span; off refuses its submit, on again takes it.
    for ([_]i32{ -1, 3 }) |b| try testing.expectEqual(@as(c_int, -1), c.q3ld_warm_config(b));
    try testing.expectEqual(@as(i64, -1), c.q3ld_warm_cancel(0, -1));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_warm_config(0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_submit_warm(fd, size, 1, 1, 1, &offs, &rows, &lens, 24));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_warm_config(1));
    // A well-formed raw job lands: its two one-part ranges, byte for byte.
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, -1, 3, page + 7, lens[0..1], lens[1..2], &dst, 10));
    try pool.wait(10, 2, 10 * std.time.ns_per_s);
    try testing.expectEqualSlices(u8, f.image[3..13], dst[0..10]);
    try testing.expectEqualSlices(u8, f.image[page + 7 ..][0..10], dst[10..20]);
    for (10..12) |t| try testing.expectEqual(Status.ok, pool.result(@intCast(t)).status);
}

test "dsv41 io cov: a job past its deadline publishes deadline then skipped; a zero-length range finishes before any check; a far deadline reads" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(8 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32 });
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var dst: [500]u8 = @splat(0xAA);
    // Deadline 0 is past at the range's first check: nothing read, nothing written.
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, 0, 500, 3 * page + 7, &.{300}, &.{200}, &dst, 0));
    try pool.wait(0, 2, 10 * std.time.ns_per_s);
    const late = pool.result(0);
    try testing.expectEqual(Status.deadline, late.status);
    try testing.expectEqual(@as(i64, 0), late.payload);
    try testing.expectEqual(@as(i64, 0), late.preadv_calls);
    try testing.expectEqual(Status.skipped, pool.result(1).status);
    try testing.expect(std.mem.allEqual(u8, &dst, 0xAA));
    // A zero-length gate/up range has nothing to read: OK past the deadline; its down range is late.
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, 0, 500, 3 * page + 7, &.{0}, &.{200}, &dst, 2));
    try pool.wait(2, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(2).status);
    try testing.expectEqual(@as(i64, 0), pool.result(2).payload);
    try testing.expectEqual(Status.deadline, pool.result(3).status);
    // A deadline an hour away reads as no deadline.
    const far = c.q3ld_monotonic_ns() + 3600 * std.time.ns_per_s;
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, far, 500, 3 * page + 7, &.{300}, &.{200}, &dst, 4));
    try pool.wait(4, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(4).status);
    try testing.expectEqual(Status.ok, pool.result(5).status);
    try testing.expectEqualSlices(u8, f.image[500..800], dst[0..300]);
    try testing.expectEqualSlices(u8, f.image[3 * page + 7 ..][0..200], dst[300..500]);
}

test "dsv41 io cov: ranges at and past the end of file, zero-length parts, a short file size, a range over many stagings and a bad descriptor" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(6 * page + 100);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    const S: u64 = f.ufd.size;
    const dst = try testing.allocator.alloc(u8, 4 * page);
    defer testing.allocator.free(dst);
    const Case = struct { gu: u64, gl: []const i64, dl: []const i64 = &.{10}, size: i64, fd: i32, want: Status, payload: i64, calls: ?i64 = null };
    const cases = [_]Case{
        // Straddles the end: the 50 bytes that exist land, then the range is short; its down range is skipped.
        .{ .gu = S - 50, .gl = &.{100}, .size = size, .fd = fd, .want = .short, .payload = 50, .calls = 1 },
        // Starts at the end, or a page past it: short, nothing read.
        .{ .gu = S, .gl = &.{10}, .size = size, .fd = fd, .want = .short, .payload = 0, .calls = 0 },
        .{ .gu = S + page, .gl = &.{10}, .size = size, .fd = fd, .want = .short, .payload = 0, .calls = 0 },
        // The caller's file size bounds the read, whatever the file holds.
        .{ .gu = 0, .gl = &.{200}, .size = 100, .fd = fd, .want = .short, .payload = 100 },
        // Zero-length parts (a leading one, one between, a trailing one) are skipped; the rest lands.
        .{ .gu = 3, .gl = &.{ 0, 7, 0 }, .dl = &.{ 0, 5 }, .size = size, .fd = fd, .want = .ok, .payload = 7 },
        // Unaligned, three stagings and a page wide: four or more aligned reads.
        .{ .gu = page - 1, .gl = &.{ page + 2, 2 * page + 1 }, .size = size, .fd = fd, .want = .ok, .payload = 3 * page + 3 },
        // A bad descriptor: the preadv's errno.
        .{ .gu = 0, .gl = &.{10}, .size = size, .fd = -1, .want = .os_error, .payload = 0, .calls = 1 },
    };
    for (cases, 0..) |cs, i| {
        @memset(dst, 0xAA);
        const t: u32 = @intCast(2 * i);
        const down: u64 = 2 * page + 9;
        try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, cs.fd, cs.size, -1, cs.gu, down, cs.gl, cs.dl, dst, t));
        try pool.wait(t, 2, 10 * std.time.ns_per_s);
        const r = pool.result(t);
        testing.expectEqual(cs.want, r.status) catch |e| {
            std.debug.print("case {d}\n", .{i});
            return e;
        };
        try testing.expectEqual(cs.payload, r.payload);
        if (cs.calls) |n| try testing.expectEqual(n, r.preadv_calls);
        var gl_total: usize = 0;
        for (cs.gl) |l| gl_total += @intCast(l);
        const got: usize = @intCast(r.payload);
        // The landed bytes are the file's (zero-length parts take no room), the rest of the rows untouched.
        if (got > 0) try testing.expectEqualSlices(u8, f.image[cs.gu..][0..got], dst[0..got]);
        try testing.expect(std.mem.allEqual(u8, dst[got..gl_total], 0xAA));
        if (cs.want == .ok) {
            try testing.expectEqual(Status.ok, pool.result(t + 1).status);
            var dl_total: usize = 0;
            for (cs.dl) |l| dl_total += @intCast(l);
            try testing.expectEqualSlices(u8, f.image[down..][0..dl_total], dst[gl_total..][0..dl_total]);
        } else {
            try testing.expectEqual(Status.skipped, pool.result(t + 1).status);
        }
        if (cs.want == .os_error) try testing.expectEqual(@as(i64, @intFromEnum(std.posix.E.BADF)), r.errno);
        if (cs.gu == page - 1) try testing.expect(r.preadv_calls >= 4);
    }
}

test "dsv41 io cov: jobs queued behind a held worker stay pending (a resubmit is refused), then drain in order once a quiesce frees it" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(1, 2);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    // The only worker takes the pre-read's gate/up range and waits at its bind point; its down range stays queued.
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(40 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted1);
    var d = try Dests.init(3, &spec_lens);
    defer testing.allocator.free(d.buf);
    var firsts: [3]u32 = undefined;
    const bases = [3]u64{ 2 * page + 1, 10 * page + 2, 20 * page + 3 };
    for (&firsts, bases, 0..) |*fst, b, i| fst.* = try R96.submit(pool, f.ufd, &.{b}, &.{b + spec_gu_len}, d.rows[i..][0..1], &spec_lens);
    for (firsts) |fst| for (0..2) |k| try testing.expectEqual(Status.pending, pool.result(fst + @as(u32, @intCast(k))).status);
    var scratch: [20]u8 = undefined;
    try testing.expectEqual(@as(c_int, -2), rawSubmit(pool, f.ufd.fd, @intCast(f.ufd.size), -1, 0, 100, &.{10}, &.{10}, &scratch, firsts[1]));
    // Quiesce expires the unbound ranges (the queued one freed, the held one cancelled at its bind point) and drains.
    const seq0 = c.q3ld_seq();
    try testing.expectEqual(@as(c_int, 0), c.q3ld_quiesce(10 * std.time.ns_per_s));
    for (firsts) |fst| try pool.wait(fst, 2, 10 * std.time.ns_per_s);
    for (bases, 0..) |b, i| try d.expectRecord(i, f.image, b, b + spec_gu_len, &spec_lens);
    var order: [6]u32 = undefined;
    const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
    try testing.expectEqualSlices(u32, &.{ firsts[0], firsts[0] + 1, firsts[1], firsts[1] + 1, firsts[2], firsts[2] + 1 }, got);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 1), pool.counter(.pre_started));
    try testing.expectEqual(@as(i64, 0), pool.counter(.pre_served));
    try testing.expectEqual(@as(i64, 1), pool.readGauge()[1]);
    try testing.expect(preIdle(pool));
}

test "dsv41 io cov: the pre-read class refuses bad geometry and a re-arm under live ranges; it caps ranges at a staging buffer and its table" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(160 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var l: [n_components]i64 = undefined;
    for (spec_lens, &l) |x, *y| y.* = @intCast(x);
    // Disarming an unarmed class is a no-op; bad counts and a missing length list are refused.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_pre_config(0, 0, null));
    for ([_][2]i32{ .{ 0, 1 }, .{ 1, 0 }, .{ 7, 1 }, .{ 1, 7 }, .{ -1, 1 } }) |g| try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(g[0], g[1], &l));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(6, 3, null));
    try R96.armPreRead(pool, &spec_lens);
    // Malformed calls: a negative count, a negative length.
    const b0 = [1]i64{@intCast(4 * page)};
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(fd, size, 1, -1, &b0, &l));
    var neg = l;
    neg[7] = -5;
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(fd, size, 1, 1, &b0, &neg));
    // Two ranges, both workers held at their bind points: a re-arm is refused while they live.
    const hold = [1]i64{@intCast(150 * page + 3)};
    try testing.expectEqual(@as(i32, 2), c.q3ld_pre_read_lens(fd, size, 100, 1, &hold, &l));
    try waitFor(pool, preStarted2);
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(6, 3, &l));
    // The stock entry point (the armed lengths) on the same record refreshes the live ranges: nothing new.
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_pre_read(fd, size, 100, 1, &hold));
    // A gate/up range wider than a staging buffer (4 pages) is not pre-read; its down range is.
    var wide = l;
    wide[0] = @intCast(5 * page);
    try testing.expectEqual(@as(i32, 1), c.q3ld_pre_read_lens(fd, size, 100, 1, &b0, &wide));
    try testing.expectEqual(@as(i64, 1), pool.counter(.pre_noslot));
    // 20 records want 40 ranges; the table has 29 entries left: every gate/up, then 9 downs.
    var many: [20]i64 = undefined;
    for (&many, 0..) |*b, i| b.* = @intCast(10 * page + i * 5 * page + 1);
    try testing.expectEqual(@as(i32, 29), c.q3ld_pre_read_lens(fd, size, 100, 20, &many, &l));
    try testing.expectEqual(@as(i64, 12), pool.counter(.pre_noslot));
    try testing.expectEqual(@as(i64, 32), pool.counter(.pre_issued));
    var st: [pre_state_w * max_pre]i64 = undefined;
    try testing.expectEqual(@as(i32, 1), c.q3ld_pre_state(&st));
    var queued: usize = 0;
    var inflight: usize = 0;
    for (0..max_pre) |i| {
        queued += @intFromBool(st[i * pre_state_w] == 1);
        inflight += @intFromBool(st[i * pre_state_w] == 2);
    }
    try testing.expectEqual(@as(usize, 30), queued);
    try testing.expectEqual(@as(usize, 2), inflight);
    // The call's settle expires all 32: the held two cancelled at their bind points, the queued freed.
    _ = try pool.specStep(f.ufd, 100, &.{}, 0);
    try waitFor(pool, preIdle);
    try testing.expectEqual(@as(i64, 32), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_started));
    // Disarmed, the class refuses a pre-read.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_pre_config(0, 0, null));
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(fd, size, 101, 1, &b0, &l));
}

test "dsv41 io cov: horizon-N steps keep, promote and drop queued records; a full table refuses; a live base refreshes; a pre-read cancels a queued record" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(160 * page);
    defer f.deinit();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var pool = try specPool(2, 4);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    // Both workers held at pre-range bind points (tag 100, past every step below): demand busy 2, so the queue rule
    // keeps every unclaimed record QUEUED.
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(150 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted2);
    const rec = struct {
        fn at(i: u64) i64 {
            return @intCast(2 * std.heap.pageSize() + i * 6 * std.heap.pageSize() + i);
        }
    }.at;
    const A = rec(0);
    const B = rec(1);
    const C = rec(2);
    const D = rec(3);
    const E = rec(4);
    // Horizon 1 (A, B: tag 1, class 0); horizon 2 (C, D: tag 2, class 1).
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.ufd, 0, &.{ A, B }, 0));
    const no_val = [1]i64{0};
    try testing.expectEqual(@as(i32, 2), q3raw.q3ld_spec_stepn(fd, size, 0, 2, &[2]i32{ 0, 0 }, &no_val, &[2]i32{ 0, 2 }, &[2]i64{ C, D }));
    // The next call (cur 1): A and B settle; C is still a candidate for the next layer (kept, promoted to class 0),
    // D is not (dropped); E is issued two ahead.
    try testing.expectEqual(@as(i32, 1), q3raw.q3ld_spec_stepn(fd, size, 1, 2, &[2]i32{ 1, 0 }, &[1]i64{C}, &[2]i32{ 0, 1 }, &[1]i64{E}));
    try testing.expectEqual(@as(i64, 3), pool.counter(.expired));
    try testing.expectEqual(@as(i64, 1), counterAt(pool, @intFromEnum(io.Counter.h_kept) + 1));
    try testing.expectEqual(@as(i64, 1), counterAt(pool, @intFromEnum(io.Counter.h_dropped) + 1));
    try testing.expectEqual(@as(i64, 2), counterAt(pool, @intFromEnum(io.Counter.h_submitted)));
    try testing.expectEqual(@as(i64, 3), counterAt(pool, @intFromEnum(io.Counter.h_submitted) + 1));
    var raw_st: [spec_state_w * max_spec]i64 = undefined;
    try testing.expectEqual(@as(i32, 4), c.q3ld_spec_state(&raw_st));
    var seen: u32 = 0;
    for (0..4) |i| {
        const w = raw_st[i * spec_state_w ..][0..spec_state_w];
        if (w[0] == 0) continue;
        try testing.expectEqual(@as(i64, 1), w[0]);
        if (w[2] == C) {
            try testing.expectEqualSlices(i64, &.{ 2, 0, 2 }, &.{ w[1], w[8], w[10] });
            seen += 1;
        } else if (w[2] == E) {
            try testing.expectEqualSlices(i64, &.{ 3, 1, 2 }, &.{ w[1], w[8], w[10] });
            seen += 1;
        }
    }
    try testing.expectEqual(@as(u32, 2), seen);
    // Four slots, C and E live: two of three new records find a slot, the third none (no landed slot to reclaim).
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.ufd, 1, &.{ rec(5), rec(6), rec(7) }, 0));
    try testing.expectEqual(@as(i64, 1), pool.counter(.noslot));
    // A live base is refreshed, never queued twice (both h1 entry points).
    try testing.expectEqual(@as(u32, 0), try pool.specStep(f.ufd, 1, &.{C}, 0));
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_spec_step(fd, size, 1, 1, &[1]i64{C}));
    try testing.expectEqual(@as(i64, 2), pool.counter(.refreshed));
    // A record length past the configured one is refused.
    try testing.expectError(error.SpecRefused, pool.specStep(f.ufd, 1, &.{A}, spec_rec_len + 1));
    // A pre-read of the queued C cancels it (its bytes are pre-read instead): two ranges queued.
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{C}, &spec_lens));
    try testing.expectEqual(@as(i64, 1), pool.counter(.cancelled_by_demand));
    try testing.expectEqual(@as(i64, 0), pool.counter(.started));
    // The stop's quiesce frees the queued records and the held workers.
}

test "dsv41 io cov: a failed job skip-cancels the pre-ranges bound to its later ranges" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    defer clearFaults();
    try R96.armPreRead(pool, &spec_lens);
    const ra: u64 = 2 * page + 5;
    const rb: u64 = 20 * page + 9;
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 1, &.{@intCast(rb)}, &spec_lens));
    try waitFor(pool, preStarted2);
    injectFault(ra / page * page, 2, 0);
    var d = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{ ra, rb }, &.{ ra + spec_gu_len, rb + spec_gu_len }, d.rows[0..2], &spec_lens);
    try pool.wait(first, 4, 10 * std.time.ns_per_s);
    const want = [4]Status{ .os_error, .skipped, .skipped, .skipped };
    for (want, 0..) |w, k| try testing.expectEqual(w, pool.result(first + @as(u32, @intCast(k))).status);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_bound));
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_skip_cancels));
    try testing.expectEqual(@as(i64, 0), pool.counter(.pre_served));
    try waitFor(pool, preIdle);
}

test "dsv41 io cov: the event class's arming refusals, the gate ring's bounds, and a host release past every gate" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 128 });
    defer pool.stop();
    var word: i64 align(8) = 0;
    const w: u64 = @intFromPtr(&word);
    var st: [10]i64 = undefined;
    // Off while off is a no-op; a bad kind, a null object, timeouts outside 1 ms .. 600 s.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_ev_config(0, 0, 0, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(3, w, std.time.ns_per_s, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, 0, std.time.ns_per_s, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, w, std.time.ns_per_ms - 1, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, w, 600 * std.time.ns_per_s + 1, 0));
    // Armed at 5: the event's value is taken as handed over; a second arm is refused.
    try pool.armEvent(.host, w, 60 * std.time.ns_per_s, 5);
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_ev_state(&st));
    try testing.expectEqualSlices(i64, &.{ 2, 0, 0, 0, 0, 5, 5, 5, 1, 60 * std.time.ns_per_s }, &st);
    try testing.expectError(error.EventRefused, pool.armEvent(.host, w, std.time.ns_per_s, 0));
    try testing.expectEqual(@as(i64, 0), word);
    // Disarmed (the watchdog joined): release and gates are refused; it arms again.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_ev_config(0, 0, 0, 0));
    _ = q3raw.q3ld_ev_state(&st);
    try testing.expectEqual(@as(i64, 0), st[0]);
    try testing.expectEqual(@as(i64, 0), st[8]);
    try testing.expectEqual(@as(i32, -1), c.q3ld_ev_release(1));
    try testing.expectError(error.GateRefused, pool.registerGates(&.{1}, &.{0}, &.{}));
    try pool.armEvent(.host, w, 60 * std.time.ns_per_s, 0);
    // A gate over a pending ticket heads the ring; a second gate may not wait for that ticket, nor for a negative count.
    pool.res[60 * res_w] = @intFromEnum(Status.pending);
    defer pool.res[60 * res_w] = 0;
    try pool.registerGates(&.{1}, &.{1}, &.{60});
    try testing.expectError(error.GateInvalid, pool.registerGates(&.{2}, &.{1}, &.{60}));
    try testing.expectError(error.GateInvalid, pool.registerGates(&.{2}, &.{-1}, &.{}));
    // 255 satisfied gates queue behind the unsatisfied head: the ring is full at 256 live.
    var values: [io.max_gates - 1]u64 = undefined;
    for (&values, 0..) |*v, i| v.* = 2 + i;
    const zeros: [io.max_gates - 1]i32 = @splat(0);
    try pool.registerGates(&values, &zeros, &.{});
    try testing.expectEqual(@as(i32, io.max_gates), q3raw.q3ld_ev_state(&st));
    try testing.expectEqualSlices(i64, &.{ 1, 1, 0 }, st[2..5]);
    try testing.expectEqual(@as(i64, 0), word);
    try testing.expectError(error.GatesFull, pool.registerGates(&.{257}, &.{0}, &.{}));
    // Live gates refuse a disarm.
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(0, 0, 0, 0));
    // The host release forces the head (the only gate still waiting): the prefix runs to the last gate.
    try testing.expectEqual(@as(i32, 1), c.q3ld_ev_release(256));
    try testing.expectEqual(@as(i64, 256), word);
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_ev_state(&st));
    try testing.expectEqual(@as(i64, 1), pool.counter(.ev_host_released));
    try testing.expectEqual(@as(i64, io.max_gates - 1), pool.counter(.ev_immediate));
    // With no live gate, a released value goes to the event too, and later gates must rise above it.
    try testing.expectEqual(@as(i32, 0), c.q3ld_ev_release(300));
    try testing.expectEqual(@as(i64, 300), word);
    try testing.expectError(error.GateInvalid, pool.registerGates(&.{300}, &.{0}, &.{}));
    try pool.registerGates(&.{301}, &.{0}, &.{});
    try testing.expectEqual(@as(i64, 301), word);
}

test "dsv41 io cov: a gate forced before its ticket lands ignores the late publish; the next gate's ticket still satisfies it" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(1, 2);
    defer pool.stop();
    var word: i64 align(8) = 0;
    try pool.armEvent(.host, @intFromPtr(&word), 60 * std.time.ns_per_s, 0);
    try R96.armPreRead(pool, &spec_lens);
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(40 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted1);
    // The job queues behind the held worker: both tickets pending, each under a gate.
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const base: u64 = 4 * page + 1;
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.registerGates(&.{ 1, 2 }, &.{ 1, 1 }, &.{ first, first + 1 });
    try testing.expectEqual(@as(i32, 1), c.q3ld_ev_release(1));
    try testing.expectEqual(@as(i64, 1), word);
    // The quiesce frees the worker: the forced gate's ticket publishes into nothing, the second gate is satisfied.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_quiesce(10 * std.time.ns_per_s));
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(@as(i64, 2), @atomicLoad(i64, &word, .acquire));
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    try testing.expectEqual(@as(i64, 1), pool.counter(.ev_host_released));
    try testing.expectEqual(@as(i64, 2), pool.counter(.ev_signals));
    try testing.expectEqual(@as(i64, 0), pool.counter(.ev_wd_forced));
}

test "dsv41 io cov: qos with spin reads a record and waits through the spin; the test event log names submit, start and the spec delay hook" {
    if (std.c.getenv("DSV41_TEST_READER_SCHED") != null) return error.SkipZigTest;
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = 4 * page, .tickets = 32, .sched = .{ .qos = true, .spin = true } });
    defer pool.stop();
    const Log = struct {
        var buf: [5 * 16]i64 = undefined;
        var off: [5]i64 = undefined;
    };
    _ = q3raw.q3ld_test_events(&Log.buf, 16);
    defer _ = q3raw.q3ld_test_events(&Log.off, 0);
    q3raw.q3ld_test_spec_delay(0);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const base: u64 = 3 * page + 11;
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    // Submit (kind 1: the job queued, no worker busy), then the worker's start (kind 2: the queue empty again).
    const n: usize = @intCast(q3raw.q3ld_test_events(null, 0));
    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqualSlices(i64, &.{ 1, 1, 0 }, Log.buf[0..3]);
    try testing.expectEqualSlices(i64, &.{ 2, 0, 0 }, Log.buf[5..8]);
    try testing.expectEqual(@as(i64, first), Log.buf[4]);
    try testing.expectEqual(@as(i64, first), Log.buf[9]);
    try testing.expect(Log.buf[8] >= Log.buf[3]);
}

fn specStateIs(base: i64, state: i64) bool {
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == base and s.state == state) return true;
    return false;
}

fn failedAt(base: i64) bool {
    return specStateIs(base, 4);
}

test "dsv41 io cov: a speculative record past the end fails; one the file ends inside lands short and serves only what landed" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(20 * page);
    defer f.deinit();
    var pool = try specPool(1, 4);
    defer pool.stop();
    const fd = f.ufd.fd;
    const S: i64 = @intCast(f.ufd.size);
    const pg: i64 = @intCast(page);
    // Past the end of the file: no span to read, the record fails.
    try testing.expectEqual(@as(i32, 1), c.q3ld_spec_step_len(fd, S, 0, 1, &[1]i64{S + pg}, spec_rec_len));
    try waitFor(S + pg, failedAt);
    // A caller's size past the real end: the record lands the 3000 bytes the file holds.
    const short = S - 3000;
    const big = S + 10 * pg;
    try testing.expectEqual(@as(i32, 1), c.q3ld_spec_step_len(fd, big, 0, 1, &[1]i64{short}, spec_rec_len));
    try waitFor(short, landedAt);
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == short) try testing.expectEqual(@as(i64, 3000), s.landed);
    try testing.expectEqual(@as(i64, 1), pool.counter(.failed));
    // A range inside what landed is copied out (no preadv); one past it is read the stock way and is short.
    var dst: [4000]u8 = @splat(0xAA);
    const s_u: u64 = @intCast(short);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, big, -1, s_u, s_u + 1000, &.{1000}, &.{500}, &dst, 0));
    try pool.wait(0, 2, 10 * std.time.ns_per_s);
    for (0..2) |k| {
        try testing.expectEqual(Status.ok, pool.result(@intCast(k)).status);
        try testing.expectEqual(@as(i64, 0), pool.result(@intCast(k)).preadv_calls);
    }
    try testing.expectEqualSlices(u8, f.image[s_u..][0..1500], dst[0..1500]);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, big, -1, s_u, s_u + 1000, &.{4000}, &.{10}, &dst, 2));
    try pool.wait(2, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.short, pool.result(2).status);
    try testing.expectEqual(@as(i64, 3000), pool.result(2).payload);
    try testing.expect(pool.result(2).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 2), pool.counter(.adopt_ranges));
}

fn specStarted(n: i64) type {
    return struct {
        fn f(p: *Pool) bool {
            return p.counter(.started) >= n;
        }
    };
}

fn parked1(p: *Pool) bool {
    return p.counter(.parks) >= 1;
}

fn paused1(p: *Pool) bool {
    return p.counter(.pauses) >= 1;
}

fn abandoned2(p: *Pool) bool {
    return p.counter(.abandoned) >= 2;
}

fn preStarted4(p: *Pool) bool {
    return p.counter(.pre_started) >= 4;
}

test "dsv41 io cov: an in-flight record parks (class 1) or pauses (class 0) when demand turns busy; a drop or settle abandons it; a claim waits for its chunk" {
    // Timing-guarded: each record's first chunk sleeps 400 ms in the pool (an injected rule), and the test moves demand
    // busy (two workers held at pre-range bind points) inside that window. A box that stalls the test thread past it
    // skips the step, never asserts on a race.
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    const pg: i64 = @intCast(page);
    const rec: u64 = 3 * page;
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * page, .tickets = 128, .spec = .{ .threads = 1, .slots = 4, .record_bytes = rec, .chunk_bytes = page } });
    defer pool.stop();
    defer clearFaults();
    try R96.armPreRead(pool, &spec_lens);
    var l: [n_components]i64 = undefined;
    for (spec_lens, &l) |x, *y| y.* = @intCast(x);
    const none = [1]i64{0};
    const hold_ms = 400 * std.time.ns_per_ms;

    // Park: R (horizon 2, class 1) starts with demand idle; demand turns busy during its first chunk.
    const R: i64 = 10 * pg;
    injectFault(@intCast(R), 5, hold_ms);
    try testing.expectEqual(@as(i32, 1), q3raw.q3ld_spec_stepn(fd, size, 0, 2, &[2]i32{ 0, 0 }, &none, &[2]i32{ 0, 1 }, &[1]i64{R}));
    try waitFor(pool, specStarted(1).f);
    try testing.expectEqual(@as(i32, 2), c.q3ld_pre_read_lens(fd, size, 100, 1, &[1]i64{40 * pg + 3}, &l));
    try waitFor(pool, preStarted2);
    if (pool.counter(.spec_chunks) != 0) {
        std.debug.print("SPECPROBE skipped: the first chunk ended before demand turned busy (box loaded)\n", .{});
        return error.SkipZigTest;
    }
    try waitFor(pool, parked1);
    // The next call no longer predicts R two ahead: the parked record is dropped with the chunk it read.
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_spec_stepn(fd, size, 1, 2, &[2]i32{ 0, 0 }, &none, &[2]i32{ 0, 0 }, &none));
    try testing.expectEqual(@as(i64, 1), pool.counter(.abandoned));
    try testing.expectEqual(pg, pool.counter(.abandoned_bytes));
    try testing.expectEqual(@as(i64, 1), counterAt(pool, @intFromEnum(io.Counter.h_dropped) + 1));
    // Demand idle again (a later pre-read call expires the held ranges).
    try testing.expectEqual(@as(i32, 0), c.q3ld_pre_read_lens(fd, size, 101, 0, &none, &l));
    try waitFor(pool, preIdle);

    // Pause: P (horizon 1, class 0) starts; demand turns busy during its first chunk; it waits at the boundary.
    const P: i64 = 20 * pg;
    injectFault(@intCast(P), 5, hold_ms);
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 1, &.{P}, rec));
    try waitFor(pool, specStarted(2).f);
    try testing.expectEqual(@as(i32, 2), c.q3ld_pre_read_lens(fd, size, 200, 1, &[1]i64{44 * pg + 5}, &l));
    try waitFor(pool, preStarted4);
    if (pool.counter(.spec_chunks) != 1) {
        std.debug.print("SPECPROBE skipped: P's first chunk ended before demand turned busy (box loaded)\n", .{});
        return error.SkipZigTest;
    }
    try waitFor(pool, paused1);
    // Its call's settle flags the running record: it is abandoned at the boundary it waits at.
    _ = try pool.specStep(f.ufd, 2, &.{}, rec);
    try waitFor(pool, abandoned2);
    try testing.expectEqual(@as(i32, 0), c.q3ld_pre_read_lens(fd, size, 201, 0, &none, &l));
    try waitFor(pool, preIdle);

    // A claim of an in-flight record: a deadline inside its first chunk expires the wait; a later one is served.
    const Q: i64 = 30 * pg;
    injectFault(@intCast(Q), 5, hold_ms);
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 2, &.{Q}, rec));
    try waitFor(pool, specStarted(3).f);
    var dst: [200]u8 = @splat(0xAA);
    const q: u64 = @intCast(Q);
    const t0 = c.q3ld_monotonic_ns();
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, t0 + 50 * std.time.ns_per_ms, q, q + 200, &.{100}, &.{100}, &dst, 0));
    try pool.wait(0, 2, 10 * std.time.ns_per_s);
    if (pool.result(0).status == .ok) {
        std.debug.print("SPECPROBE skipped: Q landed before its claim's deadline (box loaded)\n", .{});
        return error.SkipZigTest;
    }
    try testing.expectEqual(Status.deadline, pool.result(0).status);
    try testing.expectEqual(Status.skipped, pool.result(1).status);
    try testing.expect(std.mem.allEqual(u8, &dst, 0xAA));
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, t0 + 60 * std.time.ns_per_s, q, q + 200, &.{100}, &.{100}, &dst, 2));
    try pool.wait(2, 2, 10 * std.time.ns_per_s);
    for (2..4) |k| {
        try testing.expectEqual(Status.ok, pool.result(@intCast(k)).status);
        try testing.expectEqual(@as(i64, 0), pool.result(@intCast(k)).preadv_calls);
    }
    try testing.expectEqualSlices(u8, f.image[q..][0..100], dst[0..100]);
    try testing.expectEqualSlices(u8, f.image[q + 200 ..][0..100], dst[100..200]);
    try testing.expect(pool.counter(.adopt_waits) >= 1);
    try testing.expectEqual(@as(i64, 1), pool.counter(.claimed_inflight));
}

test "dsv41 io cov: q3ld_stop alone wakes and joins a worker held at a pre-range's bind point" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(1, 2);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(40 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted1);
    // The header's contract: q3ld_stop drains, joins and returns. Run it on a thread and give it 2 s.
    const Stopper = struct {
        var done = std.atomic.Value(bool).init(false);
        fn run() void {
            _ = c.q3ld_stop();
            done.store(true, .release);
        }
    };
    Stopper.done.store(false, .release);
    const th = try std.Thread.spawn(.{}, Stopper.run, .{});
    var t: u32 = 0;
    while (!Stopper.done.load(.acquire) and t < 2000) : (t += 1) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
    const hung = !Stopper.done.load(.acquire);
    // Rescue: a quiesce cancels the unbound range at its bind point, so the stop's join returns.
    if (hung) _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
    th.join();
    try testing.expect(!hung);
}

test "dsv41 io cov: the warm class refuses to arm on a pool configured without counters (its submits count into them)" {
    const page = std.heap.pageSize();
    const mem = try std.heap.page_allocator.alloc(u8, page);
    defer std.heap.page_allocator.free(mem);
    var res_arr: [16 * res_w]i64 = @splat(0);
    var log_arr: [16]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const staging = [1]u64{@intFromPtr(mem.ptr)};
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, null));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_start(1, &staging, @intCast(page), @intCast(page), &res_arr, 16, &log_arr, 16, &gauge));
    defer _ = c.q3ld_stop();
    // pre_config and ev_config refuse a pool without counters; the warm class must too (never submitted here).
    const rc = c.q3ld_warm_config(1);
    defer if (rc == 0) {
        _ = c.q3ld_warm_config(0);
    };
    try testing.expectEqual(@as(c_int, -1), rc);
}

test "dsv41 io cov: a refused pre-read re-arm (a negative length) leaves the armed class as it was" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    var pool = try specPool(1, 1);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    var l: [n_components]i64 = undefined;
    for (spec_lens, &l) |x, *y| y.* = @intCast(x);
    var neg = l;
    neg[7] = -1;
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(6, 3, &neg));
    // A refused call leaves the armed class as it was: the stock pre-read's ranges carry the record's geometry, so the
    // demand submit of that record binds them (with the defect they hold no component and never bind).
    const b: u64 = 4 * page + 1;
    try testing.expectEqual(@as(i32, 2), q3raw.q3ld_pre_read(f.ufd.fd, @intCast(f.ufd.size), 1, 1, &[1]i64{@intCast(b)}));
    try waitFor(pool, preStarted1);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{b}, &.{b + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, b, b + spec_gu_len, &spec_lens);
    try testing.expect(pool.counter(.pre_bound) >= 1);
}

test "dsv41 io cov: a file shorter than its job's size, ending inside a range's page, ends the range short (no retry spin)" {
    const page = std.heap.pageSize();
    // The file ends 300 bytes into its second page; the job carries the four pages it had when opened.
    var f = try PatternFile.init(page + 300);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16 });
    defer pool.stop();
    const fd = std.c.dup(f.ufd.fd);
    try testing.expect(fd >= 0);
    var closed = false;
    defer if (!closed) {
        _ = std.c.close(fd);
    };
    var dst: [510]u8 = @splat(0xAA);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, @intCast(4 * page), -1, page + 100, 0, &.{500}, &.{10}, &dst, 0));
    // 200 bytes exist past the skip: they land, then the end lies inside the same page. The range should be short.
    pool.wait(0, 2, 2 * std.time.ns_per_s) catch |e| {
        // Rescue the spinning worker: its next preadv fails on the closed descriptor.
        _ = std.c.close(fd);
        closed = true;
        try pool.wait(0, 2, 10 * std.time.ns_per_s);
        return e;
    };
    try testing.expectEqual(Status.short, pool.result(0).status);
    try testing.expectEqual(@as(i64, 200), pool.result(0).payload);
}

fn namedDemand1(_: void) bool {
    return ThreadProbe.scan("testsched").demand >= 1;
}

test "dsv41 io cov: DSV41_TEST_READER_SCHED runs a stock pool at its value (threads named); a malformed value leaves the stock pool" {
    if (std.c.getenv("DSV41_TEST_READER_SCHED") != null) return error.SkipZigTest;
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    defer _ = env.unsetenv("DSV41_TEST_READER_SCHED");
    const page = std.heap.pageSize();
    _ = env.setenv("DSV41_TEST_READER_SCHED", "qos,qos", 1);
    {
        var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16 });
        defer pool.stop();
        try testing.expectEqual(@as(u32, 0), ThreadProbe.scan("testsched-bad").named);
    }
    _ = env.setenv("DSV41_TEST_READER_SCHED", "qos", 1);
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16 });
    defer pool.stop();
    try waitFor({}, namedDemand1);
}

// The spin mode's lock-free reads of the queue length and the stop flag race the writers' stores unless both sides are
// atomic: this drives both against spinning workers. A ThreadSanitizer build of the pool (the coverage hook with a
// -fsanitize=thread object) reports any such race here; the plain build checks the bytes.
test "dsv41 io cov: spin mode: submits and a stop against spinning workers land every job" {
    if (std.c.getenv("DSV41_TEST_READER_SCHED") != null) return error.SkipZigTest;
    const page = std.heap.pageSize();
    var f = try PatternFile.init(32 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * page, .tickets = 64, .sched = .{ .qos = true, .spin = true } });
    defer pool.stop();
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    for (0..200) |i| {
        const base: u64 = (i % 20) * page + i;
        const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
        try pool.wait(first, 2, 10 * std.time.ns_per_s);
        try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    }
}
