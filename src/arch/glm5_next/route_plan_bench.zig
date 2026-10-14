//! The prompt call's host plan between the routing barrier's readback and the first MoE commit, on the CPU, driven by a
//! real route trace (GLM5_NEXT_ROUTE_PLAN_TRACE=<prefix>: the profile build's `<prefix>.json` + `.prompt.u16`; skipped
//! without it). glm5_next's stream (its bridge, the overlap schedule early_whole_trim with prompt lever 1's relocation,
//! wide depth 2, `max_route_ids` groups) over a one-layer synthetic bank of tiny records with 288 experts (the plan's
//! arithmetic is the bank's experts and rows, not its record bytes), host slot rows, the pool's pread workers. Each traced
//! routed layer replays as a request's first call of its layer (residents forgotten, prompt counts zeroed, as
//! `Source.beginPrefill` leaves them), the read-ahead posted beforehand with the call's own count ranking (P1's predictor
//! is the next router on the layer's input; here an oracle of the same width). The executor's steps (deepseek_v41
//! experts.zig `runWideCore`, glm5_next's wide route: seed, hot_first, depth 2, read_ahead) are timed one by one:
//! distinct, count, seed, order, route (the first `depth` routes: the whole call's plans and the first posts), scan (group
//! 0's three bank passes over the call's ids). Prints `GLM5N_ROUTE_PLAN` lines (ms per layer by step, median of `reps`).

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../../sdk_ext.zig");
const bank_mod = @import("bank.zig");
const experts = @import("experts.zig");
const shared = @import("../deepseek_v41/experts.zig");
const mimo_stream = @import("../mimo_v2/stream.zig");
const testing = std.testing;

const B = experts.B;
const StreamSource = B.StreamSource;
const max_route_ids = B.max_route_ids;
const n_experts = 288;
const top_k = experts.top_k;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const Step = enum { distinct, count, seed, order, route, scan };
/// The executor's code: `full` the scan before glm5next/route-plan (every bank pass over the whole call, two passes for
/// distinct and count), `grouped` runWideCore's now (one pass for distinct and count, the rows bucketed by group once:
/// `shared.bucketRowsByGroup`, each bank pass over its group's rows).
pub const Mode = enum { full, grouped };
const n_steps = @typeInfo(Step).@"enum".field_names.len;

/// One routed layer's synthetic bank: 288 records of (64, 32) NVFP4 projections, one page each.
const OneLayer = struct {
    tmp: std.testing.TmpDir,
    owner: bank_mod.Bank,
    sdk_bank: bank_mod.SDKBank,

    fn init(f: *OneLayer, a: std.mem.Allocator, io: std.Io) !void {
        f.tmp = std.testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        try writeBank(a, io, f.tmp.dir);
        var buf: [512]u8 = undefined;
        const dir = buf[0..try f.tmp.dir.realPath(io, &buf)];
        f.owner = try bank_mod.Bank.open(a, io, dir);
        errdefer f.owner.deinit();
        f.sdk_bank = try bank_mod.SDKBank.init(a, &f.owner);
    }

    fn deinit(f: *OneLayer) void {
        f.sdk_bank.deinit();
        f.owner.deinit();
        f.tmp.cleanup();
    }
};

/// experts.bin (zeros) and its manifest, as bank_test.zig's writer lays them out.
fn writeBank(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    const g = try bank_mod.Geometry.init(64, 32, n_experts, 3, 1);
    const data = try a.alloc(u8, @intCast(g.total_bytes));
    defer a.free(data);
    @memset(data, 0);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.print(a, "{{\"format\":\"{s}\",\"data_file\":\"experts.bin\",\"hidden_size\":64,\"intermediate_size\":32,\"num_experts\":{d},\"first_moe_layer\":3,\"num_moe_layers\":1,", .{ bank_mod.format, n_experts });
    try text.print(a, "\"layout\":\"{s}\",\"alignment\":{d},\"record_bytes\":{d},\"record_stride\":{d},\"gate_up_bytes\":{d},\"down_bytes\":{d},\"quantization\":{{\"mode\":\"nvfp4\",\"bits\":4,\"group_size\":16}},", .{ bank_mod.layout, bank_mod.page, g.logical_bytes, g.record_bytes, g.gate_up_bytes, g.logical_bytes - g.gate_up_bytes });
    try text.appendSlice(a, "\"source\":{\"repository\":\"synthetic\",\"revision\":\"0123456789abcdef0123456789abcdef01234567\"},\"parts\":[");
    for (g.segments, 0..) |s, c| {
        try text.print(a, "{s}{{\"name\":\"{s}\",\"dtype\":\"{t}\",\"shape\":[", .{ if (c == 0) "" else ",", @as(bank_mod.Component, @fromBackingInt(@intCast(c))).name(), s.dtype });
        for (s.shape[0..s.rank], 0..) |d, k| try text.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try text.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ s.offset, s.length });
    }
    try text.appendSlice(a, "],\"records\":[");
    for (0..n_experts) |i| {
        var d: [32]u8 = undefined;
        Sha256.hash(data[i * g.record_bytes ..][0..@intCast(g.logical_bytes)], &d, .{});
        try text.print(a, "{s}{{\"layer\":3,\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{x}\"}}", .{ if (i == 0) "" else ",", i, i * g.record_bytes, g.logical_bytes, d });
    }
    var whole: [32]u8 = undefined;
    Sha256.hash(data, &whole, .{});
    try text.print(a, "],\"output_files\":{{\"experts.bin\":{{\"size\":{d},\"sha256\":\"{x}\"}}}}}}", .{ g.total_bytes, whole });
    try dir.writeFile(io, .{ .sub_path = "experts.bin", .data = data });
    try dir.writeFile(io, .{ .sub_path = "expert-manifest.json", .data = text.items });
}

/// The executor's host state of one call (runWideCore's `wide` lists), reused across layers as the executor reuses them.
const Wide = struct {
    first: std.ArrayList(i32) = .empty,
    distinct: std.ArrayList(u16) = .empty,
    count: std.ArrayList(u32) = .empty,
    slot: std.ArrayList(u32) = .empty,
    act_row: std.ArrayList(u32) = .empty,
    pos: std.ArrayList(u32) = .empty,
    grp_of: std.ArrayList(u16) = .empty,
    grp_off: std.ArrayList(u32) = .empty,
    inv: std.ArrayList(u32) = .empty,

    fn deinit(w: *Wide, a: std.mem.Allocator) void {
        inline for (.{ "first", "distinct", "count", "slot", "act_row", "pos", "grp_of", "grp_off", "inv" }) |f| @field(w, f).deinit(a);
    }
};

/// What one call's plan produced (for the byte-identity test): the order, the groups' plans and windows, the scan's lists.
pub const Record = struct {
    distinct: std.ArrayList(u16) = .empty,
    plans: std.ArrayList(u8) = .empty,
    scan: std.ArrayList(u32) = .empty,

    fn deinit(r: *Record, a: std.mem.Allocator) void {
        r.distinct.deinit(a);
        r.plans.deinit(a);
        r.scan.deinit(a);
    }
};

/// One call of `ids` (rows x top-k) through the executor's host steps; `ns` gets each step's time; `rec` (optional) the
/// call's results. Then every group consumed (reads waited, released, the next routed) and the stream settled.
fn runCall(a: std.mem.Allocator, ss: *StreamSource, w: *Wide, ids: []const u16, mode: Mode, ns: *[n_steps]u64, rec: ?*Record) !void {
    const layer: u32 = 3;
    const k = top_k;
    // P1: the read-ahead posted before the barrier (the call's own ranking, an oracle predictor of the same width).
    {
        var counts: [n_experts]u32 = @splat(0);
        for (ids) |e| counts[e] += 1;
        var ranked: [n_experts]u16 = undefined;
        const hot = sdk_ext.expert.policy.rankHottest(&counts, &ranked);
        try ss.readAheadSeed(layer, hot);
    }
    var t = nowNs();
    switch (mode) {
        .full => {
            try w.first.resize(a, n_experts);
            @memset(w.first.items, -1);
            w.distinct.clearRetainingCapacity();
            for (ids) |e| if (w.first.items[e] < 0) {
                w.first.items[e] = @intCast(w.distinct.items.len);
                try w.distinct.append(a, e);
            };
            ns[@intFromEnum(Step.distinct)] = nowNs() - t;
            t = nowNs();
            try w.count.resize(a, n_experts);
            @memset(w.count.items, 0);
            for (ids) |e| w.count.items[e] += 1;
            ns[@intFromEnum(Step.count)] = nowNs() - t;
        },
        .grouped => {
            // runWideCore's one pass (the feed counts).
            try w.first.resize(a, n_experts);
            @memset(w.first.items, -1);
            w.distinct.clearRetainingCapacity();
            try w.distinct.ensureTotalCapacity(a, n_experts);
            try w.count.resize(a, n_experts);
            @memset(w.count.items, 0);
            for (ids) |e| {
                if (w.first.items[e] < 0) {
                    w.first.items[e] = @intCast(w.distinct.items.len);
                    w.distinct.appendAssumeCapacity(e);
                }
                w.count.items[e] += 1;
            }
            ns[@intFromEnum(Step.distinct)] = nowNs() - t;
            ns[@intFromEnum(Step.count)] = 0;
        },
    }
    t = nowNs();
    try ss.awaitReadAhead(layer);
    try ss.seedPrefill(layer, ids);
    ns[@intFromEnum(Step.seed)] = nowNs() - t;
    t = nowNs();
    ss.hotOrder(layer, w.distinct.items, w.count.items);
    for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
    ns[@intFromEnum(Step.order)] = nowNs() - t;
    const depth: usize = mimo_stream.wide_depth;
    const n_distinct = w.distinct.items.len;
    const n_groups = (n_distinct + max_route_ids - 1) / max_route_ids;
    const groupOf = struct {
        fn of(d: []const u16, gi: usize) []const u16 {
            return d[gi * max_route_ids .. @min((gi + 1) * max_route_ids, d.len)];
        }
    }.of;
    var calls: [8]?*StreamSource.Call = @splat(null);
    // `grouped`: the rows bucketed by group (runWideCore's `groupRows`, glm5_next's groups: no seed alignment), charged
    // to the scan.
    var bucket_ns: u64 = 0;
    if (mode == .grouped) {
        const tb = nowNs();
        try w.grp_of.resize(a, n_experts);
        for (w.distinct.items, 0..) |e, i| w.grp_of.items[e] = @intCast(i / max_route_ids);
        try w.grp_off.resize(a, n_groups + 1);
        try w.inv.resize(a, ids.len);
        shared.bucketRowsByGroup(ids, w.distinct.items, w.grp_of.items, w.count.items, w.grp_off.items, w.inv.items);
        bucket_ns = nowNs() - tb;
    }
    t = nowNs();
    for (0..@min(depth, n_groups)) |gi| calls[gi % depth] = try ss.route(layer, groupOf(w.distinct.items, gi), &.{});
    ns[@intFromEnum(Step.route)] = nowNs() - t;
    for (0..n_groups) |gi| {
        const start = gi * max_route_ids;
        const group = groupOf(w.distinct.items, gi);
        const call = calls[gi % depth].?;
        const sv = ss.served(call);
        if (rec) |r| {
            const route = call.route.?;
            try r.plans.appendSlice(a, std.mem.asBytes(&route.plan));
            try r.plans.append(a, route.window);
            try r.plans.appendSlice(a, std.mem.sliceAsBytes(route.hit_slots[0..route.plan.n_hits]));
        }
        t = nowNs();
        for ([_]B.BankKind{ .base, .ext, .transient }) |kind| {
            w.slot.clearRetainingCapacity();
            w.act_row.clearRetainingCapacity();
            w.pos.clearRetainingCapacity();
            switch (mode) {
                .full => for (ids, 0..) |e, row| {
                    const fi: usize = @intCast(w.first.items[e]);
                    if (fi < start or fi >= start + group.len) continue;
                    const ref = sv.refs[fi - start];
                    if (ref.bank != kind) continue;
                    try w.slot.append(a, ref.row);
                    try w.act_row.append(a, @intCast(row / @as(usize, k)));
                    try w.pos.append(a, @intCast(row));
                },
                .grouped => for (w.inv.items[w.grp_off.items[gi]..w.grp_off.items[gi + 1]]) |r| {
                    const row: usize = r;
                    const e = ids[row];
                    const fi: usize = @intCast(w.first.items[e]);
                    const ref = sv.refs[fi - start];
                    if (ref.bank != kind) continue;
                    try w.slot.append(a, ref.row);
                    try w.act_row.append(a, @intCast(row / @as(usize, k)));
                    try w.pos.append(a, @intCast(row));
                },
            }
            if (rec) |r| {
                try r.scan.append(a, @intCast(w.slot.items.len));
                try r.scan.appendSlice(a, w.slot.items);
                try r.scan.appendSlice(a, w.act_row.items);
                try r.scan.appendSlice(a, w.pos.items);
            }
        }
        if (gi == 0) ns[@intFromEnum(Step.scan)] = nowNs() - t + bucket_ns;
        for (0..sv.n_parts) |p| {
            try ss.waitGu(call, @intCast(p));
            try ss.waitDown(call, @intCast(p));
        }
        ss.release(call);
        calls[gi % depth] = null;
        if (gi + depth < n_groups) calls[gi % depth] = try ss.route(layer, groupOf(w.distinct.items, gi + depth), &.{});
    }
    if (rec) |r| try r.distinct.appendSlice(a, w.distinct.items);
    try ss.flush();
    try ss.stream.settleRoutes();
    _ = try ss.forgetResidents();
    ss.resetPromptCounts();
}

const Trace = struct {
    layers: u32,
    tokens: u32,
    prompt: []u16,

    fn load(a: std.mem.Allocator, io: std.Io, prefix: []const u8) !Trace {
        var path_buf: [1024]u8 = undefined;
        const hdr_path = try std.fmt.bufPrint(&path_buf, "{s}.json", .{prefix});
        const hdr_text = try std.Io.Dir.cwd().readFileAlloc(io, hdr_path, a, .limited(1 << 22));
        defer a.free(hdr_text);
        const hdr = try std.json.parseFromSlice(std.json.Value, a, hdr_text, .{});
        defer hdr.deinit();
        const layers: u32 = @intCast(hdr.value.object.get("layers").?.integer);
        const tokens: u32 = @intCast(hdr.value.object.get("prompt_tokens").?.integer);
        if (hdr.value.object.get("experts").?.integer != n_experts or hdr.value.object.get("top_k").?.integer != top_k) return error.TraceGeometry;
        const p_path = try std.fmt.bufPrint(&path_buf, "{s}.prompt.u16", .{prefix});
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p_path, a, .limited(1 << 30));
        defer a.free(bytes);
        if (bytes.len != @as(usize, layers) * tokens * top_k * 2) return error.TraceSize;
        const prompt = try a.alloc(u16, bytes.len / 2);
        @memcpy(std.mem.sliceAsBytes(prompt), bytes);
        return .{ .layers = layers, .tokens = tokens, .prompt = prompt };
    }

    fn layer(t: *const Trace, l: usize) []const u16 {
        const n = @as(usize, t.tokens) * top_k;
        return t.prompt[l * n ..][0..n];
    }
};

fn openSource(a: std.mem.Allocator, f: *OneLayer, rows: []const u32) !StreamSource {
    var opts = B.options(rows, .host);
    opts.transient_release = true;
    try B.applyReadSchedule(&opts, experts.read_schedule);
    opts.prompt_relocate = experts.prompt_relocate;
    const stream = try B.Stream.init(a, &f.sdk_bank, opts);
    errdefer stream.deinit();
    var ss = StreamSource.init(stream);
    try ss.installReadSchedule(experts.read_schedule, mimo_stream.default_read_schedule_threshold);
    return ss;
}

test "glm5_next route plan: the prompt call's host steps on the route trace's real routes, ms per layer by step (GLM5_NEXT_ROUTE_PLAN_TRACE)" {
    const prefix = std.mem.span(std.c.getenv("GLM5_NEXT_ROUTE_PLAN_TRACE") orelse return error.SkipZigTest);
    const a = std.heap.c_allocator;
    const io = testing.io;
    const trace = try Trace.load(a, io, prefix);
    defer a.free(trace.prompt);
    var f: OneLayer = undefined;
    try f.init(a, io);
    defer f.deinit();
    const rows = [_]u32{126};
    var ss = try openSource(a, &f, &rows);
    defer ss.stream.deinit();
    var w: Wide = .{};
    defer w.deinit(a);
    const reps = 5;
    // Interleaved by layer and rep (full, grouped), so both modes see the same machine state.
    var per: [2][n_steps][reps]u64 = undefined;
    var med_sum: [2][n_steps]f64 = @splat(@splat(0));
    for (0..trace.layers) |l| {
        for (0..reps) |r| for ([_]Mode{ .full, .grouped }, 0..) |mode, m| {
            var ns: [n_steps]u64 = @splat(0);
            try runCall(a, &ss, &w, trace.layer(l), mode, &ns, null);
            for (0..n_steps) |s| per[m][s][r] = ns[s];
        };
        for (0..2) |m| {
            var line: [n_steps]f64 = undefined;
            var total: f64 = 0;
            for (0..n_steps) |s| {
                std.sort.pdq(u64, &per[m][s], {}, std.sort.asc(u64));
                line[s] = @as(f64, @floatFromInt(per[m][s][reps / 2])) / 1e6;
                med_sum[m][s] += line[s];
                total += line[s];
            }
            std.debug.print("GLM5N_ROUTE_PLAN mode={t} layer={d} distinct={d:.3} count={d:.3} seed={d:.3} order={d:.3} route={d:.3} scan={d:.3} total={d:.3} ms\n", .{ @as(Mode, @enumFromInt(m)), l + 3, line[0], line[1], line[2], line[3], line[4], line[5], total });
        }
    }
    const nl: f64 = @floatFromInt(trace.layers);
    for (0..2) |m| {
        var tot: f64 = 0;
        for (med_sum[m]) |s| tot += s;
        const x = med_sum[m];
        std.debug.print("GLM5N_ROUTE_PLAN mean mode={t} distinct={d:.3} count={d:.3} seed={d:.3} order={d:.3} route={d:.3} scan={d:.3} total={d:.3} ms per layer; tokens={d} rows={d} reps={d}\n", .{ @as(Mode, @enumFromInt(m)), x[0] / nl, x[1] / nl, x[2] / nl, x[3] / nl, x[4] / nl, x[5] / nl, tot / nl, trace.tokens, rows[0], reps });
    }
}

test "glm5_next route plan: the grouped scan's plan is byte-identical to the full scan's on every traced layer (order, groups' plans, windows, hit slots, every bank pass's slots, act rows and positions)" {
    const prefix = std.mem.span(std.c.getenv("GLM5_NEXT_ROUTE_PLAN_TRACE") orelse return error.SkipZigTest);
    const a = std.heap.c_allocator;
    const io = testing.io;
    const trace = try Trace.load(a, io, prefix);
    defer a.free(trace.prompt);
    var f: OneLayer = undefined;
    try f.init(a, io);
    defer f.deinit();
    const rows = [_]u32{126};
    var ss = try openSource(a, &f, &rows);
    defer ss.stream.deinit();
    var w: Wide = .{};
    defer w.deinit(a);
    var bytes: u64 = 0;
    for (0..trace.layers) |l| {
        var want: Record = .{};
        defer want.deinit(a);
        var got: Record = .{};
        defer got.deinit(a);
        var ns: [n_steps]u64 = undefined;
        try runCall(a, &ss, &w, trace.layer(l), .full, &ns, &want);
        try runCall(a, &ss, &w, trace.layer(l), .grouped, &ns, &got);
        try testing.expectEqualSlices(u16, want.distinct.items, got.distinct.items);
        try testing.expectEqualSlices(u8, want.plans.items, got.plans.items);
        try testing.expectEqualSlices(u32, want.scan.items, got.scan.items);
        bytes += want.plans.items.len + 4 * want.scan.items.len + 2 * want.distinct.items.len;
    }
    std.debug.print("GLM5N_ROUTE_PLAN identical layers={d} bytes={d}\n", .{ trace.layers, bytes });
}

test "glm5_next route plan: bucketRowsByGroup keeps every group's rows in row order, with or without the counts" {
    // 3 groups over 5 experts; rows of 2 ids.
    const ids = [_]u16{ 4, 0, 2, 4, 1, 3, 0, 2, 4, 1 };
    const distinct = [_]u16{ 4, 0, 2, 1, 3 };
    const grp_of = [_]u16{ 0, 1, 1, 2, 0 };
    var count: [5]u32 = @splat(0);
    for (ids) |e| count[e] += 1;
    for ([_]?[]const u32{ &count, null }) |c| {
        var off: [4]u32 = undefined;
        var out: [ids.len]u32 = undefined;
        shared.bucketRowsByGroup(&ids, &distinct, &grp_of, c, &off, &out);
        try testing.expectEqualSlices(u32, &.{ 0, 5, 9, 10 }, &off);
        // Group 0 = experts 4, 0; group 1 = 2, 1; group 2 = 3: each group's id positions ascending.
        try testing.expectEqualSlices(u32, &.{ 0, 1, 3, 6, 8, 2, 4, 7, 9, 5 }, &out);
    }
}
