//! K6 `decode_commits` / K7 `decode_miss_issue` CPU proofs (mimo/k6-commits, decode_barrier.zig). No device, no lock:
//! the trace backend over MiMo's real stream source (host slot rows, the lookahead class, a host event: MiMo's gated
//! route), and the MLX CPU stream (the poll and in-place reads against the host route's; the scheduler's decode lane end
//! to end under every route pair).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const shared = @import("../deepseek_v41/experts.zig");
const stream_mod = @import("stream.zig");
const source_mod = @import("source.zig");
const prefill_math = @import("math.zig");
const decode_barrier = @import("decode_barrier.zig");
const vl = @import("verify_lane_test.zig");

const testing = std.testing;
const Tr = ops.TraceOps;
const StreamSource = stream_mod.StreamSource;
const top_k = vl.top_k;
const n_experts = vl.n_experts;

const Arm = enum { hitpass, fused };
const Pair = struct { commits: decode_barrier.Commits, issue: decode_barrier.MissIssue };
const pairs = [_]Pair{
    .{ .commits = .mlx, .issue = .host },
    .{ .commits = .layer, .issue = .host },
    .{ .commits = .mlx, .issue = .event },
    .{ .commits = .layer, .issue = .event },
};

/// Decode-lane calls per run: layers 1 and 2 alternating, M = 4 (the verify) every third pair, M = 1 (the serial token)
/// otherwise; a decode-phase wide call (M = 5, `Source.lutAfterWide`'s whole LUT rewrite) after call `wide_after`.
const n_calls = 36;
const wide_after = 13;

fn xorshift(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

/// Each row's top-8: distinct experts of 64, drawn so that rows overlap and the 8 resident rows per layer churn
/// (admissions and evictions on most calls).
fn drawRows(state: *u32, m: usize, out: []u16) void {
    for (0..m) |r| {
        var seen: [n_experts]bool = @splat(false);
        var j: usize = 0;
        while (j < top_k) {
            // a hot band (experts 0..15) half the time: hits and repeats across rows
            const e: u16 = @intCast(if (xorshift(state) % 2 == 0) xorshift(state) % 16 else xorshift(state) % n_experts);
            if (seen[e]) continue;
            seen[e] = true;
            out[r * top_k + j] = e;
            j += 1;
        }
    }
}

/// One decode call's census (what the routes change by design).
const Census = struct {
    layer: u32,
    m: u32,
    c0: usize,
    c1: usize,
    evals: usize,
    host_reads: usize,
    log_from: usize,
    log_to: usize,
};

/// Everything the routes must leave identical: per call the stream's host-side counters, every live route's plan
/// (ids' slots, hits, misses, loads, evictions, placement order, parts), every routed layer's slots and the LUT the
/// layer's next call reads.
const Facts = std.ArrayList(u64);

fn put(f: *Facts, a: std.mem.Allocator, v: anytype) !void {
    try f.append(a, @as(u64, @intCast(v)));
}

fn recordFacts(f: *Facts, a: std.mem.Allocator, ss: *StreamSource) !void {
    // The host-side counters only (the pool's pre-read / speculation counters move with the read workers' timing).
    const st = ss.stats();
    inline for (.{ "route_calls", "expert_cache_hits", "expert_cache_misses", "expert_cache_evictions", "persistent_loads", "transient_loads", "loads_skipped", "gates" }) |name| try put(f, a, @field(st, name));
    for (&ss.stream.routes) |*r| {
        if (r.state == .free) continue;
        try put(f, a, r.layer);
        try put(f, a, r.plan.n_ids);
        for (r.plan.slots[0..r.plan.n_ids]) |s| try put(f, a, s);
        for (r.plan.hitsOf()) |e| try put(f, a, e);
        try put(f, a, @as(u64, 0xffff));
        for (r.plan.missesOf()) |e| try put(f, a, e);
        try put(f, a, @as(u64, 0xffff));
        for (r.plan.loadsOf()) |l| {
            try put(f, a, l.expert);
            try put(f, a, l.slot);
            try put(f, a, @intFromBool(l.persistent));
        }
        for (r.plan.evictionsOf()) |ev| {
            try put(f, a, ev.slot);
            try put(f, a, ev.previous);
            try put(f, a, ev.next);
        }
        for (r.order[0..r.plan.n_loads], r.reads[0..r.plan.n_loads]) |o, rd| {
            try put(f, a, o);
            try put(f, a, @intFromBool(rd));
        }
        try put(f, a, r.n_parts);
        for (r.partsOf()) |p| {
            try put(f, a, p.first);
            try put(f, a, p.n);
            try put(f, a, p.n_reads);
        }
    }
    for (1..3) |l| {
        const pol = ss.policy(@intCast(l));
        for (pol.slot_to_expert[0..pol.capacity]) |e| try put(f, a, e);
    }
}

/// The hit pass's next LUT of `layer` (the one its next call reads) against the policy's whole map; returns it as facts.
fn checkLut(f: *Facts, a: std.mem.Allocator, ss: *const StreamSource, ex: anytype, g: *Tr, layer: u32) !void {
    var want: [n_experts]u32 = undefined;
    ss.residentLut(layer, &want);
    const got = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[layer][ex.dev_parity[layer]]))));
    try testing.expectEqualSlices(u32, &want, got);
    for (got) |v| try put(f, a, v);
}

const Run = struct {
    facts: Facts = .empty,
    census: [n_calls]Census = undefined,
    log: decode_barrier.TraceLog,
    lut_entries: u64 = 0,

    fn deinit(r: *Run, a: std.mem.Allocator) void {
        r.facts.deinit(a);
        r.log.deinit();
    }
};

fn routeRun(a: std.mem.Allocator, arm: Arm, pair: Pair, run: *Run) !void {
    const rows = [_]u32{ 8, 8 };
    const f = try a.create(vl.Fixture);
    defer a.destroy(f);
    try f.initWith(a, 32, 32, &rows, .host, true, true);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    if (pair.issue == .event) try ss.installLutDelta();
    var g = Tr.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = shared.ExpertsWith(Tr, StreamSource, vl.LaneMath, .{ .prefill = true, .gated = true, .lookahead = true });
    const c: @import("config.zig").Config = .{ .hidden_size = 32, .moe_intermediate_size = 32, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 32, n_experts }, .float32), .bias = try g.input(&.{n_experts}, .float32) };
    var math: vl.LaneMath = .{ .hidden = 32, .inter = 32, .barrier = try decode_barrier.Routes(Tr).of(pair.commits, pair.issue) };
    math.barrier.log = &run.log;
    var ex = try Ex.initWith(a, &g, &ss, math, &c, .{ .gates = &gates, .banked = true, .devroute = arm == .hitpass, .wide = .{ .seed = true, .hot_first = true, .depth = stream_mod.wide_depth } });
    defer ex.deinit();

    // The script: the prompt's M = 4 chunk, then every decode call's rows (and the decode-phase wide call's).
    var bufs: [n_calls + 2][5 * top_k]u16 = undefined;
    var calls: [n_calls + 2][]const u16 = undefined;
    var state: u32 = 0x6b36_c0de;
    drawRows(&state, 4, &bufs[0]);
    calls[0] = bufs[0][0 .. 4 * top_k];
    var k: usize = 1;
    for (0..n_calls) |ci| {
        const m: usize = if ((ci / 2) % 3 == 0) 4 else 1;
        drawRows(&state, m, &bufs[k]);
        calls[k] = bufs[k][0 .. m * top_k];
        k += 1;
        if (ci == wide_after) {
            drawRows(&state, 5, &bufs[k]);
            calls[k] = bufs[k][0 .. 5 * top_k];
            k += 1;
        }
    }
    var script: vl.Script = .{ .calls = calls[0..k] };
    g.host_values = script.values();

    const x4 = try g.input(&.{ 4, 32 }, .bfloat16);
    const idx4 = try g.input(&.{ 4, 8 }, .int32);
    _ = try ex.at(1).routedParts(&g, x4, idx4);
    ex.releaseParts(&g);
    try ex.flush();
    try ex.grow(&g, &rows);
    try testing.expect(ss.decoding());

    for (0..n_calls) |ci| {
        const layer: u32 = if (ci % 2 == 0) 1 else 2;
        const m: u32 = if ((ci / 2) % 3 == 0) 4 else 1;
        const x = try g.input(&.{ @intCast(m), 32 }, .bfloat16);
        const idx = try g.input(&.{ @intCast(m), 8 }, .int32);
        const n0 = g.nodes.items.len;
        const cs: Census = .{ .layer = layer, .m = m, .c0 = g.commit_spans.items.len, .c1 = 0, .evals = g.evals.items.len, .host_reads = 0, .log_from = run.log.entries.items.len, .log_to = 0 };
        const out = try ex.at(layer).routed(&g, x, idx);
        try testing.expectEqualSlices(c_int, &.{ @intCast(m), 8, 32 }, g.shapeOf(out).slice());
        var reads: usize = 0;
        for (g.nodes.items[n0..]) |nd| reads += @intFromBool(nd.op == .host_read);
        run.census[ci] = cs;
        run.census[ci].c1 = g.commit_spans.items.len;
        run.census[ci].evals = g.evals.items.len - cs.evals;
        run.census[ci].host_reads = reads;
        run.census[ci].log_to = run.log.entries.items.len;
        try recordFacts(&run.facts, a, &ss);
        if (arm == .hitpass) try checkLut(&run.facts, a, &ss, &ex, &g, layer);
        if (layer == 2) try ex.flush();
        if (ci == wide_after) {
            // A decode-phase wide call (no decode caller makes one; `Source.forward` rewrites its layer's LUT whole after
            // it, `lutAfterWide`): the delta's next rewrites of that layer must still land on the policy's map.
            const x5 = try g.input(&.{ 5, 32 }, .bfloat16);
            const idx5 = try g.input(&.{ 5, 8 }, .int32);
            _ = try ex.at(1).routed(&g, x5, idx5);
            if (arm == .hitpass) try source_mod.lutAfterRoute(&ex, &g, 1);
            try ex.flush();
            try recordFacts(&run.facts, a, &ss);
            if (arm == .hitpass) try checkLut(&run.facts, a, &ss, &ex, &g, 1);
        }
    }
    try ex.flush();
}

fn expectStructure(arm: Arm, pair: Pair, run: *const Run) !void {
    for (run.census) |cs| {
        const log = run.log.entries.items[cs.log_from..cs.log_to];
        const commits = cs.c1 - cs.c0;
        const has_scores = cs.layer == 1; // the last routed layer has no next router to predict
        // The commits: the hit pass's barrier set and device wave (2); off the hit pass the event route commits the set
        // ahead of its poll (1), and the host hit wave adds one when the call has hits.
        switch (arm) {
            .hitpass => try testing.expectEqual(@as(usize, 2), commits),
            .fused => {
                const ahead: usize = @intFromBool(pair.issue == .event);
                try testing.expect(commits == ahead or commits == ahead + 1);
            },
        }
        var n_poll: usize = 0;
        var n_reads: usize = 0;
        var n_scopes: usize = 0;
        for (log) |e| switch (e.what) {
            .poll => n_poll += 1,
            .read_ids, .read_scores => n_reads += 1,
            .scope_begin => n_scopes += 1,
            .scope_end => {},
        };
        switch (pair.issue) {
            .host => {
                // MLX's eval + wait + copy: the scores' read (layer 1), the ids' read; the set's wait (an evalAll) with scores.
                try testing.expectEqual(@as(usize, if (has_scores) 2 else 1), cs.host_reads);
                try testing.expectEqual(@as(usize, @intFromBool(has_scores)), cs.evals);
                try testing.expectEqual(@as(usize, 0), n_poll + n_reads);
            },
            .event => {
                // No MLX eval, wait or host read: one poll after the set's commit, then the in-place reads.
                try testing.expectEqual(@as(usize, 0), cs.host_reads);
                try testing.expectEqual(@as(usize, 0), cs.evals);
                try testing.expectEqual(@as(usize, 1), n_poll);
                try testing.expectEqual(@as(usize, if (has_scores) 2 else 1), n_reads);
                var at_poll: usize = 0;
                for (log, 0..) |e, i| if (e.what == .poll) {
                    at_poll = e.commits;
                    // the reads follow the poll, ids last (the route reads them next)
                    try testing.expect(i + n_reads < log.len + 1);
                    try testing.expectEqual(decode_barrier.Primitive.read_ids, log[i + n_reads].what);
                    if (has_scores) try testing.expectEqual(decode_barrier.Primitive.read_scores, log[i + 1].what);
                };
                // the set (and the hit pass's wave) are committed before the poll
                try testing.expectEqual(cs.c0 + (if (arm == .hitpass) @as(usize, 2) else 1), at_poll);
            },
            .invalid => unreachable,
        }
        switch (pair.commits) {
            .mlx => try testing.expectEqual(@as(usize, 0), n_scopes),
            .layer => {
                // Every commit of the call inside a scope (a scope never left open, never nested).
                try testing.expect(n_scopes >= 1);
                var open: ?usize = null;
                var covered: usize = 0;
                for (log) |e| switch (e.what) {
                    .scope_begin => {
                        try testing.expect(open == null);
                        open = e.commits;
                    },
                    .scope_end => {
                        covered += e.commits - open.?;
                        open = null;
                    },
                    else => {},
                };
                try testing.expect(open == null);
                try testing.expectEqual(commits, covered);
            },
            .invalid => unreachable,
        }
    }
}

test "mimo barrier routes: the four route pairs leave every route, plan, slot and next LUT identical; commits per call, scopes, polls and reads as designed (trace backend, MiMo's gated stream source, M = 1 and 4)" {
    const a = testing.allocator;
    for ([_]Arm{ .hitpass, .fused }) |arm| {
        var runs: [pairs.len]Run = undefined;
        for (&runs) |*r| r.* = .{ .log = .{ .a = a } };
        defer for (&runs) |*r| r.deinit(a);
        for (pairs, &runs) |p, *r| try routeRun(a, arm, p, r);
        for (pairs, &runs) |p, *r| {
            try expectStructure(arm, p, r);
            // Exactness: the same records, in the same order, from the same ids, as the control.
            try testing.expectEqualSlices(u64, runs[0].facts.items, r.facts.items);
        }
        var commits_m1: usize = 0;
        var commits_m4: usize = 0;
        for (runs[3].census) |cs| {
            if (cs.m == 1) commits_m1 += cs.c1 - cs.c0 else commits_m4 += cs.c1 - cs.c0;
        }
        std.debug.print("MIMO_K6_TRACE arm={s} calls={d} facts={d} commits(layer+event): M1={d} over {d} calls, M4={d} over {d} calls; every pair's facts identical to the control's\n", .{ @tagName(arm), n_calls, runs[0].facts.items.len, commits_m1, n_calls - n_calls / 3, commits_m4, n_calls / 3 });
    }
}

test "mimo barrier routes: the LUT delta keeps every next LUT the policy's map over 400 calls of admissions, evictions and decode-phase wide calls (trace backend, hit pass)" {
    const a = testing.allocator;
    const rows = [_]u32{ 4, 4 };
    const f = try a.create(vl.Fixture);
    defer a.destroy(f);
    try f.initWith(a, 32, 32, &rows, .host, false, false);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    try ss.installLutDelta();
    var g = Tr.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = shared.ExpertsWith(Tr, StreamSource, vl.LaneMath, .{ .prefill = true });
    const c: @import("config.zig").Config = .{ .hidden_size = 32, .moe_intermediate_size = 32, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    const math: vl.LaneMath = .{ .hidden = 32, .inter = 32, .barrier = try decode_barrier.Routes(Tr).of(.layer, .event) };
    var ex = try Ex.initWith(a, &g, &ss, math, &c, .{ .banked = true, .devroute = true });
    defer ex.deinit();
    const n = 400;
    var bufs: [n][5 * top_k]u16 = undefined;
    var calls: [n][]const u16 = undefined;
    var state: u32 = 0x1a7e_5eed;
    for (0..n) |i| {
        const m: usize = if (i % 37 == 36) 5 else if (i % 3 == 0) 4 else 1;
        drawRows(&state, m, &bufs[i]);
        calls[i] = bufs[i][0 .. m * top_k];
    }
    var script: vl.Script = .{ .calls = &calls };
    g.host_values = script.values();
    try ex.grow(&g, &rows);
    var writes: u64 = 0;
    var changed: u64 = 0;
    for (0..n) |i| {
        const layer: u32 = if (i % 2 == 0) 1 else 2;
        const m: c_int = @intCast(calls[i].len / top_k);
        const before = ss.stats();
        const rel = layer - 1;
        _ = try ex.at(layer).routed(&g, try g.input(&.{ m, 32 }, .bfloat16), try g.input(&.{ m, 8 }, .int32));
        if (m > 4) try source_mod.lutAfterRoute(&ex, &g, layer);
        const after = ss.stats();
        changed += (after.persistent_loads - before.persistent_loads) + (after.expert_cache_evictions - before.expert_cache_evictions);
        writes += ss.lut_dirty[rel][ex.dev_parity[layer] +% 1].count();
        var want: [n_experts]u32 = undefined;
        ss.residentLut(layer, &want);
        try testing.expectEqualSlices(u32, &want, std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[layer][ex.dev_parity[layer]])))));
        try ex.flush();
    }
    try testing.expect(changed > 100);
    std.debug.print("MIMO_K7_LUT calls={d} slot_changes={d} lut_equal_after_every_call=true pending_entries_other_lut={d} (whole-map rewrite: {d} entries per call)\n", .{ n, changed, writes, n_experts });
}

test "mimo barrier routes: a barrier set committed ahead of its call (a row-group schedule's first half) takes the call's packaging: one scoped commit under layer, a plain one under mlx (trace backend)" {
    const a = testing.allocator;
    for ([_]decode_barrier.Commits{ .mlx, .layer }) |commits| {
        var g = Tr.init(a);
        defer g.deinit();
        var log: decode_barrier.TraceLog = .{ .a = a };
        defer log.deinit();
        var r = try decode_barrier.Routes(Tr).of(commits, .event);
        r.log = &log;
        const ids = try g.input(&.{ 2, 8 }, .int32);
        const sc = try g.input(&.{ 2, 64 }, .float32);
        try r.commitSet(&g, &.{ ids, sc });
        try testing.expectEqual(@as(usize, 1), g.commit_spans.items.len);
        try testing.expectEqualSlices(u32, &.{ ids, sc }, g.committed.items);
        switch (commits) {
            .layer => {
                try testing.expectEqual(@as(usize, 2), log.entries.items.len);
                try testing.expectEqual(decode_barrier.Primitive.scope_begin, log.entries.items[0].what);
                try testing.expectEqual(@as(usize, 0), log.entries.items[0].commits);
                try testing.expectEqual(decode_barrier.Primitive.scope_end, log.entries.items[1].what);
                try testing.expectEqual(@as(usize, 1), log.entries.items[1].commits);
            },
            else => try testing.expectEqual(@as(usize, 0), log.entries.items.len),
        }
    }
}

// ── MLX CPU stream ──

/// A router-shaped barrier on the CPU stream (`module.route`'s ids: F32 scores, argpartition top-8 sliced, uint32 ->
/// int32; the predictor's F32 [M, 64] scores): fresh arrays per call, so each route evaluates its own.
fn barrierGraph(g: *ops.MlxOps, x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, m: c_int) ![3]mlx.mlx_array {
    const scores = try g.add(try g.sigmoid(try g.matmul(try g.astype(x, .float32), w)), bias);
    var part = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_argpartition_axis(&part, try g.neg(scores), top_k - 1, -1, g.s));
    const sel = try g.slice(try g.adopt(part), &.{ 0, 0 }, &.{ m, top_k }, &.{ 1, 1 });
    const ids = try g.astype(try g.astype(sel, .uint32), .int32);
    const pred = try g.add(try g.sigmoid(try g.matmul(try g.astype(x, .float32), try g.transpose(w))), bias);
    const dev = try g.matmul(try g.astype(x, .float32), w);
    return .{ ids, pred, dev };
}

test "mimo barrier routes CPU: the event route's poll and in-place reads return the host route's ids and scores bit for bit; the layer scope raises and restores the device's commit limits (MLX CPU stream)" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    // The limits: a scope's raise is visible through MLX's own getter and the previous limits come back.
    const before = try decode_barrier.limits();
    const prev = try decode_barrier.swapLimits(decode_barrier.raised);
    const inside = try decode_barrier.limits();
    _ = try decode_barrier.swapLimits(prev);
    try testing.expectEqual(before, prev);
    try testing.expectEqual(decode_barrier.raised, inside);
    try testing.expectEqual(before, try decode_barrier.limits());
    try testing.expectEqual(.cpu_stream, try decode_barrier.selfCheck(s));
    var state: u32 = 0x00c0_ffee;
    var wbuf: [64 * n_experts]f32 = undefined;
    for (&wbuf) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(xorshift(&state) % 2001)) - 1000)) / 1000;
    var bbuf: [n_experts]f32 = undefined;
    for (&bbuf) |*v| v.* = @as(f32, @floatFromInt(xorshift(&state) % 101)) / 1000;
    const w = g.keep(try g.hostArray(std.mem.sliceAsBytes(&wbuf), &.{ 64, n_experts }, .float32));
    defer g.release(w);
    const bias = g.keep(try g.hostArray(std.mem.sliceAsBytes(&bbuf), &.{n_experts}, .float32));
    defer g.release(bias);
    const host = decode_barrier.Routes(ops.MlxOps).control;
    for ([_]decode_barrier.Commits{ .mlx, .layer }) |commits| for ([_]c_int{ 1, 4 }) |m| for ([_]bool{ false, true }) |with_dev| {
        var xbuf: [4 * 64]f32 = undefined;
        for (&xbuf) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(xorshift(&state) % 2001)) - 1000)) / 500;
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xbuf[0 .. @as(usize, @intCast(m)) * 64]), &.{ m, 64 }, .float32), .bfloat16);
        const ev = try decode_barrier.Routes(ops.MlxOps).of(commits, .event);
        const nid: usize = @as(usize, @intCast(m)) * top_k;
        const nsc: usize = @as(usize, @intCast(m)) * n_experts;
        // the control: the host route on its own graph
        const ga = try barrierGraph(&g, x, w, bias, m);
        var ids_a: [4 * top_k]u16 = undefined;
        var sc_a: [4 * n_experts]f32 = undefined;
        var scores_a: []const f32 = &.{};
        try host.commit(&g, &.{ ga[0], ga[1] }, if (with_dev) ga[2] else null);
        const got_a = try host.read(&g, &.{ ga[0], ga[1] }, ga[0], ga[1], ids_a[0..nid], sc_a[0..nsc], &scores_a);
        // the event route on an identical graph: committed, polled, read in place
        const gb = try barrierGraph(&g, x, w, bias, m);
        var ids_b: [4 * top_k]u16 = undefined;
        var sc_b: [4 * n_experts]f32 = undefined;
        var scores_b: []const f32 = &.{};
        try ev.commit(&g, &.{ gb[0], gb[1] }, if (with_dev) gb[2] else null);
        const got_b = try ev.read(&g, &.{ gb[0], gb[1] }, gb[0], gb[1], ids_b[0..nid], sc_b[0..nsc], &scores_b);
        try testing.expectEqualSlices(u16, got_a, got_b);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(scores_a), std.mem.sliceAsBytes(scores_b));
        try testing.expectEqual(nsc, scores_b.len);
        // The poll left the set available: a later host read is a wait only, and reads the same values.
        for ([_]mlx.mlx_array{ gb[0], gb[1] }) |y| {
            var ok = false;
            try mlx.check(mlx._mlx_array_is_available(&ok, y));
            try testing.expect(ok);
        }
        var ids_c: [4 * top_k]u16 = undefined;
        try testing.expectEqualSlices(u16, got_b, try g.hostIds(gb[0], ids_c[0..nid]));
        try testing.expectEqual(before, try decode_barrier.limits());
        g.reset();
    };
}

test "mimo barrier routes CPU: the scheduler's decode lane under every route pair returns the control's rows bit for bit with the same routes and stream counters (MLX CPU stream, M = 1..4, stock math)" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const H: u32 = 64;
    const I: u32 = 64;
    const rows = [_]u32{ 8, 8 };
    var control: [4][]u32 = undefined;
    var control_counters: [4][6]u64 = undefined;
    var have_control = false;
    defer if (have_control) for (control) |x| a.free(x);
    for (pairs) |p| {
        var g = try ops.MlxOps.init(a, s);
        defer g.deinit();
        const f = try a.create(vl.Fixture);
        defer a.destroy(f);
        try f.init(a, H, I, &rows, .{ .mlx = s }, false);
        defer f.deinit(a);
        var ss = StreamSource.init(f.stream);
        if (p.issue == .event) try ss.installLutDelta();
        const Ex = shared.ExpertsWith(ops.MlxOps, StreamSource, prefill_math.Math, .{ .prefill = true });
        const c: @import("config.zig").Config = .{ .hidden_size = H, .moe_intermediate_size = I, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
        var math = try prefill_math.Math.init(a, f.geom, source_mod.expertActivation);
        math.barrier = try decode_barrier.Routes(ops.MlxOps).of(p.commits, p.issue);
        var ex = try Ex.initWith(a, &g, &ss, math, &c, .{});
        defer {
            ex.math.deinit(&g);
            ex.deinit();
        }
        try ex.grow(&g, &rows);
        var state: u32 = 0x7e51_f00d;
        var xin: [4 * H]f32 = undefined;
        for (&xin) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(xorshift(&state) % 2001)) - 1000)) / 4000;
        var idsin: [4 * top_k]i32 = undefined;
        for (vl.verify_ids, 0..) |r, t| for (r, 0..) |e, j| {
            idsin[t * top_k + j] = e;
        };
        var counters: [4][6]u64 = undefined;
        for ([_]usize{ 1, 2, 3, 4 }, 0..) |m, mi| {
            const st0 = ss.stats();
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin[0 .. m * H]), &.{ @intCast(m), @intCast(H) }, .float32), .bfloat16);
            const id = try g.hostArray(std.mem.sliceAsBytes(idsin[0 .. m * top_k]), &.{ @intCast(m), top_k }, .int32);
            const routed = try ex.at(1).routed(&g, x, id);
            try g.evalAll(&.{routed});
            try ex.flush();
            const st1 = ss.stats();
            counters[mi] = .{ st1.route_calls - st0.route_calls, st1.expert_cache_hits - st0.expert_cache_hits, st1.expert_cache_misses - st0.expert_cache_misses, st1.persistent_loads - st0.persistent_loads, st1.transient_loads - st0.transient_loads, st1.expert_cache_evictions - st0.expert_cache_evictions };
            const n: usize = @intCast(g.shapeOf(routed).numel());
            const vals = try a.alloc(f32, n);
            defer a.free(vals);
            _ = try g.hostF32(try g.astype(routed, .float32), vals);
            const bits = try a.alloc(u32, n);
            for (vals, bits) |v, *b| b.* = @bitCast(v);
            if (p.commits == .mlx and p.issue == .host) {
                // pairs[0]: the control's rows and counters, kept for the other pairs
                control[mi] = bits;
                control_counters[mi] = counters[mi];
                if (mi == 3) have_control = true;
            } else {
                defer a.free(bits);
                try testing.expectEqualSlices(u32, control[mi], bits);
                try testing.expectEqualSlices(u64, &control_counters[mi], &counters[mi]);
            }
            g.reset();
        }
        std.debug.print("MIMO_K6_CPU commits={s} issue={s} rows_bit_equal_to_control=true counters(route,hits,misses,persistent,transient,evictions) M1={any} M4={any}\n", .{ @tagName(p.commits), @tagName(p.issue), counters[0], counters[3] });
    }
}
