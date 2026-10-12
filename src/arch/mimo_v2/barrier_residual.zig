//! MIMO_BARRIER_RESIDUAL (profile builds only; `sdk.plugin_profile`, the MTP profile's switch): where the verify's
//! host-side routing barrier time goes, per routed layer, from the GPU timeline's own rows (`dsv41_verify_timeline`,
//! read after `settle`). In every other build the storage is `void` and every entry compiles to nothing.
//!
//! Per verify and routed layer L (host stamps of `Experts.run`: call, barrier, route, hit, end) the host span from the
//! previous layer's last submit (its `end`; the verify's begin for the first) to L's barrier return is split, exactly,
//! into the GPU's busy union inside it and its idle rest:
//!   gpu_prev     buffers committed before L's call (the previous layer's hit wave and miss parts, its combine's tail)
//!   gpu_bar      the barrier set: committed by the return and complete before it (residual, attention, router,
//!                predictor and the previous layer's join / combine, in MLX's command-buffer splits)
//!   gpu_wave     L's later buffers running before the return (the hit wave committed ahead of the wait)
//!   gpu_gated    rows the timeline repaired (wait-only buffers of gated miss parts: an event wait, not kernels)
//!   idle_queued  GPU idle while the next buffer to run was committed and not started (a command-buffer boundary)
//!   idle_starved GPU idle before the next buffer to run was committed (the host behind the GPU)
//! host = gpu (the union; the classes may overlap) + idle_queued + idle_starved, by construction. Also: `call_barrier`
//! (L's call to its return: DECODE_TIMERS' barrier bucket, which the summary reconciles with the MTP profile's
//! `verify_barrier_ms`), `wake` (the barrier set's last GPU end to the host's return), the buffers committed per layer
//! and, of them, the barrier set's. Host reads per verify (`noteRoute`, counted between `verifyBegin` / `verifyEnd`):
//! one routed-ids read and one barrier wait per route; one predictor-scores read per route but the last routed layer's
//! (`ExpertsWith.predictorGate`: the decode-only lookahead scores the NEXT layer's router, read in the same wait); plus
//! the verify's one decision read (its logits) per round.
const std = @import("std");
const profile = @import("profile.zig");
const mimo = @import("config.zig");
const timeline = @import("../deepseek_v41/verify_timeline.zig");

pub const enabled = profile.enabled;

const Row = timeline.Row;
const Layer = timeline.Layer;
const Verify = timeline.Verify;
const Phase = timeline.Phase;

/// Per routed layer kind (MiMo's `layer_types`), then the round's totals.
pub const Kind = enum(u1) { sliding, full };

/// Sums (ns, counts) over the layers (or rounds) they cover.
pub const Acc = struct {
    n: u64 = 0,
    host: u64 = 0,
    call_barrier: u64 = 0,
    gpu: u64 = 0,
    gpu_prev: u64 = 0,
    gpu_bar: u64 = 0,
    gpu_wave: u64 = 0,
    gpu_gated: u64 = 0,
    idle_queued: u64 = 0,
    idle_starved: u64 = 0,
    wake: u64 = 0,
    commits: u64 = 0,
    bar_commits: u64 = 0,

    fn add(a: *Acc, b: Acc) void {
        inline for (@typeInfo(Acc).@"struct".field_names) |f| @field(a, f) += @field(b, f);
    }
};

/// One verify's totals (ns): its wall, the routed layers' sums, the span before the first call and after the last end.
pub const Round = struct { wall: u64 = 0, pre: u64 = 0, post: u64 = 0, layers: Acc = .{} };

const Class = enum(u3) { prev, bar, wave, gated };
const Span = struct { a: u64, b: u64, class: Class };

/// The route counter (profile builds): between the verify's begin and end.
const Reads = struct { in_verify: bool = false, routes: u64 = 0, verifies: u64 = 0 };
var reads: if (enabled) Reads else void = if (enabled) .{} else {};

pub inline fn verifyBegin() void {
    if (comptime !enabled) return;
    reads.in_verify = true;
    reads.verifies += 1;
}

pub inline fn verifyEnd() void {
    if (comptime enabled) reads.in_verify = false;
}

/// One decode-phase route (`mtp_profile.unionNote`, from `StreamSource.route` and its missfree twin): its routing
/// barrier read the routed ids (and the predictor's scores, but on the last routed layer) in one wait.
pub inline fn noteRoute() void {
    if (comptime !enabled) return;
    if (!reads.in_verify) return;
    reads.routes += 1;
}

pub fn resetReads() void {
    if (comptime enabled) reads = .{};
}

fn kindOf(layer: usize) Kind {
    const c: mimo.Config = .{};
    return if (layer < c.layer_types.len and c.layer_types[layer] == .full) .full else .sliding;
}

fn t(l: Layer, p: timeline.Point) u64 {
    return l.t[@intFromEnum(p)];
}

/// The union length of `spans` (sorted by start) restricted to `class` (null: every class).
fn unionOf(spans: []const Span, class: ?Class) u64 {
    var busy: u64 = 0;
    var cur: ?[2]u64 = null;
    for (spans) |sp| {
        if (class) |c| if (sp.class != c) continue;
        if (cur) |cv| {
            if (sp.a <= cv[1]) {
                cur = .{ cv[0], @max(cv[1], sp.b) };
                continue;
            }
            busy += cv[1] - cv[0];
        }
        cur = .{ sp.a, sp.b };
    }
    if (cur) |cv| busy += cv[1] - cv[0];
    return busy;
}

/// One verify's routed layers over its rows `rs` (that cycle's verify-phase rows, commit order). `scratch` holds the
/// clipped spans of one layer's host span (more than fit are dropped from the GPU sums and counted in `truncated`).
pub fn addVerify(rs: []const Row, v: Verify, ls: []const Layer, per: *[2]Acc, scratch: []Span, truncated: *u64) Round {
    var round: Round = .{};
    if (v.end <= v.begin) return round;
    round.wall = v.end - v.begin;
    var prev_end: u64 = v.begin;
    var first = true;
    for (ls, 0..) |l, li| {
        const call = t(l, .call);
        const barrier = t(l, .barrier);
        const end = t(l, .end);
        if (call == 0 or barrier < call or end < barrier) continue;
        if (first) round.pre = call -| v.begin;
        first = false;
        var acc: Acc = .{ .n = 1 };
        const lo = @min(prev_end, call);
        const hi = barrier;
        acc.host = hi - lo;
        acc.call_barrier = barrier - call;
        var n: usize = 0;
        var bar_end: u64 = 0;
        for (rs) |r| {
            if (r.host_commit > lo and r.host_commit <= end) {
                acc.commits += 1;
            }
            if (r.host_commit > hi) continue;
            const ended = r.gpu_end != 0 and r.gpu_end <= hi;
            const class: Class = if (r.status & timeline.repaired_status != 0) .gated else if (r.host_commit < call) .prev else if (ended) .bar else .wave;
            if (class == .bar or (class == .gated and r.host_commit >= call and ended)) {
                acc.bar_commits += @intFromBool(class == .bar);
                bar_end = @max(bar_end, r.gpu_end);
            }
            if (r.gpu_end <= r.gpu_start) continue;
            const a = @max(r.gpu_start, lo);
            const b = @min(r.gpu_end, hi);
            if (b <= a) continue;
            if (n == scratch.len) {
                truncated.* += 1;
                continue;
            }
            scratch[n] = .{ .a = a, .b = b, .class = class };
            n += 1;
        }
        const spans = scratch[0..n];
        std.mem.sort(Span, spans, {}, struct {
            fn lt(_: void, x: Span, y: Span) bool {
                return x.a < y.a;
            }
        }.lt);
        acc.gpu = unionOf(spans, null);
        acc.gpu_prev = unionOf(spans, .prev);
        acc.gpu_bar = unionOf(spans, .bar);
        acc.gpu_wave = unionOf(spans, .wave);
        acc.gpu_gated = unionOf(spans, .gated);
        // The idle rest, gap by gap: queued when a buffer committed by the gap's start began only at or after its end.
        var cursor = lo;
        var i: usize = 0;
        while (cursor < hi) {
            // the next busy interval of the union at or after `cursor`
            var nb: ?[2]u64 = null;
            while (i < spans.len) : (i += 1) {
                if (spans[i].b <= cursor) continue;
                var s0 = @max(spans[i].a, cursor);
                var s1 = spans[i].b;
                var j = i + 1;
                while (j < spans.len and spans[j].a <= s1) : (j += 1) s1 = @max(s1, spans[j].b);
                if (s0 < cursor) s0 = cursor;
                nb = .{ s0, s1 };
                i = j;
                break;
            }
            const gap_end = if (nb) |x| x[0] else hi;
            if (gap_end > cursor) {
                // The buffers that ran after the gap: the earliest of their commits splits it into starved (nothing
                // committed yet) and queued (committed, not yet started: a command-buffer boundary, a launch).
                var first_commit: u64 = std.math.maxInt(u64);
                for (rs) |r| if (r.gpu_end > r.gpu_start and r.gpu_start >= gap_end) {
                    first_commit = @min(first_commit, r.host_commit);
                };
                const starved = @min(gap_end, @max(first_commit, cursor)) - cursor;
                acc.idle_starved += starved;
                acc.idle_queued += (gap_end - cursor) - starved;
            }
            cursor = if (nb) |x| @max(x[1], gap_end) else hi;
        }
        acc.wake = if (bar_end != 0) hi -| bar_end else 0;
        per[@intFromEnum(kindOf(li))].add(acc);
        round.layers.add(acc);
        prev_end = end;
    }
    round.post = if (!first) v.end -| prev_end else 0;
    return round;
}

const max_cycles = timeline.max_cycles;
const span_cap = 4096;
const State = struct {
    starts: [max_cycles + 1]u32 = undefined,
    index: [timeline.max_buffers]u32 = undefined,
    picked: [span_cap * 4]Row = undefined,
    spans: [span_cap]Span = undefined,
};
var state: if (enabled) State else void = undefined;

/// The verify-phase rows of each cycle, in commit order (one pass; `out` borrows the static index).
fn byCycle(rs: []const Row, n_cycles: usize) void {
    @memset(state.starts[0 .. n_cycles + 1], 0);
    for (rs) |r| {
        if (timeline.rowPhase(r) != @intFromEnum(Phase.verify)) continue;
        const c = timeline.rowCycle(r) orelse continue;
        if (c < n_cycles) state.starts[c + 1] += 1;
    }
    for (1..n_cycles + 1) |c| state.starts[c] += state.starts[c - 1];
    var fill: [max_cycles]u32 = undefined;
    @memcpy(fill[0..n_cycles], state.starts[0..n_cycles]);
    for (rs, 0..) |r, ri| {
        if (timeline.rowPhase(r) != @intFromEnum(Phase.verify)) continue;
        const c = timeline.rowCycle(r) orelse continue;
        if (c >= n_cycles) continue;
        state.index[fill[c]] = @intCast(ri);
        fill[c] += 1;
    }
}

/// The summary over every stored verify (after `timeline.settle`), with the MTP profile's mean `verify_barrier_ms` to
/// reconcile against. One `MIMO_BARRIER_RESIDUAL {...}` line.
pub fn line(buf: []u8, profile_barrier_ms: f64) []const u8 {
    if (comptime !enabled) return buf[0..0];
    const st = timeline.stored();
    const n_cycles = @min(st.verifies.len, max_cycles);
    byCycle(st.rows, n_cycles);
    var per: [2]Acc = .{ .{}, .{} };
    var tot: Round = .{};
    var rounds: u64 = 0;
    var truncated: u64 = 0;
    var picked_over: u64 = 0;
    for (0..n_cycles) |c| {
        const lo = state.starts[c];
        const hi = state.starts[c + 1];
        const n = @min(hi - lo, state.picked.len);
        picked_over += (hi - lo) - n;
        for (state.index[lo .. lo + n], 0..) |ri, k| state.picked[k] = st.rows[ri];
        const r = addVerify(state.picked[0..n], st.verifies[c], st.layers[c][0..st.n_layers], &per, &state.spans, &truncated);
        if (r.wall == 0) continue;
        rounds += 1;
        tot.wall += r.wall;
        tot.pre += r.pre;
        tot.post += r.post;
        tot.layers.add(r.layers);
    }
    var w: std.Io.Writer = .fixed(buf);
    writeLine(&w, rounds, tot, per, profile_barrier_ms, truncated + picked_over) catch return buf[0..0];
    return w.buffered();
}

fn ms(x: u64, n: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e6 / @as(f64, @floatFromInt(@max(n, 1)));
}

fn per1(x: u64, n: u64) f64 {
    return @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(@max(n, 1)));
}

pub fn writeLine(w: *std.Io.Writer, rounds: u64, tot: Round, per: [2]Acc, profile_barrier_ms: f64, truncated: u64) !void {
    const L = tot.layers;
    const cb = ms(L.call_barrier, rounds);
    try w.writeAll("MIMO_BARRIER_RESIDUAL {");
    try w.print("\"rounds\": {d}, \"layers_per_round\": {d:.2}, \"truncated\": {d}, ", .{ rounds, per1(L.n, rounds), truncated });
    try w.print("\"reconcile\": {{\"call_to_barrier_ms\": {d:.3}, \"verify_barrier_ms\": {d:.3}, \"residual_ms\": {d:.3}}}, ", .{ cb, profile_barrier_ms, profile_barrier_ms - cb });
    try w.print("\"per_round_ms\": {{\"wall\": {d:.3}, \"pre\": {d:.3}, \"post\": {d:.3}, \"host\": {d:.3}, \"gpu\": {d:.3}, \"gpu_prev\": {d:.3}, \"gpu_bar\": {d:.3}, \"gpu_wave\": {d:.3}, \"gpu_gated\": {d:.3}, \"idle_queued\": {d:.3}, \"idle_starved\": {d:.3}, \"wake\": {d:.3}}}, ", .{ ms(tot.wall, rounds), ms(tot.pre, rounds), ms(tot.post, rounds), ms(L.host, rounds), ms(L.gpu, rounds), ms(L.gpu_prev, rounds), ms(L.gpu_bar, rounds), ms(L.gpu_wave, rounds), ms(L.gpu_gated, rounds), ms(L.idle_queued, rounds), ms(L.idle_starved, rounds), ms(L.wake, rounds) });
    try w.print("\"per_round\": {{\"commits\": {d:.2}, \"barrier_commits\": {d:.2}", .{ per1(L.commits, rounds), per1(L.bar_commits, rounds) });
    if (comptime enabled) {
        const routes = per1(reads.routes, reads.verifies);
        try w.print(", \"routes\": {d:.2}, \"ids_reads\": {d:.2}, \"scores_reads\": {d:.2}, \"barrier_waits\": {d:.2}, \"decision_reads\": 1", .{ routes, routes, @max(routes - 1, 0), routes });
    }
    try w.writeAll("}, \"per_layer_ms\": {");
    for ([_]Kind{ .sliding, .full }, 0..) |k, ki| {
        const a = per[@intFromEnum(k)];
        try w.print("{s}\"{s}\": {{\"layers\": {d}, \"host\": {d:.4}, \"call_to_barrier\": {d:.4}, \"gpu\": {d:.4}, \"gpu_prev\": {d:.4}, \"gpu_bar\": {d:.4}, \"gpu_wave\": {d:.4}, \"gpu_gated\": {d:.4}, \"idle_queued\": {d:.4}, \"idle_starved\": {d:.4}, \"wake\": {d:.4}, \"commits\": {d:.2}, \"barrier_commits\": {d:.2}}}", .{ if (ki == 0) "" else ", ", @tagName(k), a.n, ms(a.host, a.n), ms(a.call_barrier, a.n), ms(a.gpu, a.n), ms(a.gpu_prev, a.n), ms(a.gpu_bar, a.n), ms(a.gpu_wave, a.n), ms(a.gpu_gated, a.n), ms(a.idle_queued, a.n), ms(a.idle_starved, a.n), ms(a.wake, a.n), per1(a.commits, a.n), per1(a.bar_commits, a.n) });
    }
    try w.writeAll("}}");
}

const testing = std.testing;

test "mimo barrier residual: a hand-made verify splits each layer's host span into GPU classes and idle, exactly" {
    const tg = timeline.tagOf(0, .verify);
    // Verify [0, 1000). Layer 1 (sliding): call 10, barrier 300, route 302, hit 303, end 305.
    //   bar buffers B1a committed 20 (GPU 30..150), B1b committed 25 (GPU 152..280); hit wave W1 committed 28 (GPU 280..500),
    //   empty E1 committed 29 (GPU 501..501).
    // Layer 5 (full): call 360, barrier 900, route 901, hit 902, end 905.
    //   bar B5 committed 380 (GPU 520..850: queued behind W1, then a 20 ns boundary), wave W5 committed 385 (GPU 850..990).
    //   the GPU idles 500..520 while B5 was committed at 380: a queued boundary.
    const rs = [_]Row{
        .{ .host_commit = 20, .gpu_start = 30, .gpu_end = 150, .tag = tg },
        .{ .host_commit = 25, .gpu_start = 152, .gpu_end = 280, .tag = tg },
        .{ .host_commit = 28, .gpu_start = 280, .gpu_end = 500, .tag = tg },
        .{ .host_commit = 29, .gpu_start = 501, .gpu_end = 501, .tag = tg },
        .{ .host_commit = 380, .gpu_start = 520, .gpu_end = 850, .tag = tg },
        .{ .host_commit = 385, .gpu_start = 850, .gpu_end = 990, .tag = tg },
    };
    var ls: [8]Layer = @splat(.{});
    ls[1] = .{ .t = .{ 10, 300, 302, 303, 305 } };
    ls[5] = .{ .t = .{ 360, 900, 901, 902, 905 } };
    var per: [2]Acc = .{ .{}, .{} };
    var scratch: [16]Span = undefined;
    var truncated: u64 = 0;
    const r = addVerify(&rs, .{ .begin = 0, .end = 1000 }, &ls, &per, &scratch, &truncated);
    try testing.expectEqual(@as(u64, 0), truncated);
    try testing.expectEqual(@as(u64, 1000), r.wall);
    try testing.expectEqual(@as(u64, 10), r.pre);
    try testing.expectEqual(@as(u64, 95), r.post);
    const s = per[@intFromEnum(Kind.sliding)];
    const f = per[@intFromEnum(Kind.full)];
    // layer 1: host [0, 300) (the verify's begin to its return; its call at 10)
    try testing.expectEqual(@as(u64, 300), s.host);
    try testing.expectEqual(@as(u64, 290), s.call_barrier);
    try testing.expectEqual(@as(u64, 120 + 128), s.gpu_bar);
    try testing.expectEqual(@as(u64, 20), s.gpu_wave); // W1 280..300 before the return
    try testing.expectEqual(@as(u64, 0), s.gpu_prev);
    try testing.expectEqual(@as(u64, 20), s.wake); // B1b ended 280, the host returned at 300
    try testing.expectEqual(@as(u64, 4), s.commits);
    try testing.expectEqual(@as(u64, 2), s.bar_commits);
    // idle: the gap [0, 30) is starved until B1a's commit (20), then queued to its start (30); 150..152 is a queued
    // boundary (B1b, committed at 25, started at 152)
    try testing.expectEqual(@as(u64, 20), s.idle_starved);
    try testing.expectEqual(@as(u64, 12), s.idle_queued);
    try testing.expectEqual(s.host, s.gpu + s.idle_queued + s.idle_starved);
    // layer 5: host [305, 900): W1's tail 305..500 (prev), the boundary 500..520 (queued: B5 committed 380 < 500),
    // 501..501 empty, B5 520..850 (bar), W5 850..900 (wave)
    try testing.expectEqual(@as(u64, 595), f.host);
    try testing.expectEqual(@as(u64, 540), f.call_barrier);
    try testing.expectEqual(@as(u64, 195), f.gpu_prev);
    try testing.expectEqual(@as(u64, 330), f.gpu_bar);
    try testing.expectEqual(@as(u64, 50), f.gpu_wave);
    try testing.expectEqual(@as(u64, 20), f.idle_queued);
    try testing.expectEqual(@as(u64, 0), f.idle_starved);
    try testing.expectEqual(f.host, f.gpu + f.idle_queued + f.idle_starved);
    try testing.expectEqual(@as(u64, 50), f.wake);
    try testing.expectEqual(@as(u64, 2), f.commits);
    try testing.expectEqual(@as(u64, 1), f.bar_commits);
    // the round's totals are the layers' sums
    try testing.expectEqual(s.host + f.host, r.layers.host);
    try testing.expectEqual(@as(u64, 2), r.layers.n);
    // the line reconciles the call-to-barrier sum with a profile's barrier mean
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeLine(&w, 1, r, per, 0.000831, 0);
    const text = w.buffered();
    try testing.expect(std.mem.startsWith(u8, text, "MIMO_BARRIER_RESIDUAL {\"rounds\": 1, \"layers_per_round\": 2.00"));
    try testing.expect(std.mem.indexOf(u8, text, "\"call_to_barrier_ms\": 0.001, \"verify_barrier_ms\": 0.001, \"residual_ms\": 0.000") != null);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text["MIMO_BARRIER_RESIDUAL ".len..], .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("per_layer_ms").?.object.get("full") != null);
}

test "mimo barrier residual: a repaired wait-only row is gated, not barrier work; a span past the scratch is counted" {
    const tg = timeline.tagOf(0, .verify);
    const rs = [_]Row{
        .{ .host_commit = 5, .gpu_start = 10, .gpu_end = 40, .tag = tg },
        .{ .host_commit = 6, .gpu_start = 40, .gpu_end = 90, .tag = tg, .status = timeline.repaired_status },
        .{ .host_commit = 7, .gpu_start = 90, .gpu_end = 100, .tag = tg },
    };
    var ls: [2]Layer = @splat(.{});
    ls[1] = .{ .t = .{ 1, 100, 101, 102, 103 } };
    var per: [2]Acc = .{ .{}, .{} };
    var scratch: [2]Span = undefined;
    var truncated: u64 = 0;
    _ = addVerify(&rs, .{ .begin = 0, .end = 200 }, &ls, &per, &scratch, &truncated);
    const s = per[@intFromEnum(Kind.sliding)];
    try testing.expectEqual(@as(u64, 1), truncated);
    try testing.expectEqual(@as(u64, 30), s.gpu_bar);
    try testing.expectEqual(@as(u64, 50), s.gpu_gated);
    var scratch3: [3]Span = undefined;
    var per3: [2]Acc = .{ .{}, .{} };
    truncated = 0;
    _ = addVerify(&rs, .{ .begin = 0, .end = 200 }, &ls, &per3, &scratch3, &truncated);
    try testing.expectEqual(@as(u64, 0), truncated);
    try testing.expectEqual(@as(u64, 40), per3[0].gpu_bar);
    try testing.expectEqual(per3[0].host, per3[0].gpu + per3[0].idle_queued + per3[0].idle_starved);
}

test "mimo barrier residual: a served build compiles every hook to nothing" {
    if (comptime enabled) return error.SkipZigTest;
    verifyBegin();
    noteRoute();
    verifyEnd();
    var buf: [16]u8 = undefined;
    try testing.expectEqual(@as(usize, 0), line(&buf, 1).len);
    try testing.expectEqual(@as(usize, 0), @sizeOf(@TypeOf(state)));
}
