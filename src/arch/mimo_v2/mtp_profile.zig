//! The MTP draft lane's per-round profile, for PROFILE builds only (`sdk.plugin_profile`, the one switch that also turns
//! on DECODE_TIMERS and the GPU timeline). In every other build `enabled` is false: the storage is `void` and every call
//! below compiles to nothing (the served round carries no clock, counter or branch for it).
//!
//! A round's host time is split on DECODE_TIMERS' own buckets (`dsv41_decode_timers.zig`: the round charges
//! draft_build / draft_wait / draft_host / verify / verify_eval / accept / trim / tail, the shared decode lane inside
//! the verify charges barrier / route / read_wait); this file takes each bucket's delta per round, with the expert
//! stream's counters (misses, bytes, lookahead claims, pre-reads) and the verify's union per routed layer. The run's
//! summary is one `[mtp-profile] {...}` line (means, p50, p90); with `MIMO_ROUTE_TRACE` armed the rounds also land in
//! `<prefix>-r<k>.rounds.tsv`. `MIMO_GPU_TIMELINE=1` (read once at construction) also installs DSV's verify GPU
//! timeline (`dsv41_verify_timeline.zig`, cycle = round) for the request: its `VERIFY_GPU_TIMELINE {...}` line splits
//! the verify's GPU time into work and the gated waits on expert bytes, and with the trace armed its raw rows land in
//! `<prefix>-r<k>.gpu-timeline.json`.
const std = @import("std");
const mlx = @import("sdk").mlx;
const profile = @import("profile.zig");
const dt = @import("../deepseek_v41/decode_timers.zig");
const timeline = @import("../deepseek_v41/verify_timeline.zig");
const barrier_residual = @import("barrier_residual.zig");
const gpu_warm = @import("gpu_warm.zig");
const gpu_power = @import("gpu_power.zig");
const expert = @import("../../sdk_ext.zig").expert;
const log = std.log.scoped(.mimo);

pub const enabled = profile.enabled;
comptime {
    if (dt.enabled != enabled or timeline.enabled != enabled) @compileError("mimo_v2: the MTP profile, DECODE_TIMERS and the GPU timeline follow one switch");
}

/// Rounds kept per request (16,384 tokens at one token per round); later rounds count in `dropped`.
pub const max_rounds: u32 = 16384;
/// The GPU timeline's layer table: MiMo's 48 absolute layers (the shared lane stamps absolute layer ids).
pub const timeline_layers: u32 = 48;

/// One round. `*_ns` with a DECODE_TIMERS bucket's name is that bucket's delta over the round; the stream counters are
/// deltas of `expert.Stats` over the round.
pub const Round = struct {
    rows: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    round_ns: u64 = 0,
    draft_build_ns: u64 = 0,
    draft_wait_ns: u64 = 0,
    draft_host_ns: u64 = 0,
    verify_ns: u64 = 0,
    barrier_ns: u64 = 0,
    route_ns: u64 = 0,
    read_wait_ns: u64 = 0,
    verify_eval_ns: u64 = 0,
    accept_ns: u64 = 0,
    trim_ns: u64 = 0,
    tail_ns: u64 = 0,
    routed_calls: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    bytes: u64 = 0,
    claimed: u64 = 0,
    pre_issued: u64 = 0,
    pre_served: u64 = 0,
    spec_issued: u64 = 0,
    read_wall_ns: u64 = 0,
    union_sum: u64 = 0,
    union_max: u64 = 0,
};

const State = struct {
    rounds: [max_rounds]Round = undefined,
    n: u32 = 0,
    dropped: u64 = 0,
    // the round in progress
    t0: u64 = 0,
    dt0: @TypeOf(dt.ns) = undefined,
    calls0: u64 = 0,
    st0: expert.Stats = .{},
    union_sum: u64 = 0,
    union_max: u64 = 0,
    // the request's GPU timeline: wanted (construction env), installed (first request), recording this request
    gpu_wanted: bool = false,
    gpu_installed: bool = false,
    gpu_on: bool = false,
    gpu_truncated: bool = false,
    stream: ?mlx.mlx_stream = null,
    finished: bool = true,
    // X12: the warm engine (`gpu_keep_warm: "waits"`) and its counters at the decode phase's begin
    warm: ?*const gpu_warm.Warm = null,
    warm_route: gpu_warm.Route = .off,
    warm0: gpu_warm.Stats = .{},
    // X12: the GPU's energy / DVFS / thermal counters (IOReport), opened at the first decode, sampled at its begin
    power: ?gpu_power.Sampler = null,
    power_tried: bool = false,
    power0: ?*anyopaque = null,
};
var state: if (enabled) State else void = if (enabled) .{} else {};
var scratch: if (enabled) [max_rounds]f64 else void = undefined;
var warm_spans: if (enabled) [1 << 14][2]u64 else void = undefined;
var warm_per: if (enabled) [timeline.max_cycles]f64 else void = undefined;

/// X12: per round, the GPU busy union inside the verify window of every recorded buffer and of the model's queue's alone
/// (`Row.queue` 1; the warm engine commits on its own queue): what the other queue's buffers alone added.
pub const WarmBusy = struct { rounds: u32 = 0, other_rows: u64 = 0, truncated: u32 = 0, all_ms_mean: f64 = 0, model_ms_mean: f64 = 0, warm_only_ms_mean: f64 = 0, warm_only_ms_p50: f64 = 0 };

fn unionNs(spans: [][2]u64) u64 {
    std.mem.sort([2]u64, spans, {}, struct {
        fn lt(_: void, x: [2]u64, y: [2]u64) bool {
            return x[0] < y[0];
        }
    }.lt);
    var busy: u64 = 0;
    var cur: ?[2]u64 = null;
    for (spans) |sp| {
        if (cur) |cu| {
            if (sp[0] <= cu[1]) {
                cur = .{ cu[0], @max(cu[1], sp[1]) };
                continue;
            }
            busy += cu[1] - cu[0];
        }
        cur = sp;
    }
    if (cur) |cu| busy += cu[1] - cu[0];
    return busy;
}

/// Pure over the stored rows (repaired) and verifies; `spans` per round's scratch, `per` the rounds' warm-only ms. Rows
/// are in commit order and their cycles non-decreasing (a buffer committed off the inference thread can read the tag of
/// the cycle before: each round's scan reaches `slack` rows past both ends).
pub fn warmBusy(rs: []const timeline.Row, vs: []const timeline.Verify, spans: [][2]u64, per: []f64) WarmBusy {
    const slack: usize = 256;
    var out: WarmBusy = .{};
    var all_ns: u64 = 0;
    var model_ns: u64 = 0;
    var n_per: usize = 0;
    for (rs) |r| out.other_rows += @intFromBool(r.queue != 1);
    var cursor: usize = 0;
    for (vs, 0..) |v, ci| {
        const cyc: u32 = @intCast(ci);
        while (cursor < rs.len and (timeline.rowCycle(rs[cursor]) orelse 0) < cyc) cursor += 1;
        var end = cursor;
        while (end < rs.len and (timeline.rowCycle(rs[end]) orelse 0) <= cyc) end += 1;
        if (v.end <= v.begin) continue;
        const lo = cursor -| slack;
        const hi = @min(rs.len, end + slack);
        // the model's rows first, then the other queue's: two unions over the same scratch
        var n: usize = 0;
        var n_model: usize = 0;
        for ([_]bool{ true, false }) |model_pass| {
            for (rs[lo..hi]) |r| {
                if (timeline.rowCycle(r) != cyc or (r.queue == 1) != model_pass or r.gpu_end <= r.gpu_start) continue;
                const a = @max(r.gpu_start, v.begin);
                const b = @min(r.gpu_end, v.end);
                if (b <= a) continue;
                if (n == spans.len) {
                    out.truncated += 1;
                    break;
                }
                spans[n] = .{ a, b };
                n += 1;
            }
            if (model_pass) n_model = n;
        }
        const m = unionNs(spans[0..n_model]);
        const all = unionNs(spans[0..n]);
        all_ns += all;
        model_ns += m;
        if (n_per < per.len) {
            per[n_per] = @as(f64, @floatFromInt(all -| m)) / 1e6;
            n_per += 1;
        }
        out.rounds += 1;
    }
    if (out.rounds > 0) {
        const r: f64 = @floatFromInt(out.rounds);
        out.all_ms_mean = @as(f64, @floatFromInt(all_ns)) / 1e6 / r;
        out.model_ms_mean = @as(f64, @floatFromInt(model_ns)) / 1e6 / r;
        out.warm_only_ms_mean = out.all_ms_mean - out.model_ms_mean;
        out.warm_only_ms_p50 = quantile(per[0..n_per], 0.5);
    }
    return out;
}

test "mimo gpu warm timeline: per round, the busy time the other queue's buffers alone add over the model's (verify windows, commit-order rows, a straggler from the round before)" {
    const T = timeline.Row;
    const tag = timeline.tagOf;
    // round 0 window [100, 200): model 100..140 and 150..190; warm 130..160 (adds 140..150 = 10) and 195..230 (adds 195..200 = 5)
    // round 1 window [300, 400): model 300..350; a warm buffer committed late with round 0's tag (not round 1's); warm
    // 360..380 (adds 20)
    const rows = [_]T{
        .{ .gpu_start = 100, .gpu_end = 140, .tag = tag(0, .verify), .queue = 1 },
        .{ .gpu_start = 130, .gpu_end = 160, .tag = tag(0, .verify), .queue = 0 },
        .{ .gpu_start = 150, .gpu_end = 190, .tag = tag(0, .verify), .queue = 1 },
        .{ .gpu_start = 300, .gpu_end = 350, .tag = tag(1, .verify), .queue = 1 },
        .{ .gpu_start = 195, .gpu_end = 230, .tag = tag(0, .verify), .queue = 0 },
        .{ .gpu_start = 360, .gpu_end = 380, .tag = tag(1, .verify), .queue = 0 },
    };
    const vs = [_]timeline.Verify{ .{ .begin = 100, .end = 200 }, .{ .begin = 300, .end = 400 } };
    var spans: [16][2]u64 = undefined;
    var per: [4]f64 = undefined;
    const wb = warmBusy(&rows, &vs, &spans, &per);
    try std.testing.expectEqual(@as(u32, 2), wb.rounds);
    try std.testing.expectEqual(@as(u64, 3), wb.other_rows);
    try std.testing.expectEqual(@as(u32, 0), wb.truncated);
    // round 0: all = 100..190 (90) + 195..200 (5) = 95, model 80 -> 15; round 1: all 70, model 50 -> 20
    try std.testing.expectApproxEqAbs(@as(f64, (95.0 + 70.0) / 2.0 / 1e6), wb.all_ms_mean, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (80.0 + 50.0) / 2.0 / 1e6), wb.model_ms_mean, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 17.5 / 1e6), wb.warm_only_ms_mean, 1e-12);
}

pub const Stamp = dt.Stamp;
pub const now = dt.now;
pub const charge = dt.charge;

/// Construction: the GPU timeline is wanted (`MIMO_GPU_TIMELINE=1`, read once by the module).
pub fn wantGpuTimeline(on: bool) void {
    if (comptime enabled) state.gpu_wanted = on;
}

/// The decode phase of a lane request begins (after the phase change): the rounds restart; the GPU timeline is installed
/// when wanted, once per process (the cell's one request).
pub fn decodeBegin(s: mlx.mlx_stream, decode_budget: u64) void {
    if (comptime !enabled) return;
    state.n = 0;
    state.dropped = 0;
    state.finished = false;
    state.gpu_on = false;
    state.gpu_truncated = false;
    state.stream = s;
    barrier_residual.resetReads();
    if (!state.gpu_wanted) return;
    if (state.gpu_installed) {
        log.info("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"installed for the first request only\"}}", .{});
        return;
    }
    const cycles: u32 = @intCast(@min(decode_budget, timeline.max_cycles));
    timeline.install(s, cycles + 1, timeline_layers) catch |e| {
        log.info("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"install: {s}\"}}", .{@errorName(e)});
        return;
    };
    state.gpu_installed = true;
    state.gpu_on = true;
}

/// X12: the decode phase's warm engine (null on the control): its counters now, for the request's `MIMO_GPU_WARM` line.
pub fn warmBegin(w: ?*const gpu_warm.Warm, route: gpu_warm.Route) void {
    if (comptime !enabled) return;
    state.warm = w;
    state.warm_route = route;
    if (w) |x| state.warm0 = x.stats();
    // Every arm (off and waits) samples the GPU's energy, DVFS and thermal counters over the decode phase.
    if (!state.power_tried) {
        state.power_tried = true;
        state.power = gpu_power.Sampler.open() catch |e| blk: {
            log.info("MIMO_GPU_POWER_UNAVAILABLE {{\"reason\": \"{s}: {s}\"}}", .{ @errorName(e), gpu_power.lastError() });
            break :blk null;
        };
    }
    if (state.power) |*p| {
        if (state.power0) |old| p.free(old);
        state.power0 = p.sample();
    }
}

/// A round begins: the buckets' and the stream's marks; the GPU timeline's cycle (stopped at its cycle table's end).
pub fn roundBegin(st: expert.Stats) Stamp {
    if (comptime !enabled) return {};
    if (state.gpu_on) {
        if (timeline.cycles == 0) if (timeline.unavailableReason()) |why| {
            var buf: [1024]u8 = undefined;
            log.info("{s}", .{timeline.unavailableLine(&buf, why)});
            timeline.uninstall();
            state.gpu_on = false;
        };
        if (state.gpu_on and timeline.cycles >= timeline.max_cycles) {
            timeline.disarm();
            timeline.active = false;
            state.gpu_truncated = true;
        }
        timeline.cycleBegin();
    }
    state.dt0 = dt.ns;
    state.calls0 = dt.routed_calls;
    state.st0 = st;
    state.union_sum = 0;
    state.union_max = 0;
    state.t0 = dt.now();
    return state.t0;
}

pub inline fn verifyBegin() void {
    if (comptime !enabled) return;
    timeline.verifyBegin();
    barrier_residual.verifyBegin();
}

pub inline fn verifyEnd() void {
    if (comptime !enabled) return;
    timeline.verifyEnd();
    barrier_residual.verifyEnd();
}

/// One decode-phase route (`StreamSource.route`): the distinct experts of its ids (the rows' union).
pub inline fn unionNote(ids: []const u16) void {
    if (comptime !enabled) return;
    var seen: std.StaticBitSet(512) = .empty;
    var n: u64 = 0;
    for (ids) |e| if (e >= 512 or !seen.isSet(e)) {
        if (e < 512) seen.set(e);
        n += 1;
    };
    state.union_sum += n;
    state.union_max = @max(state.union_max, n);
    barrier_residual.noteRoute();
}

/// The round ends: its record (rows = 1 + drafted).
pub fn roundEnd(st: expert.Stats, drafted: u32, accepted: u32) void {
    if (comptime !enabled) return;
    const t = dt.now();
    if (state.n >= max_rounds) {
        state.dropped += 1;
        return;
    }
    var r: Round = .{
        .rows = drafted + 1,
        .drafted = drafted,
        .accepted = accepted,
        .round_ns = t - state.t0,
        .routed_calls = dt.routed_calls - state.calls0,
        .hits = st.expert_cache_hits -| state.st0.expert_cache_hits,
        .misses = st.expert_cache_misses -| state.st0.expert_cache_misses,
        .bytes = st.expert_bytes_read -| state.st0.expert_bytes_read,
        .claimed = st.claimed -| state.st0.claimed,
        .pre_issued = st.pre_issued -| state.st0.pre_issued,
        .pre_served = st.pre_served -| state.st0.pre_served,
        .spec_issued = st.spec_issued -| state.st0.spec_issued,
        .read_wall_ns = st.read_wall_ns -| state.st0.read_wall_ns,
        .union_sum = state.union_sum,
        .union_max = state.union_max,
    };
    inline for (@typeInfo(dt.Bucket).@"enum".field_names, 0..) |name, i| {
        if (@hasField(Round, name ++ "_ns")) @field(r, name ++ "_ns") = dt.ns[i] - state.dt0[i];
    }
    state.rounds[state.n] = r;
    state.n += 1;
}

pub fn rounds() u32 {
    if (comptime !enabled) return 0;
    return state.n;
}

// ── the summary ──

const ms = 1e6;
/// The summary's per-round quantities (one definition each; REPORT.md lists them).
pub const Field = struct { name: []const u8, f: *const fn (Round) f64 };
fn f(x: u64) f64 {
    return @floatFromInt(x);
}
pub const fields = [_]Field{
    .{ .name = "round_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.round_ns) / ms;
        }
    }.g },
    .{ .name = "draft_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.draft_build_ns + r.draft_wait_ns + r.draft_host_ns) / ms;
        }
    }.g },
    .{ .name = "draft_build_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.draft_build_ns) / ms;
        }
    }.g },
    .{ .name = "draft_wait_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.draft_wait_ns) / ms;
        }
    }.g },
    .{ .name = "draft_host_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.draft_host_ns) / ms;
        }
    }.g },
    .{ .name = "verify_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.verify_ns) / ms;
        }
    }.g },
    .{ .name = "verify_barrier_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.barrier_ns) / ms;
        }
    }.g },
    .{ .name = "verify_route_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.route_ns) / ms;
        }
    }.g },
    .{ .name = "verify_read_wait_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.read_wait_ns) / ms;
        }
    }.g },
    .{ .name = "verify_encode_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.verify_ns -| (r.barrier_ns + r.route_ns + r.read_wait_ns)) / ms;
        }
    }.g },
    .{ .name = "verify_eval_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.verify_eval_ns) / ms;
        }
    }.g },
    .{ .name = "accept_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.accept_ns) / ms;
        }
    }.g },
    .{ .name = "trim_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.trim_ns) / ms;
        }
    }.g },
    .{ .name = "tail_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.tail_ns) / ms;
        }
    }.g },
    .{ .name = "other_ms", .f = &struct {
        fn g(r: Round) f64 {
            const charged = r.draft_build_ns + r.draft_wait_ns + r.draft_host_ns + r.verify_ns + r.verify_eval_ns + r.accept_ns + r.trim_ns + r.tail_ns;
            return f(r.round_ns -| charged) / ms;
        }
    }.g },
    .{ .name = "rows", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.rows);
        }
    }.g },
    .{ .name = "accepted", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.accepted);
        }
    }.g },
    .{ .name = "routed_calls", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.routed_calls);
        }
    }.g },
    .{ .name = "misses", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.misses);
        }
    }.g },
    .{ .name = "hits", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.hits);
        }
    }.g },
    .{ .name = "read_mb", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.bytes) / 1e6;
        }
    }.g },
    .{ .name = "read_wall_ms", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.read_wall_ns) / ms;
        }
    }.g },
    .{ .name = "union_per_layer", .f = &struct {
        fn g(r: Round) f64 {
            return if (r.routed_calls == 0) 0 else f(r.union_sum) / f(r.routed_calls);
        }
    }.g },
    .{ .name = "union_max", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.union_max);
        }
    }.g },
    .{ .name = "ahead_claimed", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.claimed);
        }
    }.g },
    .{ .name = "pre_served", .f = &struct {
        fn g(r: Round) f64 {
            return f(r.pre_served);
        }
    }.g },
    .{ .name = "ahead_coverage", .f = &struct {
        fn g(r: Round) f64 {
            return if (r.misses == 0) 0 else f(r.claimed) / f(r.misses);
        }
    }.g },
};

fn quantile(xs: []f64, q: f64) f64 {
    if (xs.len == 0) return 0;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    // nearest rank
    const rank: usize = @intFromFloat(@ceil(q * @as(f64, @floatFromInt(xs.len))));
    return xs[@max(rank, 1) - 1];
}

/// `[mtp-profile] {...}` over `rs`: counts, then the mean / p50 / p90 of every field.
pub fn line(buf: []u8, rs: []const Round, dropped: u64, xs: []f64) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    writeLine(&w, rs, dropped, xs) catch return buf[0..0];
    return w.buffered();
}

fn writeLine(w: *std.Io.Writer, rs: []const Round, dropped: u64, xs: []f64) !void {
    var drafted: u64 = 0;
    var accepted: u64 = 0;
    for (rs) |r| {
        drafted += r.drafted;
        accepted += r.accepted;
    }
    const n: f64 = @floatFromInt(@max(rs.len, 1));
    try w.print("[mtp-profile] {{\"rounds\": {d}, \"dropped\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"accepted_per_round\": {d:.3}, \"tokens_per_round\": {d:.3}", .{ rs.len, dropped, drafted, accepted, f(accepted) / n, (f(accepted) + f(rs.len)) / n });
    inline for (.{ "mean", "p50", "p90" }) |kind| {
        try w.print(", \"{s}\": {{", .{kind});
        for (fields, 0..) |fd, i| {
            for (rs, xs[0..rs.len]) |r, *x| x.* = fd.f(r);
            const v = if (comptime std.mem.eql(u8, kind, "mean")) blk: {
                var s: f64 = 0;
                for (xs[0..rs.len]) |x| s += x;
                break :blk s / n;
            } else quantile(xs[0..rs.len], if (comptime std.mem.eql(u8, kind, "p50")) 0.5 else 0.9);
            try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", fd.name, v });
        }
        try w.writeAll("}");
    }
    try w.writeAll("}");
}

/// The per-round file: a header of `Round`'s fields, then one row per round (raw ns and counts).
pub fn writeTsv(w: *std.Io.Writer, rs: []const Round) !void {
    try w.writeAll("round");
    inline for (@typeInfo(Round).@"struct".field_names) |name| try w.writeAll("\t" ++ name);
    try w.writeAll("\n");
    for (rs, 0..) |r, i| {
        try w.print("{d}", .{i});
        inline for (@typeInfo(Round).@"struct".field_names) |name| try w.print("\t{d}", .{@field(r, name)});
        try w.writeAll("\n");
    }
}

/// The request's end (the lane's receipt, or the decode phase's end, whichever comes first; once): the summary line,
/// the GPU timeline's line, and with `base` (the armed trace's `<prefix>-r<k>`) the per-round and GPU files.
pub fn finish(a: std.mem.Allocator, io: std.Io, base: ?[]const u8) void {
    if (comptime !enabled) return;
    if (state.finished) return;
    state.finished = true;
    if (state.n == 0 and !state.gpu_on) return;
    const rs = state.rounds[0..state.n];
    var buf: [8192]u8 = undefined;
    log.info("{s}", .{line(&buf, rs, state.dropped, &scratch)});
    // X12: the warm engine's request (episodes, chunks, its GPU time), from its own counters (the warm thread's).
    if (state.warm) |w| {
        var wb: [512]u8 = undefined;
        log.info("{s}", .{gpu_warm.line(&wb, state.warm_route, w.stats().since(state.warm0))});
    }
    // X12: the GPU's energy, mean active MHz and thermal limiting over the decode phase (both arms).
    if (state.power) |*p| if (state.power0) |p0| {
        const p1 = p.sample();
        defer p.free(p1);
        if (p.delta(p0, p1)) |d| {
            var pb: [1024]u8 = undefined;
            var w: std.Io.Writer = .fixed(&pb);
            w.writeAll("MIMO_GPU_POWER ") catch {};
            gpu_power.json(&w, d, p.mhz[0..p.n_mhz]) catch {};
            log.info("{s}", .{w.buffered()});
        } else |e| log.info("MIMO_GPU_POWER_UNAVAILABLE {{\"reason\": \"delta: {s}: {s}\"}}", .{ @errorName(e), gpu_power.lastError() });
        p.free(p0);
        state.power0 = null;
    };
    if (base) |b| writeFile(a, io, b, ".rounds.tsv", rs, writeTsvTo);
    if (state.gpu_on) {
        const pending = timeline.settle(state.stream.?, 2000);
        // MiMo's gated miss parts commit wait-only buffers (GPUStartTime 0): their starts are the queue's before any
        // span is summed (`timeline.repairRows`); the JSON keeps them marked.
        const rep = timeline.repairUnstarted();
        log.info("VERIFY_GPU_TIMELINE_REPAIR {{\"unstarted\": {d}, \"repaired\": {d}, \"zero_length\": {d}, \"status_bit\": {d}}}", .{ rep.unstarted, rep.repaired, rep.zero_length, timeline.repaired_status });
        var gb: [16384]u8 = undefined;
        log.info("{s}", .{timeline.line(&gb, pending)});
        // X12: the busy time the other queue's buffers alone added per round (the warm engine's chunks: the line above
        // counts them in its busy union; its gpu_work subtracts only the gated waits they overlap).
        const st = timeline.stored();
        const wb = warmBusy(st.rows, st.verifies, &warm_spans, &warm_per);
        log.info("MIMO_GPU_WARM_TIMELINE {{\"rounds\": {d}, \"other_queue_rows\": {d}, \"truncated\": {d}, \"per_round_ms\": {{\"busy_all_mean\": {d:.3}, \"busy_model_mean\": {d:.3}, \"warm_only_mean\": {d:.3}, \"warm_only_p50\": {d:.3}}}}}", .{ wb.rounds, wb.other_rows, wb.truncated, wb.all_ms_mean, wb.model_ms_mean, wb.warm_only_ms_mean, wb.warm_only_ms_p50 });
        if (state.gpu_truncated) log.info("VERIFY_GPU_TIMELINE_TRUNCATED {{\"cycles\": {d}}}", .{timeline.max_cycles});
        // The verify barrier's attribution over the same rows, reconciled with the rounds' mean barrier bucket.
        var barrier_ns: u64 = 0;
        for (rs) |r| barrier_ns += r.barrier_ns;
        const barrier_ms = @as(f64, @floatFromInt(barrier_ns)) / 1e6 / @as(f64, @floatFromInt(@max(rs.len, 1)));
        var bb: [4096]u8 = undefined;
        log.info("{s}", .{barrier_residual.line(&bb, barrier_ms)});
        if (base) |b| writeFile(a, io, b, ".gpu-timeline.json", rs, writeGpuTo);
        timeline.uninstall();
        state.gpu_on = false;
    }
}

fn writeTsvTo(w: *std.Io.Writer, rs: []const Round) anyerror!void {
    return writeTsv(w, rs);
}

fn writeGpuTo(w: *std.Io.Writer, _: []const Round) anyerror!void {
    return timeline.writeJson(w);
}

fn writeFile(a: std.mem.Allocator, io: std.Io, base: []const u8, ext: []const u8, rs: []const Round, comptime body: fn (*std.Io.Writer, []const Round) anyerror!void) void {
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    body(&aw.writer, rs) catch |e| return log.err("[mtp-profile] {s}: {s}", .{ ext, @errorName(e) });
    const path = std.fmt.allocPrint(a, "{s}{s}", .{ base, ext }) catch return;
    defer a.free(path);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = aw.written() }) catch |e| log.err("[mtp-profile] write {s}: {s}", .{ path, @errorName(e) });
}

test "mimo mtp profile: the summary line's counts, means and nearest-rank p50 / p90; the per-round file's columns" {
    var rs: [10]Round = undefined;
    for (&rs, 0..) |*r, i| r.* = .{
        .rows = 4,
        .drafted = 3,
        .accepted = @intCast(i % 4),
        .round_ns = (i + 1) * 10_000_000,
        .draft_build_ns = 1_000_000,
        .draft_wait_ns = 2_000_000,
        .draft_host_ns = 0,
        .verify_ns = 5_000_000,
        .barrier_ns = 3_000_000,
        .route_ns = 1_000_000,
        .misses = 100,
        .claimed = 25,
        .routed_calls = 47,
        .union_sum = 47 * 20,
        .union_max = 25,
        .bytes = 2_000_000,
    };
    var xs: [10]f64 = undefined;
    var buf: [8192]u8 = undefined;
    const l = line(&buf, &rs, 0, &xs);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, l["[mtp-profile] ".len..], .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 10), o.get("rounds").?.integer);
    // accepted 0,1,2,3,0,1,2,3,0,1 = 13
    try std.testing.expectApproxEqAbs(@as(f64, 1.3), o.get("accepted_per_round").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2.3), o.get("tokens_per_round").?.float, 1e-9);
    const mean = o.get("mean").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 55), mean.get("round_ms").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 3), mean.get("draft_ms").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1), mean.get("verify_encode_ms").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 20), mean.get("union_per_layer").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), mean.get("ahead_coverage").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), mean.get("read_mb").?.float, 1e-9);
    // round_ms 10..100: p50 = the 5th = 50, p90 = the 9th = 90; other = round - (draft 3 + verify 5)
    try std.testing.expectApproxEqAbs(@as(f64, 50), o.get("p50").?.object.get("round_ms").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 90), o.get("p90").?.object.get("round_ms").?.float, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 82), o.get("p90").?.object.get("other_ms").?.float, 1e-9);
    for (fields) |fd| try std.testing.expect(o.get("mean").?.object.get(fd.name) != null);
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    try writeTsv(&aw.writer, rs[0..2]);
    var it = std.mem.splitScalar(u8, aw.written(), '\n');
    const head = it.next().?;
    try std.testing.expect(std.mem.startsWith(u8, head, "round\trows\tdrafted\taccepted\tround_ns\tdraft_build_ns"));
    try std.testing.expectEqual(@as(usize, @typeInfo(Round).@"struct".field_names.len + 1), std.mem.count(u8, head, "\t") + 1);
    try std.testing.expect(std.mem.startsWith(u8, it.next().?, "0\t4\t3\t0\t10000000\t1000000"));
}

test "mimo mtp profile: a served build compiles every hook to nothing" {
    if (enabled) return error.SkipZigTest;
    try std.testing.expect(@TypeOf(state) == void);
    try std.testing.expect(@TypeOf(roundBegin(.{})) == void);
    unionNote(&.{ 1, 2, 3 });
    roundEnd(.{}, 3, 2);
    try std.testing.expectEqual(@as(u32, 0), rounds());
}

test "mimo mtp profile: the GPU timeline repairs wait-only buffers (GPUStartTime 0) before the gated waits are summed" {
    // The first MiMo MTP cell's raw rows (prof-mtp-bf16-20261008 mtp-typ-normal, 1,696 of 117,350 rows): status 4,
    // GPUStartTime 0, GPUEndTime set, committed behind a layer's hit pass. The hit pass commits its wave before the
    // barrier's host read, so no buffer lands between the route and hit stamps (hit_span null) and the gated wait
    // started at the first post-hit buffer's start: 0. T = a realistic mach-absolute origin.
    const tl = timeline;
    const T: u64 = 1_900_000_000_000_000;
    const tag: u64 = @as(u64, 1) << 8 | @as(u64, @intFromEnum(tl.Phase.verify));
    var rs = [_]tl.Row{
        .{ .host_commit = T + 111, .gpu_start = T + 111, .gpu_end = T + 114, .tag = tag, .queue = 1, .status = 4 }, // L0 barrier ids
        .{ .host_commit = T + 112, .gpu_start = T + 114, .gpu_end = T + 130, .tag = tag, .queue = 1, .status = 4 }, // L0 hit pass
        .{ .host_commit = T + 123, .gpu_start = 0, .gpu_end = T + 160, .tag = tag, .queue = 1, .status = 4 }, // wait for gu
        .{ .host_commit = T + 124, .gpu_start = T + 160, .gpu_end = T + 170, .tag = tag, .queue = 1, .status = 4 }, // gated gate/up
        .{ .host_commit = T + 124, .gpu_start = 0, .gpu_end = T + 175, .tag = tag, .queue = 1, .status = 4 }, // wait for down
        .{ .host_commit = T + 124, .gpu_start = T + 175, .gpu_end = T + 185, .tag = tag, .queue = 1, .status = 4 }, // gated down
        .{ .host_commit = T + 201, .gpu_start = T + 201, .gpu_end = T + 203, .tag = tag, .queue = 1, .status = 4 }, // L1 barrier
        .{ .host_commit = T + 202, .gpu_start = T + 203, .gpu_end = T + 220, .tag = tag, .queue = 1, .status = 4 }, // L1 hit pass
        .{ .host_commit = T + 209, .gpu_start = T + 221, .gpu_end = T + 221, .tag = tag, .queue = 1, .status = 4 }, // empty
    };
    var ls: [2]tl.Layer = .{
        .{ .t = .{ T + 110, T + 120, T + 121, T + 122, T + 125 }, .missed = true, .gate_gu = 7, .gate_down = 9 },
        .{ .t = .{ T + 200, T + 205, T + 206, T + 207, T + 208 } },
    };
    const sig = [_]tl.Signal{ .{ 7, @intCast(T + 158), @intCast(T + 157), 1 }, .{ 9, @intCast(T + 174), @intCast(T + 173), 1 } };
    var spans_buf: [16][2]u64 = undefined; // the timeline's Span
    var per: [2]tl.PerLayer = undefined;
    const v: tl.Verify = .{ .begin = T + 100, .end = T + 400 };
    // Raw: the wait runs from time 0 (the cell's bytes_wait_ms ~9.7e9) and gpu_work saturates to 0.
    const raw = tl.cycleSum(&rs, v, &ls, 0, &sig, &spans_buf, &per);
    try std.testing.expect(per[0].hit_span == null);
    try std.testing.expect(raw.wait > raw.wall);
    try std.testing.expectEqual(@as(u64, 0), raw.work);
    // Repaired: each wait-only row starts at the previous row's end (the queue runs rows in commit order).
    const rep = tl.repairRows(&rs);
    try std.testing.expectEqual(tl.Repair{ .unstarted = 2, .repaired = 2, .zero_length = 1 }, rep);
    try std.testing.expectEqual(T + 130, rs[2].gpu_start);
    try std.testing.expectEqual(T + 170, rs[4].gpu_start);
    try std.testing.expectEqual(@as(u64, 4 | tl.repaired_status), rs[2].status);
    try std.testing.expectEqual(@as(u64, 4), rs[3].status);
    const s = tl.cycleSum(&rs, v, &ls, 0, &sig, &spans_buf, &per);
    // busy 111..185 + 201..220 = 74 + 19; waits from the queue's arrival (130) to the down / gu signals (174 / 158)
    try std.testing.expectEqual(@as(u64, 300), s.wall);
    try std.testing.expectEqual(@as(u64, 93), s.busy);
    try std.testing.expectEqual(@as(?u64, 44), per[0].wait);
    try std.testing.expectEqual(@as(?u64, 28), per[0].wait_gu);
    try std.testing.expectEqual(@as(u64, 93 - 44), s.work);
    try std.testing.expect(s.wait <= s.busy and s.busy <= s.wall);
    // Idempotent: a second pass finds nothing to repair (the starts are set).
    try std.testing.expectEqual(tl.Repair{ .unstarted = 0, .repaired = 0, .zero_length = 1 }, tl.repairRows(&rs));
}
