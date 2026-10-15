//! glm5_next's profile build (`sdk.plugin_profile`: the host's `-Dplugin-profile=true`, never a served build): MiMo's
//! diagnostics on GLM-5.3-Flash's serial lane. Each switch is read once at construction (`arm`):
//!   `GLM5_NEXT_ROUTE_TRACE=<prefix>`  the request's route trace over the bank's 288 experts (route_trace.zig `TraceOf(u16)`,
//!                                     files `<prefix>-r<k>.*`), the prompt timeline's TSVs beside it;
//!   `GLM5_NEXT_PREFILL_TIMELINE=1`    MiMo's prompt-pass timeline (prefill_timeline.zig, glm5_next's stages: the DSA split);
//!   `GLM5_NEXT_GPU_TIMELINE=1`        DSV's verify GPU timeline over the serial decode's forwards (one forward a cycle with
//!                                     one verify; the first `verify_timeline.max_cycles` forwards of the first request).
//! Every request also prints MiMo's per-phase lines (`GLM5_NEXT_PROFILE`, `_AHEAD`, `_AHEAD_LOOKAHEAD`, `_IO`) and, at the
//! decode's end, DSV's `DECODE_TIMERS` (a serial cycle is a token). The state lives here (no Module field); in a served
//! build every entry below compiles to nothing.
//!
//! Its own memory, outside the arch's bill (the cell reserves it on the baseline, `GLM_PROFILE_RESERVE_GB` 0.25): the
//! prompt timeline's static rows, marks, tiles and signals (~12 MB) and the verify timeline's (~24 MB), the route trace's
//! request buffers (u16 ids: ~11 MB of prompt ids at 16,384 tokens, ~11 MB of decode ids and ~16 MB of predictor records
//! at its 4,096 decode calls), the TSV text and the GPU timeline's JSON (each at most ~15 MB, built once at an end): at
//! most ~0.1 GB.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const expert = @import("../../sdk_ext.zig").expert;
const mimo_profile = @import("../mimo_v2/profile.zig");
const ptl = @import("../mimo_v2/prefill_timeline.zig");
const dt = @import("../deepseek_v41/decode_timers.zig");
const timeline = @import("../deepseek_v41/verify_timeline.zig");
const experts = @import("experts.zig");

pub const enabled = mimo_profile.enabled;

comptime {
    if (dt.enabled != enabled or timeline.enabled != enabled or ptl.enabled != enabled) @compileError("glm5_next: the profile probes follow one switch (sdk.plugin_profile)");
}

pub const tag = "GLM5_NEXT";
/// The serial decode's GPU timeline holds at most this many forwards (the shared timeline's cycle table).
pub const gpu_cycles_cap: u32 = timeline.max_cycles;

const Phase = enum { none, prefill, decode };

const State = struct {
    a: std.mem.Allocator = undefined,
    io: std.Io = undefined,
    trace: ?*experts.Trace = null,
    gpu_wanted: bool = false,
    gpu_installed: bool = false,
    gpu_on: bool = false,
    gpu_truncated: bool = false,
    stream: ?mlx.mlx_stream = null,
    phase: Phase = .none,
    s0: expert.Stats = .{},
    steps: u64 = 0,
    record_bytes: u64 = 1,
    hidden: u32 = 0,
    inter: u32 = 0,
    layers: u32 = 0,
};
var state: if (enabled) State else void = if (enabled) .{} else {};

/// Construction (after the source): every switch read once, each armed one's line printed; an unreadable value refuses
/// the load by name.
pub fn arm(a: std.mem.Allocator, io: std.Io, src: anytype, n_layers: u32, s: mlx.mlx_stream) !void {
    if (comptime !enabled) return;
    const geo = src.bank.geometry;
    state = .{ .a = a, .io = io, .stream = s, .record_bytes = geo.record_bytes, .hidden = geo.hidden_size, .inter = geo.intermediate_size, .layers = n_layers };
    if (std.c.getenv(tag ++ "_ROUTE_TRACE")) |p| {
        const prefix = std.mem.span(p);
        const t = try a.create(experts.Trace);
        errdefer a.destroy(t);
        t.* = try experts.Trace.init(a, io, prefix, geo.num_moe_layers, geo.num_experts, geo.first_moe_layer, geo.record_bytes);
        state.trace = t;
        src.stream_source.trace = t;
        log.info(tag ++ "_ROUTE_TRACE armed: prefix={s} format={s}\n", .{ prefix, experts.Trace.trace_format });
    }
    if (ptl.wantFor(io, .glm5_next) catch return error.Glm5NextPrefillTimelineSetting)
        log.info(tag ++ "_PREFILL_TIMELINE armed: first prompt pass, files beside the route trace={}\n", .{state.trace != null});
    if (std.c.getenv(tag ++ "_GPU_TIMELINE")) |p| {
        const v = std.mem.span(p);
        if (std.mem.eql(u8, v, "1")) {
            state.gpu_wanted = true;
            log.info(tag ++ "_GPU_TIMELINE armed: serial forwards, cycles_cap={d}\n", .{gpu_cycles_cap});
        } else if (!std.mem.eql(u8, v, "0")) {
            log.err(tag ++ "_GPU_TIMELINE={s} refused: 0 | 1\n", .{v});
            return error.Glm5NextGpuTimelineSetting;
        }
    }
}

/// The module's end: the trace freed (the source still alive).
pub fn disarm(src: anytype) void {
    if (comptime !enabled) return;
    if (state.trace) |t| {
        src.stream_source.trace = null;
        t.deinit();
        state.a.destroy(t);
        state.trace = null;
    }
}

/// A request's prompt pass begins (the source in its prompt phase): its trace, the phase's counters, the timeline.
pub fn promptBegin(src: anytype, ids: []const u32, reserve: u64) void {
    if (comptime !enabled) return;
    if (state.trace) |t| t.begin(ids, reserve -| ids.len) catch |e| log.err(tag ++ "_ROUTE_TRACE begin: {s}\n", .{@errorName(e)});
    startPhase(.prefill, src);
    ptl.begin(state.stream.?);
}

/// The prompt pass's end (its logits evaluated, before the phase change): the timeline's line and files, the phase lines.
pub fn promptEnd(src: anytype) void {
    if (comptime !enabled) return;
    const base = requestBase();
    defer if (base) |b| state.a.free(b);
    ptl.finish(state.a, base, state.hidden, state.inter);
    endPhase(src);
}

/// The decode phase begins (after the phase change): its counters, the residents' snapshot, the GPU timeline (the first
/// request's, at most `budget` forwards and `gpu_cycles_cap`).
pub fn decodeBegin(src: anytype, budget: u64) void {
    if (comptime !enabled) return;
    startPhase(.decode, src);
    state.steps = 0;
    if (state.trace) |t| traceResidents(src, t, false);
    if (!state.gpu_wanted) return;
    if (state.gpu_installed) {
        log.info("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"installed for the first request only\"}}\n", .{});
        return;
    }
    const cycles: u32 = @intCast(@min(@max(budget, 1), gpu_cycles_cap));
    timeline.install(state.stream.?, cycles + 1, state.layers) catch |e| {
        log.info("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"install: {s}\"}}\n", .{@errorName(e)});
        return;
    };
    state.gpu_installed = true;
    state.gpu_on = true;
    state.gpu_truncated = false;
}

/// A serial step's forward begins (its input token `id`): the trace's token, the GPU timeline's cycle and verify.
pub fn stepBegin(id: u32) dt.Stamp {
    if (comptime !enabled) return {};
    if (state.trace) |t| t.decodeToken(id);
    if (state.gpu_on) {
        // After the first forward the hook must have seen its commits (before it, nothing has committed since the
        // install at the phase change).
        if (timeline.cycles == 1) if (timeline.unavailableReason()) |why| {
            var buf: [1024]u8 = undefined;
            log.info("{s}\n", .{timeline.unavailableLine(&buf, why)});
            timeline.uninstall();
            state.gpu_on = false;
        };
        if (state.gpu_on and timeline.active and timeline.cycles >= timeline.max_cycles) {
            timeline.disarm();
            timeline.active = false;
            state.gpu_truncated = true;
        }
        timeline.cycleBegin();
        timeline.verifyBegin();
    }
    return dt.now();
}

/// The step's forward ended (its logits and states evaluated, the source flushed).
pub fn stepEnd(t0: dt.Stamp) void {
    if (comptime !enabled) return;
    _ = dt.charge(.verify, t0);
    if (state.gpu_on) timeline.verifyEnd();
    if (state.phase == .decode) state.steps += 1;
}

/// An MTP round's end (its verify was the step's forward, `stepBegin` traced t1): the round's accepted drafts and its
/// further tokens in the trace (`tokens` = [t1, the accepted drafts]).
pub fn roundEnd(accepted: u32, tokens: []const u32) void {
    if (comptime !enabled) return;
    const t = state.trace orelse return;
    t.roundAccepted(accepted);
    for (tokens[1..]) |id| t.decodeToken(id);
}

/// The request's end (the decode phase's): the phase lines, `DECODE_TIMERS`, the GPU timeline's line and file, the
/// trace's files.
pub fn decodeEnd(src: anytype) void {
    if (comptime !enabled) return;
    if (state.phase != .decode) return;
    endPhase(src);
    dt.cycles = state.steps;
    var buf: [2048]u8 = undefined;
    log.info("{s}\n", .{dt.line(&buf)});
    dt.reset();
    const base = requestBase();
    defer if (base) |b| state.a.free(b);
    if (state.gpu_on) {
        const pending = timeline.settle(state.stream.?, 2000);
        const rep = timeline.repairUnstarted();
        log.info("VERIFY_GPU_TIMELINE_REPAIR {{\"unstarted\": {d}, \"repaired\": {d}, \"zero_length\": {d}, \"status_bit\": {d}}}\n", .{ rep.unstarted, rep.repaired, rep.zero_length, timeline.repaired_status });
        var gb: [16384]u8 = undefined;
        log.info("{s}\n", .{timeline.line(&gb, pending)});
        if (state.gpu_truncated) log.info("VERIFY_GPU_TIMELINE_TRUNCATED {{\"cycles\": {d}}}\n", .{timeline.max_cycles});
        if (base) |b| writeGpuJson(b);
        timeline.uninstall();
        state.gpu_on = false;
    }
    if (state.trace) |t| if (t.active) {
        traceResidents(src, t, true);
        t.write() catch |e| log.err(tag ++ "_ROUTE_TRACE write failed: {s}\n", .{@errorName(e)});
    };
}

/// `<prefix>-r<k>` of the request the armed trace is recording (owned by the caller), else null.
fn requestBase() ?[]u8 {
    const t = state.trace orelse return null;
    if (!t.active) return null;
    return t.basePath(state.a) catch null;
}

fn startPhase(phase: Phase, src: anytype) void {
    state.phase = phase;
    state.s0 = src.stats();
    mimo_profile.reset();
    dt.reset();
}

/// MiMo's phase lines (module.zig `endDiagnosticPhase`) under glm5_next's tag, over the phase's settled counters.
fn endPhase(src: anytype) void {
    const name = @tagName(state.phase);
    const before = state.s0;
    const after = src.stats();
    mimo_profile.reportReadBytesAs(tag, name, state.record_bytes, after.expert_bytes_read - before.expert_bytes_read);
    log.info(tag ++ "_AHEAD phase={s} ahead_posted={d} ahead_hits={d} ahead_demand={d} ahead_bytes={d} ahead_dropped={d}\n", .{
        name,                                     after.ahead_posted - before.ahead_posted, after.ahead_hits - before.ahead_hits,
        after.ahead_demand - before.ahead_demand, after.ahead_bytes - before.ahead_bytes,
        after.ahead_dropped - before.ahead_dropped,
    });
    log.info(tag ++ "_AHEAD_LOOKAHEAD phase={s} spec_issued={d} spec_landed={d} spec_bytes={d} claimed={d} adopt_ranges={d} adopt_bytes={d} pre_issued={d} pre_served={d} pre_expired={d} gates={d} gates_forced={d}\n", .{
        name,                                   after.spec_issued - before.spec_issued,   after.spec_landed - before.spec_landed,
        after.spec_bytes - before.spec_bytes,   after.claimed - before.claimed,           after.adopt_ranges - before.adopt_ranges,
        after.adopt_bytes - before.adopt_bytes, after.pre_issued - before.pre_issued,     after.pre_served - before.pre_served,
        after.pre_expired - before.pre_expired, after.gates - before.gates,               after.gates_forced - before.gates_forced,
    });
    log.info(tag ++ "_IO phase={s} bytes={d} planned_loads={d} skipped_loads={d} hits={d} misses={d} read_range_sum_s={d:.6} read_inflight_wall_s={d:.6}\n", .{
        name,                                                                                                  after.expert_bytes_read - before.expert_bytes_read,
        (after.persistent_loads + after.transient_loads) - (before.persistent_loads + before.transient_loads), after.loads_skipped - before.loads_skipped,
        after.expert_cache_hits - before.expert_cache_hits,                                                    after.expert_cache_misses - before.expert_cache_misses,
        after.expert_read_seconds - before.expert_read_seconds,                                                @as(f64, @floatFromInt(after.read_wall_ns - before.read_wall_ns)) / 1e9,
    });
    if (state.trace) |t| {
        const io: @import("../mimo_v2/route_trace.zig").Io = .{ .hits = after.expert_cache_hits - before.expert_cache_hits, .misses = after.expert_cache_misses - before.expert_cache_misses, .bytes = after.expert_bytes_read - before.expert_bytes_read, .records = (after.expert_bytes_read - before.expert_bytes_read) / state.record_bytes };
        if (state.phase == .prefill) t.io_prefill = io else t.io_decode = io;
    }
    state.phase = .none;
}

/// Every routed layer's persistent slots into the trace: at the decode's start (`end` false) or its end.
fn traceResidents(src: anytype, t: *experts.Trace, end: bool) void {
    const geo = src.bank.geometry;
    for (0..geo.num_moe_layers) |rel| {
        const policy = src.stream_source.policy(geo.first_moe_layer + @as(u32, @intCast(rel)));
        t.snapshot(end, @intCast(rel), policy.slot_to_expert[0..policy.capacity]);
    }
}

fn writeGpuJson(base: []const u8) void {
    var aw: std.Io.Writer.Allocating = .init(state.a);
    defer aw.deinit();
    timeline.writeJson(&aw.writer) catch |e| return log.err(tag ++ "_GPU_TIMELINE json: {s}\n", .{@errorName(e)});
    const path = std.fmt.allocPrint(state.a, "{s}.gpu-timeline.json", .{base}) catch return;
    defer state.a.free(path);
    std.Io.Dir.cwd().writeFile(state.io, .{ .sub_path = path, .data = aw.written() }) catch |e| log.err(tag ++ "_GPU_TIMELINE write {s}: {s}\n", .{ path, @errorName(e) });
}

test "glm5_next profile: a served build compiles every entry to nothing" {
    if (enabled) return error.SkipZigTest;
    try std.testing.expect(@TypeOf(state) == void);
    try std.testing.expect(@TypeOf(stepBegin(7)) == void);
    stepEnd({});
}
