//! beam's profile build (`sdk.plugin_profile`: the host's `-Dplugin-profile=true`, never a served build): the shared
//! probes on Beam's serial lane, as glm5_next arms them (glm5_next/profile.zig), plus two raw logs for the cost
//! inventory. Each switch is read once (`armEarly` at the construction's start, `arm` at its end):
//!   `BEAM_PREFILL_TIMELINE=1`  MiMo's prompt-pass timeline (prefill_timeline.zig `.beam`: MiMo's stages), the first
//!                              prompt pass of the process;
//!   `BEAM_GPU_TIMELINE=1`      DSV's verify GPU timeline over the serial decode's forwards (one forward a cycle, the
//!                              first request's first `verify_timeline.max_cycles` steps);
//!   `BEAM_KERNEL_LOG=<steps>`  every compute dispatch of the inference thread (its kernel's function name, threadgroups,
//!                              threads per threadgroup and the command buffer that carried it: csrc/dsv41_cb_timeline.mm
//!                              `dsv41kl_*`) in the timed prompt pass and the first <steps> decode steps; with
//!                              MLX_MAX_OPS_PER_BUFFER=0 each command buffer holds one MLX primitive, so its GPU span is
//!                              that primitive's;
//!   `BEAM_PROFILE_OUT=<prefix>` where the raw rows land: `<prefix>.prefill-{rows,kernels,reads}.tsv` (and the timeline's
//!                              own `.prefill-{layers,tiles,cbs}.tsv`), `<prefix>.decode-{timeline.json,kernels.tsv,
//!                              reads.tsv}`.
//! The read pool's preadv log (every call's [start, end), bytes and thread class: csrc/q3_lookahead4_exl3.c
//! `q3ld_test_read_log`) runs with either timeline. Every request also prints MiMo's per-phase lines (`BEAM_PROFILE`,
//! `BEAM_AHEAD`, `BEAM_AHEAD_LOOKAHEAD`, `BEAM_IO`) and, at the decode's end, DSV's `DECODE_TIMERS` (a step is a cycle).
//! The state lives here (no Module field); in a served build every entry below compiles to nothing.
//!
//! Its own memory, outside the arch's bill: the prompt timeline's static rows (~29 MB at 2^19 rows), the verify
//! timeline's (~59 MB at 2^20 rows), the kernel log (64 MB of static rows, touched as filled), the read log (32 MB), the
//! TSV and JSON text built once at each phase's end (at most ~0.2 GB).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const mimo_profile = @import("../mimo_v2/profile.zig");
const ptl = @import("../mimo_v2/prefill_timeline.zig");
const dt = @import("../deepseek_v41/decode_timers.zig");
const timeline = @import("../deepseek_v41/verify_timeline.zig");

pub const enabled = mimo_profile.enabled;

comptime {
    if (dt.enabled != enabled or timeline.enabled != enabled or ptl.enabled != enabled) @compileError("beam: the profile probes follow one switch (sdk.plugin_profile)");
}

pub const tag = "BEAM";

/// One dispatch of the kernel log (the shim's `Dsv41klDispatch`).
pub const Dispatch = extern struct { row: u32, name: u32, grid: [3]u32, group: [3]u32 };
/// The read log's entries: [start ns, end ns, bytes (or the call's negative result), thread class].
pub const max_reads = 1 << 20;

const c = if (enabled) struct {
    pub extern fn dsv41kl_hook() c_int;
    pub extern fn dsv41kl_arm(on: c_int) void;
    pub extern fn dsv41kl_rows(out: *[4]u32) [*]const Dispatch;
    pub extern fn dsv41kl_names() [*]const u8;
    pub extern fn q3ld_test_read_log(buf: ?[*]i64, cap: i64, out_dropped: ?*i64) i64;
} else struct {};

const Phase = enum { none, prefill, decode };

const State = struct {
    a: std.mem.Allocator = undefined,
    io: std.Io = undefined,
    stream: ?mlx.mlx_stream = null,
    out: ?[]u8 = null,
    prefill_wanted: bool = false,
    gpu_wanted: bool = false,
    gpu_installed: bool = false,
    gpu_on: bool = false,
    gpu_truncated: bool = false,
    /// `BEAM_KERNEL_LOG`: the hook is installed; the decode steps the log covers.
    kernels: bool = false,
    kernel_steps: u64 = 0,
    kernels_on: bool = false,
    reads_on: bool = false,
    phase: Phase = .none,
    s0: @import("../../sdk_ext.zig").expert.Stats = .{},
    /// A draft round's start and its first chunk's (`roundBegin`, `chunkBegin`); the round's draft is not charged yet.
    round_t0: dt.Stamp = if (dt.enabled) 0 else {},
    chunk_t0: dt.Stamp = if (dt.enabled) 0 else {},
    round_drafting: bool = false,
    record_bytes: u64 = 1,
    hidden: u32 = 0,
    inter: u32 = 0,
    layers: u32 = 0,
};
var state: if (enabled) State else void = if (enabled) .{} else {};
const Storage = struct {
    var reads: [max_reads][4]i64 = undefined;
    /// Each cycle's open (`cycleOpen`: a serial step's forward begin, or a draft round's start), the timeline's clock.
    var opens: [timeline.max_cycles]u64 = undefined;
};

fn envValue(comptime name: [:0]const u8) ?[]const u8 {
    const p = std.c.getenv(name.ptr) orelse return null;
    return std.mem.span(p);
}

/// The construction's start (before any kernel is made): `BEAM_KERNEL_LOG` installs the kernel log's hook, so every
/// pipeline the model makes is named. An unreadable value refuses the load by name.
pub fn armEarly() !void {
    if (comptime !enabled) return;
    const v = envValue(tag ++ "_KERNEL_LOG") orelse return;
    const steps = std.fmt.parseInt(u64, v, 10) catch {
        log.err(tag ++ "_KERNEL_LOG={s} refused: the decode steps to log (0: the prompt pass only)\n", .{v});
        return error.BeamKernelLogSetting;
    };
    const rc = c.dsv41kl_hook();
    if (rc != 0) {
        log.err(tag ++ "_KERNEL_LOG: the hook failed at step {d}\n", .{rc});
        return error.BeamKernelLogHook;
    }
    state.kernels = true;
    state.kernel_steps = steps;
}

/// Construction (after the source): every other switch read once, each armed one's line printed.
pub fn arm(a: std.mem.Allocator, io: std.Io, src: anytype, n_layers: u32, s: mlx.mlx_stream) !void {
    if (comptime !enabled) return;
    const geo = src.bank.geometry;
    const kernels = state.kernels;
    const kernel_steps = state.kernel_steps;
    state = .{ .a = a, .io = io, .stream = s, .record_bytes = geo.record_bytes, .hidden = geo.hidden_size, .inter = geo.intermediate_size, .layers = n_layers, .kernels = kernels, .kernel_steps = kernel_steps };
    if (envValue(tag ++ "_PROFILE_OUT")) |p| state.out = try a.dupe(u8, p);
    state.prefill_wanted = ptl.wantFor(io, .beam) catch return error.BeamPrefillTimelineSetting;
    if (envValue(tag ++ "_GPU_TIMELINE")) |v| {
        if (std.mem.eql(u8, v, "1")) {
            state.gpu_wanted = true;
        } else if (!std.mem.eql(u8, v, "0")) {
            log.err(tag ++ "_GPU_TIMELINE={s} refused: 0 | 1\n", .{v});
            return error.BeamGpuTimelineSetting;
        }
    }
    log.info(tag ++ "_PROFILE armed: prefill_timeline={} gpu_timeline={} (cycles_cap={d}) kernel_log={} (decode_steps={d}) out={s}\n", .{ state.prefill_wanted, state.gpu_wanted, timeline.max_cycles, state.kernels, state.kernel_steps, state.out orelse "(none)" });
}

/// The module's end.
pub fn disarm() void {
    if (comptime !enabled) return;
    if (state.kernels_on) c.dsv41kl_arm(0);
    if (state.reads_on) _ = c.q3ld_test_read_log(null, -1, null);
    if (state.out) |o| state.a.free(o);
    state.out = null;
}

fn readsBegin() void {
    _ = c.q3ld_test_read_log(@ptrCast(&Storage.reads), max_reads, null);
    state.reads_on = true;
}

/// A request's prompt pass begins (the source in its prompt phase): the phase's counters, the timeline, the logs.
pub fn promptBegin(src: anytype) void {
    if (comptime !enabled) return;
    startPhase(.prefill, src);
    ptl.begin(state.stream.?);
    if (!state.prefill_wanted) return;
    readsBegin();
    if (state.kernels) {
        c.dsv41kl_arm(1);
        state.kernels_on = true;
    }
}

/// The prompt pass's end (its logits evaluated, before the phase change): the timeline's line and files, the raw rows,
/// the phase lines.
pub fn promptEnd(src: anytype) void {
    if (comptime !enabled) return;
    if (state.phase != .prefill) return;
    const was_recording = state.prefill_wanted and state.reads_on;
    ptl.finish(state.a, state.out, state.hidden, state.inter);
    if (was_recording) {
        if (state.kernels_on) {
            c.dsv41kl_arm(0);
            state.kernels_on = false;
        }
        if (state.out) |base| {
            writePrefillRows(base);
            writeKernels(base, ".prefill-kernels.tsv");
            writeReads(base, ".prefill-reads.tsv");
        }
        // The prompt's reads stay logged only for its window: the prefill timeline is the process's first pass only.
        state.prefill_wanted = false;
    }
    endPhase(src);
}

/// The decode phase begins (after the phase change): its counters, the GPU timeline (the first request's, at most
/// `budget` forwards and `max_cycles`), the logs.
pub fn decodeBegin(src: anytype, budget: u64) void {
    if (comptime !enabled) return;
    startPhase(.decode, src);
    if (!state.gpu_wanted) return;
    if (state.gpu_installed) {
        log.info("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"installed for the first request only\"}}\n", .{});
        return;
    }
    const cycles: u32 = @intCast(@min(@max(budget, 1), timeline.max_cycles));
    timeline.install(state.stream.?, cycles + 1, state.layers) catch |e| {
        log.info("VERIFY_GPU_TIMELINE_UNAVAILABLE {{\"reason\": \"install: {s}\"}}\n", .{@errorName(e)});
        return;
    };
    state.gpu_installed = true;
    state.gpu_on = true;
    state.gpu_truncated = false;
    readsBegin();
    if (state.kernels and state.kernel_steps > 0) {
        c.dsv41kl_arm(1);
        state.kernels_on = true;
    }
}

/// A cycle opens (a serial step's forward, or a draft round's start): the GPU timeline's checks and its cycle (the
/// buffers committed until `verifyBegin` carry the `draft` phase), the cycle's open stamp; the kernel log stops past its
/// steps.
fn cycleOpen() void {
    if (!state.gpu_on) return;
    // After the first forward the hook must have seen its commits (before it, nothing has committed since the install
    // at the phase change).
    if (timeline.cycles == 1) if (timeline.unavailableReason()) |why| {
        var buf: [1024]u8 = undefined;
        log.info("{s}\n", .{timeline.unavailableLine(&buf, why)});
        timeline.uninstall();
        state.gpu_on = false;
        return;
    };
    if (timeline.active and timeline.cycles >= timeline.max_cycles) {
        timeline.disarm();
        timeline.active = false;
        state.gpu_truncated = true;
    }
    if (state.kernels_on and timeline.cycles >= state.kernel_steps) {
        c.dsv41kl_arm(0);
        state.kernels_on = false;
    }
    if (timeline.active) Storage.opens[timeline.cycles] = timeline.now();
    timeline.cycleBegin();
}

/// A serial step's forward begins: a cycle and its verify window at once.
pub fn stepBegin() dt.Stamp {
    if (comptime !enabled) return {};
    cycleOpen();
    if (state.gpu_on) timeline.verifyBegin();
    return dt.now();
}

/// The step's forward ended (its logits committed, the source flushed).
pub fn stepEnd(t0: dt.Stamp) void {
    if (comptime !enabled) return;
    _ = dt.charge(.verify, t0);
    dt.countCycle();
    if (state.gpu_on) timeline.verifyEnd();
}

/// A draft round begins (module.zig `openRound`, lane.zig `roundWith`): one cycle for the round. The drafter's buffers
/// carry the cycle's `draft` phase, the verify chunks' the `verify` phase, the commit's appends the `rest` phase.
pub fn roundBegin() void {
    if (comptime !enabled) return;
    cycleOpen();
    dt.countCycle();
    state.round_t0 = dt.now();
    state.round_drafting = true;
}

/// A verify chunk begins (module.zig `verifyChunk`): the round's draft (its graph, eval and host reads) is charged
/// once, and the cycle's verify window opens at its first chunk (verify_timeline.zig keeps the first begin).
pub fn chunkBegin() void {
    if (comptime !enabled) return;
    if (state.round_drafting) {
        _ = dt.charge(.draft_build, state.round_t0);
        state.round_drafting = false;
    }
    if (state.gpu_on) timeline.verifyBegin();
    state.chunk_t0 = dt.now();
}

/// A verify chunk's routes released (module.zig `flushVerify`, after the lane's decision eval).
pub fn chunkEnd() void {
    if (comptime !enabled) return;
    _ = dt.charge(.verify, state.chunk_t0);
    if (state.gpu_on) timeline.verifyEnd();
}

/// The request's end (the decode phase's): the phase lines, `DECODE_TIMERS`, the GPU timeline's line and files, the
/// raw logs.
pub fn decodeEnd(src: anytype) void {
    if (comptime !enabled) return;
    if (state.phase != .decode) return;
    endPhase(src);
    var buf: [2048]u8 = undefined;
    log.info("{s}\n", .{dt.line(&buf)});
    dt.reset();
    if (!state.gpu_on) return;
    if (state.kernels_on) {
        c.dsv41kl_arm(0);
        state.kernels_on = false;
    }
    const pending = timeline.settle(state.stream.?, 2000);
    const rep = timeline.repairUnstarted();
    log.info("VERIFY_GPU_TIMELINE_REPAIR {{\"unstarted\": {d}, \"repaired\": {d}, \"zero_length\": {d}, \"status_bit\": {d}}}\n", .{ rep.unstarted, rep.repaired, rep.zero_length, timeline.repaired_status });
    var gb: [16384]u8 = undefined;
    log.info("{s}\n", .{timeline.line(&gb, pending)});
    if (state.gpu_truncated) log.info("VERIFY_GPU_TIMELINE_TRUNCATED {{\"cycles\": {d}}}\n", .{timeline.max_cycles});
    if (state.out) |base| {
        writeGpuJson(base);
        writeDecodeRows(base);
        writeKernels(base, ".decode-kernels.tsv");
        writeReads(base, ".decode-reads.tsv");
    }
    timeline.uninstall();
    state.gpu_on = false;
    if (state.reads_on) {
        _ = c.q3ld_test_read_log(null, -1, null);
        state.reads_on = false;
    }
}

fn startPhase(phase: Phase, src: anytype) void {
    state.phase = phase;
    state.s0 = src.stats();
    mimo_profile.reset();
    dt.reset();
}

/// MiMo's phase lines (glm5_next's `endPhase`) under Beam's tag, over the phase's settled counters.
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
    // perf/beam-misspath (diag): the speculative class's schedule, M1's co-read included; M2a's split copies.
    log.info(tag ++ "_SPEC phase={s} claimed_inflight={d} co_reads={d} abandoned={d} abandoned_bytes={d} cancelled={d} expired={d} discarded={d} split_ranges={d}\n", .{
        name,                                                       after.spec_claimed_inflight - before.spec_claimed_inflight,
        after.spec_co_reads - before.spec_co_reads,                 after.spec_abandoned - before.spec_abandoned,
        after.spec_abandoned_bytes - before.spec_abandoned_bytes,   after.spec_cancelled - before.spec_cancelled,
        after.spec_expired - before.spec_expired,                   after.spec_discarded - before.spec_discarded,
        after.split_ranges - before.split_ranges,
    });
    log.info(tag ++ "_IO phase={s} bytes={d} planned_loads={d} skipped_loads={d} hits={d} misses={d} read_range_sum_s={d:.6} read_inflight_wall_s={d:.6}\n", .{
        name,                                                                                                  after.expert_bytes_read - before.expert_bytes_read,
        (after.persistent_loads + after.transient_loads) - (before.persistent_loads + before.transient_loads), after.loads_skipped - before.loads_skipped,
        after.expert_cache_hits - before.expert_cache_hits,                                                    after.expert_cache_misses - before.expert_cache_misses,
        after.expert_read_seconds - before.expert_read_seconds,                                                @as(f64, @floatFromInt(after.read_wall_ns - before.read_wall_ns)) / 1e9,
    });
    state.phase = .none;
}

fn writeOut(base: []const u8, ext: []const u8, text: []const u8) void {
    const path = std.fmt.allocPrint(state.a, "{s}{s}", .{ base, ext }) catch return;
    defer state.a.free(path);
    std.Io.Dir.cwd().writeFile(state.io, .{ .sub_path = path, .data = text }) catch |e| log.err(tag ++ "_PROFILE write {s}: {s}\n", .{ path, @errorName(e) });
}

fn writeGpuJson(base: []const u8) void {
    var aw: std.Io.Writer.Allocating = .init(state.a);
    defer aw.deinit();
    timeline.writeJson(&aw.writer) catch |e| return log.err(tag ++ "_GPU_TIMELINE json: {s}\n", .{@errorName(e)});
    writeOut(base, ".decode-timeline.json", aw.written());
}

/// The decode's command-buffer rows in commit order (the kernel log's `row` is the line's index; `queue` 0 = another
/// queue's, the keep-warm engine's), each step's forward window and each routed layer's host stamps, the read pool's
/// event signals: absolute ns (the read log's clock).
fn writeDecodeRows(base: []const u8) void {
    const st = timeline.stored();
    {
        var aw: std.Io.Writer.Allocating = .init(state.a);
        defer aw.deinit();
        const w = &aw.writer;
        w.writeAll("row\tcycle\tphase\thost_commit_ns\tgpu_start_ns\tgpu_end_ns\thost_done_ns\tstatus\tqueue\n") catch return;
        for (st.rows, 0..) |r, i| {
            const cyc: i64 = if (timeline.rowCycle(r)) |x| x else -1;
            w.print("{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{ i, cyc, timeline.rowPhase(r), r.host_commit, r.gpu_start, r.gpu_end, r.host_done, r.status, r.queue }) catch return;
        }
        writeOut(base, ".decode-rows.tsv", aw.written());
    }
    {
        var aw: std.Io.Writer.Allocating = .init(state.a);
        defer aw.deinit();
        const w = &aw.writer;
        w.writeAll("cycle\tbegin_ns\tend_ns\topen_ns\n") catch return;
        for (st.verifies, 0..) |vf, i| w.print("{d}\t{d}\t{d}\t{d}\n", .{ i, vf.begin, vf.end, Storage.opens[i] }) catch return;
        writeOut(base, ".decode-steps.tsv", aw.written());
    }
    {
        var aw: std.Io.Writer.Allocating = .init(state.a);
        defer aw.deinit();
        const w = &aw.writer;
        w.writeAll("cycle\tlayer\tcall_ns\tbarrier_ns\troute_ns\thit_ns\tend_ns\tread_gauge_call_ns\tread_gauge_end_ns\tmissed\tgate_gu\tgate_down\n") catch return;
        for (st.layers, 0..) |ls, ci| for (ls[0..st.n_layers], 0..) |l, li| {
            if (l.t[0] == 0) continue;
            w.print("{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{ ci, li, l.t[0], l.t[1], l.t[2], l.t[3], l.t[4], l.read_call, l.read_end, @intFromBool(l.missed), l.gate_gu, l.gate_down }) catch return;
        };
        writeOut(base, ".decode-layers.tsv", aw.written());
    }
    {
        var aw: std.Io.Writer.Allocating = .init(state.a);
        defer aw.deinit();
        const w = &aw.writer;
        w.writeAll("value\tsignal_call_ns\tbytes_landed_ns\tcall_ns\n") catch return;
        for (timeline.storedSignals()) |sg| w.print("{d}\t{d}\t{d}\t{d}\n", .{ sg[0], sg[1], sg[2], sg[3] }) catch return;
        writeOut(base, ".decode-signals.tsv", aw.written());
    }
}

/// The prompt pass's command-buffer rows in commit order (the kernel log's `row` is the line's index), absolute ns.
fn writePrefillRows(base: []const u8) void {
    var aw: std.Io.Writer.Allocating = .init(state.a);
    defer aw.deinit();
    const w = &aw.writer;
    w.writeAll("row\thost_commit_ns\tgpu_start_ns\tgpu_end_ns\thost_done_ns\ttag\tstatus\tqueue\n") catch return;
    for (ptl.storedRows(), 0..) |r, i| w.print("{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{ i, r.host_commit, r.gpu_start, r.gpu_end, r.host_done, r.tag, r.status, r.queue }) catch return;
    writeOut(base, ".prefill-rows.tsv", aw.written());
}

fn writeKernels(base: []const u8, ext: []const u8) void {
    var n: [4]u32 = undefined;
    const rows = c.dsv41kl_rows(&n);
    const names = c.dsv41kl_names();
    var aw: std.Io.Writer.Allocating = .init(state.a);
    defer aw.deinit();
    const w = &aw.writer;
    w.print("# dispatches {d} dropped {d} names_bytes {d} pipelines {d}\n", .{ n[0], n[1], n[2], n[3] }) catch return;
    w.writeAll("i\trow\tname\tgx\tgy\tgz\ttx\tty\ttz\n") catch return;
    for (rows[0..n[0]], 0..) |d, i| {
        const name = std.mem.span(@as([*:0]const u8, @ptrCast(names + d.name)));
        w.print("{d}\t{d}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{ i, d.row, name, d.grid[0], d.grid[1], d.grid[2], d.group[0], d.group[1], d.group[2] }) catch return;
    }
    writeOut(base, ext, aw.written());
    log.info(tag ++ "_KERNEL_LOG {s}: dispatches={d} dropped={d} pipelines={d}\n", .{ ext, n[0], n[1], n[3] });
}

fn writeReads(base: []const u8, ext: []const u8) void {
    var dropped: i64 = 0;
    const n: usize = @intCast(@max(c.q3ld_test_read_log(null, 0, &dropped), 0));
    var aw: std.Io.Writer.Allocating = .init(state.a);
    defer aw.deinit();
    const w = &aw.writer;
    w.print("# reads {d} dropped {d}\n", .{ n, dropped }) catch return;
    w.writeAll("t0_ns\tt1_ns\tbytes\tclass\n") catch return;
    for (Storage.reads[0..@min(n, max_reads)]) |e| w.print("{d}\t{d}\t{d}\t{d}\n", .{ e[0], e[1], e[2], e[3] }) catch return;
    writeOut(base, ext, aw.written());
    log.info(tag ++ "_READ_LOG {s}: reads={d} dropped={d}\n", .{ ext, n, dropped });
}

test "beam profile: a served build compiles every entry to nothing" {
    if (enabled) return error.SkipZigTest;
    try std.testing.expect(@TypeOf(state) == void);
    try std.testing.expect(@TypeOf(stepBegin()) == void);
    stepEnd({});
    roundBegin();
    chunkBegin();
    chunkEnd();
    try armEarly();
}
