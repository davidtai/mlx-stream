//! MiMo K6 `decode_commits` and K7 `decode_miss_issue` (mimo/k6-commits): the decode lane's command-buffer commits and
//! its routing barrier's wait, two construction-time routes installed into MiMo's math (`Math.barrier`) and run by the
//! shared scheduler's comptime seams (`ExpertsWith.run`: `decodeCommit`, `decodeRead`, `decodeWave`). A math without
//! them (DeepSeek's) compiles the scheduler's inline sequence, unchanged.
//!
//! Per routed decode-lane call (the serial token, M = 1, and the MTP verify, M <= 4, `Source.decodeLane`) the scheduler
//! builds the barrier's arrays (`bar`: the routed ids, then the lookahead predictor's scores) and, on the hit pass
//! (DEVROUTE), the device hit wave; then:
//!   `decodeCommit(bar, dev)`  the commits ahead of the wait,
//!   `decodeRead(bar, ...)`    the wait and the host's reads of the ids and scores (the route then issues the reads),
//!   `decodeWave(outs)`        the host hit wave's commit (the fused / stock routes: the hits' rows after the route).
//!
//! K6 `decode_commits`:
//!   "mlx" (the control): MLX's own command-buffer cuts. `CommandEncoder::needs_commit` (MLX 0.32.3 device.cpp:511)
//!   commits after any primitive once the open buffer holds > 50 dispatches or > 50 MB of distinct inputs (this M5 Max);
//!   the decode lane's weights (q_proj 100 MB, o_proj 67 MB BF16) and slot banks (> 1 GB each) cut after nearly every
//!   big kernel: the barrier set in 2 (sliding) / 3 (full) buffers, the hit wave after gate/up and after down, and the
//!   wave's eval epilogue (its event signal + finalize) in a third, otherwise empty buffer. Measured: 5.15 / 6.0
//!   buffers per sliding / full layer, 249 per verify round (cells/barrier1-bf16-20261008, MIMO_BARRIER_RESIDUAL).
//!   "layer": the device's two commit limits are raised for the call's commits only (`mimo_cb_swap_limits`,
//!   csrc/mimo_commit_shim.inc) and restored before the call returns: each eval commits once, at its epilogue. The
//!   barrier set [the previous layer's gated miss parts, join, combine, residual; this layer's attention, router,
//!   predictor] is one buffer, the hit wave one, nothing near-empty: 2 per layer. The miss parts stay what they are
//!   (event-gated lazy waves that ride the next barrier set: MiMo never commits them on their own). Every other eval
//!   of the process (the prompt pass, the logits, the drafts) keeps MLX's cuts.
//! K7 `decode_miss_issue`:
//!   "host" (the control): `evalAll(bar)` + `hostF32` + `hostIds`: MLX's `array::wait`, a blocking
//!   `waitUntilSignaledValue` on the barrier eval's MTLSharedEvent (44-65 us from the set's GPU end to the return).
//!   "event": the barrier's arrays are committed asynchronously (the hit pass already does), the host polls that same
//!   event (`_mlx_array_is_available`: MLX's own non-blocking test of the value the wait blocks on; it detaches the
//!   event and raises its error like the wait), then copies the ids and scores out of their evaluated buffers in place
//!   (no eval call). The route issues the layer's miss reads as soon as the GPU's signal is visible, not after the
//!   kernel's wake. The decode LUT rewrite after the route writes only the entries a route changed
//!   (`stream.StreamSource.installLutDelta`).
//! Exact by construction: the same graphs, the same arrays committed in the same order on the same stream, the same ids
//! and scores read from the same buffers, the same route. Only the command-buffer packaging (K6) and the host's wake
//! (K7) move. Per call, never per layer: a schedule of several row groups per layer (mimo/pipelined-verify) runs each
//! group's call through the same three seams (and commits a set it queues ahead of its call through `commitSet`); the
//! limits are restored before each call returns and each poll reads its own eval's event (MLX makes a fresh event per
//! eval), so interleaved groups cannot see each other's state.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");

pub const Commits = enum { mlx, layer, invalid };
pub const MissIssue = enum { host, event, invalid };

/// A settings string's route (`"mlx"` / `"layer"`; `"host"` / `"event"`); anything else is `.invalid`, refused by name
/// in `settings.Config.validate`.
pub fn parseCommits(s: []const u8) Commits {
    const c = std.meta.stringToEnum(Commits, s) orelse return .invalid;
    return c;
}

pub fn parseMissIssue(s: []const u8) MissIssue {
    const i = std.meta.stringToEnum(MissIssue, s) orelse return .invalid;
    return i;
}

// ── the commit limits (csrc/mimo_commit_shim.inc) ──
const shim = if (builtin.os.tag == .macos) struct {
    extern fn mimo_cb_swap_limits(ops_: c_int, mb: c_int, prev_ops: *c_int, prev_mb: *c_int) c_int;
    extern fn mimo_cb_limits(ops_: *c_int, mb: *c_int) c_int;
    extern fn mimo_cb_selfcheck(s: mlx.mlx_stream) c_int;
    extern fn mimo_cb_last_error() [*:0]const u8;
} else struct {
    fn mimo_cb_swap_limits(_: c_int, _: c_int, _: *c_int, _: *c_int) c_int {
        return -1;
    }
    fn mimo_cb_limits(_: *c_int, _: *c_int) c_int {
        return -1;
    }
    fn mimo_cb_selfcheck(_: mlx.mlx_stream) c_int {
        return -1;
    }
    fn mimo_cb_last_error() [*:0]const u8 {
        return "no Metal device";
    }
};

/// The device's commit limits (ops, MB), as MLX's own getter reads them.
pub const Limits = struct { ops: c_int, mb: c_int };

/// The limits inside a "layer" scope: no size or op count cuts a decode-lane eval; its epilogue commits it once.
pub const raised: Limits = .{ .ops = std.math.maxInt(c_int), .mb = std.math.maxInt(c_int) };

pub fn limits() !Limits {
    var l: Limits = undefined;
    if (shim.mimo_cb_limits(&l.ops, &l.mb) != 0) return error.MimoCommitLimits;
    return l;
}

/// Writes `to` and returns the limits it replaced.
pub fn swapLimits(to: Limits) !Limits {
    var prev: Limits = undefined;
    if (shim.mimo_cb_swap_limits(to.ops, to.mb, &prev.ops, &prev.mb) != 0) return error.MimoCommitLimits;
    return prev;
}

/// Once at construction, on the stream the decode lane runs on (`Source`, under the caller's lease): on a GPU stream,
/// libmlx's compiled `needs_commit` must read the limits this route writes (`mimo_cb_selfcheck`); a CPU stream has no
/// command buffer to cut (nothing to prove). Refuses the "layer" route by name otherwise.
pub fn selfCheck(s: mlx.mlx_stream) !enum { proven, cpu_stream } {
    return switch (shim.mimo_cb_selfcheck(s)) {
        0 => .proven,
        1 => .cpu_stream,
        else => {
            std.log.scoped(.mimo).err("decode_commits=layer refused: {s}", .{std.mem.span(shim.mimo_cb_last_error())});
            return error.MimoDecodeCommitsSelfCheck;
        },
    };
}

/// The poll's bound: a barrier set that has not signalled by then fails the call (a GPU that stopped; MLX's own wait
/// never returns in that case).
pub const poll_timeout_ns: u64 = 30 * std.time.ns_per_s;

/// The trace backend's record of the routes' primitives (the CPU proofs): each at the trace's node and commit counts.
pub const Primitive = enum { scope_begin, scope_end, poll, read_scores, read_ids };
pub const Entry = struct { what: Primitive, node: usize, commits: usize, evals: usize };
pub const TraceLog = struct {
    a: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    pub fn deinit(l: *TraceLog) void {
        l.entries.deinit(l.a);
    }
    /// The entries recorded since `from` (a call's).
    pub fn since(l: *const TraceLog, from: usize) []const Entry {
        return l.entries.items[from..];
    }
};

/// The routes over backend `G` (`ops.MlxOps` served; `ops.TraceOps` for the CPU proofs), bound once by `of`.
pub fn Routes(comptime G: type) type {
    comptime {
        if (G != ops.MlxOps and G != ops.TraceOps) @compileError("decode_barrier: an MLX or trace backend");
    }
    const T = G.T;
    const traced = G == ops.TraceOps;
    return struct {
        const Self = @This();
        pub const CommitFn = fn (r: *const Self, g: *G, bar: []const T, dev: ?T) anyerror!void;
        pub const ReadFn = fn (r: *const Self, g: *G, bar: []const T, indices: T, sc: ?T, ids_out: []u16, score_out: []f32, scores: *[]const f32) anyerror![]const u16;
        pub const WaveFn = fn (r: *const Self, g: *G, outs: []const T) anyerror!void;

        commits: Commits = .mlx,
        issue: MissIssue = .host,
        commit_fn: *const CommitFn = commitDev,
        read_fn: *const ReadFn = readHost,
        wave_fn: *const WaveFn = waveMlx,
        /// A barrier set committed ahead of its call (`commitSet`: a row-group schedule's first half), same packaging as
        /// the call's own commits.
        set_fn: *const WaveFn = waveMlx,
        /// The trace backend's log (tests set it); `void` on MLX.
        log: if (traced) ?*TraceLog else void = if (traced) null else {},

        /// The control: MLX's cuts, the blocking wait (the scheduler's inline sequence, op for op).
        pub const control: Self = .{};

        /// The construction-time binding; an `.invalid` route refuses by name.
        pub fn of(c: Commits, i: MissIssue) !Self {
            if (c == .invalid) return error.MimoDecodeCommits;
            if (i == .invalid) return error.MimoDecodeMissIssue;
            const layer = c == .layer;
            return .{
                .commits = c,
                .issue = i,
                .commit_fn = switch (i) {
                    .host => if (layer) &commitDevScoped else &commitDev,
                    .event => if (layer) &commitBarScoped else &commitBar,
                    .invalid => unreachable,
                },
                .read_fn = switch (i) {
                    .host => if (layer) &readHostScoped else &readHost,
                    .event => &readEvent,
                    .invalid => unreachable,
                },
                .wave_fn = if (layer) &waveScoped else &waveMlx,
                .set_fn = if (layer) &waveScoped else &waveMlx,
            };
        }

        pub fn commit(r: *const Self, g: *G, bar: []const T, dev: ?T) !void {
            return r.commit_fn(r, g, bar, dev);
        }

        pub fn read(r: *const Self, g: *G, bar: []const T, indices: T, sc: ?T, ids_out: []u16, score_out: []f32, scores: *[]const f32) ![]const u16 {
            return r.read_fn(r, g, bar, indices, sc, ids_out, score_out, scores);
        }

        pub fn wave(r: *const Self, g: *G, outs: []const T) !void {
            return r.wave_fn(r, g, outs);
        }

        /// Per group, not per layer: a schedule that commits a call's barrier set before the call (two row groups per
        /// layer, each group's set queued while the host serves the other) commits it here, so "layer" keeps it one
        /// buffer; the call's own `commit` then finds it scheduled (MLX's async_eval returns at once) and commits only
        /// the hit wave, and `read` polls the set's own eval event.
        pub fn commitSet(r: *const Self, g: *G, bar: []const T) !void {
            return r.set_fn(r, g, bar);
        }

        // ── commits ahead of the wait ──

        /// The control ("host"): DEVROUTE's barrier set, then its hit wave; without a device wave nothing (the host
        /// read's eval commits the set).
        fn commitDev(_: *const Self, g: *G, bar: []const T, dev: ?T) anyerror!void {
            if (dev) |d| {
                try g.asyncEval(bar);
                try g.asyncEval(&.{d});
            }
        }

        fn commitDevScoped(r: *const Self, g: *G, bar: []const T, dev: ?T) anyerror!void {
            const prev = try r.scopeBegin(g);
            defer r.scopeEnd(g, prev);
            return commitDev(r, g, bar, dev);
        }

        /// "event": the barrier set is always committed ahead of the poll (the hit pass's order: then its wave).
        fn commitBar(_: *const Self, g: *G, bar: []const T, dev: ?T) anyerror!void {
            try g.asyncEval(bar);
            if (dev) |d| try g.asyncEval(&.{d});
        }

        fn commitBarScoped(r: *const Self, g: *G, bar: []const T, dev: ?T) anyerror!void {
            const prev = try r.scopeBegin(g);
            defer r.scopeEnd(g, prev);
            return commitBar(r, g, bar, dev);
        }

        // ── the wait and the host's reads ──

        /// The control: the scheduler's sequence (a wait when the set is committed; else the eval commits it).
        fn readHost(_: *const Self, g: *G, bar: []const T, indices: T, sc: ?T, ids_out: []u16, score_out: []f32, scores: *[]const f32) anyerror![]const u16 {
            if (bar.len > 1) try g.evalAll(bar);
            if (sc) |s| scores.* = try g.hostF32(s, score_out);
            return g.hostIds(indices, ids_out);
        }

        /// "layer" + "host": the read's eval commits the set when nothing did (the fused / stock routes), in one buffer.
        fn readHostScoped(r: *const Self, g: *G, bar: []const T, indices: T, sc: ?T, ids_out: []u16, score_out: []f32, scores: *[]const f32) anyerror![]const u16 {
            const prev = try r.scopeBegin(g);
            defer r.scopeEnd(g, prev);
            return readHost(r, g, bar, indices, sc, ids_out, score_out, scores);
        }

        /// "event": poll the set's eval event, then read the ids and scores in place.
        fn readEvent(r: *const Self, g: *G, bar: []const T, indices: T, sc: ?T, ids_out: []u16, score_out: []f32, scores: *[]const f32) anyerror![]const u16 {
            try r.poll(g, bar);
            if (sc) |s| scores.* = try r.readF32(g, s, score_out);
            return r.readIds(g, indices, ids_out);
        }

        // ── the host hit wave (the fused / stock routes) ──

        fn waveMlx(_: *const Self, g: *G, outs: []const T) anyerror!void {
            return g.asyncEval(outs);
        }

        fn waveScoped(r: *const Self, g: *G, outs: []const T) anyerror!void {
            const prev = try r.scopeBegin(g);
            defer r.scopeEnd(g, prev);
            return g.asyncEval(outs);
        }

        // ── the backend's primitives ──

        fn note(r: *const Self, g: *const G, what: Primitive) !void {
            if (comptime !traced) return;
            const l = r.log orelse return;
            try l.entries.append(l.a, .{ .what = what, .node = g.nodes.items.len, .commits = g.commit_spans.items.len, .evals = g.evals.items.len });
        }

        fn scopeBegin(r: *const Self, g: *G) !Limits {
            if (comptime traced) {
                try r.note(g, .scope_begin);
                return raised;
            }
            return swapLimits(raised);
        }

        /// Restores the limits the scope replaced (a failed restore leaves the process's cuts raised: fail loudly).
        fn scopeEnd(r: *const Self, g: *G, prev: Limits) void {
            if (comptime traced) {
                r.note(g, .scope_end) catch @panic("decode_barrier: trace log out of memory");
                return;
            }
            _ = swapLimits(prev) catch @panic("decode_barrier: the commit limits could not be restored");
        }

        fn poll(r: *const Self, g: *G, bar: []const T) !void {
            if (comptime traced) return r.note(g, .poll);
            for (bar) |x| try pollReady(x);
        }

        fn readIds(r: *const Self, g: *G, x: T, out: []u16) ![]const u16 {
            if (comptime traced) {
                try r.note(g, .read_ids);
                if (g.dtypeOf(x) != .int32) return error.HostIdsDtype;
                if (g.shapeOf(x).numel() != @as(i64, @intCast(out.len))) return error.HostIdsSize;
                const hv = g.host_values orelse return error.NoHostValues;
                try hv.ids(hv.ctx, out);
                return out;
            }
            return readIdsInPlace(x, out);
        }

        fn readF32(r: *const Self, g: *G, x: T, out: []f32) ![]const f32 {
            if (comptime traced) {
                try r.note(g, .read_scores);
                if (g.dtypeOf(x) != .float32) return error.HostReadDtype;
                if (g.shapeOf(x).numel() != @as(i64, @intCast(out.len))) return error.HostReadSize;
                const hv = g.host_values orelse return error.NoHostValues;
                try (hv.f32s orelse return error.NoHostValues)(hv.ctx, out);
                return out;
            }
            return readF32InPlace(x, out);
        }
    };
}

/// Spins until `x`'s eval event is signalled (`array::is_available`: MLX's non-blocking test of the value `array::wait`
/// blocks on; once signalled it detaches the event, raising the eval's error as the wait does, and marks the array
/// available). `x` was committed by this call's `decodeCommit` (the route installs the commit with the poll).
pub fn pollReady(x: mlx.mlx_array) !void {
    var spins: u32 = 0;
    var t0: u64 = 0;
    while (true) {
        var ok = false;
        try mlx.check(mlx._mlx_array_is_available(&ok, x));
        if (ok) return;
        std.atomic.spinLoopHint();
        spins +%= 1;
        if (spins & 0xffff == 0) {
            const now = monotonicNs();
            if (t0 == 0) t0 = now else if (now - t0 > poll_timeout_ns) return error.MimoBarrierPollTimeout;
        }
    }
}

fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The routed ids of an available int32 array, row-major, read in place through its strides (`MlxOps.hostIds` without
/// its eval call: the poll made it available).
pub fn readIdsInPlace(x: mlx.mlx_array, out: []u16) ![]const u16 {
    if (mlx.mlx_array_size(x) != out.len) return error.HostIdsSize;
    std.debug.assert(mlx.mlx_array_dtype(x) == .int32);
    const nd = mlx.mlx_array_ndim(x);
    ops.copyStrided(i32, u16, mlx.mlx_array_data_int32(x) orelse return error.MlxNoData, mlx.mlx_array_shape(x)[0..nd], mlx.mlx_array_strides(x)[0..nd], out);
    return out;
}

/// The F32 values of an available array, row-major, in place (`MlxOps.hostF32` without its eval call).
pub fn readF32InPlace(x: mlx.mlx_array, out: []f32) ![]const f32 {
    if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
    std.debug.assert(mlx.mlx_array_dtype(x) == .float32);
    const nd = mlx.mlx_array_ndim(x);
    ops.copyStrided(f32, f32, mlx.mlx_array_data_float32(x) orelse return error.MlxNoData, mlx.mlx_array_shape(x)[0..nd], mlx.mlx_array_strides(x)[0..nd], out);
    return out;
}
