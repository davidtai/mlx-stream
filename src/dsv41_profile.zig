//! DSV4.1 native decode-cycle attribution, for profiling window binaries only.
//!
//! `Profiled(Ops)` wraps the model's graph backend (`deepseek_v41_ops.MlxOps` or `TraceOps`,
//! or any backend with their methods) and attributes each DSpark cycle to the categories of
//! the Python residue note (R/q3-decode-cycle-residue-20260927.md sec. 2a, 2c and the verify
//! host-stamp arm of sec. 7): the loop's timers (draft, verify forward, accept, commit) and
//! its glue, and inside the verify forward the host-stamp split's per layer-call buckets
//! (barrier build / encode, barrier eval, route readback, runner, drain commit and wake, read
//! wait, layer tail, forward head / tail), per layer.
//!
//! Time is the host's awake clock (CLOCK_UPTIME_RAW on macOS, the clock under Python's
//! perf_counter_ns), read at every phase boundary and around every sync the loop makes
//! (evalAll, host reads, asyncEval): the Python arm's phase timers and EVAL / AEVAL stamps.
//! A phase's time is exclusive (charged to the innermost open phase), so a cycle's phases
//! and its glue add up to its wall exactly. Counters are totals per cycle and tag, reset at
//! `cycleBegin`; nothing is kept per launch.
//!
//! Selection is compile-time. The hooks (`cycleBegin` / `cycleEnd`, `beginPhase` /
//! `endPhase`, `beginLayer` / `endLayer`) take the loop's backend pointer and compile to
//! nothing unless it is a `Profiled` type, so the serving build carries no counter, clock
//! read or branch; `Profiled` itself refuses to exist outside a test build or a root that
//! declares `pub const dsv41_profile_window = true;` (the serving root does not).

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

/// The categories. The first five are the loop's phases (the roots of a cycle's phase
/// stack); the rest are the verify forward's buckets.
pub const Tag = enum(u8) {
    /// the draft block (3 DSpark stages, head, markov), its eval, host reads, the lookup
    draft,
    /// one verify chunk: the target forward and its eval (Python `verify_ms`)
    verify,
    /// the target's decision on a chunk: the argmax (typical flags) sync and acceptance
    accept,
    /// trim, seed the draft windows, the evals that settle them (Python `commit_ms`)
    commit,
    /// the expert source's settle / unpin at the cycle end
    flush,
    /// forward head: positions, embedding, Engram rows
    fwd_head,
    /// one layer-call, open while the layer runs; exclusive = its graph build outside the
    /// MoE call and the tail (Engram, HC, attention, router): the barrier build / encode
    layer,
    /// the routed MoE call; exclusive = the runner glue (served, release, the join)
    moe,
    /// the routing barrier: predictor scores, the eval, the route readback
    barrier,
    /// the expert source's route: plan, slot lookups, read issue
    route,
    /// the residents' gate/up and down waves and their async evals
    hit_wave,
    /// blocking waits for expert bytes (waitGu / waitDown)
    read_wait,
    /// a miss part's gate/up or down wave and its async eval (the drain commit)
    drain,
    /// the layer's tail after the MoE (Seg3 / HC post)
    layer_tail,
    /// forward tail: final norm, the logits head graph
    fwd_tail,
};

pub const n_tags = std.meta.fieldNames(Tag).len;
/// The slot of time and counts with no phase open (inside a cycle: the loop's glue).
pub const glue = n_tags;
pub const n_slots = n_tags + 1;
pub const max_depth = 8;
pub const max_layers = 64;

fn ix(t: Tag) usize {
    return @backingInt(t);
}

pub fn slotName(s: usize) []const u8 {
    return if (s == glue) "glue" else @tagName(@as(Tag, @fromBackingInt(@intCast(s))));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// One slot's totals: exclusive wall time, the part of it spent blocked in syncs, and counts.
pub const Counts = struct {
    /// exclusive wall ns on the host clock
    host_ns: u64 = 0,
    /// of which blocked in evalAll (GPU-inclusive)
    eval_ns: u64 = 0,
    /// of which blocked in host reads (hostIds / hostU32 / hostF32 / hostBool / hostArgmax)
    read_ns: u64 = 0,
    /// of which inside asyncEval (encode + commit)
    async_ns: u64 = 0,
    /// phase entries
    calls: u32 = 0,
    /// kernel launches (launch / launchPrepared)
    launches: u32 = 0,
    /// compiled regions run (tape)
    regions: u32 = 0,
    evals: u32 = 0,
    reads: u32 = 0,
    async_evals: u32 = 0,
    /// other graph ops built
    ops: u32 = 0,

    fn add(a: *Counts, b: Counts) void {
        inline for (comptime std.meta.fieldNames(Counts)) |f| @field(a, f) += @field(b, f);
    }
};

/// Per (layer, tag) totals over the run.
pub const LayerCounts = struct { host_ns: u64 = 0, eval_ns: u64 = 0, read_ns: u64 = 0, calls: u32 = 0 };

/// One cycle (or the run's sum of cycles).
pub const Cycle = struct {
    wall_ns: u64 = 0,
    /// inclusive time per root (the outermost open phase); glue in slot `glue`: sums to wall
    root_ns: [n_slots]u64 = @splat(0),
    /// evalAll + host-read time under each root (the waits on the GPU)
    sync_ns: [n_slots]u64 = @splat(0),
    /// read_wait time under each root (verify: the read wait of the verify forward)
    wait_ns: [n_slots]u64 = @splat(0),
    tags: [n_slots]Counts = @splat(.{}),
    /// clock reads the profiler made in the cycle (its own cost: stamps x clock_ns)
    stamps: u64 = 0,
    unbalanced: bool = false,

    fn add(a: *Cycle, b: *const Cycle) void {
        a.wall_ns += b.wall_ns;
        a.stamps += b.stamps;
        for (0..n_slots) |s| {
            a.root_ns[s] += b.root_ns[s];
            a.sync_ns[s] += b.sync_ns[s];
            a.wait_ns[s] += b.wait_ns[s];
            a.tags[s].add(b.tags[s]);
        }
        a.unbalanced = a.unbalanced or b.unbalanced;
    }
};

/// The profiler's clock: the host's awake clock, or a caller-driven one (tests).
pub const Clock = union(enum) {
    awake: std.Io,
    manual: *const u64,

    pub fn now(c: Clock) u64 {
        return switch (c) {
            .awake => |io| @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds),
            .manual => |t| t.*,
        };
    }
};

pub const SyncKind = enum { eval, read, async_eval };

/// The attribution state of one backend (one generation thread).
pub const Profiler = struct {
    a: Allocator,
    clock: Clock,
    stack: [max_depth]Tag = undefined,
    depth: u8 = 0,
    /// phases begun past max_depth (not attributed; their ends are absorbed)
    overflow: u32 = 0,
    last: u64,
    in_cycle: bool = false,
    layer: ?u16 = null,
    cycle_t0: u64 = 0,
    cur: Cycle = .{},
    /// every cycle's sum (wall_ns = the sum of the walls)
    total: Cycle = .{},
    n_cycles: u64 = 0,
    first_wall_ns: u64 = 0,
    /// phases and syncs outside any cycle (prefill, the start-of-decode boundary)
    outside: [n_slots]Counts = @splat(.{}),
    per_layer: [max_layers][n_tags]LayerCounts = @splat(@splat(.{})),
    layer_calls: [max_layers]u32 = @splat(0),
    /// the receipt's cycles, in order (capacity fixed at init)
    stored: []Cycle,
    n_stored: usize = 0,
    dropped: u64 = 0,
    /// phase ends with none open, cycles ended with phases open, nested cycle begins
    unbalanced: u32 = 0,
    /// one clock read, calibrated at init (the Python arm's "stamp cost")
    clock_ns: f64 = 0,

    pub fn init(a: Allocator, clock: Clock, max_cycles: usize) !Profiler {
        const t0 = clock.now();
        for (0..256) |_| std.mem.doNotOptimizeAway(clock.now());
        const t1 = clock.now();
        return .{ .a = a, .clock = clock, .last = t1, .clock_ns = @as(f64, @floatFromInt(t1 - t0)) / 257.0, .stored = try a.alloc(Cycle, max_cycles) };
    }

    /// A clock read (counted: the profiler's own cost per cycle).
    pub fn now(p: *Profiler) u64 {
        if (p.in_cycle) p.cur.stamps += 1;
        return p.clock.now();
    }

    pub fn deinit(p: *Profiler) void {
        p.a.free(p.stored);
        p.* = undefined;
    }

    fn slot(p: *const Profiler) usize {
        return if (p.depth > 0) ix(p.stack[p.depth - 1]) else glue;
    }

    fn root(p: *const Profiler) usize {
        return if (p.depth > 0) ix(p.stack[0]) else glue;
    }

    fn counts(p: *Profiler) *Counts {
        return if (p.in_cycle) &p.cur.tags[p.slot()] else &p.outside[p.slot()];
    }

    /// Charges the time since the last boundary to the innermost open phase.
    fn charge(p: *Profiler, t: u64) void {
        const d = t -| p.last;
        p.last = t;
        const s = p.slot();
        if (!p.in_cycle) {
            p.outside[s].host_ns += d;
            return;
        }
        p.cur.tags[s].host_ns += d;
        const r = p.root();
        p.cur.root_ns[r] += d;
        if (s == ix(.read_wait)) p.cur.wait_ns[r] += d;
        if (p.layer) |l| if (s != glue) {
            p.per_layer[l][s].host_ns += d;
        };
    }

    fn push(p: *Profiler, tag: Tag) void {
        if (p.depth == max_depth) {
            p.overflow += 1;
            return;
        }
        p.stack[p.depth] = tag;
        p.depth += 1;
        p.counts().calls += 1;
        if (p.in_cycle) if (p.layer) |l| {
            p.per_layer[l][ix(tag)].calls += 1;
        };
    }

    pub fn begin(p: *Profiler, tag: Tag) void {
        p.charge(p.now());
        p.push(tag);
    }

    pub fn end(p: *Profiler) void {
        p.charge(p.now());
        if (p.overflow > 0) {
            p.overflow -= 1;
        } else if (p.depth == 0) {
            p.unbalanced += 1;
        } else {
            p.depth -= 1;
        }
    }

    /// Opens layer-call `l` (a `.layer` phase whose time and syncs also count per layer).
    pub fn beginLayer(p: *Profiler, l: u32) void {
        p.charge(p.now());
        p.layer = if (p.in_cycle and l < max_layers) @intCast(l) else null;
        if (p.layer) |x| p.layer_calls[x] += 1;
        p.push(.layer);
    }

    pub fn endLayer(p: *Profiler) void {
        p.end();
        p.layer = null;
    }

    /// A sync that began at `t0` has returned: its time is part of the innermost phase's.
    pub fn synced(p: *Profiler, comptime kind: SyncKind, t0: u64) void {
        const d = p.now() -| t0;
        const c = p.counts();
        switch (kind) {
            .eval => {
                c.eval_ns += d;
                c.evals += 1;
            },
            .read => {
                c.read_ns += d;
                c.reads += 1;
            },
            .async_eval => {
                c.async_ns += d;
                c.async_evals += 1;
            },
        }
        if (!p.in_cycle or kind == .async_eval) return;
        p.cur.sync_ns[p.root()] += d;
        const s = p.slot();
        if (p.layer) |l| if (s != glue) switch (kind) {
            .eval => p.per_layer[l][s].eval_ns += d,
            .read => p.per_layer[l][s].read_ns += d,
            .async_eval => unreachable,
        };
    }

    pub fn launch(p: *Profiler) void {
        p.counts().launches += 1;
    }

    pub fn region(p: *Profiler) void {
        p.counts().regions += 1;
    }

    pub fn op(p: *Profiler) void {
        p.counts().ops += 1;
    }

    pub fn cycleBegin(p: *Profiler) void {
        const t = p.clock.now();
        p.charge(t);
        if (p.in_cycle) p.unbalanced += 1;
        if (p.depth != 0 or p.overflow != 0) {
            p.unbalanced += 1;
            p.depth = 0;
            p.overflow = 0;
        }
        p.layer = null;
        p.cur = .{ .stamps = 1 };
        p.cycle_t0 = t;
        p.in_cycle = true;
    }

    pub fn cycleEnd(p: *Profiler) void {
        const t = p.now();
        p.charge(t);
        if (!p.in_cycle) {
            p.unbalanced += 1;
            return;
        }
        if (p.depth != 0 or p.overflow != 0) {
            p.cur.unbalanced = true;
            p.unbalanced += 1;
            p.depth = 0;
            p.overflow = 0;
        }
        p.layer = null;
        p.cur.wall_ns = t - p.cycle_t0;
        if (p.n_cycles == 0) p.first_wall_ns = p.cur.wall_ns;
        p.total.add(&p.cur);
        p.n_cycles += 1;
        if (p.n_stored < p.stored.len) {
            p.stored[p.n_stored] = p.cur;
            p.n_stored += 1;
        } else p.dropped += 1;
        p.in_cycle = false;
    }

    // ── Receipt ──

    /// One JSON line per stored cycle: wall, the roots' inclusive ms (the loop's timers),
    /// the verify's read wait, and per tag {host_ms, eval_ms, read_ms, async_ms, launches,
    /// regions, evals, reads, async_evals, ops, calls} (tags with no time and no calls left out).
    pub fn writeCycles(p: *const Profiler, w: *std.Io.Writer) std.Io.Writer.Error!void {
        for (p.stored[0..p.n_stored], 0..) |*c, i| try writeCycle(w, i, c);
    }

    /// The summary: the residue note's sec. 2a terms, the host-stamp split's buckets (sec. 7),
    /// every tag, and the per-layer table, ms per cycle over every cycle recorded.
    pub fn writeSummary(p: *const Profiler, w: *std.Io.Writer, label: []const u8) std.Io.Writer.Error!void {
        const n: f64 = @floatFromInt(@max(p.n_cycles, 1));
        const t = &p.total;
        const per = struct {
            fn c(nn: f64, ns: u64) f64 {
                return ms(ns) / nn;
            }
        }.c;
        const tg = struct {
            fn host(x: *const Cycle, tag: Tag) u64 {
                return x.tags[ix(tag)].host_ns;
            }
        }.host;
        try w.print("# Native decode cycle attribution ({s})\n\n", .{label});
        const rest: f64 = if (p.n_cycles > 1) ms(t.wall_ns - p.first_wall_ns) / @as(f64, @floatFromInt(p.n_cycles - 1)) else 0;
        try w.print("cycles {d} (stored {d}, dropped {d}), unbalanced {d}; ms/cycle {d:.3} (first cycle {d:.3} ms; the rest {d:.3} ms/cycle)\n", .{ p.n_cycles, p.n_stored, p.dropped, p.unbalanced, per(n, t.wall_ns), ms(p.first_wall_ns), rest });
        const stamps = @as(f64, @floatFromInt(t.stamps)) / n;
        try w.print("clock reads {d:.0} per cycle at {d:.1} ns each (calibrated at init): overhead ≈ {d:.3} ms/cycle\n\n", .{ stamps, p.clock_ns, stamps * p.clock_ns / 1e6 });
        const verify = t.root_ns[ix(.verify)];
        const vwait = t.wait_ns[ix(.verify)];
        try w.print("| term | {s} | source (tags) |\n|---|---:|---|\n", .{label});
        try w.print("| draft | {d:.3} | draft (inclusive; GPU-inclusive waits {d:.3}) |\n", .{ per(n, t.root_ns[ix(.draft)]), per(n, t.sync_ns[ix(.draft)]) });
        try w.print("| verify forward | {d:.3} | verify (inclusive: fwd_head, layer, moe, barrier, route, hit_wave, read_wait, drain, layer_tail, fwd_tail) |\n", .{per(n, verify)});
        try w.print("| … of which read wait | {d:.3} | read_wait under verify |\n", .{per(n, vwait)});
        try w.print("| … of which non-read verify | {d:.3} | verify − read wait |\n", .{per(n, verify - vwait)});
        try w.print("| accept | {d:.3} | accept |\n", .{per(n, t.root_ns[ix(.accept)])});
        try w.print("| commit | {d:.3} | commit |\n", .{per(n, t.root_ns[ix(.commit)])});
        try w.print("| flush (expert source settle, cycle end) | {d:.3} | flush |\n", .{per(n, t.root_ns[ix(.flush)])});
        var other: u64 = 0;
        for (ix(.fwd_head)..n_tags) |s| other += t.root_ns[s];
        if (other != 0) try w.print("| other phases opened at the cycle's root | {d:.3} | verify-level tags outside a loop phase |\n", .{per(n, other)});
        try w.print("| loop glue (in the cycle, no phase open) | {d:.3} | glue |\n", .{per(n, t.root_ns[glue])});
        try w.print("| **ms/cycle** | **{d:.3}** | cycle wall |\n\n", .{per(n, t.wall_ns)});
        const vsync = t.sync_ns[ix(.verify)];
        try w.print("VERIFY FORWARD {d:.3} ms/cycle = waits {d:.3} (evals + host reads {d:.3}, read wait {d:.3}) + host {d:.3}\n\n", .{ per(n, verify), per(n, vsync + vwait), per(n, vsync), per(n, vwait), per(n, verify -| (vsync + vwait)) });
        // the host-stamp split (sec. 7), exclusive ms per cycle
        const calls: f64 = @floatFromInt(@max(t.tags[ix(.layer)].calls, 1));
        const b = t.tags[ix(.barrier)];
        const rows = [_]struct { name: []const u8, ns: u64, wait: u64, src: []const u8, per_call: bool = true }{
            .{ .name = "barrier build/encode", .ns = tg(t, .layer) + (b.host_ns -| (b.eval_ns + b.read_ns)), .wait = 0, .src = "layer + barrier host" },
            .{ .name = "barrier eval (encode + GPU + wake)", .ns = b.eval_ns, .wait = b.eval_ns, .src = "barrier eval" },
            .{ .name = "route readback", .ns = b.read_ns, .wait = b.read_ns, .src = "barrier reads" },
            .{ .name = "runner (route, hit waves, glue)", .ns = tg(t, .route) + tg(t, .hit_wave) + tg(t, .moe), .wait = t.tags[ix(.route)].eval_ns + t.tags[ix(.hit_wave)].eval_ns + t.tags[ix(.moe)].eval_ns + t.tags[ix(.route)].read_ns + t.tags[ix(.hit_wave)].read_ns + t.tags[ix(.moe)].read_ns, .src = "route + hit_wave + moe" },
            .{ .name = "drain commit and wake", .ns = tg(t, .drain), .wait = t.tags[ix(.drain)].eval_ns + t.tags[ix(.drain)].read_ns, .src = "drain" },
            .{ .name = "[wait] read wait", .ns = tg(t, .read_wait), .wait = tg(t, .read_wait), .src = "read_wait" },
            .{ .name = "layer tail", .ns = tg(t, .layer_tail), .wait = t.tags[ix(.layer_tail)].eval_ns + t.tags[ix(.layer_tail)].read_ns, .src = "layer_tail" },
            .{ .name = "forward head", .ns = tg(t, .fwd_head), .wait = t.tags[ix(.fwd_head)].eval_ns + t.tags[ix(.fwd_head)].read_ns, .src = "fwd_head", .per_call = false },
            .{ .name = "forward tail", .ns = tg(t, .fwd_tail), .wait = t.tags[ix(.fwd_tail)].eval_ns + t.tags[ix(.fwd_tail)].read_ns, .src = "fwd_tail", .per_call = false },
            .{ .name = "between layers + the verify's own eval", .ns = tg(t, .verify), .wait = t.tags[ix(.verify)].eval_ns + t.tags[ix(.verify)].read_ns, .src = "verify (exclusive)", .per_call = false },
        };
        try w.print("| bucket (exclusive) | ms/cycle | per layer-call mean µs | of which waits ms/cycle | source (tags) |\n|---|---:|---:|---:|---|\n", .{});
        for (rows) |r| {
            try w.print("| {s} | {d:.3} | ", .{ r.name, per(n, r.ns) });
            if (r.per_call) try w.print("{d:.1}", .{ms(r.ns) * 1e3 / calls});
            try w.print(" | {d:.3} | {s} |\n", .{ per(n, r.wait), r.src });
        }
        try w.print("\nlayer-calls {d:.1} per cycle\n\n", .{@as(f64, @floatFromInt(t.tags[ix(.layer)].calls)) / n});
        // every tag
        try w.print("| tag | host ms | eval ms | read ms | async ms | calls | launches | regions | evals | reads | async evals | ops |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n", .{});
        for (0..n_slots) |s| {
            const c = t.tags[s];
            if (c.calls == 0 and c.host_ns == 0) continue;
            try w.print("| {s} | {d:.3} | {d:.3} | {d:.3} | {d:.3} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.2} |\n", .{ slotName(s), per(n, c.host_ns), per(n, c.eval_ns), per(n, c.read_ns), per(n, c.async_ns), cnt(c.calls, n), cnt(c.launches, n), cnt(c.regions, n), cnt(c.evals, n), cnt(c.reads, n), cnt(c.async_evals, n), cnt(c.ops, n) });
        }
        // per layer, mean µs per layer-call
        try w.print("\n| layer | calls | barrier build/encode | barrier eval | route readback | runner | drain commit and wake | read wait | layer tail |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n", .{});
        for (0..max_layers) |l| {
            const k = p.layer_calls[l];
            if (k == 0) continue;
            const L = &p.per_layer[l];
            const kf: f64 = @floatFromInt(k);
            const bl = L[ix(.barrier)];
            const build = L[ix(.layer)].host_ns + (bl.host_ns -| (bl.eval_ns + bl.read_ns));
            const runner = L[ix(.route)].host_ns + L[ix(.hit_wave)].host_ns + L[ix(.moe)].host_ns;
            try w.print("| {d} | {d} | {d:.1} | {d:.1} | {d:.1} | {d:.1} | {d:.1} | {d:.1} | {d:.1} |\n", .{ l, k, us(build, kf), us(bl.eval_ns, kf), us(bl.read_ns, kf), us(runner, kf), us(L[ix(.drain)].host_ns, kf), us(L[ix(.read_wait)].host_ns, kf), us(L[ix(.layer_tail)].host_ns, kf) });
        }
        var out: Counts = .{};
        for (p.outside) |c| out.add(c);
        try w.print("\noutside cycles: {d:.3} ms host (evals {d}, reads {d}, launches {d})\n", .{ ms(out.host_ns), out.evals, out.reads, out.launches });
    }

    /// One line for chain logs: the sec. 2a terms, ms per cycle.
    pub fn writeSummaryLine(p: *const Profiler, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const n: f64 = @floatFromInt(@max(p.n_cycles, 1));
        const t = &p.total;
        const v = t.root_ns[ix(.verify)];
        const rw = t.wait_ns[ix(.verify)];
        try w.print("NATIVE_CYCLE {{\"cycles\":{d},\"ms_cycle\":{d:.3},\"draft\":{d:.3},\"verify\":{d:.3},\"read_wait\":{d:.3},\"non_read_verify\":{d:.3},\"accept\":{d:.3},\"commit\":{d:.3},\"flush\":{d:.3},\"glue\":{d:.3},\"unbalanced\":{d}}}\n", .{ p.n_cycles, ms(t.wall_ns) / n, ms(t.root_ns[ix(.draft)]) / n, ms(v) / n, ms(rw) / n, ms(v - rw) / n, ms(t.root_ns[ix(.accept)]) / n, ms(t.root_ns[ix(.commit)]) / n, ms(t.root_ns[ix(.flush)]) / n, ms(t.root_ns[glue]) / n, p.unbalanced });
    }
};

fn cnt(x: u32, n: f64) f64 {
    return @as(f64, @floatFromInt(x)) / n;
}

fn us(ns: u64, k: f64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3 / k;
}

fn writeCycle(w: *std.Io.Writer, i: usize, c: *const Cycle) std.Io.Writer.Error!void {
    try w.print("{{\"cycle\":{d},\"wall_ms\":{d:.4},\"unbalanced\":{s},\"roots\":{{", .{ i, ms(c.wall_ns), if (c.unbalanced) "true" else "false" });
    var first = true;
    for (0..n_slots) |s| if (c.root_ns[s] != 0) {
        try w.print("{s}\"{s}\":{d:.4}", .{ if (first) "" else ",", slotName(s), ms(c.root_ns[s]) });
        first = false;
    };
    try w.print("}},\"verify_read_wait_ms\":{d:.4},\"verify_sync_ms\":{d:.4},\"tags\":{{", .{ ms(c.wait_ns[ix(.verify)]), ms(c.sync_ns[ix(.verify)]) });
    first = true;
    for (0..n_slots) |s| {
        const t = c.tags[s];
        if (t.calls == 0 and t.host_ns == 0) continue;
        try w.print("{s}\"{s}\":{{\"host_ms\":{d:.4},\"eval_ms\":{d:.4},\"read_ms\":{d:.4},\"async_ms\":{d:.4},\"launches\":{d},\"regions\":{d},\"evals\":{d},\"reads\":{d},\"async_evals\":{d},\"ops\":{d},\"calls\":{d}}}", .{ if (first) "" else ",", slotName(s), ms(t.host_ns), ms(t.eval_ns), ms(t.read_ns), ms(t.async_ns), t.launches, t.regions, t.evals, t.reads, t.async_evals, t.ops, t.calls });
        first = false;
    }
    try w.writeAll("}}\n");
}

// ── Selection (compile time) ──

/// `Root` may instantiate `Profiled`: a test build, or a root that declares
/// `pub const dsv41_profile_window = true;` (a profiling window binary). The serving root
/// (src/main.zig) does not, so a serving build that named `Profiled` would not compile.
pub fn profilingAllowed(comptime Root: type, comptime is_test: bool) bool {
    if (is_test) return true;
    if (!@hasDecl(Root, "dsv41_profile_window")) return false;
    return Root.dsv41_profile_window == true;
}

pub const window_build = profilingAllowed(@import("root"), builtin.is_test);

pub fn isProfiled(comptime G: type) bool {
    return @typeInfo(G) == .@"struct" and @hasDecl(G, "dsv41_profiled");
}

/// The backend under a `Profiled` wrapper (G itself otherwise): for the model's
/// backend-specific branches (`Base(G) == ops.MlxOps`).
pub fn Base(comptime G: type) type {
    return if (isProfiled(G)) G.Inner else G;
}

/// The base backend of `g` (`g` itself unless it is a `Profiled` backend).
pub inline fn base(g: anytype) *Base(std.meta.Child(@TypeOf(g))) {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) return &g.inner;
    return g;
}

/// For the serving arm: refuses (at compile time) a profiled backend.
pub fn assertServing(comptime G: type) void {
    if (comptime isProfiled(G)) @compileError("dsv41 profile: the serving build instantiated " ++ @typeName(G));
}

/// `G` declares `name` as a function (a wrapper declares an absent capability as `{}`).
pub fn isFn(comptime G: type, comptime name: []const u8) bool {
    return @hasDecl(G, name) and @typeInfo(@TypeOf(@field(G, name))) == .@"fn";
}

// ── The hooks the decode loop calls (nothing unless the backend is profiled) ──

pub inline fn cycleBegin(g: anytype) void {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) g.prof.cycleBegin();
}

pub inline fn cycleEnd(g: anytype) void {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) g.prof.cycleEnd();
}

pub inline fn beginPhase(g: anytype, comptime tag: Tag) void {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) g.prof.begin(tag);
}

pub inline fn endPhase(g: anytype) void {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) g.prof.end();
}

pub inline fn beginLayer(g: anytype, l: u32) void {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) g.prof.beginLayer(l);
}

pub inline fn endLayer(g: anytype) void {
    if (comptime isProfiled(std.meta.Child(@TypeOf(g)))) g.prof.endLayer();
}

// ── The wrapper ──

/// The profiling backend over `Ops`: every method of `Ops` with the same signature (taken
/// from `Ops` at compile time); launches, compiled regions, evals, host reads and async evals
/// counted and the syncs timed into the open phase; everything else forwarded. Owns `inner`.
/// Only a test build or a profiling window root may instantiate it (`profilingAllowed`).
pub fn Profiled(comptime Ops: type) type {
    if (!window_build) @compileError("dsv41 profile: Profiled(" ++ @typeName(Ops) ++ ") outside a profiling window build; the serving build must not link the profiler (a window root declares `pub const dsv41_profile_window = true;`)");
    return struct {
        const Self = @This();
        pub const T = Ops.T;
        pub const Inner = Ops;
        pub const dsv41_profiled = {};
        const prepared = isFn(Ops, "prepareLaunch");

        inner: Ops,
        prof: Profiler,

        /// `max_cycles`: the receipt's capacity (cycles past it are counted, not stored).
        pub fn init(a: Allocator, inner: Ops, clock: Clock, max_cycles: usize) !Self {
            return .{ .inner = inner, .prof = try Profiler.init(a, clock, max_cycles) };
        }

        /// Frees the profiler and the backend (its `deinit`, when `Ops` has a public one).
        pub fn deinit(g: *Self) void {
            g.prof.deinit();
            if (comptime isFn(Ops, "deinit")) g.inner.deinit();
        }

        pub fn base(g: *Self) *Ops {
            return &g.inner;
        }

        pub fn beginPhase(g: *Self, tag: Tag) void {
            g.prof.begin(tag);
        }

        pub fn endPhase(g: *Self) void {
            g.prof.end();
        }

        fn P(comptime name: []const u8, comptime i: usize) type {
            return @typeInfo(@TypeOf(@field(Ops, name))).@"fn".param_types[i].?;
        }

        fn R(comptime name: []const u8) type {
            return @typeInfo(@TypeOf(@field(Ops, name))).@"fn".return_type.?;
        }

        // counted and timed

        pub fn launch(g: *Self, k: P("launch", 1), inputs: P("launch", 2), cfg: P("launch", 3), out: P("launch", 4)) R("launch") {
            g.prof.launch();
            return g.inner.launch(k, inputs, cfg, out);
        }

        pub fn tape(g: *Self, comptime Body: type, ctx: *const Body.Ctx, inputs: []const T, out: []T) @TypeOf(g.inner.tape(Body, ctx, inputs, out)) {
            g.prof.region();
            return g.inner.tape(Body, ctx, inputs, out);
        }

        pub fn evalAll(g: *Self, xs: P("evalAll", 1)) R("evalAll") {
            const t0 = g.prof.now();
            defer g.prof.synced(.eval, t0);
            return g.inner.evalAll(xs);
        }

        pub fn asyncEval(g: *Self, xs: P("asyncEval", 1)) R("asyncEval") {
            const t0 = g.prof.now();
            defer g.prof.synced(.async_eval, t0);
            return g.inner.asyncEval(xs);
        }

        pub fn hostIds(g: *Self, x: P("hostIds", 1), out: P("hostIds", 2)) R("hostIds") {
            const t0 = g.prof.now();
            defer g.prof.synced(.read, t0);
            return g.inner.hostIds(x, out);
        }

        pub fn hostU32(g: *Self, x: P("hostU32", 1), out: P("hostU32", 2)) R("hostU32") {
            const t0 = g.prof.now();
            defer g.prof.synced(.read, t0);
            return g.inner.hostU32(x, out);
        }

        pub fn hostF32(g: *Self, x: P("hostF32", 1), out: P("hostF32", 2)) R("hostF32") {
            const t0 = g.prof.now();
            defer g.prof.synced(.read, t0);
            return g.inner.hostF32(x, out);
        }

        pub fn hostBool(g: *Self, x: P("hostBool", 1), out: P("hostBool", 2)) R("hostBool") {
            const t0 = g.prof.now();
            defer g.prof.synced(.read, t0);
            return g.inner.hostBool(x, out);
        }

        pub fn hostArgmax(g: *Self, x: P("hostArgmax", 1)) R("hostArgmax") {
            const t0 = g.prof.now();
            defer g.prof.synced(.read, t0);
            return g.inner.hostArgmax(x);
        }

        // prepared launches, when the backend prepares them (`{}` otherwise: RowPlans then
        // takes the per-call launch, as on the backend itself)

        pub const Prepared = if (prepared) Ops.Prepared else void;
        pub const prepareLaunch = if (prepared) prepareLaunchP else {};
        pub const launchPrepared = if (prepared) launchPreparedP else {};
        pub const releasePrepared = if (prepared) releasePreparedP else {};

        fn prepareLaunchP(g: *Self, k: P("prepareLaunch", 1), cfg: P("prepareLaunch", 2)) R("prepareLaunch") {
            return g.inner.prepareLaunch(k, cfg);
        }

        fn launchPreparedP(g: *Self, p: P("launchPrepared", 1), inputs: P("launchPrepared", 2), out: P("launchPrepared", 3)) R("launchPrepared") {
            g.prof.launch();
            return g.inner.launchPrepared(p, inputs, out);
        }

        fn releasePreparedP(g: *Self, p: P("releasePrepared", 1)) R("releasePrepared") {
            return g.inner.releasePrepared(p);
        }

        // forwarded: one line per method of deepseek_v41_ops.MlxOps / TraceOps at m1 44a910e
        // (a method added there fails the profiling build by name until it is added here);
        // graph-building methods count as ops
        pub fn reset(g: *Self) R("reset") {
            return g.inner.reset();
        }

        pub fn mark(g: *const Self) R("mark") {
            return g.inner.mark();
        }

        pub fn resetTo(g: *Self, a1: P("resetTo", 1)) R("resetTo") {
            return g.inner.resetTo(a1);
        }

        pub fn keep(g: *Self, a1: P("keep", 1)) R("keep") {
            return g.inner.keep(a1);
        }

        pub fn release(g: *Self, a1: P("release", 1)) R("release") {
            return g.inner.release(a1);
        }

        pub fn drop(g: *Self, a1: P("drop", 1)) R("drop") {
            return g.inner.drop(a1);
        }

        pub fn dropKept(g: *Self, a1: P("dropKept", 1)) R("dropKept") {
            return g.inner.dropKept(a1);
        }

        pub fn adopt(g: *Self, a1: P("adopt", 1)) R("adopt") {
            return g.inner.adopt(a1);
        }

        pub fn hostArray(g: *Self, a1: P("hostArray", 1), a2: P("hostArray", 2), a3: P("hostArray", 3)) R("hostArray") {
            g.prof.op();
            return g.inner.hostArray(a1, a2, a3);
        }

        pub fn dtypeOf(g: *Self, a1: P("dtypeOf", 1)) R("dtypeOf") {
            return g.inner.dtypeOf(a1);
        }

        pub fn shapeOf(g: *Self, a1: P("shapeOf", 1)) R("shapeOf") {
            return g.inner.shapeOf(a1);
        }

        pub fn scalar(g: *Self, a1: P("scalar", 1), a2: P("scalar", 2)) R("scalar") {
            g.prof.op();
            return g.inner.scalar(a1, a2);
        }

        pub fn arange(g: *Self, a1: P("arange", 1), a2: P("arange", 2), a3: P("arange", 3), a4: P("arange", 4)) R("arange") {
            g.prof.op();
            return g.inner.arange(a1, a2, a3, a4);
        }

        pub fn ones(g: *Self, a1: P("ones", 1), a2: P("ones", 2)) R("ones") {
            g.prof.op();
            return g.inner.ones(a1, a2);
        }

        pub fn zeros(g: *Self, a1: P("zeros", 1), a2: P("zeros", 2)) R("zeros") {
            g.prof.op();
            return g.inner.zeros(a1, a2);
        }

        pub fn full(g: *Self, a1: P("full", 1), a2: P("full", 2), a3: P("full", 3)) R("full") {
            g.prof.op();
            return g.inner.full(a1, a2, a3);
        }

        pub fn astype(g: *Self, a1: P("astype", 1), a2: P("astype", 2)) R("astype") {
            g.prof.op();
            return g.inner.astype(a1, a2);
        }

        pub fn add(g: *Self, a1: P("add", 1), a2: P("add", 2)) R("add") {
            g.prof.op();
            return g.inner.add(a1, a2);
        }

        pub fn sub(g: *Self, a1: P("sub", 1), a2: P("sub", 2)) R("sub") {
            g.prof.op();
            return g.inner.sub(a1, a2);
        }

        pub fn mul(g: *Self, a1: P("mul", 1), a2: P("mul", 2)) R("mul") {
            g.prof.op();
            return g.inner.mul(a1, a2);
        }

        pub fn div(g: *Self, a1: P("div", 1), a2: P("div", 2)) R("div") {
            g.prof.op();
            return g.inner.div(a1, a2);
        }

        pub fn floorDiv(g: *Self, a1: P("floorDiv", 1), a2: P("floorDiv", 2)) R("floorDiv") {
            g.prof.op();
            return g.inner.floorDiv(a1, a2);
        }

        pub fn maximum(g: *Self, a1: P("maximum", 1), a2: P("maximum", 2)) R("maximum") {
            g.prof.op();
            return g.inner.maximum(a1, a2);
        }

        pub fn minimum(g: *Self, a1: P("minimum", 1), a2: P("minimum", 2)) R("minimum") {
            g.prof.op();
            return g.inner.minimum(a1, a2);
        }

        pub fn power(g: *Self, a1: P("power", 1), a2: P("power", 2)) R("power") {
            g.prof.op();
            return g.inner.power(a1, a2);
        }

        pub fn logaddexp(g: *Self, a1: P("logaddexp", 1), a2: P("logaddexp", 2)) R("logaddexp") {
            g.prof.op();
            return g.inner.logaddexp(a1, a2);
        }

        pub fn less(g: *Self, a1: P("less", 1), a2: P("less", 2)) R("less") {
            g.prof.op();
            return g.inner.less(a1, a2);
        }

        pub fn lessEqual(g: *Self, a1: P("lessEqual", 1), a2: P("lessEqual", 2)) R("lessEqual") {
            g.prof.op();
            return g.inner.lessEqual(a1, a2);
        }

        pub fn greater(g: *Self, a1: P("greater", 1), a2: P("greater", 2)) R("greater") {
            g.prof.op();
            return g.inner.greater(a1, a2);
        }

        pub fn equal(g: *Self, a1: P("equal", 1), a2: P("equal", 2)) R("equal") {
            g.prof.op();
            return g.inner.equal(a1, a2);
        }

        pub fn greaterEqual(g: *Self, a1: P("greaterEqual", 1), a2: P("greaterEqual", 2)) R("greaterEqual") {
            g.prof.op();
            return g.inner.greaterEqual(a1, a2);
        }

        pub fn logicalAnd(g: *Self, a1: P("logicalAnd", 1), a2: P("logicalAnd", 2)) R("logicalAnd") {
            g.prof.op();
            return g.inner.logicalAnd(a1, a2);
        }

        pub fn logicalOr(g: *Self, a1: P("logicalOr", 1), a2: P("logicalOr", 2)) R("logicalOr") {
            g.prof.op();
            return g.inner.logicalOr(a1, a2);
        }

        pub fn matmul(g: *Self, a1: P("matmul", 1), a2: P("matmul", 2)) R("matmul") {
            g.prof.op();
            return g.inner.matmul(a1, a2);
        }

        pub fn neg(g: *Self, a1: P("neg", 1)) R("neg") {
            g.prof.op();
            return g.inner.neg(a1);
        }

        pub fn square(g: *Self, a1: P("square", 1)) R("square") {
            g.prof.op();
            return g.inner.square(a1);
        }

        pub fn sqrt(g: *Self, a1: P("sqrt", 1)) R("sqrt") {
            g.prof.op();
            return g.inner.sqrt(a1);
        }

        pub fn rsqrt(g: *Self, a1: P("rsqrt", 1)) R("rsqrt") {
            g.prof.op();
            return g.inner.rsqrt(a1);
        }

        pub fn exp(g: *Self, a1: P("exp", 1)) R("exp") {
            g.prof.op();
            return g.inner.exp(a1);
        }

        pub fn sigmoid(g: *Self, a1: P("sigmoid", 1)) R("sigmoid") {
            g.prof.op();
            return g.inner.sigmoid(a1);
        }

        pub fn cos(g: *Self, a1: P("cos", 1)) R("cos") {
            g.prof.op();
            return g.inner.cos(a1);
        }

        pub fn sin(g: *Self, a1: P("sin", 1)) R("sin") {
            g.prof.op();
            return g.inner.sin(a1);
        }

        pub fn abs(g: *Self, a1: P("abs", 1)) R("abs") {
            g.prof.op();
            return g.inner.abs(a1);
        }

        pub fn transpose(g: *Self, a1: P("transpose", 1)) R("transpose") {
            g.prof.op();
            return g.inner.transpose(a1);
        }

        pub fn where(g: *Self, a1: P("where", 1), a2: P("where", 2), a3: P("where", 3)) R("where") {
            g.prof.op();
            return g.inner.where(a1, a2, a3);
        }

        pub fn clip(g: *Self, a1: P("clip", 1), a2: P("clip", 2), a3: P("clip", 3)) R("clip") {
            g.prof.op();
            return g.inner.clip(a1, a2, a3);
        }

        pub fn einsum(g: *Self, a1: P("einsum", 1), a2: P("einsum", 2)) R("einsum") {
            g.prof.op();
            return g.inner.einsum(a1, a2);
        }

        pub fn qmm(g: *Self, a1: P("qmm", 1), a2: P("qmm", 2), a3: P("qmm", 3), a4: P("qmm", 4)) R("qmm") {
            g.prof.op();
            return g.inner.qmm(a1, a2, a3, a4);
        }

        pub fn dequantize(g: *Self, a1: P("dequantize", 1), a2: P("dequantize", 2), a3: P("dequantize", 3)) R("dequantize") {
            g.prof.op();
            return g.inner.dequantize(a1, a2, a3);
        }

        pub fn quantize(g: *Self, a1: P("quantize", 1), a2: P("quantize", 2)) R("quantize") {
            g.prof.op();
            return g.inner.quantize(a1, a2);
        }

        pub fn reshape(g: *Self, a1: P("reshape", 1), a2: P("reshape", 2)) R("reshape") {
            g.prof.op();
            return g.inner.reshape(a1, a2);
        }

        pub fn transposeAxes(g: *Self, a1: P("transposeAxes", 1), a2: P("transposeAxes", 2)) R("transposeAxes") {
            g.prof.op();
            return g.inner.transposeAxes(a1, a2);
        }

        pub fn broadcastTo(g: *Self, a1: P("broadcastTo", 1), a2: P("broadcastTo", 2)) R("broadcastTo") {
            g.prof.op();
            return g.inner.broadcastTo(a1, a2);
        }

        pub fn expandDims(g: *Self, a1: P("expandDims", 1), a2: P("expandDims", 2)) R("expandDims") {
            g.prof.op();
            return g.inner.expandDims(a1, a2);
        }

        pub fn slice(g: *Self, a1: P("slice", 1), a2: P("slice", 2), a3: P("slice", 3), a4: P("slice", 4)) R("slice") {
            g.prof.op();
            return g.inner.slice(a1, a2, a3, a4);
        }

        pub fn sliceUpdateDyn(g: *Self, a1: P("sliceUpdateDyn", 1), a2: P("sliceUpdateDyn", 2), a3: P("sliceUpdateDyn", 3)) R("sliceUpdateDyn") {
            g.prof.op();
            return g.inner.sliceUpdateDyn(a1, a2, a3);
        }

        pub fn concat(g: *Self, a1: P("concat", 1), a2: P("concat", 2)) R("concat") {
            g.prof.op();
            return g.inner.concat(a1, a2);
        }

        pub fn stack(g: *Self, a1: P("stack", 1), a2: P("stack", 2)) R("stack") {
            g.prof.op();
            return g.inner.stack(a1, a2);
        }

        pub fn take(g: *Self, a1: P("take", 1), a2: P("take", 2), a3: P("take", 3)) R("take") {
            g.prof.op();
            return g.inner.take(a1, a2, a3);
        }

        pub fn takeAlongAxis(g: *Self, a1: P("takeAlongAxis", 1), a2: P("takeAlongAxis", 2), a3: P("takeAlongAxis", 3)) R("takeAlongAxis") {
            g.prof.op();
            return g.inner.takeAlongAxis(a1, a2, a3);
        }

        pub fn sum(g: *Self, a1: P("sum", 1), a2: P("sum", 2), a3: P("sum", 3)) R("sum") {
            g.prof.op();
            return g.inner.sum(a1, a2, a3);
        }

        pub fn mean(g: *Self, a1: P("mean", 1), a2: P("mean", 2), a3: P("mean", 3)) R("mean") {
            g.prof.op();
            return g.inner.mean(a1, a2, a3);
        }

        pub fn max(g: *Self, a1: P("max", 1), a2: P("max", 2), a3: P("max", 3)) R("max") {
            g.prof.op();
            return g.inner.max(a1, a2, a3);
        }

        pub fn softmax(g: *Self, a1: P("softmax", 1), a2: P("softmax", 2)) R("softmax") {
            g.prof.op();
            return g.inner.softmax(a1, a2);
        }

        pub fn sort(g: *Self, a1: P("sort", 1), a2: P("sort", 2)) R("sort") {
            g.prof.op();
            return g.inner.sort(a1, a2);
        }

        pub fn argsort(g: *Self, a1: P("argsort", 1), a2: P("argsort", 2)) R("argsort") {
            g.prof.op();
            return g.inner.argsort(a1, a2);
        }

        pub fn argpartition(g: *Self, a1: P("argpartition", 1), a2: P("argpartition", 2), a3: P("argpartition", 3)) R("argpartition") {
            g.prof.op();
            return g.inner.argpartition(a1, a2, a3);
        }

        pub fn cumsum(g: *Self, a1: P("cumsum", 1), a2: P("cumsum", 2)) R("cumsum") {
            g.prof.op();
            return g.inner.cumsum(a1, a2);
        }

        pub fn repeat(g: *Self, a1: P("repeat", 1), a2: P("repeat", 2), a3: P("repeat", 3)) R("repeat") {
            g.prof.op();
            return g.inner.repeat(a1, a2, a3);
        }

        pub fn silu(g: *Self, a1: P("silu", 1)) R("silu") {
            g.prof.op();
            return g.inner.silu(a1);
        }

        pub fn softplus(g: *Self, a1: P("softplus", 1)) R("softplus") {
            g.prof.op();
            return g.inner.softplus(a1);
        }

        pub fn hadamard(g: *Self, a1: P("hadamard", 1), a2: P("hadamard", 2)) R("hadamard") {
            g.prof.op();
            return g.inner.hadamard(a1, a2);
        }

        pub fn gatherQmm(g: *Self, a1: P("gatherQmm", 1), a2: P("gatherQmm", 2), a3: P("gatherQmm", 3), a4: P("gatherQmm", 4), a5: P("gatherQmm", 5)) R("gatherQmm") {
            g.prof.op();
            return g.inner.gatherQmm(a1, a2, a3, a4, a5);
        }

        pub fn argmax(g: *Self, a1: P("argmax", 1), a2: P("argmax", 2)) R("argmax") {
            g.prof.op();
            return g.inner.argmax(a1, a2);
        }

        pub fn logsumexp(g: *Self, a1: P("logsumexp", 1), a2: P("logsumexp", 2), a3: P("logsumexp", 3)) R("logsumexp") {
            g.prof.op();
            return g.inner.logsumexp(a1, a2, a3);
        }

        pub fn input(g: *Self, a1: P("input", 1), a2: P("input", 2)) R("input") {
            g.prof.op();
            return g.inner.input(a1, a2);
        }

        pub fn node(g: *const Self, a1: P("node", 1)) R("node") {
            return g.inner.node(a1);
        }

        pub fn opsSince(g: *const Self, a1: P("opsSince", 1), a2: P("opsSince", 2)) R("opsSince") {
            return g.inner.opsSince(a1, a2);
        }

        pub fn kernel(g: *Self, a1: P("kernel", 1), a2: P("kernel", 2)) R("kernel") {
            g.prof.op();
            return g.inner.kernel(a1, a2);
        }

        pub fn eventAlias(g: *Self, a1: P("eventAlias", 1), a2: P("eventAlias", 2), a3: P("eventAlias", 3)) R("eventAlias") {
            g.prof.op();
            return g.inner.eventAlias(a1, a2, a3);
        }
    };
}

// ── Tests ──

const testing = std.testing;
const ns_ms: u64 = 1_000_000;

/// A host backend for the profiler tests: every call logged; its evals, host reads and async
/// evals cost `eval_ns` / `read_ns` / `async_ns` on the manual clock (or sleep them on `io`).
const TestOps = struct {
    pub const T = u32;
    pub const Prepared = struct { k: u16, cfg: u32 };
    pub const Ev = union(enum) { launch: u16, prepared: u16, region: [*]const u8, eval: usize, async_eval: usize, read: u32, op: u32 };

    a: Allocator,
    clock: *u64,
    io: ?std.Io = null,
    eval_ns: u64 = 0,
    read_ns: u64 = 0,
    async_ns: u64 = 0,
    log: std.ArrayList(Ev) = .empty,
    live: isize = 0,
    next: u32 = 1,

    pub fn deinit(t: *TestOps) void {
        t.log.deinit(t.a);
    }

    fn spend(t: *TestOps, ns: u64) void {
        spendOn(t.clock, t.io, ns);
    }

    fn outs(t: *TestOps, out: []T) void {
        for (out) |*o| {
            o.* = t.next;
            t.next += 1;
        }
    }

    pub fn launch(t: *TestOps, k: u16, inputs: []const T, cfg: u32, out: []T) !void {
        _ = inputs;
        _ = cfg;
        t.outs(out);
        try t.log.append(t.a, .{ .launch = k });
    }

    pub fn prepareLaunch(t: *TestOps, k: u16, cfg: u32) !Prepared {
        t.live += 1;
        return .{ .k = k, .cfg = cfg };
    }

    pub fn launchPrepared(t: *TestOps, p: *const Prepared, inputs: []const T, out: []T) !void {
        _ = inputs;
        t.outs(out);
        try t.log.append(t.a, .{ .prepared = p.k });
    }

    pub fn releasePrepared(t: *TestOps, _: *Prepared) void {
        t.live -= 1;
    }

    pub fn tape(t: *TestOps, comptime Body: type, ctx: *const Body.Ctx, inputs: []const T, out: []T) !void {
        _ = ctx;
        _ = inputs;
        t.outs(out);
        try t.log.append(t.a, .{ .region = @typeName(Body).ptr });
    }

    pub fn evalAll(t: *TestOps, xs: []const T) !void {
        t.spend(t.eval_ns);
        try t.log.append(t.a, .{ .eval = xs.len });
    }

    pub fn asyncEval(t: *TestOps, xs: []const T) !void {
        t.spend(t.async_ns);
        try t.log.append(t.a, .{ .async_eval = xs.len });
    }

    pub fn hostU32(t: *TestOps, x: T, out: []u32) ![]const u32 {
        t.spend(t.read_ns);
        @memset(out, x);
        try t.log.append(t.a, .{ .read = x });
        return out;
    }

    pub fn add(t: *TestOps, x: T, y: T) !T {
        _ = y;
        try t.log.append(t.a, .{ .op = x });
        t.next += 1;
        return t.next - 1;
    }
};

/// Host work between calls: the manual clock advances, or the thread sleeps on `io`.
fn spendOn(clock: *u64, io: ?std.Io, ns: u64) void {
    if (ns == 0) return;
    if (io) |i| {
        std.Io.sleep(i, .fromNanoseconds(ns), .awake) catch {};
    } else clock.* += ns;
}

const Seg = struct {
    pub const Ctx = struct { n: u32 };
};

/// Host time the synthetic cycle spends between its calls (ns).
const Work = struct { draft: u64, glue_a: u64, fwd_head: u64, layer: u64, route: u64, read_wait: u64, moe: u64, fwd_tail: u64, glue_b: u64 };

const work_ms: Work = .{ .draft = 2 * ns_ms, .glue_a = ns_ms / 2, .fwd_head = ns_ms / 10, .layer = ns_ms, .route = ns_ms / 5, .read_wait = 3 * ns_ms, .moe = ns_ms / 20, .fwd_tail = ns_ms / 10, .glue_b = 3 * ns_ms / 10 };

/// One cycle as the native loop runs it (draft, verify over `layers` layer-calls with the
/// routed MoE's barrier / route / read wait / drain, accept, commit), with the hooks at the
/// model lane's hook points; on a plain backend the hooks are nothing.
fn syntheticCycle(g: anytype, clock: *u64, io: ?std.Io, w: Work, prep: *const TestOps.Prepared, layers: u32) !void {
    const seg: Seg.Ctx = .{ .n = 1 };
    var out: [2]u32 = undefined;
    var buf: [4]u32 = undefined;
    cycleBegin(g);
    beginPhase(g, .draft);
    spendOn(clock, io, w.draft);
    for (0..3) |i| try g.launch(@intCast(i), &.{}, 0, out[0..1]);
    try g.tape(Seg, &seg, &.{}, out[0..1]);
    try g.evalAll(&.{ 1, 2 });
    _ = try g.hostU32(7, buf[0..2]);
    endPhase(g);
    spendOn(clock, io, w.glue_a);
    beginPhase(g, .verify);
    beginPhase(g, .fwd_head);
    spendOn(clock, io, w.fwd_head);
    _ = try g.add(1, 2);
    endPhase(g);
    for (0..layers) |l| {
        beginLayer(g, @intCast(l));
        spendOn(clock, io, w.layer);
        try g.tape(Seg, &seg, &.{}, out[0..1]);
        try g.tape(Seg, &seg, &.{}, out[0..1]);
        beginPhase(g, .moe);
        beginPhase(g, .barrier);
        try g.evalAll(&.{3});
        _ = try g.hostU32(9, buf[0..1]);
        endPhase(g);
        beginPhase(g, .route);
        spendOn(clock, io, w.route);
        endPhase(g);
        beginPhase(g, .read_wait);
        spendOn(clock, io, w.read_wait);
        endPhase(g);
        beginPhase(g, .drain);
        try g.launchPrepared(prep, &.{}, out[0..1]);
        try g.launchPrepared(prep, &.{}, out[0..1]);
        try g.asyncEval(&.{4});
        endPhase(g);
        spendOn(clock, io, w.moe);
        endPhase(g);
        beginPhase(g, .layer_tail);
        try g.tape(Seg, &seg, &.{}, out[0..1]);
        endPhase(g);
        endLayer(g);
    }
    beginPhase(g, .fwd_tail);
    spendOn(clock, io, w.fwd_tail);
    endPhase(g);
    try g.evalAll(&.{5});
    endPhase(g);
    beginPhase(g, .accept);
    _ = try g.hostU32(1, buf[0..1]);
    endPhase(g);
    beginPhase(g, .commit);
    try g.evalAll(&.{6});
    endPhase(g);
    spendOn(clock, io, w.glue_b);
    cycleEnd(g);
}

fn tagOf(c: *const Cycle, t: Tag) Counts {
    return c.tags[ix(t)];
}

test "dsv41 kernels profile: a synthetic cycle's time, launches, regions and syncs land in their tags (manual clock, exact)" {
    const a = testing.allocator;
    var clock: u64 = 1000;
    const P = Profiled(TestOps);
    var g = try P.init(a, .{ .a = a, .clock = &clock, .eval_ns = 3 * ns_ms / 2, .read_ns = ns_ms / 4, .async_ns = ns_ms / 10 }, .{ .manual = &clock }, 4);
    defer g.deinit();
    var prep = try g.prepareLaunch(7, 11);
    try syntheticCycle(&g, &clock, null, work_ms, &prep, 2);
    try syntheticCycle(&g, &clock, null, work_ms, &prep, 2);
    g.releasePrepared(&prep);
    try testing.expectEqual(@as(isize, 0), g.inner.live);
    const p = &g.prof;
    try testing.expectEqual(@as(u64, 2), p.n_cycles);
    try testing.expectEqual(@as(usize, 2), p.n_stored);
    try testing.expectEqual(@as(u32, 0), p.unbalanced);
    const c = &p.stored[1];
    // the loop's terms (inclusive per root) and the wall: 3.75 + 0.5 + 13.9 + 0.25 + 1.5 + 0.3
    try testing.expectEqual(@as(u64, 20_200_000), c.wall_ns);
    try testing.expectEqual(@as(u64, 3_750_000), c.root_ns[ix(.draft)]);
    try testing.expectEqual(@as(u64, 13_900_000), c.root_ns[ix(.verify)]);
    try testing.expectEqual(@as(u64, 250_000), c.root_ns[ix(.accept)]);
    try testing.expectEqual(@as(u64, 1_500_000), c.root_ns[ix(.commit)]);
    try testing.expectEqual(@as(u64, 800_000), c.root_ns[glue]);
    var sum: u64 = 0;
    for (c.root_ns) |x| sum += x;
    try testing.expectEqual(c.wall_ns, sum);
    var excl: u64 = 0;
    for (c.tags) |x| excl += x.host_ns;
    try testing.expectEqual(c.wall_ns, excl);
    // the verify's read wait and its GPU waits (2 barriers + its own eval)
    try testing.expectEqual(@as(u64, 6_000_000), c.wait_ns[ix(.verify)]);
    try testing.expectEqual(@as(u64, 5_000_000), c.sync_ns[ix(.verify)]);
    // per tag: exclusive time, the syncs inside it, counts
    const d = tagOf(c, .draft);
    try testing.expectEqual(@as(u64, 3_750_000), d.host_ns);
    try testing.expectEqual(@as(u64, 1_500_000), d.eval_ns);
    try testing.expectEqual(@as(u64, 250_000), d.read_ns);
    try testing.expectEqual([_]u32{ 1, 3, 1, 1, 1, 0, 0 }, [_]u32{ d.calls, d.launches, d.regions, d.evals, d.reads, d.async_evals, d.ops });
    const v = tagOf(c, .verify);
    try testing.expectEqual([_]u64{ 1_500_000, 1_500_000 }, [_]u64{ v.host_ns, v.eval_ns });
    try testing.expectEqual(@as(u64, 100_000), tagOf(c, .fwd_head).host_ns);
    try testing.expectEqual(@as(u32, 1), tagOf(c, .fwd_head).ops);
    const l = tagOf(c, .layer);
    try testing.expectEqual([_]u64{ 2_000_000, 0 }, [_]u64{ l.host_ns, l.eval_ns });
    try testing.expectEqual([_]u32{ 2, 4, 0 }, [_]u32{ l.calls, l.regions, l.launches });
    const b = tagOf(c, .barrier);
    try testing.expectEqual([_]u64{ 3_500_000, 3_000_000, 500_000 }, [_]u64{ b.host_ns, b.eval_ns, b.read_ns });
    try testing.expectEqual([_]u32{ 2, 2, 2 }, [_]u32{ b.calls, b.evals, b.reads });
    try testing.expectEqual(@as(u64, 400_000), tagOf(c, .route).host_ns);
    try testing.expectEqual(@as(u64, 6_000_000), tagOf(c, .read_wait).host_ns);
    const dr = tagOf(c, .drain);
    try testing.expectEqual([_]u64{ 200_000, 200_000 }, [_]u64{ dr.host_ns, dr.async_ns });
    try testing.expectEqual([_]u32{ 2, 4, 2 }, [_]u32{ dr.calls, dr.launches, dr.async_evals });
    try testing.expectEqual(@as(u64, 100_000), tagOf(c, .moe).host_ns);
    try testing.expectEqual([_]u32{ 2, 2 }, [_]u32{ tagOf(c, .layer_tail).calls, tagOf(c, .layer_tail).regions });
    try testing.expectEqual(@as(u64, 100_000), tagOf(c, .fwd_tail).host_ns);
    try testing.expectEqual([_]u64{ 250_000, 250_000 }, [_]u64{ tagOf(c, .accept).host_ns, tagOf(c, .accept).read_ns });
    try testing.expectEqual([_]u64{ 1_500_000, 1_500_000 }, [_]u64{ tagOf(c, .commit).host_ns, tagOf(c, .commit).eval_ns });
    try testing.expectEqual(@as(u32, 0), c.tags[glue].calls);
    // per layer (both cycles): one call per cycle each, the read wait and the barrier eval per layer-call
    for (0..2) |i| {
        try testing.expectEqual(@as(u32, 2), p.layer_calls[i]);
        try testing.expectEqual(@as(u64, 6_000_000), p.per_layer[i][ix(.read_wait)].host_ns);
        try testing.expectEqual(@as(u64, 3_000_000), p.per_layer[i][ix(.barrier)].eval_ns);
        try testing.expectEqual(@as(u64, 2_000_000), p.per_layer[i][ix(.layer)].host_ns);
    }
    try testing.expectEqual(@as(u32, 0), p.layer_calls[2]);
    // run totals = the sum of the cycles
    try testing.expectEqual(@as(u64, 40_400_000), p.total.wall_ns);
    try testing.expectEqual(@as(u32, 8), p.total.tags[ix(.drain)].launches);
}

test "dsv41 kernels profile: the wrapper adds no launch, region, eval or read to the backend (and the hooks are nothing on a plain one)" {
    const a = testing.allocator;
    var c1: u64 = 0;
    var c2: u64 = 0;
    var plain: TestOps = .{ .a = a, .clock = &c1, .eval_ns = 1000, .read_ns = 100, .async_ns = 10 };
    defer plain.deinit();
    var g = try Profiled(TestOps).init(a, .{ .a = a, .clock = &c2, .eval_ns = 1000, .read_ns = 100, .async_ns = 10 }, .{ .manual = &c2 }, 1);
    defer g.deinit();
    var p1 = try plain.prepareLaunch(3, 5);
    var p2 = try g.prepareLaunch(3, 5);
    try syntheticCycle(&plain, &c1, null, work_ms, &p1, 3);
    try syntheticCycle(&g, &c2, null, work_ms, &p2, 3);
    plain.releasePrepared(&p1);
    g.releasePrepared(&p2);
    // the same calls in the same order, and the same simulated time: the hooks read no clock
    try testing.expectEqual(plain.log.items.len, g.inner.log.items.len);
    for (plain.log.items, g.inner.log.items) |x, y| try testing.expect(std.meta.eql(x, y));
    try testing.expectEqual(c1, c2);
    var launches: u32 = 0;
    for (plain.log.items) |e| launches += switch (e) {
        .launch, .prepared => 1,
        else => 0,
    };
    var counted: u32 = 0;
    for (g.prof.stored[0].tags) |t| counted += t.launches;
    try testing.expectEqual(@as(u32, 3 + 2 * 3), launches);
    try testing.expectEqual(launches, counted);
}

test "dsv41 kernels profile: the host clock attributes real sleeps to their tags, and the tags add up to the wall" {
    const a = testing.allocator;
    const io = testing.io;
    var unused: u64 = 0;
    const us_: u64 = 1000;
    const w: Work = .{ .draft = 1500 * us_, .glue_a = 200 * us_, .fwd_head = 0, .layer = 400 * us_, .route = 0, .read_wait = 2000 * us_, .moe = 0, .fwd_tail = 0, .glue_b = 300 * us_ };
    var g = try Profiled(TestOps).init(a, .{ .a = a, .clock = &unused, .io = io, .eval_ns = 1000 * us_, .read_ns = 0, .async_ns = 0 }, .{ .awake = io }, 1);
    defer g.deinit();
    var prep = try g.prepareLaunch(0, 0);
    try syntheticCycle(&g, &unused, io, w, &prep, 1);
    g.releasePrepared(&prep);
    const c = &g.prof.stored[0];
    var sum: u64 = 0;
    for (c.root_ns) |x| sum += x;
    try testing.expectEqual(c.wall_ns, sum);
    // every sleep lands in its own tag (at least its length; a sleep may overrun, never shrink)
    try testing.expect(tagOf(c, .draft).host_ns >= w.draft + 1000 * us_);
    try testing.expect(tagOf(c, .draft).eval_ns >= 1000 * us_);
    try testing.expect(tagOf(c, .layer).host_ns >= w.layer);
    try testing.expect(tagOf(c, .read_wait).host_ns >= w.read_wait);
    try testing.expect(tagOf(c, .barrier).eval_ns >= 1000 * us_);
    try testing.expect(c.tags[glue].host_ns >= w.glue_a + w.glue_b);
    try testing.expect(c.wall_ns >= 7500 * us_ and c.wall_ns < 2000 * ns_ms);
    try testing.expectEqual(c.wait_ns[ix(.verify)], tagOf(c, .read_wait).host_ns);
}

test "dsv41 kernels profile: the receipt is one JSON object per cycle and a summary in the residue note's columns" {
    const a = testing.allocator;
    var clock: u64 = 0;
    var g = try Profiled(TestOps).init(a, .{ .a = a, .clock = &clock, .eval_ns = 3 * ns_ms / 2, .read_ns = ns_ms / 4, .async_ns = ns_ms / 10 }, .{ .manual = &clock }, 1);
    defer g.deinit();
    var prep = try g.prepareLaunch(1, 1);
    try syntheticCycle(&g, &clock, null, work_ms, &prep, 2);
    try syntheticCycle(&g, &clock, null, work_ms, &prep, 2);
    g.releasePrepared(&prep);
    // capacity 1: the second cycle counted, not stored
    try testing.expectEqual(@as(u64, 1), g.prof.dropped);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try g.prof.writeCycles(&aw.writer);
    const text = aw.written();
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "\n"));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, std.mem.trimEnd(u8, text, "\n"), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqual(@as(i64, 0), o.get("cycle").?.integer);
    try testing.expectApproxEqAbs(@as(f64, 20.2), o.get("wall_ms").?.float, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 13.9), o.get("roots").?.object.get("verify").?.float, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.0), o.get("verify_read_wait_ms").?.float, 1e-9);
    const tags = o.get("tags").?.object;
    const drain = tags.get("drain").?.object;
    try testing.expectEqual(@as(i64, 4), drain.get("launches").?.integer);
    try testing.expectApproxEqAbs(@as(f64, 0.2), drain.get("async_ms").?.float, 1e-9);
    try testing.expectEqual(@as(i64, 4), tags.get("layer").?.object.get("regions").?.integer);
    try testing.expectEqual(@as(i64, 2), tags.get("barrier").?.object.get("evals").?.integer);
    try testing.expectApproxEqAbs(@as(f64, 0.8), tags.get("glue").?.object.get("host_ms").?.float, 1e-9);
    try testing.expect(tags.get("flush") == null);
    // the summary: sec. 2a's terms, the split's buckets, the per-layer table, one chain line
    var sw: std.Io.Writer.Allocating = .init(a);
    defer sw.deinit();
    try g.prof.writeSummary(&sw.writer, "native");
    const s = sw.written();
    for ([_][]const u8{
        "| term | native | source (tags) |",
        "| draft | 3.750 |",
        "| verify forward | 13.900 |",
        "| … of which read wait | 6.000 |",
        "| … of which non-read verify | 7.900 |",
        "| accept | 0.250 |",
        "| commit | 1.500 |",
        "| loop glue (in the cycle, no phase open) | 0.800 |",
        "| **ms/cycle** | **20.200** |",
        "VERIFY FORWARD 13.900 ms/cycle = waits 11.000 (evals + host reads 5.000, read wait 6.000) + host 2.900",
        "| barrier build/encode | 2.000 | 1000.0 |",
        "| barrier eval (encode + GPU + wake) | 3.000 | 1500.0 | 3.000 |",
        "| route readback | 0.500 | 250.0 | 0.500 |",
        "| [wait] read wait | 6.000 | 3000.0 | 6.000 |",
        "| forward head | 0.100 |  | 0.000 | fwd_head |",
        "| 0 | 2 | 1000.0 | 1500.0 | 250.0 |",
    }) |want| {
        if (std.mem.indexOf(u8, s, want) == null) {
            std.debug.print("summary lacks: {s}\n{s}\n", .{ want, s });
            return error.TestUnexpectedResult;
        }
    }
    var lw: std.Io.Writer.Allocating = .init(a);
    defer lw.deinit();
    try g.prof.writeSummaryLine(&lw.writer);
    try testing.expectEqualStrings("NATIVE_CYCLE {\"cycles\":2,\"ms_cycle\":20.200,\"draft\":3.750,\"verify\":13.900,\"read_wait\":6.000,\"non_read_verify\":7.900,\"accept\":0.250,\"commit\":1.500,\"flush\":0.000,\"glue\":0.800,\"unbalanced\":0}\n", lw.written());
}

test "dsv41 kernels profile: the serving build cannot link it, and unbalanced phases are counted, not attributed" {
    // Profiled exists in a test build or under a root that opts in; a serving root does not
    try testing.expect(!profilingAllowed(struct {}, false));
    try testing.expect(!profilingAllowed(struct {
        pub const dsv41_profile_window = false;
    }, false));
    try testing.expect(profilingAllowed(struct {
        pub const dsv41_profile_window = true;
    }, false));
    try testing.expect(profilingAllowed(struct {}, true));
    comptime {
        assertServing(TestOps);
        std.debug.assert(!isProfiled(TestOps));
        std.debug.assert(isProfiled(Profiled(TestOps)));
        std.debug.assert(Base(Profiled(TestOps)) == TestOps);
        std.debug.assert(Base(TestOps) == TestOps);
        // the prepared capability follows the backend's (RowPlans reads it with isFn)
        std.debug.assert(isFn(Profiled(TestOps), "prepareLaunch"));
        std.debug.assert(!isFn(Profiled(struct {
            pub const T = u32;
        }), "prepareLaunch"));
    }
    const a = testing.allocator;
    var clock: u64 = 0;
    var g = try Profiled(TestOps).init(a, .{ .a = a, .clock = &clock }, .{ .manual = &clock }, 2);
    defer g.deinit();
    try testing.expectEqual(&g.inner, base(&g));
    endPhase(&g);
    try testing.expectEqual(@as(u32, 1), g.prof.unbalanced);
    cycleBegin(&g);
    beginPhase(&g, .verify);
    clock += 5;
    cycleEnd(&g);
    try testing.expect(g.prof.stored[0].unbalanced);
    try testing.expectEqual(@as(u32, 2), g.prof.unbalanced);
    try testing.expectEqual(@as(u64, 5), g.prof.stored[0].root_ns[ix(.verify)]);
    // the next cycle starts clean
    cycleBegin(&g);
    clock += 3;
    cycleEnd(&g);
    try testing.expect(!g.prof.stored[1].unbalanced);
    try testing.expectEqual(@as(u64, 3), g.prof.stored[1].root_ns[glue]);
}
