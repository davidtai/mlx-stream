//! X12 microbench, device family `gpuwarm` (test builds only: `MIMO_GPUWARM_GPU=1` + `_GPU_WINDOW_LOCKED=1`, its own
//! ReleaseFast binary, a declared bound). The question it answers before any ABBA: does GPU idle lengthen the next hit
//! wave, and does the route's warm kernel stop it?
//!
//! The wave: the decode lane's served hit-wave kernels (K1 `mimo_moe_k1_gateup` + `mimo_moe_k1_down`, the union texts)
//! over a real-shape MXFP4 slot bank (3 banks x 8 slots of 4096 x 2048 experts, 321 MB), 32 pairs (a 4-row verify's
//! top-8) over 24 distinct slots, as `moe.Fused` dispatches it in decode: ~0.6 ms, bandwidth-bound.
//! The idle: decode's form, an event-gated wait. Each wave's input passes the stream's MTLSharedEvent wait for a fresh
//! value (`dsv41ev_wait`, its own command buffer), the wave is committed behind it, and the host hands the event the
//! value `idle` ms after the commit (the read pool's role), then waits for the wave. The GPU sits idle, blocked on the
//! event, for `idle` ms before every wave: 0 (dense: every wave's value handed before its commit, two waves in flight),
//! 1, 2 (decode's per-layer gap) and 4 ms.
//! The warm arms: off (the control); narrow (the route's form: one simdgroup, `gpu_warm.Warm`, armed with the wave's
//! value after the commit, stopping at the event); wide (64 threadgroups: a diagnostic, never routed).
//! Per train: 12 unrecorded waves, then 48 recorded. Per wave: its GPU span (the first GPU start to the last GPU end of
//! its buffers, Metal's GPUStartTime / GPUEndTime through the commit probe, the bench thread's commits only), the GPU's
//! resume after the event (the wave's GPU start minus the host's signal), and the GPU gap before it. Per train: the
//! warm engine's chunks and their GPU time, the IOReport GPU energy, active-time mean MHz and thermal limiting
//! (`gpu_power`), and the last wave's output bits against the reference (exact: the warm kernel touches only its own
//! buffer). Three repetitions, the arm order reversed in the middle one.
//!
//! The verdict (declared here, before any run), each in at least 2 of the 3 repetitions, at decode's 2 ms gap:
//!   LENGTHENS: warm off, the median wave is longer than the dense median by more than 3 % and more than 15 us;
//!   RECOVERS:  the narrow warm's median wave is back within a third of that gap;
//!   SHORTENS:  the narrow warm's median (resume + wave: the wave's GPU end after the host's signal, what a decode layer
//!              waits for once its bytes land) is below warm off's by more than 2 % and more than 10 us.
//! The lever is ALIVE when idle lengthens the wave or the warm shortens the layer's wait-to-done; otherwise it is dead
//! and the window's driver stops before the ABBA.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");
const moe = @import("moe.zig");
const bank_mod = @import("bank.zig");
const expert_event = @import("../../sdk_ext.zig").expert.event;
const gpu_warm = @import("gpu_warm.zig");
const gpu_power = @import("gpu_power.zig");

const T = mlx.mlx_array;
const Proj = @import("../../sdk_ext.zig").quant.GatherQmm.Arrays(T);
const testing = std.testing;

// ── the command-buffer probe (csrc/mimo_gpu_probe.inc; test binaries only) ──
const probe = struct {
    extern fn mimo_gpu_probe_install() i32;
    extern fn mimo_gpu_probe_arm_thread(rows: [*]f64, cap: u32) void;
    extern fn mimo_gpu_probe_disarm() void;
    extern fn mimo_gpu_probe_counts(out: *[4]u32) void;
};

pub const slots: u32 = 8;
pub const n_banks: u32 = 3;
pub const pairs: u32 = 32;
pub const distinct: u32 = 24;
pub const warmup_waves: u32 = 12;
pub const waves: u32 = 48;
pub const reps: u32 = 3;
pub const idles_ms = [_]u32{ 1, 2, 4 };
pub const Warm = enum { off, narrow, wide };
/// The decode round's gated layers (cells/k2-20261009 VERIFY_GPU_TIMELINE median `gated_layers` 38) and GPU work
/// (`gpu_work_ms` 44.8 normal cell): the verdict's ms/round extrapolations.
pub const gated_layers_per_round: f64 = 38;
pub const gpu_work_ms_per_round: f64 = 44.8;
const probe_cap: u32 = 4096;

/// Device: the bank 3 x 8 x 13,369,344 B = 321 MB, x / h / y < 1 MiB, the warm buffers 2 x 32 KiB; host staging one
/// projection (33.6 MB) at a time; MLX transients 64 MiB, JIT and allocator slack 512 MiB.
pub fn bound() u64 {
    return @as(u64, n_banks) * slots * 13_369_344 + (64 << 20) + (512 << 20) + 34 * 1_000_000;
}
pub const mlx_limit: usize = 1536 << 20;

fn xorshift(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

fn projection(a: std.mem.Allocator, g: *ops.MlxOps, out: u32, in: u32, state: *u32) !Proj {
    const words = try a.alloc(u32, slots * out * (in / 8));
    defer a.free(words);
    const scales = try a.alloc(u8, slots * out * (in / 32));
    defer a.free(scales);
    for (words) |*w| w.* = xorshift(state);
    // E8M0 near 2^-4 .. 2^0, the real banks' range (moe.zig's self-check fixture)
    for (scales) |*sc| sc.* = @intCast(123 + xorshift(state) % 5);
    const w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(slots), @intCast(out), @intCast(in / 8) }, .uint32);
    const sc = try g.hostArray(scales, &.{ @intCast(slots), @intCast(out), @intCast(in / 32) }, .uint8);
    try g.evalAll(&.{ w, sc });
    return .{ .w = g.keep(w), .scales = g.keep(sc) };
}

fn median(xs: []f64) f64 {
    return quantile(xs, 0.5);
}

fn quantile(xs: []f64, q: f64) f64 {
    if (xs.len == 0) return 0;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const i: usize = @min(xs.len - 1, @as(usize, @intFromFloat(@floor(q * @as(f64, @floatFromInt(xs.len - 1)) + 0.5))));
    return xs[i];
}

fn sleepUntil(t: u64) void {
    while (true) {
        const now = gpu_warm.nowNs();
        if (now >= t) return;
        const left = t - now;
        if (left > 300 * std.time.ns_per_us) {
            const ts = std.c.timespec{ .sec = 0, .nsec = @intCast(left - 250 * std.time.ns_per_us) };
            _ = std.c.nanosleep(&ts, null);
        } else std.atomic.spinLoopHint();
    }
}

const Wave = struct { c0: u32, c1: u32, t_sig: u64 };
const Train = struct {
    wave_us: [waves]f64 = undefined,
    resume_us: [waves]f64 = undefined,
    /// The wave's GPU end after the host's signal (resume + wave): what a decode layer waits for after its bytes land.
    done_us: [waves]f64 = undefined,
    gap_us: [waves]f64 = undefined,
    n: usize = 0,
    warm: gpu_warm.Stats = .{},
    power: ?gpu_power.Delta = null,
    exact: bool = true,
};

const Bench = struct {
    a: std.mem.Allocator,
    g: *ops.MlxOps,
    s: mlx.mlx_stream,
    fused: *const moe.Fused,
    banks: *const [moe.n_banks]moe.Arrays,
    x: T,
    ids: T,
    ev: expert_event.Event,
    value: u64 = 0,
    rows: []f64,
    narrow: *gpu_warm.Warm,
    wide: *gpu_warm.Warm,
    power: ?gpu_power.Sampler,
    reference: []u32,
    have_reference: bool = false,

    fn counts() [4]u32 {
        var k: [4]u32 = undefined;
        probe.mimo_gpu_probe_counts(&k);
        return k;
    }

    /// The wave behind the event's next value: (the gated input's handle to free, the output).
    fn build(b: *Bench) !struct { xw: T, y: T } {
        b.value += 1;
        var outs = [_]T{mlx.mlx_array_new()};
        try expert_event.wait(&.{b.x}, b.ev, b.value, &.{}, false, b.s, &outs);
        const h = try b.fused.gateUp(b.g, outs[0], b.ids, b.banks);
        const y = try b.fused.down(b.g, h, b.ids, b.banks);
        return .{ .xw = outs[0], .y = y };
    }

    /// One wave's span from its rows: the earliest GPU start and the latest GPU end of the bench thread's buffers it
    /// committed (wait-only rows carry no start and are skipped).
    fn span(b: *const Bench, w: Wave) ?[2]f64 {
        var lo: f64 = std.math.inf(f64);
        var hi: f64 = 0;
        for (w.c0..w.c1) |i| {
            const r = b.rows[3 * i ..][0..3];
            if (r[1] <= 0 or r[2] <= 0) continue;
            lo = @min(lo, r[1]);
            hi = @max(hi, r[2]);
        }
        return if (hi > lo) .{ lo, hi } else null;
    }

    fn settleProbe(_: *const Bench) !void {
        var spins: u32 = 0;
        while (true) {
            const k = counts();
            if (k[1] >= @min(k[0], probe_cap)) break;
            spins += 1;
            if (spins > 5_000_000) return error.GpuWarmBenchProbePending;
            std.Thread.yield() catch {};
        }
        if (counts()[2] != 0) return error.GpuWarmBenchProbeDropped;
    }

    /// The output's bits against the reference (the first train's last wave).
    fn checkExact(b: *Bench, y: T) !bool {
        const n: usize = pairs * moe.hidden;
        const vals = try b.a.alloc(f32, n);
        defer b.a.free(vals);
        _ = try b.g.hostF32(try b.g.astype(y, .float32), vals);
        if (!b.have_reference) {
            for (vals, b.reference) |v, *r| r.* = @bitCast(v);
            b.have_reference = true;
            return true;
        }
        for (vals, b.reference) |v, r| if (@as(u32, @bitCast(v)) != r) return false;
        return true;
    }

    fn engine(b: *Bench, warm: Warm) ?*gpu_warm.Warm {
        return switch (warm) {
            .off => null,
            .narrow => b.narrow,
            .wide => b.wide,
        };
    }

    /// One train: `idle_ms` = 0 is dense (each value handed before its wave's commit, two waves in flight).
    fn train(b: *Bench, idle_ms: u32, warm: Warm) !Train {
        var out: Train = .{};
        const eng = b.engine(warm);
        const w0 = if (eng) |e| e.stats() else gpu_warm.Stats{};
        const p0 = if (b.power) |*p| p.sample() else null;
        defer if (b.power) |*p| p.free(p0);
        var meta: [warmup_waves + waves]Wave = undefined;
        try mlx.check(mlx.mlx_synchronize(b.s));
        probe.mimo_gpu_probe_arm_thread(b.rows.ptr, probe_cap);
        var last: T = .{ .ctx = null };
        if (idle_ms == 0) {
            var prev: ?T = null;
            for (0..warmup_waves + waves) |i| {
                const wv = try b.build();
                gpu_warm.signalEvent(b.ev.object, b.value);
                try b.g.asyncEval(&.{wv.xw});
                const c0 = counts()[0];
                try b.g.asyncEval(&.{wv.y});
                meta[i] = .{ .c0 = c0, .c1 = counts()[0], .t_sig = 0 };
                const kept = b.g.keep(wv.y);
                _ = mlx.mlx_array_free(wv.xw);
                b.g.reset();
                // two waves in flight: the previous one is waited only after this one is committed
                if (prev) |p| {
                    try b.g.evalAll(&.{p});
                    b.g.release(p);
                }
                prev = kept;
            }
            try b.g.evalAll(&.{prev.?});
            last = prev.?;
        } else {
            for (0..warmup_waves + waves) |i| {
                const wv = try b.build();
                try b.g.asyncEval(&.{wv.xw});
                const c0 = counts()[0];
                try b.g.asyncEval(&.{wv.y});
                const c1 = counts()[0];
                const t_commit = gpu_warm.nowNs();
                if (eng) |e| e.arm(b.value);
                sleepUntil(t_commit + @as(u64, idle_ms) * std.time.ns_per_ms);
                const t_sig = gpu_warm.nowNs();
                gpu_warm.signalEvent(b.ev.object, b.value);
                try b.g.evalAll(&.{wv.y});
                meta[i] = .{ .c0 = c0, .c1 = c1, .t_sig = t_sig };
                if (i + 1 == warmup_waves + waves) last = b.g.keep(wv.y);
                _ = mlx.mlx_array_free(wv.xw);
                b.g.reset();
            }
        }
        probe.mimo_gpu_probe_disarm();
        try b.settleProbe();
        if (eng) |e| {
            // the engine parks once the last value is handed (its chunks in flight retire)
            var spins: u32 = 0;
            while (!e.isParked()) : (spins += 1) {
                if (spins > 2_000_000) return error.GpuWarmBenchEngineBusy;
                std.Thread.yield() catch {};
            }
            out.warm = e.stats().since(w0);
        }
        if (b.power) |*p| {
            const p1 = p.sample();
            defer p.free(p1);
            out.power = p.delta(p0, p1) catch null;
        }
        out.exact = try b.checkExact(last);
        b.g.release(last);
        var prev_end: f64 = 0;
        for (meta, 0..) |m, i| {
            const sp = b.span(m) orelse return error.GpuWarmBenchNoWaveRows;
            defer prev_end = sp[1];
            if (i < warmup_waves) continue;
            out.wave_us[out.n] = (sp[1] - sp[0]) * 1e6;
            out.resume_us[out.n] = if (m.t_sig != 0) (sp[0] - @as(f64, @floatFromInt(m.t_sig)) * 1e-9) * 1e6 else 0;
            out.done_us[out.n] = if (m.t_sig != 0) (sp[1] - @as(f64, @floatFromInt(m.t_sig)) * 1e-9) * 1e6 else 0;
            out.gap_us[out.n] = if (prev_end > 0) (sp[0] - prev_end) * 1e6 else 0;
            out.n += 1;
        }
        return out;
    }
};

fn printTrain(rep: u32, idle_ms: u32, warm: Warm, t: *Train, mhz: []const u32) void {
    const n = t.n;
    const w50 = median(t.wave_us[0..n]);
    const w10 = quantile(t.wave_us[0..n], 0.1);
    const w90 = quantile(t.wave_us[0..n], 0.9);
    const r50 = median(t.resume_us[0..n]);
    const d50 = median(t.done_us[0..n]);
    const g50 = median(t.gap_us[0..n]);
    const chunk_us = if (t.warm.retired > 0) @as(f64, @floatFromInt(t.warm.warm_gpu_ns)) / @as(f64, @floatFromInt(t.warm.retired)) / 1e3 else 0;
    var pb: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&pb);
    if (t.power) |d| {
        gpu_power.json(&w, d, mhz) catch {};
    } else {
        w.writeAll("null") catch {};
    }
    std.debug.print("MIMO_GPUWARM_BENCH rep={d} idle_ms={d} warm={s} waves={d} wave_us_p50={d:.1} wave_us_p10={d:.1} wave_us_p90={d:.1} resume_us_p50={d:.1} done_us_p50={d:.1} gap_us_p50={d:.1} warm_episodes={d} warm_chunks={d} warm_chunk_us={d:.1} warm_iters={d} warm_stuck={d} warm_failed={} exact={} power={s}\n", .{ rep, idle_ms, @tagName(warm), n, w50, w10, w90, r50, d50, g50, t.warm.episodes, t.warm.chunks, chunk_us, t.warm.iters, t.warm.stuck, t.warm.failed, t.exact, w.buffered() });
}

test "mimo GPU warm: the hit wave after 0 / 1 / 2 / 4 ms of event-gated GPU idle with the warm kernel off / on (narrow, the route's) / wide (MIMO_GPUWARM_GPU, guarded window)" {
    _ = std.c.getenv("MIMO_GPUWARM_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("MIMO_GPUWARM_BENCH bound_bytes={d} mlx_limit={d} slots={d}x{d} pairs={d} distinct={d} warmup={d} waves={d} reps={d} idles_ms=0,1,2,4 chunk_target_us={d} narrow_groups={d} wide_groups={d}\n", .{ bound(), mlx_limit, n_banks, slots, pairs, distinct, warmup_waves, waves, reps, gpu_warm.chunk_target_ns / 1000, gpu_warm.route_groups, gpu_warm.wide_groups });
    const a = testing.allocator;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, mlx_limit));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const geometry = try bank_mod.Geometry.init(moe.hidden, moe.inter, 256, 1, 47);
    var fused = try moe.Fused.initFamily(geometry, slots, .k1);
    defer fused.deinit();
    var state: u32 = 0x6770_7761;
    var banks: [moe.n_banks]moe.Arrays = undefined;
    for (&banks) |*bk| bk.* = .{
        .gate = try projection(a, &g, moe.inter, moe.hidden, &state),
        .up = try projection(a, &g, moe.inter, moe.hidden, &state),
        .down = try projection(a, &g, moe.hidden, moe.inter, &state),
    };
    defer for (banks) |bk| for ([_]Proj{ bk.gate, bk.up, bk.down }) |p| {
        g.release(p.w);
        g.release(p.scales.?);
    };
    // the 4 verify rows' token rows, each pair its token's; pair p reads slot p mod 24 (8 slots read twice)
    var tok: [4 * moe.hidden]f32 = undefined;
    for (&tok) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(xorshift(&state) % 2001)) - 1000)) / 4000;
    var xs: [pairs * moe.hidden]f32 = undefined;
    for (0..pairs) |p| @memcpy(xs[p * moe.hidden ..][0..moe.hidden], tok[(p / moe.top_k) * moe.hidden ..][0..moe.hidden]);
    var idv: [pairs]u32 = undefined;
    for (&idv, 0..) |*id, p| {
        const slot: u32 = @intCast(p % distinct);
        id.* = (slot / slots) << moe.row_bits | (slot % slots);
    }
    const x = g.keep(try g.astype(try g.hostArray(std.mem.sliceAsBytes(&xs), &.{ pairs, moe.hidden }, .float32), .bfloat16));
    defer g.release(x);
    const ids = g.keep(try g.hostArray(std.mem.sliceAsBytes(&idv), &.{pairs}, .uint32));
    defer g.release(ids);
    try g.evalAll(&.{ x, ids });
    g.reset();
    const ev = try expert_event.createMetal();
    if (probe.mimo_gpu_probe_install() != 0) return error.GpuWarmBenchProbeInstall;
    const rows = try a.alloc(f64, 3 * probe_cap);
    defer a.free(rows);
    var nb = try gpu_warm.Metal.init(gpu_warm.route_groups, ev.object);
    const narrow = gpu_warm.Warm.create(a, nb, .{}) catch |e| {
        nb.deinit();
        return e;
    };
    defer narrow.destroy();
    var wb = try gpu_warm.Metal.init(gpu_warm.wide_groups, ev.object);
    const wide = gpu_warm.Warm.create(a, wb, .{}) catch |e| {
        wb.deinit();
        return e;
    };
    defer wide.destroy();
    const power: ?gpu_power.Sampler = gpu_power.Sampler.open() catch |e| blk: {
        std.debug.print("MIMO_GPUWARM_POWER unavailable reason={s}: {s}\n", .{ @errorName(e), gpu_power.lastError() });
        break :blk null;
    };
    const mhz: []const u32 = if (power) |*p| p.mhz[0..p.n_mhz] else &.{};
    const reference = try a.alloc(u32, pairs * moe.hidden);
    defer a.free(reference);
    var b: Bench = .{ .a = a, .g = &g, .s = s, .fused = &fused, .banks = &banks, .x = x, .ids = ids, .ev = ev, .rows = rows, .narrow = narrow, .wide = wide, .power = power, .reference = reference };

    // [rep][idle index][warm]; idle index 0 = dense (warm off only)
    var med: [reps][1 + idles_ms.len][3]f64 = undefined;
    var res: [reps][1 + idles_ms.len][3]f64 = undefined;
    var done: [reps][1 + idles_ms.len][3]f64 = undefined;
    var all_exact = true;
    for (0..reps) |ri| {
        const rep: u32 = @intCast(ri);
        var dense = try b.train(0, .off);
        printTrain(rep, 0, .off, &dense, mhz);
        med[ri][0] = .{ median(dense.wave_us[0..dense.n]), 0, 0 };
        res[ri][0] = .{ 0, 0, 0 };
        done[ri][0] = .{ 0, 0, 0 };
        all_exact = all_exact and dense.exact;
        for (idles_ms, 1..) |idle, ii| {
            const order: [3]Warm = if (rep == 1) .{ .wide, .narrow, .off } else .{ .off, .narrow, .wide };
            for (order) |wm| {
                var t = try b.train(idle, wm);
                printTrain(rep, idle, wm, &t, mhz);
                med[ri][ii][@intFromEnum(wm)] = median(t.wave_us[0..t.n]);
                res[ri][ii][@intFromEnum(wm)] = median(t.resume_us[0..t.n]);
                done[ri][ii][@intFromEnum(wm)] = median(t.done_us[0..t.n]);
                all_exact = all_exact and t.exact;
                if (wm != .off) try testing.expect(!t.warm.failed);
            }
        }
    }
    // The table (medians over the repetitions) and the declared verdict.
    var lengthens_votes: u32 = 0;
    var recovers_votes: u32 = 0;
    var wide_votes: u32 = 0;
    var done_votes: u32 = 0;
    for (0..reps) |ri| {
        const d = med[ri][0][0];
        const off2 = med[ri][2][@intFromEnum(Warm.off)];
        const nar2 = med[ri][2][@intFromEnum(Warm.narrow)];
        const wid2 = med[ri][2][@intFromEnum(Warm.wide)];
        const gap = off2 - d;
        const long = gap > 0.03 * d and gap > 15;
        lengthens_votes += @intFromBool(long);
        recovers_votes += @intFromBool(long and nar2 - d <= gap / 3);
        wide_votes += @intFromBool(long and wid2 - d <= gap / 3);
        // the layer's view: the wave's end after the bytes land (resume + wave), narrow vs off at the decode gap
        const done_off = done[ri][2][@intFromEnum(Warm.off)];
        const done_nar = done[ri][2][@intFromEnum(Warm.narrow)];
        done_votes += @intFromBool(done_off - done_nar > 0.02 * done_off and done_off - done_nar > 10);
    }
    for (idles_ms, 1..) |idle, ii| {
        var dv: [reps]f64 = undefined;
        var ov: [reps]f64 = undefined;
        var nv: [reps]f64 = undefined;
        var wv: [reps]f64 = undefined;
        var orv: [reps]f64 = undefined;
        var nrv: [reps]f64 = undefined;
        var odv: [reps]f64 = undefined;
        var ndv: [reps]f64 = undefined;
        for (0..reps) |ri| {
            odv[ri] = done[ri][ii][@intFromEnum(Warm.off)];
            ndv[ri] = done[ri][ii][@intFromEnum(Warm.narrow)];
            dv[ri] = med[ri][0][0];
            ov[ri] = med[ri][ii][@intFromEnum(Warm.off)];
            nv[ri] = med[ri][ii][@intFromEnum(Warm.narrow)];
            wv[ri] = med[ri][ii][@intFromEnum(Warm.wide)];
            orv[ri] = res[ri][ii][@intFromEnum(Warm.off)];
            nrv[ri] = res[ri][ii][@intFromEnum(Warm.narrow)];
        }
        const d = median(&dv);
        const o = median(&ov);
        const n = median(&nv);
        const wdt = median(&wv);
        std.debug.print("MIMO_GPUWARM_TABLE idle_ms={d} dense_wave_us={d:.1} off_wave_us={d:.1} narrow_wave_us={d:.1} wide_wave_us={d:.1} off_vs_dense_pct={d:.2} narrow_vs_dense_pct={d:.2} wide_vs_dense_pct={d:.2} off_resume_us={d:.1} narrow_resume_us={d:.1} off_done_us={d:.1} narrow_done_us={d:.1}\n", .{ idle, d, o, n, wdt, (o - d) / d * 100, (n - d) / d * 100, (wdt - d) / d * 100, median(&orv), median(&nrv), median(&odv), median(&ndv) });
    }
    // ms/round predictions at the decode gap (2 ms): the wave delta x gated layers, and the slowdown share of the round's
    // GPU work (every post-wait part at the same clock), resume included.
    var dv: [reps]f64 = undefined;
    var ov: [reps]f64 = undefined;
    var nv: [reps]f64 = undefined;
    var orv: [reps]f64 = undefined;
    var nrv: [reps]f64 = undefined;
    for (0..reps) |ri| {
        dv[ri] = med[ri][0][0];
        ov[ri] = med[ri][2][@intFromEnum(Warm.off)];
        nv[ri] = med[ri][2][@intFromEnum(Warm.narrow)];
        orv[ri] = res[ri][2][@intFromEnum(Warm.off)];
        nrv[ri] = res[ri][2][@intFromEnum(Warm.narrow)];
    }
    const d2 = median(&dv);
    const o2 = median(&ov);
    const n2 = median(&nv);
    const per_layer_us = (o2 - n2) + (median(&orv) - median(&nrv));
    const wave_ms = per_layer_us * gated_layers_per_round / 1000;
    const share_ms = gpu_work_ms_per_round * (o2 - n2) / d2 + (median(&orv) - median(&nrv)) * gated_layers_per_round / 1000;
    const lengthens = lengthens_votes >= 2;
    const shortens = done_votes >= 2;
    std.debug.print("MIMO_GPUWARM_VERDICT idle_lengthens_wave={s} votes={d}/{d} narrow_recovers={s} votes={d}/{d} wide_recovers={s} votes={d}/{d} warm_shortens_done={s} votes={d}/{d} lever_alive={s} rule=\"lengthens: off@2ms median wave > dense median by > 3 % and > 15 us; recovers: narrow within 1/3 of that gap; shortens: narrow@2ms median (resume + wave) below off's by > 2 % and > 10 us; alive = lengthens or shortens; each in >= 2 of 3 reps\" per_layer_saving_us={d:.1} predicted_round_saving_ms_wave={d:.2} predicted_round_saving_ms_gpu_work_share={d:.2} gated_layers={d:.0} gpu_work_ms={d:.1} all_exact={}\n", .{ if (lengthens) "yes" else "no", lengthens_votes, reps, if (recovers_votes >= 2) "yes" else "no", recovers_votes, reps, if (wide_votes >= 2) "yes" else "no", wide_votes, reps, if (shortens) "yes" else "no", done_votes, reps, if (lengthens or shortens) "yes" else "no", per_layer_us, wave_ms, share_ms, gated_layers_per_round, gpu_work_ms_per_round, all_exact });
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("MIMO_GPUWARM_BENCH peak_bytes={d} bound_bytes={d}\n", .{ peak, bound() });
    try testing.expect(all_exact);
    try testing.expect(peak <= bound());
}
