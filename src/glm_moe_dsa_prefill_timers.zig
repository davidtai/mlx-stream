//! GLM-5.3's prompt-pass phase split, for PROFILE builds only (`-Dplugin-profile=true`): each phase of a prompt
//! forward's layer evaluated on its own and its wall time charged to it, accumulated over one prompt pass and printed
//! at its end. In every other build `enabled` is false and each call below compiles to nothing.

const std = @import("std");
const bo = @import("build_flags.zig");
const ops = @import("deepseek_v41_ops.zig");
const log = @import("sdk").log;
const mlx = @import("sdk").mlx;

pub const enabled: bool = if (@hasDecl(bo, "glm_prefill_timers")) bo.glm_prefill_timers else false;

/// The phases of a prompt layer: the read-ahead's predictor, the q / kv / indexer projections and the cache append,
/// the indexer's selection, the attention core, the output projection, the router, the routed experts (split below),
/// the shared expert or the dense MLP, the residual and the layer's eval.
pub const Phase = enum { seed, proj, index, attn, oproj, router, routed, mlp, rest };
/// The routed call's split: the routing barrier (the ids to the host, their counts and order), the read-ahead's landing,
/// the seed and the routes (reads planned and posted), the read waits, the groups' graphs, their compute drains, the
/// staged hot experts' copies into resident rows, the join, the combine's build.
pub const Routed = enum { barrier, ahead, route, wait, encode, compute, promote, join, combine, commit };

const n_phases = @typeInfo(Phase).@"enum".field_names.len;
const n_routed = @typeInfo(Routed).@"enum".field_names.len;

pub var ns: [n_phases]u64 = @splat(0);
pub var routed_ns: [n_routed]u64 = @splat(0);
/// Attention core per layer kind: full (with the indexer) and shared layers.
pub var attn_full_ns: u64 = 0;
pub var attn_shared_ns: u64 = 0;
var last: u64 = 0;
/// DIAGNOSTIC (waits inventory): the host's waits on the GPU (`MlxOps.waits`) charged to each phase, the routed split's
/// charges counted, and a round trip (one tiny eval on an idle GPU, then a synchronize) timed at every 8th layer.
pub var waits: [n_phases]u64 = @splat(0);
pub var routed_n: [n_routed]u64 = @splat(0);
var last_w: u64 = 0;
var w_begin: u64 = 0;
var rt_n: u64 = 0;
var rt_eval_ns: u64 = 0;
var rt_eval_max: u64 = 0;
var rt_sync_ns: u64 = 0;

/// DIAGNOSTIC (GPU timeline): every command buffer the prompt pass commits, with Metal's GPU start and end (the
/// profile build's commit hook, csrc/dsv41_cb_timeline.mm), tagged with the site whose eval committed it; after the
/// prompt, the GPU's busy and idle time, each idle gap charged to the site of the buffer that ended it.
pub const Site = enum(u8) { none = 0, trunk = 1, group = 2, drain = 3, tail = 4, after = 5, probe = 6 };
const Row = @import("dsv41_verify_timeline.zig").Row;
const tl_cap: u32 = 1 << 16;
var tl_rows: [tl_cap]Row = undefined;
var tl_on: bool = false;
extern fn dsv41tl_mlx_buffer(s: mlx.mlx_stream) ?*anyopaque;
extern fn dsv41tl_install(buf: ?*anyopaque, rows: [*]Row, cap: u32) c_int;
extern fn dsv41tl_tag(tag: u64) void;
extern fn dsv41tl_counts(out: *[3]u32) void;

/// The site of the next commits: routed layer `layer`, bank layer `side`, group `idx`.
pub inline fn tag(site: Site, layer: u32, side: u32, idx: usize) void {
    if (comptime !enabled) return;
    if (!tl_on) return;
    dsv41tl_tag((@as(u64, layer) + 1) << 24 | @as(u64, side) << 16 | @as(u64, @intFromEnum(site)) << 8 | @as(u64, @min(idx, 255)));
}

fn summarize(rs: []const Row, dropped: u32) void {
    // Classes: the site, and for groups: 0 the first group of bank layer 0, 1 the first of bank layer 1, 2 the rest.
    const n_cls = 10;
    const names = [n_cls][]const u8{ "untagged", "trunk", "group0 side0", "group0 side1", "groups after", "drain", "tail", "after", "probe", "other" };
    var busy: [n_cls]u64 = @splat(0);
    var gap: [n_cls]u64 = @splat(0);
    var gap_n: [n_cls]u64 = @splat(0);
    var gap_max: [n_cls]u64 = @splat(0);
    var bufs: [n_cls]u64 = @splat(0);
    var host_late: [n_cls]u64 = @splat(0);
    var end_max: u64 = 0;
    var first: u64 = 0;
    var busy_all: u64 = 0;
    var bad: u64 = 0;
    for (rs, 0..) |r, i| {
        if (r.gpu_end <= r.gpu_start or r.gpu_start == 0) {
            bad += 1;
            continue;
        }
        const site: u8 = @truncate(r.tag >> 8);
        const side: u8 = @truncate(r.tag >> 16);
        const idx: u8 = @truncate(r.tag);
        const cls: usize = switch (site) {
            0 => 0,
            1 => 1,
            2 => if (idx == 0) (if (side == 0) @as(usize, 2) else 3) else 4,
            3 => 5,
            4 => 6,
            5 => 7,
            6 => 8,
            else => 9,
        };
        bufs[cls] += 1;
        busy[cls] += r.gpu_end - r.gpu_start;
        if (i == 0 or first == 0) {
            first = r.gpu_start;
            end_max = r.gpu_end;
            busy_all += r.gpu_end - r.gpu_start;
            continue;
        }
        if (r.gpu_start > end_max) {
            const d = r.gpu_start - end_max;
            gap[cls] += d;
            gap_n[cls] += 1;
            gap_max[cls] = @max(gap_max[cls], d);
            // The host committed it after the GPU had gone idle: the host's own time made the gap.
            if (r.host_commit > end_max) host_late[cls] += r.host_commit - end_max;
            busy_all += r.gpu_end - r.gpu_start;
        } else if (r.gpu_end > end_max) {
            busy_all += r.gpu_end - end_max;
        }
        end_max = @max(end_max, r.gpu_end);
    }
    const span = end_max -| first;
    const f = struct {
        fn s(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e9;
        }
        fn ms(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6;
        }
    };
    log.info("glm_moe_dsa profile timeline: {d} command buffers ({d} dropped, {d} without GPU times), GPU span {d:.3} s, busy {d:.3} s, idle {d:.3} s\n", .{ rs.len, dropped, bad, f.s(span), f.s(busy_all), f.s(span -| busy_all) });
    for (0..n_cls) |c| {
        if (bufs[c] == 0) continue;
        log.info("glm_moe_dsa profile timeline:   {s}: {d} buffers, busy {d:.3} s; idle before them {d:.3} s in {d} gaps (largest {d:.3} ms), of it {d:.3} s committed after the GPU went idle\n", .{ names[c], bufs[c], f.s(busy[c]), f.s(gap[c]), gap_n[c], f.ms(gap_max[c]), f.s(host_late[c]) });
    }
}

pub const Stamp = if (enabled) u64 else void;

pub inline fn now() Stamp {
    if (comptime !enabled) return {};
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The phase clock's start (a layer's first phase).
pub inline fn start() void {
    if (comptime !enabled) return;
    last = now();
}

/// `xs` evaluated, the time since the previous phase's end charged to `p`; returns the time charged.
pub inline fn phase(g: *ops.MlxOps, xs: []const ops.MlxOps.T, p: Phase) !u64 {
    if (comptime !enabled) return 0;
    // DIAGNOSTIC (waits inventory): no eval at a phase's end, so the timers read the prompt pass's own waits.
    _ = xs;
    const t = now();
    const d = t - last;
    ns[@intFromEnum(p)] += d;
    last = t;
    waits[@intFromEnum(p)] += g.waits -| last_w;
    last_w = g.waits;
    return d;
}

/// The prompt pass's first wait count (`MlxOps.waits`).
pub fn begin(g: *ops.MlxOps) void {
    if (comptime !enabled) return;
    w_begin = g.waits;
    last_w = g.waits;
    const rc = dsv41tl_install(dsv41tl_mlx_buffer(g.s), &tl_rows, tl_cap);
    tl_on = rc == 0;
    if (!tl_on) log.info("glm_moe_dsa profile timeline: not installed ({d})\n", .{rc});
    tag(.trunk, 0, 0, 0);
}

/// At every 8th layer, after its eval (the GPU idle): one tiny eval and a synchronize timed; their waits left out of
/// the phases' counts.
pub fn roundTrip(g: *ops.MlxOps, l: u32) !void {
    if (comptime !enabled) return;
    if (l % 8 != 4) return;
    const one = [_]f32{1.0};
    const x = try g.hostArray(std.mem.sliceAsBytes(&one), &.{1}, .float32);
    const y = try g.add(x, x);
    tag(.probe, l, 0, 0);
    const t0 = now();
    try g.evalAll(&.{y});
    const t1 = now();
    _ = mlx.mlx_synchronize(g.s);
    const t2 = now();
    rt_n += 1;
    rt_eval_ns += t1 - t0;
    rt_eval_max = @max(rt_eval_max, t1 - t0);
    rt_sync_ns += t2 - t1;
    last_w = g.waits;
}

/// The time since `t0` charged to the routed call's `r` (the routed phase's total is its own).
pub inline fn chargeRouted(r: Routed, t0: Stamp) void {
    if (comptime !enabled) return;
    routed_ns[@intFromEnum(r)] += now() - t0;
    routed_n[@intFromEnum(r)] += 1;
}

pub inline fn chargeAttn(full: bool, d: u64) void {
    if (comptime !enabled) return;
    if (full) attn_full_ns += d else attn_shared_ns += d;
}

fn sec(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e9;
}

/// The prompt pass's split, one line, then the counters zeroed.
pub fn report(tokens: u64, g: *ops.MlxOps) void {
    if (comptime !enabled) return;
    if (tl_on) {
        dsv41tl_tag(0);
        _ = mlx.mlx_synchronize(g.s);
        var k: [3]u32 = undefined;
        const tw = now();
        while (true) {
            dsv41tl_counts(&k);
            if (k[1] >= @min(k[0], tl_cap) or now() - tw > 5 * std.time.ns_per_s) break;
            std.Thread.yield() catch {};
        }
        summarize(tl_rows[0..@min(k[0], tl_cap)], k[2]);
        tl_on = false;
    }
    {
        var in_phases: u64 = 0;
        for (waits) |x| in_phases += x;
        const total = g.waits - w_begin - rt_n;
        log.info("glm_moe_dsa profile waits: prompt {d} tokens, {d} host waits on the GPU (MlxOps.waits, the round trips left out): in the phases {d} (seed {d}, proj {d}, index {d}, attn {d}, oproj {d}, router {d}, routed {d}, mlp {d}, rest {d}), after the layers {d}; routed charges: barrier {d}, ahead {d}, route {d}, wait {d}, encode {d} (of them asyncEval {d}), compute {d}, promote {d}, join {d}, combine {d}; round trip on an idle GPU over {d} layers: eval mean {d:.3} ms (max {d:.3}), synchronize after it mean {d:.3} ms\n", .{
            tokens,                                        total,                                     in_phases,
            waits[@intFromEnum(Phase.seed)],               waits[@intFromEnum(Phase.proj)],           waits[@intFromEnum(Phase.index)],
            waits[@intFromEnum(Phase.attn)],               waits[@intFromEnum(Phase.oproj)],          waits[@intFromEnum(Phase.router)],
            waits[@intFromEnum(Phase.routed)],             waits[@intFromEnum(Phase.mlp)],            waits[@intFromEnum(Phase.rest)],
            total -| in_phases,                            routed_n[@intFromEnum(Routed.barrier)],    routed_n[@intFromEnum(Routed.ahead)],
            routed_n[@intFromEnum(Routed.route)],          routed_n[@intFromEnum(Routed.wait)],       routed_n[@intFromEnum(Routed.encode)],
            routed_n[@intFromEnum(Routed.commit)],
            routed_n[@intFromEnum(Routed.compute)],        routed_n[@intFromEnum(Routed.promote)],    routed_n[@intFromEnum(Routed.join)],
            routed_n[@intFromEnum(Routed.combine)],        rt_n,                                      if (rt_n > 0) @as(f64, @floatFromInt(rt_eval_ns / rt_n)) / 1e6 else 0,
            @as(f64, @floatFromInt(rt_eval_max)) / 1e6,    if (rt_n > 0) @as(f64, @floatFromInt(rt_sync_ns / rt_n)) / 1e6 else 0,
        });
        waits = @splat(0);
        routed_n = @splat(0);
        rt_n = 0;
        rt_eval_ns = 0;
        rt_eval_max = 0;
        rt_sync_ns = 0;
    }
    var total: u64 = 0;
    for (ns) |x| total += x;
    const r = struct {
        fn of(b: Routed) f64 {
            return sec(routed_ns[@intFromEnum(b)]);
        }
    }.of;
    log.info("glm_moe_dsa profile: prompt {d} tokens, phases {d:.2} s: seed {d:.2}, proj {d:.2}, index {d:.2}, attn {d:.2} (full {d:.2}, shared {d:.2}), oproj {d:.2}, router {d:.2}, routed {d:.2} (barrier {d:.2}, ahead {d:.2}, route {d:.2}, wait {d:.2}, encode {d:.2} (of it asyncEval {d:.2}), compute {d:.2}, promote {d:.2}, join {d:.2}, combine {d:.2}), mlp {d:.2}, rest {d:.2}\n", .{
        tokens,                            sec(total),
        sec(ns[@intFromEnum(Phase.seed)]), sec(ns[@intFromEnum(Phase.proj)]),
        sec(ns[@intFromEnum(Phase.index)]), sec(ns[@intFromEnum(Phase.attn)]),
        sec(attn_full_ns),                 sec(attn_shared_ns),
        sec(ns[@intFromEnum(Phase.oproj)]), sec(ns[@intFromEnum(Phase.router)]),
        sec(ns[@intFromEnum(Phase.routed)]), r(.barrier),
        r(.ahead),                         r(.route),
        r(.wait),                          r(.encode),
        r(.commit),
        r(.compute),                       r(.promote),
        r(.join),                          r(.combine),
        sec(ns[@intFromEnum(Phase.mlp)]),
        sec(ns[@intFromEnum(Phase.rest)]),
    });
    ns = @splat(0);
    routed_ns = @splat(0);
    attn_full_ns = 0;
    attn_shared_ns = 0;
}
