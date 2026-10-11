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
pub const Routed = enum { barrier, ahead, route, wait, encode, compute, promote, join, combine };

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
}

/// At every 8th layer, after its eval (the GPU idle): one tiny eval and a synchronize timed; their waits left out of
/// the phases' counts.
pub fn roundTrip(g: *ops.MlxOps, l: u32) !void {
    if (comptime !enabled) return;
    if (l % 8 != 4) return;
    const one = [_]f32{1.0};
    const x = try g.hostArray(std.mem.sliceAsBytes(&one), &.{1}, .float32);
    const y = try g.add(x, x);
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
    {
        var in_phases: u64 = 0;
        for (waits) |x| in_phases += x;
        const total = g.waits - w_begin - rt_n;
        log.info("glm_moe_dsa profile waits: prompt {d} tokens, {d} host waits on the GPU (MlxOps.waits, the round trips left out): in the phases {d} (seed {d}, proj {d}, index {d}, attn {d}, oproj {d}, router {d}, routed {d}, mlp {d}, rest {d}), after the layers {d}; routed charges: barrier {d}, ahead {d}, route {d}, wait {d}, encode {d}, compute {d}, promote {d}, join {d}, combine {d}; round trip on an idle GPU over {d} layers: eval mean {d:.3} ms (max {d:.3}), synchronize after it mean {d:.3} ms\n", .{
            tokens,                                        total,                                     in_phases,
            waits[@intFromEnum(Phase.seed)],               waits[@intFromEnum(Phase.proj)],           waits[@intFromEnum(Phase.index)],
            waits[@intFromEnum(Phase.attn)],               waits[@intFromEnum(Phase.oproj)],          waits[@intFromEnum(Phase.router)],
            waits[@intFromEnum(Phase.routed)],             waits[@intFromEnum(Phase.mlp)],            waits[@intFromEnum(Phase.rest)],
            total -| in_phases,                            routed_n[@intFromEnum(Routed.barrier)],    routed_n[@intFromEnum(Routed.ahead)],
            routed_n[@intFromEnum(Routed.route)],          routed_n[@intFromEnum(Routed.wait)],       routed_n[@intFromEnum(Routed.encode)],
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
    log.info("glm_moe_dsa profile: prompt {d} tokens, phases {d:.2} s: seed {d:.2}, proj {d:.2}, index {d:.2}, attn {d:.2} (full {d:.2}, shared {d:.2}), oproj {d:.2}, router {d:.2}, routed {d:.2} (barrier {d:.2}, ahead {d:.2}, route {d:.2}, wait {d:.2}, encode {d:.2}, compute {d:.2}, promote {d:.2}, join {d:.2}, combine {d:.2}), mlp {d:.2}, rest {d:.2}\n", .{
        tokens,                            sec(total),
        sec(ns[@intFromEnum(Phase.seed)]), sec(ns[@intFromEnum(Phase.proj)]),
        sec(ns[@intFromEnum(Phase.index)]), sec(ns[@intFromEnum(Phase.attn)]),
        sec(attn_full_ns),                 sec(attn_shared_ns),
        sec(ns[@intFromEnum(Phase.oproj)]), sec(ns[@intFromEnum(Phase.router)]),
        sec(ns[@intFromEnum(Phase.routed)]), r(.barrier),
        r(.ahead),                         r(.route),
        r(.wait),                          r(.encode),
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
