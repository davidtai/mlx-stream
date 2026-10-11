//! GLM-5.3's prompt-pass phase split, for PROFILE builds only (`-Dplugin-profile=true`). In every other build `enabled`
//! is false and each call below compiles to nothing.
//!
//! DIAGNOSTIC (not for merge), the GPU idle inventory: no eval at a phase's end (the GPU queue as the shipped build
//! fills it); every command buffer of the prompt pass recorded through the commit hook (csrc/dsv41_cb_timeline.mm: its
//! host commit, Metal's GPU start and end), tagged by the site that committed it; the host's state marked at each
//! step of the layer (`host`). After the pass each GPU idle gap is charged to the site of the buffer that ended it and,
//! inside the gap, to the host's state over time (the part after that buffer's commit to "commit to start").
//! `GLM53_FINE_LAYERS` (a comma list of layers): those layers run op by op, each op evaluated alone (one command
//! buffer), its GPU time and wall time reported by op.

const std = @import("std");
const bo = @import("build_flags.zig");
const ops = @import("deepseek_v41_ops.zig");
const sdk_ext = @import("sdk_ext.zig");
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

const clock = sdk_ext.profile.Clock(enabled);
pub const Stamp = clock.Stamp;
pub const now = clock.now;

// ── the GPU timeline and the host's marks ──

/// The site of a command buffer: the layer's trunk, a group (its index and bank layer in the tag), a drain, the
/// layer's tail, after the layers, an op of a fine layer.
pub const Site = enum(u8) { none = 0, trunk = 1, group = 2, drain = 3, tail = 4, after = 5, fine = 6 };
/// What the host is doing: building the trunk's graph, the read-ahead's predictor, the staged misses' routes, in the
/// routing barrier (MLX encoding the trunk, then the wait), planning a bank layer's groups, waiting for the read-ahead,
/// waiting for a group's reads, building a group's graph, in a group's asyncEval (MLX's encoding and commit), waiting
/// in a drain, the drain's release and promotion hand-off, routing ahead, waiting for the copier, the join's build, the
/// tail's build, the tail's asyncEval, after the layers, a fine layer's op.
pub const Host = enum(u8) { none, trunk, seed, stage, barrier, plan, ahead, read_wait, encode, commit, drain_wait, drain_post, route, copier_wait, join, tail, tail_commit, after, fine };
const n_host = @typeInfo(Host).@"enum".field_names.len;

const Row = @import("dsv41_verify_timeline.zig").Row;
const tl_cap: u32 = 1 << 17;
var tl_rows: [tl_cap]Row = undefined;
var tl_on: bool = false;
extern fn dsv41tl_mlx_buffer(s: mlx.mlx_stream) ?*anyopaque;
extern fn dsv41tl_install(buf: ?*anyopaque, rows: [*]Row, cap: u32) c_int;
extern fn dsv41tl_tag(tag: u64) void;
extern fn dsv41tl_counts(out: *[3]u32) void;
extern fn dsv41tl_now() u64;

const hm_cap: u32 = 1 << 20;
var hm_t: [hm_cap]u64 = undefined;
var hm_s: [hm_cap]Host = undefined;
var hm_n: u32 = 0;
var hm_cur: Host = .none;

/// The fine layers (`GLM53_FINE_LAYERS`), the one running now and its op labels' totals.
pub const Op = enum(u8) { q_a, q_a_norm, q_b, q_rope, iq_proj, iq_rope, kv_a, kv_norm, k_rope, ik_proj, ik_norm, ik_rope, p_proj, p_index, p_attn, o_transpose, o_proj, p_oproj, post_norm, p_router, p_routed, mlp_gate, mlp_silu, mlp_up, mlp_mul, mlp_down, p_mlp, tail };
const n_ops = @typeInfo(Op).@"enum".field_names.len;
var fine_set: std.StaticBitSet(128) = .initEmpty();
var fine_cur: ?u32 = null;
var fine_label: Op = .q_a;
var fine_wall: [128][n_ops]u64 = undefined;
var fine_n: [128][n_ops]u32 = undefined;

fn tagWord(site: Site, layer: u32, side: u32, idx: usize) u64 {
    return (@as(u64, layer) + 1) << 24 | @as(u64, side) << 16 | @as(u64, @intFromEnum(site)) << 8 | @as(u64, @min(idx, 255));
}

/// The site of the next commits: layer `layer`, bank layer `side`, group `idx` (in a fine layer the op's label stays).
pub inline fn tag(site: Site, layer: u32, side: u32, idx: usize) void {
    if (comptime !enabled) return;
    if (!tl_on or fine_cur != null) return;
    dsv41tl_tag(tagWord(site, layer, side, idx));
}

/// The host's state from now.
pub inline fn host(s: Host) void {
    if (comptime !enabled) return;
    if (!tl_on or hm_n >= hm_cap) return;
    if (fine_cur != null and s != .fine) {
        hm_cur = s;
        return;
    }
    hm_t[hm_n] = dsv41tl_now();
    hm_s[hm_n] = s;
    hm_n += 1;
    hm_cur = s;
}

/// The host's state `s` from now; returns the state it replaced (`leave` restores it).
pub inline fn enter(s: Host) Host {
    if (comptime !enabled) return .none;
    const p = hm_cur;
    host(s);
    return p;
}

pub inline fn leave(p: Host) void {
    if (comptime !enabled) return;
    host(p);
}

fn copierWait(on: bool) void {
    const S = struct {
        var saved: Host = .none;
    };
    if (on) S.saved = enter(.copier_wait) else leave(S.saved);
}

/// A prompt layer's start: its trunk's tag and state, and whether it runs op by op.
pub fn layerStart(l: u32) void {
    if (comptime !enabled) return;
    fine_cur = if (tl_on and l < 128 and fine_set.isSet(l)) l else null;
    if (fine_cur != null) {
        host(.fine);
        fine_label = .q_a;
        dsv41tl_tag(tagWord(.fine, l, 0, @intFromEnum(Op.q_a)));
        return;
    }
    tag(.trunk, l, 0, 0);
    host(.trunk);
}

/// The layers' end: the tag and state after them.
pub fn layersEnd() void {
    if (comptime !enabled) return;
    fine_cur = null;
    tag(.after, 0, 0, 0);
    host(.after);
}

pub inline fn fine() bool {
    if (comptime !enabled) return false;
    return fine_cur != null;
}

/// In a fine layer: `xs` evaluated alone under label `o` (its command buffers tagged with it), its wall time charged.
pub fn op(g: *ops.MlxOps, xs: []const ops.MlxOps.T, o: Op) !void {
    if (comptime !enabled) return;
    const l = fine_cur orelse return;
    fine_label = o;
    dsv41tl_tag(tagWord(.fine, l, 0, @intFromEnum(o)));
    const t0 = dsv41tl_now();
    if (xs.len > 0) try g.evalAll(xs);
    fine_wall[l][@intFromEnum(o)] += dsv41tl_now() - t0;
    fine_n[l][@intFromEnum(o)] += 1;
}

/// The phase clock's start (a layer's first phase).
pub inline fn start() void {
    if (comptime !enabled) return;
    last = now();
}

const phase_op = [n_phases]?Op{ null, .p_proj, .p_index, .p_attn, .p_oproj, .p_router, .p_routed, .p_mlp, .tail };

/// The time since the previous phase's end charged to `p` (no eval: the queue as shipped; a fine layer evaluates `xs`
/// under the phase's label); returns the time charged.
pub inline fn phase(g: *ops.MlxOps, xs: []const ops.MlxOps.T, p: Phase) !u64 {
    if (comptime !enabled) return 0;
    if (fine_cur != null) if (phase_op[@intFromEnum(p)]) |o| try op(g, xs, o);
    const t = now();
    const d = t - last;
    ns[@intFromEnum(p)] += d;
    last = t;
    return d;
}

/// The time since `t0` charged to the routed call's `r` (the routed phase's total is its own).
pub inline fn chargeRouted(r: Routed, t0: Stamp) void {
    if (comptime !enabled) return;
    routed_ns[@intFromEnum(r)] += now() - t0;
}

pub inline fn chargeAttn(full: bool, d: u64) void {
    if (comptime !enabled) return;
    if (full) attn_full_ns += d else attn_shared_ns += d;
}

/// The prompt pass's start: the commit hook installed on MLX's stream, the marks and the fine layers reset.
pub fn begin(g: *ops.MlxOps) void {
    if (comptime !enabled) return;
    const rc = dsv41tl_install(dsv41tl_mlx_buffer(g.s), &tl_rows, tl_cap);
    tl_on = rc == 0;
    if (!tl_on) log.info("glm_moe_dsa profile timeline: not installed ({d})\n", .{rc});
    hm_n = 0;
    hm_cur = .none;
    fine_cur = null;
    fine_set = .initEmpty();
    fine_wall = @splat(@splat(0));
    fine_n = @splat(@splat(0));
    if (std.c.getenv("GLM53_FINE_LAYERS")) |v| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |s| {
            const x = std.fmt.parseInt(u32, s, 10) catch continue;
            if (x < 128) fine_set.set(x);
        }
    }
    sdk_ext.expert.stream.diag_copier_wait = &copierWait;
    host(.trunk);
    tag(.trunk, 0, 0, 0);
}

const n_cls = 9;
const cls_names = [n_cls][]const u8{ "untagged", "layer start (trunk)", "router sync (group 0, bank layer 0)", "bank layer 1 start (its group 0)", "group drains (groups after)", "drain-tagged", "layer end (tail)", "after the layers", "fine layers" };

fn classOf(t: u64) usize {
    const site: u8 = @truncate(t >> 8);
    const side: u8 = @truncate(t >> 16);
    const idx: u8 = @truncate(t);
    return switch (site) {
        0 => 0,
        1 => 1,
        2 => if (idx == 0) (if (side == 0) @as(usize, 2) else 3) else 4,
        3 => 5,
        4 => 6,
        5 => 7,
        6 => 8,
        else => 0,
    };
}

fn s_(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e9;
}

fn ms_(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e6;
}

/// The host's states over [x, y) added to `acc` (marks in time order; `at` the search's start, advanced).
fn chargeHost(x: u64, y: u64, acc: *[n_host]u64, at: *u32) void {
    if (y <= x) return;
    while (at.* + 1 < hm_n and hm_t[at.* + 1] <= x) at.* += 1;
    if (hm_n == 0) {
        acc[@intFromEnum(Host.none)] += y - x;
        return;
    }
    var i = at.*;
    var t = x;
    while (t < y) {
        if (hm_t[i] > t) {
            // Before the first mark.
            const e = @min(y, hm_t[i]);
            acc[@intFromEnum(Host.none)] += e - t;
            t = e;
            continue;
        }
        const next: u64 = if (i + 1 < hm_n) hm_t[i + 1] else std.math.maxInt(u64);
        const e = @min(y, next);
        acc[@intFromEnum(hm_s[i])] += e - t;
        t = e;
        if (t < y) i += 1;
    }
}

fn summarize(rs: []const Row, dropped: u32) void {
    var gap: [n_cls]u64 = @splat(0);
    var gap_n: [n_cls]u64 = @splat(0);
    var gap_max: [n_cls]u64 = @splat(0);
    var busy: [n_cls]u64 = @splat(0);
    var bufs: [n_cls]u64 = @splat(0);
    var by_host: [n_cls][n_host]u64 = @splat(@splat(0));
    var c2s: [n_cls]u64 = @splat(0);
    var end_max: u64 = 0;
    var first: u64 = 0;
    var busy_all: u64 = 0;
    var bad: u64 = 0;
    var at: u32 = 0;
    // Fine layers: per layer and op label, the GPU's busy time (each op ran alone).
    var fine_gpu: [128][n_ops]u64 = @splat(@splat(0));
    var fine_bufs: [128][n_ops]u32 = @splat(@splat(0));
    // Per layer: the tail's commit, the first trunk commit after it (the host's time the tail's GPU work has to cover),
    // the tail's busy share.
    var tail_commit: ?u64 = null;
    var tail_to_trunk: u64 = 0;
    var tail_to_trunk_n: u64 = 0;
    for (rs) |r| {
        if (r.gpu_end <= r.gpu_start or r.gpu_start == 0) {
            bad += 1;
            continue;
        }
        const cls = classOf(r.tag);
        bufs[cls] += 1;
        if (cls == 8) {
            const l: usize = @intCast(((r.tag >> 24) - 1) & 127);
            const o: usize = @min(@as(usize, @as(u8, @truncate(r.tag))), n_ops - 1);
            fine_gpu[l][o] += r.gpu_end - r.gpu_start;
            fine_bufs[l][o] += 1;
        }
        if (cls == 6) tail_commit = r.host_commit else if (cls == 1) if (tail_commit) |tc| {
            if (r.host_commit > tc) tail_to_trunk += r.host_commit - tc;
            tail_to_trunk_n += 1;
            tail_commit = null;
        };
        if (first == 0) {
            first = r.gpu_start;
            end_max = r.gpu_end;
            busy_all += r.gpu_end - r.gpu_start;
            busy[cls] += r.gpu_end - r.gpu_start;
            continue;
        }
        if (r.gpu_start > end_max) {
            const a = end_max;
            const b = r.gpu_start;
            gap[cls] += b - a;
            gap_n[cls] += 1;
            gap_max[cls] = @max(gap_max[cls], b - a);
            const c = @min(b, @max(a, r.host_commit));
            chargeHost(a, c, &by_host[cls], &at);
            c2s[cls] += b - c;
            busy_all += r.gpu_end - r.gpu_start;
            busy[cls] += r.gpu_end - r.gpu_start;
        } else if (r.gpu_end > end_max) {
            busy_all += r.gpu_end - end_max;
            busy[cls] += r.gpu_end - end_max;
        }
        end_max = @max(end_max, r.gpu_end);
    }
    const span = end_max -| first;
    log.info("glm_moe_dsa profile timeline: {d} command buffers ({d} dropped, {d} without GPU times), {d} host marks, GPU span {d:.3} s, busy {d:.3} s, idle {d:.3} s\n", .{ rs.len, dropped, bad, hm_n, s_(span), s_(busy_all), s_(span -| busy_all) });
    var w: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer w.deinit();
    for (0..n_cls) |c| {
        if (bufs[c] == 0) continue;
        w.clearRetainingCapacity();
        for (0..n_host) |h| if (by_host[c][h] >= 100_000) {
            w.writer.print(", {s} {d:.3}", .{ @tagName(@as(Host, @enumFromInt(h))), s_(by_host[c][h]) }) catch {};
        };
        log.info("glm_moe_dsa profile idle: {s}: {d} buffers, busy share {d:.3} s; idle before them {d:.3} s in {d} gaps (largest {d:.3} ms): commit to start {d:.3}{s}\n", .{ cls_names[c], bufs[c], s_(busy[c]), s_(gap[c]), gap_n[c], ms_(gap_max[c]), s_(c2s[c]), w.written() });
    }
    log.info("glm_moe_dsa profile idle: the tail's commit to the next layer's first trunk commit: {d:.3} s over {d} layers ({d:.3} ms a layer)\n", .{ s_(tail_to_trunk), tail_to_trunk_n, if (tail_to_trunk_n > 0) ms_(tail_to_trunk / tail_to_trunk_n) else 0 });
    for (0..128) |l| {
        if (!fine_set.isSet(l)) continue;
        w.clearRetainingCapacity();
        var tot_gpu: u64 = 0;
        var tot_wall: u64 = 0;
        for (0..n_ops) |o| if (fine_n[l][o] > 0 or fine_bufs[l][o] > 0) {
            tot_gpu += fine_gpu[l][o];
            tot_wall += fine_wall[l][o];
            w.writer.print(" {s} {d:.3}/{d:.3}", .{ @tagName(@as(Op, @enumFromInt(o))), ms_(fine_gpu[l][o]), ms_(fine_wall[l][o]) }) catch {};
            if (fine_bufs[l][o] > 1) w.writer.print(" ({d} bufs)", .{fine_bufs[l][o]}) catch {};
            w.writer.writeAll(";") catch {};
        };
        log.info("glm_moe_dsa profile fine layer {d}: GPU {d:.3} ms of {d:.3} ms wall; op GPU/wall ms:{s}\n", .{ l, ms_(tot_gpu), ms_(tot_wall), w.written() });
    }
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
        const tw = dsv41tl_now();
        while (true) {
            dsv41tl_counts(&k);
            if (k[1] >= @min(k[0], tl_cap) or dsv41tl_now() - tw > 5 * std.time.ns_per_s) break;
            std.Thread.yield() catch {};
        }
        summarize(tl_rows[0..@min(k[0], tl_cap)], k[2]);
        tl_on = false;
        sdk_ext.expert.stream.diag_copier_wait = null;
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
