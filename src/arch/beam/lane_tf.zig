//! DIAG, not for merge (branch diag/beam-dflash-tf): the drafter teacher-forced over a prompt that ends with a text the
//! lane served. M2 of the draft lane's acceptance-gap plan (work/beam-eval/dflash/m2/).
//!
//! With BEAM_LANE_TF_START=s in the server's environment, an installed lane's prompt pass of n rows (s + 1 < n, from
//! position 0 in one call, n at most the drafter's window less one, so its captures hold every row) runs this check
//! before its captures go to the drafter's context as usual. The drafter's context restarts with rows [0, s) of the pass.
//! Then at every position p in [s, n - 1), for every depth d of BEAM_LANE_TF_DEPTHS (default 3,7,15): the block
//! [ids[p], mask x d] at p and its greedy path, matched against ids[p + 1 ..] (the prefix that equals the text); then
//! row p joins the context. Per depth, one log line of counts: the matched prefix at every position, by draft position;
//! and the lane's rounds walked over those prefixes (from s: k = min(d, n - 1 - p) drafts, a accepted, the next round at
//! p + a + 1), in the lane receipt's counters. The context is the prompt pass's, one call: equal rates with a served
//! request over the same text clear the served path's context (rows, positions, anchor).

const std = @import("std");
const sdk = @import("sdk");
const log = sdk.log;
const graph = @import("graph.zig");
const drafter = @import("drafter.zig");
const lane_mod = @import("lane.zig");

const T = graph.T;
const max_depths = 4;

pub const Plan = struct {
    start: u32,
    depths: [max_depths]u32 = @splat(0),
    n_depths: u32 = 0,
};

/// The check's plan from the environment, or null (no BEAM_LANE_TF_START, or a word that does not parse).
pub fn plan() ?Plan {
    const sv = std.c.getenv("BEAM_LANE_TF_START") orelse return null;
    const start = std.fmt.parseInt(u32, std.mem.span(sv), 10) catch return null;
    if (start == 0) return null;
    var p: Plan = .{ .start = start };
    const ds: []const u8 = if (std.c.getenv("BEAM_LANE_TF_DEPTHS")) |d| std.mem.span(d) else "3,7,15";
    var it = std.mem.tokenizeScalar(u8, ds, ',');
    while (it.next()) |w| {
        if (p.n_depths == max_depths) return null;
        const d = std.fmt.parseInt(u32, w, 10) catch return null;
        if (d < 1 or d + 1 > drafter.max_block) return null;
        p.depths[p.n_depths] = d;
        p.n_depths += 1;
    }
    return if (p.n_depths > 0) p else null;
}

/// Whether a prompt call runs the check: the request's first call from position 0, the whole prompt in it, past the
/// start, every row in the captures.
pub fn applies(pl: Plan, n_ids: usize, start: u64, call_rows: usize, window: u32) bool {
    return start == 0 and call_rows == n_ids and n_ids > pl.start + 1 and n_ids <= window - 1;
}

/// The check over `ids` with the prompt pass's `caps` (every row). The drafter's context is restarted at the end: the
/// caller's append of the captures rebuilds it.
pub fn run(m: anytype, l: *lane_mod.Lane, pl: Plan, ids: []const u32, caps: *const drafter.Captures) !void {
    const dr = l.dr;
    const g = &l.g;
    const c = &dr.c;
    const a = m.gpa;
    const n: u32 = @intCast(ids.len);
    if (!caps.complete() or caps.rows != n) return error.BeamLaneTfCaptures;
    var xs: [drafter.max_targets]T = undefined;
    const sl = caps.slice(&xs);
    var acc: [max_depths][]u8 = undefined;
    var n_alloc: usize = 0;
    defer for (acc[0..n_alloc]) |x| a.free(x);
    for (0..pl.n_depths) |i| {
        acc[i] = try a.alloc(u8, n);
        @memset(acc[i], 0);
        n_alloc += 1;
    }
    const t0 = m.nowNs();
    defer dr.restart();
    dr.restart();
    {
        const mk = g.mark();
        defer g.resetTo(mk);
        try dr.appendContext(g, sl, 0, 0, pl.start);
    }
    var sb: [(drafter.max_block - 1) * drafter.max_top_k * drafter.max_top_k]f32 = undefined;
    var cb: [(drafter.max_block - 1) * drafter.max_top_k]u32 = undefined;
    var path: [drafter.max_block]u32 = undefined;
    var p: u32 = pl.start;
    while (p + 1 < n) : (p += 1) {
        for (pl.depths[0..pl.n_depths], 0..) |d, di| {
            const mk = g.mark();
            defer g.resetTo(mk);
            const dd = try dr.draft(g, a, &m.w, ids[p], p, d, false);
            var roots: [3]T = undefined;
            try g.evalAll(dd.roots(&roots));
            const scores = try g.hostF32(dd.scores, sb[0 .. @as(usize, d) * c.top_k * c.top_k]);
            const cand = try g.hostU32(dd.cand, cb[0 .. @as(usize, d) * c.top_k]);
            drafter.pathOf(scores, cand, d, c.top_k, path[0..d], null);
            const cmp = @min(d, n - 1 - p);
            var k: u32 = 0;
            while (k < cmp and path[k] == ids[p + 1 + k]) k += 1;
            acc[di][p] = @intCast(k);
        }
        const mk = g.mark();
        defer g.resetTo(mk);
        try dr.appendContext(g, sl, p, p, 1);
    }
    const ns = m.nowNs() - t0;
    for (pl.depths[0..pl.n_depths], 0..) |d, di| report(d, pl.start, n, acc[di], ns);
}

/// One depth's line: every position's matched prefix by draft position, and the lane's rounds walked over them.
fn report(d: u32, s: u32, n: u32, acc: []const u8, ns: u64) void {
    var every: lane_mod.Stats = .{};
    var p: u32 = s;
    var positions: u64 = 0;
    while (p + 1 < n) : (p += 1) {
        const cmp = @min(d, n - 1 - p);
        positions += 1;
        every.addPositions(cmp, acc[p]);
    }
    var walk: lane_mod.Stats = .{};
    p = s;
    while (p + 1 < n) {
        const k = @min(d, n - 1 - p);
        const got: u32 = @min(acc[p], k);
        walk.base.add(k, got);
        walk.addPositions(k, got);
        p += got + 1;
    }
    var eb: [512]u8 = undefined;
    var wb: [512]u8 = undefined;
    const r: f64 = @floatFromInt(@max(walk.base.rounds, 1));
    log.info("beam lane tf: depth={d} start={d} rows={d} positions={d} every_position={s} rounds={d} drafted={d} accepted={d} corrected={d} tokens_per_round={d:.3} rounds_by_position={s} check_s={d:.1}\n", .{
        d,                   s,                                 n,                  positions,           every.positions(&eb),
        walk.base.rounds,    walk.base.drafted,                 walk.base.accepted, walk.base.corrected, @as(f64, @floatFromInt(walk.base.generated)) / r,
        walk.positions(&wb), @as(f64, @floatFromInt(ns)) / 1e9,
    });
}
