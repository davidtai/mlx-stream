//! DIAG, not for merge (branch diag/beam-tf-rows): the serial model's view of every row of a prompt pass, for a
//! teacher-forced score of a served text at every position (work/beam-eval/dflash/typical-stability.md).
//!
//! With BEAM_TF_ROWS_FROM=f in the server's environment, the served prompt pass's last call (at least 2 rows, a row at
//! position f or later before its last row) is forwarded as served, with the trunk's final residual stream kept
//! (`graph.Head.hidden`). The pass's own logits are then the last row's, by the served head's ops (the row's slice, then
//! `graph.logitsOf`), so the request is served as before. Then the head goes over the call's rows at positions f and
//! later, the last row excluded (it has no next id), in tiles of `tile` rows: the final norm, lm_head and the soft cap
//! (the served logits, f32), p = their softmax at temperature 1, unfiltered. One log line a row:
//!
//!     beam tf row: n=<prompt ids> p=<position> t=<ids[p + 1]> top=<argmax> rank=<logits above t's> lp=<ln p(t)>
//!     lp1=<ln p(top)> h=<entropy of p, nats>
//!
//! and one line a call: `beam tf rows: n= call_start= rows= scored= from= s=`.

const std = @import("std");
const sdk = @import("sdk");
const log = sdk.log;
const graph = @import("graph.zig");

const T = graph.T;
const tile: usize = 128;

/// The first position the check scores (BEAM_TF_ROWS_FROM), or null: off.
pub fn from() ?u64 {
    const sv = std.c.getenv("BEAM_TF_ROWS_FROM") orelse return null;
    return std.fmt.parseInt(u64, std.mem.span(sv), 10) catch null;
}

/// Whether the call [at, end) at positions [start, start + end - at) runs the check: the prompt's last call, 2 rows or
/// more, a row at position f or later before its last row.
pub fn applies(f: u64, n_ids: usize, at: usize, end: usize, start: u32) bool {
    return end == n_ids and end - at >= 2 and @as(u64, start) + @as(u64, @intCast(end - at)) - 1 > f;
}

/// The call [at, end) of `ids` at positions [start, start + rows), forwarded as served; its rows' lines; the last row's
/// logits f32 [1, vocab], a kept handle the caller frees.
pub fn lastCall(m: anytype, ids: []const u32, at: usize, end: usize, start: u32, f: u64) !T {
    const g = &m.g;
    const c = &m.c.model;
    const rows = end - at;
    const hidden: c_int = @intCast(c.hidden_size);
    const x = try graph.forward(g, m.gpa, c, &m.w, m.routes, ids[at..end], start, &m.cache, m.src, .hidden, {}, {});
    defer g.release(x);
    const out = blk: {
        const mk = g.mark();
        defer g.resetTo(mk);
        const last = try g.slice(x, &.{ @intCast(rows - 1), 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
        break :blk g.keep(try graph.logitsOf(g, c, &m.w, last));
    };
    errdefer g.release(out);
    report(m, x, ids, at, rows, start, f) catch |e| log.warn("beam tf rows: {s}\n", .{@errorName(e)});
    return out;
}

/// The lines of the call's rows at positions f and later, the last row excluded: row r (position start + r) and its
/// next id ids[at + r + 1].
fn report(m: anytype, x: T, ids: []const u32, at: usize, rows: usize, start: u32, f: u64) !void {
    const g = &m.g;
    const c = &m.c.model;
    const hidden: c_int = @intCast(c.hidden_size);
    const hi: usize = rows - 1;
    const lo: usize = if (f > start) @min(@as(usize, @intCast(f - start)), hi) else 0;
    var tb: [tile]i32 = undefined;
    var amb: [tile]u32 = undefined;
    var rkb: [tile]u32 = undefined;
    var lpb: [tile]f32 = undefined;
    var l1b: [tile]f32 = undefined;
    var hb: [tile]f32 = undefined;
    const t0 = m.nowNs();
    var r: usize = lo;
    while (r < hi) {
        const k: usize = @min(tile, hi - r);
        const mk = g.mark();
        defer g.resetTo(mk);
        for (0..k) |i| tb[i] = @intCast(ids[at + r + i + 1]);
        const kc: c_int = @intCast(k);
        const xs = try g.slice(x, &.{ @intCast(r), 0 }, &.{ @intCast(r + k), hidden }, &.{ 1, 1 });
        const lg = try graph.logitsOf(g, c, &m.w, xs);
        const tid = try g.hostArray(std.mem.sliceAsBytes(tb[0..k]), &.{ kc, 1 }, .int32);
        const lse = try g.logsumexp(lg, -1, true);
        const lt = try g.takeAlongAxis(lg, tid, -1);
        const am = try g.argmax(lg, -1);
        const above = try g.astype(try g.sum(try g.astype(try g.greater(lg, lt), .int32), -1, false), .uint32);
        const lp = try g.reshape(try g.sub(lt, lse), &.{-1});
        const lp1 = try g.reshape(try g.sub(try g.max(lg, -1, true), lse), &.{-1});
        const logp = try g.sub(lg, lse);
        const ent = try g.neg(try g.sum(try g.mul(try g.exp(logp), logp), -1, false));
        try g.evalAll(&.{ am, above, lp, lp1, ent });
        _ = try g.hostU32(am, amb[0..k]);
        _ = try g.hostU32(above, rkb[0..k]);
        _ = try g.hostF32(lp, lpb[0..k]);
        _ = try g.hostF32(lp1, l1b[0..k]);
        _ = try g.hostF32(ent, hb[0..k]);
        for (0..k) |i| {
            const pos: u64 = @as(u64, start) + @as(u64, @intCast(r + i));
            log.info("beam tf row: n={d} p={d} t={d} top={d} rank={d} lp={d:.6} lp1={d:.6} h={d:.6}\n", .{
                ids.len, pos, tb[i], amb[i], rkb[i], @as(f64, lpb[i]), @as(f64, l1b[i]), @as(f64, hb[i]),
            });
        }
        r += k;
    }
    log.info("beam tf rows: n={d} call_start={d} rows={d} scored={d} from={d} s={d:.2}\n", .{
        ids.len, start, rows, hi - lo, @as(u64, start) + @as(u64, @intCast(lo)), @as(f64, @floatFromInt(m.nowNs() - t0)) / 1e9,
    });
}
