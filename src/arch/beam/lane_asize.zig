//! DIAG (diag/beam-dflash-asize, not for merge): three draft rules' acceptance at the same served positions, on a sampled
//! request with the point-mass path, armed by `BEAM_LANE_ASIZE=1` in the server's environment. At draft position t of a
//! round whose drafts before t were all accepted (so verify row t - 1 holds the target's own next-token distribution),
//! with p the target's filtered row there (`acceptance.filteredRows`: the request's temperature and top_p) and q =
//! softmax(the selector's row at t / T_q) over its k candidates (the predecessor: the anchor at t = 1, else the
//! point-mass path's pick at t - 1):
//!
//! - pm: p(d_t), the point-mass draft's acceptance probability (sample-then-verify);
//! - A: the sum over the candidates of min(p, q): rejection sampling's acceptance probability for a draft drawn from q
//!   (kept with min(1, p / q), the residual (p - q)+ drawn on a rejection);
//! - B: the overlap of p's and q's inverse-CDF intervals in token-id order at one uniform (`draft_path: sampled`);
//! - real: the point-mass draft's realized acceptance (its mean is pm's).
//!
//! At the request's end one line: per position the rounds counted, the rates (A at T_q 0.5, the request's temperature
//! and 1.0), the candidates' p mass, and each rule's tokens a round, 1 + a1 + a1 a2 + a1 a2 a3, from the conditional
//! rates.

const std = @import("std");
const sdk = @import("sdk");
const log = sdk.log;
const acc = @import("../../sdk_ext.zig").acceptance;
const graph = @import("graph.zig");
const drafter = @import("drafter.zig");

const G = graph.G;
const T = graph.T;

/// Draft positions counted (depth 3: the verify's one chunk of 4 rows).
pub const max_t = 3;
/// q's temperatures for A: 0.5, the request's own (slot 1, B's too), 1.0.
pub const n_temps = 3;

pub const Sums = struct {
    temperature: f32 = 0,
    top_p: f32 = 1,
    rounds: u64 = 0,
    n: [max_t]u64 = @splat(0),
    real: [max_t]u64 = @splat(0),
    pm: [max_t]f64 = @splat(0),
    a: [n_temps][max_t]f64 = @splat(@splat(0)),
    b: [max_t]f64 = @splat(0),
    mass: [max_t]f64 = @splat(0),
};

var armed_flag: ?bool = null;

/// `BEAM_LANE_ASIZE=1` in the environment (read once).
pub fn armed() bool {
    if (armed_flag) |v| return v;
    const v = if (std.c.getenv("BEAM_LANE_ASIZE")) |e| std.mem.eql(u8, std.mem.span(e), "1") else false;
    armed_flag = v;
    return v;
}

fn lessById(ids: []const u32, a: u32, b: u32) bool {
    return ids[a] < ids[b];
}

/// A round's first verify chunk: `logits` f32 [rows, vocab], `scores` [k][top_k][top_k] and `cand` [k][top_k] (the
/// selector's, on the host), `pick` the point-mass path's candidate index at each position, `accepted` the chunk's
/// accepted drafts.
pub fn observe(s: *Sums, g: *G, logits: T, sm: acc.Params, scores: []const f32, cand: []const u32, pick: []const u32, k: u32, top_k: u32, accepted: u32) !void {
    const rows: u32 = @intCast(g.shapeOf(logits).dim(0));
    const nt: usize = @min(@min(k, rows), max_t);
    if (nt == 0) return;
    const kk: usize = top_k;
    const m = g.mark();
    defer g.resetTo(m);
    var ib: [max_t * drafter.max_top_k]i32 = undefined;
    const n_i = nt * kk;
    for (ib[0..n_i], cand[0..n_i]) |*x, c| x.* = @intCast(c);
    const idx = try g.hostArray(std.mem.sliceAsBytes(ib[0..n_i]), &.{ @intCast(nt), @intCast(kk) }, .int32);
    const f = try acc.filteredRows(g, logits, sm);
    const vocab = g.shapeOf(f.p).dim(1);
    const pr = try g.slice(f.p, &.{ 0, 0 }, &.{ @intCast(nt), vocab }, &.{ 1, 1 });
    const gp = try g.takeAlongAxis(pr, idx, -1);
    const gc = try g.takeAlongAxis(try g.cumsum(pr, -1), idx, -1);
    try g.evalAll(&.{ gp, gc });
    var pb: [max_t * drafter.max_top_k]f32 = undefined;
    var cb: [max_t * drafter.max_top_k]f32 = undefined;
    const pv = try g.hostF32(gp, pb[0..n_i]);
    const cv = try g.hostF32(gc, cb[0..n_i]);
    s.temperature = sm.temperature;
    s.top_p = sm.top_p;
    s.rounds += 1;
    const tq = [n_temps]f64{ 0.5, sm.temperature, 1.0 };
    var j: usize = 0;
    for (0..nt) |t| {
        // Draft t + 1 counts when drafts 1 .. t were accepted (its verify row is the target's own distribution).
        if (accepted < t) break;
        const row = scores[(t * kk + j) * kk ..][0..kk];
        const ids = cand[t * kk ..][0..kk];
        const p = pv[t * kk ..][0..kk];
        const cum = cv[t * kk ..][0..kk];
        s.n[t] += 1;
        s.real[t] += @intFromBool(accepted > t);
        s.pm[t] += p[pick[t]];
        for (p) |x| s.mass[t] += @as(f64, x);
        var mx: f32 = row[0];
        for (row[1..]) |v| mx = @max(mx, v);
        for (tq, 0..) |temp, ti| {
            var q: [drafter.max_top_k]f64 = undefined;
            var z: f64 = 0;
            for (row, q[0..kk]) |v, *x| {
                x.* = @exp(@as(f64, v - mx) / temp);
                z += x.*;
            }
            var sa: f64 = 0;
            for (q[0..kk], p) |x, px| sa += @min(x / z, @as(f64, px));
            s.a[ti][t] += sa;
            if (ti != 1) continue;
            // B: p's and q's inverse-CDF intervals in token-id order (the target's draw: the first id whose cumulative
            // mass reaches the uniform; the draft's: the same over the candidates).
            var ord: [drafter.max_top_k]u32 = undefined;
            for (ord[0..kk], 0..) |*o, i| o.* = @intCast(i);
            std.mem.sort(u32, ord[0..kk], ids, lessById);
            var cq: f64 = 0;
            var ov: f64 = 0;
            for (ord[0..kk]) |i| {
                const lo_q = cq / z;
                cq += q[i];
                const hi_q = cq / z;
                const hi_p: f64 = cum[i];
                const lo_p: f64 = hi_p - p[i];
                ov += @max(0, @min(hi_q, hi_p) - @max(lo_q, lo_p));
            }
            s.b[t] += ov;
        }
        j = pick[t];
    }
}

fn rate(sum: f64, n: u64) f64 {
    return if (n == 0) 0 else sum / @as(f64, @floatFromInt(n));
}

/// Tokens a round from the conditional rates: 1 + a1 + a1 a2 + a1 a2 a3.
fn perRound(r: [max_t]f64) f64 {
    var tot: f64 = 1;
    var run: f64 = 1;
    for (r) |x| {
        run *= x;
        tot += run;
    }
    return tot;
}

/// The request's line (nothing without a counted round).
pub fn report(s: *const Sums, name: []const u8) void {
    if (s.rounds == 0) return;
    var buf: [2048]u8 = undefined;
    var n: usize = 0;
    var rr: [6][max_t]f64 = undefined;
    for (0..max_t) |t| {
        rr[0][t] = rate(@floatFromInt(s.real[t]), s.n[t]);
        rr[1][t] = rate(s.pm[t], s.n[t]);
        rr[2][t] = rate(s.a[0][t], s.n[t]);
        rr[3][t] = rate(s.a[1][t], s.n[t]);
        rr[4][t] = rate(s.a[2][t], s.n[t]);
        rr[5][t] = rate(s.b[t], s.n[t]);
        const w = std.fmt.bufPrint(buf[n..], " t{d}: n={d} real={d:.4} pm={d:.4} A_T0.5={d:.4} A_Treq={d:.4} A_T1.0={d:.4} B={d:.4} mass={d:.4};", .{ t + 1, s.n[t], rr[0][t], rr[1][t], rr[2][t], rr[3][t], rr[4][t], rr[5][t], rate(s.mass[t], s.n[t]) }) catch break;
        n += w.len;
    }
    log.info("beam lane asize: lane=\"{s}\" T={d:.2} top_p={d:.2} rounds={d}{s} tokens_per_round real={d:.3} pm={d:.3} A_T0.5={d:.3} A_Treq={d:.3} A_T1.0={d:.3} B={d:.3}\n", .{ name, s.temperature, s.top_p, s.rounds, buf[0..n], perRound(rr[0]), perRound(rr[1]), perRound(rr[2]), perRound(rr[3]), perRound(rr[4]), perRound(rr[5]) });
}
