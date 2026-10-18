//! Beam's draft lane (G3 `draft_lane`): Reflection's DFlash2 drafter (drafter.zig) over the module's own state, on the
//! shared acceptance (`sdk_ext.acceptance`) through MiMo's draft-lane seam (`mimo_v2/draft.zig`: the round's draft count,
//! the decision graph and its read, the gate, the counters).
//!
//! A round at the host's spec invariant (the trunk holds prompt + emitted; t1, the next token, at position p, not in it):
//!   1. draft: k = min(depth, the host's accepted cap, the positions left after t1) (`draftCount`); the drafter's block
//!      of depth + 1 rows at p over its context (which ends at p) and the selector's scores in one eval; the host walks
//!      the greedy path; the gate (`draft_gate` > 0) keeps the leading drafts whose cumulative drafter probability stays
//!      at or above theta (`gatedDrafts`);
//!   2. verify: [t1, d1 .. dk] in chunks of at most `experts.decode_lane_rows` rows (DSpark's shape, deepseek_v41/
//!      dspark_loop.zig), each chunk one decode-lane forward with every row's logits and the target layers' rows captured
//!      (graph.zig `forward`'s `capture`); its decision graph (`decisionGraph` at the chunk's first position) folded into
//!      the chunk's one eval; a chunk that holds a rejection (or the bonus row) ends the verify;
//!   3. commit: the rejected rows trimmed (`Module.trim`: at most a chunk's rows less one, so the rings' 17 rows of slack
//!      cover it), the committed rows' captures appended to the drafter's context (lazy: the next draft's eval runs it);
//!      [t1, the accepted drafts] and the next token go to the host.
//!
//! The rule is a construction value (`settings.mtp_acceptance`, `mtp_typical_delta`; `acceptGeneral`'s): `exact`, the
//! default (a greedy request's tokens are each verify row's argmax; a sampled one's are the target's position-keyed
//! draws, a draft kept when it is the draw: distributed as the target), or `typical` (Medusa's typical acceptance on every
//! request: lossy, an opt-in). The drafter proposes its greedy path (a point mass), or on a sampled request with
//! `draft_path: sampled` a path drawn at the target's own uniforms (drafter.zig `sampledPathOf`); `acceptance` needs no
//! q under either: each emitted token is the target's draw.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const acc = @import("../../sdk_ext.zig").acceptance;
const expert = @import("../../sdk_ext.zig").expert;
const mimo_draft = @import("../mimo_v2/draft.zig");
const settings = @import("settings.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const drafter = @import("drafter.zig");

const G = graph.G;
const T = graph.T;

/// A verify chunk's rows: the decode lane's widest call (MiMo's math: 4 rows x top-8 = 32 route ids, one route).
pub const chunk_rows: u32 = experts.decode_lane_rows;
/// A round's chunks at most (1 + the trained block's 15 drafts).
pub const max_chunks = (drafter.max_block + chunk_rows - 1) / chunk_rows;

comptime {
    if (drafter.max_block > acc.max_rows) @compileError("beam: a verify would not fit the acceptance graphs' rows");
    if (settings.max_draft_depth + 1 > drafter.max_block) @compileError("beam: a served depth past the drafter's block");
}

/// The chunks of a verify of `rows` rows: [start, end) of each, at most `chunk_rows` rows a chunk.
pub fn chunkEnd(start: u32, rows: u32) u32 {
    return @min(start + chunk_rows, rows);
}

/// The lane's counters over one request (MiMo's, plus the chunks, the gate's drops, typical's deviations and the time).
pub const Stats = struct {
    base: mimo_draft.Stats = .{},
    /// Verify chunks run (each one trunk forward).
    chunks: u64 = 0,
    /// Drafts the gate dropped before the verify.
    gated: u64 = 0,
    /// Typical: accepted drafts that are not their row's own token (the target's argmax, or its draw on a sampled
    /// request): where the emitted tokens leave what `exact` emits at the same prefix.
    deviated: u64 = 0,
    draft_ns: u64 = 0,
    verify_ns: u64 = 0,
    commit_ns: u64 = 0,
    /// By draft position i (0: the first draft): the rounds that verified a draft there, and those that accepted it.
    pos_drafted: [drafter.max_block - 1]u64 = @splat(0),
    pos_accepted: [drafter.max_block - 1]u64 = @splat(0),

    /// A round of `k` verified drafts and `accepted` accepted ones in the counters by position.
    pub fn addPositions(st: *Stats, k: u32, accepted: u32) void {
        std.debug.assert(accepted <= k and k <= st.pos_drafted.len);
        for (st.pos_drafted[0..k]) |*x| x.* += 1;
        for (st.pos_accepted[0..accepted]) |*x| x.* += 1;
    }

    /// The acceptance by draft position: `accepted/verified` at each position that a round verified, in order.
    pub fn positions(st: *const Stats, buf: []u8) []const u8 {
        var n: usize = 0;
        for (st.pos_drafted, st.pos_accepted, 0..) |d, acc_n, i| {
            if (d == 0) break;
            const w = std.fmt.bufPrint(buf[n..], "{s}{d}/{d}", .{ if (i == 0) "" else ",", acc_n, d }) catch break;
            n += w.len;
        }
        return buf[0..n];
    }

    /// The receipt's line: MiMo's (rounds, drafts, accepted and tokens a round, the stream's misses and bytes a round),
    /// then the chunks, the gate, typical's deviations, the phases' time a round.
    pub fn line(st: *const Stats, buf: []u8, name: []const u8, now: expert.Stats) []const u8 {
        const head = st.base.line(buf, name, now);
        const r: f64 = @floatFromInt(@max(st.base.rounds, 1));
        const tail = std.fmt.bufPrint(buf[head.len..], " chunks={d} chunks_per_round={d:.3} gated={d} deviated={d} draft_ms_per_round={d:.2} verify_ms_per_round={d:.2} commit_ms_per_round={d:.2}", .{
            st.chunks, @as(f64, @floatFromInt(st.chunks)) / r, st.gated, st.deviated, @as(f64, @floatFromInt(st.draft_ns)) / 1e6 / r, @as(f64, @floatFromInt(st.verify_ns)) / 1e6 / r, @as(f64, @floatFromInt(st.commit_ns)) / 1e6 / r,
        }) catch return head;
        return buf[0 .. head.len + tail.len];
    }
};

/// The host side of a round's verify: the outcome of its chunks so far (`acceptance.prefix` per chunk).
pub const Walk = struct {
    accepted: u32 = 0,
    /// Rows verified (every chunk run).
    verified: u32 = 0,
    next: ?u32 = null,

    /// Chunk [start, end) decided: `pre` its prefix over the chunk's drafts. True when the verify ends here.
    pub fn add(w: *Walk, start: u32, end: u32, pre: acc.Prefix) bool {
        std.debug.assert(start == w.verified and pre.accepted <= end - start);
        w.verified = end;
        w.accepted += pre.accepted;
        w.next = pre.next;
        return pre.next != null;
    }

    /// The rows the trunk drops (the verified rows past t1 and the accepted drafts).
    pub fn trim(w: *const Walk) u32 {
        return w.verified - (w.accepted + 1);
    }
};

/// The drafts of chunk [start, end) of a verify of `k` drafts: one a row (row r decides draft r + 1), the bonus row
/// (r = k) none.
pub fn chunkDrafts(drafts: []const u32, start: u32, end: u32) []const u32 {
    return drafts[start..@min(end, @as(u32, @intCast(drafts.len)))];
}

pub const Lane = struct {
    a: std.mem.Allocator,
    dr: *drafter.Drafter,
    rule: acc.Rule,
    /// Drafts a round at most (`settings.draft_depth`); the block is depth + 1 rows.
    depth: u32,
    gate: f32,
    /// The drafter's path on a sampled request (`settings.draft_path`).
    path: settings.DraftPath = .greedy,
    /// The decision graphs' and the draft's backend (its own scope, reset every round).
    g: G,
    stats: Stats = .{},
    /// The host armed the next request (`arm`), and the running request is armed (`begin`): its prompt pass captures
    /// the context and every serial step appends to it.
    pending: bool = false,
    active: bool = false,
    name_buf: [64]u8 = undefined,
    name_len: usize = 0,

    /// Construction: the drafter from the pack's verified sidecar, the rule and the depth (within its block).
    pub fn init(a: std.mem.Allocator, io: std.Io, dir: []const u8, cfg: *const settings.Config, s: mlx.mlx_stream) !*Lane {
        const dr = try drafter.Drafter.load(a, io, dir, &cfg.model, s);
        errdefer dr.deinit();
        if (cfg.draft_depth + 1 > dr.c.block) {
            log.err("beam: draft_depth {d} is past the drafter's block of {d} less one\n", .{ cfg.draft_depth, dr.c.block });
            return error.BeamDraftDepthOverBlock;
        }
        var g = try G.init(a, s);
        errdefer g.deinit();
        const self = try a.create(Lane);
        self.* = .{ .a = a, .dr = dr, .rule = cfg.draftRule(), .depth = cfg.draft_depth, .gate = cfg.draft_gate, .path = cfg.draft_path, .g = g };
        self.name_len = (nameOf(&self.name_buf, self.rule, self.depth, self.gate, self.path) catch unreachable).len;
        return self;
    }

    pub fn deinit(self: *Lane) void {
        self.dr.deinit();
        self.g.deinit();
        self.a.destroy(self);
    }

    /// The lane's label: `dflash2 exact depth 7`, `dflash2 typical 0.4 depth 3`, a gate named (`gate 0.5`), the sampled
    /// path named (`path sampled`).
    pub fn nameOf(buf: []u8, rule: acc.Rule, depth: u32, gate: f32, path: settings.DraftPath) ![]const u8 {
        var n = (switch (rule) {
            .exact => try std.fmt.bufPrint(buf, "dflash2 exact depth {d}", .{depth}),
            .typical => |t| try std.fmt.bufPrint(buf, "dflash2 typical {d} depth {d}", .{ t.delta, depth }),
        }).len;
        if (gate != 0) n += (try std.fmt.bufPrint(buf[n..], " gate {d}", .{gate})).len;
        if (path == .sampled) n += (try std.fmt.bufPrint(buf[n..], " path sampled", .{})).len;
        return buf[0..n];
    }

    pub fn name(self: *const Lane) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    /// The host's arming of the next request (`draft_lane.arm`): a clean request drafts.
    pub fn arm(self: *Lane, clean: bool) void {
        self.pending = clean;
    }

    /// The request's prompt pass starts: its arming taken, the counters and the drafter's context restarted.
    pub fn begin(self: *Lane) void {
        self.active = self.pending;
        self.pending = false;
        self.stats = .{};
        self.dr.restart();
    }

    /// The prompt call's (or an extend's) captures: each target layer's last window - 1 rows.
    pub fn promptCaptures(self: *const Lane) drafter.Captures {
        return .{ .targets = self.dr.c.targetSlice(), .tail = self.dr.c.window - 1 };
    }

    /// A step's or a verify chunk's captures: a reference to each target layer's rows.
    pub fn rowCaptures(self: *const Lane) drafter.Captures {
        return .{ .targets = self.dr.c.targetSlice() };
    }

    /// A forward's captures that end at position `end` appended to the context (rows [end - caps.rows, end)); with
    /// `eval`, the rings evaluated now (a prompt call: its captures go after it). Releases the captures.
    pub fn appendCaptures(self: *Lane, g: *G, caps: *drafter.Captures, end: u64, eval: bool) !void {
        defer caps.release();
        if (!caps.complete()) return error.BeamCapturesIncomplete;
        var xs: [drafter.max_targets]T = undefined;
        try self.dr.appendContext(g, caps.slice(&xs), 0, end - caps.rows, caps.rows);
        if (eval) try self.dr.evalRings(g);
    }
};

/// One round (the module's `draft_lane.round`): [t1, the accepted drafts] (owned by `a`) and the next token.
/// `oracle` (tests only, `{}` when served): drafts in place of the drafter's path (the target's own continuation).
pub fn roundWith(m: anytype, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams, oracle: anytype) !sdk.DraftRound {
    const self = m.lane.?;
    std.debug.assert(self.active);
    const dr = self.dr;
    const c = &dr.c;
    const g = &self.g;
    defer g.reset();
    const p = m.position();
    const t0 = m.nowNs();
    m.openRound(t0);
    var k = mimo_draft.draftCount(dr.ready(p), self.depth, accepted_cap, m.reserve, p);
    const params = acc.Params.of(sampling);
    var ids: [drafter.max_block]u32 = undefined;
    ids[0] = t1;
    if (k > 0) {
        if (comptime @TypeOf(oracle) == void) {
            // The draft's arrays (its rings' views among them) go once the host holds the path: before the verify and
            // the commit's appends, so the rings' next writes donate their slots (mimo_v2/kv.zig's invariant).
            const dm = g.mark();
            defer g.resetTo(dm);
            const d = try dr.draft(g, a, &m.w, t1, p, self.depth, self.gate > 0);
            var roots: [3]T = undefined;
            try g.evalAll(d.roots(&roots));
            var sb: [(drafter.max_block - 1) * drafter.max_top_k * drafter.max_top_k]f32 = undefined;
            var cb: [(drafter.max_block - 1) * drafter.max_top_k]u32 = undefined;
            const n_sc = @as(usize, self.depth) * c.top_k * c.top_k;
            const scores = try g.hostF32(d.scores, sb[0..n_sc]);
            const cand = try g.hostU32(d.cand, cb[0 .. @as(usize, self.depth) * c.top_k]);
            var path: [drafter.max_block]u32 = undefined;
            var pick: [drafter.max_block]u32 = undefined;
            // A sampled request under `draft_path: sampled` draws the path at the target's uniforms; else the greedy walk.
            const drawn: ?acc.Params = if (self.path == .sampled) acc.Params.active(params) else null;
            if (drawn) |sm|
                drafter.sampledPathOf(scores, cand, self.depth, c.top_k, sm.temperature, sm.seed, p, path[0..self.depth], pick[0..self.depth])
            else
                drafter.pathOf(scores, cand, self.depth, c.top_k, path[0..self.depth], pick[0..self.depth]);
            @memcpy(ids[1 .. k + 1], path[0..k]);
            if (d.probs) |pr| {
                var pb: [(drafter.max_block - 1) * drafter.max_top_k]f32 = undefined;
                const probs = try g.hostF32(pr, pb[0 .. @as(usize, self.depth) * c.top_k]);
                var q: [drafter.max_block]f32 = undefined;
                for (q[0..k], 0..) |*x, t| x.* = probs[t * c.top_k + pick[t]];
                const kept = mimo_draft.gatedDrafts(q[0..k], self.gate);
                self.stats.gated += k - kept;
                k = kept;
            }
        } else {
            k = @min(k, @as(u32, @intCast(oracle.len)));
            @memcpy(ids[1 .. k + 1], oracle[0..k]);
            // The context rows still have to land before the verify reads the rings.
            try dr.evalRings(g);
        }
    }
    const t_draft = m.nowNs();
    const drafts = ids[1 .. k + 1];
    const rows: u32 = k + 1;
    var caps: [max_chunks]drafter.Captures = undefined;
    var n_chunks: u32 = 0;
    defer for (caps[0..n_chunks]) |*cp| cp.release();
    var walk: Walk = .{};
    var deviated: u32 = 0;
    var start: u32 = 0;
    while (start < rows) {
        const end = chunkEnd(start, rows);
        caps[n_chunks] = self.rowCaptures();
        n_chunks += 1;
        const logits = try m.verifyChunk(ids[start..end], &caps[n_chunks - 1]);
        defer _ = mlx.mlx_array_free(logits);
        const cd = chunkDrafts(drafts, start, end);
        const dec = try mimo_draft.decisionGraph(g, logits, cd, self.rule, params, p + start);
        try g.evalAll(&.{ dec.targets, dec.flags orelse dec.targets });
        try m.flushVerify();
        const pre = try mimo_draft.readDecisionRows(g, dec, cd, end - start);
        if (self.rule == .typical and pre.accepted > 0) {
            var tb: [chunk_rows]u32 = undefined;
            const targets = try g.hostU32(dec.targets, tb[0 .. end - start]);
            for (cd[0..pre.accepted], targets[0..pre.accepted]) |x, t| deviated += @intFromBool(x != t);
        }
        const done = walk.add(start, end, pre);
        start = end;
        if (done) break;
    }
    const t_verify = m.nowNs();
    const next = walk.next orelse return error.BeamVerifyNoNextToken;
    try m.trim(walk.trim());
    // The committed rows [p, p + accepted + 1) into the drafter's context, chunk by chunk (lazy).
    var left: u32 = walk.accepted + 1;
    for (caps[0..n_chunks], 0..) |*cp, ci| {
        if (left == 0) break;
        const n = @min(left, cp.rows);
        var xs: [drafter.max_targets]T = undefined;
        try dr.appendContext(g, cp.slice(&xs), 0, p + ci * chunk_rows, n);
        left -= n;
    }
    std.debug.assert(left == 0);
    const tokens = try a.dupe(u32, ids[0 .. walk.accepted + 1]);
    const t_end = m.nowNs();
    self.stats.base.add(k, walk.accepted);
    self.stats.addPositions(k, walk.accepted);
    self.stats.chunks += n_chunks;
    self.stats.deviated += deviated;
    self.stats.draft_ns += t_draft - t0;
    self.stats.verify_ns += t_verify - t_draft;
    self.stats.commit_ns += t_end - t_verify;
    m.noteRound(walk.accepted + 1, t0);
    return .{ .tokens = tokens, .accepted = walk.accepted, .next_token = next };
}

const testing = std.testing;

test "beam lane: a verify's chunks, their drafts and the walk: early exit at the first rejection, trims within a chunk, the bonus row last" {
    try testing.expectEqual(@as(u32, 4), chunk_rows);
    try testing.expectEqual(@as(usize, 4), max_chunks);
    // Depth 7: rows [t1, d1 .. d7] = 8 rows in chunks [0, 4), [4, 8); depth 15: four chunks, the last [12, 16).
    try testing.expectEqual(@as(u32, 4), chunkEnd(0, 8));
    try testing.expectEqual(@as(u32, 8), chunkEnd(4, 8));
    try testing.expectEqual(@as(u32, 16), chunkEnd(12, 16));
    try testing.expectEqual(@as(u32, 3), chunkEnd(0, 3));
    const drafts = [_]u32{ 10, 11, 12, 13, 14, 15, 16 };
    try testing.expectEqualSlices(u32, &.{ 10, 11, 12, 13 }, chunkDrafts(&drafts, 0, 4));
    try testing.expectEqualSlices(u32, &.{ 14, 15, 16 }, chunkDrafts(&drafts, 4, 8));
    // Every draft accepted: chunk 1 holds no further row (next null), chunk 2's bonus row is the next token; no trim.
    var w: Walk = .{};
    try testing.expect(!w.add(0, 4, acc.prefix(chunkDrafts(&drafts, 0, 4), &.{ 10, 11, 12, 13 }, null)));
    try testing.expect(w.add(4, 8, acc.prefix(chunkDrafts(&drafts, 4, 8), &.{ 14, 15, 16, 99 }, null)));
    try testing.expectEqual(@as(u32, 7), w.accepted);
    try testing.expectEqual(@as(?u32, 99), w.next);
    try testing.expectEqual(@as(u32, 0), w.trim());
    // A rejection at d6 (row 5): chunk 2 ends the verify; rows 6 and 7 trimmed (8 verified, 5 + 1 kept).
    w = .{};
    _ = w.add(0, 4, acc.prefix(chunkDrafts(&drafts, 0, 4), &.{ 10, 11, 12, 13 }, null));
    try testing.expect(w.add(4, 8, acc.prefix(chunkDrafts(&drafts, 4, 8), &.{ 14, 77, 16, 99 }, null)));
    try testing.expectEqual(@as(u32, 5), w.accepted);
    try testing.expectEqual(@as(?u32, 77), w.next);
    try testing.expectEqual(@as(u32, 2), w.trim());
    // A rejection in chunk 1: chunk 2 never runs; the trim is within the chunk (at most its rows less one).
    w = .{};
    try testing.expect(w.add(0, 4, acc.prefix(chunkDrafts(&drafts, 0, 4), &.{ 55, 11, 12, 13 }, null)));
    try testing.expectEqual(@as(u32, 0), w.accepted);
    try testing.expectEqual(@as(u32, 3), w.trim());
    // Every trim of a depth-15 verify fits a chunk.
    for (0..16) |acc_n| {
        const rejected_chunk = @min(acc_n, 15) / chunk_rows;
        const verified = @min(16, (rejected_chunk + 1) * chunk_rows);
        if (acc_n < 15) try testing.expect(verified - (acc_n + 1) < chunk_rows);
    }
}

test "beam lane: the acceptance by draft position counts each round's verified and accepted drafts" {
    var st: Stats = .{};
    st.addPositions(3, 3);
    st.addPositions(3, 1);
    st.addPositions(2, 0);
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("2/3,1/3,1/2", st.positions(&buf));
    const none: Stats = .{};
    try testing.expectEqualStrings("", none.positions(&buf));
}

test "beam lane: the label names the rule, the delta, the depth, a gate and the sampled path" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("dflash2 exact depth 7", try Lane.nameOf(&buf, .exact, 7, 0, .greedy));
    try testing.expectEqualStrings("dflash2 typical 0.4 depth 3", try Lane.nameOf(&buf, .{ .typical = .{ .delta = 0.4 } }, 3, 0, .greedy));
    try testing.expectEqualStrings("dflash2 exact depth 15 gate 0.5", try Lane.nameOf(&buf, .exact, 15, 0.5, .greedy));
    try testing.expectEqualStrings("dflash2 exact depth 3 path sampled", try Lane.nameOf(&buf, .exact, 3, 0, .sampled));
    try testing.expectEqualStrings("dflash2 exact depth 3 gate 0.5 path sampled", try Lane.nameOf(&buf, .exact, 3, 0.5, .sampled));
}

test "beam lane: the receipt adds the chunks, the gate, the deviations and the phases' time to MiMo's line" {
    var st: Stats = .{};
    st.base.add(7, 5);
    st.base.add(7, 7);
    st.chunks = 4;
    st.gated = 1;
    st.deviated = 2;
    st.draft_ns = 10_000_000;
    st.verify_ns = 200_000_000;
    st.commit_ns = 2_000_000;
    var buf: [512]u8 = undefined;
    const line = st.line(&buf, "dflash2 exact depth 7", .{});
    try testing.expect(std.mem.startsWith(u8, line, "lane=\"dflash2 exact depth 7\" rounds=2 drafted=14 accepted=12 generated=14 corrected=1 accepted_per_round=6.000 tokens_per_round=7.000"));
    try testing.expect(std.mem.endsWith(u8, line, " chunks=4 chunks_per_round=2.000 gated=1 deviated=2 draft_ms_per_round=5.00 verify_ms_per_round=100.00 commit_ms_per_round=1.00"));
}
