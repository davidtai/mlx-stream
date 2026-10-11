//! MiMo-V2.6-Flash's MTP draft lane (Phase 2d): the arch's `DraftLane` over module-owned state, on the shared
//! acceptance module (`sdk_ext.acceptance`).
//!
//! A round at the host's spec invariant (the trunk holds prompt + emitted, t1 = the next token not in it, position P):
//!   1. draft: the trained heads chain d1..dk at the last committed position q = P - 1 (`mtp.draft`), k <=
//!      min(depth, the host's accepted cap, the reserved positions left) (depth: `settings.draft_depth`, 1..3, default
//!      the trained 3; heads past it never run); one eval brings the k ids to the host;
//!   2. verify: [t1, d1..dk] (M = 1 + k <= 1 + depth <= 4 rows) through the trunk at P.., the same decode-phase expert stream (the
//!      source's M-row path: each layer reads the union of the rows' routes once), every row's F32 logits and
//!      final-normed hidden; the acceptance graph (`decisionGraph`: argmax rows plus `typicalFlags` for a greedy
//!      request, `typicalRows` / `sampledRows` for a sampled one) folds into the verify's one eval;
//!   3. decide: `acceptance.prefix` -> a accepted drafts and the next token (a correction or the bonus);
//!   4. commit: the trunk drops the k - a rejected rows (`Module.trim`: the KV rollback), the heads' history takes the
//!      kept rows' hiddens and tokens; returns [t1, d1..da] and the next token.
//!
//! The tier is a construction value (`settings.draft_tier`, `draft_delta`) and applies to every request
//! (`acceptance.acceptGeneral`): `exact` (the control: a greedy request's emitted tokens are the verify rows' argmax, a
//! sampled request's are exact speculative sampling with point-mass drafts, distributed as its target) or `typical`
//! (delta 0.3 by default, DSV's served delta: a greedy request tests each draft on the row's T = 1 softmax with the
//! argmax correction, a sampled request on its own tempered, filtered target with the correction and bonus drawn from
//! it: Medusa's typical acceptance). Per round only the request's own sampling varies (greedy or sampled, the
//! accepted cap, the positions left): no eligibility or fallback branch.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const acc = @import("../../sdk_ext.zig").acceptance;
const expert = @import("../../sdk_ext.zig").expert;
const ops = @import("../deepseek_v41/ops.zig");
const settings = @import("settings.zig");
const mtp = @import("mtp.zig");
const tm = @import("module.zig");
/// Profile builds only (`sdk.plugin_profile`): the round's DECODE_TIMERS buckets, its record and the route trace's round.
/// In a served build every call below compiles to nothing.
const mp = @import("mtp_profile.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const drop = tm.drop;

pub const verify_rows = mtp.verify_rows;
comptime {
    _ = mp; // its tests run with the lane's (src/tests.zig imports this file)
    if (verify_rows > @import("route_trace.zig").max_call_rows) @compileError("mimo_v2: the route trace would not keep the verify's rows");
}
/// Rows a verify can roll back (every draft rejected): the sliding caches keep this many rows past their window.
pub const rollback_rows: u32 = mtp.max_drafts;
/// The final-normed tail a committed forward hands the heads' history (a prompt's last rows, a serial step's one).
pub const history_rows: u32 = mtp.ring_rows;

comptime {
    // The setting's trained depth is the heads the pack carries; the verify and rollback widths stay the trained ones.
    if (settings.trained_draft_depth != mtp.max_drafts or mtp.max_drafts > mtp.n_heads) @compileError("mimo_v2: draft depth != the trained heads");
}

/// The lane's installed draft depth: the setting, never past the trained heads (`settings.validate` refuses 0 and > 3).
pub fn depthOf(c: settings.Config) u32 {
    return @min(c.draft_depth, mtp.max_drafts);
}

/// The tier's rule, from the construction settings.
pub fn ruleOf(c: settings.Config) acc.Rule {
    return switch (c.draft_tier) {
        .exact => .exact,
        .typical => .{ .typical = .{ .delta = c.draft_delta } },
        .invalid => unreachable, // settings.validate refuses it before construction.
    };
}

/// The lane's counters over one request (reset with the request). The expert stream's counters are read at the
/// decode phase's start and at the receipt, never per round.
pub const Stats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    generated: u64 = 0,
    /// Rounds whose next token is a correction (a draft rejected).
    corrected: u64 = 0,
    stream_start: expert.Stats = .{},

    pub fn add(st: *Stats, drafted: u32, accepted: u32) void {
        st.rounds += 1;
        st.drafted += drafted;
        st.accepted += accepted;
        st.generated += accepted + 1;
        st.corrected += @intFromBool(accepted < drafted);
    }

    /// The receipt's line: rounds, drafts, accepted/round, and the stream's misses and bytes per round over the decode.
    pub fn line(st: Stats, buf: []u8, name: []const u8, now: expert.Stats) []const u8 {
        const r: f64 = @floatFromInt(@max(st.rounds, 1));
        const misses = now.expert_cache_misses -| st.stream_start.expert_cache_misses;
        const bytes = now.expert_bytes_read -| st.stream_start.expert_bytes_read;
        return std.fmt.bufPrint(buf, "lane=\"{s}\" rounds={d} drafted={d} accepted={d} generated={d} corrected={d} accepted_per_round={d:.3} tokens_per_round={d:.3} misses={d} misses_per_round={d:.2} bytes={d} bytes_per_round={d:.0}", .{
            name,                                                        st.rounds,                                               st.drafted,                                        st.accepted, st.generated, st.corrected,
            @as(f64, @floatFromInt(st.accepted)) / r,                    @as(f64, @floatFromInt(st.generated)) / r,               misses,                                            @as(f64, @floatFromInt(misses)) / r,
            bytes,                                                       @as(f64, @floatFromInt(bytes)) / r,
        }) catch buf[0..0];
    }
};

/// The verify's decision arrays: the rows' tokens (argmax, or the sampled path's draws) and the per-draft test (null:
/// the exact rule, target == draft).
pub fn Decision(comptime T: type) type {
    return struct { targets: T, flags: ?T };
}

/// The decision graph over the verify rows `row2` [1 + drafts, V] (any op backend `acceptance` takes;
/// `acceptance.acceptGeneral` is its oracle): a sampled request (`Params.active`) takes `sampledRows` (exact) or
/// `typicalRows` (typical) at `base` (t1's absolute position); a greedy one the argmax rows and, on the typical tier,
/// `typicalFlags`.
pub fn decisionGraph(g: anytype, row2: anytype, drafts: []const u32, rule: acc.Rule, params: ?acc.Params, base: u64) !Decision(@TypeOf(row2)) {
    if (acc.Params.active(params)) |sm| {
        const sr = switch (rule) {
            .exact => try acc.sampledRows(g, row2, sm, base, drafts),
            .typical => |t| try acc.typicalRows(g, row2, sm, base, drafts, t),
        };
        return .{ .targets = sr.tok, .flags = sr.accept };
    }
    const targets = try g.astype(try g.argmax(row2, -1), .uint32);
    const flags = switch (rule) {
        .exact => null,
        .typical => |t| if (drafts.len == 0) null else try acc.typicalFlags(g, row2, drafts, t),
    };
    return .{ .targets = targets, .flags = flags };
}

/// The evaluated decision on the host: the accepted count and the next token (`acceptance.prefix`).
pub fn readDecision(g: anytype, d: anytype, drafts: []const u32) !acc.Prefix {
    var tb: [acc.max_rows]u32 = undefined;
    var fb: [acc.max_rows]bool = undefined;
    const targets = try g.hostU32(d.targets, tb[0 .. drafts.len + 1]);
    const flags: ?[]const bool = if (d.flags) |f| try g.hostBool(f, fb[0..drafts.len]) else null;
    return acc.prefix(drafts, targets, flags);
}

/// A round's drafts at position `p` (t1 = x_p): the installed depth (`depthOf`), the host's accepted cap, the reserved
/// positions after t1 (the verify writes p .. p + k); none when the heads' history does not reach p.
pub fn draftCount(ready: bool, depth: u32, accepted_cap: u32, reserve: u64, p: u64) u32 {
    if (!ready) return 0;
    return @intCast(@min(@min(@as(u64, depth), accepted_cap), reserve -| (p + 1)));
}

/// The host side of one round over the decision (`acceptance.prefix`): the rows the trunk rolls back, the tokens the
/// round returns ([t1, accepted drafts]) and the next token.
pub const Commit = struct { trim: u32, emitted: u32, next: u32 };

pub fn commitOf(drafted: u32, p: acc.Prefix) Commit {
    std.debug.assert(p.accepted <= drafted);
    return .{ .trim = drafted - p.accepted, .emitted = p.accepted + 1, .next = p.next.? };
}

pub const Lane = struct {
    a: std.mem.Allocator,
    s: Stream,
    heads: mtp.Heads,
    rule: acc.Rule,
    /// Drafts per round at most (`depthOf`, construction).
    depth: u32,
    name_buf: [48]u8 = undefined,
    name_len: usize = 0,
    state: mtp.State = .{},
    /// The decision graph's backend (its own scope, reset every round).
    g: ops.MlxOps,
    stats: Stats = .{},

    /// Construction: the heads from the pack's verified sidecar, the tier's rule.
    /// `fp8`: the module's FP8 resident route (fp8res packs; null on the BF16 / affine tiers), bound by the heads' FP8
    /// projections at load.
    pub fn init(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: settings.Config, s: Stream, fp8: ?*const @import("fp8_linear.zig").Kernels) !*Lane {
        try c.validate();
        var heads = try mtp.Heads.load(a, io, dir, c.mimo, fp8);
        errdefer heads.deinit();
        var g = try ops.MlxOps.init(a, s);
        errdefer g.deinit();
        const self = try a.create(Lane);
        self.* = .{ .a = a, .s = s, .heads = heads, .rule = ruleOf(c), .depth = depthOf(c), .g = g };
        self.name_len = (nameOf(&self.name_buf, self.rule, self.depth) catch unreachable).len;
        return self;
    }

    /// The lane's label: the tier (and delta); a depth below the trained one is named (`mtp typical 0.3 depth 2`).
    pub fn nameOf(buf: []u8, rule: acc.Rule, depth: u32) ![]const u8 {
        const tier = switch (rule) {
            .exact => try std.fmt.bufPrint(buf, "mtp exact", .{}),
            .typical => |t| try std.fmt.bufPrint(buf, "mtp typical {d}", .{t.delta}),
        };
        if (depth == mtp.max_drafts) return tier;
        const tail = try std.fmt.bufPrint(buf[tier.len..], " depth {d}", .{depth});
        return buf[0 .. tier.len + tail.len];
    }

    pub fn deinit(self: *Lane) void {
        self.state.deinit();
        self.heads.deinit();
        self.g.deinit();
        self.a.destroy(self);
    }

    pub fn name(self: *const Lane) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    /// A new request: the heads' rows and history go; the counters restart.
    pub fn resetRequest(self: *Lane) void {
        self.state.deinit();
        self.stats = .{};
    }

    /// `Module.history` with the lane installed: a committed forward's final-normed tail and its tokens.
    pub fn recordHistory(m: *tm.Module, p0: u64, ids: []const u32, normed: Array) anyerror!void {
        const self = m.lane.?;
        try self.state.record(m.s, p0, normed, ids);
    }

    fn trunkOf(m: *const tm.Module) mtp.Trunk {
        return .{
            .a = m.a,
            .s = m.s,
            .embed = m.embed,
            .lm_head = m.head,
            .rope_freq = m.sliding_rope_freq,
            .value_scale = m.value_scale,
            .ones = m.norm_ones,
            .eps = m.c.rms_norm_eps,
            .heads = m.c.n_heads,
            .kv_heads = m.c.swa_n_kv_heads,
            .head_dim = m.c.head_dim,
            .v_dim = m.c.value_head_dim,
            .rope_dim = m.c.rope_head_dim,
            .window = m.c.window,
        };
    }

    /// One round (the module's `draft_lane.round`): [t1, the accepted drafts] (owned by `a`) and the next token.
    pub fn round(m: *tm.Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
        // The host runs a round only on an armed lane (`blockSize` > 0: the lane is installed) after a prompt pass:
        // invariants of the host contract, held by the lane tests.
        std.debug.assert(m.lane != null and m.pos > 0);
        const self = m.lane.?;
        const p = m.pos;
        var tt = if (comptime mp.enabled) mp.roundBegin(m.source.stream_source.stats()) else mp.now();
        try self.state.pushTokens(p, &.{t1});
        // Drafts: the installed depth, the host's cap, the reserved positions after t1; none when the heads' history
        // does not reach this position (it restarts from this round's commit).
        const k = draftCount(self.state.ready(p), self.depth, accepted_cap, m.reserve, p);
        var ids: [verify_rows]u32 = undefined;
        ids[0] = t1;
        {
            var dev: [mtp.max_drafts]Array = undefined;
            const trunk = trunkOf(m);
            try mtp.draft(&trunk, &self.heads, &self.state, p - 1, k, dev[0..k]);
            defer for (dev[0..k]) |d| drop(d);
            tt = mp.charge(.draft_build, tt);
            if (k > 0) {
                const vec = mlx.mlx_vector_array_new_data(&dev, k);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                tt = mp.charge(.draft_wait, tt);
                for (dev[0..k], ids[1 .. k + 1]) |d, *x| {
                    const ptr = mlx.mlx_array_data_uint32(d) orelse return error.MimoDraftRead;
                    x.* = ptr[0];
                }
            }
            tt = mp.charge(.draft_host, tt);
        }
        const drafts = ids[1 .. k + 1];
        mp.verifyBegin();
        const v = try m.verifyRows(ids[0 .. k + 1]);
        defer drop(v.logits);
        defer drop(v.hidden);
        tt = mp.charge(.verify, tt);
        const g = &self.g;
        defer g.reset();
        const d = try decisionGraph(g, v.logits, drafts, self.rule, acc.Params.of(sampling), p);
        const roots = [_]Array{ d.targets, d.flags orelse d.targets, v.hidden };
        try g.evalAll(&roots);
        tt = mp.charge(.verify_eval, tt);
        try m.flushVerify();
        const pre = try readDecision(g, d, drafts);
        const plan = commitOf(k, pre);
        const kept = pre.accepted;
        tt = mp.charge(.accept, tt);
        mp.verifyEnd();
        try m.trim(plan.trim);
        tt = mp.charge(.trim, tt);
        const committed = try tm.slice2(v.hidden, 0, @intCast(kept + 1), 0, @intCast(m.c.hidden_size), m.s);
        defer drop(committed);
        try self.state.recordHidden(m.s, p, committed);
        try self.state.pushTokens(p + 1, drafts[0..kept]);
        const tokens = try a.dupe(u32, ids[0..plan.emitted]);
        self.stats.add(k, kept);
        if (comptime mp.enabled) {
            _ = mp.charge(.tail, tt);
            mp.roundEnd(m.source.stream_source.stats(), k, kept);
            m.traceRound(kept, tokens);
        }
        return .{ .tokens = tokens, .accepted = kept, .next_token = plan.next };
    }
};

const testing = std.testing;

fn cpuStream() Stream {
    return mlx.mlx_default_cpu_stream_new();
}

/// Seeded verify rows: each row a leader (spread per set), drafts leader / runner / random as the module's tests.
fn rows(seed: u64, comptime w: usize, comptime v: usize, lg: *[w * v]f32, drafts: *[w - 1]u32, spread: f32) void {
    var rng: std.Random.DefaultPrng = .init(seed);
    for (lg) |*x| x.* = rng.random().floatNorm(f32) * spread;
    for (0..w) |r| {
        const lead = rng.random().uintLessThan(u32, @as(u32, v));
        lg[r * v + lead] += 1.0 + 6.0 * rng.random().float(f32);
        if (r < w - 1) drafts[r] = switch (rng.random().uintLessThan(u32, 4)) {
            0, 1 => lead,
            2 => (lead + 1) % @as(u32, v),
            else => rng.random().uintLessThan(u32, @as(u32, v)),
        };
    }
}

test "mimo draft: the lane's decision graph agrees with acceptance.accept on synthetic verify rows (exact, typical 0.3, typical 0.5; greedy, seeded sampled and card sampled; MIMO_MTP_MLX, guarded window, MLX CPU stream)" {
    try mtp.requireMlxWindow();
    const a = testing.allocator;
    var g = try ops.MlxOps.init(a, cpuStream());
    defer g.deinit();
    const W = verify_rows;
    const V = 64;
    const rules = [_]acc.Rule{ .exact, .{ .typical = .{ .delta = 0.3 } }, .{ .typical = .{ .delta = 0.5 } } };
    const requests = [_]?acc.Params{ null, .{ .temperature = 0.8, .seed = 11 }, .{ .temperature = 0.7, .top_k = 20, .top_p = 0.9, .min_p = 0.02, .seed = 78 }, .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 } };
    var totals: [rules.len][requests.len]u32 = @splat(@splat(0));
    for (0..24) |set| {
        var lg: [W * V]f32 = undefined;
        var drafts: [W - 1]u32 = undefined;
        rows(0x51_7a + set, W, V, &lg, &drafts, 0.4 + 0.15 * @as(f32, @floatFromInt(set % 8)));
        for ([_]usize{ 0, 1, 3 }) |k| {
            const base: u64 = 1000 + set * 7;
            for (rules, 0..) |rule, ri| for (requests, 0..) |req, qi| {
                defer g.reset();
                const row2 = try g.hostArray(std.mem.sliceAsBytes(lg[0 .. (k + 1) * V]), &.{ @as(c_int, @intCast(k + 1)), V }, .float32);
                const d = try decisionGraph(&g, row2, drafts[0..k], rule, req, base);
                try g.evalAll(&.{ d.targets, d.flags orelse d.targets });
                const got = try readDecision(&g, d, drafts[0..k]);
                const want = acc.acceptGeneral(rule, req, lg[0 .. (k + 1) * V], drafts[0..k], null, base);
                testing.expectEqual(want.n_accepted, got.accepted) catch |e| {
                    std.debug.print("set {d} k {d} rule {d} req {d}\n", .{ set, k, ri, qi });
                    return e;
                };
                try testing.expectEqual(want.next_token, got.next.?);
                totals[ri][qi] += got.accepted;
            };
        }
    }
    // Greedy requests: typical 0.3 >= typical 0.5 >= exact. Sampled requests take the tier too: typical 0.3 >= 0.5
    // (the same draws, a lower floor), and typical is not exact speculative sampling.
    try testing.expect(totals[1][0] >= totals[2][0] and totals[2][0] >= totals[0][0] and totals[1][0] > totals[0][0]);
    for (1..requests.len) |qi| try testing.expect(totals[1][qi] >= totals[2][qi] and totals[1][qi] != totals[0][qi]);
}

test "mimo draft: settings choose the rule once (exact control, typical at the setting's delta)" {
    var c: settings.Config = .{};
    try testing.expectEqual(acc.Rule{ .typical = .{ .delta = 0.5 } }, ruleOf(c));
    c.draft_delta = 0.3;
    try testing.expectEqual(acc.Rule{ .typical = .{ .delta = 0.3 } }, ruleOf(c));
    c.draft_tier = .exact;
    try testing.expectEqual(acc.Rule.exact, ruleOf(c));
    c.draft_tier = .typical;
    c.draft_delta = 0.5;
    try testing.expectEqual(acc.Rule{ .typical = .{ .delta = 0.5 } }, ruleOf(c));
}

test "mimo draft: lane stats count rounds, drafts, accepted, generated and corrections; the receipt reads the stream's deltas per round" {
    var st: Stats = .{ .stream_start = .{ .expert_cache_misses = 100, .expert_bytes_read = 1_000 } };
    st.add(3, 3);
    st.add(3, 1);
    st.add(0, 0);
    st.add(2, 0);
    try testing.expectEqual(@as(u64, 4), st.rounds);
    try testing.expectEqual(@as(u64, 8), st.drafted);
    try testing.expectEqual(@as(u64, 4), st.accepted);
    try testing.expectEqual(@as(u64, 8), st.generated);
    try testing.expectEqual(@as(u64, 2), st.corrected);
    var buf: [512]u8 = undefined;
    const line = st.line(&buf, "mtp typical 0.3", .{ .expert_cache_misses = 140, .expert_bytes_read = 9_000 });
    try testing.expectEqualStrings("lane=\"mtp typical 0.3\" rounds=4 drafted=8 accepted=4 generated=8 corrected=2 accepted_per_round=1.000 tokens_per_round=2.000 misses=40 misses_per_round=10.00 bytes=8000 bytes_per_round=2000", line);
}

test "mimo draft: the round's host plan against acceptance.accept on synthetic rows: drafts per round, rollback rows, emitted tokens and the next token (exact, typical 0.3, typical 0.5; greedy, seeded sampled and card sampled; host)" {
    // Drafts: depth 3, the host's cap, the reserve after t1, none without history.
    try testing.expectEqual(@as(u32, 3), draftCount(true, 3, std.math.maxInt(u32), 1000, 10));
    try testing.expectEqual(@as(u32, 1), draftCount(true, 3, 1, 1000, 10));
    try testing.expectEqual(@as(u32, 2), draftCount(true, 3, 9, 13, 10));
    try testing.expectEqual(@as(u32, 0), draftCount(true, 3, 9, 11, 10));
    try testing.expectEqual(@as(u32, 0), draftCount(false, 3, 9, 1000, 10));
    // The plan from the module's reference round (the device graph's oracle, `acceptance.accept`) on seeded rows.
    const W = verify_rows;
    const V = 64;
    const rules = [_]acc.Rule{ .exact, .{ .typical = .{ .delta = 0.3 } }, .{ .typical = .{ .delta = 0.5 } } };
    const requests = [_]?acc.Params{ null, .{ .temperature = 0.8, .seed = 11 }, .{ .temperature = 0.7, .top_k = 20, .top_p = 0.9, .min_p = 0.02, .seed = 78 }, .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 } };
    var totals: [rules.len][requests.len]u32 = @splat(@splat(0));
    for (0..24) |set| {
        var lg: [W * V]f32 = undefined;
        var drafts: [W - 1]u32 = undefined;
        rows(0x51_7a + set, W, V, &lg, &drafts, 0.4 + 0.15 * @as(f32, @floatFromInt(set % 8)));
        for ([_]u32{ 0, 1, 3 }) |k| for (rules, 0..) |rule, ri| for (requests, 0..) |req, qi| {
            const r = acc.acceptGeneral(rule, req, lg[0 .. (k + 1) * V], drafts[0..k], null, 1000 + set * 7);
            const plan = commitOf(k, .{ .accepted = r.n_accepted, .next = r.next_token });
            try testing.expectEqual(k - r.n_accepted, plan.trim);
            try testing.expectEqual(r.n_accepted + 1, plan.emitted);
            try testing.expectEqual(r.next_token, plan.next);
            try testing.expect(plan.trim <= rollback_rows and plan.emitted <= verify_rows);
            totals[ri][qi] += r.n_accepted;
        };
    }
    // Greedy requests: typical 0.3 >= typical 0.5 >= exact. Sampled requests take the tier too: typical 0.3 >= 0.5
    // (the same draws, a lower floor), and typical is not exact speculative sampling.
    try testing.expect(totals[1][0] >= totals[2][0] and totals[2][0] >= totals[0][0] and totals[1][0] > totals[0][0]);
    for (1..requests.len) |qi| try testing.expect(totals[1][qi] >= totals[2][qi] and totals[1][qi] != totals[0][qi]);
    std.debug.print("MIMO_MTP_ACCEPT greedy exact={d} typical0.3={d} typical0.5={d} (of 24 sets x drafts 0+1+3)\n", .{ totals[0][0], totals[1][0], totals[2][0] });
}

test "mimo draft: depth 3 (the default) keeps today's plan: the drafts per round equal the trained-depth count for every cap, reserve and history state, and the lane's label is unchanged" {
    const c: settings.Config = .{};
    try testing.expectEqual(@as(u32, 3), depthOf(c));
    const today = struct {
        // The count before `draft_depth` (8c48acd): min(3, the host's cap, the reserve after t1); none without history.
        fn f(ready: bool, accepted_cap: u32, reserve: u64, p: u64) u32 {
            if (!ready) return 0;
            return @intCast(@min(@min(@as(u64, 3), accepted_cap), reserve -| (p + 1)));
        }
    }.f;
    for ([_]bool{ true, false }) |ready| for ([_]u32{ 0, 1, 2, 3, 4, 15, std.math.maxInt(u32) }) |cap| for ([_]u64{ 0, 10, 11, 12, 13, 14, 15, 1000 }) |reserve| {
        try testing.expectEqual(today(ready, cap, reserve, 10), draftCount(ready, depthOf(c), cap, reserve, 10));
    };
    var buf: [48]u8 = undefined;
    try testing.expectEqualStrings("mtp typical 0.5", try Lane.nameOf(&buf, ruleOf(c), depthOf(c)));
    const at03: settings.Config = .{ .draft_delta = 0.3 };
    try testing.expectEqualStrings("mtp typical 0.3", try Lane.nameOf(&buf, ruleOf(at03), depthOf(at03)));
    try testing.expectEqualStrings("mtp exact", try Lane.nameOf(&buf, .exact, 3));
    try testing.expectEqualStrings("mtp typical 0.3 depth 1", try Lane.nameOf(&buf, ruleOf(at03), 1));
    try testing.expectEqualStrings("mtp exact depth 2", try Lane.nameOf(&buf, .exact, 2));
}

test "mimo draft: depth 1 and 2 draft at most depth per round (under the cap and reserve), verify 1 + depth rows, and the acceptance round's plan stays inside them (exact, typical 0.3, typical 0.5, seeded sampled; host)" {
    const V = 64;
    const W = verify_rows;
    const rules = [_]acc.Rule{ .exact, .{ .typical = .{ .delta = 0.3 } }, .{ .typical = .{ .delta = 0.5 } } };
    const requests = [_]?acc.Params{ null, .{ .temperature = 0.8, .seed = 11 }, .{ .temperature = 0.7, .top_k = 20, .top_p = 0.9, .min_p = 0.02, .seed = 78 } };
    for ([_]u32{ 1, 2, 3 }) |depth| {
        var c: settings.Config = .{ .draft_depth = depth };
        try c.validate();
        try testing.expectEqual(depth, depthOf(c));
        // The depth bounds the round; the host's cap and the reserve still bound it below the depth.
        try testing.expectEqual(depth, draftCount(true, depth, std.math.maxInt(u32), 1000, 10));
        try testing.expectEqual(@as(u32, @min(depth, 1)), draftCount(true, depth, 1, 1000, 10));
        try testing.expectEqual(@as(u32, @min(depth, 2)), draftCount(true, depth, 9, 13, 10));
        try testing.expectEqual(@as(u32, 0), draftCount(true, depth, 9, 11, 10));
        try testing.expectEqual(@as(u32, 0), draftCount(false, depth, 9, 1000, 10));
        var accepted: [rules.len]u32 = @splat(0);
        for (0..24) |set| {
            var lg: [W * V]f32 = undefined;
            var drafts: [W - 1]u32 = undefined;
            rows(0x51_7a + set, W, V, &lg, &drafts, 0.4 + 0.15 * @as(f32, @floatFromInt(set % 8)));
            const k = draftCount(true, depth, std.math.maxInt(u32), 1000, 10);
            // The round's verify: [t1, d1..dk] = 1 + depth rows of the trained four.
            try testing.expectEqual(depth + 1, k + 1);
            for (rules, 0..) |rule, ri| for (requests) |req| {
                const r = acc.accept(rule, req, lg[0 .. (k + 1) * V], drafts[0..k], null, 1000 + set * 7);
                const plan = commitOf(k, .{ .accepted = r.n_accepted, .next = r.next_token });
                try testing.expect(r.n_accepted <= depth);
                try testing.expectEqual(k - r.n_accepted, plan.trim);
                try testing.expect(plan.trim <= depth and plan.trim <= rollback_rows);
                try testing.expectEqual(r.n_accepted + 1, plan.emitted);
                try testing.expect(plan.emitted <= depth + 1);
                try testing.expectEqual(r.next_token, plan.next);
                if (req == null) accepted[ri] += r.n_accepted;
                // A shallower greedy round is the deeper round's prefix: the same rows decide the same first drafts.
                if (depth < 3 and req == null) {
                    const deep = acc.accept(rule, req, lg[0 .. 4 * V], drafts[0..3], null, 1000 + set * 7);
                    if (deep.n_accepted < depth) {
                        try testing.expectEqual(deep.n_accepted, r.n_accepted);
                        try testing.expectEqual(deep.next_token, r.next_token);
                    } else try testing.expectEqual(depth, r.n_accepted);
                }
            };
        }
        try testing.expect(accepted[1] >= accepted[2] and accepted[2] >= accepted[0]);
        std.debug.print("MIMO_MTP_DEPTH depth={d} greedy accepted exact={d} typical0.3={d} typical0.5={d} (of 24 sets)\n", .{ depth, accepted[0], accepted[1], accepted[2] });
    }
}
