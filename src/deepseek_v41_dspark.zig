//! The DSpark-direct decode cycle's host half (Python
//! `deepseek_v41_dspark_decode._decode_cycles` as the lane of record runs it:
//! `hybrid_install`'s causal lookup to depth 7 and one 8-row verify chunk,
//! `q3_typical_candidate`'s #475 typical rule on the greedy accept seam): the
//! verify schedule, the draft length after the confidence early stop, the
//! lookup extension, acceptance (greedy or typical), the run's counters and
//! what a cycle commits.
//!
//! One cycle: draft `k_cap` ids from `main_h` and the primary token; keep
//! `k_eff` (confidence early stop); extend a full native proposal from the
//! committed history (`Lookup`); verify `[primary, drafts]` chunk by chunk
//! (stopping once a row rejects); commit = trim the unaccepted verified rows,
//! seed the draft's stage windows with `main_hidden[0 .. accepted + 1]`, next
//! primary = the correction (or bonus), next `main_h` = `main_hidden[accepted]`.

const std = @import("std");

pub const max_block = 16;

pub const Error = error{ VerifySchedule, DraftDepth };

/// `_normalize_verify_chunks`: the construction-time schedule partitions the
/// `k_cap + 1` verify rows exactly (null = one chunk of all of them).
pub fn verifyChunks(k_cap: u32, chunks: ?[]const u32, buf: *[max_block + 1]u32) Error![]const u32 {
    if (k_cap + 1 > buf.len) return error.DraftDepth;
    const cs = chunks orelse {
        buf[0] = k_cap + 1;
        return buf[0..1];
    };
    if (cs.len == 0 or cs.len > buf.len) return error.VerifySchedule;
    var sum: u32 = 0;
    for (cs, 0..) |w, i| {
        if (w == 0) return error.VerifySchedule;
        sum += w;
        buf[i] = w;
    }
    if (sum != k_cap + 1) return error.VerifySchedule;
    return buf[0..cs.len];
}

/// `_effective_draft_len`: the leading run of drafts whose sigmoid confidence
/// (f32, compared as the Python float it becomes) clears `threshold`, at least
/// one (a cycle always verifies a draft).
pub fn effectiveDraftLen(conf: []const f32, k: u32, threshold: ?f64) u32 {
    const t = threshold orelse return k;
    if (k == 0) return 0;
    var keep: u32 = 0;
    for (conf[0..k]) |p| {
        if (@as(f64, p) >= t) keep += 1 else break;
    }
    return @max(1, keep);
}

/// A verify chunk's rows `[start, end)` of the block `[primary, d1 .. d_keff]`.
pub fn chunkRows(chunks: []const u32, i: usize, k_eff: u32) ?[2]u32 {
    var start: u32 = 0;
    for (chunks[0..i]) |w| start += w;
    if (start >= k_eff + 1) return null;
    return .{ start, @min(start + chunks[i], k_eff + 1) };
}

pub const Outcome = struct {
    accepted: u32 = 0,
    /// The target's token at the first rejecting row (or the bonus row).
    correction: ?u32 = null,
    /// Verify rows forwarded (chunks after the rejecting one never run).
    verified: u32 = 0,

    /// Rows `Model.trim` drops: the verified rows past the accepted run and its correction.
    pub fn trimRows(o: Outcome) u32 {
        return o.verified - (o.accepted + 1);
    }

    /// The `main_hidden` row the next draft starts from.
    pub fn nextMainRow(o: Outcome) u32 {
        return o.accepted;
    }
};

/// The run's counters (`DSparkDecodeStats`): depth i counts the cycles that
/// reached / accepted the draft at depth i (0-based; native depths first, then
/// the lookup's).
pub const Stats = struct {
    speculative_depth: u32 = 0,
    cycles: u32 = 0,
    verify_calls: u32 = 0,
    drafted_tokens: u32 = 0,
    accepted_drafts: u32 = 0,
    rejected_drafts: u32 = 0,
    correction_tokens: u32 = 0,
    bonus_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    drafted_by_depth: [max_block]u32 = @splat(0),
    accepted_by_depth: [max_block]u32 = @splat(0),

    pub fn acceptRate(s: *const Stats) f64 {
        return if (s.drafted_tokens == 0) 0 else @as(f64, @floatFromInt(s.accepted_drafts)) / @as(f64, @floatFromInt(s.drafted_tokens));
    }

    pub fn tokensPerCycle(s: *const Stats) f64 {
        return if (s.cycles == 0) 0 else @as(f64, @floatFromInt(s.generated_tokens)) / @as(f64, @floatFromInt(s.cycles));
    }

    pub fn acceptRateAt(s: *const Stats, depth: usize) ?f64 {
        const d = s.drafted_by_depth[depth];
        return if (d == 0) null else @as(f64, @floatFromInt(s.accepted_by_depth[depth])) / @as(f64, @floatFromInt(d));
    }

    /// A finished cycle: a correction when a reached draft was rejected, else the bonus.
    pub fn endCycle(s: *Stats, o: Outcome, k_eff: u32) void {
        s.cycles += 1;
        if (o.accepted < k_eff) {
            s.rejected_drafts += 1;
            s.correction_tokens += 1;
        } else s.bonus_tokens += 1;
    }
};

/// How a reached draft is accepted: the target's argmax (greedy, the exact
/// tier) or #475's typicality on the target row (the typical tier; the
/// correction / bonus stays the argmax either way).
pub const Acceptance = union(enum) {
    greedy,
    typical: Typical,
};

/// `accept <=> p_t(draft) > min(eps, delta * exp(-H(p_t)))`, p_t = softmax of the
/// verify row at temperature 1 (evaluated on the device; the host sees the flags).
pub const Typical = struct { delta: f32, eps: f32 = 1.0 };

/// One verify chunk's acceptance (the greedy branch of `_decode_cycles`, whose
/// compare the typical tier swaps for its flags): `target[r]` = the argmax of
/// the chunk's row r; `typical[r]` (drafted rows only; null = greedy) = the
/// typical decision. Returns true once the cycle has its correction.
pub fn acceptChunk(o: *Outcome, st: *Stats, drafts: []const u32, k_eff: u32, rows: [2]u32, target: []const u32, typical: ?[]const bool) bool {
    o.verified = rows[1];
    for (target, 0..) |t, local| {
        const depth = rows[0] + local;
        if (depth < k_eff) {
            st.drafted_by_depth[depth] += 1;
            st.drafted_tokens += 1;
            const ok = if (typical) |ty| ty[local] else t == drafts[depth];
            if (ok) {
                o.accepted += 1;
                st.accepted_by_depth[depth] += 1;
                st.accepted_drafts += 1;
                continue;
            }
        }
        o.correction = t;
        return true;
    }
    return false;
}

/// `lookup.LookupExtension`: causal extensions of a complete native proposal
/// from the committed history (the prompt, then every committed token). A
/// full proposal found in the history is extended by up to `extra_tokens`
/// tokens that followed its occurrence with the longest matching context
/// (>= `minimum_context` tokens before it equal the history's tail), the
/// earliest on ties.
pub const Lookup = struct {
    a: std.mem.Allocator,
    /// The key map and its position lists: bump-allocated, freed with the lookup.
    arena: std.heap.ArenaAllocator,
    history: std.ArrayList(u32) = .empty,
    /// 5-token key -> the positions that followed it, chronological.
    ends: std.AutoHashMapUnmanaged([key_len]u32, std.ArrayList(u32)) = .empty,
    indexed_end: usize = key_len,
    minimum_context: u32,
    extra_tokens: u32,

    pub const key_len = 5;
    const max_context = 32;

    /// Any prompt length (`LookupExtension`): positions are indexed from
    /// `key_len` on, as the history grows past it.
    /// `reserve`: the positions the request can reach (prompt + tokens), the
    /// history and the key map sized to them once; 0 lets them grow.
    pub fn init(a: std.mem.Allocator, prompt: []const u32, minimum_context: u32, extra_tokens: u32, reserve: usize) !Lookup {
        var l: Lookup = .{ .a = a, .arena = .init(a), .minimum_context = minimum_context, .extra_tokens = extra_tokens };
        errdefer l.deinit();
        try l.history.ensureTotalCapacity(a, @max(reserve, prompt.len));
        try l.ends.ensureTotalCapacity(l.arena.allocator(), @intCast(@max(reserve, prompt.len)));
        l.history.appendSliceAssumeCapacity(prompt);
        return l;
    }

    pub fn deinit(self: *Lookup) void {
        self.arena.deinit();
        self.history.deinit(self.a);
        self.* = undefined;
    }

    /// Appends committed tokens and indexes every position that now has a continuation.
    pub fn appendCommitted(self: *Lookup, tokens: []const u32) !void {
        try self.history.appendSlice(self.a, tokens);
        const h = self.history.items;
        const ka = self.arena.allocator();
        var end = self.indexed_end;
        while (end < h.len) : (end += 1) {
            const gop = try self.ends.getOrPut(ka, h[end - key_len ..][0..key_len].*);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(ka, @intCast(end));
        }
        self.indexed_end = h.len;
    }

    /// The proposal `native`, extended when it is a full key found in the history.
    pub fn extend(self: *const Lookup, native: []const u32, out: []u32) []u32 {
        @memcpy(out[0..native.len], native);
        if (native.len != key_len) return out[0..native.len];
        const list = self.ends.get(native[0..key_len].*) orelse return out[0..native.len];
        const h = self.history.items;
        var best_context: usize = self.minimum_context - 1;
        var best_end: ?usize = null;
        for (list.items) |e| {
            const end: usize = e;
            var context: usize = 0;
            while (context < max_context and end - key_len > context and h[end - key_len - 1 - context] == h[h.len - 1 - context]) context += 1;
            if (context > best_context) {
                best_context = context;
                best_end = end;
            }
        }
        const e = best_end orelse return out[0..native.len];
        const n = @min(self.extra_tokens, h.len - e);
        @memcpy(out[native.len..][0..n], h[e..][0..n]);
        return out[0 .. native.len + n];
    }
};

const testing = std.testing;

test "dsv41 dspark: the verify schedule partitions K + 1 rows, the early stop keeps a leading run" {
    var buf: [max_block + 1]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{6}, try verifyChunks(5, null, &buf));
    try testing.expectEqualSlices(u32, &.{ 2, 4 }, try verifyChunks(5, &.{ 2, 4 }, &buf));
    try testing.expectError(error.VerifySchedule, verifyChunks(5, &.{ 2, 3 }, &buf));
    try testing.expectError(error.VerifySchedule, verifyChunks(5, &.{ 6, 0 }, &buf));
    const conf = [_]f32{ 0.9, 0.7, 0.4, 0.8, 0.9 };
    try testing.expectEqual(@as(u32, 5), effectiveDraftLen(&conf, 5, null));
    try testing.expectEqual(@as(u32, 2), effectiveDraftLen(&conf, 5, 0.5));
    try testing.expectEqual(@as(u32, 1), effectiveDraftLen(&conf, 5, 0.95)); // always one draft
    // Compared as Python floats: f32(0.7) < 0.7 fails the gate an f32 compare would pass.
    try testing.expectEqual(@as(u32, 1), effectiveDraftLen(&.{ 0.9, 0.7 }, 2, 0.7));
    try testing.expectEqual(@as(u32, 0), effectiveDraftLen(&conf, 0, 0.5));
}

test "dsv41 dspark: greedy acceptance commits the accepted run and its correction, trims the rest" {
    const drafts = [_]u32{ 11, 12, 13, 14, 15 };
    // One 6-row chunk; the target agrees at depths 0 and 1, then says 99.
    var st: Stats = .{};
    var o: Outcome = .{};
    try testing.expect(acceptChunk(&o, &st, &drafts, 5, .{ 0, 6 }, &.{ 11, 12, 99, 14, 15, 7 }, null));
    try testing.expectEqual(@as(u32, 2), o.accepted);
    try testing.expectEqual(@as(?u32, 99), o.correction);
    try testing.expectEqual(@as(u32, 3), o.trimRows()); // 6 verified - (2 + 1) kept
    try testing.expectEqual(@as(u32, 2), o.nextMainRow());
    // Everything accepted: the last row is the bonus, nothing to trim.
    var all: Outcome = .{};
    try testing.expect(acceptChunk(&all, &st, &drafts, 5, .{ 0, 6 }, &.{ 11, 12, 13, 14, 15, 42 }, null));
    try testing.expectEqual(@as(u32, 5), all.accepted);
    try testing.expectEqual(@as(?u32, 42), all.correction);
    try testing.expectEqual(@as(u32, 0), all.trimRows());
    // A staged [2, 4] schedule stops after a rejection in its first chunk: 2 rows verified.
    var staged: Outcome = .{};
    var buf: [max_block + 1]u32 = undefined;
    const chunks = try verifyChunks(5, &.{ 2, 4 }, &buf);
    const r0 = chunkRows(chunks, 0, 5).?;
    try testing.expect(acceptChunk(&staged, &st, &drafts, 5, r0, &.{ 50, 60 }, null));
    try testing.expectEqual(@as(u32, 0), staged.accepted);
    try testing.expectEqual(@as(u32, 2), staged.verified);
    try testing.expectEqual(@as(u32, 1), staged.trimRows());
    // An early stop at k_eff 2 truncates the last chunk: rows [2, 3) of a [2, 4] schedule.
    try testing.expectEqual([2]u32{ 2, 3 }, chunkRows(chunks, 1, 2).?);
    try testing.expect(chunkRows(chunks, 1, 1) == null);
}

test "dsv41 dspark: typical flags decide acceptance, the argmax stays the correction and the bonus" {
    const drafts = [_]u32{ 11, 12, 13, 14, 15, 16, 17 };
    var st: Stats = .{};
    // 8 rows (5 native + 2 lookup drafts): typical at depths 0..2 though the argmax differs at 1.
    var o: Outcome = .{};
    try testing.expect(acceptChunk(&o, &st, &drafts, 7, .{ 0, 8 }, &.{ 11, 90, 13, 91, 15, 16, 17, 5 }, &.{ true, true, true, false, true, true, true }));
    try testing.expectEqual(@as(u32, 3), o.accepted);
    try testing.expectEqual(@as(?u32, 91), o.correction);
    st.endCycle(o, 7);
    // All seven typical: the bonus is the last row's argmax.
    var all: Outcome = .{};
    try testing.expect(acceptChunk(&all, &st, &drafts, 7, .{ 0, 8 }, &.{ 0, 0, 0, 0, 0, 0, 0, 5 }, &.{ true, true, true, true, true, true, true }));
    try testing.expectEqual(@as(?u32, 5), all.correction);
    st.endCycle(all, 7);
    try testing.expectEqualSlices(u32, &.{ 2, 2, 2, 2, 1, 1, 1 }, st.drafted_by_depth[0..7]);
    try testing.expectEqualSlices(u32, &.{ 2, 2, 2, 1, 1, 1, 1 }, st.accepted_by_depth[0..7]);
    try testing.expectEqual(@as(u32, 11), st.drafted_tokens);
    try testing.expectEqual(@as(u32, 10), st.accepted_drafts);
    try testing.expectEqual(@as(u32, 2), st.cycles);
    try testing.expectEqual(@as(u32, 1), st.correction_tokens);
    try testing.expectEqual(@as(u32, 1), st.bonus_tokens);
    try testing.expectEqual(@as(?f64, 0.5), st.acceptRateAt(3));
}

test "dsv41 dspark: the lookup extends a full proposal from its earliest longest-context occurrence" {
    const a = testing.allocator;
    // history: 7 8 | 1 2 3 4 5 6 9 | 8 | 1 2 3 4 5 70 71 | 9 7 8   (then the primary 8 is committed)
    var lk = try Lookup.init(a, &.{ 7, 8, 1, 2, 3, 4, 5, 6, 9, 8, 1, 2, 3, 4, 5, 70, 71, 9, 7 }, 2, 2, 0);
    defer lk.deinit();
    try lk.appendCommitted(&.{8});
    var buf: [8]u32 = undefined;
    // [1..5] follows "7 8" at its first occurrence (context 2) and "9 8" at its second (context 1).
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 9 }, lk.extend(&.{ 1, 2, 3, 4, 5 }, &buf));
    // Short proposals (the confidence stop) and unknown keys are never extended.
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, lk.extend(&.{ 1, 2, 3, 4 }, &buf));
    try testing.expectEqualSlices(u32, &.{ 2, 3, 4, 5, 9 }, lk.extend(&.{ 2, 3, 4, 5, 9 }, &buf));
    // Below minimum_context: "3 4 5 70 71" occurs once, and the token before it (2) is not the tail's (8).
    try testing.expectEqualSlices(u32, &.{ 3, 4, 5, 70, 71 }, lk.extend(&.{ 3, 4, 5, 70, 71 }, &buf));
    // A prompt shorter than the key (Python indexes from end 5 on as the history grows).
    var short = try Lookup.init(a, &.{ 1, 2, 3 }, 2, 2, 0);
    defer short.deinit();
    try short.appendCommitted(&.{ 4, 5, 6, 1, 2, 3, 4, 5 });
    try testing.expectEqual(@as(usize, 6), short.ends.count()); // ends 5..10
    // Its only earlier occurrence ends at 5 with no token before it: no context, no extension.
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5 }, short.extend(&.{ 1, 2, 3, 4, 5 }, &buf));
}

// DSV41_LOOKUP_FIXTURE=<json from the reference runtime's dump_dsv41_lookup_fixture.py>
test "dsv41 dspark: the lookup replays the lane's LookupExtension call for call" {
    const path = std.mem.span(std.c.getenv("DSV41_LOOKUP_FIXTURE") orelse return error.SkipZigTest);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, testing.allocator, .limited(64 << 20));
    defer testing.allocator.free(text);
    try replayLookup(text);
}

test "dsv41 dspark: the lookup replays one recorded scenario of the lane's LookupExtension (embedded)" {
    try replayLookup(@embedFile("fixtures/dsv41_lookup_fixture_one.json"));
}

fn replayLookup(text: []const u8) !void {
    const a = testing.allocator;
    const Op = struct { op: []const u8, tokens: []const u32 = &.{}, native: []const u32 = &.{}, out: []const u32 = &.{} };
    const Scenario = struct { seed: u32, prompt: []const u32, ops: []const Op };
    const Fixture = struct { format: []const u8, minimum_context: u32, extra_tokens: u32, scenarios: []const Scenario };
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    try testing.expectEqualStrings("mlx-serve-dsv41-lookup-fixture-v1", f.format);
    var calls: usize = 0;
    var extended: usize = 0;
    for (f.scenarios) |sc| {
        var lk = try Lookup.init(a, sc.prompt, f.minimum_context, f.extra_tokens, 0);
        defer lk.deinit();
        var buf: [16]u32 = undefined;
        for (sc.ops, 0..) |op, i| {
            if (std.mem.eql(u8, op.op, "append")) {
                try lk.appendCommitted(op.tokens);
                continue;
            }
            const got = lk.extend(op.native, &buf);
            testing.expectEqualSlices(u32, op.out, got) catch |e| {
                std.debug.print("scenario {d}, op {d} differs\n", .{ sc.seed, i });
                return e;
            };
            calls += 1;
            extended += @intFromBool(got.len > op.native.len);
        }
    }
    std.debug.print("dsv41 dspark: {d} lookup calls equal the lane's ({d} extended)\n", .{ calls, extended });
}

// DSV41_DSPARK_RECEIPT=<a lane receipt's comparison json (stats + acceptance)>
test "dsv41 dspark: the counters have the lane receipt's shape and derived rates" {
    const path = std.mem.span(std.c.getenv("DSV41_DSPARK_RECEIPT") orelse return error.SkipZigTest);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, testing.allocator, .limited(64 << 20));
    defer testing.allocator.free(text);
    try checkReceipt(text);
}

test "dsv41 dspark: the counters have a lane receipt's shape and derived rates (embedded stats)" {
    try checkReceipt(@embedFile("fixtures/dsv41_dspark_receipt_stats.json"));
}

fn checkReceipt(text: []const u8) !void {
    const a = testing.allocator;
    const RStats = struct {
        speculative_depth: u32,
        cycles: u32,
        verify_calls: u32,
        verify_chunks: []const u32,
        drafted_tokens: u32,
        accepted_drafts: u32,
        rejected_drafts: u32,
        correction_tokens: u32,
        bonus_tokens: u32,
        generated_tokens: u32,
        drafted_by_depth: []const u32,
        accepted_by_depth: []const u32,
        accept_rate: f64,
        tokens_per_cycle: f64,
        accept_rate_by_depth: []const ?f64,
    };
    const Depth = struct { depth: u32, mechanism: []const u8, reached: u32, accepted: u32 };
    const Receipt = struct { stats: RStats, acceptance: struct { cycles: u32, depths: []const Depth } };
    const parsed = try std.json.parseFromSlice(Receipt, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const r = parsed.value.stats;
    // The lane of record: 5 native + 2 lookup depths, one 8-row verify chunk per cycle.
    try testing.expectEqual(@as(u32, 7), r.speculative_depth);
    try testing.expectEqualSlices(u32, &.{8}, r.verify_chunks);
    try testing.expectEqual(@as(usize, 7), r.drafted_by_depth.len);
    for (parsed.value.acceptance.depths) |d| try testing.expectEqualStrings(if (d.depth <= 5) "native" else "lookup", d.mechanism);
    var st: Stats = .{ .speculative_depth = r.speculative_depth, .cycles = r.cycles, .verify_calls = r.verify_calls, .drafted_tokens = r.drafted_tokens, .accepted_drafts = r.accepted_drafts, .rejected_drafts = r.rejected_drafts, .correction_tokens = r.correction_tokens, .bonus_tokens = r.bonus_tokens, .generated_tokens = r.generated_tokens };
    @memcpy(st.drafted_by_depth[0..7], r.drafted_by_depth);
    @memcpy(st.accepted_by_depth[0..7], r.accepted_by_depth);
    try expectShape(&st, 7);
    try testing.expectEqual(r.verify_calls, st.cycles);
    try testing.expectApproxEqAbs(r.accept_rate, st.acceptRate(), 1e-15);
    try testing.expectApproxEqAbs(r.tokens_per_cycle, st.tokensPerCycle(), 1e-15);
    for (r.accept_rate_by_depth, 0..) |v, i| {
        if (v) |x| try testing.expectApproxEqAbs(x, st.acceptRateAt(i).?, 1e-15) else try testing.expect(st.acceptRateAt(i) == null);
    }
}

/// What every run's counters satisfy (the receipt's own relations).
fn expectShape(st: *const Stats, depth: u32) !void {
    var drafted: u32 = 0;
    var accepted: u32 = 0;
    for (st.drafted_by_depth[0..depth], st.accepted_by_depth[0..depth], 0..) |d, acc, i| {
        drafted += d;
        accepted += acc;
        try testing.expect(acc <= d);
        // A cycle reaches depth i + 1 only through an accepted depth i.
        if (i + 1 < depth) try testing.expect(st.drafted_by_depth[i + 1] <= acc);
    }
    try testing.expectEqual(st.drafted_tokens, drafted);
    try testing.expectEqual(st.accepted_drafts, accepted);
    try testing.expectEqual(st.drafted_tokens, st.accepted_drafts + st.rejected_drafts);
    try testing.expectEqual(st.rejected_drafts, st.correction_tokens);
    try testing.expectEqual(st.cycles, st.correction_tokens + st.bonus_tokens);
    // Every cycle emits its accepted drafts and one target token; the prompt's pick is the +1.
    try testing.expect(st.generated_tokens <= st.accepted_drafts + st.cycles + 1);
}

/// A synthetic target over a 23-token vocabulary: the greedy next token of a history (its last two tokens).
fn synthTarget(h: []const u32) u32 {
    const y: u32 = if (h.len > 1) h[h.len - 2] else 0;
    return (h[h.len - 1] * 7 + y * 3 + 1) % 23;
}

/// A drafter right where a hash of the position says (about 3 in 4), wrong (the target + 1) elsewhere.
fn synthDraft(h: []const u32) struct { id: u32, conf: f32 } {
    const t = synthTarget(h);
    const hit = (h.len * 2654435761) % 4 != 0;
    return .{ .id = if (hit) t else (t + 1) % 23, .conf = if (hit) 0.9 else 0.3 };
}

/// Runs the host half's cycles (draft 5, early stop, lookup to 7, verify by `schedule`, accept, commit) until `n` tokens.
fn runCycles(a: std.mem.Allocator, prompt: []const u32, n: usize, schedule: ?[]const u32, threshold: ?f64, typical: bool, st: *Stats) !std.ArrayList(u32) {
    var hist: std.ArrayList(u32) = .empty;
    defer hist.deinit(a);
    try hist.appendSlice(a, prompt);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var lk = try Lookup.init(a, prompt, 2, 2, 0);
    defer lk.deinit();
    try lk.appendCommitted(&.{});
    var buf: [max_block + 1]u32 = undefined;
    const chunks = try verifyChunks(7, schedule, &buf);
    st.speculative_depth = 7;
    while (out.items.len < n) {
        var drafts: [8]u32 = undefined;
        var conf: [5]f32 = undefined;
        var scratch: std.ArrayList(u32) = .empty;
        defer scratch.deinit(a);
        try scratch.appendSlice(a, hist.items);
        for (0..5) |i| {
            const d = synthDraft(scratch.items);
            drafts[i] = d.id;
            conf[i] = d.conf;
            try scratch.append(a, d.id);
        }
        var k_eff = effectiveDraftLen(&conf, 5, threshold);
        var ext: [8]u32 = undefined;
        const proposal = lk.extend(drafts[0..k_eff], &ext);
        @memcpy(drafts[0..proposal.len], proposal);
        k_eff = @intCast(proposal.len);
        // The target's argmax at every verify row of [primary, drafts], and typical flags that also pass odd misses.
        var rows_t: [8]u32 = undefined;
        var flags: [8]bool = undefined;
        scratch.clearRetainingCapacity();
        try scratch.appendSlice(a, hist.items);
        for (0..k_eff + 1) |r| {
            rows_t[r] = synthTarget(scratch.items);
            if (r < k_eff) {
                flags[r] = rows_t[r] == drafts[r] or (drafts[r] % 2 == 1);
                try scratch.append(a, drafts[r]);
            }
        }
        var o: Outcome = .{};
        st.verify_calls += 1;
        for (0..chunks.len) |i| {
            const rows = chunkRows(chunks, i, k_eff) orelse break;
            if (acceptChunk(&o, st, drafts[0..k_eff], k_eff, rows, rows_t[rows[0]..rows[1]], if (typical) flags[rows[0]..rows[1]] else null)) break;
        }
        st.endCycle(o, k_eff);
        // The commit: the accepted run and the correction (or the bonus), which is always the target's argmax there.
        try testing.expectEqual(rows_t[o.accepted], o.correction.?);
        try testing.expect(o.verified >= o.accepted + 1);
        const committed = o.trimRows() + o.accepted + 1;
        try testing.expectEqual(o.verified, committed);
        var c: [9]u32 = undefined;
        @memcpy(c[0..o.accepted], drafts[0..o.accepted]);
        c[o.accepted] = o.correction.?;
        try out.appendSlice(a, c[0 .. o.accepted + 1]);
        try hist.appendSlice(a, c[0 .. o.accepted + 1]);
        try lk.appendCommitted(c[0 .. o.accepted + 1]);
    }
    st.generated_tokens = @intCast(out.items.len);
    return out;
}

test "dsv41 dspark: greedy cycles over every verify schedule, early stop and the lookup commit the target's own greedy stream" {
    const a = testing.allocator;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9 };
    const n = 96;
    // The serial reference: the target's argmax, one token at a time.
    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(a);
    try ref.appendSlice(a, &prompt);
    for (0..n + 8) |_| try ref.append(a, synthTarget(ref.items));
    const schedules = [_]?[]const u32{ null, &.{ 2, 6 }, &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, &.{ 5, 3 } };
    for (schedules) |sched| for ([_]?f64{ null, 0.5 }) |thr| {
        var st: Stats = .{};
        var out = try runCycles(a, &prompt, n, sched, thr, false, &st);
        defer out.deinit(a);
        try testing.expectEqualSlices(u32, ref.items[prompt.len..][0..out.items.len], out.items);
        try expectShape(&st, 7);
        try testing.expect(st.accepted_drafts > st.cycles and st.tokensPerCycle() > 1.0);
        try testing.expectEqual(st.cycles, st.verify_calls);
    };
}

test "dsv41 dspark: typical cycles commit each accepted run's drafts and the argmax at the first unaccepted row" {
    const a = testing.allocator;
    var st: Stats = .{};
    var out = try runCycles(a, &.{ 2, 7, 1, 8, 2, 8 }, 80, &.{ 3, 5 }, null, true, &st);
    defer out.deinit(a);
    try expectShape(&st, 7);
    // Its flags pass the argmax's misses with an odd id too: more accepted than the drafter's three in four.
    try testing.expect(st.acceptRate() > 0.75);
}

test "dsv41 dspark: the counters' rates on an empty run, and the schedule's depth and width bounds" {
    const st: Stats = .{};
    try testing.expectEqual(@as(f64, 0), st.acceptRate());
    try testing.expectEqual(@as(f64, 0), st.tokensPerCycle());
    try testing.expectEqual(@as(?f64, null), st.acceptRateAt(0));
    var buf: [max_block + 1]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{max_block + 1}, try verifyChunks(max_block, null, &buf));
    try testing.expectError(error.DraftDepth, verifyChunks(max_block + 1, null, &buf));
    try testing.expectError(error.VerifySchedule, verifyChunks(4, &.{}, &buf));
    const ones: [max_block + 2]u32 = @splat(1);
    try testing.expectError(error.VerifySchedule, verifyChunks(max_block, &ones, &buf));
    // The lookup's extension stops at the history's end.
    var lk = try Lookup.init(testing.allocator, &.{ 7, 1, 2, 3, 4, 5, 6, 9 }, 1, 4, 16);
    defer lk.deinit();
    try lk.appendCommitted(&.{7});
    var out: [12]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 9, 7 }, lk.extend(&.{ 1, 2, 3, 4, 5 }, &out));
}
