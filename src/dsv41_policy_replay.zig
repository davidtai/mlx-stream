//! #22 / #23 pricing on the host: a native residency-policy trace (`dsv41-policytrace-v1`, lane B's profile v3
//! POLICY_TRACE: <receipt>.policytrace.bin, u32 words) replayed at a chosen decode rows vector through
//!   - the shipped policy (`expert_policy.LayerPolicy`, every recorded call re-run; the grow takes the vector's rows),
//!   - LRU, and
//!   - Belady (offline: evict the resident whose next use is farthest; a bound, it needs the future),
//! the last two from the shipped policy's residents at the grow over the same decode calls. The rows vectors: the trace's
//! own (uniform), prompt_stats (`arm.DecodeRows` over the replayed prompt counts) and decode_first16 (the uniform run's
//! own misses over cycles 2..17, the m / r^3 marginal). Prints misses per cycle per layer, one table per quantity.
//! CPU only; no MLX. Run: DSV41_POLICY_TRACE=<.policytrace.bin> with the dsv41 test binary.
const std = @import("std");
const expert_policy = @import("sdk").expert.policy;
const arm = @import("deepseek_v41_arm.zig");

/// The record kinds of dsv41-policytrace-v1 (lane B's `dsv41_policy_trace.Kind`).
pub const Kind = enum(u8) { init = 0, seed = 1, plan = 2, invalidate = 3, admit = 4, grow = 5, forget = 6, mark = 7 };

pub const max_layers = 64;
/// decode_first16 reads cycles first_lo ..= first_hi (cycle 1 is the first-touch burst).
pub const first_lo: u32 = 2;
pub const first_hi: u32 = 17;

pub const Layer = struct {
    /// The trace's prompt rows (init) and decode rows (grow).
    prompt_rows: u32 = 0,
    decode_rows: u32 = 0,
    shipped: u64 = 0,
    lru: u64 = 0,
    belady: u64 = 0,
    /// Shipped misses over cycles first_lo ..= first_hi (decode_first16's input).
    first: u64 = 0,
    /// Distinct decode experts not resident at the grow: every policy misses them (the compulsory floor).
    compulsory: u64 = 0,
};

/// One phase's plan totals, as the stream counts them (lane B's `PhaseTotals`).
pub const PhaseTotals = struct { plans: u64 = 0, hits: u64 = 0, misses: u64 = 0, evictions: u64 = 0, persistent_loads: u64 = 0, transient_loads: u64 = 0 };

pub const Result = struct {
    layers: []Layer,
    n_layers: u32,
    n_experts: u32,
    /// Decode cycles seen (distinct cycle values of decode plans).
    cycles: u32,
    /// Per layer the prompt counts at the grow (`LayerPolicy.prefill_freq`), n_layers x n_experts.
    prompt_counts: []u32,
    admit_mismatches: u32 = 0,
    map_mismatches: u32 = 0,
    /// The shipped policy's plans per segment: construction (before the prompt's mark), the prompt, the decode.
    construction: PhaseTotals = .{},
    prefill: PhaseTotals = .{},
    decode: PhaseTotals = .{},

    pub fn deinit(r: *Result, a: std.mem.Allocator) void {
        a.free(r.layers);
        a.free(r.prompt_counts);
    }

    pub fn total(r: *const Result, comptime field: []const u8) u64 {
        var n: u64 = 0;
        for (r.layers[0..r.n_layers]) |l| n += @field(l, field);
        return n;
    }
};

fn unpack(ws: []const u32, n: usize, out: []u16) []u16 {
    for (0..n) |k| out[k] = @truncate(ws[k / 2] >> @intCast(16 * (k % 2)));
    return out[0..n];
}

/// One decode call of a layer: its distinct experts.
const Call = []const u16;

/// The trace replayed with each layer's grow at `rows[l]` (null: the trace's own); LRU and Belady from the grow's
/// residents over the same calls.
pub fn run(a: std.mem.Allocator, ws: []const u32, rows: ?[]const u32) !Result {
    var pols: [max_layers]?expert_policy.LayerPolicy = @splat(null);
    defer for (&pols) |*p| if (p.*) |*x| x.deinit(a);
    var calls: [max_layers]std.ArrayList(Call) = @splat(.empty);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    defer for (&calls) |*c| c.deinit(a);
    var start: [max_layers][]const u16 = @splat(&.{});
    var cap: [max_layers]u32 = @splat(0);
    const layers = try a.alloc(Layer, max_layers);
    errdefer a.free(layers);
    @memset(layers, .{});
    var res: Result = .{ .layers = layers, .n_layers = 0, .n_experts = 0, .cycles = 0, .prompt_counts = &.{} };
    errdefer if (res.prompt_counts.len > 0) a.free(res.prompt_counts);
    const ids = try a.alloc(u16, 1 << 17);
    defer a.free(ids);
    var plan: expert_policy.Plan = .{};
    var seen_cycles = std.AutoHashMap(u32, void).init(a);
    defer seen_cycles.deinit();
    var prompt_started = false;
    var i: usize = 0;
    while (i + 2 <= ws.len) {
        const kind: Kind = std.enums.fromInt(Kind, ws[i] >> 24) orelse return error.TraceKind;
        const layer = ws[i] & 0xFFFFFF;
        const n = ws[i + 1];
        if (i + 2 + n > ws.len) return error.TraceTruncated;
        const p = ws[i + 2 ..][0..n];
        i += 2 + n;
        if (kind == .mark) {
            prompt_started = true;
            continue;
        }
        if (layer >= max_layers) return error.TraceLayer;
        res.n_layers = @max(res.n_layers, layer + 1);
        if (kind == .init) {
            if (pols[layer]) |*x| x.deinit(a);
            pols[layer] = try expert_policy.LayerPolicy.init(a, p[0], p[1]);
            res.n_experts = p[0];
            layers[layer].prompt_rows = p[1];
            continue;
        }
        const pol = &(pols[layer] orelse return error.TraceNoInit);
        switch (kind) {
            .init, .mark => unreachable,
            .seed => pol.prepareSeed(unpack(p[1..], p[0], ids)),
            .plan => {
                const n_ids = p[3];
                const n_held = p[4];
                const id_words = (n_ids + 1) / 2;
                const phase: expert_policy.Phase = @enumFromInt(p[0]);
                const call_ids = unpack(p[5..], n_ids, ids);
                // Held slots past a smaller capacity hold nothing in the counterfactual.
                var held: [expert_policy.max_route_ids * 8]u32 = undefined;
                var nh: usize = 0;
                for (p[5 + id_words ..][0..n_held]) |s| if (s < pol.capacity and nh < held.len) {
                    held[nh] = s;
                    nh += 1;
                };
                pol.planWith(call_ids, phase, &plan, .{ .transient_base = p[1], .held = held[0..nh] });
                const pt = if (phase == .decode) &res.decode else if (prompt_started) &res.prefill else &res.construction;
                pt.plans += 1;
                pt.hits += plan.n_hits;
                pt.misses += plan.n_misses;
                pt.evictions += plan.n_evictions;
                pt.persistent_loads += plan.n_persistent;
                pt.transient_loads += plan.n_loads - plan.n_persistent;
                if (phase == .decode) {
                    const cycle = p[2];
                    try seen_cycles.put(cycle, {});
                    layers[layer].shipped += plan.n_misses;
                    if (cycle >= first_lo and cycle <= first_hi) layers[layer].first += plan.n_misses;
                    try calls[layer].append(a, try distinct(aa, call_ids));
                }
            },
            .invalidate => pol.invalidate(@intCast(p[0])),
            .admit => {
                const n_in = p[0];
                const in_words = (n_in + 1) / 2;
                const n_ad = p[2];
                var out: [expert_policy.max_route_ids * 4]expert_policy.LayerPolicy.ReadAhead = undefined;
                const got = pol.admitReadAhead(unpack(p[3..], n_in, ids), out[0..@min(p[1], out.len)]);
                const want = p[3 + in_words ..][0 .. 2 * n_ad];
                var same = got.len == n_ad;
                if (same) for (got, 0..) |g, k| {
                    same = same and g.expert == want[2 * k] and g.slot == want[2 * k + 1];
                };
                res.admit_mismatches += @intFromBool(!same);
            },
            .grow => {
                layers[layer].decode_rows = p[0];
                const c = if (rows) |r| r[layer] else p[0];
                try pol.grow(c);
                cap[layer] = c;
                if (rows == null) {
                    const map = unpack(p[2..], p[1], ids);
                    res.map_mismatches += @intFromBool(!std.mem.eql(u16, map, pol.slot_to_expert[0..map.len]));
                }
                var st: std.ArrayList(u16) = .empty;
                for (pol.slot_to_expert[0..pol.capacity]) |e| if (e != expert_policy.no_expert) try st.append(aa, e);
                start[layer] = st.items;
                if (res.prompt_counts.len == 0) {
                    res.prompt_counts = try a.alloc(u32, max_layers * @as(usize, pol.n_experts));
                    @memset(res.prompt_counts, 0);
                }
                @memcpy(res.prompt_counts[layer * pol.n_experts ..][0..pol.n_experts], pol.prefill_freq);
            },
            .forget => _ = pol.forgetAll(),
        }
    }
    res.cycles = seen_cycles.count();
    for (0..res.n_layers) |l| {
        layers[l].lru = try lruMisses(a, start[l], calls[l].items, cap[l], res.n_experts);
        layers[l].belady = try beladyMisses(a, start[l], calls[l].items, cap[l], res.n_experts);
        layers[l].compulsory = try compulsoryMisses(a, start[l], calls[l].items, res.n_experts);
    }
    return res;
}

fn distinct(a: std.mem.Allocator, xs: []const u16) ![]const u16 {
    var out: std.ArrayList(u16) = .empty;
    for (xs) |x| if (std.mem.indexOfScalar(u16, out.items, x) == null) try out.append(a, x);
    return out.items;
}

/// LRU over set requests: each call's experts are used together (missing ones loaded), the least recently used
/// resident outside the call evicted when full.
pub fn lruMisses(a: std.mem.Allocator, start: []const u16, calls: []const Call, cap: u32, n_experts: u32) !u64 {
    const stamp = try a.alloc(u64, n_experts);
    defer a.free(stamp);
    const res = try a.alloc(bool, n_experts);
    defer a.free(res);
    @memset(stamp, 0);
    @memset(res, false);
    var occ: u32 = 0;
    for (start) |e| {
        res[e] = true;
        occ += 1;
    }
    var clock: u64 = 1;
    var misses: u64 = 0;
    for (calls) |c| {
        clock += 1;
        for (c) |e| if (!res[e]) {
            misses += 1;
            if (occ >= cap) {
                var victim: ?u16 = null;
                for (0..n_experts) |x| if (res[x] and stamp[x] != clock and std.mem.indexOfScalar(u16, c, @intCast(x)) == null) {
                    if (victim == null or stamp[x] < stamp[victim.?]) victim = @intCast(x);
                };
                res[victim orelse return error.CallWiderThanRows] = false;
                occ -= 1;
            }
            res[e] = true;
            occ += 1;
        };
        for (c) |e| stamp[e] = clock;
    }
    return misses;
}

/// Belady over set requests: the resident outside the call whose next use is farthest (never: farthest of all;
/// ties by expert id) is evicted.
pub fn beladyMisses(a: std.mem.Allocator, start: []const u16, calls: []const Call, cap: u32, n_experts: u32) !u64 {
    const res = try a.alloc(bool, n_experts);
    defer a.free(res);
    @memset(res, false);
    var occ: u32 = 0;
    for (start) |e| {
        res[e] = true;
        occ += 1;
    }
    var misses: u64 = 0;
    for (calls, 0..) |c, t| {
        for (c) |e| if (!res[e]) {
            misses += 1;
            if (occ >= cap) {
                var victim: ?u16 = null;
                var far: usize = 0;
                for (0..n_experts) |x| if (res[x] and std.mem.indexOfScalar(u16, c, @intCast(x)) == null) {
                    const nu = nextUse(calls, t + 1, @intCast(x));
                    if (victim == null or nu > far) {
                        victim = @intCast(x);
                        far = nu;
                    }
                };
                res[victim orelse return error.CallWiderThanRows] = false;
                occ -= 1;
            }
            res[e] = true;
            occ += 1;
        };
    }
    return misses;
}

/// Distinct experts the calls use that were not resident at the start.
pub fn compulsoryMisses(a: std.mem.Allocator, start: []const u16, calls: []const Call, n_experts: u32) !u64 {
    const seen = try a.alloc(bool, n_experts);
    defer a.free(seen);
    @memset(seen, false);
    for (start) |e| seen[e] = true;
    var n: u64 = 0;
    for (calls) |c| for (c) |e| if (!seen[e]) {
        seen[e] = true;
        n += 1;
    };
    return n;
}

fn nextUse(calls: []const Call, from: usize, e: u16) usize {
    for (calls[from..], from..) |c, t| if (std.mem.indexOfScalar(u16, c, e) != null) return t;
    return std.math.maxInt(usize);
}

/// decode_first16's rows: the bounded top-K of `arm.DecodeRows` with each layer's row r worth m_l / r^3, m_l = the
/// uniform run's misses over cycles 2..17 (`Layer.first`). Same floor, cap, total and ties as prompt_stats.
pub fn firstRows(a: std.mem.Allocator, r: *const Result, out: []u32) !void {
    const L = r.n_layers;
    const u = r.layers[0].decode_rows;
    const S = arm.decode_rows_shift_cap;
    const Cand = struct { m: u64, row: u32, layer: u32 };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(a);
    var k: u64 = @as(u64, L) * u;
    for (0..L) |l| {
        const lo = @max(r.layers[l].prompt_rows, u -| S);
        const hi = @min(r.n_experts, u + S);
        out[l] = lo;
        k -= lo;
        var row = lo + 1;
        while (row <= hi) : (row += 1) try cands.append(a, .{ .m = r.layers[l].first, .row = row, .layer = @intCast(l) });
    }
    std.sort.pdq(Cand, cands.items, {}, struct {
        fn lt(_: void, x: Cand, y: Cand) bool {
            const rx: u128 = x.row;
            const ry: u128 = y.row;
            const vx = @as(u128, x.m) * ry * ry * ry;
            const vy = @as(u128, y.m) * rx * rx * rx;
            if (vx != vy) return vx > vy;
            if (x.row != y.row) return x.row < y.row;
            return x.layer < y.layer;
        }
    }.lt);
    for (cands.items[0..@intCast(k)]) |c| out[c.layer] += 1;
}

/// prompt_stats' rows (`arm.DecodeRows`) from the replayed prompt counts.
pub fn promptRows(a: std.mem.Allocator, r: *const Result, out: []u32) !void {
    var dr = try arm.DecodeRows.init(a, r.n_layers, r.n_experts);
    defer dr.deinit(a);
    const prompt = try a.alloc(u32, r.n_layers);
    defer a.free(prompt);
    for (0..r.n_layers) |l| {
        @memcpy(dr.layerCounts(l), r.prompt_counts[l * r.n_experts ..][0..r.n_experts]);
        prompt[l] = r.layers[l].prompt_rows;
    }
    @memcpy(out, try dr.plan(prompt, r.layers[0].decode_rows));
}

// ── the test-side trace writer (the format, for synthetic traces) ──

pub const Writer = struct {
    ws: std.ArrayList(u32) = .empty,

    pub fn deinit(w: *Writer, a: std.mem.Allocator) void {
        w.ws.deinit(a);
    }

    fn put(w: *Writer, a: std.mem.Allocator, kind: Kind, layer: u32, fixed: []const u32, h: []const u16, tail: []const u32) !void {
        try w.ws.append(a, @as(u32, @intFromEnum(kind)) << 24 | layer);
        try w.ws.append(a, @intCast(fixed.len + (h.len + 1) / 2 + tail.len));
        try w.ws.appendSlice(a, fixed);
        var k: usize = 0;
        while (k < h.len) : (k += 2) try w.ws.append(a, @as(u32, h[k]) | @as(u32, if (k + 1 < h.len) h[k + 1] else 0) << 16);
        try w.ws.appendSlice(a, tail);
    }
    pub fn init(w: *Writer, a: std.mem.Allocator, layer: u32, n_experts: u32, capacity: u32) !void {
        try w.put(a, .init, layer, &.{ n_experts, capacity }, &.{}, &.{});
    }
    pub fn seed(w: *Writer, a: std.mem.Allocator, layer: u32, ids: []const u16) !void {
        try w.put(a, .seed, layer, &.{@intCast(ids.len)}, ids, &.{});
    }
    pub fn plan(w: *Writer, a: std.mem.Allocator, layer: u32, phase: expert_policy.Phase, cycle: u32, ids: []const u16) !void {
        try w.put(a, .plan, layer, &.{ @intFromEnum(phase), 0, cycle, @intCast(ids.len), 0 }, ids, &.{});
    }
    pub fn grow(w: *Writer, a: std.mem.Allocator, layer: u32, capacity: u32, map: []const u16) !void {
        try w.put(a, .grow, layer, &.{ capacity, @intCast(map.len) }, map, &.{});
    }
    pub fn mark(w: *Writer, a: std.mem.Allocator) !void {
        try w.put(a, .mark, 0, &.{0}, &.{}, &.{});
    }
};

fn printTable(r: *const Result, name: []const u8, comptime field: []const u8) void {
    const cyc: f64 = @floatFromInt(@max(r.cycles, 1));
    std.debug.print("\n{s}: misses per cycle per layer ({s})\n", .{ name, field });
    for (r.layers[0..r.n_layers], 0..) |l, i| std.debug.print("{s}{d:.2}", .{ if (i == 0) "  " else " ", @as(f64, @floatFromInt(@field(l, field))) / cyc });
    std.debug.print("\n  total {d:.2}\n", .{@as(f64, @floatFromInt(r.total(field))) / cyc});
}

const testing = std.testing;

test "dsv41 policy replay: LRU and Belady on known set-request sequences" {
    const a = testing.allocator;
    // Capacity 2 from {1, 2}: LRU evicts 1 for 3, 2 for 1, 1 for 2 (3 misses); Belady evicts 2 for 3 (its next use is
    // farthest) and keeps 1 (2 misses).
    const calls = [_]Call{ &.{3}, &.{1}, &.{3}, &.{2} };
    try testing.expectEqual(@as(u64, 3), try lruMisses(a, &.{ 1, 2 }, &calls, 2, 8));
    try testing.expectEqual(@as(u64, 2), try beladyMisses(a, &.{ 1, 2 }, &calls, 2, 8));
    // Enough rows: compulsory misses only.
    try testing.expectEqual(@as(u64, 1), try lruMisses(a, &.{ 1, 2 }, &calls, 3, 8));
    try testing.expectEqual(@as(u64, 1), try beladyMisses(a, &.{ 1, 2 }, &calls, 3, 8));
    // A call's own experts are never victims.
    const wide = [_]Call{ &.{ 4, 5 }, &.{ 1, 4 } };
    try testing.expectEqual(@as(u64, 3), try lruMisses(a, &.{ 1, 2 }, &wide, 2, 8));
}

test "dsv41 policy replay: a synthetic trace replays the shipped policy as LayerPolicy does, and a per-layer vector moves only residency" {
    const a = testing.allocator;
    var w: Writer = .{};
    defer w.deinit(a);
    // Two layers, 16 experts, 2 prompt rows, 4 decode rows. Layer 0 decodes over 3 experts, layer 1 over 8.
    for (0..2) |l| try w.init(a, @intCast(l), 16, 2);
    try w.mark(a);
    try w.seed(a, 0, &.{ 1, 1, 2 });
    try w.seed(a, 1, &.{ 5, 6, 5 });
    try w.plan(a, 0, .prefill, 0, &.{ 1, 2 });
    try w.plan(a, 1, .prefill, 0, &.{ 5, 6 });
    // The live policy alongside, for the trace's own grow maps.
    var p0 = try expert_policy.LayerPolicy.init(a, 16, 2);
    defer p0.deinit(a);
    var p1 = try expert_policy.LayerPolicy.init(a, 16, 2);
    defer p1.deinit(a);
    var plan: expert_policy.Plan = .{};
    p0.prepareSeed(&.{ 1, 1, 2 });
    p1.prepareSeed(&.{ 5, 6, 5 });
    p0.planWith(&.{ 1, 2 }, .prefill, &plan, .{});
    p1.planWith(&.{ 5, 6 }, .prefill, &plan, .{});
    try p0.grow(4);
    try p1.grow(4);
    try w.grow(a, 0, 4, p0.slot_to_expert[0..4]);
    try w.grow(a, 1, 4, p1.slot_to_expert[0..4]);
    var want: [2]u64 = .{ 0, 0 };
    var rng = std.Random.DefaultPrng.init(22);
    for (1..21) |cycle| for (0..2) |l| {
        var ids: [3]u16 = undefined;
        const span: u16 = if (l == 0) 3 else 8;
        for (&ids) |*e| e.* = rng.random().uintLessThan(u16, span) + @as(u16, if (l == 0) 1 else 5);
        try w.plan(a, @intCast(l), .decode, @intCast(cycle), &ids);
        const pol = if (l == 0) &p0 else &p1;
        pol.planWith(&ids, .decode, &plan, .{});
        want[l] += plan.n_misses;
    };
    var r = try run(a, w.ws.items, null);
    defer r.deinit(a);
    try testing.expectEqual(@as(u32, 2), r.n_layers);
    try testing.expectEqual(@as(u32, 20), r.cycles);
    try testing.expectEqual(@as(u32, 0), r.map_mismatches + r.admit_mismatches);
    try testing.expectEqual(want[0], r.layers[0].shipped);
    try testing.expectEqual(want[1], r.layers[1].shipped);
    // Belady never misses more than LRU or the shipped policy at the same rows.
    for (r.layers[0..2]) |l| try testing.expect(l.compulsory <= l.belady and l.belady <= l.lru and l.belady <= l.shipped);
    try testing.expectEqual(want[0] + want[1], r.decode.misses);
    try testing.expectEqual(@as(u64, 40), r.decode.plans);
    try testing.expectEqual(@as(u64, 2), r.prefill.plans);
    // Rows moved to the wide layer (same total): its misses fall, layer 0's (3 experts in 3 rows) stay compulsory.
    var moved = try run(a, w.ws.items, &.{ 3, 5 });
    defer moved.deinit(a);
    try testing.expect(moved.layers[1].shipped <= r.layers[1].shipped);
    try testing.expect(moved.total("belady") <= r.total("belady"));
    // decode_first16 gives the rows to the layer that missed in cycles 2..17.
    var fr: [2]u32 = undefined;
    try firstRows(a, &r, &fr);
    try testing.expectEqual(@as(u32, 8), fr[0] + fr[1]);
    try testing.expect(fr[1] >= fr[0]);
}

// DSV41_POLICY_TRACE=<.policytrace.bin> (host): the native trace at its own rows, prompt_stats' and decode_first16's;
// shipped / LRU / Belady misses per cycle per layer, one table per quantity.
test "dsv41 policy replay: the native trace priced at uniform, prompt_stats and decode_first16 rows" {
    const path = std.mem.span(std.c.getenv("DSV41_POLICY_TRACE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 30));
    defer a.free(bytes);
    if (bytes.len % 4 != 0) return error.TraceNotWords;
    const ws = try a.alloc(u32, bytes.len / 4);
    defer a.free(ws);
    for (ws, 0..) |*x, k| x.* = std.mem.readInt(u32, bytes[4 * k ..][0..4], .little);
    var base = try run(a, ws, null);
    defer base.deinit(a);
    std.debug.print("\nDSV41_POLICY_REPLAY {{\"layers\": {d}, \"cycles\": {d}, \"admit_mismatches\": {d}, \"map_mismatches\": {d}", .{ base.n_layers, base.cycles, base.admit_mismatches, base.map_mismatches });
    inline for (.{ "construction", "prefill", "decode" }) |ph| {
        const x = @field(base, ph);
        std.debug.print(", \"{s}\": {{\"plans\": {d}, \"hits\": {d}, \"misses\": {d}, \"evictions\": {d}, \"persistent_loads\": {d}, \"transient_loads\": {d}}}", .{ ph, x.plans, x.hits, x.misses, x.evictions, x.persistent_loads, x.transient_loads });
    }
    std.debug.print("}}\n", .{});
    // The receipt's decode_stream.misses, when given: the acceptance before any alternative is read.
    if (std.c.getenv("DSV41_POLICY_DECODE_MISSES")) |v| try testing.expectEqual(try std.fmt.parseInt(u64, std.mem.span(v), 10), base.decode.misses);
    const rows = try a.alloc(u32, base.n_layers);
    defer a.free(rows);
    inline for (.{ "shipped", "lru", "belady", "compulsory" }) |f| printTable(&base, "uniform", f);
    try promptRows(a, &base, rows);
    std.debug.print("\nprompt_stats rows: {any}\n", .{rows});
    var ps = try run(a, ws, rows);
    defer ps.deinit(a);
    inline for (.{ "shipped", "lru", "belady", "compulsory" }) |f| printTable(&ps, "prompt_stats", f);
    try firstRows(a, &base, rows);
    std.debug.print("\ndecode_first16 rows: {any}\n", .{rows});
    var df = try run(a, ws, rows);
    defer df.deinit(a);
    inline for (.{ "shipped", "lru", "belady", "compulsory" }) |f| printTable(&df, "decode_first16", f);
    try testing.expectEqual(@as(u32, 0), base.map_mismatches);
    try testing.expectEqual(@as(u32, 0), base.admit_mismatches);
}
