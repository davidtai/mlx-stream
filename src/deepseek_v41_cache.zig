//! DeepSeek-V4.1 per-sequence attention state (Python `deepseek_v41_cache.py`):
//! the window, compressed-KV, index-key and compressor-frontier lanes of each
//! layer, their trim / rollback seam, and the prefill chunk geometry. The
//! backing is a construction-time route; every route hands the attention the
//! same rows the full-history store would (the ring keeps a contiguous suffix
//! whose dropped rows no query can reach). Generic over the op backend `G`.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");

const sdk_kv = @import("sdk_ext.zig").kv;
const sdk_testing = @import("sdk_ext.zig").kv_testing;
comptime {
    std.debug.assert(ops.max_dims == sdk_kv.max_dims);
}

/// The lanes, their routes and caps live in the SDK's KV seam (`sdk_ext.kv`); the names stay here for the arch.
pub const Route = sdk_kv.Route;
pub const Geometry = sdk_kv.Geometry;
pub const boundedCompCap = sdk_kv.boundedCompCap;
pub const boundedLatentCap = sdk_kv.boundedLatentCap;
pub const Error = sdk_kv.Error;
pub const Lanes = sdk_kv.Lanes;

/// One layer's attention state (Python `LayerAttentionCache` + its
/// `CompressorState`), lanes picked at construction.
pub fn LayerState(comptime G: type) type {
    return struct {
        const Self = @This();
        const L = Lanes(G);
        pub const T = G.T;

        offset: u32 = 0,
        window_size: u32,
        ratio: u32,
        kv_source: bool,
        bounded_max_kv: ?u32 = null,
        window: L.Window,
        compress: L.Store,
        index: L.Store,
        /// The compressor frontier (`raw_kv`, `raw_score`) of a ratio > 1 kv source. On the ring routes a `Ring` of
        /// window `ratio`: a push pools only the groups it completes, whose rows start within `ratio - 1` rows of the
        /// fed length, and a verify's rollback re-exposes at most its own rows, so the ring's retained rows (ratio plus
        /// the verify margin and slack) hold every row a later push reads. Elsewhere a plain store of every fed row.
        frontier: ?struct { kv: L.Window, score: L.Window } = null,

        pub const Mark = struct { offset: u32, window: u32, compress: u32, index: u32, frontier: u32 };

        pub fn init(li: v41.LayerInfo, window_size: u32, geo: Geometry) Self {
            const ratio: u32 = li.ratio;
            var self: Self = .{
                .window_size = window_size,
                .ratio = ratio,
                .kv_source = li.kv_source,
                .window = .{ .store = .{ .concat = .{} } },
                .compress = .{ .concat = .{} },
                .index = .{ .concat = .{} },
            };
            var frontier_lane: L.Window = .{ .store = .{ .concat = .{} } };
            switch (geo.route) {
                .full_history => {},
                .chunk_grow => {
                    self.window = .{ .store = .{ .grow = L.Grow.init(256, null) } };
                    self.compress = .{ .grow = L.Grow.init(256, null) };
                    self.index = .{ .grow = L.Grow.init(256, null) };
                },
                .window_ring => {
                    self.window = .{ .ring = L.Ring.init(window_size, geo) };
                    const icap = geo.max_kv orelse 256;
                    self.compress = .{ .grow = L.Grow.init(icap, null) };
                    self.index = .{ .grow = L.Grow.init(icap, null) };
                    frontier_lane = .{ .ring = L.Ring.init(@max(ratio, 1), geo) };
                },
                .bounded => {
                    self.bounded_max_kv = geo.max_kv;
                    self.window = .{ .ring = L.Ring.init(window_size, geo) };
                    const cc = boundedCompCap(geo.max_kv, ratio);
                    self.compress = .{ .grow = L.Grow.init(cc orelse 256, cc) };
                    self.index = .{ .grow = L.Grow.init(cc orelse 256, cc) };
                    frontier_lane = .{ .ring = L.Ring.init(@max(ratio, 1), geo) };
                },
            }
            if (li.kv_source and ratio > 1) self.frontier = .{ .kv = frontier_lane, .score = frontier_lane };
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.window.deinit(g);
            self.compress.deinit(g);
            self.index.deinit(g);
            if (self.frontier) |*f| {
                f.kv.deinit(g);
                f.score.deinit(g);
            }
        }

        /// The longest sequence `canAdmit` lets through (null: unbounded). A
        /// constant of the layer's geometry: a state checks its minimum once.
        pub fn admitLimit(self: *const Self) ?u32 {
            const m = self.bounded_max_kv orelse return null;
            var lim: ?u32 = null;
            if (self.kv_source and self.ratio >= 1) if (boundedCompCap(m, self.ratio)) |cap| {
                const l = (cap + 1) * self.ratio - 1; // new_len / ratio <= cap
                lim = if (lim) |x| @min(x, l) else l;
            };
            return lim;
        }

        /// `assert_can_admit`: an over-cap forward fails before any lane is written.
        pub fn canAdmit(self: *const Self, n: u32) Error!void {
            const m = self.bounded_max_kv orelse return;
            const new_len = self.offset + n;
            if (self.kv_source and self.ratio >= 1) if (boundedCompCap(m, self.ratio)) |cap| if (new_len / self.ratio > cap) return error.BoundedLaneFull;
        }

        pub fn nFed(self: *const Self) u32 {
            return if (self.frontier) |f| f.kv.rows() else 0;
        }

        /// The groups a push completes: `kv` / `score` appended, then the completed groups' rows (fed rows
        /// [g_before * ratio, g_after * ratio)) as views of the lanes (a ring's view starts at its drop offset).
        pub const Groups = struct { kv: T, score: T, groups: u32 };

        pub fn frontierGroups(self: *Self, g: *G, kv: T, score: T) !?Groups {
            const f = &self.frontier.?;
            const r = self.ratio;
            const n_before = f.kv.rows();
            try f.kv.append(g, kv);
            try f.score.append(g, score);
            const g_before = n_before / r;
            const g_after = f.kv.rows() / r;
            if (g_after == g_before) return null;
            const drop = f.kv.dropOffset();
            // The ring keeps the incomplete group's rows by construction (see `frontier`); checked, never taken.
            if (g_before * r < drop) return error.FrontierRowsDropped;
            const lo = g_before * r - drop;
            const hi = g_after * r - drop;
            return .{ .kv = try L.rowsSlice(g, (try f.kv.view(g)).?, lo, hi), .score = try L.rowsSlice(g, (try f.score.view(g)).?, lo, hi), .groups = g_after - g_before };
        }

        /// `CompressorState.push`: append the fp32 projections, return the
        /// softmax-gated pooled latents of the groups this push completed.
        pub fn frontierPush(self: *Self, g: *G, kv: T, score: T) !?T {
            const gr = (try self.frontierGroups(g, kv, score)) orelse return null;
            const sh = g.shapeOf(kv);
            const grp: [4]c_int = .{ sh.d[0], @intCast(gr.groups), @intCast(self.ratio), sh.d[2] };
            const gk = try g.reshape(gr.kv, &grp);
            const gs = try g.reshape(gr.score, &grp);
            return try g.sum(try g.mul(gk, try g.softmax(gs, 2)), 2, false);
        }

        /// `advance`: the entry offset after a forward's appends.
        pub fn advance(self: *Self, n: u32) void {
            self.offset += n;
        }

        /// Whether `trim(n)` can restore every lane (the ring keeps the window of
        /// the next query).
        pub fn canTrim(self: *const Self, n: u32) bool {
            if (n > self.offset) return false;
            if (self.frontier) |f| if (n > f.kv.rows() or !f.kv.canTruncateTo(f.kv.rows() - n)) return false;
            return switch (self.window) {
                .ring => |r| r.canTruncateToLength(self.offset - n),
                .store => true,
            };
        }

        /// `trim(n)`: back to `offset - n` tokens; 0 (no change) when a ring
        /// cannot recover that far (the session-restore miss contract).
        pub fn trim(self: *Self, g: *G, n: u32) !u32 {
            if (n == 0) return 0;
            if (n > self.offset) return error.TrimPastStart;
            const new_len = self.offset - n;
            // The frontier ring's rule with the window's: a rollback either lands whole or is a clean miss.
            if (self.frontier) |f| if (n <= f.kv.rows() and !f.kv.canTruncateTo(f.kv.rows() - n)) return 0;
            switch (self.window) {
                .ring => |*r| {
                    if (!r.canTruncateToLength(new_len)) return 0;
                    try r.truncateToLength(new_len);
                },
                .store => |*s| try s.truncate(g, new_len),
            }
            if (self.kv_source and self.ratio >= 1) {
                const groups = new_len / self.ratio;
                try self.compress.truncate(g, groups);
                try self.index.truncate(g, groups);
                if (self.frontier) |*f| {
                    if (n > f.kv.rows()) return error.TrimPastStart;
                    const keep = f.kv.rows() - n;
                    try f.kv.truncateTo(g, keep);
                    try f.score.truncateTo(g, keep);
                }
            }
            self.offset = new_len;
            return n;
        }

        pub fn mark(self: *const Self) Mark {
            return .{
                .offset = self.offset,
                .window = switch (self.window) {
                    .store => |s| s.rows(),
                    .ring => |r| r.logicalLen(),
                },
                .compress = self.compress.rows(),
                .index = self.index.rows(),
                .frontier = self.nFed(),
            };
        }

        pub fn rollback(self: *Self, g: *G, m: Mark) !void {
            switch (self.window) {
                .ring => |*r| try r.truncateToLength(m.offset),
                .store => |*s| try s.truncate(g, m.window),
            }
            try self.compress.truncate(g, m.compress);
            try self.index.truncate(g, m.index);
            if (self.frontier) |*f| {
                try f.kv.truncateTo(g, m.frontier);
                try f.score.truncateTo(g, m.frontier);
            }
            self.offset = m.offset;
        }

        /// The layer's state at a prompt's end, to continue a later prompt that starts with it (a conversation's next
        /// turn): the window and frontier rings copied (`Window.snapshot`, a turn deeper than any ring's margin), the
        /// stores' rows (append-only, so a truncate restores them).
        pub const Boundary = struct {
            offset: u32,
            window: L.Window.Snap,
            compress: u32,
            index: u32,
            frontier: ?struct { kv: L.Window.Snap, score: L.Window.Snap } = null,

            pub fn deinit(self: *Boundary, g: *G) void {
                self.window.deinit(g);
                if (self.frontier) |*f| {
                    f.kv.deinit(g);
                    f.score.deinit(g);
                }
            }

            /// The copies to evaluate before the state is written again (at most three).
            pub fn arrays(self: *const Boundary, out: []T) usize {
                var k: usize = 0;
                if (self.window.array()) |x| {
                    out[k] = x;
                    k += 1;
                }
                if (self.frontier) |*f| for ([_]?T{ f.kv.array(), f.score.array() }) |a| if (a) |x| {
                    out[k] = x;
                    k += 1;
                };
                return k;
            }
        };

        pub fn boundary(self: *const Self, g: *G) !Boundary {
            var b: Boundary = .{ .offset = self.offset, .window = try self.window.snapshot(g), .compress = self.compress.rows(), .index = self.index.rows() };
            errdefer b.deinit(g);
            if (self.frontier) |*f| b.frontier = .{ .kv = try f.kv.snapshot(g), .score = try f.score.snapshot(g) };
            return b;
        }

        /// Back to `b` (spent): every lane as it stood at the boundary, whatever was appended or compacted since.
        pub fn restoreBoundary(self: *Self, g: *G, b: Boundary) !void {
            try self.window.restore(g, b.window);
            try self.compress.truncate(g, b.compress);
            try self.index.truncate(g, b.index);
            if (self.frontier) |*f| {
                try f.kv.restore(g, b.frontier.?.kv);
                try f.score.restore(g, b.frontier.?.score);
            }
            self.offset = b.offset;
        }
    };
}

// ── prefill chunk geometry (host) ──

pub const default_chunk_target_bytes: f64 = 8e9;

/// `_prefill_score_bytes_per_row`: one `[H, T]` f32 score row, T = s plus the
/// smallest positive ratio's compressed rows.
pub fn prefillScoreBytesPerRow(c: *const v41.Config, s: u64) u64 {
    var min_ratio: u64 = 0;
    for (c.layers[0 .. c.n_layers + c.dspark.n_stages]) |li| {
        if (li.ratio > 0 and (min_ratio == 0 or li.ratio < min_ratio)) min_ratio = li.ratio;
    }
    const n_comp = if (min_ratio > 0) s / min_ratio else 0;
    return @as(u64, c.n_heads) * (s + n_comp) * 4;
}

/// `_resolve_prefill_chunk`: an explicit chunk wins, else the largest query
/// chunk whose score transient stays under `target_bytes`. A result `>= s`
/// (or `<= 0` explicit) is one-shot.
pub fn resolvePrefillChunk(c: *const v41.Config, s: u64, explicit: ?i64, target_bytes: f64) i64 {
    if (explicit) |e| return e;
    const per_row = prefillScoreBytesPerRow(c, s);
    if (per_row == 0) return @intCast(s);
    const chunk: u64 = @intFromFloat(@floor(@max(target_bytes, 1e9) / @as(f64, @floatFromInt(per_row))));
    return @intCast(@max(1, @min(chunk, s)));
}

/// The served tier's K16 span (attention over the selected keys, `routes.selected_keys`): the chunk the served wave
/// holds under `target_bytes`, never narrower than the standard 16,384-token prompt's span under the stock rule.
/// The stock rule (`resolvePrefillChunk`) budgets the full-score attention row [H, s + s / ratio] f32, an array the
/// served tier never builds; its span falls as 1 / s (59 rows at 256K, 14 at 1M), so the K16 pass's chunks, each a
/// graph build and a fence per layer, grow as s^2 (pass3ew: 512K and 1M ran ~10-13x slower per token at equal depth
/// than 256K and missed the 6,000 s limit). The served chunk's score-sized arrays per row: the attention's
/// [H, selected_keys + 1] f32 (`PrefillBill.waveBytes`, served) and the indexer's score over every position read,
/// 4 B a position (IDX_CHUNKED_SELECT bounds a chunk's at its block budget). Its span is
///   min(target / (H (selected_keys + 1) 4 + 4 s), max(stock(s), stock(16,384)))
/// so every prompt up to 16,384 keeps the stock span byte for byte (the served-wave span is wider there), and a longer
/// one keeps 16,384's span (953 rows) while the served wave allows it (~1,800 rows at 1M).
pub fn servedSpanRows(n_heads: u64, selected_keys: u64, min_ratio: u64, s: u64, target_bytes: f64) u64 {
    if (s == 0) return 0;
    const t = @max(target_bytes, 1e9);
    const stock = struct {
        fn of(h: u64, r: u64, n: u64, tb: f64) u64 {
            const per_row = h * (n + (if (r > 0) n / r else 0)) * 4;
            if (per_row == 0) return n;
            return @max(1, @min(@as(u64, @intFromFloat(@floor(tb / @as(f64, @floatFromInt(per_row))))), n));
        }
    }.of;
    const floor_rows = @max(stock(n_heads, min_ratio, s, t), stock(n_heads, min_ratio, fill_span_prompt, t));
    const served_per_row = n_heads * (selected_keys + 1) * 4 + 4 * s;
    const served: u64 = @intFromFloat(@floor(t / @as(f64, @floatFromInt(served_per_row))));
    return @max(1, @min(@min(served, floor_rows), s));
}

/// The prompt whose stock span the served rule keeps as its floor (the standard 16K cell's).
pub const fill_span_prompt: u64 = 16384;

/// `resolvePrefillChunk` at the tier's attention: the served span (`servedSpanRows`) when it attends the selected keys,
/// else the stock rule.
pub fn resolvePrefillChunkFor(c: *const v41.Config, s: u64, explicit: ?i64, target_bytes: f64, selected_keys: bool) i64 {
    if (explicit) |e| return e;
    if (!selected_keys) return resolvePrefillChunk(c, s, null, target_bytes);
    var min_ratio: u64 = 0;
    for (c.layers[0 .. c.n_layers + c.dspark.n_stages]) |li| {
        if (li.ratio > 0 and (min_ratio == 0 or li.ratio < min_ratio)) min_ratio = li.ratio;
    }
    return @intCast(servedSpanRows(c.n_heads, @as(u64, c.window) + c.index_topk, min_ratio, s, target_bytes));
}

/// The query spans of a forward over `s` tokens (`[start, end)` pairs).
pub fn prefillSpans(a: std.mem.Allocator, s: u32, chunk: i64) ![][2]u32 {
    if (chunk <= 0 or chunk >= s) {
        const one = try a.alloc([2]u32, 1);
        one[0] = .{ 0, s };
        return one;
    }
    const k: u32 = @intCast(chunk);
    const n = (s + k - 1) / k;
    const out = try a.alloc([2]u32, n);
    for (out, 0..) |*sp, i| sp.* = .{ @intCast(i * k), @min(s, @as(u32, @intCast((i + 1) * k))) };
    return out;
}

/// The prompt's sub-chunk: a prompt longer than this runs as module calls of at most about this many rows, each
/// extending the request's state (upstream deepseek_v4's `PREFILL_SUB`, deepseek_v4.zig:8054-8077, which sub-chunks its
/// prefill inside `extendState` at 512 and exposes `prefillSub()` to server.zig's prefill memory guard). Here the
/// sub-chunk is a call of the layer-major pass, so it is wide: the 16,384-token standard cell stays one call.
pub const prefill_sub: u64 = 16384;

/// The rows of each sub-chunk call of a prompt of `n` tokens whose spans are `span` rows (the chunk rule at the
/// WHOLE prompt's length, pinned for every call): `sub` rounded down to a multiple of `span`, so each call's spans
/// are the one-call pass's spans at the same positions. `n` when the prompt is one call.
pub fn prefillSubWidth(n: u64, span: u64, sub: u64) u64 {
    if (n <= sub or span == 0 or span >= sub) return n;
    return sub - sub % span;
}

/// The widest call of `prefillSubCalls` (the bill's rows): a tail of at most one span rides on the call before it
/// (alone it would be a one-span call: the single-span forward, not the layer-major pass of the one-call run).
pub fn prefillSubWidest(n: u64, span: u64, sub: u64) u64 {
    const w = prefillSubWidth(n, span, sub);
    if (w >= n) return n;
    const r = n % w;
    return if (r > 0 and r <= span) w + r else w;
}

/// A sub-chunk call's rows x the positions it reads, at most (one span over, from the merged short tail: the bill's
/// `PrefillBill.selectionRowPositions`). The call's index selection is held per row and position read across the layer
/// (pass3ep, 256K: 4.99 B a row and position, 16 calls, residual 0.23 GB), so a call deep in a long prompt runs fewer
/// rows: 16,384 rows up to 262,144 positions (every prompt up to 256K unchanged), then fewer (8,192 at 512K, 4,096 at 1M).
pub const prefill_sub_area: u64 = prefill_selection_budget_bytes / selection_pos_bytes;

/// The index selection a prompt call holds per row and compressed position across a layer: each K16 chunk's kept
/// selection (its top-k mask, candidates and window memo, carried to the layer's routed call) over every chunk of the
/// call, 4.99 B a row and position read (pass3ep at 256K: 16 sub-chunk calls, residual 0.23 GB). IDX_CHUNKED_SELECT
/// does not lower it: the indexer launches per chunk (`indexerSelect` sees the chunk's rows: 59 at 256K, 14 at 1M), and a
/// chunk's 5 x rows x N stays under its block budget, so every served chunk keeps the one launch (pass3et's 256K kill
/// with the select merged). The bill and the call area read this one constant.
pub const selection_pos_bytes: u64 = 5;

/// The selection's budget per sub-chunk call, fixed in bytes (21.47 GB: 16,384 rows over 262,144 positions at 5 B, so
/// every prompt up to 256K runs its calls as before); the area in rows x positions follows `selection_pos_bytes`.
pub const prefill_selection_budget_bytes: u64 = 21_474_836_480;

/// The rows of a sub-chunk call starting at position `start`: the largest w <= `sub` with w x (start + w) <=
/// `prefill_sub_area`, rounded down to a multiple of `span` (at least one span) so its spans are the one-call pass's.
pub fn prefillSubRowsAt(start: u64, span: u64, sub: u64) u64 {
    const disc = start * start + 4 * prefill_sub_area;
    var w = (std.math.sqrt(disc) - start) / 2;
    while (w > 0 and w * (start + w) > prefill_sub_area) w -= 1;
    while ((w + 1) * (start + w + 1) <= prefill_sub_area) w += 1;
    w = @min(w, sub);
    if (span > 0 and span < sub) w = @max(w - w % span, span);
    return @max(w, 1);
}

/// The calls (`[start, end)`) a prompt of `n` tokens runs as: one call up to the sub-chunk, else calls of
/// `prefillSubRowsAt` rows from each start (`prefillSubWidth` rows while rows x positions stay within the area), the
/// short tail (at most one span) merged into the call before it.
pub fn prefillSubCalls(a: std.mem.Allocator, n: u32, span: u64, sub: u64) ![][2]u32 {
    if (prefillSubWidth(n, span, sub) >= n) {
        const one = try a.alloc([2]u32, 1);
        one[0] = .{ 0, n };
        return one;
    }
    var out: std.ArrayList([2]u32) = .empty;
    errdefer out.deinit(a);
    var at: u64 = 0;
    while (at < n) {
        var end = @min(@as(u64, n), at + prefillSubRowsAt(at, span, sub));
        if (end < n and n - end <= span) end = n;
        try out.append(a, .{ @intCast(at), @intCast(end) });
        at = end;
    }
    return out.toOwnedSlice(a);
}

test "dsv41 cache: the prompt's sub-chunk calls: one call up to the sub-chunk, then span-aligned calls, the short tail merged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    // The standard cell: one call (16,384 rows, its 18 spans unchanged).
    const s16: u64 = @intCast(resolvePrefillChunk(&c, 16384, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(usize, 1), (try prefillSubCalls(a, 16384, s16, prefill_sub)).len);
    try testing.expectEqual(@as(u64, 16384), prefillSubWidest(16384, s16, prefill_sub));
    // Every longer prompt: calls whose starts are span boundaries of the one-call pass, covering it exactly.
    for ([_]u32{ 16385, 32768, 65536, 131072, 100_003 }) |n| {
        const span: u64 = @intCast(resolvePrefillChunk(&c, n, null, default_chunk_target_bytes));
        const calls = try prefillSubCalls(a, n, span, prefill_sub);
        try testing.expectEqual(@as(u32, 0), calls[0][0]);
        try testing.expectEqual(n, calls[calls.len - 1][1]);
        var widest: u64 = 0;
        for (calls, 0..) |cl, i| {
            if (i > 0) try testing.expectEqual(calls[i - 1][1], cl[0]);
            try testing.expectEqual(@as(u64, 0), cl[0] % span);
            // Every call is wider than one span (the layer-major pass), none past the sub-chunk and a span.
            try testing.expect(cl[1] - cl[0] > span or calls.len == 1);
            try testing.expect(cl[1] - cl[0] <= prefill_sub + span);
            widest = @max(widest, cl[1] - cl[0]);
        }
        try testing.expectEqual(widest, prefillSubWidest(n, span, prefill_sub));
    }
    // 128K: a 119-row span; eight 16,303-row calls (137 spans each), then the last 648 rows.
    const s128: u64 = @intCast(resolvePrefillChunk(&c, 131072, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(u64, 119), s128);
    try testing.expectEqual(@as(u64, 16303), prefillSubWidth(131072, s128, prefill_sub));
    const c128 = try prefillSubCalls(a, 131072, s128, prefill_sub);
    try testing.expectEqual(@as(usize, 9), c128.len);
    try testing.expectEqual([2]u32{ 130424, 131072 }, c128[8]);
    // A tail of at most one span rides on the call before it.
    const m = try prefillSubCalls(a, 2 * 60 + 25, 30, 60);
    try testing.expectEqual(@as(usize, 2), m.len);
    try testing.expectEqual([2]u32{ 60, 145 }, m[1]);
    try testing.expectEqual(@as(u64, 85), prefillSubWidest(145, 30, 60));
    const t = try prefillSubCalls(a, 2 * 60 + 31, 30, 60);
    try testing.expectEqual(@as(usize, 3), t.len);
    try testing.expectEqual([2]u32{ 120, 151 }, t[2]);
}

test "dsv41 cache: a sub-chunk call's rows x positions stay within the area: 256K unchanged, fewer rows deeper" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    for ([_]u32{ 131072, 262144, 524288, 1048576, 700_001 }) |n| {
        const span: u64 = @intCast(resolvePrefillChunk(&c, n, null, default_chunk_target_bytes));
        const calls = try prefillSubCalls(a, n, span, prefill_sub);
        var widest: u64 = 0;
        for (calls, 0..) |cl, i| {
            const rows: u64 = cl[1] - cl[0];
            if (i > 0) try testing.expectEqual(calls[i - 1][1], cl[0]);
            try testing.expectEqual(@as(u64, 0), cl[0] % span);
            try testing.expect(rows <= prefill_sub + span);
            // rows x positions read: within the area, one span over at most (the merged tail).
            try testing.expect(rows * cl[1] <= prefill_sub_area + span * cl[1]);
            widest = @max(widest, rows);
        }
        try testing.expectEqual(n, calls[calls.len - 1][1]);
        // The widest call is the first (the bill's rows): unchanged by the area.
        try testing.expectEqual(prefillSubWidest(n, span, prefill_sub), widest);
        if (n <= 262144) try testing.expectEqual((n + prefillSubWidth(n, span, prefill_sub) - 1) / prefillSubWidth(n, span, prefill_sub) - @intFromBool(n % prefillSubWidth(n, span, prefill_sub) != 0 and n % prefillSubWidth(n, span, prefill_sub) <= span), calls.len);
    }
    // The rows at a start: the root of w (start + w) <= area, a span multiple.
    try testing.expectEqual(@as(u64, 16384), prefillSubRowsAt(0, 0, prefill_sub));
    try testing.expectEqual(@as(u64, 16384), prefillSubRowsAt(245_760, 0, prefill_sub));
    const w = prefillSubRowsAt(1_040_000, 14, prefill_sub);
    try testing.expect(w * (1_040_000 + w) <= prefill_sub_area and (w + 14) * (1_040_000 + w + 14) > prefill_sub_area and w % 14 == 0);
}

// ── tests: a row-id backend checks the lanes' reachable rows on the host ──

const testing = std.testing;

/// The SDK's row-id backend (`sdk.testing.RowOps`): every append / compaction / view / trim checked against the
/// absolute positions it must hold.
const RowOps = sdk_testing.RowOps;

const RS = LayerState(RowOps);

/// The window view must be exactly positions [drop, logical) and hold every
/// row the queries of the last append can reach.
fn expectWindow(g: *RowOps, st: *const RS, logical: u32) !void {
    const v = (try st.window.view(g)) orelse return error.TestUnexpectedResult;
    const r = g.rows(v);
    const drop = st.window.dropOffset();
    try testing.expectEqual(logical - drop, @as(u32, @intCast(r.len)));
    for (r, 0..) |id, j| try testing.expectEqual(@as(i64, @intCast(drop + j)), id);
}

test "dsv41 cache: a prompt's boundary restores the ring whatever decode appended since; the next turn reads every row a cold prompt would" {
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    const li: v41.LayerInfo = .{ .ratio = 0 };
    for ([_]u32{ 0, 1 }) |cut| {
        var st = RS.init(li, 128, .{ .route = .window_ring });
        defer st.deinit(&g);
        // Turn 1's prompt: 300 rows in chunks.
        var pos: u32 = 0;
        for ([_]u32{ 120, 120, 60 }) |n| {
            try st.window.append(&g, try g.range(pos, pos + n));
            st.advance(n);
            pos += n;
        }
        var b = try st.boundary(&g);
        var spent = false;
        defer if (!spent) b.deinit(&g);
        // Decode: 250 single rows, the ring compacting several times past the boundary's rows.
        for (0..250) |_| {
            try st.window.append(&g, try g.range(pos, pos + 1));
            st.advance(1);
            pos += 1;
        }
        // The next turn: back to the boundary (spent), `cut` ids short (a thinking turn re-renders the last prompt id),
        // then 70 new rows: the ring holds exactly the rows a cold 300 - cut + 70 prompt holds.
        try st.restoreBoundary(&g, b);
        spent = true;
        try testing.expectEqual(@as(u32, 300), st.offset);
        if (cut > 0) try testing.expectEqual(cut, try st.trim(&g, cut));
        const at = 300 - cut;
        try st.window.append(&g, try g.range(at, at + 70));
        st.advance(70);
        try expectWindow(&g, &st, at + 70);
        // Two turns on: a second boundary, decode, restore, extend.
        const b2 = try st.boundary(&g);
        for (0..40) |k| {
            try st.window.append(&g, try g.range(at + 70 + @as(u32, @intCast(k)), at + 71 + @as(u32, @intCast(k))));
            st.advance(1);
        }
        try st.restoreBoundary(&g, b2);
        try st.window.append(&g, try g.range(at + 70, at + 90));
        st.advance(20);
        try expectWindow(&g, &st, at + 90);
    }
}

test "dsv41 cache: the window ring keeps every reachable row across prefill chunks, decode, verify and trims" {
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    const li: v41.LayerInfo = .{ .ratio = 0 };
    var st = RS.init(li, 128, .{ .route = .window_ring });
    defer st.deinit(&g);
    var full = RS.init(li, 128, .{ .route = .full_history });
    defer full.deinit(&g);
    // prefill in chunks wider and narrower than the ring, then decode / verify with rejections
    const Step = struct { n: u32, trim: u32 = 0 };
    var steps: [36]Step = @splat(.{ .n = 6, .trim = 2 });
    steps[0..6].* = .{ .{ .n = 300 }, .{ .n = 40 }, .{ .n = 1 }, .{ .n = 6, .trim = 4 }, .{ .n = 1 }, .{ .n = 6, .trim = 5 } };
    var pos: u32 = 0;
    for (steps) |sp| {
        const x = try g.range(pos, pos + sp.n);
        try st.window.append(&g, x);
        try full.window.append(&g, x);
        st.advance(sp.n);
        full.advance(sp.n);
        pos += sp.n;
        try expectWindow(&g, &st, pos);
        // Every query of this append reaches back window-1 rows: all resident.
        try testing.expect(pos - sp.n + 1 >= st.window.dropOffset() + st.window_size or st.window.dropOffset() == 0);
        if (sp.trim > 0) {
            try testing.expectEqual(sp.trim, try st.trim(&g, sp.trim));
            _ = try full.trim(&g, sp.trim);
            pos -= sp.trim;
            try expectWindow(&g, &st, pos);
        }
    }
    // The ring stayed bounded: two buffers of window + 8 + 8 + 64 rows after the wide prefill chunk shrank back.
    try testing.expectEqual(@as(u32, 128 + 8 + 8 + 64), st.window.ring.phys_cap);
    // The same script on the bare ring lane, against the concatenated store; a rollback past the window refused by name.
    var ring: Lanes(RowOps).Window = .{ .ring = Lanes(RowOps).Ring.init(128, .{ .route = .window_ring }) };
    defer ring.deinit(&g);
    var lane_steps: [steps.len + 1]sdk_testing.LaneStep = undefined;
    for (steps, lane_steps[0..steps.len]) |sp, *ls| ls.* = .{ .n = sp.n, .trim = sp.trim };
    lane_steps[steps.len] = .{ .n = 1, .trim = 200 };
    try sdk_testing.expectLaneEquivalence(&g, &ring, &lane_steps, 128 - 1);
    // A rollback past the resident window is a clean miss, never a wrong read.
    try testing.expectEqual(@as(u32, 0), try st.trim(&g, 200));
    try testing.expectEqual(pos, st.offset);
    // Before the first compaction every row is resident: a short sequence rolls back freely.
    var short = RS.init(li, 128, .{ .route = .window_ring });
    defer short.deinit(&g);
    try short.window.append(&g, try g.range(0, 40));
    short.advance(40);
    try testing.expectEqual(@as(u32, 6), try short.trim(&g, 6));
    try expectWindow(&g, &short, 34);
}

test "dsv41 cache: grow and bounded lanes read like the concatenated store, trims keep capacity" {
    const L = Lanes(RowOps);
    for ([_]L.Grow{ L.Grow.init(256, null), L.Grow.init(708, 708) }) |lane0| {
        var g: RowOps = .{ .gpa = testing.allocator };
        defer g.deinit();
        var lane: L.Store = .{ .grow = lane0 };
        defer lane.deinit(&g);
        // Every append against the concatenated store, row for row (sdk.testing's lane equivalence).
        try sdk_testing.expectLaneEquivalence(&g, &lane, &.{ .{ .n = 333 }, .{ .n = 1 }, .{ .n = 1 }, .{ .n = 6 }, .{ .n = 17 }, .{ .n = 300 } }, 0);
        const allocs = g.allocs;
        try lane.truncate(&g, 500);
        try lane.append(&g, try g.range(500, 510));
        try testing.expectEqual(allocs, g.allocs); // a trim keeps the capacity
        for (g.rows((try lane.view(&g)).?), 0..) |id, j| try testing.expectEqual(@as(i64, @intCast(j)), id);
        if (lane.grow.bounded_cap != null) try testing.expectError(error.BoundedLaneFull, lane.append(&g, try g.range(0, 200)));
    }
    // W107 caps at max_kv 700, ratio 2: 358 groups (the frontier is a ring, no cap of its own); admission refuses
    // before any write.
    try testing.expectEqual(@as(?u32, 358), boundedCompCap(700, 2));
    const li: v41.LayerInfo = .{ .ratio = 2, .kv_source = true, .index_source = true, .mode = .full };
    var st = RS.init(li, 128, .{ .route = .bounded, .max_kv = 700 });
    st.offset = 700;
    try testing.expectError(error.BoundedLaneFull, st.canAdmit(18));
    try st.canAdmit(17);
    // The per-state limit agrees with the per-forward check at every length around it,
    // for every compression ratio the model has (with and without the frontier).
    for ([_]u8{ 1, 2, 4, 128 }) |ratio| for ([_]u32{ 1, 7, 700, 4096 }) |max_kv| {
        const lr: v41.LayerInfo = .{ .ratio = ratio, .kv_source = true, .index_source = true, .mode = .full };
        var s2 = RS.init(lr, 128, .{ .route = .bounded, .max_kv = max_kv });
        const lim = s2.admitLimit().?;
        for (lim -| 40..lim + 40) |len| {
            s2.offset = 0;
            const ok = if (s2.canAdmit(@intCast(len))) |_| true else |_| false;
            try testing.expectEqual(len <= lim, ok);
        }
    };
    const unbounded = RS.init(li, 128, .{ .route = .full_history });
    try testing.expectEqual(@as(?u32, null), unbounded.admitLimit());
}

test "dsv41 cache: the compressor frontier pools each group once, across chunks and a trim" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    const li: v41.LayerInfo = .{ .ratio = 4, .kv_source = true, .index_source = true, .mode = .full };
    var st = LayerState(ops.TraceOps).init(li, 128, .{ .route = .window_ring });
    defer st.deinit(&g);
    const want = [_]?c_int{ 2, null, 1, null };
    for ([_]c_int{ 9, 2, 1, 3 }, want) |n, w| {
        const kv = try g.input(&.{ 1, n, 512 }, .float32);
        const pooled = try st.frontierPush(&g, kv, kv);
        if (w) |groups| {
            try testing.expect(g.shapeOf(pooled.?).eql(ops.Shape.of(&.{ 1, groups, 512 })));
        } else try testing.expect(pooled == null);
    }
    try testing.expectEqual(@as(u32, 15), st.nFed());
    st.offset = 15;
    // Trim 5: the frontier re-exposes the partial group of 10 fed rows.
    try testing.expectEqual(@as(u32, 5), try st.trim(&g, 5));
    try testing.expectEqual(@as(u32, 10), st.nFed());
    const kv = try g.input(&.{ 1, 2, 512 }, .float32);
    try testing.expect(g.shapeOf((try st.frontierPush(&g, kv, kv)).?).eql(ops.Shape.of(&.{ 1, 1, 512 })));
}

test "dsv41 cache: the frontier ring hands every push its completed groups' rows, as the full store does, across prefill chunks, verify blocks and trims" {
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    for ([_]u8{ 2, 4 }) |ratio| {
        const li: v41.LayerInfo = .{ .ratio = ratio, .kv_source = true, .index_source = true, .mode = .full };
        var st = RS.init(li, 128, .{ .route = .bounded, .max_kv = 20000 });
        defer st.deinit(&g);
        var full = RS.init(li, 128, .{ .route = .full_history });
        defer full.deinit(&g);
        try testing.expect(st.frontier.?.kv == .ring and full.frontier.?.kv == .store);
        // K16's chunks (953 and a ragged tail), then decode / verify blocks with rejected rows trimmed.
        const Step = struct { n: u32, trim: u32 = 0 };
        var steps: [40]Step = @splat(.{ .n = 6, .trim = 2 });
        steps[0..8].* = .{ .{ .n = 953 }, .{ .n = 953 }, .{ .n = 953 }, .{ .n = 183 }, .{ .n = 1 }, .{ .n = 6, .trim = 5 }, .{ .n = 8, .trim = 7 }, .{ .n = 3 } };
        var pos: u32 = 0;
        for (steps) |sp| {
            const x = try g.range(pos, pos + sp.n);
            const got = try st.frontierGroups(&g, x, x);
            const want = try full.frontierGroups(&g, x, x);
            try testing.expectEqual(want == null, got == null);
            if (got) |gr| {
                try testing.expectEqual(want.?.groups, gr.groups);
                try testing.expectEqualSlices(i64, g.rows(want.?.kv), g.rows(gr.kv));
                try testing.expectEqualSlices(i64, g.rows(want.?.score), g.rows(gr.score));
                // The rows of groups g_before .. g_after: every one fed, each group's rows once.
                const first = g.rows(gr.kv)[0];
                try testing.expectEqual(@as(i64, 0), @mod(first, ratio));
                for (g.rows(gr.kv), 0..) |id, j| try testing.expectEqual(first + @as(i64, @intCast(j)), id);
            }
            st.advance(sp.n);
            full.advance(sp.n);
            pos += sp.n;
            if (sp.trim > 0) {
                try testing.expect(st.canTrim(sp.trim));
                try testing.expectEqual(sp.trim, try st.trim(&g, sp.trim));
                _ = try full.trim(&g, sp.trim);
                pos -= sp.trim;
                try testing.expectEqual(full.nFed(), st.nFed());
            }
        }
        // The ring stayed small: two buffers of ratio + 8 + 8 + 64 rows once the wide chunks passed.
        try testing.expectEqual(@as(u32, ratio + 8 + 8 + 64), st.frontier.?.kv.ring.phys_cap);
        // A rollback deeper than the ring keeps is a clean miss (the window's rule), never a wrong group.
        try testing.expect(!st.canTrim(200));
        try testing.expectEqual(@as(u32, 0), try st.trim(&g, 200));
        try testing.expectEqual(full.nFed(), st.nFed());
    }
}

test "dsv41 cache: prefill chunks follow the Python shape-aware derivation" {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    // Goldens from `_derive_prefill_chunk` (8 GB target; ratio-1 layers double T).
    try testing.expectEqual(@as(i64, 953), resolvePrefillChunk(&c, 16384, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(i64, 2048), resolvePrefillChunk(&c, 2048, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(i64, 238), resolvePrefillChunk(&c, 65536, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(i64, 32), resolvePrefillChunk(&c, 16384, 32, default_chunk_target_bytes));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spans = try prefillSpans(arena.allocator(), 2000, 953);
    try testing.expectEqual(@as(usize, 3), spans.len);
    try testing.expectEqual([2]u32{ 1906, 2000 }, spans[2]);
    try testing.expectEqual(@as(usize, 1), (try prefillSpans(arena.allocator(), 1, 953)).len);
    // The 16K cell's prompt: the lane of record's 17 x 953 + 183 chunks.
    const cell = try prefillSpans(arena.allocator(), 16384, resolvePrefillChunk(&c, 16384, null, default_chunk_target_bytes));
    try testing.expectEqual(@as(usize, 18), cell.len);
    for (cell[0..17]) |sp| try testing.expectEqual(@as(u32, 953), sp[1] - sp[0]);
    try testing.expectEqual([2]u32{ 16201, 16384 }, cell[17]);
}
