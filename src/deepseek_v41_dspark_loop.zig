//! The DSpark-direct decode loop on the native path (Python
//! `dspark_generate` + `_decode_cycles` as the lane of record runs it: the
//! hybrid causal lookup to depth 7 in one verify chunk, greedy or #475 typical
//! acceptance). The prompt runs in forwards of at most `prompt_chunk` rows
//! (every routed call a decode-lane call), its main hiddens seed the draft
//! head once; each cycle drafts, verifies `[primary, drafts]`, accepts, and
//! commits by trimming the target and seeding the draft windows.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const graph = @import("deepseek_v41_graph.zig");
const mdl = @import("deepseek_v41_model.zig");
const ds = @import("deepseek_v41_dspark.zig");
const dh = @import("deepseek_v41_dspark_head.zig");
const first_cycle = @import("dsv41_decode_first.zig");
const dt = @import("dsv41_decode_timers.zig");
const draft_routes = @import("dsv41_draft_routes.zig");
const timeline = @import("dsv41_verify_timeline.zig");

pub const Config = struct {
    /// Requested native draft depth (capped by the head's block size).
    k_request: u32 = 5,
    /// The confidence early stop (the runner's 0.5); null keeps every draft.
    confidence_threshold: ?f64 = 0.5,
    /// `hybrid_install`'s causal lookup: a full native proposal extended from the history.
    lookup: ?struct { minimum_context: u32 = 2, extra_tokens: u32 = 2 } = .{},
    acceptance: ds.Acceptance = .greedy,
    /// Rows per prompt forward; `whole_prompt` runs it as one forward, chunked
    /// by the model's own rule (every chunk wider than a route takes runs
    /// through the wide lane: the served prompt pass).
    prompt_chunk: u32 = 8,
    max_tokens: u32,
    stop_ids: []const u32 = &.{},
    /// DRAFT_AHEAD (exact): during each verify's wait, the next draft block is built for the outcome "every draft
    /// accepted" (its windows seeded and main row taken from the verify's last row); committed only when the decision
    /// is that outcome, its primary written into its persistent inputs first; otherwise discarded uncommitted.
    draft_ahead: bool = false,
};

pub const Finish = enum { length, stop };

/// `Config.prompt_chunk`: the whole prompt in one forward.
pub const whole_prompt: u32 = std.math.maxInt(u32);

/// What a request's strategy keeps alive from its prompt pass into decode (the native bill's `prompt_state`,
/// decode phase), as `Loop.prefillImpl` leaves it: the seed gathers its own copies (`copyRows`), so what stays is
/// the next draft's main row (`main_h`, f32 `[1, n_main x hidden]`) and each stage's window (f32 `[rows, head_dim]`,
/// at most `c.window` rows); the prompt's main taps and main KV are freed at the prefill's reset. Stated here,
/// beside the code that retains it, so a change to the seed changes the bill with it (the served bank: 847,872 B
/// for any prompt of at least the window; the views it replaced held P x 67,584 B, 1,107,296,256 B at 16K).
pub fn seedRetainedBytes(c: *const v41.Config, prompt_tokens: u64) u64 {
    var n_main: u64 = 0;
    for (c.layers[0..c.n_layers]) |li| n_main += @intFromBool(li.dspark_target);
    const window_rows: u64 = @min(prompt_tokens, c.window);
    return n_main * c.hidden_size * 4 + @as(u64, c.dspark.n_stages) * window_rows * c.head_dim * 4;
}

/// One cycle's decisions, as the Python oracle fixture records them.
pub const CycleLog = struct {
    primary: u32,
    native: [ds.max_block]u32 = undefined,
    n_native: u32 = 0,
    /// Drafts kept by the confidence early stop, before the lookup.
    k_native: u32 = 0,
    /// The host values the cycle read: sigmoid confidences, the verify rows'
    /// argmax (all chunks) and the typical flags.
    conf: [ds.max_block]f32 = undefined,
    targets: [ds.max_block + 1]u32 = undefined,
    n_targets: u32 = 0,
    flags: [ds.max_block + 1]bool = undefined,
    n_flags: u32 = 0,
    drafts: [ds.max_block]u32 = undefined,
    k_eff: u32 = 0,
    accepted: u32 = 0,
    correction: u32 = 0,
    verified: u32 = 0,
    trimmed: u32 = 0,
    /// Set by a caller that classifies divergences (the window harness): each
    /// verify row's top two ids and f32 logits and its rms, indexed like `targets`.
    want_top: bool = false,
    top_ids: [ds.max_block + 1][2]u32 = undefined,
    top_logits: [ds.max_block + 1][2]f32 = undefined,
    rms: [ds.max_block + 1]f32 = undefined,
};

/// A cycle's phases for a decode-profile run's host stamps (`Loop.cycleStamped`): a stamper's
/// `mark(p)` charges the host time since its previous mark to `p`.
pub const Phase = enum { draft, verify, decide, commit, tail };

inline fn mark(stamp: anytype, p: Phase) void {
    if (@TypeOf(stamp) != void) stamp.mark(p);
}

/// The owner's call at the prompt's last trunk chunk (the Module's tail release route).
pub const PromptTail = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

pub fn Loop(comptime G: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        pub const M = mdl.Model(G);
        pub const H = dh.Head(G);

        g: *G,
        model: *M,
        head: *const H,
        st: *M.State,
        caches: []H.Cache,
        cfg: Config,
        lookup: ?ds.Lookup = null,
        stats: ds.Stats = .{},
        k_cap: u32,
        max_rows: u32,
        /// `main_hidden` of the next draft's main token (kept across resets).
        main_h: ?T = null,
        primary: u32 = 0,
        /// The lookup's history already ends with `primary` (the cell's prefill and every cycle
        /// append it); the shell's prompt pass and serial steps commit only forwarded tokens.
        lookup_has_primary: bool = false,
        /// Called once per prompt at its last trunk chunk, after the logits' argmax and before the seed (`PromptTail`).
        tail: ?PromptTail = null,
        /// The verify chunk's eval, bound at `init` (`Config.draft_ahead`): the eval alone (stock), or committed, the
        /// next draft built ahead while it runs, then waited (DRAFT_AHEAD).
        verify_wait: *const VerifyWaitFn = verifyWaitStock,
        /// DRAFT_AHEAD: the block built during the last verify's wait, and whether that verify's outcome installed it.
        ahead: ?Ahead = null,
        /// DRAFT_AHEAD's persistent inputs (kept from the first prebuild to `deinit`).
        ahead_in: ?H.AheadIn = null,

        const VerifyWaitFn = fn (self: *Self, ev: []const T, verify_hidden: T, n_block: u32, whole: bool) anyerror!void;
        const max_ahead_stages = 8;
        const Ahead = struct { draft: H.Draft, caches: [max_ahead_stages]H.Cache, n_block: u32, ready: bool = false };

        /// The draft depth and the widest verify a request under `cfg` reaches with `head`.
        pub const Shapes = struct { k_cap: u32, max_rows: u32 };
        pub fn shapesOf(head: *const H, cfg: Config) Shapes {
            const k_cap = @min(cfg.k_request, head.blockSize());
            const extra: u32 = if (cfg.lookup) |l| (if (k_cap == ds.Lookup.key_len) l.extra_tokens else 0) else 0;
            return .{ .k_cap = k_cap, .max_rows = k_cap + extra + 1 };
        }

        pub fn init(g: *G, model: *M, head: *const H, st: *M.State, caches: []H.Cache, cfg: Config) Self {
            const sh = shapesOf(head, cfg);
            var self: Self = .{ .g = g, .model = model, .head = head, .st = st, .caches = caches, .cfg = cfg, .k_cap = sh.k_cap, .max_rows = sh.max_rows };
            self.stats.speculative_depth = sh.max_rows - 1;
            if (cfg.draft_ahead and sh.k_cap > 0 and caches.len <= max_ahead_stages) self.verify_wait = verifyWaitAhead;
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.dropAhead();
            if (self.ahead_in) |in| {
                self.g.release(in.patch);
                self.g.release(in.prev);
            }
            if (self.main_h) |x| self.g.release(x);
            if (self.lookup) |*l| l.deinit();
        }

        /// DRAFT_AHEAD as bound at `init`.
        pub fn draftAhead(self: *const Self) bool {
            return self.verify_wait == &verifyWaitAhead;
        }

        fn verifyWaitStock(self: *Self, ev: []const T, _: T, _: u32, _: bool) !void {
            try self.g.evalAll(ev);
        }

        /// DRAFT_AHEAD: the verify committed, the next block built on the host while the GPU runs it (a verify in one
        /// chunk: `whole`), then the wait.
        fn verifyWaitAhead(self: *Self, ev: []const T, verify_hidden: T, n_block: u32, whole: bool) !void {
            if (whole) {
                try self.g.asyncEval(ev);
                try self.prebuild(verify_hidden, n_block);
            }
            try self.g.evalAll(ev);
        }

        /// The next draft block for "every draft accepted": the windows seeded with all `n_block` verified rows (into
        /// the ahead caches; the loop's untouched), the main row = the last verified row, the block over the persistent
        /// inputs (primary not yet written). Its outputs kept; nothing committed.
        fn prebuild(self: *Self, verify_hidden: T, n_block: u32) !void {
            const g = self.g;
            self.dropAhead();
            const in = self.ahead_in orelse blk: {
                var buf: [64 * 1024]u8 = undefined;
                var fba = std.heap.FixedBufferAllocator.init(&buf);
                const x = try self.head.aheadInputs(g, fba.allocator(), self.model.embed);
                self.ahead_in = x;
                break :blk x;
            };
            var ah: Ahead = .{ .draft = undefined, .caches = @splat(.{}), .n_block = n_block };
            const n_st = self.caches.len;
            try self.head.seedMainInto(g, try sliceRows(g, verify_hidden, 0, @intCast(n_block)), self.caches, ah.caches[0..n_st]);
            errdefer for (ah.caches[0..n_st]) |*c| c.deinit(g);
            const mh = try sliceRows(g, verify_hidden, @intCast(n_block - 1), @intCast(n_block));
            var d = try self.head.draftBlockAhead(g, mh, try self.head.aheadInput(g, self.model.embed, in), ah.caches[0..n_st], self.model.head);
            d.ids = g.keep(d.ids);
            d.logits = g.keep(d.logits);
            d.conf = g.keep(d.conf);
            ah.draft = d;
            self.ahead = ah;
        }

        /// At the commit: the ahead block's outcome happened (every verified row kept) -> its windows replace the
        /// loop's and it is the next draft; otherwise it is discarded, never committed.
        fn takeAhead(self: *Self, kept: u32) bool {
            if (self.ahead == null) return false;
            const ah = &self.ahead.?;
            if (kept + 1 != ah.n_block) {
                self.dropAhead();
                return false;
            }
            for (self.caches, ah.caches[0..self.caches.len]) |*c, *n| {
                if (c.window) |w| self.g.release(w);
                c.* = n.*;
                n.* = .{};
            }
            ah.ready = true;
            return true;
        }

        /// Releases an ahead block that will not run (its windows when not installed, its kept outputs).
        fn dropAhead(self: *Self) void {
            if (self.ahead == null) return;
            const ah = &self.ahead.?;
            for (ah.caches[0..self.caches.len]) |*c| c.deinit(self.g);
            releaseDraft(self.g, &ah.draft);
            self.ahead = null;
        }

        fn releaseDraft(g: *G, d: *H.Draft) void {
            g.release(d.ids);
            g.release(d.logits);
            g.release(d.conf);
            draft_routes.drop(G, g, &d.stage_ids);
        }

        fn setMain(self: *Self, x: T) void {
            const k = self.g.keep(x);
            if (self.main_h) |old| self.g.release(old);
            self.main_h = k;
        }

        fn sliceRows(g: *G, x: T, lo: c_int, hi: c_int) !T {
            const s = g.shapeOf(x);
            return g.slice(x, &.{ 0, lo, 0 }, &.{ s.d[0], hi, s.d[2] }, &.{ 1, 1, 1 });
        }

        /// Rows `lo..hi` (axis 1) as a gather: a fresh buffer of those rows alone (a slice is a view that
        /// keeps its whole source buffer alive; a gather never aliases or takes over its input).
        fn copyRows(g: *G, x: T, lo: c_int, hi: c_int) !T {
            return g.take(x, try g.arange(@floatFromInt(lo), @floatFromInt(hi), 1, .int32), 1);
        }

        fn windows(self: *const Self, ws: *[8]T) []const T {
            var n: usize = 0;
            for (self.caches) |c| if (c.window) |w| {
                ws[n] = w;
                n += 1;
            };
            return ws[0..n];
        }

        fn evalWindows(self: *Self) !void {
            var ws: [8]T = undefined;
            try self.g.evalAll(self.windows(&ws));
        }

        fn dispatchWindows(self: *Self) !void {
            var ws: [8]T = undefined;
            try self.g.asyncEval(self.windows(&ws));
        }

        fn isStop(self: *const Self, tok: u32) bool {
            return std.mem.indexOfScalar(u32, self.cfg.stop_ids, tok) != null;
        }

        /// `dspark_generate`'s prompt pass: the forwards, the primary pick,
        /// `_seed_prefill_state` (one seed over every prompt row).
        pub fn prefill(self: *Self, a: std.mem.Allocator, ex: anytype, prompt: []const u32) !u32 {
            return (try self.prefillImpl(a, ex, prompt, false)).primary;
        }

        /// `prefill` for the shell (its Generator picks every token itself): the same seed, the last
        /// row's logits handed back (kept; the caller releases them), the lookup's history the
        /// forwarded prompt only.
        pub fn prefillLogits(self: *Self, a: std.mem.Allocator, ex: anytype, prompt: []const u32) !T {
            return (try self.prefillImpl(a, ex, prompt, true)).logits.?;
        }

        /// A serial forward of committed `ids` that keeps the strategy in step: their main taps
        /// append to the draft windows (`seedMain`, as the prompt's did), the last one is the next
        /// draft's main row, the lookup commits them. The shell's prompt is `prefillLogits` over all
        /// but its last token and this over the last: the seed equals `prefill`'s over the whole
        /// prompt (the windows append per row). The last row's logits, kept.
        pub fn extendLogits(self: *Self, a: std.mem.Allocator, ex: anytype, ids: []const u32) !T {
            _ = a;
            const g = self.g;
            const r = try self.model.forward(g, self.st, ids, .{ .logits = .last, .main_hidden = true }, ex, graph.NoProbe{});
            try M.fence(g, self.st, &.{ r.logits.?, r.main_hidden.? });
            try ex.flush();
            // A split prompt's final part (`tail` set only then): after its routed work, before the seed.
            if (self.tail) |t| try t.run(t.ctx);
            const logits = g.keep(r.logits.?);
            errdefer g.release(logits);
            const mains = r.main_hidden.?;
            try self.head.seedMain(g, mains, self.caches);
            const n: c_int = @intCast(ids.len);
            self.setMain(try sliceRows(g, mains, n - 1, n));
            try self.evalWindows();
            try g.evalAll(&.{self.main_h.?});
            if (self.lookup) |*l| try l.appendCommitted(if (self.lookup_has_primary) ids[1..] else ids);
            self.lookup_has_primary = false;
            g.reset();
            return logits;
        }

        const Prefilled = struct { primary: u32, logits: ?T };

        fn prefillImpl(self: *Self, a: std.mem.Allocator, ex: anytype, prompt: []const u32, shell: bool) !Prefilled {
            const keep_logits = shell;
            const g = self.g;
            var kept: ?T = null;
            errdefer if (kept) |x| g.release(x);
            const chunk = self.cfg.prompt_chunk;
            var mains: std.ArrayList(T) = .empty;
            defer {
                for (mains.items) |x| g.release(x);
                mains.deinit(a);
            }
            var i: usize = 0;
            while (i < prompt.len) {
                const end = @min(i + chunk, prompt.len);
                const last = end == prompt.len;
                const r = try self.model.forward(g, self.st, prompt[i..end], .{ .logits = if (last) .last else .none, .main_hidden = true }, ex, graph.NoProbe{});
                try M.fence(g, self.st, &.{ if (last) r.logits.? else r.hidden, r.main_hidden.? });
                try ex.flush();
                try mains.append(a, g.keep(r.main_hidden.?));
                if (last) {
                    self.primary = try g.hostArgmax(r.logits.?);
                    if (keep_logits) kept = g.keep(r.logits.?);
                    if (self.tail) |t| try t.run(t.ctx);
                }
                g.reset();
                i = end;
            }
            const all = if (mains.items.len == 1) mains.items[0] else try g.concat(mains.items, 1);
            try self.head.seedMain(g, all, self.caches);
            const n: c_int = @intCast(prompt.len);
            // The seed keeps copies, not views: the last row sliced from the prompt's main taps and each
            // stage's window sliced from its main KV would hold those whole buffers (P x 67,584 B, 1.11 GB
            // at 16K) until the first round replaces them, through the phase change. A gather writes fresh
            // buffers of the views' own rows (the same bytes); the prompt's go at the reset below.
            self.setMain(try copyRows(g, all, n - 1, n));
            for (self.caches) |*c| if (c.window) |w| {
                c.window = g.keep(try copyRows(g, w, 0, g.shapeOf(w).dim(1)));
                g.release(w);
            };
            try self.evalWindows();
            try g.evalAll(&.{self.main_h.?});
            g.reset();
            if (self.cfg.lookup) |l| {
                // Reserved to the state's admitted length when bounded (the request's positions).
                self.lookup = try ds.Lookup.init(a, prompt, l.minimum_context, l.extra_tokens, self.st.max_len orelse 0);
                // The cell's run commits its own primary; the shell's Generator commits the tokens it picks.
                if (!shell) try self.lookup.?.appendCommitted(&.{self.primary});
            }
            self.lookup_has_primary = !shell;
            return .{ .primary = self.primary, .logits = kept };
        }

        /// The target's decision on a verify chunk, as a graph over its (lazy) logits: the rows' argmax and,
        /// for the typical tier, `_TYPICAL_DECIDE`'s flags on the drafted rows. The verify's eval realises it.
        const Decision = struct { tt: T, typical: ?T = null, width: u32, drafted: u32 = 0 };

        fn decideGraph(self: *Self, logits: T, drafts: []const u32, rows: [2]u32, k_eff: u32) !Decision {
            const g = self.g;
            const s = g.shapeOf(logits);
            const vocab = s.dim(-1);
            const width: u32 = @intCast(s.dim(1));
            const row2 = try g.reshape(logits, &.{ @intCast(width), vocab });
            const tt = try g.argmax(row2, -1);
            const typ = switch (self.cfg.acceptance) {
                .greedy => return .{ .tt = tt, .width = width },
                .typical => |t| t,
            };
            const drafted: u32 = @min(k_eff -| rows[0], width);
            if (drafted == 0) return .{ .tt = tt, .width = width };
            const lg = try g.astype(try g.slice(row2, &.{ 0, 0 }, &.{ @intCast(drafted), vocab }, &.{ 1, 1 }), .float32);
            const log_p = try g.sub(lg, try g.logsumexp(lg, -1, true));
            const p = try g.exp(log_p);
            const entropy = try g.neg(try g.sum(try g.mul(p, log_p), -1, false));
            const floor = try g.minimum(try g.scalar(typ.eps, .float32), try g.mul(try g.scalar(typ.delta, .float32), try g.exp(try g.neg(entropy))));
            var idb: [ds.max_block]i32 = undefined;
            for (idb[0..drafted], drafts[rows[0]..][0..drafted]) |*d, v| d.* = @intCast(v);
            const ids = try g.hostArray(std.mem.sliceAsBytes(idb[0..drafted]), &.{ @intCast(drafted), 1 }, .int32);
            const typical = try g.greater(try g.reshape(try g.takeAlongAxis(p, ids, -1), &.{-1}), floor);
            return .{ .tt = tt, .typical = typical, .width = width, .drafted = drafted };
        }

        /// The evaluated decision's host values: the argmax per row into `target`; the typical tier's flags
        /// (empty when no row of the chunk is drafted), null on the greedy tier.
        fn decideRead(self: *Self, d: Decision, target: []u32, flags: []bool) !?[]const bool {
            _ = try self.g.hostU32(d.tt, target[0..d.width]);
            return switch (self.cfg.acceptance) {
                .greedy => null,
                .typical => if (d.typical) |ty| try self.g.hostBool(ty, flags[0..d.drafted]) else flags[0..0],
            };
        }

        /// The install warm-up (the lane's pipelines warmed at install): every
        /// compiled region traced once at the shapes the cycles serve, before the
        /// first request. On a scratch state and scratch draft windows: a verify
        /// forward of each row count 1 .. `max_rows` (logits every row, main
        /// hidden kept), the windows seeded from each, then one draft block.
        /// Nothing it builds outlives it; the routed calls go through `ex`.
        /// `peaks` (optional, `max_rows + 1` entries): the transient bytes each
        /// shape raised MLX's high-water mark by, measured once here (row counts
        /// 1 .. `max_rows`, then the draft block): the bill's per-shape terms.
        pub fn warm(self: *Self, a: std.mem.Allocator, ex: anytype, peaks: ?[]u64) !void {
            return warmShapes(self.g, a, self.model, self.head, self.k_cap, self.max_rows, ex, peaks);
        }

        /// The install warm-up of a model and head a caller constructs (the
        /// module's init, before any request): every forward width 1 ..
        /// max(`widths`, the widest verify of a request under `cfg`) (the
        /// compiled regions trace per width up to their row bound), then a draft
        /// block when `cfg` drafts. Returns the per-shape peaks (owned): one per
        /// width, then the draft block's (0 when none).
        pub fn warmFor(g: *G, a: std.mem.Allocator, model: *M, head: *const H, ex: anytype, cfg: Config, widths: u32) ![]u64 {
            const sh = shapesOf(head, cfg);
            const rows = @max(sh.max_rows, widths);
            const peaks = try a.alloc(u64, rows + 1);
            errdefer a.free(peaks);
            @memset(peaks, 0);
            try warmShapes(g, a, model, head, sh.k_cap, rows, ex, peaks);
            return peaks;
        }

        const max_warm_rows = 64;

        fn warmShapes(g: *G, a: std.mem.Allocator, model: *M, head: *const H, k_cap: u32, rows: u32, ex: anytype, peaks: ?[]u64) !void {
            if (rows > max_warm_rows) return error.WarmTooWide;
            if (peaks) |p| std.debug.assert(p.len >= rows + @intFromBool(k_cap > 0));
            var st = try model.newState();
            defer st.deinit(g, a);
            var caches: [8]H.Cache = @splat(.{});
            const n_st = head.nStages();
            defer for (caches[0..n_st]) |*x| x.deinit(g);
            var ids: [max_warm_rows]u32 = @splat(1);
            var main: ?T = null;
            defer if (main) |x| g.release(x);
            for (1..rows + 1) |m| {
                const base = g.peakFrom();
                const r = try model.forward(g, &st, ids[0..m], .{ .logits = .all, .main_hidden = true }, ex, graph.NoProbe{});
                try g.evalAll(&.{ r.logits.?, r.main_hidden.? });
                try head.seedMain(g, r.main_hidden.?, caches[0..n_st]);
                var ws: [8]T = undefined;
                var nw: usize = 0;
                for (caches[0..n_st]) |c| if (c.window) |w| {
                    ws[nw] = w;
                    nw += 1;
                };
                try g.evalAll(ws[0..nw]);
                if (main == null) {
                    main = g.keep(try sliceRows(g, r.main_hidden.?, 0, 1));
                    try g.evalAll(&.{main.?});
                }
                if (peaks) |p| p[m - 1] = g.peakAbove(base);
                try ex.flush();
                g.reset();
            }
            if (k_cap > 0) {
                const base = g.peakFrom();
                const d = try head.draftBlock(g, main.?, 1, caches[0..n_st], model.embed, model.head);
                try g.evalAll(&.{ d.ids, d.logits, d.conf });
                if (peaks) |p| p[rows] = g.peakAbove(base);
                g.reset();
            }
        }

        /// A logged verify chunk's rows, for the tie-flip rule: the top two ids
        /// and f32 logits of each row and the row's rms (one extra sync, only
        /// when the caller asks: `CycleLog.want_top`).
        fn topTwo(self: *Self, logits: T, width: u32, ids: [][2]u32, vals: [][2]f32, rms: []f32) !void {
            const g = self.g;
            const vocab = g.shapeOf(logits).dim(-1);
            const w: c_int = @intCast(width);
            const rows = try g.astype(try g.reshape(logits, &.{ w, vocab }), .float32);
            const first_id = try g.argmax(rows, -1);
            const m1 = try g.max(rows, -1, false);
            const col = try g.reshape(try g.arange(0, @floatFromInt(vocab), 1, .uint32), &.{ 1, vocab });
            const masked = try g.where(try g.equal(col, try g.reshape(first_id, &.{ w, 1 })), try g.scalar(-std.math.inf(f64), .float32), rows);
            const second_id = try g.argmax(masked, -1);
            const m2 = try g.max(masked, -1, false);
            const r = try g.sqrt(try g.mean(try g.square(rows), -1, false));
            try g.evalAll(&.{ first_id, m1, second_id, m2, r });
            var a1: [ds.max_block + 1]u32 = undefined;
            var a2: [ds.max_block + 1]u32 = undefined;
            var v1: [ds.max_block + 1]f32 = undefined;
            var v2: [ds.max_block + 1]f32 = undefined;
            _ = try g.hostU32(first_id, a1[0..width]);
            _ = try g.hostU32(second_id, a2[0..width]);
            _ = try g.hostF32(m1, v1[0..width]);
            _ = try g.hostF32(m2, v2[0..width]);
            _ = try g.hostF32(r, rms[0..width]);
            for (ids[0..width], vals[0..width], 0..) |*id, *v, i| {
                id.* = .{ a1[i], a2[i] };
                v.* = .{ v1[i], v2[i] };
            }
        }

        /// One cycle; returns null to continue, or how the run finished.
        pub fn cycle(self: *Self, ex: anytype, out: *std.ArrayList(u32), a: std.mem.Allocator, log: ?*CycleLog) !?Finish {
            return self.cycleStamped(ex, out, a, log, {});
        }

        /// `cycle` with a stamper's marks at its phase ends (a decode-profile run only; `{}` compiles
        /// them out, as `cycle` passes).
        /// One cycle's commit, the part every driver shares: the drafts (the head's, the lookup's
        /// extension), the verify of [primary, drafts] in chunks of `max_rows`, the decision
        /// (greedy or typical, the correction), the target trimmed to [primary, the first `kept`
        /// accepted drafts] (`kept` = min(accepted, `accepted_cap`)), the draft windows seeded.
        /// `next` is the token after the kept run: the correction, or past a cap the next accepted
        /// draft. Neither is in the state.
        const Core = struct { drafts: [ds.max_block]u32, n_drafts: u32, kept: u32, next: u32, verify_hidden: T };

        /// A0 (c)'s D1, profile builds only, in the decode's first cycle before its own draft block:
        ///   1. the state that block's eval would otherwise realise (the main row and the stage windows: the prompt's
        ///      seed), evaluated and timed apart;
        ///   2. one draft block whose output is dropped: its build and wait timed, the bytes MLX took for it counted,
        ///      and its arrays freed (`resetTo`) so the cycle's own block can reuse their buffers.
        /// `draftBlock` reads the caches as `[]const Cache` and the head as const, so the cycle's block and every id after
        /// it are unchanged. Returns MLX's (active, cache) bytes before the cycle's own block.
        fn droppedDraft(self: *Self) ![2]u64 {
            const g = self.g;
            const dev = comptime G == ops.MlxOps;
            var pend: [9]T = undefined;
            var np: usize = 0;
            pend[np] = self.main_h.?;
            np += 1;
            for (self.caches) |c| if (c.window) |w| {
                if (np == pend.len) return error.TooManyStages;
                pend[np] = w;
                np += 1;
            };
            var t = dt.now();
            try g.evalAll(pend[0..np]);
            first_cycle.cycle1_pending_ns = dt.now() - t;
            const m = g.mark();
            const before = first_cycle.memNow(dev);
            t = dt.now();
            var d = try self.head.draftBlock(g, self.main_h.?, self.primary, self.caches, self.model.embed, self.model.head);
            const sig = try g.sigmoid(try g.astype(d.conf, .float32));
            const built = dt.now();
            try g.evalAll(&.{ d.ids, sig });
            const waited = dt.now();
            draft_routes.drop(G, g, &d.stage_ids);
            first_cycle.recordDraft(0, built - t, waited - built, before, first_cycle.memNow(dev));
            g.resetTo(m);
            return first_cycle.memNow(dev);
        }

        fn core(self: *Self, ex: anytype, log: ?*CycleLog, stamp: anytype, accepted_cap: u32) !Core {
            const g = self.g;
            const st = &self.stats;
            var drafts_buf: [ds.max_block]u32 = undefined;
            var drafts: []const u32 = &.{};
            var k_eff: u32 = 0;
            var native: [ds.max_block]u32 = undefined;
            timeline.cycleBegin();
            var tt = dt.now();
            if (self.k_cap > 0) {
                // A0 (c)'s D1 (profile builds, the decode's first cycle): the pending state and a dropped draft block
                // first, timed apart and charged to no bucket; this block is then the second (`droppedDraft`).
                const doubled = first_cycle.enabled and first_cycle.firstDraft();
                var mem0: [2]u64 = .{ 0, 0 };
                if (comptime first_cycle.enabled) {
                    if (doubled) {
                        mem0 = try self.droppedDraft();
                        tt = dt.now();
                    }
                }
                const t_build = tt;
                // DRAFT_AHEAD: the block the last verify's outcome installed, its primary written in place first.
                const ahead = if (self.ahead) |ah| (if (ah.ready) ah else null) else null;
                if (ahead != null) try self.head.aheadWrite(g, self.model.embed, self.ahead_in.?, self.primary);
                var d = if (ahead) |ah| ah.draft else try self.head.draftBlock(g, self.main_h.?, self.primary, self.caches, self.model.embed, self.model.head);
                const bs = self.head.blockSize();
                // CYCLE_TRIM draftfold: the confidence sigmoid is realised by the draft's own eval (one
                // sync), which also realises the main row and window update the previous commit left.
                const sig = try g.sigmoid(try g.astype(d.conf, .float32));
                tt = dt.charge(.draft_build, tt);
                const t_wait = tt;
                try g.evalAll(&.{ d.ids, sig });
                tt = dt.charge(.draft_wait, tt);
                // Profile builds: the stages' routed ids, realised by that eval, read in place.
                try draft_routes.take(G, g, &d.stage_ids);
                if (comptime first_cycle.enabled) {
                    if (doubled) first_cycle.recordDraft(1, t_wait - t_build, tt - t_wait, mem0, first_cycle.memNow(G == ops.MlxOps));
                }
                _ = try g.hostU32(d.ids, native[0..bs]);
                var conf: [ds.max_block]f32 = undefined;
                _ = try g.hostF32(sig, conf[0..bs]);
                // The ahead block's kept outputs are read: released (its arrays freed with the cycle's).
                if (ahead != null) {
                    releaseDraft(g, &d);
                    self.ahead = null;
                }
                k_eff = ds.effectiveDraftLen(conf[0..bs], self.k_cap, self.cfg.confidence_threshold);
                if (log) |lg| {
                    lg.k_native = k_eff;
                    @memcpy(lg.conf[0..bs], conf[0..bs]);
                }
                drafts = if (self.lookup) |*l| l.extend(native[0..k_eff], &drafts_buf) else blk: {
                    @memcpy(drafts_buf[0..k_eff], native[0..k_eff]);
                    break :blk drafts_buf[0..k_eff];
                };
                if (log) |lg| {
                    lg.n_native = bs;
                    @memcpy(lg.native[0..bs], native[0..bs]);
                }
                k_eff = @intCast(drafts.len);
            }
            tt = dt.charge(.draft_host, tt);
            mark(stamp, .draft);
            // Verify [primary, drafts] in chunks of max_rows, stopping at the correction.
            var block: [ds.max_block + 1]u32 = undefined;
            block[0] = self.primary;
            @memcpy(block[1..][0..drafts.len], drafts);
            const n_block: u32 = @intCast(drafts.len + 1);
            var o: ds.Outcome = .{};
            var hiddens: [ds.max_block + 1]T = undefined;
            var n_hidden: usize = 0;
            var start: u32 = 0;
            timeline.verifyBegin();
            while (start < n_block) {
                const end = @min(start + self.max_rows, n_block);
                const r = try self.model.forward(g, self.st, block[start..end], .{ .logits = .all, .main_hidden = true }, ex, graph.NoProbe{});
                // The decision folded into the verify's eval (one sync): its graph over the lazy logits.
                const dec = try self.decideGraph(r.logits.?, drafts, .{ start, end }, k_eff);
                tt = dt.charge(.verify, tt);
                var ev: [3]T = .{ dec.tt, r.main_hidden.?, undefined };
                var n_ev: usize = 2;
                if (dec.typical) |ty| {
                    ev[2] = ty;
                    n_ev = 3;
                }
                try self.verify_wait(self, ev[0..n_ev], r.main_hidden.?, n_block, start == 0 and end == n_block);
                tt = dt.charge(.verify_eval, tt);
                mark(stamp, .verify);
                st.verify_calls += 1;
                hiddens[n_hidden] = r.main_hidden.?;
                n_hidden += 1;
                var target: [ds.max_block + 1]u32 = undefined;
                var flags: [ds.max_block + 1]bool = undefined;
                const typ = try self.decideRead(dec, &target, &flags);
                if (log) |lg| {
                    if (lg.want_top) try self.topTwo(r.logits.?, end - start, lg.top_ids[lg.n_targets..], lg.top_logits[lg.n_targets..], lg.rms[lg.n_targets..]);
                    @memcpy(lg.targets[lg.n_targets..][0 .. end - start], target[0 .. end - start]);
                    lg.n_targets += end - start;
                    if (typ) |ty| {
                        @memcpy(lg.flags[lg.n_flags..][0..ty.len], ty);
                        lg.n_flags += @intCast(ty.len);
                    }
                }
                const done = ds.acceptChunk(&o, st, drafts, k_eff, .{ start, end }, target[0 .. end - start], typ);
                tt = dt.charge(.accept, tt);
                mark(stamp, .decide);
                start = end;
                if (done) break;
            }
            timeline.verifyEnd();
            const correction = o.correction.?; // acceptChunk sets it on the chunk that ends the verify
            st.endCycle(o, k_eff);
            const verify_hidden = if (n_hidden == 1) hiddens[0] else try g.concat(hiddens[0..n_hidden], 1);
            const kept = @min(o.accepted, accepted_cap);
            const next = if (kept < o.accepted) drafts[kept] else correction;
            const trimmed = o.verified - (kept + 1);
            // Commit: keep [primary, d1 .. d_kept] in the target, seed the draft windows. CYCLE_TRIM gap:
            // the window update is dispatched, not waited; the round boundary's host work (the tail, the
            // caller's, the next draft's build) runs under it, and the next draft's eval waits for it.
            try self.model.trim(g, self.st, trimmed);
            tt = dt.charge(.trim, tt);
            // DRAFT_AHEAD: the windows seeded ahead when every verified row is kept; otherwise seeded here (stock).
            if (!self.takeAhead(kept)) try self.head.seedMain(g, try sliceRows(g, verify_hidden, 0, @intCast(kept + 1)), self.caches);
            tt = dt.charge(.seed, tt);
            try self.dispatchWindows();
            _ = dt.charge(.window, tt);
            mark(stamp, .commit);
            if (log) |lg| {
                lg.primary = self.primary;
                lg.k_eff = k_eff;
                lg.accepted = o.accepted;
                lg.correction = correction;
                lg.verified = o.verified;
                lg.trimmed = trimmed;
                @memcpy(lg.drafts[0..drafts.len], drafts);
            }
            var c: Core = .{ .drafts = undefined, .n_drafts = @intCast(drafts.len), .kept = kept, .next = next, .verify_hidden = verify_hidden };
            @memcpy(c.drafts[0..drafts.len], drafts);
            return c;
        }

        /// The cell's emitter over `core`: emit the accepted drafts and the correction up to
        /// `cfg.max_tokens`; a stop id is emitted, then the run ends.
        pub fn cycleStamped(self: *Self, ex: anytype, out: *std.ArrayList(u32), a: std.mem.Allocator, log: ?*CycleLog, stamp: anytype) !?Finish {
            const g = self.g;
            const st = &self.stats;
            const c = try self.core(ex, log, stamp, std.math.maxInt(u32));
            const tt = dt.now();
            var emitted: [ds.max_block + 1]u32 = undefined;
            @memcpy(emitted[0..c.kept], c.drafts[0..c.kept]);
            emitted[c.kept] = c.next;
            const base = out.items.len;
            var finish: ?Finish = null;
            for (emitted[0 .. c.kept + 1]) |tok| {
                if (out.items.len >= self.cfg.max_tokens) break;
                try out.append(a, tok);
                if (self.isStop(tok)) {
                    finish = .stop;
                    break;
                }
            }
            st.generated_tokens = @intCast(out.items.len);
            if (self.lookup) |*l| try l.appendCommitted(out.items[base..]);
            if (finish == null and out.items.len >= self.cfg.max_tokens) finish = .length;
            if (finish == null) {
                self.primary = c.next;
                // The next draft's main row, realised by that draft's eval.
                self.setMain(try sliceRows(g, c.verify_hidden, @intCast(c.kept), @intCast(c.kept + 1)));
            }
            try ex.flush();
            try cycleEnd(ex);
            g.reset();
            _ = dt.charge(.tail, tt);
            dt.countCycle();
            mark(stamp, .tail);
            return finish;
        }

        /// A decode cycle's end for the routed source (option (b)'s clock), when the hook has one.
        fn cycleEnd(ex: anytype) !void {
            if (comptime @hasDecl(@TypeOf(ex.*), "cycleEnd")) try ex.cycleEnd();
        }

        /// The shell's round over `core` (the Generator's v2 spec invariant: the state holds the
        /// prompt and every emitted token; `t1`, the next token, is not in it). Verifies [t1,
        /// drafts], keeps at most `accepted_cap` accepted drafts, and returns [t1, the kept drafts]
        /// (owned by `a`) with the next token (the correction; not in the state). It never stops:
        /// EOS, stop strings and the token budget are the caller's.
        pub const Round = struct { tokens: []u32, accepted: u32, next_token: u32 };

        pub fn round(self: *Self, ex: anytype, a: std.mem.Allocator, t1: u32, accepted_cap: u32, log: ?*CycleLog, stamp: anytype) !Round {
            const g = self.g;
            if (self.lookup) |*l| if (!self.lookup_has_primary) try l.appendCommitted(&.{t1});
            self.lookup_has_primary = true;
            self.primary = t1;
            const c = try self.core(ex, log, stamp, accepted_cap);
            const tt = dt.now();
            const tokens = try a.alloc(u32, c.kept + 1);
            errdefer a.free(tokens);
            tokens[0] = t1;
            @memcpy(tokens[1..], c.drafts[0..c.kept]);
            self.stats.generated_tokens += c.kept + 1;
            if (self.lookup) |*l| {
                try l.appendCommitted(c.drafts[0..c.kept]);
                try l.appendCommitted(&.{c.next});
            }
            self.primary = c.next;
            // The next draft's main row, realised by that draft's eval.
            self.setMain(try sliceRows(g, c.verify_hidden, @intCast(c.kept), @intCast(c.kept + 1)));
            try ex.flush();
            try cycleEnd(ex);
            g.reset();
            _ = dt.charge(.tail, tt);
            dt.countCycle();
            mark(stamp, .tail);
            return .{ .tokens = tokens, .accepted = c.kept, .next_token = c.next };
        }

        /// The run after `prefill`: cycles until the token cap or a stop id.
        pub fn run(self: *Self, ex: anytype, out: *std.ArrayList(u32), a: std.mem.Allocator) !Finish {
            if (self.isStop(self.primary)) return .stop;
            while (true) {
                if (try self.cycle(ex, out, a, null)) |f| return f;
            }
        }
    };
}

// ── Tests ──

const testing = std.testing;
const TraceOps = ops.TraceOps;
const routes = @import("deepseek_v41_routes.zig");
const xp = @import("deepseek_v41_experts.zig");

/// Host reads of a scripted run: routed ids from a PRNG, the prompt's pick,
/// and per cycle the draft ids, their sigmoid confidences, the verify
/// targets and (typical) the flags, in the loop's read order.
const Script = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(20260928),
    n_experts: u16,
    pick: u32,
    u32s: []const []const u32,
    f32s: []const []const f32,
    bools: []const []const bool = &.{},
    nu: usize = 0,
    nf: usize = 0,
    nb: usize = 0,

    fn values(self: *Script) TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax, .u32s = u32s_, .f32s = f32s_, .bools = bools_ };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        for (out) |*o| o.* = s.rng.random().uintLessThan(u16, s.n_experts);
    }
    fn argmax(ctx: *anyopaque) anyerror!u32 {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return s.pick;
    }
    fn next(comptime E: type, list: []const []const E, i: *usize, out: []E) !void {
        if (i.* >= list.len) return error.ScriptExhausted;
        if (list[i.*].len != out.len) return error.ScriptShape;
        @memcpy(out, list[i.*]);
        i.* += 1;
    }
    fn u32s_(ctx: *anyopaque, out: []u32) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(u32, s.u32s, &s.nu, out);
    }
    fn f32s_(ctx: *anyopaque, out: []f32) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(f32, s.f32s, &s.nf, out);
    }
    fn bools_(ctx: *anyopaque, out: []bool) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(bool, s.bools, &s.nb, out);
    }
};

const Rig = struct {
    m: *mdl.Mini,
    g: TraceOps,
    lookup: mdl.SpecLookup,
    model: *Loop(TraceOps).M,
    head: *Loop(TraceOps).H,
    st: Loop(TraceOps).M.State,
    caches: [4]Loop(TraceOps).H.Cache = @splat(.{}),
    src: xp.FakeSource,
    ex: xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath),

    fn init(rig: *Rig) !void {
        const a = testing.allocator;
        rig.m = try mdl.Mini.init();
        rig.g = TraceOps.init(a);
        rig.caches = @splat(.{});
        rig.lookup = .{ .g = &rig.g, .spec = rig.m.spec };
        const c = &rig.m.c;
        rig.model = try Loop(TraceOps).M.init(a, &rig.g, rig.m.c, try routes.parse(&.{}, null), &rig.lookup, &rig.m.src);
        rig.head = try Loop(TraceOps).H.init(a, &rig.g, rig.m.c, .{}, &rig.lookup);
        rig.st = try rig.model.newState();
        var rows: [8]u32 = @splat(0);
        var grown: [8]u32 = @splat(@intCast(c.n_routed_experts));
        rig.src = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows[0..c.n_layers] });
        rig.ex = try xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath).init(a, &rig.g, &rig.src, .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) }, c);
        try rig.ex.grow(&rig.g, grown[0..c.n_layers]);
    }

    fn deinit(rig: *Rig) void {
        for (rig.caches[0..rig.head.nStages()]) |*c| c.deinit(&rig.g);
        rig.ex.deinit();
        rig.src.deinit();
        rig.st.deinit(&rig.g, testing.allocator);
        rig.head.deinit(&rig.g);
        rig.model.deinit(&rig.g);
        rig.g.deinit();
        rig.m.deinit();
    }
};

test "dsv41 dspark loop: the mini model's cycles draft, verify, accept, trim and seed as _decode_cycles does" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    const c = &rig.m.c;
    try testing.expectEqual(@as(u32, 2), rig.head.blockSize());
    // Four cycles (block 2, confidence 0.5, greedy): a reject at depth 1; an early
    // stop to one draft and its bonus; a stop to one draft rejected; both accepted
    // (the bonus is cut by max_tokens 6).
    var script: Script = .{
        .n_experts = @intCast(c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 2), lp.k_cap);
    try testing.expectEqual(@as(u32, 3), lp.max_rows);
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    try testing.expectEqual(@as(u32, 3), try lp.prefill(a, &rig.ex, &prompt));
    try testing.expectEqual(@as(u32, 20), rig.caches[0].offset);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var logs: [4]CycleLog = @splat(.{ .primary = 0 });
    for (&logs, 0..) |*lg, i| {
        const f = try lp.cycle(&rig.ex, &out, a, lg);
        try testing.expectEqual(@as(?Finish, if (i == 3) .length else null), f);
    }
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    const want = [_][4]u32{ .{ 2, 1, 9, 1 }, .{ 1, 1, 12, 0 }, .{ 1, 0, 20, 1 }, .{ 2, 2, 40, 0 } };
    for (logs, want) |lg, w| {
        try testing.expectEqual(w[0], lg.k_eff);
        try testing.expectEqual(w[1], lg.accepted);
        try testing.expectEqual(w[2], lg.correction);
        try testing.expectEqual(w[3], lg.trimmed);
    }
    const st = lp.stats;
    try testing.expectEqualSlices(u32, &.{ 4, 2 }, st.drafted_by_depth[0..2]);
    try testing.expectEqualSlices(u32, &.{ 3, 1 }, st.accepted_by_depth[0..2]);
    try testing.expectEqual(@as(u32, 4), st.cycles);
    try testing.expectEqual(@as(u32, 4), st.verify_calls);
    try testing.expectEqual(@as(u32, 2), st.correction_tokens);
    try testing.expectEqual(@as(u32, 2), st.bonus_tokens);
    try testing.expectEqual(@as(u32, 6), st.generated_tokens);
    // The target and the draft windows both hold the prompt plus every committed row.
    try testing.expectEqual(@as(u32, 28), rig.st.offset);
    try testing.expectEqual(@as(u32, 28), rig.caches[0].offset);
    try testing.expectEqual(script.u32s.len, script.nu);
    try testing.expectEqual(script.f32s.len, script.nf);
    // Every verify forward routed through the source: 3 prompt forwards + 4 verifies per layer.
    try testing.expectEqual(@as(u64, 7 * c.n_layers), rig.src.stats().route_calls);
}

test "dsv41 dspark loop: A0 (c) D1 (profile builds): the first cycle evaluates its pending state, then a draft block dropped and freed, then its own; the tokens are unchanged" {
    if (comptime !first_cycle.enabled) return error.SkipZigTest;
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const g = &rig.g;
    first_cycle.startDecode();
    defer first_cycle.phase = .build;
    for (0..4) |i| {
        const e0 = g.evals.items.len;
        const f0 = g.freed.items.len;
        _ = try lp.cycle(&rig.ex, &out, a, null);
        first_cycle.endCycle();
        const evals = g.evals.items[e0..];
        if (i == 0) {
            // the pending state, the dropped block, the cycle's own block, the verify
            try testing.expectEqual(@as(usize, 4), evals.len);
            // the dropped block's arrays, built between the first two syncs, are freed before the cycle's own block
            var dropped_freed = false;
            for (g.freed.items[f0..]) |r| dropped_freed = dropped_freed or (r.from == evals[0] and r.to == evals[1]);
            try testing.expect(dropped_freed);
        } else try testing.expectEqual(@as(usize, 2), evals.len);
    }
    try testing.expect(!first_cycle.firstDraft());
    // the same tokens as the cycles without D1 (the first test's script, every host value read once)
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    try testing.expectEqual(script.u32s.len, script.nu);
    try testing.expectEqual(script.f32s.len, script.nf);
}

/// `Script` with the routed ids drawn from a counter and every ids read logged (its values, per call).
const IdLog = struct {
    base: Script,
    next: u32 = 0,
    vals: [8192]u16 = undefined,
    n_vals: usize = 0,
    starts: [1024]usize = undefined,
    n_calls: usize = 0,

    fn values(self: *IdLog) TraceOps.HostValues {
        var v = self.base.values();
        v.ctx = self;
        v.ids = ids;
        v.argmax = argmax;
        v.u32s = u32s_;
        v.f32s = f32s_;
        v.bools = bools_;
        return v;
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *IdLog = @ptrCast(@alignCast(ctx));
        s.starts[s.n_calls] = s.n_vals;
        s.n_calls += 1;
        for (out) |*o| {
            o.* = @intCast((s.next * 5 + 3) % s.base.n_experts);
            s.next += 1;
            s.vals[s.n_vals] = o.*;
            s.n_vals += 1;
        }
    }
    fn call(s: *const IdLog, i: usize) []const u16 {
        const end = if (i + 1 < s.n_calls) s.starts[i + 1] else s.n_vals;
        return s.vals[s.starts[i]..end];
    }
    fn argmax(ctx: *anyopaque) anyerror!u32 {
        return Script.argmax(&@as(*IdLog, @ptrCast(@alignCast(ctx))).base);
    }
    fn u32s_(ctx: *anyopaque, out: []u32) anyerror!void {
        return Script.u32s_(&@as(*IdLog, @ptrCast(@alignCast(ctx))).base, out);
    }
    fn f32s_(ctx: *anyopaque, out: []f32) anyerror!void {
        return Script.f32s_(&@as(*IdLog, @ptrCast(@alignCast(ctx))).base, out);
    }
    fn bools_(ctx: *anyopaque, out: []bool) anyerror!void {
        return Script.bools_(&@as(*IdLog, @ptrCast(@alignCast(ctx))).base, out);
    }
};

test "dsv41 dspark loop: draft routes (profile builds): each cycle records every stage's routed ids, read after the draft's eval; the tokens are unchanged" {
    if (comptime !draft_routes.enabled) return error.SkipZigTest;
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    const c = &rig.m.c;
    const dsc = c.dspark;
    var log: IdLog = .{ .base = .{
        .n_experts = @intCast(dsc.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    } };
    rig.g.host_values = log.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    try draft_routes.install(.{ .n_stages = dsc.n_stages, .n_experts = dsc.n_routed_experts, .top_k = dsc.n_experts_per_tok, .block = dsc.block_size }, 6);
    defer draft_routes.uninstall();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    // a cycle's ids reads: its draft stages' (after the draft's eval), then the verify's routed layers
    var first_call: [4]usize = undefined;
    for (&first_call) |*f| {
        f.* = log.n_calls;
        _ = try lp.cycle(&rig.ex, &out, a, null);
        try testing.expectEqual(f.* + dsc.n_stages + c.n_layers, log.n_calls);
    }
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    try testing.expectEqual(@as(u32, 4), draft_routes.cycles);
    // the stream is the stage reads, in order; the histogram is their fold
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try draft_routes.writeStreamJson(&aw.writer);
    var want: std.Io.Writer.Allocating = .init(a);
    defer want.deinit();
    var hand: [4][512]u32 = @splat(@splat(0));
    try want.writer.print("{{\"stages\": {d}, \"experts_per_stage\": {d}, \"top_k\": {d}, \"block\": {d}, \"global_id\": \"stage * experts_per_stage + id\", \"cycles\": [", .{ dsc.n_stages, dsc.n_routed_experts, dsc.n_experts_per_tok, dsc.block_size });
    for (first_call, 0..) |f, ci| {
        try want.writer.writeAll(if (ci == 0) "[" else ", [");
        for (0..dsc.n_stages) |st| {
            const v = log.call(f + st);
            try testing.expectEqual(@as(usize, dsc.block_size * dsc.n_experts_per_tok), v.len);
            try want.writer.writeAll(if (st == 0) "[" else ", [");
            for (v, 0..) |e, i| {
                try want.writer.print("{s}{d}", .{ if (i == 0) "" else ",", e });
                hand[st][e] += 1;
            }
            try want.writer.writeAll("]");
        }
        try want.writer.writeAll("]");
    }
    try want.writer.writeAll("]}");
    try testing.expectEqualStrings(want.written(), aw.written());
    var cs: [512]u32 = undefined;
    for (0..dsc.n_stages) |st| {
        _ = draft_routes.counts(st, cs[0..dsc.n_routed_experts]);
        try testing.expectEqualSlices(u32, hand[st][0..dsc.n_routed_experts], cs[0..dsc.n_routed_experts]);
    }
    // the cell's lines over this run (the host dry run of their format)
    var hb: [8192]u8 = undefined;
    for (0..dsc.n_stages) |st| std.debug.print("NATIVE {s}\n", .{draft_routes.histLine(&hb, st)});
    std.debug.print("NATIVE {s}\n", .{draft_routes.lruLine(&hb)});
}

test "dsv41 dspark loop: verify timeline (profile builds): every verify is stamped, each routed layer's submits in order inside it" {
    if (comptime !timeline.enabled) return error.SkipZigTest;
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    const c = &rig.m.c;
    var script: Script = .{
        .n_experts = @intCast(c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    // the prompt's routed calls come before the arming: no verify, no stamp
    timeline.armHost(c.n_layers);
    defer timeline.uninstall();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    for (0..4) |_| _ = try lp.cycle(&rig.ex, &out, a, null);
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    try testing.expectEqual(@as(u32, 4), timeline.cycles);
    for (0..4) |ci| {
        const v = timeline.verifyOf(ci);
        try testing.expect(v.begin > 0 and v.end >= v.begin);
        for (timeline.layersOf(ci)) |l| {
            try testing.expect(l.t[0] >= v.begin and l.t[4] <= v.end);
            for (l.t[1..], l.t[0..4]) |later, earlier| try testing.expect(later >= earlier and earlier > 0);
        }
    }
    // the cell's line over this run (no shim armed: no buffer rows; the host dry run of its format)
    var tb: [8192]u8 = undefined;
    std.debug.print("NATIVE {s}\n", .{timeline.line(&tb, 0)});
}

test "dsv41 dspark loop: CYCLE_TRIM: one eval per draft with its sigmoid, the commit's window update dispatched and waited by the next draft" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const g = &rig.g;
    for (0..4) |_| {
        const n0 = g.nodes.items.len;
        const e0 = g.evals.items.len;
        _ = try lp.cycle(&rig.ex, &out, a, null);
        const nodes = g.nodes.items;
        const evals = g.evals.items[e0..];
        // Two syncs per greedy cycle: the draft and the verify. The draft's ids and confidence
        // sigmoid are both read right after its eval, with nothing built between (realised in it).
        try testing.expectEqual(@as(usize, 2), evals.len);
        for (nodes[n0..evals[0]]) |nd| try testing.expect(nd.op != .host_read);
        try testing.expectEqual(ops.Op.host_read, nodes[evals[0]].op);
        try testing.expectEqual(ops.Op.host_read, nodes[evals[0] + 1].op);
        // The last GPU commit of the cycle is the window update, after its last sync.
        var last_async: usize = 0;
        for (nodes[n0..], n0..) |nd, i| {
            if (nd.op == .async_eval) last_async = i;
        }
        try testing.expect(last_async >= evals[evals.len - 1]);
    }
    // The same tokens as the unfolded cycles (the first test's script).
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
}

test "dsv41 dspark loop: DRAFT_AHEAD: the next block built in the verify's wait runs only when every draft was accepted; the same tokens" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    rig.g.record_host = true;
    // CYCLE_TRIM's script: cycle 1 accepts 1 of 2, cycle 2 accepts 1 of 1 (every draft), cycle 3 rejects its one draft,
    // cycle 4 runs to the cap.
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    const n_st = rig.head.nStages();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .k_request = 5, .lookup = null, .max_tokens = 6, .draft_ahead = true });
    defer lp.deinit();
    try testing.expect(lp.draftAhead());
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const g = &rig.g;
    // After each cycle: whether its verify's outcome installed the ahead block (every verified row kept).
    const installed = [_]bool{ false, true, false };
    for (0..4) |cy| {
        const e0 = g.evals.items.len;
        const n0 = g.nodes.items.len;
        const used = lp.ahead != null and lp.ahead.?.ready;
        try testing.expectEqual(cy == 2, used);
        const fin = try lp.cycle(&rig.ex, &out, a, null);
        // Still two syncs per cycle (the draft's, the verify's); the verify is committed before its wait.
        const evals = g.evals.items[e0..];
        try testing.expectEqual(@as(usize, 2), evals.len);
        var commits_before_verify: usize = 0;
        for (g.nodes.items[evals[0]..evals[1]]) |nd| commits_before_verify += @intFromBool(nd.op == .async_eval);
        try testing.expect(commits_before_verify >= 1);
        // A used ahead block was written with this cycle's primary before its eval: no block built in the cycle
        // before that eval but the stock route's own (none when ahead).
        if (used) {
            var built_before_eval: usize = 0;
            for (g.nodes.items[n0..evals[0]]) |nd| built_before_eval += @intFromBool(nd.op != .async_eval and nd.op != .host_read);
            try testing.expect(built_before_eval < 8);
        }
        if (cy < installed.len) try testing.expectEqual(installed[cy], lp.ahead != null and lp.ahead.?.ready);
        if (fin != null) break;
    }
    // The same tokens as the stock cycles (CYCLE_TRIM's).
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    // The ahead block used (cycle 3's) was written with its cycle's primary: 12, cycle 2's correction.
    const prev = try g.hostBytes(lp.ahead_in.?.prev);
    try testing.expectEqual(@as(i32, 12), std.mem.bytesToValue(i32, prev[0..4]));
}

test "dsv41 dspark loop: VERIFY_ENCODE hoist: a verify forward hands each routed call its shared expert, gate weights and HC tail" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{ .n_experts = @intCast(rig.m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    // Each routed call's hoist, as the hook receives it: its length, and whether its first array is
    // the shared expert's output (f32, the call's rows x hidden) and its second the gate weights.
    const Recorder = struct {
        ex: @TypeOf(&rig.ex),
        lens: std.ArrayList(usize) = .empty,
        shaped: bool = true,
        const Rec = @This();
        pub fn at(r: *Rec, l: u32) Hook {
            return .{ .r = r, .l = l };
        }
        const Hook = struct {
            r: *Rec,
            l: u32,
            pub fn routed(h: Hook, g: *TraceOps, xf: u32, idx: u32) !u32 {
                try h.r.lens.append(testing.allocator, std.math.maxInt(usize));
                return h.r.ex.at(h.l).routed(g, xf, idx);
            }
            pub fn routedHoist(h: Hook, g: *TraceOps, xf: u32, idx: u32, hoist: []const u32) !u32 {
                try h.r.lens.append(testing.allocator, hoist.len);
                if (hoist.len > 0) {
                    const rows = g.shapeOf(xf).dim(0);
                    const s = g.shapeOf(hoist[0]);
                    h.r.shaped = h.r.shaped and g.dtypeOf(hoist[0]) == .float32 and s.dim(0) == rows and s.dim(1) == g.shapeOf(xf).dim(1);
                    h.r.shaped = h.r.shaped and g.shapeOf(hoist[1]).eql(g.shapeOf(idx));
                }
                return h.r.ex.at(h.l).routedHoist(g, xf, idx, hoist);
            }
        };
    };
    var rec: Recorder = .{ .ex = &rig.ex };
    defer rec.lens.deinit(a);
    var ids: [12]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
    // A verify-width forward (3 rows): every layer hands 4 arrays (shared, weights, post, comb).
    _ = try rig.model.forward(&rig.g, &rig.st, ids[0..3], .{ .logits = .all, .main_hidden = true }, &rec, graph.NoProbe{});
    try rig.ex.flush();
    const n_layers = rig.m.c.n_layers;
    try testing.expectEqual(@as(usize, n_layers), rec.lens.items.len);
    for (rec.lens.items) |n| try testing.expectEqual(@as(usize, 4), n);
    try testing.expect(rec.shaped);
    // Wider than the decode rows: nothing is handed over.
    rec.lens.clearRetainingCapacity();
    _ = try rig.model.forward(&rig.g, &rig.st, ids[0..9], .{ .logits = .last }, &rec, graph.NoProbe{});
    try rig.ex.flush();
    try testing.expectEqual(@as(usize, n_layers), rec.lens.items.len);
    for (rec.lens.items) |n| try testing.expectEqual(@as(usize, 0), n);
}

test "dsv41 dspark loop: the shell's prompt (all but the last token, then the last) and its rounds give the cell's tokens" {
    // The cell: `prefill` over the whole prompt, then `cycle`s (the mini model's script above).
    // The shell: `prefillLogits` over all but the last prompt token, `extendLogits` over the last,
    // the Generator's pick (its argmax), then `round`s under the v2 invariant with the budget cap.
    const a = testing.allocator;
    const script_of = struct {
        fn f(n_experts: u16) Script {
            return .{
                .n_experts = n_experts,
                .pick = 3,
                .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 10, 11 }, &.{ 10, 12 }, &.{ 13, 14 }, &.{ 20, 21 }, &.{ 30, 31 }, &.{ 30, 31, 40 } },
                .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 0.9, 0.3 }, &.{ 0.2, 0.9 }, &.{ 0.9, 0.9 } },
            };
        }
    }.f;
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    const budget: usize = 6;

    var cell: Rig = undefined;
    try cell.init();
    defer cell.deinit();
    var cs = script_of(@intCast(cell.m.c.n_routed_experts));
    cell.g.host_values = cs.values();
    var lc = Loop(TraceOps).init(&cell.g, cell.model, cell.head, &cell.st, cell.caches[0..cell.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = budget });
    defer lc.deinit();
    const primary = try lc.prefill(a, &cell.ex, &prompt);
    var cell_out: std.ArrayList(u32) = .empty;
    defer cell_out.deinit(a);
    try testing.expectEqual(Finish.length, try lc.run(&cell.ex, &cell_out, a));

    var sh: Rig = undefined;
    try sh.init();
    defer sh.deinit();
    var ss = script_of(@intCast(sh.m.c.n_routed_experts));
    sh.g.host_values = ss.values();
    var ls = Loop(TraceOps).init(&sh.g, sh.model, sh.head, &sh.st, sh.caches[0..sh.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = std.math.maxInt(u32) });
    defer ls.deinit();
    const l0 = try ls.prefillLogits(a, &sh.ex, prompt[0 .. prompt.len - 1]);
    sh.g.release(l0);
    const l1 = try ls.extendLogits(a, &sh.ex, prompt[prompt.len - 1 ..]);
    const t1 = try sh.g.hostArgmax(l1);
    sh.g.release(l1);
    // The shell's prompt seeds as the whole prompt does: the target and every draft window at 20 rows.
    try testing.expectEqual(primary, t1);
    try testing.expectEqual(@as(u32, 20), sh.st.offset);
    for (sh.caches[0..sh.head.nStages()]) |c| try testing.expectEqual(@as(u32, 20), c.offset);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var next = t1;
    var rounds: usize = 0;
    while (out.items.len < budget) {
        const first = rounds == 0;
        const cap: u32 = @intCast(budget - out.items.len - @intFromBool(!first));
        const r = try ls.round(&sh.ex, a, next, cap, null, {});
        defer a.free(r.tokens);
        try testing.expectEqual(next, r.tokens[0]);
        for (r.tokens[@intFromBool(first)..]) |tok| if (out.items.len < budget) try out.append(a, tok);
        next = r.next_token;
        rounds += 1;
        if (out.items.len + 1 == budget) try out.append(a, next);
    }
    try testing.expectEqualSlices(u32, cell_out.items, out.items);
    try testing.expectEqualSlices(u32, &.{ 5, 9, 10, 12, 20, 30 }, out.items);
    try testing.expectEqual(@as(usize, 4), rounds);
    // The last round kept one of its two accepted drafts (the budget cap): one row fewer than the
    // cell's uncapped cycle (28), in the target and in every draft window.
    try testing.expectEqual(@as(u32, 27), sh.st.offset);
    for (sh.caches[0..sh.head.nStages()]) |c| try testing.expectEqual(@as(u32, 27), c.offset);
    try testing.expectEqual(cs.nu, ss.nu);
    try testing.expectEqual(cs.nf, ss.nf);
}

test "dsv41 dspark loop: the K33 draft block replays the eager one from regions built at construction" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{ .n_experts = @intCast(rig.m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    const n_st = rig.head.nStages();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    // A K33 head over the same residents (its regions are built here, at construction).
    const k33 = try Loop(TraceOps).H.init(a, &rig.g, rig.m.c, .{ .draft_rows = graph.draft_compile_max_rows }, &rig.lookup);
    defer k33.deinit(&rig.g);
    const e0 = rig.g.nodes.items.len;
    const de = try rig.head.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e1 = rig.g.nodes.items.len;
    const dk = try k33.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e2 = rig.g.nodes.items.len;
    inline for (.{ "ids", "logits", "conf" }) |f| {
        try testing.expect(rig.g.shapeOf(@field(de, f)).eql(rig.g.shapeOf(@field(dk, f))));
        try testing.expectEqual(rig.g.dtypeOf(@field(de, f)), rig.g.dtypeOf(@field(dk, f)));
    }
    const count = struct {
        fn f(g: *const TraceOps, from: usize, to: usize, op: ops.Op) usize {
            var n: usize = 0;
            for (g.nodes.items[from..to]) |nd| n += @intFromBool(nd.op == op);
            return n;
        }
    }.f;
    // Eager: no region. K33: per stage the HC attention prep, the main KV, the QKV and
    // output prep, the HC ffn prep, the gate prefix, the MoE combine and the HC post;
    // then a markov step per draft and the confidence.
    try testing.expectEqual(@as(usize, 0), count(&rig.g, e0, e1, .tape_begin));
    try testing.expectEqual(8 * n_st + rig.head.blockSize() + 1, count(&rig.g, e1, e2, .tape_begin));
    // The regions hold the eager body's heavy ops.
    // (take: the markov embeds are gathered once per step on both paths.)
    inline for (.{ ops.Op.qmm, ops.Op.gather_qmm, ops.Op.softmax, ops.Op.argmax, ops.Op.matmul, ops.Op.take }) |op| {
        try testing.expectEqual(count(&rig.g, e0, e1, op), count(&rig.g, e1, e2, op));
    }
}

/// A lookup that records every name asked for and every name dropped.
const DropLookup = struct {
    inner: *const mdl.SpecLookup,
    got: std.ArrayList([]u8) = .empty,
    dropped: std.ArrayList([]u8) = .empty,

    pub fn get(self: *DropLookup, name: []const u8) ?u32 {
        self.got.append(testing.allocator, testing.allocator.dupe(u8, name) catch @panic("oom")) catch @panic("oom");
        return self.inner.get(name);
    }

    pub fn drop(self: *DropLookup, name: []const u8) void {
        self.dropped.append(testing.allocator, testing.allocator.dupe(u8, name) catch @panic("oom")) catch @panic("oom");
    }

    fn has(list: []const []u8, name: []const u8) bool {
        for (list) |x| if (std.mem.eql(u8, x, name)) return true;
        return false;
    }

    fn deinit(self: *DropLookup) void {
        for (self.got.items) |x| testing.allocator.free(x);
        for (self.dropped.items) |x| testing.allocator.free(x);
        self.got.deinit(testing.allocator);
        self.dropped.deinit(testing.allocator);
    }
};

test "dsv41 dspark loop: each draft stage is one wave, freed at the stage's end" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{ .n_experts = @intCast(rig.m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    const n_st = rig.head.nStages();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    const e0: u32 = @intCast(rig.g.nodes.items.len);
    const waves0 = rig.g.freed.items.len;
    const d = try rig.head.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const waves = rig.g.freed.items[waves0..];
    try testing.expectEqual(@as(usize, n_st), waves.len);
    var prev = e0;
    for (waves) |w| {
        try testing.expect(w.from >= prev and w.to > w.from);
        prev = w.to;
    }
    // forward_head (the base logits, the markov steps, the confidence) builds after the last stage.
    try testing.expect(d.ids >= prev and d.logits >= prev and d.conf >= prev);
}

test "dsv41 dspark loop: DRAFT_STAGED: the same draft block's graph, each stage's outputs committed once built" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{ .n_experts = @intCast(rig.m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    const H = Loop(TraceOps).H;
    // The stock head (the rig's) and a staged one over the same weights; bound at construction, off by default.
    const staged = try H.initWith(a, &rig.g, rig.m.c, .{}, &rig.lookup, .{ .staged_commit = true });
    defer staged.deinit(&rig.g);
    try testing.expect(!rig.head.stagedCommit() and staged.stagedCommit());
    const n_st = rig.head.nStages();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var spans: [3]usize = undefined;
    var waves: [2][]const TraceOps.Freed = undefined;
    for ([_]*H{ rig.head, staged }, 0..) |hd, k| {
        spans[k] = rig.g.nodes.items.len;
        const w0 = rig.g.freed.items.len;
        _ = try hd.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
        waves[k] = rig.g.freed.items[w0..];
    }
    spans[2] = rig.g.nodes.items.len;
    // Exact by construction: every node other than a commit in the same order, op, dtype and shape.
    var kept: [2]std.ArrayList(TraceOps.Node) = .{ .empty, .empty };
    defer for (&kept) |*l| l.deinit(a);
    var commits: [2]usize = .{ 0, 0 };
    for (0..2) |k| for (rig.g.nodes.items[spans[k]..spans[k + 1]]) |nd| {
        if (nd.op == .async_eval) commits[k] += 1 else try kept[k].append(a, nd);
    };
    try testing.expectEqual(kept[0].items.len, kept[1].items.len);
    for (kept[0].items, kept[1].items) |x, y| try testing.expect(x.op == y.op and x.dtype == y.dtype and x.shape.eql(y.shape));
    // Stock commits nothing inside the block; staged commits once per stage, each inside its stage's wave (after the
    // stage's nodes, before the wave's reset), so the head is built after every stage was committed.
    try testing.expectEqual(@as(usize, 0), commits[0]);
    try testing.expectEqual(@as(usize, n_st), commits[1]);
    var j: usize = 0;
    for (rig.g.nodes.items[spans[1]..spans[2]], spans[1]..) |nd, i| if (nd.op == .async_eval) {
        try testing.expect(i > waves[1][j].from and i < waves[1][j].to);
        j += 1;
    };
}

/// The bytes MlxOps holds over one traced call `[from, to)` whose released
/// ranges are `ranges` (layer waves, and the score sub-waves nested in them):
/// `reset` = every allocating node until the call's reset (one reset per
/// forward, the 8b72bd27 harness); `layer` = the widest layer wave holding all
/// its nodes to the layer's reset (69fe3eb); `wave` = the widest layer wave with
/// its score chains released at last use (their outputs kept, at most their
/// two largest arrays live at once, as MLX frees a chain); `outside` = the
/// call's nodes outside every wave.
const Held = struct {
    reset: u64,
    outside: u64,
    layer: u64,
    wave: u64,
    widest_at: usize,

    fn allocating(n: TraceOps.Node) u64 {
        switch (n.op) {
            .input, .host, .scalar, .reshape, .transpose, .transpose_axes, .broadcast_to, .expand_dims, .slice, .tape_begin, .tape_end => return 0,
            else => return @as(u64, @intCast(n.shape.numel())) * ops.dtypeSize(n.dtype),
        }
    }

    /// A released chain's live bound: its two largest arrays, and its output (the last array) kept.
    fn chain(g: *const TraceOps, r: TraceOps.Freed) struct { live: u64, out: u64 } {
        var a: u64 = 0;
        var b: u64 = 0;
        var out: u64 = 0;
        for (g.nodes.items[r.from..r.to]) |n| {
            const x = allocating(n);
            if (x == 0) continue;
            out = x;
            if (x > a) {
                b = a;
                a = x;
            } else if (x > b) b = x;
        }
        return .{ .live = a + b, .out = out };
    }

    fn of(g: *const TraceOps, from: usize, to: usize, ranges: []const TraceOps.Freed) Held {
        var h: Held = .{ .reset = graph.heldBytes(g, from, to).sum, .outside = 0, .layer = 0, .wave = 0, .widest_at = 0 };
        var in_waves: u64 = 0;
        var k: usize = 0;
        for (ranges, 0..) |w, i| {
            const top = for (ranges) |o| {
                if (o.from <= w.from and w.to <= o.to and (o.from != w.from or o.to != w.to)) break false;
            } else true;
            if (!top) continue;
            const all = graph.heldBytes(g, w.from, w.to).sum;
            in_waves += all;
            var kept = all;
            var live: u64 = 0;
            for (ranges) |r| {
                if (r.from >= w.from and r.to <= w.to and (r.from != w.from or r.to != w.to)) {
                    const ch = chain(g, r);
                    kept = kept - graph.heldBytes(g, r.from, r.to).sum + ch.out;
                    live = @max(live, ch.live);
                }
            }
            if (all > h.layer) h.layer = all;
            if (kept + live > h.wave) {
                h.wave = kept + live;
                h.widest_at = k;
            }
            _ = i;
            k += 1;
        }
        h.outside = h.reset - in_waves;
        return h;
    }

    /// Peak live bytes of [from, to) for waves nested to any depth (K16: layer > chunk > score chain):
    /// its own arrays (outside every child wave) stay live; a child wave's arrays are freed at its end
    /// except its survivors (a leaf's output: a score chain's last array; a non-leaf's survivors are
    /// the arrays it kept, not visible to the trace: 0 here, billed by construction); at any moment at
    /// most one child is open. peak = own + sum(survivors) + max(child peak - child survivor).
    const Nested = struct { all: u64, peak: u64, survivor: u64 };
    fn nested(g: *const TraceOps, from: usize, to: usize, ranges: []const TraceOps.Freed, depth: u8) Nested {
        const all = graph.heldBytes(g, from, to).sum;
        var own = all;
        var surv: u64 = 0;
        var worst: u64 = 0;
        var n_children: usize = 0;
        for (ranges) |w| {
            if (!(w.from >= from and w.to <= to and (w.from != from or w.to != to))) continue;
            // a maximal child: no other range inside [from, to) strictly contains it
            const maximal = for (ranges) |o| {
                if (o.from >= from and o.to <= to and (o.from != from or o.to != to) and o.from <= w.from and w.to <= o.to and (o.from != w.from or o.to != w.to)) break false;
            } else true;
            if (!maximal) continue;
            n_children += 1;
            const ch = if (depth < 6) nested(g, w.from, w.to, ranges, depth + 1) else Nested{ .all = graph.heldBytes(g, w.from, w.to).sum, .peak = graph.heldBytes(g, w.from, w.to).sum, .survivor = 0 };
            own -|= ch.all;
            surv += ch.survivor;
            worst = @max(worst, ch.peak -| ch.survivor);
        }
        if (n_children == 0) {
            const c = chain(g, .{ .from = @intCast(from), .to = @intCast(to) });
            return .{ .all = all, .peak = c.live + c.out, .survivor = c.out };
        }
        return .{ .all = all, .peak = own + surv + worst, .survivor = 0 };
    }

    fn print(h: Held, what: []const u8) void {
        std.debug.print("dsv41 held: {s}: one reset per forward {d} B; layer waves {d} B; score chains released {d} B ({d} B outside the waves + the widest wave, #{d}, {d} B)\n", .{ what, h.reset, h.outside + h.layer, h.outside + h.wave, h.outside, h.widest_at, h.wave });
    }
};

// Bank mode (host only, the trace backend; DSV41_BANK): the window harnesses'
// forwards on the bank's own config, residents bound from the resident spec,
// the Engram rows read from the bank, the routed calls through the EXL3 chain
// over 8 slot rows per layer (the harness's). The held bytes per forward with
// one reset per forward against one wave per layer: the static bound of the
// M3 / M5 re-runs (the head's f32 promotion inside the matmul is not a node:
// 2.65 GB at each forward's head, both sides).
test "dsv41 dspark loop: the bank's forwards hold one layer's arrays per wave, not the forward's" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 held: {s}\n", .{diag.message()});
    const c = try v41.Config.load(a, io, bank, &diag);
    const eng = @import("deepseek_v41_engram.zig");
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/engram-token-map.u32", .{bank}), &c, &diag);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
    const L = Loop(TraceOps);
    const model_ = try L.M.init(a, &g, c, try routes.parse(&.{}, null), &lookup, &src);
    defer model_.deinit(&g);
    const head = try L.H.init(a, &g, c, .{}, &lookup);
    defer head.deinit(&g);
    const nl = c.n_layers;
    const rows0 = try aa.alloc(u32, nl);
    @memset(rows0, 0);
    const rows8 = try aa.alloc(u32, nl);
    @memset(rows8, 8);
    var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows0 });
    defer fsrc.deinit();
    const Chain = xp.EagerChain(TraceOps, xp.TraceGemv);
    var ex = try xp.Experts(TraceOps, xp.FakeSource, Chain).init(a, &g, &fsrc, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try ex.grow(&g, rows8);
    var script: Script = .{ .n_experts = @intCast(c.n_routed_experts), .pick = 1, .u32s = &.{}, .f32s = &.{} };
    g.host_values = script.values();
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);

    // M3: the prompt in forwards of 8 rows (the last with its row's logits), then a decode forward.
    var st = try model_.newState();
    defer st.deinit(&g, a);
    var m3_prompt: Held = undefined;
    var i: usize = 0;
    while (i < prompt.len) : (i += 8) {
        const last = i + 8 == prompt.len;
        const f0 = g.nodes.items.len;
        const w0 = g.freed.items.len;
        const r = try model_.forward(&g, &st, prompt[i .. i + 8], .{ .logits = if (last) .last else .none }, &ex, graph.NoProbe{});
        const h = Held.of(&g, f0, g.nodes.items.len, g.freed.items[w0..]);
        if (i == 0 or h.reset > m3_prompt.reset) m3_prompt = h;
        try L.M.fence(&g, &st, &.{if (last) r.logits.? else r.hidden});
        try ex.flush();
        g.reset();
    }
    const f0 = g.nodes.items.len;
    const w0 = g.freed.items.len;
    _ = try model_.forward(&g, &st, prompt[0..1], .{ .logits = .last }, &ex, graph.NoProbe{});
    const m3_decode = Held.of(&g, f0, g.nodes.items.len, g.freed.items[w0..]);
    try ex.flush();
    g.reset();
    m3_prompt.print("M3 prompt forward (8 rows)");
    m3_decode.print("M3 decode forward (1 row)");

    // M5: the loop's prefill seeds the draft stages; then a verify forward of 8 rows (every
    // row's logits and the DSpark taps) and a draft block.
    var st5 = try model_.newState();
    defer st5.deinit(&g, a);
    const n_st = head.nStages();
    var caches: [4]L.H.Cache = @splat(.{});
    defer for (caches[0..n_st]) |*cc| cc.deinit(&g);
    var lp = L.init(&g, model_, head, &st5, caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    _ = try lp.prefill(a, &ex, &prompt);
    const v0 = g.nodes.items.len;
    const vw0 = g.freed.items.len;
    const ver = try model_.forward(&g, &st5, prompt[0..8], .{ .logits = .all, .main_hidden = true }, &ex, graph.NoProbe{});
    const m5_verify = Held.of(&g, v0, g.nodes.items.len, g.freed.items[vw0..]);
    try L.M.fence(&g, &st5, &.{ ver.logits.?, ver.main_hidden.? });
    try ex.flush();
    g.reset();
    const d0 = g.nodes.items.len;
    const dw0 = g.freed.items.len;
    _ = try head.draftBlock(&g, lp.main_h.?, 1, caches[0..n_st], model_.embed, model_.head);
    const m5_draft = Held.of(&g, d0, g.nodes.items.len, g.freed.items[dw0..]);
    g.reset();
    m5_verify.print("M5 verify forward (8 rows, every row's logits, the taps)");
    m5_draft.print("M5 draft block");

    // One wave per layer (stage): the forward's waves are its layers.
    // The grouped wo_a dequantized per call and its f32 cast (201 MB a layer) sit in every
    // forward's one-reset total, and in no more than one wave at a time.
    const woa_layer: u64 = @as(u64, c.o_groups) * c.o_lora_rank * (@as(u64, c.n_heads) * c.head_dim / c.o_groups) * (2 + 4);
    for ([_]Held{ m3_prompt, m3_decode, m5_verify }) |h| {
        try testing.expect(h.reset >= nl * woa_layer);
        try testing.expect(h.wave >= woa_layer and h.outside + h.wave < 2 * woa_layer + (256 << 20));
    }
    try testing.expect(m5_draft.reset >= n_st * woa_layer and m5_draft.layer < 2 * woa_layer);
}

// Bank mode (host only, the trace backend; DSV41_BANK): the 16K cell's prompt as the model chunks it (the Python
// rule: 17 x 953 + 183), each chunk one forward over the state the earlier chunks built, the routed calls a
// stand-in (the prefill's routed transients are the wide lane's own). Per chunk: the widest layer wave with
// every array held to the layer's reset (69fe3eb) against the score chains released at last use; the stock
// tier (masked-full attention, the parity harnesses') and the served tier (K30 selected keys).
test "dsv41 dspark loop: the bank's 16K prompt chunks hold at most two score blocks per chain, not the layer's" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 held: {s}\n", .{diag.message()});
    const c = try v41.Config.load(a, io, bank, &diag);
    const eng = @import("deepseek_v41_engram.zig");
    const kvc = @import("deepseek_v41_cache.zig");
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/engram-token-map.u32", .{bank}), &c, &diag);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    const prompt = try aa.alloc(u32, 16384);
    for (prompt, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
    const spans = try kvc.prefillSpans(aa, 16384, kvc.resolvePrefillChunk(&c, 16384, null, kvc.default_chunk_target_bytes));
    try testing.expectEqual(@as(usize, 18), spans.len);
    // K16 on the served tier (the construction switch the integration lane adds): every layer over all
    // chunks before the next, one wave per layer, so a layer holds every chunk's arrays at once.
    var served_k16 = routes.served;
    served_k16.layer_major = true;
    for ([_]struct { name: []const u8, tier: routes.Tier }{ .{ .name = "stock", .tier = routes.stock }, .{ .name = "served", .tier = routes.served }, .{ .name = "served-k16", .tier = served_k16 } }) |t| {
        var g = TraceOps.init(a);
        defer g.deinit();
        const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
        var kd: @import("exl3_kernels.zig").Diag = .{};
        var reg = try @import("exl3_kernels.zig").Registry.init(a, &@import("exl3_kernels.zig").embedded, @import("exl3_kernels.zig").manifest_sha256, &kd);
        defer reg.deinit();
        const model_ = try Loop(TraceOps).M.initWith(a, &g, c, t.tier, &lookup, &src, .{ .registry = &reg });
        defer model_.deinit(&g);
        var st = try model_.newState();
        defer st.deinit(&g, a);
        const stand: graph.StandIn(TraceOps) = .{ .scale = try g.input(&.{@intCast(c.n_routed_experts)}, .float32) };
        var worst: Held = undefined;
        var worst_i: usize = 0;
        // Chunk-major: one forward per chunk, as the served prompt pass runs them. K16: ONE forward over the
        // prompt (the model spans it into the same chunks and runs every layer over all of them).
        const k16_spans = [_][2]u32{.{ 0, 16384 }};
        const drive: []const [2]u32 = if (t.tier.layer_major) &k16_spans else spans;
        for (drive, 0..) |sp, i| {
            const f0 = g.nodes.items.len;
            const w0 = g.freed.items.len;
            _ = try model_.forward(&g, &st, prompt[sp[0]..sp[1]], .{ .logits = if (i + 1 == drive.len) .last else .none }, stand, graph.NoProbe{});
            if (t.tier.layer_major) {
                // Three wave levels (layer > chunk > score chain): the nested peak, and the chunks' kept
                // Halves (their survivors, invisible to the trace) by construction.
                const nst = Held.nested(&g, f0, g.nodes.items.len, g.freed.items[w0..], 0);
                const halves: u64 = 16384 * (@as(u64, c.hc_mult) * c.hidden_size * 4 + c.hidden_size * 4 + 3 * @as(u64, c.hc_mult) * 4 + @as(u64, c.hc_mult) * c.hc_mult * 4);
                const billed = v41.PrefillBill.of(&c, t.tier.kv).withIndexLaunch(t.tier.routes.prefill_index).layerMajorWaveBytes(16384, .served);
                std.debug.print("dsv41 held: served-k16 tier, 16K one forward: nested peak {d} B + kept halves {d} B = {d} B (built {d} B); billed layer-major wave {d} B\n", .{ nst.peak, halves, nst.peak + halves, nst.all, billed });
                // The bill covers the trace. Tightness: with the prefill core and indexer the attention side
                // falls under the routed group's sub-wave, whose bill term (moeRowCap rows of routed outputs,
                // joined input, combine and HC post) the trace's routed stand-in does not materialize, so the
                // trace cannot bound it from above here; the chunk-major tiers keep their ratio rules below.
                try testing.expect(nst.peak + halves <= billed);
                if (!t.tier.routes.prefill_index) try testing.expect(billed <= nst.peak + halves + (nst.peak + halves) / 2);
                // wire_tables' prompt wave in buffers: every node of the widest layer wave (no frees inside it
                // credited), the nodes outside the layer waves, and each chunk's kept Halves (four arrays).
                const end = g.nodes.items.len;
                const freed = g.freed.items[w0..];
                var in_layers: u64 = 0;
                var widest: u64 = 0;
                for (freed, 0..) |w, wi| {
                    if (w.to <= w.from) continue;
                    const inner = for (freed, 0..) |o, oi| {
                        if (oi != wi and o.from <= w.from and w.to <= o.to and (o.from != w.from or o.to != w.to)) break true;
                    } else false;
                    if (inner) continue;
                    const n = graph.heldArrays(&g, w.from, w.to, false);
                    in_layers += n;
                    widest = @max(widest, n);
                }
                const k16_arrays = (graph.heldArrays(&g, f0, end, false) -| in_layers) + widest + 4 * spans.len;
                std.debug.print("DSV41_WIRE_ARRAYS_K16 {{\"prompt_wave\": {d}}}\n", .{k16_arrays});
                try testing.expect(k16_arrays <= @import("deepseek_v41_bill.zig").wire_arrays_prompt_wave);
            }
            const h = if (t.tier.layer_major) Held{ .reset = 0, .outside = 0, .layer = 0, .wave = 0, .widest_at = 0 } else Held.of(&g, f0, g.nodes.items.len, g.freed.items[w0..]);
            if (i == 0 or h.outside + h.layer > worst.outside + worst.layer) {
                worst = h;
                worst_i = i;
            }
            if (i + 2 >= drive.len) {
                var nb: [96]u8 = undefined;
                h.print(try std.fmt.bufPrint(&nb, "{s} tier, 16K chunk {d} ({d} rows)", .{ t.name, i, sp[1] - sp[0] }));
            }
        }
        var nb: [96]u8 = undefined;
        worst.print(try std.fmt.bufPrint(&nb, "{s} tier, 16K widest chunk {d}", .{ t.name, worst_i }));
        // Released score chains: the stock tier's masked-full chunk keeps two 8 GB score blocks of its
        // five (plus the indexer's), under half its layer wave; K30's gathered chain under 3 / 5 of it.
        // K16's forward is one call over every chunk: its held bytes are the reading here (no chain-share rule).
        // The prefill core (served) gathers nothing: the indexer's score chain is the widest wave left,
        // still under the layer's.
        if (std.mem.eql(u8, t.name, "stock")) try testing.expect(2 * worst.wave < worst.layer) else if (!t.tier.layer_major) {
            if (t.tier.routes.prefill_attn) try testing.expect(worst.wave < worst.layer) else try testing.expect(5 * worst.wave < 3 * worst.layer);
        }
        // A 5-row verify after the prompt: the served tier's head is C11's m1rows with the headpad
        // (a [6, dim] bf16 concat), the stock tier's the dense head.
        const n5 = g.nodes.items.len;
        const r5 = try model_.forward(&g, &st, prompt[0..5], .{ .logits = .all }, stand, graph.NoProbe{});
        try testing.expect(g.shapeOf(r5.logits.?).eql(ops.Shape.of(&.{ 1, 5, @intCast(c.vocab_size) })));
        var padded = false;
        for (g.nodes.items[n5..]) |nd| padded = padded or (nd.op == .concat and nd.shape.eql(ops.Shape.of(&.{ 6, @intCast(c.hidden_size) })));
        try testing.expectEqual(!std.mem.eql(u8, t.name, "stock"), padded);
    }
}

test "dsv41 dspark loop: a pinned subset head keeps only its experts, maps every routed id through its lut and otherwise drafts as the full head" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    // The mini head (one stage) at 4 experts, so a subset can keep some, leave some out and
    // move a kept expert to another slot: the full and the compact head over the same residents.
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var c4 = rig.m.c;
    c4.dspark.n_routed_experts = 4;
    const c = &c4;
    const lookup4: mdl.SpecLookup = .{ .g = &rig.g, .spec = try v41.residentSpec(arena.allocator(), c) };
    const full = try Loop(TraceOps).H.init(a, &rig.g, c4, .{}, &lookup4);
    defer full.deinit(&rig.g);
    const n_st = full.nStages();
    try testing.expectEqual(@as(usize, 1), n_st);
    const n: u16 = 4;
    // Stage 0 keeps {1, 3}: 1 -> slot 0, 3 -> slot 1, the left-out 0 and 2 -> slot 0.
    const text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"kind\": \"test\", \"n_experts\": 4, \"selected\": [[1, 3]]}";
    try rig.m.tmp.dir.writeFile(testing.io, .{ .sub_path = "subset.json", .data = text });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/subset.json", .{root[0..try rig.m.tmp.dir.realPath(testing.io, &root)]});
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    var sub = try dh.Subset.load(a, testing.io, .{ .path = path, .sha256 = &hex }, null);
    defer sub.deinit();
    var dl: DropLookup = .{ .inner = &lookup4 };
    defer dl.deinit();
    rig.g.record_host = true;
    const compact = try Loop(TraceOps).H.initWith(a, &rig.g, c4, .{}, &dl, .{ .subset = &sub });
    defer compact.deinit(&rig.g);
    // Only the kept experts' arrays are asked for (in ascending order: slot i is kept
    // expert i); every per-expert array is dropped, a left-out one before anything read it.
    var nb: [96]u8 = undefined;
    for (sub.selected, 0..) |kept, st| {
        var slot: usize = 0;
        for (0..n) |e| inline for (.{ "w1", "w3", "w2" }) |w| inline for (.{ "weight", "scales" }) |part| {
            const name = try std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}." ++ w ++ "." ++ part, .{ st, e });
            const is_kept = std.mem.indexOfScalar(u16, kept, @intCast(e)) != null;
            try testing.expectEqual(is_kept, DropLookup.has(dl.got.items, name));
            try testing.expect(DropLookup.has(dl.dropped.items, name));
        };
        for (dl.got.items) |name| {
            var buf2: [96]u8 = undefined;
            const want = try std.fmt.bufPrint(&buf2, "mtp.{d}.ffn.experts.", .{st});
            if (!std.mem.startsWith(u8, name, want) or !std.mem.endsWith(u8, name, ".w1.weight")) continue;
            const e = try std.fmt.parseInt(u16, name[want.len..std.mem.indexOfScalarPos(u8, name, want.len, '.').?], 10);
            try testing.expectEqual(kept[slot], e);
            slot += 1;
        }
        try testing.expectEqual(kept.len, slot);
        // The compact banks and the lut (`positions.get(expert, 0)`, run_full.py:92).
        const stg = compact.stages[st];
        inline for (.{ "w1", "w3", "w2" }) |w| try testing.expectEqual(@as(c_int, @intCast(kept.len)), rig.g.shapeOf(@field(stg.experts, w).w).dim(0));
        const lut = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, rig.g.hostBytesOf(stg.lut.?).?));
        try testing.expectEqual(@as(usize, n), lut.len);
        for (0..n) |e| {
            const want: i32 = if (std.mem.indexOfScalar(u16, kept, @intCast(e))) |i| @intCast(i) else 0;
            try testing.expectEqual(want, lut[e]);
        }
    }
    try testing.expect(compact.identity == .compact);
    try testing.expectEqualSlices(u8, &sub.sha256, &compact.identity.compact);
    var lut4: [4]i32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&lut4), rig.g.hostBytesOf(compact.stages[0].lut.?).?);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1 }, &lut4);
    try testing.expect(full.identity == .full and full.pruned_bytes == 0 and full.stages[0].lut == null);
    // The credit: the left-out experts' bytes (2 of them) at the head's per-expert bytes.
    try testing.expectEqual(@as(u64, 2), sub.pruned());
    try testing.expectEqual(sub.pruned() * dh.expertBytes(c), compact.pruned_bytes);
    // One expert's six arrays, as bound, are `expertBytes`.
    var per_expert: u64 = 0;
    inline for (.{ "w1", "w3", "w2" }) |w| inline for (.{ "weight", "scales" }) |part| {
        const x = lookup4.get("mtp.0.ffn.experts.0." ++ w ++ "." ++ part).?;
        per_expert += @intCast(rig.g.shapeOf(x).numel() * @as(i64, @intCast(ops.dtypeSize(rig.g.dtypeOf(x)))));
    };
    try testing.expectEqual(per_expert, dh.expertBytes(c));
    // A draft block: the same graph as the full head's, plus each stage MoE's lut gather.
    var script: Script = .{ .n_experts = @intCast(c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..n_st], .{ .lookup = null, .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    const e0 = rig.g.nodes.items.len;
    _ = try full.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e1 = rig.g.nodes.items.len;
    _ = try compact.draftBlock(&rig.g, lp.main_h.?, 3, rig.caches[0..n_st], rig.model.embed, rig.model.head);
    const e2 = rig.g.nodes.items.len;
    try testing.expectEqual(e1 - e0 + n_st, e2 - e1);
    inline for (@typeInfo(ops.Op).@"enum".field_names) |name| {
        const op = @field(ops.Op, name);
        var nf: usize = 0;
        var nc: usize = 0;
        for (rig.g.nodes.items[e0..e1]) |nd| nf += @intFromBool(nd.op == op);
        for (rig.g.nodes.items[e1..e2]) |nd| nc += @intFromBool(nd.op == op);
        if (op == .take) {
            try testing.expectEqual(nf + n_st, nc);
        } else try testing.expectEqual(nf, nc);
    }
}

test "dsv41 dspark loop: a logged cycle that wants the tie-flip rule reads each verify row's top two and rms" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    // One cycle (block 2, greedy): drafts 5, 6; verify rows [3, 5, 6] -> targets 5, 9, 7 (accepts 1).
    // Then the logged top two: ids [5, 9, 7] / [8, 2, 4], logits [2, 1, 3] / [1.5, 0.5, 2.5], rms [4, 4, 4].
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 9, 7 }, &.{ 5, 9, 7 }, &.{ 8, 2, 4 } },
        .f32s = &.{ &.{ 0.9, 0.8 }, &.{ 2, 1, 3 }, &.{ 1.5, 0.5, 2.5 }, &.{ 4, 4, 4 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 6 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var lg: CycleLog = .{ .primary = 0, .want_top = true };
    const e0 = rig.g.nodes.items.len;
    _ = try lp.cycle(&rig.ex, &out, a, &lg);
    try testing.expectEqual(@as(usize, 4), script.nu);
    try testing.expectEqual(@as(usize, 4), script.nf);
    try testing.expectEqualSlices(u32, &.{ 5, 9 }, out.items);
    try testing.expectEqual([2]u32{ 9, 2 }, lg.top_ids[1]);
    try testing.expectEqual([2]f32{ 1, 0.5 }, lg.top_logits[1]);
    try testing.expectEqual(@as(f32, 4), lg.rms[2]);
    // The verify's own argmax, plus the logged first and second (masked) argmax.
    var n_argmax: usize = 0;
    var n_where: usize = 0;
    for (rig.g.nodes.items[e0..]) |nd| {
        n_argmax += @intFromBool(nd.op == .argmax);
        n_where += @intFromBool(nd.op == .where);
    }
    try testing.expect(n_argmax >= 3 and n_where >= 1);
}

test "dsv41 dspark loop: the install warm-up traces every region at the served shapes; cycles at every row count then trace none" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    // Every decode region route on: ATTN / HC / SMALL_STAGES compile and the K33 draft regions.
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_ATTN_COMPILE", "1" }, .{ "MTPLX_DSV41_HC_COMPILE", "1" }, .{ "MTPLX_DSV41_SMALL_STAGES_FUSED", "1" }, .{ "MTPLX_DSV41_DRAFT_COMPILE", "1" } }, null);
    const model = try Loop(TraceOps).M.init(a, &rig.g, rig.m.c, tier, &rig.lookup, &rig.m.src);
    defer model.deinit(&rig.g);
    const head = try Loop(TraceOps).H.init(a, &rig.g, rig.m.c, tier.routes, &rig.lookup);
    defer head.deinit(&rig.g);
    var st = try model.newState();
    defer st.deinit(&rig.g, a);
    var caches: [4]Loop(TraceOps).H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&rig.g);
    // Three cycles verifying 2, 2 and 3 rows (block 2, confidence 0.5, greedy, no lookup; the confidence stop
    // keeps at least one draft, so 2 .. max_rows are the served row counts).
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 5, 7 }, &.{ 8, 9 }, &.{ 8, 10 }, &.{ 11, 12 }, &.{ 11, 12, 13 } },
        .f32s = &.{ &.{ 0.2, 0.2 }, &.{ 0.9, 0.2 }, &.{ 0.9, 0.9 } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, model, head, &st, caches[0..head.nStages()], .{ .k_request = 5, .lookup = null, .max_tokens = 64 });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 3), lp.max_rows);
    const before = rig.g.compiles;
    var peaks: [4]u64 = @splat(1);
    try lp.warm(a, &rig.ex, &peaks);
    try testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0 }, &peaks); // the trace backend holds no device memory
    try testing.expect(rig.g.compiles > before);
    try testing.expectEqual(@as(u32, 0), st.offset);
    // The prompt pass traces its own shapes (8-row forwards here); the cycles then replay warmed traces only.
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(1 + i % 30);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    const warmed = rig.g.compiles;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var verified: [3]u32 = undefined;
    for (&verified) |*v| {
        var lg: CycleLog = .{ .primary = 0 };
        _ = try lp.cycle(&rig.ex, &out, a, &lg);
        v.* = lg.verified;
    }
    try testing.expectEqualSlices(u32, &.{ 2, 2, 3 }, &verified);
    try testing.expectEqual(warmed, rig.g.compiles);
}

test "dsv41 dspark loop: the seed keeps its copies only: the bank's retained prompt state is 847,872 B at 16K" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 seed: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank, &diag);
    // main_h 1 x 3 x 5120 x 4 = 61,440; the windows 3 x 128 x 512 x 4 = 786,432 (the views held 16,384 x 67,584).
    try testing.expectEqual(@as(u64, 847_872), seedRetainedBytes(&c, 16_384));
    try testing.expectEqual(seedRetainedBytes(&c, 16_384), seedRetainedBytes(&c, c.window));
    // A prompt shorter than the window keeps only its rows.
    try testing.expectEqual(@as(u64, 61_440 + 3 * 64 * 512 * 4), seedRetainedBytes(&c, 64));
}

test "dsv41 dspark loop: on a bounded state the lookup's history and key map are reserved to the request's positions" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var st = try rig.model.newStateWith(rig.model.boundedKv(64));
    defer st.deinit(&rig.g, a);
    var caches: [4]Loop(TraceOps).H.Cache = @splat(.{});
    defer for (caches[0..rig.head.nStages()]) |*x| x.deinit(&rig.g);
    var script: Script = .{ .n_experts = @intCast(rig.m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &st, caches[0..rig.head.nStages()], .{ .max_tokens = 8 });
    defer lp.deinit();
    var prompt: [9]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(i + 1);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    const lk = &lp.lookup.?;
    try testing.expect(lk.history.capacity >= st.max_len.?);
    try testing.expect(lk.ends.capacity() >= st.max_len.?);
}

/// The loop on MLX with the served expert source, analysed on the host (never run): the
/// MLX backend's warm-up measurement, the logged verify reads and the cycle compile.
fn mlxSmoke(lp: *Loop(ops.MlxOps), a: std.mem.Allocator, ex: *xp.ExpertsWith(ops.MlxOps, xp.StreamSource, xp.QuantMath(ops.MlxOps, @import("exl3_quant.zig").Accepted(ops.MlxOps)), .{ .prefill = true }), out: *std.ArrayList(u32)) !void {
    var peaks: [ds.max_block + 2]u64 = undefined;
    try lp.warm(a, ex, &peaks);
    var lg: CycleLog = .{ .primary = 0, .want_top = true };
    _ = try lp.cycle(ex, out, a, &lg);
}

test "dsv41 dspark loop: the MLX instantiation of the loop analyses (host, nothing runs)" {
    try testing.expect(@TypeOf(&mlxSmoke) != void);
}

/// A wide route that records each call: its layer (routes are built in layer
/// order), rows, act rows and slots, in the order the forward makes them.
const WideLog = struct {
    const quant = @import("sdk_ext.zig").quant;
    var next_layer: u32 = 0;
    var order: [512]u32 = undefined;
    var n_order: usize = 0;
    layer: u32,
    calls: u32 = 0,
    rows: u32 = 0,
    finishes: u32 = 0,
    ok: bool = true,

    pub fn init() WideLog {
        next_layer += 1;
        return .{ .layer = next_layer - 1 };
    }
    pub fn call(self: *WideLog, g: *TraceOps, act: u32, r: quant.PrefillRows, bank: xp.BankArraysOf(u32)) !u32 {
        self.calls += 1;
        self.rows += @intCast(r.slot.len);
        order[n_order] = self.layer;
        n_order += 1;
        const tokens: u32 = @intCast(g.shapeOf(act).d[0]);
        const cap: u32 = @intCast(g.shapeOf(bank.gate.code).d[0]);
        const act_row = r.act_row.?;
        self.ok = self.ok and g.dtypeOf(act) == .bfloat16 and act_row.len == r.slot.len;
        for (r.slot, act_row) |slot, row| self.ok = self.ok and slot < cap and row < tokens;
        return g.input(&.{ @intCast(r.slot.len), g.shapeOf(act).d[1] }, .float32);
    }
    pub fn finish(self: *WideLog, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

test "dsv41 dspark loop: the served prompt pass is one forward the model chunks; its wide chunks run through the wide lane, chunk-major" {
    const a = testing.allocator;
    const m = try mdl.Mini.init();
    defer m.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = m.spec };
    // The model's own chunk rule, pinned small (30 rows) so a mini prompt spans several chunks.
    const model = try Loop(TraceOps).M.init(a, &g, m.c, try routes.parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "30" }}, null), &lookup, &m.src);
    defer model.deinit(&g);
    const head = try Loop(TraceOps).H.init(a, &g, m.c, .{}, &lookup);
    defer head.deinit(&g);
    const nl = m.c.n_layers;
    const k = m.c.n_experts_per_tok;
    var rows0: [8]u32 = @splat(@intCast(m.c.n_routed_experts));
    var src = try xp.FakeSource.init(a, .{ .hidden = m.c.hidden_size, .inter = m.c.moe_intermediate_size, .n_experts = m.c.n_routed_experts, .rows = rows0[0..nl] });
    defer src.deinit();
    WideLog.next_layer = 0;
    WideLog.n_order = 0;
    var logs: [8]WideLog = undefined;
    for (logs[0..nl]) |*l| l.* = WideLog.init();
    const Math = xp.WithPrefillRoutes(TraceOps, xp.TraceMath, WideLog);
    const Ex = xp.ExpertsWith(TraceOps, xp.FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(a, &g, &src, .{ .d = .{ .hidden = @intCast(m.c.hidden_size), .inter = @intCast(m.c.moe_intermediate_size) }, .routes = logs[0..nl] }, &m.c);
    defer ex.deinit();
    var script: Script = .{ .n_experts = @intCast(m.c.n_routed_experts), .pick = 3, .u32s = &.{}, .f32s = &.{} };
    g.host_values = script.values();
    // 70 prompt rows: spans 30 / 30 / 10. The first two are wider than a route takes (30 x k > 48), the last is not.
    const n_prompt = 70;
    try testing.expect(30 * k > xp.max_route_ids and 10 * k <= xp.max_route_ids);
    var prompt: [n_prompt]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast(1 + i % 50);
    // The request's KV bounded to its positions: the prompt, 4 tokens and one verify block.
    var st = try model.newStateWith(model.boundedKv(n_prompt + 4 + 8));
    defer st.deinit(&g, a);
    const max_len = st.max_len orelse return error.TestUnexpectedResult;
    try testing.expect(max_len >= n_prompt + 4 + 8);
    // The bounded state's n-gram history is reserved to its admitted length at state build.
    try testing.expect(st.hash.?.hist.capacity >= max_len);
    var caches: [4]Loop(TraceOps).H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);
    var lp = Loop(TraceOps).init(&g, model, head, &st, caches[0..head.nStages()], .{ .lookup = null, .max_tokens = 4, .prompt_chunk = whole_prompt });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 3), try lp.prefill(a, &ex, &prompt));
    try testing.expectEqual(@as(u32, n_prompt), st.offset);
    try testing.expectEqual(@as(u32, n_prompt), caches[0].offset);
    // Every layer's wide route took the two wide chunks' rows (60 x k), act rows indexing the chunk,
    // slots inside the bank, bf16 act; every call drains every layer's waves (the quant's finishPrefill).
    var wide_calls: u32 = 0;
    for (logs[0..nl]) |r| wide_calls += r.calls;
    for (logs[0..nl]) |r| {
        try testing.expect(r.ok);
        try testing.expectEqual(@as(u32, 60 * k), r.rows);
        try testing.expectEqual(wide_calls, r.finishes);
    }
    // Chunk-major: chunk 0 through layers 0 .. L-1, then chunk 1 (a layer's calls in a chunk adjacent).
    var runs: [64]u32 = undefined;
    var n_runs: usize = 0;
    for (WideLog.order[0..WideLog.n_order]) |l| {
        if (n_runs == 0 or runs[n_runs - 1] != l) {
            runs[n_runs] = l;
            n_runs += 1;
        }
    }
    try testing.expectEqual(@as(usize, 2 * nl), n_runs);
    for (runs[0..n_runs], 0..) |l, i| try testing.expectEqual(@as(u32, @intCast(i % nl)), l);
    // The narrow last chunk takes the decode lane: one route per layer per chunk, wide or not.
    try testing.expectEqual(@as(u64, 3 * nl), src.stats().route_calls);
    // A forward past the request's positions is refused before any lane is written.
    const room: usize = max_len - st.offset;
    const too_many = try a.alloc(u32, room + 1);
    defer a.free(too_many);
    @memset(too_many, 1);
    try testing.expectError(error.BoundedLaneFull, model.forward(&g, &st, too_many, .{ .logits = .none }, &ex, graph.NoProbe{}));
    try testing.expectEqual(@as(u32, n_prompt), st.offset);
}

test "dsv41 dspark loop: typical flags accept what the argmax rejects; the correction stays the argmax" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 7, 8, 9 }, &.{ 11, 12 }, &.{ 13, 14, 15 } },
        .f32s = &.{ &.{ 0.9, 0.9 }, &.{ 0.9, 0.9 } },
        // cycle 1: both drafts typical though the argmax disagrees: bonus 9;
        // cycle 2: depth 0 typical, depth 1 not: the correction is that row's argmax, 14.
        .bools = &.{ &.{ true, true }, &.{ true, false } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .lookup = null, .acceptance = .{ .typical = .{ .delta = 0.5 } }, .max_tokens = 5 });
    defer lp.deinit();
    var prompt: [9]u32 = @splat(4);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    try testing.expectEqual(Finish.length, try lp.run(&rig.ex, &out, a));
    try testing.expectEqualSlices(u32, &.{ 5, 6, 9, 11, 14 }, out.items);
    try testing.expectEqual(@as(u32, 2), script.nb);
    // The typical decision ran on the device: logsumexp over the drafted rows, one flag per draft.
    var n_lse: usize = 0;
    for (rig.g.nodes.items) |nd| n_lse += @intFromBool(nd.op == .logsumexp);
    try testing.expectEqual(@as(usize, 2), n_lse);
}

test "dsv41 dspark loop: the typical decision is realised by the verify's eval (two syncs per cycle), then read" {
    const a = testing.allocator;
    var rig: Rig = undefined;
    try rig.init();
    defer rig.deinit();
    var script: Script = .{
        .n_experts = @intCast(rig.m.c.n_routed_experts),
        .pick = 3,
        .u32s = &.{ &.{ 5, 6 }, &.{ 7, 8, 9 }, &.{ 11, 12 }, &.{ 13, 14, 15 } },
        .f32s = &.{ &.{ 0.9, 0.9 }, &.{ 0.9, 0.9 } },
        .bools = &.{ &.{ true, true }, &.{ true, false } },
    };
    rig.g.host_values = script.values();
    var lp = Loop(TraceOps).init(&rig.g, rig.model, rig.head, &rig.st, rig.caches[0..rig.head.nStages()], .{ .lookup = null, .acceptance = .{ .typical = .{ .delta = 0.5 } }, .max_tokens = 5 });
    defer lp.deinit();
    var prompt: [9]u32 = @splat(4);
    _ = try lp.prefill(a, &rig.ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const g = &rig.g;
    for (0..2) |_| {
        const n0 = g.nodes.items.len;
        const e0 = g.evals.items.len;
        _ = try lp.cycle(&rig.ex, &out, a, null);
        const evals = g.evals.items[e0..];
        const nodes = g.nodes.items;
        // The draft's eval and the verify's, which also realises the decision (argmax, logsumexp, flags).
        try testing.expectEqual(@as(usize, 2), evals.len);
        var lse: usize = 0;
        for (nodes[n0..evals[1]]) |nd| lse += @intFromBool(nd.op == .logsumexp);
        try testing.expectEqual(@as(usize, 1), lse);
        // Its reads follow at once: the argmax, then the flags, nothing built between.
        try testing.expectEqual(ops.Op.host_read, nodes[evals[1]].op);
        try testing.expectEqual(ops.Op.host_read, nodes[evals[1] + 1].op);
    }
    // The decisions of the unfolded decide (the test above's script).
    try testing.expectEqualSlices(u32, &.{ 5, 6, 9, 11, 14 }, out.items);
}

// DSV41_DSPARK_CYCLES_FIXTURE=<json from R/exl3/runtime/dump_dsv41_dspark_cycles.py>
test "dsv41 dspark loop: the lane's recorded cycles replay decision for decision" {
    const path = std.mem.span(std.c.getenv("DSV41_DSPARK_CYCLES_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const Cycle = struct {
        primary: u32,
        draft_ids: []const u32,
        conf_sigmoid: []const f32,
        k_eff_native: u32,
        native: []const u32,
        drafts: []const u32,
        targets: []const []const u32,
        flags: []const []const bool = &.{},
        verified: u32,
        kept: u32,
        emitted: []const u32 = &.{},
    };
    const Fixture = struct {
        format: []const u8,
        arm: []const u8,
        delta: ?f64 = null,
        confidence_threshold: f64,
        config: std.json.Value,
        prompt: []const u32,
        max_tokens: u32,
        tokens: []const u32,
        cycles: []const Cycle,
        stats: struct { drafted_by_depth: []const u32, accepted_by_depth: []const u32, cycles: u32, bonus_tokens: u32, correction_tokens: u32 },
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    try testing.expectEqualStrings("mlx-serve-dsv41-dspark-cycles-v1", f.format);
    const cfg_json = try std.json.Stringify.valueAlloc(a, f.config, .{});
    defer a.free(cfg_json);
    var diag: v41.Diag = .{};
    const c = v41.Config.parse(a, cfg_json, &diag) catch |e| {
        std.debug.print("fixture config refused: {s}\n", .{diag.message()});
        return e;
    };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const M = Loop(TraceOps).M;
    const model = try M.init(a, &g, c, try routes.parse(&.{}, null), &lookup, null);
    defer model.deinit(&g);
    const head = try Loop(TraceOps).H.init(a, &g, c, .{}, &lookup);
    defer head.deinit(&g);
    var st = try model.newState();
    defer st.deinit(&g, a);
    var caches: [4]Loop(TraceOps).H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);
    var rows: [8]u32 = @splat(0);
    var grown: [8]u32 = @splat(@intCast(c.n_routed_experts));
    var src = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows[0..c.n_layers] });
    defer src.deinit();
    var ex = try xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath).init(a, &g, &src, .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) }, &c);
    defer ex.deinit();
    try ex.grow(&g, grown[0..c.n_layers]);
    // The loop's host reads, in its order: per cycle the draft ids, their sigmoid
    // confidences, then per verify chunk the argmax rows (and the typical flags).
    var u32s: std.ArrayList([]const u32) = .empty;
    defer u32s.deinit(a);
    var f32s: std.ArrayList([]const f32) = .empty;
    defer f32s.deinit(a);
    var bools: std.ArrayList([]const bool) = .empty;
    defer bools.deinit(a);
    for (f.cycles) |cy| {
        try u32s.append(a, cy.draft_ids);
        try f32s.append(a, cy.conf_sigmoid);
        for (cy.targets, 0..) |t, i| {
            try u32s.append(a, t);
            if (i < cy.flags.len) try bools.append(a, cy.flags[i]);
        }
    }
    var script: Script = .{ .n_experts = @intCast(c.n_routed_experts), .pick = f.tokens[0], .u32s = u32s.items, .f32s = f32s.items, .bools = bools.items };
    g.host_values = script.values();
    const acceptance: ds.Acceptance = if (std.mem.eql(u8, f.arm, "typical")) .{ .typical = .{ .delta = @floatCast(f.delta.?) } } else .greedy;
    var lp = Loop(TraceOps).init(&g, model, head, &st, caches[0..head.nStages()], .{ .confidence_threshold = f.confidence_threshold, .acceptance = acceptance, .max_tokens = f.max_tokens - 1 });
    defer lp.deinit();
    try testing.expectEqual(f.tokens[0], try lp.prefill(a, &ex, f.prompt));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    for (f.cycles, 0..) |cy, i| {
        var lg: CycleLog = .{ .primary = 0 };
        const fin = try lp.cycle(&ex, &out, a, &lg);
        errdefer std.debug.print("cycle {d} differs\n", .{i});
        try testing.expectEqual(cy.primary, lg.primary);
        try testing.expectEqual(cy.k_eff_native, lg.k_native);
        try testing.expectEqualSlices(u32, cy.drafts, lg.drafts[0..lg.k_eff]);
        try testing.expectEqual(cy.kept - 1, lg.accepted);
        try testing.expectEqual(cy.verified, lg.verified);
        try testing.expectEqual(cy.verified - cy.kept, lg.trimmed);
        // Uncut cycles emit the accepted drafts and their correction / bonus.
        if (cy.emitted.len == lg.accepted + 1) try testing.expectEqual(cy.emitted[lg.accepted], lg.correction);
        try testing.expectEqual(i + 1 == f.cycles.len, fin != null);
    }
    try testing.expectEqualSlices(u32, f.tokens[1..], out.items);
    try testing.expectEqualSlices(u32, f.stats.drafted_by_depth, lp.stats.drafted_by_depth[0..f.stats.drafted_by_depth.len]);
    try testing.expectEqualSlices(u32, f.stats.accepted_by_depth, lp.stats.accepted_by_depth[0..f.stats.accepted_by_depth.len]);
    try testing.expectEqual(f.stats.cycles, lp.stats.cycles);
    try testing.expectEqual(f.stats.bonus_tokens, lp.stats.bonus_tokens);
    try testing.expectEqual(f.stats.correction_tokens, lp.stats.correction_tokens);
    std.debug.print("dsv41 dspark loop: {d} recorded {s} cycles replay decision for decision ({d} tokens)\n", .{ f.cycles.len, f.arm, f.tokens.len });
}

// DSV41_DSPARK_MINI_CONFIG=<dump_dsv41_dspark_cycles.py --dump-config json>
test "dsv41 dspark loop: the fixture's mini config builds the model and head, and a full proposal takes the lookup" {
    const path = std.mem.span(std.c.getenv("DSV41_DSPARK_MINI_CONFIG") orelse return error.SkipZigTest);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(text);
    try miniConfigCycles(text);
}

test "dsv41 dspark loop: the m4 mini config builds the model and head, and a full proposal takes the lookup (embedded)" {
    try miniConfigCycles(@embedFile("fixtures/dsv41_dspark_mini_config.json"));
}

fn miniConfigCycles(text: []const u8) !void {
    const a = testing.allocator;
    var diag: v41.Diag = .{};
    const c = v41.Config.parse(a, text, &diag) catch |e| {
        std.debug.print("mini config refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expectEqual(@as(u32, 5), c.dspark.block_size);
    try testing.expectEqual(@as(u32, 0), c.engram.n_layers);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const L = Loop(TraceOps);
    const model = try L.M.init(a, &g, c, try routes.parse(&.{}, null), &lookup, null);
    defer model.deinit(&g);
    const head = try L.H.init(a, &g, c, .{}, &lookup);
    defer head.deinit(&g);
    try testing.expectEqual(@as(usize, 2), head.nStages());
    var st = try model.newState();
    defer st.deinit(&g, a);
    var caches: [4]L.H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);
    var rows: [8]u32 = @splat(0);
    var grown: [8]u32 = @splat(@intCast(c.n_routed_experts));
    var src = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows[0..c.n_layers] });
    defer src.deinit();
    var ex = try xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath).init(a, &g, &src, .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) }, &c);
    defer ex.deinit();
    try ex.grow(&g, grown[0..c.n_layers]);
    // Prompt "... 9 | 1 2 3 4 5 6 7 | ... 9": after the pick 9, the draft 1 2 3 4 5 is extended by 6 7.
    const prompt = [_]u32{ 30, 31, 8, 9, 1, 2, 3, 4, 5, 6, 7, 40, 41, 8 };
    var script: Script = .{
        .n_experts = @intCast(c.n_routed_experts),
        .pick = 9,
        .u32s = &.{ &.{ 1, 2, 3, 4, 5 }, &.{ 1, 2, 3, 4, 5, 6, 50, 51 } },
        .f32s = &.{&.{ 0.9, 0.9, 0.9, 0.9, 0.9 }},
    };
    g.host_values = script.values();
    var lp = L.init(&g, model, head, &st, caches[0..head.nStages()], .{ .max_tokens = 64 });
    defer lp.deinit();
    try testing.expectEqual(@as(u32, 8), lp.max_rows);
    try testing.expectEqual(@as(u32, 7), lp.stats.speculative_depth);
    _ = try lp.prefill(a, &ex, &prompt);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var lg: CycleLog = .{ .primary = 0 };
    try testing.expect(try lp.cycle(&ex, &out, a, &lg) == null);
    try testing.expectEqual(@as(u32, 5), lg.k_native);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 7 }, lg.drafts[0..lg.k_eff]);
    // The target keeps 1..6 and says 50 at the second lookup depth.
    try testing.expectEqual(@as(u32, 6), lg.accepted);
    try testing.expectEqual(@as(u32, 50), lg.correction);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 50 }, out.items);
    try testing.expectEqualSlices(u32, &.{ 1, 1, 1, 1, 1, 1, 1 }, lp.stats.drafted_by_depth[0..7]);
    try testing.expectEqualSlices(u32, &.{ 1, 1, 1, 1, 1, 1, 0 }, lp.stats.accepted_by_depth[0..7]);
}

// DSV41_PHASE0B_MLX=1 only (a GPU-lock-held run: any MLX array creates the Metal device).
test "dsv41 smoke 0b: the seed's row copy owns its rows and equals the view (MLX)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const mlx = @import("sdk").mlx;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(testing.allocator, s);
    defer g.deinit();
    var vals: [2 * 5 * 3]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const x = try g.hostArray(std.mem.sliceAsBytes(&vals), &.{ 2, 5, 3 }, .float32);
    const L = Loop(ops.MlxOps);
    const view = try L.sliceRows(&g, x, 1, 4);
    const copy = try L.copyRows(&g, x, 1, 4);
    try g.evalAll(&.{ view, copy });
    var a: [2 * 3 * 3]f32 = undefined;
    var b: [2 * 3 * 3]f32 = undefined;
    try testing.expectEqualSlices(f32, try g.hostF32(view, &a), try g.hostF32(copy, &b));
    // The copy's data is its own buffer, not the source's (a view keeps the whole source alive).
    try testing.expect(mlx.mlx_array_data_float32(copy) != mlx.mlx_array_data_float32(view));
}
