//! The DeepSeek-V4.1 text model over the trunk graphs: residents bound once
//! (every layer, embedding, final norm, head, Engram), per-sequence state, the
//! forward over a span (one shot, chunk-major, or K16 layer-major prefill; a
//! decode token; a DSpark verify block) and its trim / mark / rollback.
//! Generic over the op backend; the routed experts are the caller's source
//! (`routed(g, xf, indices) -> [n, k, dim]`, the expert streamer when served).
//!
//! DSpark seam (M2): a verify forward runs `[t, d1 .. dK]` with
//! `.{ .logits = .all, .main_hidden = true }`; the decode loop accepts `a`
//! drafts and calls `trim(K - a)`; the draft reads `main_hidden`'s committed rows.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const graph = @import("deepseek_v41_graph.zig");
const kr = @import("dsv41_kernel_routes.zig");
const kvc = @import("deepseek_v41_cache.zig");
const eng = @import("deepseek_v41_engram.zig");
const xk = @import("exl3_kernels.zig");
const routes = @import("deepseek_v41_routes.zig");
const expert_policy = @import("sdk_ext.zig").expert.policy;
/// PROFILE builds only: P1's read-ahead record (compiles to nothing otherwise).
const prof = @import("dsv41_prefill_timers.zig");
const ngram = @import("ngram_table.zig");

/// K16: each chunk's DSpark main tap is evaluated in its chunk fence (`forwardLayerMajor`), so the tap's mean does
/// not hold the layer's input stream (hc x the tap's bytes) to the forward's end. The bill reads this declaration
/// (`@hasDecl`): with it, the routed group's live hc-width streams are one (the HC post's matmul output) on every
/// layer; without it, the DSpark target layers keep their input streams through the prompt pass.
pub const main_taps_in_chunk_fence = true;

/// K16's chunk timeline marks (`probeFence`): a chunk's host build starts, its fence starts (the build ends), its fence
/// returns. The fence profile (D14's price) reads them without syncs of its own.
pub const FenceAt = enum { build, wait, done };

pub const Want = struct {
    /// Head rows: none, the last position (a prefill), or every row (decode, verify).
    logits: enum { none, last, all } = .all,
    /// DSpark `main_hidden`: the target layers' attention inputs (mean over the
    /// hc copies), concatenated.
    main_hidden: bool = false,
};

pub const Error = error{ EngramSourceRequired, TrimTooDeep, MissingWeight, NameTooLong, EmbeddingRetired, EmbeddingRowsMismatch };

/// `_derive_moe_row_cap`: rows one K16 routed call may carry.
/// JOINLESS takes a wide call only (the experts' wide lane: more than one route of ids).
const joinless_min_ids = v41.PrefillBill.joinless_min_ids;

pub fn moeRowCap(c: *const v41.Config, target_bytes: f64) u64 {
    const per_row: u64 = @as(u64, c.n_experts_per_tok) * c.hidden_size * 4;
    return @max(1, @as(u64, @intFromFloat(@floor(@max(target_bytes, 1e9) / @as(f64, @floatFromInt(per_row))))));
}

pub fn Model(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const T = G.T;
        pub const Tr = graph.Trunk(G);
        pub const Cache = Tr.Cache;

        gpa: std.mem.Allocator,
        c: v41.Config,
        tier: routes.Tier,
        layers: []graph.LayerW(T),
        inv_swa: T,
        inv_yarn: T,
        /// The input embedding: the resident table until `retireEmbedding`
        /// (the prompt pass's fence), then its rows on the host.
        embed: Embed,
        norm_w: T,
        head: Tr.HeadW,
        engram: ?EngramBind = null,
        /// Arrays the model made (inverse frequencies, the quantized head, f32 wo_a).
        owned: std.ArrayList(T) = .empty,
        /// The tier's RC kernel routes (C12 ...), bound at `initWith`; empty on a stock tier.
        kx: Tr.Kernels = .{},
        /// C11: the verify head's m1rows route (rows <= 8), over the dense bf16 head.
        head_rows: ?kr.HeadRows(G) = null,
        /// HEAD_MODE mxfp8 on RCPROJ (rows <= 8: the verify rows and the draft block), over the quantized head.
        head_mx: ?kr.HeadMx(G) = null,
        /// C29: the Engram wkv's M-invariant rows route per Engram slot (rows <= 8).
        engram_m1: [eng.max_layers]?kr.Mxfp8Rows(G) = @splat(null),

        /// `posted`: the prompt pass gathers each Engram layer's rows ahead of it on the row source's poster
        /// threads (`eng.RowSource.post`), the first before the first layer, each later one once the layer
        /// before it took its own; the blocking read otherwise.
        const EngramBind = struct { src: *const eng.RowSource, w: [eng.max_layers]graph.EngramW(T), posted: bool = false };

        /// The input embedding's source. Either way a lookup of `ids` is the
        /// table's rows, byte for byte, `[1, n, dim]` in the table's dtype:
        /// the resident table, or its rows read from the checkpoint shard
        /// (`qwen4_exp.NgramTable.openTensor`, past the page cache).
        pub const Embed = union(enum) {
            table: T,
            rows: *ngram.NgramTable,

            /// `ids` embedded `[1, n, dim]`; host buffers come from `a`.
            pub fn of(self: Embed, g: *G, a: std.mem.Allocator, ids: []const u32, dim: u32) !T {
                const n: c_int = @intCast(ids.len);
                switch (self) {
                    .table => |w| {
                        const v = try a.alloc(i32, ids.len);
                        for (v, ids) |*d, s| d.* = @intCast(s);
                        return Tr.embed(g, w, try g.hostArray(std.mem.sliceAsBytes(v), &.{ 1, n }, .int32));
                    },
                    .rows => |r| {
                        const buf = try a.alloc(u8, ids.len * @as(usize, r.dim) * 2);
                        try r.gatherRaw(ids, buf);
                        return g.hostArray(buf, &.{ 1, n, @intCast(dim) }, .bfloat16);
                    },
                }
            }
        };
        pub const State = struct {
            offset: u32 = 0,
            layers: []Cache,
            hash: ?eng.HashState = null,
            /// The longest sequence every lane admits (null: unbounded), from the layers' geometry.
            max_len: ?u32 = null,
            /// Host scratch of a forward of at most `scratch_rows` rows (the decode / verify lane).
            scratch: []u8 = &.{},
            /// A prompt run as sub-chunk calls (`kvc.prefillSubCalls`): every call's spans at the chunk rule of the
            /// WHOLE prompt (set for the prompt's calls only), so each call runs the one-call pass's spans.
            span_chunk: ?i64 = null,

            pub fn deinit(self: *State, g: *G, gpa: std.mem.Allocator) void {
                for (self.layers) |*l| l.deinit(g);
                gpa.free(self.layers);
                gpa.free(self.scratch);
                if (self.hash) |*h| h.deinit(gpa);
            }
        };

        /// Rows a forward may run on the state's scratch (the decode lane's
        /// widest call: a verify block of 8 rows).
        pub const scratch_rows = 8;
        comptime {
            std.debug.assert(scratch_rows <= routes.min_prefill_chunk);
            // The ring's verify margin runs from this forward's rows to the box's top: never an empty box.
            std.debug.assert(scratch_rows <= routes.ring_lever_box.max_verify_max);
        }

        pub const Mark = struct { offset: u32, layers: []Cache.Mark };

        pub const Result = struct {
            /// The final-normed hidden `[1, s, dim]`.
            hidden: T,
            logits: ?T = null,
            main_hidden: ?T = null,
        };

        /// Bind every resident once. `lookup.get(name) ?T` resolves checkpoint
        /// (and Engram sidecar) names; `engram_src` is required when the config
        /// has Engram layers. The model must not move afterwards (compiled
        /// regions key on `&self.c`).
        pub fn init(gpa: std.mem.Allocator, g: *G, c: v41.Config, tier: routes.Tier, lookup: anytype, engram_src: ?*const eng.RowSource) !*Self {
            return initWith(gpa, g, c, tier, lookup, engram_src, .{});
        }

        /// `registry`: the accepted trunk routes' registry, required when the tier binds kernel
        /// routes (refused otherwise: TierNeedsKernels, before any resident binds).
        pub const Options = struct { registry: ?*const xk.Registry = null };

        pub fn initWith(gpa: std.mem.Allocator, g: *G, c: v41.Config, tier: routes.Tier, lookup: anytype, engram_src: ?*const eng.RowSource, opts: Options) !*Self {
            if (Tr.Kernels.needed(&tier.routes) and opts.registry == null) return error.TierNeedsKernels;
            if (tier.routes.rc_head_mxfp8) {
                if (tier.routes.head != .mxfp8) return error.HeadMxNeedsMxfp8;
                if (opts.registry == null) return error.HeadMxNeedsKernels;
            }
            // The ring's verify margin holds the widest block a forward appends (`scratch_rows`), and every ring lever
            // sits in the box the bill's ring tests cover: refused here, once, by name.
            try routes.checkRingGeometry(tier.kv, scratch_rows);
            const self = try gpa.create(Self);
            self.* = .{ .gpa = gpa, .c = c, .tier = tier, .layers = &.{}, .inv_swa = undefined, .inv_yarn = undefined, .embed = undefined, .norm_w = undefined, .head = undefined };
            errdefer self.deinit(g);
            const cp = &self.c;
            self.layers = try gpa.alloc(graph.LayerW(T), cp.n_layers);
            for (self.layers, 0..) |*lw, l| {
                lw.* = try bindLayer(lookup, cp.layers[l], @intCast(l));
                if (tier.routes.wo_a_f32) lw.wo_a_dense = try self.own(g, try Tr.woaDenseF32(g, cp, lw.wo_a));
                if (tier.routes.dense_rc) try self.stackSharedGateUp(g, lookup, lw, @intCast(l));
            }
            // The prefill core's sink views, once per layer (`W.sink4`).
            if (tier.routes.prefill_attn) for (self.layers) |*lw| {
                lw.sink4 = try self.own(g, try Tr.sinkView(g, cp, lw.attn_sink));
            };
            // DENSE16 o-projection: its rhs index pair, once, shared by every layer (`W.oproj_idx`).
            if (tier.routes.prefill_oproj) {
                const oi = try Tr.oprojIndices(g, cp);
                const owned: [2]T = .{ try self.own(g, oi[0]), try self.own(g, oi[1]) };
                for (self.layers) |*lw| lw.oproj_idx = owned;
            }
            self.inv_swa = try self.own(g, try Tr.swaInvFreq(g, cp));
            self.inv_yarn = try self.own(g, try Tr.yarnInvFreq(g, cp));
            self.embed = .{ .table = try req(lookup, "embed.weight") };
            self.norm_w = try req(lookup, "norm.weight");
            const head_w = try req(lookup, "head.weight");
            self.head = switch (tier.routes.head) {
                .f32, .bf16 => .{ .dense = head_w },
                .mxfp8 => blk: {
                    const q = try Tr.quantizeHead(g, head_w);
                    break :blk .{ .mxfp8 = .{ .w = try self.own(g, q.w), .s = try self.own(g, q.s), .mode = .mxfp8 } };
                },
            };
            if (cp.engram.n_layers > 0) {
                const src = engram_src orelse return error.EngramSourceRequired;
                var bind: EngramBind = .{ .src = src, .w = undefined };
                var b: [96]u8 = undefined;
                for (cp.engram.layer_ids[0..cp.engram.n_layers], 0..) |l, i| {
                    bind.w[i] = .{
                        .wkv = .{ .w = try reqf(lookup, &b, "layers.{d}.engram.wkv.weight", .{l}), .s = try reqf(lookup, &b, "layers.{d}.engram.wkv.scales", .{l}), .mode = .mxfp8 },
                        .q_weight = try reqf(lookup, &b, "layers.{d}.engram.q_weight", .{l}),
                        .k_weight = try reqf(lookup, &b, "layers.{d}.engram.k_weight", .{l}),
                    };
                }
                self.engram = bind;
            }
            if (opts.registry) |reg| {
                self.kx = try Tr.Kernels.init(gpa, g, reg, &self.c, &self.tier.routes, self.layers);
                if (tier.routes.rc_head) {
                    // The kernel reads the bf16 head (HEAD_MODE bf16: the served head's weight as bound).
                    if (tier.routes.head != .bf16) return error.HeadRowsNeedsBf16;
                    self.head_rows = kr.HeadRows(G).init(g, reg, self.head.dense, null) catch |e| return if (e == error.RouteInput) error.HeadRowsGeometry else e;
                }
                if (tier.routes.rc_mxfp8_rows) if (self.engram) |en| {
                    for (0..cp.engram.n_layers) |i| self.engram_m1[i] = try Tr.m1Site(g, reg, .engram_wkv, en.w[i].wkv);
                };
                if (tier.routes.rc_head_mxfp8) {
                    self.head_mx = kr.HeadMx(G).init(g, reg, self.head.mxfp8.w, self.head.mxfp8.s, null) catch |e| return if (e == error.RouteInput) error.HeadMxGeometry else e;
                }
            }
            try Tr.prepareRegions(g, &self.c, &self.tier.routes, self.tier.layer_major);
            try g.evalAll(self.owned.items);
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            for (self.owned.items) |x| g.release(x);
            self.owned.deinit(self.gpa);
            self.kx.deinit(g);
            if (self.head_rows) |*x| x.deinit(g);
            if (self.head_mx) |*x| x.deinit(g);
            for (&self.engram_m1) |*x| if (x.*) |*r| r.deinit(g);
            self.gpa.free(self.layers);
            self.gpa.destroy(self);
        }

        /// DENSE_RC: the layer's shared gate and up stacked `[2 I, H / 4]` (and their scales), evaluated, with
        /// `sh_w1` / `sh_w3` rebound as its halves (views); a mutable lookup that can forget arrays drops the two
        /// originals here, one layer at a time (else the owner drops them after construction: `droppedSharedGateUp`).
        fn stackSharedGateUp(self: *Self, g: *G, lookup: anytype, lw: *graph.LayerW(T), l: u32) !void {
            const w1 = lw.sh_w1;
            const w3 = lw.sh_w3;
            if (w1.mode != .mxfp8 or w3.mode != .mxfp8) return error.DenseRcMode;
            const w = try self.own(g, try g.concat(&.{ w1.w, w3.w }, 0));
            const sc = try self.own(g, try g.concat(&.{ w1.s, w3.s }, 0));
            try g.evalAll(&.{ w, sc });
            const n = g.shapeOf(w1.w).dim(0);
            const half = struct {
                fn of(g_: *G, x: T, lo: c_int, hi: c_int) !T {
                    const sh = g_.shapeOf(x);
                    return g_.slice(x, &.{ lo, 0 }, &.{ hi, sh.dim(1) }, &.{ 1, 1 });
                }
            }.of;
            lw.sh_w13 = .{ .w = w, .s = sc, .mode = .mxfp8 };
            lw.sh_w1 = .{ .w = try self.own(g, try half(g, w, 0, n)), .s = try self.own(g, try half(g, sc, 0, n)), .mode = .mxfp8 };
            lw.sh_w3 = .{ .w = try self.own(g, try half(g, w, n, 2 * n)), .s = try self.own(g, try half(g, sc, n, 2 * n)), .mode = .mxfp8 };
            if (comptime canDrop(@TypeOf(lookup))) {
                var b: [96]u8 = undefined;
                inline for (.{ "w1", "w3" }) |nm| inline for (.{ "weight", "scales" }) |part| {
                    lookup.drop(try std.fmt.bufPrint(&b, "layers.{d}.ffn.shared_experts." ++ nm ++ "." ++ part, .{l}));
                };
            }
        }

        fn canDrop(comptime L: type) bool {
            return switch (@typeInfo(L)) {
                .pointer => |p| !p.attrs.@"const" and @hasDecl(p.child, "drop"),
                else => false,
            };
        }

        fn own(self: *Self, g: *G, x: T) !T {
            const k = g.keep(x);
            try self.owned.append(self.gpa, k);
            return k;
        }

        fn req(lookup: anytype, name: []const u8) !T {
            return lookup.get(name) orelse error.MissingWeight;
        }

        fn reqf(lookup: anytype, buf: []u8, comptime fmt: []const u8, args: anytype) !T {
            return req(lookup, std.fmt.bufPrint(buf, fmt, args) catch return error.NameTooLong);
        }

        fn reqQ(lookup: anytype, buf: []u8, comptime base: []const u8, args: anytype) !graph.Q(T) {
            return .{ .w = try reqf(lookup, buf, base ++ ".weight", args), .s = try reqf(lookup, buf, base ++ ".scales", args), .mode = .mxfp8 };
        }

        /// One trunk layer's residents by checkpoint name.
        pub fn bindLayer(lookup: anytype, li: v41.LayerInfo, l: u32) !graph.LayerW(T) {
            return bindBlock(lookup, "layers", li, l);
        }

        /// A decoder block's residents under `pfx` (`layers`, or `mtp` for the DSpark stages).
        pub fn bindBlock(lookup: anytype, comptime pfx: []const u8, li: v41.LayerInfo, l: u32) !graph.LayerW(T) {
            var b: [192]u8 = undefined;
            var lw: graph.LayerW(T) = .{
                .attn_norm = try reqf(lookup, &b, pfx ++ ".{d}.attn_norm.weight", .{l}),
                .ffn_norm = try reqf(lookup, &b, pfx ++ ".{d}.ffn_norm.weight", .{l}),
                .hc_attn_fn = try reqf(lookup, &b, pfx ++ ".{d}.hc_attn_fn", .{l}),
                .hc_attn_base = try reqf(lookup, &b, pfx ++ ".{d}.hc_attn_base", .{l}),
                .hc_attn_scale = try reqf(lookup, &b, pfx ++ ".{d}.hc_attn_scale", .{l}),
                .hc_ffn_fn = try reqf(lookup, &b, pfx ++ ".{d}.hc_ffn_fn", .{l}),
                .hc_ffn_base = try reqf(lookup, &b, pfx ++ ".{d}.hc_ffn_base", .{l}),
                .hc_ffn_scale = try reqf(lookup, &b, pfx ++ ".{d}.hc_ffn_scale", .{l}),
                .attn_sink = try reqf(lookup, &b, pfx ++ ".{d}.attn.attn_sink", .{l}),
                .q_norm = try reqf(lookup, &b, pfx ++ ".{d}.attn.q_norm.weight", .{l}),
                .kv_norm = try reqf(lookup, &b, pfx ++ ".{d}.attn.kv_norm.weight", .{l}),
                .wq_a = try reqQ(lookup, &b, pfx ++ ".{d}.attn.wq_a", .{l}),
                .wq_b = try reqQ(lookup, &b, pfx ++ ".{d}.attn.wq_b", .{l}),
                .wkv = try reqQ(lookup, &b, pfx ++ ".{d}.attn.wkv", .{l}),
                .wo_a = try reqQ(lookup, &b, pfx ++ ".{d}.attn.wo_a", .{l}),
                .wo_b = try reqQ(lookup, &b, pfx ++ ".{d}.attn.wo_b", .{l}),
                .gate_w = try reqf(lookup, &b, pfx ++ ".{d}.ffn.gate.weight", .{l}),
                .gate_bias = try reqf(lookup, &b, pfx ++ ".{d}.ffn.gate.bias", .{l}),
                .sh_w1 = try reqQ(lookup, &b, pfx ++ ".{d}.ffn.shared_experts.w1", .{l}),
                .sh_w2 = try reqQ(lookup, &b, pfx ++ ".{d}.ffn.shared_experts.w2", .{l}),
                .sh_w3 = try reqQ(lookup, &b, pfx ++ ".{d}.ffn.shared_experts.w3", .{l}),
            };
            if (li.kv_source) {
                lw.comp = .{
                    .wkv = try reqf(lookup, &b, pfx ++ ".{d}.attn.compressor.wkv.weight", .{l}),
                    .wgate = if (li.ratio > 1) try reqf(lookup, &b, pfx ++ ".{d}.attn.compressor.wgate.weight", .{l}) else null,
                    .norm = try reqf(lookup, &b, pfx ++ ".{d}.attn.compressor.norm.weight", .{l}),
                };
                lw.idx_k = .{ .wk = try reqf(lookup, &b, pfx ++ ".{d}.attn.indexer.wk.weight", .{l}), .k_norm = try reqf(lookup, &b, pfx ++ ".{d}.attn.indexer.k_norm.weight", .{l}) };
            }
            if (li.index_source) {
                lw.idx_q = .{ .wq_b = try reqQ(lookup, &b, pfx ++ ".{d}.attn.indexer.wq_b", .{l}), .weights_proj = try reqf(lookup, &b, pfx ++ ".{d}.attn.indexer.weights_proj.weight", .{l}) };
            }
            return lw;
        }

        /// A fresh sequence: lanes per the tier's KV route, its own n-gram history.
        pub fn newState(self: *const Self) !State {
            return self.newStateWith(self.tier.kv);
        }

        /// A fresh sequence whose KV lanes follow `kv`: the tier's route, or a
        /// request's own bounded route (`boundedKv`).
        pub fn newStateWith(self: *const Self, kv: kvc.Geometry) !State {
            const cs = try self.gpa.alloc(Cache, self.c.n_layers);
            errdefer self.gpa.free(cs);
            var max_len: ?u32 = null;
            for (cs, 0..) |*lc, l| {
                lc.* = Cache.init(self.c.layers[l], self.c.window, kv);
                if (lc.admitLimit()) |m| max_len = if (max_len) |x| @min(x, m) else m;
            }
            // The n-gram history of a bounded state is reserved to its admitted length: a step never grows it.
            var hash: ?eng.HashState = null;
            if (self.engram != null) {
                hash = .{};
                if (max_len) |m| try hash.?.hist.ensureTotalCapacity(self.gpa, m);
            }
            errdefer if (hash) |*h| h.deinit(self.gpa);
            return .{ .layers = cs, .hash = hash, .max_len = max_len, .scratch = try self.gpa.alloc(u8, self.scratchBytes(scratch_rows)) };
        }

        /// The bounded route (W107 lanes) for a request of at most `max_positions`
        /// positions (its prompt, its tokens and one verify block): the window a
        /// ring, the compress / index / frontier lanes sized once to it at their
        /// first write and never grown; a forward past it is refused by name
        /// (`BoundedLaneFull`) before any lane is written.
        pub fn boundedKv(self: *const Self, max_positions: u32) kvc.Geometry {
            var kv = self.tier.kv;
            kv.route = .bounded;
            kv.max_kv = max_positions;
            return kv;
        }

        /// Host bytes a forward of `rows` rows allocates (the embed ids or, after
        /// the fence, the embedding rows; the Engram rows, per Engram layer its
        /// ids / codes / scales), each rounded up to the allocator's worst alignment.
        fn scratchBytes(self: *const Self, rows: usize) usize {
            const pad = 16;
            var n: usize = @max(rows * @sizeOf(i32), rows * @as(usize, self.c.hidden_size) * 2) + pad;
            if (self.engram) |en| {
                const cols = en.src.hashing.cols();
                const hd: usize = en.src.bank.head_dim;
                n += rows * en.src.perToken() * @sizeOf(i64) + pad;
                n += self.c.engram.n_layers * (rows * cols * @sizeOf(i64) + rows * cols * hd + rows * cols * (hd / 32) + 3 * pad);
                // A posted gather's ids, records and job (`forwardSpan`'s decode-width posts).
                if (en.posted) n += self.c.engram.n_layers * (rows * cols * @sizeOf(i64) + rows * cols * @as(usize, en.src.bank.record_bytes) + @sizeOf(eng.RowSource.Posted) + 3 * pad);
            }
            return n;
        }

        fn invFor(self: *const Self, li: v41.LayerInfo) T {
            return if (li.ratio > 0) self.inv_yarn else self.inv_swa;
        }

        /// The Engram add of layer slot `slot` for `n` positions (`rows` from the span's hash).
        fn engramLayer(self: *const Self, g: *G, a: std.mem.Allocator, slot: usize, h: T, rows: []const i64, n: usize) !T {
            const en = &self.engram.?;
            const cols = en.src.hashing.cols();
            const hd: usize = en.src.bank.head_dim;
            const ids = try a.alloc(i64, n * cols);
            const codes = try a.alloc(u8, n * cols * hd);
            const scales = try a.alloc(u8, n * cols * (hd / 32));
            try en.src.read(slot, rows, n, ids, codes, scales);
            return self.engramApply(g, slot, h, codes, scales, n);
        }

        /// `engramLayer` over a posted gather's records (the same bytes; `take` waits for them).
        fn engramLayerPosted(self: *const Self, g: *G, a: std.mem.Allocator, slot: usize, h: T, p: *eng.RowSource.Posted, n: usize) !T {
            const en = &self.engram.?;
            const cols = en.src.hashing.cols();
            const hd: usize = en.src.bank.head_dim;
            const codes = try a.alloc(u8, n * cols * hd);
            const scales = try a.alloc(u8, n * cols * (hd / 32));
            try en.src.take(p, codes, scales);
            return self.engramApply(g, slot, h, codes, scales, n);
        }

        fn engramApply(self: *const Self, g: *G, slot: usize, h: T, codes: []u8, scales: []u8, n: usize) !T {
            const en = &self.engram.?;
            const cols = en.src.hashing.cols();
            const hd: usize = en.src.bank.head_dim;
            const nr: c_int = @intCast(n * cols);
            const ca = try g.hostArray(codes, &.{ nr, @intCast(hd / 4) }, .uint32);
            const sa = try g.hostArray(scales, &.{ nr, @intCast(hd / 32) }, .uint8);
            const er = try Tr.engramRows(g, ca, sa, 1, @intCast(n), @intCast(cols));
            return Tr.engramApplyM1(g, &self.c, en.w[slot], if (self.engram_m1[slot]) |*x| x else null, h, er);
        }

        /// A span's posted gathers, drained (their poster done with the memory) and released (`a` = post's).
        fn dropPosts(self: *const Self, a: std.mem.Allocator, posts: []?*eng.RowSource.Posted) void {
            for (posts) |p| if (p) |x| {
                self.engram.?.src.drain(x);
                self.engram.?.src.release(a, x);
            };
        }

        /// A slot's per-chunk posted gathers (`gpa`-owned; null until posted).
        const PostList = []?*eng.RowSource.Posted;

        /// Every chunk's gather of Engram slot `slot`, posted in chunk order into `list` (its entries null).
        fn postSlot(self: *const Self, list: PostList, slot: usize, rows: []const []const i64, spans: []const [2]u32) !void {
            for (spans, rows, list) |sp, r, *p| p.* = try self.engram.?.src.post(self.gpa, slot, r, sp[1] - sp[0]);
        }

        /// Drain and free a slot's posted gathers (every one taken on the way through; an abort's too).
        fn releasePosts(self: *const Self, list: PostList) void {
            for (list) |p| if (p) |x| {
                self.engram.?.src.drain(x);
                self.engram.?.src.release(self.gpa, x);
            };
            self.gpa.free(list);
        }

        /// `mean(h.astype(f32), axis=2).astype(h.dtype)`.
        fn mainOf(g: *G, h: T) !T {
            return g.astype(try g.mean(try g.astype(h, .float32), 2, false), g.dtypeOf(h));
        }

        fn embedSpan(self: *const Self, g: *G, a: std.mem.Allocator, ids: []const u32) !Tr.Out {
            return Tr.expandEmbedding(g, &self.c, try self.embed.of(g, a, ids, self.c.hidden_size));
        }

        /// Device bytes the model builds at construction beyond the checkpoint's
        /// residents (the bill's resident term): W97's dense f32 wo_a per layer, an
        /// mxfp8 head's codes and scales (the Module then drops the dense head:
        /// `droppedBytes`). The rotary tables (< 1 MB) aside.
        pub fn builtBytes(self: *const Self) u64 {
            const c = &self.c;
            var n: u64 = 0;
            if (self.tier.routes.wo_a_f32) n += @as(u64, c.n_layers) * graph.woaDenseBytes(c);
            if (self.tier.routes.head == .mxfp8) n += @as(u64, c.vocab_size) * c.hidden_size * 33 / 32;
            if (self.tier.routes.dense_rc) n += graph.sharedGateUpBytes(c);
            return n;
        }

        /// Checkpoint residents the Module drops once the model is built: the dense bf16 head under HEAD_MODE
        /// mxfp8 (its quantized codes and scales are in `builtBytes`). The bill's resident term less these.
        pub fn droppedBytes(self: *const Self) u64 {
            var n: u64 = if (self.tier.routes.head == .mxfp8) @as(u64, self.c.vocab_size) * self.c.hidden_size * 2 else 0;
            if (self.tier.routes.dense_rc) n += graph.sharedGateUpBytes(&self.c);
            return n;
        }

        /// The input table's bytes (bf16 `[vocab, dim]`): what retiring it frees,
        /// the admission's post-prefill embedding credit.
        pub fn embeddingBytes(self: *const Self) u64 {
            return @as(u64, self.c.vocab_size) * self.c.hidden_size * 2;
        }

        /// The prompt pass's fence (`embedding_install.retire`): later lookups
        /// read `rows` (the table's rows on the host) and the table is handed
        /// back for its owner to free. Once per model; `rows` must be the
        /// table's layout (bf16 `[vocab, dim]`) and outlive the model.
        pub fn retireEmbedding(self: *Self, g: *G, rows: *ngram.NgramTable) !T {
            const w = switch (self.embed) {
                .table => |w| w,
                .rows => return error.EmbeddingRetired,
            };
            const s = g.shapeOf(w);
            if (g.dtypeOf(w) != .bfloat16 or !s.eql(ops.Shape.of(&.{ @intCast(self.c.vocab_size), @intCast(self.c.hidden_size) })) or
                rows.bits != 16 or rows.rows != self.c.vocab_size or rows.dim != self.c.hidden_size)
                return error.EmbeddingRowsMismatch;
            self.embed = .{ .rows = rows };
            return w;
        }

        fn engramRowsFor(self: *const Self, st: *State, a: std.mem.Allocator, ids: []const u32) ![]const i64 {
            const en = &(self.engram orelse return &.{});
            const rows = try a.alloc(i64, ids.len * en.src.perToken());
            try en.src.advance(self.gpa, &st.hash.?, ids, rows);
            return rows;
        }

        /// `_forward_span`: every layer over one span; returns the final-normed hidden.
        fn forwardSpan(self: *const Self, g: *G, a: std.mem.Allocator, st: *State, ids: []const u32, want_main: bool, routed: anytype, probe: anytype, main_out: *?T) !T {
            const c = &self.c;
            const rt = &self.tier.routes;
            const n: u32 = @intCast(ids.len);
            const positions = try g.arange(@floatFromInt(st.offset), @floatFromInt(st.offset + n), 1, .int32);
            const e = try self.embedSpan(g, a, ids);
            const rows = try self.engramRowsFor(st, a, ids);
            // ENGRAM=prefetch at decode width: every Engram slot's gather posted before the first layer (each
            // slot's table runs its own; per table the order is the blocking path's), each taken at its layer.
            var posts: [eng.max_layers]?*eng.RowSource.Posted = @splat(null);
            defer self.dropPosts(a, &posts);
            if (self.engram) |en| if (en.posted and n <= scratch_rows) {
                for (posts[0..en.src.hashing.n_layers], 0..) |*p, sl| p.* = try en.src.post(a, sl, rows, n);
            };
            var h = e.h;
            var pm = e.pre_mix;
            var shared: Tr.Share = .{};
            var mains: [8]T = undefined;
            var n_main: usize = 0;
            // One wave per layer: what a layer builds is freed at its end, what
            // the next layers read is carried (`Tr.Carry`).
            var carry: Tr.Carry = .{};
            errdefer carry.release(g);
            errdefer for (mains[0..n_main]) |x| g.release(x);
            for (self.layers, 0..) |*lw, l| {
                const li = c.layers[l];
                const wave = g.mark();
                if (li.engram_slot) |slot| h = if (posts[slot]) |p| try self.engramLayerPosted(g, a, slot, h, p, n) else try self.engramLayer(g, a, slot, h, rows, n);
                if (want_main and li.dspark_target) {
                    mains[n_main] = g.keep(try mainOf(g, h));
                    n_main += 1;
                }
                const out = try Tr.layer(g, probe, c, rt, self.kx.at(l), li, lw, self.invFor(li), h, pm, positions, &st.layers[l], &shared, routed.at(@intCast(l)));
                h = out.h;
                pm = out.pre_mix;
                carry.persist(g, &h, &pm, &shared);
                g.resetTo(wave);
            }
            for (st.layers) |*lc| lc.advance(n);
            st.offset += n;
            main_out.* = if (n_main > 0) try g.concat(mains[0..n_main], -1) else null;
            const fin = try Tr.finalNorm(g, c, h, pm, self.norm_w);
            // The tail's graph holds what it reads.
            for (mains[0..n_main]) |x| g.release(x);
            carry.release(g);
            return fin;
        }

        /// The arrays a span fence settles: every lane's backing plus `extra`.
        pub fn fence(g: *G, st: *State, extra: []const T) !void {
            var list: [512]T = undefined;
            var k: usize = 0;
            for (extra) |x| {
                list[k] = x;
                k += 1;
            }
            for (st.layers) |*lc| {
                if (try lc.window.view(g)) |x| {
                    list[k] = x;
                    k += 1;
                }
                if (try lc.compress.view(g)) |x| {
                    list[k] = x;
                    k += 1;
                }
                if (try lc.index.view(g)) |x| {
                    list[k] = x;
                    k += 1;
                }
            }
            try g.evalAll(list[0..k]);
        }

        /// The target forward over `ids` from the state's offset: one shot, or
        /// chunked by `_resolve_prefill_chunk` (K16 layer-major when the tier
        /// asks). The result's arrays live until the backend's next reset.
        pub fn forward(self: *Self, g: *G, st: *State, ids: []const u32, want: Want, routed: anytype, probe: anytype) !Result {
            const n: u32 = @intCast(ids.len);
            if (st.max_len) |m| if (st.offset + n > m) return error.BoundedLaneFull;
            // A decode / verify forward runs on the state's scratch; a prefill span on an arena.
            var fba: std.heap.FixedBufferAllocator = .init(st.scratch);
            // A prompt's host transients (the embedding rows gathered from the host table, 16,384 x 5,120 x 2 B at 16K, the
            // routing and Engram tables) come from pages mapped for this forward and unmapped at its end: through the host's
            // libc malloc each freed large block stayed dirty in the footprint in libmalloc's large cache (pass3el:
            // MALLOC_LARGE +0.53 GB after the first prompt, held for the process's life; malloc_zone_pressure_relief returns
            // none of it).
            var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
            defer arena.deinit();
            const a = if (n <= scratch_rows) fba.allocator() else arena.allocator();
            const chunk = kvc.resolvePrefillChunk(&self.c, n, st.span_chunk orelse self.tier.prefill_chunk, self.tier.chunk_target_bytes);
            var hidden: T = undefined;
            var main: ?T = null;
            if (chunk <= 0 or chunk >= n) {
                hidden = try self.forwardSpan(g, a, st, ids, want.main_hidden, routed, probe, &main);
            } else if (self.tier.layer_major) {
                const r = try self.forwardLayerMajor(g, a, st, ids, try kvc.prefillSpans(a, n, chunk), want.main_hidden, routed, probe);
                hidden = r.hidden;
                main = r.main;
            } else {
                // Chunk-major: each span through every layer, settled before the next.
                const spans = try kvc.prefillSpans(a, n, chunk);
                const outs = try a.alloc(T, spans.len);
                const mains = try a.alloc(?T, spans.len);
                for (spans, outs, mains) |sp, *o, *m| {
                    var mh: ?T = null;
                    const h = try self.forwardSpan(g, a, st, ids[sp[0]..sp[1]], want.main_hidden, routed, probe, &mh);
                    try fence(g, st, if (mh) |x| &.{ h, x } else &.{h});
                    o.* = g.keep(h);
                    m.* = if (mh) |x| g.keep(x) else null;
                    g.reset();
                }
                hidden = try g.concat(outs, 1);
                if (want.main_hidden) {
                    const ms = try a.alloc(T, spans.len);
                    for (mains, ms) |m, *x| x.* = m.?;
                    main = try g.concat(ms, 1);
                }
                for (outs) |x| g.release(x);
                for (mains) |m| if (m) |x| g.release(x);
            }
            var res: Result = .{ .hidden = hidden, .main_hidden = main };
            switch (want.logits) {
                .none => {},
                .all => res.logits = try self.headOf(g, hidden),
                .last => {
                    const s = g.shapeOf(hidden);
                    const last = try g.slice(hidden, &.{ 0, s.d[1] - 1, 0 }, s.slice(), &.{ 1, 1, 1 });
                    res.logits = try self.headOf(g, last);
                },
            }
            return res;
        }

        /// The head under its route: C11's m1rows (bf16) or RCPROJ (mxfp8) at <= 8 rows when bound (a phase route), else
        /// `Tr.head`.
        fn headOf(self: *const Self, g: *G, x: T) !T {
            if (self.head_rows) |*hr| {
                const s = g.shapeOf(x);
                if (s.d[0] * s.d[1] <= graph.rc_max_rows) return Tr.headRows(g, hr, x);
            }
            if (self.head_mx) |*hm| {
                const s = g.shapeOf(x);
                if (s.d[0] * s.d[1] <= graph.rc_max_rows) return Tr.headMx(g, hm, x);
            }
            return Tr.head(g, &self.tier.routes, x, self.head);
        }

        /// Greedy AR (the M3 token-parity run): the prompt in forwards of at
        /// most `chunk` rows, then one token per forward; `out[0]` is the
        /// prompt's pick. `ex` is the routed-experts executor (`at`, `flush`),
        /// flushed after each forward's eval; `observer` (or `{}`) gets each
        /// evaluated logits row (`step(g, logits)`).
        pub fn greedy(self: *Self, g: *G, st: *State, prompt: []const u32, chunk: u32, ex: anytype, out: []u32, observer: anytype) !void {
            std.debug.assert(prompt.len > 0 and chunk > 0 and out.len > 0);
            var i: usize = 0;
            var next: u32 = 0;
            while (i < prompt.len) {
                const end = @min(i + chunk, prompt.len);
                const last = end == prompt.len;
                const r = try self.forward(g, st, prompt[i..end], .{ .logits = if (last) .last else .none }, ex, graph.NoProbe{});
                try fence(g, st, &.{if (last) r.logits.? else r.hidden});
                try ex.flush();
                if (last) {
                    if (@TypeOf(observer) != void) try observer.step(g, r.logits.?);
                    next = try g.hostArgmax(r.logits.?);
                }
                g.reset();
                i = end;
            }
            for (out, 0..) |*o, t| {
                o.* = next;
                if (t + 1 == out.len) break;
                const r = try self.forward(g, st, &.{next}, .{ .logits = .last }, ex, graph.NoProbe{});
                try fence(g, st, &.{r.logits.?});
                try ex.flush();
                if (@TypeOf(observer) != void) try observer.step(g, r.logits.?);
                next = try g.hostArgmax(r.logits.?);
                g.reset();
            }
        }

        /// P1's predictor chunks per GPU round trip: their transients live together (about 25 MB a 953-row chunk, 100 MB
        /// a batch), before the layer's wave opens, so under the wave's own bound.
        pub const predict_batch = 4;

        /// P1: layer `l`'s predictor pass before its attention: per chunk `Tr.predictIds` and one host read; the counts'
        /// ranking (`expert_policy.rankHottest`, the seed's order) to the hook, which reads that seed ahead. Profile
        /// stage "moe.predict" ("moe.predict.in" before it: the previous layer's carried-over work).
        fn predictSeed(self: *const Self, g: *G, l: usize, lw: *const Tr.W, hook: anytype, hs: []const T, pms: []const T, probe: anytype) !void {
            const c = &self.c;
            const gpa = self.gpa;
            try probe.put("moe.predict.in", hs[0]);
            // Host scratch off the forward's allocator (a short forward's is the state's fixed scratch).
            const counts = try gpa.alloc(u32, c.n_routed_experts);
            defer gpa.free(counts);
            @memset(counts, 0);
            const ranked = try gpa.alloc(u16, c.n_routed_experts);
            defer gpa.free(ranked);
            var rows: usize = 0;
            for (hs) |h| rows = @max(rows, @as(usize, @intCast(g.shapeOf(h).dim(1))));
            const ids = try gpa.alloc(u16, rows * c.n_experts_per_tok);
            defer gpa.free(ids);
            const wave = g.mark();
            defer g.resetTo(wave);
            const wf = if (self.tier.routes.predict_bf16) lw.gate_w else try g.astype(lw.gate_w, .float32);
            // `predict_batch` chunks' predictions evaluated together, one GPU round trip for all of them; their ids are
            // then read in place (an evaluated array's host read makes no round trip).
            var start: usize = 0;
            while (start < hs.len) : (start += predict_batch) {
                const end = @min(start + predict_batch, hs.len);
                const batch = g.mark();
                defer g.resetTo(batch);
                var idxs: [predict_batch]T = undefined;
                for (hs[start..end], pms[start..end], idxs[0 .. end - start]) |h, pm, *idx| idx.* = try Tr.predictIds(g, c, self.kx.at(l), lw, h, pm, wf);
                try g.evalAll(idxs[0 .. end - start]);
                for (idxs[0 .. end - start]) |idx| {
                    const n: usize = @intCast(g.shapeOf(idx).numel());
                    for (try g.hostIds(idx, ids[0..n])) |e| counts[e] += 1;
                }
            }
            prof.recordPrediction(l, counts);
            try hook.readAheadSeed(expert_policy.rankHottest(counts, ranked));
            try probe.put("moe.predict", wf);
        }

        /// K16 `_forward_layer_major`: every layer over all chunks before the next;
        /// the gate and shared expert per chunk, the routed call batched across
        /// chunks (row-capped), the ffn combine the compiled `_PREFILL_HC_POST`.
        /// Each chunk's published lanes as views of the layer's final lanes, up to the rows the lanes held after that
        /// chunk's publish (`publishCompressed` read `view` there: the same rows, never rewritten after). One lane
        /// version per kv source, written in place chunk by chunk (nothing else holds the version a write replaces).
        fn laneViewsAtLayerEnd(g: *G, lc: *Cache, lane_rows: []const [2]u32, shareds: []Tr.Share) !void {
            const cv = try lc.compress.view(g);
            const iv = try lc.index.view(g);
            for (shareds, lane_rows) |*sh, n| {
                sh.compress_kv = if (cv) |v| try lanePrefix(g, v, n[0]) else null;
                sh.index_k = if (iv) |v| try lanePrefix(g, v, n[1]) else null;
            }
        }

        /// Rows [0, n) of a lane view (null for none, the view itself for all of it).
        fn lanePrefix(g: *G, v: T, n: u32) !?T {
            const s = g.shapeOf(v);
            if (n == 0) return null;
            if (n == s.dim(1)) return v;
            var start: [8]c_int = @splat(0);
            var stop: [8]c_int = undefined;
            const strides: [8]c_int = @splat(1);
            @memcpy(stop[0..s.n], s.slice());
            stop[1] = @intCast(n);
            return try g.slice(v, start[0..s.n], stop[0..s.n], strides[0..s.n]);
        }

        fn forwardLayerMajor(self: *const Self, g: *G, a: std.mem.Allocator, st: *State, ids: []const u32, spans: []const [2]u32, want_main: bool, routed: anytype, probe: anytype) !struct { hidden: T, main: ?T } {
            const c = &self.c;
            const rt = &self.tier.routes;
            const nc = spans.len;
            const offset0 = st.offset;
            const hs = try a.alloc(T, nc);
            const pms = try a.alloc(T, nc);
            const poss = try a.alloc(T, nc);
            const rows = try a.alloc([]const i64, nc);
            const shareds = try a.alloc(Tr.Share, nc);
            const mains = try a.alloc([8]T, nc);
            var n_main: usize = 0;
            // The Engram gathers ahead of the layers that read them: the first Engram slot's chunk by chunk as
            // the embedding hashes it, each later slot's once the slot before it is taken (one slot held).
            const posting = if (self.engram) |en| en.posted else false;
            var posts: [eng.max_layers]?PostList = @splat(null);
            defer for (&posts) |*ps| if (ps.*) |list| {
                self.releasePosts(list);
                ps.* = null;
            };
            if (posting) {
                posts[0] = try self.gpa.alloc(?*eng.RowSource.Posted, nc);
                @memset(posts[0].?, null);
            }
            // The embedding's own wave: only each chunk's (kept) h, pre_mix and positions survive it.
            const embed_wave = g.mark();
            for (spans, 0..) |sp, i| {
                const e = try self.embedSpan(g, a, ids[sp[0]..sp[1]]);
                hs[i] = g.keep(e.h);
                pms[i] = g.keep(e.pre_mix);
                poss[i] = g.keep(try g.arange(@floatFromInt(offset0 + sp[0]), @floatFromInt(offset0 + sp[1]), 1, .int32));
                rows[i] = try self.engramRowsFor(st, a, ids[sp[0]..sp[1]]);
                if (posting) posts[0].?[i] = try self.engram.?.src.post(self.gpa, 0, rows[i], sp[1] - sp[0]);
                shareds[i] = .{};
            }
            try g.evalAll(hs);
            g.resetTo(embed_wave);
            try probe.put("embed", hs[0]);
            // Each chunk's shared runtime outlives the per-layer reset (`Tr.Carry`).
            const carries = try a.alloc(Tr.Carry, nc);
            @memset(carries, .{});
            defer for (carries) |*k| k.release(g);
            const cap = moeRowCap(c, self.tier.chunk_target_bytes);
            const halves = try a.alloc(Tr.Half, nc);
            const xfs = try a.alloc(T, nc);
            const routes_ = try a.alloc(Tr.Route, nc);
            const dim: c_int = @intCast(c.hidden_size);
            // A kv source's lane rows after each chunk's publish (`laneViewsAtLayerEnd`).
            const lane_rows = try a.alloc([2]u32, nc);
            // P1 (the routed hook's construction option, read once): each layer's predicted seed read ahead during its attention.
            const has_ahead = comptime @hasDecl(@TypeOf(routed.at(0)), "readAheadSeed");
            const read_ahead = if (comptime has_ahead) routed.at(0).readAhead() else false;
            for (self.layers, 0..) |*lw, l| {
                const li = c.layers[l];
                const lc = &st.layers[l];
                if (comptime has_ahead) {
                    // The predictor's stages charge the layer's first chunk (not the previous layer's last).
                    probeChunk(probe, 0);
                    if (read_ahead) try self.predictSeed(g, l, lw, routed.at(@intCast(l)), hs, pms, probe);
                }
                // One wave per layer (freed at its end; hs, pms and the chunks' shared runtime carried).
                const layer_wave = g.mark();
                for (spans, 0..) |sp, i| {
                    // One sub-wave per chunk: its attention side is freed before the next chunk's, only
                    // its Half and its shared runtime kept to the layer's routed call (as chunk-major
                    // keeps one chunk's layer at a time), so a layer never holds every chunk's arrays.
                    probeChunk(probe, i);
                    probeFence(probe, .build);
                    const wave = g.mark();
                    var h = hs[i];
                    if (li.engram_slot) |slot| {
                        h = if (posting)
                            try self.engramLayerPosted(g, a, slot, h, (posts[slot] orelse return error.EngramPostMissing)[i] orelse return error.EngramPostMissing, sp[1] - sp[0])
                        else
                            try self.engramLayer(g, a, slot, h, rows[i], sp[1] - sp[0]);
                        // The profile's own stage for the Engram read and add (else it lands in attn.pre).
                        try probe.put("engram.add", h);
                    }
                    const tap: ?T = if (want_main and li.dspark_target) g.keep(try mainOf(g, h)) else null;
                    if (tap) |t| mains[i][n_main] = t;
                    halves[i] = try Tr.attnAndMoeInput(g, probe, c, rt, self.kx.at(l), li, lw, self.invFor(li), h, pms[i], poss[i], lc, &shareds[i]);
                    // Every kept array evaluated before the reset: a lazy one would hold its whole graph. The main
                    // tap too (`main_taps_in_chunk_fence`): its mean would hold the layer's input stream to the end.
                    const hf = halves[i];
                    var settle: [6]T = .{ hf.moe_in, hf.ffn_pre, hf.h1, hf.post, hf.comb, undefined };
                    var n_settle: usize = 5;
                    if (tap) |t| {
                        settle[5] = t;
                        n_settle = 6;
                    }
                    probeFence(probe, .wait);
                    try fence(g, st, settle[0..n_settle]);
                    probeFence(probe, .done);
                    // The profile's split of the chunk's carry-over: the fence's evaluation, then the keeps and
                    // the wave's frees (each probe re-reads an evaluated kept array: its segment is the host work).
                    try probe.put("chunk.fence", hf.moe_in);
                    keepHalf(g, &halves[i]);
                    // A kv source's published lanes are not carried per chunk: a carried view would pin this chunk's
                    // version of each lane, so the next chunk's write could not donate it and every chunk would copy
                    // and keep a whole lane (pass3ds: 137 lane versions, 27.8 GB at 128K). The chunk's rows are
                    // recorded and its view of the final lanes carried at the layer's end (`laneViewsAtLayerEnd`).
                    if (li.kv_source) {
                        lane_rows[i] = .{ lc.compress.rows(), lc.index.rows() };
                        shareds[i].compress_kv = null;
                        shareds[i].index_k = null;
                    }
                    carries[i].persistShared(g, &shareds[i]);
                    // Nothing reads the layer's input stream past the fence (the Half carries the residual, the tap is
                    // settled): on its route it goes here, not at the chunk's HC post.
                    if (rt.input_stream_early_release) g.release(hs[i]);
                    g.resetTo(wave);
                    try probe.put("chunk.frees", halves[i].moe_in);
                }
                if (want_main and li.dspark_target) n_main += 1;
                // Per chunk: the resident gate (M == the chunk, as chunk-major).
                for (halves, xfs, routes_, 0..) |hf, *xf, *r, ri| {
                    probeChunk(probe, ri);
                    xf.* = try g.reshape(hf.moe_in, &.{ -1, dim });
                    r.* = try Tr.router(g, probe, c, rt, self.kx.at(l), lw, xf.*);
                }
                // The routed call over consecutive chunks up to the row cap.
                var i: usize = 0;
                while (i < nc) {
                    var j = i;
                    var n_rows: u64 = 0;
                    while (j < nc) : (j += 1) {
                        const r_: u64 = @intCast(g.shapeOf(xfs[j]).dim(0));
                        if (j > i and n_rows + r_ > cap) break;
                        n_rows += r_;
                    }
                    // One sub-wave per routed group: its routed outputs, combines and HC posts are freed
                    // once the group's new hidden states are evaluated and kept.
                    probeChunk(probe, i);
                    const group_wave = g.mark();
                    const cat_xf = if (j - i == 1) xfs[i] else try g.concat(xfs[i..j], 0);
                    const idxs = try a.alloc(T, j - i);
                    for (routes_[i..j], idxs) |r, *d| d.* = r.indices;
                    const cat_idx = if (j - i == 1) routes_[i].indices else try g.concat(idxs, 0);
                    // PREFILL_HOST shared: the routing barrier (the ids evaluated, as the routed call's
                    // host read needs them), then each chunk's shared expert started on the GPU so it
                    // runs while the host plans the routed waves (it depends on the MoE input only).
                    const pre_shared = try a.alloc(?T, j - i);
                    @memset(pre_shared, null);
                    if (rt.prefill_host_shared) {
                        try g.evalAll(&.{cat_idx});
                        for (i..j, pre_shared) |k, *ps| ps.* = try g.astype(try Tr.sharedExpertPrompt(g, c, rt, lw, xfs[k]), .float32);
                        const started = try a.alloc(T, j - i);
                        for (pre_shared, started) |ps, *st_| st_.* = ps.?;
                        try g.asyncEval(started);
                        // The profile's own stage for the group's shared experts (the last one's eval waits for all,
                        // one queue), else their GPU time lands in the routed call's first drain (base_seed).
                        try probe.put("moe.shared", started[started.len - 1]);
                    }
                    const hook = routed.at(@intCast(l));
                    const lk = self.kx.at(l);
                    const top: c_int = @intCast(c.n_experts_per_tok);
                    // JOINLESS on a wide call (a routed source that hands out its unjoined outputs):
                    // each chunk's combine reads the routed rows in place (its loc rows).
                    const has_parts = comptime @hasDecl(@TypeOf(hook), "routedParts");
                    var parts: ?struct { outs: []const T, loc: T } = null;
                    var ro: T = undefined;
                    if (comptime has_parts) {
                        if (lk.joinless != null and n_rows * c.n_experts_per_tok > joinless_min_ids) {
                            const pt = try hook.routedParts(g, cat_xf, cat_idx);
                            parts = .{ .outs = pt.outs, .loc = pt.loc };
                        }
                    }
                    if (parts) |pt| {
                        try probe.put("moe.routed", pt.loc);
                        // The profile's own stage for the wide call's merged sources, evaluated before the first
                        // combine reads them (else the merge lands in moe.y).
                        try probeMerge(probe, pt.outs);
                    } else {
                        ro = try hook.routed(g, cat_xf, cat_idx);
                        // The profile's own stage for the group's routed compute (else it lands in moe.shared).
                        try probe.put("moe.routed", ro);
                    }
                    // The profile's grouped mode: the group's halves at its start (their handles go at the HC post builds
                    // below, h1 / post / comb with each post's evaluation).
                    probeGroupHalves(probe, halves[i..j], i);
                    if (parts != null) probeGroupMerge(probe, hook);
                    // Each chunk's rows and MoE output shape and dtype, read before the inputs may be released.
                    const nks = try a.alloc(c_int, j - i);
                    const mo_shape = try a.alloc(ops.Shape, j - i);
                    const mo_dt = try a.alloc(ops.Dtype, j - i);
                    for (i..j, nks, mo_shape, mo_dt) |k, *nk, *ms, *md| {
                        nk.* = g.shapeOf(xfs[k]).dim(0);
                        ms.* = g.shapeOf(halves[k].moe_in);
                        md.* = g.dtypeOf(halves[k].moe_in);
                    }
                    // PREFILL_INPUT_RELEASE: the group's MoE inputs (each chunk's moe_in, its row view and their concat)
                    // are dead once the wide call's waves drained and the shared experts were issued from them: freed
                    // here, before the group's final evaluation.
                    const released = rt.prefill_input_release and parts != null;
                    if (released) {
                        if (j - i > 1) g.drop(cat_xf);
                        for (i..j) |k| {
                            g.drop(xfs[k]);
                            halves[k].moe_in = g.dropKept(halves[k].moe_in);
                        }
                    }
                    var pos: c_int = 0;
                    for (i..j) |k| {
                        probeChunk(probe, k);
                        const nk = nks[k - i];
                        const y = if (parts) |pt| blk: {
                            const loc = if (j - i == 1) pt.loc else try g.slice(pt.loc, &.{ pos, 0, 0 }, &.{ pos + nk, top, 2 }, &.{ 1, 1, 1 });
                            const shared = pre_shared[k - i] orelse try g.astype(try Tr.sharedExpertPrompt(g, c, rt, lw, xfs[k]), .float32);
                            break :blk try lk.joinless.?.call(g, pt.outs, loc, try g.astype(routes_[k].weights, .float32), shared);
                        } else blk: {
                            const rs = g.shapeOf(ro);
                            const part = if (j - i == 1) ro else try g.slice(ro, &.{ pos, 0, 0 }, &.{ pos + nk, rs.d[1], rs.d[2] }, &.{ 1, 1, 1 });
                            break :blk try Tr.combineRouted(g, probe, c, rt, lk, lw, part, routes_[k].weights, xfs[k], pre_shared[k - i]);
                        };
                        pos += nk;
                        const mo = try g.reshape(try g.astype(y, mo_dt[k - i]), mo_shape[k - i].slice());
                        try probe.put("moe.y", mo);
                        const next = try Tr.prefillHcPost(g, c, self.kx.at(l), mo, halves[k]);
                        try probe.put("out.h", next);
                        if (!rt.input_stream_early_release) g.release(hs[k]);
                        hs[k] = g.keep(next);
                        g.release(pms[k]);
                        pms[k] = g.keep(halves[k].ffn_pre);
                        releaseHalf(g, &halves[k]);
                    }
                    // The profile's mark before the group's final evaluation (the merge, the combines, the HC posts and the new
                    // streams at once, as the timed pass runs it), then that evaluation's own stage.
                    probeGroupEval(probe, if (parts) |pt| pt.outs else &.{}, if (parts) |pt| pt.loc else null, pre_shared, hs[i..j], if (released) null else cat_xf);
                    try g.evalAll(hs[i..j]);
                    probeChunk(probe, i);
                    try probe.put("group.eval", hs[i]);
                    probeChunk(probe, j - 1);
                    if (comptime has_parts) {
                        if (parts != null) hook.releaseParts(g);
                    }
                    g.resetTo(group_wave);
                    try probe.put("group.frees", hs[i]);
                    i = j;
                }
                try g.evalAll(hs);
                if (li.kv_source) try laneViewsAtLayerEnd(g, lc, lane_rows, shareds);
                for (shareds, carries) |*sh, *k| k.persistShared(g, sh);
                g.resetTo(layer_wave);
                // This layer's gathers are taken and its waves evaluated: free them, post the next slot's.
                if (posting) if (li.engram_slot) |slot| {
                    self.releasePosts(posts[slot].?);
                    posts[slot] = null;
                    if (slot + 1 < c.engram.n_layers) {
                        posts[slot + 1] = try self.gpa.alloc(?*eng.RowSource.Posted, nc);
                        @memset(posts[slot + 1].?, null);
                        try self.postSlot(posts[slot + 1].?, slot + 1, rows, spans);
                    }
                };
                try probe.put("layer.end", hs[0]);
            }
            for (st.layers) |*lc| lc.advance(@intCast(ids.len));
            st.offset += @intCast(ids.len);
            const outs = try a.alloc(T, nc);
            for (outs, 0..) |*o, i| o.* = try Tr.finalNorm(g, c, hs[i], pms[i], self.norm_w);
            const hidden = try g.concat(outs, 1);
            var main: ?T = null;
            if (n_main > 0) {
                const parts = try a.alloc(T, nc);
                for (parts, 0..) |*p, i| p.* = try g.concat(mains[i][0..n_main], -1);
                main = try g.concat(parts, 1);
            }
            for (0..nc) |i| {
                g.release(hs[i]);
                g.release(pms[i]);
                g.release(poss[i]);
                for (mains[i][0..n_main]) |x| g.release(x);
            }
            return .{ .hidden = hidden, .main = main };
        }

        /// K16's per-chunk sub-wave keeps a chunk's Half past its reset (released after its HC post).
        /// A probe that attributes by chunk (the profile's) learns which K16 chunk the next stages belong to;
        /// compiled out for every other probe (NoProbe in timed builds).
        /// A probe that marks K16's chunk timeline (the fence profile) gets each mark; compiled out for every other
        /// probe (NoProbe in timed builds, the stage profile).
        fn probeFence(probe: anytype, comptime at: FenceAt) void {
            const P = @TypeOf(probe);
            if (comptime @typeInfo(P) == .pointer and @hasDecl(@typeInfo(P).pointer.child, "fenceMark")) probe.fenceMark(at);
        }

        fn probeChunk(probe: anytype, i: usize) void {
            const P = @TypeOf(probe);
            if (comptime @typeInfo(P) == .pointer and @hasDecl(@typeInfo(P).pointer.child, "atChunk")) probe.atChunk(i);
        }

        /// A probe that splits the wide call's merge (the profile's) evaluates the merged sources on their own stage,
        /// MLX's active and cache read around them; compiled out for every other probe (NoProbe in timed builds).
        fn probeMerge(probe: anytype, outs: []const T) !void {
            const P = @TypeOf(probe);
            if (comptime @typeInfo(P) == .pointer and @hasDecl(@typeInfo(P).pointer.child, "merge")) try probe.merge(outs);
        }

        /// A probe that measures the routed group's final evaluation whole (the profile's grouped mode) reads the group's
        /// halves at its start and, right before `evalAll(hs[i..j])`, MLX's active mark and the geometry the evaluation
        /// reads and writes; compiled out for every other probe (NoProbe in timed builds).
        fn probeGroupHalves(probe: anytype, halves: []const Tr.Half, first_chunk: usize) void {
            const P = @TypeOf(probe);
            if (comptime @typeInfo(P) == .pointer and @hasDecl(@typeInfo(P).pointer.child, "groupHalves")) probe.groupHalves(halves, first_chunk);
        }

        fn probeGroupEval(probe: anytype, outs: []const T, loc: ?T, shared: []const ?T, next: []const T, cat_xf: ?T) void {
            const P = @TypeOf(probe);
            if (comptime @typeInfo(P) == .pointer and @hasDecl(@typeInfo(P).pointer.child, "groupEval")) probe.groupEval(outs, loc, shared, next, cat_xf);
        }

        /// The group's JOINLESS merge (the hook's record, profile builds) for a probe that reads it.
        fn probeGroupMerge(probe: anytype, hook: anytype) void {
            const P = @TypeOf(probe);
            if (comptime @typeInfo(P) == .pointer and @hasDecl(@typeInfo(P).pointer.child, "groupMerge") and @hasDecl(@TypeOf(hook), "lastMerge")) probe.groupMerge(hook.lastMerge());
        }

        fn keepHalf(g: *G, hf: *Tr.Half) void {
            inline for (@typeInfo(Tr.Half).@"struct".field_names) |name| @field(hf, name) = g.keep(@field(hf, name));
        }

        fn releaseHalf(g: *G, hf: *Tr.Half) void {
            inline for (@typeInfo(Tr.Half).@"struct".field_names) |name| g.release(@field(hf, name));
        }

        /// Drop the last `n` tokens from every lane and the n-gram history, all or
        /// nothing (a ring that cannot recover that far refuses before any change).
        pub fn trim(self: *const Self, g: *G, st: *State, n: u32) !void {
            _ = self;
            if (n == 0) return;
            for (st.layers) |*lc| if (!lc.canTrim(n)) return error.TrimTooDeep;
            for (st.layers) |*lc| _ = try lc.trim(g, n);
            if (st.hash) |*h| h.trim(n);
            st.offset -= n;
        }

        pub fn mark(self: *const Self, a: std.mem.Allocator, st: *const State) !Mark {
            _ = self;
            const ms = try a.alloc(Cache.Mark, st.layers.len);
            for (ms, st.layers) |*m, *lc| m.* = lc.mark();
            return .{ .offset = st.offset, .layers = ms };
        }

        /// The request's state at a prompt's end (`Cache.Boundary` per layer, the n-gram history's length), to continue
        /// a later prompt that starts with this one. Its copies are evaluated here (no later write reaches them).
        pub const Boundary = struct {
            offset: u32,
            layers: []Cache.Boundary,
            hash_len: usize,

            pub fn deinit(self: *Boundary, g: *G, a: std.mem.Allocator) void {
                for (self.layers) |*l| l.deinit(g);
                a.free(self.layers);
                self.layers = &.{};
            }
        };

        pub fn boundary(self: *const Self, g: *G, a: std.mem.Allocator, st: *const State) !Boundary {
            _ = self;
            const ls = try a.alloc(Cache.Boundary, st.layers.len);
            var n: usize = 0;
            errdefer {
                for (ls[0..n]) |*l| l.deinit(g);
                a.free(ls);
            }
            for (st.layers) |*lc| {
                ls[n] = try lc.boundary(g);
                n += 1;
            }
            var arrs: std.ArrayList(T) = .empty;
            defer arrs.deinit(a);
            for (ls) |*l| {
                var buf: [3]T = undefined;
                try arrs.appendSlice(a, buf[0..l.arrays(&buf)]);
            }
            if (arrs.items.len > 0) try g.evalAll(arrs.items);
            return .{ .offset = st.offset, .layers = ls, .hash_len = if (st.hash) |h| h.hist.items.len else 0 };
        }

        /// Back to `b` (spent, its layers freed): every lane and the n-gram history as at the boundary.
        pub fn restoreBoundary(self: *const Self, g: *G, a: std.mem.Allocator, st: *State, b: *Boundary) !void {
            _ = self;
            if (b.offset > st.offset or b.layers.len != st.layers.len) return error.BoundaryAhead;
            for (st.layers, b.layers) |*lc, lb| try lc.restoreBoundary(g, lb);
            a.free(b.layers);
            b.layers = &.{};
            if (st.hash) |*h| h.trim(@intCast(h.hist.items.len - b.hash_len));
            st.offset = b.offset;
        }

        pub fn rollback(self: *const Self, g: *G, st: *State, m: Mark) !void {
            _ = self;
            if (m.offset > st.offset) return error.TrimTooDeep;
            for (st.layers) |*lc| if (m.offset != lc.offset and !lc.canTrim(lc.offset - m.offset)) return error.TrimTooDeep;
            for (st.layers, m.layers) |*lc, lm| try lc.rollback(g, lm);
            if (st.hash) |*h| h.trim(st.offset - m.offset);
            st.offset = m.offset;
        }
    };
}

// ── tests: the whole mini model through the trace backend ──

const testing = std.testing;
const TraceOps = ops.TraceOps;
const TM = Model(TraceOps);

/// Resident names -> trace inputs of the spec's dtype and shape (test helper).
pub const SpecLookup = struct {
    g: *TraceOps,
    spec: []const v41.Param,

    fn find(self: *const SpecLookup, name: []const u8) ?struct { p: v41.Param, scales: bool } {
        for (self.spec) |p| switch (p.kind) {
            .dense => if (std.mem.eql(u8, p.name, name)) return .{ .p = p, .scales = false },
            .quant => {
                if (std.mem.startsWith(u8, name, p.name) and name.len > p.name.len and name[p.name.len] == '.') {
                    const rest = name[p.name.len + 1 ..];
                    if (std.mem.eql(u8, rest, "weight")) return .{ .p = p, .scales = false };
                    if (std.mem.eql(u8, rest, "scales")) return .{ .p = p, .scales = true };
                }
            },
        };
        return null;
    }

    pub fn get(self: *const SpecLookup, name: []const u8) ?u32 {
        const f = self.find(name) orelse return null;
        return switch (f.p.kind) {
            .dense => |d| blk: {
                var sh: [2]c_int = undefined;
                for (0..d.rank) |i| sh[i] = @intCast(d.shape[i]);
                break :blk self.g.input(sh[0..d.rank], stToDtype(d.dtype)) catch null;
            },
            .quant => |q| if (f.scales)
                self.g.input(&.{ @intCast(q.out), @intCast(q.in / 32) }, .uint8) catch null
            else
                self.g.input(&.{ @intCast(q.out), @intCast(q.in * v41.quantBits(q.mode) / 32) }, .uint32) catch null,
        };
    }
};

fn stToDtype(d: v41.StDtype) ops.Dtype {
    return switch (d) {
        .BF16 => .bfloat16,
        .F32 => .float32,
        .U8 => .uint8,
        .U32 => .uint32,
        else => .float16,
    };
}

/// The routed stand-in's shape: unweighted `[n, k, dim]` f32.
/// A wide routed source on the trace backend that hands out its outputs unjoined (test helper).
const PartsHook = struct {
    outs: *[1]u32,
    k: c_int,
    pub fn routed(_: PartsHook, g: *TraceOps, xf: u32, idx: u32) !u32 {
        return g.input(&.{ g.shapeOf(xf).dim(0), g.shapeOf(idx).dim(1), g.shapeOf(xf).dim(1) }, .float32);
    }
    pub fn routedParts(h: PartsHook, g: *TraceOps, xf: u32, _: u32) !struct { outs: []const u32, loc: u32 } {
        const n = g.shapeOf(xf).dim(0);
        h.outs[0] = try g.input(&.{ n * h.k, g.shapeOf(xf).dim(1) }, .float32);
        return .{ .outs = h.outs[0..1], .loc = try g.input(&.{ n, h.k, 2 }, .int32) };
    }
    pub fn releaseParts(_: PartsHook, _: *TraceOps) void {}
};

const TraceRouted = struct {
    pub fn at(self: TraceRouted, _: u32) TraceRouted {
        return self;
    }

    pub fn routed(_: TraceRouted, g: *TraceOps, xf: u32, indices: u32) !u32 {
        return g.input(&.{ g.shapeOf(xf).dim(0), g.shapeOf(indices).dim(1), g.shapeOf(xf).dim(1) }, .float32);
    }
};

/// The mini config with its Engram bank on disk and its resident spec (test helper).
pub const Mini = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    c: v41.Config,
    src: eng.RowSource,
    spec: []v41.Param,

    pub fn init() !*Mini {
        const m = try testing.allocator.create(Mini);
        errdefer testing.allocator.destroy(m);
        m.arena = std.heap.ArenaAllocator.init(testing.allocator);
        const a = m.arena.allocator();
        m.tmp = std.testing.tmpDir(.{});
        var rbuf: [512]u8 = undefined;
        const root = try a.dupe(u8, rbuf[0..try m.tmp.dir.realPath(testing.io, &rbuf)]);
        const map_path = try eng.writeMiniBank(a, &m.tmp, root, .{});
        const json = try v41.testConfigJson(testing.allocator, .mini);
        defer testing.allocator.free(json);
        m.c = try v41.Config.parse(testing.allocator, json, null);
        m.src = try eng.RowSource.open(testing.allocator, testing.io, root, map_path, &m.c, null);
        const spec = try v41.residentSpec(a, &m.c);
        const espec = try v41.engramSpec(a, &m.c);
        m.spec = try std.mem.concat(a, v41.Param, &.{ spec, espec });
        return m;
    }

    pub fn deinit(m: *Mini) void {
        m.src.deinit();
        m.tmp.cleanup();
        m.arena.deinit();
        testing.allocator.destroy(m);
    }
};

test "dsv41 model: the mini model binds every resident and runs prefill, decode and a verify block" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    var tier = try routes.parse(&.{ .{ "MTPLX_DSV41_WINDOW_RING", "1" }, .{ "MTPLX_DSV41_SELECTED_KEYS", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    tier.routes.head = .mxfp8;
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    var ids: [20]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    // Prefill 20 tokens in chunks of 8 (3 spans); only the last row gets logits.
    const pre = try model_.forward(&g, &st, &ids, .{ .logits = .last, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
    try testing.expect(g.shapeOf(pre.hidden).eql(ops.Shape.of(&.{ 1, 20, 64 })));
    try testing.expect(g.shapeOf(pre.logits.?).eql(ops.Shape.of(&.{ 1, 1, 64 })));
    try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(pre.logits.?));
    // One DSpark target layer (4) of width 64.
    try testing.expect(g.shapeOf(pre.main_hidden.?).eql(ops.Shape.of(&.{ 1, 20, 64 })));
    try testing.expectEqual(@as(u32, 20), st.offset);
    try testing.expectEqual(@as(u32, 20), st.layers[3].compress.rows()); // ratio 1: one row per token
    try testing.expectEqual(@as(u32, 10), st.layers[1].compress.rows()); // ratio 2
    try testing.expectEqual(@as(usize, 20), st.hash.?.hist.items.len);
    // A verify block of K + 1 = 6 rows: every row's logits, then accept 2 of 5 drafts.
    const ver = try model_.forward(&g, &st, ids[0..6], .{ .logits = .all, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
    try testing.expect(g.shapeOf(ver.logits.?).eql(ops.Shape.of(&.{ 1, 6, 64 })));
    try model_.trim(&g, &st, 5 - 2);
    try testing.expectEqual(@as(u32, 23), st.offset);
    for (st.layers) |lc| try testing.expectEqual(@as(u32, 23), lc.offset);
    try testing.expectEqual(@as(u32, 11), st.layers[1].compress.rows());
    try testing.expectEqual(@as(usize, 23), st.hash.?.hist.items.len);
    try testing.expectEqual(@as(u32, 11), st.layers[1].nFed() / 2);
    // Mark, decode two tokens, roll back: every lane and the history return.
    const mk = try model_.mark(testing.allocator, &st);
    defer testing.allocator.free(mk.layers);
    _ = try model_.forward(&g, &st, ids[0..1], .{}, TraceRouted{}, graph.NoProbe{});
    _ = try model_.forward(&g, &st, ids[1..2], .{}, TraceRouted{}, graph.NoProbe{});
    try model_.rollback(&g, &st, mk);
    try testing.expectEqual(@as(u32, 23), st.offset);
    try testing.expectEqual(@as(u32, 23), st.layers[3].compress.rows());
    try testing.expectEqual(@as(usize, 23), st.hash.?.hist.items.len);
}

test "dsv41 model: the mxfp8 head's RCPROJ route is refused off the mxfp8 codec, without kernels and off the head site's geometry" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var tier = try routes.parse(&.{ .{ "MTPLX_DSV41_WINDOW_RING", "1" }, .{ "MTPLX_DSV41_SELECTED_KEYS", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    tier.routes.rc_head_mxfp8 = true;
    tier.routes.head = .bf16;
    try testing.expectError(error.HeadMxNeedsMxfp8, TM.initWith(testing.allocator, &g, m.c, tier, &lookup, &m.src, .{ .registry = &reg }));
    tier.routes.head = .mxfp8;
    try testing.expectError(error.HeadMxNeedsKernels, TM.initWith(testing.allocator, &g, m.c, tier, &lookup, &m.src, .{}));
    // The mini head (64 x 64) is not RCPROJ's head site (129280 x 5120).
    try testing.expectError(error.HeadMxGeometry, TM.initWith(testing.allocator, &g, m.c, tier, &lookup, &m.src, .{ .registry = &reg }));
}

test "dsv41 model: ENGRAM=prefetch at decode width: the forward's posted Engram gathers feed the blocking read's bytes" {
    // One run per route over its own mini bank (identical bytes): a prompt in chunks of 8, then a verify of 3.
    const Run = struct {
        fn of(m: *Mini, g: *TraceOps, posted: bool) !void {
            var tier = try routes.parse(&.{ .{ "MTPLX_DSV41_WINDOW_RING", "1" }, .{ "MTPLX_DSV41_SELECTED_KEYS", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
            tier.routes.head = .mxfp8;
            const lookup: SpecLookup = .{ .g = g, .spec = m.spec };
            const model_ = try TM.init(testing.allocator, g, m.c, tier, &lookup, &m.src);
            defer model_.deinit(g);
            if (posted) {
                try m.src.enablePosting();
                model_.engram.?.posted = true;
            }
            var st = try model_.newState();
            defer st.deinit(g, testing.allocator);
            var ids: [20]u32 = undefined;
            for (&ids, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
            _ = try model_.forward(g, &st, &ids, .{ .logits = .last }, TraceRouted{}, graph.NoProbe{});
            _ = try model_.forward(g, &st, ids[0..3], .{ .logits = .all }, TraceRouted{}, graph.NoProbe{});
        }
    };
    const blocking = try Mini.init();
    defer blocking.deinit();
    const posting = try Mini.init();
    defer posting.deinit();
    var gb = TraceOps.init(testing.allocator);
    defer gb.deinit();
    var gp = TraceOps.init(testing.allocator);
    defer gp.deinit();
    gb.record_host = true;
    gp.record_host = true;
    try Run.of(blocking, &gb, false);
    try Run.of(posting, &gp, true);
    // The same graphs, and every host array (the Engram codes and scales among them) the same bytes.
    try testing.expectEqual(gb.nodes.items.len, gp.nodes.items.len);
    var n_host: usize = 0;
    for (gb.nodes.items, gp.nodes.items, 0..) |nb, np, i| {
        try testing.expectEqual(nb.op, np.op);
        if (nb.op != .host) continue;
        n_host += 1;
        try testing.expectEqualSlices(u8, gb.hostBytesOf(@intCast(i)).?, gp.hostBytesOf(@intCast(i)).?);
    }
    try testing.expect(n_host > 0);
    // The posted run never took the blocking read (its record scratch untouched); the blocking run did.
    try testing.expect(blocking.src.recs.?.items.len > 0);
    try testing.expectEqual(@as(usize, 0), posting.src.recs.?.items.len);
    // The row caches saw the same gathers.
    for (0..blocking.c.engram.n_layers) |li| try testing.expectEqual(blocking.src.cacheStats(li), posting.src.cacheStats(li));
}

test "dsv41 model: K16 settles each chunk's DSpark main tap in its chunk fence (the tap holds no input stream)" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    var ids: [20]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
    const mark = g.nodes.items.len;
    const e0 = g.evaluated.items.len;
    _ = try model_.forward(&g, &st, &ids, .{ .logits = .last, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
    // The taps: the stream's mean over its hc copies, [1, rows, hidden], one per chunk of the target layer (the
    // mini config's layer 4: chunks of 8, 8 and 4 rows). Each is settled by an eval before the next chunk's.
    const dim: c_int = @intCast(m.c.hidden_size);
    var taps: usize = 0;
    for (g.nodes.items[mark..], mark..) |nd, at| {
        if (nd.op != .mean or nd.shape.n != 3 or nd.shape.d[2] != dim or nd.shape.d[0] != 1) continue;
        taps += 1;
        try testing.expect(std.mem.indexOfScalar(u32, g.evaluated.items[e0..], @intCast(at)) != null);
    }
    try testing.expectEqual(@as(usize, 3), taps);
    try testing.expect(main_taps_in_chunk_fence);
    // The bill's kept taps (`PrefillBill.layerMajorWaveTerms`: n_main x seq x hidden x stream_bytes): a tap is the
    // stream's dtype (kv16's bf16 stream: bf16, never f32).
    inline for (.{ ops.Dtype.bfloat16, ops.Dtype.float32 }) |dt| {
        const hs = try g.input(&.{ 1, 8, @intCast(m.c.hc_mult), @intCast(m.c.hidden_size) }, dt);
        try testing.expectEqual(dt, g.dtypeOf(try TM.mainOf(&g, hs)));
    }
}

test "dsv41 model: K16's input-stream release (route): each chunk's layer input goes at its chunk fence, before the layer's routed call; the ops are the same either way" {
    const m = try Mini.init();
    defer m.deinit();
    // Per layer: the streams its HC posts produced (the next layer's inputs) and the releases made by its first routed call.
    const Rec = struct {
        g: *TraceOps,
        layer: usize = 0,
        outs: [8]std.ArrayList(u32) = @splat(.empty),
        released_at_routed: [8]?usize = @splat(null),
        pub fn put(self: *@This(), name: []const u8, x: anytype) !void {
            if (std.mem.eql(u8, name, "out.h")) try self.outs[self.layer].append(testing.allocator, x);
            if (std.mem.eql(u8, name, "moe.routed") and self.released_at_routed[self.layer] == null) self.released_at_routed[self.layer] = self.g.released.items.len;
            if (std.mem.eql(u8, name, "layer.end")) self.layer += 1;
        }
        fn deinit(self: *@This()) void {
            for (&self.outs) |*o| o.deinit(testing.allocator);
        }
    };
    var seqs: [2]?[]ops.Op = .{ null, null };
    defer for (seqs) |sq| if (sq) |x| testing.allocator.free(x);
    for ([_]bool{ false, true }, &seqs) |early, *seq| {
        var g = TraceOps.init(testing.allocator);
        defer g.deinit();
        const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
        var tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
        tier.routes.input_stream_early_release = early;
        const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
        defer model_.deinit(&g);
        var st = try model_.newState();
        defer st.deinit(&g, testing.allocator);
        var rec: Rec = .{ .g = &g };
        defer rec.deinit();
        var ids: [20]u32 = undefined;
        for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
        const mark = g.nodes.items.len;
        _ = try model_.forward(&g, &st, &ids, .{ .logits = .last, .main_hidden = true }, TraceRouted{}, &rec);
        seq.* = try g.opsSince(testing.allocator, mark);
        // Layers 1..: each input stream (the previous layer's HC post output, one per chunk) is released exactly once,
        // before the layer's first routed call on the route, after it without.
        try testing.expectEqual(m.c.n_layers, rec.layer);
        for (1..m.c.n_layers) |l| {
            const at = rec.released_at_routed[l].?;
            try testing.expectEqual(@as(usize, 3), rec.outs[l - 1].items.len);
            for (rec.outs[l - 1].items) |x| {
                try testing.expectEqual(early, std.mem.indexOfScalar(u32, g.released.items[0..at], x) != null);
                try testing.expectEqual(@as(usize, 1), std.mem.count(u32, g.released.items, &.{x}));
            }
        }
    }
    // Lifetime only: the same op sequence with the route on and off.
    try testing.expectEqualSlices(ops.Op, seqs[0].?, seqs[1].?);
}

test "dsv41 model: PREFILL_INPUT_RELEASE (route): each routed group's MoE inputs are dropped after the wide call and never read again; the ops are the same either way" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    var c = try v41.Config.parse(testing.allocator, json, null);
    // Two real-geometry layers (JOINLESS's combine is derived for them), no Engram.
    c.n_layers = 2;
    c.engram.n_layers = 0;
    for (c.layers[0..2]) |*li| li.engram_slot = null;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = try v41.residentSpec(arena.allocator(), &c);
    const k: c_int = @intCast(c.n_experts_per_tok);
    // A wide routed source that hands out its outputs unjoined (one output, every assignment's row in it).
    const Parts = struct {
        outs: [1]u32 = undefined,
        k: c_int,
        pub fn at(self: *@This(), _: u32) PartsHook {
            return .{ .outs = &self.outs, .k = self.k };
        }
    };
    var seqs: [2]?[]ops.Op = .{ null, null };
    defer for (seqs) |sq| if (sq) |x| testing.allocator.free(x);
    for ([_]bool{ false, true }, &seqs) |rel, *seq| {
        var g = TraceOps.init(testing.allocator);
        defer g.deinit();
        const lookup: SpecLookup = .{ .g = &g, .spec = spec };
        var tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
        tier.routes.prefill_joinless = true;
        tier.routes.prefill_host_shared = true;
        tier.routes.prefill_input_release = rel;
        const model_ = try TM.initWith(testing.allocator, &g, c, tier, &lookup, null, .{ .registry = &reg });
        defer model_.deinit(&g);
        var st = try model_.newState();
        defer st.deinit(&g, testing.allocator);
        var src: Parts = .{ .k = k };
        var ids: [20]u32 = undefined;
        for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 1000);
        const mark = g.nodes.items.len;
        _ = try model_.forward(&g, &st, &ids, .{ .logits = .last }, &src, graph.NoProbe{});
        seq.* = try g.opsSince(testing.allocator, mark);
        // Off: nothing dropped. On: per layer, three chunks' moe_in and row views and the group's concat, none of them
        // read (shape, dtype or evaluation) after the drop.
        try testing.expectEqual(@as(usize, if (rel) 2 * (3 + 3 + 1) else 0), g.dropped.items.len);
        try testing.expectEqual(@as(u32, 0), g.use_after_drop);
        if (rel) {
            // The detector itself: a read of a dropped array counts.
            _ = g.shapeOf(g.dropped.items[0]);
            try testing.expectEqual(@as(u32, 1), g.use_after_drop);
        }
    }
    // Lifetime only: the same op sequence with the route on and off.
    try testing.expectEqualSlices(ops.Op, seqs[0].?, seqs[1].?);
}

test "dsv41 model: K16 marks each chunk's build, fence start and fence end for a fence probe (D14's timeline), in order" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    const Rec = struct {
        marks: std.ArrayList(FenceAt) = .empty,
        evals_at_wait: std.ArrayList(usize) = .empty,
        g: *TraceOps,
        pub fn put(_: *@This(), _: []const u8, _: anytype) !void {}
        pub fn fenceMark(self: *@This(), at: FenceAt) void {
            self.marks.append(testing.allocator, at) catch unreachable;
            if (at != .build) self.evals_at_wait.append(testing.allocator, self.g.evals.items.len) catch unreachable;
        }
    };
    var rec: Rec = .{ .g = &g };
    defer rec.marks.deinit(testing.allocator);
    defer rec.evals_at_wait.deinit(testing.allocator);
    var ids: [20]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
    _ = try model_.forward(&g, &st, &ids, .{ .logits = .last, .main_hidden = true }, TraceRouted{}, &rec);
    // 5 layers x 3 chunks (8, 8, 4 rows), each build -> wait -> done.
    const n = m.c.n_layers * 3;
    try testing.expectEqual(n * 3, rec.marks.items.len);
    for (0..n) |k| try testing.expectEqualSlices(FenceAt, &.{ .build, .wait, .done }, rec.marks.items[k * 3 ..][0..3]);
    // The fence's eval falls between its wait and done marks (one eval: the fence).
    for (0..n) |k| try testing.expectEqual(rec.evals_at_wait.items[k * 2] + 1, rec.evals_at_wait.items[k * 2 + 1]);
}

test "dsv41 model: K16 layer-major prefill runs every layer over all chunks, one compiled combine per chunk" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    var ids: [20]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
    const mark = g.nodes.items.len;
    const r = try model_.forward(&g, &st, &ids, .{ .logits = .last, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
    try testing.expect(g.shapeOf(r.hidden).eql(ops.Shape.of(&.{ 1, 20, 64 })));
    try testing.expect(g.shapeOf(r.main_hidden.?).eql(ops.Shape.of(&.{ 1, 20, 64 })));
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    // 5 layers x 3 chunks of `_PREFILL_HC_POST`; nothing else compiled at 8-row chunks.
    try testing.expectEqual(@as(usize, 5 * 3), std.mem.count(ops.Op, seq, &.{.tape_begin}));
    try testing.expectEqual(@as(u32, 20), st.offset);
    try testing.expectEqual(@as(u32, 10), st.layers[1].compress.rows());
    // The engram needs its row source; a model without it refuses at construction.
    try testing.expectError(error.EngramSourceRequired, TM.init(testing.allocator, &g, m.c, tier, &lookup, null));
}

test "dsv41 model: a prompt's sub-chunk calls run the one-call pass's spans over the same state (1, 2 and 3 calls)" {
    const a = testing.allocator;
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(a, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    const span: u64 = 8;
    const sub: u64 = 16;
    // 12 tokens: one call; 36: two (the 4-row tail rides on the second); 50: three.
    for ([_]u32{ 12, 36, 50 }, [_]usize{ 1, 2, 3 }) |n, n_calls| {
        var ids: [64]u32 = undefined;
        for (ids[0..n], 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
        const calls = try kvc.prefillSubCalls(a, n, span, sub);
        defer a.free(calls);
        try testing.expectEqual(n_calls, calls.len);
        // The one call.
        var one = try model_.newState();
        defer one.deinit(&g, a);
        const m1 = g.nodes.items.len;
        _ = try model_.forward(&g, &one, ids[0..n], .{ .logits = .last, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
        const seq1 = try g.opsSince(a, m1);
        defer a.free(seq1);
        // The sub-chunk calls over one state, the spans pinned to the whole prompt's chunk rule.
        var st = try model_.newState();
        defer st.deinit(&g, a);
        st.span_chunk = @intCast(span);
        var tapes: usize = 0;
        for (calls) |c| {
            const mk = g.nodes.items.len;
            const r = try model_.forward(&g, &st, ids[c[0]..c[1]], .{ .logits = .last, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
            try testing.expect(g.shapeOf(r.main_hidden.?).eql(ops.Shape.of(&.{ 1, @as(c_int, @intCast(c[1] - c[0])), 64 })));
            const seq = try g.opsSince(a, mk);
            defer a.free(seq);
            tapes += std.mem.count(ops.Op, seq, &.{.tape_begin});
        }
        // The same spans: one compiled HC post per layer and span, ceil(n / 8) spans either way.
        const n_spans = (n + span - 1) / span;
        try testing.expectEqual(m.c.n_layers * n_spans, std.mem.count(ops.Op, seq1, &.{.tape_begin}));
        try testing.expectEqual(m.c.n_layers * n_spans, tapes);
        // The same state: the offset, every layer's window / compressed / index rows, the Engram history.
        try testing.expectEqual(one.offset, st.offset);
        for (one.layers, st.layers) |*x, *y| {
            try testing.expectEqual(x.window.rows(), y.window.rows());
            try testing.expectEqual(x.compress.rows(), y.compress.rows());
            try testing.expectEqual(x.index.rows(), y.index.rows());
        }
        try testing.expectEqualSlices(i64, one.hash.?.hist.items, st.hash.?.hist.items);
    }
    // The pin is what the calls read: a 4-row pin over the tier's 8 doubles the spans of a 16-row call.
    var st = try model_.newState();
    defer st.deinit(&g, a);
    st.span_chunk = 4;
    var ids: [16]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast(i + 1);
    const mk = g.nodes.items.len;
    _ = try model_.forward(&g, &st, &ids, .{ .logits = .last }, TraceRouted{}, graph.NoProbe{});
    const seq = try g.opsSince(a, mk);
    defer a.free(seq);
    try testing.expectEqual(m.c.n_layers * 4, std.mem.count(ops.Op, seq, &.{.tape_begin}));
}

test "dsv41 model: a conversation's next turn from the last prompt's boundary has the state of a cold prompt of the whole conversation (two and three turns)" {
    const a = testing.allocator;
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(a);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(a, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var conv: [96]u32 = undefined;
    for (&conv, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    // Turn prompts end at 24, 52 and 90; each turn decodes 6 ids that the next prompt does not keep (`cut` 1: the
    // thinking turn's last prompt id re-rendered), then the next prompt continues from the boundary.
    for ([_]u32{ 0, 1 }) |cut| {
        var st = try model_.newStateWith(model_.boundedKv(200));
        defer st.deinit(&g, a);
        const ends = [_]u32{ 24, 52, 90 };
        _ = try model_.forward(&g, &st, conv[0..ends[0]], .{ .logits = .last }, TraceRouted{}, graph.NoProbe{});
        var prev = ends[0];
        for (ends[1..]) |e| {
            var b = try model_.boundary(&g, a, &st);
            defer b.deinit(&g, a);
            var junk: [6]u32 = .{ 60, 61, 62, 63, 1, 2 };
            for (&junk) |*j| _ = try model_.forward(&g, &st, j[0..1], .{ .logits = .last }, TraceRouted{}, graph.NoProbe{});
            try model_.restoreBoundary(&g, a, &st, &b);
            try testing.expectEqual(prev, st.offset);
            if (cut > 0) try model_.trim(&g, &st, cut);
            _ = try model_.forward(&g, &st, conv[prev - cut .. e], .{ .logits = .last }, TraceRouted{}, graph.NoProbe{});
            // The same state as a cold prompt of the whole conversation so far.
            var cold = try model_.newStateWith(model_.boundedKv(200));
            defer cold.deinit(&g, a);
            _ = try model_.forward(&g, &cold, conv[0..e], .{ .logits = .last }, TraceRouted{}, graph.NoProbe{});
            try testing.expectEqual(cold.offset, st.offset);
            for (cold.layers, st.layers) |*x, *y| {
                try testing.expectEqual(x.window.rows(), y.window.rows());
                try testing.expectEqual(x.compress.rows(), y.compress.rows());
                try testing.expectEqual(x.index.rows(), y.index.rows());
            }
            try testing.expectEqualSlices(i64, cold.hash.?.hist.items, st.hash.?.hist.items);
            prev = e;
        }
    }
}

/// Records each stage a pass publishes and the chunk the profile charges it to (PrefillProbe's attribution).
const StageProbe = struct {
    names: std.ArrayList([]const u8) = .empty,
    chunks: std.ArrayList(usize) = .empty,
    cur: usize = 0,
    group_halves: std.ArrayList(usize) = .empty,
    group_next: std.ArrayList(usize) = .empty,
    fn deinit(self: *StageProbe) void {
        self.names.deinit(testing.allocator);
        self.chunks.deinit(testing.allocator);
        self.group_halves.deinit(testing.allocator);
        self.group_next.deinit(testing.allocator);
    }
    pub fn atChunk(self: *StageProbe, i: usize) void {
        self.cur = i;
    }
    pub fn put(self: *StageProbe, name: []const u8, _: anytype) !void {
        try self.names.append(testing.allocator, name);
        try self.chunks.append(testing.allocator, self.cur);
    }
    /// The grouped profile's hooks (no evaluation): recorded in the stage order with the group's sizes.
    pub fn groupHalves(self: *StageProbe, halves: anytype, _: usize) void {
        self.names.append(testing.allocator, "group.halves") catch unreachable;
        self.chunks.append(testing.allocator, self.cur) catch unreachable;
        self.group_halves.append(testing.allocator, halves.len) catch unreachable;
    }
    pub fn groupEval(self: *StageProbe, _: anytype, _: anytype, _: anytype, next: anytype, _: anytype) void {
        self.names.append(testing.allocator, "group.mark") catch unreachable;
        self.chunks.append(testing.allocator, self.cur) catch unreachable;
        self.group_next.append(testing.allocator, next.len) catch unreachable;
    }
    fn count(self: *const StageProbe, name: []const u8) usize {
        var n: usize = 0;
        for (self.names.items) |x| n += @intFromBool(std.mem.eql(u8, x, name));
        return n;
    }
};

test "dsv41 model: the K16 profile charges a chunk's carry-over to its own stages (combine, HC post, fence, frees, layer end); out.h once per chunk and layer" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    var ids: [20]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 5 + 1) % 64);
    var p: StageProbe = .{};
    defer p.deinit();
    _ = try model_.forward(&g, &st, &ids, .{ .logits = .last, .main_hidden = true }, TraceRouted{}, &p);
    const nl: usize = m.c.n_layers;
    const nc = 3; // 8 + 8 + 4 rows
    try testing.expectEqualStrings("embed", p.names.items[0]);
    // out.h once per chunk and layer, layer-major, each charged to its own chunk (the header's chunks: 3, not 0).
    var outs: std.ArrayList(usize) = .empty;
    defer outs.deinit(testing.allocator);
    for (p.names.items, p.chunks.items) |x, ch| if (std.mem.eql(u8, x, "out.h")) try outs.append(testing.allocator, ch);
    try testing.expectEqual(nl * nc, outs.items.len);
    for (outs.items, 0..) |ch, k| try testing.expectEqual(k % nc, ch);
    for ([_][]const u8{ "chunk.fence", "chunk.frees", "moe.y", "attn.in" }) |x| try testing.expectEqual(nl * nc, p.count(x));
    try testing.expectEqual(nl, p.count("layer.end"));
    try testing.expect(p.count("group.frees") >= nl);
    // attn.in follows only a carry-over stage of its own, and every moe.y is its chunk's HC post's predecessor.
    for (p.names.items, p.chunks.items, 0..) |x, ch, k| {
        if (std.mem.eql(u8, x, "attn.in")) {
            const prev = p.names.items[k - 1];
            var ok = false;
            for ([_][]const u8{ "embed", "chunk.frees", "layer.end", "engram.add" }) |want| ok = ok or std.mem.eql(u8, prev, want);
            try testing.expect(ok);
        }
        if (std.mem.eql(u8, x, "out.h")) {
            try testing.expectEqualStrings("moe.y", p.names.items[k - 1]);
            try testing.expectEqual(ch, p.chunks.items[k - 1]);
        }
    }
    // (served run 19) the grouped profile's points: per routed group, the halves read after its routed call (before the first
    // combine), the mark after the last HC post's build, then group.eval (the group's one evaluation, charged to its first
    // chunk) right before group.frees (charged to its last chunk, as before); the halves and the new streams are the
    // group's chunks.
    const ng = p.count("group.frees");
    for ([_][]const u8{ "group.halves", "group.mark", "group.eval" }) |x| try testing.expectEqual(ng, p.count(x));
    try testing.expectEqualSlices(usize, p.group_halves.items, p.group_next.items);
    var first: usize = 0;
    for (p.names.items, p.chunks.items, 0..) |x, ch, k| {
        if (std.mem.eql(u8, x, "group.halves")) {
            try testing.expectEqualStrings("moe.routed", p.names.items[k - 1]);
            // The group's first combine follows (its shared expert's own point first on the combine path without parts).
            var y = k + 1;
            while (!std.mem.eql(u8, p.names.items[y], "moe.y")) : (y += 1) try testing.expectEqualStrings("moe.shared", p.names.items[y]);
            first = p.chunks.items[y];
        }
        if (std.mem.eql(u8, x, "group.mark")) try testing.expectEqualStrings("out.h", p.names.items[k - 1]);
        if (std.mem.eql(u8, x, "group.eval")) {
            try testing.expectEqualStrings("group.mark", p.names.items[k - 1]);
            try testing.expectEqualStrings("group.frees", p.names.items[k + 1]);
            try testing.expectEqual(first, ch);
            try testing.expectEqual(p.chunks.items[k - 1], p.chunks.items[k + 1]);
        }
    }
    for (p.group_halves.items) |n| try testing.expect(n > 0);
}

test "dsv41 model: the AR dry path routes every layer call of every forward through the expert source" {
    const xp = @import("deepseek_v41_experts.zig");
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{}, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    // The harness's source: no prefill rows, grown before the first forward
    // (every call a decode route), all four experts resident after growth.
    const nl = m.c.n_layers;
    var rows0: [8]u32 = @splat(0);
    var rows1: [8]u32 = @splat(4);
    var src = try xp.FakeSource.init(testing.allocator, .{ .hidden = m.c.hidden_size, .inter = m.c.moe_intermediate_size, .n_experts = m.c.n_routed_experts, .rows = rows0[0..nl] });
    defer src.deinit();
    const Ex = xp.Experts(TraceOps, xp.FakeSource, xp.TraceMath);
    var ex = try Ex.init(testing.allocator, &g, &src, .{ .hidden = @intCast(m.c.hidden_size), .inter = @intCast(m.c.moe_intermediate_size) }, &m.c);
    defer ex.deinit();
    try ex.grow(&g, rows1[0..nl]);
    const Host = struct {
        rng: std.Random.DefaultPrng,
        n: u16,
        picks: u32 = 0,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            for (out) |*o| o.* = h.rng.random().uintLessThan(u16, h.n);
        }
        fn argmax(ctx: *anyopaque) anyerror!u32 {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            h.picks += 1;
            return (h.picks * 5 + 1) % 64;
        }
    };
    var host: Host = .{ .rng = std.Random.DefaultPrng.init(20260928), .n = @intCast(m.c.n_routed_experts) };
    g.host_values = .{ .ctx = &host, .ids = Host.ids, .argmax = Host.argmax };
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*d, i| d.* = @intCast((i * 7 + 3) % 64);
    var out: [4]u32 = undefined;
    // 20 prompt tokens in forwards of 8, 8, 4 rows; then 3 one-token forwards.
    try model_.greedy(&g, &st, &prompt, 8, &ex, &out, {});
    try testing.expectEqualSlices(u32, &.{ 6, 11, 16, 21 }, &out);
    try testing.expectEqual(@as(u32, 23), st.offset);
    const forwards = 3 + 3;
    try testing.expectEqual(@as(u64, forwards * nl), src.stats().route_calls);
    try testing.expectEqual(@as(usize, 0), src.liveCalls());
    // Per forward: every layer routes and releases its call once, in layer order.
    var n_route: usize = 0;
    var n_release: usize = 0;
    var layer_next: u32 = 0;
    for (src.log.items) |e| switch (e.kind) {
        .route => {
            try testing.expectEqual(layer_next, e.layer);
            layer_next = (layer_next + 1) % nl;
            n_route += 1;
        },
        .release => n_release += 1,
        .wait_gu, .wait_down => try testing.expectEqual((layer_next + nl - 1) % nl, e.layer),
        .flush, .grow, .gate, .read_ahead, .await_read_ahead => {},
    };
    try testing.expectEqual(n_route, n_release);
    try testing.expectEqual(@as(u32, 4), host.picks);
    // A prompt forward wider than the decode lane (top-2 x 25 rows > 48 ids) is refused by route.
    try testing.expectError(error.PrefillLaneNotPorted, model_.greedy(&g, &st, &(@as([25]u32, @splat(1))), 25, &ex, &out, {}));
}

test "dsv41 model: a prompt forward wider than a route takes runs every layer's routed call through the wide lane" {
    const xp = @import("deepseek_v41_experts.zig");
    const quant = @import("sdk_ext.zig").quant;
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const model_ = try TM.init(testing.allocator, &g, m.c, try routes.parse(&.{}, null), &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    const nl = m.c.n_layers;
    var rows0: [8]u32 = @splat(4);
    var src = try xp.FakeSource.init(testing.allocator, .{ .hidden = m.c.hidden_size, .inter = m.c.moe_intermediate_size, .n_experts = m.c.n_routed_experts, .rows = rows0[0..nl] });
    defer src.deinit();
    // A wide route that counts its calls and returns the DIG route's output shape.
    const Count = struct {
        calls: u32 = 0,
        rows: u32 = 0,
        pub fn call(self: *@This(), gg: *TraceOps, act: u32, r: quant.PrefillRows, _: xp.BankArraysOf(u32)) !u32 {
            self.calls += 1;
            self.rows += @intCast(r.slot.len);
            return gg.input(&.{ @intCast(r.slot.len), gg.shapeOf(act).d[1] }, .float32);
        }
        pub fn finish(_: *@This(), _: *TraceOps) !void {}
    };
    var counts: [8]Count = @splat(.{});
    const Math = xp.WithPrefillRoutes(TraceOps, xp.TraceMath, Count);
    const Ex = xp.ExpertsWith(TraceOps, xp.FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.init(testing.allocator, &g, &src, .{ .d = .{ .hidden = @intCast(m.c.hidden_size), .inter = @intCast(m.c.moe_intermediate_size) }, .routes = counts[0..nl] }, &m.c);
    defer ex.deinit();
    const Host = struct {
        n: u16,
        next: u16 = 0,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            for (out) |*o| {
                o.* = h.next % h.n;
                h.next += 1;
            }
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return 7;
        }
    };
    var host: Host = .{ .n = @intCast(m.c.n_routed_experts) };
    g.host_values = .{ .ctx = &host, .ids = Host.ids, .argmax = Host.argmax };
    // 25 prompt rows x top-k in one forward: every layer's call is wide; then two decode forwards.
    const k = m.c.n_experts_per_tok;
    try testing.expect(25 * k > xp.max_route_ids);
    var out: [3]u32 = undefined;
    try model_.greedy(&g, &st, &(@as([25]u32, @splat(1))), 25, &ex, &out, {});
    try testing.expectEqualSlices(u32, &.{ 7, 7, 7 }, &out);
    try testing.expectEqual(@as(u32, 27), st.offset);
    for (counts[0..nl]) |r| {
        try testing.expectEqual(@as(u32, 1), r.calls);
        try testing.expectEqual(@as(u32, 25 * k), r.rows);
    }
    // One route per layer for the wide forward (4 experts: one group), one per layer per decode forward.
    try testing.expectEqual(@as(u64, 3 * nl), src.stats().route_calls);
}

test "dsv41 model: P1: each layer's predictor pass counts its chunks' predicted ids, hands the seed's ranking to the hook, and the call lands it before routing" {
    const xp = @import("deepseek_v41_experts.zig");
    const quant = @import("sdk_ext.zig").quant;
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    const nl = m.c.n_layers;
    const n: u16 = @intCast(m.c.n_routed_experts);
    const k = m.c.n_experts_per_tok;
    var rows0: [8]u32 = @splat(2);
    var src = try xp.FakeSource.init(testing.allocator, .{ .hidden = m.c.hidden_size, .inter = m.c.moe_intermediate_size, .n_experts = n, .rows = rows0[0..nl] });
    defer src.deinit();
    const Count = struct {
        pub fn call(_: *@This(), gg: *TraceOps, act: u32, r: quant.PrefillRows, _: xp.BankArraysOf(u32)) !u32 {
            return gg.input(&.{ @intCast(r.slot.len), gg.shapeOf(act).d[1] }, .float32);
        }
        pub fn finish(_: *@This(), _: *TraceOps) !void {}
    };
    var counts: [8]Count = @splat(.{});
    const Math = xp.WithPrefillRoutes(TraceOps, xp.TraceMath, Count);
    const Ex = xp.ExpertsWith(TraceOps, xp.FakeSource, Math, .{ .prefill = true });
    var ex = try Ex.initWith(testing.allocator, &g, &src, .{ .d = .{ .hidden = @intCast(m.c.hidden_size), .inter = @intCast(m.c.moe_intermediate_size) }, .routes = counts[0..nl] }, &m.c, .{ .wide = .{ .seed = true, .read_ahead = true } });
    defer ex.deinit();
    // Every host read of ids (a predictor chunk's, a routing barrier's) is the same pattern: 0, 3, 2, 1, ...
    const Host = struct {
        n: u16,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            for (out, 0..) |*o, i| o.* = @intCast((i * 3) % h.n);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return 7;
        }
    };
    var host: Host = .{ .n = n };
    g.host_values = .{ .ctx = &host, .ids = Host.ids, .argmax = Host.argmax };
    // 25 prompt rows in chunks of 8, 8, 8, 1: the predictor's counts per layer are the four reads' ids.
    var want_counts: [4]u32 = @splat(0);
    for ([_]usize{ 8, 8, 8, 1 }) |rows| for (0..rows * k) |i| {
        want_counts[(i * 3) % n] += 1;
    };
    var rank_buf: [4]u16 = undefined;
    const want = expert_policy.rankHottest(&want_counts, &rank_buf);
    try testing.expectEqualSlices(u16, &.{ 0, 3, 1, 2 }, want);
    var out: [1]u32 = undefined;
    try model_.greedy(&g, &st, &(@as([25]u32, @splat(1))), 25, &ex, &out, {});
    // Per layer: the ranking's head admitted (2 rows), in order; read ahead, landed, then the layer's routes.
    try testing.expectEqual(@as(usize, 2 * nl), src.ahead_log.items.len);
    for (0..nl) |l| try testing.expectEqualSlices(u16, want[0..2], src.ahead_log.items[2 * l ..][0..2]);
    var stage: [8]u8 = @splat(0);
    for (src.log.items) |e| switch (e.kind) {
        .read_ahead => {
            try testing.expectEqual(@as(u8, 0), stage[e.layer]);
            try testing.expectEqual(@as(u32, 2), e.part);
            stage[e.layer] = 1;
        },
        .await_read_ahead => {
            try testing.expectEqual(@as(u8, 1), stage[e.layer]);
            stage[e.layer] = 2;
        },
        .route => try testing.expectEqual(@as(u8, 2), stage[e.layer]),
        else => {},
    };
    for (stage[0..nl]) |s_| try testing.expectEqual(@as(u8, 2), s_);
    // The seed the call chose is the read-ahead's pair: each layer's first route hits both.
    try testing.expectEqual(@as(u64, 2 * nl), src.stats().expert_cache_hits);
}

test "dsv41 model: P1's predictor reads its chunks' ids in batches: one eval a batch of four, every chunk's ids counted" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
    const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
    defer model_.deinit(&g);
    const n: u16 = @intCast(m.c.n_routed_experts);
    // Every host read of ids: 0, 3, 6, ... mod n, the chunk's rows x top-k of them.
    const Host = struct {
        n: u16,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            for (out, 0..) |*o, i| o.* = @intCast((i * 3) % h.n);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return 0;
        }
    };
    var host: Host = .{ .n = n };
    g.host_values = .{ .ctx = &host, .ids = Host.ids, .argmax = Host.argmax };
    const Hook = struct {
        ranked: std.ArrayList(u16) = .empty,
        pub fn readAheadSeed(self: *@This(), r: []const u16) !void {
            try self.ranked.appendSlice(testing.allocator, r);
        }
    };
    var hook: Hook = .{};
    defer hook.ranked.deinit(testing.allocator);
    // Six chunks of 8, 8, 8, 8, 8 and 3 rows: two batches (4 + 2), six host reads.
    const rows = [_]c_int{ 8, 8, 8, 8, 8, 3 };
    var hs: [rows.len]u32 = undefined;
    var pms: [rows.len]u32 = undefined;
    for (rows, &hs, &pms) |r, *h, *pm| {
        h.* = try g.input(&.{ 1, r, @intCast(m.c.hc_mult), @intCast(m.c.hidden_size) }, .float32);
        pm.* = try g.input(&.{ 1, r, @intCast(m.c.hc_mult) }, .float32);
    }
    const e0 = g.evals.items.len;
    const n0 = g.nodes.items.len;
    try model_.predictSeed(&g, 0, &model_.layers[0], &hook, &hs, &pms, graph.NoProbe{});
    try testing.expectEqual(@as(usize, 2), g.evals.items.len - e0);
    // Every chunk's ids are read after its batch's eval (in place) and before the next batch's.
    const ev = g.evals.items[e0..];
    var reads: usize = 0;
    for (g.nodes.items[n0..], n0..) |nd, at| if (nd.op == .host_read) {
        const batch = reads / TM.predict_batch;
        try testing.expect(at >= ev[batch]);
        if (batch + 1 < ev.len) try testing.expect(at < ev[batch + 1]);
        reads += 1;
    };
    try testing.expectEqual(@as(usize, rows.len), reads);
    // The counts are every chunk's ids: the ranking the hook got is the one of the summed pattern.
    var want: [512]u32 = @splat(0);
    for (rows) |r| for (0..@as(usize, @intCast(r)) * m.c.n_experts_per_tok) |i| {
        want[(i * 3) % n] += 1;
    };
    var buf: [512]u16 = undefined;
    try testing.expectEqualSlices(u16, expert_policy.rankHottest(want[0..n], buf[0..n]), hook.ranked.items);
}

test "dsv41 model: P1's predictor in bf16 (Routes.predict_bf16): the GEMM on the gate as stored, no f32 copy of the gate; the evals, reads and seed are the f32 route's" {
    const m = try Mini.init();
    defer m.deinit();
    const n: c_int = @intCast(m.c.n_routed_experts);
    const H: c_int = @intCast(m.c.hidden_size);
    const Host = struct {
        n: u16,
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            for (out, 0..) |*o, i| o.* = @intCast((i * 5) % h.n);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return 0;
        }
    };
    const Hook = struct {
        ranked: std.ArrayList(u16) = .empty,
        pub fn readAheadSeed(self: *@This(), r: []const u16) !void {
            try self.ranked.appendSlice(testing.allocator, r);
        }
    };
    // Three chunks (one batch); no row count equals the expert count, so the shapes below name the gate alone.
    const rows = [_]c_int{ 8, 8, 3 };
    const Seen = struct { gemm_f32: usize = 0, gemm_bf16: usize = 0, gate_to_f32: usize = 0, evals: usize = 0, reads: usize = 0 };
    var seen: [2]Seen = .{ .{}, .{} };
    var seeds: [2]std.ArrayList(u16) = .{ .empty, .empty };
    defer for (&seeds) |*x| x.deinit(testing.allocator);
    for ([_]bool{ false, true }, &seen, &seeds) |bf16, *sn, *seed| {
        var g = TraceOps.init(testing.allocator);
        defer g.deinit();
        const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
        var tier = try routes.parse(&.{ .{ "MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1" }, .{ "MTPLX_DSV41_PREFILL_CHUNK", "8" } }, null);
        tier.routes.predict_bf16 = bf16;
        const model_ = try TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src);
        defer model_.deinit(&g);
        var host: Host = .{ .n = @intCast(n) };
        g.host_values = .{ .ctx = &host, .ids = Host.ids, .argmax = Host.argmax };
        var hook: Hook = .{};
        defer hook.ranked.deinit(testing.allocator);
        var hs: [rows.len]u32 = undefined;
        var pms: [rows.len]u32 = undefined;
        for (rows, &hs, &pms) |r, *h, *pm| {
            h.* = try g.input(&.{ 1, r, @intCast(m.c.hc_mult), H }, .float32);
            pm.* = try g.input(&.{ 1, r, @intCast(m.c.hc_mult) }, .float32);
        }
        const e0 = g.evals.items.len;
        const n0 = g.nodes.items.len;
        try model_.predictSeed(&g, 0, &model_.layers[0], &hook, &hs, &pms, graph.NoProbe{});
        sn.evals = g.evals.items.len - e0;
        for (g.nodes.items[n0..]) |nd| switch (nd.op) {
            .matmul => if (nd.shape.n == 2 and nd.shape.dim(1) == n) switch (nd.dtype) {
                .float32 => sn.gemm_f32 += 1,
                .bfloat16 => sn.gemm_bf16 += 1,
                else => {},
            },
            .astype => if (nd.dtype == .float32 and nd.shape.n == 2 and nd.shape.dim(0) == n and nd.shape.dim(1) == H) {
                sn.gate_to_f32 += 1;
            },
            .host_read => sn.reads += 1,
            else => {},
        };
        try seed.appendSlice(testing.allocator, hook.ranked.items);
    }
    // f32 (the served default): one f32 copy of the gate for the pass, each chunk's GEMM in f32.
    try testing.expectEqual(Seen{ .gemm_f32 = rows.len, .gate_to_f32 = 1, .evals = 1, .reads = rows.len }, seen[0]);
    // bf16: no copy, each chunk's GEMM in bf16; the same evals and reads, and from the same ids the same seed.
    try testing.expectEqual(Seen{ .gemm_bf16 = rows.len, .evals = 1, .reads = rows.len }, seen[1]);
    try testing.expectEqualSlices(u16, seeds[0].items, seeds[1].items);
}

test "dsv41 model: the routed row cap follows _derive_moe_row_cap" {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    // 8e9 // (6 x 5120 x 4): the whole 16,384-token prompt is one routed call.
    try testing.expectEqual(@as(u64, 65104), moeRowCap(&c, kvc.default_chunk_target_bytes));
}

/// The model over MLX with the checkpoint's weights map and the routed
/// stand-in: referenced (never called) by the test below so the MLX
/// instantiation is analysed on the host.
fn mlxSmoke(gpa: std.mem.Allocator, g: *ops.MlxOps, c: v41.Config, tier: routes.Tier, w: *const @import("sdk").Weights, src: ?*const eng.RowSource, routed: graph.StandIn(ops.MlxOps)) !void {
    const M = Model(ops.MlxOps);
    const m = try M.init(gpa, g, c, tier, w, src);
    defer m.deinit(g);
    var st = try m.newState();
    defer st.deinit(g, gpa);
    _ = try m.forward(g, &st, &.{ 1, 2, 3 }, .{ .logits = .last, .main_hidden = true }, routed, graph.NoProbe{});
    try m.trim(g, &st, 1);
    const mk = try m.mark(gpa, &st);
    defer gpa.free(mk.layers);
    try m.rollback(g, &st, mk);
}

test "dsv41 model: each layer of a forward is one wave, freed at the layer's end; the tail builds after the last" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    const model_ = try TM.init(testing.allocator, &g, m.c, try routes.parse(&.{}, null), &lookup, &m.src);
    defer model_.deinit(&g);
    var st = try model_.newState();
    defer st.deinit(&g, testing.allocator);
    const ids = [_]u32{ 3, 10, 17, 24, 31, 38 };
    // A verify-shaped forward (every row's logits and the DSpark taps), then a decode forward.
    for ([_][]const u32{ &ids, ids[0..1] }) |span| {
        const first: u32 = @intCast(g.nodes.items.len);
        const waves0 = g.freed.items.len;
        const r = try model_.forward(&g, &st, span, .{ .logits = .all, .main_hidden = true }, TraceRouted{}, graph.NoProbe{});
        // The layer waves, in order (a prefill-width layer's score chains would be sub-waves inside it).
        var layers: [16]TraceOps.Freed = undefined;
        var n_layers: usize = 0;
        var n_sub: usize = 0;
        var prev = first;
        for (g.freed.items[waves0..]) |w| {
            try testing.expect(w.from >= first and w.to > w.from);
            if (w.from < prev) {
                // A later wave that contains the earlier ones: the layer closing over its sub-waves.
                while (n_layers > 0 and layers[n_layers - 1].from >= w.from) n_layers -= 1;
                layers[n_layers] = w;
                n_layers += 1;
            } else {
                layers[n_layers] = w;
                n_layers += 1;
            }
            prev = w.to;
        }
        for (g.freed.items[waves0..]) |w| {
            for (layers[0..n_layers]) |l| {
                if (w.from >= l.from and w.to <= l.to and (w.from != l.from or w.to != l.to)) {
                    n_sub += 1;
                    break;
                }
            }
        }
        try testing.expectEqual(@as(usize, m.c.n_layers), n_layers);
        // Decode / verify widths keep their score chains in the layer wave (prefill widths release them).
        try testing.expectEqual(@as(usize, 0), n_sub);
        prev = layers[n_layers - 1].to;
        // The final norm, the head and the taps' concat come after the last wave.
        try testing.expect(r.logits.? >= prev and r.main_hidden.? >= prev and r.hidden >= prev);
    }
}

test "dsv41 model: the MLX instantiation of the model analyses (host, nothing runs)" {
    try testing.expect(@TypeOf(&mlxSmoke) != void);
}

test "dsv41 model: a ring geometry below the widest forward or outside the tested box is refused at construction" {
    const m = try Mini.init();
    defer m.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const lookup: SpecLookup = .{ .g = &g, .spec = m.spec };
    // A tier built past the parser (which refuses these levers by name) meets the same box here. A verify margin under
    // the widest forward (scratch_rows, 8): a verify block would push read rows out of the ring.
    var tier = try routes.parse(&.{.{ "MTPLX_DSV41_WINDOW_RING", "1" }}, null);
    tier.kv.max_verify = TM.scratch_rows - 1;
    try testing.expectError(error.RingVerifyBelowForward, TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src));
    // A lever past the box the bill's ring tests cover.
    tier.kv.max_verify = TM.scratch_rows;
    tier.kv.headroom = routes.ring_lever_box.headroom_max + 1;
    try testing.expectError(error.RingLeverRange, TM.init(testing.allocator, &g, m.c, tier, &lookup, &m.src));
}
