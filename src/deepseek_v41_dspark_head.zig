//! The DSpark draft head (Python `deepseek_v41_dspark.DSparkHead`; K33 draft
//! compile when the routes set `draft_rows`). Three stages under `mtp.{0,1,2}`, each a V4.1
//! decoder block whose attention is a pure sliding window over the backbone's
//! committed main hiddens (`Cache`, seeded by `seedMain`) plus the block's own
//! draft rows, and whose MoE is its own 128-expert top-3 mxfp4 switch, resident
//! (`gather_qmm`). Stage 0 projects the target-layer hiddens (`main_proj`,
//! `main_norm`); the last stage's head autoregresses the markov bias over the
//! block and scores each draft (`confidence_head`).

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const graph = @import("deepseek_v41_graph.zig");
const mdl = @import("deepseek_v41_model.zig");
const xk = @import("exl3_kernels.zig");
const kr = @import("dsv41_kernel_routes.zig");
const draft_routes = @import("dsv41_draft_routes.zig");
const sdk = @import("sdk");
const xsc = sdk.expert.slot_cache;
const expert_io = sdk.expert.io;
const expert_policy = sdk.expert.policy;
const expert_stream = @import("expert_stream.zig");

/// Bytes of one DSpark head expert (gate, up and down, mxfp4 with one e8m0
/// scale per 32 weights): 18,800,640 on the bank of record, the stack of
/// record's per-expert saving (`MTP_PRUNED_BYTES / 201`).
pub fn expertBytes(c: *const v41.Config) u64 {
    return 3 * @as(u64, c.moe_intermediate_size) * c.hidden_size * 17 / 32;
}

/// Resident bytes a subset of the head's experts leaves out (0: the full head).
pub fn prunedBytes(c: *const v41.Config, subset: ?*const Subset) u64 {
    const s = subset orelse return 0;
    return s.pruned() * expertBytes(c);
}

/// Which head a run drafted with (receipts): every expert resident, or the
/// compact head of a pinned subset.
pub const Identity = union(enum) {
    full,
    compact: [32]u8,

    pub fn text(self: Identity, buf: *[72]u8) []const u8 {
        return switch (self) {
            .full => "full",
            .compact => |sha| std.fmt.bufPrint(buf, "compact:{s}", .{&std.fmt.bytesToHex(sha, .lower)}) catch unreachable,
        };
    }
};

pub fn Head(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const T = G.T;
        const Tr = graph.Trunk(G);
        const Q = graph.Q(T);

        /// A stage's routed experts stacked `[E, out, in]` (the `SwitchGLU`
        /// banks): gate = w1, up = w3, down = w2.
        pub const Experts = struct { w1: Q, w3: Q, w2: Q };
        /// `lut`: with a subset, each routed id's slot in the compact banks.
        pub const Stage = struct { w: graph.LayerW(T), experts: Experts, lut: ?T = null };

        /// Construction choices (default: the full head, every expert resident).
        pub const Options = struct {
            /// Keep only these experts per stage (a pinned subset file; the
            /// stack of record's compact head is one, a trace-derived ceiling).
            subset: ?*const Subset = null,
            /// The accepted trunk routes' registry: required when the routes bind DRAFTRC (C16).
            registry: ?*const xk.Registry = null,
            /// DRAFTCACHE: the stages' experts served from this cache's slot banks (`DraftCache`), none resident.
            cache: ?*DraftCache = null,
            /// HEAD_MODE mxfp8 on RCPROJ (the model's, over the quantized head the draft shares): required when the routes
            /// carry `rc_head_mxfp8`, and then the block's head pass runs on it (block_size <= 8 rows).
            head_mx: ?*const kr.HeadMx(G) = null,
            /// DRAFT_STAGED (exact: the same graph, only its commits move): each stage's outputs committed as soon as
            /// the stage is built, so the GPU runs stage s while the host builds the later stages and the head.
            staged_commit: bool = false,
        };

        const DP = kr.DraftProj(G);

        /// C16 DRAFTRC, one stage: the verify side's RC kernels over the stage's own modules.
        /// `attn_dt` is the stream dtype into the stage's attention: bf16 at stage 0 (the block's
        /// embedding), f32 after (the first HC post is f32: the attention output is). `wkv_main`
        /// and `main_proj` take main_x / the target's hidden at either dtype (bf16 at the verify
        /// rows, f32 from the prompt pass): the dtype route of a runtime phase.
        pub const StageRc = struct {
            attn_dt: ops.Dtype,
            wq_a: DP,
            wq_b: DP,
            wkv: DP,
            wkv_main: [2]DP,
            woa: DP,
            wo_b: DP,
            router: kr.Router(G),
            premix: [2]kr.Premix(G),
            shared: graph.SharedRc(G),

            fn init(g: *G, reg: *const xk.Registry, w: *const graph.LayerW(T), s: usize) !StageRc {
                const dt: ops.Dtype = if (s == 0) .bfloat16 else .float32;
                var wq_a = try DP.init(g, reg, .wq_a, dt, w.wq_a.w, w.wq_a.s, null);
                errdefer wq_a.deinit(g);
                var wq_b = try DP.init(g, reg, .wq_b, dt, w.wq_b.w, w.wq_b.s, null);
                errdefer wq_b.deinit(g);
                var wkv = try DP.init(g, reg, .wkv, dt, w.wkv.w, w.wkv.s, null);
                errdefer wkv.deinit(g);
                var km0 = try DP.init(g, reg, .wkv, .bfloat16, w.wkv.w, w.wkv.s, null);
                errdefer km0.deinit(g);
                var km1 = try DP.init(g, reg, .wkv, .float32, w.wkv.w, w.wkv.s, null);
                errdefer km1.deinit(g);
                var woa = try DP.init(g, reg, .woa, .float32, w.wo_a.w, w.wo_a.s, null);
                errdefer woa.deinit(g);
                var wo_b = try DP.init(g, reg, .wo_b, .float32, w.wo_b.w, w.wo_b.s, null);
                errdefer wo_b.deinit(g);
                var router = try kr.Router(G).init(g, reg, w.gate_w, w.gate_bias, null);
                errdefer router.deinit(g);
                var pa = try kr.Premix(G).init(g, reg, w.hc_attn_fn, null);
                errdefer pa.deinit(g);
                var pf = try kr.Premix(G).init(g, reg, w.hc_ffn_fn, null);
                errdefer pf.deinit(g);
                return .{ .attn_dt = dt, .wq_a = wq_a, .wq_b = wq_b, .wkv = wkv, .wkv_main = .{ km0, km1 }, .woa = woa, .wo_b = wo_b, .router = router, .premix = .{ pa, pf }, .shared = try graph.SharedRc(G).init(g, reg, .float32, w) };
            }

            fn deinit(self: *StageRc, g: *G) void {
                inline for (.{ &self.wq_a, &self.wq_b, &self.wkv, &self.wkv_main[0], &self.wkv_main[1], &self.woa, &self.wo_b }) |x| x.deinit(g);
                self.router.deinit(g);
                for (&self.premix) |*x| x.deinit(g);
                self.shared.deinit(g);
            }

            fn mainKv(self: *const StageRc, g: *G, main_x: T) !T {
                return self.wkv_main[@intFromBool(g.dtypeOf(main_x) == .float32)].linear(g, main_x);
            }
        };

        /// C16: the draft block's RC routes (weight-free ones shared by the stages).
        pub const DraftRc = struct {
            main_proj: [2]DP,
            sinkhorn: kr.Sinkhorn(G),
            tape_bf16: kr.HcTape(G),
            tape_f32: kr.HcTape(G),
            tape_mixed: kr.HcTapeMixed(G),
            stages: std.ArrayList(StageRc) = .empty,

            fn init(gpa: std.mem.Allocator, g: *G, reg: *const xk.Registry, head: *const Self) !*DraftRc {
                const c = &head.c;
                // Baked by the texts: D 5120, hc 4, 20 Sinkhorn iterations, the eps pair; the draft router's
                // 128-expert top-3 variant (`__n128_top3`: sqrt(softplus), normalised, x 1.5).
                if (c.hidden_size != 5120 or c.hc_mult != 4 or c.hc_sinkhorn_iters != 20 or @as(f32, @floatCast(c.rms_norm_eps)) != @as(f32, 1e-20) or @as(f32, @floatCast(c.hc_eps)) != @as(f32, 1e-6)) return error.DraftRcGeometry;
                if (head.mc.n_routed_experts != 128 or head.mc.n_experts_per_tok != 3 or !head.mc.norm_topk_prob or head.mc.routed_scaling_factor != 1.5) return error.DraftRcGeometry;
                const d = try gpa.create(DraftRc);
                errdefer gpa.destroy(d);
                d.main_proj[0] = try DP.init(g, reg, .main_proj, .bfloat16, head.main_proj.w, head.main_proj.s, null);
                errdefer d.main_proj[0].deinit(g);
                d.main_proj[1] = try DP.init(g, reg, .main_proj, .float32, head.main_proj.w, head.main_proj.s, null);
                errdefer d.main_proj[1].deinit(g);
                d.sinkhorn = try kr.Sinkhorn(G).init(g, reg);
                errdefer d.sinkhorn.deinit(g);
                d.tape_bf16 = try kr.HcTape(G).init(g, reg, .bfloat16, null);
                errdefer d.tape_bf16.deinit(g);
                d.tape_f32 = try kr.HcTape(G).init(g, reg, .float32, null);
                errdefer d.tape_f32.deinit(g);
                d.tape_mixed = try kr.HcTapeMixed(G).init(g, reg, null);
                errdefer d.tape_mixed.deinit(g);
                d.stages = .empty;
                errdefer {
                    for (d.stages.items) |*x| x.deinit(g);
                    d.stages.deinit(gpa);
                }
                try d.stages.ensureTotalCapacity(gpa, head.stages.len);
                for (head.stages, 0..) |*st, s| d.stages.appendAssumeCapacity(try StageRc.init(g, reg, &st.w, s));
                return d;
            }

            fn deinit(self: *DraftRc, gpa: std.mem.Allocator, g: *G) void {
                for (self.stages.items) |*x| x.deinit(g);
                self.stages.deinit(gpa);
                self.tape_mixed.deinit(g);
                self.tape_f32.deinit(g);
                self.tape_bf16.deinit(g);
                self.sinkhorn.deinit(g);
                for (&self.main_proj) |*x| x.deinit(g);
                gpa.destroy(self);
            }

            /// Stage `s`'s layer view: stage 0's attention prep on the bf16 stream, its ffn prep the
            /// mixed call (f32 x over the bf16 residual); every other tape f32.
            fn layer(self: *const DraftRc, s: usize) graph.LayerKernels(G) {
                const sr = &self.stages.items[s];
                return .{
                    .sinkhorn = &self.sinkhorn,
                    .router = &sr.router,
                    .premix_attn = &sr.premix[0],
                    .premix_ffn = &sr.premix[1],
                    .tape = &self.tape_f32,
                    .tape_attn = if (s == 0) &self.tape_bf16 else &self.tape_f32,
                    .tape_mixed = if (s == 0) &self.tape_mixed else null,
                    .shared = &sr.shared,
                };
            }
        };

        /// `DSparkStageCache`: the last `window` post-RoPE main-KV rows of a
        /// stage and the count of main tokens seen.
        pub const Cache = struct {
            window: ?T = null,
            offset: u32 = 0,

            pub fn deinit(self: *Cache, g: *G) void {
                if (self.window) |w| g.release(w);
                self.* = .{};
            }

            /// `append_main`: keep the last `size` rows, advance the offset.
            fn appendMain(self: *Cache, g: *G, main_kv: T, size: u32) !void {
                const next = try self.appended(g, main_kv, size);
                if (self.window) |w| g.release(w);
                self.* = next;
            }

            /// `appendMain`'s window and offset as a new cache (its window kept); this one unchanged.
            fn appended(self: *const Cache, g: *G, main_kv: T, size: u32) !Cache {
                var all = if (self.window) |w| try g.concat(&.{ w, main_kv }, 1) else main_kv;
                const s = g.shapeOf(all);
                const rows = s.dim(1);
                if (rows > size) all = try g.slice(all, &.{ 0, rows - @as(c_int, @intCast(size)), 0 }, s.slice(), &.{ 1, 1, 1 });
                return .{ .window = g.keep(all), .offset = self.offset + @as(u32, @intCast(g.shapeOf(main_kv).dim(1))) };
            }
        };

        pub const Draft = struct {
            /// The block's drafted ids `[1, block_size]` (uint32).
            ids: T,
            /// The draft logits `[1, block_size, vocab]` f32.
            logits: T,
            /// `confidence_head` scores `[1, block_size]` f32 (before the sigmoid).
            conf: T,
            /// Profile builds with the draft routes active: each stage's routed ids, kept (`draft_routes.take`).
            stage_ids: draft_routes.StageIds(T) = draft_routes.noIds(T),
        };

        /// K33 (`MTPLX_DSV41_DRAFT_COMPILE`): at rows <= `draft_rows` the stages
        /// replay compiled regions (the backbone's K22 / K4 ones, the draft's
        /// main-KV, markov step and confidence), byte-identical to the eager
        /// bodies; 0 runs the eager bodies. The regions are built at `init`.
        pub const DraftKv = struct {
            pub const region: ops.Region = .draft_kv;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            /// in main_x, cos, sin, kv_norm, wkv (words, scales): `rope(rmsnorm(wkv(m)))`.
            pub fn run(g: *G, c: *const Ctx, in: []const T, out: []T) !void {
                const kv = try Tr.rmsnorm(g, try g.qmm(in[0], in[4], in[5], .mxfp8), in[3], c.rms_norm_eps);
                out[0] = try Tr.ropeLast(g, kv, .{ .cos = in[1], .sin = in[2] }, false);
            }
        };

        pub const MarkovStep = struct {
            pub const region: ops.Region = .markov_step;
            pub const Ctx = v41.Config;
            pub const n_out = 3;
            /// in token, base row, markov embed, markov head: out (logits row, embed, argmax).
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                const me = try Tr.embed(g, in[2], in[0]);
                const li = try g.add(in[1], try Tr.linear(g, me, in[3]));
                out[0] = li;
                out[1] = me;
                out[2] = try g.argmax(li, -1);
            }
        };

        pub const Confidence = struct {
            pub const region: ops.Region = .confidence;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            /// in hidden, markov embed, proj: `(concat(h, me).f32 @ w.f32.T)` without its last axis.
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                const h = try g.astype(try g.concat(&.{ in[0], in[1] }, -1), .float32);
                const conf = try g.matmul(h, try g.transpose(try g.astype(in[2], .float32)));
                const s = g.shapeOf(conf);
                out[0] = try g.reshape(conf, s.slice()[0 .. s.n - 1]);
            }
        };

        gpa: std.mem.Allocator,
        c: v41.Config,
        /// The stages' MoE shape: `n_routed_experts` / `n_experts_per_tok` of the DSpark head.
        mc: v41.Config,
        /// The head codec shared with the target (`Routes.head`: f32, bf16 or mxfp8).
        rt: graph.Routes,
        /// The stages' routes: K22 / K4 at rows <= `rt.draft_rows` when K33 is on.
        stage_rt: graph.Routes,
        stages: []Stage,
        main_proj: Q,
        main_norm: T,
        norm: T,
        markov_embed: T,
        markov_head: T,
        conf_proj: T,
        inv_swa: T,
        owned: std.ArrayList(T) = .empty,
        /// Host bytes of one block's input lookup (its ids, or its embedding rows
        /// once the target's table is on the host), allocated once.
        block_scratch: []u8 = &.{},
        identity: Identity = .full,
        /// Resident bytes the subset left out (0 for the full head).
        pruned_bytes: u64 = 0,
        /// C16's routes (`rt.rc_draft`), built at `initWith`.
        rc: ?*DraftRc = null,
        /// `Options.head_mx`, bound when the routes carry `rc_head_mxfp8`.
        head_mx: ?*const kr.HeadMx(G) = null,
        /// DRAFTCACHE (`Options.cache`; borrowed): the stages route through its slots.
        cache: ?*DraftCache = null,
        /// A built stage's commit, bound at construction (`Options.staged_commit`): none (the stock route: the block's
        /// one eval commits every stage) or its outputs committed at once (DRAFT_STAGED).
        stage_commit: *const StageCommitFn = stageCommitNone,

        const StageCommitFn = fn (g: *G, outs: []const T) anyerror!void;

        fn stageCommitNone(_: *G, _: []const T) !void {}

        fn stageCommitAsync(g: *G, outs: []const T) !void {
            try g.asyncEval(outs);
        }

        /// DRAFT_STAGED as bound at construction.
        pub fn stagedCommit(self: *const Self) bool {
            return self.stage_commit == &stageCommitAsync;
        }

        /// Binds the residents once and stacks each stage's experts into its
        /// switch banks. `rt` carries the head codec the draft head shares with the target.
        pub fn init(gpa: std.mem.Allocator, g: *G, c: v41.Config, rt: graph.Routes, lookup: anytype) !*Self {
            return initWith(gpa, g, c, rt, lookup, .{});
        }

        /// `init` with a subset: each stage stacks only its kept experts (in
        /// ascending order: slot `i` is kept expert `i`) and maps every routed id
        /// through its `lut` (an id outside the subset takes slot 0, as the
        /// stack of record's compact head does); a dropping lookup forgets
        /// every per-expert array, so a left-out expert is never read.
        pub fn initWith(gpa: std.mem.Allocator, g: *G, c: v41.Config, rt: graph.Routes, lookup: anytype, opts: Options) !*Self {
            const ds = c.dspark;
            if (ds.n_stages == 0 or ds.block_size == 0) return error.NoDsparkHead;
            if (opts.subset) |sub| if (sub.n_experts != ds.n_routed_experts or sub.selected.len != ds.n_stages) return error.SubsetGeometry;
            if (opts.cache) |dc| if (opts.subset != null or dc.n_stages != ds.n_stages or dc.n_experts != ds.n_routed_experts) return error.DraftCacheGeometry;
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            var mc = c;
            mc.n_routed_experts = ds.n_routed_experts;
            mc.n_experts_per_tok = ds.n_experts_per_tok;
            const stage_rt: graph.Routes = if (rt.draft_rows > 0) .{ .attn_rows = rt.draft_rows, .hc_rows = rt.draft_rows } else .{};
            self.* = .{ .gpa = gpa, .c = c, .mc = mc, .rt = rt, .stage_rt = stage_rt, .stages = &.{}, .main_proj = undefined, .main_norm = undefined, .norm = undefined, .markov_embed = undefined, .markov_head = undefined, .conf_proj = undefined, .inv_swa = undefined };
            if (opts.staged_commit) self.stage_commit = stageCommitAsync;
            errdefer self.deinitOwned(g);
            self.stages = try gpa.alloc(Stage, ds.n_stages);
            errdefer gpa.free(self.stages);
            const M = mdl.Model(G);
            var b: [160]u8 = undefined;
            if (ds.n_routed_experts > 512) return error.TooManyExperts;
            for (self.stages, 0..) |*st, s| {
                st.* = .{ .w = try M.bindBlock(lookup, "mtp", c.layers[c.n_layers + s], @intCast(s)), .experts = undefined };
                // W97 on the draft's attention too (DSparkAttention inherits `_o_lora_dense_weight`).
                if (rt.wo_a_f32) st.w.wo_a_dense = try self.own(g, try Tr.woaDenseF32(g, &self.c, st.w.wo_a));
                const first_owned = self.owned.items.len;
                var all: [512]u16 = undefined;
                for (all[0..ds.n_routed_experts], 0..) |*e, i| e.* = @intCast(i);
                const kept: []const u16 = if (opts.subset) |sub| sub.selected[s] else all[0..ds.n_routed_experts];
                // DRAFTCACHE: the stage's banks are the cache's slot arrays; the per-expert arrays are dropped unread.
                if (opts.cache) |dc| {
                    inline for (.{ "w1", "w3", "w2" }, 0..) |name, pi| {
                        @field(st.experts, name) = .{ .w = try cacheBank(g, dc, s, 2 * pi), .s = try cacheBank(g, dc, s, 2 * pi + 1), .mode = .mxfp4 };
                        if (comptime canDrop(@TypeOf(lookup))) for (0..ds.n_routed_experts) |e| {
                            lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                            lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                        };
                    }
                } else inline for (.{ "w1", "w3", "w2" }) |name| {
                    var ws: [512]T = undefined;
                    var ss: [512]T = undefined;
                    for (kept, 0..) |e, i| {
                        ws[i] = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                        ss[i] = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                    }
                    const n = kept.len;
                    @field(st.experts, name) = .{ .w = try self.own(g, try g.stack(ws[0..n], 0)), .s = try self.own(g, try g.stack(ss[0..n], 0)), .mode = .mxfp4 };
                    // The stacks hold the kept arrays until they evaluate; a lookup that can
                    // forget them frees each stage's inputs once its stacks are built (and a
                    // left-out expert's before anything read it).
                    if (comptime canDrop(@TypeOf(lookup))) for (0..ds.n_routed_experts) |e| {
                        lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".weight", .{ s, e }));
                        lookup.drop(try std.fmt.bufPrint(&b, "mtp.{d}.ffn.experts.{d}." ++ name ++ ".scales", .{ s, e }));
                    };
                }
                if (opts.subset) |sub| {
                    var l: [512]i32 = undefined;
                    sub.lut(s, l[0..ds.n_routed_experts]);
                    st.lut = try self.own(g, try g.hostArray(std.mem.sliceAsBytes(l[0..ds.n_routed_experts]), &.{@intCast(ds.n_routed_experts)}, .int32));
                }
                try g.evalAll(self.owned.items[first_owned..]);
            }
            self.cache = opts.cache;
            if (opts.subset) |sub| {
                self.identity = .{ .compact = sub.sha256 };
                self.pruned_bytes = prunedBytes(&self.c, sub);
            }
            const last = ds.n_stages - 1;
            self.main_proj = .{ .w = try need(lookup, "mtp.0.main_proj.weight"), .s = try need(lookup, "mtp.0.main_proj.scales"), .mode = .mxfp8 };
            self.main_norm = try need(lookup, "mtp.0.main_norm.weight");
            self.norm = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.norm.weight", .{last}));
            self.markov_embed = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.markov_head.embed.weight", .{last}));
            self.markov_head = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.markov_head.head.weight", .{last}));
            self.conf_proj = try need(lookup, try std.fmt.bufPrint(&b, "mtp.{d}.confidence_head.proj.weight", .{last}));
            self.inv_swa = try self.own(g, try Tr.swaInvFreq(g, &self.c));
            self.block_scratch = try gpa.alloc(u8, @max(ds.block_size * @sizeOf(i32), @as(usize, ds.block_size) * c.hidden_size * 2) + 16);
            errdefer gpa.free(self.block_scratch);
            if (rt.rc_head_mxfp8) {
                self.head_mx = opts.head_mx orelse return error.DraftNeedsHeadMx;
                if (ds.block_size > kr.HeadMx(G).max_rows) return error.DraftHeadMxRows;
            }
            if (rt.rc_draft) {
                const reg = opts.registry orelse return error.DraftNeedsKernels;
                self.rc = DraftRc.init(gpa, g, reg, self) catch |e| return if (e == error.RouteInput) error.DraftRcGeometry else e;
            }
            errdefer if (self.rc) |d| d.deinit(gpa, g);
            if (rt.draft_rows > 0) {
                try Tr.prepareRegions(g, &self.mc, &self.stage_rt, false);
                inline for (.{ DraftKv, MarkovStep, Confidence }) |B| try g.prepareTape(B, &self.mc);
            }
            try g.evalAll(self.owned.items);
            return self;
        }

        fn canDrop(comptime L: type) bool {
            return switch (@typeInfo(L)) {
                .pointer => |p| @hasDecl(p.child, "drop"),
                else => false,
            };
        }

        /// Component `k` of stage `s`'s slot bank: the cache's MLX array (borrowed), or on the trace backend a leaf of its shape.
        fn cacheBank(g: *G, dc: *const DraftCache, s: usize, k: usize) !T {
            const grp = dc.geom.group_of[s];
            if (comptime G == ops.MlxOps) {
                if (dc.cache.memory != .mlx) return error.DraftCacheMemory;
                return dc.cache.arrays[grp][k];
            }
            var shape: [3]c_int = undefined;
            shape[0] = @intCast(dc.cache.geom.rows(grp));
            @memcpy(shape[1..3], &dc.geom.shapes[k]);
            return g.input(&shape, dc.geom.comps[k].dtype);
        }

        fn need(lookup: anytype, name: []const u8) !T {
            return lookup.get(name) orelse error.MissingWeight;
        }

        fn own(self: *Self, g: *G, x: T) !T {
            const k = g.keep(x);
            try self.owned.append(self.gpa, k);
            return k;
        }

        fn deinitOwned(self: *Self, g: *G) void {
            for (self.owned.items) |x| g.release(x);
            self.owned.deinit(self.gpa);
        }

        pub fn deinit(self: *Self, g: *G) void {
            if (self.rc) |d| d.deinit(self.gpa, g);
            self.deinitOwned(g);
            self.gpa.free(self.block_scratch);
            self.gpa.free(self.stages);
            self.gpa.destroy(self);
        }

        pub fn nStages(self: *const Self) usize {
            return self.stages.len;
        }

        /// Device bytes the head builds beyond the checkpoint's residents (the
        /// bill's resident term): W97's dense f32 wo_a per stage. Its expert
        /// stacks replace the per-expert arrays they drop (no net bytes).
        pub fn builtBytes(self: *const Self) u64 {
            return if (self.rt.wo_a_f32) @as(u64, self.stages.len) * graph.woaDenseBytes(&self.c) else 0;
        }

        pub fn blockSize(self: *const Self) u32 {
            return self.c.dspark.block_size;
        }

        /// `main_x = main_norm(main_proj(main_hidden))` (stage 0).
        fn mainProject(self: *const Self, g: *G, main_hidden: T) !T {
            const p = if (self.rc) |d| (if (rowsOf(g, main_hidden) <= graph.rc_max_rows) try d.main_proj[@intFromBool(g.dtypeOf(main_hidden) == .float32)].linear(g, main_hidden) else null) else null;
            return Tr.rmsnorm(g, p orelse try Tr.qlinear(g, main_hidden, self.main_proj), self.main_norm, self.c.rms_norm_eps);
        }

        fn rowsOf(g: *G, x: T) c_int {
            const s = g.shapeOf(x);
            return s.d[0] * s.d[1];
        }

        /// C16's stage routes at <= 8 rows of `x`, else null (the stock / K33 bodies).
        fn stageRc(self: *const Self, g: *G, x: T, s: usize) ?*const StageRc {
            const d = self.rc orelse return null;
            return if (rowsOf(g, x) <= graph.rc_max_rows) &d.stages.items[s] else null;
        }

        /// A stage's main KV: `rope(rmsnorm(wkv(main_x)))` at the main positions `[offset, offset + S)`.
        fn mainKv(self: *const Self, g: *G, st: *const Stage, s: usize, main_x: T, offset: u32) !T {
            const S = g.shapeOf(main_x).dim(1);
            const pos = try g.arange(@floatFromInt(offset), @floatFromInt(offset + @as(u32, @intCast(S))), 1, .int32);
            const cs = try Tr.cosSin(g, self.inv_swa, pos);
            const kv = if (self.stageRc(g, main_x, s)) |sr| try sr.mainKv(g, main_x) else try Tr.qlinear(g, main_x, st.w.wkv);
            return Tr.ropeLast(g, try Tr.rmsnorm(g, kv, st.w.kv_norm, self.c.rms_norm_eps), cs, false);
        }

        /// `seedMain`'s statements into `out` (each window kept), `caches` unchanged: DRAFT_AHEAD's seed for an outcome
        /// not yet decided. The caller installs `out` (releasing the old windows) or releases `out`'s windows.
        pub fn seedMainInto(self: *const Self, g: *G, main_hidden: T, caches: []const Cache, out: []Cache) !void {
            const main_x = try self.mainProject(g, main_hidden);
            for (self.stages, caches, out[0..caches.len], 0..) |*st, *cache, *o, s| {
                o.* = try cache.appended(g, try self.mainKv(g, st, s, main_x, cache.offset), self.c.window);
            }
        }

        /// DRAFT_AHEAD's block inputs, built before the primary is known and written in place before the block is
        /// committed (`Loop`'s ahead inputs): `raw` = what `Embed.of` gives for `[primary, noise, ...]`, `prev` = the
        /// markov chain's first id `[primary]` (int32).
        pub const BlockInput = struct { raw: T, prev: T };

        /// DRAFT_AHEAD's persistent inputs (kept; the caller releases both): `patch` = the block's ids `[1, n]` int32
        /// (a table embedding) or its embedded rows `[1, n, dim]` bf16 (host rows), every position the noise token's;
        /// `prev` = `[1]` int32. `aheadWrite` puts a primary at position 0 of both before the block is committed.
        pub const AheadIn = struct { patch: T, prev: T };

        pub fn aheadInputs(self: *const Self, g: *G, a: std.mem.Allocator, embed: mdl.Model(G).Embed) !AheadIn {
            const ds = self.c.dspark;
            var block_ids: [64]u32 = undefined;
            const ids = try self.blockIds(ds.noise_token_id, &block_ids);
            const n: c_int = @intCast(ids.len);
            const patch = switch (embed) {
                .table => blk: {
                    var v: [64]i32 = undefined;
                    for (v[0..ids.len], ids) |*d, s| d.* = @intCast(s);
                    break :blk try g.hostArray(std.mem.sliceAsBytes(v[0..ids.len]), &.{ 1, n }, .int32);
                },
                .rows => |r| blk: {
                    const buf = try a.alloc(u8, ids.len * @as(usize, r.dim) * 2);
                    defer a.free(buf);
                    try r.gatherRaw(ids, buf);
                    break :blk try g.hostArray(buf, &.{ 1, n, @intCast(self.c.hidden_size) }, .bfloat16);
                },
            };
            const z = [1]i32{@intCast(ds.noise_token_id)};
            const prev = try g.hostArray(std.mem.sliceAsBytes(&z), &.{1}, .int32);
            return .{ .patch = g.keep(patch), .prev = g.keep(prev) };
        }

        /// `Embed.of`'s result over the persistent `patch`: the table's gather of the ids, or the rows themselves.
        pub fn aheadInput(_: *const Self, g: *G, embed: mdl.Model(G).Embed, in: AheadIn) !BlockInput {
            return .{ .raw = switch (embed) {
                .table => |w| try Tr.embed(g, w, in.patch),
                .rows => in.patch,
            }, .prev = in.prev };
        }

        /// Writes `primary` into position 0 of the persistent inputs (its id, or its embedding row read from the host
        /// rows), in place: only while no committed command reads them (the last block that did was waited).
        pub fn aheadWrite(_: *const Self, g: *G, embed: mdl.Model(G).Embed, in: AheadIn, primary: u32) !void {
            const id: i32 = @intCast(primary);
            switch (embed) {
                .table => @memcpy((try g.hostBytes(in.patch))[0..4], std.mem.asBytes(&id)),
                .rows => |r| try r.gatherRaw(&.{primary}, (try g.hostBytes(in.patch))[0 .. @as(usize, r.dim) * 2]),
            }
            @memcpy((try g.hostBytes(in.prev))[0..4], std.mem.asBytes(&id));
        }

        /// `draftBlock` over `in` (DRAFT_AHEAD), every stage left uncommitted (the inputs are written later).
        pub fn draftBlockAhead(self: *const Self, g: *G, main_hidden: T, in: BlockInput, caches: []const Cache, head_w: Tr.HeadW) !Draft {
            const main_x = try self.mainProject(g, main_hidden);
            return self.draftBlockOn(g, main_x, in.raw, 0, in.prev, caches, head_w, stageCommitNone);
        }

        /// `seed_main`: every stage appends the committed rows' main KV to its window.
        pub fn seedMain(self: *const Self, g: *G, main_hidden: T, caches: []Cache) !void {
            const main_x = try self.mainProject(g, main_hidden);
            for (self.stages, caches, 0..) |*st, *cache, s| {
                try cache.appendMain(g, try self.mainKv(g, st, s, main_x, cache.offset), self.c.window);
            }
        }

        /// `DSparkAttention.__call__` (draft): the draft queries over the window,
        /// this cycle's main KV (not appended) and the block's own draft KV.
        fn attention(self: *const Self, g: *G, st: *const Stage, s: usize, x: T, main_x: T, cache: *const Cache) !T {
            const c = &self.c;
            const w = &st.w;
            const S = g.shapeOf(main_x).dim(1);
            const main_kv = if (self.stageRc(g, main_x, s) != null) try self.mainKv(g, st, s, main_x, cache.offset) else if (g.shapeOf(main_x).dim(0) * S <= @as(c_int, @intCast(self.rt.draft_rows))) blk: {
                const mpos = try g.arange(@floatFromInt(cache.offset), @floatFromInt(cache.offset + @as(u32, @intCast(S))), 1, .int32);
                const mcs = try Tr.cosSin(g, self.inv_swa, mpos);
                var o: [1]T = undefined;
                try g.tape(DraftKv, &self.mc, &.{ main_x, mcs.cos, mcs.sin, w.kv_norm, w.wkv.w, w.wkv.s }, &o);
                break :blk o[0];
            } else try self.mainKv(g, st, s, main_x, cache.offset);
            var win = if (cache.window) |wd| try g.concat(&.{ wd, main_kv }, 1) else main_kv;
            const ws = g.shapeOf(win);
            const wr = ws.dim(1);
            const size: c_int = @intCast(c.window);
            if (wr > size) win = try g.slice(win, &.{ 0, wr - size, 0 }, ws.slice(), &.{ 1, 1, 1 });
            const wp = g.shapeOf(win).dim(1);
            const sx = g.shapeOf(x);
            const b = sx.d[0];
            const t = sx.d[1];
            const base = cache.offset + @as(u32, @intCast(g.shapeOf(main_x).dim(1)));
            const dpos = try g.arange(@floatFromInt(base), @floatFromInt(base + @as(u32, @intCast(t))), 1, .int32);
            const cs = try Tr.cosSin(g, self.inv_swa, dpos);
            // C16 at <= 8 rows: the projections on the draft FMA kernel at the stage's stream dtype
            // (the glue statements the stock ones), ahead of K33's tapes.
            const rc = self.stageRc(g, x, s);
            const compiled = rc == null and b * t <= @as(c_int, @intCast(self.rt.draft_rows));
            var q: T = undefined;
            var kv: T = undefined;
            if (rc) |sr| {
                if (std.debug.runtime_safety) std.debug.assert(g.dtypeOf(x) == sr.attn_dt);
                const qr = try Tr.rmsnorm(g, try sr.wq_a.linear(g, x), w.q_norm, c.rms_norm_eps);
                q = try Tr.ropeLast(g, try g.reshape(try sr.wq_b.linear(g, qr), &.{ b, t, @intCast(c.n_heads), @intCast(c.head_dim) }), cs, false);
                kv = try Tr.ropeLast(g, try Tr.rmsnorm(g, try sr.wkv.linear(g, x), w.kv_norm, c.rms_norm_eps), cs, false);
            } else if (compiled) {
                var o3: [3]T = undefined;
                try g.tape(Tr.QkvPrep, &self.mc, &.{ x, cs.cos, cs.sin, w.q_norm, w.kv_norm, w.wq_a.w, w.wq_a.s, w.wq_b.w, w.wq_b.s, w.wkv.w, w.wkv.s }, &o3);
                q = o3[0];
                kv = o3[2];
            } else {
                const qr = try Tr.rmsnorm(g, try Tr.qlinear(g, x, w.wq_a), w.q_norm, c.rms_norm_eps);
                q = try Tr.ropeLast(g, try g.reshape(try Tr.qlinear(g, qr, w.wq_b), &.{ b, t, @intCast(c.n_heads), @intCast(c.head_dim) }), cs, false);
                kv = try Tr.ropeLast(g, try Tr.rmsnorm(g, try Tr.qlinear(g, x, w.wkv), w.kv_norm, c.rms_norm_eps), cs, false);
            }
            const keys = try g.concat(&.{ win, kv }, 1);
            const attend = try g.ones(&.{ b, t, wp + t }, .bool_);
            const o = try Tr.sparseAttend(g, c, w, q, keys, attend);
            if (rc) |sr| {
                // The o-LoRA on the packed wo_a (f32 x: the attention output), then wo_b.
                const o1 = try g.astype(try Tr.ropeLast(g, o, cs, true), .float32);
                const o2 = try sr.woa.call(g, try g.reshape(o1, &.{ b * t, -1 }));
                return sr.wo_b.linear(g, try g.reshape(o2, &.{ b, t, -1 }));
            }
            const w_ol = try Tr.woaDense(g, c, w);
            if (compiled) {
                var o1: [1]T = undefined;
                try g.tape(Tr.OutPrep, &self.mc, &.{ o, cs.cos, cs.sin, w_ol, w.wo_b.w, w.wo_b.s }, &o1);
                return o1[0];
            }
            return Tr.outProj(g, c, o, cs, w_ol, w.wo_b, false);
        }

        /// The stage's resident `SwitchGLU` with `ClampedSwiGLU` (mlx_lm arg order:
        /// the activation gets (up, gate)); unsorted below 64 routed ids.
        pub const Resident = struct {
            ex: *const Experts,
            limit: f64,
            /// The compact banks' slot of every routed id (a subset head).
            lut: ?T = null,
            /// Profile builds: where the stage's routed ids go (before the lut), when the draft routes are active.
            capture: Capture = no_capture,

            pub fn at(self: Resident, _: u32) Resident {
                return self;
            }

            pub fn routed(self: Resident, g: *G, xf: T, routed_ids: T) !T {
                const si = g.shapeOf(routed_ids);
                if (si.numel() >= 64) return error.SortedSwitchNotPorted;
                keepIds(self.capture, routed_ids);
                // `_CompactMTPExpertSwitch.__call__`: `mapped = mx.take(LUT, indices)`.
                const indices = if (self.lut) |l| try g.take(l, routed_ids, 0) else routed_ids;
                return switchGlu(g, self.ex, self.limit, xf, indices);
            }
        };

        /// The `SwitchGLU` with `ClampedSwiGLU` over `ex` at `indices` [rows, k] (unsorted).
        fn switchGlu(g: *G, ex: *const Experts, limit: f64, xf: T, indices: T) !T {
            const s = g.shapeOf(xf);
            const si = g.shapeOf(indices);
            const x = try g.reshape(xf, &.{ s.d[0], 1, 1, s.d[1] });
            var up = try g.gatherQmm(x, ex.w3.w, ex.w3.s, indices, .mxfp4);
            var gate = try g.gatherQmm(x, ex.w1.w, ex.w1.s, indices, .mxfp4);
            if (limit > 0) {
                const dt = g.dtypeOf(up);
                up = try g.clip(up, try g.scalar(-limit, dt), try g.scalar(limit, dt));
                gate = try g.minimum(gate, try g.scalar(limit, g.dtypeOf(gate)));
            }
            const h = try g.mul(try g.silu(gate), up);
            const y = try g.gatherQmm(h, ex.w2.w, ex.w2.s, indices, .mxfp4);
            return g.reshape(y, &.{ si.d[0], si.d[1], s.d[1] });
        }

        /// DRAFTCACHE: the stage's routed ids read on the host (the routing barrier), planned on the cache (every
        /// miss read into its slot before the call returns), and the switch over the slot banks at those slots.
        pub const Cached = struct {
            ex: *const Experts,
            limit: f64,
            dc: *DraftCache,
            /// The stage's group in the cache and its experts' id offset there (`DraftGeometry`, fixed at construction).
            group: u32,
            offset: u16,
            /// Profile builds: as `Resident.capture`.
            capture: Capture = no_capture,

            pub fn at(self: Cached, _: u32) Cached {
                return self;
            }

            pub fn routed(self: Cached, g: *G, xf: T, routed_ids: T) !T {
                const si = g.shapeOf(routed_ids);
                const n: usize = @intCast(si.numel());
                if (n > expert_policy.max_route_ids) return error.DraftCacheRouteWidth;
                keepIds(self.capture, routed_ids);
                var id_buf: [expert_policy.max_route_ids]u16 = undefined;
                // The shared pool's bill (H + 15 rows) rests on this barrier: it completes the previous stage's gathers before any row is reused.
                _ = try g.hostIds(routed_ids, id_buf[0..n]);
                for (id_buf[0..n]) |*e| e.* += self.offset;
                var slots: [expert_policy.max_route_ids]u32 = undefined;
                try self.dc.cache.route(self.group, id_buf[0..n], slots[0..n]);
                var sl: [expert_policy.max_route_ids]i32 = undefined;
                for (sl[0..n], slots[0..n]) |*d, v| d.* = @intCast(v);
                const indices = try g.hostArray(std.mem.sliceAsBytes(sl[0..n]), si.slice(), .int32);
                return switchGlu(g, self.ex, self.limit, xf, indices);
            }
        };

        /// The stage's installed expert source: resident banks, or DRAFTCACHE's.
        pub const Source = union(enum) {
            resident: Resident,
            cached: Cached,

            pub fn at(self: Source, _: u32) Source {
                return self;
            }

            pub fn routed(self: Source, g: *G, xf: T, routed_ids: T) !T {
                return switch (self) {
                    inline else => |x| x.routed(g, xf, routed_ids),
                };
            }
        };

        /// Profile builds: a stage's routed-ids slot (`Draft.stage_ids`); void elsewhere.
        const Capture = if (draft_routes.enabled) ?*?T else void;
        const no_capture: Capture = if (draft_routes.enabled) null else {};

        fn keepIds(capture: Capture, routed_ids: T) void {
            if (comptime draft_routes.enabled) if (capture) |c| {
                c.* = routed_ids;
            };
        }

        fn sourceOf(self: *const Self, st: *const Stage, s: usize, capture: Capture) Source {
            if (self.cache) |dc| return .{ .cached = .{ .ex = &st.experts, .limit = self.c.swiglu_limit, .dc = dc, .group = dc.geom.group_of[s], .offset = @intCast(dc.geom.offset_of[s]), .capture = capture } };
            return .{ .resident = .{ .ex = &st.experts, .limit = self.c.swiglu_limit, .lut = st.lut, .capture = capture } };
        }

        /// `DSparkBlock.__call__` (draft): HC attention prep, the draft
        /// attention, HC ffn prep, the stage MoE, HC post.
        fn stage(self: *const Self, g: *G, st: *const Stage, s: usize, h: T, pre_mix: T, main_x: T, cache: *const Cache, capture: Capture) !Tr.Out {
            const c = &self.c;
            const w = &st.w;
            if (self.rc) |d| if (self.stageRc(g, h, s) != null) {
                // C16: the HC tapes, premix, Sinkhorn, router and shared expert of the stage's view.
                const lk = d.layer(s);
                const a = try Tr.hcAttnPrep(g, c, lk, h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm);
                const ao = try self.attention(g, st, s, a[0], main_x, cache);
                const f = try Tr.hcFfnPrep(g, c, lk, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm);
                const mo = try Tr.moe(g, graph.NoProbe{}, &self.mc, &self.stage_rt, lk, w, f[0], self.sourceOf(st, s, capture));
                return .{ .h = try Tr.hcPostRoute(g, c, &self.stage_rt, lk, mo, f[1], f[2], f[3]), .pre_mix = f[4] };
            };
            const sh = g.shapeOf(h);
            const use = sh.d[0] * sh.d[1] <= @as(c_int, @intCast(self.stage_rt.hc_rows));
            var a: [4]T = undefined;
            if (use) {
                try g.tape(Tr.HcAttnPrep, &self.mc, &.{ h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm }, &a);
            } else a = try Tr.hcAttnPrep(g, c, .{}, h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm);
            const ao = try self.attention(g, st, s, a[0], main_x, cache);
            var f: [5]T = undefined;
            if (use) {
                try g.tape(Tr.HcFfnPrep, &self.mc, &.{ ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm }, &f);
            } else f = try Tr.hcFfnPrep(g, c, .{}, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm);
            const mo = try Tr.moe(g, graph.NoProbe{}, &self.mc, &self.stage_rt, .{}, w, f[0], self.sourceOf(st, s, capture));
            if (use) {
                var o: [1]T = undefined;
                try g.tape(Tr.HcPost, &self.mc, &.{ mo, f[1], f[2], f[3] }, &o);
                return .{ .h = o[0], .pre_mix = f[4] };
            }
            return .{ .h = try Tr.hcPost(g, mo, f[1], f[2], f[3]), .pre_mix = f[4] };
        }

        /// `draft_block`: embed `[primary, noise, ...]`, the stages (threading
        /// `main_x`), then `forward_head`: the base logits, the markov
        /// autoregression (greedy) and the confidence scores.
        pub fn draftBlock(self: *const Self, g: *G, main_hidden: T, primary: u32, caches: []const Cache, embed: mdl.Model(G).Embed, head_w: Tr.HeadW) !Draft {
            const c = &self.c;
            const main_x = try self.mainProject(g, main_hidden);
            var block_ids: [64]u32 = undefined;
            const ids = try self.blockIds(primary, &block_ids);
            var fba = std.heap.FixedBufferAllocator.init(self.block_scratch);
            return self.draftBlockOn(g, main_x, try embed.of(g, fba.allocator(), ids, c.hidden_size), primary, null, caches, head_w, self.stage_commit);
        }

        /// The block's token ids `[primary, noise, ...]` (`block_size`).
        pub fn blockIds(self: *const Self, primary: u32, out: *[64]u32) ![]const u32 {
            const ds = self.c.dspark;
            if (ds.block_size > out.len) return error.BlockTooWide;
            out[0] = primary;
            for (out[1..ds.block_size]) |*d| d.* = @intCast(ds.noise_token_id);
            return out[0..ds.block_size];
        }

        /// `draftBlock`'s body over the projected main row and the embedded block (`raw`, `Embed.of`'s result); the
        /// markov chain's first id is `prev` when given, else `[primary]` from the host at its place in the stock body.
        fn draftBlockOn(self: *const Self, g: *G, main_x: T, raw: T, primary: u32, prev_in: ?T, caches: []const Cache, head_w: Tr.HeadW, commit: *const StageCommitFn) !Draft {
            const c = &self.c;
            const ds = c.dspark;
            const bs: c_int = @intCast(ds.block_size);
            var ids: [64]i32 = undefined;
            if (ds.block_size > ids.len) return error.BlockTooWide;
            ids[0] = @intCast(primary);
            const e = try Tr.expandEmbedding(g, c, raw);
            var cur: Tr.Out = e;
            // One wave per stage, as the trunk's layers (`Tr.Carry`).
            var carry: Tr.Carry = .{};
            errdefer carry.release(g);
            var stage_ids = draft_routes.noIds(T);
            errdefer draft_routes.drop(G, g, &stage_ids);
            for (self.stages, caches, 0..) |*st, *cache, s| {
                const wave = g.mark();
                const capture: Capture = if (comptime draft_routes.enabled) (if (draft_routes.active and s < stage_ids.len) &stage_ids[s] else null) else {};
                cur = try self.stage(g, st, s, cur.h, cur.pre_mix, main_x, cache, capture);
                carry.persist(g, &cur.h, &cur.pre_mix, null);
                // DRAFT_STAGED: the stage on the GPU while the host builds the rest (stock: nothing here).
                try commit(g, &.{ cur.h, cur.pre_mix });
                // the stage's routed ids outlive its wave (read after the block's eval)
                if (comptime draft_routes.enabled) if (s < stage_ids.len) if (stage_ids[s]) |x| {
                    stage_ids[s] = g.keep(x);
                };
                g.resetTo(wave);
            }
            // forward_head
            const x = try Tr.hcPre(g, cur.h, cur.pre_mix);
            carry.release(g);
            const hn = try Tr.rmsnorm(g, x, self.norm, c.rms_norm_eps);
            const base = if (self.head_mx) |hm| try Tr.headMx(g, hm, hn) else try Tr.head(g, &self.rt, hn, head_w);
            const vocab = g.shapeOf(base).dim(-1);
            var prev = prev_in orelse try g.hostArray(std.mem.sliceAsBytes(ids[0..1]), &.{1}, .int32);
            var outs: [64]T = undefined;
            var logit_cols: [64]T = undefined;
            var embeds: [64]T = undefined;
            const n: usize = ds.block_size;
            const compiled = bs <= @as(c_int, @intCast(self.rt.draft_rows));
            for (0..n) |i| {
                const row = try g.reshape(try g.slice(base, &.{ 0, @intCast(i), 0 }, &.{ 1, @intCast(i + 1), vocab }, &.{ 1, 1, 1 }), &.{ 1, vocab });
                if (compiled) {
                    var o3: [3]T = undefined;
                    try g.tape(MarkovStep, &self.mc, &.{ prev, row, self.markov_embed, self.markov_head }, &o3);
                    logit_cols[i] = o3[0];
                    embeds[i] = o3[1];
                    prev = o3[2];
                } else {
                    const me = try Tr.embed(g, self.markov_embed, prev);
                    const li = try g.add(row, try Tr.linear(g, me, self.markov_head));
                    logit_cols[i] = li;
                    embeds[i] = me;
                    prev = try g.argmax(li, -1);
                }
                outs[i] = prev;
            }
            const conf = if (compiled) blk: {
                // The markov steps' own embeds (each step's gather, not a second one).
                var o: [1]T = undefined;
                try g.tape(Confidence, &self.mc, &.{ x, try g.stack(embeds[0..n], 1), self.conf_proj }, &o);
                break :blk o[0];
            } else blk: {
                const markov = try g.stack(embeds[0..n], 1);
                const hcat = try g.astype(try g.concat(&.{ x, markov }, -1), .float32);
                break :blk try g.matmul(hcat, try g.transpose(try g.astype(self.conf_proj, .float32)));
            };
            return .{
                .ids = try g.stack(outs[0..n], 1),
                .logits = try g.stack(logit_cols[0..n], 1),
                .conf = try g.reshape(conf, &.{ 1, bs }),
                .stage_ids = stage_ids,
            };
        }
    };
}

// ── DRAFTCACHE: the stages' experts behind an exact adaptive cache ──
//
// Each stage keeps `Hs` persistent slots (the hot set's, filled by the streamer's decode policy online) and one
// block's distinct ids of transient slots; the 2,304 per-expert tensors stay in their shards and are read past the
// page cache by the streamer's read pool into the slot rows on a miss. The drafts are those of the resident head:
// same ids, same bytes, the same unsorted `gather_qmm`, only the bank row differs.

/// A record's parts in read order: (w1, w3, w2) x (weight, scales), each projection one pool job.
pub const draft_parts = [_][]const u8{ "w1.weight", "w1.scales", "w3.weight", "w3.scales", "w2.weight", "w2.scales" };
pub const max_draft_stages = 8;
/// The aux ring the served route reserves on the stream's pool (`expert_io.Options.aux_tickets`): two tickets per pool
/// job; a read runs in batches of the ring's size, one batch waited before the next.
pub const draft_aux_tickets: u32 = 192;

/// The hot set's persistent slots per stage: `hot` split as evenly as possible, the first stages taking the remainder.
pub fn hotSplit(hot: u32, n_stages: u32, out: []u32) void {
    for (out[0..n_stages], 0..) |*o, s| o.* = hot / n_stages + @intFromBool(s < hot % n_stages);
}

/// Transient slots per stage: the most distinct ids one draft block routes (block x top-k, at most every expert).
pub fn draftTransient(c: *const v41.Config) u32 {
    return @min(c.dspark.block_size * c.dspark.n_experts_per_tok, c.dspark.n_routed_experts);
}

/// The per-row shape of each part: w1 / w3 [I, H / 8] u32 and [I, H / 32] u8, w2 [H, I / 8] and [H, I / 32].
fn partShape(c: *const v41.Config, k: usize) [2]c_int {
    const I: c_int = @intCast(c.moe_intermediate_size);
    const Hd: c_int = @intCast(c.hidden_size);
    const out_dim = if (k < 4) I else Hd;
    const in_dim = if (k < 4) Hd else I;
    return .{ out_dim, if (k % 2 == 0) @divExact(in_dim, 8) else @divExact(in_dim, 32) };
}

/// The slot pool's form: one bank per stage (the hot count split evenly), or one bank all stages share (one policy
/// over every stage's experts, global id = stage x experts + expert; one block's transient rows serve every stage).
pub const DraftPool = enum { per_stage, shared };

/// The cache's geometry at `hot` (owned by `DraftCache`, or the caller's for a bill). Refuses by name when a group's
/// slots would hold every expert it serves (no byte saved).
pub const DraftGeometry = struct {
    comps: [draft_parts.len]xsc.Component = undefined,
    shapes: [draft_parts.len][2]c_int = undefined,
    caps: [max_draft_stages]u32 = undefined,
    pool: DraftPool = .per_stage,
    /// The residency policy (construction-time route; shipped by default).
    policy: xsc.PolicyKind = .shipped,
    n_stages: u32 = 0,
    /// The cache's groups: one per stage, or one shared.
    n_groups: u32 = 0,
    /// Experts per group: a stage's, or every stage's when shared.
    n_experts: u32 = 0,
    transient: u32 = 0,
    /// Each stage's group and the offset of its expert 0 in that group's ids.
    group_of: [max_draft_stages]u32 = undefined,
    offset_of: [max_draft_stages]u32 = undefined,

    pub fn of(c: *const v41.Config, hot: u32, pool: DraftPool) error{DraftCacheGeometry}!DraftGeometry {
        const ds = c.dspark;
        if (ds.n_stages == 0 or ds.n_stages > max_draft_stages) return error.DraftCacheGeometry;
        var d: DraftGeometry = .{ .pool = pool, .n_stages = ds.n_stages, .transient = draftTransient(c) };
        switch (pool) {
            .per_stage => {
                d.n_groups = ds.n_stages;
                d.n_experts = ds.n_routed_experts;
                hotSplit(hot, ds.n_stages, &d.caps);
                for (0..ds.n_stages) |st| {
                    d.group_of[st] = @intCast(st);
                    d.offset_of[st] = 0;
                }
            },
            .shared => {
                d.n_groups = 1;
                d.n_experts = ds.n_stages * ds.n_routed_experts;
                d.caps[0] = hot;
                for (0..ds.n_stages) |st| {
                    d.group_of[st] = 0;
                    d.offset_of[st] = @intCast(st * ds.n_routed_experts);
                }
            },
        }
        if (d.n_experts > expert_policy.no_expert) return error.DraftCacheGeometry;
        for (d.caps[0..d.n_groups]) |cap| if (cap + d.transient >= d.n_experts) return error.DraftCacheGeometry;
        for (&d.shapes, 0..) |*sh, k| sh.* = partShape(c, k);
        return d;
    }

    /// Point the components at this value's shapes (after it reached its final address).
    pub fn geometry(d: *DraftGeometry) xsc.Geometry {
        for (&d.comps, &d.shapes, 0..) |*cp, *sh, k| cp.* = .{
            .bytes = @as(u64, @intCast(sh[0])) * @as(u64, @intCast(sh[1])) * @as(u64, if (k % 2 == 0) 4 else 1),
            .shape = sh,
            .dtype = if (k % 2 == 0) .uint32 else .uint8,
        };
        return .{ .n_experts = d.n_experts, .policy = d.policy, .components = &d.comps, .capacity = d.caps[0..d.n_groups], .transient = d.transient };
    }

    /// The bill's term: every group's slot arrays, each rounded to the allocator's page.
    pub fn billBytes(d: *DraftGeometry) u64 {
        return d.geometry().billBytes();
    }
};

/// The bill's DRAFTCACHE term at `hot` in `pool`'s form (replacing the resident experts).
pub fn draftCacheBytes(c: *const v41.Config, hot: u32, pool: DraftPool) error{DraftCacheGeometry}!u64 {
    var d = try DraftGeometry.of(c, hot, pool);
    return d.billBytes();
}

/// Every part of stages [0, n_stages) placed at its checkpoint tensor in `cache` (stage s's expert e at its group's id
/// offset + e, `geom`), each shard opened once past the page cache; a tensor of another dtype or shape than the
/// geometry's is refused by name.
pub fn placeParts(cache: *xsc.Cache, a: std.mem.Allocator, ck: *const v41.Checkpoint, geom: *const DraftGeometry, n_stages: u32, n_stage_experts: u32) !void {
    const shapes = &geom.shapes;
    var file_of: [64]?u16 = @splat(null);
    var name: [96]u8 = undefined;
    for (0..n_stages) |s| for (0..n_stage_experts) |e| for (draft_parts, 0..) |part, k| {
        const n = try std.fmt.bufPrint(&name, "mtp.{d}.ffn.experts.{d}.{s}", .{ s, e, part });
        const t = ck.tensors.get(n) orelse return error.MissingWeight;
        const want: v41.StDtype = if (k % 2 == 0) .U32 else .U8;
        const sh = shapes[k];
        if (t.dtype != want or t.rank != 2 or t.shape[0] != @as(u64, @intCast(sh[0])) or t.shape[1] != @as(u64, @intCast(sh[1]))) return error.DraftCacheTensor;
        if (t.shard >= file_of.len) return error.DraftCacheTensor;
        const f = file_of[t.shard] orelse blk: {
            const path = try ck.shardPath(a, t.shard);
            defer a.free(path);
            const idx = try cache.openFile(path);
            file_of[t.shard] = idx;
            break :blk idx;
        };
        cache.setLoc(geom.group_of[s], geom.offset_of[s] + e, k, .{ .file = f, .offset = t.begin });
    };
}

/// DRAFTCACHE's state: the geometry the cache's slices point into, and the cache over the checkpoint's shards.
pub const DraftCache = struct {
    a: std.mem.Allocator,
    geom: DraftGeometry,
    n_stages: u32,
    n_experts: u32,
    hot: u32,
    cache: *xsc.Cache,
    /// `cache.stats` at the previous `takeRequestStats` (zero: construction).
    request_base: expert_stream.Stats = .{},

    /// The statistics since the previous call (the first: since `cache.stats` was last zeroed; the Module zeroes it
    /// after the seed), for one request's receipt; the cache itself carries over to the next request.
    pub fn takeRequestStats(self: *DraftCache) expert_stream.Stats {
        const now = self.cache.stats;
        var d: expert_stream.Stats = .{};
        inline for (@typeInfo(expert_stream.Stats).@"struct".field_names) |n| @field(d, n) = @field(now, n) - @field(self.request_base, n);
        self.request_base = now;
        return d;
    }

    /// The policy alone, no slot memory and no file (the trace backend's stand-in).
    pub fn planOnly(a: std.mem.Allocator, c: *const v41.Config, hot: u32, pool: DraftPool) !*DraftCache {
        return planOnlyWith(a, c, hot, pool, .shipped);
    }

    pub fn planOnlyWith(a: std.mem.Allocator, c: *const v41.Config, hot: u32, pool: DraftPool, policy: xsc.PolicyKind) !*DraftCache {
        const self = try a.create(DraftCache);
        errdefer a.destroy(self);
        var geom = try DraftGeometry.of(c, hot, pool);
        geom.policy = policy;
        self.* = .{ .a = a, .geom = geom, .n_stages = c.dspark.n_stages, .n_experts = c.dspark.n_routed_experts, .hot = hot, .cache = undefined };
        self.cache = try xsc.Cache.init(a, self.geom.geometry(), .none, null);
        return self;
    }

    /// The cache at `hot` over `ck`'s shards (each opened past the page cache), every part's place checked against the
    /// header (dtype, shape, inside the file) by name before any read.
    pub fn open(a: std.mem.Allocator, ck: *const v41.Checkpoint, c: *const v41.Config, hot: u32, form: DraftPool, memory: xsc.Memory, pool: ?*expert_io.Pool) !*DraftCache {
        return openWith(a, ck, c, hot, form, .shipped, memory, pool);
    }

    pub fn openWith(a: std.mem.Allocator, ck: *const v41.Checkpoint, c: *const v41.Config, hot: u32, form: DraftPool, policy: xsc.PolicyKind, memory: xsc.Memory, pool: ?*expert_io.Pool) !*DraftCache {
        const self = try a.create(DraftCache);
        errdefer a.destroy(self);
        var geom = try DraftGeometry.of(c, hot, form);
        geom.policy = policy;
        self.* = .{ .a = a, .geom = geom, .n_stages = c.dspark.n_stages, .n_experts = c.dspark.n_routed_experts, .hot = hot, .cache = undefined };
        // The served route reads on its own tickets (the pool's aux ring): no ticket of the stream's demand ring is reused.
        if (memory == .mlx and (pool == null or pool.?.auxTickets() < draft_aux_tickets)) return error.DraftCacheTickets;
        self.cache = try xsc.Cache.init(a, self.geom.geometry(), memory, pool);
        errdefer self.cache.deinit();
        try placeParts(self.cache, a, ck, &self.geom, self.n_stages, self.n_experts);
        try self.cache.checkLocs();
        return self;
    }

    /// Construction, after the install warm-up (which routed its draft block through the cache): the policies
    /// anew, then the hot slots seeded with first ids (prompt-independent: any id is a priori as good as any other,
    /// and a seeded slot can only spare a first-use miss): per stage, each stage's first `Hs`; shared, the stages'
    /// ids interleaved (stage 0 id 0, stage 1 id 0, ..., then id 1, ...) until `hot`. Returns the records read.
    pub fn seedFirstIds(self: *DraftCache) !u32 {
        try self.cache.forgetAll();
        var ids: [max_draft_stages * 512]u16 = undefined;
        var n: u32 = 0;
        switch (self.geom.pool) {
            .per_stage => for (0..self.n_stages) |s| {
                for (ids[0..self.geom.caps[s]], 0..) |*d, i| d.* = @intCast(i);
                n += try self.cache.seed(s, ids[0..self.geom.caps[s]]);
            },
            .shared => {
                const cap = self.geom.caps[0];
                for (ids[0..cap], 0..) |*d, j| d.* = @intCast((j % self.n_stages) * self.n_experts + j / self.n_stages);
                n += try self.cache.seed(0, ids[0..cap]);
            },
        }
        return n;
    }

    pub fn deinit(self: *DraftCache) void {
        self.cache.deinit();
        self.a.destroy(self);
    }
};

// ── The compact head's data: a pinned subset of each stage's experts ──
//
// The stack of record's compact DSpark head keeps some of each stage's experts
// resident and maps every routed id to its compact slot (an id outside the
// subset takes slot 0: `_CompactMTPExpertSwitch`, tcq_runner/packed/
// run_full.py:89-93, 107-109). The subset is data, loaded and pinned by its
// sha256 (default: none, the full head).
//
// File (`mlx-serve-expert-subset-v1`): `{"format", "kind", "n_experts",
// "selected": [[ids of stage 0], ...], ...}`; any other field documents it.

pub const subset_format = "mlx-serve-expert-subset-v1";

pub const SubsetError = error{ SubsetFile, SubsetNotPinned, SubsetFormat, SubsetIds, OutOfMemory };

/// Where a subset file is and the sha256 (hex) it must have.
pub const SubsetPin = struct { path: []const u8, sha256: []const u8 };

pub const Subset = struct {
    arena: std.heap.ArenaAllocator,
    sha256: [32]u8,
    kind: []const u8,
    n_experts: u32,
    /// Per block, the kept experts in ascending order; slot `i` of the
    /// block's compact bank is expert `selected[block][i]`.
    selected: []const []const u16,

    /// The file at `pin.path`, refused unless its sha256 is `pin.sha256`
    /// (hex) and every block's ids are ascending, unique, below `n_experts`
    /// and at least one.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, pin: SubsetPin, diag: ?*SubsetDiag) SubsetError!Subset {
        var self: Subset = .{ .arena = std.heap.ArenaAllocator.init(gpa), .sha256 = undefined, .kind = "", .n_experts = 0, .selected = &.{} };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        const text = std.Io.Dir.cwd().readFileAlloc(io, pin.path, a, .limited(16 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.SubsetFile, "{s}: {t}", .{ pin.path, e }),
        };
        std.crypto.hash.sha2.Sha256.hash(text, &self.sha256, .{});
        const hex = std.fmt.bytesToHex(self.sha256, .lower);
        if (!std.ascii.eqlIgnoreCase(&hex, pin.sha256))
            return refuse(diag, error.SubsetNotPinned, "{s}: sha256 {s}, pinned {s}", .{ pin.path, &hex, pin.sha256 });
        const Json = struct { format: []const u8, kind: []const u8 = "", n_experts: u32, selected: []const []const u16 };
        const j = std.json.parseFromSliceLeaky(Json, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.SubsetFormat, "{s}: {t}", .{ pin.path, e }),
        };
        if (!std.mem.eql(u8, j.format, subset_format)) return refuse(diag, error.SubsetFormat, "{s}: format \"{s}\", not {s}", .{ pin.path, j.format, subset_format });
        if (j.n_experts == 0 or j.selected.len == 0) return refuse(diag, error.SubsetFormat, "{s}: no blocks or no experts", .{pin.path});
        for (j.selected, 0..) |ids, b| {
            if (ids.len == 0) return refuse(diag, error.SubsetIds, "{s}: block {d} keeps no expert", .{ pin.path, b });
            for (ids, 0..) |e, i| {
                if (e >= j.n_experts) return refuse(diag, error.SubsetIds, "{s}: block {d} keeps expert {d} of {d}", .{ pin.path, b, e, j.n_experts });
                if (i > 0 and ids[i - 1] >= e) return refuse(diag, error.SubsetIds, "{s}: block {d} is not ascending at {d}", .{ pin.path, b, i });
            }
        }
        self.kind = j.kind;
        self.n_experts = j.n_experts;
        self.selected = j.selected;
        return self;
    }

    pub fn deinit(self: *Subset) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Block `block`'s slot for every routed id `0..n_experts`: its position
    /// in the kept list, 0 for an id outside it.
    pub fn lut(self: *const Subset, block: usize, out: []i32) void {
        std.debug.assert(out.len == self.n_experts);
        @memset(out, 0);
        for (self.selected[block], 0..) |e, i| out[e] = @intCast(i);
    }

    /// Experts the subset leaves out, over every block.
    pub fn pruned(self: *const Subset) u64 {
        var n: u64 = 0;
        for (self.selected) |ids| n += self.n_experts - ids.len;
        return n;
    }

    pub fn shaHex(self: *const Subset) [64]u8 {
        return std.fmt.bytesToHex(self.sha256, .lower);
    }
};

/// A refusal's message (the caller names the subset in its own diag).
pub const SubsetDiag = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const SubsetDiag) []const u8 {
        return self.buf[0..self.len];
    }
};

fn refuse(diag: ?*SubsetDiag, err: SubsetError, comptime fmt: []const u8, args: anytype) SubsetError {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

// ── Tests (host) ──

const testing = std.testing;

fn writeSubsetFile(tmp: *std.testing.TmpDir, name: []const u8, text: []const u8, path_buf: []u8) ![]const u8 {
    try tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = text });
    var root: [512]u8 = undefined;
    return std.fmt.bufPrint(path_buf, "{s}/{s}", .{ root[0..try tmp.dir.realPath(testing.io, &root)], name });
}

fn subsetSha(text: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

test "dsv41 dspark head: a pinned subset file loads; every routed id maps to its compact slot, the rest to slot 0" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const text =
        \\{"format": "mlx-serve-expert-subset-v1", "kind": "test", "n_experts": 8, "note": "any field documents",
        \\ "selected": [[1, 4, 6], [0, 7]]}
    ;
    var pb: [700]u8 = undefined;
    const path = try writeSubsetFile(&tmp, "s.json", text, &pb);
    const sha = subsetSha(text);
    var s = try Subset.load(testing.allocator, testing.io, .{ .path = path, .sha256 = &sha }, null);
    defer s.deinit();
    try testing.expectEqualStrings("test", s.kind);
    try testing.expectEqual(@as(u64, 5 + 6), s.pruned());
    try testing.expectEqualStrings(&sha, &s.shaHex());
    var l: [8]i32 = undefined;
    s.lut(0, &l);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 1, 0, 2, 0 }, &l);
    s.lut(1, &l);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 0, 0, 0, 1 }, &l);
}

test "dsv41 dspark head: an unpinned, malformed or unordered subset file is refused by name" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pb: [700]u8 = undefined;
    var diag: SubsetDiag = .{};
    const good = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[1, 4]]}";
    const path = try writeSubsetFile(&tmp, "g.json", good, &pb);
    const zero: [64]u8 = @splat('0');
    try testing.expectError(error.SubsetNotPinned, Subset.load(testing.allocator, testing.io, .{ .path = path, .sha256 = &zero }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "pinned") != null);
    const cases = [_]struct { text: []const u8, err: SubsetError }{
        .{ .text = "{\"format\": \"other\", \"n_experts\": 8, \"selected\": [[1]]}", .err = error.SubsetFormat },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[4, 1]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[1, 1]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[8]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8, \"selected\": [[]]}", .err = error.SubsetIds },
        .{ .text = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 8}", .err = error.SubsetFormat },
    };
    for (cases, 0..) |cs, i| {
        var nb: [16]u8 = undefined;
        var pb2: [700]u8 = undefined;
        const p = try writeSubsetFile(&tmp, try std.fmt.bufPrint(&nb, "c{d}.json", .{i}), cs.text, &pb2);
        const sha = subsetSha(cs.text);
        try testing.expectError(cs.err, Subset.load(testing.allocator, testing.io, .{ .path = p, .sha256 = &sha }, null));
    }
    try testing.expectError(error.SubsetFile, Subset.load(testing.allocator, testing.io, .{ .path = "/nonexistent/s.json", .sha256 = &zero }, null));
}

test "dsv41 dspark head: DRAFTRC binds the draft's routes on the real geometry, each stage at its stream dtype, and the block takes them at 5 rows" {
    const a = testing.allocator;
    const TraceOps = ops.TraceOps;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    var g = TraceOps.init(a);
    defer g.deinit();
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const H = Head(TraceOps);
    const rt: graph.Routes = .{ .rc_draft = true, .head = .bf16, .draft_rows = graph.draft_compile_max_rows };
    try testing.expectError(error.DraftNeedsKernels, H.init(a, &g, c, rt, &lookup));
    const h = try H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg });
    defer h.deinit(&g);
    // Stage 0 reads the block's bf16 embedding (its attention prep bf16, its ffn prep the mixed
    // call); stages 1.. run on the f32 stream.
    const d = h.rc.?;
    try testing.expectEqual(@as(usize, 3), d.stages.items.len);
    try testing.expect(d.layer(0).tape_attn.? == &d.tape_bf16 and d.layer(0).tape_mixed.? == &d.tape_mixed and d.layer(0).tape.? == &d.tape_f32);
    try testing.expect(d.layer(2).tape_attn.? == &d.tape_f32 and d.layer(2).tape_mixed == null and d.layer(2).router.? == &d.stages.items[2].router);
    try testing.expectEqual(ops.Dtype.bfloat16, d.stages.items[0].attn_dt);
    try testing.expectEqual(ops.Dtype.float32, d.stages.items[1].attn_dt);
    const caches = try arena.allocator().alloc(H.Cache, h.nStages());
    for (caches) |*x| x.* = .{};
    defer for (caches) |*x| x.deinit(&g);
    const vocab: c_int = @intCast(c.vocab_size);
    const table = try g.input(&.{ vocab, 5120 }, .bfloat16);
    // The commit seed from the prompt pass (f32 hiddens, 3 rows): main_proj and each stage's main
    // KV at f32 x.
    var l0 = g.launched.items.len;
    try h.seedMain(&g, try g.input(&.{ 1, 3, 15360 }, .float32), caches);
    try testing.expectEqual(@as(usize, 1 + 3), g.launchesOf(l0, .q3drc_mxfp8_fma_f32x));
    // A draft block from a verify row's bf16 hidden.
    l0 = g.launched.items.len;
    const n0 = g.nodes.items.len;
    const out = try h.draftBlock(&g, try g.input(&.{ 1, 1, 15360 }, .bfloat16), 7, caches, .{ .table = table }, .{ .dense = table });
    try testing.expect(g.shapeOf(out.logits).eql(ops.Shape.of(&.{ 1, 5, vocab })));
    for (g.nodes.items[n0..]) |nd| try testing.expect(nd.op != .qmm);
    const L = struct {
        fn of(tg: *const TraceOps, from: usize, k: xk.Kernel) usize {
            return tg.launchesOf(from, k);
        }
    }.of;
    // main_proj at bf16 x (the draft-only plan); the main KV and stage 0's wq_a / wq_b / wkv at bf16;
    // the other 21 projections at f32 x (stages 1-2's attention, every wo_a / wo_b / shared w1, w3, w2).
    try testing.expectEqual(@as(usize, 1), L(&g, l0, .q3rc_mxfp8_fma__draft));
    try testing.expectEqual(@as(usize, 3 + 3), L(&g, l0, .q3rc_mxfp8_fma));
    try testing.expectEqual(@as(usize, 6 + 15), L(&g, l0, .q3drc_mxfp8_fma_f32x));
    try testing.expectEqual(@as(usize, 1), L(&g, l0, .q3ht_collapse_norm));
    try testing.expectEqual(@as(usize, 2), L(&g, l0, .q3ht_collapse_norm__f32));
    try testing.expectEqual(@as(usize, 1), L(&g, l0, .q3ht_combine_collapse_norm__f32_rbf16));
    try testing.expectEqual(@as(usize, 2), L(&g, l0, .q3ht_combine_collapse_norm__f32));
    try testing.expectEqual(@as(usize, 3), L(&g, l0, .q3ht_combine__f32));
    try testing.expectEqual(@as(usize, 6), L(&g, l0, .q3ht_mixfin));
    try testing.expectEqual(@as(usize, 6), L(&g, l0, .q3dk_sinkhorn16_hc4_it20));
    try testing.expectEqual(@as(usize, 6), L(&g, l0, .q3rc_premix_part));
    try testing.expectEqual(@as(usize, 3), L(&g, l0, .q3rc_router_tail__n128_top3));
    // Off the draft's geometry: refused by name.
    var bad = c;
    bad.dspark.n_experts_per_tok = 4;
    try testing.expectError(error.DraftRcGeometry, H.initWith(a, &g, bad, rt, &lookup, .{ .registry = &reg }));
}

/// The trace backend's script: each routing barrier's ids, `rows` rows of distinct top-k ids drawn from a seeded
/// stream (the same seed replays the same ids).
const ScriptIds = struct {
    prng: std.Random.DefaultPrng,
    n_experts: u16,
    k: u16,

    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *ScriptIds = @ptrCast(@alignCast(ctx));
        const r = self.prng.random();
        var i: usize = 0;
        while (i < out.len) : (i += 1) {
            const row0 = i - i % self.k;
            while (true) {
                const e = r.uintLessThan(u16, self.n_experts);
                if (std.mem.indexOfScalar(u16, out[row0..i], e) == null) {
                    out[i] = e;
                    break;
                }
            }
        }
    }

    fn argmax(_: *anyopaque) anyerror!u32 {
        return 0;
    }

    fn values(self: *ScriptIds) ops.TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
};

/// The DRAFTCACHE trace check at `pool`: the head binds the cache's banks, each block gathers at a twin cache's slots
/// for the replayed ids, the banks are within the bill, and each block is evaluated (the loop's draft read) before the
/// next block's first routing barrier.
fn checkCachedHead(pool: DraftPool) !void {
    const a = testing.allocator;
    const TraceOps = ops.TraceOps;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    var g = TraceOps.init(a);
    defer g.deinit();
    g.record_host = true;
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const H = Head(TraceOps);
    const rt: graph.Routes = .{ .rc_draft = true, .head = .bf16, .draft_rows = graph.draft_compile_max_rows };
    const hot: u32 = 128;
    const dc = try DraftCache.planOnly(a, &c, hot, pool);
    defer dc.deinit();
    // The twin replays the same ids on its own policy: the slots the graph must gather at.
    const twin = try DraftCache.planOnly(a, &c, hot, pool);
    defer twin.deinit();
    try testing.expectEqual(@as(u32, 128), try dc.seedFirstIds());
    _ = try twin.seedFirstIds();
    const h = try H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg, .cache = dc });
    defer h.deinit(&g);
    const rows: [3]u32 = switch (pool) {
        .per_stage => .{ 43 + 15, 43 + 15, 42 + 15 },
        .shared => .{ 128 + 15, 128 + 15, 128 + 15 },
    };
    for (h.stages, rows) |st, r| {
        try testing.expect(g.shapeOf(st.experts.w1.w).eql(ops.Shape.of(&.{ @intCast(r), 2304, 640 })));
        try testing.expect(g.shapeOf(st.experts.w2.s).eql(ops.Shape.of(&.{ @intCast(r), 5120, 72 })));
        try testing.expectEqual(ops.Dtype.uint8, g.dtypeOf(st.experts.w3.s));
    }
    for (g.nodes.items) |nd| try testing.expect(nd.op != .stack or nd.shape.dim(0) != 128);
    // The route's device allocations: each group's six slot arrays (the stages of a shared pool bind the same ones),
    // within the bill's term by its rounding only.
    var banks: u64 = 0;
    for (h.stages, 0..) |st, si| {
        if (si > 0 and dc.geom.group_of[si] == dc.geom.group_of[si - 1]) continue;
        inline for (.{ "w1", "w3", "w2" }) |name| for ([_]u32{ @field(st.experts, name).w, @field(st.experts, name).s }) |x| {
            banks += @as(u64, @intCast(g.shapeOf(x).numel())) * ops.dtypeSize(g.dtypeOf(x));
        };
    }
    const term = try draftCacheBytes(&c, hot, pool);
    try testing.expect(banks <= term and term - banks < 18 * xsc.alloc_page_bytes);
    const caches = try arena.allocator().alloc(H.Cache, h.nStages());
    for (caches) |*x| x.* = .{};
    defer for (caches) |*x| x.deinit(&g);
    const table = try g.input(&.{ @intCast(c.vocab_size), 5120 }, .bfloat16);
    var script: ScriptIds = .{ .prng = std.Random.DefaultPrng.init(5), .n_experts = 128, .k = 3 };
    var replay: ScriptIds = .{ .prng = std.Random.DefaultPrng.init(5), .n_experts = 128, .k = 3 };
    g.host_values = script.values();
    try h.seedMain(&g, try g.input(&.{ 1, 3, 15360 }, .float32), caches);
    const n_block0 = g.nodes.items.len;
    var last_eval: ?usize = null;
    for (0..4) |_| {
        const n0 = g.nodes.items.len;
        const d = try h.draftBlock(&g, try g.input(&.{ 1, 1, 15360 }, .bfloat16), 7, caches, .{ .table = table }, .{ .dense = table });
        // Per stage: one routing barrier, then one [5, 3] int32 host array: the twin's slots for the same ids.
        var stage: usize = 0;
        for (g.nodes.items[n0..], n0..) |nd, i| {
            if (nd.op == .host_read and stage == 0) if (last_eval) |le| try testing.expect(le <= i);
            if (nd.op != .host or !nd.shape.eql(ops.Shape.of(&.{ 5, 3 }))) continue;
            try testing.expectEqual(ops.Dtype.int32, nd.dtype);
            var ids: [15]u16 = undefined;
            try ScriptIds.ids(&replay, &ids);
            for (&ids) |*e| e.* += @intCast(dc.geom.offset_of[stage]);
            var want: [15]u32 = undefined;
            try twin.cache.route(dc.geom.group_of[stage], &ids, &want);
            var want_i: [15]i32 = undefined;
            for (&want_i, want) |*w, v| w.* = @intCast(v);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want_i), g.hostBytesOf(@intCast(i)).?);
            stage += 1;
        }
        try testing.expectEqual(@as(usize, 3), stage);
        // The loop reads the block's drafts (an eval over everything its last stage's gathers feed) before it verifies
        // and drafts again: the next block's rows are written only after it.
        try g.evalAll(&.{ d.ids, d.conf });
        last_eval = g.nodes.items.len;
    }
    try testing.expectEqual(twin.cache.stats.expert_cache_misses, dc.cache.stats.expert_cache_misses);
    // Beyond the resident head's block, the route allocates per block only its three [5, 3] int32 index arrays.
    var host_bytes: u64 = 0;
    for (g.nodes.items[n_block0..]) |nd| if (nd.op == .host) {
        host_bytes += @as(u64, @intCast(nd.shape.numel())) * ops.dtypeSize(nd.dtype);
    };
    try testing.expect(dc.cache.stats.expert_cache_misses > 0 and dc.cache.stats.expert_cache_hits > 0);
    // The resident head gathers at the router's ids: no host read, no host index array.
    const h0 = try H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg });
    defer h0.deinit(&g);
    try testing.expect(g.shapeOf(h0.stages[0].experts.w1.w).eql(ops.Shape.of(&.{ 128, 2304, 640 })));
    const n0 = g.nodes.items.len;
    _ = try h0.draftBlock(&g, try g.input(&.{ 1, 1, 15360 }, .bfloat16), 7, caches, .{ .table = table }, .{ .dense = table });
    var host0: u64 = 0;
    for (g.nodes.items[n0..]) |nd| {
        try testing.expect(nd.op != .host_read);
        if (nd.op == .host) host0 += @as(u64, @intCast(nd.shape.numel())) * ops.dtypeSize(nd.dtype);
    }
    try testing.expectEqual(4 * (host0 + 3 * 15 * 4), host_bytes);
}

test "dsv41 dspark head: DRAFTCACHE binds the stages to the cache's slot banks and gathers each block at the cache's slots for the router's ids" {
    try checkCachedHead(.per_stage);
}

test "dsv41 dspark head: DRAFTCACHE shared pool: every stage binds one bank of H + 15 rows and gathers at the shared policy's slots for its offset ids" {
    try checkCachedHead(.shared);
}

test "dsv41 dspark head: draft routes (profile builds): every expert source hands the block each stage's router ids, not the cache's slots" {
    if (comptime !draft_routes.enabled) return error.SkipZigTest;
    const a = testing.allocator;
    const TraceOps = ops.TraceOps;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const H = Head(TraceOps);
    const rt: graph.Routes = .{ .rc_draft = true, .head = .bf16, .draft_rows = graph.draft_compile_max_rows };
    try draft_routes.install(.{ .n_stages = c.dspark.n_stages, .n_experts = c.dspark.n_routed_experts, .top_k = c.dspark.n_experts_per_tok, .block = c.dspark.block_size }, 8);
    defer draft_routes.uninstall();
    // the resident head, then the cache's two forms (per stage; one pool whose ids carry a stage offset)
    for (0..3) |form| {
        // one backend per head (each binds its own compiled-region contexts)
        var g = TraceOps.init(a);
        defer g.deinit();
        const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
        const table = try g.input(&.{ @intCast(c.vocab_size), 5120 }, .bfloat16);
        var script: ScriptIds = .{ .prng = std.Random.DefaultPrng.init(5), .n_experts = 128, .k = 3 };
        g.host_values = script.values();
        const dc: ?*DraftCache = switch (form) {
            0 => null,
            1 => try DraftCache.planOnly(a, &c, 128, .per_stage),
            else => try DraftCache.planOnly(a, &c, 128, .shared),
        };
        defer if (dc) |x| x.deinit();
        if (dc) |x| _ = try x.seedFirstIds();
        const h = try H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg, .cache = dc });
        defer h.deinit(&g);
        const caches = try arena.allocator().alloc(H.Cache, h.nStages());
        for (caches) |*x| x.* = .{};
        defer for (caches) |*x| x.deinit(&g);
        try h.seedMain(&g, try g.input(&.{ 1, 3, 15360 }, .float32), caches);
        var d = try h.draftBlock(&g, try g.input(&.{ 1, 1, 15360 }, .bfloat16), 7, caches, .{ .table = table }, .{ .dense = table });
        for (d.stage_ids[0..c.dspark.n_stages]) |k| {
            const x = k orelse return error.TestUnexpectedResult;
            // the router's [5, 3] int32 ids, never the slot array the cache's arm builds on the host
            try testing.expect(g.shapeOf(x).eql(ops.Shape.of(&.{ 5, 3 })));
            try testing.expectEqual(ops.Dtype.int32, g.dtypeOf(x));
            try testing.expect(g.nodes.items[x].op != .host);
        }
        draft_routes.drop(TraceOps, &g, &d.stage_ids);
    }
}

test "dsv41 dspark head: DRAFTCACHE geometry: the even split, the bill's bytes, refused by name when a stage's slots hold every expert" {
    const a = testing.allocator;
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    const rec: u64 = 18_800_640;
    try testing.expectEqual(rec, expertBytes(&c));
    var caps: [3]u32 = undefined;
    for ([_]u32{ 96, 128, 201, 256 }, [_][3]u32{ .{ 32, 32, 32 }, .{ 43, 43, 42 }, .{ 67, 67, 67 }, .{ 86, 85, 85 } }) |hot, want| {
        hotSplit(hot, 3, &caps);
        try testing.expectEqualSlices(u32, &want, &caps);
        // Weights are page multiples; each stage's three scales arrays round up to the 16 KiB page.
        var round: u64 = 0;
        for (want) |cap| {
            const r: u64 = cap + 15;
            round += 3 * (std.mem.alignForward(u64, r * 368_640, 16384) - r * 368_640);
        }
        try testing.expectEqual((hot + 45) * rec + round, try draftCacheBytes(&c, hot, .per_stage));
        try testing.expect(round < 9 * 16384);
        // Shared: one bank of hot + 15 rows (its three scales arrays rounded).
        const r: u64 = hot + 15;
        try testing.expectEqual((hot + 15) * rec + 3 * (std.mem.alignForward(u64, r * 368_640, 16384) - r * 368_640), try draftCacheBytes(&c, hot, .shared));
    }
    const none = try draftCacheBytes(&c, 0, .per_stage);
    try testing.expect(none >= 45 * rec and none < 45 * rec + 9 * 16384);
    try testing.expectError(error.DraftCacheGeometry, draftCacheBytes(&c, 3 * 113, .per_stage));
    try testing.expect((try draftCacheBytes(&c, 3 * 112, .per_stage)) < 384 * rec);
    try testing.expectError(error.DraftCacheGeometry, draftCacheBytes(&c, 384 - 15, .shared));
    try testing.expect((try draftCacheBytes(&c, 384 - 16, .shared)) < 384 * rec);
    // The shared geometry: every stage in group 0 at its offset; the seed interleaves the stages.
    const d = try DraftGeometry.of(&c, 201, .shared);
    try testing.expectEqual(@as(u32, 1), d.n_groups);
    try testing.expectEqual(@as(u32, 384), d.n_experts);
    try testing.expectEqualSlices(u32, &.{ 0, 128, 256 }, d.offset_of[0..3]);
    const dc = try DraftCache.planOnly(a, &c, 7, .shared);
    defer dc.deinit();
    try testing.expectEqual(@as(u32, 7), try dc.seedFirstIds());
    for ([_]u16{ 0, 128, 256, 1, 129, 257, 2 }) |e| try testing.expect(dc.cache.slotOf(0, e) != null);
    try testing.expect(dc.cache.slotOf(0, 130) == null);
    // The shared slots are keyed by (stage, expert): stage 1's expert 2 (global 130) misses although stage 0's expert 2
    // is resident.
    var slots: [1]u32 = undefined;
    const miss0 = dc.cache.stats.expert_cache_misses;
    try testing.expect(dc.cache.slotOf(0, 2) != null);
    try dc.cache.route(0, &.{@intCast(dc.geom.offset_of[1] + 2)}, &slots);
    try testing.expectEqual(miss0 + 1, dc.cache.stats.expert_cache_misses);
    for ([_]u32{ 0, 1, 2 }) |st| try testing.expectEqual(@as(u32, 0), dc.geom.group_of[st]);
    // Per request: the first take counts from construction (the seed and the route above), the next only its own
    // routes; the residents carry over (expert 130 hits in request 2).
    const r1 = dc.takeRequestStats();
    try testing.expectEqual(dc.cache.stats.expert_cache_misses, r1.expert_cache_misses);
    try testing.expectEqual(@as(u64, 1), r1.route_calls);
    var slots2: [2]u32 = undefined;
    try dc.cache.route(0, &.{ 130, 3 }, &slots2);
    const r2 = dc.takeRequestStats();
    try testing.expectEqual(@as(u64, 1), r2.route_calls);
    try testing.expectEqual(@as(u64, 1), r2.expert_cache_hits);
    try testing.expectEqual(@as(u64, 1), r2.expert_cache_misses);
    try testing.expectEqual(r1.expert_cache_misses + 1, dc.cache.stats.expert_cache_misses);
}

// DSV41_BANK=<bank> (host): stage 0's real records through the cache at 2 hot + 3 transient slots (94 MB of host rows),
// 24 routes of 3 ids forcing evictions: every routed id's six slot rows equal its tensors read past the page cache.
test "dsv41 dspark head: DRAFTCACHE on the bank: every routed id's slot rows are its checkpoint tensors through evictions" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 draft cache bank: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, testing.io, bank, &vd);
    var ck = try v41.Checkpoint.openIndexed(a, testing.io, bank, &vd);
    defer ck.deinit();
    var dg = try DraftGeometry.of(&c, 0, .per_stage);
    var geom = dg.geometry();
    geom.capacity = &.{2};
    geom.transient = 3;
    var pool = try expert_io.Pool.start(a, .{ .workers = 2, .tickets = 64 });
    defer pool.stop();
    const cache = try xsc.Cache.init(a, geom, .host, pool);
    defer cache.deinit();
    try placeParts(cache, a, &ck, &dg, 1, c.dspark.n_routed_experts);
    try cache.checkLocs();
    try testing.expectEqual(@as(u32, 2), try cache.seed(0, &.{ 0, 1 }));
    const want = try a.alloc(u8, 5_898_240);
    defer a.free(want);
    var name: [96]u8 = undefined;
    var script: ScriptIds = .{ .prng = std.Random.DefaultPrng.init(17), .n_experts = 8, .k = 3 };
    var fds: [64]std.c.fd_t = @splat(-1);
    defer for (fds) |fd| if (fd >= 0) {
        _ = std.c.close(fd);
    };
    for (0..24) |_| {
        var ids: [3]u16 = undefined;
        try ScriptIds.ids(&script, &ids);
        var slots: [3]u32 = undefined;
        try cache.route(0, &ids, &slots);
        for (ids, slots) |e, slot| for (draft_parts, 0..) |part, k| {
            const t = ck.tensors.get(try std.fmt.bufPrint(&name, "mtp.0.ffn.experts.{d}.{s}", .{ e, part })).?;
            if (fds[t.shard] < 0) {
                const path = try ck.shardPath(a, t.shard);
                defer a.free(path);
                fds[t.shard] = try @import("io_util").openNoCache(path.ptr, .{});
            }
            const buf = want[0..@intCast(t.end - t.begin)];
            try @import("io_util").readAligned(fds[t.shard], buf, t.begin);
            try testing.expectEqualSlices(u8, buf, cache.row(0, k, slot));
        };
    }
    try testing.expect(cache.stats.expert_cache_evictions > 0);
    std.debug.print("dsv41 draft cache bank: {d} routes, {d} hits, {d} misses, {d} evictions, {d} B read\n", .{ cache.stats.route_calls, cache.stats.expert_cache_hits, cache.stats.expert_cache_misses, cache.stats.expert_cache_evictions, cache.stats.expert_bytes_read });
}

// DSV41_PHASE0B_MLX=1 DSV41_BANK=<bank> (device, the window's smoke step): stage 0's switch over the cache's MLX slots
// (8 hot + 15 transient rows, misses read by the pool; per stage, then shared with a stage-1 route between stage-0 routes)
// against the resident bank of all 128 experts (stacked as the head does), 16 blocks of 5 rows x top-3 forcing
// evictions per form, f32 and bf16 inputs: outputs equal bit for bit. Peak
// device ~5.3 GB (the 2.41 GB bank, its 2.41 GB of sources while it stacks, 0.43 GB of slots); a few seconds.
test "dsv41 smoke 0b: DRAFTCACHE: a draft stage's switch over the cache's slots equals the resident bank's, bit for bit, on real records" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse {
        std.debug.print("\ndraft cache smoke: DSV41_PHASE0B_MLX without DSV41_BANK (the real records): refused\n", .{});
        return error.TestUnexpectedResult;
    });
    const mlx = @import("mlx");
    const model_io = @import("deepseek_v41_host.zig").model;
    const G = ops.MlxOps;
    const T = G.T;
    const H = Head(G);
    const a = testing.allocator;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, dir, &vd);
    var ck = try v41.Checkpoint.openIndexed(a, testing.io, dir, &vd);
    defer ck.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    // The resident bank: stage 0's shard loaded past the page cache, its 128 experts stacked per part.
    var w = model_io.Weights.init(a);
    defer w.deinit();
    const t0 = ck.tensors.get("mtp.0.ffn.experts.0.w1.weight").?;
    const shard = try ck.shardPath(a, t0.shard);
    defer a.free(shard);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try model_io.loadSafetensorsFile(a, &w, shard.ptr, cpu, .{ .nocache = true });
    var res: H.Experts = undefined;
    var name: [96]u8 = undefined;
    inline for (.{ "w1", "w3", "w2" }) |proj| {
        var ws: [128]T = undefined;
        var ss: [128]T = undefined;
        for (0..c.dspark.n_routed_experts) |e| {
            ws[e] = w.get(try std.fmt.bufPrint(&name, "mtp.0.ffn.experts.{d}." ++ proj ++ ".weight", .{e})).?;
            ss[e] = w.get(try std.fmt.bufPrint(&name, "mtp.0.ffn.experts.{d}." ++ proj ++ ".scales", .{e})).?;
        }
        @field(res, proj) = .{ .w = g.keep(try g.stack(ws[0..c.dspark.n_routed_experts], 0)), .s = g.keep(try g.stack(ss[0..c.dspark.n_routed_experts], 0)), .mode = .mxfp4 };
        try g.evalAll(&.{ @field(res, proj).w, @field(res, proj).s });
    }
    defer inline for (.{ "w1", "w3", "w2" }) |proj| {
        g.release(@field(res, proj).w);
        g.release(@field(res, proj).s);
    };
    w.deinit();
    w = model_io.Weights.init(a);
    // The cache: 8 hot + 15 transient MLX rows, reads through a pool, the first 8 ids seeded. Per stage: stage 0's bank;
    // shared: one bank over stages 0 and 1, a stage-1 route (ids + 128) before every stage-0 route, so stage 0 reads
    // rows stage 1 just used. Only stage 0 is compared (its resident bank is the one loaded).
    for ([_]DraftPool{ .per_stage, .shared }) |form| {
        var dg = try DraftGeometry.of(&c, 0, form);
        var geom = dg.geometry();
        geom.capacity = &.{8};
        var pool = try expert_io.Pool.start(a, .{ .workers = 4, .tickets = 256 });
        defer pool.stop();
        const cache = try xsc.Cache.init(a, geom, .{ .mlx = s }, pool);
        defer cache.deinit();
        // Shared: every stage's ids sit in the one group, so every stage is placed (checkLocs walks them all).
        try placeParts(cache, a, &ck, &dg, if (form == .shared) c.dspark.n_stages else 1, c.dspark.n_routed_experts);
        try cache.checkLocs();
        _ = try cache.seed(0, &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
        const cached: H.Experts = .{ .w1 = .{ .w = cache.arrays[0][0], .s = cache.arrays[0][1], .mode = .mxfp4 }, .w3 = .{ .w = cache.arrays[0][2], .s = cache.arrays[0][3], .mode = .mxfp4 }, .w2 = .{ .w = cache.arrays[0][4], .s = cache.arrays[0][5], .mode = .mxfp4 } };
        var script: ScriptIds = .{ .prng = std.Random.DefaultPrng.init(23), .n_experts = 40, .k = 3 };
        var prng = std.Random.DefaultPrng.init(29);
        var xs: [5 * 5120]f32 = undefined;
        var out_r: [15 * 5120]f32 = undefined;
        var out_c: [15 * 5120]f32 = undefined;
        for (0..16) |blk| {
            for (&xs) |*v| v.* = prng.random().floatNorm(f32);
            const mark = g.mark();
            defer g.resetTo(mark);
            var xf = try g.hostArray(std.mem.sliceAsBytes(&xs), &.{ 5, 5120 }, .float32);
            if (blk % 2 == 1) xf = try g.astype(xf, .bfloat16);
            var ids: [15]u16 = undefined;
            try ScriptIds.ids(&script, &ids);
            var ids_i: [15]i32 = undefined;
            for (&ids_i, ids) |*d, v| d.* = v;
            const ids_dev = try g.hostArray(std.mem.sliceAsBytes(&ids_i), &.{ 5, 3 }, .int32);
            var slots: [15]u32 = undefined;
            if (form == .shared) {
                var other: [15]u16 = undefined;
                try ScriptIds.ids(&script, &other);
                for (&other) |*e| e.* += 128;
                try cache.route(0, &other, &slots);
            }
            try cache.route(0, &ids, &slots);
            var sl: [15]i32 = undefined;
            for (&sl, slots) |*d, v| d.* = @intCast(v);
            const idx = try g.hostArray(std.mem.sliceAsBytes(&sl), &.{ 5, 3 }, .int32);
            const yr = try g.astype(try H.switchGlu(&g, &res, c.swiglu_limit, xf, ids_dev), .float32);
            const yc = try g.astype(try H.switchGlu(&g, &cached, c.swiglu_limit, xf, idx), .float32);
            _ = try g.hostF32(yr, &out_r);
            _ = try g.hostF32(yc, &out_c);
            try testing.expectEqualSlices(u32, @ptrCast(&out_r), @ptrCast(&out_c));
        }
        std.debug.print("\ndraft cache smoke ({t}): 16 blocks bit for bit (f32 and bf16 inputs); {d} hits, {d} misses, {d} evictions, {d} B read\n", .{ form, cache.stats.expert_cache_hits, cache.stats.expert_cache_misses, cache.stats.expert_cache_evictions, cache.stats.expert_bytes_read });
        try testing.expect(cache.stats.expert_cache_evictions > 0);
    }
}

// DSV41_BANK=<bank> (host): the served DraftCache.open over the real checkpoint for both pools at H 128, no slot memory
// (placement and the construction check only: nothing read or allocated): every stage's every expert is placed, each
// pair in its own shard, stage 2's last expert where its pool keys it.
test "dsv41 dspark head: DRAFTCACHE on the bank: both pools place every stage's experts in their shards at construction" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 draft cache placement: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, testing.io, bank, &vd);
    var ck = try v41.Checkpoint.openIndexed(a, testing.io, bank, &vd);
    defer ck.deinit();
    const last_name = "mtp.2.ffn.experts.127.w2.scales";
    const last = ck.tensors.get(last_name).?;
    for ([_]DraftPool{ .per_stage, .shared }) |form| {
        const dc = try DraftCache.open(a, &ck, &c, 128, form, .none, null);
        defer dc.deinit();
        // Three shards (stages 0 / 1 / 2), each opened once.
        try testing.expectEqual(@as(usize, 3), dc.cache.files.items.len);
        const grp = dc.geom.group_of[2];
        const id = dc.geom.offset_of[2] + 127;
        try testing.expectEqual(@as(u32, if (form == .shared) 383 else 127), id);
        const loc = dc.cache.locs[(grp * dc.cache.geom.n_experts + id) * draft_parts.len + 5];
        try testing.expectEqual(last.begin, loc.offset);
        // Stage 0's expert 0 and stage 2's expert 127 read from different files.
        const first = dc.cache.locs[(@as(usize, dc.geom.group_of[0]) * dc.cache.geom.n_experts + dc.geom.offset_of[0]) * draft_parts.len];
        try testing.expect(first.file != loc.file);
    }
}

// DSV41_DRAFT_REPLAY=<receipt.json>[,<receipt.json>...] (host, CPU only): each receipt's draft route stream (per cycle,
// per stage, the block's routed ids) replayed through the served policy (planOnly, first-ids seed) for both pools at
// H 96 / 128 / 201 / 256. Misses split into compulsory (an id neither seeded nor routed before) and capacity; the cycle
// price at the decode lane's figures: 1.41 ms per miss + 0.55 ms of routing barriers per cycle against 0.65 ms per
// decode row gained (the 7.29 GB bill rows).
test "dsv41 dspark head: DRAFTCACHE replay of recorded draft routes (misses per cycle per pool and H)" {
    const list = std.mem.span(std.c.getenv("DSV41_DRAFT_REPLAY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    const Rec = struct { draft_route_stream: struct { stages: u32, experts_per_stage: u32, cycles: []const []const []const u16 } };
    const stock_rows: f64 = 171;
    const Rows = struct { hot: u32, per_stage: f64, shared: f64 };
    const rows = [_]Rows{ .{ .hot = 96, .per_stage = 180, .shared = 181 }, .{ .hot = 128, .per_stage = 179, .shared = 180 }, .{ .hot = 201, .per_stage = 176, .shared = 177 }, .{ .hot = 256, .per_stage = 174, .shared = 175 } };
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |path| {
        const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(256 << 20));
        defer a.free(text);
        const parsed = try std.json.parseFromSlice(Rec, a, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const ds = parsed.value.draft_route_stream;
        const n_cycles = ds.cycles.len;
        var distinct: [max_draft_stages]u32 = @splat(0);
        {
            var used: [max_draft_stages * 512]bool = @splat(false);
            for (ds.cycles) |cy| for (cy, 0..) |ids, s| for (ids) |e| {
                const g = s * ds.experts_per_stage + e;
                if (!used[g]) distinct[s] += 1;
                used[g] = true;
            };
        }
        std.debug.print("\nDSV41_DRAFT_REPLAY {{\"receipt\": \"{s}\", \"cycles\": {d}, \"distinct\": [{d}, {d}, {d}]}}\n", .{ std.fs.path.basename(path), n_cycles, distinct[0], distinct[1], distinct[2] });
        for (rows) |rw| for ([_]DraftPool{ .per_stage, .shared }) |form| {
            const dc = try DraftCache.planOnly(a, &c, rw.hot, form);
            defer dc.deinit();
            _ = try dc.seedFirstIds();
            var seen: [max_draft_stages * 512]bool = @splat(false);
            for (0..dc.geom.n_groups) |grp| for (dc.cache.policies[grp].residents()) |e| {
                if (e == expert_policy.no_expert) continue;
                // A group's ids are its stage's (per stage) or global (shared): back to a global id.
                const g: usize = if (form == .shared) e else grp * ds.experts_per_stage + e;
                seen[g] = true;
            };
            dc.cache.stats = .{};
            var compulsory: u64 = 0;
            var max_cycle: u64 = 0;
            for (ds.cycles) |cy| {
                const m0 = dc.cache.stats.expert_cache_misses;
                for (cy, 0..) |ids, s| {
                    var buf: [expert_policy.max_route_ids]u16 = undefined;
                    var first: [expert_policy.max_route_ids]bool = undefined;
                    for (ids, 0..) |e, i| {
                        buf[i] = @intCast(dc.geom.offset_of[s] + e);
                        const g = s * ds.experts_per_stage + e;
                        first[i] = !seen[g];
                    }
                    // Unique first uses in this route: each a compulsory miss.
                    for (ids, 0..) |e, i| if (first[i]) {
                        const g = s * ds.experts_per_stage + e;
                        if (!seen[g]) compulsory += 1;
                        seen[g] = true;
                    };
                    var slots: [expert_policy.max_route_ids]u32 = undefined;
                    try dc.cache.route(dc.geom.group_of[s], buf[0..ids.len], slots[0..ids.len]);
                }
                max_cycle = @max(max_cycle, dc.cache.stats.expert_cache_misses - m0);
            }
            const misses = dc.cache.stats.expert_cache_misses;
            const cyc: f64 = @floatFromInt(n_cycles);
            const mpc = @as(f64, @floatFromInt(misses)) / cyc;
            const gained = (if (form == .shared) rw.shared else rw.per_stage) - stock_rows;
            const net = 1.41 * mpc + 0.55 - 0.65 * gained;
            std.debug.print("DSV41_DRAFT_REPLAY {{\"pool\": \"{t}\", \"hot\": {d}, \"misses\": {d}, \"compulsory\": {d}, \"capacity\": {d}, \"misses_per_cycle\": {d:.3}, \"compulsory_per_cycle\": {d:.3}, \"capacity_per_cycle\": {d:.3}, \"max_misses_cycle\": {d}, \"rows_gained\": {d}, \"net_ms_per_cycle\": {d:.3}}}\n", .{ form, rw.hot, misses, compulsory, misses - compulsory, mpc, @as(f64, @floatFromInt(compulsory)) / cyc, @as(f64, @floatFromInt(misses - compulsory)) / cyc, max_cycle, gained, net });
        };
    }
}

test "dsv41 dspark head: HEAD_MODE mxfp8 on RCPROJ runs the block's head pass on the model's head route, refused unbound" {
    const a = testing.allocator;
    const TraceOps = ops.TraceOps;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    var g = TraceOps.init(a);
    defer g.deinit();
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
    const H = Head(TraceOps);
    const rt: graph.Routes = .{ .rc_draft = true, .head = .mxfp8, .rc_head_mxfp8 = true, .draft_rows = graph.draft_compile_max_rows };
    try testing.expectError(error.DraftNeedsHeadMx, H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg }));
    const vocab: c_int = @intCast(c.vocab_size);
    const qw = try g.input(&.{ vocab, 5120 / 4 }, .uint32);
    const qs = try g.input(&.{ vocab, 5120 / 32 }, .uint8);
    var hm = try kr.HeadMx(TraceOps).init(&g, &reg, qw, qs, null);
    defer hm.deinit(&g);
    const h = try H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg, .head_mx = &hm });
    defer h.deinit(&g);
    const caches = try arena.allocator().alloc(H.Cache, h.nStages());
    for (caches) |*x| x.* = .{};
    defer for (caches) |*x| x.deinit(&g);
    const table = try g.input(&.{ vocab, 5120 }, .bfloat16);
    try h.seedMain(&g, try g.input(&.{ 1, 3, 15360 }, .float32), caches);
    const l0 = g.launched.items.len;
    const n0 = g.nodes.items.len;
    const out = try h.draftBlock(&g, try g.input(&.{ 1, 1, 15360 }, .bfloat16), 7, caches, .{ .table = table }, .{ .mxfp8 = .{ .w = qw, .s = qs, .mode = .mxfp8 } });
    try testing.expect(g.shapeOf(out.logits).eql(ops.Shape.of(&.{ 1, 5, vocab })));
    // The head pass is the one extra RCPROJ launch (DRAFTRC's 3 + 3 bf16 calls, then the head at 5 rows); no MLX qmm.
    for (g.nodes.items[n0..]) |nd| try testing.expect(nd.op != .qmm);
    try testing.expectEqual(@as(usize, 3 + 3 + 1), g.launchesOf(l0, .q3rc_mxfp8_fma));
}

/// One recorded draft route stream (pass3bz's receipts, `draft_route_stream`): per cycle, per stage, the block's ids.
const RouteStream = struct { stages: u32, experts_per_stage: u32, cycles: []const []const []const u16 };

/// Total draft misses of `stream` through a shared pool at `hot` under `policy`, from an empty cache (no seed).
fn replayMisses(a: std.mem.Allocator, c: *const v41.Config, stream: RouteStream, hot: u32, policy: xsc.PolicyKind) !u64 {
    const dc = try DraftCache.planOnlyWith(a, c, hot, .shared, policy);
    defer dc.deinit();
    for (stream.cycles) |cy| for (cy, 0..) |ids, s| {
        var buf: [expert_policy.max_route_ids]u16 = undefined;
        for (ids, 0..) |e, i| buf[i] = @intCast(dc.geom.offset_of[s] + e);
        var slots: [expert_policy.max_route_ids]u32 = undefined;
        try dc.cache.route(0, buf[0..ids.len], slots[0..ids.len]);
    };
    return dc.cache.stats.expert_cache_misses;
}

test "dsv41 dspark head: DRAFTCACHE policies on the recorded draft routes reproduce the decode lane's replay (shipped and LRU, shared, from empty)" {
    const a = testing.allocator;
    const json = try v41.testConfigJson(a, .real);
    defer a.free(json);
    const c = try v41.Config.parse(a, json, null);
    const Want = struct { text: []const u8, hot: u32, shipped: u64, lru: u64 };
    const fastest = @embedFile("../fixtures/dsv41_draft_routes_fastest_20261001.json");
    const standard = @embedFile("../fixtures/dsv41_draft_routes_standard_20261001.json");
    // The decode lane's replay (decode note sec. 35): the Python oracle of the streamer's policy, LRU at H + 15.
    for ([_]Want{
        .{ .text = fastest, .hot = 96, .shipped = 246, .lru = 195 },
        .{ .text = fastest, .hot = 128, .shipped = 167, .lru = 156 },
        .{ .text = standard, .hot = 96, .shipped = 424, .lru = 334 },
        .{ .text = standard, .hot = 128, .shipped = 259, .lru = 229 },
    }) |w| {
        const parsed = try std.json.parseFromSlice(RouteStream, a, w.text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const shipped = try replayMisses(a, &c, parsed.value, w.hot, .shipped);
        const lru = try replayMisses(a, &c, parsed.value, w.hot, .lru);
        std.debug.print("DSV41_DRAFT_POLICY_REPLAY {{\"cycles\": {d}, \"hot\": {d}, \"shipped\": {d}, \"lru\": {d}}}\n", .{ parsed.value.cycles.len, w.hot, shipped, lru });
        try testing.expectEqual(w.shipped, shipped);
        try testing.expectEqual(w.lru, lru);
    }
}
