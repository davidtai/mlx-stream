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
                inline for (.{ "w1", "w3", "w2" }) |name| {
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

        /// Profile builds: a stage's routed-ids slot (`Draft.stage_ids`); void elsewhere.
        const Capture = if (draft_routes.enabled) ?*?T else void;
        const no_capture: Capture = if (draft_routes.enabled) null else {};

        fn keepIds(capture: Capture, routed_ids: T) void {
            if (comptime draft_routes.enabled) if (capture) |c| {
                c.* = routed_ids;
            };
        }

        fn sourceOf(self: *const Self, st: *const Stage, capture: Capture) Resident {
            return .{ .ex = &st.experts, .limit = self.c.swiglu_limit, .lut = st.lut, .capture = capture };
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
                const mo = try Tr.moe(g, graph.NoProbe{}, &self.mc, &self.stage_rt, lk, w, f[0], self.sourceOf(st, capture));
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
            const mo = try Tr.moe(g, graph.NoProbe{}, &self.mc, &self.stage_rt, .{}, w, f[0], self.sourceOf(st, capture));
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
            std.debug.assert(ds.block_size <= out.len); // the config bounds dspark_block_size to [1, 64]
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
            std.debug.assert(ds.block_size <= ids.len);
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
pub const SubsetDiag = @import("sdk").Diag;

fn refuse(diag: ?*SubsetDiag, err: SubsetError, comptime fmt: []const u8, args: anytype) SubsetError {
    if (diag) |d| d.set(fmt, args);
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

test "dsv41 dspark head: draft routes (profile builds): the expert source hands the block each stage's router ids" {
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
    {
        var g = TraceOps.init(a);
        defer g.deinit();
        const lookup: mdl.SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(arena.allocator(), &c) };
        const table = try g.input(&.{ @intCast(c.vocab_size), 5120 }, .bfloat16);
        var script: ScriptIds = .{ .prng = std.Random.DefaultPrng.init(5), .n_experts = 128, .k = 3 };
        g.host_values = script.values();
        const h = try H.initWith(a, &g, c, rt, &lookup, .{ .registry = &reg });
        defer h.deinit(&g);
        const caches = try arena.allocator().alloc(H.Cache, h.nStages());
        for (caches) |*x| x.* = .{};
        defer for (caches) |*x| x.deinit(&g);
        try h.seedMain(&g, try g.input(&.{ 1, 3, 15360 }, .float32), caches);
        var d = try h.draftBlock(&g, try g.input(&.{ 1, 1, 15360 }, .bfloat16), 7, caches, .{ .table = table }, .{ .dense = table });
        for (d.stage_ids[0..c.dspark.n_stages]) |k| {
            const x = k orelse return error.TestUnexpectedResult;
            // the router's [5, 3] int32 ids, not a host array
            try testing.expect(g.shapeOf(x).eql(ops.Shape.of(&.{ 5, 3 })));
            try testing.expectEqual(ops.Dtype.int32, g.dtypeOf(x));
            try testing.expect(g.nodes.items[x].op != .host);
        }
        draft_routes.drop(TraceOps, &g, &d.stage_ids);
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
