//! DeepSeek-V4.1 trunk modules as graph builders over an op
//! backend (`deepseek_v41_ops.zig`). Each function transliterates our Python
//! stock eager path (`mtplx/models/deepseek_v41.py`, `_moe`, `_cache`, every
//! MTPLX_DSV41_* lever off) op for op: same ops, order, dtypes and the same
//! `mx.compile` regions, so the Python runtime stays a bitwise oracle.
//! The routed experts are a caller-supplied `routed` source (the expert
//! streamer when served; a stand-in or the Python dump's output in parity
//! runs); everything else is here.

const std = @import("std");
const sdk = @import("sdk");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const kvc = @import("deepseek_v41_cache.zig");
const xk = @import("exl3_kernels.zig");
const kr = @import("dsv41_kernel_routes.zig");

const Dtype = ops.Dtype;

pub fn Q(comptime T: type) type {
    return struct { w: T, s: T, mode: sdk.QuantMode = .mxfp8 };
}

/// One trunk layer's residents (handles owned by the weights map).
pub fn LayerW(comptime T: type) type {
    return struct {
        attn_norm: T,
        ffn_norm: T,
        hc_attn_fn: T,
        hc_attn_base: T,
        hc_attn_scale: T,
        hc_ffn_fn: T,
        hc_ffn_base: T,
        hc_ffn_scale: T,
        attn_sink: T,
        q_norm: T,
        kv_norm: T,
        wq_a: Q(T),
        wq_b: Q(T),
        wkv: Q(T),
        wo_a: Q(T),
        wo_b: Q(T),
        /// kv sources: the gated pooling compressor (wgate only when ratio > 1).
        comp: ?struct { wkv: T, wgate: ?T, norm: T } = null,
        /// kv sources: the index-key projection.
        idx_k: ?struct { wk: T, k_norm: T } = null,
        /// index sources: the indexer queries and head weights.
        idx_q: ?struct { wq_b: Q(T), weights_proj: T } = null,
        /// W97: the grouped wo_a dequantized once to f32 `[g, rank, in]`.
        wo_a_dense: ?T = null,
        /// DENSE16 o-projection (`Routes.prefill_oproj`): the two gather_qmm rhs index arrays (uint32), wo_a's
        /// groups `arange(o_groups)` and wo_b's `[0]`, built once at construction and shared by every layer
        /// (Python builds them once too; per call they were 4 of the route's 9 launches).
        oproj_idx: ?[2]T = null,
        /// The prefill core (`Routes.prefill_attn`): the attention sink as its `[1, 1, H, 1]` f32 view, built once at
        /// construction (the checkpoint stores it f32: the cast was already free, the per-call reshape was not).
        sink4: ?T = null,
        gate_w: T,
        gate_bias: T,
        sh_w1: Q(T),
        sh_w2: Q(T),
        sh_w3: Q(T),
        /// DENSE_RC: the shared expert's gate and up stacked `[2 I, H / 4]` (gate rows first), built once by the model;
        /// `sh_w1` / `sh_w3` are then its two halves (views).
        sh_w13: ?Q(T) = null,
    };
}

pub fn EngramW(comptime T: type) type {
    return struct { wkv: Q(T), q_weight: T, k_weight: T };
}

/// What a source layer hands down the stack within ONE forward (Python
/// `SharedAttentionRuntime`); borrowed references, dropped with the forward.
pub fn Shared(comptime T: type) type {
    return struct {
        compress_kv: ?T = null,
        index_k: ?T = null,
        topk_mask: ?T = null,
        candidates: ?T = null,
        /// K30: the index source's selection as `[b, s, k]` row indices (-1 pads).
        selected_idx: ?T = null,
        /// The window attend mask, identical for every layer of one forward
        /// (same positions, window rows and drop offset: K24's memo).
        win_mask: ?T = null,
        win_rows: u32 = 0,
        win_drop: u32 = 0,
        /// The prefill core's window rows `[s, W]` and their validity (`_window_selected_idx`), identical for
        /// every layer of one forward or K16 chunk (same key as the mask's): built by its first layer and reused.
        win_idx: ?T = null,
        win_valid: ?T = null,
        win_sel_rows: u32 = 0,
        win_sel_drop: u32 = 0,
        /// C22 rope (DISPATCH_FUSE): the forward's (or K16 chunk's) RoPE tables per inv_freq family, built by the
        /// family's first layer and reused: every layer reads the same positions and its family's inv_freq
        /// (`Model.invFor`: YaRN when `li.ratio > 0`, else SWA).
        cos_swa: ?T = null,
        sin_swa: ?T = null,
        cos_yarn: ?T = null,
        sin_yarn: ?T = null,
        /// The positions each family's tables were built from (`arrayKey`; 0: none): a forward's positions array is one
        /// handle for all its layers, so a later forward's (a new array) rebuilds them.
        rope_swa_key: usize = 0,
        rope_yarn_key: usize = 0,
    };
}

/// The trunk's construction-time routes: the Python `MTPLX_DSV41_*` levers
/// that are pure MLX. The default is the stock eager path (every lever off);
/// kernel levers are refused where the routes are built (`deepseek_v41_routes.zig`).
pub const Routes = struct {
    /// C12: the RCTAIL sinkhorn (the 16-lane kernel up to 32 matrices, the stock K3 text
    /// above; bitwise the stock op chain, manifest note), bound in `Trunk.Kernels`.
    rc_sinkhorn: bool = false,
    /// C13: the RCTAIL router (split-K gate GEMV + the stock top-k tail) at rows <= 8, bound
    /// per layer in `Trunk.Kernels` (the prefill widths keep the K22 / eager gate: a phase route).
    rc_router: bool = false,
    /// C13: the RCTAIL HC premix (split-K f32 GEMV of the [24, 20480] HC fn) at rows <= 8,
    /// bound per layer for the attn and ffn mixes.
    rc_premix: bool = false,
    /// C14: RCPROJ mxfp8 + woarc at rows <= 8: wq_a, wkv, wq_b, wo_b and the grouped wo_a on
    /// the packed weights (the M-invariant FMA kernel, bf16 in / out), bound per layer; the
    /// prefill widths keep the K22 / eager chain.
    rc_proj: bool = false,
    /// C15: HCTAPE at rows <= 8 (the fused HC combine / collapse / norm tail and the mixes'
    /// split, one weight-free route over the bf16 stream C14 keeps); needs rc_proj.
    rc_tape: bool = false,
    /// A9 K36 (ATTN_FUSED_PROJ): the projection chain's glue at rows <= 8 as four kernels (the
    /// q-latent RMSNorm, KV RMSNorm + k_pe RoPE, query RoPE, the output's inverse RoPE to bf16),
    /// bound per layer over its norm weights; rides C14's route (needs rc_proj).
    rc_fused_proj: bool = false,
    /// C11: the verify head at rows <= 8 on `dsv41_head_m1rows` (the bf16 head, the M = 1
    /// order per row), M 5 / 7 padded to M + 1 (RCTAIL headpad: the kernel's odd-M cliff; every
    /// row is the kernel's M-invariant row); bound by the model over its head weight.
    rc_head: bool = false,
    /// HEAD_MODE mxfp8's head on RCPROJ at <= 8 rows (verify and draft); needs `head == .mxfp8`.
    rc_head_mxfp8: bool = false,
    /// C16 DRAFTRC: the DSpark draft block's RC routes at rows <= 8 (proj at bf16 / f32 x, the HC
    /// tapes at the draft's stream dtypes, the 128-expert router, the premix, the Sinkhorn), bound
    /// by the draft head; the draft head's own head call stays MLX's (RCTAIL drafthead).
    rc_draft: bool = false,
    /// The prefill attention core (ATTNHALF ropefuse): at rows above `attn_compile_max_rows` the
    /// selected-keys attention of a layer as the kernel lane's QK / softmax / PV launches, the query
    /// roped in the QK load and o handed on inverse-roped in the o-LoRA group layout (no gathered
    /// KVg, no eager core); one route per (query dtype, window-store dtype, compressed) kind.
    prefill_attn: bool = false,
    /// The prefill indexer (ATTNHALF idxscore + INDEX_TOPK select): at rows above
    /// `attn_compile_max_rows` an index source's reach-masked scores in one launch and its top
    /// `index_topk` as ascending indices and the mask in one more (no eager einsum / relu / sum,
    /// no argpartition, no mask-to-index argsort).
    prefill_index: bool = false,
    /// The prefill HC norms (ATTN hcnorm: rsqrt of the stream's mean square, and the pre-collapse
    /// + RMSNorm) at rows >= `hc_norm_min_rows`, one kernel each in the stock reduction order.
    prefill_hc: bool = false,
    /// The prefill MoE combine (SMALLK): routed x weights summed over the experts + shared, one f32
    /// kernel at rows above `attn_compile_max_rows`.
    prefill_combine: bool = false,
    /// DENSE16 o-projection after the prefill core: the grouped o-LoRA as one bf16 gather_qmm over
    /// the packed mxfp8 wo_a (one expert per group) and wo_b as a bf16 qmm, widened to f32 (no f32
    /// einsum over the dense f32 wo_a).
    prefill_oproj: bool = false,
    /// kv16-opt: DENSE16 returns wo_b's bf16 gather_qmm product as is. The attention side joins the stream in its dtype
    /// (bf16 under kv16), so the f32 widening and the narrowing after it were a round trip of the same words (bf16 ->
    /// f32 is exact): value-identical, two [rows, 5120] arrays fewer per chunk and layer. false: the f32 widening.
    prefill_oproj_bf16: bool = false,
    /// PREFILL_HOST shared (K16): after the routing barrier each chunk's shared expert (f32) is
    /// started on the GPU before the routed call, so it runs while the host plans the waves;
    /// the combine reads it (the same expression: exact). Billed in the layer-major wave.
    prefill_host_shared: bool = false,
    /// JOINLESS (K16): the combine reads the wide call's unjoined outputs in place (each
    /// assignment's (output, row)); no concatenate / take of the routed rows. Exact vs SMALLK.
    prefill_joinless: bool = false,
    /// HCPOST (Q3_PREFILL_ATTN hcpost): the attention side's HC post above `rc_max_rows` (C15's tape below) as the
    /// compiled `_hc_post_impl` (HcPost's region, the closure K16's ffn combine already runs), not the eager chain;
    /// the single-span pass's MoE-side combine too (`hcPostRoute`; K16's is always compiled). Exact.
    prefill_hc_post: bool = false,
    /// K16: each chunk's layer input stream released at its chunk fence (nothing reads it after: the Half carries the
    /// residual, the tap is settled), not at its routed group's HC post; the routed groups then hold one hc-width
    /// stream. Lifetime only: the same ops.
    input_stream_early_release: bool = false,
    /// K16: each routed group's MoE inputs (the chunks' moe_in, their row views and concat) freed once the wide call's
    /// waves drained and the host-shared experts were issued from them, before the group's final evaluation. Lifetime
    /// only: the same ops. Needs JOINLESS and the host shared experts (the combine then reads neither input).
    prefill_input_release: bool = false,
    /// The shared expert's middle (clamps, silu, product) as C22's compiled SharedMid region at prompt widths (rows
    /// above rc_max_rows); decode widths keep the op chain. The device probe found the region equal
    /// to the op chain word for word at 953 and 183 f32 rows (kbench hcpostx1ff57a9e, SHAREDMIDX).
    prefill_shared_mid: bool = false,
    /// PREFILL_HCPOST: both HC combines at prompt widths (rows above rc_max_rows) in one pass (`kr.HcPostTf32`) instead
    /// of the compiled HcPost region, its words by construction on the region's numerics (needs prefill_hc_post: the
    /// region is the reference, checked at construction); decode widths keep their routes.
    prefill_hcpost: bool = false,
    /// P1's predictor GEMM in the gate's stored bf16 (MLX accumulates in f32) instead of an f32 copy of the gate: the
    /// seed it reads ahead may differ near ties. Exact outputs: the router decides the routes, the predictor only reads.
    predict_bf16: bool = false,
    /// ENGRAM=prefetch (K16): the prompt pass's Engram gathers posted ahead of their layers on the row
    /// source's poster threads (the blocking read's bytes, in its order). Exact.
    engram_posted: bool = false,
    /// C28 MINVARIANT smallm_all at rows <= 8 (the verify / decode forwards): the compressor wkv /
    /// wgate, the indexer wk and weights_proj as MLX's M = 1 GEMV order per row, bound per layer over
    /// its bf16 weights (needs rc_proj: the bf16 stream at these rows; RCTAIL keeps the premix / gate).
    rc_smallm: bool = false,
    /// C29 MINVARIANT mxfp8 rows (m1order) at rows <= 8: the indexer wq_b, the shared expert
    /// (w1 / w3 / w2) and the Engram wkv in the M = 1 mxfp8 qmv order per row (RCPROJ keeps wq_a /
    /// wkv / wq_b / wo_b); bound per layer (the Engram's by the model).
    rc_mxfp8_rows: bool = false,
    /// DENSE_RC (kbench v7 / v7b, rounding-class; needs rc_mxfp8_rows): the shared expert's gate and up as ONE RCPROJ
    /// launch over their stacked weights (`LayerW.sh_w13`); C29's other sites stay on m1rows. Bound at construction.
    dense_rc: bool = false,
    /// C27 INDEX_TOPK=metal at rows <= 8: an index source's select as one dispatch (the prefill
    /// route's kernel at the verify rows).
    rc_index_topk: bool = false,
    /// C23 ATTN_FUSE softmax at rows <= 8: the selected-keys core's scale / mask / sink softmax as
    /// one kernel (one threadgroup per row and head); QK and PV stay MLX's.
    rc_attn_softmax: bool = false,
    /// K30: each query gathers its window rows and the selected compressed rows.
    selected_keys: bool = false,
    /// W50 lean prefill score: the scale folded into q, the sink into the denominator.
    lean_prefill_score: bool = false,
    /// The compiled regions, each at rows <= its bound (0: the route is off; the
    /// on value is the lane's cap): a call tests its rows only.
    /// W97: the K30 core (the selection then padded to index_topk).
    core_rows: u32 = 0,
    /// K22: attention qkv / out prep, gate prefix and MoE combine.
    attn_rows: u32 = 0,
    /// K4: the Hyper-Connection prep and combine.
    hc_rows: u32 = 0,
    /// K35: the layer's small stages as three segments.
    small_rows: u32 = 0,
    /// K33: the DSpark draft stages' pure chains (the draft head applies it).
    draft_rows: u32 = 0,
    /// W97: wo_a dequantized to f32 once at binding (byte-identical).
    wo_a_f32: bool = false,
    head: Head = .f32,

    /// `MTPLX_DSV41_HEAD_MODE`: unset = `head(x.astype(f32))`; bf16 = a bf16
    /// GEMV cast to f32 after; mxfp8 = the head quantized once, `quantized_matmul`.
    pub const Head = enum { f32, bf16, mxfp8 };
};

/// W97's dense f32 grouped wo_a of one attention block (`[g, rank, in]`, f32).
pub fn woaDenseBytes(c: *const v41.Config) u64 {
    return @as(u64, c.o_groups) * c.o_lora_rank * (@as(u64, c.n_heads) * c.head_dim / c.o_groups) * 4;
}

pub const attn_compile_max_rows = 32;
/// The prefill HC norms' first row count (the lane's HC_MIN_ROWS).
pub const hc_norm_min_rows = 32;
/// A forward wider than this releases its score chains inside the layer
/// (`closeScores`): the prefill widths, where a chain's arrays are score-sized;
/// a decode / verify forward keeps no per-layer host calls for it.
pub const score_wave_min_rows = attn_compile_max_rows;
pub const core_compile_max_rows = 8;
pub const hc_compile_max_rows = 7;
pub const small_stages_max_rows = 7;
pub const draft_compile_max_rows = 32;

/// A probe that records nothing (serving).
pub const NoProbe = struct {
    pub fn put(_: NoProbe, _: []const u8, _: anytype) !void {}
};

/// The decode / verify rows the RC routes' kernels take (the Python `MAX_ROWS`); wider
/// forwards (prefill) keep the stock path.
pub const rc_max_rows = 8;

/// The kernel routes one HC mix calls (null: the stock op chain).
pub fn MixKernels(comptime G: type) type {
    return struct {
        sinkhorn: ?*const kr.Sinkhorn(G) = null,
        premix: ?*const kr.Premix(G) = null,
        /// The prefill HC norms by stream dtype (0: bf16, 1: f32; `Routes.prefill_hc`).
        norm: [2]?*const kr.HcNorm(G) = .{ null, null },
    };
}

/// C28 / C29: one layer's M-invariant sites at rows <= 8 (null where the layer has no such weight or
/// the route is off): the compressor projections (cmp_f32 on the ratio-2 layers' f32 x, cmp_bf16 on
/// ratio 1), the indexer wk (the latent's dtype), weights_proj, the indexer wq_b and the shared expert.
pub fn MinvSites(comptime G: type) type {
    return struct {
        const Self = @This();
        cmp_wkv: ?kr.SmallM(G) = null,
        cmp_wgate: ?kr.SmallM(G) = null,
        wk: ?kr.SmallM(G) = null,
        wproj: ?kr.SmallM(G) = null,
        idx_wq_b: ?kr.Mxfp8Rows(G) = null,
        sh_w1: ?kr.Mxfp8Rows(G) = null,
        sh_w3: ?kr.Mxfp8Rows(G) = null,
        sh_w2: ?kr.Mxfp8Rows(G) = null,
        /// DENSE_RC: gate and up in one RCPROJ launch over `LayerW.sh_w13` (sh_w1 / sh_w3 then unbound).
        sh_w13: ?kr.Mxfp8Rows(G) = null,

        pub fn deinit(self: *Self, g: *G) void {
            inline for (.{ &self.cmp_wkv, &self.cmp_wgate, &self.wk, &self.wproj }) |x| if (x.*) |*r| r.deinit(g);
            inline for (.{ &self.idx_wq_b, &self.sh_w1, &self.sh_w3, &self.sh_w2, &self.sh_w13 }) |x| if (x.*) |*r| r.deinit(g);
        }
    };
}

/// DENSE_RC's stacked shared gate | up over every layer: the two mxfp8 weights' codes and e8m0 scales (I x H x 33 / 32
/// each), what the model builds and the checkpoint's originals it drops.
pub fn sharedGateUpBytes(c: *const v41.Config) u64 {
    return @as(u64, c.n_layers) * 2 * @as(u64, c.moe_intermediate_size) * c.hidden_size * 33 / 32;
}

/// C14: one layer's RCPROJ sites (`kr.RcSite` minus the head), over its packed mxfp8 pairs.
pub fn RcProjs(comptime G: type) type {
    return struct {
        const Self = @This();
        wq_a: kr.RcProj(G),
        wkv: kr.RcProj(G),
        wq_b: kr.RcProj(G),
        wo_b: kr.RcProj(G),
        woa: kr.RcProj(G),

        const names = @typeInfo(Self).@"struct".field_names;

        /// A pair off the site's pinned geometry (or not mxfp8) is refused by name.
        pub fn init(g: *G, reg: *const xk.Registry, w: *const LayerW(G.T)) !Self {
            var s: Self = undefined;
            var built: usize = 0;
            errdefer inline for (names, 0..) |name, i| {
                if (i < built) @field(s, name).deinit(g);
            };
            inline for (names) |name| {
                const q: Q(G.T) = if (comptime std.mem.eql(u8, name, "woa")) w.wo_a else @field(w, name);
                if (q.mode != .mxfp8) return error.RcProjGeometry;
                @field(s, name) = kr.RcProj(G).init(g, reg, @field(kr.RcSite, name), q.w, q.s, null) catch |e| return if (e == error.RouteInput) error.RcProjGeometry else e;
                built += 1;
            }
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            inline for (names) |name| @field(self, name).deinit(g);
        }
    };
}

/// C16: a shared expert's three projections on the draft FMA kernel (w1 / w3 at site shared_w13,
/// w2 at shared_w2), at the call site's x dtype.
pub fn SharedRc(comptime G: type) type {
    return struct {
        w1: kr.DraftProj(G),
        w3: kr.DraftProj(G),
        w2: kr.DraftProj(G),

        pub fn init(g: *G, reg: *const xk.Registry, x_dtype: Dtype, w: *const LayerW(G.T)) !@This() {
            var w1 = try kr.DraftProj(G).init(g, reg, .shared_w13, x_dtype, w.sh_w1.w, w.sh_w1.s, null);
            errdefer w1.deinit(g);
            var w3 = try kr.DraftProj(G).init(g, reg, .shared_w13, x_dtype, w.sh_w3.w, w.sh_w3.s, null);
            errdefer w3.deinit(g);
            return .{ .w1 = w1, .w3 = w3, .w2 = try kr.DraftProj(G).init(g, reg, .shared_w2, x_dtype, w.sh_w2.w, w.sh_w2.s, null) };
        }

        pub fn deinit(self: *@This(), g: *G) void {
            self.w1.deinit(g);
            self.w3.deinit(g);
            self.w2.deinit(g);
        }
    };
}

/// The rounding-class tier's kernel routes one layer calls (null: the stock op chain):
/// a view of the model's `Trunk(G).Kernels`, bound once at construction.
pub fn LayerKernels(comptime G: type) type {
    return struct {
        const Self = @This();
        sinkhorn: ?*const kr.Sinkhorn(G) = null,
        router: ?*const kr.Router(G) = null,
        premix_attn: ?*const kr.Premix(G) = null,
        premix_ffn: ?*const kr.Premix(G) = null,
        proj: ?*const RcProjs(G) = null,
        /// The HC tape (C15) of the ffn prep and the MoE-side combine, and of the attention prep
        /// unless `tape_attn` names another (the draft's stage 0: a bf16 stream into its first prep).
        tape: ?*const kr.HcTape(G) = null,
        tape_attn: ?*const kr.HcTape(G) = null,
        /// The ffn prep's fused call on an f32 x over a bf16 residual (the draft's stage 0).
        tape_mixed: ?*const kr.HcTapeMixed(G) = null,
        fused: ?*const kr.FusedProj(G) = null,
        /// The prefill attention core of this layer's kind (`Routes.prefill_attn`).
        prefill_attn: ?*const kr.PrefillAttn(G) = null,
        /// The prefill indexer's score and select (`Routes.prefill_index`; one of each per trunk).
        idx_score: ?*const kr.IdxScore(G) = null,
        index_topk: ?*const kr.IndexTopk(G) = null,
        /// The prefill HC norms by stream dtype (0: bf16, 1: f32) and the prefill combine.
        hc_norm: [2]?*const kr.HcNorm(G) = .{ null, null },
        combine: ?*const kr.SmallKCombine(G) = null,
        joinless: ?*const kr.JoinlessCombine(G) = null,
        /// PREFILL_HCPOST's one-pass combine (both sites, prompt widths).
        hcpost: ?*const kr.HcPostTf32(G) = null,
        /// C27-C29 / C23 at rows <= 8: this layer's M-invariant sites, the select, the softmax.
        minv: ?*const MinvSites(G) = null,
        decode_topk: ?*const kr.IndexTopk(G) = null,
        attn_softmax: ?*const kr.AttnSoftmax(G) = null,
        /// C16: the shared expert's projections (the draft's; the trunk's shared expert is stock).
        shared: ?*const SharedRc(G) = null,

        pub fn attnMix(self: Self) MixKernels(G) {
            return .{ .sinkhorn = self.sinkhorn, .premix = self.premix_attn, .norm = self.hc_norm };
        }

        pub fn ffnMix(self: Self) MixKernels(G) {
            return .{ .sinkhorn = self.sinkhorn, .premix = self.premix_ffn, .norm = self.hc_norm };
        }
    };
}

pub fn Trunk(comptime G: type) type {
    return struct {
        pub const T = G.T;
        pub const W = LayerW(T);
        pub const Cache = kvc.LayerState(G);
        pub const Share = Shared(T);
        pub const Mixes = struct { pre: T, post: T, comb: T };
        pub const CosSin = struct { cos: T, sin: T };
        pub const Out = struct { h: T, pre_mix: T };
        pub const LK = LayerKernels(G);
        pub const MK = MixKernels(G);

        /// The kernel routes the tier's RC members bind, over the accepted trunk routes'
        /// registry: built once (a geometry the kernels do not take refused by name), then
        /// read by each layer through `at`.
        pub const Kernels = struct {
            gpa: ?std.mem.Allocator = null,
            sinkhorn: ?kr.Sinkhorn(G) = null,
            /// C13, per layer: the router over its gate, the attn / ffn premixes over its HC fns.
            router: std.ArrayList(kr.Router(G)) = .empty,
            premix: std.ArrayList([2]kr.Premix(G)) = .empty,
            /// C14, per layer: the five RCPROJ sites.
            proj: std.ArrayList(RcProjs(G)) = .empty,
            /// C15: the HC tail over the bf16 stream (every layer; weights are call inputs).
            tape: ?kr.HcTape(G) = null,
            /// A9, per layer: the fused projection glue over its q / kv norm weights.
            fused: std.ArrayList(kr.FusedProj(G)) = .empty,
            /// The prefill attention cores by kind (0: layer 0's bf16 query and window, window only;
            /// 1: f32, window only; 2: f32, compressed) and each layer's kind.
            prefill_attn: [3]?kr.PrefillAttn(G) = .{ null, null, null },
            prefill_attn_kind: [v41.max_layers]u8 = @splat(0),
            /// The prefill indexer's score and select.
            idx_score: ?kr.IdxScore(G) = null,
            index_topk: ?kr.IndexTopk(G) = null,
            hc_norm: [2]?kr.HcNorm(G) = .{ null, null },
            combine: ?kr.SmallKCombine(G) = null,
            joinless: ?kr.JoinlessCombine(G) = null,
            hcpost: ?kr.HcPostTf32(G) = null,
            /// C28 / C29, per layer: the M-invariant sites; C27 the verify select; C23 the softmax.
            minv: std.ArrayList(MinvSites(G)) = .empty,
            decode_topk: ?kr.IndexTopk(G) = null,
            attn_softmax: ?kr.AttnSoftmax(G) = null,

            pub fn needed(rt: *const Routes) bool {
                return rt.rc_sinkhorn or rt.rc_router or rt.rc_premix or rt.rc_proj or rt.rc_tape or rt.rc_fused_proj or rt.rc_head or rt.prefill_attn or rt.prefill_index or rt.prefill_hc or rt.prefill_combine or rt.prefill_joinless or rt.prefill_hcpost or rt.rc_smallm or rt.rc_mxfp8_rows or rt.rc_index_topk or rt.rc_attn_softmax;
            }

            /// `layers`: the model's bound layer weights (the router and premix routes keep
            /// references to each layer's gate and HC fns; nothing is copied).
            pub fn init(gpa: std.mem.Allocator, g: *G, reg: *const xk.Registry, c: *const v41.Config, rt: *const Routes, layers: []const W) !Kernels {
                var k: Kernels = .{ .gpa = gpa };
                errdefer k.deinit(g);
                if (rt.rc_sinkhorn) {
                    // q3dk_sinkhorn16_hc4_it20 / mtplx_dsv4_sinkhorn_hc4_it20: hc 4, 20 iterations, eps 1e-6 baked in.
                    if (c.hc_mult != 4 or c.hc_sinkhorn_iters != 20 or @as(f32, @floatCast(c.hc_eps)) != @as(f32, 1e-6)) return error.SinkhornGeometry;
                    k.sinkhorn = try kr.Sinkhorn(G).init(g, reg);
                }
                if (rt.rc_router) {
                    // q3rc_router_tail bakes 384 experts over 5120, sqrt(softplus) / temp 1, top-6,
                    // normalised, x 1.5 (the Python `router_config_check` constants).
                    if (c.n_routed_experts != 384 or c.hidden_size != 5120 or c.n_experts_per_tok != 6 or !c.norm_topk_prob or c.routed_scaling_factor != 1.5) return error.RouterGeometry;
                    try k.router.ensureTotalCapacity(gpa, layers.len);
                    for (layers) |*w| k.router.appendAssumeCapacity(try kr.Router(G).init(g, reg, w.gate_w, w.gate_bias, null));
                }
                if (rt.rc_premix) {
                    // q3rc_premix_part / _fin: N (2 + hc) hc = 24 over K hc x hidden = 20480.
                    if (c.hcMix() != 24 or c.hc_mult * c.hidden_size != 20480) return error.PremixGeometry;
                    try k.premix.ensureTotalCapacity(gpa, layers.len);
                    for (layers) |*w| {
                        var pa = try kr.Premix(G).init(g, reg, w.hc_attn_fn, null);
                        errdefer pa.deinit(g);
                        k.premix.appendAssumeCapacity(.{ pa, try kr.Premix(G).init(g, reg, w.hc_ffn_fn, null) });
                    }
                }
                if (rt.rc_proj) {
                    try k.proj.ensureTotalCapacity(gpa, layers.len);
                    for (layers) |*w| k.proj.appendAssumeCapacity(try RcProjs(G).init(g, reg, w));
                }
                if (rt.rc_tape) {
                    // The q3ht texts take the bf16 stream (OT), which the RCPROJ outputs keep at
                    // these rows; without them the stream turns f32 after the first attention.
                    if (!rt.rc_proj) return error.HcTapeStream;
                    // Baked: D 5120, hc 4, RMSNorm eps 1e-20, HC eps 1e-6 (`HC_CONFIG`).
                    if (c.hidden_size != 5120 or c.hc_mult != 4 or @as(f32, @floatCast(c.rms_norm_eps)) != @as(f32, 1e-20) or @as(f32, @floatCast(c.hc_eps)) != @as(f32, 1e-6)) return error.HcTapeGeometry;
                    k.tape = try kr.HcTape(G).init(g, reg, .bfloat16, null);
                }
                if (rt.rc_fused_proj) {
                    // The glue sits between the RCPROJ projections (bf16 in / out at these rows).
                    if (!rt.rc_proj) return error.FusedProjNeedsProj;
                    // Baked: 64 heads x 512, RoPE 64, the q latent 1280 (the kv latent = head_dim).
                    if (c.n_heads != 64 or c.head_dim != 512 or c.rope_head_dim != 64 or c.q_lora_rank != 1280) return error.FusedProjGeometry;
                    try k.fused.ensureTotalCapacity(gpa, layers.len);
                    // The fused RMSNorms bind the model's eps: the route takes the registered 1e-20 only.
                    const eps: f32 = @floatCast(c.rms_norm_eps);
                    if (eps != @as(f32, 1e-20)) return error.FusedProjEps;
                    for (layers) |*w| k.fused.appendAssumeCapacity(kr.FusedProj(G).init(g, reg, w.q_norm, w.kv_norm, eps, null) catch |e| return if (e == error.RouteInput) error.FusedProjGeometry else e);
                }
                if (rt.prefill_attn) {
                    // The kernel lane admits only the geometry its texts were derived for (each field
                    // compared by name, RouteInput): the model's, from its config.
                    const geo = prefillGeometry(c);
                    // The prompt pass's dtypes (kv16): every layer reads the bf16 stream (bf16 query and window rows);
                    // compressed rows are stored bf16 (the reference's `compress_kv_cache`).
                    for (c.layers[0..c.n_layers], 0..) |li, l| {
                        const kind: u8 = if (l == 0) 0 else if (li.ratio > 0) 2 else 1;
                        if (l == 0 and li.ratio > 0) return error.RouteInput;
                        k.prefill_attn_kind[l] = kind;
                        if (k.prefill_attn[kind] == null) {
                            const dt: ops.Dtype = .bfloat16;
                            k.prefill_attn[kind] = try kr.PrefillAttn(G).init(g, reg, &geo, .rope, dt, dt, kind == 2, null);
                        }
                    }
                }
                if (rt.prefill_index) {
                    const geo = prefillGeometry(c);
                    k.idx_score = try kr.IdxScore(G).init(reg, &geo, null);
                    k.index_topk = try kr.IndexTopk(G).init(g, reg, &geo, null);
                }
                if (rt.prefill_hc) {
                    const geo = prefillGeometry(c);
                    const eps: f32 = @floatCast(c.rms_norm_eps);
                    k.hc_norm[0] = try kr.HcNorm(G).init(g, reg, &geo, .bfloat16, eps, null);
                    k.hc_norm[1] = try kr.HcNorm(G).init(g, reg, &geo, .float32, eps, null);
                }
                if (rt.prefill_combine) {
                    const geo = prefillGeometry(c);
                    k.combine = try kr.SmallKCombine(G).init(reg, &geo, null);
                }
                if (rt.prefill_joinless) {
                    const geo = prefillGeometry(c);
                    k.joinless = try kr.JoinlessCombine(G).init(reg, &geo, null);
                }
                if (rt.prefill_hcpost) {
                    if (!rt.prefill_hc_post) return error.HcPostNeedsRegion;
                    const geo = prefillGeometry(c);
                    k.hcpost = try kr.HcPostTf32(G).init(reg, &geo, null);
                }
                if (rt.dense_rc and !rt.rc_mxfp8_rows) return error.DenseRcNeedsRows;
                if (rt.rc_smallm or rt.rc_mxfp8_rows) {
                    // The sites' x is the bf16 stream at these rows (C14 keeps it bf16).
                    if (!rt.rc_proj) return error.MinvNeedsProj;
                    if (layers.len != c.n_layers) return error.MinvLayers;
                    try k.minv.ensureTotalCapacity(gpa, layers.len);
                    for (layers, 0..) |*w, l| {
                        const li = c.layers[l];
                        k.minv.appendAssumeCapacity(.{});
                        const ms = &k.minv.items[l];
                        if (rt.rc_smallm) {
                            if (w.comp) |cp| {
                                const site: kr.SmallMSite = if (li.ratio == 1) .cmp_bf16 else .cmp_f32;
                                ms.cmp_wkv = try kr.SmallM(G).init(g, reg, site, cp.wkv, null);
                                if (cp.wgate) |wg| ms.cmp_wgate = try kr.SmallM(G).init(g, reg, site, wg, null);
                            }
                            if (w.idx_k) |ik| ms.wk = try kr.SmallM(G).init(g, reg, if (li.ratio == 1) .wk_bf16 else .wk_f32, ik.wk, null);
                            if (w.idx_q) |iq| ms.wproj = try kr.SmallM(G).init(g, reg, .wproj, iq.weights_proj, null);
                        }
                        if (rt.dense_rc) {
                            const w13 = w.sh_w13 orelse return error.DenseRcNeedsStack;
                            if (w.idx_q) |iq| ms.idx_wq_b = try m1Site(g, reg, .indexer_wq_b, iq.wq_b);
                            ms.sh_w13 = try rcSite(g, reg, .shared_w1_w3, w13);
                            ms.sh_w2 = try m1Site(g, reg, .shared_w2, w.sh_w2);
                        } else if (rt.rc_mxfp8_rows) {
                            if (w.idx_q) |iq| ms.idx_wq_b = try m1Site(g, reg, .indexer_wq_b, iq.wq_b);
                            ms.sh_w1 = try m1Site(g, reg, .shared_w1_w3, w.sh_w1);
                            ms.sh_w3 = try m1Site(g, reg, .shared_w1_w3, w.sh_w3);
                            ms.sh_w2 = try m1Site(g, reg, .shared_w2, w.sh_w2);
                        }
                    }
                }
                if (rt.rc_index_topk) {
                    const geo = prefillGeometry(c);
                    k.decode_topk = try kr.IndexTopk(G).init(g, reg, &geo, null);
                }
                if (rt.rc_attn_softmax) {
                    // The kernel's static scale is head_dim 512's (0.0441941738); 64 heads per row.
                    if (c.n_heads != kr.AttnSoftmax(G).heads or c.head_dim != 512) return error.AttnSoftmaxGeometry;
                    k.attn_softmax = try kr.AttnSoftmax(G).init(g, reg, null);
                }
                return k;
            }

            pub fn deinit(self: *Kernels, g: *G) void {
                if (self.sinkhorn) |*x| x.deinit(g);
                for (self.router.items) |*x| x.deinit(g);
                for (self.premix.items) |*p| for (p) |*x| x.deinit(g);
                for (self.proj.items) |*x| x.deinit(g);
                if (self.tape) |*x| x.deinit(g);
                for (self.fused.items) |*x| x.deinit(g);
                for (&self.prefill_attn) |*x| if (x.*) |*r| r.deinit(g);
                if (self.index_topk) |*x| x.deinit(g);
                for (&self.hc_norm) |*x| if (x.*) |*r| r.deinit(g);
                for (self.minv.items) |*x| x.deinit(g);
                if (self.decode_topk) |*x| x.deinit(g);
                if (self.attn_softmax) |*x| x.deinit(g);
                if (self.gpa) |a| {
                    self.router.deinit(a);
                    self.premix.deinit(a);
                    self.proj.deinit(a);
                    self.fused.deinit(a);
                    self.minv.deinit(a);
                }
                self.* = .{};
            }

            pub fn at(self: *const Kernels, l: usize) LK {
                return .{
                    .sinkhorn = if (self.sinkhorn) |*x| x else null,
                    .router = if (self.router.items.len > 0) &self.router.items[l] else null,
                    .premix_attn = if (self.premix.items.len > 0) &self.premix.items[l][0] else null,
                    .premix_ffn = if (self.premix.items.len > 0) &self.premix.items[l][1] else null,
                    .proj = if (self.proj.items.len > 0) &self.proj.items[l] else null,
                    .tape = if (self.tape) |*x| x else null,
                    .tape_attn = if (self.tape) |*x| x else null,
                    .fused = if (self.fused.items.len > 0) &self.fused.items[l] else null,
                    .prefill_attn = if (self.prefill_attn[self.prefill_attn_kind[l]]) |*x| x else null,
                    .idx_score = if (self.idx_score) |*x| x else null,
                    .index_topk = if (self.index_topk) |*x| x else null,
                    .hc_norm = .{ if (self.hc_norm[0]) |*x| x else null, if (self.hc_norm[1]) |*x| x else null },
                    .combine = if (self.combine) |*x| x else null,
                    .joinless = if (self.joinless) |*x| x else null,
                    .hcpost = if (self.hcpost) |*x| x else null,
                    .minv = if (self.minv.items.len > 0) &self.minv.items[l] else null,
                    .decode_topk = if (self.decode_topk) |*x| x else null,
                    .attn_softmax = if (self.attn_softmax) |*x| x else null,
                };
            }
        };

        /// The arrays a layer hands the next: its hidden and pre-mix, and every
        /// array of the shared runtime (the published compressed lanes, masks,
        /// selections, the window mask memo).
        const share_arrays = blk: {
            const info = @typeInfo(Share).@"struct";
            var names: []const []const u8 = &.{};
            for (info.field_names, info.field_types) |name, ty| {
                if (ty == ?T) names = names ++ &[_][]const u8{name};
            }
            break :blk names;
        };

        /// One wave per layer (the backend's `mark` / `resetTo`, the kernels'
        /// wave lifecycle). MlxOps holds every op output until a reset, so a
        /// forward reset only at its end holds all its layers' intermediates at
        /// once; the grouped wo_a alone, dequantized per call (bf16 67 MB, its
        /// f32 cast 134 MB), is 8.05 GB over 40 layers (the M3 AR run's MLX peak,
        /// 27.66 GB against 18.3 planned). The Python lane frees each array at
        /// its last use. Before a wave's `resetTo`, `persist` turns what later
        /// layers read into kept handles (a slot then holds its kept handle; a
        /// handle is released when a later wave replaces it); `release` drops
        /// them once the forward's tail holds what it reads.
        pub const Carry = struct {
            h: ?T = null,
            pre_mix: ?T = null,
            shared: [share_arrays.len]?T = @splat(null),

            pub fn persist(self: *Carry, g: *G, h: *T, pre_mix: *T, shared: ?*Share) void {
                persistOne(g, h, &self.h);
                persistOne(g, pre_mix, &self.pre_mix);
                if (shared) |s| self.persistShared(g, s);
            }

            pub fn persistShared(self: *Carry, g: *G, s: *Share) void {
                inline for (share_arrays, 0..) |name, i| persistSlot(g, &@field(s, name), &self.shared[i]);
            }

            pub fn release(self: *Carry, g: *G) void {
                drop(g, &self.h);
                drop(g, &self.pre_mix);
                for (&self.shared) |*k| drop(g, k);
            }

            fn persistOne(g: *G, slot: *T, kept: *?T) void {
                var v: ?T = slot.*;
                persistSlot(g, &v, kept);
                slot.* = v.?;
            }

            fn persistSlot(g: *G, slot: *?T, kept: *?T) void {
                const cur = slot.* orelse return drop(g, kept);
                if (kept.*) |k| if (std.meta.eql(k, cur)) return;
                const fresh = g.keep(cur);
                drop(g, kept);
                kept.* = fresh;
                slot.* = fresh;
            }

            fn drop(g: *G, kept: *?T) void {
                if (kept.*) |x| g.release(x);
                kept.* = null;
            }
        };

        /// A score sub-wave (inside a layer's wave): `outs`, built since `m`, survive
        /// as fresh handles; everything else the chain built is released before the
        /// eval, so MLX frees each score-sized array at its last use, as the Python
        /// lane does (at most two score blocks live) instead of holding all of them
        /// to the layer's reset (a 953-row prefill chunk's attention and indexer
        /// arrays are 8 GB each at 16K).
        fn closeScores(g: *G, m: ops.Mark, outs: []const *T) !void {
            std.debug.assert(outs.len <= 3);
            var kept: [3]T = undefined;
            for (outs, 0..) |o, i| kept[i] = g.keep(o.*);
            g.resetTo(m);
            for (outs, 0..) |o, i| o.* = try g.adopt(kept[i]);
        }

        /// A weak Python float against `like` (MLX `to_array(v, like.dtype)`).
        fn sf(g: *G, v: f64, like: T) !T {
            const d = g.dtypeOf(like);
            return g.scalar(v, if (ops.isFloat(d)) d else .float32);
        }

        fn sliceLast(g: *G, x: T, lo: c_int, hi: c_int) !T {
            const s = g.shapeOf(x);
            var start: [ops.max_dims]c_int = @splat(0);
            var stop: [ops.max_dims]c_int = undefined;
            const strides: [ops.max_dims]c_int = @splat(1);
            @memcpy(stop[0..s.n], s.slice());
            start[s.n - 1] = lo;
            stop[s.n - 1] = hi;
            return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
        }

        /// `x[..., i]`.
        fn lastIndex(g: *G, x: T, i: c_int) !T {
            const s = g.shapeOf(x);
            return g.reshape(try sliceLast(g, x, i, i + 1), s.d[0 .. s.n - 1]);
        }

        /// `x[i]` of a 1-D array: a 0-d view.
        fn index0(g: *G, x: T, i: c_int) !T {
            return g.reshape(try sliceLast(g, x, i, i + 1), &.{});
        }

        /// Python `_rmsnorm`.
        pub fn rmsnorm(g: *G, x: T, w: T, eps: f64) !T {
            const dt = g.dtypeOf(x);
            var xf = try g.astype(x, .float32);
            const v = try g.mean(try g.square(xf), -1, true);
            xf = try g.mul(xf, try g.rsqrt(try g.add(v, try sf(g, eps, v))));
            return g.astype(try g.mul(try g.astype(w, .float32), xf), dt);
        }

        /// A dense `nn.Linear` without bias: `x @ W.T`.
        pub fn linear(g: *G, x: T, w: T) !T {
            return g.matmul(x, try g.transpose(w));
        }

        pub fn qlinear(g: *G, x: T, q: Q(T)) !T {
            return g.qmm(x, q.w, q.s, q.mode);
        }

        /// Python `_swa_inv_freq`: `1.0 / (rope_theta ** (arange(0, rd, 2) / rd))`.
        pub fn swaInvFreq(g: *G, c: *const v41.Config) !T {
            const rd: f64 = @floatFromInt(c.rope_head_dim);
            const e = try g.div(try g.arange(0, rd, 2, .float32), try g.scalar(rd, .float32));
            const p = try g.power(try g.scalar(c.rope_theta, .float32), e);
            return g.div(try g.scalar(1.0, .float32), p);
        }

        /// Python `_compress_inv_freq` -> V4 `_yarn_inv_freq` at compress_rope_theta.
        pub fn yarnInvFreq(g: *G, c: *const v41.Config) !T {
            const dim: f64 = @floatFromInt(c.rope_head_dim);
            const base = c.compress_rope_theta;
            const e = try g.div(try g.arange(0, dim, 2, .float32), try g.scalar(dim, .float32));
            var freqs = try g.div(try g.scalar(1.0, .float32), try g.power(try g.scalar(base, .float32), e));
            const y = c.yarn;
            const orig: f64 = @floatFromInt(y.original_seq_len);
            const lr = yarnRamp(dim, base, orig, y.beta_fast, y.beta_slow);
            const ramp0 = try g.div(try g.sub(try g.arange(0, dim / 2, 1, .float32), try g.scalar(lr.low, .float32)), try g.scalar(lr.high - lr.low, .float32));
            const ramp = try g.clip(ramp0, try g.scalar(0.0, .float32), try g.scalar(1.0, .float32));
            const smooth = try g.sub(try g.scalar(1.0, .float32), ramp);
            const a = try g.mul(try g.div(freqs, try g.scalar(y.factor, .float32)), try g.sub(try g.scalar(1, .float32), smooth));
            freqs = try g.add(a, try g.mul(freqs, smooth));
            return freqs;
        }

        /// C22 rope (DISPATCH_FUSE): `cosSin` once per forward (or K16 chunk) and inv_freq family, reused by the
        /// family's later layers (the same positions and inv_freq: the same arrays, so the same values).
        fn ropeTables(g: *G, shared: *Share, li: v41.LayerInfo, inv_freq: T, positions: T) !CosSin {
            const yarn = li.ratio > 0;
            const cos_slot = if (yarn) &shared.cos_yarn else &shared.cos_swa;
            const sin_slot = if (yarn) &shared.sin_yarn else &shared.sin_swa;
            const key_slot = if (yarn) &shared.rope_yarn_key else &shared.rope_swa_key;
            const key = arrayKey(positions);
            if (cos_slot.*) |cs| if (key_slot.* == key) return .{ .cos = cs, .sin = sin_slot.*.? };
            const cs = try cosSin(g, inv_freq, positions);
            cos_slot.* = cs.cos;
            sin_slot.* = cs.sin;
            key_slot.* = key;
            return cs;
        }

        /// An array handle's identity (non-zero): the trace backend's node id, MLX's array pointer.
        fn arrayKey(x: T) usize {
            return if (comptime T == u32) @as(usize, x) + 1 else @intFromPtr(x.ctx);
        }

        /// Python `_cos_sin(inv_freq, positions)`.
        pub fn cosSin(g: *G, inv_freq: T, positions: T) !CosSin {
            const ang = try g.mul(try g.expandDims(try g.astype(positions, .float32), 1), try g.expandDims(inv_freq, 0));
            return .{ .cos = try g.cos(ang), .sin = try g.sin(ang) };
        }

        /// V4 `_apply_interleaved_rope`: adjacent pairs rotated in f32, stored at x's dtype.
        pub fn interleavedRope(g: *G, x: T, cs: T, sn: T) !T {
            const shape = g.shapeOf(x);
            const dt = g.dtypeOf(x);
            var pairs = shape;
            pairs.d[shape.n - 1] = @divExact(shape.d[shape.n - 1], 2);
            pairs.d[shape.n] = 2;
            pairs.n += 1;
            const xp = try g.reshape(x, pairs.slice());
            const x0 = try lastIndex(g, xp, 0);
            const x1 = try lastIndex(g, xp, 1);
            const r0 = try g.sub(try g.mul(x0, cs), try g.mul(x1, sn));
            const r1 = try g.add(try g.mul(x0, sn), try g.mul(x1, cs));
            const out = try g.stack(&.{ r0, r1 }, -1);
            return g.astype(try g.reshape(out, shape.slice()), dt);
        }

        /// Python `_rope_last`: RoPE the last `2 * half` dims; `inverse` conjugates.
        pub fn ropeLast(g: *G, x: T, cs: CosSin, inverse: bool) !T {
            const shape = g.shapeOf(x);
            const csh = g.shapeOf(cs.cos);
            const half = csh.dim(-1);
            const rd = half * 2;
            const D = shape.dim(-1);
            const lead = try sliceLast(g, x, 0, D - rd);
            const tail = try sliceLast(g, x, D - rd, D);
            const extra: usize = shape.n - 3;
            var bs: [ops.max_dims]c_int = @splat(1);
            bs[0] = csh.dim(0);
            bs[extra + 1] = half;
            const c = try g.reshape(cs.cos, bs[0 .. extra + 2]);
            var s = try g.reshape(cs.sin, bs[0 .. extra + 2]);
            if (inverse) s = try g.neg(s);
            const roped = try interleavedRope(g, tail, c, s);
            if (D - rd == 0) return roped;
            return g.concat(&.{ lead, roped }, -1);
        }

        /// V4 `_sinkhorn_ops`: row softmax, then alternating column / row normalisation.
        pub fn sinkhorn(g: *G, comb: T, iters: u32, eps: f64) !T {
            var cb = try g.softmax(comb, -1);
            cb = try g.add(cb, try sf(g, eps, cb));
            cb = try g.div(cb, try g.add(try g.sum(cb, -2, true), try sf(g, eps, cb)));
            for (1..iters) |_| {
                cb = try g.div(cb, try g.add(try g.sum(cb, -1, true), try sf(g, eps, cb)));
                cb = try g.div(cb, try g.add(try g.sum(cb, -2, true), try sf(g, eps, cb)));
            }
            return cb;
        }

        /// `DecoderLayer._mixes` + `hc_split_sinkhorn`: the pre / post / Sinkhorn comb mixes.
        pub fn hcMixes(g: *G, c: *const v41.Config, mk: MK, x: T, fnw: T, base: T, scale: T) !Mixes {
            const hc: c_int = @intCast(c.hc_mult);
            const xf = try g.astype(x, .float32);
            const sh = g.shapeOf(xf);
            var fs = sh;
            fs.n -= 1;
            fs.d[fs.n - 1] = hc * sh.dim(-1);
            const flat = try g.reshape(xf, fs.slice());
            const rs = if (hcNormFor(mk.norm, g, x)) |hn| try hn.rsqrt(g, x) else try g.rsqrt(try g.add(try g.mean(try g.square(flat), -1, true), try g.scalar(c.rms_norm_eps, .float32)));
            // C13 hcpremix at <= 8 rows (the kernel's plans; wider: the stock GEMM).
            const mm = if (mk.premix) |k| (if (rowsOf(g, flat, 1) <= rc_max_rows) try k.mm(g, flat) else null) else null;
            const mixes = try g.mul(mm orelse try g.matmul(flat, try g.transpose(try g.astype(fnw, .float32))), rs);
            const total: c_int = @intCast(c.hcMix());
            const eps = c.hc_eps;
            const pre_in = try g.add(try g.mul(try sliceLast(g, mixes, 0, hc), try index0(g, scale, 0)), try sliceLast(g, base, 0, hc));
            const pre = try g.add(try g.sigmoid(pre_in), try sf(g, eps, pre_in));
            const post_in = try g.add(try g.mul(try sliceLast(g, mixes, hc, 2 * hc), try index0(g, scale, 1)), try sliceLast(g, base, hc, 2 * hc));
            const post = try g.mul(try g.scalar(2.0, .float32), try g.sigmoid(post_in));
            var comb = try g.add(try g.mul(try sliceLast(g, mixes, 2 * hc, total), try index0(g, scale, 2)), try sliceLast(g, base, 2 * hc, total));
            const ms = g.shapeOf(comb);
            var cshape = ms;
            cshape.d[ms.n - 1] = hc;
            cshape.d[ms.n] = hc;
            cshape.n += 1;
            comb = try g.reshape(comb, cshape.slice());
            const sk = if (mk.sinkhorn) |k| try k.call(g, comb) else try sinkhorn(g, comb, c.hc_sinkhorn_iters, eps);
            return .{ .pre = pre, .post = post, .comb = sk };
        }

        /// The prefill HC norm for stream `x` [1, S, hc, dim] at S >= hc_norm_min_rows (by its dtype).
        fn hcNormFor(ns: [2]?*const kr.HcNorm(G), g: *G, x: T) ?*const kr.HcNorm(G) {
            const sh = g.shapeOf(x);
            if (sh.n != 4 or sh.d[0] != 1 or sh.d[1] < hc_norm_min_rows) return null;
            return switch (g.dtypeOf(x)) {
                .bfloat16 => ns[0],
                .float32 => ns[1],
                else => null,
            };
        }

        /// `DecoderLayer._hc_pre`: collapse the hc copies with the threaded pre mix.
        pub fn hcPre(g: *G, x: T, pre_mix: T) !T {
            const y = try g.sum(try g.mul(try g.expandDims(pre_mix, -1), try g.astype(x, .float32)), 2, false);
            return g.astype(y, g.dtypeOf(x));
        }

        /// V4 `_hc_post_impl`: `post * x + sum_j comb[j, k] * residual[j]`, at x's dtype.
        pub fn hcPost(g: *G, x: T, residual: T, post: T, comb: T) !T {
            const dt = g.dtypeOf(x);
            const xf = try g.astype(x, .float32);
            const rf = try g.astype(residual, .float32);
            const term = try g.mul(try g.expandDims(post, -1), try g.expandDims(xf, -2));
            const mixed = try g.einsum("...jk,...jd->...kd", &.{ comb, rf });
            return g.astype(try g.add(term, mixed), dt);
        }

        /// `_forward_span` prologue: the embedding rows expanded to hc copies
        /// (a broadcast view) and the identity pre mix.
        pub fn expandEmbedding(g: *G, c: *const v41.Config, rows: T) !Out {
            const s = g.shapeOf(rows);
            const hc: c_int = @intCast(c.hc_mult);
            const h = try g.broadcastTo(try g.expandDims(rows, 2), &.{ s.d[0], s.d[1], hc, s.d[2] });
            const pm = try g.concat(&.{ try g.ones(&.{ s.d[0], s.d[1], 1 }, .float32), try g.zeros(&.{ s.d[0], s.d[1], hc - 1 }, .float32) }, -1);
            return .{ .h = h, .pre_mix = try g.astype(pm, .float32) };
        }

        /// `nn.Embedding`: `weight[ids]`.
        pub fn embed(g: *G, weight: T, ids: T) !T {
            return g.take(weight, ids, 0);
        }

        /// `_forward_span` epilogue: collapse with the last pre mix, then the final norm.
        pub fn finalNorm(g: *G, c: *const v41.Config, h: T, pre_mix: T, norm_w: T) !T {
            const y = try g.sum(try g.mul(try g.expandDims(pre_mix, -1), try g.astype(h, .float32)), 2, false);
            return rmsnorm(g, try g.astype(y, g.dtypeOf(h)), norm_w, c.rms_norm_eps);
        }

        pub const HeadW = union(enum) { dense: T, mxfp8: Q(T) };

        /// `Model._apply_head` under the head route.
        pub fn head(g: *G, rt: *const Routes, x: T, hw: HeadW) !T {
            return switch (rt.head) {
                .f32 => linear(g, try g.astype(x, .float32), hw.dense),
                .bf16 => g.astype(try linear(g, try g.astype(x, g.dtypeOf(hw.dense)), hw.dense), .float32),
                .mxfp8 => g.astype(try qlinear(g, try g.astype(x, .float32), hw.mxfp8), .float32),
            };
        }

        /// C11: the head on `dsv41_head_m1rows` (x [..., K] with <= 8 rows, cast to the head's bf16)
        /// -> f32 logits [..., N]; M 5 / 7 run as M + 1 rows (a zero row appended, first M kept).
        pub fn headRows(g: *G, hr: *const kr.HeadRows(G), x: T) !T {
            const sh = g.shapeOf(x);
            const k = sh.dim(-1);
            const m = rowsOf(g, x, 1);
            var x2 = try g.reshape(try g.astype(x, .bfloat16), &.{ m, k });
            const pad = m == 5 or m == 7;
            if (pad) x2 = try g.concat(&.{ x2, try g.zeros(&.{ 1, k }, .bfloat16) }, 0);
            var y = try hr.call(g, x2);
            const n = g.shapeOf(y).dim(-1);
            if (pad) y = try g.slice(y, &.{ 0, 0 }, &.{ m, n }, &.{ 1, 1 });
            var out = sh;
            out.d[out.n - 1] = n;
            return g.astype(try g.reshape(y, out.slice()), .float32);
        }

        /// HEAD_MODE mxfp8 on RCPROJ (x [..., K] with <= 8 rows, cast to bf16) -> f32 logits [..., N].
        pub fn headMx(g: *G, hm: *const kr.HeadMx(G), x: T) !T {
            const sh = g.shapeOf(x);
            const k = sh.dim(-1);
            const m = rowsOf(g, x, 1);
            const y = try hm.call(g, try g.reshape(try g.astype(x, .bfloat16), &.{ m, k }));
            var out = sh;
            out.d[out.n - 1] = g.shapeOf(y).dim(-1);
            return g.astype(try g.reshape(y, out.slice()), .float32);
        }

        /// `_MXFP8Head.__init__`: the dense head quantized once (mxfp8 gs32).
        pub fn quantizeHead(g: *G, head_w: T) !Q(T) {
            const q = try g.quantize(head_w, .mxfp8);
            return .{ .w = q.w, .s = q.s, .mode = .mxfp8 };
        }

        /// W97 `_o_lora_dense_weight` cached: dequantize, reshape, f32.
        pub fn woaDenseF32(g: *G, c: *const v41.Config, wo_a: Q(T)) !T {
            const d = try g.reshape(try g.dequantize(wo_a.w, wo_a.s, wo_a.mode), &.{ @intCast(c.o_groups), @intCast(c.o_lora_rank), -1 });
            return g.astype(d, .float32);
        }

        /// `Attention._window_attend`: view row j is absolute position `drop + j`.
        fn windowAttend(g: *G, c: *const v41.Config, positions: T, t_len: c_int, drop: u32, b: c_int, s: c_int) !T {
            const wpos = try g.add(try g.scalar(@floatFromInt(drop), .int32), try g.arange(0, @floatFromInt(t_len), 1, .int32));
            const qp = try g.expandDims(positions, 1);
            const wp = try g.expandDims(wpos, 0);
            const inside = try g.logicalAnd(try g.lessEqual(wp, qp), try g.greater(wp, try g.sub(qp, try g.scalar(@floatFromInt(c.window), .int32))));
            return g.broadcastTo(try g.expandDims(inside, 0), &.{ b, s, t_len });
        }

        /// The prefill core's window selection, built by a forward's (a K16 chunk's) first layer and reused.
        fn windowSelection(g: *G, c: *const v41.Config, shared: *Share, positions: T, t_len: c_int, drop: u32) !struct { idx: T, valid: T } {
            if (shared.win_idx) |idx| if (shared.win_sel_rows == t_len and shared.win_sel_drop == drop) return .{ .idx = idx, .valid = shared.win_valid.? };
            const sel = try windowSelectedIdx(g, c, positions, t_len, drop);
            shared.win_idx = sel.idx;
            shared.win_valid = sel.valid;
            shared.win_sel_rows = @intCast(t_len);
            shared.win_sel_drop = drop;
            return .{ .idx = sel.idx, .valid = sel.valid };
        }

        /// The forward's window mask, built by its first layer and reused.
        fn windowMask(g: *G, c: *const v41.Config, shared: *Share, positions: T, t_len: c_int, drop: u32, b: c_int, s: c_int) !T {
            if (shared.win_mask) |m| if (shared.win_rows == t_len and shared.win_drop == drop) return m;
            const m = try windowAttend(g, c, positions, t_len, drop, b, s);
            shared.win_mask = m;
            shared.win_rows = @intCast(t_len);
            shared.win_drop = drop;
            return m;
        }

        /// `_topk_rows`: the k highest per row, ties to the lowest index.
        fn topkRows(g: *G, score: T, k: c_int) !T {
            const n = g.shapeOf(score).dim(-1);
            if (k >= n) return g.greater(score, try sf(g, -std.math.inf(f64), score));
            // ranked[..., k-1] of the descending sort == sorted[..., n-k].
            const thr = try sliceLast(g, try g.sort(score, -1), n - k, n - k + 1);
            const gt = try g.greater(score, thr);
            const eq = try g.equal(score, thr);
            const n_gt = try g.sum(try g.astype(gt, .int32), -1, true);
            const tie_rank = try g.sub(try g.cumsum(try g.astype(eq, .int32), -1), try g.scalar(1, .int32));
            const quota = try g.sub(try g.scalar(@floatFromInt(k), .int32), n_gt);
            return g.logicalOr(gt, try g.logicalAnd(eq, try g.less(tie_rank, quota)));
        }

        /// `_select_candidate_blocks`: the best `topk_blocks` blocks of compressed
        /// positions per query, the query's newest block pinned in.
        fn candidateBlocks(g: *G, c: *const v41.Config, logits: T, compress_lens: T) !T {
            const sh = g.shapeOf(logits);
            const b = sh.d[0];
            const s = sh.d[1];
            const width = sh.d[2];
            const bs: c_int = @intCast(c.candidate_block_size);
            const pad = @mod(-width, bs);
            var l = logits;
            if (pad != 0) {
                const ninf = try sf(g, -std.math.inf(f64), l);
                l = try g.concat(&.{ l, try g.full(&.{ b, s, pad }, ninf, g.dtypeOf(l)) }, -1);
            }
            const nb = @divExact(width + pad, bs);
            const scores0 = try g.max(try g.reshape(l, &.{ b, s, nb, bs }), -1, false);
            const last = try g.floorDiv(try g.sub(compress_lens, try g.scalar(1, .int32)), try g.scalar(@floatFromInt(bs), .int32));
            const blocks = try g.expandDims(try g.expandDims(try g.arange(0, @floatFromInt(nb), 1, .int32), 0), 0);
            const pin = try g.equal(blocks, try g.expandDims(try g.expandDims(last, 0), -1));
            const scores = try g.where(pin, try g.scalar(std.math.inf(f64), g.dtypeOf(scores0)), scores0);
            const kb = @min(@as(c_int, @intCast(c.candidate_topk_blocks)), nb);
            var keep = try topkRows(g, scores, kb);
            keep = try g.logicalAnd(keep, try g.greater(scores, try sf(g, -std.math.inf(f64), scores)));
            return sliceLast(g, try g.repeat(keep, bs, -1), 0, width);
        }

        /// `Indexer.keys`: wk -> k_norm -> RoPE tail of a pre-RoPE latent.
        fn indexerKeys(g: *G, c: *const v41.Config, w: *const W, ms: ?*const MinvSites(G), latent: T, cs: CosSin) !T {
            const s_wk: ?*const kr.SmallM(G) = if (ms) |m| (if (m.wk) |*s| s else null) else null;
            const k = try rmsnorm(g, try smallLinear(g, s_wk, latent, w.idx_k.?.wk), w.idx_k.?.k_norm, c.rms_norm_eps);
            return ropeLast(g, k, cs, false);
        }

        /// `idx` (the prefill indexer only): the selection as ascending indices, -1 padded [1, S, k].
        const Selection = struct { mask: T, cand: ?T, idx: ?T = null };

        /// Each row's reach over `n_comp` compressed rows: [1, S, n_comp] (index < compress_lens).
        fn reachMask(g: *G, compress_lens: T, n_comp: c_int) !T {
            const ar = try g.expandDims(try g.expandDims(try g.arange(0, @floatFromInt(n_comp), 1, .int32), 0), 0);
            return g.less(ar, try g.expandDims(try g.expandDims(compress_lens, 0), -1));
        }

        /// The stock index score: sum_h relu(q_h . k_n) w_h (f32), -inf past each row's reach.
        fn indexScoreStock(g: *G, q: T, index_k: T, wts: T, compress_lens: T, n_comp: c_int) !T {
            var score = try g.einsum("bshd,btd->bsht", &.{ try g.astype(q, .float32), try g.astype(index_k, .float32) });
            score = try g.mul(try g.maximum(score, try sf(g, 0.0, score)), try g.expandDims(try g.astype(wts, .float32), -1));
            score = try g.sum(score, 2, false);
            return g.where(try reachMask(g, compress_lens, n_comp), score, try sf(g, -std.math.inf(f64), score));
        }
        /// The prefill indexer's launches (one prompt row block, b = 1).
        const PrefillIndex = struct { score: *const kr.IdxScore(G), topk: *const kr.IndexTopk(G) };

        /// `Indexer.select`: score the compressed rows, keep the top `index_topk`.
        fn indexerSelect(g: *G, p: anytype, c: *const v41.Config, w: *const W, lk: LK, x: T, qr: T, index_k: T, cs: CosSin, compress_lens: T, n_comp: c_int, candidates: ?T, set_candidates: bool, pi: ?PrefillIndex) !Selection {
            const sh = g.shapeOf(x);
            const iq = w.idx_q.?;
            const H: c_int = @intCast(c.index_n_heads);
            const D: c_int = @intCast(c.index_head_dim);
            const ms = lk.minv;
            const s_wq_b: ?*const kr.Mxfp8Rows(G) = if (ms) |m| (if (m.idx_wq_b) |*s| s else null) else null;
            const s_wproj: ?*const kr.SmallM(G) = if (ms) |m| (if (m.wproj) |*s| s else null) else null;
            var q = try g.reshape(try m1Linear(g, s_wq_b, qr, iq.wq_b), &.{ sh.d[0], sh.d[1], H, D });
            q = try ropeLast(g, q, cs, false);
            const softmax_scale = std.math.pow(f64, @floatFromInt(c.index_head_dim), -0.5);
            const wts0 = try smallLinear(g, s_wproj, x, iq.weights_proj);
            const wts = try g.mul(wts0, try sf(g, softmax_scale * std.math.pow(f64, @floatFromInt(c.index_n_heads), -0.5), wts0));
            if (pi) |ix| {
                // One launch: sum_h relu(q_h . k_n) w_h, -inf past each row's reach.
                var score = try ix.score.call(g, try g.astype(q, .float32), index_k, try g.astype(wts, .float32), compress_lens);
                var cand: ?T = null;
                if (set_candidates) {
                    cand = try candidateBlocks(g, c, score, compress_lens);
                } else if (candidates) |cm| {
                    score = try g.where(cm, score, try sf(g, -std.math.inf(f64), score));
                }
                try p.put("attn.index_score", score);
                const s_ = sh.d[1];
                const r = try ix.topk.select(g, try g.reshape(score, &.{ s_, n_comp }), compress_lens);
                return .{ .mask = try g.reshape(r[1], &.{ 1, s_, n_comp }), .cand = cand, .idx = try g.expandDims(r[0], 0) };
            }
            var score = try indexScoreStock(g, q, index_k, wts, compress_lens, n_comp);
            const reach = try reachMask(g, compress_lens, n_comp);
            var cand: ?T = null;
            if (set_candidates) {
                cand = try candidateBlocks(g, c, score, compress_lens);
            } else if (candidates) |cm| {
                score = try g.where(cm, score, try sf(g, -std.math.inf(f64), score));
            }
            try p.put("attn.index_score", score);
            // C27 at the verify rows (one prompt row block): the select as one dispatch.
            if (lk.decode_topk) |tk| if (sh.d[0] == 1 and sh.d[1] <= rc_max_rows) {
                const r = try tk.select(g, try g.reshape(score, &.{ sh.d[1], n_comp }), compress_lens);
                return .{ .mask = try g.reshape(r[1], &.{ 1, sh.d[1], n_comp }), .cand = cand, .idx = try g.expandDims(r[0], 0) };
            };
            const k = @min(@as(c_int, @intCast(c.index_topk)), n_comp);
            const mask = try g.logicalAnd(try topkRows(g, score, k), reach);
            return .{ .mask = mask, .cand = cand };
        }

        /// `Compressor.pool` + `CompressorState.push`: the normed, pre-RoPE latents
        /// of the groups this call completes (null when none completed).
        fn compressorPool(g: *G, c: *const v41.Config, li: v41.LayerInfo, w: *const W, ms: ?*const MinvSites(G), x: T, cache: *Cache) !?T {
            const comp = w.comp.?;
            const s_wkv: ?*const kr.SmallM(G) = if (ms) |m| (if (m.cmp_wkv) |*s| s else null) else null;
            const s_wgate: ?*const kr.SmallM(G) = if (ms) |m| (if (m.cmp_wgate) |*s| s else null) else null;
            if (li.ratio == 1) return try rmsnorm(g, try smallLinear(g, s_wkv, x, comp.wkv), comp.norm, c.rms_norm_eps);
            const xf = try g.astype(x, .float32);
            const proj = try smallLinear(g, s_wkv, xf, comp.wkv);
            const score = try smallLinear(g, s_wgate, xf, comp.wgate.?);
            const pooled = try cache.frontierPush(g, proj, score) orelse return null;
            return try rmsnorm(g, pooled, comp.norm, c.rms_norm_eps);
        }

        /// `Attention._publish_compressed`: pool, RoPE at group positions, index
        /// keys, append, publish to the forward's shared runtime.
        fn publishCompressed(g: *G, p: anytype, c: *const v41.Config, li: v41.LayerInfo, w: *const W, ms: ?*const MinvSites(G), inv_freq: T, x: T, cache: *Cache, shared: *Share) !void {
            if (try compressorPool(g, c, li, w, ms, x, cache)) |lat| {
                const n_prev: c_int = @intCast(cache.compress.rows());
                const n_new = g.shapeOf(lat).dim(1);
                const gpos = try g.mul(try g.arange(@floatFromInt(n_prev), @floatFromInt(n_prev + n_new), 1, .int32), try g.scalar(@floatFromInt(li.ratio), .int32));
                const gcs = try cosSin(g, inv_freq, gpos);
                const cnew = try ropeLast(g, lat, gcs, false);
                const inew = try indexerKeys(g, c, w, ms, lat, gcs);
                try p.put("attn.comp_latent", lat);
                try p.put("attn.compress_new", cnew);
                try p.put("attn.index_new", inew);
                try cache.compress.append(g, try g.astype(cnew, .bfloat16));
                try cache.index.append(g, try g.astype(inew, .float32));
            }
            shared.compress_kv = try cache.compress.view(g);
            shared.index_k = try cache.index.view(g);
        }

        const Compressed = struct { kv: T, mask: T };

        /// `Attention._compressed`: the CSA2 mode dispatch. Under K30 an index
        /// source also publishes its selection as gather indices.
        fn compressed(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, li: v41.LayerInfo, w: *const W, inv_freq: T, x: T, qr: T, positions: T, cs: CosSin, cache: *Cache, shared: *Share, pi: ?PrefillIndex) !?Compressed {
            if (li.kv_source) try publishCompressed(g, p, c, li, w, lk.minv, inv_freq, x, cache, shared);
            const ckv = shared.compress_kv orelse return null;
            const n_comp = g.shapeOf(ckv).dim(1);
            var mask: T = undefined;
            if (li.index_source) {
                const lens = try g.floorDiv(try g.add(positions, try g.scalar(1, .int32)), try g.scalar(@floatFromInt(li.ratio), .int32));
                const cand = if (li.candidate_source) null else shared.candidates;
                const scores: ?ops.Mark = if (rowsOf(g, x, 1) > score_wave_min_rows) g.mark() else null;
                var sel = try indexerSelect(g, p, c, w, lk, x, qr, shared.index_k.?, cs, lens, n_comp, cand, li.candidate_source, pi);
                if (scores) |m| {
                    if (sel.idx) |*ix| {
                        if (sel.cand) |*cd| try closeScores(g, m, &.{ &sel.mask, cd, ix }) else try closeScores(g, m, &.{ &sel.mask, ix });
                    } else if (sel.cand) |*cd| try closeScores(g, m, &.{ &sel.mask, cd }) else try closeScores(g, m, &.{&sel.mask});
                }
                shared.topk_mask = sel.mask;
                if (li.candidate_source) shared.candidates = sel.cand;
                mask = sel.mask;
                if (rt.selected_keys) {
                    // A fixed-shape core pads the selection to index_topk.
                    const k: c_int = if (rt.core_rows > 0) @intCast(c.index_topk) else @min(@as(c_int, @intCast(c.index_topk)), n_comp);
                    shared.selected_idx = if (sel.idx) |ix| try padIdx(g, ix, k) else try maskToTopkIdx(g, mask, k);
                    try p.put("attn.selected_idx", shared.selected_idx.?);
                }
            } else {
                // The config refuses a reuse layer with no index source before it.
                mask = shared.topk_mask.?;
            }
            try p.put("attn.topk_mask", mask);
            return .{ .kv = ckv, .mask = mask };
        }

        /// `Attention._sparse_attend_oneshot` (f32 score path): one softmax over
        /// window + compressed rows with the per-head value-0 sink column.
        pub fn sparseAttend(g: *G, c: *const v41.Config, w: *const W, q: T, keys: T, attend: T) !T {
            const qs = g.shapeOf(q);
            const H: c_int = qs.d[2];
            const tk = g.shapeOf(keys).dim(1);
            const scale = std.math.pow(f64, @floatFromInt(c.head_dim), -0.5);
            var scores = try g.einsum("bshd,btd->bsht", &.{ try g.astype(q, .float32), try g.astype(keys, .float32) });
            scores = try g.mul(scores, try sf(g, scale, scores));
            scores = try g.where(try g.expandDims(attend, 2), scores, try sf(g, -std.math.inf(f64), scores));
            const sink = try g.reshape(try g.astype(w.attn_sink, .float32), &.{ 1, 1, H, 1 });
            const sink_b = try g.broadcastTo(sink, &.{ qs.d[0], qs.d[1], H, 1 });
            const full = try g.concat(&.{ scores, sink_b }, -1);
            const wts = try sliceLast(g, try g.softmax(full, -1), 0, tk);
            return g.einsum("bsht,btd->bshd", &.{ wts, try g.astype(keys, .float32) });
        }

        /// The value-0 sink softmax with the sink folded into the denominator
        /// (reference `_k_sparse_attn`): `scores` are scaled and masked f32.
        fn sinkSoftmaxPv(g: *G, w: *const W, scores: T, values: T, pv: [:0]const u8) !T {
            const H = g.shapeOf(scores).dim(2);
            const sink = try g.reshape(try g.astype(w.attn_sink, .float32), &.{ 1, 1, H, 1 });
            const m = try g.maximum(try g.max(scores, -1, true), sink);
            const ex = try g.exp(try g.sub(scores, m));
            const denom = try g.add(try g.sum(ex, -1, true), try g.exp(try g.sub(sink, m)));
            return g.div(try g.einsum(pv, &.{ ex, values }), denom);
        }

        /// W50 lean prefill score (`fuse_scale`, `fold_sink`): the scale folded
        /// into q, no sink column. Reassociation-class vs `sparseAttend`.
        fn sparseAttendLean(g: *G, c: *const v41.Config, w: *const W, q: T, keys: T, attend: T) !T {
            const scale = std.math.pow(f64, @floatFromInt(c.head_dim), -0.5);
            const qd = try g.mul(q, try sf(g, scale, q));
            var scores = try g.einsum("bshd,btd->bsht", &.{ try g.astype(qd, .float32), try g.astype(keys, .float32) });
            scores = try g.where(try g.expandDims(attend, 2), scores, try sf(g, -std.math.inf(f64), scores));
            return sinkSoftmaxPv(g, w, scores, try g.astype(keys, .float32), "bsht,btd->bshd");
        }

        /// `_window_selected_idx`: each query's window rows as view indices
        /// `[s, W]` (absolute minus `drop`) and their validity.
        fn windowSelectedIdx(g: *G, c: *const v41.Config, positions: T, t_len: c_int, drop: u32) !struct { idx: T, valid: T } {
            const W_: c_int = @intCast(c.window);
            const qp = try g.reshape(positions, &.{ -1, 1 });
            const base = try g.maximum(try g.sub(qp, try g.scalar(@floatFromInt(W_ - 1), .int32)), try g.scalar(0, .int32));
            const idx = try g.add(base, try g.reshape(try g.arange(0, @floatFromInt(W_), 1, .int32), &.{ 1, W_ }));
            const logical: f64 = @floatFromInt(@as(i64, drop) + t_len);
            const d: f64 = @floatFromInt(drop);
            const valid = try g.logicalAnd(try g.logicalAnd(try g.lessEqual(idx, qp), try g.less(idx, try g.scalar(logical, .int32))), try g.greaterEqual(idx, try g.scalar(d, .int32)));
            return .{ .idx = try g.astype(try g.sub(idx, try g.scalar(d, .int32)), .int32), .valid = valid };
        }

        /// `_mask_to_topk_idx`: the True positions of each row, ascending,
        /// padded with -1 to `k` columns.
        /// The prefill select's min(index_topk, N) indices padded with -1 to `k` (as `maskToTopkIdx`).
        fn padIdx(g: *G, idx: T, k: c_int) !T {
            const sh = g.shapeOf(idx);
            if (sh.d[2] >= k) return idx;
            return g.concat(&.{ idx, try g.full(&.{ sh.d[0], sh.d[1], k - sh.d[2] }, try g.scalar(-1, .int32), .int32) }, -1);
        }

        fn maskToTopkIdx(g: *G, mask: T, k: c_int) !T {
            const sh = g.shapeOf(mask);
            const n = sh.d[2];
            const ar = try g.arange(0, @floatFromInt(n), 1, .int32);
            const keys = try g.where(mask, try g.reshape(ar, &.{ 1, 1, n }), try g.reshape(try g.add(try g.scalar(@floatFromInt(n), .int32), ar), &.{ 1, 1, n }));
            var order = try g.astype(try g.argsort(keys, -1), .int32);
            if (k <= n) {
                order = try sliceLast(g, order, 0, k);
            } else {
                order = try g.concat(&.{ order, try g.full(&.{ sh.d[0], sh.d[1], k - n }, try g.scalar(-1, .int32), .int32) }, -1);
            }
            const count = try g.sum(try g.astype(mask, .int32), -1, true);
            const valid = try g.less(try g.reshape(try g.arange(0, @floatFromInt(k), 1, .int32), &.{ 1, 1, k }), count);
            return g.where(valid, order, try g.scalar(-1, .int32));
        }

        /// `_gather_rows`: `source[b, idx]` as `[b, s, k, d]` (pads read row 0).
        fn gatherRows(g: *G, source: T, idx: T, valid: T) !T {
            const ss = g.shapeOf(source);
            const is = g.shapeOf(idx);
            const b = ss.d[0];
            const n = ss.d[1];
            const idx_c = try g.where(valid, idx, try g.scalar(0, .int32));
            const offs = try g.reshape(try g.mul(try g.arange(0, @floatFromInt(b), 1, .int32), try g.scalar(@floatFromInt(n), .int32)), &.{ b, 1, 1 });
            const flat = try g.reshape(try g.add(idx_c, offs), &.{-1});
            const rows = try g.take(try g.reshape(source, &.{ b * n, ss.d[2] }), flat, 0);
            return g.reshape(rows, &.{ b, is.d[1], is.d[2], ss.d[2] });
        }

        /// K30 `_sparse_attend_selected`: gather each query's window rows and
        /// selected compressed rows, one sink softmax over them (lean casts).
        fn sparseAttendSelected(g: *G, c: *const v41.Config, rt: *const Routes, w: *const W, sm: ?*const kr.AttnSoftmax(G), q: T, window: T, drop: u32, comp_kv: ?T, comp_idx: ?T, positions: T) !T {
            const qs = g.shapeOf(q);
            const b = qs.d[0];
            const s = qs.d[1];
            const W_: c_int = @intCast(c.window);
            const t_len = g.shapeOf(window).dim(1);
            const sel = try windowSelectedIdx(g, c, positions, t_len, drop);
            const win_idx = try g.broadcastTo(try g.expandDims(sel.idx, 0), &.{ b, s, W_ });
            const win_valid = try g.broadcastTo(try g.expandDims(sel.valid, 0), &.{ b, s, W_ });
            var kvg = try gatherRows(g, window, win_idx, win_valid);
            var valid = win_valid;
            if (comp_kv != null and comp_idx != null) {
                const cv = try g.greaterEqual(comp_idx.?, try g.scalar(0, .int32));
                const rows_ = try gatherRows(g, comp_kv.?, comp_idx.?, cv);
                kvg = try g.concat(&.{ kvg, rows_ }, 2);
                valid = try g.concat(&.{ valid, cv }, 2);
            }
            if (b * s <= rt.core_rows) {
                var o: [1]T = undefined;
                try g.tape(AttnCore, c, &.{ q, kvg, valid, w.attn_sink }, &o);
                return o[0];
            }
            if (sm) |k| if (b == 1 and s <= rc_max_rows) return attnCoreFused(g, k, q, kvg, valid, w.attn_sink);
            return attnCore(g, c, q, kvg, valid, w.attn_sink, true);
        }

        /// `_attn_core_impl` (and the eager core of `_sparse_attend_selected`):
        /// QK, mask, value-0 sink softmax, PV, all f32. `lean` casts KVg once.
        /// C23: `attnCore` (lean) with the scale / mask / sink softmax as the ATTN_FUSE kernel: QK
        /// unscaled -> (ex, denom) -> PV / denom (the same expression; one threadgroup per row, head).
        fn attnCoreFused(g: *G, sm: *const kr.AttnSoftmax(G), q: T, kvg: T, valid: T, sink_w: T) !T {
            const kf = try g.astype(kvg, .float32);
            const qk = try g.einsum("bshd,bskd->bshk", &.{ try g.astype(q, .float32), kf });
            const H = g.shapeOf(q).dim(2);
            const sink = try g.reshape(try g.astype(sink_w, .float32), &.{ 1, 1, H, 1 });
            const r = try sm.call(g, qk, valid, sink);
            return g.div(try g.einsum("bshk,bskd->bshd", &.{ r[0], kf }), r[1]);
        }

        fn attnCore(g: *G, c: *const v41.Config, q: T, kvg: T, valid: T, sink_w: T, lean: bool) !T {
            const scale = std.math.pow(f64, @floatFromInt(c.head_dim), -0.5);
            const kf = try g.astype(kvg, .float32);
            var scores = try g.einsum("bshd,bskd->bshk", &.{ try g.astype(q, .float32), kf });
            scores = try g.mul(scores, try sf(g, scale, scores));
            scores = try g.where(try g.expandDims(valid, 2), scores, try sf(g, -std.math.inf(f64), scores));
            const H = g.shapeOf(q).dim(2);
            const sink = try g.reshape(try g.astype(sink_w, .float32), &.{ 1, 1, H, 1 });
            const m = try g.maximum(try g.max(scores, -1, true), sink);
            const ex = try g.exp(try g.sub(scores, m));
            const denom = try g.add(try g.sum(ex, -1, true), try g.exp(try g.sub(sink, m)));
            const values = if (lean) kf else try g.astype(kvg, .float32);
            return g.div(try g.einsum("bshk,bskd->bshd", &.{ ex, values }), denom);
        }

        /// W97 `_attn_core_compiled`: in q, kvg, valid, sink.
        const AttnCore = struct {
            pub const region: ops.Region = .attn_core;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            pub fn run(g: *G, ctx: *const Ctx, in: []const T, out: []T) !void {
                out[0] = try attnCore(g, ctx, in[0], in[1], in[2], in[3], false);
            }
        };

        /// K22 `_attn_qkv_prep_impl`: in x, qcos, qsin, q_norm, kv_norm, then
        /// wq_a, wq_b, wkv as (words, scales); out q, qr, kv_new.
        pub const QkvPrep = struct {
            pub const region: ops.Region = .qkv_prep;
            pub const Ctx = v41.Config;
            pub const n_out = 3;
            pub fn run(g: *G, c: *const Ctx, in: []const T, out: []T) !void {
                const cs: CosSin = .{ .cos = in[1], .sin = in[2] };
                const qr = try rmsnorm(g, try g.qmm(in[0], in[5], in[6], .mxfp8), in[3], c.rms_norm_eps);
                var qs = g.shapeOf(qr);
                qs.d[qs.n - 1] = @intCast(c.n_heads);
                qs.d[qs.n] = @intCast(c.head_dim);
                qs.n += 1;
                out[0] = try ropeLast(g, try g.reshape(try g.qmm(qr, in[7], in[8], .mxfp8), qs.slice()), cs, false);
                out[1] = qr;
                out[2] = try ropeLast(g, try rmsnorm(g, try g.qmm(in[0], in[9], in[10], .mxfp8), in[4], c.rms_norm_eps), cs, false);
            }
        };

        /// K22 `_attn_out_prep_impl`: in o, qcos, qsin, the dense grouped wo_a,
        /// wo_b (words, scales); out the attention output.
        pub const OutPrep = struct {
            pub const region: ops.Region = .out_prep;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            pub fn run(g: *G, ctx: *const Ctx, in: []const T, out: []T) !void {
                out[0] = try outProj(g, ctx, in[0], .{ .cos = in[1], .sin = in[2] }, in[3], .{ .w = in[4], .s = in[5] }, true);
            }
        };

        /// The query-RoPE removal, grouped o-LoRA and `wo_b`. `flat` keeps the
        /// compiled tape's flatten / unflatten pair (the eager body reshapes once).
        pub fn outProj(g: *G, c: *const v41.Config, o0: T, cs: CosSin, w_ol: T, wo_b: Q(T), flat: bool) !T {
            const s0 = g.shapeOf(o0);
            const G_: c_int = @intCast(c.o_groups);
            var o1 = try ropeLast(g, o0, cs, true);
            if (flat) o1 = try g.reshape(o1, &.{ s0.d[0], s0.d[1], s0.d[2] * s0.d[3] });
            o1 = try g.reshape(o1, &.{ s0.d[0], s0.d[1], G_, -1 });
            const o2 = try g.einsum("bsgd,grd->bsgr", &.{ try g.astype(o1, .float32), try g.astype(w_ol, .float32) });
            return qlinear(g, try g.reshape(o2, &.{ s0.d[0], s0.d[1], -1 }), wo_b);
        }

        /// One construction self-check: a route's name and its bool scalar (all within tolerance / equal).
        pub const RouteCheck = struct { name: []const u8, ok: T };

        /// Uniform [-amp, amp) host data at `shape` (f32, cast to `d`), from `scratch`.
        fn checkFill(g: *G, rr: std.Random, buf: []f32, shape: []const c_int, amp: f32, d: Dtype) !T {
            var n: usize = 1;
            for (shape) |x| n *= @intCast(x);
            if (buf.len < n) return error.PrefillCheckScratch;
            for (buf[0..n]) |*v| v.* = (rr.float(f32) * 2 - 1) * amp;
            return g.astype(try g.hostArray(std.mem.sliceAsBytes(buf[0..n]), shape, .float32), d);
        }

        /// |got - want| <= tol x (1 + |want|) everywhere (f32 compare), as one bool scalar.
        fn checkClose(g: *G, got: T, want: T, tol: f64) !T {
            const a = try g.astype(got, .float32);
            const b = try g.astype(want, .float32);
            const err = try g.sub(try g.abs(try g.sub(a, b)), try g.mul(try g.add(try g.abs(b), try sf(g, 1.0, b)), try sf(g, tol, b)));
            return g.lessEqual(try g.max(try g.reshape(err, &.{-1}), 0, false), try sf(g, 0.0, b));
        }

        pub fn checkCloseOf(g: *G, got: T, want: T, tol: f64) !T {
            return checkClose(g, got, want, tol);
        }

        /// Every element equal, as one bool scalar.
        fn checkEqual(g: *G, got: T, want: T) !T {
            var cnt: usize = 1;
            for (g.shapeOf(got).slice()) |d| cnt *= @intCast(d);
            const n: f64 = @floatFromInt(cnt);
            const eq = try g.sum(try g.astype(try g.reshape(try g.equal(got, want), &.{-1}), .int32), 0, false);
            return g.equal(eq, try g.scalar(n, .int32));
        }

        /// The prefill call sites' construction self-checks against the stock chain (the attention
        /// core by kind, the indexer's score and select, the HC norms by stream dtype, the combine),
        /// on deterministic host data at prompt widths; one RouteCheck per installed route into `out`.
        /// Tolerances: the score and the f32 norms / combine 1e-3 x (1 + |stock|) (reduction order),
        /// bf16 norm outputs 2e-2; the select's indices equal the stock top-k on the stock score.
        pub fn prefillRoutesCheck(g: *G, c: *const v41.Config, rt: *const Routes, kx: *const Kernels, layers: []const W, scratch: []f32, out: []RouteCheck) !usize {
            var n: usize = 0;
            const attn = try prefillAttnCheck(g, c, kx, layers, scratch);
            const attn_names = [_][]const u8{ "attention core, layer 0 kind", "attention core, f32 window kind", "attention core, compressed kind" };
            for (attn, attn_names) |ok, name| if (ok) |x| {
                out[n] = .{ .name = name, .ok = x };
                n += 1;
            };
            var rng = std.Random.DefaultPrng.init(0x5eed_d542);
            const r = rng.random();
            const S: c_int = 64;
            if (kx.idx_score) |*sc| {
                // Rows at positions 2048.. of a ratio-2 layer: every row reaches ~1,024 compressed rows
                // (> index_topk, so the select ranks).
                const N: c_int = 1056;
                const IH: c_int = @intCast(c.index_n_heads);
                const ID: c_int = @intCast(c.index_head_dim);
                const q = try checkFill(g, r, scratch, &.{ 1, S, IH, ID }, 1.0, .float32);
                const ik = try checkFill(g, r, scratch, &.{ 1, N, ID }, 1.0, .float32);
                const wts = try checkFill(g, r, scratch, &.{ 1, S, IH }, 0.1, .float32);
                const pos = try g.arange(2048, 2048 + @as(f64, @floatFromInt(S)), 1, .int32);
                const lens = try g.floorDiv(try g.add(pos, try g.scalar(1, .int32)), try g.scalar(2, .int32));
                const want = try indexScoreStock(g, q, ik, wts, lens, N);
                const got = try sc.call(g, q, ik, wts, lens);
                const reach = try reachMask(g, lens, N);
                out[n] = .{ .name = "indexer score", .ok = try checkClose(g, try g.where(reach, got, try sf(g, 0.0, got)), try g.where(reach, want, try sf(g, 0.0, want)), 1e-3) };
                n += 1;
                if (kx.index_topk) |*tk| {
                    const k: c_int = @min(@as(c_int, @intCast(c.index_topk)), N);
                    const stock_idx = try maskToTopkIdx(g, try g.logicalAnd(try topkRows(g, want, k), reach), k);
                    const sel = try tk.select(g, try g.reshape(want, &.{ S, N }), lens);
                    out[n] = .{ .name = "indexer select", .ok = try checkEqual(g, try g.expandDims(sel[0], 0), stock_idx) };
                    n += 1;
                }
            }
            const hc: c_int = @intCast(c.hc_mult);
            const dim: c_int = @intCast(c.hidden_size);
            for (kx.hc_norm, 0..) |maybe, di| if (maybe) |*hn| {
                const dt: Dtype = if (di == 0) .bfloat16 else .float32;
                const h = try checkFill(g, r, scratch, &.{ 1, S, hc, dim }, 1.0, dt);
                const pre = try g.add(try checkFill(g, r, scratch, &.{ 1, S, hc }, 0.5, .float32), try g.scalar(0.5, .float32));
                const flat = try g.reshape(try g.astype(h, .float32), &.{ 1, S, hc * dim });
                const rs = try g.rsqrt(try g.add(try g.mean(try g.square(flat), -1, true), try g.scalar(c.rms_norm_eps, .float32)));
                out[n] = .{ .name = if (di == 0) "HC rsqrt, bf16 stream" else "HC rsqrt, f32 stream", .ok = try checkClose(g, try hn.rsqrt(g, h), rs, 1e-3) };
                n += 1;
                const w = layers[0].attn_norm;
                const want = try rmsnorm(g, try hcPre(g, h, pre), w, c.rms_norm_eps);
                out[n] = .{ .name = if (di == 0) "HC pre-norm, bf16 stream" else "HC pre-norm, f32 stream", .ok = try checkClose(g, try hn.preNorm(g, h, pre, w), want, if (di == 0) 2e-2 else 1e-3) };
                n += 1;
            };
            if (rt.prefill_oproj) {
                // DENSE16 against the f32 grouped o-LoRA on layer 1's weights (bf16 rounding: 3e-2).
                const in_: c_int = @intCast(c.n_heads * c.head_dim / c.o_groups);
                const og = try checkFill(g, r, scratch, &.{ @intCast(c.o_groups), S, in_ }, 1.0, .float32);
                const w = &layers[1];
                out[n] = .{ .name = "o-projection DENSE16", .ok = try checkClose(g, try outProjDense16(g, c, og, w, 1, S, .float32), try outProjGrouped(g, c, og, try woaDense(g, c, w), w.wo_b, 1, S), 3e-2) };
                n += 1;
            }
            if (rt.prefill_hc_post) {
                // HCPOST: the compiled `_hc_post_impl` against the eager chain at a prompt width, bit for bit, on
                // both stream dtypes (the f32 attention output over a bf16 or an f32 residual: the two fused texts).
                const x = try checkFill(g, r, scratch, &.{ 1, S, dim }, 1.0, .float32);
                const post = try checkFill(g, r, scratch, &.{ 1, S, hc }, 1.0, .float32);
                const comb = try checkFill(g, r, scratch, &.{ 1, S, hc, hc }, 0.5, .float32);
                for ([_]Dtype{ .bfloat16, .float32 }, [_][]const u8{ "HC post compiled, bf16 stream", "HC post compiled, f32 stream" }) |dt, name| {
                    const res = try checkFill(g, r, scratch, &.{ 1, S, hc, dim }, 1.0, dt);
                    var o: [1]T = undefined;
                    try g.tape(HcPost, c, &.{ x, res, post, comb }, &o);
                    out[n] = .{ .name = name, .ok = try checkEqual(g, o[0], try hcPost(g, x, res, post, comb)) };
                    n += 1;
                }
            }
            if (kx.joinless) |*jl| {
                // Three outputs (40 + 24 + 64 rows) holding 64 rows x top-k assignments in a
                // deterministic shuffle; against take(concat(outs), inverse) and the stock combine.
                const top: c_int = @intCast(c.n_experts_per_tok);
                const n_as: usize = @intCast(S * top);
                const rows = [_]usize{ 80, 48, n_as - 128 };
                var outs: [3]T = undefined;
                for (&outs, rows) |*o, rr| o.* = try checkFill(g, r, scratch, &.{ @intCast(rr), dim }, 1.0, .float32);
                var perm: [64 * 8]u32 = undefined;
                for (perm[0..n_as], 0..) |*p, q| p.* = @intCast(q);
                r.shuffle(u32, perm[0..n_as]);
                // Assignment perm[j] is joined row j.
                var loc: [64 * 8 * 2]i32 = undefined;
                var inv: [64 * 8]u32 = undefined;
                var j: usize = 0;
                for (rows, 0..) |rr, src| for (0..rr) |row| {
                    loc[2 * perm[j]] = @intCast(src);
                    loc[2 * perm[j] + 1] = @intCast(row);
                    inv[perm[j]] = @intCast(j);
                    j += 1;
                };
                const locT = try g.hostArray(std.mem.sliceAsBytes(loc[0 .. 2 * n_as]), &.{ S, top, 2 }, .int32);
                const invT = try g.hostArray(std.mem.sliceAsBytes(inv[0..n_as]), &.{@intCast(n_as)}, .uint32);
                const ro = try g.reshape(try g.take(try g.concat(&outs, 0), invT, 0), &.{ S, top, dim });
                const wt = try checkFill(g, r, scratch, &.{ S, top }, 1.0, .float32);
                const sh = try checkFill(g, r, scratch, &.{ S, dim }, 1.0, .float32);
                out[n] = .{ .name = "JOINLESS combine", .ok = try checkClose(g, try jl.call(g, &outs, locT, wt, sh), try moeCombine(g, ro, wt, sh), 1e-3) };
                n += 1;
            }
            if (kx.combine) |*cb| {
                const top: c_int = @intCast(c.n_experts_per_tok);
                const ro = try checkFill(g, r, scratch, &.{ S, top, dim }, 1.0, .float32);
                const wt = try checkFill(g, r, scratch, &.{ S, top }, 1.0, .float32);
                const sh = try checkFill(g, r, scratch, &.{ S, dim }, 1.0, .float32);
                out[n] = .{ .name = "MoE combine", .ok = try checkClose(g, try cb.call(g, ro, wt, sh), try moeCombine(g, ro, wt, sh), 1e-3) };
                n += 1;
            }
            if (kx.hcpost) |*hp| {
                // PREFILL_HCPOST: the one-pass combine == the compiled region, every element, on both residual dtypes and on
                // kv16's all-bf16 stream (x and the residual bf16, h bf16).
                for ([_][2]Dtype{ .{ .float32, .float32 }, .{ .float32, .bfloat16 }, .{ .bfloat16, .bfloat16 } }) |dts| {
                    const rdt = dts[1];
                    const x = try checkFill(g, r, scratch, &.{ 1, S, dim }, 4.0, dts[0]);
                    const res = try checkFill(g, r, scratch, &.{ 1, S, hc, dim }, 4.0, rdt);
                    const post = try g.add(try checkFill(g, r, scratch, &.{ 1, S, hc }, 1.0, .float32), try g.scalar(1.0, .float32));
                    const comb = try g.add(try checkFill(g, r, scratch, &.{ 1, S, hc, hc }, 0.5, .float32), try g.scalar(0.5, .float32));
                    var want: [1]T = undefined;
                    try g.tape(HcPost, c, &.{ x, res, post, comb }, &want);
                    out[n] = .{ .name = if (dts[0] == .bfloat16) "HC post, bf16 stream" else if (rdt == .float32) "HC post, f32 residual" else "HC post, bf16 residual", .ok = try checkEqual(g, try hcPostFused(g, hp, x, res, post, comb), want[0]) };
                    n += 1;
                }
            }
            return n;
        }

        /// The verify-row routes' construction self-checks against the stock chain (C28 per site kind,
        /// C29 per site, C27, C23), on deterministic host data at M = 5 rows: the M-invariant GEMVs
        /// against MLX's matmul / qmm (1e-3 x (1 + |stock|) on f32 outputs, 2e-2 on bf16 ones: the
        /// reduction order), the select's indices equal to the stock top-k on the stock score, the
        /// fused softmax core against the lean core (1e-3). One RouteCheck per checked route into `out`.
        pub fn decodeRoutesCheck(g: *G, c: *const v41.Config, kx: *const Kernels, layers: []const W, scratch: []f32, out: []RouteCheck) !usize {
            var n: usize = 0;
            var rng = std.Random.DefaultPrng.init(0x5eed_d543);
            const r = rng.random();
            const M: c_int = 5;
            // C28: the first layer carrying each site kind.
            var seen: std.EnumSet(kr.SmallMSite) = .empty;
            for (kx.minv.items, 0..) |*ms, l| {
                const w = &layers[l];
                const Pair = struct { s: *const ?kr.SmallM(G), w: ?T };
                const pairs = [_]Pair{
                    .{ .s = &ms.cmp_wkv, .w = if (w.comp) |cp| cp.wkv else null },
                    .{ .s = &ms.cmp_wgate, .w = if (w.comp) |cp| cp.wgate else null },
                    .{ .s = &ms.wk, .w = if (w.idx_k) |ik| ik.wk else null },
                    .{ .s = &ms.wproj, .w = if (w.idx_q) |iq| iq.weights_proj else null },
                };
                for (pairs) |pr| if (pr.s.*) |*sm| {
                    if (seen.contains(sm.site)) continue;
                    seen.insert(sm.site);
                    const f32_site = sm.site == .cmp_f32 or sm.site == .wk_f32;
                    const K = g.shapeOf(pr.w.?).dim(1);
                    const x = try checkFill(g, r, scratch, &.{ M, K }, 1.0, if (f32_site) .float32 else .bfloat16);
                    out[n] = .{ .name = @tagName(sm.site), .ok = try checkClose(g, try sm.call(g, x), try linear(g, x, pr.w.?), if (f32_site) 1e-3 else 2e-2) };
                    n += 1;
                };
            }
            // C29: the first layer's sites (every layer carries the shared expert; index sources wq_b).
            var wq_b_done = false;
            for (kx.minv.items, 0..) |*ms, l| {
                const w = &layers[l];
                if (!wq_b_done) if (ms.idx_wq_b) |*s| {
                    const x = try checkFill(g, r, scratch, &.{ M, @intCast(c.q_lora_rank) }, 1.0, .bfloat16);
                    out[n] = .{ .name = "indexer_wq_b", .ok = try checkClose(g, try s.call(g, x), try qlinear(g, x, w.idx_q.?.wq_b), 2e-2) };
                    n += 1;
                    wq_b_done = true;
                };
                if (l == 0) if (ms.sh_w1) |*s1| {
                    const x = try checkFill(g, r, scratch, &.{ M, @intCast(c.hidden_size) }, 1.0, .bfloat16);
                    out[n] = .{ .name = "shared_w1_w3", .ok = try checkClose(g, try s1.call(g, x), try qlinear(g, x, w.sh_w1), 2e-2) };
                    n += 1;
                    const h = try checkFill(g, r, scratch, &.{ M, @intCast(c.moe_intermediate_size) }, 1.0, .bfloat16);
                    out[n] = .{ .name = "shared_w2", .ok = try checkClose(g, try ms.sh_w2.?.call(g, h), try qlinear(g, h, w.sh_w2), 2e-2) };
                    n += 1;
                };
                if (l == 0) if (ms.sh_w13) |*s13| {
                    // DENSE_RC: the stacked launch against the two stock projections side by side.
                    const x = try checkFill(g, r, scratch, &.{ M, @intCast(c.hidden_size) }, 1.0, .bfloat16);
                    const want = try g.concat(&.{ try qlinear(g, x, w.sh_w1), try qlinear(g, x, w.sh_w3) }, -1);
                    out[n] = .{ .name = "shared_w13 rc", .ok = try checkClose(g, try s13.call(g, x), want, 2e-2) };
                    n += 1;
                    const h = try checkFill(g, r, scratch, &.{ M, @intCast(c.moe_intermediate_size) }, 1.0, .bfloat16);
                    out[n] = .{ .name = "shared_w2", .ok = try checkClose(g, try ms.sh_w2.?.call(g, h), try qlinear(g, h, w.sh_w2), 2e-2) };
                    n += 1;
                };
            }
            if (kx.decode_topk) |*tk| {
                const N: c_int = 1056;
                const raw = try checkFill(g, r, scratch, &.{ 1, M, N }, 1.0, .float32);
                const keep = try g.greater(try checkFill(g, r, scratch, &.{ 1, M, N }, 1.0, .float32), try g.scalar(-0.8, .float32));
                const pos = try g.arange(2048, 2048 + @as(f64, @floatFromInt(M)), 1, .int32);
                const lens = try g.floorDiv(try g.add(pos, try g.scalar(1, .int32)), try g.scalar(2, .int32));
                const reach = try reachMask(g, lens, N);
                const score = try g.where(try g.logicalAnd(keep, reach), raw, try sf(g, -std.math.inf(f64), raw));
                const k: c_int = @min(@as(c_int, @intCast(c.index_topk)), N);
                const want = try maskToTopkIdx(g, try g.logicalAnd(try topkRows(g, score, k), reach), k);
                const sel = try tk.select(g, try g.reshape(score, &.{ M, N }), lens);
                out[n] = .{ .name = "verify select", .ok = try checkEqual(g, try g.expandDims(sel[0], 0), want) };
                n += 1;
            }
            if (kx.attn_softmax) |*sm| {
                const H: c_int = @intCast(c.n_heads);
                const D: c_int = @intCast(c.head_dim);
                const Kk: c_int = 640;
                const q = try checkFill(g, r, scratch, &.{ 1, M, H, D }, 0.2, .float32);
                const kvg = try checkFill(g, r, scratch, &.{ 1, M, Kk, D }, 1.0, .bfloat16);
                const valid = try g.greater(try checkFill(g, r, scratch, &.{ 1, M, Kk }, 1.0, .float32), try g.scalar(-0.5, .float32));
                const sink_w = layers[0].attn_sink;
                out[n] = .{ .name = "verify softmax", .ok = try checkClose(g, try attnCoreFused(g, sm, q, kvg, valid, sink_w), try attnCore(g, c, q, kvg, valid, sink_w, true), 1e-3) };
                n += 1;
            }
            return n;
        }

        /// The prefill attention core's construction self-check against the stock chain: per installed
        /// kind, on its first layer's weights, 64 prompt rows over a 64-row window (and 32 compressed
        /// rows, each row selecting the ones its position reaches), deterministic host data. The stock
        /// o (roped q, K30 gather, eager core, the inverse RoPE) and the core's grouped o are compared
        /// elementwise: |core - stock| <= tol x (1 + |stock|), tol 2e-3 on the f32 kinds and 3e-2 on
        /// layer 0's bf16 one (its stock chain rounds q / window rows to bf16 too). Returns one bool
        /// scalar per installed kind (null: not installed); the caller evaluates and reads them once.
        pub fn prefillAttnCheck(g: *G, c: *const v41.Config, kx: *const Kernels, layers: []const W, scratch: []f32) ![3]?T {
            const S: c_int = 64;
            const Nc: c_int = 32;
            const H: c_int = @intCast(c.n_heads);
            const hd: c_int = @intCast(c.head_dim);
            const n_q: usize = @intCast(S * H * hd);
            if (scratch.len < n_q) return error.PrefillAttnCheckScratch;
            var out: [3]?T = .{ null, null, null };
            var rng = std.Random.DefaultPrng.init(0x5eed_d541);
            const r = rng.random();
            const stock_rt: Routes = .{};
            const positions = try g.arange(0, @floatFromInt(S), 1, .int32);
            for (0..3) |kind| {
                const core = if (kx.prefill_attn[kind]) |*x| x else continue;
                var l: usize = 0;
                while (l < c.n_layers and kx.prefill_attn_kind[l] != kind) l += 1;
                if (l == c.n_layers) continue;
                const li = c.layers[l];
                const w = &layers[l];
                const dt: Dtype = .bfloat16;
                const fill = struct {
                    fn f(g_: *G, rr: std.Random, buf: []f32, shape: []const c_int, amp: f32, d: Dtype) !T {
                        var n: usize = 1;
                        for (shape) |x| n *= @intCast(x);
                        for (buf[0..n]) |*v| v.* = (rr.float(f32) * 2 - 1) * amp;
                        return g_.astype(try g_.hostArray(std.mem.sliceAsBytes(buf[0..n]), shape, .float32), d);
                    }
                }.f;
                const q = try fill(g, r, scratch, &.{ 1, S, H, hd }, 4.0, dt);
                const window = try fill(g, r, scratch, &.{ 1, S, hd }, 1.0, dt);
                var cmp: ?[2]T = null;
                if (kind == 2) {
                    const ckv = try fill(g, r, scratch, &.{ 1, Nc, hd }, 1.0, .bfloat16);
                    var idx: [64 * 32]i32 = undefined;
                    for (0..64) |si| for (0..32) |j| {
                        idx[si * 32 + j] = if (j < (si + 1) / 2) @intCast(j) else -1;
                    };
                    cmp = .{ ckv, try g.hostArray(std.mem.sliceAsBytes(&idx), &.{ 1, S, Nc }, .int32) };
                }
                const inv = if (li.ratio > 0) try yarnInvFreq(g, c) else try swaInvFreq(g, c);
                const cs = try cosSin(g, inv, positions);
                // Stock: the roped query, K30's gathered keys, the eager core, the inverse RoPE.
                const o0 = try sparseAttendSelected(g, c, &stock_rt, w, null, try ropeLast(g, q, cs, false), window, 0, if (cmp) |x| x[0] else null, if (cmp) |x| x[1] else null, positions);
                const o1 = try g.astype(try ropeLast(g, o0, cs, true), .float32);
                const want = try g.transposeAxes(try g.reshape(o1, &.{ S, @intCast(c.o_groups), -1 }), &.{ 1, 0, 2 });
                // The core: the un-roped query, the window rows as view indices.
                const sel = try windowSelectedIdx(g, c, positions, S, 0);
                const sink = try g.reshape(try g.astype(w.attn_sink, .float32), &.{ 1, 1, H, 1 });
                const got = try core.attend(g, q, window, sel.idx, sel.valid, cmp, sink, .{ cs.cos, cs.sin });
                const tol: f64 = if (kind == 0) 3e-2 else 2e-3;
                const err = try g.sub(try g.abs(try g.sub(got, want)), try g.mul(try g.add(try g.abs(want), try sf(g, 1.0, want)), try sf(g, tol, want)));
                out[kind] = try g.lessEqual(try g.max(try g.reshape(err, &.{-1}), 0, false), try sf(g, 0.0, want));
            }
            return out;
        }

        /// C29: one mxfp8 site's M-invariant rows route over its packed pair (the mode is the site's).
        pub fn m1Site(g: *G, reg: *const xk.Registry, site: kr.M1Site, q: Q(T)) !kr.Mxfp8Rows(G) {
            if (q.mode != .mxfp8) return error.Mxfp8RowsMode;
            return kr.Mxfp8Rows(G).init(g, reg, site, q.w, q.s, null);
        }

        /// DENSE_RC: one mxfp8 site on RCPROJ (`kr.rcRowsGeom`) over its packed pair.
        pub fn rcSite(g: *G, reg: *const xk.Registry, site: kr.M1Site, q: Q(T)) !kr.Mxfp8Rows(G) {
            if (q.mode != .mxfp8) return error.Mxfp8RowsMode;
            return kr.Mxfp8Rows(G).initRc(g, reg, site, q.w, q.s, null) catch |e| return if (e == error.RouteInput) error.DenseRcGeometry else e;
        }

        /// `linear(x, w)`, at rows <= 8 on the C28 site when bound (x's leading dims flattened to M).
        fn smallLinear(g: *G, site: ?*const kr.SmallM(G), x: T, w: T) !T {
            if (site) |s| {
                const sh = g.shapeOf(x);
                const m = rowsOf(g, x, 1);
                if (m <= rc_max_rows) {
                    var os = sh;
                    os.d[os.n - 1] = g.shapeOf(w).dim(0);
                    return g.reshape(try s.call(g, try g.reshape(x, &.{ m, sh.dim(-1) })), os.slice());
                }
            }
            return linear(g, x, w);
        }

        /// `qlinear(x, q)`, at rows <= 8 on the C29 site when bound.
        fn m1Linear(g: *G, site: ?*const kr.Mxfp8Rows(G), x: T, q: Q(T)) !T {
            if (site) |s| {
                const sh = g.shapeOf(x);
                const m = rowsOf(g, x, 1);
                if (m <= rc_max_rows) {
                    var os = sh;
                    os.d[os.n - 1] = g.shapeOf(q.w).dim(0);
                    return g.reshape(try s.call(g, try g.reshape(x, &.{ m, sh.dim(-1) })), os.slice());
                }
            }
            return qlinear(g, x, q);
        }

        /// The model's prefill geometry, as the kernel lane's routes compare it.
        fn prefillGeometry(c: *const v41.Config) kr.PrefillGeometry {
            return .{ .n_heads = c.n_heads, .head_dim = c.head_dim, .rope_head_dim = c.rope_head_dim, .window = c.window, .index_topk = c.index_topk, .index_n_heads = c.index_n_heads, .index_head_dim = c.index_head_dim, .n_experts_per_tok = c.n_experts_per_tok, .hidden = c.hidden_size, .hc_mult = c.hc_mult };
        }

        /// `W.sink4`: a layer's attention sink as the prefill core's `[1, 1, H, 1]` f32 view.
        pub fn sinkView(g: *G, c: *const v41.Config, attn_sink: T) !T {
            return g.reshape(try g.astype(attn_sink, .float32), &.{ 1, 1, @intCast(c.n_heads), 1 });
        }

        /// `W.oproj_idx`: wo_a's group indices `arange(o_groups)` and wo_b's `[0]`, uint32, built once.
        pub fn oprojIndices(g: *G, c: *const v41.Config) ![2]T {
            return .{
                try g.astype(try g.arange(0, @floatFromInt(c.o_groups), 1, .int32), .uint32),
                try g.astype(try g.arange(0, 1, 1, .int32), .uint32),
            };
        }

        /// DENSE16 `outProj` after the prefill core: og f32 [g, S, in] -> bf16, the grouped o-LoRA as
        /// gather_qmm over wo_a's packed [g, rank, in] view (rhs = arange(g)), [S, g x rank], then
        /// wo_b's qmm in `out_dt` (bf16: the product as is, `Routes.prefill_oproj_bf16`; f32: widened).
        pub fn outProjDense16(g: *G, c: *const v41.Config, og: T, w: *const W, b: c_int, s: c_int, out_dt: Dtype) !T {
            const G_: c_int = @intCast(c.o_groups);
            const R: c_int = @intCast(c.o_lora_rank);
            const ws = g.shapeOf(w.wo_a.w);
            const ss = g.shapeOf(w.wo_a.s);
            const wa = try g.reshape(w.wo_a.w, &.{ G_, R, ws.dim(-1) });
            const sa = try g.reshape(w.wo_a.s, &.{ G_, R, ss.dim(-1) });
            // The rhs indices, built once at construction (`W.oproj_idx`; the route sets them by construction).
            const oi = w.oproj_idx.?;
            const o2 = try g.gatherQmm(try g.astype(og, .bfloat16), wa, sa, oi[0], w.wo_a.mode);
            // [S, g x rank] (the bf16 copy), then wo_b as the lane does: one gather_qmm over its [1, out,
            // in / 4] view with rhs [0] (the NAX gather kernel at the chunk's rows), widened to f32.
            const flat = try g.reshape(try g.transposeAxes(o2, &.{ 1, 0, 2 }), &.{ 1, b * s, G_ * R });
            const bs_ = g.shapeOf(w.wo_b.w);
            const bss = g.shapeOf(w.wo_b.s);
            const wb = try g.reshape(w.wo_b.w, &.{ 1, bs_.dim(0), bs_.dim(1) });
            const sb = try g.reshape(w.wo_b.s, &.{ 1, bss.dim(0), bss.dim(1) });
            const y = try g.gatherQmm(flat, wb, sb, oi[1], w.wo_b.mode);
            return g.astype(try g.reshape(y, &.{ b, s, bs_.dim(0) }), out_dt);
        }

        /// `outProj` after the prefill core: o already inverse-roped as [g, S, in] f32 -> the grouped
        /// o-LoRA (the lane's einsum over its [1, S, g, in] view) and `wo_b`.
        fn outProjGrouped(g: *G, c: *const v41.Config, og: T, w_ol: T, wo_b: Q(T), b: c_int, s: c_int) !T {
            _ = c;
            const o2 = try g.einsum("gsd,grd->sgr", &.{ og, try g.astype(w_ol, .float32) });
            return qlinear(g, try g.reshape(o2, &.{ b, s, -1 }), wo_b);
        }

        /// The grouped `wo_a` as `[g, rank, in]`: bound once in f32 (W97), else
        /// dequantized per call (bf16) as `_o_lora_dense_weight`.
        pub fn woaDense(g: *G, c: *const v41.Config, w: *const W) !T {
            if (w.wo_a_dense) |d| return d;
            return g.reshape(try g.dequantize(w.wo_a.w, w.wo_a.s, w.wo_a.mode), &.{ @intCast(c.o_groups), @intCast(c.o_lora_rank), -1 });
        }

        fn rowsOf(g: *G, x: T, trailing: u8) c_int {
            const s = g.shapeOf(x);
            var r: c_int = 1;
            for (s.d[0 .. s.n - trailing]) |d| r *= d;
            return r;
        }

        /// `Attention._attend`: the projections (K22 tape at rows <= 32), the
        /// window append, masked-full (stock, W50 lean at prefill) or K30
        /// selected-key attention, the output projection.
        pub fn attention(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, li: v41.LayerInfo, w: *const W, inv_freq: T, x: T, positions: T, cache: *Cache, shared: *Share) !T {
            const sh = g.shapeOf(x);
            const b = sh.d[0];
            const s = sh.d[1];
            const H: c_int = @intCast(c.n_heads);
            const hd: c_int = @intCast(c.head_dim);
            // C14 at <= 8 rows: the RCPROJ projections (the stream is bf16 there by construction:
            // every attention and MoE output of the route is bf16); K22 / eager above.
            const rc: ?*const RcProjs(G) = if (b * s <= rc_max_rows) lk.proj else null;
            const compiled = rc == null and b * s <= rt.attn_rows;
            // The prefill attention core at prompt widths (above the compiled regions' rows; K30 keys).
            const pa: ?*const kr.PrefillAttn(G) = if (rc == null and !compiled and b * s > attn_compile_max_rows and rt.selected_keys) lk.prefill_attn else null;
            // The prefill indexer at the same widths (one prompt row block).
            const pi: ?PrefillIndex = if (b == 1 and s > attn_compile_max_rows and lk.idx_score != null) .{ .score = lk.idx_score.?, .topk = lk.index_topk.? } else null;
            const cs = if (b * s > rc_max_rows) try ropeTables(g, shared, li, inv_freq, positions) else try cosSin(g, inv_freq, positions);
            var q: T = undefined;
            var qr: T = undefined;
            var kv_new: T = undefined;
            if (rc) |pj| {
                if (std.debug.runtime_safety) std.debug.assert(g.dtypeOf(x) == .bfloat16);
                const qa = try pj.wq_a.linear(g, x);
                const kva = try pj.wkv.linear(g, x);
                if (lk.fused) |fp| {
                    // A9: the glue as the K36 kernels (bf16 stores).
                    qr = try fp.qNorm(g, qa);
                    q = try fp.ropeHeads(g, try g.reshape(try pj.wq_b.linear(g, qr), &.{ b, s, H, hd }), cs.cos, cs.sin, .fwd);
                    kv_new = try fp.kvNormRope(g, kva, cs.cos, cs.sin);
                } else {
                    qr = try rmsnorm(g, qa, w.q_norm, c.rms_norm_eps);
                    q = try ropeLast(g, try g.reshape(try pj.wq_b.linear(g, qr), &.{ b, s, H, hd }), cs, false);
                    kv_new = try ropeLast(g, try rmsnorm(g, kva, w.kv_norm, c.rms_norm_eps), cs, false);
                }
            } else if (compiled) {
                var o: [3]T = undefined;
                try g.tape(QkvPrep, c, &.{ x, cs.cos, cs.sin, w.q_norm, w.kv_norm, w.wq_a.w, w.wq_a.s, w.wq_b.w, w.wq_b.s, w.wkv.w, w.wkv.s }, &o);
                q = o[0];
                qr = o[1];
                kv_new = o[2];
            } else {
                qr = try rmsnorm(g, try qlinear(g, x, w.wq_a), w.q_norm, c.rms_norm_eps);
                const qb = try g.reshape(try qlinear(g, qr, w.wq_b), &.{ b, s, H, hd });
                // The prefill core ropes q in its QK load: hand it the un-roped query.
                q = if (pa != null) qb else try ropeLast(g, qb, cs, false);
                kv_new = try ropeLast(g, try rmsnorm(g, try qlinear(g, x, w.wkv), w.kv_norm, c.rms_norm_eps), cs, false);
            }
            try p.put("attn.qr", qr);
            try p.put("attn.q", q);
            try p.put("attn.kv_new", kv_new);
            try cache.window.append(g, kv_new);
            const window = (try cache.window.view(g)).?;
            const drop = cache.window.dropOffset();
            var o0: T = undefined;
            if (rt.selected_keys) {
                var ckv: ?T = null;
                var cidx: ?T = null;
                if (li.ratio > 0) if (try compressed(g, p, c, rt, lk, li, w, inv_freq, x, qr, positions, cs, cache, shared, pi)) |comp| {
                    ckv = comp.kv;
                    cidx = shared.selected_idx;
                };
                if (pa) |core| {
                    // o f32 [8, S, 4096]: inverse-roped, in the o-LoRA group layout.
                    const sel = try windowSelection(g, c, shared, positions, g.shapeOf(window).dim(1), drop);
                    const cmp: ?[2]T = if (ckv) |kv| .{ kv, cidx.? } else null;
                    // The sink's [1, 1, H, 1] view, built once at construction (`W.sink4`; the route sets it).
                    const sink = w.sink4.?;
                    // The core's QK scores live inside the score wave, as the eager chain's do.
                    const scores: ?ops.Mark = if (b * s > score_wave_min_rows) g.mark() else null;
                    var og = try core.attend(g, q, window, sel.idx, sel.valid, cmp, sink, .{ cs.cos, cs.sin });
                    if (scores) |m| try closeScores(g, m, &.{&og});
                    try p.put("attn.o_grouped", og);
                    const out = if (rt.prefill_oproj) try outProjDense16(g, c, og, w, b, s, if (rt.prefill_oproj_bf16) .bfloat16 else .float32) else try outProjGrouped(g, c, og, try woaDense(g, c, w), w.wo_b, b, s);
                    try p.put("attn.out", out);
                    return out;
                }
                const scores: ?ops.Mark = if (b * s > score_wave_min_rows) g.mark() else null;
                o0 = try sparseAttendSelected(g, c, rt, w, lk.attn_softmax, q, window, drop, ckv, cidx, positions);
                if (scores) |m| try closeScores(g, m, &.{&o0});
            } else {
                var attend = try windowMask(g, c, shared, positions, g.shapeOf(window).dim(1), drop, b, s);
                var keys = window;
                if (li.ratio > 0) {
                    if (try compressed(g, p, c, rt, lk, li, w, inv_freq, x, qr, positions, cs, cache, shared, pi)) |comp| {
                        keys = try g.concat(&.{ window, comp.kv }, 1);
                        attend = try g.concat(&.{ attend, comp.mask }, -1);
                    }
                }
                const scores: ?ops.Mark = if (b * s > score_wave_min_rows) g.mark() else null;
                o0 = if (s > 1 and rt.lean_prefill_score) try sparseAttendLean(g, c, w, q, keys, attend) else try sparseAttend(g, c, w, q, keys, attend);
                if (scores) |m| try closeScores(g, m, &.{&o0});
            }
            try p.put("attn.o", o0);
            if (rc) |pj| {
                // woarc: the query-RoPE removal to bf16, the grouped wo_a on its packed pair, wo_b.
                const o1 = if (lk.fused) |fp| try fp.ropeHeads(g, o0, cs.cos, cs.sin, .inv) else try g.astype(try ropeLast(g, o0, cs, true), .bfloat16);
                const o2 = try pj.woa.call(g, try g.reshape(o1, &.{ b * s, -1 }));
                const out = try pj.wo_b.linear(g, try g.reshape(o2, &.{ b, s, -1 }));
                try p.put("attn.out", out);
                return out;
            }
            const w_ol = try woaDense(g, c, w);
            const out = if (compiled) blk: {
                var o: [1]T = undefined;
                try g.tape(OutPrep, c, &.{ o0, cs.cos, cs.sin, w_ol, w.wo_b.w, w.wo_b.s }, &o);
                break :blk o[0];
            } else try outProj(g, c, o0, cs, w_ol, w.wo_b, false);
            try p.put("attn.out", out);
            return out;
        }

        pub const Route = struct { weights: T, indices: T };

        /// `Gate.__call__`'s pure prefix: f32 score GEMM / temp, sqrtsoftplus, the
        /// noaux_tc correction bias. Returns the unbiased and the biased scores.
        fn gatePrefix(g: *G, xf: T, gate_w: T, gate_bias: T) ![3]T {
            const logits = try g.div(try linear(g, try g.astype(xf, .float32), try g.astype(gate_w, .float32)), try g.scalar(1.0, .float32));
            const scores = try g.sqrt(try g.softplus(logits));
            return .{ scores, try g.add(scores, gate_bias), logits };
        }

        /// `Gate.__call__`'s selection: top-k of the biased scores in descending
        /// order, weights from the unbiased ones, normalised, x route scale.
        fn gateSelect(g: *G, c: *const v41.Config, scores: T, biased: T) !Route {
            const k: c_int = @intCast(c.n_experts_per_tok);
            const part = try sliceLast(g, try g.argpartition(try g.neg(biased), k - 1, -1), 0, k);
            const order = try g.argsort(try g.neg(try g.takeAlongAxis(biased, part, -1)), -1);
            const indices = try g.astype(try g.takeAlongAxis(part, order, -1), .int32);
            var weights = try g.takeAlongAxis(scores, indices, -1);
            if (c.norm_topk_prob and k > 1) {
                weights = try g.div(weights, try g.add(try g.sum(weights, -1, true), try g.scalar(1e-20, .float32)));
            }
            weights = try g.mul(weights, try g.scalar(c.routed_scaling_factor, .float32));
            return .{ .weights = weights, .indices = indices };
        }

        /// K22 `_gate_prefix_impl`: in xf, gate weight, bias; out scores, biased.
        const GatePrefix = struct {
            pub const region: ops.Region = .gate_prefix;
            pub const Ctx = v41.Config;
            pub const n_out = 2;
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                out[0..2].* = (try gatePrefix(g, in[0], in[1], in[2]))[0..2].*;
            }
        };

        /// `Gate.__call__`: sqrtsoftplus scores, noaux_tc biased selection,
        /// unbiased normalised weights x route scale (the prefix a K22 tape at
        /// rows <= 32; the selection eager, it feeds the routing barrier).
        pub fn router(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, w: *const W, xf: T) !Route {
            if (lk.router) |k| {
                if (g.shapeOf(xf).dim(0) <= rc_max_rows) {
                    // C13 RCTAIL router: the whole gate (split-K logits, the stock tail) at <= 8 rows.
                    const o = try k.gateTopk(g, xf);
                    try p.put("gate.indices", o[1]);
                    try p.put("gate.weights", o[0]);
                    return .{ .weights = o[0], .indices = o[1] };
                }
            }
            var pre: [2]T = undefined;
            if (g.shapeOf(xf).dim(0) <= rt.attn_rows) {
                try g.tape(GatePrefix, c, &.{ xf, w.gate_w, w.gate_bias }, &pre);
            } else {
                const e = try gatePrefix(g, xf, w.gate_w, w.gate_bias);
                pre = e[0..2].*;
                try p.put("gate.logits", e[2]);
                try p.put("gate.scores", e[0]);
            }
            const r = try gateSelect(g, c, pre[0], pre[1]);
            try p.put("gate.indices", r.indices);
            try p.put("gate.weights", r.weights);
            return r;
        }

        /// P1's predictor: layer `w`'s router over its input `h` collapsed with the attention's pre mix and normed as its
        /// MoE input is (attention's part left out), each row's top-k ids unordered, int32 [rows, k]; `wf` = the gate in
        /// f32. It chooses reads only: no output depends on it.
        pub fn predictIds(g: *G, c: *const v41.Config, lk: LK, w: *const W, h: T, pre_mix: T, wf: T) !T {
            const x = if (hcNormFor(lk.hc_norm, g, h)) |hn| try hn.preNorm(g, h, pre_mix, w.ffn_norm) else try rmsnorm(g, try hcPre(g, h, pre_mix), w.ffn_norm, c.rms_norm_eps);
            return predictTopk(g, c, try g.reshape(x, &.{ -1, @as(c_int, @intCast(c.hidden_size)) }), wf, w.gate_bias);
        }

        /// The predictor's selection over `x` [rows, dim]: `gatePrefix`'s biased scores and each row's top-k ids by
        /// `gateSelect`'s partition, unordered, int32 [rows, k]. The GEMM runs in `wf`'s dtype: the gate in f32 (the
        /// router's numerics) or as stored in bf16 (`Routes.predict_bf16`); the scores are f32 either way.
        pub fn predictTopk(g: *G, c: *const v41.Config, x: T, wf: T, bias: T) !T {
            const logits = try g.astype(try linear(g, try g.astype(x, g.dtypeOf(wf)), wf), .float32);
            const biased = try g.add(try g.sqrt(try g.softplus(logits)), bias);
            const k: c_int = @intCast(c.n_experts_per_tok);
            return g.astype(try sliceLast(g, try g.argpartition(try g.neg(biased), k - 1, -1), 0, k), .int32);
        }

        /// The shared expert's elementwise middle (`deepseek_v41_moe.py`:337-346): both projections to f32, the SwiGLU
        /// clamps, silu, the product, back to `dt`.
        fn sharedMid(g: *G, c: *const v41.Config, gate_lin: T, up_lin: T, dt: Dtype) !T {
            var gate = try g.astype(gate_lin, .float32);
            var up = try g.astype(up_lin, .float32);
            if (c.swiglu_limit > 0) {
                up = try g.clip(up, try g.scalar(-c.swiglu_limit, .float32), try g.scalar(c.swiglu_limit, .float32));
                gate = try g.minimum(gate, try g.scalar(c.swiglu_limit, .float32));
            }
            return g.astype(try g.mul(try g.silu(gate), up), dt);
        }

        /// C22 moeshared: `sharedMid` as one compiled region. In: gate, up (the projections' outputs) and the expert's
        /// input x, which only names the output dtype (the region reads none of it).
        pub const SharedMid = struct {
            pub const region: ops.Region = .shared_mid;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            pub fn run(g: *G, c: *const Ctx, in: []const T, out: []T) !void {
                out[0] = try sharedMid(g, c, in[0], in[1], g.dtypeOf(in[2]));
            }
        };

        /// `Expert.__call__` (the shared expert): clamped SwiGLU in f32.
        pub fn sharedExpert(g: *G, c: *const v41.Config, w: *const W, x: T) !T {
            return sharedExpertQ(g, c, x, w.sh_w1, w.sh_w3, w.sh_w2);
        }

        /// The shared expert at a prompt-width call: its middle as the compiled SharedMid region when the route is
        /// installed (rows above rc_max_rows), else `sharedExpert`.
        pub fn sharedExpertPrompt(g: *G, c: *const v41.Config, rt: *const Routes, w: *const W, x: T) !T {
            if (!rt.prefill_shared_mid or rowsOf(g, x, 1) <= rc_max_rows) return sharedExpert(g, c, w, x);
            var o: [1]T = undefined;
            try g.tape(SharedMid, c, &.{ try qlinear(g, x, w.sh_w1), try qlinear(g, x, w.sh_w3), x }, &o);
            return qlinear(g, o[0], w.sh_w2);
        }

        /// C16: `sharedExpertQ`'s statements with the three projections on the draft FMA kernel.
        fn sharedExpertRc(g: *G, c: *const v41.Config, s: *const SharedRc(G), x: T) !T {
            const gl = try s.w1.linear(g, x);
            const ul = try s.w3.linear(g, x);
            return s.w2.linear(g, try sharedMid(g, c, gl, ul, g.dtypeOf(x)));
        }

        /// `sharedExpertMinv` (tests).
        pub fn sharedExpertMinvFor(g: *G, c: *const v41.Config, rt: *const Routes, w: *const W, ms: ?*const MinvSites(G), x: T) !T {
            return sharedExpertMinv(g, c, rt, w, ms, x);
        }

        /// The shared expert with C29's rows at rows <= 8 when bound (else `sharedExpert`).
        fn sharedExpertMinv(g: *G, c: *const v41.Config, rt: *const Routes, w: *const W, ms: ?*const MinvSites(G), x: T) !T {
            const m = ms orelse return sharedExpertPrompt(g, c, rt, w, x);
            if (m.sh_w13) |*s13| {
                if (rowsOf(g, x, 1) > rc_max_rows) return sharedExpertPrompt(g, c, rt, w, x);
                // DENSE_RC: gate | up in one launch, then their halves.
                const gu = try m1Linear(g, s13, x, w.sh_w13.?);
                const sh = g.shapeOf(gu);
                const n: c_int = @divExact(sh.dim(-1), 2);
                var lo: [ops.max_dims]c_int = @splat(0);
                var hi = sh.d;
                const st: [ops.max_dims]c_int = @splat(1);
                hi[sh.n - 1] = n;
                const gl = try g.slice(gu, lo[0..sh.n], hi[0..sh.n], st[0..sh.n]);
                lo[sh.n - 1] = n;
                hi[sh.n - 1] = 2 * n;
                const ul = try g.slice(gu, lo[0..sh.n], hi[0..sh.n], st[0..sh.n]);
                return m1Linear(g, &m.sh_w2.?, try sharedMid(g, c, gl, ul, g.dtypeOf(x)), w.sh_w2);
            }
            if (m.sh_w1 == null or rowsOf(g, x, 1) > rc_max_rows) return sharedExpertPrompt(g, c, rt, w, x);
            const gl = try m1Linear(g, &m.sh_w1.?, x, w.sh_w1);
            const ul = try m1Linear(g, &m.sh_w3.?, x, w.sh_w3);
            return m1Linear(g, &m.sh_w2.?, try sharedMid(g, c, gl, ul, g.dtypeOf(x)), w.sh_w2);
        }

        fn sharedExpertQ(g: *G, c: *const v41.Config, x: T, w1: Q(T), w3: Q(T), w2: Q(T)) !T {
            const dt = g.dtypeOf(x);
            var gate = try g.astype(try qlinear(g, x, w1), .float32);
            var up = try g.astype(try qlinear(g, x, w3), .float32);
            if (c.swiglu_limit > 0) {
                up = try g.clip(up, try g.scalar(-c.swiglu_limit, .float32), try g.scalar(c.swiglu_limit, .float32));
                gate = try g.minimum(gate, try g.scalar(c.swiglu_limit, .float32));
            }
            const h = try g.mul(try g.silu(gate), up);
            return qlinear(g, try g.astype(h, dt), w2);
        }

        /// `_moe_combine_impl`: the weighted routed sum in f32 plus the shared output.
        /// The combine at prompt widths: SMALLK's kernel above attn_compile_max_rows when bound.
        fn combineWide(g: *G, lk: LK, ro: T, weights: T, shared: T) !T {
            if (lk.combine) |k| if (g.shapeOf(shared).dim(0) > attn_compile_max_rows)
                return k.call(g, if (g.dtypeOf(ro) == .bfloat16) ro else try g.astype(ro, .float32), try g.astype(weights, .float32), shared);
            return moeCombine(g, ro, weights, shared);
        }

        fn moeCombine(g: *G, ro: T, weights: T, shared: T) !T {
            return g.add(try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(weights, -1)), -2, false), shared);
        }

        /// K22 `_moe_combine`: in routed, weights, shared.
        const MoeCombine = struct {
            pub const region: ops.Region = .moe_combine;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                out[0] = try moeCombine(g, in[0], in[1], in[2]);
            }
        };

        /// `MoE.__call__`: gate, routed experts (`routed.routed(g, xf, indices)`
        /// returns the unweighted `[n, k, dim]` outputs), shared expert, f32 combine.
        pub fn moe(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, w: *const W, x: T, routed: anytype) !T {
            return moeWith(g, p, c, rt, lk, w, x, routed, null);
        }

        /// `routed.routed`, or, for a source that commits the arrays a call does not wait on once
        /// its reads are issued (`routedHoist`), with those (VERIFY_ENCODE hoist).
        fn routedWith(routed: anytype, g: *G, xf: T, indices: T, hoist: []const T) !T {
            if (comptime @hasDecl(@TypeOf(routed), "routedHoist")) return routed.routedHoist(g, xf, indices, hoist);
            return routed.routed(g, xf, indices);
        }

        /// `moe` with the caller's HC post and comb (`tail`), which do not wait on the routed call: at
        /// decode rows the shared expert is built first and handed over with the gate weights and
        /// `tail`, so a streaming source runs them during its read wait, not in the next routing barrier.
        pub fn moeWith(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, w: *const W, x: T, routed: anytype, tail: ?[2]T) !T {
            const sh = g.shapeOf(x);
            const dim: c_int = @intCast(c.hidden_size);
            const xf = try g.reshape(x, &.{ -1, dim });
            const r = try router(g, p, c, rt, lk, w, xf);
            const decode_rows = g.shapeOf(xf).dim(0) <= rc_max_rows;
            const rc_shared: ?*const SharedRc(G) = if (decode_rows) lk.shared else null;
            var hoist: [4]T = undefined;
            var n_hoist: usize = 0;
            if (decode_rows) {
                hoist[0] = try g.astype(if (rc_shared) |s| try sharedExpertRc(g, c, s, xf) else try sharedExpertMinv(g, c, rt, w, lk.minv, xf), .float32);
                hoist[1] = r.weights;
                n_hoist = 2;
                if (tail) |t| {
                    hoist[2..4].* = t;
                    n_hoist = 4;
                }
            }
            const ro = try routedWith(routed, g, xf, r.indices, hoist[0..n_hoist]);
            try p.put("moe.routed", ro);
            const shared = if (decode_rows) hoist[0] else try g.astype(try sharedExpertMinv(g, c, rt, w, lk.minv, xf), .float32);
            try p.put("moe.shared", shared);
            const y = if (g.shapeOf(xf).dim(0) <= rt.attn_rows) blk: {
                var o: [1]T = undefined;
                try g.tape(MoeCombine, c, &.{ ro, r.weights, shared }, &o);
                break :blk o[0];
            } else try combineWide(g, lk, ro, r.weights, shared);
            return g.reshape(try g.astype(y, g.dtypeOf(x)), sh.slice());
        }

        /// `_hc_attn_prep_impl`: the attn HC mixes, the pre-mix collapse and the
        /// attention RMSNorm. Out: attention input, pre, post, comb.
        pub fn hcAttnPrep(g: *G, c: *const v41.Config, lk: LK, h: T, pre_mix: T, fnw: T, base: T, scale: T, norm_w: T) ![4]T {
            if (lk.tape_attn) |t| if (tapeRows(g, h)) |d| {
                // C15 seg1: collapse + RMSNorm + the premix sum of squares in one kernel.
                const cn = try t.collapseNorm(g, try g.reshape(h, &.{ d.m, d.hc, d.dim }), try g.reshape(pre_mix, &.{ d.m, d.hc }), norm_w);
                const mx = try tapeMixes(g, c, t, lk.attnMix(), d, cn[0], cn[1], fnw, base, scale);
                return .{ try g.reshape(cn[2], &.{ d.b, d.s, d.dim }), mx.pre, mx.post, mx.comb };
            };
            const m = try hcMixes(g, c, lk.attnMix(), h, fnw, base, scale);
            const x = if (hcNormFor(lk.hc_norm, g, h)) |hn| try hn.preNorm(g, h, pre_mix, norm_w) else try rmsnorm(g, try hcPre(g, h, pre_mix), norm_w, c.rms_norm_eps);
            return .{ x, m.pre, m.post, m.comb };
        }

        /// `_hc_ffn_prep_impl`: the attention HC post, the ffn mixes, collapse and
        /// ffn RMSNorm. Out: moe input, h1, ffn post, ffn comb, ffn pre.
        pub fn hcFfnPrep(g: *G, c: *const v41.Config, lk: LK, attn_out: T, residual: T, attn_pre: T, attn_post: T, attn_comb: T, fnw: T, base: T, scale: T, norm_w: T) ![5]T {
            if (lk.tape) |t| if (tapeRows(g, residual)) |d| {
                // C15 seg2': the attention HC combine, collapse, RMSNorm and sum of squares fused
                // (C16's stage 0: the f32 x over the bf16 residual on its own text).
                const args = .{ try g.reshape(attn_out, &.{ d.m, d.dim }), try g.reshape(residual, &.{ d.m, d.hc, d.dim }), try g.reshape(attn_post, &.{ d.m, d.hc }), try g.reshape(attn_comb, &.{ d.m, d.hc * d.hc }), try g.reshape(attn_pre, &.{ d.m, d.hc }), norm_w };
                const f = if (lk.tape_mixed) |tm| try tm.call(g, args[0], args[1], args[2], args[3], args[4], args[5]) else try t.combineCollapseNorm(g, args[0], args[1], args[2], args[3], args[4], args[5]);
                const mx = try tapeMixes(g, c, t, lk.ffnMix(), d, f[1], f[2], fnw, base, scale);
                return .{ try g.reshape(f[3], &.{ d.b, d.s, d.dim }), try g.reshape(f[0], &.{ d.b, d.s, d.hc, d.dim }), mx.post, mx.comb, mx.pre };
            };
            return hcFfnFrom(g, c, lk, try hcPost(g, attn_out, residual, attn_post, attn_comb), attn_pre, fnw, base, scale, norm_w);
        }

        /// `hcFfnPrep` at prompt widths with the attention HC post compiled (HCPOST: HcPost's region, prepared at
        /// construction): the eager chain's ops in one region; the ffn mixes and pre-norm follow as there.
        pub fn hcFfnPrepCompiledPost(g: *G, c: *const v41.Config, lk: LK, attn_out: T, residual: T, attn_pre: T, attn_post: T, attn_comb: T, fnw: T, base: T, scale: T, norm_w: T) ![5]T {
            if (lk.hcpost) |hp| return hcFfnFrom(g, c, lk, try hcPostFused(g, hp, attn_out, residual, attn_post, attn_comb), attn_pre, fnw, base, scale, norm_w);
            var o: [1]T = undefined;
            try g.tape(HcPost, c, &.{ attn_out, residual, attn_post, attn_comb }, &o);
            return hcFfnFrom(g, c, lk, o[0], attn_pre, fnw, base, scale, norm_w);
        }

        /// The ffn side after the attention HC post `h1`: its mixes, collapse and RMSNorm.
        fn hcFfnFrom(g: *G, c: *const v41.Config, lk: LK, h1: T, attn_pre: T, fnw: T, base: T, scale: T, norm_w: T) ![5]T {
            const m = try hcMixes(g, c, lk.ffnMix(), h1, fnw, base, scale);
            const x = if (hcNormFor(lk.hc_norm, g, h1)) |hn| try hn.preNorm(g, h1, attn_pre, norm_w) else try rmsnorm(g, try hcPre(g, h1, attn_pre), norm_w, c.rms_norm_eps);
            return .{ x, h1, m.post, m.comb, m.pre };
        }

        /// The rows of an HC stream `[b, s, hc, dim]` when the C15 tape takes them (<= 8).
        const TapeDims = struct { b: c_int, s: c_int, m: c_int, hc: c_int, dim: c_int };
        fn tapeRows(g: *G, h: T) ?TapeDims {
            const sh = g.shapeOf(h);
            const m = sh.d[0] * sh.d[1];
            if (m > rc_max_rows) return null;
            return .{ .b = sh.d[0], .s = sh.d[1], .m = m, .hc = sh.d[2], .dim = sh.d[3] };
        }

        /// C15: the mixes over the tape's f32 stream copy `sfc` [M, hc dim] and its sum of squares:
        /// the premix GEMV (C13's route, else the stock GEMM), `q3ht_mixfin` (rsqrt-multiply, the
        /// split's affine / sigmoid), the Sinkhorn (C12's route, else the op chain).
        fn tapeMixes(g: *G, c: *const v41.Config, t: *const kr.HcTape(G), mk: MK, d: TapeDims, sfc: T, ssq: T, fnw: T, base: T, scale: T) !Mixes {
            const mm = if (mk.premix) |k| try k.call(g, sfc) else try g.matmul(sfc, try g.transpose(try g.astype(fnw, .float32)));
            const pr = try t.mixfin(g, mm, ssq, scale, base);
            const comb = try g.reshape(pr[2], &.{ d.b, d.s, d.hc, d.hc });
            const sk = if (mk.sinkhorn) |k| try k.call(g, comb) else try sinkhorn(g, comb, c.hc_sinkhorn_iters, c.hc_eps);
            return .{ .pre = try g.reshape(pr[0], &.{ d.b, d.s, d.hc }), .post = try g.reshape(pr[1], &.{ d.b, d.s, d.hc }), .comb = sk };
        }

        /// The MoE-side HC combine (`_hc_post_impl`): C15's `q3ht_combine` at <= 8 rows, else
        /// K4's tape at its rows or HCPOST's above 8 (the compiled region K16's combine runs), else the eager einsum.
        pub fn hcPostRoute(g: *G, c: *const v41.Config, rt: *const Routes, lk: LK, x: T, residual: T, post: T, comb: T) !T {
            if (lk.tape) |t| if (tapeRows(g, residual)) |d| {
                const h = try t.combine(g, try g.reshape(x, &.{ d.m, d.dim }), try g.reshape(residual, &.{ d.m, d.hc, d.dim }), try g.reshape(post, &.{ d.m, d.hc }), try g.reshape(comb, &.{ d.m, d.hc * d.hc }));
                return g.reshape(h, &.{ d.b, d.s, d.hc, d.dim });
            };
            const rows = rowsOf(g, residual, 2);
            if (lk.hcpost) |hp| if (rows > rc_max_rows) return hcPostFused(g, hp, x, residual, post, comb);
            if (rows <= rt.hc_rows or (rt.prefill_hc_post and rows > rc_max_rows)) {
                var o: [1]T = undefined;
                try g.tape(HcPost, c, &.{ x, residual, post, comb }, &o);
                return o[0];
            }
            return hcPost(g, x, residual, post, comb);
        }

        /// K4 / K35 seg1: in h, pre_mix, attn fn, base, scale, attn norm.
        pub const HcAttnPrep = struct {
            pub const region: ops.Region = .hc_attn_prep;
            pub const Ctx = v41.Config;
            pub const n_out = 4;
            pub fn run(g: *G, ctx: *const Ctx, in: []const T, out: []T) !void {
                out[0..4].* = try hcAttnPrep(g, ctx, .{}, in[0], in[1], in[2], in[3], in[4], in[5]);
            }
        };

        /// K4 ffn prep: in attn out, residual, attn pre, post, comb, ffn fn, base, scale, ffn norm.
        pub const HcFfnPrep = struct {
            pub const region: ops.Region = .hc_ffn_prep;
            pub const Ctx = v41.Config;
            pub const n_out = 5;
            pub fn run(g: *G, ctx: *const Ctx, in: []const T, out: []T) !void {
                out[0..5].* = try hcFfnPrep(g, ctx, .{}, in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8]);
            }
        };

        /// K4 moe combine (`_hc_post_impl`): in moe output, residual, ffn post, ffn comb.
        pub const HcPost = struct {
            pub const region: ops.Region = .hc_post;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                out[0] = try hcPost(g, in[0], in[1], in[2], in[3]);
            }
        };

        /// K35 seg2: the ffn prep, the whole gate and the shared expert. In: the
        /// HcFfnPrep inputs, gate weight, gate bias, then shared w1, w3, w2 as
        /// (words, scales). Out: xf, weights, indices, shared, h1, ffn post, ffn comb, ffn pre.
        const Seg2 = struct {
            pub const region: ops.Region = .seg2;
            pub const Ctx = v41.Config;
            pub const n_out = 8;
            pub fn run(g: *G, c: *const Ctx, in: []const T, out: []T) !void {
                const f = try hcFfnPrep(g, c, .{}, in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8]);
                const xf = try g.reshape(f[0], &.{ -1, @intCast(c.hidden_size) });
                const pre = try gatePrefix(g, xf, in[9], in[10]);
                const r = try gateSelect(g, c, pre[0], pre[1]);
                const shared = try g.astype(try sharedExpertQ(g, c, xf, .{ .w = in[11], .s = in[12] }, .{ .w = in[13], .s = in[14] }, .{ .w = in[15], .s = in[16] }), .float32);
                out[0..8].* = .{ xf, r.weights, r.indices, shared, f[1], f[2], f[3], f[4] };
            }
        };

        /// K35 seg3: the MoE combine folded into the ffn HC post. In routed,
        /// weights, shared, residual, ffn post, ffn comb.
        const Seg3 = struct {
            pub const region: ops.Region = .seg3;
            pub const Ctx = v41.Config;
            pub const n_out = 1;
            pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
                const res = g.shapeOf(in[3]);
                const y = try g.astype(try moeCombine(g, in[0], in[1], in[2]), g.dtypeOf(in[3]));
                const y3 = try g.reshape(y, &.{ res.d[0], res.d[1], res.d[3] });
                out[0] = try hcPost(g, y3, in[3], in[4], in[5]);
            }
        };

        /// Builds every compiled region the routes use, for context `c` (once, at
        /// construction: no region compiles inside a forward; a call finds its
        /// closure by region and context).
        pub fn prepareRegions(g: *G, c: *const v41.Config, rt: *const Routes, layer_major: bool) !void {
            if (rt.core_rows > 0) try g.prepareTape(AttnCore, c);
            if (rt.attn_rows > 0) inline for (.{ QkvPrep, OutPrep, GatePrefix, MoeCombine }) |B| try g.prepareTape(B, c);
            if (rt.hc_rows > 0) inline for (.{ HcAttnPrep, HcFfnPrep, HcPost }) |B| try g.prepareTape(B, c);
            if (rt.small_rows > 0) inline for (.{ HcAttnPrep, Seg2, Seg3, HcPost }) |B| try g.prepareTape(B, c);
            if (layer_major or rt.prefill_hc_post) try g.prepareTape(HcPost, c);
            if (rt.prefill_shared_mid) try g.prepareTape(SharedMid, c);
        }

        /// `DecoderLayer.__call__`: attention and MoE, each inside a
        /// Hyper-Connection pre / post, the pre mix threaded across sublayers.
        /// At decode / verify rows K35 runs three compiled segments; K4 compiles
        /// the HC prep / combine; the eager body otherwise.
        pub fn layer(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, li: v41.LayerInfo, w: *const W, inv_freq: T, h: T, pre_mix: T, positions: T, cache: *Cache, shared: *Share, routed: anytype) !Out {
            const rows = rowsOf(g, h, 2);
            if (rows <= rt.small_rows) {
                var s1: [4]T = undefined;
                try g.tape(HcAttnPrep, c, &.{ h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm }, &s1);
                try p.put("attn.x", s1[0]);
                const ao = try attention(g, p, c, rt, lk, li, w, inv_freq, s1[0], positions, cache, shared);
                var s2: [8]T = undefined;
                try g.tape(Seg2, c, &.{ ao, h, s1[1], s1[2], s1[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm, w.gate_w, w.gate_bias, w.sh_w1.w, w.sh_w1.s, w.sh_w3.w, w.sh_w3.s, w.sh_w2.w, w.sh_w2.s }, &s2);
                try p.put("gate.indices", s2[2]);
                try p.put("gate.weights", s2[1]);
                try p.put("moe.shared", s2[3]);
                const ro = try routedWith(routed, g, s2[0], s2[2], &.{ s2[3], s2[1], s2[4], s2[5], s2[6], s2[7] });
                try p.put("moe.routed", ro);
                var s3: [1]T = undefined;
                try g.tape(Seg3, c, &.{ ro, s2[1], s2[3], s2[4], s2[5], s2[6] }, &s3);
                try p.put("out.h", s3[0]);
                try p.put("out.pre_mix", s2[7]);
                return .{ .h = s3[0], .pre_mix = s2[7] };
            }
            const half = try attnAndMoeInput(g, p, c, rt, lk, li, w, inv_freq, h, pre_mix, positions, cache, shared);
            const mo = try moeWith(g, p, c, rt, lk, w, half.moe_in, routed, .{ half.post, half.comb });
            try p.put("moe.y", mo);
            const out = try hcPostRoute(g, c, rt, lk, mo, half.h1, half.post, half.comb);
            try p.put("out.h", out);
            try p.put("out.pre_mix", half.ffn_pre);
            return .{ .h = out, .pre_mix = half.ffn_pre };
        }

        /// What `attnAndMoeInput` hands the MoE and its combine.
        pub const Half = struct { moe_in: T, h1: T, post: T, comb: T, ffn_pre: T };

        /// `DecoderLayer.attn_and_moe_input`: the attention Hyper-Connection
        /// (which writes this layer's KV), the ffn mixes and the MoE input
        /// (K4 compiles both HC preps at rows <= 7).
        pub fn attnAndMoeInput(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, li: v41.LayerInfo, w: *const W, inv_freq: T, h: T, pre_mix: T, positions: T, cache: *Cache, shared: *Share) !Half {
            const hc_tape = rowsOf(g, h, 2) <= rt.hc_rows;
            var a: [4]T = undefined;
            if (hc_tape) {
                try g.tape(HcAttnPrep, c, &.{ h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm }, &a);
            } else {
                // The chunk's inputs (evaluated already): this stage takes whatever ran since the previous probe
                // (the previous chunk's fence and frees, a layer's end), so attn.pre is the HC premix alone.
                try p.put("attn.in", h);
                a = try hcAttnPrep(g, c, lk, h, pre_mix, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm);
                try p.put("attn.pre", a[1]);
                try p.put("attn.post", a[2]);
                try p.put("attn.comb", a[3]);
            }
            try p.put("attn.x", a[0]);
            const ao_raw = try attention(g, p, c, rt, lk, li, w, inv_freq, a[0], positions, cache, shared);
            var f: [5]T = undefined;
            if (hc_tape) {
                try g.tape(HcFfnPrep, c, &.{ ao_raw, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm }, &f);
            } else {
                // kv16: at prompt widths the attention output joins the stream in the stream's dtype (the reference's bf16
                // `wo_b`; `hc_post` returns its x's dtype), so the residual stream stays bf16 through every layer. Decode
                // rows keep their tapes' own dtypes (C14 / C16 already run the bf16 stream).
                const ao = if (rowsOf(g, h, 2) > rc_max_rows) try g.astype(ao_raw, g.dtypeOf(h)) else ao_raw;
                f = if (rt.prefill_hc_post and rowsOf(g, h, 2) > rc_max_rows)
                    try hcFfnPrepCompiledPost(g, c, lk, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm)
                else
                    try hcFfnPrep(g, c, lk, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm);
                try p.put("hc1.h", f[1]);
                try p.put("ffn.pre", f[4]);
                try p.put("ffn.post", f[2]);
                try p.put("ffn.comb", f[3]);
            }
            try p.put("ffn.x", f[0]);
            return .{ .moe_in = f[0], .h1 = f[1], .post = f[2], .comb = f[3], .ffn_pre = f[4] };
        }

        /// `MoE.combine_routed`: the shared expert and the f32 combine (K22 at
        /// rows <= 32) of routed rows computed elsewhere (K16's batched switch).
        pub fn combineRouted(g: *G, p: anytype, c: *const v41.Config, rt: *const Routes, lk: LK, w: *const W, ro: T, weights: T, xf: T, pre_shared: ?T) !T {
            const shared = pre_shared orelse try g.astype(try sharedExpertPrompt(g, c, rt, w, xf), .float32);
            try p.put("moe.shared", shared);
            if (g.shapeOf(xf).dim(0) <= rt.attn_rows) {
                var o: [1]T = undefined;
                try g.tape(MoeCombine, c, &.{ ro, weights, shared }, &o);
                return o[0];
            }
            return combineWide(g, lk, ro, weights, shared);
        }

        /// PREFILL_HCPOST: x [b, s, dim], residual [b, s, hc, dim], post [b, s, hc], comb [b, s, hc, hc] through the
        /// one-pass combine (rows flattened) -> [b, s, hc, dim] f32.
        fn hcPostFused(g: *G, hp: *const kr.HcPostTf32(G), x: T, residual: T, post: T, comb: T) !T {
            const rs = g.shapeOf(residual);
            const m = rs.d[0] * rs.d[1];
            // The kernel computes in f32 (the reference's hc_post: f32 math, `y.type_as(x)`): the result in x's dtype. On
            // kv16's bf16 stream (x and the residual bf16) the all-bf16 text reads x through static_cast<float> and stores
            // h rounded to bfloat, the region's `astype(.., x dtype)`: no f32 x copy, no f32 h to narrow.
            const bf16_stream = g.dtypeOf(x) == .bfloat16 and g.dtypeOf(residual) == .bfloat16;
            const xin = if (bf16_stream) x else try g.astype(x, .float32);
            const h = try hp.call(g, try g.reshape(xin, &.{ m, rs.d[3] }), try g.reshape(residual, &.{ m, rs.d[2], rs.d[3] }), try g.reshape(post, &.{ m, rs.d[2] }), try g.reshape(comb, &.{ m, rs.d[2] * rs.d[2] }));
            return g.astype(try g.reshape(h, rs.slice()), g.dtypeOf(x));
        }

        /// `_PREFILL_HC_POST`: K16's ffn combine, always the compiled `_hc_post_impl` (PREFILL_HCPOST's one pass above
        /// rc_max_rows when bound).
        pub fn prefillHcPost(g: *G, c: *const v41.Config, lk: LK, mo: T, half: Half) !T {
            if (lk.hcpost) |hp| if (rowsOf(g, half.h1, 2) > rc_max_rows) return hcPostFused(g, hp, mo, half.h1, half.post, half.comb);
            var o: [1]T = undefined;
            try g.tape(HcPost, c, &.{ mo, half.h1, half.post, half.comb }, &o);
            return o[0];
        }

        /// `NGramRowCache.dequantize` (mxfp8 records): the E4M3 code words
        /// `[n, head_dim / 4]` u32 and E8M0 scales `[n, head_dim / 32]` u8 of
        /// `B * L * cols` records -> `[B, L, cols, head_dim]` bf16.
        pub fn engramRows(g: *G, codes: T, scales: T, B: c_int, L: c_int, cols: c_int) !T {
            const dq = try g.dequantize(codes, scales, .mxfp8);
            return g.reshape(dq, &.{ B, L, cols, g.shapeOf(dq).dim(-1) });
        }

        /// `EngramV41.__call__` after the row fetch: `rows` are the dequantized
        /// `[B, L, cols, head_dim]` bank rows; returns `h + gate * value`.
        pub fn engramApply(g: *G, c: *const v41.Config, w: EngramW(T), hidden: T, rows: T) !T {
            return engramApplyM1(g, c, w, null, hidden, rows);
        }

        /// `engramApply` with C29's Engram wkv rows at rows <= 8 when bound.
        pub fn engramApplyM1(g: *G, c: *const v41.Config, w: EngramW(T), m1_wkv: ?*const kr.Mxfp8Rows(G), hidden: T, rows: T) !T {
            const hs = g.shapeOf(hidden);
            const B = hs.d[0];
            const L = hs.d[1];
            const hc: c_int = @intCast(c.hc_mult);
            const dim: c_int = @intCast(c.hidden_size);
            const kv = try m1Linear(g, m1_wkv, try g.reshape(rows, &.{ B, L, -1 }), w.wkv);
            const split = hc * dim;
            const key = try g.reshape(try g.astype(try sliceLast(g, kv, 0, split), .float32), &.{ B, L, hc, dim });
            const value = try g.astype(try sliceLast(g, kv, split, split + dim), .float32);
            const hf = try g.astype(hidden, .float32);
            const weight = try g.astype(try g.mul(w.q_weight, w.k_weight), .float32);
            const eps = c.rms_norm_eps;
            const m1 = try g.mean(try g.mul(hf, hf), -1, false);
            const m2 = try g.mean(try g.mul(key, key), -1, false);
            const rstd = try g.mul(try g.rsqrt(try g.add(m1, try sf(g, eps, m1))), try g.rsqrt(try g.add(m2, try sf(g, eps, m2))));
            const prod = try g.mul(try g.mul(hf, weight), key);
            const dot0 = try g.mul(try g.sum(prod, -1, false), rstd);
            const dot = try g.mul(dot0, try sf(g, std.math.pow(f64, @floatFromInt(c.hidden_size), -0.5), dot0));
            const mag = try g.sqrt(try g.maximum(try g.abs(dot), try sf(g, 1e-6, dot)));
            const signed = try g.where(try g.less(dot, try sf(g, 0, dot)), try g.neg(mag), mag);
            const gate = try g.sigmoid(signed);
            const contribution = try g.mul(try g.expandDims(gate, -1), try g.expandDims(value, 2));
            return g.astype(try g.add(hf, contribution), g.dtypeOf(hidden));
        }
    };
}

/// The parity dumps' routed-expert stand-in (never served): expert e's
/// output is `x * (1 + e / n)` per row, f32, the op sequence of the Python
/// `StandInSwitch` (`x.astype(f32)[:, None, :] * scale[indices][..., None]`),
/// so a chained run needs no injected expert outputs.
pub fn StandIn(comptime G: type) type {
    return struct {
        scale: G.T,

        /// `1 + e / n` rounded per f32 op, as numpy builds the Python table.
        pub fn table(buf: []f32) void {
            const n: f32 = @floatFromInt(buf.len);
            for (buf, 0..) |*v, e| v.* = 1.0 + @as(f32, @floatFromInt(e)) / n;
        }

        /// The same stand-in on every layer.
        pub fn at(self: @This(), _: u32) @This() {
            return self;
        }

        pub fn routed(self: @This(), g: *G, xf: G.T, indices: G.T) !G.T {
            const xs = try g.expandDims(try g.astype(xf, .float32), 1);
            return g.mul(xs, try g.expandDims(try g.take(self.scale, indices, 0), -1));
        }
    };
}

pub const Ramp = struct { low: f64, high: f64 };

/// The YaRN correction range (`_yarn_inv_freq`, Python floats): floor / ceil
/// of the correction dims, `high += 0.001` when they meet.
pub fn yarnRamp(dim: f64, base: f64, orig: f64, beta_fast: f64, beta_slow: f64) Ramp {
    const corr = struct {
        fn f(d: f64, b: f64, o: f64, rot: f64) f64 {
            return d * @log(o / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(corr(dim, base, orig, beta_fast)), 0);
    var high = @min(@ceil(corr(dim, base, orig, beta_slow)), dim - 1);
    if (low == high) high += 0.001;
    return .{ .low = low, .high = high };
}

// ── tests (host-only: TraceOps records shapes, dtypes and ops; no MLX array) ──

const testing = std.testing;
const TraceOps = ops.TraceOps;
const stock: Routes = .{};
const Tr = Trunk(TraceOps);
const Shape = ops.Shape;

/// Records the named stages a trunk function hands its probe.
const TraceProbe = struct {
    a: std.mem.Allocator,
    names: std.ArrayList([]const u8) = .empty,
    nodes: std.ArrayList(u32) = .empty,

    fn deinit(self: *TraceProbe) void {
        self.names.deinit(self.a);
        self.nodes.deinit(self.a);
    }

    pub fn put(self: *TraceProbe, name: []const u8, x: u32) !void {
        try self.names.append(self.a, name);
        try self.nodes.append(self.a, x);
    }

    /// The latest node recorded under `name`.
    fn get(self: *const TraceProbe, name: []const u8) ?u32 {
        var i = self.names.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.names.items[i], name)) return self.nodes.items[i];
        }
        return null;
    }
};

/// The deterministic routed stand-in's shape: unweighted `[n, k, dim]` f32.
const TraceRouted = struct {
    pub fn routed(_: TraceRouted, g: *TraceOps, xf: u32, indices: u32) !u32 {
        return g.input(&.{ g.shapeOf(xf).dim(0), g.shapeOf(indices).dim(1), g.shapeOf(xf).dim(1) }, .float32);
    }
};

fn qIn(g: *TraceOps, out: u64, in: u64, mode: sdk.QuantMode) !Q(u32) {
    const o: c_int = @intCast(out);
    const i: c_int = @intCast(in);
    const bits: c_int = @intCast(ops.quantBits(mode));
    return .{ .w = try g.input(&.{ o, @divExact(i * bits, 32) }, .uint32), .s = try g.input(&.{ o, @divExact(i, 32) }, .uint8), .mode = mode };
}

fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// A layer's residents as trace inputs, dtypes as the checkpoint stores them.
fn traceLayerW(g: *TraceOps, c: *const v41.Config, li: v41.LayerInfo) !LayerW(u32) {
    const H: u64 = c.hidden_size;
    const hd: u64 = c.head_dim;
    const mix = ci(c.hcMix());
    var w: LayerW(u32) = .{
        .attn_norm = try g.input(&.{ci(H)}, .bfloat16),
        .ffn_norm = try g.input(&.{ci(H)}, .bfloat16),
        .hc_attn_fn = try g.input(&.{ mix, ci(c.hc_mult * H) }, .float32),
        .hc_attn_base = try g.input(&.{mix}, .float32),
        .hc_attn_scale = try g.input(&.{3}, .float32),
        .hc_ffn_fn = try g.input(&.{ mix, ci(c.hc_mult * H) }, .float32),
        .hc_ffn_base = try g.input(&.{mix}, .float32),
        .hc_ffn_scale = try g.input(&.{3}, .float32),
        .attn_sink = try g.input(&.{ci(c.n_heads)}, .float32),
        .q_norm = try g.input(&.{ci(c.q_lora_rank)}, .bfloat16),
        .kv_norm = try g.input(&.{ci(hd)}, .bfloat16),
        .wq_a = try qIn(g, c.q_lora_rank, H, .mxfp8),
        .wq_b = try qIn(g, c.n_heads * hd, c.q_lora_rank, .mxfp8),
        .wkv = try qIn(g, hd, H, .mxfp8),
        .wo_a = try qIn(g, c.o_groups * c.o_lora_rank, c.n_heads * hd / c.o_groups, .mxfp8),
        .wo_b = try qIn(g, H, c.o_groups * c.o_lora_rank, .mxfp8),
        .gate_w = try g.input(&.{ ci(c.n_routed_experts), ci(H) }, .bfloat16),
        .gate_bias = try g.input(&.{ci(c.n_routed_experts)}, .float32),
        .sh_w1 = try qIn(g, c.moe_intermediate_size, H, .mxfp8),
        .sh_w2 = try qIn(g, H, c.moe_intermediate_size, .mxfp8),
        .sh_w3 = try qIn(g, c.moe_intermediate_size, H, .mxfp8),
    };
    if (li.kv_source) {
        w.comp = .{
            .wkv = try g.input(&.{ ci(hd), ci(H) }, .bfloat16),
            .wgate = if (li.ratio > 1) try g.input(&.{ ci(hd), ci(H) }, .bfloat16) else null,
            .norm = try g.input(&.{ci(hd)}, .bfloat16),
        };
        w.idx_k = .{ .wk = try g.input(&.{ ci(c.index_head_dim), ci(hd) }, .bfloat16), .k_norm = try g.input(&.{ci(c.index_head_dim)}, .bfloat16) };
    }
    if (li.index_source) {
        w.idx_q = .{ .wq_b = try qIn(g, c.index_n_heads * c.index_head_dim, c.q_lora_rank, .mxfp8), .weights_proj = try g.input(&.{ ci(c.index_n_heads), ci(H) }, .bfloat16) };
    }
    // The prefill core's sink view, as the model builds it when the route is on.
    w.sink4 = try Tr.sinkView(g, c, w.attn_sink);
    return w;
}

fn realConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

/// No `op` node in the trace since `from`.
fn noneOf(g: *const TraceOps, from: usize, op: ops.Op) bool {
    for (g.nodes.items[from..]) |nd| if (nd.op == op) return false;
    return true;
}

// Inside a GPU run (DSV41_PHASE0B_MLX=1: any MLX array creates the Metal device), seconds: P1's predictor
// selection on the GPU stream against the router's own (`gatePrefix` + `gateSelect`) over the same rows and gate.
test "dsv41 smoke 0b: the prefill shared expert's three mxfp8 qmm at the K16 chunk shapes, against a bf16 matmul ceiling (MLX, GPU stream)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const mlx = @import("sdk").mlx;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(testing.allocator, s);
    defer g.deinit();
    const c = try realConfig();
    const TrM = Trunk(ops.MlxOps);
    const io = testing.io;
    const H: usize = c.hidden_size;
    const I: usize = c.moe_intermediate_size;
    var rng = std.Random.DefaultPrng.init(0x5eed_5a3d);
    const r = rng.random();
    const buf = try testing.allocator.alloc(f32, @max(H * I, 953 * @max(H, I)));
    defer testing.allocator.free(buf);
    const Proj = struct { name: []const u8, n: usize, k: usize };
    const projs = [_]Proj{ .{ .name = "w1", .n = I, .k = H }, .{ .name = "w3", .n = I, .k = H }, .{ .name = "w2", .n = H, .k = I } };
    // The K16 chunk's rows (17 x 953 and one 183-row chunk at 16K); 8 calls per timed run, the best of 5 after a warm-up.
    for ([_]usize{ 953, 183 }) |rows| for (projs) |pj| {
        const mark = g.mark();
        defer g.resetTo(mark);
        for (buf[0 .. pj.n * pj.k]) |*v| v.* = r.floatNorm(f32) * 0.02;
        const wd = try g.astype(try g.hostArray(std.mem.sliceAsBytes(buf[0 .. pj.n * pj.k]), &.{ @intCast(pj.n), @intCast(pj.k) }, .float32), .bfloat16);
        const qw = try g.quantize(wd, .mxfp8);
        const q: Q(ops.MlxOps.T) = .{ .w = qw.w, .s = qw.s, .mode = .mxfp8 };
        for (buf[0 .. rows * pj.k]) |*v| v.* = r.floatNorm(f32);
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(buf[0 .. rows * pj.k]), &.{ @intCast(rows), @intCast(pj.k) }, .float32), .bfloat16);
        try g.evalAll(&.{ wd, q.w, q.s, x });
        for ([_][]const u8{ "mxfp8", "bf16" }) |mode| {
            var best: u64 = std.math.maxInt(u64);
            for (0..6) |run| {
                const m = g.mark();
                defer g.resetTo(m);
                var outs: [8]ops.MlxOps.T = undefined;
                const t0 = std.Io.Timestamp.now(io, .boot);
                for (&outs) |*o| o.* = if (mode[0] == 'm') try TrM.qlinear(&g, x, q) else try TrM.linear(&g, x, wd);
                try g.evalAll(&outs);
                const ns: u64 = @intCast(t0.untilNow(io, .boot).nanoseconds);
                if (run > 0) best = @min(best, ns);
            }
            const us = @as(f64, @floatFromInt(best)) / 8000.0;
            const tflops = 2.0 * @as(f64, @floatFromInt(rows * pj.n * pj.k)) / (us * 1e-6) / 1e12;
            std.debug.print("SHARED_QMM_MICROBENCH {{\"proj\": \"{s}\", \"rows\": {d}, \"k\": {d}, \"n\": {d}, \"mode\": \"{s}\", \"us\": {d:.1}, \"tflops\": {d:.2}}}\n", .{ pj.name, rows, pj.k, pj.n, mode, us, tflops });
        }
    };
}

test "dsv41 smoke 0b: P1's predictor top-k is the router's selection on the same rows (MLX, GPU stream)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const mlx = @import("sdk").mlx;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(testing.allocator, s);
    defer g.deinit();
    const c = try realConfig();
    const TrM = Trunk(ops.MlxOps);
    const rows = 64;
    const dim: usize = c.hidden_size;
    const n: usize = c.n_routed_experts;
    const k: usize = c.n_experts_per_tok;
    try testing.expect(n <= 512 and k <= 8);
    var rng = std.Random.DefaultPrng.init(0x5eed_91a1);
    const r = rng.random();
    const xs = try testing.allocator.alloc(f32, rows * dim);
    defer testing.allocator.free(xs);
    for (xs) |*v| v.* = r.floatNorm(f32);
    const ws = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(ws);
    for (ws) |*v| v.* = r.floatNorm(f32) * 0.02;
    var bs: [512]f32 = undefined;
    for (bs[0..n]) |*v| v.* = r.floatNorm(f32) * 0.1;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs), &.{ rows, @intCast(dim) }, .float32), .bfloat16);
    const w = try g.astype(try g.hostArray(std.mem.sliceAsBytes(ws), &.{ @intCast(n), @intCast(dim) }, .float32), .bfloat16);
    const bias = try g.hostArray(std.mem.sliceAsBytes(bs[0..n]), &.{@intCast(n)}, .float32);
    const pre = try TrM.gatePrefix(&g, x, w, bias);
    const route = try TrM.gateSelect(&g, &c, pre[0], pre[1]);
    const got = try TrM.predictTopk(&g, &c, x, try g.astype(w, .float32), bias);
    var want_ids: [rows * 8]u16 = undefined;
    var got_ids: [rows * 8]u16 = undefined;
    _ = try g.hostIds(route.indices, want_ids[0 .. rows * k]);
    _ = try g.hostIds(got, got_ids[0 .. rows * k]);
    for (0..rows) |i| {
        std.mem.sort(u16, want_ids[i * k ..][0..k], {}, std.sort.asc(u16));
        std.mem.sort(u16, got_ids[i * k ..][0..k], {}, std.sort.asc(u16));
    }
    try testing.expectEqualSlices(u16, want_ids[0 .. rows * k], got_ids[0 .. rows * k]);
    std.debug.print("\nP1 predictor smoke: {d} rows x {d} experts: the predictor's top-{d} sets == the router's\n", .{ rows, n, k });
}

test "dsv41 smoke 0b: P1's predictor on the gate as stored (bf16) takes its own scores' top-k, the router's wherever the rounding cannot reorder (MLX, GPU stream)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const mlx = @import("sdk").mlx;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(testing.allocator, s);
    defer g.deinit();
    const c = try realConfig();
    const TrM = Trunk(ops.MlxOps);
    // Enough rows that some sit within the bf16 rounding of the router's 6th / 7th scores: the data separates the routes.
    const rows = 1024;
    const dim: usize = c.hidden_size;
    const n: usize = c.n_routed_experts;
    const k: usize = c.n_experts_per_tok;
    try testing.expect(n <= 512 and k <= 8);
    const a = testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x5eed_91a2);
    const r = rng.random();
    const xs = try a.alloc(f32, rows * dim);
    defer a.free(xs);
    for (xs) |*v| v.* = r.floatNorm(f32);
    const ws = try a.alloc(f32, n * dim);
    defer a.free(ws);
    for (ws) |*v| v.* = r.floatNorm(f32) * 0.02;
    var bs: [512]f32 = undefined;
    for (bs[0..n]) |*v| v.* = r.floatNorm(f32) * 0.1;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs), &.{ rows, @intCast(dim) }, .float32), .bfloat16);
    // The gate as stored (bf16): what `Routes.predict_bf16` hands the predictor (no f32 copy).
    const w = try g.astype(try g.hostArray(std.mem.sliceAsBytes(ws), &.{ @intCast(n), @intCast(dim) }, .float32), .bfloat16);
    const bias = try g.hostArray(std.mem.sliceAsBytes(bs[0..n]), &.{@intCast(n)}, .float32);
    const pre = try TrM.gatePrefix(&g, x, w, bias);
    const route = try TrM.gateSelect(&g, &c, pre[0], pre[1]);
    const got = try TrM.predictTopk(&g, &c, x, w, bias);
    // The bf16 route's own scores and their top-k, by predictTopk's ops with the GEMM in the gate's dtype.
    const sb = try g.add(try g.sqrt(try g.softplus(try g.astype(try TrM.linear(&g, try g.astype(x, .bfloat16), w), .float32))), bias);
    const own = try g.astype(try TrM.sliceLast(&g, try g.argpartition(try g.neg(sb), @as(c_int, @intCast(k)) - 1, -1), 0, @intCast(k)), .int32);
    try g.evalAll(&.{ route.indices, got, own, pre[1], sb });
    const want_ids = try a.alloc(u16, rows * k);
    defer a.free(want_ids);
    const got_ids = try a.alloc(u16, rows * k);
    defer a.free(got_ids);
    const own_ids = try a.alloc(u16, rows * k);
    defer a.free(own_ids);
    const sf = try a.alloc(f32, rows * n);
    defer a.free(sf);
    const sbf = try a.alloc(f32, rows * n);
    defer a.free(sbf);
    const sorted = try a.alloc(f32, n);
    defer a.free(sorted);
    _ = try g.hostIds(route.indices, want_ids);
    _ = try g.hostIds(got, got_ids);
    _ = try g.hostIds(own, own_ids);
    _ = try g.hostF32(pre[1], sf);
    _ = try g.hostF32(sb, sbf);
    var within: usize = 0;
    var differ: usize = 0;
    var differ_beyond: usize = 0;
    for (0..rows) |i| {
        const wi = want_ids[i * k ..][0..k];
        const gi = got_ids[i * k ..][0..k];
        const oi = own_ids[i * k ..][0..k];
        std.mem.sort(u16, wi, {}, std.sort.asc(u16));
        std.mem.sort(u16, gi, {}, std.sort.asc(u16));
        std.mem.sort(u16, oi, {}, std.sort.asc(u16));
        // (a) Its own scores' selection: the GEMM ran on the gate as stored.
        try testing.expectEqualSlices(u16, oi, gi);
        // (b) The router's wherever the bf16 perturbation (eps, the row's largest |bf16 - f32| score) cannot reorder the
        // router's 6th and 7th scores (a margin over 2 eps keeps every top-k score above every other).
        const row_f = sf[i * n ..][0..n];
        const row_b = sbf[i * n ..][0..n];
        var eps: f32 = 0;
        for (row_f, row_b) |f, b| eps = @max(eps, @abs(b - f));
        @memcpy(sorted, row_f);
        std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
        const same = std.mem.eql(u16, wi, gi);
        if (!same) differ += 1;
        if (sorted[k - 1] - sorted[k] > 2 * eps) {
            if (!same) differ_beyond += 1;
        } else within += 1;
    }
    try testing.expectEqual(@as(usize, 0), differ_beyond);
    // (c) The data separates the routes (some row's sets differ), so (a) tells the GEMM's dtype apart.
    try testing.expect(differ > 0);
    std.debug.print("\nP1 predictor bf16 smoke: {d} rows x {d} experts: the predictor's top-{d} sets == its own bf16 scores' on every row, == the router's on all {d} rows beyond twice the bf16 perturbation ({d} rows within it, {d} differ)\n", .{ rows, n, k, rows - within, within, differ });
}

fn miniConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .mini);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

fn expectShape(g: *TraceOps, x: u32, want: []const c_int, dt: Dtype) !void {
    const got = g.shapeOf(x);
    if (!got.eql(Shape.of(want)) or g.dtypeOf(x) != dt) {
        std.debug.print("shape {any} {s}, want {any} {s}\n", .{ got.slice(), @tagName(g.dtypeOf(x)), want, @tagName(dt) });
        return error.TestUnexpectedResult;
    }
}

fn expectStage(g: *TraceOps, p: *const TraceProbe, name: []const u8, want: []const c_int, dt: Dtype) !void {
    const x = p.get(name) orelse {
        std.debug.print("stage {s} not recorded\n", .{name});
        return error.TestUnexpectedResult;
    };
    expectShape(g, x, want, dt) catch |e| {
        std.debug.print("  at stage {s}\n", .{name});
        return e;
    };
}

test "dsv41 graph: rmsnorm traces _rmsnorm's op sequence and keeps the input dtype" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const x = try g.input(&.{ 1, 3, 5120 }, .bfloat16);
    const w = try g.input(&.{5120}, .bfloat16);
    const mark = g.nodes.items.len;
    const y = try Tr.rmsnorm(&g, x, w, 1e-20);
    try expectShape(&g, y, &.{ 1, 3, 5120 }, .bfloat16);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    const O = ops.Op;
    try testing.expectEqualSlices(O, &.{ .astype, .square, .mean, .scalar, .add, .rsqrt, .mul, .astype, .mul, .astype }, seq);
    // An f32 input has no casts of its own (MLX's same-dtype astype is the input).
    const xf = try g.input(&.{ 1, 3, 5120 }, .float32);
    const mark2 = g.nodes.items.len;
    _ = try Tr.rmsnorm(&g, xf, w, 1e-20);
    const seq2 = try g.opsSince(testing.allocator, mark2);
    defer testing.allocator.free(seq2);
    try testing.expectEqualSlices(O, &.{ .square, .mean, .scalar, .add, .rsqrt, .mul, .astype, .mul }, seq2);
}

test "dsv41 graph: the RC sinkhorn binds on the kernels' geometry, refuses another by name, and takes every HC comb" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var c = try realConfig();
    const rt: Routes = .{ .rc_sinkhorn = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &.{});
    defer k.deinit(&g);
    const lk = k.at(0);
    try testing.expect(lk.sinkhorn != null);
    const w = try traceLayerW(&g, &c, c.layers[0]);
    const h = try g.input(&.{ 1, 3, @intCast(c.hc_mult), @intCast(c.hidden_size) }, .bfloat16);
    const n0 = g.nodes.items.len;
    const m = try Tr.hcMixes(&g, &c, lk.attnMix(), h, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale);
    try expectShape(&g, m.comb, &.{ 1, 3, 4, 4 }, .float32);
    for (g.nodes.items[n0..]) |nd| try testing.expect(nd.op != .softmax);
    var bad = c;
    bad.hc_sinkhorn_iters = 19;
    try testing.expectError(error.SinkhornGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &rt, &.{}));
    // No RC member on the tier: nothing bound, the op chain stays.
    var none = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &.{}, &.{});
    defer none.deinit(&g);
    const n = none.at(0);
    try testing.expect(n.sinkhorn == null and n.router == null and n.premix_attn == null and !Tr.Kernels.needed(&.{}));
}

test "dsv41 graph: the RC router and HC premix bind per layer on the kernels' geometry, take verify rows, and leave prefill widths stock" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const ws = [_]LayerW(u32){ try traceLayerW(&g, &c, c.layers[0]), try traceLayerW(&g, &c, c.layers[3]) };
    const rt: Routes = .{ .rc_router = true, .rc_premix = true };
    try testing.expect(Tr.Kernels.needed(&rt));
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &ws);
    defer k.deinit(&g);
    // Each layer reads its own routes (over its own gate and HC fns).
    const lk = k.at(1);
    try testing.expect(lk.router.? == &k.router.items[1] and lk.premix_attn.? == &k.premix.items[1][0] and lk.premix_ffn.? == &k.premix.items[1][1]);
    try testing.expectEqual(ws[1].gate_w, lk.router.?.w);
    try testing.expectEqual(ws[1].hc_ffn_fn, lk.premix_ffn.?.w);
    try testing.expect(lk.sinkhorn == null);
    // Verify rows (5): the whole gate is the two router kernels; each HC mix's GEMM the two premix kernels.
    const l0 = g.prepared_launches;
    var n0 = g.nodes.items.len;
    const r = try Tr.router(&g, &p, &c, &.{}, lk, &ws[1], try g.input(&.{ 5, 5120 }, .float32));
    try expectShape(&g, r.indices, &.{ 5, 6 }, .int32);
    try expectShape(&g, r.weights, &.{ 5, 6 }, .float32);
    try testing.expectEqual(r.indices, p.get("gate.indices").?);
    try testing.expectEqual(l0 + 2, g.prepared_launches);
    try testing.expect(noneOf(&g, n0, .argpartition) and noneOf(&g, n0, .matmul));
    n0 = g.nodes.items.len;
    const m = try Tr.hcMixes(&g, &c, lk.attnMix(), try g.input(&.{ 1, 5, 4, 5120 }, .bfloat16), ws[1].hc_attn_fn, ws[1].hc_attn_base, ws[1].hc_attn_scale);
    try expectShape(&g, m.comb, &.{ 1, 5, 4, 4 }, .float32);
    try testing.expectEqual(l0 + 4, g.prepared_launches);
    try testing.expect(noneOf(&g, n0, .matmul));
    // A prefill width (9 rows): the stock gate and GEMM, no launch.
    n0 = g.nodes.items.len;
    _ = try Tr.router(&g, &p, &c, &.{}, lk, &ws[1], try g.input(&.{ 9, 5120 }, .float32));
    _ = try Tr.hcMixes(&g, &c, lk.ffnMix(), try g.input(&.{ 1, 9, 4, 5120 }, .bfloat16), ws[1].hc_ffn_fn, ws[1].hc_ffn_base, ws[1].hc_ffn_scale);
    try testing.expectEqual(l0 + 4, g.prepared_launches);
    try testing.expect(!noneOf(&g, n0, .argpartition) and !noneOf(&g, n0, .matmul));
    // Another geometry is refused by name at construction.
    var bad = c;
    bad.n_routed_experts = 128;
    try testing.expectError(error.RouterGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &.{ .rc_router = true }, &ws));
    bad = c;
    bad.hc_mult = 2;
    try testing.expectError(error.PremixGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &.{ .rc_premix = true }, &ws));
}

test "dsv41 graph: the RCPROJ sites bind per layer on the packed pairs, take verify rows bf16 in and out, and leave prefill widths on the chain" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[0];
    const ws = [_]LayerW(u32){try traceLayerW(&g, &c, li)};
    // K22 prepared: the RC route takes the verify rows before the compiled prep.
    const rt: Routes = .{ .rc_proj = true, .selected_keys = true, .attn_rows = attn_compile_max_rows };
    try Tr.prepareRegions(&g, &c, &rt, false);
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &ws);
    defer k.deinit(&g);
    const lk = k.at(0);
    try testing.expectEqual(ws[0].wo_a.w, lk.proj.?.woa.w);
    try testing.expectEqual(ws[0].wo_b.s, lk.proj.?.wo_b.scales);
    const inv = try Tr.swaInvFreq(&g, &c);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    // Verify rows (5): five FMA launches (wq_a, wq_b, wkv, the packed wo_a, wo_b), no quantized
    // matmul and no wo_a dequantize; the output bf16 (the stream stays bf16).
    const l0 = g.prepared_launches;
    var n0 = g.nodes.items.len;
    const out = try Tr.attention(&g, &p, &c, &rt, lk, li, &ws[0], inv, try g.input(&.{ 1, 5, 5120 }, .bfloat16), try g.arange(0, 5, 1, .int32), &cache, &shared);
    try expectShape(&g, out, &.{ 1, 5, 5120 }, .bfloat16);
    try expectStage(&g, &p, "attn.qr", &.{ 1, 5, @intCast(c.q_lora_rank) }, .bfloat16);
    try testing.expectEqual(l0 + 5, g.prepared_launches);
    try testing.expect(noneOf(&g, n0, .qmm) and noneOf(&g, n0, .dequantize));
    // A prefill width (9 rows): no launch; K22's prep and the per-call wo_a dequantize (no W97).
    n0 = g.nodes.items.len;
    _ = try Tr.attention(&g, &p, &c, &rt, lk, li, &ws[0], inv, try g.input(&.{ 1, 9, 5120 }, .bfloat16), try g.arange(5, 14, 1, .int32), &cache, &shared);
    try testing.expectEqual(l0 + 5, g.prepared_launches);
    try testing.expect(!noneOf(&g, n0, .dequantize));
    // A pair off a site's pinned geometry, or another codec, is refused by name.
    var bad = ws;
    bad[0].wo_b = try qIn(&g, 5120, 4096, .mxfp8);
    try testing.expectError(error.RcProjGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &bad));
    bad = ws;
    bad[0].wq_a.mode = .mxfp4;
    try testing.expectError(error.RcProjGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &bad));
}

test "dsv41 graph: the HC tape binds over the bf16 stream only, takes the verify rows' HC tails, and leaves prefill widths stock" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    const ws = [_]LayerW(u32){try traceLayerW(&g, &c, c.layers[0])};
    const w = &ws[0];
    // Without RCPROJ the stream turns f32 after the first attention: refused by name.
    try testing.expectError(error.HcTapeStream, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &.{ .rc_tape = true }, &ws));
    var bad = c;
    bad.rms_norm_eps = 1e-6;
    try testing.expectError(error.HcTapeGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &.{ .rc_tape = true, .rc_proj = true }, &ws));
    const rt: Routes = .{ .rc_tape = true, .rc_proj = true, .rc_sinkhorn = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &ws);
    defer k.deinit(&g);
    const lk = k.at(0);
    try testing.expect(lk.tape.? == &k.tape.? and lk.premix_attn == null);
    const bf = Dtype.bfloat16;
    const f32_ = Dtype.float32;
    for ([_]c_int{ 5, 9 }) |rows| {
        const on = rows <= rc_max_rows;
        const h = try g.input(&.{ 1, rows, 4, 5120 }, bf);
        const l0 = g.prepared_launches;
        var n0 = g.nodes.items.len;
        // seg1: collapse_norm + mixfin (the premix unbound: the stock GEMM) + the Sinkhorn kernel.
        const a = try Tr.hcAttnPrep(&g, &c, lk, h, try g.input(&.{ 1, rows, 4 }, f32_), w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm);
        try expectShape(&g, a[0], &.{ 1, rows, 5120 }, bf);
        try expectShape(&g, a[1], &.{ 1, rows, 4 }, f32_);
        try expectShape(&g, a[3], &.{ 1, rows, 4, 4 }, f32_);
        try testing.expectEqual(l0 + @as(usize, if (on) 3 else 1), g.prepared_launches);
        try testing.expectEqual(on, noneOf(&g, n0, .rsqrt));
        // seg2': the fused combine + collapse + norm, mixfin, the Sinkhorn.
        n0 = g.nodes.items.len;
        const ao = try g.input(&.{ 1, rows, 5120 }, bf);
        const f = try Tr.hcFfnPrep(&g, &c, lk, ao, h, a[1], a[2], a[3], w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm);
        try expectShape(&g, f[0], &.{ 1, rows, 5120 }, bf);
        try expectShape(&g, f[1], &.{ 1, rows, 4, 5120 }, bf);
        try expectShape(&g, f[3], &.{ 1, rows, 4, 4 }, f32_);
        try testing.expectEqual(l0 + @as(usize, if (on) 6 else 2), g.prepared_launches);
        try testing.expectEqual(on, noneOf(&g, n0, .einsum));
        // seg3: the MoE-side combine.
        n0 = g.nodes.items.len;
        const out = try Tr.hcPostRoute(&g, &c, &rt, lk, ao, f[1], f[2], f[3]);
        try expectShape(&g, out, &.{ 1, rows, 4, 5120 }, bf);
        try testing.expectEqual(l0 + @as(usize, if (on) 7 else 2), g.prepared_launches);
        try testing.expectEqual(on, noneOf(&g, n0, .einsum));
    }
    // The whole layer at verify rows: the MoE-side combine is the tape's too (5 RCPROJ + 3 + 3 + 1).
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const lrt: Routes = .{ .rc_tape = true, .rc_proj = true, .rc_sinkhorn = true, .selected_keys = true };
    var lkx = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &lrt, &ws);
    defer lkx.deinit(&g);
    var cache = Tr.Cache.init(c.layers[0], c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    const si: StandIn(TraceOps) = .{ .scale = try g.input(&.{384}, f32_) };
    const l1 = g.prepared_launches;
    const o = try Tr.layer(&g, &p, &c, &lrt, lkx.at(0), c.layers[0], w, try Tr.swaInvFreq(&g, &c), try g.input(&.{ 1, 5, 4, 5120 }, bf), try g.input(&.{ 1, 5, 4 }, f32_), try g.arange(0, 5, 1, .int32), &cache, &shared, si);
    try expectShape(&g, o.h, &.{ 1, 5, 4, 5120 }, bf);
    try testing.expectEqual(l1 + 12, g.prepared_launches);
}

test "dsv41 graph: the K36 fused glue binds per layer beside RCPROJ, takes the verify rows' norms and RoPEs, and leaves prefill widths stock" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[0];
    const ws = [_]LayerW(u32){ try traceLayerW(&g, &c, li), try traceLayerW(&g, &c, c.layers[3]) };
    try testing.expectError(error.FusedProjNeedsProj, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &.{ .rc_fused_proj = true }, &ws));
    var bad = c;
    bad.n_heads = 32;
    try testing.expectError(error.FusedProjGeometry, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &.{ .rc_fused_proj = true, .rc_proj = true }, &ws));
    bad = c;
    bad.rms_norm_eps = 1e-6;
    try testing.expectError(error.FusedProjEps, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &.{ .rc_fused_proj = true, .rc_proj = true }, &ws));
    const rt: Routes = .{ .rc_proj = true, .rc_fused_proj = true, .selected_keys = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &ws);
    defer k.deinit(&g);
    const lk = k.at(0);
    try testing.expectEqual(ws[0].q_norm, lk.fused.?.q_norm);
    try testing.expectEqual(ws[1].kv_norm, k.at(1).fused.?.kv_norm);
    const inv = try Tr.swaInvFreq(&g, &c);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    // Verify rows (5): 5 RCPROJ + 4 glue launches; no eager RMSNorm (rsqrt) in the chain.
    const l0 = g.prepared_launches;
    var n0 = g.nodes.items.len;
    const out = try Tr.attention(&g, &p, &c, &rt, lk, li, &ws[0], inv, try g.input(&.{ 1, 5, 5120 }, .bfloat16), try g.arange(0, 5, 1, .int32), &cache, &shared);
    try expectShape(&g, out, &.{ 1, 5, 5120 }, .bfloat16);
    try expectStage(&g, &p, "attn.q", &.{ 1, 5, 64, 512 }, .bfloat16);
    try expectStage(&g, &p, "attn.kv_new", &.{ 1, 5, 512 }, .bfloat16);
    try testing.expectEqual(l0 + 9, g.prepared_launches);
    try testing.expect(noneOf(&g, n0, .rsqrt));
    // A prefill width: no launch, the eager glue.
    n0 = g.nodes.items.len;
    _ = try Tr.attention(&g, &p, &c, &rt, lk, li, &ws[0], inv, try g.input(&.{ 1, 9, 5120 }, .bfloat16), try g.arange(5, 14, 1, .int32), &cache, &shared);
    try testing.expectEqual(l0 + 9, g.prepared_launches);
    try testing.expect(!noneOf(&g, n0, .rsqrt));
}

test "dsv41 graph: the m1rows head takes rows 1..8, M 5 and 7 padded to M + 1, f32 logits of x's rows" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var hr = try kr.HeadRows(TraceOps).init(&g, &reg, try g.input(&.{ 129280, 5120 }, .bfloat16), null);
    defer hr.deinit(&g);
    for (1..9) |mu| {
        const m: c_int = @intCast(mu);
        const l0 = g.prepared_launches;
        const n0 = g.nodes.items.len;
        const y = try Tr.headRows(&g, &hr, try g.input(&.{ 1, m, 5120 }, .bfloat16));
        try expectShape(&g, y, &.{ 1, m, 129280 }, .float32);
        try testing.expectEqual(l0 + 1, g.prepared_launches);
        var padded = false;
        for (g.nodes.items[n0..]) |nd| padded = padded or (nd.op == .concat and nd.shape.eql(ops.Shape.of(&.{ m + 1, 5120 })));
        try testing.expectEqual(mu == 5 or mu == 7, padded);
    }
    // Another head shape is refused (the kernel's pinned [129280, 5120]).
    try testing.expectError(error.RouteInput, kr.HeadRows(TraceOps).init(&g, &reg, try g.input(&.{ 4096, 5120 }, .bfloat16), null));
}

test "dsv41 graph: the DENSE16 o-projection takes its rhs index pair from construction (no per-call arange or u32 cast)" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    var w = try traceLayerW(&g, &c, c.layers[1]);
    w.oproj_idx = try Tr.oprojIndices(&g, &c);
    const s: c_int = 64;
    const og = try g.input(&.{ @intCast(c.o_groups), s, @intCast(c.n_heads * c.head_dim / c.o_groups) }, .float32);
    const from = g.nodes.items.len;
    const out = try Tr.outProjDense16(&g, &c, og, &w, 1, s, .float32);
    try testing.expectEqual(@as(u8, 3), g.shapeOf(out).n);
    var aranges: usize = 0;
    var u32_casts: usize = 0;
    var gathers: usize = 0;
    for (g.nodes.items[from..]) |nd| {
        aranges += @intFromBool(nd.op == .arange);
        u32_casts += @intFromBool(nd.op == .astype and nd.dtype == .uint32);
        gathers += @intFromBool(nd.op == .gather_qmm);
    }
    try testing.expectEqual(@as(usize, 0), aranges);
    try testing.expectEqual(@as(usize, 0), u32_casts);
    // Both projections still gather through their packed views (wo_a's groups, wo_b's one expert).
    try testing.expectEqual(@as(usize, 2), gathers);
}

test "dsv41 graph: kv16-opt DENSE16 bf16 out: wo_b's product as is (no f32 widening); the f32 route widens once" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    var w = try traceLayerW(&g, &c, c.layers[1]);
    w.oproj_idx = try Tr.oprojIndices(&g, &c);
    const s: c_int = 64;
    const og = try g.input(&.{ @intCast(c.o_groups), s, @intCast(c.n_heads * c.head_dim / c.o_groups) }, .float32);
    inline for (.{ Dtype.bfloat16, Dtype.float32 }, .{ 0, 1 }) |dt, widen| {
        const from = g.nodes.items.len;
        const out = try Tr.outProjDense16(&g, &c, og, &w, 1, s, dt);
        try testing.expectEqual(dt, g.dtypeOf(out));
        var f32_casts: usize = 0;
        for (g.nodes.items[from..]) |nd| f32_casts += @intFromBool(nd.op == .astype and nd.dtype == .float32);
        try testing.expectEqual(@as(usize, widen), f32_casts);
    }
}

test "dsv41 graph: the prefill attention core takes the prompt widths per layer kind; no gathered KVg; verify widths keep the chain" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const rt: Routes = .{ .prefill_attn = true, .selected_keys = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &.{});
    defer k.deinit(&g);
    // Layer 0 (the bf16 stream, window only), layer 1 (f32, window only), layer 2 (f32, compressed).
    try testing.expectEqual(@as(u8, 0), k.prefill_attn_kind[0]);
    try testing.expectEqual(@as(u8, 1), k.prefill_attn_kind[1]);
    try testing.expectEqual(@as(u8, 2), k.prefill_attn_kind[2]);
    const Case = struct { l: usize, dt: Dtype };
    for ([_]Case{ .{ .l = 0, .dt = .bfloat16 }, .{ .l = 2, .dt = .float32 } }) |cs| {
        const li = c.layers[cs.l];
        const w = try traceLayerW(&g, &c, li);
        const inv = if (li.ratio > 0) try Tr.yarnInvFreq(&g, &c) else try Tr.swaInvFreq(&g, &c);
        var cache = Tr.Cache.init(li, c.window, .{});
        defer cache.deinit(&g);
        var shared: Tr.Share = .{};
        // A prompt width (64 rows): the core's launches; nothing gathered.
        const n0 = g.nodes.items.len;
        const out = try Tr.attention(&g, &p, &c, &rt, k.at(cs.l), li, &w, inv, try g.input(&.{ 1, 64, 5120 }, cs.dt), try g.arange(0, 64, 1, .int32), &cache, &shared);
        try expectShape(&g, out, &.{ 1, 64, 5120 }, .float32);
        // The core's launches (QK, softmax, PV: kernel nodes); nothing gathered.
        try testing.expect(!noneOf(&g, n0, .kernel));
        try testing.expect(noneOf(&g, n0, .take));
        try expectStage(&g, &p, "attn.o_grouped", &.{ 8, 64, 4096 }, .float32);
        // A verify width (8 rows) keeps the eager selected-keys chain.
        const n1 = g.nodes.items.len;
        _ = try Tr.attention(&g, &p, &c, &rt, k.at(cs.l), li, &w, inv, try g.input(&.{ 1, 8, 5120 }, cs.dt), try g.arange(64, 72, 1, .int32), &cache, &shared);
        try testing.expect(noneOf(&g, n1, .kernel));
    }
    // The construction self-check builds for every installed kind (one bool scalar each).
    var ws: [v41.max_layers]LayerW(u32) = undefined;
    for (0..c.n_layers) |l| ws[l] = try traceLayerW(&g, &c, c.layers[l]);
    const scratch = try testing.allocator.alloc(f32, 64 * 64 * 512);
    defer testing.allocator.free(scratch);
    const oks = try Tr.prefillAttnCheck(&g, &c, &k, ws[0..c.n_layers], scratch);
    for (oks) |o| {
        try testing.expect(o != null);
        try testing.expectEqual(Dtype.bool_, g.dtypeOf(o.?));
        try testing.expectEqual(@as(u8, 0), g.shapeOf(o.?).n);
    }
    // Every prefill route's check builds (all routes on: 3 core kinds, score, select, 2 x 2 HC norms, DENSE16,
    // 2 compiled HC posts, JOINLESS, the combine).
    const all: Routes = .{ .prefill_attn = true, .prefill_index = true, .prefill_hc = true, .prefill_combine = true, .prefill_joinless = true, .selected_keys = true };
    var ka = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &all, &.{});
    defer ka.deinit(&g);
    var checks: [16]Tr.RouteCheck = undefined;
    const all_o: Routes = .{ .prefill_attn = true, .prefill_index = true, .prefill_hc = true, .prefill_combine = true, .prefill_oproj = true, .prefill_joinless = true, .prefill_hc_post = true, .selected_keys = true };
    const oi = try Tr.oprojIndices(&g, &c);
    for (ws[0..c.n_layers]) |*w| w.oproj_idx = oi;
    try g.prepareTape(Tr.HcPost, &c);
    const n = try Tr.prefillRoutesCheck(&g, &c, &all_o, &ka, ws[0..c.n_layers], scratch, &checks);
    try testing.expectEqual(@as(usize, 14), n);
    for (checks[0..n]) |ck| {
        try testing.expectEqual(Dtype.bool_, g.dtypeOf(ck.ok));
        try testing.expectEqual(@as(u8, 0), g.shapeOf(ck.ok).n);
    }
    var bad = c;
    bad.window = 64;
    try testing.expectError(error.RouteInput, Tr.Kernels.init(testing.allocator, &g, &reg, &bad, &rt, &.{}));
}

test "dsv41 graph: the prefill core's window selection is built by a chunk's first layer and reused by the next" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    const rt: Routes = .{ .prefill_attn = true, .selected_keys = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &.{});
    defer k.deinit(&g);
    const li = c.layers[1];
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.swaInvFreq(&g, &c);
    // Two layers' window lanes at the same state (K16: every layer has taken the same chunks), one chunk's Share.
    var cache_a = Tr.Cache.init(li, c.window, .{});
    defer cache_a.deinit(&g);
    var cache_b = Tr.Cache.init(li, c.window, .{});
    defer cache_b.deinit(&g);
    var shared: Tr.Share = .{};
    const pos = try g.arange(0, 64, 1, .int32);
    const n0 = g.nodes.items.len;
    _ = try Tr.attention(&g, NoProbe{}, &c, &rt, k.at(1), li, &w, inv, try g.input(&.{ 1, 64, 5120 }, .float32), pos, &cache_a, &shared);
    const first = g.nodes.items.len - n0;
    const idx = shared.win_idx.?;
    const valid = shared.win_valid.?;
    try testing.expect(!noneOf(&g, n0, .arange));
    const n1 = g.nodes.items.len;
    _ = try Tr.attention(&g, NoProbe{}, &c, &rt, k.at(1), li, &w, inv, try g.input(&.{ 1, 64, 5120 }, .float32), pos, &cache_b, &shared);
    // The second layer builds no selection (its only arange was the selection's) and holds the first one's arrays.
    try testing.expect(noneOf(&g, n1, .arange));
    try testing.expect(g.nodes.items.len - n1 < first);
    try testing.expectEqual(idx, shared.win_idx.?);
    try testing.expectEqual(valid, shared.win_valid.?);
    try expectShape(&g, idx, &.{ 64, @intCast(c.window) }, .int32);
    // A lane at another length (a different key) builds its own.
    var cache_c = Tr.Cache.init(li, c.window, .{});
    defer cache_c.deinit(&g);
    _ = try Tr.attention(&g, NoProbe{}, &c, &rt, k.at(1), li, &w, inv, try g.input(&.{ 1, 40, 5120 }, .float32), try g.arange(0, 40, 1, .int32), &cache_c, &shared);
    try testing.expect(shared.win_idx.? != idx);
}

test "dsv41 graph: the prefill indexer scores and selects in two launches at prompt widths; verify widths keep the chain" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const rt: Routes = .{ .prefill_index = true, .selected_keys = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, &.{});
    defer k.deinit(&g);
    var l: usize = 0;
    while (!c.layers[l].index_source) l += 1;
    const li = c.layers[l];
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.yarnInvFreq(&g, &c);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    const n0 = g.nodes.items.len;
    _ = try Tr.attention(&g, &p, &c, &rt, k.at(l), li, &w, inv, try g.input(&.{ 1, 64, 5120 }, .float32), try g.arange(0, 64, 1, .int32), &cache, &shared);
    const n_comp: c_int = @intCast(64 / li.ratio);
    try expectStage(&g, &p, "attn.index_score", &.{ 1, 64, n_comp }, .float32);
    try expectStage(&g, &p, "attn.selected_idx", &.{ 1, 64, @min(n_comp, @as(c_int, @intCast(c.index_topk))) }, .int32);
    try testing.expect(!noneOf(&g, n0, .kernel));
    try testing.expect(noneOf(&g, n0, .argpartition) and noneOf(&g, n0, .argsort));
    // A verify width (8 rows) keeps the eager score (its einsum) and select; no launches.
    const n1 = g.nodes.items.len;
    _ = try Tr.attention(&g, &p, &c, &rt, k.at(l), li, &w, inv, try g.input(&.{ 1, 8, 5120 }, .float32), try g.arange(64, 72, 1, .int32), &cache, &shared);
    try testing.expect(noneOf(&g, n1, .kernel));
    try testing.expect(!noneOf(&g, n1, .einsum));
}

test "dsv41 graph: the verify-row routes (C23, C27-C29) bind per layer, take rows <= 8, and their self-checks build" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    var ws: [v41.max_layers]LayerW(u32) = undefined;
    for (0..c.n_layers) |l| ws[l] = try traceLayerW(&g, &c, c.layers[l]);
    const rt: Routes = .{ .rc_proj = true, .rc_smallm = true, .rc_mxfp8_rows = true, .rc_index_topk = true, .rc_attn_softmax = true, .selected_keys = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, ws[0..c.n_layers]);
    defer k.deinit(&g);
    var l: usize = 0;
    while (!(c.layers[l].index_source and c.layers[l].kv_source)) l += 1;
    const li = c.layers[l];
    const ms = &k.minv.items[l];
    try testing.expect(ms.cmp_wkv != null and ms.wk != null and ms.wproj != null and ms.idx_wq_b != null and ms.sh_w1 != null and ms.sh_w2 != null);
    const scratch = try testing.allocator.alloc(f32, 5 * 640 * 512);
    defer testing.allocator.free(scratch);
    var checks: [40]Tr.RouteCheck = undefined;
    const n = try Tr.decodeRoutesCheck(&g, &c, &k, ws[0..c.n_layers], scratch, &checks);
    try testing.expect(n >= 7);
    for (checks[0..n]) |ck| {
        try testing.expectEqual(Dtype.bool_, g.dtypeOf(ck.ok));
        try testing.expectEqual(@as(u8, 0), g.shapeOf(ck.ok).n);
    }
    // A 5-row verify attention on an index source: the sites' launches, no sort / argsort select.
    const inv = if (li.ratio > 0) try Tr.yarnInvFreq(&g, &c) else try Tr.swaInvFreq(&g, &c);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    const n0 = g.nodes.items.len;
    const out = try Tr.attention(&g, &p, &c, &rt, k.at(l), li, &ws[l], inv, try g.input(&.{ 1, 5, 5120 }, .bfloat16), try g.arange(0, 5, 1, .int32), &cache, &shared);
    try expectShape(&g, out, &.{ 1, 5, 5120 }, .bfloat16);
    try testing.expect(!noneOf(&g, n0, .kernel));
    try testing.expect(noneOf(&g, n0, .sort) and noneOf(&g, n0, .argsort));
}

test "dsv41 graph: DENSE_RC binds the mxfp8 rows sites on RCPROJ over the stacked shared gate | up, one launch each" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    var ws: [v41.max_layers]LayerW(u32) = undefined;
    for (0..c.n_layers) |l| ws[l] = try traceLayerW(&g, &c, c.layers[l]);
    const rt: Routes = .{ .rc_proj = true, .rc_smallm = true, .rc_mxfp8_rows = true, .dense_rc = true, .selected_keys = true };
    try testing.expectError(error.DenseRcNeedsStack, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, ws[0..c.n_layers]));
    const I: c_int = @intCast(c.moe_intermediate_size);
    for (ws[0..c.n_layers]) |*w| w.sh_w13 = .{ .w = try g.input(&.{ 2 * I, @intCast(c.hidden_size / 4) }, .uint32), .s = try g.input(&.{ 2 * I, @intCast(c.hidden_size / 32) }, .uint8), .mode = .mxfp8 };
    var bad = rt;
    bad.rc_mxfp8_rows = false;
    try testing.expectError(error.DenseRcNeedsRows, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &bad, ws[0..c.n_layers]));
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, ws[0..c.n_layers]);
    defer k.deinit(&g);
    var l: usize = 0;
    while (!c.layers[l].index_source) l += 1;
    const ms = &k.minv.items[l];
    try testing.expect(ms.sh_w13 != null and ms.sh_w1 == null and ms.sh_w3 == null and ms.sh_w2 != null and ms.idx_wq_b != null);
    try testing.expectEqual(xk.Kernel.q3rc_mxfp8_fma, ms.sh_w13.?.e.kernel);
    // The shared expert at 5 rows: one RCPROJ launch (gate | up stacked), w2 on m1rows, no qmm.
    const l0 = g.launched.items.len;
    const n0 = g.nodes.items.len;
    const y = try Tr.sharedExpertMinvFor(&g, &c, &rt, &ws[l], ms, try g.input(&.{ 5, 5120 }, .bfloat16));
    try expectShape(&g, y, &.{ 5, 5120 }, .bfloat16);
    try testing.expectEqual(@as(usize, 1), g.launchesOf(l0, .q3rc_mxfp8_fma));
    try testing.expectEqual(@as(usize, 1), g.launchesOf(l0, .dsv41_mxfp8_m1rows));
    try testing.expectEqual(xk.Kernel.dsv41_mxfp8_m1rows, ms.idx_wq_b.?.e.kernel);
    for (g.nodes.items[n0..]) |nd| try testing.expect(nd.op != .qmm);
}

test "dsv41 graph: HC mixes split pre / post / a Sinkhorn comb with 1 + 1 + 2 x 19 normalisations" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    const h = try g.input(&.{ 1, 3, 4, 5120 }, .bfloat16);
    const w = try traceLayerW(&g, &c, c.layers[0]);
    const mark = g.nodes.items.len;
    const m = try Tr.hcMixes(&g, &c, .{}, h, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale);
    try expectShape(&g, m.pre, &.{ 1, 3, 4 }, .float32);
    try expectShape(&g, m.post, &.{ 1, 3, 4 }, .float32);
    try expectShape(&g, m.comb, &.{ 1, 3, 4, 4 }, .float32);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    var divs: usize = 0;
    var softmaxes: usize = 0;
    for (seq) |o| switch (o) {
        .div => divs += 1,
        .softmax => softmaxes += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), softmaxes);
    try testing.expectEqual(@as(usize, 1 + 2 * 19), divs);
    const x = try Tr.hcPre(&g, h, try g.input(&.{ 1, 3, 4 }, .float32));
    try expectShape(&g, x, &.{ 1, 3, 5120 }, .bfloat16);
    const post = try Tr.hcPost(&g, try g.input(&.{ 1, 3, 5120 }, .float32), h, m.post, m.comb);
    try expectShape(&g, post, &.{ 1, 3, 4, 5120 }, .float32);
}

test "dsv41 graph: the real layer-0 (SWA) forward turns the bf16 residual f32 at its attention" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[0];
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.swaInvFreq(&g, &c);
    try expectShape(&g, inv, &.{32}, .float32);
    const rows = try g.input(&.{ 1, 5, 5120 }, .bfloat16);
    const e = try Tr.expandEmbedding(&g, &c, rows);
    try expectShape(&g, e.h, &.{ 1, 5, 4, 5120 }, .bfloat16);
    try expectShape(&g, e.pre_mix, &.{ 1, 5, 4 }, .float32);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    const pos = try g.arange(0, 5, 1, .int32);
    const out = try Tr.layer(&g, &p, &c, &stock, .{}, li, &w, inv, e.h, e.pre_mix, pos, &cache, &shared, TraceRouted{});
    try expectStage(&g, &p, "attn.x", &.{ 1, 5, 5120 }, .bfloat16);
    try expectStage(&g, &p, "attn.qr", &.{ 1, 5, 1280 }, .bfloat16);
    try expectStage(&g, &p, "attn.q", &.{ 1, 5, 64, 512 }, .bfloat16);
    try expectStage(&g, &p, "attn.kv_new", &.{ 1, 5, 512 }, .bfloat16);
    try expectStage(&g, &p, "attn.o", &.{ 1, 5, 64, 512 }, .float32);
    // o-LoRA in f32 feeds wo_b an f32 input: the attention output (and from here
    // the residual stream) is f32, as in the Python stock path.
    try expectStage(&g, &p, "attn.out", &.{ 1, 5, 5120 }, .float32);
    try expectStage(&g, &p, "hc1.h", &.{ 1, 5, 4, 5120 }, .float32);
    try expectStage(&g, &p, "gate.indices", &.{ 5, 6 }, .int32);
    try expectStage(&g, &p, "gate.weights", &.{ 5, 6 }, .float32);
    try expectStage(&g, &p, "moe.shared", &.{ 5, 5120 }, .float32);
    try expectStage(&g, &p, "moe.y", &.{ 1, 5, 5120 }, .float32);
    try expectShape(&g, out.h, &.{ 1, 5, 4, 5120 }, .float32);
    try expectShape(&g, out.pre_mix, &.{ 1, 5, 4 }, .float32);
    try testing.expect(p.get("attn.topk_mask") == null); // SWA: no compressed branch
    try expectShape(&g, (try cache.window.view(&g)).?, &.{ 1, 5, 512 }, .bfloat16);
    const fin = try Tr.finalNorm(&g, &c, out.h, out.pre_mix, try g.input(&.{5120}, .bfloat16));
    try expectShape(&g, fin, &.{ 1, 5, 5120 }, .float32);
    const logits = try Tr.head(&g, &stock, fin, .{ .dense = try g.input(&.{ 4096, 5120 }, .bfloat16) });
    try expectShape(&g, logits, &.{ 1, 5, 4096 }, .float32);
}

test "dsv41 graph: layer 2 (Full, ratio 2) pools a group every 2 tokens across prefill and decode" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[2];
    try testing.expectEqual(v41.LayerMode.full, li.mode);
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.yarnInvFreq(&g, &c);
    try expectShape(&g, inv, &.{32}, .float32);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    // prefill 5 tokens, then decode 2 single tokens
    const steps = [_]struct { s: c_int, rows: c_int, comp: c_int, fresh: bool }{
        .{ .s = 5, .rows = 5, .comp = 2, .fresh = true },
        .{ .s = 1, .rows = 6, .comp = 3, .fresh = true },
        .{ .s = 1, .rows = 7, .comp = 3, .fresh = false },
    };
    var pos0: c_int = 0;
    for (steps) |st| {
        var shared: Tr.Share = .{};
        p.names.clearRetainingCapacity();
        p.nodes.clearRetainingCapacity();
        // kv16: the bf16 stream in; the window ring and the compressed store bf16, the index keys f32.
        const x = try g.input(&.{ 1, st.s, 5120 }, .bfloat16);
        const pos = try g.arange(@floatFromInt(pos0), @floatFromInt(pos0 + st.s), 1, .int32);
        const out = try Tr.attention(&g, &p, &c, &stock, .{}, li, &w, inv, x, pos, &cache, &shared);
        try expectShape(&g, out, &.{ 1, st.s, 5120 }, .float32);
        try expectShape(&g, (try cache.window.view(&g)).?, &.{ 1, st.rows, 512 }, .bfloat16);
        try expectShape(&g, (try cache.compress.view(&g)).?, &.{ 1, st.comp, 512 }, .bfloat16);
        try expectShape(&g, (try cache.index.view(&g)).?, &.{ 1, st.comp, 128 }, .float32);
        try testing.expectEqual(p.get("attn.compress_new") != null, st.fresh);
        try expectStage(&g, &p, "attn.index_score", &.{ 1, st.s, st.comp }, .float32);
        try expectStage(&g, &p, "attn.topk_mask", &.{ 1, st.s, st.comp }, .bool_);
        try testing.expectEqual(shared.topk_mask.?, p.get("attn.topk_mask").?);
        try testing.expect(shared.candidates == null); // layer 2 is not the candidate source
        pos0 += st.s;
    }
    try testing.expectEqual(@as(u32, 7), cache.nFed());
    try expectShape(&g, (try cache.frontier.?.kv.view(&g)).?, &.{ 1, 7, 512 }, .float32);
}

test "dsv41 graph: under the window ring the attention sees a bounded window, one mask per forward" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const inv = try Tr.swaInvFreq(&g, &c);
    // layers 0 and 1 (both SWA) in lockstep: prefill 300, then 20 decode tokens
    var caches: [2]Tr.Cache = undefined;
    var ws: [2]Tr.W = undefined;
    for (&caches, &ws, 0..) |*cc, *w, l| {
        cc.* = Tr.Cache.init(c.layers[l], c.window, .{ .route = .window_ring });
        w.* = try traceLayerW(&g, &c, c.layers[l]);
    }
    defer for (&caches) |*cc| cc.deinit(&g);
    var pos0: c_int = 0;
    for (0..21) |step| {
        const s: c_int = if (step == 0) 300 else 1;
        var shared: Tr.Share = .{};
        const pos = try g.arange(@floatFromInt(pos0), @floatFromInt(pos0 + s), 1, .int32);
        var masks: [2]u32 = undefined;
        for (0..2) |l| {
            const x = try g.input(&.{ 1, s, 5120 }, .bfloat16);
            p.names.clearRetainingCapacity();
            p.nodes.clearRetainingCapacity();
            _ = try Tr.attention(&g, &p, &c, &stock, .{}, c.layers[l], &ws[l], inv, x, pos, &caches[l], &shared);
            masks[l] = shared.win_mask.?;
            caches[l].advance(@intCast(s));
        }
        try testing.expectEqual(masks[0], masks[1]); // the second layer reuses the first's mask
        pos0 += s;
        const view = (try caches[0].window.view(&g)).?;
        const rows = g.shapeOf(view).dim(1);
        try testing.expectEqual(@as(c_int, pos0), @as(c_int, @intCast(caches[0].window.dropOffset())) + rows);
        try testing.expect(rows <= 300);
        try expectShape(&g, shared.win_mask.?, &.{ 1, s, rows }, .bool_);
        try expectStage(&g, &p, "attn.o", &.{ 1, s, 64, 512 }, .float32);
    }
    // 300 prefill rows, then the first decode token compacts to window + 8 + 8 = 144 rows.
    try testing.expectEqual(@as(u32, 321 - 144 - 20), caches[1].window.dropOffset());
}

test "dsv41 graph: reuse, candidate and reindex layers share one forward's runtime (mini geometry)" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try miniConfig();
    var caches: [5]Tr.Cache = undefined;
    for (&caches, 0..) |*cc, l| cc.* = Tr.Cache.init(c.layers[l], c.window, .{});
    defer for (&caches) |*cc| cc.deinit(&g);
    var shared: Tr.Share = .{};
    const s: c_int = 9;
    const pos = try g.arange(0, 9, 1, .int32);
    const inv_c = try Tr.yarnInvFreq(&g, &c);
    for (1..5) |l| {
        const li = c.layers[l];
        const w = try traceLayerW(&g, &c, li);
        const x = try g.input(&.{ 1, s, ci(c.hidden_size) }, .float32);
        p.names.clearRetainingCapacity();
        p.nodes.clearRetainingCapacity();
        _ = try Tr.attention(&g, &p, &c, &stock, .{}, li, &w, inv_c, x, pos, &caches[l], &shared);
        switch (l) {
            // Full, ratio 2: 4 groups from 9 tokens.
            1 => try expectStage(&g, &p, "attn.topk_mask", &.{ 1, 9, 4 }, .bool_),
            // Reuse: reads layer 1's selection, computes none.
            2 => {
                try testing.expect(p.get("attn.index_score") == null);
                try testing.expectEqual(@as(u32, 0), caches[2].compress.rows());
                try expectStage(&g, &p, "attn.topk_mask", &.{ 1, 9, 4 }, .bool_);
            },
            // Full, ratio 1, the candidate source: sets the block mask.
            3 => {
                try expectStage(&g, &p, "attn.topk_mask", &.{ 1, 9, 9 }, .bool_);
                try expectShape(&g, shared.candidates.?, &.{ 1, 9, 9 }, .bool_);
            },
            // Reindex: its own queries over layer 3's keys, masked by the candidates.
            4 => {
                try expectStage(&g, &p, "attn.index_score", &.{ 1, 9, 9 }, .float32);
                try testing.expectEqual(@as(u32, 0), caches[4].compress.rows());
            },
            else => unreachable,
        }
    }
}

test "dsv41 graph: router, shared expert and Engram apply keep the Python dtypes" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const w = try traceLayerW(&g, &c, c.layers[3]);
    const xf = try g.input(&.{ 5, 5120 }, .float32);
    const r = try Tr.router(&g, &p, &c, &stock, .{}, &w, xf);
    try expectShape(&g, r.indices, &.{ 5, 6 }, .int32);
    try expectShape(&g, r.weights, &.{ 5, 6 }, .float32);
    try expectStage(&g, &p, "gate.scores", &.{ 5, 384 }, .float32);
    const sh = try Tr.sharedExpert(&g, &c, &w, xf);
    try expectShape(&g, sh, &.{ 5, 5120 }, .float32);
    const ew: EngramW(u32) = .{ .wkv = try qIn(&g, 5120 * 5, 24 * 256, .mxfp8), .q_weight = try g.input(&.{ 4, 5120 }, .float32), .k_weight = try g.input(&.{ 4, 5120 }, .float32) };
    const hid = try g.input(&.{ 1, 3, 4, 5120 }, .bfloat16);
    const rows = try g.input(&.{ 1, 3, 24, 256 }, .bfloat16);
    const e = try Tr.engramApply(&g, &c, ew, hid, rows);
    try expectShape(&g, e, &.{ 1, 3, 4, 5120 }, .bfloat16);
    // The row fetch: 3 positions x 24 records of E4M3 words + E8M0 scales -> bf16 rows.
    const codes_b: [72 * 256]u8 = @splat(0);
    const scales_b: [72 * 8]u8 = @splat(0);
    const mark = g.nodes.items.len;
    const er = try Tr.engramRows(&g, try g.hostArray(&codes_b, &.{ 72, 64 }, .uint32), try g.hostArray(&scales_b, &.{ 72, 8 }, .uint8), 1, 3, 24);
    try expectShape(&g, er, &.{ 1, 3, 24, 256 }, .bfloat16);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    try testing.expectEqualSlices(ops.Op, &.{ .host, .host, .dequantize, .reshape }, seq);
    try testing.expectError(error.HostBytes, g.hostArray(codes_b[0..8], &.{ 72, 64 }, .uint32));
}

test "dsv41 graph: host constants round as the Python floats do" {
    // Goldens from CPython (struct.pack('<f', v)).
    const f32bits = struct {
        fn b(v: f64) u32 {
            return @bitCast(@as(f32, @floatCast(v)));
        }
    }.b;
    try testing.expectEqual(@as(u32, 0x3d3504f3), f32bits(std.math.pow(f64, 512, -0.5)));
    try testing.expectEqual(@as(u32, 0x3c800000), f32bits(std.math.pow(f64, 128, -0.5) * std.math.pow(f64, 32, -0.5)));
    try testing.expectEqual(@as(u32, 0x1e3ce508), f32bits(1e-20));
    try testing.expectEqual(@as(u32, 0x358637bd), f32bits(1e-6));
    // `_yarn_inv_freq` correction range on the real geometry (dim 64, base 160000).
    try testing.expectEqual(Ramp{ .low = 15, .high = 25 }, yarnRamp(64, 160000, 65536, 32, 1));
}

test "dsv41 graph: the routed stand-in keeps the Python table and shapes" {
    var buf: [384]f32 = undefined;
    StandIn(TraceOps).table(&buf);
    try testing.expectEqual(@as(f32, 1.0), buf[0]);
    // Goldens from numpy: 1.0 + np.arange(384, dtype=np.float32) / 384.
    try testing.expectEqual(@as(u32, 0x3F805555), @as(u32, @bitCast(buf[1])));
    try testing.expectEqual(@as(u32, 0x3FFFAAAA), @as(u32, @bitCast(buf[383])));
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const si: StandIn(TraceOps) = .{ .scale = try g.input(&.{384}, .float32) };
    const out = try si.routed(&g, try g.input(&.{ 5, 5120 }, .float32), try g.input(&.{ 5, 6 }, .int32));
    try expectShape(&g, out, &.{ 5, 6, 5120 }, .float32);
}

test "dsv41 graph: K30 selected keys gather each query's window and selected rows, the selection published once" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try miniConfig();
    const rt: Routes = .{ .selected_keys = true };
    var caches: [5]Tr.Cache = undefined;
    for (&caches, 0..) |*cc, l| cc.* = Tr.Cache.init(c.layers[l], c.window, .{ .route = .window_ring });
    defer for (&caches) |*cc| cc.deinit(&g);
    const inv = try Tr.yarnInvFreq(&g, &c);
    var shared: Tr.Share = .{};
    const pos = try g.arange(0, 9, 1, .int32);
    var published: [5]?u32 = @splat(null);
    for (1..5) |l| {
        const w = try traceLayerW(&g, &c, c.layers[l]);
        const x = try g.input(&.{ 1, 9, ci(c.hidden_size) }, .float32);
        p.names.clearRetainingCapacity();
        p.nodes.clearRetainingCapacity();
        const out = try Tr.attention(&g, &p, &c, &rt, .{}, c.layers[l], &w, inv, x, pos, &caches[l], &shared);
        try expectShape(&g, out, &.{ 1, 9, ci(c.hidden_size) }, .float32);
        try expectStage(&g, &p, "attn.o", &.{ 1, 9, 2, 32 }, .float32);
        published[l] = shared.selected_idx;
    }
    // No masked-full attention: no window mask was built.
    try testing.expect(shared.win_mask == null);
    // Layer 1 (index source over 4 groups) publishes k = min(index_topk 4, 4); reuse layer 2 reads it;
    // layer 3 (index source, ratio 1) and reindex layer 4 publish their own over 9 rows.
    try expectShape(&g, published[1].?, &.{ 1, 9, 4 }, .int32);
    try testing.expectEqual(published[1].?, published[2].?);
    try expectShape(&g, published[3].?, &.{ 1, 9, 4 }, .int32);
    try testing.expect(published[4].? != published[3].?);
    // The core compile pads the selection to index_topk and runs as one region at decode rows.
    const rtc: Routes = .{ .selected_keys = true, .core_rows = core_compile_max_rows };
    const w1 = try traceLayerW(&g, &c, c.layers[1]);
    var cache1 = Tr.Cache.init(c.layers[1], c.window, .{});
    defer cache1.deinit(&g);
    var sh1: Tr.Share = .{};
    try testing.expectError(error.RegionNotPrepared, Tr.attention(&g, &p, &c, &rtc, .{}, c.layers[1], &w1, inv, try g.input(&.{ 1, 1, ci(c.hidden_size) }, .float32), try g.arange(0, 1, 1, .int32), &cache1, &sh1));
    try Tr.prepareRegions(&g, &c, &rtc, false);
    var cache2 = Tr.Cache.init(c.layers[1], c.window, .{});
    defer cache2.deinit(&g);
    var sh2: Tr.Share = .{};
    const mark = g.nodes.items.len;
    _ = try Tr.attention(&g, &p, &c, &rtc, .{}, c.layers[1], &w1, inv, try g.input(&.{ 1, 1, ci(c.hidden_size) }, .float32), try g.arange(0, 1, 1, .int32), &cache2, &sh2);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    try testing.expectEqual(@as(usize, 1), std.mem.count(ops.Op, seq, &.{.tape_begin}));
}

test "dsv41 graph: C22 DISPATCH_FUSE: RoPE tables once per family; K30 at a decode row rebuilds its selection and gathers each call" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try miniConfig();
    // rope: a forward's tables are built by each family's first layer and reused.
    var sh: Tr.Share = .{};
    const pos = try g.arange(0, 1, 1, .int32);
    const inv_s = try Tr.swaInvFreq(&g, &c);
    const inv_y = try Tr.yarnInvFreq(&g, &c);
    var swa_li = c.layers[0];
    swa_li.ratio = 0;
    var yarn_li = c.layers[0];
    yarn_li.ratio = 4;
    const m0 = g.nodes.items.len;
    const a1 = try Tr.ropeTables(&g, &sh, swa_li, inv_s, pos);
    const b1 = try Tr.ropeTables(&g, &sh, yarn_li, inv_y, pos);
    const a2 = try Tr.ropeTables(&g, &sh, swa_li, inv_s, pos);
    const b2 = try Tr.ropeTables(&g, &sh, yarn_li, inv_y, pos);
    try testing.expectEqual(a1, a2);
    try testing.expectEqual(b1, b2);
    try testing.expect(a1.cos != b1.cos);
    const rope_ops = try g.opsSince(testing.allocator, m0);
    defer testing.allocator.free(rope_ops);
    try testing.expectEqual(@as(usize, 2), std.mem.count(ops.Op, rope_ops, &.{.cos}));
    const rt: Routes = .{ .selected_keys = true };
    const w = try traceLayerW(&g, &c, c.layers[1]);
    const H = ci(c.n_heads);
    const D = ci(c.head_dim);
    const q = try g.input(&.{ 1, 1, H, D }, .float32);
    const window = try g.input(&.{ 1, ci(c.window), D }, .float32);
    const ckv = try g.input(&.{ 1, 6, D }, .float32);
    const cidx = try g.input(&.{ 1, 1, 4 }, .int32);
    // Every call builds the window selection's arange, both gathers and both gathers' offsets (two more aranges).
    for (0..2) |_| {
        const m = g.nodes.items.len;
        _ = try Tr.sparseAttendSelected(&g, &c, &rt, &w, null, q, window, 0, ckv, cidx, pos);
        const seq = try g.opsSince(testing.allocator, m);
        defer testing.allocator.free(seq);
        try testing.expectEqual(@as(usize, 3), std.mem.count(ops.Op, seq, &.{.arange}));
        try testing.expectEqual(@as(usize, 2), std.mem.count(ops.Op, seq, &.{.take}));
    }
}

test "dsv41 graph: the SharedMid region holds exactly the shared expert's middle (the op chain's ops)" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try miniConfig();
    const I = ci(c.moe_intermediate_size);
    const gl = try g.input(&.{ 5, I }, .bfloat16);
    const ul = try g.input(&.{ 5, I }, .bfloat16);
    const x = try g.input(&.{ 5, ci(c.hidden_size) }, .bfloat16);
    const on: Routes = .{ .prefill_shared_mid = true };
    const m0 = g.nodes.items.len;
    const eager = try Tr.sharedMid(&g, &c, gl, ul, .bfloat16);
    const m1 = g.nodes.items.len;
    var o: [1]TraceOps.T = undefined;
    try testing.expectError(error.RegionNotPrepared, g.tape(Tr.SharedMid, &c, &.{ gl, ul, x }, &o));
    try Tr.prepareRegions(&g, &c, &on, false);
    const m2 = g.nodes.items.len;
    try g.tape(Tr.SharedMid, &c, &.{ gl, ul, x }, &o);
    try expectShape(&g, eager, &.{ 5, I }, .bfloat16);
    try expectShape(&g, o[0], &.{ 5, I }, .bfloat16);
    const e_ops = try g.opsSince(testing.allocator, m0);
    defer testing.allocator.free(e_ops);
    const t_ops = try g.opsSince(testing.allocator, m2);
    defer testing.allocator.free(t_ops);
    // The region holds exactly the chain's ops, between its two markers.
    try testing.expectEqual(ops.Op.tape_begin, t_ops[0]);
    try testing.expectEqual(ops.Op.tape_end, t_ops[t_ops.len - 1]);
    try testing.expectEqualSlices(ops.Op, e_ops[0 .. m1 - m0], t_ops[1 .. t_ops.len - 1]);
    try testing.expectEqual(@as(usize, 1), g.compiles);
}

fn countOps(seq: []const ops.Op) [@typeInfo(ops.Op).@"enum".field_names.len]u32 {
    var n: [@typeInfo(ops.Op).@"enum".field_names.len]u32 = @splat(0);
    for (seq) |o| n[@backingInt(o)] += 1;
    return n;
}

/// The ops one real layer-0 forward records at `rows` query rows under `rt`.
fn layerOps(rt: *const Routes, rows: c_int) ![]ops.Op {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    const li = c.layers[0];
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.swaInvFreq(&g, &c);
    const h = try g.input(&.{ 1, rows, 4, 5120 }, .bfloat16);
    const pm = try g.input(&.{ 1, rows, 4 }, .float32);
    var cache = Tr.Cache.init(li, c.window, .{});
    defer cache.deinit(&g);
    var shared: Tr.Share = .{};
    try Tr.prepareRegions(&g, &c, rt, false);
    const mark = g.nodes.items.len;
    _ = try Tr.layer(&g, NoProbe{}, &c, rt, .{}, li, &w, inv, h, pm, try g.arange(0, @floatFromInt(rows), 1, .int32), &cache, &shared, TraceRouted{});
    return g.opsSince(testing.allocator, mark);
}

test "dsv41 graph: the K22 / K4 / K35 regions hold the eager ops, compiled only at decode / verify rows" {
    const O = ops.Op;
    const eager = try layerOps(&.{}, 1);
    defer testing.allocator.free(eager);
    try testing.expectEqual(@as(usize, 0), std.mem.count(O, eager, &.{.tape_begin}));
    // K22 + K4 at one row: 4 + 3 tapes over the same ops, plus the out tape's flatten reshape.
    const k22k4 = try layerOps(&.{ .attn_rows = attn_compile_max_rows, .hc_rows = hc_compile_max_rows }, 1);
    defer testing.allocator.free(k22k4);
    var want = countOps(eager);
    want[@backingInt(O.reshape)] += 1;
    var got = countOps(k22k4);
    try testing.expectEqual(@as(u32, 7), got[@backingInt(O.tape_begin)]);
    got[@backingInt(O.tape_begin)] = 0;
    got[@backingInt(O.tape_end)] = 0;
    try testing.expectEqual(want, got);
    // K35 at one row: three segments (+ the attention's own K22 tapes) over the same multiset.
    const k35 = try layerOps(&.{ .small_rows = small_stages_max_rows, .attn_rows = attn_compile_max_rows }, 1);
    defer testing.allocator.free(k35);
    got = countOps(k35);
    try testing.expectEqual(@as(u32, 3 + 2), got[@backingInt(O.tape_begin)]);
    got[@backingInt(O.tape_begin)] = 0;
    got[@backingInt(O.tape_end)] = 0;
    try testing.expectEqual(want, got);
    // Past the row caps the eager body runs: K4 / K35 stop at 7 rows, K22 at 32.
    const wide = try layerOps(&.{ .attn_rows = attn_compile_max_rows, .hc_rows = hc_compile_max_rows, .small_rows = small_stages_max_rows }, 8);
    defer testing.allocator.free(wide);
    try testing.expectEqual(@as(usize, 4), std.mem.count(O, wide, &.{.tape_begin}));
    const prefill = try layerOps(&.{ .attn_rows = attn_compile_max_rows, .hc_rows = hc_compile_max_rows, .small_rows = small_stages_max_rows }, 33);
    defer testing.allocator.free(prefill);
    try testing.expectEqual(@as(usize, 0), std.mem.count(O, prefill, &.{.tape_begin}));
}

/// K16's attention side and ffn combine over two layers (kv16: both on the bf16 stream, the attention output joining
/// it at the stream's dtype), one chunk per width in order: the HcPost region's traces (`mx.compile`'s cache).
fn k16HcPostTraces(rt: *const Routes, widths: []const c_int) !usize {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    try Tr.prepareRegions(&g, &c, rt, true);
    const inv = try Tr.swaInvFreq(&g, &c);
    var ws: [2]LayerW(u32) = undefined;
    for (&ws, 0..) |*w, l| w.* = try traceLayerW(&g, &c, c.layers[l]);
    for (widths) |s| {
        var h = try g.input(&.{ 1, s, 4, 5120 }, .bfloat16);
        var pm = try g.input(&.{ 1, s, 4 }, .float32);
        for (&ws, 0..) |*w, l| {
            const li = c.layers[l];
            var cache = Tr.Cache.init(li, c.window, .{});
            defer cache.deinit(&g);
            var shared: Tr.Share = .{};
            const half = try Tr.attnAndMoeInput(&g, NoProbe{}, &c, rt, .{}, li, w, inv, h, pm, try g.arange(0, @floatFromInt(s), 1, .int32), &cache, &shared);
            // kv16: the combine's x at the MoE input's dtype (bf16: the stream dtype of h1), as forwardLayerMajor casts
            // it; the stream stays bf16 through every layer (the reference's `y.type_as(x)`).
            try testing.expectEqual(Dtype.bfloat16, g.dtypeOf(half.moe_in));
            h = try Tr.prefillHcPost(&g, &c, .{}, try g.input(g.shapeOf(half.moe_in).slice(), g.dtypeOf(half.moe_in)), half);
            try testing.expectEqual(Dtype.bfloat16, g.dtypeOf(h));
            pm = half.ffn_pre;
        }
    }
    return g.compiles;
}

/// The single-span pass (`layer`) over two layers, one forward per width in order: the HcPost region's traces.
fn spanHcPostTraces(rt: *const Routes, widths: []const c_int) !usize {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    try Tr.prepareRegions(&g, &c, rt, false);
    const inv = try Tr.swaInvFreq(&g, &c);
    var ws: [2]LayerW(u32) = undefined;
    for (&ws, 0..) |*w, l| w.* = try traceLayerW(&g, &c, c.layers[l]);
    for (widths) |s| {
        var h = try g.input(&.{ 1, s, 4, 5120 }, .bfloat16);
        var pm = try g.input(&.{ 1, s, 4 }, .float32);
        for (&ws, 0..) |*w, l| {
            const li = c.layers[l];
            var cache = Tr.Cache.init(li, c.window, .{});
            defer cache.deinit(&g);
            var shared: Tr.Share = .{};
            const o = try Tr.layer(&g, NoProbe{}, &c, rt, .{}, li, w, inv, h, pm, try g.arange(0, @floatFromInt(s), 1, .int32), &cache, &shared, TraceRouted{});
            h = o.h;
            pm = o.pre_mix;
        }
    }
    return g.compiles;
}

test "dsv41 graph: HCPOST compiles both HC posts above 8 rows over the eager ops; 8 rows and below keep their chain; one added trace per chunk width" {
    const O = ops.Op;
    // Above 8 rows (the part's `rows > SMALL_ROWS`): two tapes, the attention side's and the single-span pass's MoE-side
    // combine (prepared by prepareRegions for the route alone, not layer-major), holding the eager chain's op multiset.
    for ([_]c_int{ 9, 32, 64 }) |rows| {
        const eager = try layerOps(&.{}, rows);
        defer testing.allocator.free(eager);
        const hp = try layerOps(&.{ .prefill_hc_post = true }, rows);
        defer testing.allocator.free(hp);
        var got = countOps(hp);
        try testing.expectEqual(@as(u32, 2), got[@backingInt(O.tape_begin)]);
        got[@backingInt(O.tape_begin)] = 0;
        got[@backingInt(O.tape_end)] = 0;
        try testing.expectEqual(countOps(eager), got);
    }
    // Decode and verify widths (8 rows and below: C15's tape where it is bound) keep their chain, op for op.
    for ([_]c_int{ 1, 8 }) |rows| {
        const e = try layerOps(&.{}, rows);
        defer testing.allocator.free(e);
        const on = try layerOps(&.{ .prefill_hc_post = true }, rows);
        defer testing.allocator.free(on);
        try testing.expectEqualSlices(O, e, on);
    }
    // The compile cost (kv16: one bf16 signature): K16's ffn combine traces the region once per chunk width (x and h1
    // bf16, post and comb f32); the attention side reuses that trace on every layer: no added trace (two chunks of 64
    // rows, a 40-row tail: two widths).
    const widths = [_]c_int{ 64, 64, 40 };
    try testing.expectEqual(@as(usize, 2), try k16HcPostTraces(&.{}, &widths));
    try testing.expectEqual(@as(usize, 2), try k16HcPostTraces(&.{ .prefill_hc_post = true }, &widths));
    // The single-span pass (short prompts, the warm-up's 9..32): both combines share the one bf16 signature per width;
    // none with the route off.
    const span = [_]c_int{ 16, 16, 12 };
    try testing.expectEqual(@as(usize, 0), try spanHcPostTraces(&.{}, &span));
    try testing.expectEqual(@as(usize, 2), try spanHcPostTraces(&.{ .prefill_hc_post = true }, &span));
}

test "dsv41 graph: PREFILL_HCPOST runs both HC combines above 8 rows as one launch (the f32 or bf16 residual text), checked against the region; 8 rows and below keep their routes" {
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    var ws: [v41.max_layers]LayerW(u32) = undefined;
    for (0..c.n_layers) |l| ws[l] = try traceLayerW(&g, &c, c.layers[l]);
    // The region is the reference: refused without it.
    try testing.expectError(error.HcPostNeedsRegion, Tr.Kernels.init(testing.allocator, &g, &reg, &c, &.{ .prefill_hcpost = true }, ws[0..c.n_layers]));
    const rt: Routes = .{ .prefill_hc_post = true, .prefill_hcpost = true };
    var k = try Tr.Kernels.init(testing.allocator, &g, &reg, &c, &rt, ws[0..c.n_layers]);
    defer k.deinit(&g);
    try Tr.prepareRegions(&g, &c, &rt, true);
    const lk = k.at(1);
    try testing.expect(lk.hcpost != null);
    const hc: c_int = @intCast(c.hc_mult);
    const dim: c_int = @intCast(c.hidden_size);
    for ([_]c_int{ 953, 183, 9 }) |S| for ([_]ops.Dtype{ .float32, .bfloat16 }) |rdt| {
        const x = try g.input(&.{ 1, S, dim }, .float32);
        const res = try g.input(&.{ 1, S, hc, dim }, rdt);
        const post = try g.input(&.{ 1, S, hc }, .float32);
        const comb = try g.input(&.{ 1, S, hc, hc }, .float32);
        const l0 = g.launched.items.len;
        const n0 = g.nodes.items.len;
        const half: Tr.Half = .{ .moe_in = x, .h1 = res, .post = post, .comb = comb, .ffn_pre = post };
        const a = try Tr.prefillHcPost(&g, &c, lk, x, half);
        const b = try Tr.hcPostRoute(&g, &c, &rt, lk, x, res, post, comb);
        for ([_]u32{ a, b }) |h| {
            try testing.expect(g.shapeOf(h).eql(ops.Shape.of(&.{ 1, S, hc, dim })));
            try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(h));
        }
        _ = l0;
        var kernels: usize = 0;
        for (g.nodes.items[n0..]) |nd| kernels += @intFromBool(nd.op == .kernel);
        try testing.expectEqual(@as(usize, 2), kernels);
        try testing.expect(noneOf(&g, n0, .tape_begin));
    };
    // kv16-opt: on the bf16 stream (x and the residual bf16) both combines read and write bf16 in their one launch:
    // no f32 x copy, no narrowing of an f32 h.
    for ([_]c_int{ 953, 183, 9 }) |S| {
        const x = try g.input(&.{ 1, S, dim }, .bfloat16);
        const res = try g.input(&.{ 1, S, hc, dim }, .bfloat16);
        const post = try g.input(&.{ 1, S, hc }, .float32);
        const comb = try g.input(&.{ 1, S, hc, hc }, .float32);
        const n0 = g.nodes.items.len;
        const half: Tr.Half = .{ .moe_in = x, .h1 = res, .post = post, .comb = comb, .ffn_pre = post };
        const a = try Tr.prefillHcPost(&g, &c, lk, x, half);
        const b = try Tr.hcPostRoute(&g, &c, &rt, lk, x, res, post, comb);
        for ([_]u32{ a, b }) |h| try testing.expectEqual(ops.Dtype.bfloat16, g.dtypeOf(h));
        var kernels: usize = 0;
        var casts: usize = 0;
        for (g.nodes.items[n0..]) |nd| {
            kernels += @intFromBool(nd.op == .kernel);
            casts += @intFromBool(nd.op == .astype);
        }
        try testing.expectEqual(@as(usize, 2), kernels);
        try testing.expectEqual(@as(usize, 0), casts);
    }
    // 8 rows and below: no launch (the region or the decode tape / chain, as before).
    {
        const S: c_int = 8;
        const res = try g.input(&.{ 1, S, hc, dim }, .float32);
        const x = try g.input(&.{ 1, S, dim }, .float32);
        const n0 = g.nodes.items.len;
        _ = try Tr.prefillHcPost(&g, &c, lk, x, .{ .moe_in = x, .h1 = res, .post = try g.input(&.{ 1, S, hc }, .float32), .comb = try g.input(&.{ 1, S, hc, hc }, .float32), .ffn_pre = x });
        try testing.expect(noneOf(&g, n0, .kernel));
    }
    // The construction check: the one-pass combine against the region on both residual dtypes.
    const scratch = try testing.allocator.alloc(f32, 64 * 64 * 512);
    defer testing.allocator.free(scratch);
    var checks: [16]Tr.RouteCheck = undefined;
    const n = try Tr.prefillRoutesCheck(&g, &c, &rt, &k, ws[0..c.n_layers], scratch, &checks);
    var found: usize = 0;
    for (checks[0..n]) |ck| found += @intFromBool(std.mem.endsWith(u8, ck.name, "residual") and std.mem.startsWith(u8, ck.name, "HC post"));
    try testing.expectEqual(@as(usize, 2), found);
}

test "dsv41 graph: PREFILL_SHAREDMID compiles the shared middle above 8 rows over the eager ops; 8 rows and below keep their chain" {
    const O = ops.Op;
    const Run = struct {
        fn ops_(rt: *const Routes, rows: c_int, regions: *usize) ![]O {
            var g = TraceOps.init(testing.allocator);
            defer g.deinit();
            const cc = try realConfig();
            try Tr.prepareRegions(&g, &cc, rt, true);
            const w = try traceLayerW(&g, &cc, cc.layers[1]);
            const x = try g.input(&.{ rows, @intCast(cc.hidden_size) }, .float32);
            const n0 = g.nodes.items.len;
            const y = try Tr.sharedExpertPrompt(&g, &cc, rt, &w, x);
            try testing.expectEqual(ops.Dtype.float32, g.dtypeOf(y));
            regions.* = g.compiles;
            return g.opsSince(testing.allocator, n0);
        }
    };
    for ([_]c_int{ 9, 183, 953 }) |rows| {
        var r_off: usize = 0;
        var r_on: usize = 0;
        const off = try Run.ops_(&.{}, rows, &r_off);
        defer testing.allocator.free(off);
        const on = try Run.ops_(&.{ .prefill_shared_mid = true }, rows, &r_on);
        defer testing.allocator.free(on);
        // The region traced once, holding the eager chain's op multiset (the markers aside).
        try testing.expectEqual(@as(usize, 0), r_off);
        try testing.expectEqual(@as(usize, 1), r_on);
        var got = countOps(on);
        try testing.expectEqual(@as(u32, 1), got[@backingInt(O.tape_begin)]);
        got[@backingInt(O.tape_begin)] = 0;
        got[@backingInt(O.tape_end)] = 0;
        try testing.expectEqual(countOps(off), got);
    }
    // Decode and verify widths keep the chain, op for op.
    for ([_]c_int{ 1, 8 }) |rows| {
        var r: usize = 0;
        const off = try Run.ops_(&.{}, rows, &r);
        defer testing.allocator.free(off);
        const on = try Run.ops_(&.{ .prefill_shared_mid = true }, rows, &r);
        defer testing.allocator.free(on);
        try testing.expectEqualSlices(O, off, on);
    }
}

test "dsv41 graph: head codecs and the cached wo_a keep the Python dtypes" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    const c = try realConfig();
    const x = try g.input(&.{ 1, 3, 5120 }, .float32);
    const hw = try g.input(&.{ 129280, 5120 }, .bfloat16);
    try expectShape(&g, try Tr.head(&g, &.{}, x, .{ .dense = hw }), &.{ 1, 3, 129280 }, .float32);
    const mark = g.nodes.items.len;
    try expectShape(&g, try Tr.head(&g, &.{ .head = .bf16 }, x, .{ .dense = hw }), &.{ 1, 3, 129280 }, .float32);
    const seq = try g.opsSince(testing.allocator, mark);
    defer testing.allocator.free(seq);
    // bf16 GEMV: cast the hidden down, matmul at bf16, cast the logits up.
    try testing.expectEqualSlices(ops.Op, &.{ .astype, .transpose, .matmul, .astype }, seq);
    const q = try Tr.quantizeHead(&g, hw);
    try expectShape(&g, q.w, &.{ 129280, 1280 }, .uint32);
    try expectShape(&g, q.s, &.{ 129280, 160 }, .uint8);
    try expectShape(&g, try Tr.head(&g, &.{ .head = .mxfp8 }, x, .{ .mxfp8 = q }), &.{ 1, 3, 129280 }, .float32);
    const w = try traceLayerW(&g, &c, c.layers[0]);
    try expectShape(&g, try Tr.woaDenseF32(&g, &c, w.wo_a), &.{ 8, 1024, 4096 }, .float32);
}

test "dsv41 graph: the W50 lean prefill score folds the sink instead of concatenating a column" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const li = c.layers[0];
    const w = try traceLayerW(&g, &c, li);
    const inv = try Tr.swaInvFreq(&g, &c);
    for ([_]bool{ false, true }) |lean| {
        var cache = Tr.Cache.init(li, c.window, .{});
        defer cache.deinit(&g);
        var shared: Tr.Share = .{};
        const mark = g.nodes.items.len;
        _ = try Tr.attention(&g, &p, &c, &.{ .lean_prefill_score = lean }, .{}, li, &w, inv, try g.input(&.{ 1, 5, 5120 }, .bfloat16), try g.arange(0, 5, 1, .int32), &cache, &shared);
        try expectStage(&g, &p, "attn.o", &.{ 1, 5, 64, 512 }, .float32);
        const seq = try g.opsSince(testing.allocator, mark);
        defer testing.allocator.free(seq);
        try testing.expectEqual(!lean, std.mem.count(ops.Op, seq, &.{.softmax}) == 1);
        try testing.expectEqual(lean, std.mem.count(ops.Op, seq, &.{.exp}) == 2);
    }
}

/// Bytes the MLX backend holds for one traced node range: MlxOps tracks every
/// op output until the per-layer `reset`, so an evaluated layer keeps all of
/// them at once. Views (reshape / transpose / expand / broadcast / slice) and
/// leaves share or own no new buffer and are left out.
pub fn heldBytes(g: *const TraceOps, from: usize, to: usize) struct { sum: u64, max: u64, max_op: ops.Op } {
    var sum: u64 = 0;
    var mx: u64 = 0;
    var mop: ops.Op = .input;
    for (g.nodes.items[from..to]) |n| {
        switch (n.op) {
            .input, .host, .scalar, .reshape, .transpose, .transpose_axes, .broadcast_to, .expand_dims, .slice, .tape_begin, .tape_end => continue,
            else => {},
        }
        const b: u64 = @as(u64, @intCast(n.shape.numel())) * ops.dtypeSize(n.dtype);
        sum += b;
        if (b > mx) {
            mx = b;
            mop = n.op;
        }
    }
    return .{ .sum = sum, .max = mx, .max_op = mop };
}

/// The buffers a traced node range allocates: every node but the views (host arrays and scalars are buffers of their
/// own here, unlike in `heldBytes`); `inputs` also counts the input leaves (each resident or state tensor).
pub fn heldArrays(g: *const TraceOps, from: usize, to: usize, inputs: bool) u64 {
    var n: u64 = 0;
    for (g.nodes.items[from..to]) |x| switch (x.op) {
        .reshape, .transpose, .transpose_axes, .broadcast_to, .expand_dims, .slice, .tape_begin, .tape_end => {},
        .input => n += @intFromBool(inputs),
        else => n += 1,
    };
    return n;
}

/// Handles with MLX's lifetimes (the trace backend frees nothing): `keep` makes
/// an untracked handle, `resetTo` kills every handle tracked since the mark.
const Handles = struct {
    pub const T = u32;
    gpa: std.mem.Allocator,
    alive: std.ArrayList(bool) = .empty,
    tracked: std.ArrayList(u32) = .empty,

    fn deinit(h: *Handles) void {
        h.alive.deinit(h.gpa);
        h.tracked.deinit(h.gpa);
    }

    fn make(h: *Handles) !u32 {
        try h.alive.append(h.gpa, true);
        const x: u32 = @intCast(h.alive.items.len - 1);
        try h.tracked.append(h.gpa, x);
        return x;
    }

    pub fn keep(h: *Handles, x: u32) u32 {
        std.debug.assert(h.alive.items[x]);
        h.alive.append(h.gpa, true) catch @panic("oom");
        return @intCast(h.alive.items.len - 1);
    }

    pub fn release(h: *Handles, x: u32) void {
        std.debug.assert(h.alive.items[x]);
        h.alive.items[x] = false;
    }

    pub fn mark(h: *const Handles) ops.Mark {
        return .{ .n = h.tracked.items.len };
    }

    pub fn resetTo(h: *Handles, m: ops.Mark) void {
        for (h.tracked.items[m.n..]) |x| h.alive.items[x] = false;
        h.tracked.shrinkRetainingCapacity(m.n);
    }

    fn nAlive(h: *const Handles) usize {
        return std.mem.count(bool, h.alive.items, &.{true});
    }
};

test "dsv41 graph: a layer wave's carry keeps what later layers read alive past its reset and leaks nothing" {
    var hs: Handles = .{ .gpa = testing.allocator };
    defer hs.deinit();
    const Tw = Trunk(Handles);
    var h = try hs.make();
    var pm = try hs.make();
    var shared: Tw.Share = .{};
    var carry: Tw.Carry = .{};
    const before = hs.nAlive();
    // Layer 0 publishes the compressed lanes and a window mask; layer 1 changes only the mask;
    // layer 2 publishes a selection; layer 3 drops the selection and changes nothing else.
    for (0..4) |l| {
        const wave = hs.mark();
        h = try hs.make();
        pm = try hs.make();
        switch (l) {
            0 => {
                shared.compress_kv = try hs.make();
                shared.index_k = try hs.make();
                shared.win_mask = try hs.make();
            },
            1 => shared.win_mask = try hs.make(),
            2 => shared.topk_mask = try hs.make(),
            else => shared.topk_mask = null,
        }
        _ = try hs.make(); // an intermediate nothing carries
        carry.persist(&hs, &h, &pm, &shared);
        hs.resetTo(wave);
        // Everything a later layer reads is alive; so is nothing else the waves made.
        for ([_]?u32{ h, pm, shared.compress_kv, shared.index_k, shared.win_mask, shared.topk_mask }) |s| if (s) |x| try testing.expect(hs.alive.items[x]);
        const carried: usize = 2 + 3 + @as(usize, @intFromBool(shared.topk_mask != null));
        try testing.expectEqual(before + carried, hs.nAlive());
    }
    carry.release(&hs);
    try testing.expectEqual(before, hs.nAlive());
}

// The window-2 stage-3 bound (host only): the chained 40-layer stock trunk at a
// 2,048-token pass, then three 1-token passes, as the parity runner drives it.
test "dsv41 graph: the all-layer chain's per-layer held bytes at a 2,048-token pass" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    var p: TraceProbe = .{ .a = testing.allocator };
    defer p.deinit();
    const c = try realConfig();
    const nl = c.n_layers;
    const ws = try testing.allocator.alloc(LayerW(u32), nl);
    defer testing.allocator.free(ws);
    const caches = try testing.allocator.alloc(Tr.Cache, nl);
    defer testing.allocator.free(caches);
    for (ws, caches, 0..) |*w, *cc, l| {
        w.* = try traceLayerW(&g, &c, c.layers[l]);
        cc.* = Tr.Cache.init(c.layers[l], c.window, .{});
    }
    defer for (caches) |*cc| cc.deinit(&g);
    const inv_s = try Tr.swaInvFreq(&g, &c);
    const inv_y = try Tr.yarnInvFreq(&g, &c);
    const stand: StandIn(TraceOps) = .{ .scale = try g.input(&.{ci(c.n_routed_experts)}, .float32) };
    const passes = [_]c_int{ 2048, 1, 1, 1 };
    var tok: c_int = 0;
    var worst: u64 = 0;
    var worst_l: usize = 0;
    for (passes, 0..) |s, pi| {
        var shared: Tr.Share = .{};
        const pos = try g.arange(@floatFromInt(tok), @floatFromInt(tok + s), 1, .int32);
        const e = try Tr.expandEmbedding(&g, &c, try g.input(&.{ 1, s, ci(c.hidden_size) }, .bfloat16));
        var h = e.h;
        var pm = e.pre_mix;
        for (0..nl) |l| {
            const li = c.layers[l];
            const from = g.nodes.items.len;
            p.names.clearRetainingCapacity();
            p.nodes.clearRetainingCapacity();
            const out = try Tr.layer(&g, &p, &c, &stock, .{}, li, &ws[l], if (li.ratio > 0) inv_y else inv_s, h, pm, pos, &caches[l], &shared, stand);
            h = out.h;
            pm = out.pre_mix;
            const hb = heldBytes(&g, from, g.nodes.items.len);
            if (pi == 0) std.debug.print("dsv41 bound: L{d} ratio {d} held {d:.2} GiB, largest {d:.2} GiB ({s})\n", .{ l, li.ratio, @as(f64, @floatFromInt(hb.sum)) / (1 << 30), @as(f64, @floatFromInt(hb.max)) / (1 << 30), @tagName(hb.max_op) });
            if (hb.sum > worst) {
                worst = hb.sum;
                worst_l = l;
            }
        }
        for (caches) |*cc| cc.advance(@intCast(s));
        tok += s;
    }
    std.debug.print("dsv41 bound: worst layer L{d} holds {d:.2} GiB at eval\n", .{ worst_l, @as(f64, @floatFromInt(worst)) / (1 << 30) });
    try testing.expect(worst > 0);
}
