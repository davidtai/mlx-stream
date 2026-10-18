//! Beam's decode glue (`decode_glue: fused`, the default; `ops` is the stock control): a decode-lane forward's router
//! (one row, or up to the decode lane's rows in the decode phase) and the lookahead's predictor in two texts
//! (src/kernels/decode_glue/, pinned by manifest_beam.json), each the stock op chain's arithmetic:
//!   - `glue_router_gemv`: the f32 product of the FFN sublayer's normed row with the layer's router weight (EPI 0: the
//!     logits), or with the next routed layer's, then its sigmoid and correction (EPI 1: the predictor's scores); each
//!     sum MLX's gemv at M = 1 (gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4, the stock route's `astype` and `matmul`), in
//!     `rows_per_group` rows of the weight a threadgroup;
//!   - `glue_router_select`: the tail (sigmoid, correction, the group top-k, the top-k, the normalized weights);
//!   - `glue_qk_norm`: a decode row's query and key heads normed (q_norm, k_norm), then RoPE on a sliding layer, or the
//!     query's scale on a full layer (graph.zig `qkStock`: two norms and two ropes, or two norms and a scale);
//!   - `glue_add_norm`: a decode row's sublayer boundary (graph.zig `boundaryStock`: the sublayer's norm, the
//!     depth-scaled add, the next sublayer's norm);
//!   - `glue_moe_combine`: a decode-lane call's routed combine with the scheduler's join folded in and the routed scale
//!     (the join's take, route.zig `combineStock`'s cast, product, sum and cast, graph.zig ScaleBy: six launches), the
//!     routed rows read in place through the join's order.
//! Stock: the router is the input cast, the GEMV and 17 ops (sigmoid, add, the groups' topk and sum, negation,
//! argpartition, the group ids, equal, sum, greater, where, negation, argpartition, take_along_axis, sum, add, divide);
//! the predictor is the cast, the GEMV, sigmoid and add. Exact: the ids, the weights, the predictor's scores, the
//! normed heads and the combined rows are the stock route's bit for bit (`install`'s self-check over the model's own
//! tables; the device tests `beam glue GPU`). At M > 1 the router's product stays MLX's matmul (MLX routes [M > 1, D] x [D, E] f32 to its GEMM: reproduced
//! only by running it) and the select text takes its logits; the predictor keeps the stock route there. Installed once
//! at construction (the geometry refused by name, the kernel objects, the launch configurations per decode-lane width,
//! the self-check); the forward then dispatches with no per-call config work.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const route = @import("../mimo_v2/route.zig");
const beam = @import("config.zig");

pub const G = ops.MlxOps;
pub const T = G.T;

/// The decode lane's glue route (`settings.DecodeGlue`).
pub const Route = @import("settings.zig").DecodeGlue;

/// The decode lane's widest call (experts.zig `decode_lane_rows`): the select text's configs.
pub const max_rows: u32 = @import("experts.zig").decode_lane_rows;
/// The router weight's rows a threadgroup of the GEMV owns (MLX's gemv owns 4; the sums are the same at any count).
pub const rows_per_group: u32 = 2;
/// The GEMV's threadgroup: MLX's bm1_bn8 block (8 simdgroups of 32 lanes, 4 columns a lane: 1,024 columns a block).
pub const gemv_threads: u32 = 256;
/// The q / k norm text's heads a threadgroup (one simdgroup a head).
pub const heads_per_group: u32 = 8;
/// The combine text's columns a threadgroup (one thread a column of a row; a narrower model takes one threadgroup).
pub const combine_threads: u32 = 256;

// ── the texts and their pins (src/kernels/decode_glue/manifest_beam.json) ──

const dir = "../../kernels/decode_glue/";

pub const texts = struct {
    pub const manifest = @embedFile(dir ++ "manifest_beam.json");
    pub const header = @embedFile(dir ++ "header_decode_glue.metal");
    pub const gemv = @embedFile(dir ++ "glue_router_gemv.metal");
    pub const select = @embedFile(dir ++ "glue_router_select.metal");
    pub const qk = @embedFile(dir ++ "glue_qk_norm.metal");
    pub const add_norm = @embedFile(dir ++ "glue_add_norm.metal");
    pub const header_combine = @embedFile(dir ++ "header_decode_glue_combine.metal");
    pub const combine = @embedFile(dir ++ "glue_moe_combine.metal");
    pub const header_q8 = @embedFile(dir ++ "header_decode_glue_q8.metal");
    pub const q8 = @embedFile(dir ++ "glue_q8_gemv.metal");
};
/// sha256 of manifest_beam.json (scripts/decode_glue_manifest.py prints it), which records every text's sha256 (checked
/// against the pins by a CPU test).
pub const manifest_sha256 = "439d0923180468f6822b9cac3ad16a14562a59c71d572faeadbf685b9b9992e2";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_decode_glue.metal", .text = texts.header, .sha256 = "6bec0858b4e0754c74a4eee95560bd78f0942ecc092cc50d50bcc7840977e484" },
    .{ .file = "glue_router_gemv.metal", .text = texts.gemv, .sha256 = "1c82d26867b1c277adb2ca003200f6c05c3ecc396126b74ac571854b202bdcb7" },
    .{ .file = "glue_router_select.metal", .text = texts.select, .sha256 = "5a7369ad5c53e6eb5b08013219ad7731cfa797863599d9be8f2c2e4715bf47f6" },
    .{ .file = "glue_qk_norm.metal", .text = texts.qk, .sha256 = "998421219baf1c0f93371655ae43517bb608061cd35eaacda59f1255932afd3a" },
    .{ .file = "glue_add_norm.metal", .text = texts.add_norm, .sha256 = "7802402dbccaa535d22b60265c0476f20cd55788572c45c7b5429d7bd836042d" },
    .{ .file = "header_decode_glue_combine.metal", .text = texts.header_combine, .sha256 = "c4761ee970f8a39a190f1d348e247e66c8ba2b112de9bfe4a267c6dcde992a67" },
    .{ .file = "glue_moe_combine.metal", .text = texts.combine, .sha256 = "836812cc11442c1494947afb502862303caf5a244c1dcb7fbd67a61e0914fb5d" },
    .{ .file = "header_decode_glue_q8.metal", .text = texts.header_q8, .sha256 = "e4273595fa3f04409b6fd06d3ae0628f322181000fce9f75ef244dbaebb4d7d8" },
    .{ .file = "glue_q8_gemv.metal", .text = texts.q8, .sha256 = "b1f432fda5cec73ab5a8d0e60606e51a5ae9f1aa9bb92cd28ea630ec1855bbd6" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.BeamGlueManifestNotPinned, error.BeamGlueTextNotPinned);
}

/// A model geometry the manifest registers (scripts/decode_glue_manifest.py `CASES`): the router's and the heads'.
pub const Geometry = struct { hidden: u32, experts: u32, groups: u32, kept_groups: u32, top_k: u32, heads: u32, kv_heads: u32, head_dim: u32 };

/// Beam 1.0's and the converter's fixture's (scripts/beam_fixture.py: hidden 256, 8 experts in 2 groups, 1 kept, top-2,
/// 4 / 2 heads of 64).
pub const registered_geometries = [_]Geometry{
    .{ .hidden = 5120, .experts = 256, .groups = 8, .kept_groups = 4, .top_k = 8, .heads = 48, .kv_heads = 8, .head_dim = 128 },
    .{ .hidden = 256, .experts = 8, .groups = 2, .kept_groups = 1, .top_k = 2, .heads = 4, .kv_heads = 2, .head_dim = 64 },
};

pub fn geometryOf(c: *const beam.Config) Geometry {
    return .{ .hidden = c.hidden_size, .experts = c.n_routed_experts, .groups = c.n_group, .kept_groups = c.topk_group, .top_k = c.n_experts_per_tok, .heads = c.n_heads, .kv_heads = c.n_kv_heads, .head_dim = c.head_dim };
}

/// The texts serve the model: its routed geometry is a registered one (each where MLX's gemv takes its bm1_bn8 route:
/// hidden >= 16 x experts, above 64) and its weights are normalized (norm_topk_prob).
pub fn registered(c: *const beam.Config) bool {
    if (!c.norm_topk_prob) return false;
    const want = geometryOf(c);
    for (registered_geometries) |r| if (std.meta.eql(r, want)) return true;
    return false;
}

// ── the installed kernels ──

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    gemv: Kernel = .{ .ctx = null },
    select: Kernel = .{ .ctx = null },
    qk: Kernel = .{ .ctx = null },
    /// The q / k norm text on a sliding layer (MODE 0) and a full one (MODE 1).
    qk_cfg: [2]Config = @splat(.{ .ctx = null }),
    add_norm: Kernel = .{ .ctx = null },
    add_norm_cfg: Config = .{ .ctx = null },
    combine: Kernel = .{ .ctx = null },
    /// Index M - 1: the combine text at a decode-lane call of M rows.
    combine_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: the routed scale (graph.zig ScaleBy's factor, `routed_scaling_factor`), then zeros.
    combine_consts: T = .{ .ctx = null },
    /// The GEMV at M = 1: the router's logits (EPI 0) and the predictor's scores (EPI 1).
    route_cfg: Config = .{ .ctx = null },
    pred_cfg: Config = .{ .ctx = null },
    /// Index M - 1: the select text at a decode-lane call of M rows.
    select_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: the weights' scale (1: Beam multiplies the experts' weighted sum by routed_scaling, graph.zig), then
    /// zeros (eight words: MLX passes a smaller input in the constant address space).
    consts: T = .{ .ctx = null },
    hidden: u32 = 0,
    experts: u32 = 0,
    top_k: u32 = 0,
    /// RoPE's log2(base) as MLX's host computes it (rope.cpp: std::log2 of the f32 base) and the norms' eps.
    rope_log2: f32 = 0,
    eps: f32 = 0,

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.gemv, k.select, k.qk, k.add_norm, k.combine }) |x| if (x.ctx != null) {
            mtl.mlx_fast_metal_kernel_free(x);
        };
        for ([_]Config{ k.route_cfg, k.pred_cfg, k.add_norm_cfg } ++ k.select_cfg ++ k.qk_cfg ++ k.combine_cfg) |cfg| if (cfg.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        for ([_]T{ k.consts, k.combine_consts }) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        k.* = .{};
    }

    /// The lookahead's predictor over these kernels (bound into the expert math: experts.zig `Source.bindPredictor`).
    pub fn predictor(k: *const Kernels) mimo_math.Predictor {
        return .{ .scores = predictorScores, .ctx = k };
    }
};

/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");

/// The launch geometry of each text (CPU-tested; `install` builds the configs from it).
pub const Launch = struct {
    grid: [3]u32,
    group: [3]u32,

    /// The GEMV: experts / rows_per_group threadgroups of 256 threads.
    pub fn gemv(experts: u32) Launch {
        return .{ .grid = .{ gemv_threads * (experts / rows_per_group), 1, 1 }, .group = .{ gemv_threads, 1, 1 } };
    }

    /// The select text: one experts-thread threadgroup per row.
    pub fn select(experts: u32, rows: u32) Launch {
        return .{ .grid = .{ experts, rows, 1 }, .group = .{ experts, 1, 1 } };
    }

    /// The q / k norm text: a simdgroup a head, `heads_per_group` heads a threadgroup.
    pub fn qk(heads: u32) Launch {
        const groups = (heads + heads_per_group - 1) / heads_per_group;
        return .{ .grid = .{ 32 * heads_per_group * groups, 1, 1 }, .group = .{ 32 * heads_per_group, 1, 1 } };
    }

    /// The boundary text: one threadgroup of MLX's rms threadgroup at the hidden size (`normThreads`).
    pub fn addNorm(hidden: u32) Launch {
        const t = normThreads(hidden);
        return .{ .grid = .{ t, 1, 1 }, .group = .{ t, 1, 1 } };
    }

    /// The combine text: a thread per column of each of the call's `rows` rows, `combine_threads` columns a threadgroup.
    pub fn combine(hidden: u32, rows: u32) Launch {
        return .{ .grid = .{ hidden, rows, 1 }, .group = .{ @min(hidden, combine_threads), 1, 1 } };
    }
};

/// MLX's rms threadgroup at width `d` (normalization.cpp RMSNorm::eval_gpu): rms_looped's 1,024 threads past
/// RMS_LOOPED_LIMIT (4,096), else rms_single_row's 32 x ceil(d / 128).
pub fn normThreads(d: u32) u32 {
    return if (d > 4096) 1024 else 32 * ((d + 127) / 128);
}

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !Kernel {
    // The inputs are the forward's fresh row-major rows and the residents (a view would be copied: ensure_row_contiguous).
    return kset.newKernel(name, ins, outs, source.ptr, header.ptr, true) orelse error.BeamGlueKernel;
}

fn templateOf(comptime names: []const [:0]const u8, values: [names.len]i32) [names.len]kset.TemplateArg {
    var out: [names.len]kset.TemplateArg = undefined;
    inline for (names, 0..) |n, i| out[i] = .{ .name = n, .value = .{ .int = values[i] } };
    return out;
}

fn gemvConfig(geo: Geometry, epi: i32) !Config {
    const l = Launch.gemv(geo.experts);
    const shape = [_]c_int{ 1, @intCast(geo.experts) };
    const t = templateOf(&.{ "D", "E", "M", "R", "EPI" }, .{ @intCast(geo.hidden), @intCast(geo.experts), 1, @intCast(rows_per_group), epi });
    return kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{&shape}, .out_dtypes = &.{.float32}, .template = &t });
}

fn qkConfig(geo: Geometry, mode: i32) !Config {
    const l = Launch.qk(geo.heads + geo.kv_heads);
    const qs = [_]c_int{ 1, @intCast(geo.heads), 1, @intCast(geo.head_dim) };
    const ks = [_]c_int{ 1, @intCast(geo.kv_heads), 1, @intCast(geo.head_dim) };
    const t = templateOf(&.{ "HQ", "HK", "HD", "HPG", "MODE" }, .{ @intCast(geo.heads), @intCast(geo.kv_heads), @intCast(geo.head_dim), @intCast(heads_per_group), mode });
    return kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{ &qs, &ks }, .out_dtypes = &.{ .bfloat16, .bfloat16 }, .template = &t });
}

fn addNormConfig(geo: Geometry) !Config {
    const l = Launch.addNorm(geo.hidden);
    const shape = [_]c_int{ 1, @intCast(geo.hidden) };
    const t = templateOf(&.{ "D", "TPG" }, .{ @intCast(geo.hidden), @intCast(normThreads(geo.hidden)) });
    return kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{ &shape, &shape }, .out_dtypes = &.{ .bfloat16, .bfloat16 }, .template = &t });
}

fn combineConfig(geo: Geometry, rows: u32) !Config {
    const l = Launch.combine(geo.hidden, rows);
    const shape = [_]c_int{ @intCast(rows), @intCast(geo.hidden) };
    const t = templateOf(&.{ "D", "K", "M" }, .{ @intCast(geo.hidden), @intCast(geo.top_k), @intCast(rows) });
    return kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{&shape}, .out_dtypes = &.{.bfloat16}, .template = &t });
}

fn selectConfig(geo: Geometry, rows: u32) !Config {
    const l = Launch.select(geo.experts, rows);
    const shape = [_]c_int{ @intCast(rows), @intCast(geo.top_k) };
    const t = templateOf(&.{ "E", "G", "TG", "K" }, .{ @intCast(geo.experts), @intCast(geo.groups), @intCast(geo.kept_groups), @intCast(geo.top_k) });
    return kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{ &shape, &shape }, .out_dtypes = &.{ .uint32, .float32 }, .template = &t });
}

/// The model's tables the self-check runs over (construction only): the first routed layer's router (its F32 weight
/// and correction bias), the next routed layer's (the predictor's), the first layer's q_norm and k_norm, and its
/// post-attention and pre-feedforward norms.
pub const CheckTables = struct { w: T, bias: T, next_w: T, next_bias: T, q_norm: T, k_norm: T, post_attn_norm: T, pre_ff_norm: T };

/// The route's construction on the module's GPU stream `s` (refused by name: `diag`): the geometry, the two texts'
/// kernel objects, the configs, then the self-check (`selfCheck`: Metal compiles each text at its first dispatch).
pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const beam.Config, tables: CheckTables, diag: *sdk.Diag) !Kernels {
    checkPins() catch |e| {
        diag.set("beam decode_glue=fused: a text is not the pinned one ({s})", .{@errorName(e)});
        return e;
    };
    if (!registered(c)) {
        diag.set("beam decode_glue=fused: the router is not one the texts register (hidden {d}, {d} experts in {d} groups, {d} kept, top-{d}, norm_topk_prob {})", .{ c.hidden_size, c.n_routed_experts, c.n_group, c.topk_group, c.n_experts_per_tok, c.norm_topk_prob });
        return error.BeamGlueGeometry;
    }
    if (!mlx.streamIsGpu(s)) {
        diag.set("beam decode_glue=fused: the module's stream is not a GPU stream", .{});
        return error.BeamGlueNotGpu;
    }
    const geo = geometryOf(c);
    var k: Kernels = .{ .hidden = geo.hidden, .experts = geo.experts, .top_k = geo.top_k, .rope_log2 = log2f(c.rope_theta), .eps = c.rms_norm_eps };
    errdefer k.deinit();
    k.gemv = try newKernel("beam_glue_router_gemv", &.{ "x", "w", "bias" }, &.{"out"}, texts.gemv, texts.header);
    k.select = try newKernel("beam_glue_router_select", &.{ "logits", "bias", "consts" }, &.{ "ids", "weights" }, texts.select, texts.header);
    k.qk = try newKernel("beam_glue_qk_norm", &.{ "qk", "qw", "kw", "consts" }, &.{ "qo", "ko" }, texts.qk, texts.header);
    k.add_norm = try newKernel("beam_glue_add_norm", &.{ "y", "r", "w1", "w2", "consts" }, &.{ "res", "out" }, texts.add_norm, texts.header);
    k.combine = try newKernel("beam_glue_moe_combine", &.{ "rows", "order", "w", "consts" }, &.{"y"}, texts.combine, texts.header_combine);
    k.add_norm_cfg = try addNormConfig(geo);
    k.route_cfg = try gemvConfig(geo, 0);
    k.pred_cfg = try gemvConfig(geo, 1);
    for (&k.select_cfg, 1..) |*cfg, m| cfg.* = try selectConfig(geo, @intCast(m));
    for (&k.combine_cfg, 1..) |*cfg, m| cfg.* = try combineConfig(geo, @intCast(m));
    for (&k.qk_cfg, 0..) |*cfg, mode| cfg.* = try qkConfig(geo, @intCast(mode));
    var cv: [8]f32 = @splat(0);
    cv[0] = 1;
    k.consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.consts.ctx == null) return error.MlxError;
    cv[0] = c.routed_scaling;
    k.combine_consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.combine_consts.ctx == null) return error.MlxError;
    selfCheck(a, s, c, &k, tables, diag) catch |err| {
        // A differing array names itself; any other failure is MLX's (a Metal compile error names the line).
        if (err != error.BeamGlueCheck) {
            var eb: [400]u8 = undefined;
            diag.set("beam decode_glue=fused: the self-check failed ({s}): {s}", .{ @errorName(err), mlx.takeError(&eb) orelse "" });
        }
        return err;
    };
    return k;
}

fn apply(g: *G, kernel: Kernel, cfg: Config, inputs: []const T, outs: []T) !void {
    var raw: [2]T = undefined;
    try kset.applyConfig(kernel, cfg, inputs, raw[0..outs.len], g.s);
    for (outs, raw[0..outs.len]) |*o, x| o.* = try g.adopt(x);
}

pub const Routing = struct { ids: T, weights: T };

/// The routing of `rows` decode-lane rows: x bf16 [rows, hidden] (the FFN sublayer's normed input), the router's F32
/// weight [experts, hidden] and F32 correction bias [experts] -> ids u32 [rows, top_k] (the scheduler's index type,
/// in the stock argpartition's order), weights f32 [rows, top_k]. Two dispatches at M = 1; at M > 1 MLX's matmul and
/// the select text.
pub fn routeRows(g: *G, k: *const Kernels, w: T, bias: T, x: T, rows: u32) !Routing {
    std.debug.assert(rows >= 1 and rows <= max_rows);
    var logits: [1]T = undefined;
    if (rows == 1) {
        try apply(g, k.gemv, k.route_cfg, &.{ x, w, bias }, &logits);
    } else logits[0] = try g.matmul(try g.astype(x, .float32), try g.transpose(w));
    var out: [2]T = undefined;
    try apply(g, k.select, k.select_cfg[rows - 1], &.{ logits[0], bias, k.consts }, &out);
    return .{ .ids = out[0], .weights = out[1] };
}

/// The lookahead's scores on the next routed layer's router (`Kernels.predictor`; mimo_math.Predictor's `scores`): x
/// bf16 [rows, hidden], the router's F32 weight [experts, hidden] and correction [experts] -> f32 [rows, experts]. One
/// dispatch at M = 1, the stock route's (`route.scoresStored`) at M > 1.
fn predictorScores(p: *const mimo_math.Predictor, g: *G, x: T, router: T, correction: T) anyerror!T {
    if (g.shapeOf(x).dim(0) != 1) return route.scoresStored(p, g, x, router, correction);
    const k: *const Kernels = @ptrCast(@alignCast(p.ctx.?));
    var out: [1]T = undefined;
    try apply(g, k.gemv, k.pred_cfg, &.{ x, router, correction }, &out);
    return out[0];
}

/// MLX's host computes RoPE's log2(base) with the C library's (rope.cpp `std::log2(base_)` on the f32 base).
extern "c" fn log2f(x: f32) f32;

pub const Heads = struct { q: T, k: T };

/// A decode row's normed query and key heads: `qk` bf16 [1, n] the fused projections' row as the GEMV wrote it (its
/// first heads * head_dim columns the query, the next kv_heads * head_dim the key: graph.zig `fuseProjections`; the text
/// reads them in place, as a row view of the GEMV's buffer would be copied first), the layer's q_norm and k_norm
/// [head_dim] bf16, at position `pos` -> q [1, heads, 1, head_dim], k [1, kv_heads, 1, head_dim] bf16: roped on a
/// sliding layer, the query times `scale` (the full layer's query scale, graph.zig `ssmaxAt`) on a full layer. One
/// dispatch (graph.zig `qkStock`'s four or three, and the two copies of the row views it would read).
pub fn qkNorm(g: *G, k: *const Kernels, qk: T, q_norm: T, k_norm: T, pos: u32, full: bool, scale: f32) !Heads {
    var cv: [8]f32 = @splat(0);
    cv[0] = k.rope_log2;
    cv[1] = scale;
    cv[2] = k.eps;
    cv[3] = @floatFromInt(pos);
    const consts = try g.hostArray(std.mem.sliceAsBytes(&cv), &.{8}, .float32);
    var out: [2]T = undefined;
    try apply(g, k.qk, k.qk_cfg[@intFromBool(full)], &.{ qk, q_norm, k_norm, consts }, &out);
    return .{ .q = out[0], .k = out[1] };
}

pub const Boundary = struct { res: T, out: T };

/// The depth-scaled add's two scales of sublayer index `i` (graph.zig `depthAdd`: 1 / sqrt(i), sqrt((i - 1) / i), each
/// rounded from f64 to f32 as `G.scalar` rounds it).
pub fn depthScales(i: u32) [2]f32 {
    const fi: f64 = @floatFromInt(i);
    return .{ @floatCast(1.0 / @sqrt(fi)), @floatCast(@sqrt((fi - 1.0) / fi)) };
}

/// A decode row's sublayer boundary: the sublayer's output `y` [1, hidden] normed by `w1`, the depth-scaled add of
/// sublayer index `i` with the residual `r` -> `res` [1, hidden] (the next residual), then `res` normed by `w2` -> `out`
/// [1, hidden] (the next sublayer's input). One dispatch (graph.zig `boundaryStock`'s three).
pub fn addNorm(g: *G, k: *const Kernels, y: T, r: T, w1: T, w2: T, i: u32) !Boundary {
    const sc = depthScales(i);
    var cv: [8]f32 = @splat(0);
    cv[0] = sc[0];
    cv[1] = sc[1];
    cv[2] = k.eps;
    const consts = try g.hostArray(std.mem.sliceAsBytes(&cv), &.{8}, .float32);
    var out: [2]T = undefined;
    try apply(g, k.add_norm, k.add_norm_cfg, &.{ y, r, w1, w2, consts }, &out);
    return .{ .res = out[0], .out = out[1] };
}

/// A decode-lane call's routed output: `rows` bf16 [R, hidden] (the call's concatenated outputs: the scheduler's
/// `routedHoistIndexed`), `order` u32 [rows * top_k] (each routed position's row in them, the join's order), `weights` f32
/// [rows, top_k] (the router's) -> bf16 [rows, hidden]: the score-weighted sum of the routed rows in router order, times
/// the routed scale (graph.zig ScaleBy), one dispatch (the join's take, route.zig `combineStock`'s four and the scale's).
pub fn combineRows(g: *G, k: *const Kernels, rows: T, order: T, weights: T, n: u32) !T {
    std.debug.assert(n >= 1 and n <= max_rows);
    var out: [1]T = undefined;
    try apply(g, k.combine, k.combine_cfg[n - 1], &.{ rows, order, weights, k.combine_consts }, &out);
    return out[0];
}

/// `combineRows`' stock route (the device tests' control and the self-check's): the join's take of `rows` by `order`,
/// route.zig `combineStock` (f32 cast, the product with the weights, the axis-1 sum, bf16), then graph.zig ScaleBy's
/// chain (bf16(f32(.) * scale)).
pub fn combineStockRows(g: *G, c: *const beam.Config, rows: T, order: T, weights: T, n: u32) !T {
    const graph = @import("graph.zig");
    const joined = try g.reshape(try g.take(rows, order, 0), &.{ @intCast(n), @intCast(c.n_experts_per_tok), @intCast(c.hidden_size) });
    return graph.scaleBy(g, try route.combineStock(g, joined, weights, c.n_experts_per_tok), try g.scalar(c.routed_scaling, .float32));
}

// ── the self-check (construction, once) ──

/// A fixed bf16 row [1, n] for the self-check: values of a few magnitudes, both signs, and a few exact zeros.
pub fn checkRow(g: *G, a: std.mem.Allocator, n: u32) !T {
    const bits = try a.alloc(u16, n);
    defer a.free(bits);
    var prng = std.Random.DefaultPrng.init(0xbea3_91ce);
    const r = prng.random();
    for (bits, 0..) |*b, i| {
        const v: f32 = if (i % 61 == 0) 0 else r.floatNorm(f32) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -4, 2)));
        b.* = ops.bf16Bits(v);
    }
    return g.hostArray(std.mem.sliceAsBytes(bits), &.{ 1, @intCast(n) }, .bfloat16);
}

/// `want` and `got` evaluated and equal byte for byte (the same dtype, shape and values in row-major order), else
/// error.BeamGlueCheck, the first differing element named in `diag` (or printed without one). Either may be a view (the
/// stock router's ids at M > 1 are a slice of its sort's rows): each is compared through its row-major copy.
pub fn sameBytes(g: *G, name: []const u8, want: T, got: T, diag: ?*sdk.Diag) !void {
    try g.evalAll(&.{ want, got });
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got) or n != mlx.mlx_array_size(got) or !std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got)))
        return mismatch(diag, "{s}: the dtype or the shape differs", .{name});
    const w = try rowMajor(g, want);
    const o = try rowMajor(g, got);
    try g.evalAll(&.{ w, o });
    const size = n * mlx.mlx_array_itemsize(want);
    const wb = (mlx.mlx_array_data_uint8(w) orelse return error.MlxNoData)[0..size];
    const ob = (mlx.mlx_array_data_uint8(o) orelse return error.MlxNoData)[0..size];
    if (std.mem.indexOfDiff(u8, wb, ob)) |i| return mismatch(diag, "{s}: element {d} of {d} differs", .{ name, i / mlx.mlx_array_itemsize(want), n });
}

/// `x` as a row-major array (MLX's `contiguous`: a copy only when `x` is a view of another layout).
fn rowMajor(g: *G, x: T) !T {
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_contiguous(&o, x, false, g.s)) catch |e| {
        _ = mlx.mlx_array_free(o);
        return e;
    };
    return g.adopt(o);
}

fn mismatch(diag: ?*sdk.Diag, comptime fmt: []const u8, args: anytype) error{BeamGlueCheck} {
    if (diag) |d| d.set("beam decode_glue=fused: the self-check failed: " ++ fmt, args) else std.debug.print("BEAM GLUE mismatch: " ++ fmt ++ "\n", args);
    return error.BeamGlueCheck;
}

/// The stock router's logits, ids and weights of `x` (graph.zig `route`'s ops, through `graph`).
fn stockRoute(g: *G, c: *const beam.Config, w: T, bias: T, x: T) !struct { logits: T, ids: T, weights: T } {
    const graph = @import("graph.zig");
    const logits = try graph.linear(g, try g.astype(x, .float32), w);
    const r = try graph.route(g, c, .{ .w = w, .bias = bias }, x);
    return .{ .logits = logits, .ids = r.indices, .weights = r.weights };
}

/// Once, at install: the texts against the stock ops on one fixed row over the model's first routed layers' tables,
/// byte for byte: the logits, the ids, the weights (M = 1), the predictor's scores; then the select text at the decode
/// lane's widest call on MLX's logits of that many rows. Every text is dispatched (Metal compiles it here).
pub fn selfCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const beam.Config, k: *const Kernels, t: CheckTables, diag: ?*sdk.Diag) !void {
    const graph = @import("graph.zig");
    var g = try G.init(a, s);
    defer g.deinit();
    // The stock ops' compiled regions on this graph (the full layer's query scale is one).
    try graph.prepareFused(&g, c);
    const x = try checkRow(&g, a, c.hidden_size);
    const stock = try stockRoute(&g, c, t.w, t.bias, x);
    var logits: [1]T = undefined;
    try apply(&g, k.gemv, k.route_cfg, &.{ x, t.w, t.bias }, &logits);
    try sameBytes(&g, "the router's logits", stock.logits, logits[0], diag);
    const fused = try routeRows(&g, k, t.w, t.bias, x, 1);
    try sameBytes(&g, "the router's ids", stock.ids, fused.ids, diag);
    try sameBytes(&g, "the router's weights", stock.weights, fused.weights, diag);
    const p = k.predictor();
    try sameBytes(&g, "the predictor's scores", try route.scoresStored(&p, &g, x, t.next_w, t.next_bias), try p.scores(&p, &g, x, t.next_w, t.next_bias), diag);
    // The decode lane's widest call: the fixed row and its negation, alternately.
    var parts: [max_rows]T = undefined;
    const neg = try g.neg(x);
    for (&parts, 0..) |*pt, i| pt.* = if (i % 2 == 0) x else neg;
    const xs = try g.concat(&parts, 0);
    const wide = try stockRoute(&g, c, t.w, t.bias, xs);
    const wf = try routeRows(&g, k, t.w, t.bias, xs, max_rows);
    try sameBytes(&g, "the router's ids at the lane's widest call", wide.ids, wf.ids, diag);
    try sameBytes(&g, "the router's weights at the lane's widest call", wide.weights, wf.weights, diag);
    // The heads on a sliding layer and a full one, over the first layer's norms: rows of the fixed row's values, at a
    // position inside the first window and one far past it.
    const qn: c_int = @intCast(c.n_heads * c.head_dim);
    const kn: c_int = @intCast(c.n_kv_heads * c.head_dim);
    // The fused projections' row: q and k at its head, then v and the gate (the text reads only the first two).
    const row = try checkRow(&g, a, @intCast(2 * qn + 2 * kn));
    const q = try g.slice(row, &.{ 0, 0 }, &.{ 1, qn }, &.{ 1, 1 });
    const kk = try g.slice(row, &.{ 0, qn }, &.{ 1, qn + kn }, &.{ 1, 1 });
    for ([_]u32{ 300, 16_957 }) |pos| for ([_]bool{ false, true }) |full| {
        const want = try graph.qkStock(&g, c, t.q_norm, t.k_norm, q, kk, 1, pos, full);
        const got = try qkNorm(&g, k, row, t.q_norm, t.k_norm, pos, full, graph.ssmaxAt(pos));
        try sameBytes(&g, if (full) "the full layer's query heads" else "the sliding layer's query heads", want.q, got.q, diag);
        try sameBytes(&g, if (full) "the full layer's key heads" else "the sliding layer's key heads", want.k, got.k, diag);
    };
    // The first layer's attention boundary (sublayer 2) and a deep one (sublayer 105) on the first layer's norms.
    const y = try checkRow(&g, a, c.hidden_size);
    const r = try g.neg(try checkRow(&g, a, c.hidden_size));
    for ([_]u32{ 2, 105 }) |i| {
        const want = try graph.boundaryStock(&g, c, y, r, t.post_attn_norm, t.pre_ff_norm, i);
        const got = try addNorm(&g, k, y, r, t.post_attn_norm, t.pre_ff_norm, i);
        try sameBytes(&g, "the boundary's residual", want.res, got.res, diag);
        try sameBytes(&g, "the boundary's normed row", want.out, got.out, diag);
    }
    // The combine at every decode-lane width over six fixed rows (a device wave's four and two miss rows that replace
    // two of its positions, as the scheduler's join orders them).
    const kk_n = c.n_experts_per_tok;
    const wave = try g.reshape(try checkRow(&g, a, 6 * c.hidden_size), &.{ 6, @intCast(c.hidden_size) });
    for (1..max_rows + 1) |m| {
        const ms = g.mark();
        defer g.resetTo(ms);
        const n: u32 = @intCast(m);
        var ord: [max_rows * @import("experts.zig").top_k]u32 = undefined;
        var wv: [max_rows * @import("experts.zig").top_k]f32 = undefined;
        for (0..n * kk_n) |pi| {
            ord[pi] = if (pi == 1) 4 else if (pi == 2) 5 else @intCast(pi % 4);
            wv[pi] = 1.0 / @as(f32, @floatFromInt(kk_n)) + @as(f32, @floatFromInt(pi % 3)) * 0.0625 - 0.0625;
        }
        const order = try g.hostArray(std.mem.sliceAsBytes(ord[0 .. n * kk_n]), &.{@intCast(n * kk_n)}, .uint32);
        const weights = try g.hostArray(std.mem.sliceAsBytes(wv[0 .. n * kk_n]), &.{ @intCast(n), @intCast(kk_n) }, .float32);
        try sameBytes(&g, "the combined routed rows", try combineStockRows(&g, c, wave, order, weights, n), try combineRows(&g, k, wave, order, weights, n), diag);
    }
}

const testing = std.testing;

test "beam glue: the embedded manifest and texts are the pinned ones" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 6);
}

test "beam glue: Beam 1.0's and the fixture's routers are the texts'; another is refused by name" {
    const c: beam.Config = .{};
    try testing.expect(registered(&c));
    try testing.expect(registered(&.{ .hidden_size = 256, .n_routed_experts = 8, .n_group = 2, .topk_group = 1, .n_experts_per_tok = 2, .n_heads = 4, .n_kv_heads = 2, .head_dim = 64 }));
    for ([_]beam.Config{
        .{ .hidden_size = 4096 },
        .{ .n_experts_per_tok = 6 },
        .{ .n_group = 4 },
        .{ .topk_group = 2 },
        .{ .n_routed_experts = 128 },
        .{ .norm_topk_prob = false },
        .{ .n_heads = 64 },
        .{ .head_dim = 64 },
    }) |bad| try testing.expect(!registered(&bad));
    // Every registered geometry is one MLX's gemv runs as bm1_bn8 (matmul.cpp `gemv_axbpy`: K >= 16 N, K > 64) in whole
    // threadgroups of the GEMV and within one threadgroup of the select text.
    for (registered_geometries) |r| {
        try testing.expect(r.head_dim == 64 or r.head_dim == 128);
        try testing.expect(r.hidden >= 16 * r.experts and r.hidden > 64 and r.hidden % 4 == 0);
        try testing.expect(r.experts % rows_per_group == 0 and r.experts % 4 == 0 and r.experts <= 1024);
        try testing.expect(r.experts % r.groups == 0 and r.experts / r.groups >= 2 and r.kept_groups <= r.groups and r.top_k <= 8);
    }
}

test "beam glue: the launch geometry (the GEMV's experts / R threadgroups of 256, the select text's threadgroup a row)" {
    const r = Launch.gemv(256);
    try testing.expectEqual([3]u32{ 256 * 256 / rows_per_group, 1, 1 }, r.grid);
    try testing.expectEqual([3]u32{ 256, 1, 1 }, r.group);
    for (1..max_rows + 1) |m| {
        const sl = Launch.select(256, @intCast(m));
        try testing.expectEqual([3]u32{ 256, @intCast(m), 1 }, sl.grid);
        try testing.expectEqual([3]u32{ 256, 1, 1 }, sl.group);
    }
    // 48 + 8 heads: 7 threadgroups of 8 simdgroups; the fixture's 4 + 2: one.
    try testing.expectEqual([3]u32{ 7 * 256, 1, 1 }, Launch.qk(56).grid);
    try testing.expectEqual([3]u32{ 256, 1, 1 }, Launch.qk(6).grid);
    // The boundary: rms_looped's 1,024 threads at 5,120; rms_single_row's 64 at the fixture's 256, 1,024 at 4,096.
    try testing.expectEqual([3]u32{ 1024, 1, 1 }, Launch.addNorm(5120).grid);
    try testing.expectEqual(@as(u32, 64), normThreads(256));
    try testing.expectEqual(@as(u32, 1024), normThreads(4096));
    // The depth scales of sublayer 2: 1 / sqrt(2) and sqrt(1 / 2), the same f32.
    const sc = depthScales(2);
    try testing.expectEqual(sc[0], sc[1]);
    // The combine: a thread a column of each row, 20 threadgroups of 256 at 5,120; the fixture's 256 in one.
    for (1..max_rows + 1) |m| {
        const cb = Launch.combine(5120, @intCast(m));
        try testing.expectEqual([3]u32{ 5120, @intCast(m), 1 }, cb.grid);
        try testing.expectEqual([3]u32{ 256, 1, 1 }, cb.group);
    }
    try testing.expectEqual([3]u32{ 256, 1, 1 }, Launch.combine(256, 1).group);
}

test "beam glue: the manifest registers the instantiations the install builds (each geometry at M = 1 and the lane's widths)" {
    const J = struct {
        kernels: []const struct { name: []const u8, instances: []const std.json.Value },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, texts.manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    for (registered_geometries) |geo| {
        var gemv_found: [2]bool = @splat(false);
        var select_found: [max_rows]bool = @splat(false);
        var qk_found: [2]bool = @splat(false);
        var add_found = false;
        var combine_found: [max_rows]bool = @splat(false);
        for (m.value.kernels) |kk| for (kk.instances) |inst| {
            const o = inst.object;
            const int = struct {
                fn f(obj: std.json.ObjectMap, key: []const u8) i64 {
                    return obj.get(key).?.integer;
                }
            }.f;
            if (std.mem.eql(u8, kk.name, "glue_router_gemv")) {
                if (int(o, "D") == geo.hidden and int(o, "E") == geo.experts and int(o, "M") == 1 and int(o, "R") == rows_per_group)
                    gemv_found[@intCast(int(o, "EPI"))] = true;
            } else if (std.mem.eql(u8, kk.name, "glue_router_select")) {
                if (int(o, "E") == geo.experts and int(o, "G") == geo.groups and int(o, "TG") == geo.kept_groups and int(o, "K") == geo.top_k)
                    select_found[@intCast(int(o, "M") - 1)] = true;
            } else if (std.mem.eql(u8, kk.name, "glue_qk_norm")) {
                if (int(o, "HQ") == geo.heads and int(o, "HK") == geo.kv_heads and int(o, "HD") == geo.head_dim and int(o, "HPG") == heads_per_group)
                    qk_found[@intCast(int(o, "MODE"))] = true;
            } else if (std.mem.eql(u8, kk.name, "glue_add_norm")) {
                if (int(o, "D") == geo.hidden and int(o, "TPG") == normThreads(geo.hidden)) add_found = true;
            } else if (std.mem.eql(u8, kk.name, "glue_moe_combine")) {
                if (int(o, "D") == geo.hidden and int(o, "K") == geo.top_k) combine_found[@intCast(int(o, "M") - 1)] = true;
            } else if (!std.mem.eql(u8, kk.name, "glue_q8_gemv")) return error.BeamGlueManifestKernel;
        };
        for (gemv_found) |f| try testing.expect(f);
        for (select_found) |f| try testing.expect(f);
        for (qk_found) |f| try testing.expect(f);
        try testing.expect(add_found);
        for (combine_found) |f| try testing.expect(f);
    }
}

test "beam glue: the texts name the shapes they are launched at" {
    try testing.expect(std.mem.indexOf(u8, texts.gemv, "threadgroup float part[M * R * 8];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.gemv, "out[size_t(m) * E + e] = glue_sigmoid(result[m][r]) + bias[e];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.select, "threadgroup float vs[E];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.select, "weights[row * K + e] = q * consts[0];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header, "fp contract") == null);
    // The q / k norm text reads the projection row in place; the combine reads the routed rows through the join's order,
    // without contraction into FMA (its own header), and scales once.
    try testing.expect(std.mem.indexOf(u8, texts.qk, "const device bfloat16_t* xin = qk + size_t(h) * HD;") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "rows[size_t(o[k]) * D + h]") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "static_cast<bfloat16_t>(static_cast<float>(routed) * consts[0])") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header_combine, "#pragma METAL fp contract(off)") != null);
}

// ── The texts on the device (BEAM_PARITY's environment: the box's suites under gpu-lock) ──

/// The device tests' gate: Beam's parity environment under the GPU lock (box-build.sh `suites -e`).
fn deviceTest() !void {
    _ = std.c.getenv("BEAM_PARITY") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
}

/// An f32 array of `shape` from `seed`: normal values times `scale`.
fn testF32(a: std.mem.Allocator, g: *G, shape: []const c_int, scale: f32, seed: u64) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try a.alloc(f32, n);
    defer a.free(v);
    var prng = std.Random.DefaultPrng.init(seed);
    for (v) |*x| x.* = prng.random().floatNorm(f32) * scale;
    return g.hostArray(std.mem.sliceAsBytes(v), shape, .float32);
}

/// A bf16 array of `shape` from `seed`: normal values times `scale` times a power of two in [2^-6, 2^3], with zeros.
fn testBf16(a: std.mem.Allocator, g: *G, shape: []const c_int, scale: f32, seed: u64) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const bits = try a.alloc(u16, n);
    defer a.free(bits);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (bits, 0..) |*b, i| {
        const v: f32 = if (i % 97 == 0) 0 else r.floatNorm(f32) * scale * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 3)));
        b.* = ops.bf16Bits(v);
    }
    return g.hostArray(std.mem.sliceAsBytes(bits), shape, .bfloat16);
}

/// The model of a registered geometry (the router's and the heads' fields; the rest Beam's).
fn configOf(geo: Geometry) beam.Config {
    return .{ .hidden_size = geo.hidden, .n_routed_experts = geo.experts, .n_group = geo.groups, .topk_group = geo.kept_groups, .n_experts_per_tok = geo.top_k, .n_heads = geo.heads, .n_kv_heads = geo.kv_heads, .head_dim = geo.head_dim };
}

test "beam glue GPU: the router's and the predictor's texts are the stock ops bit for bit (Beam's router and the fixture's; M = 1 .. 4; ties, saturation)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for (registered_geometries) |geo| {
        const c = configOf(geo);
        var g = try G.init(a, s);
        defer g.deinit();
        const d: c_int = @intCast(geo.hidden);
        const e: c_int = @intCast(geo.experts);
        // The router tables: logits of order one over the rows below; the correction a hundredth of a score.
        const w = try testF32(a, &g, &.{ e, d }, 1.0 / @sqrt(@as(f32, @floatFromInt(geo.hidden))), 11);
        const bias = try testF32(a, &g, &.{e}, 0.01, 12);
        const w2 = try testF32(a, &g, &.{ e, d }, 1.0 / @sqrt(@as(f32, @floatFromInt(geo.hidden))), 13);
        const bias2 = try testF32(a, &g, &.{e}, 0.01, 14);
        // Ties: a correction of three values (every group's choice scores tie in places, and groups tie).
        const tie_v = try a.alloc(f32, geo.experts);
        defer a.free(tie_v);
        for (tie_v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 3)) * 0.25;
        const bias_ties = try g.hostArray(std.mem.sliceAsBytes(tie_v), &.{e}, .float32);
        const hd: c_int = @intCast(geo.head_dim);
        const qn = try testBf16(a, &g, &.{hd}, 0.5, 15);
        const kn = try testBf16(a, &g, &.{hd}, 0.5, 16);
        const pn = try testBf16(a, &g, &.{d}, 0.5, 17);
        const fnw = try testBf16(a, &g, &.{d}, 0.5, 18);
        try g.evalAll(&.{ w, bias, w2, bias2, bias_ties, qn, kn, pn, fnw });
        var diag: sdk.Diag = .{};
        var k = install(a, s, &c, .{ .w = w, .bias = bias, .next_w = w2, .next_bias = bias2, .q_norm = qn, .k_norm = kn, .post_attn_norm = pn, .pre_ff_norm = fnw }, &diag) catch |err| {
            std.debug.print("beam glue install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        const p = k.predictor();
        var calls: usize = 0;
        for (0..48) |trial| {
            const m = g.mark();
            defer g.resetTo(m);
            const rows: u32 = @intCast(1 + trial % max_rows);
            // Every eighth trial: a zero row (every logit 0: the sigmoid's 0.5 everywhere, the correction decides), or
            // a row 256 times larger (sigmoid saturated at 0 and 1: exact ties).
            const scale: f32 = if (trial % 8 == 5) 256 else 1;
            const x = if (trial % 8 == 3) try g.zeros(&.{ @intCast(rows), d }, .bfloat16) else try testBf16(a, &g, &.{ @intCast(rows), d }, scale, 100 + trial);
            const b = if (trial % 4 == 1) bias_ties else bias;
            const stock = try stockRoute(&g, &c, w, b, x);
            const fused = try routeRows(&g, &k, w, b, x, rows);
            try sameBytes(&g, "ids", stock.ids, fused.ids, null);
            try sameBytes(&g, "weights", stock.weights, fused.weights, null);
            if (rows == 1) {
                var logits: [1]T = undefined;
                try apply(&g, k.gemv, k.route_cfg, &.{ x, w, b }, &logits);
                try sameBytes(&g, "logits", stock.logits, logits[0], null);
                try sameBytes(&g, "predictor scores", try route.scoresStored(&p, &g, x, w2, bias2), try p.scores(&p, &g, x, w2, bias2), null);
            }
            calls += 1;
        }
        std.debug.print("BEAM GLUE hidden {d}, {d} experts in {d} groups ({d} kept), top-{d}: {d} calls at M = 1 .. {d}, ids, weights, logits and predictor scores bit-equal\n", .{ geo.hidden, geo.experts, geo.groups, geo.kept_groups, geo.top_k, calls, max_rows });
    }
}

test "beam glue GPU: the q / k norm text is the stock norms and RoPE, or the norms and the query scale, bit for bit (both geometries, positions to 2^20)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const graph = @import("graph.zig");
    for (registered_geometries) |geo| {
        const c = configOf(geo);
        var g = try G.init(a, s);
        defer g.deinit();
        // The stock ops' compiled regions (the full layer's query scale is one).
        try graph.prepareFused(&g, &c);
        const d: c_int = @intCast(geo.hidden);
        const e: c_int = @intCast(geo.experts);
        const hd: c_int = @intCast(geo.head_dim);
        const qd: c_int = @intCast(geo.heads * geo.head_dim);
        const kd: c_int = @intCast(geo.kv_heads * geo.head_dim);
        const w = try testF32(a, &g, &.{ e, d }, 1.0 / @sqrt(@as(f32, @floatFromInt(geo.hidden))), 11);
        const bias = try testF32(a, &g, &.{e}, 0.01, 12);
        const qn = try testBf16(a, &g, &.{hd}, 0.5, 15);
        const kn = try testBf16(a, &g, &.{hd}, 0.5, 16);
        const pn = try testBf16(a, &g, &.{d}, 0.5, 17);
        const fnw = try testBf16(a, &g, &.{d}, 0.5, 18);
        try g.evalAll(&.{ w, bias, qn, kn, pn, fnw });
        var diag: sdk.Diag = .{};
        var k = install(a, s, &c, .{ .w = w, .bias = bias, .next_w = w, .next_bias = bias, .q_norm = qn, .k_norm = kn, .post_attn_norm = pn, .pre_ff_norm = fnw }, &diag) catch |err| {
            std.debug.print("beam glue install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        var calls: usize = 0;
        for ([_]u32{ 0, 1, 7, 300, 511, 512, 16_957, 32_767, 1_048_575 }, 0..) |pos, i| for ([_]bool{ false, true }) |full| {
            const m = g.mark();
            defer g.resetTo(m);
            // The fused projections' row (q, k at its head), its values of a few magnitudes, with zeros: the stock ops on
            // its row views, the text on the row itself.
            const row = try testBf16(a, &g, &.{ 1, qd + kd + kd + qd }, 1.0, 1300 + i);
            const q = try g.slice(row, &.{ 0, 0 }, &.{ 1, qd }, &.{ 1, 1 });
            const kk = try g.slice(row, &.{ 0, qd }, &.{ 1, qd + kd }, &.{ 1, 1 });
            const want = try graph.qkStock(&g, &c, qn, kn, q, kk, 1, pos, full);
            const got = try qkNorm(&g, &k, row, qn, kn, pos, full, graph.ssmaxAt(pos));
            try sameBytes(&g, "query heads", want.q, got.q, null);
            try sameBytes(&g, "key heads", want.k, got.k, null);
            calls += 1;
        };
        std.debug.print("BEAM GLUE qk_norm {d} / {d} heads of {d}: {d} rows (9 positions to 2^20, sliding and full) bit-equal\n", .{ geo.heads, geo.kv_heads, geo.head_dim, calls });
    }
}

test "beam glue GPU: the boundary text is the stock norm, depth-scaled add and norm bit for bit (both hidden sizes, every sublayer index)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const graph = @import("graph.zig");
    for (registered_geometries) |geo| {
        const c = configOf(geo);
        var g = try G.init(a, s);
        defer g.deinit();
        try graph.prepareFused(&g, &c);
        const d: c_int = @intCast(geo.hidden);
        const e: c_int = @intCast(geo.experts);
        const hd: c_int = @intCast(geo.head_dim);
        const w = try testF32(a, &g, &.{ e, d }, 1.0 / @sqrt(@as(f32, @floatFromInt(geo.hidden))), 11);
        const bias = try testF32(a, &g, &.{e}, 0.01, 12);
        const qn = try testBf16(a, &g, &.{hd}, 0.5, 15);
        const kn = try testBf16(a, &g, &.{hd}, 0.5, 16);
        const pn = try testBf16(a, &g, &.{d}, 0.5, 17);
        const fnw = try testBf16(a, &g, &.{d}, 0.5, 18);
        try g.evalAll(&.{ w, bias, qn, kn, pn, fnw });
        var diag: sdk.Diag = .{};
        var k = install(a, s, &c, .{ .w = w, .bias = bias, .next_w = w, .next_bias = bias, .q_norm = qn, .k_norm = kn, .post_attn_norm = pn, .pre_ff_norm = fnw }, &diag) catch |err| {
            std.debug.print("beam glue install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        var calls: usize = 0;
        var i: u32 = 2;
        while (i <= 2 * c.n_layers + 1) : (i += 1) {
            const m = g.mark();
            defer g.resetTo(m);
            // A sublayer's output of a few magnitudes (an attention's, an MLP's) and a residual stream of order ten.
            const y = try testBf16(a, &g, &.{ 1, d }, if (i % 2 == 0) 1.0 else 8.0, 1800 + i);
            const r = try testBf16(a, &g, &.{ 1, d }, 4.0, 1900 + i);
            const want = try graph.boundaryStock(&g, &c, y, r, pn, fnw, i);
            const got = try addNorm(&g, &k, y, r, pn, fnw, i);
            try sameBytes(&g, "residual", want.res, got.res, null);
            try sameBytes(&g, "normed row", want.out, got.out, null);
            calls += 1;
        }
        std.debug.print("BEAM GLUE add_norm hidden {d} ({d} threads): {d} sublayer boundaries bit-equal\n", .{ geo.hidden, normThreads(geo.hidden), calls });
    }
}

/// A join's order for `n` rows of `k` routed positions over `r` concatenated rows: each position's row, a device wave's
/// position row or a miss row past the wave's (the scheduler's joinOrder) or, every seventh call, a repeat.
fn testOrder(g: *G, n: u32, k: u32, r: u32, seed: u64) !T {
    var ord: [max_rows * @import("experts.zig").top_k]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (ord[0 .. n * k], 0..) |*o, p| o.* = if (seed % 7 == 0) rnd.uintLessThan(u32, r) else if (rnd.uintLessThan(u32, 4) == 0) n * k + rnd.uintLessThan(u32, r - n * k) else @as(u32, @intCast(p));
    return g.hostArray(std.mem.sliceAsBytes(ord[0 .. n * k]), &.{@intCast(n * k)}, .uint32);
}

test "beam glue GPU: the combine text is the stock join, combine and routed scale bit for bit (both geometries, M = 1 .. 4, miss rows, repeats, zeros, saturated rows)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const graph = @import("graph.zig");
    for (registered_geometries) |geo| {
        const c = configOf(geo);
        var g = try G.init(a, s);
        defer g.deinit();
        try graph.prepareFused(&g, &c);
        const d: c_int = @intCast(geo.hidden);
        const e: c_int = @intCast(geo.experts);
        const hd: c_int = @intCast(geo.head_dim);
        const w = try testF32(a, &g, &.{ e, d }, 1.0 / @sqrt(@as(f32, @floatFromInt(geo.hidden))), 11);
        const bias = try testF32(a, &g, &.{e}, 0.01, 12);
        const qn = try testBf16(a, &g, &.{hd}, 0.5, 15);
        const kn = try testBf16(a, &g, &.{hd}, 0.5, 16);
        const pn = try testBf16(a, &g, &.{d}, 0.5, 17);
        const fnw = try testBf16(a, &g, &.{d}, 0.5, 18);
        try g.evalAll(&.{ w, bias, qn, kn, pn, fnw });
        var diag: sdk.Diag = .{};
        var k = install(a, s, &c, .{ .w = w, .bias = bias, .next_w = w, .next_bias = bias, .q_norm = qn, .k_norm = kn, .post_attn_norm = pn, .pre_ff_norm = fnw }, &diag) catch |err| {
            std.debug.print("beam glue install refused: {s}\n", .{diag.message()});
            return err;
        };
        defer k.deinit();
        var calls: usize = 0;
        for (0..64) |trial| {
            const m = g.mark();
            defer g.resetTo(m);
            const rows: u32 = @intCast(1 + trial % max_rows);
            // A device wave's rows and up to three miss rows; every eighth trial rows of zeros, or rows 2^12 times
            // larger (the products' and the sum's rounding at the bf16 extremes).
            const extra: u32 = @intCast(1 + trial % 3);
            const r: u32 = rows * geo.top_k + extra;
            const scale: f32 = if (trial % 8 == 5) 4096 else 1;
            const rr = if (trial % 8 == 3) try g.zeros(&.{ @intCast(r), d }, .bfloat16) else try testBf16(a, &g, &.{ @intCast(r), d }, scale, 2000 + trial);
            const order = try testOrder(&g, rows, geo.top_k, r, 2100 + trial);
            // The router's normalized weights, or (every fourth trial) unnormalized ones with a zero.
            const wt = if (trial % 4 == 1) try g.astype(try testBf16(a, &g, &.{ @intCast(rows), @intCast(geo.top_k) }, 1.0, 2200 + trial), .float32) else (try routeRows(&g, &k, w, bias, try testBf16(a, &g, &.{ @intCast(rows), d }, 1.0, 2300 + trial), rows)).weights;
            try sameBytes(&g, "combined rows", try combineStockRows(&g, &c, rr, order, wt, rows), try combineRows(&g, &k, rr, order, wt, rows), null);
            calls += 1;
        }
        std.debug.print("BEAM GLUE combine hidden {d}, top-{d}: {d} calls at M = 1 .. {d} (miss rows, repeats, zeros, 2^12 rows) bit-equal\n", .{ geo.hidden, geo.top_k, calls, max_rows });
    }
}

test "beam glue GPU bench: a step's 49 routed combines, the stock join, combine and scale against the text (M = 1, dependent)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const graph = @import("graph.zig");
    const geo = registered_geometries[0];
    const c = configOf(geo);
    var g = try G.init(a, s);
    defer g.deinit();
    try graph.prepareFused(&g, &c);
    const d: c_int = @intCast(geo.hidden);
    const e: c_int = @intCast(geo.experts);
    const layers = 49;
    // Each layer's wave rows [8, D] (kept), its order and weights; each layer's rows plus the previous layer's output (one
    // broadcast add, both arms): the chain is dependent, as in a step.
    var waves: [layers]T = undefined;
    for (&waves, 0..) |*x, i| x.* = g.keep(try testBf16(a, &g, &.{ 8, d }, 1.0, 2400 + i));
    defer for (waves) |x| g.release(x);
    try g.evalAll(&waves);
    const w = try testF32(a, &g, &.{ e, d }, 0.014, 11);
    const bias = try testF32(a, &g, &.{e}, 0.01, 12);
    const hd: c_int = @intCast(geo.head_dim);
    const qn = try testBf16(a, &g, &.{hd}, 0.5, 15);
    const kn = try testBf16(a, &g, &.{hd}, 0.5, 16);
    const pn = try testBf16(a, &g, &.{d}, 0.5, 17);
    var diag: sdk.Diag = .{};
    var k = try install(a, s, &c, .{ .w = w, .bias = bias, .next_w = w, .next_bias = bias, .q_norm = qn, .k_norm = kn, .post_attn_norm = pn, .pre_ff_norm = pn }, &diag);
    defer k.deinit();
    const ord = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const order = g.keep(try g.hostArray(std.mem.sliceAsBytes(&ord), &.{8}, .uint32));
    defer g.release(order);
    const wts = g.keep(try g.astype(try testBf16(a, &g, &.{ 1, 8 }, 0.125, 2500), .float32));
    defer g.release(wts);
    try g.evalAll(&.{ order, wts });
    for ([_]bool{ false, true, false, true }) |text| {
        var times: [12]u64 = undefined;
        for (&times) |*tm| {
            const m = g.mark();
            defer g.resetTo(m);
            var y = try g.zeros(&.{ 1, d }, .bfloat16);
            try g.evalAll(&.{y});
            const t0 = std.Io.Timestamp.now(testing.io, .awake);
            for (waves) |wave| {
                const rows = try g.add(wave, y);
                y = if (text) try combineRows(&g, &k, rows, order, wts, 1) else try combineStockRows(&g, &c, rows, order, wts, 1);
            }
            try g.evalAll(&.{y});
            tm.* = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
        }
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        const med: f64 = @floatFromInt(times[times.len / 2]);
        std.debug.print("BEAM GLUE BENCH combine {s}: {d} layers in {d:.1} us (median of 12, graph build and encode included), {d:.2} us a layer\n", .{ if (text) "text" else "stock", layers, med / 1e3, med / 1e3 / layers });
    }
}

/// The q8 GEMV text's launch (`glue_q8_gemv`) at `n` output rows of `k` columns: `rows` rows a simdgroup, `spt`
/// simdgroups a threadgroup (MLX's affine_qmv_fast: 4 and 2), `pf` the next block's codes loaded first.
pub const Q8Plan = struct { rows: u32, spt: u32, pf: bool };

pub fn q8Kernel() !Kernel {
    return newKernel("beam_glue_q8_gemv", &.{ "x", "w", "scales", "biases" }, &.{"y"}, texts.q8, texts.header ++ texts.header_q8);
}

pub fn q8Config(n: u32, k: u32, plan: Q8Plan) !Config {
    const shape = [_]c_int{ 1, @intCast(n) };
    const t = templateOf(&.{ "K", "ROWS", "SPT", "PF" }, .{ @intCast(k), @intCast(plan.rows), @intCast(plan.spt), @intFromBool(plan.pf) });
    return kset.buildConfig(.{ .grid = .{ 32, n / plan.rows, 1 }, .threadgroup = .{ 32, plan.spt, 1 }, .out_shapes = &.{&shape}, .out_dtypes = &.{.bfloat16}, .template = &t });
}

/// The trunk's q8 shapes at Beam 1.0's geometry (N, K): the fused q / k / v / gate, o, the dense gate and up, the dense
/// down, the head.
const q8_shapes = [_][2]u32{ .{ 14336, 5120 }, .{ 5120, 6144 }, .{ 20480, 5120 }, .{ 5120, 20480 }, .{ 151936, 5120 } };
const q8_plans = [_]Q8Plan{ .{ .rows = 4, .spt = 2, .pf = false }, .{ .rows = 4, .spt = 2, .pf = true }, .{ .rows = 8, .spt = 2, .pf = true }, .{ .rows = 8, .spt = 1, .pf = true }, .{ .rows = 4, .spt = 4, .pf = true }, .{ .rows = 2, .spt = 4, .pf = true }, .{ .rows = 8, .spt = 4, .pf = true }, .{ .rows = 2, .spt = 2, .pf = true } };

fn q8Weights(a: std.mem.Allocator, g: *G, n: u32, k: u32, seed: u64) ![3]T {
    const wt = try testBf16(a, g, &.{ @intCast(n), @intCast(k) }, 0.02, seed);
    return ops.quantizeAffine(g, wt, 64, 8);
}

fn q8Apply(g: *G, kern: Kernel, cfg: Config, x: T, q: [3]T) !T {
    var out: [1]T = undefined;
    try apply(g, kern, cfg, &.{ x, q[0], q[1], q[2] }, &out);
    return out[0];
}

test "beam glue GPU: the q8 GEMV text is MLX's affine qmv bit for bit (the trunk's five shapes, every plan)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    const kern = try q8Kernel();
    defer mtl.mlx_fast_metal_kernel_free(kern);
    var checked: usize = 0;
    for (q8_shapes, 0..) |shape, si| {
        const m = g.mark();
        defer g.resetTo(m);
        // The head at 16,384 of its rows (its 151,936 are whole threadgroups of every plan too).
        const n: u32 = if (shape[0] > 100_000) 16384 else shape[0];
        const q = try q8Weights(a, &g, n, shape[1], 7000 + si);
        try g.evalAll(&q);
        for (0..3) |trial| {
            const mt = g.mark();
            defer g.resetTo(mt);
            const x = if (trial == 2) try g.zeros(&.{ 1, @intCast(shape[1]) }, .bfloat16) else try testBf16(a, &g, &.{ 1, @intCast(shape[1]) }, if (trial == 1) 64.0 else 1.0, 7100 + si * 8 + trial);
            const want = try g.quantizedMatmul(x, q[0], q[1], q[2], true, 8, 64, .affine);
            for (q8_plans) |plan| {
                const cfg = try q8Config(n, shape[1], plan);
                defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
                try sameBytes(&g, "q8 gemv", want, try q8Apply(&g, kern, cfg, x, q), null);
                checked += 1;
            }
        }
    }
    std.debug.print("BEAM GLUE q8 gemv: {d} launches over the trunk's 5 shapes x {d} plans bit-equal to MLX's affine_qmv_fast\n", .{ checked, q8_plans.len });
}

test "beam glue GPU bench: the trunk's q8 GEMVs, MLX's qmv against the text's plans (12 layers of distinct weights, dependent)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    const kern = try q8Kernel();
    defer mtl.mlx_fast_metal_kernel_free(kern);
    const layers = 12;
    // Per shape pair: a layer's q / k / v / gate (14,336 x 5,120) then its o (5,120 x 6,144), or a dense layer's up
    // (20,480 x 5,120) then its down (5,120 x 20,480); each layer's weights its own (past the caches), the chain
    // dependent: the first product's leading K columns feed the second, the second's output the next layer's first.
    const pairs = [_][2]usize{ .{ 0, 1 }, .{ 2, 3 } };
    for (pairs) |pr| {
        const n0 = q8_shapes[pr[0]][0];
        const k0 = q8_shapes[pr[0]][1];
        const n1 = q8_shapes[pr[1]][0];
        const k1 = q8_shapes[pr[1]][1];
        var w0: [layers][3]T = undefined;
        var w1: [layers][3]T = undefined;
        for (0..layers) |l| {
            const q0 = try q8Weights(a, &g, n0, k0, 8000 + l);
            const q1 = try q8Weights(a, &g, n1, k1, 8100 + l);
            for (0..3) |i| {
                w0[l][i] = g.keep(q0[i]);
                w1[l][i] = g.keep(q1[i]);
            }
            try g.evalAll(&(w0[l] ++ w1[l]));
            g.clearCache();
        }
        defer for (0..layers) |l| for (0..3) |i| {
            g.release(w0[l][i]);
            g.release(w1[l][i]);
        };
        const bytes: f64 = @floatFromInt(@as(u64, layers) * (n0 * k0 + n1 * k1) * 17 / 16);
        // Plan index -1: MLX's quantized_matmul.
        var p: i32 = -1;
        while (p < q8_plans.len) : (p += 1) {
            var cfgs: [2]Config = .{ .{ .ctx = null }, .{ .ctx = null } };
            if (p >= 0) {
                cfgs[0] = try q8Config(n0, k0, q8_plans[@intCast(p)]);
                cfgs[1] = try q8Config(n1, k1, q8_plans[@intCast(p)]);
            }
            defer for (cfgs) |cfg| if (cfg.ctx != null) {
                _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            };
            var times: [9]u64 = undefined;
            for (&times) |*tm| {
                const m = g.mark();
                defer g.resetTo(m);
                var x = try testBf16(a, &g, &.{ 1, @intCast(k0) }, 1.0, 8200);
                try g.evalAll(&.{x});
                const t0 = std.Io.Timestamp.now(testing.io, .awake);
                for (0..layers) |l| {
                    const y0 = if (p < 0) try g.quantizedMatmul(x, w0[l][0], w0[l][1], w0[l][2], true, 8, 64, .affine) else try q8Apply(&g, kern, cfgs[0], x, w0[l]);
                    // The second product's input: the first's leading k1 columns (every pair has n0 >= k1): a view for
                    // MLX's op, the row itself for the text (it reads its first K values, as the served o reads the
                    // attention's own output).
                    std.debug.assert(n0 >= k1);
                    const y1 = if (p < 0) try g.quantizedMatmul(try g.slice(y0, &.{ 0, 0 }, &.{ 1, @intCast(k1) }, &.{ 1, 1 }), w1[l][0], w1[l][1], w1[l][2], true, 8, 64, .affine) else try q8Apply(&g, kern, cfgs[1], y0, w1[l]);
                    x = y1;
                }
                try g.evalAll(&.{x});
                tm.* = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
            }
            std.mem.sort(u64, &times, {}, std.sort.asc(u64));
            const med: f64 = @floatFromInt(times[times.len / 2]);
            if (p < 0) {
                std.debug.print("BEAM GLUE BENCH q8 {d}x{d} + {d}x{d} MLX qmv: {d} layers in {d:.1} us (median of 9), {d:.0} GB/s\n", .{ n0, k0, n1, k1, layers, med / 1e3, bytes / med });
            } else {
                const pl = q8_plans[@intCast(p)];
                std.debug.print("BEAM GLUE BENCH q8 {d}x{d} + {d}x{d} text rows={d} spt={d} pf={}: {d} layers in {d:.1} us (median of 9), {d:.0} GB/s\n", .{ n0, k0, n1, k1, pl.rows, pl.spt, pl.pf, layers, med / 1e3, bytes / med });
            }
        }
    }
}

test "beam glue GPU bench: a step's routers, the stock ops against the texts at R = 1, 2, 4 (49 layers' tables, M = 1)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const geo = registered_geometries[0];
    const c = configOf(geo);
    var g = try G.init(a, s);
    defer g.deinit();
    const d: c_int = @intCast(geo.hidden);
    const e: c_int = @intCast(geo.experts);
    // A step's 49 routers (257 MB: past the GPU's caches, as in a step); layer i's predictor reads layer i + 1's.
    const layers = 49;
    var ws: [layers + 1]T = undefined;
    var bs: [layers + 1]T = undefined;
    for (&ws, &bs, 0..) |*w, *b, i| {
        w.* = g.keep(try testF32(a, &g, &.{ e, d }, 0.014, 200 + i));
        b.* = g.keep(try testF32(a, &g, &.{e}, 0.01, 300 + i));
    }
    defer for (ws, bs) |w, b| {
        g.release(w);
        g.release(b);
    };
    try g.evalAll(&ws);
    try g.evalAll(&bs);
    const qn = try testBf16(a, &g, &.{@intCast(geo.head_dim)}, 0.5, 15);
    const kn = try testBf16(a, &g, &.{@intCast(geo.head_dim)}, 0.5, 16);
    const pn = try testBf16(a, &g, &.{d}, 0.5, 17);
    const fnw = try testBf16(a, &g, &.{d}, 0.5, 18);
    var diag: sdk.Diag = .{};
    var k = try install(a, s, &c, .{ .w = ws[0], .bias = bs[0], .next_w = ws[1], .next_bias = bs[1], .q_norm = qn, .k_norm = kn, .post_attn_norm = pn, .pre_ff_norm = fnw }, &diag);
    defer k.deinit();
    const x = g.keep(try testBf16(a, &g, &.{ 1, d }, 1, 400));
    defer g.release(x);
    const p = k.predictor();
    for ([_]u32{ 0, 1, 2, 4 }) |r_rows| {
        // r_rows 0: the stock ops (graph.zig `route`, route.zig `scoresStored`); else the texts at R = r_rows.
        var cfgs: [2]Config = .{ k.route_cfg, k.pred_cfg };
        const own = r_rows != 0 and r_rows != rows_per_group;
        if (own) for (&cfgs, 0..) |*cfg, epi| {
            const l: Launch = .{ .grid = .{ gemv_threads * (geo.experts / r_rows), 1, 1 }, .group = .{ gemv_threads, 1, 1 } };
            const shape = [_]c_int{ 1, e };
            const t = templateOf(&.{ "D", "E", "M", "R", "EPI" }, .{ d, e, 1, @intCast(r_rows), @intCast(epi) });
            cfg.* = try kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{&shape}, .out_dtypes = &.{.float32}, .template = &t });
        };
        defer if (own) for (cfgs) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        var times: [12]u64 = undefined;
        for (&times) |*tm| {
            const m = g.mark();
            defer g.resetTo(m);
            var outs: [3 * layers]T = undefined;
            for (0..layers) |i| {
                if (r_rows == 0) {
                    const graph = @import("graph.zig");
                    const r = try graph.route(&g, &c, .{ .w = ws[i], .bias = bs[i] }, x);
                    outs[3 * i] = r.indices;
                    outs[3 * i + 1] = r.weights;
                    outs[3 * i + 2] = try route.scoresStored(&p, &g, x, ws[i + 1], bs[i + 1]);
                } else {
                    var logits: [1]T = undefined;
                    try apply(&g, k.gemv, cfgs[0], &.{ x, ws[i], bs[i] }, &logits);
                    var o: [2]T = undefined;
                    try apply(&g, k.select, k.select_cfg[0], &.{ logits[0], bs[i], k.consts }, &o);
                    var sc: [1]T = undefined;
                    try apply(&g, k.gemv, cfgs[1], &.{ x, ws[i + 1], bs[i + 1] }, &sc);
                    outs[3 * i] = o[0];
                    outs[3 * i + 1] = o[1];
                    outs[3 * i + 2] = sc[0];
                }
            }
            const t0 = std.Io.Timestamp.now(testing.io, .awake);
            try g.evalAll(&outs);
            tm.* = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
        }
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        const med: f64 = @floatFromInt(times[times.len / 2]);
        const bytes: f64 = @floatFromInt(@as(u64, layers) * 2 * geo.experts * geo.hidden * 4);
        std.debug.print("BEAM GLUE BENCH {s} R={d}: {d} layers (router + predictor) in {d:.1} us (median of 12), {d:.2} us a layer, {d:.0} GB/s of router weights\n", .{ if (r_rows == 0) "stock" else "texts", r_rows, layers, med / 1e3, med / 1e3 / layers, bytes / med });
    }
}
