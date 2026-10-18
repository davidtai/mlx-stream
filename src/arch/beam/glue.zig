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
//!     depth-scaled add, the next sublayer's norm).
//! Stock: the router is the input cast, the GEMV and 17 ops (sigmoid, add, the groups' topk and sum, negation,
//! argpartition, the group ids, equal, sum, greater, where, negation, argpartition, take_along_axis, sum, add, divide);
//! the predictor is the cast, the GEMV, sigmoid and add. Exact: the ids, the weights, the predictor's scores and the
//! normed heads are the stock route's bit for bit (`install`'s self-check over the model's own tables; the device tests
//! `beam glue GPU`). At M > 1 the router's product stays MLX's matmul (MLX routes [M > 1, D] x [D, E] f32 to its GEMM: reproduced
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

// ── the texts and their pins (src/kernels/decode_glue/manifest_beam.json) ──

const dir = "../../kernels/decode_glue/";

pub const texts = struct {
    pub const manifest = @embedFile(dir ++ "manifest_beam.json");
    pub const header = @embedFile(dir ++ "header_decode_glue.metal");
    pub const gemv = @embedFile(dir ++ "glue_router_gemv.metal");
    pub const select = @embedFile(dir ++ "glue_router_select.metal");
    pub const qk = @embedFile(dir ++ "glue_qk_norm.metal");
    pub const add_norm = @embedFile(dir ++ "glue_add_norm.metal");
};
/// sha256 of manifest_beam.json (scripts/decode_glue_manifest.py prints it), which records every text's sha256 (checked
/// against the pins by a CPU test).
pub const manifest_sha256 = "668af11732c2705238a6057f79f69e38454d481127e3b15f61fc00249faa28e3";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_decode_glue.metal", .text = texts.header, .sha256 = "6bec0858b4e0754c74a4eee95560bd78f0942ecc092cc50d50bcc7840977e484" },
    .{ .file = "glue_router_gemv.metal", .text = texts.gemv, .sha256 = "1c82d26867b1c277adb2ca003200f6c05c3ecc396126b74ac571854b202bdcb7" },
    .{ .file = "glue_router_select.metal", .text = texts.select, .sha256 = "5a7369ad5c53e6eb5b08013219ad7731cfa797863599d9be8f2c2e4715bf47f6" },
    .{ .file = "glue_qk_norm.metal", .text = texts.qk, .sha256 = "4118ee77c24b91bf6386e0bb28e84df8aeec27a87064387d29609339891c1067" },
    .{ .file = "glue_add_norm.metal", .text = texts.add_norm, .sha256 = "7802402dbccaa535d22b60265c0476f20cd55788572c45c7b5429d7bd836042d" },
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
        for ([_]Kernel{ k.gemv, k.select, k.qk, k.add_norm }) |x| if (x.ctx != null) {
            mtl.mlx_fast_metal_kernel_free(x);
        };
        for ([_]Config{ k.route_cfg, k.pred_cfg, k.add_norm_cfg } ++ k.select_cfg ++ k.qk_cfg) |cfg| if (cfg.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        if (k.consts.ctx != null) _ = mlx.mlx_array_free(k.consts);
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
};

/// MLX's rms threadgroup at width `d` (normalization.cpp RMSNorm::eval_gpu): rms_looped's 1,024 threads past
/// RMS_LOOPED_LIMIT (4,096), else rms_single_row's 32 x ceil(d / 128).
pub fn normThreads(d: u32) u32 {
    return if (d > 4096) 1024 else 32 * ((d + 127) / 128);
}

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !Kernel {
    // The inputs are the forward's fresh row-major rows and the residents (a view would be copied: ensure_row_contiguous).
    return kset.newKernel(name, ins, outs, source.ptr, texts.header.ptr, true) orelse error.BeamGlueKernel;
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
    k.gemv = try newKernel("beam_glue_router_gemv", &.{ "x", "w", "bias" }, &.{"out"}, texts.gemv);
    k.select = try newKernel("beam_glue_router_select", &.{ "logits", "bias", "consts" }, &.{ "ids", "weights" }, texts.select);
    k.qk = try newKernel("beam_glue_qk_norm", &.{ "q", "k", "qw", "kw", "consts" }, &.{ "qo", "ko" }, texts.qk);
    k.add_norm = try newKernel("beam_glue_add_norm", &.{ "y", "r", "w1", "w2", "consts" }, &.{ "res", "out" }, texts.add_norm);
    k.add_norm_cfg = try addNormConfig(geo);
    k.route_cfg = try gemvConfig(geo, 0);
    k.pred_cfg = try gemvConfig(geo, 1);
    for (&k.select_cfg, 1..) |*cfg, m| cfg.* = try selectConfig(geo, @intCast(m));
    for (&k.qk_cfg, 0..) |*cfg, mode| cfg.* = try qkConfig(geo, @intCast(mode));
    var cv: [8]f32 = @splat(0);
    cv[0] = 1;
    k.consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.consts.ctx == null) return error.MlxError;
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

/// A decode row's normed query and key heads: q [1, heads * head_dim], k [1, kv_heads * head_dim] bf16 (the fused
/// projections' row views), the layer's q_norm and k_norm [head_dim] bf16, at position `pos` -> q [1, heads, 1,
/// head_dim], k [1, kv_heads, 1, head_dim] bf16: roped on a sliding layer, the query times `scale` (the full layer's
/// query scale, graph.zig `ssmaxAt`) on a full layer. One dispatch (graph.zig `qkStock`'s four or three).
pub fn qkNorm(g: *G, k: *const Kernels, q: T, kk: T, q_norm: T, k_norm: T, pos: u32, full: bool, scale: f32) !Heads {
    var cv: [8]f32 = @splat(0);
    cv[0] = k.rope_log2;
    cv[1] = scale;
    cv[2] = k.eps;
    cv[3] = @floatFromInt(pos);
    const consts = try g.hostArray(std.mem.sliceAsBytes(&cv), &.{8}, .float32);
    var out: [2]T = undefined;
    try apply(g, k.qk, k.qk_cfg[@intFromBool(full)], &.{ q, kk, q_norm, k_norm, consts }, &out);
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
    const row = try checkRow(&g, a, @intCast(qn + kn));
    const q = try g.slice(row, &.{ 0, 0 }, &.{ 1, qn }, &.{ 1, 1 });
    const kk = try g.slice(row, &.{ 0, qn }, &.{ 1, qn + kn }, &.{ 1, 1 });
    for ([_]u32{ 300, 16_957 }) |pos| for ([_]bool{ false, true }) |full| {
        const want = try graph.qkStock(&g, c, t.q_norm, t.k_norm, q, kk, 1, pos, full);
        const got = try qkNorm(&g, k, q, kk, t.q_norm, t.k_norm, pos, full, graph.ssmaxAt(pos));
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
}

const testing = std.testing;

test "beam glue: the embedded manifest and texts are the pinned ones" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 4);
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
            } else return error.BeamGlueManifestKernel;
        };
        for (gemv_found) |f| try testing.expect(f);
        for (select_found) |f| try testing.expect(f);
        for (qk_found) |f| try testing.expect(f);
        try testing.expect(add_found);
    }
}

test "beam glue: the texts name the shapes they are launched at" {
    try testing.expect(std.mem.indexOf(u8, texts.gemv, "threadgroup float part[M * R * 8];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.gemv, "out[size_t(m) * E + e] = glue_sigmoid(result[m][r]) + bias[e];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.select, "threadgroup float vs[E];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.select, "weights[row * K + e] = q * consts[0];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header, "fp contract") == null);
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
            // The fused projections' row (q, k at its head), its values of a few magnitudes, with zeros.
            const row = try testBf16(a, &g, &.{ 1, qd + kd + kd + qd }, 1.0, 1300 + i);
            const q = try g.slice(row, &.{ 0, 0 }, &.{ 1, qd }, &.{ 1, 1 });
            const kk = try g.slice(row, &.{ 0, qd }, &.{ 1, qd + kd }, &.{ 1, 1 });
            const want = try graph.qkStock(&g, &c, qn, kn, q, kk, 1, pos, full);
            const got = try qkNorm(&g, &k, q, kk, qn, kn, pos, full, graph.ssmaxAt(pos));
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

test "beam glue GPU bench (DIAG, not for merge): a forward's q / k heads and sublayer boundaries, the stock ops at 1, 4 and 8 rows against the texts at 1 row (Beam 1.0's geometry, one dependency chain)" {
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
    const zero = g.keep(try g.scalar(0, .bfloat16));
    defer g.release(zero);
    // The 16K workload's decode positions: every sliding layer ropes, every full layer scales (Beam's layer pattern).
    const pos: u32 = 16_445;
    // `ops`: 0 the chain's shared ops alone (no heads, no boundaries), 1 the stock ops, 2 the texts. Each layer: the
    // attention's q / k sources made from the chain's normed row (one product with zero, an add each: shared by every
    // arm), the heads, the attention's output made from the heads' first key element (a product with zero and an add:
    // shared), then the layer's two boundaries (sublayer indices 2l + 2 and 2l + 3), each boundary's normed row and
    // residual feeding the next.
    const Arm = struct { rows: c_int, ops: u8 };
    for ([_]Arm{ .{ .rows = 1, .ops = 0 }, .{ .rows = 1, .ops = 1 }, .{ .rows = 1, .ops = 2 }, .{ .rows = 4, .ops = 0 }, .{ .rows = 4, .ops = 1 }, .{ .rows = 8, .ops = 0 }, .{ .rows = 8, .ops = 1 } }) |arm| {
        const n = arm.rows;
        const q0 = g.keep(try testBf16(a, &g, &.{ n, qd }, 1.0, 2000));
        const k0 = g.keep(try testBf16(a, &g, &.{ n, kd }, 1.0, 2001));
        const y0 = g.keep(try testBf16(a, &g, &.{ n, d }, 1.0, 2002));
        const r0 = g.keep(try testBf16(a, &g, &.{ n, d }, 4.0, 2003));
        defer for ([_]T{ q0, k0, y0, r0 }) |x| g.release(x);
        try g.evalAll(&.{ q0, k0, y0, r0 });
        var times: [15]u64 = undefined;
        for (&times) |*tm| {
            const m = g.mark();
            defer g.resetTo(m);
            var y = y0;
            var r = r0;
            for (0..c.n_layers) |l| {
                const li: u32 = @intCast(l);
                const full = c.isFull(li);
                const dep = try g.mul(try g.slice(y, &.{ 0, 0 }, &.{ 1, 1 }, &.{ 1, 1 }), zero);
                const qs = try g.add(q0, dep);
                const ks = try g.add(k0, dep);
                var first: T = ks;
                if (arm.ops == 1) {
                    const h = try graph.qkStock(&g, &c, qn, kn, qs, ks, n, pos, full);
                    first = try g.reshape(try g.slice(h.k, &.{ 0, 0, 0, 0 }, &.{ 1, 1, 1, 1 }, &.{ 1, 1, 1, 1 }), &.{ 1, 1 });
                } else if (arm.ops == 2) {
                    const h = try qkNorm(&g, &k, qs, ks, qn, kn, pos, full, graph.ssmaxAt(pos));
                    first = try g.reshape(try g.slice(h.k, &.{ 0, 0, 0, 0 }, &.{ 1, 1, 1, 1 }, &.{ 1, 1, 1, 1 }), &.{ 1, 1 });
                } else first = try g.slice(ks, &.{ 0, 0 }, &.{ 1, 1 }, &.{ 1, 1 });
                const ya = try g.add(y, try g.mul(first, zero));
                if (arm.ops == 0) {
                    y = ya;
                    continue;
                }
                var yi = ya;
                for ([_]u32{ 2 * li + 2, 2 * li + 3 }) |i| {
                    const b = if (arm.ops == 1) try graph.boundaryStock(&g, &c, yi, r, pn, fnw, i) else try addNorm(&g, &k, yi, r, pn, fnw, i);
                    r = b.res;
                    yi = b.out;
                }
                y = yi;
            }
            const t0 = std.Io.Timestamp.now(testing.io, .awake);
            try g.evalAll(&.{ y, r });
            tm.* = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
        }
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        const med: f64 = @floatFromInt(times[times.len / 2]);
        std.debug.print("BEAM GLUE BENCH CHAIN rows={d} {s}: {d} layers' q / k heads and {d} boundaries in {d:.1} us (median of {d}; min {d:.1} us)\n", .{ n, switch (arm.ops) {
            0 => "shared-ops-only",
            1 => "stock",
            else => "texts",
        }, c.n_layers, 2 * c.n_layers, med / 1e3, times.len, @as(f64, @floatFromInt(times[0])) / 1e3 });
    }
}
