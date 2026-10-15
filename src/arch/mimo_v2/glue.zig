//! K2 (mimo/k2-glue, `"decode_glue": "fused"`): a decode-lane layer's glue in four kernels instead of ~20 MLX ops, each
//! the stock op chain expression for expression (kernels/header_mimo_glue.metal):
//!   - the layer boundary, `mimo_glue_addnorm` (layer 0: `mimo_glue_norm`): the previous layer's MLP output added to the
//!     residual and this layer's input norm (stock: add, rms_norm, multiply);
//!   - after the attention: dense layer 0 `mimo_glue_addnorm`; a routed layer at M = 1 `mimo_glue_router1` (the residual
//!     add, the post-attention norm and the router's F32 GEMV in MLX gemv's own accumulation, 64 threadgroups); at M > 1
//!     `mimo_glue_addnorm32` (the same add and norm plus the router's F32 cast) and the router product stays MLX's matmul
//!     (MLX routes [M > 1, 4096] x [4096, 256] F32 to its NAX split-K GEMM on this box: reproduced only by running it);
//!   - `mimo_glue_top8`: sigmoid, correction, MLX's stable argsort's first 8, the normalised weights (stock: 10 ops).
//! Exact by construction: the routed ids and weights, the residual and the norms bit-equal to the stock ops (device
//! self-check `mimo GPU glue K2`; CPU proofs below). The route is installed once (settings `decode_glue`; `ops` is the
//! control and the default) and runs the decode lane (`Source.decodeLaneAt`: M = 1, or M <= 4 in the decode phase).
const std = @import("std");
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const mimo = @import("config.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const Kernel = mlx.mlx_fast_metal_kernel;
const kset = @import("../../sdk_ext/kernel_set.zig");

pub const Route = enum { ops, fused, invalid };

pub fn parseRoute(text: []const u8) Route {
    return std.meta.stringToEnum(Route, text) orelse .invalid;
}

/// The rows a glue kernel takes (the decode lane's M <= 4, with the verify's headroom; the self-check covers 1..8).
pub const max_rows: c_int = 8;
const rows_cap: usize = @intCast(max_rows);
const hidden: c_int = 4096;
const experts: c_int = 256;
const top_k: c_int = 8;
/// MLX gemv's threadgroups for the [256, 4096] F32 router at M = 1 (`gemv_axbpy`: K >= 16 N -> bm 1, bn 8; tm 4): 64.
pub const router1_groups: c_int = @divExact(experts, 4);

// ---- registry: the texts and their pins (kernels/manifest-glue.json) ----

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-glue.json");
    pub const header = @embedFile("kernels/header_mimo_glue.metal");
    pub const norm = @embedFile("kernels/mimo_glue_norm.metal");
    pub const addnorm = @embedFile("kernels/mimo_glue_addnorm.metal");
    pub const addnorm32 = @embedFile("kernels/mimo_glue_addnorm32.metal");
    pub const router1 = @embedFile("kernels/mimo_glue_router1.metal");
    pub const router_rows = @embedFile("kernels/mimo_glue_router_rows.metal");
    pub const top8 = @embedFile("kernels/mimo_glue_top8.metal");
};
/// sha256 of kernels/manifest-glue.json, which records every glue text's sha256 (checked against the pins by a CPU test).
pub const manifest_sha256 = "561df962d8065264a93a5b002a97c566e4173c143d42721c28b8eb5fed2cd7af";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_mimo_glue.metal", .text = texts.header, .sha256 = "75784dde518a7b706e3276c32add41325d28e5bfaa1cff03a68dc40057bf9e51" },
    .{ .file = "mimo_glue_norm.metal", .text = texts.norm, .sha256 = "3008660a5d8e36446aa0e2b4057798655ebbaa3f01496c748e7733147ffe31a6" },
    .{ .file = "mimo_glue_addnorm.metal", .text = texts.addnorm, .sha256 = "f547c88792eaca9cf98ca093c693245985516a44f88ce66e5011e3bd9944eb2c" },
    .{ .file = "mimo_glue_addnorm32.metal", .text = texts.addnorm32, .sha256 = "4f0b82b48f24b03f9481ccb0908744707b7c57be9fdba6ef683003d9e3388b29" },
    .{ .file = "mimo_glue_router1.metal", .text = texts.router1, .sha256 = "a0e10bef80dd1699d372501dd19a8006be4ada01c4362c985f57b4982d8dc988" },
    .{ .file = "mimo_glue_router_rows.metal", .text = texts.router_rows, .sha256 = "1bc126dc3d040904582a991862ec23b1ee89210f08b59cec071a0b8024101afa" },
    .{ .file = "mimo_glue_top8.metal", .text = texts.top8, .sha256 = "ec3124418c8d07f85ff8d788549c8d5e035605ce6adf17084f15e4f3550e40bd" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.MimoGlueManifestNotPinned, error.MimoGlueTextNotPinned);
}

/// The geometry the texts are written for (hidden 4096, 256 experts, top 8, eps 1e-6); anything else is refused at
/// construction.
pub fn requireGeometry(c: mimo.Config) !void {
    if (c.hidden_size != hidden or c.n_routed_experts != experts or c.n_experts_per_tok != top_k or c.rms_norm_eps != 1e-6)
        return error.MimoGlueGeometry;
}

// ---- kernels ----

pub const Kernels = struct {
    norm: Kernel = .{ .ctx = null },
    addnorm: Kernel = .{ .ctx = null },
    addnorm32: Kernel = .{ .ctx = null },
    router1: Kernel = .{ .ctx = null },
    top8: Kernel = .{ .ctx = null },

    pub fn init(c: mimo.Config) !Kernels {
        try requireGeometry(c);
        var k: Kernels = .{};
        errdefer k.deinit();
        k.norm = try makeKernel("mimo_glue_norm", &.{ "x", "w" }, &.{"n"}, texts.norm);
        k.addnorm = try makeKernel("mimo_glue_addnorm", &.{ "a", "b", "w" }, &.{ "h", "n" }, texts.addnorm);
        k.addnorm32 = try makeKernel("mimo_glue_addnorm32", &.{ "a", "b", "w" }, &.{ "h", "n", "xf" }, texts.addnorm32);
        k.router1 = try makeKernel("mimo_glue_router1", &.{ "a", "b", "w", "gate" }, &.{ "h", "n", "logits" }, texts.router1);
        k.top8 = try makeKernel("mimo_glue_top8", &.{ "logits", "bias" }, &.{ "ids", "scores" }, texts.top8);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.norm, k.addnorm, k.addnorm32, k.router1, k.top8 }) |x| if (x.ctx != null) {
            _ = mtl.mlx_fast_metal_kernel_free(x);
        };
        k.* = .{};
    }
};

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // Inputs are the trunk's fresh row-major arrays; a view would be copied first (ensure_row_contiguous).
    const k = mtl.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, texts.header, true, false);
    if (k.ctx == null) return error.MimoGlueKernel;
    return k;
}

// ---- X19 (mimo/router-bf16): the router's product for M rows from the BF16 weight ----

/// The router's product for 1 <= M <= `max_rows` rows (`mimo_glue_router_rows`): logits [M, 256] F32 of x [M, 4096] BF16
/// against the router's BF16 weight [256, 4096], every row in router1's accumulation (MLX gemv bm1 / bn8's), so a row's
/// logits are its M = 1 logits bit for bit whatever M is. Installed once (settings `predict_router: bf16`, the lookahead
/// predictor); one launch config per M, built here.
pub const RouterRows = struct {
    kernel: Kernel = .{ .ctx = null },
    configs: [rows_cap]mlx.mlx_fast_metal_kernel_config = @splat(.{ .ctx = null }),

    /// Construction: the text is written for hidden 4096 and 256 experts (refused otherwise).
    pub fn init(hidden_size: u64, n_experts: u64) !RouterRows {
        if (hidden_size != hidden or n_experts != experts) return error.MimoGlueGeometry;
        var r: RouterRows = .{};
        errdefer r.deinit();
        r.kernel = try makeKernel("mimo_glue_router_rows", &.{ "x", "gate" }, &.{"logits"}, texts.router_rows);
        for (&r.configs, 1..) |*cfg, m| {
            cfg.* = mlx.mlx_fast_metal_kernel_config_new();
            const shape = [_]c_int{ @intCast(m), experts };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, &shape, shape.len, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg.*, 256, router1_groups, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg.*, 256, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg.*, "T", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "M", @intCast(m)));
        }
        return r;
    }

    pub fn deinit(r: *RouterRows) void {
        for (r.configs) |cfg| if (cfg.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        if (r.kernel.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(r.kernel);
        r.* = .{};
    }

    /// logits [M, 256] F32 (owned by the caller); x [M, 4096] BF16 with M <= `max_rows` (the decode lane's widths).
    pub fn apply(r: *const RouterRows, x: Array, gate: Array, s: Stream) !Array {
        const m: usize = @intCast(rowsOf(x));
        std.debug.assert(m >= 1 and m <= rows_cap);
        const in_vec = mlx.mlx_vector_array_new_data(&[_]Array{ x, gate }, 2);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var res = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(res);
        try mlx.check(mtl.mlx_fast_metal_kernel_apply(&res, r.kernel, in_vec, r.configs[m - 1], s));
        var out: Array = .{};
        try mlx.check(mlx.mlx_vector_array_get(&out, res, 0));
        return out;
    }
};

const Out = struct { shape: []const c_int, dtype: mlx.mlx_dtype };

/// One launch: 256-thread threadgroups, `groups` of them along y; `typed` adds the BF16 template T.
fn apply(k: Kernel, ins: []const Array, outs_spec: []const Out, groups: c_int, typed: bool, s: Stream, outs: []Array) !void {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (outs_spec) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, o.shape.ptr, o.shape.len, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 256, groups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    if (typed) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    const in_vec = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&res, k, in_vec, c, s));
    var got: usize = 0;
    errdefer for (outs[0..got]) |o| drop(o);
    for (outs) |*o| {
        o.* = .{};
        try mlx.check(mlx.mlx_vector_array_get(o, res, got));
        got += 1;
    }
}

fn rowsOf(x: Array) c_int {
    return mlx.getShape(x)[0];
}

/// Layer 0's input norm: `rmsNorm(x, w)` (module.rmsNorm: unit-weight fast rms_norm, then the weight's multiply).
pub fn norm(k: *const Kernels, x: Array, w: Array, s: Stream) !Array {
    const m = rowsOf(x);
    var out: [1]Array = undefined;
    try apply(k.norm, &.{ x, w }, &.{.{ .shape = &.{ m, hidden }, .dtype = .bfloat16 }}, m, true, s, &out);
    return out[0];
}

/// A residual add and the next norm: `h = a + b`, `n = rmsNorm(h, w)`; with `f32`, also the router's F32 cast of n.
pub const AddNorm = struct {
    h: Array,
    n: Array,
    /// `.f32` / `.router`: the F32 cast of n (M > 1) or the router's logits [1, 256] (M = 1); else empty.
    x: Array = .{},

    pub fn deinit(p: *AddNorm) void {
        drop(p.h);
        drop(p.n);
        drop(p.x);
        p.* = .{ .h = .{}, .n = .{} };
    }
};

pub fn addNorm(k: *const Kernels, a: Array, b: Array, w: Array, s: Stream) !AddNorm {
    const m = rowsOf(a);
    var out: [2]Array = undefined;
    try apply(k.addnorm, &.{ a, b, w }, &.{ .{ .shape = &.{ m, hidden }, .dtype = .bfloat16 }, .{ .shape = &.{ m, hidden }, .dtype = .bfloat16 } }, m, true, s, &out);
    return .{ .h = out[0], .n = out[1] };
}

pub fn addNormF32(k: *const Kernels, a: Array, b: Array, w: Array, s: Stream) !AddNorm {
    const m = rowsOf(a);
    var out: [3]Array = undefined;
    try apply(k.addnorm32, &.{ a, b, w }, &.{ .{ .shape = &.{ m, hidden }, .dtype = .bfloat16 }, .{ .shape = &.{ m, hidden }, .dtype = .bfloat16 }, .{ .shape = &.{ m, hidden }, .dtype = .float32 } }, m, true, s, &out);
    return .{ .h = out[0], .n = out[1], .x = out[2] };
}

/// M = 1: the residual add, the post-attention norm and the router logits (`gate`: the router's BF16 [256, 4096]).
pub fn router1(k: *const Kernels, a: Array, b: Array, w: Array, gate: Array, s: Stream) !AddNorm {
    var out: [3]Array = undefined;
    try apply(k.router1, &.{ a, b, w, gate }, &.{ .{ .shape = &.{ 1, hidden }, .dtype = .bfloat16 }, .{ .shape = &.{ 1, hidden }, .dtype = .bfloat16 }, .{ .shape = &.{ 1, experts }, .dtype = .float32 } }, router1_groups, true, s, &out);
    return .{ .h = out[0], .n = out[1], .x = out[2] };
}

pub const Top8 = struct {
    ids: Array,
    scores: Array,
    pub fn deinit(t: *Top8) void {
        drop(t.ids);
        drop(t.scores);
    }
};

/// module.route's tail after its product: ids [M, 8] U32 and normalised weights [M, 8] F32.
pub fn top8(k: *const Kernels, logits: Array, bias: Array, s: Stream) !Top8 {
    const m = rowsOf(logits);
    var out: [2]Array = undefined;
    try apply(k.top8, &.{ logits, bias }, &.{ .{ .shape = &.{ m, top_k }, .dtype = .uint32 }, .{ .shape = &.{ m, top_k }, .dtype = .float32 } }, m, false, s, &out);
    return .{ .ids = out[0], .scores = out[1] };
}

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

// ---- host models (CPU proofs) ----
//
// Each glue kernel transliterated (its loops, indices and expressions), and each MLX kernel it replaces transliterated
// from MLX 64ea011cb's source (rms_single_row, GEMVKernel, the block merge sort, row_reduce_small, the unary Sigmoid).
// The hardware primitives are one shared model each (fma for the contracted `acc += a * b`, a fixed simd_sum tree, F32
// division and sqrt): both sides call the same one on the same lane values, so equality proves the kernels' orders and
// index maps, never the primitives (the device self-check proves those).
pub const model = struct {
    pub fn bf16Round(x: f32) f32 {
        const w: u32 = @bitCast(x);
        if (std.math.isNan(x)) return @bitCast((w | 0x0040_0000) & 0xffff_0000);
        return @bitCast((w +% 0x7fff +% ((w >> 16) & 1)) & 0xffff_0000);
    }
    pub fn bf16Bits(x: f32) u16 {
        return @intCast(@as(u32, @bitCast(bf16Round(x))) >> 16);
    }
    pub fn fromBits(b: u16) f32 {
        return @bitCast(@as(u32, b) << 16);
    }
    /// The 32-lane hardware reduction (simd_sum): a fixed halving tree.
    pub fn simdSum(v: [32]f32) f32 {
        var a = v;
        var o: usize = 16;
        while (o >= 1) : (o /= 2) {
            for (0..o) |l| a[l] = a[l] + a[l + o];
        }
        return a[0];
    }
    /// `r += simd_shuffle_down(r, o)` for o = width / 2 .. 1 on every lane at once; the value at `lane`.
    pub fn shuffleTree(v: [32]f32, width: usize, lane: usize) f32 {
        var a = v;
        var o = width / 2;
        while (o >= 1) : (o /= 2) {
            const old = a;
            for (0..32) |l| a[l] = old[l] + (if (l + o < 32) old[l + o] else old[l]);
        }
        return a[lane];
    }
    pub fn rsqrt(x: f32) f32 {
        return 1.0 / @sqrt(x);
    }
    /// MLX's Sigmoid (unary_ops.h): `y = 1 / (1 + exp(|x|))`, `x < 0 ? y : 1 - y`.
    pub fn sigmoid(x: f32) f32 {
        const y = 1.0 / (1.0 + @exp(@abs(x)));
        return if (x < 0) y else 1.0 - y;
    }
    /// sort.h LessThan: NaN after every number.
    pub fn lessThan(a: f32, b: f32) bool {
        const an = std.math.isNan(a);
        const bn = std.math.isNan(b);
        if (an or bn) return !an and bn;
        return a < b;
    }
};

const D: usize = 4096;
const E: usize = 256;

/// MLX rms_single_row at D = 4096 (1024 threads x 4) then the weight's multiply: out = bf16(bf16(x * inv) * w).
pub fn mlxRms(x: []const f32, w: []const f32, out: []f32) f32 {
    var sums: [32]f32 = undefined;
    for (0..32) |sg| {
        var lanes: [32]f32 = undefined;
        for (0..32) |l| {
            const t = sg * 32 + l;
            var acc: f32 = 0;
            for (0..4) |i| acc = @mulAdd(f32, x[t * 4 + i], x[t * 4 + i], acc);
            lanes[l] = acc;
        }
        sums[sg] = model.simdSum(lanes);
    }
    const total = model.simdSum(sums);
    const inv = model.rsqrt(total / 4096.0 + 1e-6);
    for (0..D) |c| out[c] = model.bf16Round(model.bf16Round(x[c] * inv) * w[c]);
    return inv;
}

/// header_mimo_glue.metal `glue_row_inv` + `glue_normed`, one 256-thread threadgroup in four rounds; `b` = the add.
pub fn fusedRms(a: []const f32, b: ?[]const f32, w: []const f32, h: []f32, out: []f32) f32 {
    var sums: [32]f32 = undefined;
    for (0..8) |sg| for (0..4) |r| {
        var lanes: [32]f32 = undefined;
        for (0..32) |lane| {
            const c0 = 1024 * r + 4 * (32 * sg + lane);
            var acc: f32 = 0;
            for (0..4) |i| {
                const v = if (b) |bb| model.bf16Round(a[c0 + i] + bb[c0 + i]) else a[c0 + i];
                h[c0 + i] = v;
                acc = @mulAdd(f32, v, v, acc);
            }
            lanes[lane] = acc;
        }
        sums[8 * r + sg] = model.simdSum(lanes);
    };
    const total = model.simdSum(sums);
    const inv = model.rsqrt(total / 4096.0 + 1e-6);
    for (0..D) |c| out[c] = model.bf16Round(model.bf16Round(h[c] * inv) * w[c]);
    return inv;
}

/// MLX gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4 (GEMVKernel::run) for the router: x [4096] against W [256, 4096].
pub fn mlxGemvRouter(W: []const f32, x: []const f32, out: []f32) void {
    for (0..64) |tg| {
        var part: [8][4]f32 = undefined;
        for (0..8) |sg_n| {
            var lanes: [4][32]f32 = undefined;
            for (0..32) |l| {
                var result = [4]f32{ 0, 0, 0, 0 };
                var bn = (32 * sg_n + l) * 4;
                for (0..4) |_| {
                    for (0..4) |tm| for (0..4) |tn| {
                        result[tm] = @mulAdd(f32, W[(4 * tg + tm) * D + bn + tn], x[bn + tn], result[tm]);
                    };
                    bn += 1024;
                }
                for (0..4) |tm| lanes[tm][l] = result[tm];
            }
            for (0..4) |tm| part[sg_n][tm] = model.shuffleTree(lanes[tm], 32, 0);
        }
        for (0..4) |tm| {
            var r = part[0][tm];
            for (1..8) |s2| r += part[s2][tm];
            out[4 * tg + tm] = r;
        }
    }
}

/// mimo_glue_router1.metal's product (threadgroup g, simdgroup sg, lane; rounds r).
pub fn fusedRouter1(W: []const f32, xv: []const f32, out: []f32) void {
    for (0..64) |g| {
        var part: [8][4]f32 = undefined;
        for (0..8) |sg| {
            var lanes: [4][32]f32 = undefined;
            for (0..32) |lane| {
                var result = [4]f32{ 0, 0, 0, 0 };
                for (0..4) |r| {
                    const bn = 1024 * r + 4 * (32 * sg + lane);
                    for (0..4) |tm| {
                        var inter: [4]f32 = undefined;
                        for (0..4) |tn| inter[tn] = W[(4 * g + tm) * D + bn + tn];
                        for (0..4) |tn| result[tm] = @mulAdd(f32, inter[tn], xv[bn + tn], result[tm]);
                    }
                }
                for (0..4) |tm| lanes[tm][lane] = result[tm];
            }
            for (0..4) |tm| part[sg][tm] = model.shuffleTree(lanes[tm], 32, 0);
        }
        for (0..4) |tm| {
            var r = part[0][tm];
            for (1..8) |s2| r += part[s2][tm];
            out[4 * g + tm] = r;
        }
    }
}

/// mimo_glue_router_rows.metal's product for `m` rows (xs row-major [m, 4096]; out [m, 256]): threadgroup g, simdgroup
/// sg, lane; rounds r; the rows' accumulators interleaved per weight load as the text runs them.
pub fn fusedRouterRows(W: []const f32, xs: []const f32, m: usize, out: []f32) void {
    std.debug.assert(m >= 1 and m <= rows_cap and xs.len == m * D and out.len == m * E);
    for (0..64) |g| {
        var part: [rows_cap][8][4]f32 = undefined;
        var fold: [rows_cap][4]f32 = undefined;
        for (0..8) |sg| {
            var lanes: [rows_cap][4][32]f32 = undefined;
            for (0..32) |lane| {
                var xv: [rows_cap][16]f32 = undefined;
                for (0..m) |row| for (0..4) |r| for (0..4) |i| {
                    xv[row][4 * r + i] = xs[row * D + 1024 * r + 4 * (32 * sg + lane) + i];
                };
                var result: [rows_cap][4]f32 = @splat(@splat(0));
                for (0..4) |r| {
                    const bn = 1024 * r + 4 * (32 * sg + lane);
                    for (0..4) |tm| {
                        var inter: [4]f32 = undefined;
                        for (0..4) |tn| inter[tn] = W[(4 * g + tm) * D + bn + tn];
                        for (0..m) |row| for (0..4) |tn| {
                            result[row][tm] = @mulAdd(f32, inter[tn], xv[row][4 * r + tn], result[row][tm]);
                        };
                    }
                }
                for (0..m) |row| for (0..4) |tm| {
                    lanes[row][tm][lane] = result[row][tm];
                };
            }
            for (0..m) |row| for (0..4) |tm| {
                part[row][sg][tm] = model.shuffleTree(lanes[row][tm], 32, 0);
            };
        }
        for (0..m) |row| {
            for (0..4) |tm| fold[row][tm] = part[row][0][tm];
            for (1..8) |s2| for (0..4) |tm| {
                fold[row][tm] += part[row][s2][tm];
            };
            for (0..4) |tm| out[row * E + 4 * g + tm] = fold[row][tm];
        }
    }
}

/// MLX gemv_gather_float32_bm4_bn1_sm1_sn32_tm4_tn4 (GEMVKernel::run, BN = 1: no threadgroup fold) for one row against the
/// router: the lookahead predictor's M > 1 rows on the control (`Math.predictorRows`: gather_mm over one-row batches routes
/// to gather_mv; the transposed F32 view is the [256, 4096] matrix). Threadgroup t, simdgroup s: experts 16 t + 4 s .. + 3;
/// lane l: columns 128 i + 4 l .. + 3 for i = 0 .. 31, then the halving shuffles.
pub fn mlxGemvGatherRouter(W: []const f32, x: []const f32, out: []f32) void {
    for (0..16) |tg| for (0..4) |sg| {
        var lanes: [4][32]f32 = undefined;
        for (0..32) |l| {
            var result = [4]f32{ 0, 0, 0, 0 };
            var bn = 4 * l;
            for (0..32) |_| {
                for (0..4) |tm| for (0..4) |tn| {
                    result[tm] = @mulAdd(f32, W[(16 * tg + 4 * sg + tm) * D + bn + tn], x[bn + tn], result[tm]);
                };
                bn += 128;
            }
            for (0..4) |tm| lanes[tm][l] = result[tm];
        }
        for (0..4) |tm| out[16 * tg + 4 * sg + tm] = model.shuffleTree(lanes[tm], 32, 0);
    };
}

/// MLX's ArgPartition on the GPU: its block merge sort (sort.h BlockMergeSort, bn 64 x tn 4 for 256 keys), argsort
/// ascending with LessThan; returns the key indices in sorted order.
pub fn mlxArgsort256(keys: [256]f32) [256]u32 {
    const bn = 64;
    const tn = 4;
    var vals = keys;
    var idxs: [256]u32 = undefined;
    for (&idxs, 0..) |*x, i| x.* = @intCast(i);
    // Thread sort: odd-even transposition over each thread's 4 keys.
    for (0..bn) |t| {
        const v = vals[t * tn ..][0..tn];
        const ix = idxs[t * tn ..][0..tn];
        for (0..tn) |i| {
            var j: usize = i & 1;
            while (j < tn - 1) : (j += 2) {
                if (model.lessThan(v[j + 1], v[j])) {
                    std.mem.swap(f32, &v[j + 1], &v[j]);
                    std.mem.swap(u32, &ix[j + 1], &ix[j]);
                }
            }
        }
    }
    var merge_threads: usize = 2;
    while (merge_threads <= bn) : (merge_threads *= 2) {
        const tv = vals;
        const ti = idxs;
        for (0..bn) |lid| {
            const merge_group = lid / merge_threads;
            const merge_lane = lid % merge_threads;
            const sort_sz = tn * merge_threads;
            const sort_st = tn * merge_threads * merge_group;
            const a_st = sort_st;
            const b_st = sort_st + sort_sz / 2;
            var a_sz: usize = sort_sz / 2;
            var b_sz: usize = sort_sz / 2;
            const sort_md = tn * merge_lane;
            // merge_partition
            var lo: usize = if (sort_md > b_sz) sort_md - b_sz else 0;
            var hi: usize = @min(sort_md, a_sz);
            while (lo < hi) {
                const md = lo + (hi - lo) / 2;
                if (model.lessThan(tv[b_st + sort_md - 1 - md], tv[a_st + md])) hi = md else lo = md + 1;
            }
            const partition = hi;
            const as_ = a_st + partition;
            const bs_ = b_st + sort_md - partition;
            a_sz -= partition;
            b_sz -= sort_md - partition;
            // merge_step
            var a_i: usize = 0;
            var b_i: usize = 0;
            for (0..tn) |i| {
                const a = if (a_i < a_sz) tv[as_ + a_i] else std.math.nan(f32);
                const b = if (b_i < b_sz) tv[bs_ + b_i] else std.math.nan(f32);
                const pred = (b_i < b_sz) and (a_i >= a_sz or model.lessThan(b, a));
                vals[lid * tn + i] = if (pred) b else a;
                idxs[lid * tn + i] = if (pred) ti[bs_ + b_i] else (if (a_i < a_sz) ti[as_ + a_i] else 0);
                b_i += @intFromBool(pred);
                a_i += @intFromBool(!pred);
            }
        }
    }
    return idxs;
}

/// module.route's tail on the GPU after its product: Sigmoid, + correction, negative, argpartition (the merge sort),
/// the first 8, take_along_axis, row_reduce_small's sum, + 1e-20, divide.
pub fn mlxRouteTail(logits: []const f32, bias: []const f32, ids: *[8]u32, scores: *[8]f32) void {
    var s: [256]f32 = undefined;
    var neg: [256]f32 = undefined;
    for (0..E) |e| {
        s[e] = model.sigmoid(logits[e]);
        neg[e] = -(s[e] + bias[e]);
    }
    const order = mlxArgsort256(neg);
    var picked: [8]f32 = undefined;
    for (0..8) |j| {
        ids[j] = order[j];
        picked[j] = s[order[j]];
    }
    var total: f32 = 0;
    for (picked) |p| total = p + total;
    const den = total + 1e-20;
    for (0..8) |j| scores[j] = picked[j] / den;
}

/// mimo_glue_top8.metal: the rank of each key (strictly less + equal at a lower index) places it.
pub fn fusedTop8(logits: []const f32, bias: []const f32, ids: *[8]u32, scores: *[8]f32) void {
    var s: [256]f32 = undefined;
    var vs: [256]f32 = undefined;
    for (0..E) |e| {
        const x = logits[e];
        const y = 1.0 / (1.0 + @exp(@abs(x)));
        s[e] = if (x < 0) y else 1.0 - y;
        vs[e] = -(s[e] + bias[e]);
    }
    var picked: [8]f32 = undefined;
    for (0..E) |e| {
        const v = vs[e];
        const vn = std.math.isNan(v);
        var rank: usize = 0;
        for (0..E) |j| {
            const u = vs[j];
            const un = std.math.isNan(u);
            const u_lt_v = if (un or vn) (!un and vn) else (u < v);
            const v_lt_u = if (un or vn) (!vn and un) else (v < u);
            rank += @intFromBool(u_lt_v or (j < e and !v_lt_u));
        }
        if (rank < 8) {
            ids[rank] = @intCast(e);
            picked[rank] = s[e];
        }
    }
    var total: f32 = 0;
    for (picked) |p| total = p + total;
    const den = total + 1e-20;
    for (0..8) |j| scores[j] = picked[j] / den;
}

// ---- CPU tests ----

const testing = std.testing;

const Rng = struct {
    st: u32,
    fn next(r: *Rng) u32 {
        r.st ^= r.st << 13;
        r.st ^= r.st >> 17;
        r.st ^= r.st << 5;
        return r.st;
    }
    /// Uniform in [-amp, amp), rounded to BF16.
    fn bf(r: *Rng, amp: f32) f32 {
        const u = @as(f32, @floatFromInt(r.next() >> 8)) / @as(f32, 1 << 24);
        return model.bf16Round((2 * u - 1) * amp);
    }
};

fn fillBf(r: *Rng, out: []f32, amp: f32) void {
    for (out) |*x| x.* = r.bf(amp);
}

test "mimo glue CPU: the manifest and every K2 text match their pins" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 6);
    try requireGeometry(.{});
    var odd: mimo.Config = .{};
    odd.rms_norm_eps = 1e-5;
    try testing.expectError(error.MimoGlueGeometry, requireGeometry(odd));
}

test "mimo glue CPU: launch samples: 256-thread groups, one per row (64 for the M = 1 router), output shapes by M" {
    const J = struct {
        kernels: []const struct {
            name: []const u8,
            family: []const u8,
            launch_samples: []const struct { grid: [3]c_int, threadgroup: [3]c_int, vars: []const struct { name: []const u8, value: c_int }, output_shapes: []const []const c_int },
        },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, texts.manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    var checked: usize = 0;
    for (m.value.kernels) |k| {
        if (!std.mem.eql(u8, k.family, "glue")) continue;
        for (k.launch_samples) |sample| {
            var rows: c_int = 1;
            for (sample.vars) |v| if (std.mem.eql(u8, v.name, "M")) {
                rows = v.value;
            };
            try testing.expectEqual([3]c_int{ 256, 1, 1 }, sample.threadgroup);
            const groups: c_int = if (std.mem.eql(u8, k.name, "mimo_glue_router1") or std.mem.eql(u8, k.name, "mimo_glue_router_rows")) router1_groups else rows;
            try testing.expectEqual([3]c_int{ 256, groups, 1 }, sample.grid);
            try testing.expectEqual(rows, sample.output_shapes[0][0]);
            checked += 1;
        }
    }
    try testing.expectEqual(@as(usize, 4 * 5 + 1 + 5), checked);
}

test "mimo glue CPU: the 256-thread norm (4 rounds) equals MLX rms_single_row's 1024 threads bit for bit, with and without the residual add, M = 1..8 rows" {
    const a = testing.allocator;
    var r: Rng = .{ .st = 0x9e3779b9 };
    const x = try a.alloc(f32, D);
    defer a.free(x);
    const y = try a.alloc(f32, D);
    defer a.free(y);
    const w = try a.alloc(f32, D);
    defer a.free(w);
    const h = try a.alloc(f32, D);
    defer a.free(h);
    const sum = try a.alloc(f32, D);
    defer a.free(sum);
    const want = try a.alloc(f32, D);
    defer a.free(want);
    const got = try a.alloc(f32, D);
    defer a.free(got);
    for (0..8) |row| {
        // rows of growing magnitude (the residual stream grows with depth), weights near 1
        const amp: f32 = std.math.pow(f32, 4, @floatFromInt(row)) / 16;
        fillBf(&r, x, amp);
        fillBf(&r, y, amp);
        for (w) |*v| v.* = model.bf16Round(1 + r.bf(0.5));
        // no add (layer 0's input norm)
        const inv0 = mlxRms(x, w, want);
        const inv1 = fusedRms(x, null, w, h, got);
        try testing.expectEqual(@as(u32, @bitCast(inv0)), @as(u32, @bitCast(inv1)));
        try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(got));
        // with the add: MLX's binary Add (BF16), then the norm
        for (0..D) |c| sum[c] = model.bf16Round(x[c] + y[c]);
        const inv2 = mlxRms(sum, w, want);
        const inv3 = fusedRms(x, y, w, h, got);
        try testing.expectEqual(@as(u32, @bitCast(inv2)), @as(u32, @bitCast(inv3)));
        try testing.expectEqualSlices(u32, @ptrCast(sum), @ptrCast(h));
        try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(got));
    }
}

test "mimo glue CPU: the M = 1 router product (router1) equals MLX gemv bm1/bn8's accumulation bit for bit on the real [256, 4096] shape" {
    const a = testing.allocator;
    var r: Rng = .{ .st = 0x1234567 };
    const W = try a.alloc(f32, E * D);
    defer a.free(W);
    fillBf(&r, W, 0.05);
    const x = try a.alloc(f32, D);
    defer a.free(x);
    var want: [256]f32 = undefined;
    var got: [256]f32 = undefined;
    for (0..4) |trial| {
        fillBf(&r, x, @as(f32, @floatFromInt(trial + 1)));
        mlxGemvRouter(W, x, &want);
        fusedRouter1(W, x, &got);
        try testing.expectEqualSlices(u32, @ptrCast(&want), @ptrCast(&got));
        // and the product is the [4096] dot within F32 accumulation error (the model computes the router's function)
        for (0..E) |e| {
            var t: f64 = 0;
            for (0..D) |c| t += @as(f64, W[e * D + c]) * x[c];
            try testing.expect(@abs(t - want[e]) <= 1e-4 * (1 + @abs(t)));
        }
    }
}

test "mimo glue CPU X19: router_rows gives every row its M = 1 product (router1 = MLX gemv bm1/bn8) bit for bit at M = 1..8, wherever the row sits" {
    const a = testing.allocator;
    var r: Rng = .{ .st = 0x7a19_0001 };
    const W = try a.alloc(f32, E * D);
    defer a.free(W);
    fillBf(&r, W, 0.05);
    const xs = try a.alloc(f32, rows_cap * D);
    defer a.free(xs);
    // Rows of different magnitudes (the normed rows times the post-norm weight span ~0.1 .. ~10).
    for (0..rows_cap) |row| fillBf(&r, xs[row * D ..][0..D], @as(f32, @floatFromInt(row + 1)) * 0.75);
    var one: [rows_cap][E]f32 = undefined;
    var gemv: [E]f32 = undefined;
    for (0..rows_cap) |row| {
        fusedRouter1(W, xs[row * D ..][0..D], &one[row]);
        mlxGemvRouter(W, xs[row * D ..][0..D], &gemv);
        try testing.expectEqualSlices(u32, @ptrCast(&gemv), @ptrCast(&one[row]));
    }
    var out: [rows_cap * E]f32 = undefined;
    for (1..rows_cap + 1) |m| {
        fusedRouterRows(W, xs[0 .. m * D], m, out[0 .. m * E]);
        for (0..m) |row| try testing.expectEqualSlices(u32, @ptrCast(&one[row]), @ptrCast(out[row * E ..][0..E]));
    }
    // The same row beside different companions (rows moved, the others replaced): its logits do not move.
    const swapped = try a.alloc(f32, 4 * D);
    defer a.free(swapped);
    fillBf(&r, swapped, 3);
    @memcpy(swapped[2 * D ..][0..D], xs[0..D]);
    fusedRouterRows(W, swapped, 4, out[0 .. 4 * E]);
    try testing.expectEqualSlices(u32, @ptrCast(&one[0]), @ptrCast(out[2 * E ..][0..E]));
    // The ids and weights follow (the top-8 kernel's model on bit-equal logits, and MLX's stock tail).
    var bias: [256]f32 = undefined;
    for (&bias) |*v| v.* = r.bf(0.01);
    var ids_a: [8]u32 = undefined;
    var ids_b: [8]u32 = undefined;
    var sc_a: [8]f32 = undefined;
    var sc_b: [8]f32 = undefined;
    fusedRouterRows(W, xs[0 .. 4 * D], 4, out[0 .. 4 * E]);
    for (0..4) |row| {
        fusedTop8(&one[row], &bias, &ids_a, &sc_a);
        mlxRouteTail(out[row * E ..][0..E], &bias, &ids_b, &sc_b);
        try testing.expectEqualSlices(u32, &ids_a, &ids_b);
        try testing.expectEqualSlices(u32, @ptrCast(&sc_a), @ptrCast(&sc_b));
    }
}

test "mimo glue CPU X19: the control predictor's M > 1 order (MLX gemv_gather bm4) is a different F32 order: same function within F32 rounding, not bit-equal" {
    const a = testing.allocator;
    var r: Rng = .{ .st = 0x7a19_0002 };
    const W = try a.alloc(f32, E * D);
    defer a.free(W);
    fillBf(&r, W, 0.05);
    const x = try a.alloc(f32, D);
    defer a.free(x);
    var one: [E]f32 = undefined;
    var gather: [E]f32 = undefined;
    var differ: usize = 0;
    for (0..4) |trial| {
        fillBf(&r, x, @as(f32, @floatFromInt(trial + 1)));
        fusedRouter1(W, x, &one);
        mlxGemvGatherRouter(W, x, &gather);
        for (0..E) |e| {
            if (@as(u32, @bitCast(one[e])) != @as(u32, @bitCast(gather[e]))) differ += 1;
            var t: f64 = 0;
            for (0..D) |c| t += @as(f64, W[e * D + c]) * x[c];
            try testing.expect(@abs(t - gather[e]) <= 1e-4 * (1 + @abs(t)));
        }
    }
    try testing.expect(differ > 0);
}

// The release's routers (opt-in, CPU, no MLX): MIMO_X19_PACK=<pack> (fp8res or bf16res: the routers, their corrections and
// the post-attention norms are BF16 / F32 as released in both).
test "mimo glue CPU X19 pack: on the release's 47 routers the M = 2..4 rows of router_rows are each row's M = 1 logits bit for bit and the top-8 is the M = 1 top-8; the control predictor's order differs by at most a few F32 ulps (MIMO_X19_PACK)" {
    const dir = std.mem.span(std.c.getenv("MIMO_X19_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const read = @import("fp8_linear.zig").readTensor;
    var r: Rng = .{ .st = 0x7a19_0003 };
    const W = try a.alloc(f32, E * D);
    defer a.free(W);
    const xa = try a.alloc(f32, 4 * D);
    defer a.free(xa);
    const xb = try a.alloc(f32, 4 * D);
    defer a.free(xb);
    const wn = try a.alloc(f32, D);
    defer a.free(wn);
    const h = try a.alloc(f32, D);
    defer a.free(h);
    const xs = try a.alloc(f32, 4 * D);
    defer a.free(xs);
    var name: [128]u8 = undefined;
    var rows_checked: usize = 0;
    var top8_rows: usize = 0;
    var logit_max: f32 = 0;
    var score_max: f32 = 0;
    var differ: usize = 0;
    var gather_top8_diff: usize = 0;
    var layer: usize = 1;
    while (layer < 48) : (layer += 1) {
        const wraw = try read(a, dir, "model.safetensors", try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.gate.weight", .{layer}), "BF16", &.{ 256, 4096 });
        defer a.free(wraw);
        const braw = try read(a, dir, "model.safetensors", try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{layer}), "F32", &.{256});
        defer a.free(braw);
        const nraw = try read(a, dir, "model.safetensors", try std.fmt.bufPrint(&name, "model.layers.{d}.post_attention_layernorm.weight", .{layer}), "BF16", &.{4096});
        defer a.free(nraw);
        for (std.mem.bytesAsSlice(u16, @as([]align(1) u8, wraw)), W) |b, *w| w.* = model.fromBits(b);
        for (std.mem.bytesAsSlice(u16, @as([]align(1) u8, nraw)), wn) |b, *w| w.* = model.fromBits(b);
        var bias: [256]f32 = undefined;
        for (std.mem.bytesAsSlice(f32, @as([]align(1) u8, braw)), &bias) |b, *v| v.* = b;
        // Four verify rows as the trunk makes them: the residual plus the attention output through the layer's own
        // post-attention norm (the glue's model of it), residuals of growing magnitude.
        fillBf(&r, xa, 4);
        fillBf(&r, xb, 1);
        for (0..4) |row| _ = fusedRms(xa[row * D ..][0..D], xb[row * D ..][0..D], wn, h, xs[row * D ..][0..D]);
        var one: [4][E]f32 = undefined;
        var ref: [E]f32 = undefined;
        for (0..4) |row| {
            fusedRouter1(W, xs[row * D ..][0..D], &one[row]);
            mlxGemvRouter(W, xs[row * D ..][0..D], &ref);
            try testing.expectEqualSlices(u32, @ptrCast(&ref), @ptrCast(&one[row]));
        }
        var out: [4 * E]f32 = undefined;
        for (2..5) |m| {
            fusedRouterRows(W, xs[0 .. m * D], m, out[0 .. m * E]);
            for (0..m) |row| {
                try testing.expectEqualSlices(u32, @ptrCast(&one[row]), @ptrCast(out[row * E ..][0..E]));
                var ids_m: [8]u32 = undefined;
                var ids_1: [8]u32 = undefined;
                var sc_m: [8]f32 = undefined;
                var sc_1: [8]f32 = undefined;
                fusedTop8(out[row * E ..][0..E], &bias, &ids_m, &sc_m);
                mlxRouteTail(&one[row], &bias, &ids_1, &sc_1);
                try testing.expectEqualSlices(u32, &ids_1, &ids_m);
                try testing.expectEqualSlices(u32, @ptrCast(&sc_1), @ptrCast(&sc_m));
                top8_rows += 1;
            }
            rows_checked += m;
        }
        // The control predictor's M > 1 order against the M = 1 order on the same rows (the lookahead's hint only).
        for (0..4) |row| {
            var gather: [E]f32 = undefined;
            mlxGemvGatherRouter(W, xs[row * D ..][0..D], &gather);
            for (0..E) |e| {
                if (@as(u32, @bitCast(gather[e])) != @as(u32, @bitCast(one[row][e]))) differ += 1;
                logit_max = @max(logit_max, @abs(gather[e] - one[row][e]));
                score_max = @max(score_max, @abs(model.sigmoid(gather[e]) - model.sigmoid(one[row][e])));
            }
            var ids_g: [8]u32 = undefined;
            var ids_1: [8]u32 = undefined;
            var sc: [8]f32 = undefined;
            fusedTop8(&gather, &bias, &ids_g, &sc);
            fusedTop8(&one[row], &bias, &ids_1, &sc);
            if (!std.mem.eql(u32, &ids_g, &ids_1)) gather_top8_diff += 1;
        }
    }
    std.debug.print("MIMO_X19_PACK layers=47 rows_m2_m4={d} bit_equal=yes top8_rows={d} top8_equal=yes control_predictor_vs_m1: logits_differ={d}/{d} logit_absdiff_max={e} score_absdiff_max={e} top8_rows_differ={d}/{d}\n", .{ rows_checked, top8_rows, differ, 47 * 4 * E, logit_max, score_max, gather_top8_diff, 47 * 4 });
    try testing.expectEqual(@as(usize, 47 * 9), rows_checked);
}

test "mimo glue CPU: top-8 by rank equals MLX's stable block merge sort (ids in order, weights bit for bit) on random and tie fixtures" {
    var r: Rng = .{ .st = 0xdeadbeef };
    var logits: [256]f32 = undefined;
    var bias: [256]f32 = undefined;
    var ids_a: [8]u32 = undefined;
    var ids_b: [8]u32 = undefined;
    var sc_a: [8]f32 = undefined;
    var sc_b: [8]f32 = undefined;
    var cases: usize = 0;
    var tied: usize = 0;
    for (0..400) |t| {
        const kind = t % 8;
        for (0..E) |e| {
            logits[e] = switch (kind) {
                // random logits, small corrections
                0, 1 => r.bf(4),
                // heavy ties: 5 distinct logits
                2 => @floatFromInt(@as(i32, @intCast(r.next() % 5)) - 2),
                // all equal: the ids are 0..7 in order
                3 => 0.5,
                // ties straddling the 8th place: 12 equal top keys, the rest lower
                4 => if (e % 21 == 0) 3 else r.bf(1),
                // signed zeros (sigmoid 0.5 both; the correction decides)
                5 => if (e % 2 == 0) 0.0 else -0.0,
                // NaN keys sort last (fewer than 8 numbers in one case)
                6 => if (e % 3 == 0) std.math.nan(f32) else r.bf(2),
                else => if (e < 250) std.math.nan(f32) else r.bf(2),
            };
            bias[e] = switch (kind) {
                0 => r.bf(0.01),
                2, 3, 4 => 0,
                5 => if (e % 4 < 2) 0.25 else 0.0,
                else => r.bf(0.001),
            };
        }
        mlxRouteTail(&logits, &bias, &ids_a, &sc_a);
        fusedTop8(&logits, &bias, &ids_b, &sc_b);
        try testing.expectEqualSlices(u32, &ids_a, &ids_b);
        try testing.expectEqualSlices(u32, @ptrCast(&sc_a), @ptrCast(&sc_b));
        if (kind >= 2) tied += 1;
        cases += 1;
    }
    try testing.expectEqual(@as(usize, 400), cases);
    try testing.expect(tied == 300);
    // all-equal keys: ids 0..7 in order (the lower index wins every tie)
    for (&logits) |*x| x.* = 0.25;
    for (&bias) |*x| x.* = 0;
    fusedTop8(&logits, &bias, &ids_b, &sc_b);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &ids_b);
}

test "mimo glue CPU: the models compute the stock graph: residual add bit for bit, norms within 1 BF16 ulp, ids and weights of MLX CPU's route" {
    try @import("mtp.zig").requireWindow();
    // MLX's CPU backend runs the stock ops with its own reduction orders (the GPU's are the models above), so only the
    // element-wise add is bitwise here; the rest within its rounding, on tie-free real-shaped fixtures.
    const module = @import("module.zig");
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var r: Rng = .{ .st = 0x51ed2701 };
    const rows: usize = 4;
    const n = rows * D;
    const xa = try a.alloc(f32, n);
    defer a.free(xa);
    const xb = try a.alloc(f32, n);
    defer a.free(xb);
    const w = try a.alloc(f32, D);
    defer a.free(w);
    const gate = try a.alloc(f32, E * D);
    defer a.free(gate);
    var bias: [256]f32 = undefined;
    fillBf(&r, xa, 3);
    fillBf(&r, xb, 3);
    for (w) |*v| v.* = model.bf16Round(1 + r.bf(0.5));
    fillBf(&r, gate, 0.05);
    for (&bias) |*v| v.* = r.bf(0.01);
    const B = struct {
        fn arr(al: std.mem.Allocator, v: []const f32, shape: []const c_int) !Array {
            const bits = try al.alloc(u16, v.len);
            defer al.free(bits);
            for (v, bits) |x, *b| b.* = model.bf16Bits(x);
            return mlx.mlx_array_new_data(bits.ptr, shape.ptr, @intCast(shape.len), .bfloat16);
        }
        fn host(al: std.mem.Allocator, x: Array, st: Stream) ![]f32 {
            var f: Array = .{};
            defer drop(f);
            try mlx.check(mlx.mlx_astype(&f, x, .float32, st));
            try mlx.check(mlx.mlx_array_eval(f));
            const len = mlx.mlx_array_size(f);
            const out = try al.alloc(f32, len);
            @memcpy(out, mlx.mlx_array_data_float32(f).?[0..len]);
            return out;
        }
    };
    const ma = try B.arr(a, xa, &.{ @intCast(rows), 4096 });
    defer drop(ma);
    const mb = try B.arr(a, xb, &.{ @intCast(rows), 4096 });
    defer drop(mb);
    const mw = try B.arr(a, w, &.{4096});
    defer drop(mw);
    const mg = try B.arr(a, gate, &.{ 256, 4096 });
    defer drop(mg);
    var ones: Array = .{};
    try mlx.check(mlx.mlx_ones(&ones, &.{4096}, 1, .bfloat16, s));
    defer drop(ones);
    const mbias = mlx.mlx_array_new_data(&bias, &[_]c_int{256}, 1, .float32);
    defer drop(mbias);
    const gf = try module.cast(mg, .float32, s);
    defer drop(gf);
    var gt: Array = .{};
    try mlx.check(mlx.mlx_transpose(&gt, gf, s));
    defer drop(gt);
    const sum = try module.binary(mlx.mlx_add, ma, mb, s);
    defer drop(sum);
    const normed = try module.rmsNorm(sum, mw, ones, 1e-6, s);
    defer drop(normed);
    var routes = try module.route(normed, gt, mbias, 8, s);
    defer routes.deinit();
    const h_cpu = try B.host(a, sum, s);
    defer a.free(h_cpu);
    const n_cpu = try B.host(a, normed, s);
    defer a.free(n_cpu);
    const sc_cpu = try B.host(a, routes.scores, s);
    defer a.free(sc_cpu);
    var ids_f: Array = .{};
    defer drop(ids_f);
    try mlx.check(mlx.mlx_astype(&ids_f, routes.ids, .uint32, s));
    var ids_c: Array = .{};
    defer drop(ids_c);
    try mlx.check(mlx.mlx_contiguous(&ids_c, ids_f, false, s));
    try mlx.check(mlx.mlx_array_eval(ids_c));
    const ids_cpu = mlx.mlx_array_data_uint32(ids_c).?;
    const hh = try a.alloc(f32, D);
    defer a.free(hh);
    const nn = try a.alloc(f32, D);
    defer a.free(nn);
    var logits: [256]f32 = undefined;
    var ids: [8]u32 = undefined;
    var scores: [8]f32 = undefined;
    var ulp_max: u32 = 0;
    var compared: usize = 0;
    for (0..rows) |row| {
        _ = fusedRms(xa[row * D ..][0..D], xb[row * D ..][0..D], w, hh, nn);
        try testing.expectEqualSlices(u32, @ptrCast(h_cpu[row * D ..][0..D]), @ptrCast(hh));
        for (nn, n_cpu[row * D ..][0..D]) |p, q| {
            const d = @abs(@as(i32, model.bf16Bits(p)) - @as(i32, model.bf16Bits(q)));
            ulp_max = @max(ulp_max, @as(u32, @intCast(d)));
        }
        fusedRouter1(gate, nn, &logits);
        fusedTop8(&logits, &bias, &ids, &scores);
        // A row whose 8th and 9th corrected scores sit within the two backends' rounding may select differently: skipped.
        var choice: [256]f32 = undefined;
        for (&choice, logits, bias) |*ch, l, b| ch.* = model.sigmoid(l) + b;
        std.mem.sort(f32, &choice, {}, std.sort.desc(f32));
        if (choice[7] - choice[8] < 1e-4) continue;
        compared += 1;
        // MLX CPU's argpartition leaves the first 8 in its own order: compare as sets, the weights per id.
        for (0..8) |j| {
            const id = ids_cpu[row * 8 + j];
            const at = std.mem.indexOfScalar(u32, &ids, id) orelse return error.TestUnexpectedResult;
            try testing.expectApproxEqRel(sc_cpu[row * 8 + j], scores[at], 1e-5);
        }
    }
    try testing.expect(ulp_max <= 1);
    try testing.expect(compared >= 2);
}

// ---- device self-check (pre-ship family `glue`; env + lock gated) ----

fn hostBits(a: std.mem.Allocator, x: Array, s: Stream) ![]u8 {
    var c: Array = .{};
    defer drop(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const out = try a.alloc(u8, n);
    const p: [*]const u8 = switch (mlx.mlx_array_dtype(c)) {
        .float32 => @ptrCast(mlx.mlx_array_data_float32(c).?),
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(c).?),
        .uint32 => @ptrCast(mlx.mlx_array_data_uint32(c).?),
        else => return error.MimoGlueSelfCheckDtype,
    };
    @memcpy(out, p[0..n]);
    return out;
}

/// Compares two arrays bit for bit (contiguous host copies): the count of equal elements of `width` bytes, the first
/// differing element.
pub fn bitwise(a: std.mem.Allocator, x: Array, y: Array, s: Stream) !struct { equal: usize, total: usize, first: ?usize } {
    const p = try hostBits(a, x, s);
    defer a.free(p);
    const q = try hostBits(a, y, s);
    defer a.free(q);
    if (p.len != q.len) return error.MimoGlueSelfCheckShape;
    const width = mlx.mlx_array_itemsize(x);
    var equal: usize = 0;
    var first: ?usize = null;
    var i: usize = 0;
    while (i < p.len) : (i += width) {
        if (std.mem.eql(u8, p[i..][0..width], q[i..][0..width])) equal += 1 else if (first == null) {
            first = i / width;
        }
    }
    return .{ .equal = equal, .total = p.len / width, .first = first };
}

fn randomBf16(a: std.mem.Allocator, r: *Rng, shape: []const c_int, amp: f32, center: f32) !Array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const bits = try a.alloc(u16, n);
    defer a.free(bits);
    for (bits) |*b| b.* = model.bf16Bits(center + r.bf(amp));
    return mlx.mlx_array_new_data(bits.ptr, shape.ptr, @intCast(shape.len), .bfloat16);
}

/// Static bound (MLX device bytes, one M at a time): router BF16 2 MiB + F32 copy 4 MiB, the norm weights, 8 rows of
/// inputs and outputs (< 1 MiB), the stock graph's transients (< 4 MiB), 256 MiB cache and 512 MiB compiler allowance.
pub fn selfCheckBound() u64 {
    return (2 << 20) + (4 << 20) + (64 << 10) + (16 << 20) + (256 << 20) + (512 << 20);
}

test "mimo GPU glue K2: the fused glue kernels vs the stock ops bit for bit (residual, norms, router logits, ids, weights) at M = 1..8 and on tie fixtures" {
    const module = @import("module.zig");
    const enabled = std.c.getenv("MIMO_GLUE_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ceiling: u64 = 1 << 30;
    const bound = selfCheckBound();
    std.debug.print("MIMO_GLUE_SELFCHECK static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoGlueAdmission;
    const a = testing.allocator;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    var k = try Kernels.init(.{});
    defer k.deinit();
    var r: Rng = .{ .st = 0x6b3a9e11 };
    const pre_w = try randomBf16(a, &r, &.{4096}, 0.5, 1);
    defer drop(pre_w);
    const post_w = try randomBf16(a, &r, &.{4096}, 0.5, 1);
    defer drop(post_w);
    const gate = try randomBf16(a, &r, &.{ 256, 4096 }, 0.05, 0);
    defer drop(gate);
    var bias_v: [256]f32 = undefined;
    for (&bias_v) |*v| v.* = r.bf(0.01);
    const bias = mlx.mlx_array_new_data(&bias_v, &[_]c_int{256}, 1, .float32);
    defer drop(bias);
    var ones: Array = .{};
    try mlx.check(mlx.mlx_ones(&ones, &.{4096}, 1, .bfloat16, s));
    defer drop(ones);
    // The module's router weights: the F32 transposed view (Layer.promote), evaluated.
    const gf = try module.cast(gate, .float32, s);
    defer drop(gf);
    var gt: Array = .{};
    try mlx.check(mlx.mlx_transpose(&gt, gf, s));
    defer drop(gt);
    try mlx.check(mlx.mlx_array_eval(gt));
    var failed: usize = 0;
    var m: c_int = 1;
    while (m <= max_rows) : (m += 1) {
        const h = try randomBf16(a, &r, &.{ m, 4096 }, 4, 0);
        defer drop(h);
        const mlp = try randomBf16(a, &r, &.{ m, 4096 }, 2, 0);
        defer drop(mlp);
        const attn = try randomBf16(a, &r, &.{ m, 4096 }, 1, 0);
        defer drop(attn);
        // stock: Layer.forward's ops (the previous layer's add, the input norm, the attention residual, the post norm,
        // module.route)
        const x = try module.binary(mlx.mlx_add, h, mlp, s);
        defer drop(x);
        const normed = try module.rmsNorm(x, pre_w, ones, 1e-6, s);
        defer drop(normed);
        const norm0 = try module.rmsNorm(h, pre_w, ones, 1e-6, s);
        defer drop(norm0);
        const residual = try module.binary(mlx.mlx_add, x, attn, s);
        defer drop(residual);
        const mlp_in = try module.rmsNorm(residual, post_w, ones, 1e-6, s);
        defer drop(mlp_in);
        const xf = try module.cast(mlp_in, .float32, s);
        defer drop(xf);
        var logits_stock: Array = .{};
        defer drop(logits_stock);
        try mlx.check(mlx.mlx_matmul(&logits_stock, xf, gt, s));
        var routes = try module.routeTail(logits_stock, bias, 8, s);
        defer routes.deinit();
        // fused
        const f_norm0 = try norm(&k, h, pre_w, s);
        defer drop(f_norm0);
        var pre = try addNorm(&k, h, mlp, pre_w, s);
        defer pre.deinit();
        var post = if (m == 1) try router1(&k, pre.h, attn, post_w, gate, s) else try addNormF32(&k, pre.h, attn, post_w, s);
        defer post.deinit();
        var logits: Array = .{};
        defer drop(logits);
        if (m == 1) logits = post.x else try mlx.check(mlx.mlx_matmul(&logits, post.x, gt, s));
        if (m == 1) post.x = .{};
        var t8 = try top8(&k, logits, bias, s);
        defer t8.deinit();
        var ids_u: Array = .{};
        defer drop(ids_u);
        try mlx.check(mlx.mlx_astype(&ids_u, routes.ids, .uint32, s));
        const checks = [_]struct { name: []const u8, x: Array, y: Array }{
            .{ .name = "norm0", .x = f_norm0, .y = norm0 },
            .{ .name = "boundary_h", .x = pre.h, .y = x },
            .{ .name = "boundary_n", .x = pre.n, .y = normed },
            .{ .name = "residual", .x = post.h, .y = residual },
            .{ .name = "mlp_in", .x = post.n, .y = mlp_in },
            .{ .name = "logits", .x = logits, .y = logits_stock },
            .{ .name = "ids", .x = t8.ids, .y = ids_u },
            .{ .name = "weights", .x = t8.scores, .y = routes.scores },
        };
        for (checks) |c| {
            const b = try bitwise(a, c.x, c.y, s);
            std.debug.print("MIMO_GLUE_SELFCHECK K2 M={d} {s} bitwise={d}/{d}{s}\n", .{ m, c.name, b.equal, b.total, if (b.first != null) " FAIL" else "" });
            if (b.equal != b.total) failed += 1;
        }
        if (m > 1) {
            const b = try bitwise(a, post.x, xf, s);
            std.debug.print("MIMO_GLUE_SELFCHECK K2 M={d} xf32 bitwise={d}/{d}{s}\n", .{ m, b.equal, b.total, if (b.first != null) " FAIL" else "" });
            if (b.equal != b.total) failed += 1;
        }
    }
    // Tie fixtures through both tails on identical logits: heavy ties, all equal, 12 equal top keys, signed zeros.
    for (0..4) |kind| {
        const rows: c_int = 8;
        var lv: [8 * 256]f32 = undefined;
        for (&lv, 0..) |*v, i| v.* = switch (kind) {
            0 => @floatFromInt(@as(i32, @intCast(r.next() % 5)) - 2),
            1 => 0.5,
            2 => if (i % 21 == 0) 3 else r.bf(1),
            else => if (i % 2 == 0) 0.0 else -0.0,
        };
        const lg = mlx.mlx_array_new_data(&lv, &[_]c_int{ rows, 256 }, 2, .float32);
        defer drop(lg);
        var tail = try module.routeTail(lg, bias, 8, s);
        defer tail.deinit();
        var t8 = try top8(&k, lg, bias, s);
        defer t8.deinit();
        var ids_u: Array = .{};
        defer drop(ids_u);
        try mlx.check(mlx.mlx_astype(&ids_u, tail.ids, .uint32, s));
        const bi = try bitwise(a, t8.ids, ids_u, s);
        const bw = try bitwise(a, t8.scores, tail.scores, s);
        std.debug.print("MIMO_GLUE_SELFCHECK K2 ties kind={d} ids bitwise={d}/{d} weights bitwise={d}/{d}{s}\n", .{ kind, bi.equal, bi.total, bw.equal, bw.total, if (bi.first != null or bw.first != null) " FAIL" else "" });
        if (bi.equal != bi.total or bw.equal != bw.total) failed += 1;
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_GLUE_SELFCHECK K2 done failed={d} peak={d} bound={d}\n", .{ failed, peak, bound });
    if (failed != 0) return error.MimoGlueSelfCheck;
    if (peak > bound) return error.MimoGluePeakExceeded;
}

// ---- X19 device check (pre-ship family `x19_router`: MIMO_GLUE_GPU=1 + MIMO_X19_PACK=<pack>; env + lock gated) ----

/// Static bound of the X19 device check (MLX device bytes): the 47 routers in BF16 (94 MiB) and their transposed F32
/// copies (188 MiB: the control's arrays, for the stock comparisons and the bench), the norms and corrections (< 1 MiB),
/// eight rows in and out per call (< 1 MiB), the bench's 47 per-call outputs (< 1 MiB), 32 MiB of transients, 256 MiB
/// cache and 512 MiB compiler.
pub fn x19Bound() u64 {
    return 47 * ((2 << 20) + (4 << 20)) + (3 << 20) + (32 << 20) + (256 << 20) + (512 << 20);
}

fn hostF32(a: std.mem.Allocator, x: Array, s: Stream) ![]f32 {
    var f: Array = .{};
    defer drop(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    const bytes = try hostBits(a, f, s);
    defer a.free(bytes);
    const out = try a.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn bf16Array(bits: []const u16, shape: []const c_int) Array {
    return mlx.mlx_array_new_data(bits.ptr, shape.ptr, @intCast(shape.len), .bfloat16);
}

fn randomBits(r: *Rng, out: []u16, amp: f32, center: f32) void {
    for (out) |*b| b.* = model.bf16Bits(center + r.bf(amp));
}

/// The lookahead selector's per-row ranking of a hint (scores already corrected): the 8 highest, a tie to the lower id.
fn hintTop8(scores: []const f32) [8]u32 {
    var ids: [8]u32 = undefined;
    var taken: [256]bool = @splat(false);
    for (&ids) |*out| {
        var best: usize = scores.len;
        for (scores, 0..) |v, e| {
            if (!taken[e] and (best == scores.len or v > scores[best])) best = e;
        }
        taken[best] = true;
        out.* = @intCast(best);
    }
    return ids;
}

fn evalOne(x: Array) !void {
    try mlx.check(mlx.mlx_array_eval(x));
}

fn stampNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// One release router on the device: its BF16 weight, the transposed F32 copy (the control's `Layer.promote`), the
/// correction and the post-attention norm, and the host's F32 view of the weight (the models' input).
const X19Layer = struct {
    gate: Array,
    gt: Array,
    bias: Array,
    norm: Array,
    w_host: []f32,

    fn load(a: std.mem.Allocator, dir: []const u8, layer: usize, s: Stream) !X19Layer {
        const read = @import("fp8_linear.zig").readTensor;
        const module = @import("module.zig");
        var name: [128]u8 = undefined;
        const wraw = try read(a, dir, "model.safetensors", try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.gate.weight", .{layer}), "BF16", &.{ 256, 4096 });
        defer a.free(wraw);
        const braw = try read(a, dir, "model.safetensors", try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{layer}), "F32", &.{256});
        defer a.free(braw);
        const nraw = try read(a, dir, "model.safetensors", try std.fmt.bufPrint(&name, "model.layers.{d}.post_attention_layernorm.weight", .{layer}), "BF16", &.{4096});
        defer a.free(nraw);
        const wbits = std.mem.bytesAsSlice(u16, @as([]align(1) u8, wraw));
        const w_host = try a.alloc(f32, E * D);
        errdefer a.free(w_host);
        for (wbits, w_host) |b, *w| w.* = model.fromBits(b);
        const gate = mlx.mlx_array_new_data(wraw.ptr, &[_]c_int{ 256, 4096 }, 2, .bfloat16);
        const gf = try module.cast(gate, .float32, s);
        defer drop(gf);
        var gt: Array = .{};
        try mlx.check(mlx.mlx_transpose(&gt, gf, s));
        try evalOne(gt);
        return .{
            .gate = gate,
            .gt = gt,
            .bias = mlx.mlx_array_new_data(braw.ptr, &[_]c_int{256}, 1, .float32),
            .norm = mlx.mlx_array_new_data(nraw.ptr, &[_]c_int{4096}, 1, .bfloat16),
            .w_host = w_host,
        };
    }

    fn deinit(l: *X19Layer, a: std.mem.Allocator) void {
        drop(l.gate);
        drop(l.gt);
        drop(l.bias);
        drop(l.norm);
        a.free(l.w_host);
    }
};

test "mimo GPU glue X19: router_rows vs router1 and MLX's M = 1 product bit for bit, M-invariant at M = 1..8 (random and the release's 47 routers); the BF16 predictor bit-equal to the F32 predictor at M = 1; rates and device times vs the F32 GEMM and gather_mv" {
    const module = @import("module.zig");
    const math_mod = @import("math.zig");
    const ops = @import("../deepseek_v41/ops.zig");
    const enabled = std.c.getenv("MIMO_GLUE_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const pack: ?[]const u8 = if (std.c.getenv("MIMO_X19_PACK")) |p| std.mem.span(p) else null;
    const ceiling: u64 = 2 << 30;
    const bound = x19Bound();
    std.debug.print("MIMO_X19_SELFCHECK static: bound={d} ceiling={d} pack={s}\n", .{ bound, ceiling, pack orelse "none" });
    if (bound > ceiling) return error.MimoGlueAdmission;
    const a = testing.allocator;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    var k = try Kernels.init(.{});
    defer k.deinit();
    var rows = try RouterRows.init(4096, 256);
    defer rows.deinit();
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const geometry = try @import("bank.zig").Geometry.init(4096, 2048, 256, 1, 47);
    var pred = try math_mod.Predictor.install(.bf16, geometry);
    defer pred.deinit();
    var r: Rng = .{ .st = 0x7a19_0d01 };
    var fail_minv: usize = 0;
    var fail_router1: usize = 0;
    var fail_mlx: usize = 0;
    var fail_pred: usize = 0;
    var gather_model_diff: usize = 0;
    var gather_model_rows: usize = 0;
    const ab = try a.alloc(u16, rows_cap * D);
    defer a.free(ab);
    const bb = try a.alloc(u16, rows_cap * D);
    defer a.free(bb);
    const wb = try a.alloc(u16, D);
    defer a.free(wb);
    // (1) A random router, M = 1..8: every row of router_rows is router1's logits for that row (and MLX's M = 1 product).
    {
        const gbits = try a.alloc(u16, E * D);
        defer a.free(gbits);
        randomBits(&r, gbits, 0.05, 0);
        const gate = bf16Array(gbits, &.{ 256, 4096 });
        defer drop(gate);
        const gf = try module.cast(gate, .float32, s);
        defer drop(gf);
        var gt: Array = .{};
        try mlx.check(mlx.mlx_transpose(&gt, gf, s));
        defer drop(gt);
        randomBits(&r, wb, 0.5, 1);
        const w = bf16Array(wb, &.{4096});
        defer drop(w);
        var m: usize = 1;
        while (m <= rows_cap) : (m += 1) {
            randomBits(&r, ab[0 .. m * D], 4, 0);
            randomBits(&r, bb[0 .. m * D], 1, 0);
            const ma = bf16Array(ab[0 .. m * D], &.{ @intCast(m), 4096 });
            defer drop(ma);
            const mb = bf16Array(bb[0 .. m * D], &.{ @intCast(m), 4096 });
            defer drop(mb);
            var post = try addNorm(&k, ma, mb, w, s);
            defer post.deinit();
            const lm = try rows.apply(post.n, gate, s);
            defer drop(lm);
            const lm_h = try hostF32(a, lm, s);
            defer a.free(lm_h);
            var equal: usize = 0;
            for (0..m) |row| {
                const ra = bf16Array(ab[row * D ..][0..D], &.{ 1, 4096 });
                defer drop(ra);
                const rb = bf16Array(bb[row * D ..][0..D], &.{ 1, 4096 });
                defer drop(rb);
                var one = try router1(&k, ra, rb, w, gate, s);
                defer one.deinit();
                const l1 = try hostF32(a, one.x, s);
                defer a.free(l1);
                if (std.mem.eql(u32, @ptrCast(l1), @ptrCast(lm_h[row * E ..][0..E]))) equal += 1 else fail_minv += 1;
                // MLX's own M = 1 product of that row (the control's router at M = 1) and router_rows at M = 1.
                const xf = try module.cast(one.n, .float32, s);
                defer drop(xf);
                var lmlx: Array = .{};
                try mlx.check(mlx.mlx_matmul(&lmlx, xf, gt, s));
                defer drop(lmlx);
                const b1 = try bitwise(a, lmlx, one.x, s);
                if (b1.equal != b1.total) fail_mlx += 1;
                const l1r = try rows.apply(one.n, gate, s);
                defer drop(l1r);
                const b2 = try bitwise(a, l1r, one.x, s);
                if (b2.equal != b2.total) fail_router1 += 1;
            }
            std.debug.print("MIMO_X19_SELFCHECK random M={d} rows_equal_router1={d}/{d}{s}\n", .{ m, equal, m, if (equal != m) " FAIL" else "" });
        }
    }
    // (2) The release's 47 routers: the verify widths M = 1..4 (M-invariance, the F32 GEMM's rate), the predictor routes.
    var layers: [47]X19Layer = undefined;
    var loaded: usize = 0;
    defer for (layers[0..loaded]) |*l| l.deinit(a);
    if (pack) |dir| {
        for (&layers, 1..) |*l, li| {
            l.* = try X19Layer.load(a, dir, li, s);
            loaded += 1;
        }
        var gemm_logits_equal: usize = 0;
        var gemm_logits: usize = 0;
        var gemm_ids_rows_differ: usize = 0;
        var gemm_rows: usize = 0;
        var gemm_absdiff: f32 = 0;
        var pred_absdiff: f32 = 0;
        var pred_top8_differ: usize = 0;
        var pred_rows: usize = 0;
        for (layers[0..loaded], 1..) |*l, li| {
            randomBits(&r, ab[0 .. 4 * D], 4, 0);
            randomBits(&r, bb[0 .. 4 * D], 1, 0);
            var one_h: [4][]f32 = undefined;
            var got_rows: usize = 0;
            defer for (one_h[0..got_rows]) |h| a.free(h);
            for (0..4) |row| {
                const ra = bf16Array(ab[row * D ..][0..D], &.{ 1, 4096 });
                defer drop(ra);
                const rb = bf16Array(bb[row * D ..][0..D], &.{ 1, 4096 });
                defer drop(rb);
                var one = try router1(&k, ra, rb, l.norm, l.gate, s);
                defer one.deinit();
                one_h[row] = try hostF32(a, one.x, s);
                got_rows += 1;
            }
            for (1..5) |m| {
                const ma = bf16Array(ab[0 .. m * D], &.{ @intCast(m), 4096 });
                defer drop(ma);
                const mb = bf16Array(bb[0 .. m * D], &.{ @intCast(m), 4096 });
                defer drop(mb);
                var post = try addNorm(&k, ma, mb, l.norm, s);
                defer post.deinit();
                const lm = try rows.apply(post.n, l.gate, s);
                defer drop(lm);
                const lm_h = try hostF32(a, lm, s);
                defer a.free(lm_h);
                for (0..m) |row| if (!std.mem.eql(u32, @ptrCast(one_h[row]), @ptrCast(lm_h[row * E ..][0..E]))) {
                    fail_minv += 1;
                };
                // The control's verify router at M = 2..4: MLX's F32 GEMM on the same rows (rounding-class).
                if (m > 1) {
                    const xf = try module.cast(post.n, .float32, s);
                    defer drop(xf);
                    var lg: Array = .{};
                    try mlx.check(mlx.mlx_matmul(&lg, xf, l.gt, s));
                    defer drop(lg);
                    const lg_h = try hostF32(a, lg, s);
                    defer a.free(lg_h);
                    for (lg_h, lm_h) |p, q| {
                        gemm_logits += 1;
                        if (@as(u32, @bitCast(p)) == @as(u32, @bitCast(q))) gemm_logits_equal += 1;
                        gemm_absdiff = @max(gemm_absdiff, @abs(p - q));
                    }
                    var t_gemm = try top8(&k, lg, l.bias, s);
                    defer t_gemm.deinit();
                    var t_rows = try top8(&k, lm, l.bias, s);
                    defer t_rows.deinit();
                    const ig = try hostBits(a, t_gemm.ids, s);
                    defer a.free(ig);
                    const ir = try hostBits(a, t_rows.ids, s);
                    defer a.free(ir);
                    for (0..m) |row| {
                        gemm_rows += 1;
                        if (!std.mem.eql(u8, ig[row * 32 ..][0..32], ir[row * 32 ..][0..32])) gemm_ids_rows_differ += 1;
                    }
                }
                // The predictor: the BF16 route (installed type) against the control's `Math.predictorScores` (MLX's gemv at
                // M = 1, gather_mv per row at M > 1) on the same rows and the same router.
                const mark = g.mark();
                defer g.resetTo(mark);
                const ctl = try math_mod.Math.predictorScores(&g, post.n, l.gt, l.bias);
                const got = try pred.scores(&pred, &g, post.n, l.gate, l.bias);
                const bp = try bitwise(a, got, ctl, s);
                if (m == 1 and bp.equal != bp.total) fail_pred += 1;
                const ch = try hostF32(a, ctl, s);
                defer a.free(ch);
                const gh = try hostF32(a, got, s);
                defer a.free(gh);
                for (ch, gh) |p, q| pred_absdiff = @max(pred_absdiff, @abs(p - q));
                for (0..m) |row| {
                    pred_rows += 1;
                    // The scores carry the correction already: the lookahead's per-row ranking of the hint.
                    const ids_c = hintTop8(ch[row * E ..][0..E]);
                    const ids_b = hintTop8(gh[row * E ..][0..E]);
                    if (!std.mem.eql(u32, &ids_c, &ids_b)) pred_top8_differ += 1;
                }
                // The host model of the control's M > 1 order (gather_mv), checked on row 0 against the device.
                if (m > 1) {
                    const xf = try g.astype(post.n, .float32);
                    const lgm = try math_mod.Math.predictorRows(&g, xf, l.gt, @intCast(m));
                    const lgm_h = try hostF32(a, lgm, s);
                    defer a.free(lgm_h);
                    const n_h = try hostF32(a, post.n, s);
                    defer a.free(n_h);
                    var model_out: [E]f32 = undefined;
                    mlxGemvGatherRouter(l.w_host, n_h[0..D], &model_out);
                    gather_model_rows += 1;
                    if (!std.mem.eql(u32, @ptrCast(&model_out), @ptrCast(lgm_h[0..E]))) gather_model_diff += 1;
                }
            }
            if (li == 1 or li == 24 or li == 47) std.debug.print("MIMO_X19_SELFCHECK pack layer={d} done minv_fail={d} pred_m1_fail={d}\n", .{ li, fail_minv, fail_pred });
        }
        std.debug.print("MIMO_X19_RATE verify_router M=2..4 vs the F32 GEMM: logits_bit_equal={d}/{d} logit_absdiff_max={e} top8_rows_differ={d}/{d}\n", .{ gemm_logits_equal, gemm_logits, gemm_absdiff, gemm_ids_rows_differ, gemm_rows });
        std.debug.print("MIMO_X19_RATE predictor M=1..4 bf16 vs f32: score_absdiff_max={e} hint_top8_rows_differ={d}/{d} (M = 1 bit-equal required) gather_model_rows_differ={d}/{d}\n", .{ pred_absdiff, pred_top8_differ, pred_rows, gather_model_diff, gather_model_rows });
        // (3) Device time per call over the 47 routers in one eval (the medians of 9 evals; measures, gates nothing).
        for (1..5) |m| {
            randomBits(&r, ab[0 .. m * D], 1, 0);
            const x = bf16Array(ab[0 .. m * D], &.{ @intCast(m), 4096 });
            defer drop(x);
            const xf = try module.cast(x, .float32, s);
            defer drop(xf);
            try evalOne(xf);
            var med: [4]f64 = undefined;
            for (0..4) |kind| {
                var samples: [9]f64 = undefined;
                for (0..12) |it| {
                    const mark = g.mark();
                    defer g.resetTo(mark);
                    var outs: [47]Array = undefined;
                    const t0 = stampNs();
                    for (layers[0..loaded], &outs) |*l, *o| o.* = switch (kind) {
                        0 => try g.adopt(try rows.apply(x, l.gate, s)),
                        1 => try g.matmul(xf, l.gt),
                        2 => try math_mod.Math.predictorScores(&g, x, l.gt, l.bias),
                        else => try pred.scores(&pred, &g, x, l.gate, l.bias),
                    };
                    const vec = mlx.mlx_vector_array_new_data(&outs, loaded);
                    defer _ = mlx.mlx_vector_array_free(vec);
                    try mlx.check(mlx.mlx_eval(vec));
                    const dt: f64 = @floatFromInt(stampNs() - t0);
                    if (it >= 3) samples[it - 3] = dt / 1000.0 / @as(f64, @floatFromInt(loaded));
                }
                std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
                med[kind] = samples[4];
            }
            std.debug.print("MIMO_X19_BENCH M={d} router_rows_us={d:.2} f32_router_us={d:.2} ({s}) predictor_f32_us={d:.2} predictor_bf16_us={d:.2}\n", .{ m, med[0], med[1], if (m == 1) "gemv" else "GEMM", med[2], med[3] });
        }
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const yes = struct {
        fn f(b: bool) []const u8 {
            return if (b) "yes" else "no";
        }
    }.f;
    std.debug.print("X19_EXACT minv={s} router1={s} mlx_m1={s} pred_m1={s} gather_model={s} pack={s} peak={d} bound={d}\n", .{ yes(fail_minv == 0), yes(fail_router1 == 0), yes(fail_mlx == 0), yes(fail_pred == 0), if (gather_model_rows == 0) "n/a" else yes(gather_model_diff == 0), if (pack != null) "yes" else "none", peak, bound });
    if (fail_minv + fail_router1 + fail_mlx + fail_pred != 0) return error.MimoGlueSelfCheck;
    if (peak > bound) return error.MimoGluePeakExceeded;
}
