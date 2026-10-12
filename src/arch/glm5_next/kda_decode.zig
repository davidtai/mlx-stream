//! GLM-5.3-Flash's KDA decode step in one text (`decode_kda: fused`, the default; `ops` the stock op chain, the control):
//! the host's oMLX KDA decode kernel (mlx-serve `kda_recurrence.zig` `decodeRows`, `src/kernels/kda_decode.metal`: oMLX
//! 0.7.0, Apache-2.0) in its BF16-gate variant (kernels/glm5n_kda_decode.metal): per KDA layer of the decode lane (one row,
//! or the MTP lane's served steps; the MTP verify's capture keeps the op chain) the short conv and SiLU, the gate's second
//! stages over the BF16 f_b / g_b, the q/k L2 norms, the gate and beta, the delta rule over the f32 state and the gated
//! RMSNorm run as one dispatch (one 1,024-thread threadgroup per head) where the op chain runs about twenty. The
//! projections stay outside the text (q, k, v, f_a, g_a, b and o each its own GEMV: the weight format's seam, `proj`), and
//! the text reads and writes the module's own state layout (the f32 conv tail, the f32 recurrent state), so the prompt
//! pass's op chain and the decode text hand the state to each other unchanged. Rounding-class against `ops` (the conv's
//! output rounded to bf16 before the SiLU, the L2 norms' epsilon inside the square root, the gate stages' summation
//! order). Installed once at construction (`install`: the geometry refused by name, the launches per decode-lane width,
//! one launch over the first KDA layer's tables as the compile check).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const kda = @import("kda.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// The decode lane's widest call (`experts.decode_lane_rows`); the text takes up to 8 (its threadgroup arrays).
pub const max_rows: u32 = 4;
pub const threads: c_int = 1024;
const conv_taps: u32 = 4;

comptime {
    if (max_rows != @import("experts.zig").decode_lane_rows) @compileError("glm5_next kda_decode: the configs do not cover the decode lane");
}

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-kda.json");
    pub const header = @embedFile("kernels/header_glm5n_kda.metal");
    pub const decode = @embedFile("kernels/glm5n_kda_decode.metal");
};
/// sha256 of kernels/manifest-kda.json, which records the texts' sha256 (checked against the pins by a CPU test).
pub const manifest_sha256 = "f8ba2a6e2d60bf71986cf13cb03d85a98354d4461408d12dba610f0d99812ada";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_glm5n_kda.metal", .text = texts.header, .sha256 = "b28e8e1c60e71c9031b34cf2aaeb3e4e4afac0bc795d541c96bc7af61be2a069" },
    .{ .file = "glm5n_kda_decode.metal", .text = texts.decode, .sha256 = "cdb0fafddd6734ef51874601ed2ede2a2c729bcf85b08cc6d8abeff8e5a4dc68" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.Glm5NextKdaManifestNotPinned, error.Glm5NextKdaTextNotPinned);
}

/// The model the text serves (its head width 128, the short conv's 4 taps, its threadgroup per head), refused by name.
pub fn requireGeometry(c: *const glm.Config) error{Glm5NextKdaDecodeGeometry}!void {
    if (c.kda_head_dim != 128 or c.kda_conv != conv_taps or c.kda_heads == 0 or c.nKda() == 0) return error.Glm5NextKdaDecodeGeometry;
}

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    decode: Kernel = .{ .ctx = null },
    /// Index rows - 1: the launch of a call of that many rows.
    cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: 1 / sqrt(dk), the L2 eps (1e-6), the norm eps, the gate's lower bound, 1 / dk, zeros (eight words: MLX
    /// passes a smaller input in the constant address space).
    consts: T = .{ .ctx = null },

    pub fn deinit(k: *Kernels) void {
        if (k.decode.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(k.decode);
        for (k.cfg) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        if (k.consts.ctx != null) _ = mlx.mlx_array_free(k.consts);
        k.* = .{};
    }
};

pub const input_names = [_][*:0]const u8{ "q", "k", "v", "fa", "ga", "b", "conv_w", "a_exp", "dt_bias", "norm_w", "consts", "conv_state", "state_in", "fb_w", "gb_w" };
pub const output_names = [_][*:0]const u8{ "y", "conv_state_out", "state_out" };

/// One launch's geometry: one threadgroup of `threads` per head.
pub fn grid(c: *const glm.Config) [3]c_int {
    return .{ threads * @as(c_int, @intCast(c.kda_heads)), 1, 1 };
}

/// `checks`: one KDA layer of each weight-dtype variant the model holds (`variants`): the compile check launches the text
/// over each, so every variant MLX instantiates is built once here.
pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, checks: []const *const kda.Weights, diag: *sdk.Diag) !Kernels {
    requireGeometry(c) catch |e| {
        diag.set("glm5_next decode_kda=fused: the model is not the text's (KDA head {d}, conv {d}; the text takes 128 and 4)", .{ c.kda_head_dim, c.kda_conv });
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next decode_kda=fused: the module's stream is not a GPU stream", .{});
        return error.Glm5NextKdaDecodeNotGpu;
    }
    var k: Kernels = .{};
    errdefer k.deinit();
    {
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        k.decode = mtl.mlx_fast_metal_kernel_new("glm5n_kda_decode", in_vec, out_vec, texts.decode, texts.header, true, false);
        if (k.decode.ctx == null) return error.Glm5NextKdaDecodeKernel;
    }
    const qkv: c_int = @intCast(c.kdaDim());
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const cdim: c_int = @intCast(c.kdaConvDim());
    for (&k.cfg, 1..) |*cfg, m| {
        const rows: c_int = @intCast(m);
        cfg.* = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, &[_]c_int{ 1, rows, qkv }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, &[_]c_int{ 1, @intCast(conv_taps - 1), cdim }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, &[_]c_int{ 1, hv, dk, dk }, 4, .float32));
        const gr = grid(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg.*, gr[0], gr[1], gr[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg.*, threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg.*, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "TOK", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "QKV", qkv));
    }
    const cv = [8]f32{ 1.0 / @sqrt(@as(f32, @floatFromInt(c.kda_head_dim))), 1e-6, c.rms_norm_eps, c.kda_lower_bound, 1.0 / @as(f32, @floatFromInt(c.kda_head_dim)), 0, 0, 0 };
    k.consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.consts.ctx == null) return error.MlxError;
    for (checks) |w| compileCheck(a, s, c, &k, w) catch |err| {
        var eb: [400]u8 = undefined;
        diag.set("glm5_next decode_kda=fused: the compile check failed ({s}; conv {t}, f_b {t}, g_b {t}, o_norm {t}): {s}", .{ @errorName(err), mlx.mlx_array_dtype(w.conv_raw), mlx.mlx_array_dtype(w.f_b), mlx.mlx_array_dtype(w.g_b), mlx.mlx_array_dtype(w.o_norm), mlx.takeError(&eb) orelse "" });
        return err;
    };
    return k;
}

/// The weights the text reads in their stored dtype (each a separate kernel variant for MLX: the conv, f_b, g_b, o_norm).
pub const Variant = [4]mlx.mlx_dtype;

pub fn variantOf(w: *const kda.Weights) Variant {
    return .{ mlx.mlx_array_dtype(w.conv_raw), mlx.mlx_array_dtype(w.f_b), mlx.mlx_array_dtype(w.g_b), mlx.mlx_array_dtype(w.o_norm) };
}

/// The first layer of each distinct `Variant` among `layers` into `out` (the pack of record has two: the conv F32 on
/// layers 0-2, BF16 after); returns the count, refused by name past `out.len`.
pub fn variants(layers: []const *const kda.Weights, out: []*const kda.Weights) error{Glm5NextKdaDecodeVariants}!usize {
    var n: usize = 0;
    outer: for (layers) |w| {
        const v = variantOf(w);
        for (out[0..n]) |o| if (std.mem.eql(mlx.mlx_dtype, &variantOf(o), &v)) continue :outer;
        if (n == out.len) return error.Glm5NextKdaDecodeVariants;
        out[n] = w;
        n += 1;
    }
    return n;
}

/// One launch at one row over zeros and the first KDA layer's tables, evaluated (Metal compiles the text at its first
/// dispatch and refuses a threadgroup its registers cannot hold), the output's shape checked.
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, k: *const Kernels, w: *const kda.Weights) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    const h: c_int = @intCast(c.hidden_size);
    const x = try g.zeros(&.{ 1, 1, h }, .bfloat16);
    var st: kda.State = .{};
    defer st.reset(&g);
    const y = try step(&g, k, c, w, x, &st);
    try g.evalAll(&.{ y, st.conv.?, st.rec.? });
    if (!std.mem.eql(c_int, g.shapeOf(y).slice(), &[_]c_int{ 1, 1, h })) return error.Glm5NextKdaDecodeShape;
}

/// A projection of the layer's input rows (the weight format's seam: today the BF16 matmul the op chain runs).
fn proj(g: *G, x: T, w: T) !T {
    return g.matmul(x, try g.transpose(w));
}

/// `kda.Step`'s fused route: `ctx` the module's installed `Kernels`.
pub fn run(ctx: ?*const anyopaque, g: *G, c: *const glm.Config, w: *const kda.Weights, x: T, st: *kda.State) anyerror!T {
    const k: *const Kernels = @ptrCast(@alignCast(ctx.?));
    return step(g, k, c, w, x, st);
}

/// The decode lane's KDA layer: x [1, R, hidden] (R <= `max_rows`) -> [1, R, hidden]; `st` advanced past the rows.
pub fn step(g: *G, k: *const Kernels, c: *const glm.Config, w: *const kda.Weights, x: T, st: *kda.State) !T {
    const rows: u32 = @intCast(g.shapeOf(x).dim(1));
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const conv_state = st.conv orelse try g.zeros(&.{ 1, @intCast(conv_taps - 1), @intCast(c.kdaConvDim()) }, .float32);
    const state = st.rec orelse try g.zeros(&.{ 1, hv, dk, dk }, .float32);
    const ins = [_]T{
        try proj(g, x, w.q), try proj(g, x, w.k), try proj(g, x, w.v), try proj(g, x, w.f_a), try proj(g, x, w.g_a), try proj(g, x, w.b),
        w.conv_raw,          w.a_exp,             w.dt_bias,           w.o_norm,              k.consts,              conv_state,
        state,               w.f_b,               w.g_b,
    };
    const in = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.decode, in, k.cfg[rows - 1], g.s));
    var outs: [3]T = undefined;
    for (&outs, 0..) |*o, i| {
        var y = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&y, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(y);
            return e;
        };
        o.* = try g.adopt(y);
    }
    st.set(g, outs[1], outs[2]);
    return proj(g, outs[0], w.o);
}

const testing = std.testing;

test "glm5_next kda_decode: the embedded manifest and texts are the pinned ones" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 1);
}

test "glm5_next kda_decode: GLM-5.3-Flash's geometry is the text's; one 1,024-thread threadgroup per head; another refused by name" {
    const c: glm.Config = .{};
    try requireGeometry(&c);
    try testing.expectEqual([3]c_int{ 64 * 1024, 1, 1 }, grid(&c));
    for ([_]glm.Config{ .{ .kda_head_dim = 64 }, .{ .kda_conv = 3 } }) |bad| try testing.expectError(error.Glm5NextKdaDecodeGeometry, requireGeometry(&bad));
    // The text's sections index 1,024 threads: the conv's 3 x 128, the gate stages' 2 x 128 rows x 4, the norms' 2 x 4 rows.
    try testing.expectEqual(@as(u32, 2 * 128 * 4), @as(u32, @intCast(threads)));
    try testing.expect(std.mem.indexOf(u8, texts.decode, "const device auto* wl = (which == 0 ? fb_w : gb_w) + size_t(row) * DK + ql * VPT;") != null);
    try testing.expect(std.mem.indexOf(u8, texts.decode, "conv_state_out[r * NP + gc] = static_cast<float>(win[TOK + r]);") != null);
    try testing.expectEqual(@as(usize, 15), input_names.len);
}
