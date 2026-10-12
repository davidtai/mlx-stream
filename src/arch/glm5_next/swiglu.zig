//! GLM-5.3-Flash's decode MLP activation in one text (`decode_swiglu: fused`, the default; `ops` the op chain, the
//! control): the clamped SwiGLU of the decode lane's dense MLPs (layers 0-2) and shared experts (every routed layer)
//! between their gate / up GEMVs and their down GEMV (`kernels/glm5n_swiglu.metal`: one dispatch where the op chain
//! runs five). The GEMVs stay MLX's (BF16 at the box's bandwidth) or whichever weight format's GEMV is bound: the seam.
//! f32 arithmetic and one rounding: rounding-class against the op chain (a bf16 rounding per op there). A row-count
//! route (the decode lane's rows, <= 4) installed once at construction (the widths refused by name, a launch config per
//! width and row count, one launch over zeros as the compile check); the prompt pass keeps the op chain.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// The decode lane's widest call (the MTP verify's rows).
pub const max_rows: u32 = 4;
/// Elements per thread, threads per threadgroup.
pub const per_thread: u32 = 4;
pub const threads: u32 = 256;

comptime {
    if (max_rows != @import("experts.zig").decode_lane_rows) @compileError("glm5_next swiglu: the configs do not cover the decode lane");
}

pub const texts = struct {
    pub const swiglu = @embedFile("kernels/glm5n_swiglu.metal");
};
pub const text_sha256 = "9a6fb8fec4a13e17f1de93a710c031271b2b60a01e36168def9e6c0c5bc4badf";

pub fn checkPins() !void {
    if (!std.mem.eql(u8, &kset.shaHex(texts.swiglu), text_sha256)) return error.Glm5NextSwigluTextNotPinned;
}

/// The two widths the texts serve: the dense MLP's intermediate and the shared expert's (each a whole number of
/// threadgroups at one row).
pub fn requireWidth(width: u32) error{Glm5NextSwigluWidth}!void {
    if (width == 0 or width % (per_thread * threads) != 0) return error.Glm5NextSwigluWidth;
}

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    swiglu: Kernel = .{ .ctx = null },
    /// [`Width`][rows - 1].
    cfg: [2][max_rows]Config = @splat(@splat(.{ .ctx = null })),
    widths: [2]u32 = .{ 0, 0 },
    /// f32 [8]: the limit, zeros (eight words: MLX passes a smaller input as constant).
    consts: T = .{ .ctx = null },

    pub fn deinit(k: *Kernels) void {
        if (k.swiglu.ctx != null) _ = mlx.mlx_fast_metal_kernel_free(k.swiglu);
        for (k.cfg) |row| for (row) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        if (k.consts.ctx != null) _ = mlx.mlx_array_free(k.consts);
        k.* = .{};
    }
};

pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, dense_width: u32, shared_width: u32, diag: *sdk.Diag) !Kernels {
    for ([_]u32{ dense_width, shared_width }) |wd| requireWidth(wd) catch |e| {
        diag.set("glm5_next decode_swiglu=fused: width {d} is not a multiple of {d}", .{ wd, per_thread * threads });
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next decode_swiglu=fused: the module's stream is not a GPU stream", .{});
        return error.Glm5NextSwigluNotGpu;
    }
    var k: Kernels = .{ .widths = .{ dense_width, shared_width } };
    errdefer k.deinit();
    {
        const ins = [_][*:0]const u8{ "gate", "up", "consts" };
        const outs = [_][*:0]const u8{"out"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        k.swiglu = mlx.mlx_fast_metal_kernel_new("glm5n_swiglu", in_vec, out_vec, texts.swiglu, "", true, false);
        if (k.swiglu.ctx == null) return error.Glm5NextSwigluKernel;
    }
    for (k.widths, 0..) |wd, wi| for (0..max_rows) |ri| {
        const m: c_int = @intCast(ri + 1);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        k.cfg[wi][ri] = cfg;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ m, @intCast(wd) }, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, m * @as(c_int, @intCast(wd / per_thread)), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(threads), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    };
    const cv = [8]f32{ c.swiglu_limit, 0, 0, 0, 0, 0, 0, 0 };
    k.consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.consts.ctx == null) return error.MlxError;
    compileCheck(a, s, &k) catch |err| {
        var eb: [400]u8 = undefined;
        diag.set("glm5_next decode_swiglu=fused: the compile check failed ({s}): {s}", .{ @errorName(err), mlx.takeError(&eb) orelse "" });
        return err;
    };
    return k;
}

/// One launch at each width at one row over zeros, evaluated.
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, k: *const Kernels) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    for ([_]Width{ .dense, .shared }) |w| {
        const z = try g.zeros(&.{ 1, @intCast(k.widths[@backingInt(w)]) }, .bfloat16);
        try g.evalAll(&.{try activate(&g, k, w, z, z, 1)});
    }
}

/// Which MLP the call is (its width fixed at install).
pub const Width = enum(u1) { dense, shared };

/// The clamped SwiGLU of gate / up [rows, width] in one dispatch: `rows` the decode lane's (1..4: the caller's row-count
/// route), `w` the MLP's installed width.
pub fn activate(g: *G, k: *const Kernels, w: Width, gate: T, up: T, rows: u32) !T {
    const in = mlx.mlx_vector_array_new_data(&[_]T{ gate, up, k.consts }, 3);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k.swiglu, in, k.cfg[@backingInt(w)][rows - 1], g.s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&out, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}

const testing = std.testing;

test "glm5_next swiglu: the text is the pinned one; Flash's widths are the text's, a stray width refused" {
    try checkPins();
    const c: glm.Config = .{};
    try requireWidth(c.intermediate_size);
    try requireWidth(c.moe_intermediate_size);
    try testing.expectError(error.Glm5NextSwigluWidth, requireWidth(1000));
    try testing.expectError(error.Glm5NextSwigluWidth, requireWidth(0));
    try testing.expect(std.mem.indexOf(u8, texts.swiglu, "const float lim = consts[0];") != null);
}
