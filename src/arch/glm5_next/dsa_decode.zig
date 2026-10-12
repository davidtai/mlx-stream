//! GLM-5.3-Flash's DSA decode attention in two texts (`decode_dsa: fused`, the default; `ops` the op chain, the control):
//! the sparse latent attention of the decode lane's rows (one, or the MTP verify's <= 4) over the latent rows their
//! indexer selected (`ops.selectTokens`: 2,048 pooled tokens and the query's partial pool, -1 for an unused slot), where
//! the op chain runs about fifteen dispatches (the index clamp and mask, the rows' gather, the scale, the scores' matmul,
//! the mask, the softmax, the output matmul, the reshapes). The split text (`glm5n_dsa_sparse_split`, kernels/) is the
//! split half of mlx-serve's msv_dec_attn (MiMo's dec_attn_split pattern: an online softmax in exp2 units per slice of the
//! selection, every slice's unnormalised output, running max and sum) for MLA (one latent row is every head's key and
//! value) with its keys gathered by index; MiMo's `dec_attn_merge.metal` (src/arch/mimo_v2/kernels, pinned by MiMo's
//! manifest) merges the slices into the bf16 output [1, H, S, R]. f32 throughout: rounding-class against the op chain
//! (bf16 scores and probabilities there). Installed once at construction (`install`: the geometry refused by name, the
//! launches per row count, one launch of each over zeros as the compile check).

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
/// The latent's width the texts are launched at, heads per threadgroup (one simdgroup each), keys per tile, slices.
pub const latent: u32 = 512;
pub const head_group: u32 = 8;
pub const tile_keys: u32 = 16;
pub const slices: u32 = 16;

comptime {
    if (max_rows != @import("experts.zig").decode_lane_rows) @compileError("glm5_next dsa_decode: the configs do not cover the decode lane");
}

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-dsa.json");
    pub const split = @embedFile("kernels/glm5n_dsa_sparse_split.metal");
    /// MiMo's merge text, unchanged (its own pin: mimo_v2/decode_attention.zig `text_pins`).
    pub const merge = @embedFile("../mimo_v2/kernels/dec_attn_merge.metal");
};
pub const manifest_sha256 = "95978e6ba0769224d57304b5bbcca2225480c2ebdab4e1f170ce057595d193e9";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "glm5n_dsa_sparse_split.metal", .text = texts.split, .sha256 = "b0978989050147499611e36b47d16ec077953521cb7ab4837adb36d19d694636" },
};
/// MiMo's pin of dec_attn_merge.metal (mimo_v2/decode_attention.zig): the merge is its text, byte for byte.
pub const merge_sha256 = "1c43462d81fe43b9eb6988346685518d8f812880bcba61ee7f4b42be57aa8a2a";

pub fn checkPins() !void {
    try kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.Glm5NextDsaManifestNotPinned, error.Glm5NextDsaTextNotPinned);
    if (!std.mem.eql(u8, &kset.shaHex(texts.merge), merge_sha256)) return error.Glm5NextDsaTextNotPinned;
}

/// The model the texts serve: the 512-wide latent, heads in whole groups of `head_group` (refused by name otherwise).
pub fn requireGeometry(c: *const glm.Config) error{Glm5NextDsaDecodeGeometry}!void {
    if (c.kv_lora_rank != latent or c.n_heads % head_group != 0 or c.nDsa() == 0) return error.Glm5NextDsaDecodeGeometry;
}

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    split: Kernel = .{ .ctx = null },
    merge: Kernel = .{ .ctx = null },
    /// Index rows - 1.
    split_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    merge_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: the attention scale (1 / sqrt(head_dim)), zeros (eight words: MLX passes a smaller input as constant).
    consts: T = .{ .ctx = null },

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.split, k.merge }) |x| if (x.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_free(x);
        };
        for (k.split_cfg ++ k.merge_cfg) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        if (k.consts.ctx != null) _ = mlx.mlx_array_free(k.consts);
        k.* = .{};
    }
};

/// The split's launch at `rows` query rows: `slices` x one 32 x `head_group` threadgroup per head group per row.
pub fn splitGrid(heads: u32, rows: u32) [3]c_int {
    return .{ @intCast(slices * 32 * head_group), @intCast(heads / head_group), @intCast(rows) };
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.Glm5NextDsaDecodeKernel;
    return k;
}

pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, diag: *sdk.Diag) !Kernels {
    requireGeometry(c) catch |e| {
        diag.set("glm5_next decode_dsa=fused: the model is not the texts' (latent {d}, {d} heads; the texts take a 512 latent, heads in groups of 8)", .{ c.kv_lora_rank, c.n_heads });
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next decode_dsa=fused: the module's stream is not a GPU stream", .{});
        return error.Glm5NextDsaDecodeNotGpu;
    }
    var k: Kernels = .{};
    errdefer k.deinit();
    k.split = try makeKernel("glm5n_dsa_sparse_split", &.{ "q", "lat", "sel", "consts" }, &.{ "pacc", "pm", "pl" }, texts.split, "#include <metal_simdgroup>\n");
    k.merge = try makeKernel("glm5n_dsa_merge", &.{ "pacc", "pm", "pl" }, &.{"out"}, texts.merge, "");
    const heads: c_int = @intCast(c.n_heads);
    const r: c_int = @intCast(latent);
    const ns: c_int = @intCast(slices);
    for (0..max_rows) |i| {
        const m: c_int = @intCast(i + 1);
        const sg = splitGrid(c.n_heads, @intCast(m));
        const cs = mlx.mlx_fast_metal_kernel_config_new();
        k.split_cfg[i] = cs;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cs, &[_]c_int{ heads * m, ns, r }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cs, &[_]c_int{ heads * m, ns }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cs, &[_]c_int{ heads * m, ns }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cs, sg[0], sg[1], sg[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cs, @intCast(32 * head_group), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cs, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cs, "R", r));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cs, "HG", @intCast(head_group)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cs, "BK", @intCast(tile_keys)));
        // MiMo's merge launch (decode_attention.zig `split`): one BDV-thread threadgroup per (head, row).
        const cm = mlx.mlx_fast_metal_kernel_config_new();
        k.merge_cfg[i] = cm;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cm, &[_]c_int{ 1, heads, m, r }, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cm, r, heads * m, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cm, r, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cm, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cm, "BDV", r));
    }
    const cv = [8]f32{ 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))), 0, 0, 0, 0, 0, 0, 0 };
    k.consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.consts.ctx == null) return error.MlxError;
    compileCheck(a, s, c, &k) catch |err| {
        var eb: [400]u8 = undefined;
        diag.set("glm5_next decode_dsa=fused: the compile check failed ({s}): {s}", .{ @errorName(err), mlx.takeError(&eb) orelse "" });
        return err;
    };
    return k;
}

/// One launch of each text at one row over zeros (a 64-row latent, a selection with unused slots), evaluated.
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, k: *const Kernels) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    const heads: c_int = @intCast(c.n_heads);
    const q = try g.zeros(&.{ 1, heads, 1, @intCast(latent) }, .bfloat16);
    const lat = try g.zeros(&.{ 1, 1, 64, @intCast(latent) }, .bfloat16);
    var sel: [40]i32 = undefined;
    for (&sel, 0..) |*x, i| x.* = if (i % 5 == 4) -1 else @intCast(i);
    const sa = try g.hostArray(std.mem.sliceAsBytes(&sel), &.{ 1, 1, 40 }, .int32);
    const o = try attend(&g, k, q, lat, sa, 1);
    try g.evalAll(&.{o});
    if (!std.mem.eql(c_int, g.shapeOf(o).slice(), &[_]c_int{ 1, heads, 1, @intCast(latent) })) return error.Glm5NextDsaDecodeShape;
}

/// The sparse latent attention of `rows` query rows: q_abs [1, H, rows, R] (the absorbed queries, unscaled), the latent
/// lane's view [1, 1, T, R], sel int32 [1, rows, W] -> [1, H, rows, R] bf16. Two dispatches.
pub fn attend(g: *G, k: *const Kernels, q_abs: T, lat: T, sel: T, rows: u32) !T {
    const lsh = g.shapeOf(lat);
    const ssh = g.shapeOf(sel);
    const lat2 = try g.reshape(lat, &.{ lsh.dim(-2), lsh.dim(-1) });
    const sel2 = try g.reshape(sel, &.{ @intCast(rows), ssh.dim(-1) });
    var parts: [3]T = undefined;
    {
        const in = mlx.mlx_vector_array_new_data(&[_]T{ q_abs, lat2, sel2, k.consts }, 4);
        defer _ = mlx.mlx_vector_array_free(in);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k.split, in, k.split_cfg[rows - 1], g.s));
        for (&parts, 0..) |*p, i| {
            var x = mlx.mlx_array_new();
            mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
                _ = mlx.mlx_array_free(x);
                return e;
            };
            p.* = try g.adopt(x);
        }
    }
    const in = mlx.mlx_vector_array_new_data(&parts, 3);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k.merge, in, k.merge_cfg[rows - 1], g.s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&out, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}

const testing = std.testing;

test "glm5_next dsa_decode: the embedded manifest and the split text are the pinned ones; the merge is MiMo's text" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 1);
    try testing.expect(std.mem.eql(u8, texts.merge, @embedFile("../mimo_v2/kernels/dec_attn_merge.metal")));
}

test "glm5_next dsa_decode: GLM-5.3-Flash's geometry is the texts'; the launch geometry per row count" {
    const c: glm.Config = .{};
    try requireGeometry(&c);
    try testing.expectError(error.Glm5NextDsaDecodeGeometry, requireGeometry(&.{ .kv_lora_rank = 256 }));
    try testing.expectError(error.Glm5NextDsaDecodeGeometry, requireGeometry(&.{ .n_heads = 60 }));
    for (1..max_rows + 1) |m| try testing.expectEqual([3]c_int{ 16 * 256, 8, @intCast(m) }, splitGrid(64, @intCast(m)));
    // A lane holds R / 32 dims of its head's query and output; a tile of 16 keys x 512 bf16 is 16 KiB of threadgroup memory.
    try testing.expectEqual(@as(u32, 16), latent / 32);
    try testing.expectEqual(@as(u32, 16 * 1024), tile_keys * latent * 2);
    try testing.expect(std.mem.indexOf(u8, texts.split, "const int W = sel_shape[1];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.merge, "const int NS = pm_shape[1];") != null);
}
