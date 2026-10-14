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
//! `dsa_split_form` picks the split's text at construction: `lean` (the default; glm5n_dsa_sparse_split_lean.metal, 16
//! heads a threadgroup, keys scored in groups of 4 and updated in key order, the factor-1 update without its multiply)
//! or `current` (glm5n_dsa_sparse_split.metal, 8 heads a threadgroup, the control). Both run the same floating-point
//! operations in the same order for every output (`emulate` below, the CPU test over 1 .. 4 rows): bit for bit.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
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
/// The lean form's heads a threadgroup.
pub const head_group_lean: u32 = 16;
/// The split's text (`dsa_split_form`), installed at construction.
pub const SplitForm = enum { current, lean };

pub fn headGroup(form: SplitForm) u32 {
    return switch (form) {
        .current => head_group,
        .lean => head_group_lean,
    };
}
pub const tile_keys: u32 = 16;
pub const slices: u32 = 16;

comptime {
    if (max_rows != @import("experts.zig").decode_lane_rows) @compileError("glm5_next dsa_decode: the configs do not cover the decode lane");
}

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-dsa.json");
    pub const split = @embedFile("kernels/glm5n_dsa_sparse_split.metal");
    pub const split_lean = @embedFile("kernels/glm5n_dsa_sparse_split_lean.metal");
    /// MiMo's merge text, unchanged (its own pin: mimo_v2/decode_attention.zig `text_pins`).
    pub const merge = @embedFile("../mimo_v2/kernels/dec_attn_merge.metal");
};
pub const manifest_sha256 = "a67dd8ade913ce530a5a65d6eb308536a994d043f038a93678c77355e36b7130";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "glm5n_dsa_sparse_split.metal", .text = texts.split, .sha256 = "b0978989050147499611e36b47d16ec077953521cb7ab4837adb36d19d694636" },
    .{ .file = "glm5n_dsa_sparse_split_lean.metal", .text = texts.split_lean, .sha256 = "5949c0a2da9279efa35e9ce6eff7fe72315acb90b1e59881b722067e20c186bc" },
};
/// MiMo's pin of dec_attn_merge.metal (mimo_v2/decode_attention.zig): the merge is its text, byte for byte.
pub const merge_sha256 = "1c43462d81fe43b9eb6988346685518d8f812880bcba61ee7f4b42be57aa8a2a";

pub fn checkPins() !void {
    try kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.Glm5NextDsaManifestNotPinned, error.Glm5NextDsaTextNotPinned);
    if (!std.mem.eql(u8, &kset.shaHex(texts.merge), merge_sha256)) return error.Glm5NextDsaTextNotPinned;
}

/// The model the texts serve: the 512-wide latent, heads in whole groups of both forms' head groups (refused by name
/// otherwise).
pub fn requireGeometry(c: *const glm.Config) error{Glm5NextDsaDecodeGeometry}!void {
    if (c.kv_lora_rank != latent or c.n_heads % head_group != 0 or c.n_heads % head_group_lean != 0 or c.nDsa() == 0) return error.Glm5NextDsaDecodeGeometry;
}

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    /// The split's text installed (`dsa_split_form`).
    form: SplitForm = .current,
    split: Kernel = .{ .ctx = null },
    merge: Kernel = .{ .ctx = null },
    /// Index rows - 1.
    split_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    merge_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: the attention scale (1 / sqrt(head_dim)), zeros (eight words: MLX passes a smaller input as constant).
    consts: T = .{ .ctx = null },

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.split, k.merge }) |x| if (x.ctx != null) {
            _ = mtl.mlx_fast_metal_kernel_free(x);
        };
        for (k.split_cfg ++ k.merge_cfg) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        if (k.consts.ctx != null) _ = mlx.mlx_array_free(k.consts);
        k.* = .{};
    }
};

/// The split's launch at `rows` query rows: `slices` x one 32 x `hg` threadgroup per head group per row.
pub fn splitGrid(heads: u32, rows: u32, hg: u32) [3]c_int {
    return .{ @intCast(slices * 32 * hg), @intCast(heads / hg), @intCast(rows) };
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mtl.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.Glm5NextDsaDecodeKernel;
    return k;
}

pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, form: SplitForm, diag: *sdk.Diag) !Kernels {
    requireGeometry(c) catch |e| {
        diag.set("glm5_next decode_dsa=fused: the model is not the texts' (latent {d}, {d} heads; the texts take a 512 latent, heads in groups of 8)", .{ c.kv_lora_rank, c.n_heads });
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next decode_dsa=fused: the module's stream is not a GPU stream", .{});
        return error.Glm5NextDsaDecodeNotGpu;
    }
    var k: Kernels = .{ .form = form };
    errdefer k.deinit();
    const hg = headGroup(form);
    k.split = switch (form) {
        .current => try makeKernel("glm5n_dsa_sparse_split", &.{ "q", "lat", "sel", "consts" }, &.{ "pacc", "pm", "pl" }, texts.split, "#include <metal_simdgroup>\n"),
        .lean => try makeKernel("glm5n_dsa_sparse_split_lean", &.{ "q", "lat", "sel", "consts" }, &.{ "pacc", "pm", "pl" }, texts.split_lean, "#include <metal_simdgroup>\n"),
    };
    k.merge = try makeKernel("glm5n_dsa_merge", &.{ "pacc", "pm", "pl" }, &.{"out"}, texts.merge, "");
    const heads: c_int = @intCast(c.n_heads);
    const r: c_int = @intCast(latent);
    const ns: c_int = @intCast(slices);
    for (0..max_rows) |i| {
        const m: c_int = @intCast(i + 1);
        const sg = splitGrid(c.n_heads, @intCast(m), hg);
        const cs = mlx.mlx_fast_metal_kernel_config_new();
        k.split_cfg[i] = cs;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cs, &[_]c_int{ heads * m, ns, r }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cs, &[_]c_int{ heads * m, ns }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cs, &[_]c_int{ heads * m, ns }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cs, sg[0], sg[1], sg[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cs, @intCast(32 * hg), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cs, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cs, "R", r));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cs, "HG", @intCast(hg)));
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
        try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.split, in, k.split_cfg[rows - 1], g.s));
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
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.merge, in, k.merge_cfg[rows - 1], g.s));
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
    try kset.expectManifestPins(texts.manifest, &text_pins, 2);
    try testing.expect(std.mem.eql(u8, texts.merge, @embedFile("../mimo_v2/kernels/dec_attn_merge.metal")));
}

test "glm5_next dsa_decode: GLM-5.3-Flash's geometry is the texts'; the launch geometry per row count" {
    const c: glm.Config = .{};
    try requireGeometry(&c);
    try testing.expectError(error.Glm5NextDsaDecodeGeometry, requireGeometry(&.{ .kv_lora_rank = 256 }));
    try testing.expectError(error.Glm5NextDsaDecodeGeometry, requireGeometry(&.{ .n_heads = 60 }));
    for (1..max_rows + 1) |m| {
        try testing.expectEqual([3]c_int{ 16 * 256, 8, @intCast(m) }, splitGrid(64, @intCast(m), headGroup(.current)));
        try testing.expectEqual([3]c_int{ 16 * 512, 4, @intCast(m) }, splitGrid(64, @intCast(m), headGroup(.lean)));
    }
    // A lane holds R / 32 dims of its head's query and output; a tile of 16 keys x 512 bf16 is 16 KiB of threadgroup memory.
    try testing.expectEqual(@as(u32, 16), latent / 32);
    try testing.expectEqual(@as(u32, 16 * 1024), tile_keys * latent * 2);
    try testing.expect(std.mem.indexOf(u8, texts.split, "const int W = sel_shape[1];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.merge, "const int NS = pm_shape[1];") != null);
}

// ---- `dsa_split_form`: the two split texts emulated on the CPU, operation for operation (no device) ----

/// The emulator's simd_sum: one fixed reduction over the 32 lanes, the same function for both forms (the device's own
/// order is one fixed order per instruction, the same instruction in both texts).
fn emuSimdSum(v: *const [32]f32) f32 {
    var w = v.*;
    var off: usize = 16;
    while (off > 0) : (off /= 2) {
        var n: [32]f32 = undefined;
        for (0..32) |l| n[l] = w[l] + w[l ^ off];
        w = n;
    }
    return w[0];
}

fn emuBf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

const Emu = struct {
    heads: usize,
    rows: usize,
    width: usize,
    q: []const u16, // [heads, rows, latent]
    lat: []const u16, // [lanes, latent]
    sel: []const i32, // [rows, width]
    scale: f32,
    pacc: []f32, // [heads * rows, slices, latent]
    pm: []f32,
    pl: []f32,
};

/// One text's launch over every threadgroup (x = slice, y = head group, z = row), as its source runs it: `lean` false
/// is glm5n_dsa_sparse_split.metal (HG 8), true glm5n_dsa_sparse_split_lean.metal (HG 16).
fn emulate(e: Emu, lean: bool) void {
    const R = latent;
    const DPL = R / 32;
    const BK = tile_keys;
    const NS = slices;
    const hg: usize = if (lean) head_group_lean else head_group;
    const W = e.width;
    const chunk = ((W + NS - 1) / NS + BK - 1) / BK * BK;
    const sl2 = e.scale * 1.44269504088896340736;
    var ks: [BK * R]u16 = undefined;
    var kid: [BK]i32 = undefined;
    for (0..e.rows) |qs| for (0..e.heads / hg) |y| for (0..NS) |j| {
        const k0 = j * chunk;
        const k1 = @min(W, k0 + chunk);
        // Each simdgroup's (head's) per-lane state.
        var qv: [32][32][DPL]f32 = undefined; // [sg][lane][d] (sg < hg)
        var acc: [32][32][DPL]f32 = undefined;
        var m: [32]f32 = @splat(-3.0e38);
        var l: [32]f32 = @splat(0.0);
        for (0..hg) |sg| {
            const h = y * hg + sg;
            for (0..32) |ln| for (0..DPL) |d| {
                qv[sg][ln][d] = emuBf16(e.q[(h * e.rows + qs) * R + ln * DPL + d]) * sl2;
                acc[sg][ln][d] = 0.0;
            };
        }
        var c0 = k0;
        while (c0 < k1) : (c0 += BK) {
            const rows = @min(BK, k1 - c0);
            for (0..BK) |r| {
                const src: i32 = if (r < rows) e.sel[qs * W + c0 + r] else -1;
                for (0..R) |d| ks[r * R + d] = if (src >= 0) e.lat[@as(usize, @intCast(src)) * R + d] else 0;
                kid[r] = src;
            }
            for (0..hg) |sg| {
                if (!lean) {
                    for (0..rows) |r| {
                        if (kid[r] < 0) continue;
                        var part: [32]f32 = undefined;
                        for (0..32) |ln| {
                            var s: f32 = 0.0;
                            for (0..DPL) |d| s = @mulAdd(f32, qv[sg][ln][d], emuBf16(ks[r * R + ln * DPL + d]), s);
                            part[ln] = s;
                        }
                        const s = emuSimdSum(&part);
                        const mn = @max(m[sg], s);
                        const fac = std.math.exp2(m[sg] - mn);
                        const p = std.math.exp2(s - mn);
                        l[sg] = l[sg] * fac + p;
                        for (0..32) |ln| for (0..DPL) |d| {
                            acc[sg][ln][d] = @mulAdd(f32, p, emuBf16(ks[r * R + ln * DPL + d]), acc[sg][ln][d] * fac);
                        };
                        m[sg] = mn;
                    }
                } else {
                    const GK = 4;
                    var r0: usize = 0;
                    while (r0 < rows) : (r0 += GK) {
                        var sc: [GK]f32 = undefined;
                        for (0..GK) |u| {
                            var part: [32]f32 = undefined;
                            for (0..32) |ln| {
                                var s: f32 = 0.0;
                                for (0..DPL) |d| s = @mulAdd(f32, qv[sg][ln][d], emuBf16(ks[(r0 + u) * R + ln * DPL + d]), s);
                                part[ln] = s;
                            }
                            sc[u] = emuSimdSum(&part);
                        }
                        for (0..GK) |u| {
                            const r = r0 + u;
                            if (r >= rows or kid[r] < 0) continue;
                            const s = sc[u];
                            const mn = @max(m[sg], s);
                            const fac = std.math.exp2(m[sg] - mn);
                            const p = std.math.exp2(s - mn);
                            l[sg] = l[sg] * fac + p;
                            if (fac == 1.0) {
                                for (0..32) |ln| for (0..DPL) |d| {
                                    acc[sg][ln][d] = @mulAdd(f32, p, emuBf16(ks[r * R + ln * DPL + d]), acc[sg][ln][d]);
                                };
                            } else {
                                for (0..32) |ln| for (0..DPL) |d| {
                                    acc[sg][ln][d] = @mulAdd(f32, p, emuBf16(ks[r * R + ln * DPL + d]), acc[sg][ln][d] * fac);
                                };
                            }
                            m[sg] = mn;
                        }
                    }
                }
            }
        }
        for (0..hg) |sg| {
            const h = y * hg + sg;
            const pb = (h * e.rows + qs) * NS + j;
            for (0..32) |ln| for (0..DPL) |d| {
                e.pacc[pb * R + ln * DPL + d] = acc[sg][ln][d];
            };
            e.pm[pb] = m[sg];
            e.pl[pb] = l[sg];
        }
    };
}

test "glm5_next dsa_decode: dsa_split_form lean's partials (pacc, pm, pl) are the current form's bit for bit, rows 1 .. 4, edge selections and values (CPU emulator of both texts)" {
    const a = testing.allocator;
    const R = latent;
    const lanes: usize = 4096;
    var prng = std.Random.DefaultPrng.init(0x6c65616e);
    const rng = prng.random();
    const lat = try a.alloc(u16, lanes * R);
    defer a.free(lat);
    const bf = struct {
        fn of(x: f32) u16 {
            const b: u32 = @bitCast(x);
            return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
        }
    };
    for (lat, 0..) |*x, i| x.* = bf.of(rng.floatNorm(f32) * @as(f32, if (i / R % 97 == 5) 40.0 else 1.0));
    // Rows 0 .. 7 of the lane: one repeated value (equal scores: the factor-1 update every key after the first).
    for (0..8) |r| for (0..R) |d| {
        lat[r * R + d] = bf.of(0.25);
    };
    const Case = struct { heads: usize, width: usize, kind: u8, qmul: f32 };
    const cases = [_]Case{
        .{ .heads = 16, .width = 2051, .kind = 's', .qmul = 0.5 }, // the decode lane's selection (2,048 + the tail)
        .{ .heads = 32, .width = 300, .kind = 's', .qmul = 3.0 }, // large scores: the max moves often
        .{ .heads = 32, .width = 37, .kind = 'r', .qmul = 1.0 }, // repeated key: factor 1 throughout
        .{ .heads = 32, .width = 13, .kind = 'z', .qmul = 0.0 }, // zero queries: every score 0; W < slices (empty slices)
        .{ .heads = 32, .width = 1, .kind = 's', .qmul = 1.0 }, // one key
        .{ .heads = 32, .width = 70, .kind = 'n', .qmul = 1.0 }, // every slot unused (-1)
        .{ .heads = 32, .width = 257, .kind = 'h', .qmul = 1.0 }, // half the slots unused, scattered
        .{ .heads = 32, .width = 64, .kind = 'i', .qmul = 8.0 }, // increasing scores along the selection
    };
    for (cases) |cs| for (1..max_rows + 1) |rows| {
        const q = try a.alloc(u16, cs.heads * rows * R);
        defer a.free(q);
        for (q, 0..) |*x, i| x.* = if (cs.kind == 'i') bf.of(cs.qmul * @as(f32, @floatFromInt(i % R)) / R) else bf.of(cs.qmul * rng.floatNorm(f32));
        const sel = try a.alloc(i32, rows * cs.width);
        defer a.free(sel);
        for (0..rows) |row| for (0..cs.width) |j| {
            sel[row * cs.width + j] = switch (cs.kind) {
                'r' => 3,
                'n' => -1,
                'h' => if (rng.boolean()) @intCast(8 + rng.uintLessThan(usize, lanes - 8)) else -1,
                'i' => @intCast(8 + j),
                else => if (j + 3 >= cs.width and cs.width > 3 and j + 3 - cs.width >= row) -1 else @intCast(8 + rng.uintLessThan(usize, lanes - 8)),
            };
        };
        var outs: [2][3][]f32 = undefined;
        for (&outs, 0..) |*o, f| {
            o[0] = try a.alloc(f32, cs.heads * rows * slices * R);
            o[1] = try a.alloc(f32, cs.heads * rows * slices);
            o[2] = try a.alloc(f32, cs.heads * rows * slices);
            emulate(.{ .heads = cs.heads, .rows = rows, .width = cs.width, .q = q, .lat = lat, .sel = sel, .scale = 1.0 / 16.0, .pacc = o[0], .pm = o[1], .pl = o[2] }, f == 1);
        }
        defer for (&outs) |*o| for (o) |x| a.free(x);
        for (0..3) |i| try testing.expectEqualSlices(u32, @as([*]const u32, @ptrCast(outs[0][i].ptr))[0..outs[0][i].len], @as([*]const u32, @ptrCast(outs[1][i].ptr))[0..outs[1][i].len]);
    };
}
