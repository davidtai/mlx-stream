//! L5 (design-decode-fusion-20261007 2c): MiMo's fused MXFP4 decode MoE over the three slot banks (base, ext,
//! transient), replacing the decode lane's per-bank gather_qmm gate + up + SiLU chain + gather_qmm down (9 dispatches
//! per bank group) and the 4-op weighted sum:
//!   - `mimo_moe_mx_gateup_banked`: gate + up + MiMo's SiLU (f32 `g * sigmoid(g)` of the bf16 gate, bf16 once, times
//!     the bf16 up) for every routed pair, ids packed `bank << 24 | slot row` (the shared scheduler's banked waves);
//!   - `mimo_moe_mx_down_banked`: the per-pair down rows `[P, H]` bf16 (no score fold: the arch's combine follows, and
//!     a miss part's rows replace a pair's rows before it);
//!   - `mimo_moe_combine`: the arch's f32 score-weighted sum over the TOPK routed positions in router order, bf16 once.
//! Donor: mlx-serve origin/mimo-v2 @ 7a9e392f src/moe_fp4.zig (`mlxserve_moe_fp4_gateup_mx` / `_downred_mx`).
//! Route: `Route.hitpass` (default: the fused texts with the hit pass), `.fused` (the texts, host-routed hit wave: the
//! hit pass's control) or `.stock` (the grouped gather_qmm route: L5's control), fixed at construction.
//! Class: gate/up/down rounding-class vs gather_qmm (accumulation order); combine exact by construction.
//! K1 (mimo/k1-moe-bw): `KernelFamily.k1` binds the k1 texts (`mimo_moe_k1_gateup` / `mimo_moe_k1_down`) at every wave
//! width: per member row the banked texts' arithmetic bit for bit (same lane -> code mapping, products, partial sums,
//! fold, simd_sum, epilogue), with two FP4 codes decoded per integer op, the union rule by simdgroup vote (MAXM 1 at
//! P <= 8, MAXM 4 from P >= 9), and each chunk's codes and scales loaded one chunk ahead. `.banked` is the control.
//! `k1_plan_fma` (the same texts, FMA 1: one FMA per weight, the block scale folded once per lane block) is a
//! rounding-class alternative the m4attr family times; no route binds it (the routed plan is FMA 0 by comptime check).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");
const quant = @import("../../sdk_ext.zig").quant;
const bank_mod = @import("bank.zig");

const T = mlx.mlx_array;
const Proj = quant.GatherQmm.Arrays(T);
pub const Arrays = quant.BankArrays(Proj);
pub const n_banks: usize = std.meta.fieldNames(@import("../../sdk_ext.zig").expert.BankKind).len;

/// The decode MoE route, fixed at construction (`settings.Config.decode_moe`). `.hitpass` (HITPASS, design-decode-fusion
/// 2d's DEVROUTE): the fused texts, each one-token call's hit wave dispatched from the layer's device-resident LUT
/// before the host reads the routed ids, only the miss parts built on the host; `"decode_moe": "fused"` the same texts
/// with the hit wave built after the host's route (its control); `"decode_moe": "stock"` L5's control.
pub const Route = enum { fused, stock, hitpass };

/// K1: the decode expert kernel family the fused routes (`.fused`, `.hitpass`) bind, fixed at construction
/// (`settings.Config.decode_moe_kernel`): `.banked` (the control: the banked texts at P <= 8, the union texts from P >= 9)
/// or `.k1` (the k1 texts at every width, bit-equal per member row); `.invalid` marks a setting outside them (refused by
/// name in `validate`).
pub const KernelFamily = enum { banked, k1, invalid };

/// MiMo's routed geometry the texts are instantiated for (validated once, `Fused.init`).
pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
pub const top_k: u32 = 8;
pub const group_size: u32 = 32;
/// Output rows per simdgroup, simdgroups per threadgroup (the donor's ROWS / SGS).
const rows_per_simdgroup: u32 = 4;
const simdgroups: u32 = 2;
/// The widest wave the shared scheduler hands a banked text (one route's ids), and the widest combine: M <= 4 tokens
/// (the MTP verify rows), each TOPK pairs.
pub const max_pairs: u32 = @import("../../sdk_ext.zig").expert.policy.max_route_ids;
pub const max_tokens: u32 = 4;
/// A packed id's row field (`bank << 24 | row`).
pub const row_bits: u5 = 24;

comptime {
    if (hidden % 512 != 0 or inter % 512 != 0) @compileError("mimo moe: K and I must be multiples of 512 (16 values x 32 lanes)");
    if (hidden % (rows_per_simdgroup * simdgroups) != 0 or inter % (rows_per_simdgroup * simdgroups) != 0) @compileError("mimo moe: N and H must be multiples of ROWS x SGS");
    if (max_tokens * top_k > max_pairs) @compileError("mimo moe: the verify rows' pairs must fit one route");
    // K1: the union vote covers a wave in two 32-lane ballots.
    if (max_pairs > 64) @compileError("mimo moe: the k1 union vote takes at most 64 pairs per wave");
}

/// The texts (pinned by `kernels/manifest-moe.json`; `manifest_sha256` pins the manifest).
pub const texts = struct {
    pub const header_mx = @embedFile("kernels/header_mimo_moe_mx.metal");
    pub const header_combine = @embedFile("kernels/header_mimo_moe_combine.metal");
    pub const gate_up = @embedFile("kernels/mimo_moe_mx_gateup_banked.metal");
    pub const down = @embedFile("kernels/mimo_moe_mx_down_banked.metal");
    /// M4 lever 1 (mimo/m4-attribution): the served texts. Each distinct slot of a wave is read once by its leader pair
    /// for up to `max_tokens` member pairs (the verify's rows that routed to it); per pair the banked texts' arithmetic.
    pub const header_union = @embedFile("kernels/header_mimo_moe_union.metal");
    pub const gate_up_union = @embedFile("kernels/mimo_moe_mx_gateup_union.metal");
    pub const down_union = @embedFile("kernels/mimo_moe_mx_down_union.metal");
    pub const combine = @embedFile("kernels/mimo_moe_combine.metal");
    /// K1 (mimo/k1-moe-bw): the k1 texts and their helpers (appended to header_mimo_moe_mx.metal).
    pub const header_k1 = @embedFile("kernels/header_mimo_moe_k1.metal");
    pub const gate_up_k1 = @embedFile("kernels/mimo_moe_k1_gateup.metal");
    pub const down_k1 = @embedFile("kernels/mimo_moe_k1_down.metal");
    pub const manifest = @embedFile("kernels/manifest-moe.json");
};
pub const manifest_sha256 = "58c8a5d1817f6210ff0d3ab29ff55753b5f2be87658fad1ce71eb463827dfe41";

/// A packed slot id's (bank, row), as the texts read it (`mimo_slot_bank` / `mimo_slot_row`).
pub fn slotOf(id: u32) struct { bank: u32, row: u32 } {
    return .{ .bank = id >> row_bits, .row = id & ((@as(u32, 1) << row_bits) - 1) };
}

/// The union texts' header: the banked header, then the union helpers.
const header_mx_union: [:0]const u8 = texts.header_mx ++ texts.header_union;
/// The k1 texts' header: the banked header, then the k1 helpers.
const header_mx_k1: [:0]const u8 = texts.header_mx ++ texts.header_k1;

/// A k1 text's launch geometry: output rows per simdgroup (`ROWS`), the widest union one leader serves (`MAXM`: a
/// slot's pairs beyond it take further leaders, the union rule), simdgroups per threadgroup, and the dot form (`FMA`):
/// 0 = the banked order (bit-equal per member row, the routed form), 1 = one FMA per weight with the block scale folded
/// once per lane block (rounding-class; the m4attr family times it, no route binds it: `k1_plan` is FMA 0 by comptime
/// check).
pub const K1Geometry = struct {
    rows: u32,
    maxm: u32,
    simdgroups: u32 = 2,
    fma: u32 = 0,

    fn check(g: K1Geometry, rows_out: u32) !void {
        if (g.rows == 0 or g.simdgroups == 0 or g.maxm == 0 or g.maxm > max_tokens or g.fma > 1) return error.MimoK1Geometry;
        if (rows_out % (g.rows * g.simdgroups) != 0) return error.MimoK1Geometry;
    }
};

/// The k1 family's geometry per wave, derived from the real shapes (mimo/k1-moe-bw REPORT §2): a one-token wave (P <= 8,
/// every slot distinct) keeps the banked texts' grid (4 rows per simdgroup, 2 simdgroups, the chunk's activations
/// converted once for 8 row decodes); a verify wave (P >= 9) decodes each row of a slot once and dots it with up to
/// `max_tokens` members, so gate/up holds 2 rows' decoded codes (2 x 2 x 8 half2) to stay near 100 registers, down 4.
pub const K1Plan = struct {
    serial_gate_up: K1Geometry = .{ .rows = 4, .maxm = 1 },
    serial_down: K1Geometry = .{ .rows = 4, .maxm = 1 },
    union_gate_up: K1Geometry = .{ .rows = 2, .maxm = max_tokens },
    union_down: K1Geometry = .{ .rows = 4, .maxm = max_tokens },

    /// Construction: every geometry tiles its projection's rows into whole threadgroups and names a union the texts
    /// take (1 .. max_tokens members); the one-token waves serve one member (all their slots are distinct).
    pub fn validate(p: K1Plan) !void {
        for ([_]K1Geometry{ p.serial_gate_up, p.union_gate_up }) |g| try g.check(inter);
        for ([_]K1Geometry{ p.serial_down, p.union_down }) |g| try g.check(hidden);
        if (p.serial_gate_up.maxm != 1 or p.serial_down.maxm != 1) return error.MimoK1Geometry;
    }
};
pub const k1_plan: K1Plan = .{};
/// The FMA alternative at the routed geometry (rounding-class; timed by the m4attr family and checked against the f32
/// truth in `mimo moe GPU`, never routed).
pub const k1_plan_fma: K1Plan = .{
    .serial_gate_up = .{ .rows = 4, .maxm = 1, .fma = 1 },
    .serial_down = .{ .rows = 4, .maxm = 1, .fma = 1 },
    .union_gate_up = .{ .rows = 2, .maxm = max_tokens, .fma = 1 },
    .union_down = .{ .rows = 4, .maxm = max_tokens, .fma = 1 },
};
comptime {
    // the routed k1 family is the exact form (bit-equal per member row to the banked texts)
    for ([_]K1Geometry{ k1_plan.serial_gate_up, k1_plan.serial_down, k1_plan.union_gate_up, k1_plan.union_down }) |g| {
        if (g.fma != 0) @compileError("mimo moe k1: the routed plan must be the exact (FMA 0) form");
    }
}

comptime {
    // the k1 texts' chunk walk (as the banked texts'): 512 values per chunk = 16 codes x 32 lanes, a lane's 16 codes
    // under one scale, whole scale groups per chunk
    if (group_size % 16 != 0 or 512 % group_size != 0) @compileError("mimo moe k1: a lane's 16 codes share one scale; a chunk holds whole groups");
}

/// Which gate/up + down texts a `Fused` binds per wave width (construction): `.union` (served): a wave of one token's
/// pairs (P <= TOPK: every slot distinct, the serial decode's every call) keeps the banked texts and their exact launch
/// configuration (the control's, byte for byte); a wider wave (the verify's rows, P > TOPK, slots shared across rows)
/// takes the union texts, one read per distinct slot. `.pair`: the banked texts at every width (the union texts'
/// bitwise reference in the device self-check). `.k1` (K1): the k1 texts at every width (`K1Plan`).
pub const Texts = enum { @"union", pair, k1 };

/// The widest wave that keeps the banked texts on the served route: one token's routed pairs.
pub const union_min_pairs: u32 = top_k + 1;

/// The fused route's kernels and its launch configurations, prebound per wave width at construction.
pub const Fused = struct {
    k_gate_up_pair: mlx.mlx_fast_metal_kernel = .{},
    k_down_pair: mlx.mlx_fast_metal_kernel = .{},
    k_gate_up_union: mlx.mlx_fast_metal_kernel = .{},
    k_down_union: mlx.mlx_fast_metal_kernel = .{},
    k_gate_up_k1: mlx.mlx_fast_metal_kernel = .{},
    k_down_k1: mlx.mlx_fast_metal_kernel = .{},
    k_combine: mlx.mlx_fast_metal_kernel = .{},
    /// Index P - 1 (pairs) / n - 1 (tokens): the texts and the launch configuration of a wave of P pairs.
    gate_up_k: [max_pairs]mlx.mlx_fast_metal_kernel = @splat(.{}),
    down_k: [max_pairs]mlx.mlx_fast_metal_kernel = @splat(.{}),
    gate_up_cfg: [max_pairs]mlx.mlx_fast_metal_kernel_config = @splat(.{}),
    down_cfg: [max_pairs]mlx.mlx_fast_metal_kernel_config = @splat(.{}),
    combine_cfg: [max_tokens]mlx.mlx_fast_metal_kernel_config = @splat(.{}),

    /// Construction boundary: the slot geometry the texts are instantiated for, and every bank's rows addressable by
    /// a packed id. No device work (the texts compile at their first dispatch; the self-check is the pre-ship gate).
    pub fn init(geometry: bank_mod.Geometry, widest_bank_rows: u32) !Fused {
        return initTexts(geometry, widest_bank_rows, .@"union");
    }

    /// K1: the fused route's texts of a kernel family (`settings.Config.decode_moe_kernel`): `.banked` is `init`'s,
    /// `.k1` the k1 texts at every width; `.invalid` is refused by name.
    pub fn initFamily(geometry: bank_mod.Geometry, widest_bank_rows: u32, family: KernelFamily) !Fused {
        return switch (family) {
            .banked => initTexts(geometry, widest_bank_rows, .@"union"),
            .k1 => initTexts(geometry, widest_bank_rows, .k1),
            .invalid => error.MimoDecodeMoeKernel,
        };
    }

    pub fn initTexts(geometry: bank_mod.Geometry, widest_bank_rows: u32, which: Texts) !Fused {
        return initTextsWith(geometry, widest_bank_rows, which, k1_plan);
    }

    /// `plan`: the k1 texts' geometry (`.k1` only; the served one is `k1_plan`, the m4attr family measures others).
    pub fn initTextsWith(geometry: bank_mod.Geometry, widest_bank_rows: u32, which: Texts, plan: K1Plan) !Fused {
        if (geometry.hidden_size != hidden or geometry.intermediate_size != inter) return error.MimoFusedMoeGeometry;
        if (widest_bank_rows == 0 or widest_bank_rows >= @as(u32, 1) << row_bits) return error.MimoFusedMoeSlotRows;
        if (which == .k1) try plan.validate();
        var self: Fused = .{};
        errdefer self.deinit();
        const gu_in = [_][*:0]const u8{ "x", "g0", "gs0", "u0", "us0", "g1", "gs1", "u1", "us1", "g2", "gs2", "u2", "us2", "ids" };
        const dn_in = [_][*:0]const u8{ "act", "d0", "ds0", "d1", "ds1", "d2", "ds2", "ids" };
        const cb_in = [_][*:0]const u8{ "rows", "w" };
        if (which != .k1) {
            self.k_gate_up_pair = try makeKernel("mimo_moe_mx_gateup_banked", &gu_in, texts.gate_up, texts.header_mx);
            self.k_down_pair = try makeKernel("mimo_moe_mx_down_banked", &dn_in, texts.down, texts.header_mx);
        }
        if (which == .@"union") {
            self.k_gate_up_union = try makeKernel("mimo_moe_mx_gateup_union", &gu_in, texts.gate_up_union, header_mx_union.ptr);
            self.k_down_union = try makeKernel("mimo_moe_mx_down_union", &dn_in, texts.down_union, header_mx_union.ptr);
        }
        if (which == .k1) {
            self.k_gate_up_k1 = try makeKernel("mimo_moe_k1_gateup", &gu_in, texts.gate_up_k1, header_mx_k1.ptr);
            self.k_down_k1 = try makeKernel("mimo_moe_k1_down", &dn_in, texts.down_k1, header_mx_k1.ptr);
        }
        self.k_combine = try makeKernel("mimo_moe_combine", &cb_in, texts.combine, texts.header_combine);
        for (&self.gate_up_k, &self.down_k, &self.gate_up_cfg, &self.down_cfg, 1..) |*gk, *dk, *gu, *dn, p| {
            const pairs: c_int = @intCast(p);
            if (which == .k1) {
                // a one-token wave (every slot distinct) serves one member per leader; a verify wave the union rule
                const gg = if (p >= union_min_pairs) plan.union_gate_up else plan.serial_gate_up;
                const dg = if (p >= union_min_pairs) plan.union_down else plan.serial_down;
                gk.* = self.k_gate_up_k1;
                dk.* = self.k_down_k1;
                gu.* = try config(&.{ pairs, inter }, .{ 32, @intCast(inter / gg.rows), pairs }, .{ 32, @intCast(gg.simdgroups), 1 }, &.{ .{ "ROWS", gg.rows }, .{ "K", hidden }, .{ "N", inter }, .{ "GS", group_size }, .{ "MAXM", gg.maxm }, .{ "FMA", gg.fma } });
                dn.* = try config(&.{ pairs, hidden }, .{ 32, @intCast(hidden / dg.rows), pairs }, .{ 32, @intCast(dg.simdgroups), 1 }, &.{ .{ "ROWS", dg.rows }, .{ "I", inter }, .{ "H", hidden }, .{ "GS", group_size }, .{ "MAXM", dg.maxm }, .{ "FMA", dg.fma } });
                continue;
            }
            const wide = which == .@"union" and p >= union_min_pairs;
            gk.* = if (wide) self.k_gate_up_union else self.k_gate_up_pair;
            dk.* = if (wide) self.k_down_union else self.k_down_pair;
            const gu_t = [_]TArg{ .{ "ROWS", rows_per_simdgroup }, .{ "K", hidden }, .{ "N", inter }, .{ "GS", group_size }, .{ "MAXM", max_tokens } };
            const dn_t = [_]TArg{ .{ "ROWS", rows_per_simdgroup }, .{ "I", inter }, .{ "H", hidden }, .{ "GS", group_size }, .{ "MAXM", max_tokens } };
            // the banked texts keep the control's template arguments (no MAXM)
            const n_t: usize = if (wide) gu_t.len else gu_t.len - 1;
            gu.* = try config(&.{ pairs, inter }, .{ 32, @intCast(inter / rows_per_simdgroup), pairs }, .{ 32, simdgroups, 1 }, gu_t[0..n_t]);
            dn.* = try config(&.{ pairs, hidden }, .{ 32, @intCast(hidden / rows_per_simdgroup), pairs }, .{ 32, simdgroups, 1 }, dn_t[0..n_t]);
        }
        for (&self.combine_cfg, 1..) |*cb, n| {
            const tokens: c_int = @intCast(n);
            cb.* = try config(&.{ tokens, hidden }, .{ hidden, tokens, 1 }, .{ 256, 1, 1 }, &.{ .{ "TOPK", top_k }, .{ "H", hidden } });
        }
        return self;
    }

    pub fn deinit(self: *Fused) void {
        for ([_]mlx.mlx_fast_metal_kernel{ self.k_gate_up_pair, self.k_down_pair, self.k_gate_up_union, self.k_down_union, self.k_gate_up_k1, self.k_down_k1, self.k_combine }) |k| if (k.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_free(k);
        };
        for (self.gate_up_cfg, self.down_cfg) |c, d| for ([_]mlx.mlx_fast_metal_kernel_config{ c, d }) |x| if (x.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(x);
        };
        for (self.combine_cfg) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        self.* = .{};
    }

    /// `x` bf16 [P, H] (each pair's token row), `ids` u32 [P] packed; bf16 [P, I] (the SwiGLU rows). The banks are
    /// passed whole (the texts' row-contiguous inputs: a slice would be copied whole per call).
    pub fn gateUp(self: *const Fused, g: *ops.MlxOps, x: T, ids: T, banks: *const [n_banks]Arrays) !T {
        const p: usize = @intCast(g.shapeOf(x).d[0]);
        var in: [14]T = undefined;
        in[0] = x;
        for (banks, 0..) |b, i| {
            in[1 + 4 * i] = b.gate.w;
            in[2 + 4 * i] = b.gate.scales.?;
            in[3 + 4 * i] = b.up.w;
            in[4 + 4 * i] = b.up.scales.?;
        }
        in[13] = ids;
        return apply(g, self.gate_up_k[p - 1], &in, self.gate_up_cfg[p - 1]);
    }

    /// `h` bf16 [P, I], `ids` as `gateUp`; bf16 [P, H]: each pair's down row, rounded once.
    pub fn down(self: *const Fused, g: *ops.MlxOps, h: T, ids: T, banks: *const [n_banks]Arrays) !T {
        const p: usize = @intCast(g.shapeOf(h).d[0]);
        var in: [8]T = undefined;
        in[0] = h;
        for (banks, 0..) |b, i| {
            in[1 + 2 * i] = b.down.w;
            in[2 + 2 * i] = b.down.scales.?;
        }
        in[7] = ids;
        return apply(g, self.down_k[p - 1], &in, self.down_cfg[p - 1]);
    }

    /// `routed` bf16 [n, TOPK, H] in router order (the scheduler's join), `weights` f32 [n, TOPK]; bf16 [n, H].
    pub fn combineRows(self: *const Fused, g: *ops.MlxOps, routed: T, weights: T) !T {
        const n: usize = @intCast(g.shapeOf(routed).d[0]);
        const rows = try g.reshape(routed, &.{ @intCast(n * top_k), @intCast(hidden) });
        return apply(g, self.k_combine, &.{ rows, weights }, self.combine_cfg[n - 1]);
    }
};

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MimoFusedMoeKernel;
    return k;
}

/// A text's integer template argument (name, value).
const TArg = struct { [*:0]const u8, u32 };

fn config(out_shape: []const c_int, grid: [3]c_int, group: [3]c_int, ints: []const TArg) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, group[0], group[1], group[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], @intCast(t[1])));
    return c;
}

fn apply(g: *ops.MlxOps, kernel: mlx.mlx_fast_metal_kernel, inputs: []const T, cfg: mlx.mlx_fast_metal_kernel_config) !T {
    const in = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, in, cfg, g.s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}

// ── Host references of the texts' arithmetic (CPU tests; the device self-check below compares the texts) ──

/// MLX's f32 sigmoid (unary_ops.h `Sigmoid`), as `mimo_sigmoid_f32`. Results flush subnormals to zero, as Metal's f32
/// arithmetic does (a gate below about -87.3: a +0 sigmoid, a signed-zero SiLU).
pub fn sigmoidRef(x: f32) f32 {
    const y = ftz(1 / (1 + @exp(@abs(x))));
    return if (x < 0) y else 1 - y;
}

fn ftz(v: f32) f32 {
    return if (@abs(v) < std.math.floatMin(f32)) std.math.copysign(@as(f32, 0), v) else v;
}

pub fn bf16Round(v: f32) f32 {
    const u: u32 = @bitCast(v);
    return @bitCast((u +% (0x7fff + ((u >> 16) & 1))) & 0xffff0000);
}

/// The gate/up text's epilogue: `T(gf * sigmoid(gf)) * T(u)`, gf the bf16-rounded gate (f32 SiLU, one rounding, then
/// a bf16 multiply: the product of two bf16 values rounded once).
pub fn siluMulRef(gate: f32, up: f32) f32 {
    const gf = bf16Round(gate);
    return bf16Round(bf16Round(ftz(gf * sigmoidRef(gf))) * bf16Round(up));
}

/// The combine text: per token t and column h, products `f32(row) * w` each added to 0, folded k = 0..TOPK-1.
pub fn combineRef(rows: []const f32, w: []const f32, n: usize, h: usize, out: []f32) void {
    for (0..n) |t| for (0..h) |c| {
        var acc: f32 = rows[(t * top_k) * h + c] * w[t * top_k] + 0.0;
        for (1..top_k) |k| {
            const p: f32 = rows[(t * top_k + k) * h + c] * w[t * top_k + k] + 0.0;
            acc = p + acc;
        }
        out[t * h + c] = bf16Round(acc);
    };
}

// ── K1: host references of the k1 and banked texts' per-lane arithmetic (CPU proofs; the device self-check runs the
// texts). IEEE f32 in the texts' op order: every product of a bf16 activation and an e2m1 code is exact in f32, so
// contraction into FMA cannot change a result; the order of additions is what the texts fix. ──

/// The half bits `mimo_e2m1` builds for code `c` (header_mimo_moe_mx.metal).
pub fn e2m1Bits(c: u32) u16 {
    return @intCast(((c & 0x7) << 9) | ((c & 0x8) << 12));
}

/// `mimo_k1_decode8` (header_mimo_moe_k1.metal): half2 j holds codes j (lo) and j + 4 (hi) of `w`.
pub fn k1Decode8(w: u32) [4]u32 {
    return .{
        ((w << 9) & 0x0E000E00) | ((w << 12) & 0x80008000),
        ((w << 5) & 0x0E000E00) | ((w << 8) & 0x80008000),
        ((w << 1) & 0x0E000E00) | ((w << 4) & 0x80008000),
        ((w >> 3) & 0x0E000E00) | (w & 0x80008000),
    };
}

fn halfF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

fn loHalf(w: u32) f32 {
    return halfF32(@truncate(w));
}

fn hiHalf(w: u32) f32 {
    return halfF32(@truncate(w >> 16));
}

fn bf16F32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// `mimo_fp4_scale`: E8M0 byte to f32 (x 2^-8), subnormals flushed as Metal's f32 arithmetic does.
pub fn scaleRef(b: u32) f32 {
    const v: f32 = @bitCast(if (b == 0) @as(u32, 0x400000) else b << 23);
    return ftz(ftz(v) * 0.00390625);
}

/// `mimo_fp4_dot16` (banked texts): codes 0..7 of `wx` into a, 0..7 of `wy` into b, each `+= x[j] v_j + x[j+1] v_{j+1}`.
fn bankedDot16(wx: u32, wy: u32, x: *const [16]f32) f32 {
    const e = struct {
        fn v(w: u32, j: u5) f32 {
            return halfF32(e2m1Bits((w >> j) & 0xF));
        }
    }.v;
    var a: f32 = 0;
    var b: f32 = 0;
    var j: u5 = 0;
    while (j < 8) : (j += 2) {
        a += x[j] * e(wx, 4 * j) + x[j + 1] * e(wx, 4 * j + 4);
        b += x[8 + j] * e(wy, 4 * j) + x[9 + j] * e(wy, 4 * j + 4);
    }
    return a + b;
}

/// `mimo_k1_dot16` (k1 texts): the same products and partial sums over the decoded half2s.
fn k1Dot16(h: *const [8]u32, x: *const [16]f32) f32 {
    var a: f32 = 0;
    var b: f32 = 0;
    a += x[0] * loHalf(h[0]) + x[1] * loHalf(h[1]);
    b += x[8] * loHalf(h[4]) + x[9] * loHalf(h[5]);
    a += x[2] * loHalf(h[2]) + x[3] * loHalf(h[3]);
    b += x[10] * loHalf(h[6]) + x[11] * loHalf(h[7]);
    a += x[4] * hiHalf(h[0]) + x[5] * hiHalf(h[1]);
    b += x[12] * hiHalf(h[4]) + x[13] * hiHalf(h[5]);
    a += x[6] * hiHalf(h[2]) + x[7] * hiHalf(h[3]);
    b += x[14] * hiHalf(h[6]) + x[15] * hiHalf(h[7]);
    return a + b;
}

/// One projection's slot record as the texts address it: codes u32 [rows, k / 8], E8M0 scales [rows, k / 32]; `flip` /
/// `bump` give each slot id of a test its own view of one record (codes xor `flip`, scales + `bump`).
pub const HostProj = struct {
    words: []const u32,
    scales: []const u8,
    k: u32,
    flip: u32 = 0,
    bump: u8 = 0,

    fn word(p: HostProj, i: usize) u32 {
        return p.words[i] ^ p.flip;
    }
    fn scale(p: HostProj, i: usize) u32 {
        return p.scales[i] +% p.bump;
    }
};

/// One lane's partial of row `n` as the banked texts compute it: per 512-value chunk k, the uint2 `n * KW2 + lane +
/// k / 16`, the scale `n * KG + kk / GS` (kk = k + 16 lane), the activations `float(x[kk + i])`, `mimo_fp4_dot16`.
pub fn bankedLane(p: HostProj, n: u32, x: []const u16, lane: u32) f32 {
    const kw2 = p.k / 16;
    const kg = p.k / group_size;
    var acc: f32 = 0;
    var k: u32 = 0;
    while (k < p.k) : (k += 512) {
        const kk = k + lane * 16;
        const wi = n * kw2 + lane + k / 16;
        var xr: [16]f32 = undefined;
        for (&xr, 0..) |*v, i| v.* = bf16F32(x[kk + i]);
        acc += bankedDot16(p.word(2 * wi), p.word(2 * wi + 1), &xr) * scaleRef(p.scale(n * kg + kk / group_size));
    }
    return acc;
}

/// One lane's partial of row `n` as the k1 texts compute it: codes from `gw = n * KW2 + lane` at `+ c * CW2`, the scale
/// from `gsc = n * KG + 16 lane / GS` at `+ c * CG`, the activations by `mimo_k1_x16` (two 16-byte loads, each 32-bit
/// word's halves in order), `mimo_k1_decode16`, `mimo_k1_dot16`, chunks in order.
pub fn k1Lane(p: HostProj, n: u32, x: []const u16, lane: u32) f32 {
    const kw2 = p.k / 16;
    const kg = p.k / group_size;
    const cw2: u32 = 512 / 16;
    const cgs: u32 = 512 / group_size;
    const gw = n * kw2 + lane;
    const gsc = n * kg + (lane * 16) / group_size;
    var acc: f32 = 0;
    for (0..p.k / 512) |ci| {
        const c: u32 = @intCast(ci);
        const wi = gw + c * cw2;
        var h: [8]u32 = undefined;
        h[0..4].* = k1Decode8(p.word(2 * wi));
        h[4..8].* = k1Decode8(p.word(2 * wi + 1));
        const kk = c * 512 + lane * 16;
        var xf: [16]f32 = undefined;
        for (0..8) |i| {
            const w: u32 = @as(u32, x[kk + 2 * i]) | (@as(u32, x[kk + 2 * i + 1]) << 16);
            xf[2 * i] = bf16F32(@truncate(w));
            xf[2 * i + 1] = bf16F32(@truncate(w >> 16));
        }
        acc += k1Dot16(&h, &xf) * scaleRef(p.scale(gsc + c * cgs));
    }
    return acc;
}

/// The FMA alternative's lane partial (`mimo_k1_dot16_fma`, FMA 1): per chunk the 16 products in one FMA chain in code
/// order (p = x0 v0, then p = fma(x_j, v_j, p)), then acc = fma(p, scale, acc). Host IEEE (subnormal intermediates kept).
pub fn k1LaneFma(p: HostProj, n: u32, x: []const u16, lane: u32) f32 {
    const kw2 = p.k / 16;
    const kg = p.k / group_size;
    const gw = n * kw2 + lane;
    const gsc = n * kg + (lane * 16) / group_size;
    var acc: f32 = 0;
    for (0..p.k / 512) |ci| {
        const c: u32 = @intCast(ci);
        const wi = gw + c * (512 / 16);
        var h: [8]u32 = undefined;
        h[0..4].* = k1Decode8(p.word(2 * wi));
        h[4..8].* = k1Decode8(p.word(2 * wi + 1));
        const kk = c * 512 + lane * 16;
        var q: f32 = 0;
        for (0..16) |i| {
            const word = h[(i / 8) * 4 + i % 4];
            const v = if (i % 8 < 4) loHalf(word) else hiHalf(word);
            const xv = bf16F32(x[kk + i]);
            q = if (i == 0) xv * v else @mulAdd(f32, xv, v, q);
        }
        acc = @mulAdd(f32, q, scaleRef(p.scale(gsc + c * (512 / group_size))), acc);
    }
    return acc;
}

/// A fixed simd_sum order (any order serves: the device sums the same 32 lane values in its own order for both texts).
fn simdSumRef(lanes: *const [32]f32) f32 {
    var v = lanes.*;
    var w: usize = 16;
    while (w >= 1) : (w /= 2) {
        for (0..w) |i| v[i] = v[i] + v[i + w];
    }
    return v[0];
}

/// `mimo_k1_members` on the host: the two 32-lane votes (bit q: pair q names pair e's slot), the rank below e, the
/// next `maxm` set bits from e. maxm 1: every pair alone.
pub fn k1Members(ids: []const u32, e: usize, maxm: u32, mem: []u32) u32 {
    if (maxm == 1) {
        mem[0] = @intCast(e);
        return 1;
    }
    const np = ids.len;
    var lo: u32 = 0;
    var hi: u32 = 0;
    for (0..32) |lane| {
        if (lane < np and ids[@min(lane, np - 1)] == ids[e]) lo |= @as(u32, 1) << @intCast(lane);
        if (lane + 32 < np and ids[@min(lane + 32, np - 1)] == ids[e]) hi |= @as(u32, 1) << @intCast(lane);
    }
    const e32: u32 = @intCast(e);
    const rank: u32 = if (e32 < 32) @popCount(lo & ((@as(u32, 1) << @intCast(e32)) - 1)) else @as(u32, @popCount(lo)) + @popCount(hi & ((@as(u32, 1) << @intCast(e32 - 32)) - 1));
    if (rank % maxm != 0) return 0;
    if (e32 < 32) {
        lo &= ~((@as(u32, 1) << @intCast(e32)) - 1);
    } else {
        lo = 0;
        hi &= ~((@as(u32, 1) << @intCast(e32 - 32)) - 1);
    }
    var nm: u32 = 0;
    for (0..maxm) |_| {
        if (lo != 0) {
            mem[nm] = @ctz(lo);
            nm += 1;
            lo &= lo - 1;
        } else if (hi != 0) {
            mem[nm] = 32 + @as(u32, @ctz(hi));
            nm += 1;
            hi &= hi - 1;
        }
    }
    return nm;
}

// ── CPU tests ──

const testing = std.testing;

fn hex(digest: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(digest, .lower);
}

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return hex(d);
}

test "mimo moe registry: the manifest pins every embedded text and the pin is the manifest's sha256" {
    try testing.expectEqualStrings(manifest_sha256, &sha(texts.manifest));
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, texts.manifest, .{});
    defer parsed.deinit();
    const m = parsed.value.object;
    const Pinned = struct { file: []const u8, text: []const u8 };
    const pinned = [_]Pinned{
        .{ .file = "header_mimo_moe_mx.metal", .text = texts.header_mx },
        .{ .file = "header_mimo_moe_combine.metal", .text = texts.header_combine },
        .{ .file = "mimo_moe_mx_gateup_banked.metal", .text = texts.gate_up },
        .{ .file = "mimo_moe_mx_down_banked.metal", .text = texts.down },
        .{ .file = "mimo_moe_combine.metal", .text = texts.combine },
        .{ .file = "header_mimo_moe_union.metal", .text = texts.header_union },
        .{ .file = "mimo_moe_mx_gateup_union.metal", .text = texts.gate_up_union },
        .{ .file = "mimo_moe_mx_down_union.metal", .text = texts.down_union },
        .{ .file = "header_mimo_moe_k1.metal", .text = texts.header_k1 },
        .{ .file = "mimo_moe_k1_gateup.metal", .text = texts.gate_up_k1 },
        .{ .file = "mimo_moe_k1_down.metal", .text = texts.down_k1 },
    };
    var seen: usize = 0;
    for (pinned) |p| {
        var found = false;
        for (m.get("headers").?.array.items) |h| if (std.mem.eql(u8, h.object.get("file").?.string, p.file)) {
            try testing.expectEqualStrings(h.object.get("sha256").?.string, &sha(p.text));
            found = true;
        };
        for (m.get("kernels").?.array.items) |k| {
            const s = k.object.get("source").?.object;
            if (std.mem.eql(u8, s.get("file").?.string, p.file)) {
                try testing.expectEqualStrings(s.get("sha256").?.string, &sha(p.text));
                try testing.expect(k.object.get("ensure_row_contiguous").?.bool);
                found = true;
            }
        }
        try testing.expect(found);
        seen += 1;
    }
    try testing.expectEqual(@as(usize, 7), m.get("kernels").?.array.items.len);
    try testing.expectEqual(pinned.len, seen);
}

test "mimo moe slot ids: the texts resolve the scheduler's packed ids to (bank, row) across the three banks" {
    const BankKind = @import("../../sdk_ext.zig").expert.BankKind;
    try testing.expectEqual(@as(usize, 3), n_banks);
    // The scheduler's packing (deepseek_v41_experts.zig gateUpWaveBanked): bank << 24 | row; base 0, ext 1, transient 2.
    const rows = [_]u32{ 0, 1, 7, 123, 255, 4095, (1 << 24) - 1 };
    for ([_]BankKind{ .base, .ext, .transient }) |kind| for (rows) |r| {
        const id = (@as(u32, @backingInt(kind)) << 24) | r;
        const s = slotOf(id);
        try testing.expectEqual(@as(u32, @backingInt(kind)), s.bank);
        try testing.expectEqual(r, s.row);
    };
    // The texts pick the bank arrays by the same field (0 base, 1 ext, anything else transient = bank 2).
    try testing.expect(std.mem.indexOf(u8, texts.header_mx, "return id >> 24;") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header_mx, "id & 0xFFFFFFu") != null);
    try testing.expect(std.mem.indexOf(u8, texts.gate_up, "b == 0u ? g0 : (b == 1u ? g1 : g2)") != null);
    try testing.expect(std.mem.indexOf(u8, texts.down, "b == 0u ? d0 : (b == 1u ? d1 : d2)") != null);
    // Construction refuses geometry the texts are not instantiated for, and banks a packed id cannot address.
    const mimo = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    try testing.expectError(error.MimoFusedMoeGeometry, Fused.init(try bank_mod.Geometry.init(64, 64, 64, 1, 1), 8));
    try testing.expectError(error.MimoFusedMoeSlotRows, Fused.init(mimo, 1 << 24));
    try testing.expectError(error.MimoFusedMoeSlotRows, Fused.init(mimo, 0));
}

/// The union texts' leader rule (`mimo_union_members`, header_mimo_moe_union.metal) on the host: pair `e` of a wave
/// leads when its rank among the pairs naming its slot is a multiple of `max_tokens`, and then takes itself and the
/// next `max_tokens - 1` of them in pair order. Returns the member count (0: not a leader).
pub fn unionMembers(ids: []const u32, e: usize, mem: *[max_tokens]u32) u32 {
    var rank: usize = 0;
    for (ids[0..e]) |q| rank += @intFromBool(q == ids[e]);
    if (rank % max_tokens != 0) return 0;
    var n: u32 = 0;
    for (ids[e..], e..) |q, i| {
        if (n == max_tokens) break;
        if (q == ids[e]) {
            mem[n] = @intCast(i);
            n += 1;
        }
    }
    return n;
}

test "mimo moe union: every pair of a wave is written by exactly one leader of its slot; distinct slots lead alone" {
    var state: u32 = 0x756e696f;
    // Wave shapes: M = 1 (8 distinct slots), the real M = 4 verify (32 pairs over a 24-slot union, each row's 8
    // distinct), the hit pass's misses (every pair base row 0), and the widest route (48 pairs over 9 slots).
    const Shape = struct { pairs: u32, slots: u32, distinct_per_row: bool };
    for ([_]Shape{ .{ .pairs = 8, .slots = 256, .distinct_per_row = true }, .{ .pairs = 32, .slots = 24, .distinct_per_row = true }, .{ .pairs = 32, .slots = 1, .distinct_per_row = false }, .{ .pairs = max_pairs, .slots = 9, .distinct_per_row = false } }) |shape| for (0..200) |_| {
        var ids: [max_pairs]u32 = undefined;
        for (0..shape.pairs) |p| {
            while (true) {
                const bank: u32 = nextU32(&state) % 3;
                const id = (bank << 24) | (nextU32(&state) % shape.slots);
                const id0 = if (shape.slots == 1) @as(u32, 0) else id;
                const row_start = p - p % top_k;
                if (shape.distinct_per_row and std.mem.indexOfScalar(u32, ids[row_start..p], id0) != null) continue;
                ids[p] = id0;
                break;
            }
        }
        const w = ids[0..shape.pairs];
        var written: [max_pairs]u32 = @splat(0);
        var leaders: u32 = 0;
        for (0..shape.pairs) |e| {
            var mem: [max_tokens]u32 = undefined;
            const n = unionMembers(w, e, &mem);
            if (n == 0) continue;
            leaders += 1;
            try testing.expectEqual(@as(u32, @intCast(e)), mem[0]);
            for (mem[0..n], 0..) |q, j| {
                try testing.expectEqual(w[e], w[q]);
                if (j > 0) try testing.expect(q > mem[j - 1]);
                written[q] += 1;
            }
        }
        for (written[0..shape.pairs]) |c| try testing.expectEqual(@as(u32, 1), c);
        // leaders = sum over distinct slots of ceil(count / max_tokens)
        var want: u32 = 0;
        for (0..shape.pairs) |p| if (std.mem.indexOfScalar(u32, w[0..p], w[p]) == null) {
            const c: u32 = @intCast(std.mem.count(u32, w, &.{w[p]}));
            want += (c + max_tokens - 1) / max_tokens;
        };
        try testing.expectEqual(want, leaders);
        // all distinct (M = 1): each pair leads itself alone (the banked texts' one read per pair); the served route
        // keeps the banked texts there anyway (`union_min_pairs`)
        if (shape.pairs == top_k) try testing.expectEqual(shape.pairs, leaders);
        try testing.expectEqual(shape.pairs >= union_min_pairs, shape.pairs > top_k);
    };
    // The texts carry that rule, and per member the banked texts' products, partial sums, fold and epilogues.
    const U = texts.header_union;
    try testing.expect(std.mem.indexOf(u8, U, "if (rank % uint(MAXM) != 0u) return 0;") != null);
    try testing.expect(std.mem.indexOf(u8, U, "for (uint q = e; q < np && nm < MAXM; ++q)") != null);
    try testing.expect(std.mem.indexOf(u8, U, "((cx & 0x7u) << 9) | ((cx & 0x8u) << 12)") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header_mx, "((c & 0x7u) << 9) | ((c & 0x8u) << 12)") != null);
    try testing.expect(std.mem.indexOf(u8, U, "a += x[j] * float(w[j]) + x[j + 1] * float(w[j + 1]);") != null);
    try testing.expect(std.mem.indexOf(u8, U, "b += x[8 + j] * float(w[8 + j]) + x[9 + j] * float(w[9 + j]);") != null);
    for ([_][]const u8{ "y[size_t(e) * N + row0 + r] = T(gf * mimo_sigmoid_f32(gf)) * T(u);", "const float gf = float(T(g));", "simd_sum(ag[r]) * 4194304.0f" }) |frag| try testing.expect(std.mem.indexOf(u8, texts.gate_up, frag) != null);
    for ([_][]const u8{ "y[size_t(mem[m]) * N + row0 + r] = T(gf * mimo_sigmoid_f32(gf)) * T(u);", "const float gf = float(T(g));", "simd_sum(ag[m][r]) * 4194304.0f", "ag[m][r] += mimo_fp4_dot16h(wgr, xr) * sgr;" }) |frag| try testing.expect(std.mem.indexOf(u8, texts.gate_up_union, frag) != null);
    for ([_][]const u8{ "y[size_t(mem[m]) * H + row0 + r] = T(v);", "simd_sum(acc[m][r]) * 4194304.0f", "acc[m][r] += mimo_fp4_dot16h(wdr, xr) * sdr;" }) |frag| try testing.expect(std.mem.indexOf(u8, texts.down_union, frag) != null);
}

test "mimo moe k1 decode: two codes per integer op give mimo_e2m1's half bits for every code at every position" {
    // Shift-and-mask-or with constant masks: every output bit is one input bit or 0, so decode(a | b) = decode(a) |
    // decode(b) and the 16 codes at each of the 8 positions (the one-bit words among them) determine the map.
    try testing.expectEqual([4]u32{ 0, 0, 0, 0 }, k1Decode8(0));
    for (0..8) |pos| for (0..16) |c| {
        const h = k1Decode8(@as(u32, @intCast(c)) << @intCast(4 * pos));
        for (0..4) |j| {
            const lo: u16 = if (pos < 4 and j == pos) e2m1Bits(@intCast(c)) else 0;
            const hi: u16 = if (pos >= 4 and j == pos - 4) e2m1Bits(@intCast(c)) else 0;
            try testing.expectEqual(lo, @as(u16, @truncate(h[j])));
            try testing.expectEqual(hi, @as(u16, @truncate(h[j] >> 16)));
        }
    };
    var state: u32 = 0x6b31_dec0;
    for (0..20000) |_| {
        const p = nextU32(&state);
        const q = nextU32(&state);
        const hp = k1Decode8(p);
        const hq = k1Decode8(q);
        const hpq = k1Decode8(p | q);
        for (0..4) |j| try testing.expectEqual(hp[j] | hq[j], hpq[j]);
        // and per code: code i of the word in half (i % 4), lo for i < 4
        for (0..8) |i| {
            const want = e2m1Bits((p >> @intCast(4 * i)) & 0xF);
            const got: u16 = @truncate(if (i < 4) hp[i] else hp[i - 4] >> 16);
            try testing.expectEqual(want, got);
        }
    }
    // the half values: code (s e1 e0 m) is +-{0, 0.5, 1, 1.5, 2, 3, 4, 6} x 2^-14 (the texts' x 2^22 restores 2^-14 x 2^-8)
    const mag = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (0..16) |c| {
        const v = mag[c & 7] * 0x1p-14;
        try testing.expectEqual(if (c & 8 != 0) -v else v, halfF32(e2m1Bits(@intCast(c))));
    }
    // the text carries exactly these masks and shifts
    for ([_][]const u8{
        "h[0] = as_type<half2>(((w << 9) & 0x0E000E00u) | ((w << 12) & 0x80008000u));",
        "h[1] = as_type<half2>(((w << 5) & 0x0E000E00u) | ((w << 8) & 0x80008000u));",
        "h[2] = as_type<half2>(((w << 1) & 0x0E000E00u) | ((w << 4) & 0x80008000u));",
        "h[3] = as_type<half2>(((w >> 3) & 0x0E000E00u) | (w & 0x80008000u));",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.header_k1, frag) != null);
}

test "mimo moe k1 members: the simdgroup vote names mimo_union_members' leaders and members on every wave shape; every pair written once" {
    var state: u32 = 0x6b31_7e11;
    const Shape = struct { pairs: u32, slots: u32 };
    for ([_]Shape{
        .{ .pairs = 8, .slots = 256 }, .{ .pairs = 16, .slots = 24 },      .{ .pairs = 24, .slots = 24 },        .{ .pairs = 32, .slots = 24 },
        .{ .pairs = 32, .slots = 12 }, .{ .pairs = 32, .slots = 1 },       .{ .pairs = 13, .slots = 3 },         .{ .pairs = 33, .slots = 2 },
        .{ .pairs = 3, .slots = 1 },   .{ .pairs = max_pairs, .slots = 9 }, .{ .pairs = max_pairs, .slots = 1 },
    }) |shape| for (0..300) |_| {
        var ids: [max_pairs]u32 = undefined;
        for (ids[0..shape.pairs]) |*id| {
            const bank: u32 = if (shape.slots <= 2) 0 else nextU32(&state) % 3;
            id.* = (bank << row_bits) | (nextU32(&state) % shape.slots);
        }
        const w = ids[0..shape.pairs];
        var written: [max_pairs]u32 = @splat(0);
        for (0..shape.pairs) |e| {
            var want: [max_tokens]u32 = undefined;
            var got: [max_tokens]u32 = undefined;
            const nw = unionMembers(w, e, &want);
            const ng = k1Members(w, e, max_tokens, &got);
            try testing.expectEqual(nw, ng);
            try testing.expectEqualSlices(u32, want[0..nw], got[0..ng]);
            for (got[0..ng]) |q| {
                try testing.expectEqual(w[e], w[q]);
                written[q] += 1;
            }
            var one: [1]u32 = undefined;
            try testing.expectEqual(@as(u32, 1), k1Members(w, e, 1, &one));
            try testing.expectEqual(@as(u32, @intCast(e)), one[0]);
        }
        for (written[0..shape.pairs]) |c| try testing.expectEqual(@as(u32, 1), c);
    };
    const K = texts.header_k1;
    for ([_][]const u8{
        "const bool v0 = lane < np && ids[min(lane, np - 1u)] == id;",
        "const bool v1 = q1 < np && ids[min(q1, np - 1u)] == id;",
        "if (rank % uint(MAXM) != 0u) return 0;",
        "mem[nm++] = ctz(lo);",
        "mem[nm++] = 32u + ctz(hi);",
    }) |frag| try testing.expect(std.mem.indexOf(u8, K, frag) != null);
}

test "mimo moe k1 CPU: per lane bit-equal to the banked texts at the real shapes (gate/up 2048 x 4096, down 4096 x 2048), M = 1..4 waves through the union leaders, rows M-invariant" {
    const a = testing.allocator;
    var state: u32 = 0x6b31_0001;
    // One real-shaped slot record per projection (codes random, E8M0 scales over the real banks' range plus zeros);
    // every slot id reads its own view (codes xor a per-id word, scales + a per-id bump), so a member computed from the
    // wrong slot or the wrong token's activations cannot match.
    const Rec = struct { words: []u32, scales: []u8, k: u32, rows: u32 };
    const dims = [3][2]u32{ .{ inter, hidden }, .{ inter, hidden }, .{ hidden, inter } }; // gate, up, down: rows, k
    var recs: [3]Rec = undefined;
    var made: usize = 0;
    defer for (recs[0..made]) |r| {
        a.free(r.words);
        a.free(r.scales);
    };
    for (&recs, dims) |*r, d| {
        r.* = .{ .words = try a.alloc(u32, d[0] * d[1] / 8), .scales = undefined, .k = d[1], .rows = d[0] };
        r.scales = a.alloc(u8, d[0] * d[1] / group_size) catch |err| {
            a.free(r.words);
            return err;
        };
        made += 1;
        for (r.words) |*w| w.* = nextU32(&state);
        for (r.scales, 0..) |*s, i| s.* = if (i % 499 == 0) 0 else @intCast(118 + nextU32(&state) % 10);
    }
    const View = struct {
        fn of(r: Rec, id: u32) HostProj {
            return .{ .words = r.words, .scales = r.scales, .k = r.k, .flip = id *% 0x9E3779B9, .bump = @intCast(id % 3) };
        }
    };
    // Activations: the 4 tokens' x (bf16) and each (token, k)'s down input (its SwiGLU row's stand-in).
    const tokens = max_tokens;
    const x = try a.alloc(u16, tokens * hidden);
    defer a.free(x);
    const act = try a.alloc(u16, tokens * top_k * inter);
    defer a.free(act);
    for ([_][]u16{ x, act }) |buf| for (buf) |*v| {
        const bits = nextU32(&state);
        v.* = @intCast(((bits & 1) << 15) | ((118 + (bits >> 1) % 12) << 7) | ((bits >> 8) & 0x7f));
    };
    // The routing: token t's k-th pair names expert (6 t + k) % 24 (the M = 4 shape over a 24-slot union), slot id
    // (expert % 3) << 24 | expert / 3: every token's 8 slots distinct, 2 tokens share 8 of them.
    const idOf = struct {
        fn f(t: usize, k: usize) u32 {
            const ex: u32 = @intCast((6 * t + k) % 24);
            return ((ex % 3) << row_bits) | (ex / 3);
        }
    }.f;
    // Rows: the first and last 4 of each projection (both ends of the walk; the index arithmetic is affine in the row).
    const n_rows = 8;
    const rowOf = struct {
        fn f(i: usize, rows: u32) u32 {
            return if (i < 4) @intCast(i) else rows - 8 + @as(u32, @intCast(i));
        }
    }.f;
    // The banked texts' lanes per (token, k, projection, row): the reference every wave must reproduce.
    const want = try a.alloc([32]f32, tokens * top_k * 3 * n_rows);
    defer a.free(want);
    for (0..tokens) |t| for (0..top_k) |k| for (recs, 0..) |r, pi| for (0..n_rows) |ri| {
        const xin = if (pi < 2) x[t * hidden ..][0..hidden] else act[(t * top_k + k) * inter ..][0..inter];
        const lanes = &want[((t * top_k + k) * 3 + pi) * n_rows + ri];
        for (lanes, 0..) |*v, l| v.* = bankedLane(View.of(r, idOf(t, k)), rowOf(ri, r.rows), xin, @intCast(l));
    };
    // The k1 texts through their waves: M = 1 (each token alone), 2, 3, 4 (tokens 0 .. M - 1); per pair e of the grid's
    // z, the leader rule at the plan's MAXM; each member's lanes from the LEADER's slot view and the MEMBER's inputs.
    const plan = k1_plan;
    var lanes_equal: usize = 0;
    var lanes_total: usize = 0;
    for ([_]u32{ 1, 1, 1, 1, 2, 3, 4 }, [_]u32{ 0, 1, 2, 3, 0, 0, 0 }) |m, t0| {
        const pairs = m * top_k;
        var ids: [max_pairs]u32 = undefined;
        for (0..pairs) |q| ids[q] = idOf(t0 + q / top_k, q % top_k);
        const w = ids[0..pairs];
        var leaders: u32 = 0;
        for (recs, 0..) |r, pi| {
            const g = if (pairs >= union_min_pairs) (if (pi < 2) plan.union_gate_up else plan.union_down) else (if (pi < 2) plan.serial_gate_up else plan.serial_down);
            var written: [max_pairs]u32 = @splat(0);
            for (0..pairs) |e| {
                var mem: [max_tokens]u32 = undefined;
                const nm = k1Members(w, e, g.maxm, &mem);
                if (nm == 0) continue;
                if (pi == 0) leaders += 1;
                for (mem[0..nm]) |q| {
                    written[q] += 1;
                    const t = t0 + q / top_k;
                    const k = q % top_k;
                    const xin = if (pi < 2) x[t * hidden ..][0..hidden] else act[(t * top_k + k) * inter ..][0..inter];
                    for (0..n_rows) |ri| {
                        const ref = &want[((t * top_k + k) * 3 + pi) * n_rows + ri];
                        var got: [32]f32 = undefined;
                        for (&got, 0..) |*v, l| v.* = k1Lane(View.of(r, w[e]), rowOf(ri, r.rows), xin, @intCast(l));
                        for (got, ref) |p, q2| lanes_equal += @intFromBool(@as(u32, @bitCast(p)) == @as(u32, @bitCast(q2)));
                        lanes_total += 32;
                        // the rows the lanes sum to (any simd_sum order: the lanes are the same values)
                        try testing.expectEqual(@as(u32, @bitCast(simdSumRef(ref))), @as(u32, @bitCast(simdSumRef(&got))));
                    }
                }
            }
            for (written[0..pairs]) |c| try testing.expectEqual(@as(u32, 1), c);
        }
        std.debug.print("MIMO_MOE_K1_CPU M={d} first_token={d} pairs={d} gate_up_leaders={d} lanes_bit_equal={d}/{d}\n", .{ m, t0, pairs, leaders, lanes_equal, lanes_total });
    }
    try testing.expectEqual(lanes_total, lanes_equal);
    // Per member row the texts keep the banked arithmetic: the same accumulation statements, simd_sum, x 2^22, epilogue.
    for ([_][]const u8{
        "ag[0][r] += mimo_k1_dot16(hg, xf) * mimo_fp4_scale(cs[r]);",
        "au[0][r] += mimo_k1_dot16(hu, xf) * mimo_fp4_scale(ct[r]);",
        "ag[m][r] += mimo_k1_dot16(hg[r], xf) * sgr[r];",
        "au[m][r] += mimo_k1_dot16(hu[r], xf) * sur[r];",
        "const float g = simd_sum(ag[m][r]) * 4194304.0f;",
        "const float gf = float(T(g));",
        "y[size_t(mem[m]) * N + row0 + r] = T(gf * mimo_sigmoid_f32(gf)) * T(u);",
        "const size_t row = mimo_slot_row(id) * N + row0;",
        "for (int c = 0; c < NC; ++c) {",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.gate_up_k1, frag) != null);
    for ([_][]const u8{
        "acc[0][r] += mimo_k1_dot16(hd, xf) * mimo_fp4_scale(cs[r]);",
        "acc[m][r] += mimo_k1_dot16(hd[r], xf) * sdr[r];",
        "const float v = simd_sum(acc[m][r]) * 4194304.0f;",
        "if (lane == 0) y[size_t(mem[m]) * H + row0 + r] = T(v);",
        "const size_t row = mimo_slot_row(id) * H + row0;",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.down_k1, frag) != null);
    for ([_][]const u8{
        "a += x[0] * float(h[0].x) + x[1] * float(h[1].x);",
        "b += x[14] * float(h[6].y) + x[15] * float(h[7].y);",
        "return a + b;",
        "xf[2 * i] = float(t.x);",
        "xf[2 * i + 1] = float(t.y);",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.header_k1, frag) != null);
}

test "mimo moe k1 fma: one FMA per weight (the timed alternative) stays within the summation bound of the banked order at the real shapes; the texts carry the chain" {
    const a = testing.allocator;
    var state: u32 = 0x6b31_f3a1;
    // gate's shape (2048 x 4096) and down's (4096 x 2048): 8 rows each, all 32 lanes, random codes, E8M0 over the real range
    var worst: f64 = 0;
    var differ: usize = 0;
    var rows_n: usize = 0;
    for ([_][2]u32{ .{ inter, hidden }, .{ hidden, inter } }) |d| {
        const words = try a.alloc(u32, d[0] * d[1] / 8);
        defer a.free(words);
        const scales = try a.alloc(u8, d[0] * d[1] / group_size);
        defer a.free(scales);
        for (words) |*w| w.* = nextU32(&state);
        for (scales) |*sc| sc.* = @intCast(118 + nextU32(&state) % 10);
        // activations over 32 binades (a lane's 16 products then need more than 24 bits, so the two orders round
        // differently; over the real-activation-like 12 binades of the k1 CPU test they almost never do)
        const x = try a.alloc(u16, d[1]);
        defer a.free(x);
        for (x) |*v| {
            const bits = nextU32(&state);
            v.* = @intCast(((bits & 1) << 15) | ((104 + (bits >> 1) % 32) << 7) | ((bits >> 8) & 0x7f));
        }
        const proj: HostProj = .{ .words = words, .scales = scales, .k = d[1] };
        for ([_]u32{ 0, 1, 2, 3, d[0] - 4, d[0] - 3, d[0] - 2, d[0] - 1 }) |n| {
            var lb: [32]f32 = undefined;
            var lf: [32]f32 = undefined;
            for (&lb, &lf, 0..) |*b, *f, l| {
                b.* = bankedLane(proj, n, x, @intCast(l));
                f.* = k1LaneFma(proj, n, x, @intCast(l));
            }
            // the bound: recursive summation of the row's K terms, 2 K u sum|terms| (u = 2^-24), both forms
            var abs: f64 = 0;
            for (0..d[1]) |kk| {
                const c = (words[(n * d[1] + @as(u32, @intCast(kk))) / 8] >> @intCast(4 * (kk % 8))) & 0xF;
                abs += @abs(@as(f64, bf16F32(x[kk])) * @as(f64, halfF32(e2m1Bits(c)))) * @as(f64, scaleRef(scales[(n * d[1] + @as(u32, @intCast(kk))) / group_size]));
            }
            const sb = simdSumRef(&lb);
            const sf = simdSumRef(&lf);
            const err = @abs(@as(f64, sf) - @as(f64, sb));
            const bound = 2.0 * @as(f64, @floatFromInt(d[1])) * 0x1p-24 * abs;
            try testing.expect(err <= bound);
            worst = @max(worst, err / @max(bound, 1e-300));
            differ += @intFromBool(@as(u32, @bitCast(sf)) != @as(u32, @bitCast(sb)));
            rows_n += 1;
        }
    }
    std.debug.print("MIMO_MOE_K1_FMA_CPU rows={d} differ_from_banked={d} worst_err_over_bound={d:.4}\n", .{ rows_n, differ, worst });
    // the fixture must exercise the difference (else the bound proves nothing about the FMA order)
    try testing.expect(differ > 0);
    for ([_][]const u8{
        "inline float mimo_k1_dot16_fma(thread const half2* h, thread const float* x) {",
        "float p = x[0] * float(h[0].x);",
        "p = fma(x[4], float(h[0].y), p);",
        "p = fma(x[8], float(h[4].x), p);",
        "p = fma(x[15], float(h[7].y), p);",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.header_k1, frag) != null);
    for ([_][]const u8{
        "ag[0][r] = fma(mimo_k1_dot16_fma(hg, xf), mimo_fp4_scale(cs[r]), ag[0][r]);",
        "au[m][r] = fma(mimo_k1_dot16_fma(hu[r], xf), sur[r], au[m][r]);",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.gate_up_k1, frag) != null);
    for ([_][]const u8{
        "acc[0][r] = fma(mimo_k1_dot16_fma(hd, xf), mimo_fp4_scale(cs[r]), acc[0][r]);",
        "acc[m][r] = fma(mimo_k1_dot16_fma(hd[r], xf), sdr[r], acc[m][r]);",
    }) |frag| try testing.expect(std.mem.indexOf(u8, texts.down_k1, frag) != null);
    // the routed plan is the exact form; the FMA plan validates at the routed geometry
    try k1_plan_fma.validate();
}

test "mimo moe k1 route: the k1 family binds its texts at every width; the plan's geometry and an invalid family refuse by name" {
    const geometry = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    try testing.expectError(error.MimoDecodeMoeKernel, Fused.initFamily(geometry, 8, .invalid));
    try k1_plan.validate();
    // geometry the texts cannot tile, a union past max_tokens, a one-token wave given a union
    try testing.expectError(error.MimoK1Geometry, Fused.initTextsWith(geometry, 8, .k1, .{ .union_gate_up = .{ .rows = 3, .maxm = 4 } }));
    try testing.expectError(error.MimoK1Geometry, Fused.initTextsWith(geometry, 8, .k1, .{ .union_down = .{ .rows = 4, .maxm = max_tokens + 1 } }));
    try testing.expectError(error.MimoK1Geometry, Fused.initTextsWith(geometry, 8, .k1, .{ .serial_gate_up = .{ .rows = 4, .maxm = 2 } }));
    try testing.expectError(error.MimoK1Geometry, Fused.initTextsWith(geometry, 8, .k1, .{ .union_down = .{ .rows = 4, .maxm = max_tokens, .fma = 2 } }));
    // the family's geometry per width (the plan) is what the configs carry; the texts are the k1 texts at every width
    var f = try Fused.initFamily(geometry, 8, .k1);
    defer f.deinit();
    for (f.gate_up_k, f.down_k) |gk, dk| {
        try testing.expect(gk.ctx == f.k_gate_up_k1.ctx and dk.ctx == f.k_down_k1.ctx);
    }
    var b = try Fused.initFamily(geometry, 8, .banked);
    defer b.deinit();
    for (b.gate_up_k, b.down_k, 1..) |gk, dk, p| {
        const want_g = if (p >= union_min_pairs) b.k_gate_up_union.ctx else b.k_gate_up_pair.ctx;
        const want_d = if (p >= union_min_pairs) b.k_down_union.ctx else b.k_down_pair.ctx;
        try testing.expect(gk.ctx == want_g and dk.ctx == want_d);
    }
    try testing.expect(b.k_gate_up_k1.ctx == null and f.k_gate_up_union.ctx == null and f.k_gate_up_pair.ctx == null);
}

const DecodeOps = struct {
    /// The arch's decode activation (source.zig `expertActivation`) on MLX.
    fn activation(g: *ops.MlxOps, gate: T, up: T) !T {
        return g.mul(try g.astype(try g.silu(try g.astype(gate, .float32)), .bfloat16), up);
    }
    /// The arch's decode combine (source.zig `forward`, nt = 1 lane) on MLX: f32(rows) * f32 w, sum axis 1, bf16.
    fn combine(g: *ops.MlxOps, routed: T, weights: T, n: c_int) !T {
        const expanded = try g.reshape(weights, &.{ n, @intCast(top_k), 1 });
        return g.astype(try g.sum(try g.mul(try g.astype(routed, .float32), expanded), 1, false), .bfloat16);
    }
};

fn nextU32(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

test "mimo moe SiLU: the text's f32 SiLU (one rounding, then the bf16 multiply) is MiMo's activation, not the donor's table" {
    // The case the source fixture pins: torch's F32 SiLU rounds once to 0.005889892578125; the donor's bf16 sigmoid
    // table form (`gt * sigtab[gt]`, every op in bf16) gives 0.00592041015625.
    try testing.expectEqual(@as(f32, 0.005889892578125), siluMulRef(0.01171875, 1));
    const gt = bf16Round(0.01171875);
    try testing.expectEqual(@as(f32, 0.00592041015625), bf16Round(bf16Round(gt * bf16Round(sigmoidRef(gt))) * 1));
    // Against the arch's MLX graph on the CPU stream, over a spread of bf16 gates and ups: the reference must match
    // bit for bit wherever MLX's CPU exp agrees with libm's to the rounding (reported), and never be more than one
    // bf16 ulp away.
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const n: usize = 4096;
    var gate: [n]f32 = undefined;
    var up: [n]f32 = undefined;
    var state: u32 = 0x5111e7ab;
    for (&gate, &up, 0..) |*x, *u, i| {
        // |gate| in [2^-9, 2^6), both signs: the sigmoid's both branches and its saturation. Past |g| ~ 87 the f32
        // sigmoid is subnormal and MLX's CPU exp (not the device's) decides it, so the sweep stays below 64.
        const bits = nextU32(&state);
        const e: u32 = 118 + (bits >> 4) % 15;
        x.* = bf16Round(@bitCast(((bits & 1) << 31) | (e << 23) | ((bits >> 9) & 0x7f) << 16));
        u.* = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 8);
    }
    const gx = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&gate), &.{@intCast(n)}, .float32), .bfloat16);
    const ux = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&up), &.{@intCast(n)}, .float32), .bfloat16);
    var got: [n]f32 = undefined;
    _ = try g.hostF32(try g.astype(try DecodeOps.activation(&g, gx, ux), .float32), &got);
    var exact: usize = 0;
    for (gate, up, got) |x, u, want| {
        const ref = siluMulRef(x, u);
        if (@as(u32, @bitCast(ref)) == @as(u32, @bitCast(want))) {
            exact += 1;
        } else {
            const d = @as(i64, @as(u32, @bitCast(ref)) >> 16) - @as(i64, @as(u32, @bitCast(want)) >> 16);
            if (d != 1 and d != -1) std.debug.print("MIMO_MOE_SILU mismatch gate={e} up={e} ref={e} mlx={e}\n", .{ x, u, ref, want });
            try testing.expect(d == 1 or d == -1);
        }
    }
    std.debug.print("MIMO_MOE_SILU cpu_ref_vs_mlx_cpu n={d} bit_equal={d} one_ulp={d}\n", .{ n, exact, n - exact });
    try testing.expect(exact * 100 >= n * 99);
}

test "mimo moe combine: the text's fold is bit-equal to the arch's score-weighted sum (f32 inputs, router order)" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const h: usize = hidden;
    for ([_]usize{ 1, 2, 3, 4 }) |n| {
        const rows = try a.alloc(f32, n * top_k * h);
        defer a.free(rows);
        const w = try a.alloc(f32, n * top_k);
        defer a.free(w);
        const want = try a.alloc(f32, n * h);
        defer a.free(want);
        const got = try a.alloc(f32, n * h);
        defer a.free(got);
        var state: u32 = 0x2c0b1e5d +% @as(u32, @intCast(n));
        for (rows, 0..) |*v, i| {
            const bits = nextU32(&state);
            // bf16 rows (the down text's outputs), wide exponents, signed zeros included.
            v.* = if (i % 997 == 0) -0.0 else bf16Round(@bitCast((bits & 0x80000000) | ((112 + (bits >> 3) % 24) << 23) | ((bits >> 9) & 0x7f) << 16));
        }
        // Router weights: sigmoid picks over their sum (f32, as module.route produces), one row per token.
        for (0..n) |t| {
            var sum: f32 = 0;
            for (w[t * top_k ..][0..top_k]) |*v| {
                v.* = sigmoidRef(@as(f32, @floatFromInt(@as(i32, @intCast(nextU32(&state) % 2001)) - 1000)) / 250);
                sum += v.*;
            }
            for (w[t * top_k ..][0..top_k]) |*v| v.* /= sum + 1e-20;
        }
        combineRef(rows, w, n, h, got);
        const mark = g.mark();
        const routed = try g.astype(try g.hostArray(std.mem.sliceAsBytes(rows), &.{ @intCast(n), @intCast(top_k), @intCast(h) }, .float32), .bfloat16);
        const weights = try g.hostArray(std.mem.sliceAsBytes(w), &.{ @intCast(n), @intCast(top_k) }, .float32);
        _ = try g.hostF32(try g.astype(try DecodeOps.combine(&g, routed, weights, @intCast(n)), .float32), want);
        g.resetTo(mark);
        for (want, got) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
        std.debug.print("MIMO_MOE_COMBINE n={d} hidden={d} words={d} bit_equal=true\n", .{ n, h, want.len });
    }
}

// ── Device self-check (pre-ship gate; never constructed by ordinary host checks) ──
//
// Static bound (declared before any run): 3 banks x 12 slots x 13,369,344 B = 481 MB of slot arrays (host copies freed
// after the MLX copy: <= 2 x 160 MB transient); per pair the f32 truth dequantizes three [2048|4096 x 4096|2048] slot
// views (3 x 32 MiB bf16 + 3 x 32 MiB f32 = 192 MiB), evaluated and released per pair; fused/stock outputs < 2 MiB.
// Bound: 481 MB + 320 MB + 192 MiB + MLX JIT (5 texts; the per-pair reference adds < 1 MiB of outputs) <= 1.2 GB;
// declared 1.5 GB.
const selfcheck_bound_bytes: u64 = 1_500_000_000;
const selfcheck_slots: u32 = 12;

fn relRms(got: []const f32, truth: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, truth) |x, t| {
        num += (@as(f64, x) - t) * (@as(f64, x) - t);
        den += @as(f64, t) * t;
    }
    return @sqrt(num / @max(den, 1e-30));
}

fn bankProjection(a: std.mem.Allocator, g: *ops.MlxOps, slots: u32, out: u32, in: u32, state: *u32) !Proj {
    const words = try a.alloc(u32, slots * out * (in / 8));
    defer a.free(words);
    const scales = try a.alloc(u8, slots * out * (in / 32));
    defer a.free(scales);
    for (words) |*w| w.* = nextU32(state);
    // E8M0 near 2^-4 .. 2^0, the real banks' range: the outputs stay finite in bf16.
    for (scales) |*sc| sc.* = @intCast(123 + nextU32(state) % 5);
    const w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(slots), @intCast(out), @intCast(in / 8) }, .uint32);
    const sc = try g.hostArray(scales, &.{ @intCast(slots), @intCast(out), @intCast(in / 32) }, .uint8);
    try g.evalAll(&.{ w, sc });
    return .{ .w = g.keep(w), .scales = g.keep(sc) };
}

/// One pair's f32 truth: dequantized slot views, f32 GEMVs, f32 SiLU of the f32 gate, f32 product, f32 down.
fn truthRow(g: *ops.MlxOps, bank: Arrays, row: u32, x: T) !T {
    const r: c_int = @intCast(row);
    const pick = struct {
        fn f(gg: *ops.MlxOps, p: Proj, rr: c_int, out: c_int, in: c_int) !T {
            const w = try gg.slice(p.w, &.{ rr, 0, 0 }, &.{ rr + 1, out, @divExact(in, 8) }, &.{ 1, 1, 1 });
            const s = try gg.slice(p.scales.?, &.{ rr, 0, 0 }, &.{ rr + 1, out, @divExact(in, 32) }, &.{ 1, 1, 1 });
            const d = try gg.dequantize(try gg.reshape(w, &.{ out, @divExact(in, 8) }), try gg.reshape(s, &.{ out, @divExact(in, 32) }), .mxfp4);
            return gg.astype(d, .float32);
        }
    }.f;
    const H: c_int = @intCast(hidden);
    const I: c_int = @intCast(inter);
    const xf = try g.astype(x, .float32);
    const gt = try g.matmul(xf, try g.transpose(try pick(g, bank.gate, r, I, H)));
    const ut = try g.matmul(xf, try g.transpose(try pick(g, bank.up, r, I, H)));
    const act = try g.mul(try g.silu(gt), ut);
    return g.matmul(act, try g.transpose(try pick(g, bank.down, r, H, I)));
}

test "mimo moe GPU: fused banked gate/up + down + combine vs the stock gather_qmm route at real shapes (M = 1..4)" {
    _ = std.c.getenv("MIMO_MOE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("MIMO_MOE_SELFCHECK bound_bytes={d}\n", .{selfcheck_bound_bytes});
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try ops.MlxOps.init(a, stream);
    defer g.deinit();
    const geometry = try bank_mod.Geometry.init(hidden, inter, 256, 1, 47);
    var fused = try Fused.init(geometry, selfcheck_slots);
    defer fused.deinit();
    // M4 lever 1: the banked (per-pair) texts, the union texts' bitwise reference.
    var pair = try Fused.initTexts(geometry, selfcheck_slots, .pair);
    defer pair.deinit();
    // K1 (mimo/k1-moe-bw): the k1 texts, bitwise against the banked texts on the same inputs.
    var k1 = try Fused.initFamily(geometry, selfcheck_slots, .k1);
    defer k1.deinit();
    // K1's FMA alternative (timed by the m4attr family, never routed): rounding-class, held to the stock route's tolerance.
    var k1f = try Fused.initTextsWith(geometry, selfcheck_slots, .k1, k1_plan_fma);
    defer k1f.deinit();
    var fma_worst: f64 = 0;
    const Bitwise = struct {
        /// Equal 32-bit words of two bf16 arrays (as f32) and their count.
        fn count(al: std.mem.Allocator, gg: *ops.MlxOps, x0: T, x1: T) ![2]usize {
            const n: usize = @intCast(gg.shapeOf(x0).numel());
            const b0 = try al.alloc(f32, n);
            defer al.free(b0);
            const b1 = try al.alloc(f32, n);
            defer al.free(b1);
            _ = try gg.hostF32(try gg.astype(x0, .float32), b0);
            _ = try gg.hostF32(try gg.astype(x1, .float32), b1);
            var eq: usize = 0;
            for (b0, b1) |p, q| eq += @intFromBool(@as(u32, @bitCast(p)) == @as(u32, @bitCast(q)));
            return .{ eq, n };
        }
    };
    var k1_words: usize = 0;
    var state: u32 = 0x6d6f655f;
    var banks: [n_banks]Arrays = undefined;
    for (&banks) |*b| b.* = .{
        .gate = try bankProjection(a, &g, selfcheck_slots, inter, hidden, &state),
        .up = try bankProjection(a, &g, selfcheck_slots, inter, hidden, &state),
        .down = try bankProjection(a, &g, selfcheck_slots, hidden, inter, &state),
    };
    defer for (banks) |b| for ([_]Proj{ b.gate, b.up, b.down }) |p| {
        g.release(p.w);
        g.release(p.scales.?);
    };
    g.reset();
    const H: c_int = @intCast(hidden);
    // Bank spreads: one bank (base), two (base + transient), all three; spread 4: one slot for every pair (the hit
    // pass's misses all name base row 0), spread 5: M = 4 rows over 8 distinct base slots, every slot in all 4 rows.
    const spreads = [_]u32{ 1, 2, 3, 4, 5 };
    var worst_ratio: f64 = 0;
    var union_words: usize = 0;
    for (spreads) |n_used| for (1..max_tokens + 1) |m| {
        const pairs: u32 = @intCast(m * top_k);
        var ids: [max_pairs]u32 = undefined;
        var tok: [max_pairs]i32 = undefined;
        for (0..pairs) |p| {
            const b: u32 = if (n_used == 1 or n_used >= 4) 0 else if (n_used == 2) (if (p % 2 == 0) 0 else 2) else @intCast(p % 3);
            const slot: u32 = switch (n_used) {
                4 => 0,
                5 => @intCast(p % top_k),
                else => (@as(u32, @intCast(p)) * 5 + nextU32(&state)) % selfcheck_slots,
            };
            ids[p] = (b << 24) | slot;
            tok[p] = @intCast(p / top_k);
        }
        var weights: [max_tokens * top_k]f32 = undefined;
        for (0..m) |t| {
            var sum: f32 = 0;
            for (weights[t * top_k ..][0..top_k]) |*v| {
                v.* = sigmoidRef(@as(f32, @floatFromInt(@as(i32, @intCast(nextU32(&state) % 2001)) - 1000)) / 250);
                sum += v.*;
            }
            for (weights[t * top_k ..][0..top_k]) |*v| v.* /= sum + 1e-20;
        }
        var xin: [max_tokens * hidden]f32 = undefined;
        for (xin[0 .. m * hidden]) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(nextU32(&state) % 2001)) - 1000)) / 4000;
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin[0 .. m * hidden]), &.{ @intCast(m), H }, .float32), .bfloat16);
        const w = try g.hostArray(std.mem.sliceAsBytes(weights[0 .. m * top_k]), &.{ @intCast(m), @intCast(top_k) }, .float32);
        const xp = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..pairs]), &.{@intCast(pairs)}, .int32), 0);
        const id_arr = try g.hostArray(std.mem.sliceAsBytes(ids[0..pairs]), &.{@intCast(pairs)}, .uint32);
        // Fused: 2 texts per wave + the combine.
        const f_h = try fused.gateUp(&g, xp, id_arr, &banks);
        const f_rows = try fused.down(&g, f_h, id_arr, &banks);
        // The per-pair reference on the same inputs (and on the union's own SwiGLU rows, so down is compared alone).
        const p_h = try pair.gateUp(&g, xp, id_arr, &banks);
        const p_rows = try pair.down(&g, f_h, id_arr, &banks);
        // K1: the k1 texts on the same inputs (down on the same SwiGLU rows as the banked down), and two partial waves
        // (3 pairs: the one-token texts with constant ids; 13: the union texts on an odd wave), each against the banked
        // texts at its own width.
        const k_h = try k1.gateUp(&g, xp, id_arr, &banks);
        const k_rows = try k1.down(&g, f_h, id_arr, &banks);
        const kf_rows = try k1f.down(&g, try k1f.gateUp(&g, xp, id_arr, &banks), id_arr, &banks);
        var k_sub: [2][4]T = undefined; // k1 gate/up, banked gate/up, k1 down, banked down
        var k_sub_p: [2]u32 = undefined;
        var n_sub: usize = 0;
        for ([_]u32{ 3, 13 }) |sp| if (sp < pairs) {
            const xs = try g.slice(xp, &.{ 0, 0 }, &.{ @intCast(sp), H }, &.{ 1, 1 });
            const is = try g.hostArray(std.mem.sliceAsBytes(ids[0..sp]), &.{@intCast(sp)}, .uint32);
            const ph = try pair.gateUp(&g, xs, is, &banks);
            k_sub[n_sub] = .{ try k1.gateUp(&g, xs, is, &banks), ph, try k1.down(&g, ph, is, &banks), try pair.down(&g, ph, is, &banks) };
            k_sub_p[n_sub] = sp;
            n_sub += 1;
        };
        const f_out = try fused.combineRows(&g, try g.reshape(f_rows, &.{ @intCast(m), @intCast(top_k), H }), w);
        // Stock: per pair the bank's gather_qmm (as Math.gateUp / down do per bank group), the arch's activation and combine.
        var s_parts: [max_pairs]T = undefined;
        for (0..pairs) |p| {
            const sl = slotOf(ids[p]);
            const b = banks[sl.bank];
            const row = try g.hostArray(std.mem.sliceAsBytes(&[_]u32{sl.row}), &.{1}, .uint32);
            const xr = try g.reshape(try g.slice(xp, &.{ @intCast(p), 0 }, &.{ @intCast(p + 1), H }, &.{ 1, 1 }), &.{ 1, 1, H });
            const gg = try g.reshape(try g.gatherQmm(xr, b.gate.w, b.gate.scales.?, row, .mxfp4), &.{ 1, @intCast(inter) });
            const uu = try g.reshape(try g.gatherQmm(xr, b.up.w, b.up.scales.?, row, .mxfp4), &.{ 1, @intCast(inter) });
            const hh = try DecodeOps.activation(&g, gg, uu);
            s_parts[p] = try g.reshape(try g.gatherQmm(try g.reshape(hh, &.{ 1, 1, @intCast(inter) }), b.down.w, b.down.scales.?, row, .mxfp4), &.{ 1, H });
        }
        const s_rows = try g.concat(s_parts[0..pairs], 0);
        const s_out = try DecodeOps.combine(&g, try g.reshape(s_rows, &.{ @intCast(m), @intCast(top_k), H }), w, @intCast(m));
        try g.evalAll(&.{ f_rows, f_out, s_rows, s_out, p_h, p_rows, k_h, k_rows, kf_rows });
        for (k_sub[0..n_sub]) |ks| try g.evalAll(&ks);
        // K1 vs banked: bit for bit, gate/up rows and down rows, the full wave and the partial waves.
        {
            const kh = try Bitwise.count(a, &g, k_h, p_h);
            const kd = try Bitwise.count(a, &g, k_rows, p_rows);
            var distinct: u32 = 0;
            for (0..pairs) |p| distinct += @intFromBool(std.mem.indexOfScalar(u32, ids[0..p], ids[p]) == null);
            std.debug.print("MIMO_MOE_K1 banks={d} M={d} pairs={d} distinct={d} gate_up_bitwise={d}/{d} down_bitwise={d}/{d}\n", .{ n_used, m, pairs, distinct, kh[0], kh[1], kd[0], kd[1] });
            try testing.expectEqual(kh[1], kh[0]);
            try testing.expectEqual(kd[1], kd[0]);
            k1_words += kh[1] + kd[1];
            for (k_sub[0..n_sub], k_sub_p[0..n_sub]) |ks, sp| {
                const sh = try Bitwise.count(a, &g, ks[0], ks[1]);
                const sd = try Bitwise.count(a, &g, ks[2], ks[3]);
                std.debug.print("MIMO_MOE_K1 banks={d} M={d} partial_pairs={d} gate_up_bitwise={d}/{d} down_bitwise={d}/{d}\n", .{ n_used, m, sp, sh[0], sh[1], sd[0], sd[1] });
                try testing.expectEqual(sh[1], sh[0]);
                try testing.expectEqual(sd[1], sd[0]);
                k1_words += sh[1] + sd[1];
            }
        }
        // Union vs per-pair: bit for bit, gate/up rows and down rows.
        {
            const nh = pairs * inter;
            const uh = try a.alloc(f32, nh);
            defer a.free(uh);
            const ph = try a.alloc(f32, nh);
            defer a.free(ph);
            _ = try g.hostF32(try g.astype(f_h, .float32), uh);
            _ = try g.hostF32(try g.astype(p_h, .float32), ph);
            const ur = try a.alloc(f32, pairs * hidden);
            defer a.free(ur);
            const pr = try a.alloc(f32, pairs * hidden);
            defer a.free(pr);
            _ = try g.hostF32(try g.astype(f_rows, .float32), ur);
            _ = try g.hostF32(try g.astype(p_rows, .float32), pr);
            var eq_h: usize = 0;
            for (uh, ph) |x0, x1| eq_h += @intFromBool(@as(u32, @bitCast(x0)) == @as(u32, @bitCast(x1)));
            var eq_d: usize = 0;
            for (ur, pr) |x0, x1| eq_d += @intFromBool(@as(u32, @bitCast(x0)) == @as(u32, @bitCast(x1)));
            var distinct: u32 = 0;
            for (0..pairs) |p| distinct += @intFromBool(std.mem.indexOfScalar(u32, ids[0..p], ids[p]) == null);
            std.debug.print("MIMO_MOE_UNION banks={d} M={d} pairs={d} distinct={d} gate_up_bitwise={d}/{d} down_bitwise={d}/{d}\n", .{ n_used, m, pairs, distinct, eq_h, nh, eq_d, ur.len });
            try testing.expectEqual(nh, eq_h);
            try testing.expectEqual(ur.len, eq_d);
            union_words += nh + ur.len;
        }
        const n_row = pairs * hidden;
        const fr = try a.alloc(f32, n_row);
        defer a.free(fr);
        const sr = try a.alloc(f32, n_row);
        defer a.free(sr);
        const tr = try a.alloc(f32, n_row);
        defer a.free(tr);
        _ = try g.hostF32(try g.astype(f_rows, .float32), fr);
        _ = try g.hostF32(try g.astype(s_rows, .float32), sr);
        var fo: [max_tokens * hidden]f32 = undefined;
        var so: [max_tokens * hidden]f32 = undefined;
        var to: [max_tokens * hidden]f32 = @splat(0);
        _ = try g.hostF32(try g.astype(f_out, .float32), fo[0 .. m * hidden]);
        _ = try g.hostF32(try g.astype(s_out, .float32), so[0 .. m * hidden]);
        // f32 truth per pair, evaluated and released per pair (the bound above).
        for (0..pairs) |p| {
            const mark = g.mark();
            const sl = slotOf(ids[p]);
            const xr = try g.slice(xp, &.{ @intCast(p), 0 }, &.{ @intCast(p + 1), H }, &.{ 1, 1 });
            _ = try g.hostF32(try truthRow(&g, banks[sl.bank], sl.row, xr), tr[p * hidden ..][0..hidden]);
            g.resetTo(mark);
            const t = p / top_k;
            for (0..hidden) |c| to[t * hidden + c] += tr[p * hidden + c] * weights[p];
        }
        var bitwise: usize = 0;
        for (fr, sr) |x0, x1| bitwise += @intFromBool(@as(u32, @bitCast(x0)) == @as(u32, @bitCast(x1)));
        const e_rows_f = relRms(fr, tr);
        const e_rows_s = relRms(sr, tr);
        const e_out_f = relRms(fo[0 .. m * hidden], to[0 .. m * hidden]);
        const e_out_s = relRms(so[0 .. m * hidden], to[0 .. m * hidden]);
        {
            const kfr = try a.alloc(f32, n_row);
            defer a.free(kfr);
            _ = try g.hostF32(try g.astype(kf_rows, .float32), kfr);
            const e_rows_k = relRms(kfr, tr);
            std.debug.print("MIMO_MOE_K1_FMA banks={d} M={d} pairs={d} rows_relrms fma={e:.3} banked={e:.3} stock={e:.3}\n", .{ n_used, m, pairs, e_rows_k, e_rows_f, e_rows_s });
            try testing.expect(e_rows_k <= 2 * e_rows_s + 1e-6);
            fma_worst = @max(fma_worst, e_rows_k / @max(e_rows_s, 1e-12));
        }
        std.debug.print("MIMO_MOE_SELFCHECK banks={d} M={d} pairs={d} rows_relrms fused={e:.3} stock={e:.3} out_relrms fused={e:.3} stock={e:.3} rows_bitwise_equal={d}/{d}\n", .{ n_used, m, pairs, e_rows_f, e_rows_s, e_out_f, e_out_s, bitwise, n_row });
        // Gate (design 2c): relative RMS vs f32 truth <= 2 x the stock route's (+ 1e-6 for an exact stock).
        try testing.expect(e_rows_f <= 2 * e_rows_s + 1e-6);
        try testing.expect(e_out_f <= 2 * e_out_s + 1e-6);
        worst_ratio = @max(worst_ratio, e_out_f / @max(e_out_s, 1e-12));
        g.reset();
    };
    std.debug.print("MIMO_MOE_SELFCHECK PASS worst_out_ratio={d:.3} union_vs_pair_bitwise_words={d} k1_vs_banked_bitwise_words={d} k1_fma_worst_rows_ratio={d:.3}\n", .{ worst_ratio, union_words, k1_words, fma_worst });
}

test "mimo moe route: hitpass is the served default; \"decode_moe\": \"fused\" / \"stock\" install the controls; an unknown name refuses" {
    const settings = @import("settings.zig");
    var c: settings.Config = .{};
    try testing.expectEqual(Route.hitpass, c.decode_moe);
    const fused = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"decode_moe\":\"fused\"}", .{});
    defer fused.deinit();
    c.applySettings(fused.value);
    try testing.expectEqual(Route.fused, c.decode_moe);
    const stock = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"decode_moe\":\"stock\"}", .{});
    defer stock.deinit();
    c.applySettings(stock.value);
    try testing.expectEqual(Route.stock, c.decode_moe);
    const bad = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"decode_moe\":\"other\"}", .{});
    defer bad.deinit();
    c.applySettings(bad.value);
    try testing.expectEqual(Route.stock, c.decode_moe);
    try testing.expectError(error.MimoDecodeMoe, c.validate());
}
