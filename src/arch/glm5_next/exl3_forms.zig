//! GLM-5.3-Flash's EXL3 K=3 decode forms: the decode chain of the seam (`exl3_kernels.zig` gateUp / down: in_rin, the
//! gate and up GEMVs, gu_epi, din_rin, the down GEMV, dpost; DeepSeek-V4.1's banked mul1h texts at GLM's shape) in other
//! launch geometries, each computing the seam's words (the bench's device parity checks every output word). A form is a
//! construction-time route: `install` builds the chosen form's launches for every wave width once; `gateUp` / `down`
//! keep the seam's signatures and run them with no per-call choice.
//!
//! Forms (texts in src/kernels/glm5_next_exl3_forms/; `scripts/glm5next_exl3_forms_gen.py` writes the generated ones):
//!   gate / up (`GuForm`): `seam` (two GEMV launches + gu_epi), `u8` (the GEMV at DeepSeek's gate unroll 8), `guone`
//!     (DeepSeek's gu_one: both GEMVs in one launch), `guone_pair` (`g5k3_guone_pair`: gu_one's launch with down_pair's
//!     two-row rule: two rows naming one slot share each trellis decode, the even rank computing both), `mcg` (the
//!     bench's codebook price, not a served form), `fused16` /
//!     `fused8` (`g5k3_gu_fused`: both GEMVs and gu_epi in one launch over a 128-column block, 16 simdgroups (one tile of
//!     one projection each) or 8 (one tile of both)).
//!   down (`DnForm`): `seam`, `pair` (DeepSeek's down_pair), `mcg`, `fused1` / `fused2` (din_rin, then `g5k3_dn_fused`:
//!     the GEMV and dpost in one launch over 1 or 2 128-column blocks), `pro1` / `pro2` (`g5k3_dn_pro`: din_rin, the
//!     GEMV and dpost in one launch).
//!   `maxm` (fused forms): 1, or 4 = the union rule of the NVFP4 k1 texts (a slot's pairs in one threadgroup, each
//!     trellis word decoded once for all of them).
//!   The w forms (hand-written; header_g5k3.metal `g5k3w_*`): gu_one's / the stock GEMV's launch per row with
//!     g5k3_pair's decode words on every path (one integer op a weight fewer than guone's), the lane's x as two float2
//!     loads a k-tile (four scalar loads before), and g5k3_colsum (no threadgroup memory, no barrier): gate / up `guw`
//!     (g5k3_guw, one output tile a simdgroup) and `guw2` (two tiles a simdgroup: the lane's x loads shared, two
//!     independent decode chains), `guwp` (g5k3_guwp: guone_pair's two-row rule on the w lane); down `dnw` / `dnw2`
//!     (g5k3_dnw at one / two tiles) and `dnwp` (g5k3_dnwp: down_pair's rule on the w lane). `guwh` / `guw2h` and
//!     `dnwh` / `dnw2h` route on the wave width at install (`Forms.at`): the one-row form at P <= top_k (one token's
//!     wave: no two pairs name one slot), the two-row form above; every form computes every row's words, the route only
//!     picks the faster launch for the width.
//! Every form keeps the seam's arithmetic and order per output word (fma chains per lane over the k-tiles, the column
//! sums, the 128-point butterflies, the scales, the clamped SwiGLU, one rounding on dpost's store); only where values
//! live and which thread computes them change. A form that fails the device parity is not installed.
//! Wave widths: P = 1 .. `max_pairs` (64 = 8 tokens x top-8, the NVFP4 texts' widest wave; the decode lane forms at most
//! 4 rows x 8 = 32 pairs, and the scheduler's route at most `policy.max_route_ids` = 48).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const mtl = @import("../../sdk_ext/metal_launch.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");
const xk = @import("exl3_kernels.zig");

pub const G = xk.G;
pub const T = xk.T;
const BankArrays = xk.BankArrays;
const n_banks = xk.n_banks;
const hidden = xk.hidden;
const inter = xk.inter;

pub const max_pairs: u32 = 64;

comptime {
    if (max_pairs < @import("../../sdk_ext.zig").expert.policy.max_route_ids) @compileError("glm5_next exl3 forms: the widest wave is narrower than the scheduler's route");
    if (max_pairs > 64) @compileError("glm5_next exl3 forms: the union vote takes at most 64 pairs a wave");
}

pub const GuForm = enum {
    seam,
    u8,
    guone,
    guone_pair,
    mcg,
    fused16,
    fused8,
    guw,
    guw2,
    guwp,
    guwh,
    guw2h,

    /// The w forms' route on the wave width (`Forms.at`): the one-row form at P <= top_k, the two-row form above.
    fn atP(f: GuForm, p: u32) GuForm {
        return switch (f) {
            .guwh => if (p <= top_k) .guw else .guwp,
            .guw2h => if (p <= top_k) .guw2 else .guwp,
            else => f,
        };
    }
};
pub const DnForm = enum {
    seam,
    pair,
    mcg,
    fused1,
    fused2,
    pro1,
    pro2,
    dnw,
    dnw2,
    dnwp,
    dnwh,
    dnw2h,

    fn atP(f: DnForm, p: u32) DnForm {
        return switch (f) {
            .dnwh => if (p <= top_k) .dnw else .dnwp,
            .dnw2h => if (p <= top_k) .dnw2 else .dnwp,
            else => f,
        };
    }
};
/// One token's wave: top_k pairs naming distinct slots (the w forms' route between the one-row and two-row texts).
pub const top_k: u32 = 8;
pub const Forms = struct {
    gu: GuForm = .seam,
    dn: DnForm = .seam,
    maxm: u32 = 1,
    /// The narrowest wave the fused forms take (narrower waves run the seam chain): a route on the wave width P, which
    /// varies per call, fixed per P at install (the fused forms' threadgroups cover a 128-column block each, so a
    /// one-pair wave fills 16 (gate / up) or 32 (down) of the GPU's cores; the bench measures the crossover).
    min_p: u32 = 1,

    pub fn fusedGu(f: Forms) bool {
        return f.gu == .fused16 or f.gu == .fused8;
    }
    pub fn fusedDn(f: Forms) bool {
        return f.dn == .fused1 or f.dn == .fused2 or f.dn == .pro1 or f.dn == .pro2;
    }
    /// The w forms (their column sums are g5k3_colsum's, the fused forms' register sums).
    pub fn wGu(f: Forms) bool {
        return switch (f.gu) {
            .guw, .guw2, .guwp, .guwh, .guw2h => true,
            else => false,
        };
    }
    pub fn wDn(f: Forms) bool {
        return switch (f.dn) {
            .dnw, .dnw2, .dnwp, .dnwh, .dnw2h => true,
            else => false,
        };
    }

    /// The forms a build installs, refused by name otherwise: maxm 1 or 4, 4 only on a fused form; the prologue forms at
    /// maxm 1 (each member's rotated input is 8 KiB of threadgroup memory).
    pub fn check(f: Forms) error{Glm5NextExl3Forms}!void {
        if (f.maxm != 1 and f.maxm != 4) return error.Glm5NextExl3Forms;
        if (f.maxm != 1 and !(f.fusedGu() or f.fusedDn())) return error.Glm5NextExl3Forms;
        if (f.maxm != 1 and (f.dn == .pro1 or f.dn == .pro2)) return error.Glm5NextExl3Forms;
        if (f.min_p < 1 or f.min_p > max_pairs) return error.Glm5NextExl3Forms;
        if (f.min_p != 1 and !(f.fusedGu() or f.fusedDn())) return error.Glm5NextExl3Forms;
    }

    /// The forms a wave of P pairs runs: these (the w forms' width routes resolved), or the seam chain below `min_p`.
    pub fn at(f: Forms, p: u32) Forms {
        if (p < f.min_p) return .{};
        var o = f;
        o.gu = f.gu.atP(p);
        o.dn = f.dn.atP(p);
        return o;
    }

    pub fn name(f: Forms, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "gu={t},dn={t},maxm={d},min_p={d}", .{ f.gu, f.dn, f.maxm, f.min_p }) catch "forms";
    }
};

// ── The texts ──

pub const Kernel = enum { g5k3_gu_fused, g5k3_dn_fused, g5k3_dn_pro, g5k3_guone, g5k3_pair, g5k3_gemv_u8, g5k3_gemv_mcg, g5k3_guone_pair, g5k3_guw, g5k3_dnw, g5k3_guwp, g5k3_dnwp };
const n_kernels = std.meta.fieldNames(Kernel).len;
const dir = "../../kernels/glm5_next_exl3_forms/";
pub const texts: [n_kernels][:0]const u8 = blk: {
    var out: [n_kernels][:0]const u8 = undefined;
    for (std.meta.fieldNames(Kernel), 0..) |n, i| out[i] = @embedFile(dir ++ n ++ ".metal");
    break :blk out;
};
pub const header: [:0]const u8 = @embedFile(dir ++ "header_g5k3.metal");
/// sha256 of each text (the generated ones as `scripts/glm5next_exl3_forms_gen.py` writes them; g5k3_guone_pair and the
/// fused texts by hand) and of the header.
pub const text_sha256 = [n_kernels]*const [64]u8{
    "06518c1a2675dca204da46410c37570d015b8dd94ad9fcb89d13e842f5783721",
    "42c6d677a6b6ffafde0ba9d9ef0857aad6c5eda89656f394285f9c20f08c3b14",
    "318a568f9ee22fbc2bd75127bc010b440f312ba388dc74839ba7f785c5364e57",
    "85dcbd286beb00d7ec1f80b298967859090c0f8aedf29824380a5c5027df3374",
    "0732d4840889255ad9499b30e1431fc2ff78ec19d72ebc89f5a1d9df31440110",
    "66d970fa467925ce0f5d9dec3bdba70d1e0955dbdaabbcf3e193ca2676021e5d",
    "3dbbae2ece4fb6e77241d6f9bd829886e798346c3f9126a840e105ab99a22abc",
    "60b34ea70408fbb498368d06f3326ff49c0843216b03ca863549e0ed7010c66e",
    "8b133738199d43fee1f358562506e89d9de73a73867c4ea7ce594e756695f4f6",
    "315fe9436e0847daebf9bb645bc575e826e39cfab1f0bb9dcfa5a4504a2b4bc4",
    "7d20d915f3c9a9dd266b11f19a4210b9658143eabd67f66bbe22f476e30f73fa",
    "be6a38486c7441e0cfb950af3c590ee07d222406e8b595f4f1c8a4a842258e25",
};
pub const header_sha256: *const [64]u8 = "93dbe2a9acb4268e1384dbe4d23e09a3a72c8e6bdb51c656a71c9bd1563cdc52";

pub fn checkPins() error{Glm5NextExl3FormsTextNotPinned}!void {
    for (texts, text_sha256) |t, want| if (!std.mem.eql(u8, &kset.shaHex(t), want)) return error.Glm5NextExl3FormsTextNotPinned;
    if (!std.mem.eql(u8, &kset.shaHex(header), header_sha256)) return error.Glm5NextExl3FormsTextNotPinned;
}

const Sig = struct { ins: []const [*:0]const u8, outs: []const [*:0]const u8, hdr: bool };

fn sigOf(k: Kernel) Sig {
    return switch (k) {
        .g5k3_gu_fused => .{ .ins = &.{ "xg", "xu", "ids", "code_g0", "code_g1", "code_g2", "code_u0", "code_u1", "code_u2", "og0", "og1", "og2", "ou0", "ou1", "ou2" }, .outs = &.{"out"}, .hdr = true },
        .g5k3_dn_fused => .{ .ins = &.{ "xd", "ids", "cd0", "cd1", "cd2", "rd0", "rd1", "rd2" }, .outs = &.{"out"}, .hdr = true },
        .g5k3_dn_pro => .{ .ins = &.{ "h", "ids", "cd0", "cd1", "cd2", "rn0", "rn1", "rn2", "rd0", "rd1", "rd2" }, .outs = &.{"out"}, .hdr = true },
        .g5k3_guone, .g5k3_guone_pair => .{ .ins = &.{ "xh_g", "xh_u", "ids", "code_g0", "code_g1", "code_g2", "code_u0", "code_u1", "code_u2" }, .outs = &.{ "out_g", "out_u" }, .hdr = false },
        .g5k3_pair, .g5k3_gemv_u8, .g5k3_gemv_mcg => .{ .ins = &.{ "xh", "ids", "code0", "code1", "code2" }, .outs = &.{"out"}, .hdr = false },
        .g5k3_guw, .g5k3_guwp => .{ .ins = &.{ "xh_g", "xh_u", "ids", "code_g0", "code_g1", "code_g2", "code_u0", "code_u1", "code_u2" }, .outs = &.{ "out_g", "out_u" }, .hdr = true },
        .g5k3_dnw, .g5k3_dnwp => .{ .ins = &.{ "xh", "ids", "code0", "code1", "code2" }, .outs = &.{"out"}, .hdr = true },
    };
}

// ── Launches (pure: the tests read them) ──

pub const Launch = struct {
    grid: [3]u32,
    tg: [3]u32,
    ints: [4]struct { name: [*:0]const u8, v: u32 } = undefined,
    n_int: u8 = 0,
    t_dtype: ?mlx.mlx_dtype = null,
    outs: [2]struct { shape: [2]c_int, dtype: mlx.mlx_dtype } = undefined,
    n_out: u8 = 0,

    fn int(l: Launch, name: [*:0]const u8, v: u32) Launch {
        var o = l;
        o.ints[o.n_int] = .{ .name = name, .v = v };
        o.n_int += 1;
        return o;
    }
    fn out(l: Launch, rows: u32, cols: u32, dtype: mlx.mlx_dtype) Launch {
        var o = l;
        o.outs[o.n_out] = .{ .shape = .{ @intCast(rows), @intCast(cols) }, .dtype = dtype };
        o.n_out += 1;
        return o;
    }
};

/// g5k3_gu_fused: one threadgroup per (128-column block of inter, pair); 512 threads (fused16) or 256 (fused8).
pub fn guFusedLaunch(p: u32, gu: GuForm, maxm: u32) Launch {
    const nth: u32 = if (gu == .fused16) 512 else 256;
    const l: Launch = .{ .grid = .{ inter / 128 * nth, p, 1 }, .tg = .{ nth, 1, 1 } };
    return l.int("IT", hidden / 16).int("OT", inter / 16).int("GUL", if (gu == .fused16) 0 else 1).int("MAXM", maxm).out(p, inter, .float32);
}

/// g5k3_dn_fused / g5k3_dn_pro: one threadgroup per (nb 128-column blocks of hidden, pair), 256 nb threads; the output in
/// `dt` (the seam's bf16; f32 for the bench's reference check).
pub fn dnFusedLaunch(p: u32, nb: u32, maxm: u32, dt: mlx.mlx_dtype) Launch {
    const nth: u32 = 256 * nb;
    var l: Launch = .{ .grid = .{ hidden / 128 / nb * nth, p, 1 }, .tg = .{ nth, 1, 1 }, .t_dtype = dt };
    l = l.int("IT", inter / 16).int("OT", hidden / 16).int("NB", nb).int("MAXM", maxm);
    return l.out(p, hidden, dt);
}

/// DeepSeek's gu_one at GLM's shape: grid (inter / 16 x 32, 2 P): the gate's P rows, then the up's. guone_pair takes the
/// same grid: an odd-rank row's threadgroups exit, its leader writes both rows.
pub fn guoneLaunch(p: u32) Launch {
    const l: Launch = .{ .grid = .{ inter / 16 * 32, 2 * p, 1 }, .tg = .{ 32, 1, 1 } };
    return l.int("IT", hidden / 16).int("OT", inter / 16).out(p, inter, .float32).out(p, inter, .float32);
}

/// The w forms' one-row texts (g5k3_guw: gate and up, grid y 2 P; g5k3_dnw: grid y P): `tpt` output tiles a simdgroup,
/// grid (out / 16 / tpt x 32, rows).
pub fn wLaunch(p: u32, in: u32, o: u32, tpt: u32, gu: bool) Launch {
    const l: Launch = .{ .grid = .{ o / 16 / tpt * 32, if (gu) 2 * p else p, 1 }, .tg = .{ 32, 1, 1 } };
    const k = l.int("IT", in / 16).int("OT", o / 16).int("TPT", tpt);
    return if (gu) k.out(p, o, .float32).out(p, o, .float32) else k.out(p, o, .float32);
}

/// The w forms' two-row texts (g5k3_guwp: gate and up, grid y 2 P; g5k3_dnwp: grid y P): one tile a simdgroup.
pub fn wpLaunch(p: u32, in: u32, o: u32, gu: bool) Launch {
    const l: Launch = .{ .grid = .{ o / 16 * 32, if (gu) 2 * p else p, 1 }, .tg = .{ 32, 1, 1 } };
    const k = l.int("IT", in / 16).int("OT", o / 16);
    return if (gu) k.out(p, o, .float32).out(p, o, .float32) else k.out(p, o, .float32);
}

/// One-simdgroup GEMV texts (pair, u8, mcg): grid (out / 16 x 32, P).
pub fn gemvLaunch(p: u32, in: u32, o: u32) Launch {
    const l: Launch = .{ .grid = .{ o / 16 * 32, p, 1 }, .tg = .{ 32, 1, 1 } };
    return l.int("IT", in / 16).int("OT", o / 16).out(p, o, .float32);
}

fn cfgOf(l: Launch) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.outs[0..l.n_out]) |*o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &o.shape, 2, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(l.grid[0]), @intCast(l.grid[1]), @intCast(l.grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(l.tg[0]), @intCast(l.tg[1]), @intCast(l.tg[2])));
    for (l.ints[0..l.n_int]) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name, @intCast(t.v)));
    if (l.t_dtype) |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", d));
    return c;
}

/// A seam launch (exl3_kernels.zig's builders) as a config: its outputs, grid, threadgroup, ints and T.
fn seamCfg(l: xk.Launch) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.outs[0..l.n_out]) |*o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &o.shape, o.ndim, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    for (l.ints[0..l.n_int]) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name, t.v));
    if (l.t_dtype) |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", d));
    return c;
}

fn apply(g: *G, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const T, outs: []T) !void {
    const in = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k, in, cfg, g.s));
    for (outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        o.* = try g.adopt(x);
    }
}

// ── Install ──

const Cfg = mlx.mlx_fast_metal_kernel_config;
const none: Cfg = .{ .ctx = null };
/// One wave width's prepared launches (null ctx: not used by the installed forms).
const WaveCfg = struct {
    /// the forms this width runs (`Forms.at`)
    f: Forms = .{},
    in_rin: Cfg = none,
    g1: Cfg = none,
    epi: Cfg = none,
    din: Cfg = none,
    d1: Cfg = none,
    post: Cfg = none,

    fn free(w: WaveCfg) void {
        for ([_]Cfg{ w.in_rin, w.g1, w.epi, w.din, w.d1, w.post }) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
    }
};

pub const Installed = struct {
    forms: Forms,
    out_dtype: mlx.mlx_dtype,
    seam: *const xk.Kernels,
    k: [n_kernels]mlx.mlx_fast_metal_kernel = @splat(.{ .ctx = null }),
    w: [max_pairs]WaveCfg = @splat(.{}),
    tok: [max_pairs]T = @splat(.{ .ctx = null }),

    pub fn deinit(self: *Installed) void {
        for (self.k) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (self.w) |w| w.free();
        for (self.tok) |t| if (t.ctx != null) {
            _ = mlx.mlx_array_free(t);
        };
    }

    fn kern(self: *const Installed, k: Kernel) mlx.mlx_fast_metal_kernel {
        return self.k[@intFromEnum(k)];
    }
    fn seamK(self: *const Installed, k: xk.Kernel) mlx.mlx_fast_metal_kernel {
        return self.seam.k[@intFromEnum(k)];
    }
};

/// Construction, once: the forms checked, the pins, the texts built, every wave width's launches of the chosen forms
/// prepared (`out_dtype`: down's output, the seam's bf16; the bench also installs f32 for its reference check). The
/// seam's kernels (`seam`, from `exl3_kernels.install`) run in_rin, gu_epi, din_rin, dpost and the seam GEMV.
pub fn install(a: std.mem.Allocator, seam: *const xk.Kernels, forms: Forms, out_dtype: mlx.mlx_dtype, diag: *sdk.Diag) !*Installed {
    forms.check() catch |e| {
        diag.set("glm5_next exl3 forms: {t} / {t} at maxm {d} is not an installable form", .{ forms.gu, forms.dn, forms.maxm });
        return e;
    };
    checkPins() catch |e| {
        diag.set("glm5_next exl3 forms: a text is not the pinned one", .{});
        return e;
    };
    const self = try a.create(Installed);
    self.* = .{ .forms = forms, .out_dtype = out_dtype, .seam = seam };
    errdefer {
        self.deinit();
        a.destroy(self);
    }
    inline for (comptime std.enums.values(Kernel), 0..) |k, i| {
        const sig = sigOf(k);
        const iv = mlx.mlx_vector_string_new_data(@ptrCast(sig.ins.ptr), sig.ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(@ptrCast(sig.outs.ptr), sig.outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        self.k[i] = mtl.mlx_fast_metal_kernel_new(@tagName(k), iv, ov, texts[i], if (sig.hdr) header else "", true, false);
        if (self.k[i].ctx == null) {
            diag.set("glm5_next exl3 forms: {s} was not created", .{@tagName(k)});
            return error.Glm5NextExl3FormsKernel;
        }
    }
    for (0..max_pairs) |i| {
        const p: u32 = @intCast(i + 1);
        const ef = forms.at(p);
        const nb: u32 = if (ef.dn == .fused2 or ef.dn == .pro2) 2 else 1;
        var w: WaveCfg = .{ .f = ef };
        errdefer w.free();
        w.in_rin = try seamCfg(xk.inRinLaunch(p));
        w.g1 = switch (ef.gu) {
            .seam => try seamCfg(xk.gemvLaunch(p, hidden, inter)),
            .u8, .mcg => try cfgOf(gemvLaunch(p, hidden, inter)),
            .guone, .guone_pair => try cfgOf(guoneLaunch(p)),
            .fused16, .fused8 => try cfgOf(guFusedLaunch(p, ef.gu, ef.maxm)),
            .guw => try cfgOf(wLaunch(p, hidden, inter, 1, true)),
            .guw2 => try cfgOf(wLaunch(p, hidden, inter, 2, true)),
            .guwp => try cfgOf(wpLaunch(p, hidden, inter, true)),
            .guwh, .guw2h => unreachable, // resolved by `Forms.at`
        };
        if (!ef.fusedGu()) w.epi = try seamCfg(xk.rotLaunch(p, inter, "INTER"));
        if (ef.dn != .pro1 and ef.dn != .pro2) w.din = try seamCfg(xk.rotLaunch(p, inter, "INTER"));
        w.d1 = switch (ef.dn) {
            .seam => try seamCfg(xk.gemvLaunch(p, inter, hidden)),
            .pair, .mcg => try cfgOf(gemvLaunch(p, inter, hidden)),
            .fused1, .fused2, .pro1, .pro2 => try cfgOf(dnFusedLaunch(p, nb, ef.maxm, out_dtype)),
            .dnw => try cfgOf(wLaunch(p, inter, hidden, 1, false)),
            .dnw2 => try cfgOf(wLaunch(p, inter, hidden, 2, false)),
            .dnwp => try cfgOf(wpLaunch(p, inter, hidden, false)),
            .dnwh, .dnw2h => unreachable, // resolved by `Forms.at`
        };
        if (!ef.fusedDn()) {
            // dpost at hidden with its store in out_dtype (the seam's bf16; T is the text's store type)
            var dl = xk.rotLaunch(p, hidden, "HID");
            dl.t_dtype = out_dtype;
            dl.outs[0].dtype = out_dtype;
            w.post = try seamCfg(dl);
        }
        self.w[i] = w;
        var idx: [max_pairs]i32 = undefined;
        for (idx[0..p], 0..) |*v, j| v.* = @intCast(j);
        const shape = [_]c_int{@intCast(p)};
        self.tok[i] = mlx.mlx_array_new_data(&idx, &shape, 1, .int32);
        if (self.tok[i].ctx == null) return error.MlxError;
    }
    return self;
}

fn widthOf(g: *G, x: T) !u32 {
    const p: u32 = @intCast(g.shapeOf(x).d[0]);
    if (p < 1 or p > max_pairs) return error.Glm5NextExl3WaveWidth;
    return p;
}

/// The seam's `gateUp` in the installed form: x bf16 [P, hidden], ids u32 [P] packed `bank << 24 | slot row` -> the
/// clamped SwiGLU f32 [P, inter].
pub fn gateUp(f: *const Installed, g: *G, x: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p = try widthOf(g, x);
    const w = &f.w[p - 1];
    var xs: [2]T = undefined;
    try apply(g, f.seamK(.in_rin), w.in_rin, &.{ x, f.tok[p - 1], banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
    var z: [2]T = undefined;
    switch (w.f.gu) {
        .seam, .u8, .mcg => {
            const k = switch (w.f.gu) {
                .seam => f.seamK(.gemv),
                .u8 => f.kern(.g5k3_gemv_u8),
                else => f.kern(.g5k3_gemv_mcg),
            };
            try apply(g, k, w.g1, &.{ xs[0], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code }, z[0..1]);
            try apply(g, k, w.g1, &.{ xs[1], ids, banks[0].up.code, banks[1].up.code, banks[2].up.code }, z[1..2]);
        },
        .guone => try apply(g, f.kern(.g5k3_guone), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &z),
        .guone_pair => try apply(g, f.kern(.g5k3_guone_pair), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &z),
        .guw, .guw2 => try apply(g, f.kern(.g5k3_guw), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &z),
        .guwp => try apply(g, f.kern(.g5k3_guwp), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &z),
        .guwh, .guw2h => unreachable, // resolved at install (`Forms.at`)
        .fused16, .fused8 => {
            var out: [1]T = undefined;
            try apply(g, f.kern(.g5k3_gu_fused), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code, banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout }, &out);
            return out[0];
        },
    }
    var out: [1]T = undefined;
    try apply(g, f.seamK(.gu_epi), w.epi, &.{ z[0], z[1], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
    return out[0];
}

/// The seam's `down` in the installed form: h f32 [P, inter] (gateUp's), the same ids -> [P, hidden] in the installed
/// output dtype (the seam's bf16).
pub fn down(f: *const Installed, g: *G, h: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p = try widthOf(g, h);
    const w = &f.w[p - 1];
    var out: [1]T = undefined;
    switch (w.f.dn) {
        .pro1, .pro2 => {
            try apply(g, f.kern(.g5k3_dn_pro), w.d1, &.{ h, ids, banks[0].down.code, banks[1].down.code, banks[2].down.code, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, banks[0].down.rout, banks[1].down.rout, banks[2].down.rout }, &out);
            return out[0];
        },
        else => {},
    }
    var hd: [1]T = undefined;
    try apply(g, f.seamK(.din_rin), w.din, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
    switch (w.f.dn) {
        .fused1, .fused2 => {
            try apply(g, f.kern(.g5k3_dn_fused), w.d1, &.{ hd[0], ids, banks[0].down.code, banks[1].down.code, banks[2].down.code, banks[0].down.rout, banks[1].down.rout, banks[2].down.rout }, &out);
            return out[0];
        },
        else => {},
    }
    const k = switch (w.f.dn) {
        .seam => f.seamK(.gemv),
        .pair => f.kern(.g5k3_pair),
        .dnw, .dnw2 => f.kern(.g5k3_dnw),
        .dnwp => f.kern(.g5k3_dnwp),
        else => f.kern(.g5k3_gemv_mcg),
    };
    var zd: [1]T = undefined;
    try apply(g, k, w.d1, &.{ hd[0], ids, banks[0].down.code, banks[1].down.code, banks[2].down.code }, &zd);
    try apply(g, f.seamK(.dpost), w.post, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
    return out[0];
}

// ── Tests (host only) ──

const testing = std.testing;

test "glm5_next exl3 forms: the texts are the pinned ones; the generated ones carry no DeepSeek shape literal" {
    try checkPins();
    for (texts) |t| {
        var it = std.mem.splitScalar(u8, t, '\n');
        while (it.next()) |line| {
            if (std.mem.startsWith(u8, std.mem.trimStart(u8, line, " "), "//")) continue;
            for ([_][]const u8{ "5120", "2304", "2211840", "320u", "144u" }) |lit| try testing.expect(std.mem.indexOf(u8, line, lit) == null);
        }
    }
}

test "glm5_next exl3 forms: the installable forms (maxm 4 only on a fused form, the prologue forms at maxm 1)" {
    try (Forms{}).check();
    try (Forms{ .gu = .fused16, .dn = .fused1, .maxm = 4 }).check();
    try (Forms{ .gu = .fused8, .dn = .seam, .maxm = 4 }).check();
    try (Forms{ .gu = .fused16, .dn = .pro1, .maxm = 1 }).check();
    try (Forms{ .gu = .fused8, .dn = .fused1, .maxm = 4, .min_p = 4 }).check();
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .guone, .dn = .pair, .min_p = 4 }).check());
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .fused8, .min_p = 0 }).check());
    const h: Forms = .{ .gu = .fused16, .dn = .fused2, .maxm = 4, .min_p = 4 };
    try testing.expectEqual(Forms{}, h.at(3));
    try testing.expectEqual(h, h.at(4));
    try testing.expectEqual(h, h.at(64));
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .seam, .dn = .pair, .maxm = 4 }).check());
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .fused16, .dn = .pro2, .maxm = 4 }).check());
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .fused16, .dn = .fused1, .maxm = 2 }).check());
}

test "glm5_next exl3 forms: the launches cover every output word at every wave width (one simdgroup per 16 outputs, the fused forms 8 tiles a 128-column block)" {
    for (1..max_pairs + 1) |pi| {
        const p: u32 = @intCast(pi);
        for ([_]GuForm{ .fused16, .fused8 }) |gu| for ([_]u32{ 1, 4 }) |mm| {
            const l = guFusedLaunch(p, gu, mm);
            // blocks x simdgroups x tiles per simdgroup x projections = the gate's and up's output tiles
            const sgs = l.tg[0] / 32;
            const tiles_per_sg: u32 = if (gu == .fused16) 1 else 2;
            try testing.expectEqual(2 * inter / 16, (l.grid[0] / l.tg[0]) * sgs * tiles_per_sg);
            try testing.expectEqual(p, l.grid[1]);
            try testing.expect(l.tg[0] <= 1024);
        };
        for ([_]u32{ 1, 2 }) |nb| {
            const l = dnFusedLaunch(p, nb, 1, .bfloat16);
            try testing.expectEqual(hidden / 16, (l.grid[0] / l.tg[0]) * (l.tg[0] / 32));
        }
        const go = guoneLaunch(p);
        try testing.expectEqual(2 * p, go.grid[1]);
        try testing.expectEqual(inter / 16 * 32, go.grid[0]);
    }
}

test "glm5_next exl3 forms: threadgroup memory of every fused instance fits 32 KiB" {
    // gu: Z[MAXM][2][128] f32; dn: Z[MAXM][NB][128] f32; pro: + X[MAXM][inter] f32
    for ([_]u32{ 1, 4 }) |mm| {
        try testing.expect(mm * 2 * 128 * 4 <= 32 << 10);
        for ([_]u32{ 1, 2 }) |nb| try testing.expect(mm * nb * 128 * 4 <= 32 << 10);
    }
    for ([_]u32{ 1, 2 }) |nb| try testing.expect(1 * inter * 4 + nb * 128 * 4 <= 32 << 10);
}

test "glm5_next exl3 forms: the w forms' launches cover every output word at every wave width; the width routes resolve at install" {
    for (1..max_pairs + 1) |pi| {
        const p: u32 = @intCast(pi);
        for ([_]u32{ 1, 2 }) |tpt| {
            const g = wLaunch(p, hidden, inter, tpt, true);
            try testing.expectEqual(inter / 16, g.grid[0] / 32 * tpt);
            try testing.expectEqual(2 * p, g.grid[1]);
            try testing.expectEqual(@as(u8, 2), g.n_out);
            const d = wLaunch(p, inter, hidden, tpt, false);
            try testing.expectEqual(hidden / 16, d.grid[0] / 32 * tpt);
            try testing.expectEqual(p, d.grid[1]);
            try testing.expectEqual(@as(u8, 1), d.n_out);
        }
        const gp = wpLaunch(p, hidden, inter, true);
        try testing.expectEqual(inter / 16, gp.grid[0] / 32);
        try testing.expectEqual(2 * p, gp.grid[1]);
        const dp = wpLaunch(p, inter, hidden, false);
        try testing.expectEqual(hidden / 16, dp.grid[0] / 32);
        // the width routes: one-row texts at P <= top_k, two-row above; the concrete forms unchanged
        const h = (Forms{ .gu = .guwh, .dn = .dnw2h }).at(p);
        try testing.expectEqual(if (p <= top_k) GuForm.guw else GuForm.guwp, h.gu);
        try testing.expectEqual(if (p <= top_k) DnForm.dnw2 else DnForm.dnwp, h.dn);
        try testing.expectEqual(GuForm.guw2, (Forms{ .gu = .guw2, .dn = .dnwp }).at(p).gu);
        try testing.expectEqual(DnForm.dnwp, (Forms{ .gu = .guw2, .dn = .dnwp }).at(p).dn);
        // no width resolves to a route value (the gate / up and down switches never meet one)
        for (std.enums.values(GuForm)) |gu| try testing.expect(gu.atP(p) != .guwh and gu.atP(p) != .guw2h);
        for (std.enums.values(DnForm)) |dn| try testing.expect(dn.atP(p) != .dnwh and dn.atP(p) != .dnw2h);
    }
    try (Forms{ .gu = .guw2h, .dn = .dnwh }).check();
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .guw, .dn = .dnw, .maxm = 4 }).check());
    try testing.expectError(error.Glm5NextExl3Forms, (Forms{ .gu = .guwh, .dn = .dnwh, .min_p = 4 }).check());
}
