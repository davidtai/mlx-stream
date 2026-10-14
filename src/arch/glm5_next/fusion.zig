//! glm5_next's trunk launch fusion (reports/dsv41-f39-tcq3-runtime/mimo/glm5next-audit-decode-20261014.md rank 8): FP8
//! GEMVs of one input fused into one launch, construction-time routes, each off by default (`settings.Fusion`: `off` keeps
//! the stock projections, g22-exl3's routes):
//!   fuse_kda_inputs  a KDA layer's six input projections of the same rows (q, k, v, f_a, g_a, b) in one launch
//!                    (`glm5n_fp8_gemv_kda_in`) at every call of at most `max_rows` rows (the MTP verify, a serial step,
//!                    a short prompt call): 6 -> 1 launches a KDA layer, 170 fewer a forward of Flash's 34 KDA layers
//!   fuse_gate_up     the decode lane's MLPs' gate and up projections (every routed layer's shared expert, the dense
//!                    layers' MLP) in one launch (`glm5n_fp8_gemv_pair`): 2 -> 1 an MLP, 45 fewer a forward (42 + 3)
//! Exact by construction: a launch's segments each run MiMo's FP8 GEMV code line for line (kernels/
//! header_glm5n_fp8_multi.metal `glm5n_fp8_gemv_rows`, the pinned mimo_fp8_gemv.metal's body) at their own site's plan:
//! its K partitions (KS) fix every output's summation order, and R and RG only pick which rows a simdgroup owns
//! (fp8_linear.zig `gemvPlanAt`'s contract), so every output is the stock launch's bit for bit. The outputs are the stock
//! launches' arrays (one per projection, the same shapes and dtype), so no bill term moves.
//! Installed once (`install`) over the bound residents: every site a switch fuses must be FP8 at the shapes and plans its
//! launch takes, else the load is refused by name (a named route never resolves to the stock one). A call wider than the
//! GEMV's rows (a prompt span) keeps the stock projections (the residents route's prompt path): a row-count route.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const fp8 = @import("fp8.zig");
const kda = @import("kda.zig");
const graph = @import("graph.zig");
const swiglu = @import("swiglu.zig");
const mimo_fp8 = @import("../mimo_v2/fp8_linear.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;
const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;
const Gemv = mimo_fp8.Gemv;

/// The widest call a fused launch takes: the GEMV's rows (decode, the MTP verify, a short prompt call).
pub const max_rows: usize = @intCast(fp8.gemv_max_rows);

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-fusion.json");
    pub const header = @embedFile("kernels/header_glm5n_fp8_multi.metal");
    pub const kda_in = @embedFile("kernels/glm5n_fp8_gemv_kda_in.metal");
    pub const pair = @embedFile("kernels/glm5n_fp8_gemv_pair.metal");
};
/// The header MLX compiles both texts with: MiMo's FP8 decoders (header_fp8res.metal, pinned there), then the segment
/// function.
const launch_header = mimo_fp8.texts.header ++ texts.header;
/// sha256 of kernels/manifest-fusion.json, which records every text's sha256 (checked against `text_pins` by a CPU test).
pub const manifest_sha256 = "2f1f05a61b919f75625a6ceae71c4e4b6be476fbc08e6320efbde5d0c2c1f3b3";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_glm5n_fp8_multi.metal", .text = texts.header, .sha256 = "44cac691ddd7a605823eff01acaec1f6b20a5d2a202af386241e104b3e96a914" },
    .{ .file = "glm5n_fp8_gemv_kda_in.metal", .text = texts.kda_in, .sha256 = "8dab99d935e66e9c5deaa8f48a108e719a28e4105a9cf06b9ede6455e785e506" },
    .{ .file = "glm5n_fp8_gemv_pair.metal", .text = texts.pair, .sha256 = "fe58d2cd4645d4a80ffc2ce6b54f50e3a3114e743d3433ca3a08f68c9f6ef756" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.Glm5NextFusionManifestNotPinned, error.Glm5NextFusionTextNotPinned);
}

// ---- geometry (host arithmetic; CPU-tested at the real shapes) ----

/// A KDA layer's fused launch at M rows (`glm5n_fp8_gemv_kda_in`'s template arguments): the q / k / v segments [nb, k]
/// on their sites' plan at M (rb rows a simdgroup, ksb K partitions), the f_a / g_a [na, k] and b [nc, k] segments on
/// theirs (rs, kss); every threadgroup `tgs` simdgroups: rgb = tgs / ksb row groups in a q / k / v threadgroup, rgs =
/// tgs / kss in the others.
pub const KdaInGeometry = struct {
    m: c_int,
    k: c_int,
    nb: c_int,
    rb: c_int,
    ksb: c_int,
    rgb: c_int,
    na: c_int,
    nc: c_int,
    rs: c_int,
    kss: c_int,
    rgs: c_int,
    tgs: c_int,

    /// Threadgroups of a q / k / v segment, of an f_a / g_a segment, of the b segment.
    pub fn groupsBig(g: KdaInGeometry) c_int {
        return @divExact(g.nb, g.rb * g.rgb);
    }
    pub fn groupsA(g: KdaInGeometry) c_int {
        return @divExact(g.na, g.rs * g.rgs);
    }
    pub fn groupsC(g: KdaInGeometry) c_int {
        return @divExact(g.nc, g.rs * g.rgs);
    }
    pub fn threadgroups(g: KdaInGeometry) c_int {
        return 3 * g.groupsBig() + 2 * g.groupsA() + g.groupsC();
    }
    /// The six segments in launch order (q, k, v, f_a, g_a, b): each one's rows and plan in the launch.
    pub fn segments(g: KdaInGeometry) [6]Segment {
        const big: Segment = .{ .n = g.nb, .r = g.rb, .ks = g.ksb, .rg = g.rgb };
        const a: Segment = .{ .n = g.na, .r = g.rs, .ks = g.kss, .rg = g.rgs };
        const cc: Segment = .{ .n = g.nc, .r = g.rs, .ks = g.kss, .rg = g.rgs };
        return .{ big, big, big, a, a, cc };
    }
};

/// One segment of a fused launch: its rows N and its plan in the launch (R rows a simdgroup, KS K partitions, RG row
/// groups a threadgroup).
pub const Segment = struct { n: c_int, r: c_int, ks: c_int, rg: c_int };

/// The fused launch's geometry from the sites' bound plans at M (`fp8.Fp8.plans[m]`): the q / k / v plan `big`, the
/// f_a / g_a / b plan `small`. Refused by name when one threadgroup shape cannot hold both plans' K partitions or a
/// segment does not tile (the text's static asserts).
pub fn kdaInGeometry(big: Gemv, small: Gemv, nb: u32, na: u32, nc: u32, k: u32, m: u32) error{Glm5NextFusionGeometry}!KdaInGeometry {
    if (m < 1 or m > max_rows) return error.Glm5NextFusionGeometry;
    const tgs = @max(big.rg * big.ks, small.rg * small.ks);
    if (tgs * 32 > 1024 or @rem(tgs, big.ks) != 0 or @rem(tgs, small.ks) != 0) return error.Glm5NextFusionGeometry;
    const geo: KdaInGeometry = .{
        .m = @intCast(m),
        .k = @intCast(k),
        .nb = @intCast(nb),
        .rb = big.r,
        .ksb = big.ks,
        .rgb = @divExact(tgs, big.ks),
        .na = @intCast(na),
        .nc = @intCast(nc),
        .rs = small.r,
        .kss = small.ks,
        .rgs = @divExact(tgs, small.ks),
        .tgs = tgs,
    };
    for (geo.segments()) |sg| if (!tiles(sg, geo.k)) return error.Glm5NextFusionGeometry;
    return geo;
}

/// The text's static asserts for one segment: whole 256-column steps per K partition, N whole R x RG row blocks, R
/// rows inside one 64-row scale row.
fn tiles(sg: Segment, k: c_int) bool {
    const step: c_int = @intCast(mimo_fp8.gemv_step);
    return sg.n > 0 and @rem(k, sg.ks * step) == 0 and @rem(sg.n, sg.r * sg.rg) == 0 and @rem(64, sg.r) == 0;
}

/// An MLP's fused gate / up launch at M rows (`glm5n_fp8_gemv_pair`): both [n, k] on their one plan at M, the stock
/// launch's threadgroup shape.
pub const PairGeometry = struct {
    m: c_int,
    n: c_int,
    k: c_int,
    p: Gemv,

    pub fn groups(g: PairGeometry) c_int {
        return @divExact(g.n, g.p.r * g.p.rg);
    }
};

pub fn pairGeometry(p: Gemv, n: u32, k: u32, m: u32) error{Glm5NextFusionGeometry}!PairGeometry {
    if (m < 1 or m > max_rows or p.rg * p.ks * 32 > 1024) return error.Glm5NextFusionGeometry;
    const geo: PairGeometry = .{ .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .p = p };
    if (!tiles(.{ .n = geo.n, .r = p.r, .ks = p.ks, .rg = p.rg }, geo.k)) return error.Glm5NextFusionGeometry;
    return geo;
}

/// The launch of `glm5n_fp8_gemv_kda_in` at `geo`: six outputs [M, N] bf16 in segment order, grid y one threadgroup of
/// `tgs` simdgroups per row block.
pub fn kdaInConfig(geo: KdaInGeometry) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (geo.segments()) |sg| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ geo.m, sg.n }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, geo.threadgroups() * geo.tgs, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, geo.tgs, 1));
    for ([_]struct { [*:0]const u8, c_int }{
        .{ "M", geo.m },   .{ "K", geo.k },   .{ "NB", geo.nb }, .{ "RB", geo.rb },   .{ "KSB", geo.ksb }, .{ "RGB", geo.rgb },
        .{ "NA", geo.na }, .{ "NC", geo.nc }, .{ "RS", geo.rs }, .{ "KSS", geo.kss }, .{ "RGS", geo.rgs },
    }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], t[1]));
    return cfg;
}

/// The launch of `glm5n_fp8_gemv_pair` at `geo`: two outputs [M, N] bf16 (gate, up), the stock launch's threadgroup.
pub fn pairConfig(geo: PairGeometry) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ geo.m, geo.n }, 2, .bfloat16));
    const sgs = geo.p.rg * geo.p.ks;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 2 * geo.groups() * sgs, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgs, 1));
    for ([_]struct { [*:0]const u8, c_int }{ .{ "M", geo.m }, .{ "N", geo.n }, .{ "K", geo.k }, .{ "R", geo.p.r }, .{ "KS", geo.p.ks }, .{ "RG", geo.p.rg } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], t[1]));
    return cfg;
}

pub const kda_in_inputs = [_][*:0]const u8{ "x", "w_q", "s_q", "w_k", "s_k", "w_v", "s_v", "w_fa", "s_fa", "w_ga", "s_ga", "w_b", "s_b" };
pub const kda_in_outputs = [_][*:0]const u8{ "y_q", "y_k", "y_v", "y_fa", "y_ga", "y_b" };
pub const pair_inputs = [_][*:0]const u8{ "x", "w_gate", "s_gate", "w_up", "s_up" };
pub const pair_outputs = [_][*:0]const u8{ "y_gate", "y_up" };

/// A text's kernel object through the plugin's launch cache (`plain`: MLX's own object, the CPU tests' graphs).
fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, comptime plain: bool) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const new = if (plain) mlx.mlx_fast_metal_kernel_new else mtl.mlx_fast_metal_kernel_new;
    const k = new(name, in_vec, out_vec, source, launch_header, true, false);
    if (k.ctx == null) return error.Glm5NextFusionKernel;
    return k;
}

// ---- the installed routes ----

/// A KDA layer's six input projections of x [.., R, hidden], each [.., R, out] bf16 (the residents route's shapes).
pub const KdaInputs = struct { q: T, k: T, v: T, f_a: T, g_a: T, b: T };
pub const KdaInputsFn = fn (g: *G, w: *const kda.Weights, x: T) anyerror!KdaInputs;
/// An MLP's gate and up projections of the decode lane's x [rows, hidden], each [rows, inter] bf16.
pub const GateUp = struct { gate: T, up: T };
pub const GateUpFn = fn (g: *G, gate: fp8.Linear, up: fp8.Linear, x: T, width: swiglu.Width, rows: u32) anyerror!GateUp;

/// The installed launches: each text's kernel object and its prepared configs (index M - 1), for the process (one model).
const KdaIn = struct {
    kernel: Kernel = .{ .ctx = null },
    cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    geo: [max_rows]KdaInGeometry = undefined,
};
const Pair = struct {
    kernel: Kernel = .{ .ctx = null },
    /// [`swiglu.Width`][M - 1].
    cfg: [2][max_rows]Config = @splat(@splat(.{ .ctx = null })),
    geo: [2][max_rows]PairGeometry = undefined,
};
var kda_in: KdaIn = .{};
var pair: Pair = .{};

/// The routes, set by `install` at construction, before any forward: the stock projections (the defaults) or the fused
/// launches.
var kda_route: *const KdaInputsFn = &kdaInputsStock;
var gate_up_route: *const GateUpFn = &gateUpStock;

/// A KDA layer's input projections on the installed route (kda.zig's forward and capture, kda_decode.zig's step).
pub fn kdaInputs(g: *G, w: *const kda.Weights, x: T) !KdaInputs {
    return kda_route(g, w, x);
}

/// A decode-lane MLP's gate and up on the installed route (graph.zig `mlpDecode`).
pub fn gateUp(g: *G, gate: fp8.Linear, up: fp8.Linear, x: T, width: swiglu.Width, rows: u32) !GateUp {
    return gate_up_route(g, gate, up, x, width, rows);
}

/// The routes as installed (the module's launch-fusion line).
pub fn kdaInputsEngaged() bool {
    return kda_route == &kdaInputsFused;
}
pub fn gateUpEngaged() bool {
    return gate_up_route == &gateUpFused;
}

/// `fuse_kda_inputs: off` (the control): each projection its own launch through the residents route (fp8.zig `project`).
fn kdaInputsStock(g: *G, w: *const kda.Weights, x: T) anyerror!KdaInputs {
    return .{
        .q = try fp8.project(g, w.q, x),
        .k = try fp8.project(g, w.k, x),
        .v = try fp8.project(g, w.v, x),
        .f_a = try fp8.project(g, w.f_a, x),
        .g_a = try fp8.project(g, w.g_a, x),
        .b = try fp8.project(g, w.b, x),
    };
}

/// `fuse_kda_inputs: on`: a call of at most `max_rows` rows is one launch at its rows' prepared config; a wider one (a
/// prompt span: the residents route's prompt path) keeps the stock route. The install proved every KDA layer's six
/// sites FP8 at the geometry's shapes.
fn kdaInputsFused(g: *G, w: *const kda.Weights, x: T) anyerror!KdaInputs {
    const sh = g.shapeOf(x);
    const nd: usize = sh.n;
    var m: c_int = 1;
    for (sh.d[0 .. nd - 1]) |d| m *= d;
    if (m > fp8.gemv_max_rows) return kdaInputsStock(g, w, x);
    const mi: usize = @intCast(m - 1);
    const geo = &kda_in.geo[mi];
    const ins = [_]T{
        try g.reshape(x, &.{ m, geo.k }),
        w.q.fp8.w,
        w.q.fp8.scales,
        w.k.fp8.w,
        w.k.fp8.scales,
        w.v.fp8.w,
        w.v.fp8.scales,
        w.f_a.fp8.w,
        w.f_a.fp8.scales,
        w.g_a.fp8.w,
        w.g_a.fp8.scales,
        w.b.fp8.w,
        w.b.fp8.scales,
    };
    var outs: [6]T = undefined;
    try launch(g, kda_in.kernel, kda_in.cfg[mi], &ins, &outs);
    // Each output [M, N] as the stock route hands it: x's leading dims, then N.
    var shape: [ops_be.max_dims]c_int = undefined;
    @memcpy(shape[0 .. nd - 1], sh.d[0 .. nd - 1]);
    var y: [6]T = undefined;
    for (outs, geo.segments(), &y) |o, sg, *yo| {
        shape[nd - 1] = sg.n;
        yo.* = try g.reshape(o, shape[0..nd]);
    }
    return .{ .q = y[0], .k = y[1], .v = y[2], .f_a = y[3], .g_a = y[4], .b = y[5] };
}

/// `fuse_gate_up: off` (the control): gate and up each their own launch through the residents route.
fn gateUpStock(g: *G, gate: fp8.Linear, up: fp8.Linear, x: T, _: swiglu.Width, _: u32) anyerror!GateUp {
    return .{ .gate = try fp8.project(g, gate, x), .up = try fp8.project(g, up, x) };
}

/// `fuse_gate_up: on`: one launch at the MLP's width and the call's rows (the decode lane's: `mlpDecode`'s rows <= 4,
/// inside the GEMV's). The install proved every MLP's gate and up FP8 at its width's geometry.
fn gateUpFused(g: *G, gate: fp8.Linear, up: fp8.Linear, x: T, width: swiglu.Width, rows: u32) anyerror!GateUp {
    std.debug.assert(rows >= 1 and rows <= max_rows);
    const wi: usize = @backingInt(width);
    const geo = &pair.geo[wi][rows - 1];
    const ins = [_]T{ try g.reshape(x, &.{ geo.m, geo.k }), gate.fp8.w, gate.fp8.scales, up.fp8.w, up.fp8.scales };
    var outs: [2]T = undefined;
    try launch(g, pair.kernel, pair.cfg[wi][rows - 1], &ins, &outs);
    return .{ .gate = outs[0], .up = outs[1] };
}

/// One launch of a prepared config; the outputs join `g`'s scope.
fn launch(g: *G, k: Kernel, cfg: Config, ins: []const T, outs: []T) !void {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, g.s));
    for (outs, 0..) |*o, i| {
        var y = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&y, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(y);
            return e;
        };
        o.* = try g.adopt(y);
    }
}

// ---- construction ----

pub const Switches = struct { kda_inputs: bool = false, gate_up: bool = false };

/// What a switch's install takes from the bound residents (pure: the CPU tests run it on synthetic layers): every KDA
/// layer's six input sites FP8 at q / k / v [kdaDim, hidden], f_a / g_a [kda_head_dim, hidden], b [kda_heads, hidden],
/// q / k / v on one plan and f_a / g_a / b on another at every M; every MLP's gate and up FP8 at its width [inter, hidden]
/// on one plan. Refused by name (`diag` names the first site that is not).
pub const Plan = struct {
    kda: [max_rows]KdaInGeometry = undefined,
    pair: [2][max_rows]PairGeometry = undefined,
};

pub fn plan(c: *const glm.Config, layers: []const graph.Layer, sw: Switches, diag: ?*sdk.Diag) !Plan {
    var p: Plan = .{};
    if (sw.kda_inputs) {
        const h = c.hidden_size;
        var first: ?*const kda.Weights = null;
        for (layers, 0..) |*lw, li| {
            const w = switch (lw.attn) {
                .kda => |*kw| kw,
                .dsa => continue,
            };
            const want = [6]struct { []const u8, fp8.Linear, u32 }{
                .{ "q_proj", w.q, c.kdaDim() },         .{ "k_proj", w.k, c.kdaDim() },         .{ "v_proj", w.v, c.kdaDim() },
                .{ "f_a_proj", w.f_a, c.kda_head_dim }, .{ "g_a_proj", w.g_a, c.kda_head_dim }, .{ "b_proj", w.b, c.kda_heads },
            };
            for (want) |s| {
                const f = switch (s[1]) {
                    .fp8 => |f| f,
                    .bf16 => return refuse(diag, error.Glm5NextFuseKdaInputsNotFp8, "glm5_next fuse_kda_inputs: layer {d} self_attn.{s} is BF16 (the fused launch reads FP8 sites only)", .{ li, s[0] }),
                };
                if (f.n != @as(c_int, @intCast(s[2])) or f.k != @as(c_int, @intCast(h))) return refuse(diag, error.Glm5NextFusionGeometry, "glm5_next fuse_kda_inputs: layer {d} self_attn.{s} is [{d}, {d}], not [{d}, {d}]", .{ li, s[0], f.n, f.k, s[2], h });
            }
            const f0 = first orelse w;
            first = f0;
            // q / k / v on one plan, f_a / g_a / b on another, every layer the first one's (one prepared launch per M).
            for (1..max_rows + 1) |m| {
                const big = w.q.fp8.plans[m];
                const small = w.f_a.fp8.plans[m];
                const same = std.meta.eql(w.k.fp8.plans[m], big) and std.meta.eql(w.v.fp8.plans[m], big) and
                    std.meta.eql(w.g_a.fp8.plans[m], small) and std.meta.eql(w.b.fp8.plans[m], small) and
                    std.meta.eql(f0.q.fp8.plans[m], big) and std.meta.eql(f0.f_a.fp8.plans[m], small);
                if (!same) return refuse(diag, error.Glm5NextFusionGeometry, "glm5_next fuse_kda_inputs: layer {d}'s plans at {d} rows differ (q / k / v and f_a / g_a / b each share one, every layer the first's)", .{ li, m });
            }
        }
        const f0 = first orelse return refuse(diag, error.Glm5NextFusionGeometry, "glm5_next fuse_kda_inputs: the model has no KDA layer", .{});
        for (&p.kda, 1..) |*geo, m| geo.* = kdaInGeometry(f0.q.fp8.plans[m], f0.f_a.fp8.plans[m], c.kdaDim(), c.kda_head_dim, c.kda_heads, c.hidden_size, @intCast(m)) catch |e|
            return refuse(diag, e, "glm5_next fuse_kda_inputs: no one launch holds the q / k / v and f_a / g_a / b plans at {d} rows", .{m});
    }
    if (sw.gate_up) {
        var seen = [2]bool{ false, false };
        for (layers, 0..) |*lw, li| {
            for ([_]struct { ?graph.Mlp, swiglu.Width }{ .{ lw.dense, .dense }, .{ lw.shared, .shared } }) |e| {
                const mlp = e[0] orelse continue;
                const wi: usize = @backingInt(e[1]);
                const inter: u32 = if (e[1] == .dense) c.intermediate_size else c.moe_intermediate_size * c.n_shared_experts;
                const gate = switch (mlp.gate) {
                    .fp8 => |f| f,
                    .bf16 => return refuse(diag, error.Glm5NextFuseGateUpNotFp8, "glm5_next fuse_gate_up: layer {d}'s gate_proj is BF16 (the fused launch reads FP8 sites only)", .{li}),
                };
                const up = switch (mlp.up) {
                    .fp8 => |f| f,
                    .bf16 => return refuse(diag, error.Glm5NextFuseGateUpNotFp8, "glm5_next fuse_gate_up: layer {d}'s up_proj is BF16 (the fused launch reads FP8 sites only)", .{li}),
                };
                const ni: c_int = @intCast(inter);
                const hi: c_int = @intCast(c.hidden_size);
                if (gate.n != ni or up.n != ni or gate.k != hi or up.k != hi)
                    return refuse(diag, error.Glm5NextFusionGeometry, "glm5_next fuse_gate_up: layer {d}'s gate / up are [{d}, {d}] / [{d}, {d}], not [{d}, {d}]", .{ li, gate.n, gate.k, up.n, up.k, inter, c.hidden_size });
                for (1..max_rows + 1) |m| {
                    if (!std.meta.eql(gate.plans[m], up.plans[m]) or (seen[wi] and !std.meta.eql(p.pair[wi][m - 1].p, gate.plans[m])))
                        return refuse(diag, error.Glm5NextFusionGeometry, "glm5_next fuse_gate_up: layer {d}'s gate / up plans at {d} rows differ (one launch takes one plan)", .{ li, m });
                    if (!seen[wi]) p.pair[wi][m - 1] = pairGeometry(gate.plans[m], inter, c.hidden_size, @intCast(m)) catch |err|
                        return refuse(diag, err, "glm5_next fuse_gate_up: layer {d}'s plan at {d} rows does not tile", .{ li, m });
                }
                seen[wi] = true;
            }
        }
        if (!seen[0] or !seen[1]) return refuse(diag, error.Glm5NextFusionGeometry, "glm5_next fuse_gate_up: the model lacks a dense MLP or a shared expert (dense {}, shared {})", .{ seen[0], seen[1] });
    }
    return p;
}

fn refuse(diag: ?*sdk.Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// Once, at construction (the module's GPU stream, after the residents are read): each switch on installs its text, its
/// launches at 1 .. `max_rows` rows and its route, after one launch at one row over the first site set and zero rows
/// (the compile check: Metal builds the text at its first dispatch); off keeps the stock route. Refused by name (`plan`,
/// the pins, a CPU stream, the compile check).
pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, layers: []const graph.Layer, sw: Switches, diag: *sdk.Diag) !void {
    kda_route = &kdaInputsStock;
    gate_up_route = &gateUpStock;
    if (!sw.kda_inputs and !sw.gate_up) return;
    const p = try plan(c, layers, sw, diag);
    checkPins() catch |e| {
        diag.set("glm5_next launch fusion: the embedded texts are not the pinned ones", .{});
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next launch fusion: the module's stream is not a GPU stream", .{});
        return error.Glm5NextFusionNotGpu;
    }
    if (sw.kda_inputs) {
        try installKdaIn(p.kda);
        compileCheck(a, s, c, layers, .kda_inputs) catch |e| return checkFailed(diag, "fuse_kda_inputs", e);
        kda_route = &kdaInputsFused;
        log.info("glm5_next launch fusion installed: fuse_kda_inputs (a KDA layer's q / k / v / f_a / g_a / b GEMVs one launch at <= {d} rows: {d} threadgroups of {d} simdgroups at 1 row, {d} at 4; 6 -> 1 launches a KDA layer, {d} fewer a forward); compile check passed\n", .{ max_rows, p.kda[0].threadgroups(), p.kda[0].tgs, p.kda[3].threadgroups(), 5 * c.nKda() });
    }
    if (sw.gate_up) {
        try installPair(p.pair);
        compileCheck(a, s, c, layers, .gate_up) catch |e| return checkFailed(diag, "fuse_gate_up", e);
        gate_up_route = &gateUpFused;
        log.info("glm5_next launch fusion installed: fuse_gate_up (the decode lane's MLP gate / up GEMVs one launch: dense [{d}, {d}], shared [{d}, {d}]; 2 -> 1 launches an MLP, {d} fewer a forward); compile check passed\n", .{ p.pair[0][0].n, p.pair[0][0].k, p.pair[1][0].n, p.pair[1][0].k, c.n_layers });
    }
}

fn checkFailed(diag: *sdk.Diag, comptime what: []const u8, e: anyerror) anyerror {
    var eb: [400]u8 = undefined;
    diag.set("glm5_next " ++ what ++ ": the compile check failed ({s}): {s}", .{ @errorName(e), mlx.takeError(&eb) orelse "" });
    return e;
}

/// The KDA text's kernel and its launches, once per process; the same geometry again is a no-op, another refused.
fn installKdaIn(geo: [max_rows]KdaInGeometry) !void {
    if (kda_in.kernel.ctx != null) {
        if (!std.meta.eql(kda_in.geo, geo)) return error.Glm5NextFusionGeometry;
        return;
    }
    kda_in = try buildKdaIn(geo, false);
}

fn buildKdaIn(geo: [max_rows]KdaInGeometry, comptime plain: bool) !KdaIn {
    var k: KdaIn = .{ .geo = geo };
    errdefer freeKdaIn(&k);
    k.kernel = try makeKernel("glm5n_fp8_gemv_kda_in", &kda_in_inputs, &kda_in_outputs, texts.kda_in, plain);
    for (&k.cfg, geo) |*cfg, gm| cfg.* = try kdaInConfig(gm);
    return k;
}

fn installPair(geo: [2][max_rows]PairGeometry) !void {
    if (pair.kernel.ctx != null) {
        if (!std.meta.eql(pair.geo, geo)) return error.Glm5NextFusionGeometry;
        return;
    }
    pair = try buildPair(geo, false);
}

fn buildPair(geo: [2][max_rows]PairGeometry, comptime plain: bool) !Pair {
    var k: Pair = .{ .geo = geo };
    errdefer freePair(&k);
    k.kernel = try makeKernel("glm5n_fp8_gemv_pair", &pair_inputs, &pair_outputs, texts.pair, plain);
    for (&k.cfg, geo) |*row, grow| for (row, grow) |*cfg, gm| {
        cfg.* = try pairConfig(gm);
    };
    return k;
}

/// Tests: the fused routes over `p` with MLX's own kernel objects (no launch cache: a CPU test's graphs never resolve a
/// pipeline), the routes switched by `use`; `done` frees them and restores what was installed and the stock routes.
pub const TestRoutes = struct {
    saved_kda: KdaIn,
    saved_pair: Pair,

    pub fn use(_: *TestRoutes, fused: bool) void {
        testUse(fused);
    }

    pub fn done(t: *TestRoutes) void {
        freeKdaIn(&kda_in);
        freePair(&pair);
        kda_in = t.saved_kda;
        pair = t.saved_pair;
        kda_route = &kdaInputsStock;
        gate_up_route = &gateUpStock;
    }
};

/// Tests: both routes switched to the fused launches (installed: `install` or `testInstall`) or to the stock ones.
pub fn testUse(fused: bool) void {
    kda_route = if (fused) &kdaInputsFused else &kdaInputsStock;
    gate_up_route = if (fused) &gateUpFused else &gateUpStock;
}

pub fn testInstall(p: Plan) !TestRoutes {
    const t: TestRoutes = .{ .saved_kda = kda_in, .saved_pair = pair };
    var k = try buildKdaIn(p.kda, true);
    errdefer freeKdaIn(&k);
    const pr = try buildPair(p.pair, true);
    kda_in = k;
    pair = pr;
    return t;
}

fn freeKdaIn(k: *KdaIn) void {
    if (k.kernel.ctx != null) mtl.mlx_fast_metal_kernel_free(k.kernel);
    for (k.cfg) |cfg| if (cfg.ctx != null) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    k.* = .{};
}

fn freePair(k: *Pair) void {
    if (k.kernel.ctx != null) mtl.mlx_fast_metal_kernel_free(k.kernel);
    for (k.cfg) |row| for (row) |cfg| if (cfg.ctx != null) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    k.* = .{};
}

/// One fused launch at one row over zero rows and the first fused site set, evaluated and checked for shape (Metal
/// builds the text at its first dispatch and refuses a threadgroup its registers cannot hold).
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, layers: []const graph.Layer, which: enum { kda_inputs, gate_up }) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    const h: c_int = @intCast(c.hidden_size);
    switch (which) {
        .kda_inputs => for (layers) |*lw| if (lw.attn == .kda) {
            const r = try kdaInputsFused(&g, &lw.attn.kda, try g.zeros(&.{ 1, 1, h }, .bfloat16));
            try g.evalAll(&.{ r.q, r.k, r.v, r.f_a, r.g_a, r.b });
            if (g.shapeOf(r.b).dim(2) != @as(c_int, @intCast(c.kda_heads))) return error.Glm5NextFusionShape;
            return;
        },
        .gate_up => for ([_]swiglu.Width{ .dense, .shared }) |width| {
            // The first MLP of each width (the plan found both).
            const mlp = for (layers) |*lw| {
                if (if (width == .dense) lw.dense else lw.shared) |m| break m;
            } else unreachable;
            const inter: u32 = if (width == .dense) c.intermediate_size else c.moe_intermediate_size * c.n_shared_experts;
            const r = try gateUpFused(&g, mlp.gate, mlp.up, try g.zeros(&.{ 1, h }, .bfloat16), width, 1);
            try g.evalAll(&.{ r.gate, r.up });
            if (g.shapeOf(r.up).dim(1) != @as(c_int, @intCast(inter))) return error.Glm5NextFusionShape;
        },
    }
}
