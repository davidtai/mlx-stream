//! glm5_next's trunk launch fusion (reports/dsv41-f39-tcq3-runtime/mimo/glm5next-audit-decode-20261014.md rank 8): the
//! decode trunk's launches that carry one input's work merged into one launch each, construction-time routes, each off by
//! default (`settings.Fusion`: `off` keeps the stock launches, g22-exl3's routes; the scheduler's merged event waits,
//! `merge_gate_waits`, live in deepseek_v41/experts.zig):
//!   fuse_kda_inputs  a KDA layer's six input projections of the same rows (q, k, v, f_a, g_a, b) in one launch
//!                    (`glm5n_fp8_gemv_kda_in`) at every call of at most `max_rows` rows (the MTP verify, a serial step,
//!                    a short prompt call): 6 -> 1 launches a KDA layer, 170 fewer a forward of Flash's 34 KDA layers
//!   fuse_gate_up     the decode lane's MLPs' gate and up projections (every routed layer's shared expert, the dense
//!                    layers' MLP) in one launch (`glm5n_fp8_gemv_pair`): 2 -> 1 an MLP, 45 fewer a forward (42 + 3)
//!   fuse_predictor   the lookahead predictor's epilogue (the f32 widening, the sigmoid and the correction over the next
//!                    routed layer's BF16 router product: experts.zig `scoresBf16`) in one launch (`glm5n_pred_epilogue`):
//!                    3 -> 1 a routed decode call, 84 fewer a forward (the predictor runs on the decode lane only)
//!   fuse_join        the routed call's join (the take of its outputs into router order) folded into the decode combine
//!                    (`glm5n_moe_combine_idx`: the combine text reading each routed row in place through the join's order):
//!                    the take gone, 42 fewer a forward
//! Exact by construction. The GEMV fusions: a launch's segments each run MiMo's FP8 GEMV code line for line (kernels/
//! header_glm5n_fp8_multi.metal `glm5n_fp8_gemv_rows`, the pinned mimo_fp8_gemv.metal's body) at their own site's plan:
//! its K partitions (KS) fix every output's summation order, and R and RG only pick which rows a simdgroup owns
//! (fp8_linear.zig `gemvPlanAt`'s contract), so every output is the stock launch's bit for bit. The outputs are the stock
//! launches' arrays (one per projection, the same shapes and dtype), so no bill term moves. The epilogue: element for element
//! MLX's own arithmetic of the three ops it replaces (the exact widening, unary_ops.h's Sigmoid with precise::exp, the f32
//! add); its output is the stock route's array (f32 [M, E]); no intermediate is held, so no bill term moves either. The
//! indexed combine: glm5n_moe_combine's lines but the row reads (the same bf16 values the take would copy, read in place);
//! one array fewer (the take's [n * 8, hidden] copy), no new one.
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
const glue = @import("glue.zig");
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
    pub const pred = @embedFile("kernels/glm5n_pred_epilogue.metal");
    pub const combine_idx = @embedFile("kernels/glm5n_moe_combine_idx.metal");
};
/// The header MLX compiles the GEMV texts with: MiMo's FP8 decoders (header_fp8res.metal, pinned there), then the segment
/// function. (The epilogue takes none; the indexed combine glue.zig's combine header, pinned there.)
const launch_header = mimo_fp8.texts.header ++ texts.header;
/// sha256 of kernels/manifest-fusion.json, which records every text's sha256 (checked against `text_pins` by a CPU test).
pub const manifest_sha256 = "9c7d3a95f4805e5656dc1dff4f2fbbb5eab0b35cfcd860880a3985b76b3dcc5b";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_glm5n_fp8_multi.metal", .text = texts.header, .sha256 = "44cac691ddd7a605823eff01acaec1f6b20a5d2a202af386241e104b3e96a914" },
    .{ .file = "glm5n_fp8_gemv_kda_in.metal", .text = texts.kda_in, .sha256 = "8dab99d935e66e9c5deaa8f48a108e719a28e4105a9cf06b9ede6455e785e506" },
    .{ .file = "glm5n_fp8_gemv_pair.metal", .text = texts.pair, .sha256 = "fe58d2cd4645d4a80ffc2ce6b54f50e3a3114e743d3433ca3a08f68c9f6ef756" },
    .{ .file = "glm5n_pred_epilogue.metal", .text = texts.pred, .sha256 = "82556142ed171e6c2e0c7c64934e07fd09a544f65d075068b329ff0ef0f7d4be" },
    .{ .file = "glm5n_moe_combine_idx.metal", .text = texts.combine_idx, .sha256 = "3fbae8287b70da19da3ccc4dbb2039353e83ec5b7747962f41ed768b232cc4ce" },
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

/// The launch of `glm5n_pred_epilogue` at M rows of E experts: one output [M, E] f32, one threadgroup of E threads a row.
pub fn predConfig(m: c_int, e: c_int) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ m, e }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, m * e, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, e, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "E", e));
    return cfg;
}

/// The launch of `glm5n_moe_combine_idx` at M rows: glue.zig's combine launch (one thread per column and row, 256 columns a
/// threadgroup, T / TOPK / H), one output [M, H] bf16.
pub fn combineIdxConfig(m: c_int) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const geo = glue.Geometry.combine(@intCast(m));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ m, @intCast(glue.hidden) }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, geo.grid[0], geo.grid[1], geo.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, geo.group[0], geo.group[1], geo.group[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", @intCast(glue.top_k)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", @intCast(glue.hidden)));
    return cfg;
}

pub const combine_idx_inputs = [_][*:0]const u8{ "rows", "order", "w", "shared" };
pub const combine_idx_outputs = [_][*:0]const u8{"y"};

pub const pred_inputs = [_][*:0]const u8{ "logits", "correction" };
pub const pred_outputs = [_][*:0]const u8{"scores"};

pub const kda_in_inputs = [_][*:0]const u8{ "x", "w_q", "s_q", "w_k", "s_k", "w_v", "s_v", "w_fa", "s_fa", "w_ga", "s_ga", "w_b", "s_b" };
pub const kda_in_outputs = [_][*:0]const u8{ "y_q", "y_k", "y_v", "y_fa", "y_ga", "y_b" };
pub const pair_inputs = [_][*:0]const u8{ "x", "w_gate", "s_gate", "w_up", "s_up" };
pub const pair_outputs = [_][*:0]const u8{ "y_gate", "y_up" };

/// A text's kernel object through the plugin's launch cache (`plain`: MLX's own object, the CPU tests' graphs).
fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, comptime plain: bool) !Kernel {
    return makeKernelWith(name, ins, outs, source, launch_header, plain);
}

fn makeKernelWith(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, comptime plain: bool) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const new = if (plain) mlx.mlx_fast_metal_kernel_new else mtl.mlx_fast_metal_kernel_new;
    const k = new(name, in_vec, out_vec, source, header, true, false);
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
const Pred = struct {
    kernel: Kernel = .{ .ctx = null },
    cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    experts: c_int = 0,
};
const Join = struct {
    kernel: Kernel = .{ .ctx = null },
    cfg: [max_rows]Config = @splat(.{ .ctx = null }),
};
var kda_in: KdaIn = .{};
var pair: Pair = .{};
var pred: Pred = .{};
var join: Join = .{};

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

/// `fuse_predictor`'s epilogue over the next routed layer's BF16 router product `logits` [rows, E] and its correction bias
/// f32 [E] (experts.zig `scoresBf16Fused`, bound by the module once `install` installed the text): f32 [rows, E], one
/// launch at the rows' prepared config. `rows` within the prepared ones (the caller's row-count route).
pub fn predEpilogue(g: *G, logits: T, correction: T, rows: u32) !T {
    std.debug.assert(rows >= 1 and rows <= max_rows);
    var out: [1]T = undefined;
    try launch(g, pred.kernel, pred.cfg[rows - 1], &.{ logits, correction }, &out);
    return out[0];
}

/// The epilogue's text installed (the module binds the predictor route only then).
pub fn predInstalled() bool {
    return pred.kernel.ctx != null;
}

/// `fuse_join`'s combine (experts.zig `forwardSharedIndexed`, bound by the module once `install` installed the text): the
/// call's concatenated outputs `rows` [R, hidden] bf16 read through the join's `order` u32 [nt * 8], the route weights f32
/// [nt, 8] and the shared expert's rows bf16 [nt, hidden] -> bf16 [nt, hidden], one launch at the rows' prepared config.
pub fn combineIndexed(g: *G, rows: T, order: T, weights: T, shared: T, nt: u32) !T {
    std.debug.assert(nt >= 1 and nt <= max_rows);
    var out: [1]T = undefined;
    try launch(g, join.kernel, join.cfg[nt - 1], &.{ rows, order, weights, shared }, &out);
    return out[0];
}

/// prebuild.zig: one launch of each installed fused text's instantiation at `rows` rows over zero rows and the first fused
/// sites (the GEMV texts at every width; the epilogue and the combine, one library each, at one row), the outputs appended
/// to `outs` for the caller's evaluation. Returns the launches.
pub fn prebuild(g: *G, c: *const glm.Config, layers: []const graph.Layer, rows: u32, outs: *std.ArrayList(T), a: std.mem.Allocator) !u32 {
    var n: u32 = 0;
    const h: c_int = @intCast(c.hidden_size);
    const m: c_int = @intCast(rows);
    if (kda_in.kernel.ctx != null) for (layers) |*lw| if (lw.attn == .kda) {
        const r = try kdaInputsFused(g, &lw.attn.kda, try g.zeros(&.{ 1, m, h }, .bfloat16));
        try outs.appendSlice(a, &.{ r.q, r.k, r.v, r.f_a, r.g_a, r.b });
        n += 1;
        break;
    };
    if (pair.kernel.ctx != null) for ([_]swiglu.Width{ .dense, .shared }) |width| {
        const mlp = for (layers) |*lw| {
            if (if (width == .dense) lw.dense else lw.shared) |x| break x;
        } else continue;
        const r = try gateUpFused(g, mlp.gate, mlp.up, try g.zeros(&.{ m, h }, .bfloat16), width, rows);
        try outs.appendSlice(a, &.{ r.gate, r.up });
        n += 1;
    };
    if (rows == 1 and pred.kernel.ctx != null) {
        try outs.append(a, try predEpilogue(g, try g.zeros(&.{ 1, pred.experts }, .bfloat16), try g.zeros(&.{pred.experts}, .float32), 1));
        n += 1;
    }
    if (rows == 1 and join.kernel.ctx != null) {
        const tk: c_int = @intCast(glue.top_k);
        try outs.append(a, try combineIndexed(g, try g.zeros(&.{ tk, h }, .bfloat16), try g.zeros(&.{tk}, .uint32), try g.zeros(&.{ 1, tk }, .float32), try g.zeros(&.{ 1, h }, .bfloat16), 1));
        n += 1;
    }
    return n;
}

/// The indexed combine installed (the module binds the join route only then).
pub fn joinInstalled() bool {
    return join.kernel.ctx != null;
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

pub const Switches = struct { kda_inputs: bool = false, gate_up: bool = false, predictor: bool = false, join: bool = false };

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
    if (!sw.kda_inputs and !sw.gate_up and !sw.predictor and !sw.join) return;
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
    if (sw.predictor) {
        const e: c_int = @intCast(c.n_routed_experts);
        if (e < 1 or e > 1024) {
            diag.set("glm5_next fuse_predictor: {d} experts do not fit one threadgroup a row", .{e});
            return error.Glm5NextFusionGeometry;
        }
        try installPred(e);
        compileCheck(a, s, c, layers, .predictor) catch |err| return checkFailed(diag, "fuse_predictor", err);
        log.info("glm5_next launch fusion installed: fuse_predictor (the lookahead predictor's f32 widening, sigmoid and correction one launch over [rows, {d}]; 3 -> 1 a routed decode call, {d} fewer a forward); compile check passed\n", .{ e, 2 * c.moeLayers() });
    }
    if (sw.join) {
        // The decode glue's combine geometry (glue.zig `requireGeometry`: hidden 4096, top-8): the text's.
        if (c.hidden_size != glue.hidden or c.n_experts_per_tok != glue.top_k) {
            diag.set("glm5_next fuse_join: the model is not the combine text's (hidden {d}, top-{d}; the text takes {d}, top-{d})", .{ c.hidden_size, c.n_experts_per_tok, glue.hidden, glue.top_k });
            return error.Glm5NextFusionGeometry;
        }
        try installJoin();
        compileCheck(a, s, c, layers, .join) catch |err| return checkFailed(diag, "fuse_join", err);
        log.info("glm5_next launch fusion installed: fuse_join (a routed decode call's join take folded into the combine: the rows read in place through the join's order; {d} fewer a forward); compile check passed\n", .{c.moeLayers()});
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

/// The indexed combine's text and its launches at 1 .. `max_rows` rows, once per process.
fn installJoin() !void {
    if (join.kernel.ctx != null) return;
    join = try buildJoin(false);
}

fn buildJoin(comptime plain: bool) !Join {
    var k: Join = .{};
    errdefer freeJoin(&k);
    k.kernel = try makeKernelWith("glm5n_moe_combine_idx", &combine_idx_inputs, &combine_idx_outputs, texts.combine_idx, glue.texts.header_combine, plain);
    for (&k.cfg, 1..) |*cfg, m| cfg.* = try combineIdxConfig(@intCast(m));
    return k;
}

fn freeJoin(k: *Join) void {
    if (k.kernel.ctx != null) mtl.mlx_fast_metal_kernel_free(k.kernel);
    for (k.cfg) |cfg| if (cfg.ctx != null) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    k.* = .{};
}

/// The epilogue's text and its launches at 1 .. `max_rows` rows, once per process (another expert count refused).
fn installPred(e: c_int) !void {
    if (pred.kernel.ctx != null) {
        if (pred.experts != e) return error.Glm5NextFusionGeometry;
        return;
    }
    pred = try buildPred(e, false);
}

fn buildPred(e: c_int, comptime plain: bool) !Pred {
    var k: Pred = .{ .experts = e };
    errdefer freePred(&k);
    k.kernel = try makeKernelWith("glm5n_pred_epilogue", &pred_inputs, &pred_outputs, texts.pred, "", plain);
    for (&k.cfg, 1..) |*cfg, m| cfg.* = try predConfig(@intCast(m), e);
    return k;
}

fn freePred(k: *Pred) void {
    if (k.kernel.ctx != null) mtl.mlx_fast_metal_kernel_free(k.kernel);
    for (k.cfg) |cfg| if (cfg.ctx != null) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    k.* = .{};
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
    saved_pred: Pred,
    saved_join: Join,

    pub fn use(_: *TestRoutes, fused: bool) void {
        testUse(fused);
    }

    pub fn done(t: *TestRoutes) void {
        freeKdaIn(&kda_in);
        freePair(&pair);
        freePred(&pred);
        freeJoin(&join);
        kda_in = t.saved_kda;
        pair = t.saved_pair;
        pred = t.saved_pred;
        join = t.saved_join;
        kda_route = &kdaInputsStock;
        gate_up_route = &gateUpStock;
    }
};

/// Tests: both routes switched to the fused launches (installed: `install` or `testInstall`) or to the stock ones.
pub fn testUse(fused: bool) void {
    kda_route = if (fused) &kdaInputsFused else &kdaInputsStock;
    gate_up_route = if (fused) &gateUpFused else &gateUpStock;
}

pub fn testInstall(p: Plan, experts: c_int) !TestRoutes {
    const t: TestRoutes = .{ .saved_kda = kda_in, .saved_pair = pair, .saved_pred = pred, .saved_join = join };
    var k = try buildKdaIn(p.kda, true);
    errdefer freeKdaIn(&k);
    var pr = try buildPair(p.pair, true);
    errdefer freePair(&pr);
    var pe = try buildPred(experts, true);
    errdefer freePred(&pe);
    const jn = try buildJoin(true);
    kda_in = k;
    pair = pr;
    pred = pe;
    join = jn;
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
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, layers: []const graph.Layer, which: enum { kda_inputs, gate_up, predictor, join }) !void {
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
        .predictor => {
            const e: c_int = @intCast(c.n_routed_experts);
            const r = try predEpilogue(&g, try g.zeros(&.{ 1, e }, .bfloat16), try g.zeros(&.{e}, .float32), 1);
            try g.evalAll(&.{r});
            if (g.shapeOf(r).dim(1) != e) return error.Glm5NextFusionShape;
        },
        .join => {
            const tk: c_int = @intCast(glue.top_k);
            const r = try combineIndexed(&g, try g.zeros(&.{ tk, h }, .bfloat16), try g.zeros(&.{tk}, .uint32), try g.zeros(&.{ 1, tk }, .float32), try g.zeros(&.{ 1, h }, .bfloat16), 1);
            try g.evalAll(&.{r});
            if (g.shapeOf(r).dim(1) != h) return error.Glm5NextFusionShape;
        },
    }
}
