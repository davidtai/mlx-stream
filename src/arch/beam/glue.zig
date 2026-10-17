//! Beam's decode glue (`decode_glue: fused`, the default; `ops` is the stock control): a decode-lane forward's router
//! (one row, or up to the decode lane's rows in the decode phase) and the lookahead's predictor in two texts
//! (src/kernels/decode_glue/, pinned by manifest_beam.json), each the stock op chain's arithmetic:
//!   - `glue_router_gemv`: the f32 product of the FFN sublayer's normed row with the layer's router weight (EPI 0: the
//!     logits), or with the next routed layer's, then its sigmoid and correction (EPI 1: the predictor's scores); each
//!     sum MLX's gemv at M = 1 (gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4, the stock route's `astype` and `matmul`), in
//!     `rows_per_group` rows of the weight a threadgroup;
//!   - `glue_router_select`: the tail (sigmoid, correction, the group top-k, the top-k, the normalized weights).
//! Stock: the router is the input cast, the GEMV and 17 ops (sigmoid, add, the groups' topk and sum, negation,
//! argpartition, the group ids, equal, sum, greater, where, negation, argpartition, take_along_axis, sum, add, divide);
//! the predictor is the cast, the GEMV, sigmoid and add. Exact: the ids, the weights and the predictor's scores are the
//! stock route's bit for bit (`install`'s self-check over the first routed layers' tables; the device tests `beam glue
//! GPU`). At M > 1 the router's product stays MLX's matmul (MLX routes [M > 1, D] x [D, E] f32 to its GEMM: reproduced
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

// ── the texts and their pins (src/kernels/decode_glue/manifest_beam.json) ──

const dir = "../../kernels/decode_glue/";

pub const texts = struct {
    pub const manifest = @embedFile(dir ++ "manifest_beam.json");
    pub const header = @embedFile(dir ++ "header_decode_glue.metal");
    pub const gemv = @embedFile(dir ++ "glue_router_gemv.metal");
    pub const select = @embedFile(dir ++ "glue_router_select.metal");
};
/// sha256 of manifest_beam.json (scripts/decode_glue_manifest.py prints it), which records every text's sha256 (checked
/// against the pins by a CPU test).
pub const manifest_sha256 = "1d1b8316fecab645e309d8138251184b389a30cbdc30bc97074dd808efcd2942";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_decode_glue.metal", .text = texts.header, .sha256 = "6bec0858b4e0754c74a4eee95560bd78f0942ecc092cc50d50bcc7840977e484" },
    .{ .file = "glue_router_gemv.metal", .text = texts.gemv, .sha256 = "1c82d26867b1c277adb2ca003200f6c05c3ecc396126b74ac571854b202bdcb7" },
    .{ .file = "glue_router_select.metal", .text = texts.select, .sha256 = "5a7369ad5c53e6eb5b08013219ad7731cfa797863599d9be8f2c2e4715bf47f6" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.BeamGlueManifestNotPinned, error.BeamGlueTextNotPinned);
}

/// A routed geometry the manifest registers (scripts/decode_glue_manifest.py `CASES`).
pub const Geometry = struct { hidden: u32, experts: u32, groups: u32, kept_groups: u32, top_k: u32 };

/// Beam 1.0's and the converter's fixture's (scripts/beam_fixture.py: hidden 256, 8 experts in 2 groups, 1 kept, top-2).
pub const registered_geometries = [_]Geometry{
    .{ .hidden = 5120, .experts = 256, .groups = 8, .kept_groups = 4, .top_k = 8 },
    .{ .hidden = 256, .experts = 8, .groups = 2, .kept_groups = 1, .top_k = 2 },
};

pub fn geometryOf(c: *const beam.Config) Geometry {
    return .{ .hidden = c.hidden_size, .experts = c.n_routed_experts, .groups = c.n_group, .kept_groups = c.topk_group, .top_k = c.n_experts_per_tok };
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

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.gemv, k.select }) |x| if (x.ctx != null) {
            mtl.mlx_fast_metal_kernel_free(x);
        };
        for ([_]Config{ k.route_cfg, k.pred_cfg } ++ k.select_cfg) |cfg| if (cfg.ctx != null) {
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
};

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

fn selectConfig(geo: Geometry, rows: u32) !Config {
    const l = Launch.select(geo.experts, rows);
    const shape = [_]c_int{ @intCast(rows), @intCast(geo.top_k) };
    const t = templateOf(&.{ "E", "G", "TG", "K" }, .{ @intCast(geo.experts), @intCast(geo.groups), @intCast(geo.kept_groups), @intCast(geo.top_k) });
    return kset.buildConfig(.{ .grid = l.grid, .threadgroup = l.group, .out_shapes = &.{ &shape, &shape }, .out_dtypes = &.{ .uint32, .float32 }, .template = &t });
}

/// The model's tables the self-check runs over (construction only): the first routed layer's router (its F32 weight
/// and correction bias) and the next routed layer's (the predictor's).
pub const CheckTables = struct { w: T, bias: T, next_w: T, next_bias: T };

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
    var k: Kernels = .{ .hidden = geo.hidden, .experts = geo.experts, .top_k = geo.top_k };
    errdefer k.deinit();
    k.gemv = try newKernel("beam_glue_router_gemv", &.{ "x", "w", "bias" }, &.{"out"}, texts.gemv);
    k.select = try newKernel("beam_glue_router_select", &.{ "logits", "bias", "consts" }, &.{ "ids", "weights" }, texts.select);
    k.route_cfg = try gemvConfig(geo, 0);
    k.pred_cfg = try gemvConfig(geo, 1);
    for (&k.select_cfg, 1..) |*cfg, m| cfg.* = try selectConfig(geo, @intCast(m));
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

/// `want` and `got` evaluated and equal byte for byte (the same dtype, shape and bytes), else error.BeamGlueCheck, the
/// first differing element named in `diag` (or printed without one).
pub fn sameBytes(g: *G, name: []const u8, want: T, got: T, diag: ?*sdk.Diag) !void {
    try g.evalAll(&.{ want, got });
    const n = mlx.mlx_array_size(want);
    if (mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got) or n != mlx.mlx_array_size(got) or !std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got)))
        return mismatch(diag, "{s}: the dtype or the shape differs", .{name});
    const size = n * mlx.mlx_array_itemsize(want);
    const wb = (mlx.mlx_array_data_uint8(want) orelse return error.MlxNoData)[0..size];
    const ob = (mlx.mlx_array_data_uint8(got) orelse return error.MlxNoData)[0..size];
    if (std.mem.indexOfDiff(u8, wb, ob)) |i| return mismatch(diag, "{s}: element {d} of {d} differs", .{ name, i / mlx.mlx_array_itemsize(want), n });
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
    var g = try G.init(a, s);
    defer g.deinit();
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
}

const testing = std.testing;

test "beam glue: the embedded manifest and texts are the pinned ones" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 2);
}

test "beam glue: Beam 1.0's and the fixture's routers are the texts'; another is refused by name" {
    const c: beam.Config = .{};
    try testing.expect(registered(&c));
    try testing.expect(registered(&.{ .hidden_size = 256, .n_routed_experts = 8, .n_group = 2, .topk_group = 1, .n_experts_per_tok = 2 }));
    for ([_]beam.Config{
        .{ .hidden_size = 4096 },
        .{ .n_experts_per_tok = 6 },
        .{ .n_group = 4 },
        .{ .topk_group = 2 },
        .{ .n_routed_experts = 128 },
        .{ .norm_topk_prob = false },
    }) |bad| try testing.expect(!registered(&bad));
    // Every registered geometry is one MLX's gemv runs as bm1_bn8 (matmul.cpp `gemv_axbpy`: K >= 16 N, K > 64) in whole
    // threadgroups of the GEMV and within one threadgroup of the select text.
    for (registered_geometries) |r| {
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
            } else if (int(o, "E") == geo.experts and int(o, "G") == geo.groups and int(o, "TG") == geo.kept_groups and int(o, "K") == geo.top_k) {
                select_found[@intCast(int(o, "M") - 1)] = true;
            }
        };
        for (gemv_found) |f| try testing.expect(f);
        for (select_found) |f| try testing.expect(f);
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

/// The model of a registered geometry (the router's fields; the rest Beam's).
fn configOf(geo: Geometry) beam.Config {
    return .{ .hidden_size = geo.hidden, .n_routed_experts = geo.experts, .n_group = geo.groups, .topk_group = geo.kept_groups, .n_experts_per_tok = geo.top_k };
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
        try g.evalAll(&.{ w, bias, w2, bias2, bias_ties });
        var diag: sdk.Diag = .{};
        var k = install(a, s, &c, .{ .w = w, .bias = bias, .next_w = w2, .next_bias = bias2 }, &diag) catch |err| {
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
    var diag: sdk.Diag = .{};
    var k = try install(a, s, &c, .{ .w = ws[0], .bias = bs[0], .next_w = ws[1], .next_bias = bs[1] }, &diag);
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
