//! GLM-5.3-Flash's decode-lane glue in fused texts (`decode_glue: fused`, the default; `ops` is the stock control), every
//! decode-lane forward (one row, or the MTP verify's <= 4 in the decode phase); a wide call keeps the stock glue (a phase
//! route: the prompt's rows):
//!   - each sublayer's mHC collapse and its norm in ONE dispatch with the previous sublayer's expand folded in (the host's
//!     `hcPre` / `Deferred` chain, ops.zig: oMLX's hc_pre_fused, the decode path mlx-serve's glm5_next takes): stock is
//!     the mixes' op chain, the Sinkhorn collapse kernel, the norm and the expand kernel per sublayer;
//!   - the router's product (`glm5n_glue_router`: MiMo's mimo_glue_router_rows with E experts, the router's BF16 weight)
//!     and its tail (`glm5n_glue_top8`: MiMo's mimo_glue_top8 with E experts and routed_scaling): stock is the F32 GEMV on
//!     the transposed F32 copy, sigmoid, bias, negation, argpartition, slice, take_along_axis, sum, add, divide, multiply
//!     and the scheduler's index cast;
//!   - the routed combine with the shared expert's add (`glm5n_moe_combine`: MiMo's mimo_moe_combine plus the add): stock
//!     is cast, multiply, sum, cast, add.
//! Class: rounding-class against `ops` (hcPre sums the mixes and the norm's squares in its own order; the router's product
//! runs MLX gemv's non-transposed order on the BF16 weight where the stock route's F32 GEMV runs gemv_t on the F32 copy:
//! the same values, another summation order); the top-8 tail and the combine are the stock expressions (exact given
//! their inputs). Installed once at construction (`install`: the geometry refused by name, the kernel objects, the
//! launch configurations per decode-lane width, and one launch of each text over the model's own tables as the compile
//! check, which also fixes hcPre's threadgroup width); the forward then dispatches with no per-call config work.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const ops = @import("ops.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// The decode lane's glue route (`settings.Config.decode_glue`).
pub const Route = @import("settings.zig").DecodeGlue;

/// The decode lane's widest call: the MTP verify's rows (`experts.decode_lane_rows`).
pub const max_rows: u32 = 4;
/// The hidden width the texts are written for (`GLM5N_D`), the routed top-k.
pub const hidden: u32 = 4096;
pub const top_k: u32 = 8;
/// The combine's TOPK: the routed top-k.
const combine_threads: u32 = 256;

comptime {
    if (max_rows > ops.HC_PRE_MAX_ROWS) @compileError("glm5_next glue: the decode lane is wider than hcPre's rows");
    if (max_rows != @import("experts.zig").decode_lane_rows) @compileError("glm5_next glue: the configs do not cover the decode lane");
}

// ── the texts and their pins (kernels/manifest-glue.json) ──

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-glue.json");
    pub const header = @embedFile("kernels/header_glm5n_glue.metal");
    pub const header_combine = @embedFile("kernels/header_glm5n_combine.metal");
    pub const router = @embedFile("kernels/glm5n_glue_router.metal");
    pub const top8 = @embedFile("kernels/glm5n_glue_top8.metal");
    pub const combine = @embedFile("kernels/glm5n_moe_combine.metal");
};
/// sha256 of kernels/manifest-glue.json, which records every text's sha256 (checked against the pins by a CPU test).
pub const manifest_sha256 = "85c4c7ee403118c624e6af02fd52bdf0f81afb277129052febac3b3b8bd40c46";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_glm5n_glue.metal", .text = texts.header, .sha256 = "6bab9e2ca4ae508bd2f6dec8f5638bd218ca0161f3c80bdf4dd5010acbf93833" },
    .{ .file = "header_glm5n_combine.metal", .text = texts.header_combine, .sha256 = "82e9d3c3c425164d73522e8ca59208212487ca19f8e6d628ca63e35b32d8ff16" },
    .{ .file = "glm5n_glue_router.metal", .text = texts.router, .sha256 = "c3d329868582f41f9a60d88fdd8c713eddf1ed7dd15b7b97454c919f1b4a6a0c" },
    .{ .file = "glm5n_glue_top8.metal", .text = texts.top8, .sha256 = "1c46a2abdb760b6ea9fab7a5995a1e028c523e7b86486e37298e34b8d7fb29c3" },
    .{ .file = "glm5n_moe_combine.metal", .text = texts.combine, .sha256 = "bc5d18b44cf27021a97a9324d2109dad669ba8660510c01196ff7710bb0f2c86" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.Glm5NextGlueManifestNotPinned, error.Glm5NextGlueTextNotPinned);
}

/// The model the texts serve, refused by name otherwise (once, at construction): the hidden width the texts are written
/// for, top-8, the router's experts in whole 4-expert threadgroups and one threadgroup of the top-8 (at most 1,024, whole
/// simdgroups), norm_topk_prob, and hcPre's stream (4 streams, a width in whole float4s within its threadgroup, its
/// K-slices whole).
pub fn requireGeometry(c: *const glm.Config) error{Glm5NextGlueGeometry}!void {
    const e = c.n_routed_experts;
    if (c.hidden_size != hidden or c.n_experts_per_tok != top_k or e % 4 != 0 or e % 32 != 0 or e > 1024 or e < top_k or !c.norm_topk_prob)
        return error.Glm5NextGlueGeometry;
    if (c.hc_mult != @as(u32, @intCast(ops.HC)) or c.hidden_size % 4 != 0 or c.hidden_size / 4 > 1024 or (c.hc_mult * c.hidden_size) % 16 != 0)
        return error.Glm5NextGlueGeometry;
}

// ── the installed kernels ──

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    router: Kernel = .{ .ctx = null },
    top8: Kernel = .{ .ctx = null },
    combine: Kernel = .{ .ctx = null },
    /// Index M - 1: the launch of a decode-lane call of M rows.
    router_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    top8_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    combine_cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: routed_scaling, then zeros (eight words: MLX passes a smaller input in the constant address space).
    consts: T = .{ .ctx = null },
    experts: u32 = 0,

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.router, k.top8, k.combine }) |x| if (x.ctx != null) {
            _ = mtl.mlx_fast_metal_kernel_free(x);
        };
        for (k.router_cfg ++ k.top8_cfg ++ k.combine_cfg) |cfg| if (cfg.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        if (k.consts.ctx != null) _ = mlx.mlx_array_free(k.consts);
        k.* = .{};
    }
};

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // The inputs are the forward's fresh row-major arrays and the residents (a view would be copied: ensure_row_contiguous).
    const k = mtl.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.Glm5NextGlueKernel;
    return k;
}

/// One launch configuration: the outputs (shape, dtype), the grid and threadgroup (threads), the template arguments.
const Out = struct { shape: []const c_int, dtype: mlx.mlx_dtype };
const TArg = struct { name: [*:0]const u8, value: union(enum) { int: c_int, dtype: mlx.mlx_dtype } };

fn makeConfig(outs: []const Out, grid: [3]c_int, group: [3]c_int, targs: []const TArg) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group[0], group[1], group[2]));
    for (targs) |t| switch (t.value) {
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, v)),
        .dtype => |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, d)),
    };
    return cfg;
}

/// The launch geometry of each text at a decode-lane width (CPU-tested; `install` builds the configs from it).
pub const Geometry = struct {
    grid: [3]c_int,
    group: [3]c_int,

    /// The router: E / 4 threadgroups of 256 threads (grid in threads: 256 x E / 4).
    pub fn router(experts: u32) Geometry {
        return .{ .grid = .{ 256, @intCast(experts / 4), 1 }, .group = .{ 256, 1, 1 } };
    }

    /// The top-8: one E-thread threadgroup per row.
    pub fn top8(experts: u32, rows: u32) Geometry {
        return .{ .grid = .{ @intCast(experts), @intCast(rows), 1 }, .group = .{ @intCast(experts), 1, 1 } };
    }

    /// The combine: one thread per (column, row), 256 columns a threadgroup.
    pub fn combine(rows: u32) Geometry {
        return .{ .grid = .{ @intCast(hidden), @intCast(rows), 1 }, .group = .{ combine_threads, 1, 1 } };
    }
};

/// The model's tables the compile check launches over (construction only): layer 0's attention hyper-connection and
/// input norm, the first routed layer's router (BF16 weight, F32 correction bias).
pub const CheckTables = struct { fn_rows: T, scale: T, base: T, norm_w: T, router_w: T, router_bias: T };

/// The route's construction on the module's GPU stream `s` (refused by name: `diag`): the geometry, the three texts'
/// kernel objects, every decode-lane width's configs, then the compile check: one launch of each text and of both hcPre
/// variants over zeros and the model's own tables, evaluated (Metal compiles a text at its first dispatch; hcPre's first
/// launch fixes its threadgroup width), the routed ids checked distinct and in range.
pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, tables: CheckTables, diag: *sdk.Diag) !Kernels {
    requireGeometry(c) catch |e| {
        diag.set("glm5_next decode_glue=fused: the model is not the texts' (hidden {d}, top-{d}, {d} experts, norm_topk_prob {}, hc {d}; the texts take hidden 4096, top-8, experts a multiple of 32 up to 1024)", .{ c.hidden_size, c.n_experts_per_tok, c.n_routed_experts, c.norm_topk_prob, c.hc_mult });
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next decode_glue=fused: the module's stream is not a GPU stream", .{});
        return error.Glm5NextGlueNotGpu;
    }
    const e: u32 = c.n_routed_experts;
    var k: Kernels = .{ .experts = e };
    errdefer k.deinit();
    k.router = try makeKernel("glm5n_glue_router", &.{ "x", "gate" }, &.{"logits"}, texts.router, texts.header);
    k.top8 = try makeKernel("glm5n_glue_top8", &.{ "logits", "bias", "consts" }, &.{ "ids", "weights" }, texts.top8, texts.header);
    k.combine = try makeKernel("glm5n_moe_combine", &.{ "rows", "w", "shared" }, &.{"y"}, texts.combine, texts.header_combine);
    for (0..max_rows) |i| {
        const m: c_int = @intCast(i + 1);
        const rg = Geometry.router(e);
        k.router_cfg[i] = try makeConfig(&.{.{ .shape = &.{ m, @intCast(e) }, .dtype = .float32 }}, rg.grid, rg.group, &.{
            .{ .name = "T", .value = .{ .dtype = .bfloat16 } }, .{ .name = "M", .value = .{ .int = m } }, .{ .name = "E", .value = .{ .int = @intCast(e) } },
        });
        const tg = Geometry.top8(e, @intCast(m));
        k.top8_cfg[i] = try makeConfig(&.{ .{ .shape = &.{ m, top_k }, .dtype = .int32 }, .{ .shape = &.{ m, top_k }, .dtype = .float32 } }, tg.grid, tg.group, &.{
            .{ .name = "E", .value = .{ .int = @intCast(e) } },
        });
        const cg = Geometry.combine(@intCast(m));
        k.combine_cfg[i] = try makeConfig(&.{.{ .shape = &.{ m, hidden }, .dtype = .bfloat16 }}, cg.grid, cg.group, &.{
            .{ .name = "T", .value = .{ .dtype = .bfloat16 } }, .{ .name = "TOPK", .value = .{ .int = top_k } }, .{ .name = "H", .value = .{ .int = hidden } },
        });
    }
    var cv: [8]f32 = @splat(0);
    cv[0] = c.routed_scaling;
    k.consts = mlx.mlx_array_new_data(&cv, &[_]c_int{8}, 1, .float32);
    if (k.consts.ctx == null) return error.MlxError;
    compileCheck(a, s, c, &k, tables) catch |err| {
        // MLX's own text (a Metal compile error names the line), consumed from its latch.
        var eb: [400]u8 = undefined;
        diag.set("glm5_next decode_glue=fused: the compile check failed ({s}): {s}", .{ @errorName(err), mlx.takeError(&eb) orelse "" });
        return err;
    };
    return k;
}

fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, k: *const Kernels, t: CheckTables) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    const h: c_int = @intCast(c.hidden_size);
    const hc: c_int = @intCast(c.hc_mult);
    // hcPre over a zero stream, then its deferred variant over a zero sublayer output.
    const stream = try g.zeros(&.{ 1, 1, hc, h }, .bfloat16);
    var p0 = try ops.hcPre(s, stream, null, t.fn_rows, t.scale, t.base, t.norm_w, c.hc_iters, c.hc_eps, c.rms_norm_eps, c.rms_norm_eps);
    defer p0.deinit();
    const y = try g.zeros(&.{ 1, 1, h }, .bfloat16);
    const df: ops.Deferred = .{ .y = y, .post = p0.post, .comb = p0.comb, .h = stream };
    var p1 = try ops.hcPre(s, null, &df, t.fn_rows, t.scale, t.base, t.norm_w, c.hc_iters, c.hc_eps, c.rms_norm_eps, c.rms_norm_eps);
    defer p1.deinit();
    // The router and its tail over p1's normed row, the combine over zero rows.
    const x = try g.reshape(p1.normed, &.{ 1, h });
    const r = try route(&g, k, t.router_w, t.router_bias, x, 1);
    const rows = try g.zeros(&.{ 1, @intCast(top_k), h }, .bfloat16);
    const shared = try g.zeros(&.{ 1, h }, .bfloat16);
    const y1 = try combineShared(&g, k, rows, r.weights, shared, 1);
    try g.evalAll(&.{ p1.h, r.ids, r.weights, y1 });
    var ids: [top_k]i32 = undefined;
    const got = mlx.mlx_array_data_int32(r.ids) orelse return error.MlxNoData;
    @memcpy(&ids, got[0..top_k]);
    for (ids, 0..) |id, i| {
        if (id < 0 or id >= @as(i32, @intCast(k.experts))) return error.Glm5NextGlueCheckIds;
        for (ids[0..i]) |prev| if (prev == id) return error.Glm5NextGlueCheckIds;
    }
}

fn apply(g: *G, kernel: Kernel, cfg: Config, inputs: []const T, outs: []T) !void {
    const in = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kernel, in, cfg, g.s));
    for (outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        o.* = try g.adopt(x);
    }
}

pub const Routing = struct { ids: T, weights: T };

/// The routing of `rows` decode-lane rows: x bf16 [rows, hidden] (the FFN sublayer's normed input), the router's BF16
/// weight [E, hidden] and F32 correction bias [E] -> ids i32 [rows, 8] (the scheduler's index type), weights f32 [rows, 8]
/// (normalized, scaled by routed_scaling). Two dispatches.
pub fn route(g: *G, k: *const Kernels, w: T, bias: T, x: T, rows: u32) !Routing {
    var logits: [1]T = undefined;
    try apply(g, k.router, k.router_cfg[rows - 1], &.{ x, w }, &logits);
    var out: [2]T = undefined;
    try apply(g, k.top8, k.top8_cfg[rows - 1], &.{ logits[0], bias, k.consts }, &out);
    return .{ .ids = out[0], .weights = out[1] };
}

/// The decode lane's combine with the shared expert's add: `routed` bf16 [rows, 8, hidden] in router order (the
/// scheduler's join), `weights` f32 [rows, 8], `shared` bf16 [rows, hidden] -> bf16 [rows, hidden]. One dispatch.
pub fn combineShared(g: *G, k: *const Kernels, routed: T, weights: T, shared: T, rows: u32) !T {
    const flat = try g.reshape(routed, &.{ @intCast(rows * top_k), hidden });
    var out: [1]T = undefined;
    try apply(g, k.combine, k.combine_cfg[rows - 1], &.{ flat, weights, shared }, &out);
    return out[0];
}

/// A sublayer's collapse and norm on hcPre (one dispatch), its outputs adopted by `g`: from the stream `x` [1, R, hc, D]
/// (the forward's first sublayer) or from the previous sublayer's `Deferred` (then `h` is the stream it stood for).
pub const Pre = struct { normed: T, post: T, comb: T, h: ?T };

pub fn pre(g: *G, c: *const glm.Config, x: ?T, deferred: ?*const ops.Deferred, fn_rows: T, scale: T, base: T, norm_w: T) !Pre {
    const p = try ops.hcPre(g.s, x, deferred, fn_rows, scale, base, norm_w, c.hc_iters, c.hc_eps, c.rms_norm_eps, c.rms_norm_eps);
    const out: Pre = .{
        .normed = try g.adopt(p.normed),
        .post = try g.adopt(p.post),
        .comb = try g.adopt(p.comb),
        .h = if (deferred != null) try g.adopt(p.h) else blk: {
            _ = mlx.mlx_array_free(p.h);
            break :blk null;
        },
    };
    return out;
}

const testing = std.testing;

test "glm5_next glue: the embedded manifest and texts are the pinned ones" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 3);
}

test "glm5_next glue: GLM-5.3-Flash's geometry is the texts'; another is refused by name" {
    const c: glm.Config = .{};
    try requireGeometry(&c);
    for ([_]glm.Config{
        .{ .hidden_size = 6144 },
        .{ .n_experts_per_tok = 6 },
        .{ .n_routed_experts = 290 },
        .{ .n_routed_experts = 2048 },
        .{ .norm_topk_prob = false },
    }) |bad| try testing.expectError(error.Glm5NextGlueGeometry, requireGeometry(&bad));
}

test "glm5_next glue: the launch geometry per decode-lane width (the router's 72 threadgroups, the top-8's 288 threads a row)" {
    const r = Geometry.router(288);
    try testing.expectEqual([3]c_int{ 256, 72, 1 }, r.grid);
    try testing.expectEqual([3]c_int{ 256, 1, 1 }, r.group);
    for (1..max_rows + 1) |m| {
        const t = Geometry.top8(288, @intCast(m));
        try testing.expectEqual([3]c_int{ 288, @intCast(m), 1 }, t.grid);
        try testing.expectEqual([3]c_int{ 288, 1, 1 }, t.group);
        const cb = Geometry.combine(@intCast(m));
        try testing.expectEqual([3]c_int{ 4096, @intCast(m), 1 }, cb.grid);
        try testing.expect(@rem(cb.grid[0], cb.group[0]) == 0);
    }
}

test "glm5_next glue: the texts name the shapes they are launched at (E experts, the routed_scaling const, i32 ids)" {
    try testing.expect(std.mem.indexOf(u8, texts.router, "logits[size_t(row) * E + 4 * g + tm]") != null);
    try testing.expect(std.mem.indexOf(u8, texts.top8, "threadgroup float vs[E];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.top8, "weights[row * 8 + e] = q * consts[0];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "static_cast<float>(routed) + static_cast<float>(shared[") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header_combine, "#pragma METAL fp contract(off)") != null);
    try testing.expect(std.mem.indexOf(u8, texts.header, "fp contract") == null);
}
