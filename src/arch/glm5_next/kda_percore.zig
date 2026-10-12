//! GLM-5.3-Flash's KDA prompt pass on oMLX's per-core kernel (`kernels/kda_percore.metal`: oMLX 0.7.0, Apache-2.0, its
//! header names the source): the vector-gated delta rule with the bounded-sigmoid gate over one prompt span, one
//! threadgroup per GPU core, or two where the cores are too few for one (`planFor`: the 40-core M5 Max takes Flash's
//! 64 x 128 value rows as 80 threadgroups x 104 rows, two per core). The gate exp(lb * sigmoid(exp(A_log) * (a +
//! dt_bias))) is evaluated while staging, so the f32 [rows, 8,192] gate the per-token route materializes never reaches
//! memory; the f32 state stays in registers. Ported from mlx-serve's `kda_recurrence.zig` (glm5next/kda-prefill-40c
//! b07dc2f5): the same kernel text, plan and probe.
//!
//! The route is chosen once, at construction (`install`, from Module.init): the plan for this GPU's cores, then one launch
//! over zeros at it (Metal caps a kernel's threadgroup by its register use, and MLX reports that only at evaluation).
//! Where both pass, `prompt` is the per-core route and the load logs
//! `[kda] per-core prefill recurrence engaged (<threadgroups> threadgroups x <rows> rows, <n> per core)`; where either
//! declines, `prompt` is the per-token route (`kda.recurStock`, installed by name) and the load logs why. kda.zig's
//! forward runs `prompt` on every KDA span wider than the decode lane (`lane_rows`: only the prompt pass has one) and the
//! per-token route on every other (a serial step, a decode-phase extend, a prompt span of at most `lane_rows` rows), so
//! decode runs the per-token kernel under either. The routes line names the installed route (`kda=percore|stock`).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const log = sdk.log;
const glm = @import("config.zig");
const kda = @import("kda.zig");
const experts = @import("experts.zig");

const G = kda.G;
const T = kda.T;

/// The key and value head width the kernel is written for (config.zig refuses any other at parse).
const head_dim: c_int = 128;
/// Value rows one threadgroup carries: 8 lanes each, at most two heads.
const MAX_ROWS: c_int = 128;
/// Threadgroups a core takes at most. One per core is the one-wave plan (80 cores: Flash's 64 heads x 128 rows as
/// 80 x 104). Two per core carries twice the rows (40 cores: the same 80 x 104, two threadgroups a core, run together or
/// one after the other). A row's arithmetic does not depend on the plan (its eight lanes, its staged q / k / gate / v /
/// beta and its folds are its own), so every plan gives the same bits.
const MAX_PER_CORE: c_int = 2;

/// The decode lane's widest span (experts.zig: a serial row, the MTP verify's rows): a span this wide or narrower runs
/// the per-token route; a wider one (only the prompt pass has one) runs `prompt`.
pub const lane_rows: c_int = experts.decode_lane_rows;

pub const Plan = struct { ntg: c_int, rows: c_int, per_core: c_int };

/// Threadgroups and value rows per threadgroup for `h * dv` rows on `cores` cores, or null: the fewest threadgroups per
/// core (at most `MAX_PER_CORE`) whose even split keeps every threadgroup at `MAX_ROWS` rows or fewer.
pub fn planFor(h: c_int, dv: c_int, cores: c_int) ?Plan {
    const total = h * dv;
    if (cores < 1 or total < 2) return null;
    const per_core = @divTrunc(total + cores * MAX_ROWS - 1, cores * MAX_ROWS);
    if (per_core > MAX_PER_CORE) return null;
    const ntg = @min(cores * per_core, @divTrunc(total, 2));
    const rows = 2 * @divTrunc(@divTrunc(total, 2) + ntg - 1, ntg);
    return if (rows <= MAX_ROWS) .{ .ntg = ntg, .rows = rows, .per_core = per_core } else null;
}

/// The prompt route a construction installs (the routes line's `kda=`).
pub const Route = enum { percore, stock };
/// Why a construction installs the per-token prompt route: a CPU stream, a core count no plan carries (`gpuCores` reads
/// 0 where the IORegistry has none), or a threadgroup past this GPU's cap for the kernel (the probe).
pub const Decline = enum { cpu_stream, cores, thread_cap };
pub const Choice = union(enum) { plan: Plan, declined: Decline };

/// The construction's choice before the probe: on a GPU stream, the plan for `heads` x 128 value rows on `cores` cores.
pub fn choose(gpu: bool, heads: c_int, cores: c_int) Choice {
    if (!gpu) return .{ .declined = .cpu_stream };
    return if (planFor(heads, head_dim, cores)) |p| .{ .plan = p } else .{ .declined = .cores };
}

/// The route of a span of `rows` rows: the installed prompt route past the decode lane, the per-token route within it.
pub fn spanRoute(rows: c_int) *const kda.RecurFn {
    return if (rows > lane_rows) prompt else &kda.recurStock;
}

/// The prompt route, set by `install` at construction, before any forward.
var prompt: *const kda.RecurFn = &kda.recurStock;

/// The engaged route's launch (set by `install` where it engages): the kernel object, this GPU's plan and the lower
/// bound's table.
const Engaged = struct { kernel: mlx.mlx_fast_metal_kernel, plan: Plan, lb: T };
var engaged: Engaged = undefined;
var kernel_obj: ?mlx.mlx_fast_metal_kernel = null;
var lb_table: T = .{ .ctx = null };

const kernel_source = @embedFile("kernels/kda_percore.metal");

/// The prompt route, once, at construction: the plan for this GPU's cores and one probe launch at it decide, `prompt` is
/// set to the chosen route and the load's line names it (declined: and why). Any error but the probe's thread-cap
/// refusal is the construction's.
pub fn install(g: *G, c: *const glm.Config) !Route {
    std.debug.assert(c.kda_head_dim == 128);
    const heads: c_int = @intCast(c.kda_heads);
    const cores = gpuCores();
    const plan = switch (choose(mlx.streamIsGpu(g.s), heads, @intCast(cores))) {
        .plan => |p| p,
        .declined => |why| return installStock(why, heads, cores),
    };
    const kern = try kernel();
    const lb = try lowerBound(c.kda_lower_bound);
    const launches = probe(g, kern, plan, lb, heads) catch |e| {
        _ = mlx.mlx_array_free(lb);
        return e;
    };
    if (!launches) {
        _ = mlx.mlx_array_free(lb);
        return installStock(.thread_cap, heads, cores);
    }
    if (lb_table.ctx != null) _ = mlx.mlx_array_free(lb_table);
    lb_table = lb;
    engaged = .{ .kernel = kern, .plan = plan, .lb = lb };
    prompt = &recurPercore;
    log.info("[kda] per-core prefill recurrence engaged ({d} threadgroups x {d} rows, {d} per core)\n", .{ plan.ntg, plan.rows, plan.per_core });
    return .percore;
}

fn installStock(why: Decline, heads: c_int, cores: u32) Route {
    prompt = &kda.recurStock;
    log.info("[kda] per-core prefill recurrence declined ({t}: {d} heads x 128 value rows, {d} GPU cores); the prompt pass runs the per-token recurrence\n", .{ why, heads, cores });
    return .stock;
}

/// The per-core route (`prompt` once `install` engaged it): one launch at the installed plan over the span.
fn recurPercore(g: *G, c: *const glm.Config, w: *const kda.Weights, q: T, k: T, v: T, a: T, beta: T, state: T) anyerror!kda.Recur {
    _ = c;
    return launch(g, engaged.kernel, engaged.plan, engaged.lb, q, k, v, a, beta, w.a_log, w.dt_bias, state);
}

/// The kernel object, built once (no device work: its library compiles at the probe's launch).
fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_obj) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" };
    const outs = [_][*:0]const u8{ "y", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new("glm5_next_kda_percore", iv, ov, kernel_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_obj = k;
    return k;
}

/// The gate's lower bound as the kernel reads it (`lower_bound[0]`), padded to 8 f32: MLX passes an input of fewer than 8
/// elements as a `constant` pointer and a wider one as `device`; padded, every input but the scalar `T` is `device`.
fn lowerBound(lb: f32) !T {
    const v: [8]f32 = @splat(lb);
    const x = mlx.mlx_array_new_data(&v, &[_]c_int{8}, 1, .float32);
    if (x.ctx == null) return error.MlxError;
    return x;
}

/// One launch of the per-core recurrence at `plan`: q, k, v [B, T, H, 128] (q / k normalized and scaled), `a` the gate's
/// pre-activation [B, T, H * 128], beta [B, T, H], A_log f32 [H], dt_bias f32 [H * 128], the f32 state [B, H, 128, 128]
/// -> y [B, T, H, 128] in q's dtype and the state after the last row. `a` and beta are taken in q's dtype (the trunk's
/// bf16: the casts are MLX's no-ops).
fn launch(g: *G, kern: mlx.mlx_fast_metal_kernel, plan: Plan, lb: T, q: T, k: T, v: T, a: T, beta: T, a_log: T, dt_bias: T, state: T) !kda.Recur {
    const sh = g.shapeOf(q);
    const b = sh.dim(0);
    const t = sh.dim(1);
    const h = sh.dim(2);
    const dt = mlx.mlx_array_dtype(q);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, h, head_dim }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, h, head_dim, head_dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, plan.rows * 8 * plan.ntg, 1, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, plan.rows * 8, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "Dk", head_dim }, .{ "Dv", head_dim }, .{ "H", h }, .{ "NTG", plan.ntg }, .{ "MR", plan.rows } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const t_arr = try g.adopt(mlx.mlx_array_new_int(t));
    const ins = [_]T{ q, k, v, try g.astype(a, dt), try g.astype(beta, dt), a_log, dt_bias, lb, state, t_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    const yt = try g.adopt(y);
    var s_out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&s_out, ov, 1));
    return .{ .y = yt, .state = try g.adopt(s_out) };
}

/// One launch over zeros at `plan` (one row, `heads` heads, every input at the prompt pass's dtype and passing, so the
/// library it compiles is the one the prompt pass runs), evaluated: false where this GPU refuses the threadgroup (Metal
/// caps it by the kernel's register use; MLX reports that only at evaluation).
fn probe(g: *G, kern: mlx.mlx_fast_metal_kernel, plan: Plan, lb: T, heads: c_int) !bool {
    const m = g.mark();
    defer g.resetTo(m);
    const x = try g.zeros(&.{ 1, 1, heads, head_dim }, .bfloat16);
    const a = try g.zeros(&.{ 1, 1, heads * head_dim }, .bfloat16);
    const beta = try g.zeros(&.{ 1, 1, heads }, .bfloat16);
    const a_log = try g.zeros(&.{heads}, .float32);
    const dt_bias = try g.zeros(&.{heads * head_dim}, .float32);
    const state = try g.zeros(&.{ 1, heads, head_dim, head_dim }, .float32);
    const r = try launch(g, kern, plan, lb, x, x, x, a, beta, a_log, dt_bias, state);
    g.evalAll(&.{r.state}) catch |e| {
        if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) return false;
        return e;
    };
    return true;
}

extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IORegistryEntryCreateCFProperties(entry: u32, props: *?*anyopaque, alloc: ?*anyopaque, opts: u32) i32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn CFDictionaryGetValue(dict: ?*const anyopaque, key: ?*const anyopaque) ?*const anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;

/// The GPU's cores: the IORegistry's AGXAccelerator `gpu-core-count` (mlx-serve's `status.gpuCoreCount`), 0 where it
/// cannot be read (no plan: the per-token route).
fn gpuCores() u32 {
    const matching = IOServiceMatching("AGXAccelerator") orelse return 0;
    var iter: u32 = 0;
    if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return 0;
    defer _ = IOObjectRelease(iter);
    const entry = IOIteratorNext(iter);
    if (entry == 0) return 0;
    defer _ = IOObjectRelease(entry);
    var props: ?*anyopaque = null;
    if (IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) return 0;
    defer if (props) |p| CFRelease(p);
    const key = CFStringCreateWithCString(null, "gpu-core-count", 0x08000100) orelse return 0;
    defer CFRelease(key);
    const v = CFDictionaryGetValue(props, key) orelse return 0;
    var n: i64 = 0;
    if (CFNumberGetValue(v, 4, @ptrCast(&n)) == 0) return 0;
    return if (n > 0 and n < 1 << 16) @intCast(n) else 0;
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "glm5_next kda percore CPU plan: one threadgroup per core where the rows fit, two per core where they need twice the cores" {
    // Flash (64 heads x 128) on 80 cores: 80 x 104, one per core. On 40 cores one per core would need 205 rows (past
    // 128): the same 80 x 104, two per core.
    try testing.expectEqual(Plan{ .ntg = 80, .rows = 104, .per_core = 1 }, planFor(64, 128, 80).?);
    try testing.expectEqual(Plan{ .ntg = 80, .rows = 104, .per_core = 2 }, planFor(64, 128, 40).?);
    // the plans that fit one per core
    try testing.expectEqual(Plan{ .ntg = 64, .rows = 128, .per_core = 1 }, planFor(64, 128, 64).?);
    try testing.expectEqual(Plan{ .ntg = 76, .rows = 108, .per_core = 1 }, planFor(64, 128, 76).?);
    try testing.expectEqual(Plan{ .ntg = 16, .rows = 128, .per_core = 1 }, planFor(16, 128, 16).?);
    // 32 and 38 cores: two per core; 30 cores would need three (past MAX_PER_CORE) and decline, as do 0 cores
    try testing.expectEqual(Plan{ .ntg = 64, .rows = 128, .per_core = 2 }, planFor(64, 128, 32).?);
    try testing.expectEqual(Plan{ .ntg = 76, .rows = 108, .per_core = 2 }, planFor(64, 128, 38).?);
    try testing.expect(planFor(64, 128, 30) == null);
    try testing.expect(planFor(64, 128, 0) == null);
    // every plan covers the rows in threadgroups of at most MAX_ROWS, a multiple of the cores when two per core
    for ([_]c_int{ 32, 34, 36, 38, 40, 48, 56, 64, 72, 80, 96, 128 }) |cores| {
        const p = planFor(64, 128, cores).?;
        try testing.expect(p.rows <= MAX_ROWS and @rem(p.rows, 2) == 0 and p.ntg * p.rows >= 64 * 128);
        try testing.expect(p.per_core == 1 or p.ntg == 2 * cores);
    }
}

test "glm5_next kda percore CPU plan: the kernel's row ranges at every plan cover each of Flash's 8,192 value rows once, at most MR rows and two heads a threadgroup" {
    const total: c_int = 64 * 128;
    var cover: [64 * 128]u8 = undefined;
    for ([_]c_int{ 32, 38, 40, 64, 76, 80, 128 }) |cores| {
        const p = planFor(64, 128, cores).?;
        @memset(&cover, 0);
        var tg: c_int = 0;
        while (tg < p.ntg) : (tg += 1) {
            // kda_percore.metal: r0 = 2 * ((tg * (TOTAL / 2)) / NTG), r1 = 2 * (((tg + 1) * (TOTAL / 2)) / NTG)
            const r0 = 2 * @divTrunc(tg * @divTrunc(total, 2), p.ntg);
            const r1 = 2 * @divTrunc((tg + 1) * @divTrunc(total, 2), p.ntg);
            const heads = @divTrunc(r1 - 1, 128) - @divTrunc(r0, 128) + 1;
            try testing.expect(r1 - r0 >= 2 and r1 - r0 <= p.rows and heads <= 2);
            for (@intCast(r0)..@intCast(r1)) |r| cover[r] += 1;
        }
        for (cover) |n| try testing.expectEqual(@as(u8, 1), n);
    }
}

test "glm5_next kda percore CPU route: the plan or the decline at construction; the decode lane's spans take the per-token route under either" {
    try testing.expectEqual(Choice{ .plan = .{ .ntg = 80, .rows = 104, .per_core = 2 } }, choose(true, 64, 40));
    try testing.expectEqual(Choice{ .plan = .{ .ntg = 80, .rows = 104, .per_core = 1 } }, choose(true, 64, 80));
    try testing.expectEqual(Choice{ .declined = .cores }, choose(true, 64, 30));
    try testing.expectEqual(Choice{ .declined = .cores }, choose(true, 64, 0));
    try testing.expectEqual(Choice{ .declined = .cpu_stream }, choose(false, 64, 40));
    // A span past the decode lane runs the installed prompt route, the per-core one where it engaged; every decode-lane
    // width (a serial row, the MTP verify's rows) runs the per-token route.
    const saved = prompt;
    defer prompt = saved;
    try testing.expectEqual(@as(c_int, experts.decode_lane_rows), lane_rows);
    for ([_]*const kda.RecurFn{ &kda.recurStock, &recurPercore }) |installed| {
        prompt = installed;
        var rows: c_int = 1;
        while (rows <= lane_rows) : (rows += 1) try testing.expect(spanRoute(rows) == &kda.recurStock);
        for ([_]c_int{ lane_rows + 1, 29, 1024 }) |r| try testing.expect(spanRoute(r) == installed);
    }
}

// ── the window's parity test ──────────────────────────────────────────────

/// Random kernel inputs at `h` heads of 128 and `t` rows, bf16-exact where the kernel reads 16-bit activations (the
/// host's and the GPU's inputs agree): q / k near unit norm as the normalized inputs, v and the gate's pre-activation
/// N(0, 1), beta in (0, 1), A_log and dt_bias spread as in the GLM checkpoints, a small f32 state (mlx-serve's `KdaCase`).
const Case = struct {
    h: usize,
    t: usize,
    q: []f32,
    k: []f32,
    v: []f32,
    a: []f32,
    beta: []f32,
    alog: []f32,
    dtb: []f32,
    state0: []f32,

    fn bf16(x: f32) f32 {
        const u: u32 = @bitCast(x);
        return @bitCast((u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000);
    }

    fn init(alloc: std.mem.Allocator, h: usize, t: usize, seed: u64) !Case {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const n = t * h * 128;
        var c: Case = .{ .h = h, .t = t, .q = &.{}, .k = &.{}, .v = &.{}, .a = &.{}, .beta = &.{}, .alog = &.{}, .dtb = &.{}, .state0 = &.{} };
        errdefer c.deinit(alloc);
        c.q = try alloc.alloc(f32, n);
        c.k = try alloc.alloc(f32, n);
        c.v = try alloc.alloc(f32, n);
        c.a = try alloc.alloc(f32, n);
        c.beta = try alloc.alloc(f32, t * h);
        c.alog = try alloc.alloc(f32, h);
        c.dtb = try alloc.alloc(f32, h * 128);
        c.state0 = try alloc.alloc(f32, h * 128 * 128);
        for (c.q) |*x| x.* = bf16(rand.floatNorm(f32) / 11.3);
        for (c.k) |*x| x.* = bf16(rand.floatNorm(f32) / 11.3);
        for (c.v) |*x| x.* = bf16(rand.floatNorm(f32));
        for (c.a) |*x| x.* = bf16(rand.floatNorm(f32));
        for (c.beta) |*x| x.* = bf16(rand.float(f32));
        for (c.alog) |*x| x.* = @log(1.0 + 15.0 * rand.float(f32));
        for (c.dtb) |*x| x.* = 0.1 * rand.floatNorm(f32);
        for (c.state0) |*x| x.* = 0.1 * rand.floatNorm(f32);
        return c;
    }

    fn deinit(c: *Case, alloc: std.mem.Allocator) void {
        for ([_][]f32{ c.q, c.k, c.v, c.a, c.beta, c.alog, c.dtb, c.state0 }) |x| alloc.free(x);
    }

    /// The sequential per-token recurrence (the bounded-gate delta rule) in f64: the truth both routes round.
    fn truth(c: *const Case, lb: f32, state: []f64, y: []f64) void {
        const D = 128;
        const total = c.h * D;
        for (state, c.state0) |*d, x| d.* = x;
        for (0..c.t) |tt| for (0..c.h) |hh| {
            const base = (tt * c.h + hh) * D;
            var gate: [D]f64 = undefined;
            for (0..D) |ch| {
                const x = @exp(@as(f64, c.alog[hh])) * (@as(f64, c.a[base + ch]) + c.dtb[hh * D + ch]);
                gate[ch] = @exp(@as(f64, lb) / (1.0 + @exp(-x)));
            }
            const bt: f64 = c.beta[tt * c.h + hh];
            for (0..D) |r| {
                const row = state[(hh * D + r) * D ..][0..D];
                var p: f64 = 0;
                for (0..D) |ch| {
                    row[ch] *= gate[ch];
                    p += row[ch] * c.k[base + ch];
                }
                const delta = (c.v[tt * total + hh * D + r] - p) * bt;
                var o: f64 = 0;
                for (0..D) |ch| {
                    row[ch] += c.k[base + ch] * delta;
                    o += row[ch] * c.q[base + ch];
                }
                y[tt * total + hh * D + r] = o;
            }
        };
    }
};

fn relRms(got: []const f32, ref: []const f64) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, ref) |x, r| {
        const d = @as(f64, x) - r;
        num += d * d;
        den += r * r;
    }
    return @sqrt(num / @max(den, 1e-300));
}

fn maxAbs(got: []const f32, ref: []const f64) f64 {
    var m: f64 = 0;
    for (got, ref) |x, r| m = @max(m, @abs(@as(f64, x) - r));
    return m;
}

/// `data` (f32) as an evaluated array of `dt` on `g`'s stream.
fn deviceArray(g: *G, data: []const f32, shape: []const c_int, dt: mlx.mlx_dtype) !T {
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(data), shape, .float32), dt);
    try g.evalAll(&.{x});
    return x;
}

/// Whether `x` and `y` hold the same values (MLX's array_equal).
fn sameValues(g: *G, x: T, y: T) !bool {
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_array_equal(&e, x, y, false, g.s));
    try mlx.check(mlx.mlx_array_eval(e));
    var eq = false;
    try mlx.check(mlx.mlx_array_item_bool(&eq, e));
    return eq;
}

/// The parity test's static bound (declared before any run): at its widest case (1,024 rows of 64 heads x 128) the host
/// holds the case's f32 inputs (4 x 33.6 MB + the 4.2 MB state) and the f64 truth (67.1 MB + 8.4 MB) and reads back two
/// f32 outputs (2 x 33.6 MB + 2 x 4.2 MB); the device holds the bf16 inputs (4 x 16.8 MB) and their f32 sources until the
/// case's reset (4 x 33.6 MB), the per-token route's five f32 gate stages (5 x 33.6 MB), every route's and partition's
/// output (5 x (16.8 + 4.2) MB) and the read-back casts (2 x 33.6 MB): ~0.93 GB with MLX's JIT (five instantiations).
const parity_bound_bytes: u64 = 1_200_000_000;

test "glm5_next kda percore GPU: the installed per-core prompt route vs the per-token route on random inputs (29, 100 and 1,024 rows); the other partitions bit for bit; both against the f64 delta rule" {
    _ = std.c.getenv("GLM5NEXT_KDA_PERCORE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_KDA_PERCORE_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    if (!mlx.streamIsGpu(stream)) return error.SkipZigTest;
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: glm.Config = .{};
    const heads: c_int = @intCast(c.kda_heads);
    const saved = prompt;
    defer prompt = saved;
    // The construction's install on this GPU: the lever is that it engages.
    const route = try install(&g, &c);
    std.debug.print("GLM5N_KDA_PERCORE_PARITY cores={d} route={t}\n", .{ gpuCores(), route });
    try testing.expectEqual(Route.percore, route);
    const own = engaged.plan;
    // The partitions 64, 76 and 128 cores take one per core (each its own instantiation), where this GPU launches them.
    const others = [_]Plan{ planFor(64, 128, 64).?, planFor(64, 128, 76).?, planFor(64, 128, 128).? };
    var can: [others.len]bool = undefined;
    for (others, &can) |p, *ok| ok.* = try probe(&g, engaged.kernel, p, engaged.lb, heads);
    std.debug.print("GLM5N_KDA_PERCORE_PARITY plan={d}x{d}x{d} launchable 64x128={any} 76x108={any} 128x64={any}\n", .{ own.ntg, own.rows, own.per_core, can[0], can[1], can[2] });
    var worst_y: f64 = 0;
    var worst_s: f64 = 0;
    for ([_]usize{ 29, 100, 1024 }) |t| {
        var cs = try Case.init(a, @intCast(heads), t, 43 + t);
        defer cs.deinit(a);
        const m = g.mark();
        defer g.resetTo(m);
        const tc: c_int = @intCast(t);
        const q = try deviceArray(&g, cs.q, &.{ 1, tc, heads, 128 }, .bfloat16);
        const k = try deviceArray(&g, cs.k, &.{ 1, tc, heads, 128 }, .bfloat16);
        const v = try deviceArray(&g, cs.v, &.{ 1, tc, heads, 128 }, .bfloat16);
        const pre = try deviceArray(&g, cs.a, &.{ 1, tc, heads * 128 }, .bfloat16);
        const beta = try deviceArray(&g, cs.beta, &.{ 1, tc, heads }, .bfloat16);
        const state = try deviceArray(&g, cs.state0, &.{ 1, heads, 128, 128 }, .float32);
        // The layer's tables as graph.zig binds them: A_log as an f32 table [heads], exp(A_log) repeated over
        // each head's key channels (graph.aExp's ops), dt_bias f32.
        var w: kda.Weights = undefined;
        w.a_log = try deviceArray(&g, cs.alog, &.{heads}, .float32);
        w.dt_bias = try deviceArray(&g, cs.dtb, &.{heads * 128}, .float32);
        w.a_exp = try g.exp(try g.reshape(try g.broadcastTo(try g.reshape(w.a_log, &.{ heads, 1 }), &.{ heads, 128 }), &.{heads * 128}));
        try g.evalAll(&.{w.a_exp});
        // The two routes kda.zig's forward takes: the installed prompt route (per-core) and the per-token route.
        const prompt_route = spanRoute(tc);
        const pc = try prompt_route(&g, &c, &w, q, k, v, pre, beta, state);
        const st = try kda.recurStock(&g, &c, &w, q, k, v, pre, beta, state);
        try g.evalAll(&.{ pc.y, pc.state, st.y, st.state });
        var cmp: usize = 0;
        var eq: usize = 0;
        for (others, can) |p, ok| {
            if (!ok) continue;
            const o = try launch(&g, engaged.kernel, p, engaged.lb, q, k, v, pre, beta, w.a_log, w.dt_bias, state);
            try g.evalAll(&.{ o.y, o.state });
            cmp += 2;
            eq += @as(usize, @intFromBool(try sameValues(&g, pc.y, o.y))) + @intFromBool(try sameValues(&g, pc.state, o.state));
        }
        const n_y = t * @as(usize, @intCast(heads)) * 128;
        const n_s = @as(usize, @intCast(heads)) * 128 * 128;
        const y_pc = try a.alloc(f32, n_y);
        defer a.free(y_pc);
        const y_st = try a.alloc(f32, n_y);
        defer a.free(y_st);
        const s_pc = try a.alloc(f32, n_s);
        defer a.free(s_pc);
        const s_st = try a.alloc(f32, n_s);
        defer a.free(s_st);
        _ = try g.hostF32(try g.astype(pc.y, .float32), y_pc);
        _ = try g.hostF32(try g.astype(st.y, .float32), y_st);
        _ = try g.hostF32(pc.state, s_pc);
        _ = try g.hostF32(st.state, s_st);
        const y64 = try a.alloc(f64, n_y);
        defer a.free(y64);
        const s64 = try a.alloc(f64, n_s);
        defer a.free(s64);
        cs.truth(c.kda_lower_bound, s64, y64);
        const e_y = relRms(y_pc, y64);
        const e_y_st = relRms(y_st, y64);
        const e_s = relRms(s_pc, s64);
        const e_s_st = relRms(s_st, s64);
        std.debug.print("GLM5N_KDA_PERCORE_PARITY T={d} partitions_equal={d}/{d} y_relrms percore={e:.3} per_token={e:.3} state_relrms percore={e:.3} per_token={e:.3} y_maxabs percore={e:.3} per_token={e:.3} state_maxabs percore={e:.3} per_token={e:.3}\n", .{ t, eq, cmp, e_y, e_y_st, e_s, e_s_st, maxAbs(y_pc, y64), maxAbs(y_st, y64), maxAbs(s_pc, s64), maxAbs(s_st, s64) });
        // at least one other partition (128 x 64 launches wherever 80 x 104 does), every one the same values
        try testing.expect(cmp >= 2);
        try testing.expectEqual(cmp, eq);
        // decode_kernels_parity's tolerance: the per-core route rounds the f64 truth within twice the per-token route
        try testing.expect(e_y <= 2 * e_y_st + 1e-6);
        try testing.expect(e_s <= 2 * e_s_st + 1e-6);
        worst_y = @max(worst_y, e_y / @max(e_y_st, 1e-12));
        worst_s = @max(worst_s, e_s / @max(e_s_st, 1e-12));
    }
    std.debug.print("GLM5N_KDA_PERCORE_PARITY PASS worst_y_ratio={d:.3} worst_state_ratio={d:.3}\n", .{ worst_y, worst_s });
}
