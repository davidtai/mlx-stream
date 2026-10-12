//! fp8res: the release's FP8 trunk served as shipped (David, 2026-10-09: dequantizing it to BF16 "seems needless").
//!
//! The pack (`residents_format: "fp8_block128"`, reports/.../mimo/fp8res-20261009) keeps every FP8 projection of
//! XiaomiMiMo/MiMo-V2.6-Flash-MOPD (q/k/v of the 48 layers, layer 0's dense MLP, the MTP heads' q/k/v/gate/up/down) as
//! the release's e4m3fn codes, U8 [out, in] `.weight`, with the release's f32 tile scales on a 64-row x 128-column grid,
//! F32 [out / 64, in / 128] `.weight_scale_inv`: value = code x scale, the release's own definition and the BF16 tier's
//! f32 product before its one BF16 rounding (the pack's CPU proof: bf16(code x scale) is the BF16 pack's tensor, bit
//! for bit, for all 165 projections).
//!
//! The route is installed once at construction from the pack's flag (module.zig `Module.init`, mtp.zig `Heads.load`):
//! every FP8 projection's QLinear gets `forwardFp8` bound to its fixed geometry (`Bound`); the BF16 and affine tiers
//! keep the stock route (the control). The installed route branches only on M, a runtime value: M <= 8 rows runs
//! `mimo_fp8_gemv` (decode, the MTP verify rows, the heads' rows); M > 8 rows (the prompt) runs the prompt route:
//! `mimo_fp8_dequant_bf16` writes the projection's BF16 weight (the BF16 pack's tensor, bit for bit) into a
//! four-slot LRU (`PromptCache`, released at the end of every prompt pass, billed geometrically) and MLX's stock NAX
//! GEMM multiplies it: the BF16 tier's prompt arithmetic exactly (fp8res1 bench: the fp8 NAX GEMM ran 0.74-0.93x of
//! the stock GEMM and the threadgroup-tile variant 0.22-0.83x on every site; both were removed). The
//! installed route needs no tensor units: the GEMV and the dequant are plain Metal and the prompt GEMM is MLX's own
//! route for the device (fp8res2: the NAX-only refusal of the fp8 GEMM era is gone with that GEMM).
//!
//! Accumulation contracts (f32 throughout, no weight rounding: exact to the release up to f32 summation and the one
//! BF16 rounding of each output):
//!   GEMV: every output is an f32 FMA chain of x * (code x scale), the weight rounded once to f32 (the converter's
//!     product), in a fixed order: lane-strided k (8 codes per lane per 256-column step), the xor butterfly over the
//!     32 lanes, the K partitions in order. The order depends on the site's plan (`gemvPlan`: KS) and never on M, so
//!     row m of an M-row call is the M = 1 call's row bit for bit (the verify rows reproduce the serial decode).
//!   Prompt (M > 8, installed): the BF16 tier's own arithmetic: bf16(code x scale) weights (one f32 product, one BF16
//!     rounding) under MLX's stock GEMM (f32 accumulation in its NAX tile order). Rounding-class against the GEMV and
//!     against the release (the BF16 weight rounding the decode path avoids); bit-equal to the BF16 tier's prompt GEMM
//!     for the same input rows; a row's value does not depend on the other rows (MLX's GEMM is row-independent).
//! Device self-checks: pre-ship family `fp8res` (`MIMO_FP8RES_GPU=1`); construction runs none.
const std = @import("std");
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const mimo = @import("config.zig");
const prefill_attention = @import("prefill_attention.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const Kernel = mlx.mlx_fast_metal_kernel;
const kset = @import("../../sdk_ext/kernel_set.zig");
const log = std.log.scoped(.mimo);

/// The widest call the GEMV takes: decode, the MTP verify window (M = 4), a head's rows; M > 8 is the prompt GEMM.
pub const gemv_max_rows: c_int = 8;
const scale_rows: u32 = mimo.fp8_scale_rows;
const scale_cols: u32 = mimo.fp8_scale_cols;

// ---- registry: the texts and their pins (kernels/manifest-fp8res.json) ----

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-fp8res.json");
    pub const header = @embedFile("kernels/header_fp8res.metal");
    pub const gemv = @embedFile("kernels/mimo_fp8_gemv.metal");
    /// The installed prompt route's weight writer.
    pub const dequant = @embedFile("kernels/mimo_fp8_dequant_bf16.metal");
};
/// sha256 of kernels/manifest-fp8res.json, which records every text's sha256 (checked against `text_pins` by a CPU test).
pub const manifest_sha256 = "cca35a908631a91cacc6e34554b5fb3abb36180a35ae005b5f7206aaa6541e30";
pub const text_pins = [_]kset.TextPin{
    .{ .file = "header_fp8res.metal", .text = texts.header, .sha256 = "3a6e4b978ebe35816d9e1474f9da300302a5ac46c1103a144fd91148c87e1d6d" },
    .{ .file = "mimo_fp8_gemv.metal", .text = texts.gemv, .sha256 = "69c5f783f59aa9a347375ecd6ab327420ec7f4665b0ef07db29d7e6583dfa33b" },
    .{ .file = "mimo_fp8_dequant_bf16.metal", .text = texts.dequant, .sha256 = "6530ac20bf944f5c4153e62e467a1c89de104e52855f3dbba3a6d007ac06ffdb" },
};

pub fn checkPins() !void {
    return kset.checkTextPins(texts.manifest, manifest_sha256, &text_pins, error.MimoFp8ManifestNotPinned, error.MimoFp8TextNotPinned);
}

// ---- geometry (host arithmetic; CPU-tested at the real shapes) ----

/// Codes per lane per step and the step (32 lanes): a 256-column step, two release blocks; each lane's 8 codes lie in
/// one 128-column block (one scale per lane per step).
pub const gemv_kv: u32 = 8;
pub const gemv_step: u32 = 32 * gemv_kv;
/// The simdgroups that fill the GPU once (the M5 Max: 40 cores x ~24 resident simdgroups): a site with fewer rows
/// splits K (KS) until it has them. Derived from the device shape and the real N.
pub const gemv_target_simdgroups: u32 = 1024;

/// One site's GEMV geometry: R rows per simdgroup (one 64-row scale row: 64 % R == 0), KS K partitions per row group
/// (summed in order through threadgroup memory), RG row groups per threadgroup. Fixed per (N, K) for every M: the
/// order of an output's sum depends on KS (and the text), never on M.
pub const Gemv = struct { r: c_int = 4, ks: c_int = 1, rg: c_int = 4 };

/// The plan for an [n, k] projection, or null when the text does not cover the shape.
pub fn gemvPlan(n: u32, k: u32) ?Gemv {
    const r: u32 = 4;
    if (n % scale_rows != 0 or k % scale_cols != 0 or k % gemv_step != 0 or n % r != 0) return null;
    var ks: u32 = 1;
    while (ks < 8 and (n / r) * ks < gemv_target_simdgroups and (k / gemv_step) % (ks * 2) == 0) ks *= 2;
    var rg: u32 = @max(1, 4 / ks);
    while (rg > 1 and n % (r * rg) != 0) rg /= 2;
    return .{ .r = @intCast(r), .ks = @intCast(ks), .rg = @intCast(rg) };
}

pub const Launch = struct { grid: [3]c_int, tg: [3]c_int };

/// X18 (mimo/dequant-audit; fp8res2's "M = 4 k / v fix"): which GEMV plan a call of M rows runs, fixed at install
/// (`settings.fp8_gemv_plan`). `"fixed"` (the control) runs the site's plan at every M. `"by_rows"` (the default) changes
/// only R, the rows per simdgroup, where the `fp8res_bench` gemv_batched rows (device time, 32 calls per eval) of four
/// windows put another R ahead of the site plan at the served widths (preship-next5 .. next8, medians, us per call):
///   M = 4 (the verify, head 0): R = 2 at every site, ahead in 4 of 4 windows at each: q 71.1 -> 62.5, gate / up 107.3 ->
///     88.2, down 121.1 -> 103.9, k_sliding 19.3 -> 16.9, v_sliding 18.1 -> 13.2, k_full 12.8 -> 12.3, v_full 12.2 ->
///     11.7 (the k / v sites from 0.82-0.96x of the BF16 GEMV to 0.93-1.13x);
///   M = 5 and 6 (heads 1 and 2 after a round of accepted drafts): R = 1 on the single-partition K = 4,096 sites (q, gate /
///     up): q 109.4 -> 93.7 and 137.1 -> 105.4, gate / up 159.4 -> 122.3 and 183.1 -> 142.4 (4 of 4; q at M = 5 3 of 4,
///     next5's slower floor the exception);
///   everything else keeps the site plan (the k / v sites past M = 4 and q at M = 1 .. 3 have no consistent winner; gate /
///     up and down lean R = 2 at M = 1 .. 3 in 3-4 of 4 windows by 3-17 us per call: off the verify width, not taken).
/// Exact by construction: an output's summation order is fixed by KS and the text (mimo_fp8_gemv.metal: per lane over
/// steps then codes, the xor butterfly, the K partitions in order), never by R or RG, and every entry keeps the site's KS
/// and RG; R only picks which output rows one simdgroup owns. So every width's rows stay the M = 1 call's rows bit for
/// bit (the device self-check `fp8res` runs `forward` at M = 1 .. 8 against the KS-only host emulation).
pub const GemvRoute = enum { fixed, by_rows, invalid };

pub fn gemvPlanAt(route: GemvRoute, site: Gemv, n: u32, k: u32, m: u32) Gemv {
    if (route != .by_rows) return site;
    const r: c_int = switch (m) {
        4 => 2,
        5, 6 => if (site.ks == 1 and k == 4096) 1 else site.r,
        else => site.r,
    };
    var p = site;
    p.r = r;
    // The text's static asserts hold for every entry at the real shapes (CPU test); a shape where they would not keeps the
    // site plan.
    if (n % @as(u32, @intCast(p.r * p.rg)) != 0 or @rem(64, p.r) != 0) return site;
    return p;
}

/// Threadgroups of RG x KS simdgroups (32 x (RG * KS) threads); grid y = one threadgroup per R * RG rows.
pub fn gemvLaunch(p: Gemv, n: c_int) Launch {
    const groups = @divExact(n, p.r * p.rg);
    return .{ .grid = .{ 32, groups * p.rg * p.ks, 1 }, .tg = .{ 32, p.rg * p.ks, 1 } };
}


// ---- kernels and the bound route ----

/// The kernel objects, one set per module; every FP8 projection's route borrows them (heap-owned: stable address).
pub const Kernels = struct {
    gemv: Kernel = .{ .ctx = null },
    dequant: Kernel = .{ .ctx = null },
    /// The prompt route's BF16 weights (owned by `install`; null in host fixtures, which never run a prompt row).
    prompt: ?*PromptCache = null,
    /// R2 (rows audit): the decode phase after the growth bills no prompt cache (`settings.decodeClosesPromptRoute`):
    /// the module closes the route there (`closePrompt`) and opens it at every prompt pass (`openPrompt`). No caller makes
    /// a prompt-route call while closed (`promptRun`'s debug assert; the tests hold it).
    prompt_closed: bool = false,
    /// X18 (mimo/dequant-audit): the GEMV's plan per call width, fixed at install (`settings.fp8_gemv_plan`); `bind`
    /// reads it once per projection (`Bound.plans`).
    gemv_route: GemvRoute = .by_rows,

    pub fn init() !Kernels {
        var k: Kernels = .{};
        errdefer k.deinit();
        k.gemv = try makeKernel("mimo_fp8_gemv", texts.gemv, texts.header);
        k.dequant = try makeKernelIo("mimo_fp8_dequant_bf16", &.{ "w", "scales" }, &.{"wb"}, texts.dequant, texts.header);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.gemv, k.dequant }) |x| if (x.ctx != null) {
            _ = mtl.mlx_fast_metal_kernel_free(x);
        };
        if (k.prompt) |c| c.release();
        k.* = .{};
    }
};

/// The prompt route's BF16 weights: the last `cap` FP8 projections a prompt row touched, dequantized once each
/// (`mimo_fp8_dequant_bf16`) and reused by every tile / span of the layer. A prompt layer touches one set of three at
/// a time: its attention's q / k / v (`Attention.forwardNax` / `forwardTiled`: k / v per span, q
/// per tile), then, in layer 0 only, the dense MLP's gate / up / down (`Dense.forwardTiled`, per 512-row
/// tile); the MTP heads' catch-up touches each of its six once. `cap` is fixed at install (`settings.fp8_prompt_slots`):
/// 3 (X18's default) holds one set, so every projection is dequantized once per layer per chunk (`pick`, the CPU test
/// over the real call orders); 4 is the control (fp8res2's cache: the fourth slot only ever held the previous set's
/// least recent projection, layer 0's q at most: 100,663,296 B). `release` runs at the end of every prompt pass
/// (Module.prefill / extend); the bill carries `promptCacheBytes` (the `cap` widest FP8 projections in BF16).
pub const PromptCache = struct {
    pub const max_slots = 4;
    cap: u8 = default_prompt_slots,
    keys: [max_slots]?*anyopaque = @splat(null),
    arrs: [max_slots]Array = @splat(.{}),
    stamps: [max_slots]u64 = @splat(0),
    clock: u64 = 0,

    pub fn release(c: *PromptCache) void {
        for (&c.arrs) |*x| {
            drop(x.*);
            x.* = .{};
        }
        c.keys = @splat(null);
        c.stamps = @splat(0);
    }

    /// The BF16 weight of codes `w` (borrowed: valid until the next `get` evicts it or `release`; the caller's graph
    /// holds its own reference once it uses the array).
    fn get(c: *PromptCache, k: *const Kernels, w: Array, scales: Array, n: c_int, kdim: c_int, s: Stream) !Array {
        c.clock += 1;
        const p = pick(c.keys[0..c.cap], c.stamps[0..c.cap], w.ctx);
        if (p.hit) {
            c.stamps[p.slot] = c.clock;
            return c.arrs[p.slot];
        }
        const j = p.slot;
        drop(c.arrs[j]);
        c.arrs[j] = .{};
        c.keys[j] = null;
        c.arrs[j] = try dequantRun(k.dequant, w, scales, n, kdim, s);
        c.keys[j] = w.ctx;
        c.stamps[j] = c.clock;
        return c.arrs[j];
    }

    /// The LRU's choice for `key` among the first `cap` slots: the slot holding it (a hit), else the least recently used
    /// slot (an empty slot's stamp is 0). The one policy `get` runs; the CPU test replays the real call orders through it.
    pub fn pick(keys: []const ?*anyopaque, stamps: []const u64, key: ?*anyopaque) struct { hit: bool, slot: usize } {
        for (keys, 0..) |k, i| if (k != null and k == key) return .{ .hit = true, .slot = i };
        var j: usize = 0;
        for (1..keys.len) |i| if (stamps[i] < stamps[j]) {
            j = i;
        };
        return .{ .hit = false, .slot = j };
    }
};

/// X18: the prompt cache's default capacity (`settings.fp8_prompt_slots`): one layer's set of three (see `PromptCache`).
pub const default_prompt_slots: u8 = 3;

/// The prompt route's bill: the `slots` widest FP8 projections of the config (trunk and MTP heads share the geometry) in
/// BF16, all live at once (geometric: no ordering credit). X18: `slots` is the installed capacity (3 by default: layer
/// 0's gate / up / down, 402,653,184 B; the control's 4 adds q, 503,316,480 B).
pub fn promptCacheBytes(c: mimo.Config, slots: u8) u64 {
    if (c.residents_format != .fp8_block128) return 0;
    const h: u64 = c.hidden_size;
    var sizes = [_]u64{
        @as(u64, c.n_heads) * c.head_dim * h, @as(u64, c.swa_n_kv_heads) * c.head_dim * h, @as(u64, c.swa_n_kv_heads) * c.value_head_dim * h,
        @as(u64, c.intermediate_size) * h,     @as(u64, c.intermediate_size) * h,              @as(u64, c.intermediate_size) * h,
    };
    std.mem.sort(u64, &sizes, {}, std.sort.desc(u64));
    var total: u64 = 0;
    for (sizes[0..slots]) |x| total += 2 * x;
    return total;
}

/// The BF16 weight [N, K] of an FP8 projection (one word of codes per thread).
fn dequantRun(kernel: Kernel, w: Array, scales: Array, n: c_int, kdim: c_int, s: Stream) !Array {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ n, kdim }, 2, .bfloat16));
    const words = @divExact(n * kdim, 4);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, words, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "N", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", kdim));
    const ins = [_]Array{ w, scales };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&outs, kernel, in_vec, c, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

/// The prompt route: x [M, K] @ bf16(code x scale)^T through MLX's stock matmul (the BF16 tier's QLinear arithmetic).
pub fn promptRun(kernels: *const Kernels, n: c_int, kdim: c_int, x: Array, w: Array, scales: Array, s: Stream) !Array {
    // R2: the decode phase after the growth has no billed prompt cache; its calls are M <= 8 (the GEMV), so no prompt-route
    // call reaches a closed route. An invariant of the callers, held by the tests.
    std.debug.assert(!kernels.prompt_closed);
    const wb = try kernels.prompt.?.get(kernels, w, scales, n, kdim, s);
    var wt: Array = .{};
    defer drop(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wb, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
    return out;
}

/// End of a prompt pass (Module.prefill / extend): the prompt route's weights go.
pub fn releasePrompt(k: ?*Kernels) void {
    if (k) |p| if (p.prompt) |c| c.release();
}

/// R2: the growth to the decode rows (Module, `settings.decodeClosesPromptRoute`): the cache released, the route closed.
pub fn closePrompt(k: ?*Kernels) void {
    releasePrompt(k);
    if (k) |p| p.prompt_closed = true;
}

/// R2: a prompt pass's start (Module.prefill / extend): the route open.
pub fn openPrompt(k: ?*Kernels) void {
    if (k) |p| p.prompt_closed = false;
}

fn makeKernel(name: [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !Kernel {
    return makeKernelIo(name, &.{ "x", "w", "scales" }, &.{"y"}, source, header);
}

fn makeKernelIo(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // x is read as [M, K] rows, w / scales are the residents (row-contiguous by construction).
    const k = mtl.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MimoFp8Kernel;
    return k;
}

/// X18: the install's construction-time choices (`settings.fp8_prompt_slots`, `settings.fp8_gemv_plan`).
pub const InstallOptions = struct { prompt_slots: u8 = default_prompt_slots, gemv: GemvRoute = .by_rows };

/// Construction (Module.init): the pack's flag picks the route once. `.standard` installs nothing (every QLinear keeps
/// the stock route); `.fp8_block128` installs the GEMV + prompt route on any device MLX serves.
pub fn install(a: std.mem.Allocator, c: mimo.Config, opts: InstallOptions) !?*Kernels {
    if (c.residents_format != .fp8_block128) return null;
    if (opts.prompt_slots < 3 or opts.prompt_slots > PromptCache.max_slots or opts.gemv == .invalid) return error.MimoFp8InstallOptions;
    const k = try a.create(Kernels);
    errdefer a.destroy(k);
    k.* = try Kernels.init();
    errdefer k.deinit();
    k.gemv_route = opts.gemv;
    const cache = try a.create(PromptCache);
    cache.* = .{ .cap = opts.prompt_slots };
    k.prompt = cache;
    log.info("MiMo residents_format=fp8_block128: FP8 resident route installed (GEMV M<=8 KV={d} target_simdgroups={d} fp8_gemv_plan={s}; prompt M>8 = bf16(code x scale) LRU fp8_prompt_slots={d} + stock GEMM, cache bound {d} B)", .{ gemv_kv, gemv_target_simdgroups, @tagName(opts.gemv), opts.prompt_slots, promptCacheBytes(c, opts.prompt_slots) });
    return k;
}

pub fn destroy(a: std.mem.Allocator, k: ?*Kernels) void {
    if (k) |p| {
        const cache = p.prompt;
        p.deinit();
        if (cache) |c| a.destroy(c);
        a.destroy(p);
    }
}

/// One FP8 projection's installed route: the kernels and its fixed geometry (N output rows, K input columns, the site's
/// GEMV plan and the plan each call width M = 1 .. 8 runs). Bound once at construction; nothing on the hot path
/// re-derives or re-validates it.
pub const Bound = struct {
    kernels: ?*const Kernels = null,
    n: c_int = 0,
    k: c_int = 0,
    /// The site's plan (`gemvPlan`): its KS fixes every output's summation order at every M.
    plan: Gemv = .{},
    /// X18: the plan a call of M rows runs (index M; 0 unused): `gemvPlanAt` under the installed `GemvRoute`. Every entry
    /// keeps `plan.ks` (and its RG), so every entry is the site plan's arithmetic bit for bit.
    plans: [gemv_max_rows + 1]Gemv = @splat(.{}),
};

/// Construction: an [n, k] FP8 projection's route, refused when the texts do not cover the shape.
pub fn bind(kernels: *const Kernels, n: u32, k: u32) !Bound {
    const plan = gemvPlan(n, k) orelse return error.MimoFp8Geometry;
    var b: Bound = .{ .kernels = kernels, .n = @intCast(n), .k = @intCast(k), .plan = plan };
    for (&b.plans, 0..) |*p, m| p.* = gemvPlanAt(kernels.gemv_route, plan, n, k, @intCast(m));
    return b;
}

/// x [M, K] BF16 (every QLinear call site passes rows x features) -> [M, N] BF16. M is the only runtime routing value.
pub fn forward(b: *const Bound, x: Array, w: Array, scales: Array, s: Stream) !Array {
    const m = mlx.getShape(x)[0];
    if (m <= gemv_max_rows) return gemvRun(b.kernels.?, b.plans[@intCast(m)], b.n, b.k, x, w, scales, m, s);
    return promptRun(b.kernels.?, b.n, b.k, x, w, scales, s);
}

pub fn gemvRun(kernels: *const Kernels, p: Gemv, n: c_int, kdim: c_int, x: Array, w: Array, scales: Array, m: c_int, s: Stream) !Array {
    const l = gemvLaunch(p, n);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ m, n }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    for ([_]struct { [*:0]const u8, c_int }{ .{ "M", m }, .{ "N", n }, .{ "K", kdim }, .{ "R", p.r }, .{ "KS", p.ks }, .{ "RG", p.rg } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    return apply(kernels.gemv, c, x, w, scales, s);
}

fn apply(kernel: Kernel, c: mlx.mlx_fast_metal_kernel_config, x: Array, w: Array, scales: Array, s: Stream) !Array {
    const ins = [_]Array{ x, w, scales };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&outs, kernel, in_vec, c, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

// ---- host references (the device self-checks' ground truth; CPU-tested below) ----

/// The OCP FP8 E4M3 (e4m3fn) value of a code: bias 7, subnormals at exponent 0; null for the NaN codes 0x7F / 0xFF.
pub fn e4m3(code: u8) ?f32 {
    const e: i32 = (code >> 3) & 0xF;
    const m: f32 = @floatFromInt(code & 7);
    if (e == 0xF and (code & 7) == 7) return null;
    const mag: f32 = if (e == 0) m / 8.0 * std.math.pow(f32, 2, -6) else (1 + m / 8.0) * std.math.pow(f32, 2, @floatFromInt(e - 7));
    return if (code & 0x80 != 0) -mag else mag;
}

/// The kernels' decode, emulated: (code & 127) << 7 with the sign at bit 15, read as a half and widened (e4m3 / 256).
pub fn decodeHalf(code: u8) f32 {
    const bits: u16 = (@as(u16, code & 0x7F) << 7) | (@as(u16, code & 0x80) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// The GEMV's weight (and the converter's): f32 code x scale, one rounding, as the kernel forms it.
pub fn weight32(code: u8, scale: f32) f32 {
    return decodeHalf(code) * (scale * 256);
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// Round to nearest even (no NaN reaches here).
pub fn f32ToBf16(v: f32) u16 {
    const r: u32 = @bitCast(v);
    return @truncate((r +% 0x7fff +% ((r >> 16) & 1)) >> 16);
}

/// f64 -> BF16 with one rounding: round to odd into f32 (24 >= 8 + 2 bits: no double-rounding error), then RNE.
pub fn f64ToBf16(v: f64) u16 {
    var t: f32 = @floatCast(v);
    const back: f64 = t;
    if (back != v) {
        var bits: u32 = @bitCast(t);
        if (@abs(back) > @abs(v)) bits -= 1;
        bits |= 1;
        t = @bitCast(bits);
    }
    return f32ToBf16(t);
}

fn ordered(b: u16) i32 {
    return if (b & 0x8000 != 0) -@as(i32, b & 0x7FFF) else @as(i32, b);
}

/// BF16 ulps between two values (as ordered integers).
pub fn ulps(a: u16, b: u16) u32 {
    return @intCast(@abs(ordered(a) - ordered(b)));
}

/// The weight row [K] as the GEMV forms it (code x scale, one f32 rounding).
pub fn weightRow(codes: []const u8, srow: []const f32, out: []f32) void {
    for (codes, out, 0..) |c, *w, i| w.* = weight32(c, srow[i / scale_cols]);
}

/// The GEMV's arithmetic for one output (row xr [K] BF16 bits, weight row codes [K], the row's scales [K / 128]),
/// emulated in its exact order (`mimo_fp8_gemv.metal`): per lane, fma over steps then codes; the xor butterfly; the K
/// partitions in order. Returns the f32 sum (the kernel rounds it to BF16 once).
pub fn hostGemvDot(plan: Gemv, xr: []const u16, codes: []const u8, srow: []const f32) f32 {
    var w: [16384]f32 = undefined;
    weightRow(codes, srow, w[0..codes.len]);
    return hostGemvDotW(plan, xr, w[0..codes.len]);
}

/// `hostGemvDot` over a precomputed weight row (`weightRow`).
pub fn hostGemvDotW(plan: Gemv, xr: []const u16, w: []const f32) f32 {
    const k = xr.len;
    const ks: usize = @intCast(plan.ks);
    const kp = k / ks;
    var total: f32 = 0;
    for (0..ks) |p| {
        var lanes: [32]f32 = undefined;
        for (0..32) |l| {
            var acc: f32 = 0;
            const k0 = p * kp + gemv_kv * l;
            var st: usize = 0;
            while (st < kp / gemv_step) : (st += 1) {
                const kk = k0 + st * gemv_step;
                for (0..gemv_kv) |i| acc = @mulAdd(f32, bf16ToF32(xr[kk + i]), w[kk + i], acc);
            }
            lanes[l] = acc;
        }
        for ([_]usize{ 16, 8, 4, 2, 1 }) |off| {
            var next: [32]f32 = undefined;
            for (0..32) |l| next[l] = lanes[l] + lanes[l ^ off];
            lanes = next;
        }
        total = if (p == 0) lanes[0] else total + lanes[0];
    }
    return total;
}

/// The exact reference: sum of x * (code x scale) in f64 (every product exact: 8 x 24 bits), and sum |x * w| (the f32
/// summation error bound's scale).
pub const Ref = struct { sum: f64, abs: f64 };
pub fn refDot(xr: []const u16, codes: []const u8, srow: []const f32) Ref {
    var sum: f64 = 0;
    var abs: f64 = 0;
    for (xr, codes, 0..) |xb, c, i| {
        const p = @as(f64, bf16ToF32(xb)) * @as(f64, weight32(c, srow[i / scale_cols]));
        sum += p;
        abs += @abs(p);
    }
    return .{ .sum = sum, .abs = abs };
}

/// `refDot` over f64 rows (x widened, the weight row widened): 4-wide f64 lanes; every product exact.
pub fn refDotF64(x: []const f64, w: []const f64) Ref {
    const V = @Vector(4, f64);
    var sum: V = @splat(0);
    var abs: V = @splat(0);
    var i: usize = 0;
    while (i + 4 <= x.len) : (i += 4) {
        const p = @as(V, x[i..][0..4].*) * @as(V, w[i..][0..4].*);
        sum += p;
        abs += @abs(p);
    }
    var s1 = @reduce(.Add, sum);
    var a1 = @reduce(.Add, abs);
    while (i < x.len) : (i += 1) {
        s1 += x[i] * w[i];
        a1 += @abs(x[i] * w[i]);
    }
    return .{ .sum = s1, .abs = a1 };
}

/// The GEMM's arithmetic modelled on the host: per 128-column block an f32 sum of the exact products x * code / 256
/// (sequential here; the tensor unit's internal order is its own), promoted by fma(block, scale x 256, acc).
pub fn hostBlockDot(xr: []const u16, codes: []const u8, srow: []const f32) f32 {
    var acc: f32 = 0;
    var kb: usize = 0;
    while (kb < xr.len / scale_cols) : (kb += 1) {
        var block: f32 = 0;
        for (kb * scale_cols..(kb + 1) * scale_cols) |i| block += bf16ToF32(xr[i]) * decodeHalf(codes[i]);
        acc = @mulAdd(f32, block, srow[kb] * 256, acc);
    }
    return acc;
}

/// Pass rule of one output against the exact reference: within 1 BF16 ulp of bf16(reference), or within the f32
/// summation bound gamma * sum |x * w| (+ the output's own half ulp) where cancellation leaves the sum small.
pub fn withinTolerance(out: u16, ref: f64, abs: f64, k: usize) bool {
    if (ulps(out, f64ToBf16(ref)) <= 1) return true;
    const gamma = @as(f64, @floatFromInt(2 * k)) * std.math.pow(f64, 2, -24);
    const half_ulp = @abs(@as(f64, bf16ToF32(out))) * std.math.pow(f64, 2, -8);
    return @abs(@as(f64, bf16ToF32(out)) - ref) <= gamma * abs + half_ulp;
}

// ---- fixtures (seeded; real shapes) ----

/// The real FP8 projection shapes [N, K]: q, k / v of the full and sliding layers (the MTP heads are sliding-shaped),
/// layer 0's and the heads' gate / up and down.
pub const Site = struct { label: []const u8, n: u32, k: u32 };
pub const sites = [_]Site{
    .{ .label = "q", .n = 12288, .k = 4096 },
    .{ .label = "k_full", .n = 768, .k = 4096 },
    .{ .label = "v_full", .n = 512, .k = 4096 },
    .{ .label = "k_sliding", .n = 1536, .k = 4096 },
    .{ .label = "v_sliding", .n = 1024, .k = 4096 },
    .{ .label = "gate_up", .n = 16384, .k = 4096 },
    .{ .label = "down", .n = 4096, .k = 16384 },
};

/// Codes uniform over the 254 non-NaN codes (a NaN byte is folded onto its neighbour).
pub fn fillCodes(r: std.Random, codes: []u8) void {
    r.bytes(codes);
    for (codes) |*c| if ((c.* & 0x7F) == 0x7F) {
        c.* -= 1;
    };
}
/// One scale per 64 x 128 block in [2^-17, 2^-6) with random mantissas: the release's range (its 165 projections'
/// scales span 9.4e-6 .. 0.0158, fp8res-20261009/proof-fp8res.json).
pub fn fillScales(r: std.Random, scales: []f32) void {
    for (scales) |*s| s.* = std.math.pow(f32, 2, -17 + @as(f32, @floatFromInt(r.uintLessThan(u32, 11)))) * (1 + r.float(f32));
}
/// x BF16 ~ N(0, 1.5^2).
pub fn fillX(r: std.Random, x: []u16) void {
    for (x) |*b| b.* = f32ToBf16(1.5 * r.floatNorm(f32));
}

/// Fixture data: codes uniform over the 254 non-NaN codes (every exponent, subnormals included), one scale per 64 x 128
/// block in the release's range (`fillScales`), x BF16 ~ N(0, 1.5^2).
pub const Fixture = struct {
    codes: []u8,
    scales: []f32,
    x: []u16,
    pub fn init(a: std.mem.Allocator, n: u32, k: u32, rows: u32, seed: u64) !Fixture {
        var rng: std.Random.DefaultPrng = .init(seed);
        const r = rng.random();
        const codes = try a.alloc(u8, @as(usize, n) * k);
        errdefer a.free(codes);
        fillCodes(r, codes);
        const scales = try a.alloc(f32, @as(usize, n / scale_rows) * (k / scale_cols));
        errdefer a.free(scales);
        fillScales(r, scales);
        const x = try a.alloc(u16, @as(usize, rows) * k);
        fillX(r, x);
        return .{ .codes = codes, .scales = scales, .x = x };
    }
    pub fn deinit(f: Fixture, a: std.mem.Allocator) void {
        a.free(f.codes);
        a.free(f.scales);
        a.free(f.x);
    }
    /// Row n's scales [K / 128].
    pub fn srow(f: Fixture, n: usize, k: usize) []const f32 {
        const sc = k / scale_cols;
        return f.scales[(n / scale_rows) * sc ..][0..sc];
    }
};

// ---- CPU tests (hermetic; no kernel is applied) ----

const testing = std.testing;

test "mimo fp8res CPU: the manifest and every text match their pins" {
    try checkPins();
    try kset.expectManifestPins(texts.manifest, &text_pins, 2);
}

test "mimo fp8res CPU: e4m3 decode: the kernels' half trick is the OCP value / 256 for all 254 codes; code x scale == decode x (scale x 256)" {
    var subnormal: usize = 0;
    for (0..256) |i| {
        const c: u8 = @intCast(i);
        const v = e4m3(c) orelse {
            try testing.expect((c & 0x7F) == 0x7F);
            continue;
        };
        try testing.expectEqual(v / 256, decodeHalf(c));
        subnormal += @intFromBool((c & 0x78) == 0 and (c & 7) != 0);
        // The converter's f32 product (torch: f32(code) * scale) and the kernels' (decode * (scale * 256)) agree bit
        // for bit over scales spanning the release's range.
        for ([_]f32{ 1.0, 0.0013427734375, 3.0517578125e-05, 0.000244140625 * 1.3, 0.0078125 * 1.97, 1.1920929e-07 * 3 }) |s|
            try testing.expectEqual(v * s, weight32(c, s));
    }
    try testing.expectEqual(@as(usize, 14), subnormal);
    // 448 is the largest code, 2^-9 the smallest subnormal.
    try testing.expectEqual(@as(f32, 448), e4m3(0x7E).?);
    try testing.expectEqual(std.math.pow(f32, 2, -9), e4m3(0x01).?);
    // BF16 helpers: RNE ties to even; round-to-odd keeps f64 -> BF16 single-rounded.
    try testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    try testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(@bitCast(@as(u32, 0x3F808000))));
    try testing.expectEqual(@as(u16, 0x3F82), f32ToBf16(@bitCast(@as(u32, 0x3F818000))));
    try testing.expectEqual(@as(u16, 0x3F81), f64ToBf16(1.0 + std.math.pow(f64, 2, -8) + std.math.pow(f64, 2, -40)));
    try testing.expectEqual(@as(u16, 0x3F80), f64ToBf16(1.0 + std.math.pow(f64, 2, -8)));
    try testing.expectEqual(@as(u32, 2), ulps(0x0001, 0x8001));
}

test "mimo fp8res CPU: GEMV plans at the real shapes: every site covered, K partitions fixed per site (M-invariant), one scale row per simdgroup" {
    const want = [_]Gemv{
        .{ .r = 4, .ks = 1, .rg = 4 }, // q 12,288: 3,072 simdgroups
        .{ .r = 4, .ks = 8, .rg = 1 }, // k_full 768: 192 x 8
        .{ .r = 4, .ks = 8, .rg = 1 }, // v_full 512: 128 x 8
        .{ .r = 4, .ks = 4, .rg = 1 }, // k_sliding 1,536: 384 x 4
        .{ .r = 4, .ks = 4, .rg = 1 }, // v_sliding 1,024: 256 x 4
        .{ .r = 4, .ks = 1, .rg = 4 }, // gate / up 16,384
        .{ .r = 4, .ks = 1, .rg = 4 }, // down 4,096 x 16,384: 1,024
    };
    for (sites, want) |site, w| {
        const p = gemvPlan(site.n, site.k).?;
        try testing.expectEqual(w, p);
        // The text's static asserts, on the host: whole steps per partition, whole row groups, one scale row per
        // simdgroup, threadgroups of at most 32 simdgroups.
        try testing.expect(site.k % (@as(u32, @intCast(p.ks)) * gemv_step) == 0);
        try testing.expect(site.n % @as(u32, @intCast(p.r * p.rg)) == 0 and @rem(64, p.r) == 0 and p.rg * p.ks <= 32);
        try testing.expect(@as(u32, @intCast(@divExact(@as(c_int, @intCast(site.n)), p.r) * p.ks)) >= @min(gemv_target_simdgroups, site.n / 4 * 8));
        var m: c_int = 1;
        while (m <= gemv_max_rows) : (m += 1) {
            const l = gemvLaunch(p, @intCast(site.n));
            try testing.expectEqual([3]c_int{ 32, p.rg * p.ks, 1 }, l.tg);
            try testing.expectEqual(@as(c_int, @intCast(site.n)), @divExact(l.grid[1], p.ks) * p.r);
        }
    }
    // Shapes the text does not cover are refused by the plan (and so by `bind`).
    try testing.expect(gemvPlan(12288, 4000) == null);
    try testing.expect(gemvPlan(12290, 4096) == null);
    try testing.expect(gemvPlan(96, 4096) == null);
}

test "mimo fp8res CPU: host GEMV (kernel order) and host block GEMM vs the exact f64 reference on real-shaped fixtures; M-invariant rows" {
    const a = testing.allocator;
    // Every site at its real [N, K], 8 rows; a sampled column set (the emulation is per output, its cost K each).
    var worst: u32 = 0;
    var total: usize = 0;
    var within1: usize = 0;
    for (sites, 0..) |site, si| {
        // Eight sampled weight rows of the site (their codes and their 64-row scale rows), eight activation rows.
        var prng: std.Random.DefaultPrng = .init(0xf8e5_0001 + si);
        const r = prng.random();
        const x = try a.alloc(u16, 8 * site.k);
        defer a.free(x);
        fillX(r, x);
        const codes = try a.alloc(u8, 8 * site.k);
        defer a.free(codes);
        fillCodes(r, codes);
        const scales = try a.alloc(f32, 8 * (site.k / scale_cols));
        defer a.free(scales);
        fillScales(r, scales);
        const plan = gemvPlan(site.n, site.k).?;
        for (0..8) |m| for (0..8) |j| {
            const xr = x[m * site.k ..][0..site.k];
            const wr = codes[j * site.k ..][0..site.k];
            const sr = scales[j * (site.k / scale_cols) ..][0 .. site.k / scale_cols];
            const ref = refDot(xr, wr, sr);
            const gemv_out = f32ToBf16(hostGemvDot(plan, xr, wr, sr));
            const gemm_out = f32ToBf16(hostBlockDot(xr, wr, sr));
            try testing.expect(withinTolerance(gemv_out, ref.sum, ref.abs, site.k));
            try testing.expect(withinTolerance(gemm_out, ref.sum, ref.abs, site.k));
            const u = ulps(gemv_out, f64ToBf16(ref.sum));
            worst = @max(worst, u);
            within1 += @intFromBool(u <= 1);
            total += 1;
        };
        // M-invariance: a row's emulated output is a function of that row alone (the order has no M in it): row 3
        // computed beside 7 other rows equals row 3 computed alone, by construction of the emulation and the text.
    }
    std.debug.print("MIMO_FP8RES_CPU host_gemv outputs={d} within_1_ulp={d} max_ulp={d}\n", .{ total, within1, worst });
    try testing.expect(within1 * 100 >= total * 99);
}

test "mimo fp8res CPU: the prompt route's bill is the installed capacity's widest FP8 projections in BF16 (X18 default 3: layer 0's MLP; the control's 4 adds q)" {
    var c: mimo.Config = .{};
    try testing.expectEqual(@as(u64, 0), promptCacheBytes(c, 3));
    try testing.expectEqual(@as(u64, 0), promptCacheBytes(c, 4));
    c.residents_format = .fp8_block128;
    try testing.expectEqual(@as(u64, 2 * 3 * 16384 * 4096), promptCacheBytes(c, default_prompt_slots));
    try testing.expectEqual(@as(u64, 402_653_184), promptCacheBytes(c, 3));
    try testing.expectEqual(@as(u64, 2 * (3 * 16384 * 4096 + 12288 * 4096)), promptCacheBytes(c, 4));
    try testing.expectEqual(@as(u64, 100_663_296), promptCacheBytes(c, 4) - promptCacheBytes(c, 3));
}

/// X18: one prompt chunk's FP8 calls in the trunk's order (module.zig), the prompt-route calls only (M > 8; a call of at
/// most eight rows runs the GEMV and never touches the cache): per layer the attention route's q / k / v (NAX tiled:
/// `forwardNax`, k / v per span, q per 512-row tile; stock: `forwardTiled` -> `forwardOps`, q / k / v per 512-row chunk, or once at <= 512 rows; at <= 8 rows the
/// NAX installs take `forward`, the GEMV), then layer 0's dense MLP (`Dense.forward` at <= 512 rows, else
/// `Dense.forwardTiled` per 512 rows: gate / up / down). Keys name (layer, projection); returns the distinct ones.
const LruRoute = enum { nax_tiled, stock };
const LruProj = enum(u8) { q, k, v, gate, up, down };
fn lruKey(owner: usize, p: LruProj) ?*anyopaque {
    return @ptrFromInt((owner * 8 + @intFromEnum(p) + 1) * 64);
}
fn lruCall(list: *std.ArrayList(?*anyopaque), a: std.mem.Allocator, seen: *std.AutoHashMap(usize, void), owner: usize, p: LruProj, m: u32) !void {
    if (m <= gemv_max_rows) return;
    try list.append(a, lruKey(owner, p));
    try seen.put(@intFromPtr(lruKey(owner, p)), {});
}
fn lruChunk(list: *std.ArrayList(?*anyopaque), a: std.mem.Allocator, c: mimo.Config, n: u32, route: LruRoute) !usize {
    var seen = std.AutoHashMap(usize, void).init(a);
    defer seen.deinit();
    const chunk = mimo.prefill_chunk;
    for (0..c.n_layers) |layer| {
        const mode = c.layer_types[layer];
        const nax = route != .stock;
        if (nax and n > 8) {
            const span = prefill_attention.attentionSpanRows(c.kvHeads(layer), mode, n);
            var s0: u32 = 0;
            while (s0 < n) : (s0 += span) {
                const sn = @min(span, n - s0);
                try lruCall(list, a, &seen, layer, .k, sn);
                try lruCall(list, a, &seen, layer, .v, sn);
                var t0 = s0;
                while (t0 < s0 + sn) : (t0 += prefill_attention.query_tile_rows) try lruCall(list, a, &seen, layer, .q, @min(prefill_attention.query_tile_rows, s0 + sn - t0));
            }
        } else if (!nax) {
            var t0: u32 = 0;
            const tile: u32 = if (n <= chunk) n else chunk;
            while (t0 < n) : (t0 += tile) {
                const tn = @min(tile, n - t0);
                for ([_]LruProj{ .q, .k, .v }) |p| try lruCall(list, a, &seen, layer, p, tn);
            }
        }
        if (layer == 0) {
            var t0: u32 = 0;
            const tile: u32 = if (n <= chunk) n else chunk;
            while (t0 < n) : (t0 += tile) {
                const tn = @min(tile, n - t0);
                for ([_]LruProj{ .gate, .up, .down }) |p| try lruCall(list, a, &seen, layer, p, tn);
            }
        }
    }
    return seen.count();
}
/// The cache's dequants over a call order at capacity `cap`: `PromptCache.pick` on every call, as `get` runs it.
fn lruMisses(cap: u8, calls: []const ?*anyopaque) usize {
    var keys: [PromptCache.max_slots]?*anyopaque = @splat(null);
    var stamps: [PromptCache.max_slots]u64 = @splat(0);
    var clock: u64 = 0;
    var misses: usize = 0;
    for (calls) |key| {
        clock += 1;
        const p = PromptCache.pick(keys[0..cap], stamps[0..cap], key);
        if (!p.hit) {
            misses += 1;
            keys[p.slot] = key;
        }
        stamps[p.slot] = clock;
    }
    return misses;
}

test "mimo fp8res CPU: X18 prompt cache capacity: over the trunk's real call orders (NAX tiled, stock; 9 .. 32,768-row prompts in 16,384-row chunks) and the MTP heads' catch-up, 3 slots dequantize every FP8 projection once per layer per chunk, exactly as 4 do" {
    const a = testing.allocator;
    const c: mimo.Config = .{};
    var checked: usize = 0;
    for ([_]LruRoute{ .nax_tiled, .stock }) |route| {
        for ([_]u32{ 9, 100, 512, 513, 516, 523, 1024, 4133, 16_384, 16_384 + 9, 16_384 + 600, 32_768 }) |rows| {
            var calls: std.ArrayList(?*anyopaque) = .empty;
            defer calls.deinit(a);
            // One prompt pass: `Module.forwardTokens`' chunks; the cache lives for the whole pass (released at its end).
            var expected: usize = 0;
            var start: u32 = 0;
            while (start < rows) {
                const n = @min(rows - start, mimo.prefill_max_tokens);
                expected += try lruChunk(&calls, a, c, n, route);
                start += n;
            }
            const m3 = lruMisses(3, calls.items);
            const m4 = lruMisses(4, calls.items);
            if (m3 != expected or m4 != expected) {
                std.debug.print("MIMO_FP8RES_LRU route={s} rows={d} calls={d} expected={d} slots3={d} slots4={d}\n", .{ @tagName(route), rows, calls.items.len, expected, m3, m4 });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        }
    }
    // The MTP heads' first-round catch-up (mtp.zig `layerForwardEmb`): each head's q / k / v then gate / up / down, once each.
    var heads: std.ArrayList(?*anyopaque) = .empty;
    defer heads.deinit(a);
    for (0..3) |h| for ([_]LruProj{ .q, .k, .v, .gate, .up, .down }) |p| try heads.append(a, lruKey(100 + h, p));
    try testing.expectEqual(@as(usize, 18), lruMisses(3, heads.items));
    try testing.expectEqual(@as(usize, 18), lruMisses(4, heads.items));
    std.debug.print("MIMO_FP8RES_LRU orders={d} slots3==slots4==once_per_layer_per_chunk catch_up=18/18\n", .{checked});
}

/// The GEMV text's thread -> (rows, partition, lanes) map and its per-output accumulation, transliterated from
/// `mimo_fp8_gemv.metal` (every thread of every threadgroup, in the text's loop order), for a host fixture. Returns the
/// f32 sums before the BF16 rounding, [M, N], and the times each output was written (must be 1).
fn hostGemvText(a: std.mem.Allocator, p: Gemv, m: usize, n: usize, k: usize, x: []const u16, w32: []const f32, out: []f32, writes: []u8) !void {
    const r: usize = @intCast(p.r);
    const ks: usize = @intCast(p.ks);
    const rg: usize = @intCast(p.rg);
    const kp = k / ks;
    const groups = n / (r * rg);
    @memset(writes, 0);
    // part[rg][ks][m][r] per threadgroup (the KS > 1 hand-off).
    const part = try a.alloc(f32, rg * ks * m * r);
    defer a.free(part);
    const acc = try a.alloc(f32, 32 * m * r);
    defer a.free(acc);
    for (0..groups) |tgy| {
        for (0..rg * ks) |sgi| {
            const ksi = sgi % ks;
            const rgi = sgi / ks;
            const n0 = (tgy * rg + rgi) * r;
            // Every lane's accumulators, then the xor butterfly (every lane ends with the same sum).
            for (0..32) |lane| {
                const k0 = ksi * kp + gemv_kv * lane;
                for (0..m) |mi| for (0..r) |ri| {
                    var v: f32 = 0;
                    var st: usize = 0;
                    while (st < kp / gemv_step) : (st += 1) {
                        const kk = k0 + st * gemv_step;
                        for (0..gemv_kv) |i| v = @mulAdd(f32, bf16ToF32(x[mi * k + kk + i]), w32[(n0 + ri) * k + kk + i], v);
                    }
                    acc[(lane * m + mi) * r + ri] = v;
                };
            }
            for (0..m) |mi| for (0..r) |ri| {
                var lanes: [32]f32 = undefined;
                for (0..32) |l| lanes[l] = acc[(l * m + mi) * r + ri];
                for ([_]usize{ 16, 8, 4, 2, 1 }) |off| {
                    var next: [32]f32 = undefined;
                    for (0..32) |l| next[l] = lanes[l] + lanes[l ^ off];
                    lanes = next;
                }
                if (ks == 1) {
                    out[mi * n + n0 + ri] = lanes[0];
                    writes[mi * n + n0 + ri] += 1;
                } else part[((rgi * ks + ksi) * m + mi) * r + ri] = lanes[0];
            };
        }
        if (ks > 1) for (0..rg) |rgi| {
            const n0 = (tgy * rg + rgi) * r;
            for (0..m) |mi| for (0..r) |ri| {
                var t = part[((rgi * ks) * m + mi) * r + ri];
                for (1..ks) |pi| t += part[((rgi * ks + pi) * m + mi) * r + ri];
                out[mi * n + n0 + ri] = t;
                writes[mi * n + n0 + ri] += 1;
            };
        };
    }
}

test "mimo fp8res CPU: X18 GEMV plan per width: the by_rows table keeps every site's KS and RG, meets the text's static asserts at the real shapes, and the text's thread map gives every output once with the site plan's sum bit for bit at every R" {
    // The table at the real sites (`fixed` = the site plan everywhere).
    for (sites) |site| {
        const base = gemvPlan(site.n, site.k).?;
        var m: u32 = 1;
        while (m <= gemv_max_rows) : (m += 1) {
            try testing.expectEqual(base, gemvPlanAt(.fixed, base, site.n, site.k, m));
            const p = gemvPlanAt(.by_rows, base, site.n, site.k, m);
            try testing.expectEqual(base.ks, p.ks);
            try testing.expectEqual(base.rg, p.rg);
            try testing.expect(site.n % @as(u32, @intCast(p.r * p.rg)) == 0 and @rem(64, p.r) == 0 and site.k % (@as(u32, @intCast(p.ks)) * gemv_step) == 0 and p.r * @as(c_int, @intCast(m)) <= 32);
            const want_r: c_int = if (m == 4) 2 else if ((m == 5 or m == 6) and base.ks == 1 and site.k == 4096) 1 else base.r;
            try testing.expectEqual(want_r, p.r);
            const l = gemvLaunch(p, @intCast(site.n));
            try testing.expectEqual(@as(c_int, @intCast(site.n)), @divExact(l.grid[1], p.ks) * p.r);
        }
    }
    // The text's map, every thread, at small shapes of each site class (KS 1 / 4 / 8 with their RG) and every R the table
    // takes: each output written once, equal to the KS-only host emulation (`hostGemvDotW`) bit for bit.
    const a = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xf8e5_18a0);
    const rnd = prng.random();
    const shapes = [_]struct { n: usize, k: usize, ks: c_int, rg: c_int }{
        .{ .n = 128, .k = 1024, .ks = 1, .rg = 4 },
        .{ .n = 64, .k = 4096, .ks = 4, .rg = 1 },
        .{ .n = 64, .k = 4096, .ks = 8, .rg = 1 },
    };
    var compared: usize = 0;
    for (shapes) |sh| {
        const m = 6;
        const x = try a.alloc(u16, m * sh.k);
        defer a.free(x);
        fillX(rnd, x);
        const codes = try a.alloc(u8, sh.n * sh.k);
        defer a.free(codes);
        fillCodes(rnd, codes);
        const scales = try a.alloc(f32, (sh.n / scale_rows) * (sh.k / scale_cols));
        defer a.free(scales);
        fillScales(rnd, scales);
        const w32 = try a.alloc(f32, sh.n * sh.k);
        defer a.free(w32);
        for (0..sh.n) |ni| weightRow(codes[ni * sh.k ..][0..sh.k], scales[(ni / scale_rows) * (sh.k / scale_cols) ..][0 .. sh.k / scale_cols], w32[ni * sh.k ..][0..sh.k]);
        const out = try a.alloc(f32, m * sh.n);
        defer a.free(out);
        const writes = try a.alloc(u8, m * sh.n);
        defer a.free(writes);
        for ([_]c_int{ 4, 2, 1 }) |r| {
            const p: Gemv = .{ .r = r, .ks = sh.ks, .rg = sh.rg };
            for ([_]usize{ 1, 4, 6 }) |mm| {
                try hostGemvText(a, p, mm, sh.n, sh.k, x, w32, out[0 .. mm * sh.n], writes[0 .. mm * sh.n]);
                for (0..mm) |mi| for (0..sh.n) |ni| {
                    try testing.expectEqual(@as(u8, 1), writes[mi * sh.n + ni]);
                    const want = hostGemvDotW(p, x[mi * sh.k ..][0..sh.k], w32[ni * sh.k ..][0..sh.k]);
                    try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(out[mi * sh.n + ni])));
                    compared += 1;
                };
            }
        }
    }
    std.debug.print("MIMO_FP8RES_PLAN by_rows table at 7 sites x M 1..8 ok; text map outputs={d} written_once=yes bit_equal_to_KS_emulation=yes\n", .{compared});
}

test "mimo fp8res route: the flag installs the route on any device; the standard format installs none; bind records the site's plan and every width's plan once and refuses shapes the text does not cover" {
    const a = testing.allocator;
    var c: mimo.Config = .{};
    try testing.expectEqual(@as(?*Kernels, null), try install(a, c, .{}));
    // The flag installs the route on any device (no tensor-unit requirement since the stock-GEMM prompt route): the
    // kernel objects (nothing compiles before a call) and the prompt cache, owned until `destroy`.
    c.residents_format = .fp8_block128;
    const installed = (try install(a, c, .{})).?;
    defer destroy(a, installed);
    try testing.expect(installed.gemv.ctx != null and installed.dequant.ctx != null and installed.prompt != null);
    // X18 defaults: three prompt slots, the by_rows plans; the control (4, fixed) installs as asked; anything else refuses.
    try testing.expectEqual(@as(u8, 3), installed.prompt.?.cap);
    try testing.expectEqual(GemvRoute.by_rows, installed.gemv_route);
    const control = (try install(a, c, .{ .prompt_slots = 4, .gemv = .fixed })).?;
    defer destroy(a, control);
    try testing.expectEqual(@as(u8, 4), control.prompt.?.cap);
    try testing.expectEqual(GemvRoute.fixed, control.gemv_route);
    try testing.expectError(error.MimoFp8InstallOptions, install(a, c, .{ .prompt_slots = 2 }));
    try testing.expectError(error.MimoFp8InstallOptions, install(a, c, .{ .prompt_slots = 5 }));
    try testing.expectError(error.MimoFp8InstallOptions, install(a, c, .{ .gemv = .invalid }));
    // `bind` records the site's plan and the per-width plans once; refused shapes never reach a kernel.
    var k: Kernels = .{};
    const b = try bind(&k, 12288, 4096);
    try testing.expectEqual(Gemv{ .r = 4, .ks = 1, .rg = 4 }, b.plan);
    try testing.expect(b.kernels == &k and b.n == 12288 and b.k == 4096);
    try testing.expectEqual(Gemv{ .r = 4, .ks = 1, .rg = 4 }, b.plans[1]);
    try testing.expectEqual(Gemv{ .r = 2, .ks = 1, .rg = 4 }, b.plans[4]);
    try testing.expectEqual(Gemv{ .r = 1, .ks = 1, .rg = 4 }, b.plans[5]);
    try testing.expectEqual(Gemv{ .r = 1, .ks = 1, .rg = 4 }, b.plans[6]);
    try testing.expectEqual(Gemv{ .r = 4, .ks = 1, .rg = 4 }, b.plans[8]);
    const kv = try bind(&k, 1536, 4096);
    try testing.expectEqual(Gemv{ .r = 2, .ks = 4, .rg = 1 }, kv.plans[4]);
    try testing.expectEqual(Gemv{ .r = 4, .ks = 4, .rg = 1 }, kv.plans[5]);
    k.gemv_route = .fixed;
    const f = try bind(&k, 12288, 4096);
    for (f.plans[1..]) |p| try testing.expectEqual(f.plan, p);
    try testing.expectError(error.MimoFp8Geometry, bind(&k, 12288, 4000));
    // The rows must fill whole 64-row scale blocks (the GEMV and the prompt route's dequant read them whole).
    try testing.expectError(error.MimoFp8Geometry, bind(&k, 96, 4096));
}

// ---- the real packs (opt-in, CPU, no MLX): MIMO_FP8RES_PACK=<fp8res pack> MIMO_BF16_PACK=<bf16res pack> ----

const io_util = @import("sdk").io_util;

/// One tensor of a safetensors file, read past the page cache (header-located; the caller frees).
pub fn readTensor(a: std.mem.Allocator, dir: []const u8, file: []const u8, name: []const u8, dtype: []const u8, shape: []const u64) ![]u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, file }, 0);
    defer a.free(path);
    const fd = try io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    var prefix: [8]u8 = undefined;
    try io_util.readAligned(fd, &prefix, 0);
    const hlen = std.mem.readInt(u64, &prefix, .little);
    if (hlen > 16 << 20) return error.MimoFp8PackHeader;
    const header = try a.alloc(u8, @intCast(hlen));
    defer a.free(header);
    try io_util.readAligned(fd, header, 8);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    const t = parsed.value.object.get(name) orelse return error.MimoFp8PackTensor;
    if (!std.mem.eql(u8, t.object.get("dtype").?.string, dtype)) return error.MimoFp8PackTensor;
    const dims = t.object.get("shape").?.array.items;
    if (dims.len != shape.len) return error.MimoFp8PackTensor;
    for (dims, shape) |d, want| if (d != .integer or d.integer != @as(i64, @intCast(want))) return error.MimoFp8PackTensor;
    const off = t.object.get("data_offsets").?.array.items;
    const buf = try a.alloc(u8, @intCast(off[1].integer - off[0].integer));
    errdefer a.free(buf);
    try io_util.readAligned(fd, buf, 8 + hlen + @as(u64, @intCast(off[0].integer)));
    return buf;
}

test "mimo fp8res pack: the plugin reads the fp8res pack's codes and 64 x 128 scale grids as the converter wrote them: bf16(code x scale) is the BF16 pack's tensor (MIMO_FP8RES_PACK, MIMO_BF16_PACK)" {
    const fp8_dir = std.mem.span(std.c.getenv("MIMO_FP8RES_PACK") orelse return error.SkipZigTest);
    const bf16_dir = std.mem.span(std.c.getenv("MIMO_BF16_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    // The pack's own config and sidecar manifest, parsed as construction parses them.
    var d = try std.Io.Dir.openDirAbsolute(io, fp8_dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", a, .limited(16 << 20));
    defer a.free(text);
    const c = try mimo.Config.parse(a, text, null);
    try testing.expectEqual(mimo.ResidentsFormat.fp8_block128, c.residents_format);
    const mtp_text = try d.readFileAlloc(io, "mtp-manifest.json", a, .limited(1 << 20));
    defer a.free(mtp_text);
    const mtp = @import("mtp.zig");
    const m = try mtp.Manifest.parse(a, mtp_text, c);
    for (m.storage, mtp.fp8_storages) |got, want| try testing.expect(got.eql(want));
    try testing.expectEqual(c.residentPayloadBytes(), try @import("prepared.zig").residentBytes(a, io, fp8_dir));
    // Tensors of every kind of geometry: a full layer's k and v (the 64-row grid), a sliding layer's q, layer 0's
    // down (K = 16,384), an MTP head's k and gate.
    const cases = [_]struct { file: []const u8, bf16_file: []const u8, name: []const u8, n: u64, k: u64 }{
        .{ .file = "model.safetensors", .bf16_file = "model.safetensors", .name = "model.layers.0.self_attn.k_proj", .n = 768, .k = 4096 },
        .{ .file = "model.safetensors", .bf16_file = "model.safetensors", .name = "model.layers.5.self_attn.v_proj", .n = 512, .k = 4096 },
        .{ .file = "model.safetensors", .bf16_file = "model.safetensors", .name = "model.layers.1.self_attn.q_proj", .n = 12288, .k = 4096 },
        .{ .file = "model.safetensors", .bf16_file = "model.safetensors", .name = "model.layers.0.mlp.down_proj", .n = 4096, .k = 16384 },
        .{ .file = "mtp-heads.safetensors", .bf16_file = "mtp-heads.safetensors", .name = "model.mtp.layers.1.self_attn.k_proj", .n = 1536, .k = 4096 },
        .{ .file = "mtp-heads.safetensors", .bf16_file = "mtp-heads.safetensors", .name = "model.mtp.layers.2.mlp.gate_proj", .n = 16384, .k = 4096 },
    };
    var name_buf: [160]u8 = undefined;
    for (cases) |cs| {
        const codes = try readTensor(a, fp8_dir, cs.file, try std.fmt.bufPrint(&name_buf, "{s}.weight", .{cs.name}), "U8", &.{ cs.n, cs.k });
        defer a.free(codes);
        const sraw = try readTensor(a, fp8_dir, cs.file, try std.fmt.bufPrint(&name_buf, "{s}.weight_scale_inv", .{cs.name}), "F32", &.{ cs.n / 64, cs.k / 128 });
        defer a.free(sraw);
        const braw = try readTensor(a, bf16_dir, cs.bf16_file, try std.fmt.bufPrint(&name_buf, "{s}.weight", .{cs.name}), "BF16", &.{ cs.n, cs.k });
        defer a.free(braw);
        const scales = std.mem.bytesAsSlice(f32, @as([]align(1) u8, sraw));
        const bf = std.mem.bytesAsSlice(u16, @as([]align(1) u8, braw));
        var mismatch: usize = 0;
        for (0..cs.n) |n| for (0..cs.k) |k| {
            const s = scales[(n / 64) * (cs.k / 128) + k / 128];
            if (f32ToBf16(weight32(codes[n * cs.k + k], s)) != bf[n * cs.k + k]) mismatch += 1;
        };
        std.debug.print("MIMO_FP8RES_PACK {s} [{d}, {d}] grid [{d}, {d}] bf16_mismatch={d}\n", .{ cs.name, cs.n, cs.k, cs.n / 64, cs.k / 128, mismatch });
        try testing.expectEqual(@as(usize, 0), mismatch);
    }
}

test "mimo fp8res pack X18: the per-width GEMV plans on the release's FP8 codes and scales (MIMO_FP8RES_PACK): the text map at the served widths gives every output once, bit-equal across R 4 / 2 / 1 and to the KS-only emulation" {
    const fp8_dir = std.mem.span(std.c.getenv("MIMO_FP8RES_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    // Real tensors of each site class at their real K and KS (the first 256 output rows: four 64-row scale rows; the text's
    // index map depends on N only through the threadgroup count): q and k of a sliding layer, layer 0's down (K = 16,384),
    // an MTP head's gate.
    const cases = [_]struct { file: []const u8, name: []const u8, n: u64, k: u64 }{
        .{ .file = "model.safetensors", .name = "model.layers.1.self_attn.q_proj", .n = 12288, .k = 4096 },
        .{ .file = "model.safetensors", .name = "model.layers.1.self_attn.k_proj", .n = 1536, .k = 4096 },
        .{ .file = "model.safetensors", .name = "model.layers.0.mlp.down_proj", .n = 4096, .k = 16384 },
        .{ .file = "mtp-heads.safetensors", .name = "model.mtp.layers.2.mlp.gate_proj", .n = 16384, .k = 4096 },
    };
    var prng: std.Random.DefaultPrng = .init(0xf8e5_18b0);
    const rnd = prng.random();
    var name_buf: [160]u8 = undefined;
    var compared: usize = 0;
    for (cases) |cs| {
        const codes = try readTensor(a, fp8_dir, cs.file, try std.fmt.bufPrint(&name_buf, "{s}.weight", .{cs.name}), "U8", &.{ cs.n, cs.k });
        defer a.free(codes);
        const sraw = try readTensor(a, fp8_dir, cs.file, try std.fmt.bufPrint(&name_buf, "{s}.weight_scale_inv", .{cs.name}), "F32", &.{ cs.n / 64, cs.k / 128 });
        defer a.free(sraw);
        const scales = std.mem.bytesAsSlice(f32, @as([]align(1) u8, sraw));
        const site = gemvPlan(@intCast(cs.n), @intCast(cs.k)).?;
        const n: usize = 256;
        const k: usize = @intCast(cs.k);
        const w32 = try a.alloc(f32, n * k);
        defer a.free(w32);
        const srow = try a.alloc(f32, k / scale_cols);
        defer a.free(srow);
        for (0..n) |ni| {
            for (srow, 0..) |*sv, j| sv.* = scales[(ni / scale_rows) * (k / scale_cols) + j];
            weightRow(codes[ni * k ..][0..k], srow, w32[ni * k ..][0..k]);
        }
        const x = try a.alloc(u16, 6 * k);
        defer a.free(x);
        fillX(rnd, x);
        const out = try a.alloc(f32, 6 * n);
        defer a.free(out);
        const base_out = try a.alloc(f32, 6 * n);
        defer a.free(base_out);
        const writes = try a.alloc(u8, 6 * n);
        defer a.free(writes);
        for ([_]u32{ 4, 5, 6 }) |m| {
            const mm: usize = m;
            const p = gemvPlanAt(.by_rows, site, @intCast(cs.n), @intCast(cs.k), m);
            try hostGemvText(a, site, mm, n, k, x, w32, base_out[0 .. mm * n], writes[0 .. mm * n]);
            try hostGemvText(a, p, mm, n, k, x, w32, out[0 .. mm * n], writes[0 .. mm * n]);
            for (0..mm * n) |i| {
                try testing.expectEqual(@as(u8, 1), writes[i]);
                try testing.expectEqual(@as(u32, @bitCast(base_out[i])), @as(u32, @bitCast(out[i])));
            }
            for (0..mm) |mi| for (0..n) |ni| {
                const want = hostGemvDotW(site, x[mi * k ..][0..k], w32[ni * k ..][0..k]);
                try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(out[mi * n + ni])));
            };
            compared += mm * n;
            std.debug.print("MIMO_FP8RES_PACK_X18 {s} K={d} KS={d} M={d} site_R={d} x18_R={d} rows={d} bit_equal=yes\n", .{ cs.name, k, site.ks, m, site.r, p.r, n });
        }
    }
    std.debug.print("MIMO_FP8RES_PACK_X18 outputs={d} bit_equal=yes\n", .{compared});
}

// ---- device self-checks (pre-ship family `fp8res`: MIMO_FP8RES_GPU=1, inside a guarded window) ----

/// The MLX memory limit the device checks set; their static bounds are printed and asserted against it and against the
/// measured MLX peak.
pub const selfcheck_ceiling_bytes: u64 = 3 << 30;
/// MLX cache + compiler + host allowances the families carry (preship families.tsv: 256 MiB + 512 MiB + 128 MiB).
const allowance_bytes: u64 = (256 + 512 + 128) << 20;

fn gpuGate(flag: [*:0]const u8) !void {
    _ = std.c.getenv(flag) orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
}

/// The GEMV check's peak at its widest site (gate / up, 16,384 x 4,096): device codes + scales, the BF16 path's weight
/// (device and its host staging), x and the outputs; host fixture codes and the host emulation / reference rows.
pub fn gemvSelfcheckBound() u64 {
    var worst: u64 = 0;
    for (sites) |site| {
        const nk: u64 = @as(u64, site.n) * site.k;
        const device = nk + nk / (64 * 128) * 4 + 2 * nk + 8 * site.k * 2 + 2 * 36 * @as(u64, site.n) * 2;
        const host = nk + 2 * nk + 8 * site.k * 2 + 8 * site.k * 8 + 8 * @as(u64, site.n) * (2 + 8 + 8);
        worst = @max(worst, device + host);
    }
    return worst + allowance_bytes;
}

pub const GemmCase = struct { site: usize, rows: []const c_int };
/// The prompt route check's GEMM cases: q / o-width calls over the whole prompt and the full layers' K/V spans to 16,384 rows, the
/// sliding spans and dense-MLP chunks at 512 rows, the tails (9, 523, 4,133).
pub const gemm_cases = [_]GemmCase{
    .{ .site = 0, .rows = &.{ 16_384, 4133, 512 } },
    .{ .site = 1, .rows = &.{ 16_384, 523, 9 } },
    .{ .site = 2, .rows = &.{16_384} },
    .{ .site = 3, .rows = &.{512} },
    .{ .site = 4, .rows = &.{512} },
    .{ .site = 5, .rows = &.{ 2048, 512 } },
    .{ .site = 6, .rows = &.{ 2048, 512 } },
};

/// The prompt route check's peak (`gemmSelfcheckBound`): per site its widest call's x and both outputs live with every smaller
/// call's output, the codes, scales and the BF16 path's weight (device + host staging), host x and the f64 rows.
pub fn gemmSelfcheckBound() u64 {
    var worst: u64 = 0;
    for (gemm_cases) |gc| {
        const site = sites[gc.site];
        const nk: u64 = @as(u64, site.n) * site.k;
        const big: u64 = @intCast(gc.rows[0]);
        var outs: u64 = 0;
        for (gc.rows) |r| outs += @as(u64, @intCast(r)) * site.n * 2;
        const device = big * site.k * 2 + outs + big * site.n * 2 + nk + nk / (64 * 128) * 4 + 2 * nk;
        const host = big * site.k * 2 + nk + 2 * nk + (big / 64 + 64) * site.k * 8;
        worst = @max(worst, device + host);
    }
    return worst + allowance_bytes;
}

/// Device arrays of one site's fixture: codes, scales, the BF16 path's weight bf16(code x scale) (= the BF16 pack's
/// tensor by the pack's proof), and x.
const DeviceSite = struct {
    w: Array,
    sc: Array,
    wb: Array,
    x: Array,
    fn init(a: std.mem.Allocator, f: Fixture, n: u32, k: u32, rows: u32) !DeviceSite {
        const wb_host = try a.alloc(u16, @as(usize, n) * k);
        defer a.free(wb_host);
        for (0..n) |ni| {
            const sr = f.srow(ni, k);
            for (0..k) |ki| wb_host[ni * k + ki] = f32ToBf16(weight32(f.codes[ni * k + ki], sr[ki / scale_cols]));
        }
        return .{
            .w = mlx.mlx_array_new_data(f.codes.ptr, &[_]c_int{ @intCast(n), @intCast(k) }, 2, .uint8),
            .sc = mlx.mlx_array_new_data(f.scales.ptr, &[_]c_int{ @intCast(n / scale_rows), @intCast(k / scale_cols) }, 2, .float32),
            .wb = mlx.mlx_array_new_data(wb_host.ptr, &[_]c_int{ @intCast(n), @intCast(k) }, 2, .bfloat16),
            .x = mlx.mlx_array_new_data(f.x.ptr, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, .bfloat16),
        };
    }
    fn deinit(d: DeviceSite) void {
        drop(d.w);
        drop(d.sc);
        drop(d.wb);
        drop(d.x);
    }
};

fn rowsOf(x: Array, m: c_int, s: Stream) !Array {
    const k = mlx.getShape(x)[1];
    var out: Array = .{};
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ m, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

/// The BF16 tier's stock route (module.zig QLinear `forwardDense`): x @ wb^T.
fn bf16Path(x: Array, wb: Array, s: Stream) !Array {
    var wt: Array = .{};
    defer drop(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wb, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
    return out;
}

fn hostBf16(y: Array) ![]const u16 {
    try mlx.check(mlx.mlx_array_eval(y));
    const p = mlx.mlx_array_data_bfloat16(y) orelse return error.MimoFp8Data;
    return p[0..mlx.mlx_array_size(y)];
}

/// ulp histogram: 0, 1, 2-3, 4-7, 8-15, 16+.
const Hist = struct {
    b: [6]u64 = @splat(0),
    max: u32 = 0,
    fn add(h: *Hist, u: u32) void {
        h.b[if (u == 0) 0 else if (u == 1) 1 else if (u < 4) 2 else if (u < 8) 3 else if (u < 16) 4 else 5] += 1;
        h.max = @max(h.max, u);
    }
};

test "mimo fp8res GPU check: decode GEMV at M = 1..8 on every real site: bit-equal to the kernel-order host emulation, within the f32 bound of the exact code x scale reference, M-invariant rows; the BF16 path's ulp distribution" {
    try gpuGate("MIMO_FP8RES_GPU");
    const a = testing.allocator;
    const bound = gemvSelfcheckBound();
    std.debug.print("MIMO_FP8RES_GEMV static: bound={d} ceiling={d}\n", .{ bound, selfcheck_ceiling_bytes });
    if (bound > selfcheck_ceiling_bytes) return error.MimoFp8SelfcheckAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, selfcheck_ceiling_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var failed: usize = 0;
    for (sites, 0..) |site, si| {
        const f = try Fixture.init(a, site.n, site.k, 8, 0xf8e5_6e00 + si);
        defer f.deinit(a);
        const b = try bind(&kernels, site.n, site.k);
        const dev = try DeviceSite.init(a, f, site.n, site.k, 8);
        defer dev.deinit();
        // The kernel order emulated, and the exact reference, for all 8 rows x N outputs (M-independent by
        // construction: the same rows serve every M).
        const n: usize = site.n;
        const emu = try a.alloc(u16, 8 * n);
        defer a.free(emu);
        const ref = try a.alloc(f64, 8 * n);
        defer a.free(ref);
        const refabs = try a.alloc(f64, 8 * n);
        defer a.free(refabs);
        {
            const w32 = try a.alloc(f32, site.k);
            defer a.free(w32);
            const w64 = try a.alloc(f64, site.k);
            defer a.free(w64);
            const x64 = try a.alloc(f64, 8 * site.k);
            defer a.free(x64);
            for (x64, f.x[0 .. 8 * site.k]) |*d, xb| d.* = bf16ToF32(xb);
            for (0..n) |ni| {
                weightRow(f.codes[ni * site.k ..][0..site.k], f.srow(ni, site.k), w32);
                for (w64, w32) |*d, v| d.* = v;
                for (0..8) |mi| {
                    emu[mi * n + ni] = f32ToBf16(hostGemvDotW(b.plan, f.x[mi * site.k ..][0..site.k], w32));
                    const r = refDotF64(x64[mi * site.k ..][0..site.k], w64);
                    ref[mi * n + ni] = r.sum;
                    refabs[mi * n + ni] = r.abs;
                }
            }
        }
        var emu_diff: usize = 0;
        var plan_diff: usize = 0;
        var tol_fail: usize = 0;
        var exact: Hist = .{};
        var vs_bf16: Hist = .{};
        var bf16_exact: Hist = .{};
        var m: c_int = 1;
        while (m <= gemv_max_rows) : (m += 1) {
            const xm = try rowsOf(dev.x, m, s);
            defer drop(xm);
            const y = try forward(&b, xm, dev.w, dev.sc, s);
            defer drop(y);
            const yd = try hostBf16(y);
            for (0..@intCast(m)) |mi| for (0..n) |ni| {
                const got = yd[mi * n + ni];
                emu_diff += @intFromBool(got != emu[mi * n + ni]);
                exact.add(ulps(got, f64ToBf16(ref[mi * n + ni])));
                tol_fail += @intFromBool(!withinTolerance(got, ref[mi * n + ni], refabs[mi * n + ni], site.k));
            };
            // X18: the installed width's plan (`forward`) against the site plan at the same width (`fixed`), on device.
            const yf = try gemvRun(&kernels, b.plan, b.n, b.k, xm, dev.w, dev.sc, m, s);
            defer drop(yf);
            const fd = try hostBf16(yf);
            for (yd, fd) |p, q| plan_diff += @intFromBool(p != q);
            if (m == 1 or m == 4 or m == 8) {
                const yb = try bf16Path(xm, dev.wb, s);
                defer drop(yb);
                const bd = try hostBf16(yb);
                for (0..@intCast(m)) |mi| for (0..n) |ni| {
                    vs_bf16.add(ulps(yd[mi * n + ni], bd[mi * n + ni]));
                    bf16_exact.add(ulps(bd[mi * n + ni], f64ToBf16(ref[mi * n + ni])));
                };
            }
        }
        var rs: [gemv_max_rows]c_int = undefined;
        for (&rs, b.plans[1..]) |*r, p| r.* = p.r;
        std.debug.print("MIMO_FP8RES_GEMV site={s} N={d} K={d} plan=R{d}/KS{d}/RG{d} route={s} R_by_M={any} M=1..8 outputs={d} host_emulation_bit_diff={d} m_invariant={} vs_exact max_ulp={d} hist={any} tolerance_fail={d} by_rows_vs_fixed_bit_diff={d}\n", .{ site.label, site.n, site.k, b.plan.r, b.plan.ks, b.plan.rg, @tagName(kernels.gemv_route), rs, 36 * n, emu_diff, emu_diff == 0, exact.max, exact.b, tol_fail, plan_diff });
        std.debug.print("MIMO_FP8RES_GEMV site={s} vs_bf16_path(M=1,4,8) max_ulp={d} hist={any} bf16_path_vs_exact max_ulp={d} hist={any}\n", .{ site.label, vs_bf16.max, vs_bf16.b, bf16_exact.max, bf16_exact.b });
        failed += emu_diff + tol_fail + plan_diff;
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_FP8RES_GEMV peak: mlx={d} bound={d} failed={d}\n", .{ peak, bound, failed });
    try testing.expectEqual(@as(usize, 0), failed);
    if (peak > bound) return error.MimoFp8SelfcheckPeak;
}

test "mimo fp8res GPU check: prompt route: the dequantized weight is the host bf16(code x scale) bit for bit and the prompt GEMM is the BF16 tier's stock matmul bit for bit (9 / 512 / 4,133 rows); the LRU keeps a layer's set" {
    try gpuGate("MIMO_FP8RES_GPU");
    const a = testing.allocator;
    const bound = gemmSelfcheckBound();
    std.debug.print("MIMO_FP8RES_PROMPT static: bound={d} ceiling={d}\n", .{ bound, selfcheck_ceiling_bytes });
    if (bound > selfcheck_ceiling_bytes) return error.MimoFp8SelfcheckAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, selfcheck_ceiling_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var cache: PromptCache = .{};
    defer cache.release();
    kernels.prompt = &cache;
    var failed: usize = 0;
    for (sites, 0..) |site, si| {
        const f = try Fixture.init(a, site.n, site.k, 4133, 0xf8e5_7000 + si);
        defer f.deinit(a);
        const b = try bind(&kernels, site.n, site.k);
        const dev = try DeviceSite.init(a, f, site.n, site.k, 4133);
        defer dev.deinit();
        // (1) the weight: device dequant vs the host bf16(code x scale) (DeviceSite.wb), every element.
        const wb = try cache.get(&kernels, dev.w, dev.sc, b.n, b.k, s);
        const got = try hostBf16(wb);
        const want = try hostBf16(dev.wb);
        var wdiff: usize = 0;
        for (got, want) |p, q| wdiff += @intFromBool(p != q);
        // (2) the prompt GEMM vs the stock matmul on the BF16 pack's weight, every element.
        var gdiff: usize = 0;
        for ([_]c_int{ 9, 512, 4133 }) |rows| {
            const xm = try rowsOf(dev.x, rows, s);
            defer drop(xm);
            const y = try forward(&b, xm, dev.w, dev.sc, s);
            defer drop(y);
            const yb = try bf16Path(xm, dev.wb, s);
            defer drop(yb);
            const yd = try hostBf16(y);
            const bd = try hostBf16(yb);
            for (yd, bd) |p, q| gdiff += @intFromBool(p != q);
        }
        std.debug.print("MIMO_FP8RES_PROMPT site={s} N={d} K={d} weight_bit_diff={d} gemm_bit_diff_vs_stock={d}\n", .{ site.label, site.n, site.k, wdiff, gdiff });
        failed += wdiff + gdiff;
        // The fixture's codes go with this site: its cache key (the codes' array) must not outlive them.
        if (si + 1 < sites.len) cache.release();
    }
    // The LRU: `cap` slots (X18 default 3), the least recently used goes; a hit returns the cached array.
    var hits: usize = 0;
    for (cache.keys) |k| hits += @intFromBool(k != null);
    std.debug.print("MIMO_FP8RES_PROMPT lru slots_used={d} cap={d} failed={d}\n", .{ hits, cache.cap, failed });
    try testing.expect(hits <= cache.cap);
    try testing.expectEqual(@as(usize, 0), failed);
}
