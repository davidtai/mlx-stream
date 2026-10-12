//! glm5_next's residents route (`residents_format` in the pack's config.json, read once at construction): `bf16` (the
//! default and the control: every trunk projection BF16 through MLX's matmul) or `fp8_block128`, MiMo's fp8res path
//! (src/arch/mimo_v2/fp8_linear.zig, its texts and pins unchanged) over GLM-5.3-Flash's trunk projections: e4m3fn codes U8
//! [out, in] `.weight` with F32 [out / 64, in / 128] `.weight_scale_inv` (value = code x scale, the release-style 64-row x
//! 128-column scale grid; the pack's tiles are 128 x 128, a last 64-row group one tile).
//!
//! The sites the route serves (439): every KDA layer's q / k / v / o / f_a / g_a / b, every DSA layer's q_a / q_b / kv_a
//! / o and its indexer's wk / wq_b, every MoE layer's shared expert, the dense layers' MLP. Each is FP8 when the pack's
//! `quantization` declares it so (config.zig `parseResidents`: the full set declares all 439, the pack lane's hybrid set
//! 190), else BF16 on the stock matmul. BF16 always: the lm_head, the embeddings, the mHC tables, KDA's f_b / g_b (K =
//! 128: the GEMV binds K % 256 == 0), the routers, the indexer's pool gate and weights_proj, kv_b (used per head), and the
//! MTP layer (the draft lane's residents are BF16 by its spec).
//!
//! Each FP8 projection is bound once at construction (`bind`: its GEMV plan per call width, MiMo's `gemvPlanAt`); a call
//! routes only on its rows M, a runtime value (MiMo's rule): M <= 8 (decode, the MTP verify, a short prompt call) runs
//! `mimo_fp8_gemv` (f32 FMA over code x scale, M-invariant: row m of an M-row call is the M = 1 call's row bit for bit);
//! M > 8 (the prompt) runs the BF16 path's own GEMM (`x @ w^T` through MLX's matmul) over the projection's BF16 weight,
//! written once per prompt layer by `mimo_fp8_dequant_bf16` (bf16(code x scale), one rounding) into the route's LRU
//! (`PromptCache`, MiMo's policy at `prompt_slots` slots). The cache lives one prompt layer: the layer's spans reuse its
//! weights, its end releases them (`Route.layerEnd`, after the layer's eval), so the prompt phase holds at most one
//! layer's FP8 set in BF16 (`promptBytes`, the bill's `fp8_prompt_weights`); decode holds none (the route is closed and
//! released at the phase change). Lossy against the BF16 pack (the KL gate): the prompt arithmetic is the BF16 path's on
//! bf16(code x scale) weights, the decode GEMV the release-style f32 product.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const mimo_fp8 = @import("../mimo_v2/fp8_linear.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// The release-style scale grid (MiMo's: one f32 scale per 64 output rows x 128 input columns).
pub const scale_rows: u32 = glm.fp8_scale_rows;
pub const scale_cols: u32 = glm.fp8_scale_cols;
/// The widest call the GEMV takes (MiMo's `gemv_max_rows`): decode (M = 1), the MTP verify (M <= 4), a short prompt call.
/// A wider call is the prompt route.
pub const gemv_max_rows: c_int = mimo_fp8.gemv_max_rows;
/// The prompt route's LRU capacity: the widest set of FP8 projections one prompt layer's attention touches in every span
/// (a KDA layer's q, k, v, f_a, b, g_a and o; a DSA layer's six), so a layer's spans dequantize each of its projections
/// once (MiMo's X18 rule: the cache holds one layer's set; at MiMo's 3 a KDA layer's 7 would be dequantized in every span).
/// The layer's MLP (3) then evicts its oldest attention weights; the layer's end releases the rest (`Route.layerEnd`).
pub const prompt_slots: usize = 7;

/// A trunk projection [out, in] as the construction bound it.
pub const Linear = union(enum) {
    bf16: T,
    fp8: Fp8,
};

pub const Fp8 = struct {
    w: T,
    scales: T,
    route: *Route,
    n: c_int,
    k: c_int,
    /// The GEMV plan a call of M rows runs (index M; 0 unused): MiMo's `gemvPlanAt` at the installed `by_rows`.
    plans: [mimo_fp8.gemv_max_rows + 1]mimo_fp8.Gemv,
};

/// What the bind takes for a trunk site [out, in] stored as `format` (the pack's declaration of that site: one table,
/// `graph.Weights.bind` checks the residents against it, the CPU tests the pack's shard headers): `fp8_block128` the codes
/// U8 [out, in] and the scales F32 [ceil(out / 64), in / 128]; `bf16` the weight BF16 [out, in] and no scales.
pub const SiteTensors = struct {
    pub const Dtype = enum { u8, bf16, f32 };
    weight: Dtype,
    weight_shape: [2]c_int,
    scales: ?[2]c_int,
};

pub fn siteTensors(format: glm.ResidentsFormat, out: u32, in: u32) SiteTensors {
    return switch (format) {
        .bf16 => .{ .weight = .bf16, .weight_shape = .{ @intCast(out), @intCast(in) }, .scales = null },
        .fp8_block128 => .{ .weight = .u8, .weight_shape = .{ @intCast(out), @intCast(in) }, .scales = scaleShape(out, in) },
    };
}

/// The scales' shape of an [out, in] FP8 projection (a last 64-row group one row of scales).
pub fn scaleShape(out: u32, in: u32) [2]c_int {
    return .{ @intCast((out + scale_rows - 1) / scale_rows), @intCast(in / scale_cols) };
}

/// The prompt route's BF16 weights: MiMo's `PromptCache` policy (`pick`: a hit, else the least recently used slot) at
/// `prompt_slots` slots, keyed by the codes' array.
pub const PromptCache = struct {
    keys: [prompt_slots]?*anyopaque = @splat(null),
    arrs: [prompt_slots]T = @splat(.{ .ctx = null }),
    stamps: [prompt_slots]u64 = @splat(0),
    clock: u64 = 0,

    pub fn release(c: *PromptCache) void {
        for (&c.arrs) |*x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        }
        c.keys = @splat(null);
        c.stamps = @splat(0);
    }

    /// The BF16 weight of `f` (borrowed: valid until a later `get` evicts it or `release`; the caller's graph holds its
    /// own reference once it uses the array).
    fn get(c: *PromptCache, dequant: mlx.mlx_fast_metal_kernel, f: *const Fp8, s: mlx.mlx_stream) !T {
        c.clock += 1;
        const p = mimo_fp8.PromptCache.pick(&c.keys, &c.stamps, f.w.ctx);
        if (p.hit) {
            c.stamps[p.slot] = c.clock;
            return c.arrs[p.slot];
        }
        const j = p.slot;
        if (c.arrs[j].ctx != null) _ = mlx.mlx_array_free(c.arrs[j]);
        c.arrs[j] = .{ .ctx = null };
        c.keys[j] = null;
        c.arrs[j] = try dequantRun(dequant, f.w, f.scales, f.n, f.k, s);
        c.keys[j] = f.w.ctx;
        c.stamps[j] = c.clock;
        return c.arrs[j];
    }
};

/// The installed route (heap-owned: every bound projection points here): MiMo's kernel objects (no device work until a
/// call) and the prompt route's cache.
pub const Route = struct {
    kernels: mimo_fp8.Kernels,
    cache: PromptCache = .{},
    /// Through decode the prompt route is closed (no M > 8 call reaches it; its cache is released and unbilled).
    closed: bool = false,

    pub fn install(a: std.mem.Allocator) !*Route {
        const r = try a.create(Route);
        errdefer a.destroy(r);
        r.* = .{ .kernels = try mimo_fp8.Kernels.init() };
        return r;
    }

    pub fn destroy(r: *Route, a: std.mem.Allocator) void {
        r.cache.release();
        r.kernels.deinit();
        a.destroy(r);
    }

    /// A prompt pass's start (or a decode-phase extend's prompt calls): the route open.
    pub fn open(r: *Route) void {
        r.closed = false;
    }

    /// A prompt layer's end (graph.forward, after the layer's outputs are evaluated): its BF16 weights released. No
    /// projection is shared between layers, so the next layer's dequants never wait on these, and the prompt phase holds
    /// at most one layer's set (`promptBytes`).
    pub fn layerEnd(r: *Route) void {
        r.cache.release();
    }

    /// A prompt pass's end: its BF16 weights released, the route closed through decode.
    pub fn close(r: *Route) void {
        r.cache.release();
        r.closed = true;
    }
};

/// Construction: an [n, k] FP8 projection's route, refused by name when the GEMV's plan does not cover the shape.
pub fn bind(route: *Route, w: T, scales: T, n: u32, k: u32) !Fp8 {
    const b = mimo_fp8.bind(&route.kernels, n, k) catch return error.Glm5NextFp8Geometry;
    return .{ .w = w, .scales = scales, .route = route, .n = b.n, .k = b.k, .plans = b.plans };
}

/// x [..., in] bf16 -> [..., out] bf16 through the projection's bound route. `bf16`: the stock `x @ w^T`. `fp8`: the
/// GEMV at M <= 8 rows (x viewed [M, in]), else the stock `x @ w^T` over the cached BF16 weight (the BF16 path's GEMM).
pub fn project(g: *G, l: Linear, x: T) !T {
    switch (l) {
        .bf16 => |w| return g.matmul(x, try g.transpose(w)),
        .fp8 => |*f| {
            const sh = g.shapeOf(x);
            const nd: usize = sh.n;
            var m: c_int = 1;
            for (sh.d[0 .. nd - 1]) |d| m *= d;
            if (m > gemv_max_rows) {
                // The prompt phase holds the route open (Module: `open` at a prompt pass's start, `close` at its end).
                std.debug.assert(!f.route.closed);
                const wb = try f.route.cache.get(f.route.kernels.dequant, f, g.s);
                return g.matmul(x, try g.transpose(wb));
            }
            const x2 = try g.reshape(x, &.{ m, f.k });
            const y2 = try g.adopt(try mimo_fp8.gemvRun(&f.route.kernels, f.plans[@intCast(m)], f.n, f.k, x2, f.w, f.scales, m, g.s));
            var shape: [ops_be.max_dims]c_int = undefined;
            @memcpy(shape[0 .. nd - 1], sh.d[0 .. nd - 1]);
            shape[nd - 1] = f.n;
            return g.reshape(y2, shape[0..nd]);
        },
    }
}

/// The BF16 weight [N, K] of an FP8 projection (one word of codes per thread): MiMo's `mimo_fp8_dequant_bf16` launch.
fn dequantRun(kernel: mlx.mlx_fast_metal_kernel, w: T, scales: T, n: c_int, kdim: c_int, s: mlx.mlx_stream) !T {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ n, kdim }, 2, .bfloat16));
    const words = @divExact(n * kdim, 4);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, words, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "N", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", kdim));
    const ins = [_]T{ w, scales };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernel, in_vec, c, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

/// Layer `l`'s sites: its attention's (`fp8Sites`, called per span), then its MLP's (`fp8MlpSites`); the index of each is
/// `glm.Config.siteIndex`, its bit in `fp8_mask`.
pub fn layerSites(c: *const glm.Config, l: u32, out: *[10]glm.Fp8Site) []const glm.Fp8Site {
    var buf: [7]glm.Fp8Site = undefined;
    const attn = c.fp8Sites(l, &buf);
    @memcpy(out[0..attn.len], attn);
    const mlp = c.fp8MlpSites(l);
    @memcpy(out[attn.len..][0..3], &mlp);
    return out[0 .. attn.len + 3];
}

/// Layer `l`'s FP8-declared projections in BF16, each in its own MLX buffer (page-rounded past one page): what the prompt
/// route can hold for the layer at once (its cached weights and, while its last span is in flight, the attention weights
/// its MLP's dequants evicted).
pub fn layerPromptBytes(c: *const glm.Config, l: u32, page: u64) u64 {
    var buf: [10]glm.Fp8Site = undefined;
    var total: u64 = 0;
    for (layerSites(c, l, &buf), 0..) |st, i| {
        if ((c.fp8_mask[l] >> @intCast(i)) & 1 == 1) total += bufferBytes(@as(u64, st.out) * st.in * 2, page);
    }
    return total;
}

/// The prompt route's bill (0 with no FP8-declared site: `bf16`): the widest layer's FP8 set in BF16 (the full set: layers
/// 0-2, KDA with the dense MLP, 573,046,784 B), geometric over the layers' declared site shapes. A layer's weights live
/// from its first span to its end (`Route.layerEnd`, after the layer's outputs are evaluated): the prompt phase never holds
/// two layers' sets. The decode phase bills none: the route is closed there.
pub fn promptBytes(c: *const glm.Config, page: u64) u64 {
    var widest: u64 = 0;
    for (0..c.n_layers) |l| widest = @max(widest, layerPromptBytes(c, @intCast(l), page));
    return widest;
}

fn bufferBytes(n: u64, page: u64) u64 {
    return if (n > page) std.mem.alignForward(u64, n, page) else n;
}

/// Tests: `c` with every site the route serves declared FP8 (the full set's `quantization`).
pub fn fullSet(c: glm.Config) glm.Config {
    var out = c;
    out.residents_format = .fp8_block128;
    for (0..out.n_layers) |l| {
        var buf: [10]glm.Fp8Site = undefined;
        out.fp8_mask[l] = @intCast((@as(u32, 1) << @intCast(layerSites(&out, @intCast(l), &buf).len)) - 1);
    }
    return out;
}

const testing = std.testing;

test "glm5_next fp8 residents: every trunk FP8 site has a GEMV plan (MiMo's static asserts hold at every width), the scale grid" {
    const c: glm.Config = .{};
    var total: u32 = 0;
    for (0..c.n_layers) |l| {
        var buf: [10]glm.Fp8Site = undefined;
        for (layerSites(&c, @intCast(l), &buf)) |st| {
            const p = mimo_fp8.gemvPlan(st.out, st.in) orelse {
                std.debug.print("no GEMV plan for {s} [{d}, {d}]\n", .{ st.suffix, st.out, st.in });
                return error.TestUnexpectedResult;
            };
            try testing.expect(@rem(@as(c_int, @intCast(st.in)), p.ks * @as(c_int, @intCast(mimo_fp8.gemv_step))) == 0);
            try testing.expect(@rem(@as(c_int, @intCast(st.out)), p.r * p.rg) == 0 and @rem(64, p.r) == 0);
            for (1..mimo_fp8.gemv_max_rows + 1) |m| {
                const pm = mimo_fp8.gemvPlanAt(.by_rows, p, st.out, st.in, @intCast(m));
                try testing.expectEqual(p.ks, pm.ks);
                try testing.expect(@rem(@as(c_int, @intCast(st.out)), pm.r * pm.rg) == 0 and @rem(64, pm.r) == 0);
            }
            // The pack's grid: whole 64-row scale rows and 128-column blocks (the scales [out / 64, in / 128] exactly).
            try testing.expect(st.out % scale_rows == 0 and st.in % scale_cols == 0);
            total += 1;
        }
    }
    // The pack's groups: KDA 34 x 7, DSA 11 x 4 + its indexer's 11 x 2, the shared experts 42 x 3, the dense MLPs 3 x 3.
    try testing.expectEqual(@as(u32, 238 + 44 + 22 + 126 + 9), total);
    try testing.expectEqual(total, c.fp8TrunkSites());
    // The scale grid of the pack's shapes (its shard headers, residents-fp8-manifest.json): b_proj, f_a, q_b, the DSA o.
    try testing.expectEqual([2]c_int{ 1, 32 }, scaleShape(64, 4096));
    try testing.expectEqual([2]c_int{ 2, 32 }, scaleShape(128, 4096));
    try testing.expectEqual([2]c_int{ 256, 12 }, scaleShape(16384, 1536));
    try testing.expectEqual([2]c_int{ 64, 128 }, scaleShape(4096, 16384));
    // A last 64-row group is one row of scales.
    try testing.expectEqual([2]c_int{ 3, 32 }, scaleShape(192, 4096));
}

test "glm5_next fp8 residents: the route by residents_format (bf16 binds BF16 [out, in] and no scales; fp8_block128 codes U8 [out, in] with F32 scales [out / 64, in / 128])" {
    const bf = siteTensors(.bf16, 8192, 4096);
    try testing.expectEqual(SiteTensors.Dtype.bf16, bf.weight);
    try testing.expectEqual([2]c_int{ 8192, 4096 }, bf.weight_shape);
    try testing.expect(bf.scales == null);
    const f8 = siteTensors(.fp8_block128, 8192, 4096);
    try testing.expectEqual(SiteTensors.Dtype.u8, f8.weight);
    try testing.expectEqual([2]c_int{ 8192, 4096 }, f8.weight_shape);
    try testing.expectEqual([2]c_int{ 128, 32 }, f8.scales.?);
    // The default (absent `residents_format`) is the control.
    const c: glm.Config = .{};
    try testing.expectEqual(glm.ResidentsFormat.bf16, c.residents_format);
    try testing.expectEqual(@as(u64, 0), promptBytes(&c, 16384));
}

test "glm5_next fp8 residents: the prompt route's bill is the widest layer's FP8 set in BF16 (layers 0-2: KDA + the dense MLP), at least every layer's; the hybrid set's; 0 under bf16" {
    const c = fullSet(.{});
    try testing.expectEqual(@as(u32, 439), c.fp8Declared());
    const page: u64 = 16384;
    // KDA q / k / v / o [8192, 4096] / [4096, 8192], f_a / g_a [128, 4096], b [64, 4096]; the dense gate / up / down
    // [12288, 4096] / [4096, 12288]; 2 B each.
    const kda_dense: u64 = 2 * (4 * 8192 * 4096 + 2 * 128 * 4096 + 64 * 4096 + 3 * 12288 * 4096);
    try testing.expectEqual(@as(u64, 573_046_784), kda_dense);
    try testing.expectEqual(kda_dense, promptBytes(&c, page));
    // A DSA layer (q_a [1536, 4096], q_b [16384, 1536], kv_a [512, 4096], o [4096, 16384], wk [128, 4096], wq_b [4096,
    // 1536]) with its shared expert ([2048, 4096] x 2, [4096, 2048]); a KDA layer with its shared expert.
    try testing.expectEqual(@as(u64, 2 * (1536 * 4096 + 16384 * 1536 + 512 * 4096 + 4096 * 16384 + 128 * 4096 + 4096 * 1536 + 3 * 2048 * 4096)), layerPromptBytes(&c, 3, page));
    try testing.expectEqual(@as(u64, 2 * (4 * 8192 * 4096 + 2 * 128 * 4096 + 64 * 4096 + 3 * 2048 * 4096)), layerPromptBytes(&c, 4, page));
    for (0..c.n_layers) |l| try testing.expect(layerPromptBytes(&c, @intCast(l), page) <= promptBytes(&c, page));
    // Below the seven widest sites anywhere (the cache's capacity with no layer bound: seven DSA o_proj).
    try testing.expect(promptBytes(&c, page) < 7 * 2 * 4096 * 16384);
    // The hybrid set (no KDA site, no wq_b): its widest layer is a dense one's MLP (3 x [12288, 4096]).
    var hybrid = c;
    for (0..c.n_layers) |l| {
        if (!c.isDsa(@intCast(l))) hybrid.fp8_mask[l] &= ~@as(u16, 0x7f);
        if (c.isDsa(@intCast(l))) hybrid.fp8_mask[l] &= ~@as(u16, 1 << 5);
    }
    try testing.expectEqual(@as(u32, 190), hybrid.fp8Declared());
    try testing.expectEqual(@as(u64, 2 * 3 * 12288 * 4096), promptBytes(&hybrid, page));
    try testing.expectEqual(@as(u64, 0), promptBytes(&glm.Config{}, page));
}

/// One prompt call's FP8 calls in the trunk's order (graph.forward: per layer the attention span by span, `span_rows`
/// rows each, then the MLP: the dense layers' tiles of `dense_tile_rows` rows, the shared expert over every row; the
/// layer's end), each as (layer, site index, M). A span's sites are listed in `fp8Sites` order (the forward's own order
/// differs; any order of at most `prompt_slots` distinct sites per span gives the same dequants), and a DSA span is
/// modelled calling its indexer's wq_b in every span (it does only past the dense prefix: a superset).
const CallOrder = struct {
    const Call = struct { layer: u32, site: u8, m: u32, end: bool = false };
    fn build(a: std.mem.Allocator, c: *const glm.Config, rows: u32, list: *std.ArrayList(Call)) !void {
        const graph = @import("graph.zig");
        const span: u32 = graph.span_rows;
        const tile: u32 = graph.dense_tile_rows;
        for (0..c.n_layers) |li| {
            const l: u32 = @intCast(li);
            var buf: [7]glm.Fp8Site = undefined;
            const n_attn: u8 = @intCast(c.fp8Sites(l, &buf).len);
            var s0: u32 = 0;
            while (s0 < rows) : (s0 += span) {
                const m = @min(span, rows - s0);
                for (0..n_attn) |i| try list.append(a, .{ .layer = l, .site = @intCast(i), .m = m });
            }
            if (c.isMoe(l)) {
                for (0..3) |i| try list.append(a, .{ .layer = l, .site = n_attn + @as(u8, @intCast(i)), .m = rows });
            } else {
                var t0: u32 = 0;
                const tr = if (rows <= tile) rows else tile;
                while (t0 < rows) : (t0 += tr) {
                    const m = @min(tr, rows - t0);
                    for (0..3) |i| try list.append(a, .{ .layer = l, .site = n_attn + @as(u8, @intCast(i)), .m = m });
                }
            }
            try list.append(a, .{ .layer = l, .site = 0, .m = 0, .end = true });
        }
    }
    /// The dequants `cap` slots run over the calls (`PromptCache.pick` on every prompt-route call, the release at every
    /// layer's end when `layer_end`), and the most distinct layers the cache held at once.
    fn replay(cap: usize, calls: []const Call, layer_end: bool) struct { misses: usize, layers_at_once: usize } {
        var keys: [prompt_slots]?*anyopaque = @splat(null);
        var stamps: [prompt_slots]u64 = @splat(0);
        var owner: [prompt_slots]u32 = @splat(0);
        var clock: u64 = 0;
        var misses: usize = 0;
        var most: usize = 0;
        for (calls) |cl| {
            if (cl.end) {
                if (layer_end) {
                    keys = @splat(null);
                    stamps = @splat(0);
                }
                continue;
            }
            if (cl.m <= gemv_max_rows) continue;
            clock += 1;
            const key: *anyopaque = @ptrFromInt((@as(usize, cl.layer) * 16 + cl.site + 1) * 64);
            const p = mimo_fp8.PromptCache.pick(keys[0..cap], stamps[0..cap], key);
            if (!p.hit) {
                misses += 1;
                keys[p.slot] = key;
                owner[p.slot] = cl.layer;
            }
            stamps[p.slot] = clock;
            var seen: [prompt_slots]u32 = undefined;
            var n: usize = 0;
            for (keys[0..cap], owner[0..cap]) |k, o| if (k != null and std.mem.indexOfScalar(u32, seen[0..n], o) == null) {
                seen[n] = o;
                n += 1;
            };
            most = @max(most, n);
        }
        return .{ .misses = misses, .layers_at_once = most };
    }
};

test "glm5_next fp8 residents: over the trunk's real prompt call orders (9 .. 16,384-row calls), 7 slots and the layer's release dequantize every FP8 projection once per layer per call and never hold two layers; MiMo's 3 thrash on a KDA layer's spans" {
    const a = testing.allocator;
    const c = fullSet(.{});
    for ([_]u32{ 9, 100, 1024, 1025, 2048, 4133, 16_384 }) |rows| {
        var calls: std.ArrayList(CallOrder.Call) = .empty;
        defer calls.deinit(a);
        try CallOrder.build(a, &c, rows, &calls);
        // Every projection called with M > 8 once per layer (each site's widest call: the attention's spans of >= 9
        // rows, the MLP's tiles or rows).
        var want: usize = 0;
        for (0..c.n_layers) |li| {
            var buf: [10]glm.Fp8Site = undefined;
            const n_sites = layerSites(&c, @intCast(li), &buf).len;
            for (0..n_sites) |site| {
                for (calls.items) |cl| if (!cl.end and cl.layer == li and cl.site == site and cl.m > gemv_max_rows) {
                    want += 1;
                    break;
                };
            }
        }
        const r7 = CallOrder.replay(prompt_slots, calls.items, true);
        try testing.expectEqual(want, r7.misses);
        try testing.expectEqual(@as(usize, 1), r7.layers_at_once);
        if (rows > 2 * 1024) {
            // MiMo's default capacity on GLM's order: every KDA span re-dequantizes the layer's seven.
            const r3 = CallOrder.replay(3, calls.items, true);
            try testing.expect(r3.misses > 2 * want);
            std.debug.print("GLM5N_FP8_LRU rows={d} dequants slots7={d} slots3={d}\n", .{ rows, r7.misses, r3.misses });
        }
    }
}

/// The pack's shard headers (name -> dtype, shape): pure host, F_NOCACHE (bill.zig's reader's rule), no payload read.
const Headers = struct {
    const Entry = struct { dtype: []const u8, shape: []const i64 };
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged(Entry) = .empty,

    fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Headers {
        var h: Headers = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer h.arena.deinit();
        const a = h.arena.allocator();
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const raw = try d.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20));
        const index = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
        const wm = index.object.get("weight_map").?.object;
        var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = wm.iterator();
        while (it.next()) |e| try shards.put(a, e.value_ptr.*.string, {});
        for (shards.keys()) |shard| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, shard }, 0);
            const fd = try sdk.io_util.openNoCache(path.ptr, .{});
            defer _ = std.c.close(fd);
            var prefix: [8]u8 = undefined;
            if (std.c.pread(fd, &prefix, 8, 0) != 8) return error.Glm5NextResidentShard;
            const len = std.mem.readInt(u64, &prefix, .little);
            if (len < 2 or len > 64 << 20) return error.Glm5NextResidentShard;
            const header = try a.alloc(u8, @intCast(len));
            var done: usize = 0;
            while (done < header.len) {
                const n = std.c.pread(fd, header[done..].ptr, header.len - done, @intCast(8 + done));
                if (n <= 0) return error.Glm5NextResidentShard;
                done += @intCast(n);
            }
            const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, header, .{});
            var tensors = parsed.object.iterator();
            while (tensors.next()) |t| {
                if (std.mem.eql(u8, t.key_ptr.*, "__metadata__")) continue;
                const shape = t.value_ptr.*.object.get("shape").?.array.items;
                const dims = try a.alloc(i64, shape.len);
                for (shape, dims) |v, *x| x.* = v.integer;
                try h.map.put(a, t.key_ptr.*, .{ .dtype = t.value_ptr.*.object.get("dtype").?.string, .shape = dims });
            }
        }
        return h;
    }

    fn deinit(h: *Headers) void {
        h.arena.deinit();
    }
};

test "glm5_next fp8 residents: the pack's shards carry every trunk FP8 site in its residents_format's tensors (GLM5_BANK, headers only)" {
    const dir = std.mem.span(std.c.getenv("GLM5_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const config_text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(config_text);
    var diag: glm.Diag = .{};
    const c = glm.Config.parse(a, config_text, &diag) catch |e| {
        std.debug.print("config refused: {s}\n", .{diag.message()});
        return e;
    };
    var h = try Headers.read(a, io, dir);
    defer h.deinit();
    const names = [_][]const u8{ "U8", "BF16", "F32" };
    var sites: u32 = 0;
    var scales: u32 = 0;
    var bad: u32 = 0;
    for (0..c.n_layers) |li| {
        var buf: [10]glm.Fp8Site = undefined;
        for (layerSites(&c, @intCast(li), &buf)) |st| {
            const want = siteTensors(if (c.siteIsFp8(@intCast(li), st.suffix)) .fp8_block128 else .bf16, st.out, st.in);
            var nb: [160]u8 = undefined;
            const wname = try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.{s}.weight", .{ li, st.suffix });
            const got = h.map.get(wname) orelse {
                std.debug.print("GLM5N_FP8_PACK missing {s}\n", .{wname});
                bad += 1;
                continue;
            };
            if (!std.mem.eql(u8, got.dtype, names[@intFromEnum(want.weight)]) or got.shape.len != 2 or got.shape[0] != want.weight_shape[0] or got.shape[1] != want.weight_shape[1]) {
                std.debug.print("GLM5N_FP8_PACK {s} is {s} {any}, want {s} {any}\n", .{ wname, got.dtype, got.shape, names[@intFromEnum(want.weight)], want.weight_shape });
                bad += 1;
            }
            var sb: [176]u8 = undefined;
            const sname = try std.fmt.bufPrint(&sb, "model.language_model.layers.{d}.{s}.weight_scale_inv", .{ li, st.suffix });
            if (want.scales) |ss| {
                const sg = h.map.get(sname) orelse {
                    std.debug.print("GLM5N_FP8_PACK missing {s}\n", .{sname});
                    bad += 1;
                    continue;
                };
                if (!std.mem.eql(u8, sg.dtype, "F32") or sg.shape.len != 2 or sg.shape[0] != ss[0] or sg.shape[1] != ss[1]) {
                    std.debug.print("GLM5N_FP8_PACK {s} is {s} {any}, want F32 {any}\n", .{ sname, sg.dtype, sg.shape, ss });
                    bad += 1;
                }
                scales += 1;
            } else if (h.map.get(sname) != null) {
                std.debug.print("GLM5N_FP8_PACK {s} present under residents_format bf16\n", .{sname});
                bad += 1;
            }
            sites += 1;
        }
    }
    std.debug.print("GLM5N_FP8_PACK pack={s} residents_format={s} sites={d} fp8_declared={d} scales={d} mismatches={d} prompt_bytes={d}\n", .{ dir, @tagName(c.residents_format), sites, c.fp8Declared(), scales, bad, promptBytes(&c, std.heap.pageSize()) });
    try testing.expectEqual(@as(u32, 0), bad);
    try testing.expectEqual(c.fp8TrunkSites(), sites);
    try testing.expectEqual(c.fp8Declared(), scales);
}
