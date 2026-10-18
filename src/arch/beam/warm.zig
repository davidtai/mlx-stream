//! P8 (glm5_next's prompt lever 3 for Beam): the prompt pass's first GPU work, moved to the load. The stack's profile
//! idled the GPU 1.06 s (cold) to 1.55 s (seeded) before the first request's first command buffer ran, and then built
//! its kernels at first use: layer 0's sliding attention at the 512-row tile, the first full layer's attention, the
//! 2,048-row dense tile, the first routed layer's predictor over a call and the prompt combine's text. The host applies
//! its wired policy after the load (mlx-serve `applyWiredPolicy`): MLX then adds every buffer of the load (the residents,
//! the slot arrays, the KV lanes: about 207 GB) to its residency set, and the GPU makes them resident before it runs the
//! next command buffer.
//!
//! The warm runs at the construction's end, after the usage seed:
//!   1. Under the host's default policy (`MLX_SERVE_WIRED` unset or `max`), MLX's wired limit is set to the device's
//!      recommended working set, the value the host sets after the load (its own call then changes nothing: MLX resizes
//!      its residency set only for a new limit). Under another policy the limit is the host's to set: step 1 does
//!      nothing, and the residency wait stays in the first request.
//!   2. Each kernel the prompt pass builds at first use runs once on zeros at the prompt pass's shapes: layer 0's
//!      attention over two tiles (its window mask), the first full layer's attention over two tiles (its causal SDPA and
//!      query scale), layer 0's dense MLP over one tile, the first routed layer's predictor (the router and the expert
//!      counts) over one call of `experts.call_rows` rows, the installed prompt combine over one token. The layers are
//!      read through the prompt pass's view (`graph.promptView`: the q8 tier's projections dequantized to BF16).
//!   3. The stream synchronized, the cache restarted (the rings rewound; a slab's rows past a request's length are never
//!      read), MLX's cache cleared.
//! No expert source state is touched (no route, read or slot): token-identical. Its memory is the prompt pass's (two
//! attention tiles, one dense tile, the predictor over one call), inside the request's billed terms, which hold nothing at
//! the load; the construction check runs after it.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const beam = @import("config.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");

const G = graph.G;
const T = graph.T;

/// What the warm runs, from the config alone.
pub const Plan = struct {
    /// The first sliding layer and the first full layer: their attention over `tiles` tiles each.
    sliding_layer: u32,
    full_layer: u32,
    tiles: u32,
    /// The first dense layer: its MLP over one tile.
    dense_layer: u32,
    /// The first routed layer: its predictor over `predict_rows` rows.
    routed_layer: u32,
    predict_rows: u32,
};

/// The warm's layers: Beam's layout has a sliding layer, a full layer, a dense layer and a routed layer; a layout without
/// one of them refuses the load by name (`BeamWarmLayout`). The tiles and the predictor's rows stay within `cap` (the
/// KV lanes' positions).
pub fn plan(c: *const beam.Config, cap: u32) !Plan {
    var sliding: ?u32 = null;
    var full: ?u32 = null;
    for (0..c.n_layers) |li| {
        const l: u32 = @intCast(li);
        if (c.isFull(l)) {
            if (full == null) full = l;
        } else if (sliding == null) sliding = l;
    }
    if (sliding == null or full == null or c.first_k_dense == 0 or c.first_k_dense >= c.n_layers) return error.BeamWarmLayout;
    return .{
        .sliding_layer = sliding.?,
        .full_layer = full.?,
        .tiles = @min(graph.tiles_in_flight, @max(1, cap / graph.tile_rows)),
        .dense_layer = 0,
        .routed_layer = c.first_k_dense,
        .predict_rows = @min(experts.call_rows, cap),
    };
}

/// Whether the host's wired policy is its default, `max` (mlx-serve `WiredMode`: `MLX_SERVE_WIRED` unset or `max`).
pub fn hostWiresMax() bool {
    const wired = std.c.getenv("MLX_SERVE_WIRED") orelse return true;
    return std.mem.eql(u8, std.mem.span(wired), "max");
}

pub const Report = struct { plan: Plan, ns: u64, wired_bytes: u64 };

/// The warm (construction, after the source, the cache and the usage seed; before the construction check): see the file's
/// doc. `src` is the expert source (its combine's warm only).
pub fn run(g: *G, a: std.mem.Allocator, io: std.Io, c: *const beam.Config, w: *const graph.Weights, cache: *graph.Cache, src: *experts.Source) !Report {
    const p = try plan(c, cache.cap);
    const t0 = std.Io.Timestamp.now(io, .awake);
    var wired: u64 = 0;
    // A GPU stream under the host's default policy only (a CPU stream has no residency set).
    if (hostWiresMax() and gpu_warm.gpuStream(g.s)) {
        const max_rec = mlx.maxRecommendedWorkingSet();
        if (max_rec > 0) {
            var previous: usize = 0;
            try mlx.check(mlx.mlx_set_wired_limit(&previous, max_rec));
            wired = max_rec;
        }
    }
    {
        const m = g.mark();
        defer g.resetTo(m);
        defer cache.restart();
        const hidden: c_int = @intCast(c.hidden_size);
        const tile = try g.zeros(&.{ @intCast(graph.tile_rows), hidden }, .bfloat16);
        try g.evalAll(&.{tile});
        // The attention tiles as a prompt call runs them: each committed as built, the one before it waited.
        for ([_]u32{ p.sliding_layer, p.full_layer }) |l| {
            // The prompt pass's view of the layer (the q8 tier's projections as BF16, dequantized once; a BF16 layer
            // is its own view): the same dequantize, GEMM and attention kernels as a prompt call's.
            const view = try graph.promptView(g, &w.layers[l]);
            const lw = &view;
            var prev: ?T = null;
            defer if (prev) |x| g.release(x);
            for (0..p.tiles) |i| {
                const ms = g.mark();
                const xn = try graph.rmsNorm(g, tile, lw.input_norm, c.rms_norm_eps);
                const out = g.keep(try graph.attention(g, a, c, lw, xn, @intCast(i * graph.tile_rows), cache));
                g.resetTo(ms);
                try g.asyncEval(&.{out});
                if (prev) |x| {
                    try g.evalAll(&.{x});
                    g.release(x);
                }
                prev = out;
            }
            if (prev) |x| {
                prev = null;
                try g.evalAll(&.{x});
                g.release(x);
            }
        }
        const dense_view = try graph.promptView(g, &w.layers[p.dense_layer]);
        const d = dense_view.dense orelse return error.BeamWarmLayout;
        const dense_tile = try g.zeros(&.{ @intCast(@min(graph.dense_tile_rows, p.predict_rows)), hidden }, .bfloat16);
        try g.evalAll(&.{try graph.mlp(g, d, dense_tile)});
        // The predictor over one call's rows (its router GEMM and the per-expert counts at the call's M).
        const call = try g.zeros(&.{ @intCast(p.predict_rows), hidden }, .bfloat16);
        try g.evalAll(&.{try graph.predictedCounts(g, c, &w.layers[p.routed_layer], call)});
        try src.combine_kernel.warmOne(&src.g, src.combine_route);
    }
    try mlx.check(mlx.mlx_synchronize(g.s));
    g.clearCache();
    return .{ .plan = p, .ns = @intCast(t0.untilNow(io, .awake).nanoseconds), .wired_bytes = wired };
}

const testing = std.testing;

test "beam warm: Beam's plan warms layer 0's sliding attention and layer 5's full attention over two tiles, layer 0's dense tile and layer 3's predictor over one call; a layout without one of them refuses" {
    const c: beam.Config = .{};
    const p = try plan(&c, 32_768);
    try testing.expectEqual(@as(u32, 0), p.sliding_layer);
    try testing.expectEqual(@as(u32, 5), p.full_layer);
    try testing.expectEqual(@as(u32, 2), p.tiles);
    try testing.expectEqual(@as(u32, 0), p.dense_layer);
    try testing.expectEqual(@as(u32, 3), p.routed_layer);
    try testing.expectEqual(experts.call_rows, p.predict_rows);
    // A billed context under one call: the predictor's rows and the tiles stay within the lanes.
    const small = try plan(&c, 700);
    try testing.expectEqual(@as(u32, 700), small.predict_rows);
    try testing.expectEqual(@as(u32, 1), small.tiles);
    var no_dense = c;
    no_dense.first_k_dense = 0;
    try testing.expectError(error.BeamWarmLayout, plan(&no_dense, 32_768));
}

test "beam warm: the warm's device code compiles against the arch's seams (declarations referenced, nothing run)" {
    testing.refAllDecls(@This());
    _ = &run;
}
