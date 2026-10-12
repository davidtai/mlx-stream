//! Prompt lever 3 (glm5next/prefill-levers): the prompt pass's first-use kernel builds, moved to the load. The 16K
//! profile idled 0.60 s inside the measured request on first use: layer 0's attention (the KDA texts, the
//! hyper-connection texts, the span GEMMs: 233 ms over later layers') and its dense MLP (the 2,048-row tiles' GEMMs:
//! 104 ms), layer 3's DSA attention (the indexer, top-k and sparse latent texts across the call's growing key counts:
//! ~220 ms) and its combine (the in-place text: 48 ms). MLX builds a custom kernel's library, and a pipeline for each
//! specialization, at its first dispatch; the warm dispatches each of these once at the load, at the shapes the prompt
//! pass runs: the first KDA layer's attention over two spans of `graph.span_rows` rows, the first dense layer's MLP over
//! one tile of `graph.dense_tile_rows` rows, the first DSA layer's attention over a whole prompt call's spans (positions
//! 0 to `graph.prompt_call_rows`, or the billed context's whole spans below it: the dense prefix, then the sparse rows at
//! every key count a call reaches) and the installed prompt combine over one token. Inputs are zeros; nothing is checked (a compile warm, not a self-check).
//! What it leaves: the request cache restarted (every KDA state cleared, the DSA lanes' lengths 0: their buffers keep
//! the warm's rows past any request's length, and a request reads only the positions it appended), the expert source
//! untouched (no route, read or slot), MLX's cache cleared. Token-identical: no request state is changed. Its memory is
//! the prompt pass's (two spans in flight, one dense tile), within the request's billed terms, which hold nothing at the
//! load. A prompt longer than one call reaches key counts the warm does not: those compile on first use, as before.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const glm = @import("config.zig");
const graph = @import("graph.zig");
const dsa = @import("dsa.zig");
const experts = @import("experts.zig");

const G = graph.G;
const T = graph.T;

/// What the warm runs, from the config alone.
pub const Plan = struct {
    /// The first KDA layer: its attention over `kda_spans` spans (the second carries the first's state, as a call does).
    kda_layer: u32,
    kda_spans: u32,
    /// The first dense layer: its MLP over one tile.
    dense_layer: u32,
    /// The first DSA layer: its attention over a whole prompt call's spans (the billed context's whole spans below one call).
    dsa_layer: u32,
    dsa_spans: u32,
};

/// The warm's layers: GLM-5.3-Flash's layout (a KDA layer, a dense layer, a DSA layer: layers 0, 0 and 3), the DSA spans
/// within `cap` positions (the request cache's: its lanes hold no more). A layout without one of them refuses by name
/// (the module's load: `Glm5NextWarmLayout`).
pub fn plan(c: *const glm.Config, cap: u32) !Plan {
    var kda_layer: ?u32 = null;
    var dsa_layer: ?u32 = null;
    for (0..c.n_layers) |li| {
        const l: u32 = @intCast(li);
        if (c.isDsa(l)) {
            if (dsa_layer == null) dsa_layer = l;
        } else if (kda_layer == null) kda_layer = l;
    }
    if (kda_layer == null or dsa_layer == null or c.first_k_dense == 0) return error.Glm5NextWarmLayout;
    return .{
        .kda_layer = kda_layer.?,
        .kda_spans = graph.spans_in_flight,
        .dense_layer = 0,
        .dsa_layer = dsa_layer.?,
        .dsa_spans = @min(graph.prompt_call_rows, cap) / graph.span_rows,
    };
}

pub const Report = struct { plan: Plan, ns: u64 };

/// The warm (construction, after the source and the cache, before the construction check): see the file's doc.
pub fn run(g: *G, io: std.Io, c: *const glm.Config, w: *const graph.Weights, cache: *graph.Cache, src: *experts.Source, routes: dsa.Routes) !Report {
    const p = try plan(c, cache.cap);
    const t0 = std.Io.Timestamp.now(io, .awake);
    {
        const m = g.mark();
        defer g.resetTo(m);
        defer cache.restart(g);
        const hidden: c_int = @intCast(c.hidden_size);
        const zero = try g.zeros(&.{ 1, @intCast(graph.span_rows), @intCast(c.hc_mult), hidden }, .bfloat16);
        try g.evalAll(&.{zero});
        try spans(g, c, &w.layers[p.kda_layer], zero, p.kda_spans, cache, routes);
        try spans(g, c, &w.layers[p.dsa_layer], zero, p.dsa_spans, cache, routes);
        const d = w.layers[p.dense_layer].dense orelse return error.Glm5NextWarmLayout;
        const x = try g.zeros(&.{ @intCast(graph.dense_tile_rows), hidden }, .bfloat16);
        try g.evalAll(&.{try graph.mlp(g, c, d, x)});
        try src.warmCombine();
    }
    try mlx.check(mlx.mlx_synchronize(g.s));
    g.clearCache();
    return .{ .plan = p, .ns = @intCast(t0.untilNow(io, .awake).nanoseconds) };
}

/// Layer `lw`'s attention over `n` spans of zeros at positions 0, span_rows, ...: each span committed with the layer's
/// lanes or state as a prompt call commits it, the span before it waited (two in flight).
fn spans(g: *G, c: *const glm.Config, lw: *const graph.Layer, zero: T, n: u32, cache: *graph.Cache, routes: dsa.Routes) !void {
    var prev: ?T = null;
    defer if (prev) |x| g.release(x);
    for (0..n) |i| {
        const ms = g.mark();
        const next = g.keep(try graph.attentionSpan(g, c, lw, zero, @intCast(i * graph.span_rows), cache, routes, .last));
        g.resetTo(ms);
        var bufs: [2]T = undefined;
        const b = cache.buffers(lw, &bufs);
        var commit: [3]T = undefined;
        commit[0] = next;
        @memcpy(commit[1 .. 1 + b.len], b);
        try g.asyncEval(commit[0 .. 1 + b.len]);
        if (prev) |x| {
            try g.evalAll(&.{x});
            g.release(x);
        }
        prev = next;
    }
    if (prev) |x| {
        prev = null;
        try g.evalAll(&.{x});
        g.release(x);
    }
}

const testing = std.testing;

test "glm5_next warm: Flash's plan warms layer 0's KDA attention over two spans and its dense MLP tile, layer 3's DSA attention over a whole call (16 spans: the dense prefix and every key count to 16,384) and refuses a layout without one of them" {
    const c: glm.Config = .{};
    const p = try plan(&c, 32_768);
    try testing.expectEqual(@as(u32, 0), p.kda_layer);
    try testing.expectEqual(@as(u32, 2), p.kda_spans);
    try testing.expectEqual(@as(u32, 0), p.dense_layer);
    try testing.expectEqual(@as(u32, 3), p.dsa_layer);
    try testing.expectEqual(@as(u32, 16), p.dsa_spans);
    try testing.expectEqual(@as(u32, graph.prompt_call_rows), p.dsa_spans * graph.span_rows);
    // The DSA spans reach the sparse rows (past the dense prefix: every pool fits the top-k up to `dense_end`).
    const dense_end = (c.index_topk / c.index_kpool + 1) * c.index_kpool - 1;
    try testing.expect(p.dsa_spans * graph.span_rows > dense_end);
    // A billed context under one call: its whole spans only (the lanes hold `cap` positions).
    try testing.expectEqual(@as(u32, 4), (try plan(&c, 4096)).dsa_spans);
    try testing.expectEqual(@as(u32, 4), (try plan(&c, 5000)).dsa_spans);
    try testing.expectEqual(@as(u32, 0), (try plan(&c, 1000)).dsa_spans);
    var no_dsa = c;
    no_dsa.dsa_mask = 0;
    try testing.expectError(error.Glm5NextWarmLayout, plan(&no_dsa, 32_768));
    var no_dense = c;
    no_dense.first_k_dense = 0;
    try testing.expectError(error.Glm5NextWarmLayout, plan(&no_dense, 32_768));
}

test "glm5_next warm: the warm's device code compiles against the arch's seams (declarations referenced, nothing run)" {
    testing.refAllDecls(@This());
    _ = &run;
    _ = &experts.Source.warmCombine;
}
