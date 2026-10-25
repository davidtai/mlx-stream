//! Layer-major prompt tiles; every resident projection retains its serial M=1 kernel.
const std = @import("std");
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const C = @import("config.zig").Config;
pub const Layer = struct {
    attn: attention.Weights,
    input_norm: ops.T,
    post_norm: ops.T,
    mlp: ops.Mlp,
    router: ?ops.T = null,
    correction: ?ops.T = null,
    down_bias: ?ops.T = null,
};

pub const Routes = struct {
    ids: [32]u16 = undefined,
    weights: [32]f32 = undefined,
    n: usize,
};

pub fn routeRows(g: *ops.G, c: *const C, x: ops.T, router: ops.T, correction: ops.T) !Routes {
    const shape = g.shapeOf(x);
    if (shape.n != 2 or shape.d[0] < 1 or shape.d[0] > 8) return error.MiniMaxChunkShape;
    const count: usize = @intCast(shape.d[0]);
    var selections: [8]ops.T = undefined;
    var mixtures: [8]ops.T = undefined;
    for (0..count) |i| {
        const h = try ops.row(g, x, @intCast(i));
        const scores = try g.sigmoid(try g.astype(try ops.linear(g, try g.astype(h, g.dtypeOf(router)), router), .float32));
        const ranked = try g.argpartition(try g.mul(try g.add(scores, correction), try g.scalar(-1, .float32)), 3, -1);
        const selected = try g.slice(ranked, &.{ 0, 0 }, &.{ 1, 4 }, &.{ 1, 1 });
        var weights = try g.takeAlongAxis(scores, selected, -1);
        weights = try g.div(weights, try g.add(try g.sum(weights, -1, true), try g.scalar(1e-20, .float32)));
        weights = try g.astype(try g.mul(weights, try g.scalar(c.routed_scaling_factor, .float32)), g.dtypeOf(h));
        selections[i] = selected;
        mixtures[i] = weights;
    }
    var routes: Routes = .{ .n = 4 * count };
    _ = try g.hostIds(try g.astype(try g.concat(selections[0..count], 0), .int32), routes.ids[0..routes.n]);
    _ = try g.hostF32(try g.astype(try g.concat(mixtures[0..count], 0), .float32), routes.weights[0..routes.n]);
    return routes;
}

pub fn forwardLayers(g: *ops.G, a: std.mem.Allocator, c: *const C, layers: []const Layer, caches: []attention.Cache, input: ops.T, position: u32, source: anytype, comptime chunked: bool) !ops.T {
    var x = g.keep(input);
    errdefer g.release(x);
    for (layers, caches, 0..) |layer, *cache, index| {
        const mark = g.mark();
        const normalized = if (chunked) try ops.normRows(g, x, layer.input_norm, c) else try ops.norm(g, x, layer.input_norm, c);
        const attended = if (chunked) try attention.forwardChunk(g, a, c, layer.attn, cache, normalized, position) else try attention.forward(g, a, c, layer.attn, cache, normalized, position);
        const next = try g.add(x, attended);
        const h = if (chunked) try ops.normRows(g, next, layer.post_norm, c) else try ops.norm(g, next, layer.post_norm, c);
        var y = if (chunked) try layer.mlp.applyRows(g, h, c) else try layer.mlp.apply(g, h, c);
        if (layer.router) |router| {
            const routes = try routeRows(g, c, h, router, layer.correction orelse return error.MiniMaxRouterCorrectionMissing);
            const routed = if (chunked)
                try source.applyChunk(g, c, @intCast(index), h, routes.ids[0..routes.n], routes.weights[0..routes.n], layer.down_bias)
            else
                try source.apply(g, c, @intCast(index), h, routes.ids[0..routes.n], routes.weights[0..routes.n], layer.down_bias);
            y = try g.add(routed, y);
            if (comptime @hasDecl(@TypeOf(source.*), "observeRoutes")) try source.observeRoutes(index, position, routes);
        }
        const output = try g.add(next, y);
        try g.evalAll(&.{output});
        if (comptime @hasDecl(@TypeOf(source.*), "observeLayer")) try source.observeLayer(g, index, position, output);
        const held = g.keep(output);
        g.resetTo(mark);
        g.release(x);
        x = held;
    }
    return x;
}
