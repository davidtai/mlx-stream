//! beam's route recorder (a diagnostic): with `BEAM_ROUTES` set at construction (the host's convention,
//! `expert.routes.dirOf`: `/dir`, or any other value for /tmp), each request's decode routes go to
//! `<dir>/beam-routes-<pid>-<n>.bin` (sdk_ext/expert/routes.zig, version 4) at its end: per step and routed layer the
//! routed ids, what served each, the lookahead's reads, the plan's stamp and four predictions (each the router's top 16
//! by score); per step four routes with no ids for the layers before the first routed layer, and the step's input token.
//!
//! The slots of a routed layer L's route (x1 = L's residual after its attention, h = rmsNorm(x1, pre_ff_norm L), the
//! routed call's input; router t = sigmoid(f32 x @ W_t^T) + bias_t):
//!   0  router L+1 on h (the lookahead's own scores; the stream records them)
//!   1  router L+2 on h
//!   2  router L+3 on h
//!   3  router L+1 on rmsNorm(x1, pre_ff_norm L+1)
//! The pseudo routes (layer -4 the embedding row, -3 .. -1 the dense layers 0 .. 2; x1 the dense layer's residual after
//! its attention, the embedding row itself for -4): slots 0 .. 2 router 3 / 4 / 5 (the first three routed layers) on
//! rmsNorm(x1, pre_ff_norm of that layer); slot 3 router 3 on the dense layer's MLP input (none for -4). Their stamp is
//! the step's start (the host knows no dense layer's end: decode has no barrier there).
//!
//! The predictions are kept lazy through the forward and evaluated once after its logits (one more host wait a step, the
//! diagnostic's), so the stamps of the routed calls are the served schedule's. Compiled into profile builds and the
//! tests only (`expert.stream.recording`); a served build has no branch here.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const sdk_ext = @import("../../sdk_ext.zig");
const routes = sdk_ext.expert.routes;

pub const enabled = sdk_ext.expert.stream.recording;
pub const env = "BEAM_ROUTES";

const T = mlx.mlx_array;

/// One prediction kept through the forward: the route it belongs to (a routed layer's stream index, or a pseudo layer
/// below 0), its slot, the lazy scores [1, experts] f32.
const Kept = struct { layer: i16, slot: u8, scores: T };

const State = struct {
    a: std.mem.Allocator = undefined,
    rec: ?*routes.Recorder = null,
    dir: []const u8 = "",
    kept: std.ArrayList(Kept) = .empty,
    step_t0: u64 = 0,
    n_experts: u32 = 0,
    scratch: []f32 = &.{},
};
var state: if (enabled) State else void = if (enabled) .{} else {};

pub fn on() bool {
    if (comptime !enabled) return false;
    return state.rec != null;
}

/// Construction (after the source): the switch read once; the recorder installed on the source's stream.
pub fn arm(a: std.mem.Allocator, stream: anytype, n_layers: u32, n_experts: u32, top_k: u32) !void {
    if (comptime !enabled) return;
    const dir = routes.dirOf(std.c.getenv(env)) orelse return;
    const rec = try routes.Recorder.init(a, n_layers, n_experts, top_k, 1);
    errdefer rec.deinit();
    const scratch = try a.alloc(f32, n_experts);
    state = .{ .a = a, .rec = rec, .dir = dir, .n_experts = n_experts, .scratch = scratch };
    stream.recorder = rec;
    log.info("beam: {s} is set: each request's decode routes are written under {s} (route recorder v{d}, {d} prediction slots)\n", .{ env, dir, routes.version, routes.pred_slots });
}

pub fn disarm(stream: anytype) void {
    if (comptime !enabled) return;
    const rec = state.rec orelse return;
    stream.recorder = null;
    for (state.kept.items) |k| _ = mlx.mlx_array_free(k.scores);
    state.kept.deinit(state.a);
    state.a.free(state.scratch);
    rec.deinit();
    state = .{};
}

/// A request's start (its prompt pass).
pub fn begin(prompt_tokens: usize) void {
    if (comptime !enabled) return;
    const rec = state.rec orelse return;
    rec.begin(@intCast(prompt_tokens), 0);
}

/// A decode step's forward begins.
pub fn stepBegin() void {
    if (comptime !enabled) return;
    if (state.rec == null) return;
    state.step_t0 = routes.nowNs();
}

/// One prediction of the forward (a kept handle the state owns until `stepEnd`).
pub fn keep(layer: i16, slot: u8, scores: T) !void {
    if (comptime !enabled) return;
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, scores);
    state.kept.append(state.a, .{ .layer = layer, .slot = slot, .scores = out }) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
}

/// The step's forward ended (its logits evaluated): every kept prediction evaluated at once and recorded on its route
/// (a pseudo layer's route made here, stamped at the step's start), then the step. A failure is logged and the step's
/// predictions dropped; the request goes on.
pub fn stepEnd(g: anytype, token: u32, wall_ns: u64) void {
    if (comptime !enabled) return;
    const rec = state.rec orelse return;
    defer {
        for (state.kept.items) |k| _ = mlx.mlx_array_free(k.scores);
        state.kept.clearRetainingCapacity();
    }
    record(g, rec) catch |e| log.warn("beam: the route recorder dropped a step's predictions: {s}\n", .{@errorName(e)});
    rec.endStep(1, 0, 0, 1, wall_ns);
    rec.tokenOfLast(token);
}

fn record(g: anytype, rec: *routes.Recorder) !void {
    if (state.kept.items.len == 0) return;
    const xs = try state.a.alloc(T, state.kept.items.len);
    defer state.a.free(xs);
    for (state.kept.items, xs) |k, *x| x.* = k.scores;
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    var pseudo: [4]u32 = @splat(routes.none);
    for (state.kept.items) |k| {
        var idx: u32 = routes.none;
        if (k.layer >= 0) {
            idx = rec.routeOf(@intCast(k.layer));
        } else {
            const pi: usize = @intCast(-@as(i32, k.layer) - 1);
            if (pi < pseudo.len) {
                if (pseudo[pi] == routes.none) pseudo[pi] = rec.pseudo(k.layer, state.step_t0);
                idx = pseudo[pi];
            }
        }
        if (idx == routes.none) continue;
        if (mlx.mlx_array_size(k.scores) != state.n_experts or mlx.mlx_array_dtype(k.scores) != .float32) continue;
        rec.predict(idx, k.slot, try g.hostF32(k.scores, state.scratch));
    }
}

/// The request's end: the file, one line. A failure is logged; the request is not failed.
pub fn write(io: std.Io) void {
    if (comptime !enabled) return;
    const rec = state.rec orelse return;
    if (!rec.any()) return;
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/beam-routes-{d}-{d}.bin", .{ state.dir, std.c.getpid(), routes.nextFile() }) catch return;
    rec.write(state.a, io, path) catch |e| {
        log.warn("beam: the routes were not written to {s}: {s}\n", .{ path, @errorName(e) });
        return;
    };
    log.info("beam: routes written to {s} ({d} routes, {d} steps, {d} dropped)\n", .{ path, rec.routes.items.len, rec.steps.items.len, rec.dropped });
}
