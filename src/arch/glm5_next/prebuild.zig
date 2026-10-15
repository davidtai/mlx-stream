//! glm5_next's decode pipeline pre-build, unconditional, at construction (the GPU stream, after every route is installed,
//! before the server takes a request): MLX builds a custom kernel's Metal library and pipeline at the first evaluation of
//! each instantiation (its template arguments, its inputs' dtypes and classes), so without this the decode lane's first
//! rounds build them inside the first request. Here each decode-lane instantiation of the kernels below is launched once
//! at every decode-lane width (1 .. `max_rows` rows) over zero rows and the bound weights, the batch evaluated and dropped:
//!   - MiMo's FP8 GEMV (`mimo_fp8_gemv`) at every distinct FP8 trunk site shape (fp8.zig `project`, the plan of each width),
//!     whichever route uses it: the stock projections, the KDA o_proj, the MLPs' down, the DSA sites;
//!   - the installed launch-fusion texts (fusion.zig `prebuild`);
//!   - the fused glue's router, top-8 and combine (glue.zig), the fused SwiGLU at both MLP widths (swiglu.zig).
//! Not a forward and not a self-check: no output is read, no request state (KV, KDA state, expert slots, the lane) is
//! touched; only MLX's library and pipeline caches gain entries (each instantiation's library is MLX's own, built by MLX's
//! own apply and eval). Exact by construction: no value of any request changes. Transient memory: the zero rows and the
//! outputs of one width at a time (< 4 MB), freed with MLX's cache before the next construction step.
//! The module adds the expert texts (experts.zig `prebuildExperts`, exl3_math.zig `Math.prebuild`): the exl3 family's
//! prompt waves at a full wave and a one-row tail, and its installed decode chain at every width (zero rows, slot row 0 of
//! the first routed layer's base bank, read only). The prompt pass's other texts are warm.zig's (prompt lever 3).
//! Not pre-built (their instantiations depend on runtime values or need request state): the hcPre chain (its deferred
//! input), the DSA decode texts (the key count), the KDA verify capture recurrence and the MTP draft chain's kernels.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const glm = @import("config.zig");
const graph = @import("graph.zig");
const fp8 = @import("fp8.zig");
const glue = @import("glue.zig");
const swiglu = @import("swiglu.zig");
const fusion = @import("fusion.zig");

const G = graph.G;
const T = graph.T;

/// The decode lane's widths (the MTP verify's rows at most).
pub const max_rows: u32 = swiglu.max_rows;

pub const Result = struct { gemv: u32 = 0, fusion: u32 = 0, glue: u32 = 0, swiglu: u32 = 0, experts: u32 = 0, ns: u64 = 0 };

/// The FP8 trunk sites of every layer (attention, then MLP), each projection once.
fn sitesOf(lw: *const graph.Layer, out: *[16]fp8.Linear) []const fp8.Linear {
    var n: usize = 0;
    switch (lw.attn) {
        .kda => |*w| for ([_]fp8.Linear{ w.q, w.k, w.v, w.f_a, w.g_a, w.b, w.o }) |l| {
            out[n] = l;
            n += 1;
        },
        .dsa => |*w| for ([_]fp8.Linear{ w.q_a, w.q_b, w.kv_a, w.o, w.idx_wq_b, w.idx_wk }) |l| {
            out[n] = l;
            n += 1;
        },
    }
    for ([_]?graph.Mlp{ lw.dense, lw.shared }) |m| if (m) |mlp| for ([_]fp8.Linear{ mlp.gate, mlp.up, mlp.down }) |l| {
        out[n] = l;
        n += 1;
    };
    return out[0..n];
}

pub fn run(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, c: *const glm.Config, w: *const graph.Weights, routes: graph.Routes) !Result {
    var r: Result = .{};
    if (!mlx.streamIsGpu(s)) return r;
    const t0 = std.Io.Timestamp.now(io, .awake);
    var g = try G.init(a, s);
    defer g.deinit();
    // The distinct FP8 site shapes (one bound projection each).
    var shapes: [32]fp8.Linear = undefined;
    var n_shapes: usize = 0;
    for (w.layers) |*lw| {
        var buf: [16]fp8.Linear = undefined;
        for (sitesOf(lw, &buf)) |l| switch (l) {
            .bf16 => {},
            .fp8 => |f| {
                for (shapes[0..n_shapes]) |x| {
                    if (x.fp8.n == f.n and x.fp8.k == f.k) break;
                } else {
                    if (n_shapes == shapes.len) return error.Glm5NextPrebuildSites;
                    shapes[n_shapes] = l;
                    n_shapes += 1;
                }
            },
        };
    }
    const h: c_int = @intCast(c.hidden_size);
    var m: u32 = 1;
    while (m <= max_rows) : (m += 1) {
        const mark = g.mark();
        defer g.resetTo(mark);
        var outs: std.ArrayList(T) = .empty;
        defer outs.deinit(a);
        const mi: c_int = @intCast(m);
        for (shapes[0..n_shapes]) |l| {
            try outs.append(a, try fp8.project(&g, l, try g.zeros(&.{ 1, mi, l.fp8.k }, .bfloat16)));
            r.gemv += 1;
        }
        r.fusion += try fusion.prebuild(&g, c, w.layers, m, &outs, a);
        if (routes.glue) |k| {
            const rt = for (w.layers) |*lw| {
                if (lw.router) |x| break x;
            } else return error.Glm5NextPrebuildRouter;
            const ro = try glue.route(&g, k, rt.w, rt.bias, try g.zeros(&.{ mi, h }, .bfloat16), m);
            try outs.appendSlice(a, &.{ ro.ids, ro.weights });
            try outs.append(a, try glue.combineShared(&g, k, try g.zeros(&.{ mi, @intCast(glue.top_k), h }, .bfloat16), try g.zeros(&.{ mi, @intCast(glue.top_k) }, .float32), try g.zeros(&.{ mi, h }, .bfloat16), m));
            r.glue += 3;
        }
        if (routes.swiglu) |k| for ([_]swiglu.Width{ .dense, .shared }) |wd| {
            const width: c_int = @intCast(k.widths[@backingInt(wd)]);
            const z = try g.zeros(&.{ mi, width }, .bfloat16);
            try outs.append(a, try swiglu.activate(&g, k, wd, z, z, m));
            r.swiglu += 1;
        };
        try g.evalAll(outs.items);
    }
    _ = mlx.mlx_clear_cache();
    r.ns = @intCast(t0.untilNow(io, .awake).nanoseconds);
    return r;
}

const testing = std.testing;

test "glm5_next prebuild: the sites of a KDA, a DSA and an MLP layer, each projection once (7, 6, 3)" {
    var route: fp8.Route = .{ .kernels = .{} };
    const Lin = struct {
        fn of(r: *fp8.Route, n: u32, k: u32) !fp8.Linear {
            return .{ .fp8 = try fp8.bind(r, .{ .ctx = null }, .{ .ctx = null }, n, k) };
        }
    };
    const c: glm.Config = .{};
    var kw: @import("kda.zig").Weights = undefined;
    inline for (.{ "q", "k", "v", "f_a", "g_a", "b" }) |f| @field(kw, f) = try Lin.of(&route, 128, 4096);
    kw.o = try Lin.of(&route, 4096, 8192);
    const mlp: graph.Mlp = .{ .gate = try Lin.of(&route, 2048, 4096), .up = try Lin.of(&route, 2048, 4096), .down = try Lin.of(&route, 4096, 2048) };
    const layer: graph.Layer = .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = kw }, .shared = mlp, .slot = 0 };
    var buf: [16]fp8.Linear = undefined;
    const sites = sitesOf(&layer, &buf);
    try testing.expectEqual(@as(usize, 10), sites.len);
    try testing.expectEqual(@as(c_int, 8192), sites[6].fp8.k);
    try testing.expectEqual(@as(c_int, 2048), sites[9].fp8.k);
    try testing.expectEqual(@as(u32, 4), max_rows);
    _ = c;
    try sdk.testing.expectNoDevice();
}
