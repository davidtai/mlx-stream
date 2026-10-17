//! Beam's trunk on the GPU stream: the residents bound from the pack's BF16 shards by their original names, the decode
//! state (each sliding layer's window ring, each full layer's slab), and the forward of rows at positions
//! [start, start + rows) through every layer (modeling_beam.py `BeamDecoderLayer`):
//!
//!   h = post_attention_layernorm(attention(input_layernorm(x)));        x = depth_add(h, x, 2 (l + 1))
//!   h = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)));  x = depth_add(h, x, 2 (l + 1) + 1)
//!
//! with depth_add(h, r, i) = h / sqrt(i) + r * sqrt((i - 1) / i). The attention: q / k / v projections, RMSNorm over each
//! head's 128 (q_norm, k_norm), RoPE (theta 10,000, rotate-half) on the sliding layers; no positions on the full layers
//! and the query scaled by (1 + (position + 1) / 2048) ^ 0.1 in f32; GQA attention (a 512-key window on the sliding
//! layers, causal on the full ones) through MLX's SDPA; the output gated per head by sigmoid(gate_proj(x) + gate_bias)
//! before o_proj. The MLP: a dense SwiGLU on the leading dense layers; on the rest the router (f32 logits on the f32
//! weight, sigmoid, the correction bias for the choice only, DeepSeek-V3's group top-k: the top-`topk_group` of
//! `n_group` groups by the sum of each group's top-2, then the top-8 of their experts; the weights the chosen experts'
//! sigmoid scores normalized to sum 1), the routed experts through MiMo's expert route (experts.zig), the weighted sum
//! times `routed_scaling_factor`. Every SwiGLU in f32 (`swiglu_f32`). The head: the final norm, the untied lm_head, the
//! soft cap 30 tanh(x / 30) in f32.
//!
//! A prompt call (at most `experts.call_rows` rows) runs layer by layer: P1's predicted seed posted before a routed
//! layer's attention (its router on the layer's input under its pre-feedforward norm), the attention in tiles of
//! `tile_rows` rows with at most `tiles_in_flight` committed (each sliding ring's append is at most one tile: the ring's
//! headroom), the dense MLP in tiles, the MoE over every row at once (the wide lane), the layer's output evaluated at
//! its end. A decode step is one row: its routed calls take the decode lane, whose routing barrier evaluates the layer up
//! to its router; nothing else is evaluated before the logits. A decode-lane call's router runs the decode glue's texts
//! where the construction installed them (glue.zig, `Routes.glue`); a prompt call's runs the stock ops.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const expert_policy = sdk_ext.expert.policy;
const ops_be = @import("../deepseek_v41/ops.zig");
const mimo = @import("../mimo_v2/config.zig");
const kv = @import("../mimo_v2/kv.zig");
const beam = @import("config.zig");
const experts = @import("experts.zig");
const glue = @import("glue.zig");
/// Profile builds only: MiMo's prompt-pass timeline's stage marks (a no-op outside a recorded prompt pass).
const ptl = @import("../mimo_v2/prefill_timeline.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// A prompt call's attention tile: the sliding rings' append bound (`mimo_v2/kv.zig` `ring_geometry`'s headroom).
pub const tile_rows: u32 = mimo.prefill_chunk;
pub const tiles_in_flight: u32 = 2;
/// A prompt call's dense MLP in tiles of this many rows, at most `tiles_in_flight` committed.
pub const dense_tile_rows: u32 = 2048;

comptime {
    if (kv.ring_geometry.headroom < tile_rows) @compileError("beam: an attention tile overruns the ring's headroom");
    if (experts.call_rows % tile_rows != 0) @compileError("beam: a prompt call is not whole attention tiles");
}

pub const Attention = struct { q: T, k: T, v: T, o: T, gate: T, gate_bias: T, q_norm: T, k_norm: T };
pub const Mlp = struct { gate: T, up: T, down: T };
/// The router: the F32 weight [experts, hidden] (the reference keeps it f32) and the F32 correction bias [experts].
pub const Router = struct { w: T, bias: T };

pub const Layer = struct {
    input_norm: T,
    post_attn_norm: T,
    pre_ff_norm: T,
    post_ff_norm: T,
    attn: Attention,
    full: bool,
    /// The layer's index among the full layers (its slab) or the sliding layers (its ring).
    slot: u32,
    dense: ?Mlp = null,
    router: ?Router = null,
};

/// A tensor of the pack the trunk reads; every other one (a routed expert's, the quantizer's scales) is dropped from the
/// weight map before the residents are read (module.zig) and never billed (bill.zig).
pub fn keptResident(c: *const beam.Config, name: []const u8) bool {
    for ([_][]const u8{ "lm_head.weight", "model.embed_tokens.weight", "model.norm.weight" }) |n|
        if (std.mem.eql(u8, name, n)) return true;
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return false;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
    const l = std.fmt.parseInt(u32, rest[0..dot], 10) catch return false;
    if (l >= c.n_layers) return false;
    const suffix = rest[dot + 1 ..];
    for (layer_suffixes) |s| if (std.mem.eql(u8, suffix, s)) return true;
    if (c.isMoe(l)) {
        for (moe_suffixes) |s| if (std.mem.eql(u8, suffix, s)) return true;
    } else for (dense_suffixes) |s| if (std.mem.eql(u8, suffix, s)) return true;
    return false;
}

const layer_suffixes = [_][]const u8{
    "input_layernorm.weight",    "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    "self_attn.q_proj.weight",   "self_attn.k_proj.weight",         "self_attn.v_proj.weight",          "self_attn.o_proj.weight",
    "self_attn.gate_proj.weight", "self_attn.gate_bias",            "self_attn.q_norm.weight",          "self_attn.k_norm.weight",
};
const dense_suffixes = [_][]const u8{ "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight" };
const moe_suffixes = [_][]const u8{ "mlp.gate.weight", "mlp.gate.e_score_correction_bias" };

/// The tensors the trunk binds per layer kind (the bill's census).
pub fn residentTensors(c: *const beam.Config) u64 {
    return 3 + @as(u64, c.n_layers) * layer_suffixes.len + @as(u64, c.first_k_dense) * dense_suffixes.len + @as(u64, c.moeLayers()) * moe_suffixes.len;
}

/// The residents. The router tables converted to F32 (a pack that stores them BF16) and the gate biases in the gate's
/// dtype are this struct's own arrays (`owned`), built once at bind; every other array is the host's weight map's.
pub const Weights = struct {
    embed: T,
    norm: T,
    head: T,
    layers: []Layer,
    owned: std.ArrayList(T) = .empty,

    pub fn deinit(w: *Weights, a: std.mem.Allocator) void {
        for (w.owned.items) |x| _ = mlx.mlx_array_free(x);
        w.owned.deinit(a);
        a.free(w.layers);
        w.* = undefined;
    }

    /// The owned arrays' bytes at most, as MLX allocates them: the routers' F32 copies (only when the pack stores them in
    /// another dtype; billed always) and the gate biases (`router_f32`, `trunk_f32_tables`).
    pub fn ownedBytes(c: *const beam.Config, page: u64) struct { tables: u64, routers: u64 } {
        return .{
            .tables = @as(u64, c.n_layers) * bufferBytes(@as(u64, c.n_heads) * 4, page),
            .routers = @as(u64, c.moeLayers()) * (bufferBytes(@as(u64, c.n_routed_experts) * c.hidden_size * 4, page) + bufferBytes(@as(u64, c.n_routed_experts) * 4, page)),
        };
    }

    /// The owned arrays' count at most (the bill's buffer census).
    pub fn ownedArrays(c: *const beam.Config) u64 {
        return c.n_layers + 2 * c.moeLayers();
    }

    /// Every resident by name and shape (refused by name otherwise: once, at construction).
    pub fn bind(a: std.mem.Allocator, s: mlx.mlx_stream, map: *const sdk.Weights, c: *const beam.Config, diag: ?*sdk.Diag) !Weights {
        const layers = try a.alloc(Layer, c.n_layers);
        errdefer a.free(layers);
        const h: c_int = @intCast(c.hidden_size);
        const vocab: c_int = @intCast(c.vocab_size);
        var w: Weights = .{
            .embed = try need(map, "model.embed_tokens.weight", &.{ vocab, h }, diag),
            .norm = try need(map, "model.norm.weight", &.{h}, diag),
            .head = try need(map, "lm_head.weight", &.{ vocab, h }, diag),
            .layers = layers,
        };
        errdefer {
            for (w.owned.items) |x| _ = mlx.mlx_array_free(x);
            w.owned.deinit(a);
        }
        const hq: c_int = @intCast(c.n_heads * c.head_dim);
        const hk: c_int = @intCast(c.n_kv_heads * c.head_dim);
        const hd: c_int = @intCast(c.head_dim);
        var n_full: u32 = 0;
        var n_sliding: u32 = 0;
        for (layers, 0..) |*lw, li| {
            const l: u32 = @intCast(li);
            var pb: [64]u8 = undefined;
            const pre = std.fmt.bufPrint(&pb, "model.layers.{d}.", .{l}) catch unreachable;
            const L = struct {
                pre: []const u8,
                map: *const sdk.Weights,
                diag: ?*sdk.Diag,
                fn get(self: @This(), comptime suffix: []const u8, shape: []const c_int) !T {
                    var nb: [128]u8 = undefined;
                    return need(self.map, std.fmt.bufPrint(&nb, "{s}" ++ suffix, .{self.pre}) catch unreachable, shape, self.diag);
                }
            };
            const lg: L = .{ .pre = pre, .map = map, .diag = diag };
            const full = c.isFull(l);
            const gate = try lg.get("self_attn.gate_proj.weight", &.{ hq, h });
            lw.* = .{
                .input_norm = try lg.get("input_layernorm.weight", &.{h}),
                .post_attn_norm = try lg.get("post_attention_layernorm.weight", &.{h}),
                .pre_ff_norm = try lg.get("pre_feedforward_layernorm.weight", &.{h}),
                .post_ff_norm = try lg.get("post_feedforward_layernorm.weight", &.{h}),
                .attn = .{
                    .q = try lg.get("self_attn.q_proj.weight", &.{ hq, h }),
                    .k = try lg.get("self_attn.k_proj.weight", &.{ hk, h }),
                    .v = try lg.get("self_attn.v_proj.weight", &.{ hk, h }),
                    .o = try lg.get("self_attn.o_proj.weight", &.{ h, hq }),
                    .gate = gate,
                    // [heads] as [1, 1, heads, 1] in the gate's dtype (the reference adds it to the bf16 gate).
                    .gate_bias = try w.own(a, try headBias(s, try lg.get("self_attn.gate_bias", &.{@intCast(c.n_heads)}), mlx.mlx_array_dtype(gate))),
                    .q_norm = try lg.get("self_attn.q_norm.weight", &.{hd}),
                    .k_norm = try lg.get("self_attn.k_norm.weight", &.{hd}),
                },
                .full = full,
                .slot = if (full) n_full else n_sliding,
            };
            if (full) n_full += 1 else n_sliding += 1;
            if (c.isMoe(l)) {
                const e: c_int = @intCast(c.n_routed_experts);
                lw.router = .{
                    .w = try w.f32Table(a, s, try lg.get("mlp.gate.weight", &.{ e, h })),
                    .bias = try w.f32Table(a, s, try lg.get("mlp.gate.e_score_correction_bias", &.{e})),
                };
            } else {
                const inter: c_int = @intCast(c.intermediate_size);
                lw.dense = .{
                    .gate = try lg.get("mlp.gate_proj.weight", &.{ inter, h }),
                    .up = try lg.get("mlp.up_proj.weight", &.{ inter, h }),
                    .down = try lg.get("mlp.down_proj.weight", &.{ h, inter }),
                };
            }
        }
        return w;
    }

    fn own(w: *Weights, a: std.mem.Allocator, x: T) !T {
        w.owned.append(a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    /// A router table as F32: the stored array when it is F32 (the reference's dtype), else an owned F32 copy.
    fn f32Table(w: *Weights, a: std.mem.Allocator, s: mlx.mlx_stream, x: T) !T {
        if (mlx.mlx_array_dtype(x) == .float32) return x;
        return w.own(a, try f32Copy(s, x));
    }

    /// The gates the expert scheduler's lookahead reads, one per model layer (a dense layer's empty): the router's
    /// stored F32 weight [experts, hidden] (route.zig `scoresStored`) and its correction bias.
    pub fn gates(w: *const Weights, a: std.mem.Allocator) ![]experts.Gate {
        const out = try a.alloc(experts.Gate, w.layers.len);
        for (out, w.layers) |*gt, lw| gt.* = if (lw.router) |r| .{ .w = r.w, .bias = r.bias } else .{ .w = .{ .ctx = null }, .bias = .{ .ctx = null } };
        return out;
    }
};

fn refuse(diag: ?*sdk.Diag, err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    if (diag) |d| d.set("beam: " ++ fmt, args);
    return err;
}

/// An MLX buffer of `n` bytes as the Metal allocator takes it: past one page, rounded up to whole pages.
pub const bufferBytes = @import("../../sdk_ext/safetensors.zig").bufferBytes;

/// The resident `name`, its shape the served form's (refused by name otherwise: once, at construction).
fn need(map: *const sdk.Weights, name: []const u8, shape: []const c_int, diag: ?*sdk.Diag) !T {
    const x = map.get(name) orelse return refuse(diag, error.BeamResidentMissing, "the residents lack {s}", .{name});
    if (!std.mem.eql(c_int, mlx.getShape(x), shape)) return refuse(diag, error.BeamResidentShape, "{s} is {any}, the served form's {any}", .{ name, mlx.getShape(x), shape });
    return x;
}

/// `x` as f32 in a buffer of its own, evaluated: never the weight map's buffer.
fn f32Copy(s: mlx.mlx_stream, x: T) !T {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, f, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The gate bias [heads] as [1, 1, heads, 1] in `dt`, evaluated, in a buffer of its own.
fn headBias(s: mlx.mlx_stream, b: T, dt: mlx.mlx_dtype) !T {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, b, dt, s));
    const heads: c_int = mlx.getShape(b)[0];
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, c, &[_]c_int{ 1, 1, heads, 1 }, 4, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, r, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

// ── The decode state ──

/// A full layer's K and V slabs [1, kv heads, cap, head_dim] bf16, allocated once at the billed positions, written by
/// position (MLX's slice_update: in place when the slab is the only reference, as it is between steps).
pub const Slab = struct {
    k: ?T = null,
    v: ?T = null,

    fn deinit(self: *Slab, g: *G) void {
        if (self.k) |x| g.release(x);
        if (self.v) |x| g.release(x);
        self.* = .{};
    }

    /// Writes `n` rows at `pos` and returns the views [0, pos + n) (scoped to the caller's mark).
    fn append(self: *Slab, g: *G, k: T, v: T, pos: u32, n: u32) !struct { k: T, v: T } {
        const sk = g.shapeOf(self.k.?);
        const start = [_]i32{ 0, 0, @intCast(pos), 0 };
        const st = try g.hostArray(std.mem.sliceAsBytes(&start), &.{4}, .int32);
        const nk = try g.sliceUpdateDyn(self.k.?, k, st);
        const nv = try g.sliceUpdateDyn(self.v.?, v, st);
        const end: c_int = @intCast(pos + n);
        const view_k = try g.slice(nk, &.{ 0, 0, 0, 0 }, &.{ sk.d[0], sk.d[1], end, sk.d[3] }, &.{ 1, 1, 1, 1 });
        const view_v = try g.slice(nv, &.{ 0, 0, 0, 0 }, &.{ sk.d[0], sk.d[1], end, sk.d[3] }, &.{ 1, 1, 1, 1 });
        g.release(self.k.?);
        g.release(self.v.?);
        self.k = g.keep(nk);
        self.v = g.keep(nv);
        return .{ .k = view_k, .v = view_v };
    }
};

/// The decode state of one request: each sliding layer's ring (MiMo's BF16 window ring over the SDK lanes,
/// `mimo_v2/kv.zig` `SlidingRing`), each full layer's slab, bounded at `cap` positions.
pub const Cache = struct {
    a: std.mem.Allocator,
    rings: []kv.SlidingRing,
    slabs: []Slab,
    len: u32 = 0,
    cap: u32,

    pub fn init(a: std.mem.Allocator, g: *G, c: *const beam.Config, cap: u32) !Cache {
        const rings = try a.alloc(kv.SlidingRing, c.nSliding());
        errdefer a.free(rings);
        var n_rings: usize = 0;
        errdefer for (rings[0..n_rings]) |*r| r.deinit();
        for (rings) |*r| {
            r.* = try kv.SlidingRing.allocate(c.n_kv_heads, c.head_dim, c.head_dim, c.sliding_window, g.s);
            n_rings += 1;
        }
        const slabs = try a.alloc(Slab, c.nFull());
        errdefer a.free(slabs);
        for (slabs) |*sl| sl.* = .{};
        const self: Cache = .{ .a = a, .rings = rings, .slabs = slabs, .cap = cap };
        errdefer for (self.slabs) |*sl| sl.deinit(g);
        // Every lane at its bound (bf16 zeros), evaluated, once: a request's appends then write in place.
        const m = g.mark();
        defer g.resetTo(m);
        var news: std.ArrayList(T) = .empty;
        defer news.deinit(a);
        for (self.slabs) |*sl| {
            const shape = [_]c_int{ 1, @intCast(c.n_kv_heads), @intCast(cap), @intCast(c.head_dim) };
            sl.k = g.keep(try g.zeros(&shape, .bfloat16));
            sl.v = g.keep(try g.zeros(&shape, .bfloat16));
            try news.appendSlice(a, &.{ sl.k.?, sl.v.? });
        }
        for (self.rings) |*r| for ([_]*kv.SlidingRing.Lane{ &r.k, &r.v }) |lane| for (lane.bufs) |b| if (b) |x| try news.append(a, x);
        if (news.items.len > 0) try g.evalAll(news.items);
        return self;
    }

    pub fn deinit(self: *Cache, g: *G) void {
        for (self.rings) |*r| r.deinit();
        for (self.slabs) |*sl| sl.deinit(g);
        self.a.free(self.rings);
        self.a.free(self.slabs);
        self.* = undefined;
    }

    /// A new request from position 0: the rings rewound (their slots kept), the slabs' rows unread past the length.
    pub fn restart(self: *Cache) void {
        for (self.rings) |*r| r.rewind();
        self.len = 0;
    }
};

// ── The forward ──

pub fn linear(g: *G, x: T, w: T) !T {
    return g.matmul(x, try g.transpose(w));
}

pub fn rmsNorm(g: *G, x: T, w: T, eps: f32) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, g.s));
    return g.adopt(o);
}

fn rope(g: *G, x: T, dims: u32, base: f32, offset: u32) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&o, x, @intCast(dims), false, mlx.mlx_optional_float.some(base), 1.0, @intCast(offset), .{}, g.s));
    return g.adopt(o);
}

fn tanh(g: *G, x: T) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&o, x, g.s));
    return g.adopt(o);
}

fn sdpa(g: *G, q: T, k: T, v: T, scale: f32, mode: [*:0]const u8, mask: T) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, mask, .{}, false, g.s));
    return g.adopt(o);
}

/// `x` bf16 times the f32 array `s` (broadcast), rounded once to bf16 (a bf16 tensor times a Python float, or the f32
/// query scale, in the reference).
pub fn scaleBy(g: *G, x: T, s: T) !T {
    return g.astype(try g.mul(try g.astype(x, .float32), s), .bfloat16);
}

/// `scaleBy` as one compiled kernel (`ScaleBy`): the f32 copy and the f32 product in registers.
fn scaled(g: *G, c: *const beam.Config, x: T, s: T) !T {
    var o: [1]T = undefined;
    try g.tape(ScaleBy, c, &.{ x, s }, &o);
    return o[0];
}

const ScaleBy = struct {
    pub const region: ops_be.Region = .beam_scale;
    pub const Ctx = beam.Config;
    pub const n_out = 1;
    pub const shapeless = true;
    pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
        out[0] = try scaleBy(g, in[0], in[1]);
    }
};

/// The depth-scaled residual add of sublayer index `i`: h / sqrt(i) + r * sqrt((i - 1) / i), each product rounded to
/// bf16 (the reference's two scalar multiplies), then their bf16 sum. One compiled kernel (`DepthAdd`): the scales are
/// its inputs, so every sublayer runs the same kernel.
pub fn depthAdd(g: *G, c: *const beam.Config, h: T, r: T, i: u32) !T {
    const fi: f64 = @floatFromInt(i);
    var o: [1]T = undefined;
    try g.tape(DepthAdd, c, &.{ h, r, try g.scalar(1.0 / @sqrt(fi), .float32), try g.scalar(@sqrt((fi - 1.0) / fi), .float32) }, &o);
    return o[0];
}

/// `depthAdd`'s op chain (the compiled region's body, and the device test's control): bf16(f32(h) * a) + bf16(f32(r) *
/// b), `a` and `b` f32 scalars.
pub fn depthAddChain(g: *G, h: T, r: T, a: T, b: T) !T {
    return g.add(try scaleBy(g, h, a), try scaleBy(g, r, b));
}

/// `depthAdd` as one compiled kernel (MLX `compile`, shapeless): the chain's casts and products in registers, in the
/// chain's order; neither f32 product nor bf16 product is written.
const DepthAdd = struct {
    pub const region: ops_be.Region = .beam_depth_add;
    pub const Ctx = beam.Config;
    pub const n_out = 1;
    pub const shapeless = true;
    pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
        out[0] = try depthAddChain(g, in[0], in[1], in[2], in[3]);
    }
};

/// The trunk's compiled regions on `g` (once, at construction; `c` outlives `g`), each run once on one row so that MLX
/// builds its kernels at the load and not in the first request.
pub fn prepareFused(g: *G, c: *const beam.Config) !void {
    try g.prepareTape(DepthAdd, c);
    try g.prepareTape(ScaleBy, c);
    try g.prepareTape(Softcap, c);
    const m = g.mark();
    defer g.resetTo(m);
    const row = try g.zeros(&.{ 1, @intCast(c.hidden_size) }, .bfloat16);
    // The full layers' query: [1, heads, rows, head_dim] against [1, 1, rows, 1] (its own signature).
    const q = try g.zeros(&.{ 1, @intCast(c.n_heads), 1, @intCast(c.head_dim) }, .bfloat16);
    const logits = try g.zeros(&.{ 1, @intCast(c.vocab_size) }, .bfloat16);
    try g.evalAll(&.{ try depthAdd(g, c, row, row, 2), try scaled(g, c, row, try g.scalar(c.routed_scaling, .float32)), try scaled(g, c, q, try ssmaxScale(g, 0, 1)), try softcap(g, c, logits) });
}

/// The full layers' query scale at rows [pos, pos + n): (1 + (p + 1) / 2048) ^ 0.1, f32 [1, 1, n, 1].
fn ssmaxScale(g: *G, pos: u32, n: u32) !T {
    var buf: [tile_rows]f32 = undefined;
    std.debug.assert(n <= tile_rows);
    for (buf[0..n], 0..) |*x, i| {
        const p: f32 = @floatFromInt(pos + @as(u32, @intCast(i)));
        x.* = @exp(beam.ssmax_power * std.math.log1p((p + 1) / beam.ssmax_base));
    }
    return g.hostArray(std.mem.sliceAsBytes(buf[0..n]), &.{ 1, 1, @intCast(n), 1 }, .float32);
}

/// A sliding tile's mask over its ring view (rows at `pos`, keys from `first`): MiMo's banded causal rule.
fn windowMask(g: *G, a: std.mem.Allocator, pos: u32, n: u32, first: u32, keys: u32, window: u32) !T {
    const buf = try a.alloc(bool, @as(usize, n) * keys);
    defer a.free(buf);
    for (0..n) |r| for (0..keys) |k| {
        buf[r * keys + k] = mimo.attends(pos + r, first + k, window);
    };
    // MLX copies host data into its own buffer at the array's creation.
    return g.hostArray(std.mem.sliceAsBytes(buf), &.{ @intCast(n), @intCast(keys) }, .bool_);
}

/// One attention tile of layer `lw`: `xn` [n, hidden] (the input norm's rows) at positions [pos, pos + n) -> [n, hidden].
pub fn attention(g: *G, a: std.mem.Allocator, c: *const beam.Config, lw: *const Layer, xn: T, pos: u32, cache: *Cache) !T {
    const n: c_int = @intCast(g.shapeOf(xn).dim(0));
    const nu: u32 = @intCast(n);
    const heads: c_int = @intCast(c.n_heads);
    const kvh: c_int = @intCast(c.n_kv_heads);
    const hd: c_int = @intCast(c.head_dim);
    const at = &lw.attn;
    const heads4 = struct {
        fn f(gg: *G, x: T, rows: c_int, hh: c_int, d: c_int) !T {
            return gg.transposeAxes(try gg.reshape(x, &.{ 1, rows, hh, d }), &.{ 0, 2, 1, 3 });
        }
    }.f;
    // [1, heads, n, d]: each head normed over its d (q_norm / k_norm), then RoPE on a sliding layer.
    var q = try heads4(g, try rmsNorm(g, try g.reshape(try linear(g, xn, at.q), &.{ n, heads, hd }), at.q_norm, c.rms_norm_eps), n, heads, hd);
    var k = try heads4(g, try rmsNorm(g, try g.reshape(try linear(g, xn, at.k), &.{ n, kvh, hd }), at.k_norm, c.rms_norm_eps), n, kvh, hd);
    const v = try heads4(g, try linear(g, xn, at.v), n, kvh, hd);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    var out: T = undefined;
    if (lw.full) {
        q = try scaled(g, c, q, try ssmaxScale(g, pos, nu));
        const view = try cache.slabs[lw.slot].append(g, k, v, pos, nu);
        out = if (n > 1) try sdpa(g, q, view.k, view.v, scale, "causal", .{}) else try sdpa(g, q, view.k, view.v, scale, "", .{});
    } else {
        q = try rope(g, q, c.head_dim, c.rope_theta, pos);
        k = try rope(g, k, c.head_dim, c.rope_theta, pos);
        const view = try cache.rings[lw.slot].append(k, v, nu);
        const vk = try g.adopt(view.k);
        const vv = try g.adopt(view.v);
        if (n > 1) {
            const span = mimo.windowView(pos, nu, c.sliding_window);
            out = try sdpa(g, q, vk, vv, scale, "array", try windowMask(g, a, pos, nu, @intCast(span.first_key), span.key_count, c.sliding_window));
        } else out = try sdpa(g, q, vk, vv, scale, "", .{});
    }
    // [1, n, heads, d], gated per head: attn * sigmoid(gate_proj(x) + gate_bias).
    const o = try g.transposeAxes(out, &.{ 0, 2, 1, 3 });
    const gate = try g.sigmoid(try g.add(try g.reshape(try linear(g, xn, at.gate), &.{ 1, n, heads, hd }), at.gate_bias));
    const gated = try g.mul(o, gate);
    return linear(g, try g.reshape(gated, &.{ n, heads * hd }), at.o);
}

/// The dense SwiGLU (f32 activation, `swiglu_f32`), then down.
pub fn mlp(g: *G, m: Mlp, x: T) !T {
    const act = try sdk_ext.quant.activation(G, g, .swiglu_f32, try linear(g, x, m.gate), try linear(g, x, m.up));
    return linear(g, act, m.down);
}

/// A prompt call's dense MLP over `x` [rows, hidden] in tiles of `dense_tile_rows` rows: each tile committed as built
/// and the tile before it waited (at most `tiles_in_flight` tiles' intermediates live), the tiles joined.
fn mlpTiles(g: *G, a: std.mem.Allocator, c: *const beam.Config, m: Mlp, x: T, rows: u32) !T {
    if (rows <= dense_tile_rows) return mlp(g, m, x);
    const hidden: c_int = @intCast(c.hidden_size);
    const n_tiles: usize = (rows + dense_tile_rows - 1) / dense_tile_rows;
    const outs = try a.alloc(T, n_tiles);
    defer a.free(outs);
    var kept: usize = 0;
    defer for (outs[0..kept]) |o| g.release(o);
    for (outs, 0..) |*o, i| {
        const lo: c_int = @intCast(i * dense_tile_rows);
        const hi: c_int = @intCast(@min(rows, (i + 1) * dense_tile_rows));
        const ms = g.mark();
        o.* = g.keep(try mlp(g, m, try g.slice(x, &.{ lo, 0 }, &.{ hi, hidden }, &.{ 1, 1 })));
        kept += 1;
        g.resetTo(ms);
        try g.asyncEval(&.{o.*});
        if (i > 0) try g.evalAll(&.{outs[i - 1]});
    }
    return g.concat(outs, 0);
}

pub const Routing = struct { indices: T, weights: T };

/// The forward's construction-time routes: the decode glue's texts (glue.zig; null under `decode_glue: ops`, or before
/// the construction installed them).
pub const Routes = struct {
    glue: ?*const glue.Kernels = null,
};

/// A call's routing of `x` [n, hidden]: the decode glue's texts on a decode-lane call (`decode`) where installed, else
/// the stock ops (`route`). The two give the same ids and weights bit for bit (glue.zig).
fn routeOf(g: *G, c: *const beam.Config, routes: Routes, r: Router, x: T, decode: bool) !Routing {
    if (decode) if (routes.glue) |k| {
        const out = try glue.routeRows(g, k, r.w, r.bias, x, @intCast(g.shapeOf(x).dim(0)));
        return .{ .indices = out.ids, .weights = out.weights };
    };
    return route(g, c, r, x);
}

/// DeepSeek-V3's noaux_tc group top-k (modeling_beam.py `BeamTopKRouter`): f32 logits on the F32 weight, sigmoid; the
/// choice scores (+ the correction bias) in `n_group` groups, each group scored by the sum of its top-2; the
/// `topk_group` best groups kept (the others' choice scores -inf); the top-k of the kept choice scores; the weights the
/// chosen experts' sigmoid scores, normalized to sum 1 (+ 1e-20). x [n, hidden] -> ids u32 [n, k], weights f32 [n, k].
/// `routed_scaling_factor` multiplies the experts' weighted sum (the caller), not these weights.
pub fn route(g: *G, c: *const beam.Config, r: Router, x: T) !Routing {
    const s = try routerScores(g, r, x);
    const inds = try select(g, c, s.choice);
    return .{ .indices = inds, .weights = try weightsAt(g, c, s.scores, inds) };
}

const Scores = struct { scores: T, choice: T };

/// The sigmoid scores of the f32 logits and the choice scores (+ the correction bias), f32 [n, experts].
fn routerScores(g: *G, r: Router, x: T) !Scores {
    const scores = try g.sigmoid(try linear(g, try g.astype(x, .float32), r.w));
    return .{ .scores = scores, .choice = try g.add(scores, r.bias) };
}

/// The group top-k of the choice scores: u32 [n, k] (in no order).
fn select(g: *G, c: *const beam.Config, choice: T) !T {
    const n = g.shapeOf(choice).dim(0);
    const e: c_int = @intCast(c.n_routed_experts);
    const ng: c_int = @intCast(c.n_group);
    const eg: c_int = @intCast(c.groupExperts());
    const k: c_int = @intCast(c.n_experts_per_tok);
    const tg: c_int = @intCast(c.topk_group);
    const grouped = try g.reshape(choice, &.{ n, ng, eg });
    var top2 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_topk_axis(&top2, grouped, 2, -1, g.s));
    const group_scores = try g.sum(try g.adopt(top2), -1, false);
    const gpart = try g.argpartition(try g.neg(group_scores), tg - 1, -1);
    const keep_groups = try g.slice(gpart, &.{ 0, 0 }, &.{ n, tg }, &.{ 1, 1 });
    // [n, tg, 1] against every group id [1, 1, ng]: a group is kept when any kept index names it (MLX sums bools as
    // int32).
    const ids_row = try g.reshape(try g.arange(0, @floatFromInt(c.n_group), 1, .uint32), &.{ 1, 1, ng });
    const hit = try g.equal(try g.reshape(try g.astype(keep_groups, .uint32), &.{ n, tg, 1 }), ids_row);
    const kept = try g.greater(try g.sum(hit, 1, false), try g.scalar(0, .int32));
    // The other groups' choice scores -inf, on the grouped view (the where broadcasts the [n, ng, 1] mask itself: no
    // [n, experts] mask is built), then back to [n, experts] (a view of the where's output).
    const masked = try g.reshape(try g.where(try g.reshape(kept, &.{ n, ng, 1 }), grouped, try g.scalar(-std.math.inf(f64), .float32)), &.{ n, e });
    const part = try g.argpartition(try g.neg(masked), k - 1, -1);
    return g.astype(try g.slice(part, &.{ 0, 0 }, &.{ n, k }, &.{ 1, 1 }), .uint32);
}

/// The chosen experts' sigmoid scores, normalized to sum 1 (+ 1e-20) under `norm_topk_prob`: f32 [n, k].
fn weightsAt(g: *G, c: *const beam.Config, scores: T, inds: T) !T {
    const w = try g.takeAlongAxis(scores, inds, -1);
    if (!c.norm_topk_prob) return w;
    return g.div(w, try g.add(try g.sum(w, -1, true), try g.scalar(1e-20, .float32)));
}

/// The parity test's routes (parity.zig): every routed layer's expert ids by position, the reference's, taken in place
/// of the router's choice; the router's own choice at each of them recorded beside (one read back per routed layer: a
/// test's forward only). Layout of both: [routed layer][position][k].
pub const ForcedRoutes = struct {
    ids: []const u32,
    chosen: []u32,
    positions: usize,
    k: usize,

    fn at(f: *const ForcedRoutes, moe: usize, pos: usize) usize {
        return (moe * f.positions + pos) * f.k;
    }
};

/// Routed layer `l`'s routing of rows [start, start + n) under forced routes: the forced ids, the weights at them; the
/// router's own choice (the call's route: `routeOf`) evaluated and recorded.
fn forcedRoute(g: *G, c: *const beam.Config, routes: Routes, r: Router, x: T, f: *const ForcedRoutes, l: u32, start: u32, decode: bool) !Routing {
    const n: usize = @intCast(g.shapeOf(x).dim(0));
    const k: c_int = @intCast(c.n_experts_per_tok);
    const moe = l - c.first_k_dense;
    if (f.k != c.n_experts_per_tok or start + n > f.positions) return error.BeamForcedRoutes;
    const s = try routerScores(g, r, x);
    const own = (try routeOf(g, c, routes, r, x, decode)).indices;
    try g.evalAll(&.{own});
    _ = try g.readU32(own, f.chosen[f.at(moe, start)..][0 .. n * f.k]);
    const inds = try g.hostArray(std.mem.sliceAsBytes(f.ids[f.at(moe, start)..][0 .. n * f.k]), &.{ @intCast(n), k }, .uint32);
    return .{ .indices = inds, .weights = try weightsAt(g, c, s.scores, inds) };
}

/// Each expert's routed rows in `ids` (u32 [rows, k]) against every expert id (`all_ids`, u32 [1, n]): int32 [n].
fn expertCounts(g: *G, ids: T, rows: c_int, k: c_int, all_ids: T) !T {
    return g.sum(try g.equal(try g.reshape(ids, &.{ rows * k, 1 }), all_ids), 0, false);
}

/// P1: routed layer `l`'s predicted experts (its router on the layer's input under its pre-feedforward norm, counted
/// per expert on the GPU, one small read back), ranked hottest first and posted to the stream, which reads the seed
/// into the layer's free persistent rows while the attention runs. The prediction only picks reads: token-exact.
fn postReadAhead(g: *G, c: *const beam.Config, lw: *const Layer, l: u32, x: T, src: anytype) !void {
    const m = g.mark();
    defer g.resetTo(m);
    const n = g.shapeOf(x).dim(0);
    const ne: c_int = @intCast(c.n_routed_experts);
    const r = try route(g, c, lw.router.?, try rmsNorm(g, x, lw.pre_ff_norm, c.rms_norm_eps));
    const all_ids = try g.reshape(try g.arange(0, @floatFromInt(c.n_routed_experts), 1, .uint32), &.{ 1, ne });
    const counts = try g.astype(try expertCounts(g, r.indices, n, @intCast(c.n_experts_per_tok), all_ids), .uint32);
    try g.evalAll(&.{counts});
    const host = try g.readU32(counts, src.ahead_counts[0..c.n_routed_experts]);
    try src.readAheadSeed(l, expert_policy.rankHottest(host, &src.ahead_ranked));
}

/// The head's rows: the last row's logits (served), or every row's (the parity test's).
pub const Head = enum { last, all };

/// One forward of `ids` at positions [start, start + ids.len) through every layer (each layer's state advanced in
/// `cache`), the routed experts through `src`, the construction's `routes`: the last row's logits f32 [1, vocab]
/// (soft-capped), a kept handle the caller frees; under `head == .all` every row's, f32 [rows, vocab]. `force` is `{}`
/// when served, or the parity test's `*const ForcedRoutes` (its own comptime instance: the served forward has no branch
/// on it).
pub fn forward(g: *G, a: std.mem.Allocator, c: *const beam.Config, w: *const Weights, routes: Routes, ids: []const u32, start: u32, cache: *Cache, src: anytype, comptime head: Head, force: anytype) !T {
    const rows: u32 = @intCast(ids.len);
    if (rows == 0 or rows > experts.call_rows) return error.BeamCallRows;
    if (start + rows > cache.cap) return error.ContextOverBill;
    const prompt = !src.decodeLaneAt(rows);
    const m0 = g.mark();
    defer g.resetTo(m0);
    const hidden: c_int = @intCast(c.hidden_size);
    var x: T = blk: {
        const idx = try g.hostArray(std.mem.sliceAsBytes(ids), &.{@intCast(rows)}, .uint32);
        break :blk g.keep(try g.take(w.embed, idx, 0));
    };
    defer g.release(x);
    const n_tiles: usize = (rows + tile_rows - 1) / tile_rows;
    const outs = try a.alloc(T, n_tiles);
    defer a.free(outs);
    for (w.layers, 0..) |*lw, li| {
        const l: u32 = @intCast(li);
        const m = g.mark();
        defer g.resetTo(m);
        if (comptime ptl.enabled) {
            ptl.layer(l);
            ptl.stage(.predict);
        }
        if (prompt and lw.router != null) try postReadAhead(g, c, lw, l, x, src);
        if (comptime ptl.enabled) ptl.stage(.attn);
        // The attention, tile by tile (the rings and slabs carry from one tile to the next); in a prompt call each tile
        // is committed as built and the tile before it waited, so at most `tiles_in_flight` run at once.
        const xn = try rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
        var n_kept: usize = 0;
        defer for (outs[0..n_kept]) |o| g.release(o);
        for (0..n_tiles) |i| {
            const lo: u32 = @intCast(i * tile_rows);
            const hi: u32 = @min(rows, lo + tile_rows);
            const ms = g.mark();
            const tile = if (n_tiles == 1) xn else try g.slice(xn, &.{ @intCast(lo), 0 }, &.{ @intCast(hi), hidden }, &.{ 1, 1 });
            outs[i] = g.keep(try attention(g, a, c, lw, tile, start + lo, cache));
            n_kept += 1;
            g.resetTo(ms);
            if (!prompt) continue;
            try g.asyncEval(&.{outs[i]});
            if (i > 0) try g.evalAll(&.{outs[i - 1]});
        }
        const attn = if (n_tiles == 1) outs[0] else try g.concat(outs[0..n_tiles], 0);
        if (comptime ptl.enabled) ptl.stage(if (lw.dense != null) .dense else .router);
        const x1 = try depthAdd(g, c, try rmsNorm(g, attn, lw.post_attn_norm, c.rms_norm_eps), x, beam.Config.residualIndex(l, false));
        const h = try rmsNorm(g, x1, lw.pre_ff_norm, c.rms_norm_eps);
        const f = if (lw.dense) |d| (if (prompt) try mlpTiles(g, a, c, d, h, rows) else try mlp(g, d, h)) else blk: {
            const r = if (comptime @TypeOf(force) == void) try routeOf(g, c, routes, lw.router.?, h, !prompt) else try forcedRoute(g, c, routes, lw.router.?, h, force, l, start, !prompt);
            if (prompt) try g.asyncEval(&.{ r.indices, r.weights });
            const routed = try g.adopt(try src.forward(l, h, r.indices, r.weights));
            break :blk try scaled(g, c, routed, try g.scalar(c.routed_scaling, .float32));
        };
        const x2 = try depthAdd(g, c, try rmsNorm(g, f, lw.post_ff_norm, c.rms_norm_eps), x1, beam.Config.residualIndex(l, true));
        const next = g.keep(x2);
        g.release(x);
        x = next;
        // MiMo's tail: a prompt layer's output evaluated before the next layer starts (bounding its graph), then the
        // pooled scratch's purge. A decode step's stays lazy: the next routing barrier evaluates it.
        if (prompt) {
            if (comptime ptl.enabled) ptl.stage(.tail);
            g.resetTo(m);
            for (outs[0..n_kept]) |o| g.release(o);
            n_kept = 0;
            try g.evalAll(&.{x});
            src.promptLayerEnd();
        }
    }
    if (comptime ptl.enabled) ptl.leaveLayers();
    cache.len = start + rows;
    if (head == .all) return g.keep(try logitsOf(g, c, w, x));
    const last = try g.slice(x, &.{ @intCast(rows - 1), 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
    return g.keep(try logitsOf(g, c, w, last));
}

/// The head over `x` [r, hidden]: the final norm, lm_head (bf16), then the soft cap in f32: cap * tanh(logits / cap), one
/// compiled kernel (`Softcap`).
pub fn logitsOf(g: *G, c: *const beam.Config, w: *const Weights, x: T) !T {
    return softcap(g, c, try linear(g, try rmsNorm(g, x, w.norm, c.rms_norm_eps), w.head));
}

fn softcap(g: *G, c: *const beam.Config, logits: T) !T {
    const cap: f64 = c.final_logit_softcapping;
    var o: [1]T = undefined;
    try g.tape(Softcap, c, &.{ logits, try g.scalar(1.0 / cap, .float32), try g.scalar(cap, .float32) }, &o);
    return o[0];
}

/// The soft cap's op chain (the region's body, the device test's control): tanh(f32(logits) * inv) * cap, f32.
pub fn softcapChain(g: *G, logits: T, inv: T, cap: T) !T {
    return g.mul(try tanh(g, try g.mul(try g.astype(logits, .float32), inv)), cap);
}

/// The soft cap as one compiled kernel (shapeless): the f32 copy, the two products and the tanh in registers; only the
/// f32 logits are written.
const Softcap = struct {
    pub const region: ops_be.Region = .beam_softcap;
    pub const Ctx = beam.Config;
    pub const n_out = 1;
    pub const shapeless = true;
    pub fn run(g: *G, _: *const Ctx, in: []const T, out: []T) !void {
        out[0] = try softcapChain(g, in[0], in[1], in[2]);
    }
};

const testing = std.testing;

test "beam graph: the kept residents are the trunk's (no routed expert, no quantizer scale)" {
    const c: beam.Config = .{};
    for ([_][]const u8{
        "lm_head.weight",                                  "model.embed_tokens.weight",                     "model.norm.weight",
        "model.layers.0.mlp.gate_proj.weight",             "model.layers.3.mlp.gate.weight",                "model.layers.51.mlp.gate.e_score_correction_bias",
        "model.layers.5.self_attn.gate_bias",              "model.layers.5.post_feedforward_layernorm.weight",
    }) |n| try testing.expect(keptResident(&c, n));
    for ([_][]const u8{
        "model.layers.3.mlp.experts.0.gate_proj.weight",       "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.3.mlp.gate_proj.weight",                 "model.layers.0.mlp.gate.weight",
        "model.layers.52.input_layernorm.weight",              "model.layers.x.input_layernorm.weight",
        "model.layers.3.mlp.experts.0.down_proj.input_scale",
    }) |n| try testing.expect(!keptResident(&c, n));
    // 3 + 52 x 12 + 3 x 3 + 49 x 2 tensors.
    try testing.expectEqual(@as(u64, 3 + 52 * 12 + 9 + 98), residentTensors(&c));
}

test "beam graph: the call geometry and the full layers' query scale" {
    try testing.expectEqual(@as(u32, 512), tile_rows);
    try testing.expect(experts.call_rows % tile_rows == 0);
    // (1 + (p + 1) / 2048) ^ 0.1 at p = 0 and p = 2047: 1.0000488 and 2^0.1.
    const p0 = @exp(beam.ssmax_power * std.math.log1p(@as(f32, 1) / beam.ssmax_base));
    try testing.expectApproxEqRel(@as(f32, 1.0000488), p0, 1e-6);
    const p1 = @exp(beam.ssmax_power * std.math.log1p(@as(f32, 2048) / beam.ssmax_base));
    try testing.expectApproxEqRel(std.math.pow(f32, 2, 0.1), p1, 1e-6);
}

// ── The fused chains on the device (BEAM_PARITY's environment: the box's suites under gpu-lock) ──

/// The device tests' gate: Beam's parity environment under the GPU lock (box-build.sh `suites -e`).
fn deviceTest() !void {
    _ = std.c.getenv("BEAM_PARITY") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
}

/// A bf16 array of `shape` from a fixed seed: values of a few magnitudes around `scale`, with signed zeros, the bf16
/// extremes of the served range and values where the sigmoid and tanh saturate.
fn testBf16(a: std.mem.Allocator, g: *G, shape: []const c_int, scale: f32, seed: u64) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const bits = try a.alloc(u16, n);
    defer a.free(bits);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const special = [_]f32{ 0.0, -0.0, 1e-30, -1e-30, 60.0, -60.0, 3.0e4, -3.0e4, 0.5, -0.5 };
    for (bits, 0..) |*b, i| {
        const v: f32 = if (i % 97 == 0) special[(i / 97) % special.len] else r.floatNorm(f32) * scale * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 3)));
        b.* = ops_be.bf16Bits(v);
    }
    return g.hostArray(std.mem.sliceAsBytes(bits), shape, .bfloat16);
}

/// `x` as a row-major array (MLX's `contiguous`: a copy only when `x` is a view of another layout).
fn rowMajor(g: *G, x: T) !T {
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_contiguous(&o, x, false, g.s)) catch |e| {
        _ = mlx.mlx_array_free(o);
        return e;
    };
    return g.adopt(o);
}

/// `want` and `got` evaluated and equal bit for bit (same dtype and shape, every byte): their element count; the first
/// differing element named otherwise.
fn sameBits(name: []const u8, g: *G, want: T, got: T) !usize {
    try g.evalAll(&.{ want, got });
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const w = try rowMajor(g, want);
    const o = try rowMajor(g, got);
    try g.evalAll(&.{ w, o });
    const bytes = mlx.mlx_array_size(w) * mlx.mlx_array_itemsize(w);
    const wb = (mlx.mlx_array_data_uint8(w) orelse return error.MlxNoData)[0..bytes];
    const ob = (mlx.mlx_array_data_uint8(o) orelse return error.MlxNoData)[0..bytes];
    if (std.mem.indexOfDiff(u8, wb, ob)) |i| {
        std.debug.print("BEAM FUSED {s}: element {d} differs\n", .{ name, i / mlx.mlx_array_itemsize(w) });
        return error.BeamFusedBits;
    }
    return mlx.mlx_array_size(w);
}

/// `sameBits`, and a line.
fn expectBits(name: []const u8, g: *G, want: T, got: T) !void {
    const n = try sameBits(name, g, want, got);
    std.debug.print("BEAM FUSED {s} {any}: {d} elements bit-equal\n", .{ name, mlx.getShape(want), n });
}

test "beam graph GPU: the depth-scaled add's one kernel is the op chain bit for bit (a step's row, a tile, a call)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: beam.Config = .{};
    try prepareFused(&g, &c);
    const h: c_int = @intCast(c.hidden_size);
    // Every sublayer's scales at one row; the first, a middle and the last sublayer at a tile and a whole call.
    for ([_]c_int{ 1, 512, 16_384 }, 0..) |rows, ri| {
        var i: u32 = 2;
        var sublayers: u32 = 0;
        var elements: usize = 0;
        while (i <= 2 * c.n_layers + 1) : (i += if (rows == 1) 1 else 51) {
            const m = g.mark();
            defer g.resetTo(m);
            const x = try testBf16(a, &g, &.{ rows, h }, 2.0, 300 + ri * 1000 + i);
            const r = try testBf16(a, &g, &.{ rows, h }, 2.0, 400 + ri * 1000 + i);
            const fi: f64 = @floatFromInt(i);
            const want = try depthAddChain(&g, x, r, try g.scalar(1.0 / @sqrt(fi), .float32), try g.scalar(@sqrt((fi - 1.0) / fi), .float32));
            elements += try sameBits("depth_add", &g, want, try depthAdd(&g, &c, x, r, i));
            sublayers += 1;
        }
        std.debug.print("BEAM FUSED depth_add {{ {d}, {d} }}: {d} sublayers, {d} elements bit-equal\n", .{ rows, h, sublayers, elements });
    }
}

test "beam graph GPU: the routed output's scale is one kernel, the op chain bit for bit (a step's row, a call)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: beam.Config = .{};
    try prepareFused(&g, &c);
    const h: c_int = @intCast(c.hidden_size);
    for ([_]c_int{ 1, 61, 16_384 }, 0..) |rows, i| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try testBf16(a, &g, &.{ rows, h }, 1.0, 500 + i);
        const s = try g.scalar(c.routed_scaling, .float32);
        try expectBits("routed_scale", &g, try scaleBy(&g, x, s), try scaled(&g, &c, x, s));
    }
}

test "beam graph GPU: the full layers' query scale is one kernel, the op chain bit for bit (a step, a tile)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: beam.Config = .{};
    try prepareFused(&g, &c);
    const heads: c_int = @intCast(c.n_heads);
    const hd: c_int = @intCast(c.head_dim);
    // The query as the attention hands it: the normed heads [rows, heads, hd] seen as [1, heads, rows, hd].
    for ([_]u32{ 1, 512 }, [_]u32{ 30_000, 1024 }, 0..) |rows, pos, i| {
        const m = g.mark();
        defer g.resetTo(m);
        const n: c_int = @intCast(rows);
        const x = try testBf16(a, &g, &.{ n, heads, hd }, 1.0, 600 + i);
        const q = try g.transposeAxes(try g.reshape(x, &.{ 1, n, heads, hd }), &.{ 0, 2, 1, 3 });
        const s = try ssmaxScale(&g, pos, rows);
        try expectBits("query_scale", &g, try scaleBy(&g, q, s), try scaled(&g, &c, q, s));
    }
}

test "beam graph GPU: the soft cap is one kernel, the op chain bit for bit (a step's logits, a parity call's)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: beam.Config = .{};
    try prepareFused(&g, &c);
    const vocab: c_int = @intCast(c.vocab_size);
    for ([_]c_int{ 1, 40 }, 0..) |rows, i| {
        const m = g.mark();
        defer g.resetTo(m);
        const l = try testBf16(a, &g, &.{ rows, vocab }, 12.0, 700 + i);
        const cap: f64 = c.final_logit_softcapping;
        const want = try softcapChain(&g, l, try g.scalar(1.0 / cap, .float32), try g.scalar(cap, .float32));
        try expectBits("softcap", &g, want, try softcap(&g, &c, l));
    }
}

test "beam graph GPU: the group top-k picks the experts its earlier form (the cast hits, the broadcast mask) picked" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: beam.Config = .{};
    const e: c_int = @intCast(c.n_routed_experts);
    const ng: c_int = @intCast(c.n_group);
    const eg: c_int = @intCast(c.groupExperts());
    const k: c_int = @intCast(c.n_experts_per_tok);
    const tg: c_int = @intCast(c.topk_group);
    for ([_]c_int{ 1, 16_384 }, 0..) |n, i| {
        const m = g.mark();
        defer g.resetTo(m);
        // Choice scores as the router makes them: sigmoid scores plus a small correction, f32 [n, experts].
        const z = try g.astype(try testBf16(a, &g, &.{ n, e }, 2.0, 800 + i), .float32);
        const choice = try g.add(try g.sigmoid(z), try g.astype(try testBf16(a, &g, &.{e}, 0.01, 900 + i), .float32));
        // The earlier form: the hits cast to int32 before the sum, the group mask broadcast to [n, experts] (a copy).
        const grouped = try g.reshape(choice, &.{ n, ng, eg });
        var top2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_topk_axis(&top2, grouped, 2, -1, g.s));
        const gpart = try g.argpartition(try g.neg(try g.sum(try g.adopt(top2), -1, false)), tg - 1, -1);
        const keep = try g.slice(gpart, &.{ 0, 0 }, &.{ n, tg }, &.{ 1, 1 });
        const ids_row = try g.reshape(try g.arange(0, @floatFromInt(c.n_group), 1, .uint32), &.{ 1, 1, ng });
        const hit = try g.equal(try g.reshape(keep, &.{ n, tg, 1 }), ids_row);
        const kept = try g.greater(try g.sum(try g.astype(hit, .int32), 1, false), try g.scalar(0, .int32));
        const mask = try g.reshape(try g.broadcastTo(try g.reshape(kept, &.{ n, ng, 1 }), &.{ n, ng, eg }), &.{ n, e });
        const masked = try g.where(mask, choice, try g.scalar(-std.math.inf(f64), .float32));
        const want = try g.slice(try g.argpartition(try g.neg(masked), k - 1, -1), &.{ 0, 0 }, &.{ n, k }, &.{ 1, 1 });
        try expectBits("group_topk", &g, want, try select(&g, &c, choice));
    }
}

test "beam graph GPU: the f32 SwiGLU's one kernel is the op chain bit for bit (routed rows, dense rows)" {
    try deviceTest();
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try G.init(a, stream);
    defer g.deinit();
    const c: beam.Config = .{};
    const inter: c_int = @intCast(c.moe_intermediate_size);
    const dense: c_int = @intCast(c.intermediate_size);
    // A decode wave's group, a prompt tile, a dense decode row, a dense prompt tile.
    for ([_][2]c_int{ .{ 8, inter }, .{ 1024, inter }, .{ 1, dense }, .{ 2048, dense } }, 0..) |shape, i| {
        const m = g.mark();
        defer g.resetTo(m);
        const gate = try testBf16(a, &g, &shape, 3.0, 100 + i);
        const up = try testBf16(a, &g, &shape, 3.0, 200 + i);
        try expectBits("swiglu_f32", &g, try sdk_ext.quant.swigluF32Chain(G, &g, gate, up), try sdk_ext.quant.activation(G, &g, .swiglu_f32, gate, up));
    }
}
