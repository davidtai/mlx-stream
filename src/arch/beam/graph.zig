//! Beam's trunk on the GPU stream: the residents bound from the pack's shards by their original names (the release's
//! BF16, or the q8 tier's affine 8-bit projections: `residents_format`), the decode state (each sliding layer's window
//! ring, each full layer's slab), and the forward of rows at positions [start, start + rows) through every layer
//! (modeling_beam.py `BeamDecoderLayer`):
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
//! to its router; nothing else is evaluated before the logits.
//!
//! The residents route (`Linear`, `project`), fixed by the pack's `residents_format`: a BF16 projection is MLX's matmul
//! on the transposed weight (the control). A q8 projection is MLX's `quantized_matmul` (affine, 8 bits, group 64) in a
//! decode step: its qmv reads 1.0625 B a weight in place of 2 (the trunk's GEMVs 8.36 ms a step against BF16's 14.27 ms,
//! residents-q8 microbench on the M5 Ultra). A prompt layer reads each q8 projection as its BF16 weight, dequantized once
//! for the layer (`promptView`: MiMo's and glm5_next's prompt route, the stock GEMM over the dequantized weight): the
//! prompt pass keeps BF16's GEMMs (the 16K prompt's projections 2,376 ms against BF16's 2,395 ms; MLX's quantized GEMM
//! on every tile 2,474 ms).

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

/// A trunk projection [out, in] as the pack stores it (`beam.Config.residents_format`): the BF16 weight, or the q8 tier's
/// MLX affine 8-bit group 64 (U32 codes [out, in / 4], BF16 scales and biases [out, in / 64]).
pub const Linear = union(enum) {
    bf16: T,
    q8: Q8,
};
pub const Q8 = struct { w: T, scales: T, biases: T };

pub const Attention = struct { q: Linear, k: Linear, v: Linear, o: Linear, gate: Linear, gate_bias: T, q_norm: T, k_norm: T };
pub const Mlp = struct { gate: Linear, up: Linear, down: Linear };
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

/// A tensor of the pack the trunk reads; every other one (a routed expert's, the NVFP4 quantizer's scales) is dropped
/// from the weight map before the residents are read (module.zig) and never billed (bill.zig). Under the q8 tier a
/// projection's scales and biases are kept with its weight.
pub fn keptResident(c: *const beam.Config, name: []const u8) bool {
    if (c.residents_format == .affine8_g64) {
        inline for (.{ ".scales", ".biases" }) |tail| {
            if (std.mem.endsWith(u8, name, tail)) {
                var buf: [192]u8 = undefined;
                const w = std.fmt.bufPrint(&buf, "{s}.weight", .{name[0 .. name.len - tail.len]}) catch return false;
                return isProjection(c, w);
            }
        }
    }
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

/// Every layer's projections (the q8 tier's sites, with the dense layers' `dense_suffixes` and the head).
const attn_projections = [_][]const u8{ "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.gate_proj.weight" };
const layer_suffixes = attn_projections ++ [_][]const u8{
    "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    "self_attn.gate_bias",    "self_attn.q_norm.weight",         "self_attn.k_norm.weight",
};
const dense_suffixes = [_][]const u8{ "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight" };
const moe_suffixes = [_][]const u8{ "mlp.gate.weight", "mlp.gate.e_score_correction_bias" };

/// `name` is the weight of a trunk projection: the head, a layer's attention projection, a dense layer's MLP projection.
fn isProjection(c: *const beam.Config, name: []const u8) bool {
    if (std.mem.eql(u8, name, "lm_head.weight")) return true;
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return false;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
    const l = std.fmt.parseInt(u32, rest[0..dot], 10) catch return false;
    if (l >= c.n_layers) return false;
    const suffix = rest[dot + 1 ..];
    for (attn_projections) |s| if (std.mem.eql(u8, suffix, s)) return true;
    if (!c.isMoe(l)) for (dense_suffixes) |s| if (std.mem.eql(u8, suffix, s)) return true;
    return false;
}

/// The trunk's projections (the head, 5 a layer, 3 a dense layer).
pub fn projections(c: *const beam.Config) u64 {
    return 1 + @as(u64, c.n_layers) * attn_projections.len + @as(u64, c.first_k_dense) * dense_suffixes.len;
}

/// The tensors the trunk binds per layer kind (the bill's census): under the q8 tier each projection is three.
pub fn residentTensors(c: *const beam.Config) u64 {
    const bf16 = 3 + @as(u64, c.n_layers) * layer_suffixes.len + @as(u64, c.first_k_dense) * dense_suffixes.len + @as(u64, c.moeLayers()) * moe_suffixes.len;
    return if (c.residents_format == .affine8_g64) bf16 + 2 * projections(c) else bf16;
}

/// The residents. The router tables converted to F32 (a pack that stores them BF16) and the gate biases in the gate's
/// dtype are this struct's own arrays (`owned`), built once at bind; every other array is the host's weight map's.
pub const Weights = struct {
    embed: T,
    norm: T,
    head: Linear,
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
        const fmt = c.residents_format;
        var w: Weights = .{
            .embed = try need(map, "model.embed_tokens.weight", &.{ vocab, h }, diag),
            .norm = try need(map, "model.norm.weight", &.{h}, diag),
            .head = try linearOf(map, "lm_head", vocab, h, fmt, diag),
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
                fmt: beam.ResidentsFormat,
                fn get(self: @This(), comptime suffix: []const u8, shape: []const c_int) !T {
                    var nb: [128]u8 = undefined;
                    return need(self.map, std.fmt.bufPrint(&nb, "{s}" ++ suffix, .{self.pre}) catch unreachable, shape, self.diag);
                }
                /// The projection `proj` ([out, in]) in the pack's format.
                fn lin(self: @This(), comptime proj: []const u8, out: c_int, in: c_int) !Linear {
                    var nb: [128]u8 = undefined;
                    return linearOf(self.map, std.fmt.bufPrint(&nb, "{s}" ++ proj, .{self.pre}) catch unreachable, out, in, self.fmt, self.diag);
                }
            };
            const lg: L = .{ .pre = pre, .map = map, .diag = diag, .fmt = fmt };
            const full = c.isFull(l);
            // The gate bias in the dtype of the gate's output: the activations' BF16 under either format.
            const gate_bias_dtype: mlx.mlx_dtype = .bfloat16;
            lw.* = .{
                .input_norm = try lg.get("input_layernorm.weight", &.{h}),
                .post_attn_norm = try lg.get("post_attention_layernorm.weight", &.{h}),
                .pre_ff_norm = try lg.get("pre_feedforward_layernorm.weight", &.{h}),
                .post_ff_norm = try lg.get("post_feedforward_layernorm.weight", &.{h}),
                .attn = .{
                    .q = try lg.lin("self_attn.q_proj", hq, h),
                    .k = try lg.lin("self_attn.k_proj", hk, h),
                    .v = try lg.lin("self_attn.v_proj", hk, h),
                    .o = try lg.lin("self_attn.o_proj", h, hq),
                    .gate = try lg.lin("self_attn.gate_proj", hq, h),
                    // [heads] as [1, 1, heads, 1] in the gate's dtype (the reference adds it to the bf16 gate).
                    .gate_bias = try w.own(a, try headBias(s, try lg.get("self_attn.gate_bias", &.{@intCast(c.n_heads)}), gate_bias_dtype)),
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
                    .gate = try lg.lin("mlp.gate_proj", inter, h),
                    .up = try lg.lin("mlp.up_proj", inter, h),
                    .down = try lg.lin("mlp.down_proj", h, inter),
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

/// `need` with the dtype the format stores.
fn needOf(map: *const sdk.Weights, name: []const u8, shape: []const c_int, dt: mlx.mlx_dtype, diag: ?*sdk.Diag) !T {
    const x = try need(map, name, shape, diag);
    if (mlx.mlx_array_dtype(x) != dt) return refuse(diag, error.BeamResidentDtype, "{s} is {s}, the served form's {s}", .{ name, @tagName(mlx.mlx_array_dtype(x)), @tagName(dt) });
    return x;
}

/// The projection `prefix` [out, in] as `fmt` stores it: `bf16` the weight [out, in]; `affine8_g64` the codes U32
/// [out, in / 4], the scales and biases BF16 [out, in / 64] (refused by name otherwise).
fn linearOf(map: *const sdk.Weights, prefix: []const u8, out: c_int, in: c_int, fmt: beam.ResidentsFormat, diag: ?*sdk.Diag) !Linear {
    var nb: [160]u8 = undefined;
    if (fmt == .bf16) return .{ .bf16 = try need(map, std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}) catch unreachable, &.{ out, in }, diag) };
    const groups: c_int = @divExact(in, @as(c_int, beam.q8_group));
    const words: c_int = @divExact(in * @as(c_int, beam.q8_bits), 32);
    return .{ .q8 = .{
        .w = try needOf(map, std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}) catch unreachable, &.{ out, words }, .uint32, diag),
        .scales = try needOf(map, std.fmt.bufPrint(&nb, "{s}.scales", .{prefix}) catch unreachable, &.{ out, groups }, .bfloat16, diag),
        .biases = try needOf(map, std.fmt.bufPrint(&nb, "{s}.biases", .{prefix}) catch unreachable, &.{ out, groups }, .bfloat16, diag),
    } };
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

/// x [.., in] -> [.., out] through the projection as stored: `bf16` x @ w^T (MLX's matmul, the control); `q8` MLX's
/// `quantized_matmul` (affine, 8 bits, group 64, transposed weights: its qmv at decode's rows, its qmm on a wide call).
pub fn project(g: *G, l: Linear, x: T) !T {
    return switch (l) {
        .bf16 => |w| linear(g, x, w),
        .q8 => |q| g.quantizedMatmul(x, q.w, q.scales, q.biases, true, beam.q8_bits, beam.q8_group, .affine),
    };
}

/// A prompt layer's view of `lw`: each q8 projection of its attention and its dense MLP as the BF16 weight MLX's
/// `dequantize` writes (bf16(s x q + b)), a node of the layer's graph made here once: every tile of the layer reads it,
/// the layer's end (the forward's mark) releases it. MiMo's and glm5_next's prompt route (the stock GEMM over a
/// projection's BF16 weight, written once per prompt layer) with the forward's scope as its cache: one layer's set at a
/// time (bill.zig `promptViewBytes`). A BF16 projection is its own view.
pub fn promptView(g: *G, lw: *const Layer) !Layer {
    var v = lw.*;
    inline for (.{ "q", "k", "v", "o", "gate" }) |f| @field(v.attn, f) = try bf16View(g, @field(lw.attn, f));
    if (lw.dense) |d| v.dense = .{ .gate = try bf16View(g, d.gate), .up = try bf16View(g, d.up), .down = try bf16View(g, d.down) };
    return v;
}

fn bf16View(g: *G, l: Linear) !Linear {
    return switch (l) {
        .bf16 => l,
        .q8 => |q| .{ .bf16 = try g.dequantizeWith(q.w, q.scales, q.biases, beam.q8_bits, beam.q8_group, .affine) },
    };
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

/// `x` bf16 times an f32 constant, rounded once to bf16 (a bf16 tensor times a Python float in the reference).
fn scaleBf16(g: *G, x: T, v: f64) !T {
    return g.astype(try g.mul(try g.astype(x, .float32), try g.scalar(v, .float32)), .bfloat16);
}

/// The depth-scaled residual add of sublayer index `i`: h / sqrt(i) + r * sqrt((i - 1) / i), each product rounded to
/// bf16 (the reference's two scalar multiplies), then their bf16 sum.
pub fn depthAdd(g: *G, h: T, r: T, i: u32) !T {
    const fi: f64 = @floatFromInt(i);
    return g.add(try scaleBf16(g, h, 1.0 / @sqrt(fi)), try scaleBf16(g, r, @sqrt((fi - 1.0) / fi)));
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
    var q = try heads4(g, try rmsNorm(g, try g.reshape(try project(g, at.q, xn), &.{ n, heads, hd }), at.q_norm, c.rms_norm_eps), n, heads, hd);
    var k = try heads4(g, try rmsNorm(g, try g.reshape(try project(g, at.k, xn), &.{ n, kvh, hd }), at.k_norm, c.rms_norm_eps), n, kvh, hd);
    const v = try heads4(g, try project(g, at.v, xn), n, kvh, hd);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    var out: T = undefined;
    if (lw.full) {
        q = try g.astype(try g.mul(try g.astype(q, .float32), try ssmaxScale(g, pos, nu)), .bfloat16);
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
    const gate = try g.sigmoid(try g.add(try g.reshape(try project(g, at.gate, xn), &.{ 1, n, heads, hd }), at.gate_bias));
    const gated = try g.mul(o, gate);
    return project(g, at.o, try g.reshape(gated, &.{ n, heads * hd }));
}

/// The dense SwiGLU (f32 activation, `swiglu_f32`), then down.
pub fn mlp(g: *G, m: Mlp, x: T) !T {
    const act = try sdk_ext.quant.activation(G, g, .swiglu_f32, try project(g, m.gate, x), try project(g, m.up, x));
    return project(g, m.down, act);
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
    // [n, tg, 1] against every group id [1, 1, ng]: a group is kept when any kept index names it.
    const ids_row = try g.reshape(try g.arange(0, @floatFromInt(c.n_group), 1, .uint32), &.{ 1, 1, ng });
    const hit = try g.equal(try g.reshape(try g.astype(keep_groups, .uint32), &.{ n, tg, 1 }), ids_row);
    const kept = try g.greater(try g.sum(try g.astype(hit, .int32), 1, false), try g.scalar(0, .int32));
    const mask = try g.reshape(try g.broadcastTo(try g.reshape(kept, &.{ n, ng, 1 }), &.{ n, ng, eg }), &.{ n, e });
    const masked = try g.where(mask, choice, try g.scalar(-std.math.inf(f64), .float32));
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
/// router's own choice evaluated and recorded.
fn forcedRoute(g: *G, c: *const beam.Config, r: Router, x: T, f: *const ForcedRoutes, l: u32, start: u32) !Routing {
    const n: usize = @intCast(g.shapeOf(x).dim(0));
    const k: c_int = @intCast(c.n_experts_per_tok);
    const moe = l - c.first_k_dense;
    if (f.k != c.n_experts_per_tok or start + n > f.positions) return error.BeamForcedRoutes;
    const s = try routerScores(g, r, x);
    const own = try select(g, c, s.choice);
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
/// `cache`), the routed experts through `src`: the last row's logits f32 [1, vocab] (soft-capped), a kept handle the
/// caller frees; under `head == .all` every row's, f32 [rows, vocab]. `force` is `{}` when served, or the parity test's
/// `*const ForcedRoutes` (its own comptime instance: the served forward has no branch on it).
pub fn forward(g: *G, a: std.mem.Allocator, c: *const beam.Config, w: *const Weights, ids: []const u32, start: u32, cache: *Cache, src: anytype, comptime head: Head, force: anytype) !T {
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
    for (w.layers, 0..) |*stored, li| {
        const l: u32 = @intCast(li);
        const m = g.mark();
        defer g.resetTo(m);
        // A prompt layer reads its q8 projections as BF16 weights dequantized once here (`promptView`); a decode step
        // reads them as stored.
        const view = if (prompt) try promptView(g, stored) else stored.*;
        const lw = &view;
        if (prompt and lw.router != null) try postReadAhead(g, c, lw, l, x, src);
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
        const x1 = try depthAdd(g, try rmsNorm(g, attn, lw.post_attn_norm, c.rms_norm_eps), x, beam.Config.residualIndex(l, false));
        const h = try rmsNorm(g, x1, lw.pre_ff_norm, c.rms_norm_eps);
        const f = if (lw.dense) |d| (if (prompt) try mlpTiles(g, a, c, d, h, rows) else try mlp(g, d, h)) else blk: {
            const r = if (comptime @TypeOf(force) == void) try route(g, c, lw.router.?, h) else try forcedRoute(g, c, lw.router.?, h, force, l, start);
            if (prompt) try g.asyncEval(&.{ r.indices, r.weights });
            const routed = try g.adopt(try src.forward(l, h, r.indices, r.weights));
            break :blk try scaleBf16(g, routed, c.routed_scaling);
        };
        const x2 = try depthAdd(g, try rmsNorm(g, f, lw.post_ff_norm, c.rms_norm_eps), x1, beam.Config.residualIndex(l, true));
        const next = g.keep(x2);
        g.release(x);
        x = next;
        // MiMo's tail: a prompt layer's output evaluated before the next layer starts (bounding its graph), then the
        // pooled scratch's purge. A decode step's stays lazy: the next routing barrier evaluates it.
        if (prompt) {
            g.resetTo(m);
            for (outs[0..n_kept]) |o| g.release(o);
            n_kept = 0;
            try g.evalAll(&.{x});
            src.promptLayerEnd();
        }
    }
    cache.len = start + rows;
    if (head == .all) return g.keep(try logitsOf(g, c, w, x));
    const last = try g.slice(x, &.{ @intCast(rows - 1), 0 }, &.{ @intCast(rows), hidden }, &.{ 1, 1 });
    return g.keep(try logitsOf(g, c, w, last));
}

/// The head over `x` [r, hidden]: the final norm, lm_head (its output bf16), then the soft cap in f32:
/// cap * tanh(logits / cap).
pub fn logitsOf(g: *G, c: *const beam.Config, w: *const Weights, x: T) !T {
    const logits = try g.astype(try project(g, w.head, try rmsNorm(g, x, w.norm, c.rms_norm_eps)), .float32);
    const cap: f64 = c.final_logit_softcapping;
    return g.mul(try tanh(g, try g.mul(logits, try g.scalar(1.0 / cap, .float32))), try g.scalar(cap, .float32));
}

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
    try testing.expectEqual(@as(u64, 734), residentTensors(&c));
}

test "beam graph: the q8 tier keeps each projection's scales and biases, and nothing else's; 270 projections of three tensors" {
    const c: beam.Config = .{ .residents_format = .affine8_g64 };
    for ([_][]const u8{
        "lm_head.scales",                         "lm_head.biases",                            "model.layers.0.mlp.down_proj.scales",
        "model.layers.51.self_attn.gate_proj.biases", "model.layers.7.self_attn.k_proj.scales", "model.layers.7.self_attn.o_proj.weight",
    }) |n| try testing.expect(keptResident(&c, n));
    for ([_][]const u8{
        "model.embed_tokens.scales",             "model.layers.3.mlp.gate.scales",     "model.layers.3.mlp.gate_proj.scales",
        "model.layers.5.self_attn.q_norm.scales", "model.layers.52.self_attn.q_proj.biases", "model.layers.3.mlp.experts.0.gate_proj.scales",
    }) |n| try testing.expect(!keptResident(&c, n));
    try testing.expect(!keptResident(&beam.Config{}, "lm_head.scales"));
    try testing.expectEqual(@as(u64, 1 + 52 * 5 + 3 * 3), projections(&c));
    try testing.expectEqual(@as(u64, 734 + 2 * 270), residentTensors(&c));
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
