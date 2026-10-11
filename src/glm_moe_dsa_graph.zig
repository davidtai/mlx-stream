//! GLM-5.3's trunk in MLX ops (`deepseek_v41_ops.MlxOps`), op for op the reference `glm_moe_dsa.py`: the quantized
//! embedding, RMSNorm, MLA with the latent cache (the prompt path expands the latent per head through `embed_q` /
//! `unembed_out`; a one-row forward absorbs them into the query and the output), the interleaved rope, the lightning
//! indexer (fp32 scores, relu, the head weights at `n_heads^-0.5 * head_dim^-0.5`, the top `index_topk` keys; bypassed
//! while the keys fit it), its selection carried from each full layer to the shared layers after it within one forward,
//! the sigmoid router with its correction bias (fp32, top-k, normalized, scaled), the dense MLPs, the shared expert, the
//! final norm and the quantized head. The routed experts are the driver's (`glm_moe_dsa_experts`): this file hands
//! it the MoE input, the routed ids and their weights and adds what it returns. On the DSA kernels (`Routes.dsa`) a
//! prompt's attention runs in spans (`glm_moe_dsa.promptSpanRows`): the indexer's scores on the tensor units, each
//! row's top `index_topk` keys kept as ids, and the absorbed attention over exactly those keys (the reference's L == 1
//! form, its scores and softmax over the selected keys only). Elsewhere it runs the reference's prompt form in query
//! blocks whose score arrays stay under `glm_moe_dsa.score_budget_bytes`. The draft lane's verify (`Want.verify`) runs
//! the projections and the experts over all its rows and the attention one row at a time, row i over the keys up to its
//! own position with its own selection, as a serial step runs it: on MLX's CPU its rows' logits equal the serial steps'.
//! A projection may be dense (`QLinear.dense`, the MTP layer's BF16 tensors as published).

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const ops = @import("deepseek_v41_ops.zig");
const glm = @import("glm_moe_dsa.zig");
const cache_mod = @import("glm_moe_dsa_cache.zig");
const pt = @import("glm_moe_dsa_prefill_timers.zig");
const dsa = sdk.dsa;

pub const G = ops.MlxOps;
pub const T = G.T;
pub const Cache = cache_mod.Cache(G);

/// An affine projection (`nn.QuantizedLinear`; per head for `QuantizedMultiLinear`): its tensors and its bits. Bits 0:
/// a dense one (`nn.Linear`, mlx-lm's `MultiLinear`), the weight `[..., out, in]` as stored or a view of it, `s` / `b`
/// unread.
pub const QLinear = struct {
    w: T,
    s: T,
    b: T,
    bits: u32,

    pub fn dense(w: T) QLinear {
        return .{ .w = w, .s = w, .b = w, .bits = 0 };
    }
};
pub const Mlp = struct { gate: QLinear, up: QLinear, down: QLinear };
pub const Indexer = struct { wq_b: T, wk: T, k_norm_w: T, k_norm_b: T, weights_proj: T };
/// The router (`mlp.gate`) as stored, and its correction bias.
pub const Router = struct { w: T, bias: T };

pub const Layer = struct {
    input_norm: T,
    post_norm: T,
    q_a: QLinear,
    q_a_norm: T,
    q_b: QLinear,
    kv_a: QLinear,
    kv_a_norm: T,
    embed_q: QLinear,
    unembed_out: QLinear,
    o: QLinear,
    indexer: ?Indexer = null,
    dense: ?Mlp = null,
    router: ?Router = null,
    shared: ?Mlp = null,
    /// The routed layer's index in the bank (sparse layers).
    bank_layer: ?u32 = null,
};

/// The residents the trunk binds, handles into the host's weight map (never freed here).
pub const Weights = struct {
    embed: QLinear,
    norm: T,
    lm_head: QLinear,
    layers: []Layer,

    pub fn deinit(self: *Weights, a: std.mem.Allocator) void {
        a.free(self.layers);
        self.* = undefined;
    }

    /// Every resident by name (the shard headers were checked against the spec, `glm_moe_dsa.checkResidents`).
    pub fn bind(a: std.mem.Allocator, w: *const sdk.Weights, c: *const glm.Config, diag: ?*glm.Diag) !Weights {
        const layers = try a.alloc(Layer, c.n_layers);
        errdefer a.free(layers);
        var nb: [128]u8 = undefined;
        for (layers, 0..) |*lw, li| {
            const l: u32 = @intCast(li);
            const p = try std.fmt.bufPrint(&nb, "model.layers.{d}", .{l});
            var pb: [128]u8 = undefined;
            @memcpy(pb[0..p.len], p);
            const pre = pb[0..p.len];
            lw.* = .{
                .input_norm = try get(w, pre, "input_layernorm.weight", diag),
                .post_norm = try get(w, pre, "post_attention_layernorm.weight", diag),
                .q_a = try quant(w, c, pre, "self_attn.q_a_proj", diag),
                .q_a_norm = try get(w, pre, "self_attn.q_a_layernorm.weight", diag),
                .q_b = try quant(w, c, pre, "self_attn.q_b_proj", diag),
                .kv_a = try quant(w, c, pre, "self_attn.kv_a_proj_with_mqa", diag),
                .kv_a_norm = try get(w, pre, "self_attn.kv_a_layernorm.weight", diag),
                .embed_q = try quant(w, c, pre, "self_attn.embed_q", diag),
                .unembed_out = try quant(w, c, pre, "self_attn.unembed_out", diag),
                .o = try quant(w, c, pre, "self_attn.o_proj", diag),
            };
            if (c.isFull(l)) lw.indexer = .{
                .wq_b = try get(w, pre, "self_attn.indexer.wq_b.weight", diag),
                .wk = try get(w, pre, "self_attn.indexer.wk.weight", diag),
                .k_norm_w = try get(w, pre, "self_attn.indexer.k_norm.weight", diag),
                .k_norm_b = try get(w, pre, "self_attn.indexer.k_norm.bias", diag),
                .weights_proj = try get(w, pre, "self_attn.indexer.weights_proj.weight", diag),
            };
            if (c.isSparse(l)) {
                lw.router = .{ .w = try get(w, pre, "mlp.gate.weight", diag), .bias = try get(w, pre, "mlp.gate.e_score_correction_bias", diag) };
                lw.shared = .{ .gate = try quant(w, c, pre, "mlp.shared_experts.gate_proj", diag), .up = try quant(w, c, pre, "mlp.shared_experts.up_proj", diag), .down = try quant(w, c, pre, "mlp.shared_experts.down_proj", diag) };
                lw.bank_layer = c.bankLayer(l);
            } else {
                lw.dense = .{ .gate = try quant(w, c, pre, "mlp.gate_proj", diag), .up = try quant(w, c, pre, "mlp.up_proj", diag), .down = try quant(w, c, pre, "mlp.down_proj", diag) };
            }
        }
        return .{ .embed = try quant(w, c, "", "model.embed_tokens", diag), .norm = try get(w, "", "model.norm.weight", diag), .lm_head = try quant(w, c, "", "lm_head", diag), .layers = layers };
    }

    fn get(w: *const sdk.Weights, prefix: []const u8, rest: []const u8, diag: ?*glm.Diag) !T {
        var buf: [192]u8 = undefined;
        const name = if (prefix.len == 0) rest else try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, rest });
        return w.get(name) orelse {
            if (diag) |d| d.set("{s}: not in the loaded weights", .{name});
            return error.TensorMissing;
        };
    }

    fn quant(w: *const sdk.Weights, c: *const glm.Config, prefix: []const u8, rest: []const u8, diag: ?*glm.Diag) !QLinear {
        var buf: [192]u8 = undefined;
        const path = if (prefix.len == 0) rest else try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, rest });
        return .{ .w = try get(w, path, "weight", diag), .s = try get(w, path, "scales", diag), .b = try get(w, path, "biases", diag), .bits = c.quant.bitsOf(path) };
    }
};

// ── ops the trunk adds over the backend (each array joins the backend's scope) ──

fn adopt(g: *G, rc: c_int, r: mlx.mlx_array) !T {
    mlx.check(rc) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// `mx.fast.rms_norm(x, w, eps)` (`nn.RMSNorm`).
pub fn rmsNorm(g: *G, x: T, w: T, eps: f32) !T {
    var r = mlx.mlx_array_new();
    return adopt(g, mlx.mlx_fast_rms_norm(&r, x, w, eps, g.s), r);
}

/// `mx.fast.layer_norm(x, w, b, eps)` (`nn.LayerNorm`).
pub fn layerNorm(g: *G, x: T, w: T, b: T, eps: f32) !T {
    var r = mlx.mlx_array_new();
    return adopt(g, mlx.mlx_fast_layer_norm(&r, x, w, b, eps, g.s), r);
}

/// `nn.RoPE(dims, traditional=True, base)` at position `offset` over x's second-to-last axis (the rows), the first
/// `dims` features rotated in interleaved pairs.
pub fn rope(g: *G, x: T, dims: u32, base: f32, offset: u32) !T {
    var r = mlx.mlx_array_new();
    return adopt(g, mlx.mlx_fast_rope(&r, x, @intCast(dims), true, mlx.mlx_optional_float.some(base), 1.0, @intCast(offset), .{}, g.s), r);
}

/// `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)` with an additive array mask.
pub fn sdpa(g: *G, q: T, k: T, v: T, scale: f32, mask: T) !T {
    var r = mlx.mlx_array_new();
    return adopt(g, mlx.mlx_fast_scaled_dot_product_attention(&r, q, k, v, scale, "array", mask, .{}, false, g.s), r);
}

/// `mx.put_along_axis(a, idx, values, axis)`.
pub fn putAlongAxis(g: *G, a: T, idx: T, values: T, axis: c_int) !T {
    var r = mlx.mlx_array_new();
    return adopt(g, mlx.mlx_put_along_axis(&r, a, idx, values, axis, g.s), r);
}

pub fn qlinear(g: *G, x: T, q: QLinear) !T {
    if (q.bits == 0) return g.matmul(x, try swapLast(g, q.w));
    return g.quantizedMatmul(x, q.w, q.s, q.b, true, q.bits, glm.group_size, .affine);
}

/// The latent projections' other direction (`QuantizedMultiLinear(x, transpose=False)`).
fn qlinearT(g: *G, x: T, q: QLinear) !T {
    if (q.bits == 0) return g.matmul(x, q.w);
    return g.quantizedMatmul(x, q.w, q.s, q.b, false, q.bits, glm.group_size, .affine);
}

/// `nn.Linear` without bias, the weight as stored.
pub fn linear(g: *G, x: T, w: T) !T {
    return g.matmul(x, try g.transpose(w));
}

/// `x[..., lo:hi]` along the last axis.
fn lastSlice(g: *G, x: T, lo: c_int, hi: c_int) !T {
    const s = g.shapeOf(x);
    var start: [ops.max_dims]c_int = @splat(0);
    var stop: [ops.max_dims]c_int = undefined;
    const strides: [ops.max_dims]c_int = @splat(1);
    @memcpy(stop[0..s.n], s.slice());
    start[s.n - 1] = lo;
    stop[s.n - 1] = hi;
    return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
}

/// `x[..., lo:hi, :]` along the second-to-last axis.
pub fn rowSlice(g: *G, x: T, lo: c_int, hi: c_int) !T {
    const s = g.shapeOf(x);
    var start: [ops.max_dims]c_int = @splat(0);
    var stop: [ops.max_dims]c_int = undefined;
    const strides: [ops.max_dims]c_int = @splat(1);
    @memcpy(stop[0..s.n], s.slice());
    start[s.n - 2] = lo;
    stop[s.n - 2] = hi;
    return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
}

/// `x.swapaxes(-1, -2)`.
pub fn swapLast(g: *G, x: T) !T {
    const n = g.shapeOf(x).n;
    var axes: [ops.max_dims]c_int = undefined;
    for (0..n) |i| axes[i] = @intCast(i);
    axes[n - 1] = @intCast(n - 2);
    axes[n - 2] = @intCast(n - 1);
    return g.transposeAxes(x, axes[0..n]);
}

/// The causal mask of query rows `[q0, q0 + rows)` over keys `[0, keys)`: bool `[rows, keys]`.
fn causalMask(g: *G, q0: u32, rows: u32, keys: u32) !T {
    const qi = try g.reshape(try g.arange(@floatFromInt(q0), @floatFromInt(q0 + rows), 1, .int32), &.{ @intCast(rows), 1 });
    const ki = try g.reshape(try g.arange(0, @floatFromInt(keys), 1, .int32), &.{ 1, @intCast(keys) });
    return g.greaterEqual(qi, ki);
}

/// `mx.finfo(bfloat16).min`.
const bf16_min: f64 = -3.3895313892515355e38;

// ── the forward ──

/// One forward's query blocks (the same for every layer) and the selection each full layer leaves its shared layers:
/// per block its top-k key indices (kept: `[1, 1, rows, index_topk]` u32 in the reference's form, `[1, rows,
/// index_topk]` i32 on the DSA kernels), or null where every causal key is attended.
pub const Carry = struct {
    a: std.mem.Allocator,
    bounds: []u32,
    topk: []?T,

    /// `rows` query rows over at most `keys` keys: a prompt on the DSA kernels in spans of `glm.promptSpanRows`, else
    /// in blocks of `glm.queryBlockRows`.
    pub fn init(a: std.mem.Allocator, c: *const glm.Config, rows: u32, keys: u32, kernels: bool) !Carry {
        const step: u32 = @intCast(if (kernels and rows > 1) glm.promptSpanRows(c, rows, keys) else @min(rows, glm.queryBlockRows(@max(c.n_heads, c.index_n_heads), keys)));
        const n = (rows + step - 1) / step;
        const bounds = try a.alloc(u32, n + 1);
        errdefer a.free(bounds);
        for (bounds, 0..) |*b, i| b.* = @min(@as(u32, @intCast(i)) * step, rows);
        const topk = try a.alloc(?T, n);
        @memset(topk, null);
        return .{ .a = a, .bounds = bounds, .topk = topk };
    }

    /// One block per row (the verify's per-row attention).
    pub fn perRow(a: std.mem.Allocator, rows: u32) !Carry {
        const bounds = try a.alloc(u32, rows + 1);
        errdefer a.free(bounds);
        for (bounds, 0..) |*b, i| b.* = @intCast(i);
        const topk = try a.alloc(?T, rows);
        @memset(topk, null);
        return .{ .a = a, .bounds = bounds, .topk = topk };
    }

    pub fn deinit(self: *Carry, g: *G) void {
        for (self.topk) |t| if (t) |x| g.release(x);
        self.a.free(self.topk);
        self.a.free(self.bounds);
    }

    pub fn nBlocks(self: *const Carry) usize {
        return self.topk.len;
    }

    fn set(self: *Carry, g: *G, i: usize, x: ?T) void {
        if (self.topk[i]) |old| g.release(old);
        self.topk[i] = if (x) |v| g.keep(v) else null;
    }
};


/// The indexer of full layer `l` over the forward's rows: per query block the top `index_topk` keys by the fp32
/// score (relu'd per head, weighted by `weights_proj` at `n_heads^-0.5 * head_dim^-0.5`, summed over heads), masked
/// causally on a prompt; null where the block's keys fit `index_topk` (every causal key is attended). Kept in `carry`.
pub fn select(g: *G, c: *const glm.Config, ix: Indexer, x: T, iq: T, ik_all: T, start: u32, rows: u32, carry: *Carry) !void {
    const keys = start + rows;
    const topk = c.index_topk;
    if (keys <= topk) {
        for (0..carry.nBlocks()) |i| carry.set(g, i, null);
        return;
    }
    const ih: f64 = @floatFromInt(c.index_n_heads);
    const ihd: f64 = @floatFromInt(c.index_head_dim);
    const wts = try g.mul(try g.astype(try linear(g, x, ix.weights_proj), .float32), try g.scalar(std.math.pow(f64, ih, -0.5) * std.math.pow(f64, ihd, -0.5), .float32));
    const k32 = try g.astype(ik_all, .float32);
    for (0..carry.nBlocks()) |i| {
        const b0 = carry.bounds[i];
        const b1 = carry.bounds[i + 1];
        const nb = start + b1;
        if (nb <= topk) {
            carry.set(g, i, null);
            continue;
        }
        const m = g.mark();
        const qb = try g.astype(try rowSlice(g, iq, @intCast(b0), @intCast(b1)), .float32);
        var s = try g.matmul(qb, try swapLast(g, try rowSlice(g, k32, 0, @intCast(nb))));
        s = try g.maximum(s, try g.scalar(0, .float32));
        const wb = try g.slice(wts, &.{ @intCast(b0), 0 }, &.{ @intCast(b1), @intCast(c.index_n_heads) }, &.{ 1, 1 });
        const w4 = try g.reshape(try g.transposeAxes(wb, &.{ 1, 0 }), &.{ 1, @intCast(c.index_n_heads), @intCast(b1 - b0), 1 });
        s = try g.sum(try g.mul(s, w4), 1, true);
        if (rows > 1) s = try g.where(try causalMask(g, start + b0, b1 - b0, nb), s, try g.scalar(-std.math.inf(f64), .float32));
        const kth: c_int = @intCast(nb - topk);
        const idx = try lastSlice(g, try g.argpartition(s, kth, -1), kth, @intCast(nb));
        carry.set(g, i, idx);
        g.resetTo(m);
    }
}

/// `select` on the DSA kernels, a prompt's spans: the head-summed scores of each span's rows over the keys it reaches
/// on the tensor units (f32, the products of the same bf16 values accumulated in f32), the top `index_topk` kept as
/// i32 ids `[1, rows, index_topk]`; null where the span's keys fit `index_topk`.
fn selectSpans(g: *G, c: *const glm.Config, ix: Indexer, x: T, iq: T, ik_all: T, start: u32, carry: *Carry) !void {
    const topk = c.index_topk;
    const ih: c_int = @intCast(c.index_n_heads);
    const ihd: c_int = @intCast(c.index_head_dim);
    const ihf: f64 = @floatFromInt(c.index_n_heads);
    const ihdf: f64 = @floatFromInt(c.index_head_dim);
    const wts = try g.mul(try g.astype(try linear(g, x, ix.weights_proj), .float32), try g.scalar(std.math.pow(f64, ihf, -0.5) * std.math.pow(f64, ihdf, -0.5), .float32));
    // The kernel reads the queries row-major [1, rows, heads, head_dim] and the keys [1, keys, head_dim].
    const iq_rows = try g.transposeAxes(iq, &.{ 0, 2, 1, 3 });
    const ik = try g.reshape(ik_all, &.{ 1, g.shapeOf(ik_all).dim(2), ihd });
    for (0..carry.nBlocks()) |i| {
        const b0 = carry.bounds[i];
        const b1 = carry.bounds[i + 1];
        const nb = start + b1;
        if (nb <= topk) {
            carry.set(g, i, null);
            continue;
        }
        const m = g.mark();
        const r: c_int = @intCast(b1 - b0);
        const qb = try g.slice(iq_rows, &.{ 0, @intCast(b0), 0, 0 }, &.{ 1, @intCast(b1), ih, ihd }, &.{ 1, 1, 1, 1 });
        const wb = try g.reshape(try g.slice(wts, &.{ @intCast(b0), 0 }, &.{ @intCast(b1), ih }, &.{ 1, 1 }), &.{ 1, r, ih });
        const kb = try g.slice(ik, &.{ 0, 0, 0 }, &.{ 1, @intCast(nb), ihd }, &.{ 1, 1, 1 });
        const scores = try g.adopt((try dsa.indexerScores(g.s, qb, wb, kb, @intCast(start + b0), 1)) orelse return error.DsaKernelDeclined);
        const kth: c_int = @intCast(nb - topk);
        const idx = try lastSlice(g, try g.argpartition(scores, kth, -1), kth, @intCast(nb));
        carry.set(g, i, try g.astype(idx, .int32));
        g.resetTo(m);
    }
}

/// The prompt's attention on the DSA kernels, span by span: the query absorbed into the latent through `embed_q`, its
/// scores over the span's selected keys (the latent rows and their roped keys; every causal key where the span's keys
/// fit `index_topk`) and their softmax on the tensor units, the latent output back through `unembed_out`. `[1, heads,
/// rows, v_head_dim]`.
fn attendSpans(g: *G, c: *const glm.Config, lw: *const Layer, q_nope: T, q_pe: T, kv_all: T, pe_all: T, start: u32, scale: f32, carry: *Carry) !T {
    const kvr: c_int = @intCast(c.kv_lora_rank);
    const rp: c_int = @intCast(c.qk_rope_head_dim);
    var outs: std.ArrayList(T) = .empty;
    defer outs.deinit(carry.a);
    errdefer for (outs.items) |o| g.release(o);
    for (0..carry.nBlocks()) |i| {
        const b0 = carry.bounds[i];
        const b1 = carry.bounds[i + 1];
        const nb: c_int = @intCast(start + b1);
        const r: c_int = @intCast(b1 - b0);
        const m = g.mark();
        const q_lat = try qlinear(g, try rowSlice(g, q_nope, @intCast(b0), @intCast(b1)), lw.embed_q);
        const qp = try rowSlice(g, q_pe, @intCast(b0), @intCast(b1));
        // Where the span's keys fit the top-k every causal key is attended: the kernel drops a key past its query.
        const idx = carry.topk[i] orelse try g.broadcastTo(try g.reshape(try g.arange(0, @floatFromInt(nb), 1, .int32), &.{ 1, 1, nb }), &.{ 1, r, nb });
        const lat = try g.slice(kv_all, &.{ 0, 0, 0, 0 }, &.{ 1, 1, nb, kvr }, &.{ 1, 1, 1, 1 });
        const pe = try g.slice(pe_all, &.{ 0, 0, 0, 0 }, &.{ 1, 1, nb, rp }, &.{ 1, 1, 1, 1 });
        const o_lat = try g.adopt((try dsa.sparseLatentRope(g.s, q_lat, qp, lat, pe, idx, scale)) orelse return error.DsaKernelDeclined);
        try outs.append(carry.a, g.keep(try qlinear(g, o_lat, lw.unembed_out)));
        g.resetTo(m);
    }
    if (outs.items.len == 1) return g.adopt(outs.items[0]);
    const out = try g.concat(outs.items, 2);
    for (outs.items) |o| g.release(o);
    outs.clearRetainingCapacity();
    return out;
}

/// Layer `lw`'s query over its input `x [rows, hidden]` (normed) at positions `[start, start + rows)`: the q latent
/// (the indexer's query input), per head the nope part and the roped part, and on a full layer the indexer's roped
/// query.
pub const Query = struct { qr: T, q_nope: T, q_pe: T, iq: ?T = null };

pub fn queryOf(g: *G, c: *const glm.Config, lw: *const Layer, x: T, start: u32, rows: u32) !Query {
    const heads: c_int = @intCast(c.n_heads);
    const qhd: c_int = @intCast(c.qHeadDim());
    const nope: c_int = @intCast(c.qk_nope_head_dim);
    const n: c_int = @intCast(rows);
    const qr = try rmsNorm(g, try qlinear(g, x, lw.q_a), lw.q_a_norm, glm.latent_norm_eps);
    const q = try g.transposeAxes(try g.reshape(try qlinear(g, qr, lw.q_b), &.{ 1, n, heads, qhd }), &.{ 0, 2, 1, 3 });
    var out: Query = .{ .qr = qr, .q_nope = try lastSlice(g, q, 0, nope), .q_pe = try rope(g, try lastSlice(g, q, nope, qhd), c.qk_rope_head_dim, c.rope_theta, start) };
    if (lw.indexer) |ix| out.iq = try rope(g, try g.transposeAxes(try g.reshape(try linear(g, qr, ix.wq_b), &.{ 1, n, @intCast(c.index_n_heads), @intCast(c.index_head_dim) }), &.{ 0, 2, 1, 3 }), c.qk_rope_head_dim, c.rope_theta, start);
    return out;
}

/// Layer `lw`'s new KV rows from `x [rows, hidden]` (normed) at `[start, start + rows)`: the normalized latent
/// `[1, rows, kv_lora_rank]`, the roped key `[1, rows, rope]` and, on a full layer, the indexer's roped key
/// `[1, rows, index_head_dim]`, in the cache's shapes.
pub const Keys = struct { latent: T, k_pe: T, index: ?T = null };

pub fn keysOf(g: *G, c: *const glm.Config, lw: *const Layer, x: T, start: u32, rows: u32) !Keys {
    const rd: u32 = c.qk_rope_head_dim;
    const rp: c_int = @intCast(rd);
    const kvr: c_int = @intCast(c.kv_lora_rank);
    const n: c_int = @intCast(rows);
    const ckv = try qlinear(g, x, lw.kv_a);
    const latent = try rmsNorm(g, try lastSlice(g, ckv, 0, kvr), lw.kv_a_norm, glm.latent_norm_eps);
    const k_pe = try rope(g, try g.reshape(try lastSlice(g, ckv, kvr, kvr + rp), &.{ 1, 1, n, rp }), rd, c.rope_theta, start);
    var out: Keys = .{ .latent = try g.reshape(latent, &.{ 1, n, kvr }), .k_pe = try g.reshape(k_pe, &.{ 1, n, rp }) };
    if (lw.indexer) |ix| {
        const ihd: c_int = @intCast(c.index_head_dim);
        const ik = try rope(g, try g.reshape(try layerNorm(g, try linear(g, x, ix.wk), ix.k_norm_w, ix.k_norm_b, glm.index_norm_eps), &.{ 1, 1, n, ihd }), rd, c.rope_theta, start);
        out.index = try g.reshape(ik, &.{ 1, n, ihd });
    }
    return out;
}

/// One query row's absorbed attention (the reference's L == 1 form): the query through embed_q into the latent over
/// the keys `kv` / `pe` (`[1, 1, n, ·]`) or the `topk` rows of them, one shared key and value head, the output back
/// through unembed_out: `[1, heads, 1, v_head_dim]`.
pub fn absorbedRow(g: *G, c: *const glm.Config, lw: *const Layer, q_nope: T, q_pe: T, kv: T, pe: T, topk: ?T) !T {
    const kvr: c_int = @intCast(c.kv_lora_rank);
    const rp: c_int = @intCast(c.qk_rope_head_dim);
    const scale64: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(c.qHeadDim())));
    var kv_sel = kv;
    var pe_sel = pe;
    if (topk) |t| {
        const kk: c_int = @intCast(c.index_topk);
        const idx = try g.reshape(t, &.{ 1, 1, kk, 1 });
        kv_sel = try g.takeAlongAxis(kv, try g.broadcastTo(idx, &.{ 1, 1, kk, kvr }), 2);
        pe_sel = try g.takeAlongAxis(pe, try g.broadcastTo(idx, &.{ 1, 1, kk, rp }), 2);
    }
    const scores = try g.matmul(try g.mul(q_pe, try g.scalar(scale64, g.dtypeOf(q_pe))), try swapLast(g, pe_sel));
    const qa = try qlinear(g, q_nope, lw.embed_q);
    return qlinear(g, try sdpa(g, qa, kv_sel, kv_sel, @floatCast(scale64), scores), lw.unembed_out);
}

/// How a forward of several rows attends: `.auto` (one row absorbed, several in the prompt form), or `.per_row` (each
/// row absorbed over the keys up to its own position, as serial steps run it: the draft lane's verify).
pub const Form = enum { auto, per_row };

/// Layer `l`'s attention over its input `x [rows, hidden]` (normed) at positions `[start, start + rows)`: the new KV
/// appended, the indexer's selection made (full layers) or read from `carry` (shared), the output `[rows, hidden]`.
/// `.per_row` takes a `Carry.perRow` (one block per row). A prompt (`.auto`) on the DSA kernels (`kernels`) runs
/// `selectSpans` and `attendSpans`.
pub fn attention(g: *G, c: *const glm.Config, lw: *const Layer, l: u32, x: T, start: u32, rows: u32, cache: *Cache, carry: *Carry, form: Form, kernels: bool) !T {
    const heads: c_int = @intCast(c.n_heads);
    const qhd: c_int = @intCast(c.qHeadDim());
    const n: c_int = @intCast(rows);
    const q = try queryOf(g, c, lw, x, start, rows);
    const k = try keysOf(g, c, lw, x, start, rows);
    try cache.append(g, l, k.latent, k.k_pe, k.index);
    const kv_all = try cache.latentView(g, l);
    const pe_all = try cache.ropeView(g, l);
    if (pt.enabled and rows > 1) {
        var bufs: [3]T = undefined;
        _ = try pt.phase(g, cache.buffers(l, &bufs), .proj);
        _ = try pt.phase(g, &.{ q.q_nope, q.q_pe, q.qr }, .proj);
    }
    const spans = kernels and form == .auto and rows > 1;
    if (lw.indexer) |ix| {
        const ik_all = try cache.indexView(g, l);
        if (spans) try selectSpans(g, c, ix, x, q.iq.?, ik_all, start, carry) else try select(g, c, ix, x, q.iq.?, ik_all, start, rows, carry);
    }
    if (pt.enabled and rows > 1) {
        var sel: std.ArrayList(T) = .empty;
        defer sel.deinit(carry.a);
        for (carry.topk) |t| if (t) |x_| try sel.append(carry.a, x_);
        _ = try pt.phase(g, sel.items, .index);
    }
    var out: T = undefined;
    if (spans) {
        out = try attendSpans(g, c, lw, q.q_nope, q.q_pe, kv_all, pe_all, start, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(qhd)))), carry);
    } else if (rows == 1) {
        out = try absorbedRow(g, c, lw, q.q_nope, q.q_pe, kv_all, pe_all, carry.topk[0]);
    } else if (form == .per_row) {
        var outs: std.ArrayList(T) = .empty;
        defer outs.deinit(carry.a);
        for (0..rows) |i| {
            const r: c_int = @intCast(i);
            const nk: c_int = @intCast(start + i + 1);
            try outs.append(carry.a, try absorbedRow(g, c, lw, try rowSlice(g, q.q_nope, r, r + 1), try rowSlice(g, q.q_pe, r, r + 1), try rowSlice(g, kv_all, 0, nk), try rowSlice(g, pe_all, 0, nk), carry.topk[i]));
        }
        out = try g.concat(outs.items, 2);
    } else {
        // The prompt form: the latent expanded per head (k through embed_q, v through unembed_out), in query blocks.
        const scale64: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(qhd)));
        const scale: f32 = @floatCast(scale64);
        const qs = try g.scalar(scale64, g.dtypeOf(q.q_pe));
        const k_all = try qlinearT(g, kv_all, lw.embed_q);
        const v_all = try qlinear(g, kv_all, lw.unembed_out);
        var outs: std.ArrayList(T) = .empty;
        defer outs.deinit(carry.a);
        errdefer for (outs.items) |o| g.release(o);
        for (0..carry.nBlocks()) |i| {
            const b0 = carry.bounds[i];
            const b1 = carry.bounds[i + 1];
            const nb = start + b1;
            const m = g.mark();
            const qn = try rowSlice(g, q.q_nope, @intCast(b0), @intCast(b1));
            const qp = try rowSlice(g, q.q_pe, @intCast(b0), @intCast(b1));
            var pe = try g.matmul(try g.mul(qp, qs), try swapLast(g, try rowSlice(g, pe_all, 0, @intCast(nb))));
            var mask = try causalMask(g, start + b0, b1 - b0, nb);
            if (carry.topk[i]) |t| {
                const sm = try putAlongAxis(g, try g.zeros(&.{ 1, 1, @intCast(b1 - b0), @intCast(nb) }, .bool_), t, try g.scalar(1, .bool_), -1);
                mask = try g.logicalAnd(sm, mask);
            }
            pe = try g.where(mask, pe, try g.scalar(bf16_min, g.dtypeOf(pe)));
            const o = try sdpa(g, qn, try rowSlice(g, k_all, 0, @intCast(nb)), try rowSlice(g, v_all, 0, @intCast(nb)), scale, pe);
            try outs.append(carry.a, g.keep(o));
            g.resetTo(m);
        }
        if (outs.items.len == 1) {
            out = try g.adopt(outs.items[0]);
        } else {
            out = try g.concat(outs.items, 2);
            for (outs.items) |o| g.release(o);
        }
        outs.clearRetainingCapacity();
    }
    if (pt.enabled and rows > 1) pt.chargeAttn(lw.indexer != null, try pt.phase(g, &.{out}, .attn));
    const o2 = try g.reshape(try g.transposeAxes(out, &.{ 0, 2, 1, 3 }), &.{ n, heads * @as(c_int, @intCast(c.v_head_dim)) });
    return qlinear(g, o2, lw.o);
}

/// `down(silu(gate(x)) * up(x))` (the dense MLP and the shared expert).
pub fn mlp(g: *G, m: Mlp, x: T) !T {
    return qlinear(g, try g.mul(try g.silu(try qlinear(g, x, m.gate)), try qlinear(g, x, m.up)), m.down);
}

/// One routed layer's routing: the top-k expert ids `[n, k]` (int32) and their weights `[n, k]` (f32).
pub const Routing = struct { indices: T, weights: T };

/// `group_expert_select` at n_group 1: fp32 logits `x @ w.T`, sigmoid, the correction bias for the choice, the
/// top-k by argpartition, the sigmoid scores of the chosen normalized (`norm_topk_prob`) and scaled.
pub fn route(g: *G, c: *const glm.Config, r: Router, x: T) !Routing {
    const k: c_int = @intCast(c.n_experts_per_tok);
    const logits = try g.matmul(try g.astype(x, .float32), try g.transpose(try g.astype(r.w, .float32)));
    const sig = try g.sigmoid(logits);
    const sc = try g.add(sig, r.bias);
    const inds = try lastSlice(g, try g.argpartition(try g.neg(sc), k - 1, -1), 0, k);
    var w = try g.takeAlongAxis(sig, inds, -1);
    if (c.norm_topk_prob) w = try g.div(w, try g.sum(w, -1, true));
    w = try g.mul(w, try g.scalar(c.routed_scaling_factor, .float32));
    return .{ .indices = try g.astype(inds, .int32), .weights = w };
}

/// The routing criterion of router `r` on `x` (sigmoid + bias, f32 `[n, n_experts]`): the lookahead's predictor of the
/// next routed layer.
pub fn routerScores(g: *G, r: Router, x: T) !T {
    const logits = try g.matmul(try g.astype(x, .float32), try g.transpose(try g.astype(r.w, .float32)));
    return g.astype(try g.add(try g.sigmoid(logits), r.bias), .float32);
}

/// The quantized embedding of `ids` (`nn.QuantizedEmbedding`): the rows' codes, scales and biases dequantized.
pub fn embed(g: *G, w: *const Weights, ids: []const u32) !T {
    const idx = try g.hostArray(std.mem.sliceAsBytes(ids), &.{@intCast(ids.len)}, .uint32);
    const e = w.embed;
    return g.dequantizeWith(try g.take(e.w, idx, 0), try g.take(e.s, idx, 0), try g.take(e.b, idx, 0), e.bits, glm.group_size, .affine);
}

/// The last row's logits `[1, vocab]`: the final norm and the quantized head.
pub fn head(g: *G, c: *const glm.Config, w: *const Weights, h: T) !T {
    const rows = g.shapeOf(h).dim(0);
    const last = try g.slice(h, &.{ rows - 1, 0 }, &.{ rows, @intCast(c.hidden_size) }, &.{ 1, 1 });
    return qlinear(g, try rmsNorm(g, last, w.norm, c.rms_norm_eps), w.lm_head);
}

/// The experts read ahead for routed layer `lw` (P1's predictor): the router on the layer's input before its
/// attention (the post-attention norm of the residual), each row's top-k counted, hottest first (ties by id).
pub fn predictSeed(g: *G, a: std.mem.Allocator, c: *const glm.Config, lw: *const Layer, h: T, rows: u32, out: *std.ArrayList(u16)) !void {
    const xp = try rmsNorm(g, h, lw.post_norm, c.rms_norm_eps);
    const rt = try route(g, c, lw.router.?, xp);
    const ids = try a.alloc(u16, rows * c.n_experts_per_tok);
    defer a.free(ids);
    _ = try g.hostIds(rt.indices, ids);
    const counts = try a.alloc(u32, c.n_routed_experts);
    defer a.free(counts);
    @memset(counts, 0);
    for (ids) |e| counts[e] += 1;
    out.clearRetainingCapacity();
    for (counts, 0..) |n, e| if (n > 0) try out.append(a, @intCast(e));
    std.sort.pdq(u16, out.items, @as([]const u32, counts), struct {
        fn lt(cnt: []const u32, p: u16, q: u16) bool {
            return if (cnt[p] != cnt[q]) cnt[p] > cnt[q] else p < q;
        }
    }.lt);
}

/// A prompt's attention runs on the DSA kernels (`Routes.dsa`): the model's dims fit them, the stream is a GPU's and
/// both kernels build and run there (a GPU without the tensor units refuses the build).
pub fn dsaKernelsRun(gpa: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config) !bool {
    if (!glm.dsaKernelsFit(c) or !mlx.streamIsGpu(s)) return false;
    var g = try G.init(gpa, s);
    defer g.deinit();
    const had_error = mlx.errorPending();
    runDsaProbe(&g, c) catch |e| {
        var buf: [512]u8 = undefined;
        const msg = if (had_error) null else mlx.takeError(&buf);
        sdk.log.info("glm_moe_dsa: the DSA kernels do not run on this device ({s}: {s}): the prompt's attention runs in the reference's form\n", .{ @errorName(e), msg orelse "" });
        return false;
    };
    return true;
}

fn runDsaProbe(g: *G, c: *const glm.Config) !void {
    const h: c_int = @intCast(c.n_heads);
    const ih: c_int = @intCast(c.index_n_heads);
    const q = try g.zeros(&.{ 1, h, 1, @intCast(c.kv_lora_rank) }, .bfloat16);
    const qp = try g.zeros(&.{ 1, h, 1, @intCast(c.qk_rope_head_dim) }, .bfloat16);
    const lat = try g.zeros(&.{ 1, 1, 1, @intCast(c.kv_lora_rank) }, .bfloat16);
    const pe = try g.zeros(&.{ 1, 1, 1, @intCast(c.qk_rope_head_dim) }, .bfloat16);
    const idx = try g.zeros(&.{ 1, 1, 1 }, .int32);
    const o = try g.adopt((try dsa.sparseLatentRope(g.s, q, qp, lat, pe, idx, 1)) orelse return error.DsaKernelDeclined);
    const iq = try g.zeros(&.{ 1, 1, ih, @intCast(c.index_head_dim) }, .bfloat16);
    const w = try g.zeros(&.{ 1, 1, ih }, .float32);
    const ik = try g.zeros(&.{ 1, 1, @intCast(c.index_head_dim) }, .bfloat16);
    const s = try g.adopt((try dsa.indexerScores(g.s, iq, w, ik, 0, 1)) orelse return error.DsaKernelDeclined);
    try g.evalAll(&.{ o, s });
}

/// The forward's construction-time routes.
pub const Routes = struct {
    /// P1: each routed layer of a wide prompt call reads its predicted experts ahead during its attention.
    read_ahead: bool = false,
    /// Decode calls hand the driver the next routed layer's scores (the stream's lookahead).
    lookahead: bool = false,
    /// The widest call the driver's decode lane takes (the stream's `max_route_ids`).
    max_route_ids: u32 = 48,
    /// A prompt's attention on the DSA kernels (the module's probe passed on this device).
    dsa: bool = false,
};

/// What a forward returns beyond the last row's logits.
pub const Want = struct {
    /// The draft lane's verify: every row's logits, the attention per row (`Form.per_row`), and the decode lane's
    /// routes (no per-layer evaluation, the lookahead's scores).
    verify: bool = false,
    /// Every row's final-normed hidden `[rows, hidden]` kept (the MTP layer's input).
    hidden: bool = false,
};

/// A forward's kept results: the logits (`[1, vocab]`, or `[rows, vocab]` under `Want.verify`) and the final-normed
/// hidden under `Want.hidden`; the caller frees both.
pub const Out = struct { logits: T, hidden: ?T = null };

/// One forward of `ids` at positions `[start, start + ids.len)` through every layer (each layer's KV appended to
/// `cache`), the routed experts through `ex` (`call`, `readAheadSeed`): the last row's logits `[1, vocab]`, a kept
/// handle the caller frees. A prompt forward evaluates each layer before the next (its waves freed per layer).
pub fn forward(g: *G, a: std.mem.Allocator, c: *const glm.Config, w: *const Weights, ids: []const u32, start: u32, cache: *Cache, ex: anytype, rt: Routes) !T {
    return (try forwardRows(g, a, c, w, ids, start, cache, ex, rt, .{})).logits;
}

/// `forward` with `want`'s results.
pub fn forwardRows(g: *G, a: std.mem.Allocator, c: *const glm.Config, w: *const Weights, ids: []const u32, start: u32, cache: *Cache, ex: anytype, rt: Routes, want: Want) !Out {
    const rows: u32 = @intCast(ids.len);
    const prompt = rows > 1 and !want.verify;
    var carry = if (want.verify) try Carry.perRow(a, rows) else try Carry.init(a, c, rows, start + rows, rt.dsa);
    defer carry.deinit(g);
    var seed: std.ArrayList(u16) = .empty;
    defer seed.deinit(a);
    const m0 = g.mark();
    defer g.resetTo(m0);
    var h = g.keep(try embed(g, w, ids));
    defer g.release(h);
    const wide = rows * c.n_experts_per_tok > rt.max_route_ids;
    for (w.layers, 0..) |*lw, li| {
        const l: u32 = @intCast(li);
        const m = g.mark();
        if (pt.enabled and prompt) pt.start();
        // The predictor runs only where the read-ahead has empty rows to fill.
        if (rt.read_ahead and wide and lw.bank_layer != null and ex.readAheadRoom(lw.bank_layer.?) > 0) {
            try predictSeed(g, a, c, lw, h, rows, &seed);
            try ex.readAheadSeed(lw.bank_layer.?, seed.items);
        }
        // A wide prompt call's misses read before its attention (`Experts.stageMisses`): at 8 rows per expert it routes
        // nearly every expert.
        if (prompt and wide and lw.bank_layer != null and rows * c.n_experts_per_tok >= 8 * c.n_routed_experts) try ex.stageMisses(lw.bank_layer.?);
        errdefer ex.dropStaged();
        if (pt.enabled and prompt) _ = try pt.phase(g, &.{}, .seed);
        const x = try rmsNorm(g, h, lw.input_norm, c.rms_norm_eps);
        const h1 = try g.add(h, try attention(g, c, lw, l, x, start, rows, cache, &carry, if (want.verify) .per_row else .auto, rt.dsa));
        if (pt.enabled and prompt) _ = try pt.phase(g, &.{h1}, .oproj);
        const x2 = try rmsNorm(g, h1, lw.post_norm, c.rms_norm_eps);
        const f = if (lw.dense) |d| try mlp(g, d, x2) else blk: {
            const r = try route(g, c, lw.router.?, x2);
            if (pt.enabled and prompt) _ = try pt.phase(g, &.{ x2, r.indices, r.weights }, .router);
            const next: ?T = if (rt.lookahead and !prompt and li + 1 < w.layers.len and w.layers[li + 1].router != null) try routerScores(g, w.layers[li + 1].router.?, x2) else null;
            // A decode-lane call takes the shared expert built first and runs it during its read wait; the sum is the same.
            const shared: ?T = if (!wide) try mlp(g, lw.shared.?, x2) else null;
            const routed = try ex.call(g, lw.bank_layer.?, x2, r.indices, r.weights, next, if (shared) |sh| &[_]T{sh} else &.{});
            if (pt.enabled and prompt) _ = try pt.phase(g, &.{routed}, .routed);
            break :blk try g.add(routed, shared orelse try mlp(g, lw.shared.?, x2));
        };
        if (pt.enabled and prompt) _ = try pt.phase(g, &.{f}, .mlp);
        const next_h = g.keep(try g.add(h1, f));
        if (prompt) {
            var bufs: [3]T = undefined;
            var evals: std.ArrayList(T) = .empty;
            defer evals.deinit(a);
            try evals.append(a, next_h);
            try evals.appendSlice(a, cache.buffers(l, &bufs));
            for (carry.topk) |t| if (t) |x_| try evals.append(a, x_);
            try g.evalAll(evals.items);
            if (pt.enabled) _ = try pt.phase(g, &.{}, .rest);
        }
        g.release(h);
        h = next_h;
        g.resetTo(m);
    }
    cache.commit(rows);
    if (!want.verify and !want.hidden) return .{ .logits = g.keep(try head(g, c, w, h)) };
    // Row by row the same as `head`: the final norm per row, the head over the rows wanted.
    const normed = try rmsNorm(g, h, w.norm, c.rms_norm_eps);
    const n: c_int = @intCast(rows);
    const lg = try qlinear(g, if (want.verify) normed else try g.slice(normed, &.{ n - 1, 0 }, &.{ n, @intCast(c.hidden_size) }, &.{ 1, 1 }), w.lm_head);
    return .{ .logits = g.keep(lg), .hidden = if (want.hidden) g.keep(normed) else null };
}

const testing = std.testing;

/// The kernels' dims on a small trunk: 64 heads, the 512-wide latent with a 64-wide roped key, 4 indexer heads of
/// 128, the top 64 keys; a full layer and a shared one (the routed third is never run), every projection 8-bit affine.
fn kernelDimsConfigJson(a: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model_type\":\"glm_moe_dsa\",\"attention_bias\":false,\"eos_token_id\":[1],\"first_k_dense_replace\":2,\"hidden_act\":\"silu\"," ++
        "\"hidden_size\":128,\"index_head_dim\":128,\"index_n_heads\":4,\"index_topk\":64,\"indexer_rope_interleave\":true,\"intermediate_size\":128,\"kv_lora_rank\":512," ++
        "\"max_position_embeddings\":4096,\"moe_intermediate_size\":64,\"moe_layer_freq\":1,\"n_group\":1,\"n_routed_experts\":16,\"n_shared_experts\":1,\"norm_topk_prob\":true," ++
        "\"num_attention_heads\":64,\"num_experts_per_tok\":8,\"num_hidden_layers\":3,\"num_key_value_heads\":64,\"q_lora_rank\":64,\"qk_nope_head_dim\":64,\"qk_rope_head_dim\":64," ++
        "\"rms_norm_eps\":1e-05,\"rope_interleave\":true,\"rope_parameters\":{{\"rope_theta\":10000,\"rope_type\":\"default\"}},\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\"," ++
        "\"tie_word_embeddings\":false,\"topk_group\":1,\"topk_method\":\"noaux_tc\",\"v_head_dim\":64,\"vocab_size\":256," ++
        "\"indexer_types\":[\"full\",\"shared\",\"shared\"],\"mlp_layer_types\":[\"dense\",\"dense\",\"sparse\"],\"quantization\":{{\"group_size\":64,\"bits\":8}}}}", .{});
}

const TestWeights = struct {
    rand: std.Random,

    fn dense(tw: TestWeights, g: *G, shape: []const c_int, sd: f32) !T {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        const v = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(v);
        for (v) |*x| x.* = tw.rand.floatNorm(f32) * sd;
        return g.astype(try g.hostArray(std.mem.sliceAsBytes(v), shape, .float32), .bfloat16);
    }

    fn quant(tw: TestWeights, g: *G, shape: []const c_int, sd: f32) !QLinear {
        const w = try tw.dense(g, shape, sd);
        var vec = mlx.mlx_vector_array{ .ctx = null };
        try mlx.check(mlx.mlx_quantize(&vec, w, mlx.mlx_optional_int.some(glm.group_size), mlx.mlx_optional_int.some(8), "affine", .{}, g.s));
        defer _ = mlx.mlx_vector_array_free(vec);
        var parts: [3]T = undefined;
        for (&parts, 0..) |*p, i| {
            p.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(p, vec, i));
            p.* = try g.adopt(p.*);
        }
        return .{ .w = parts[0], .s = parts[1], .b = parts[2], .bits = 8 };
    }
};

test "glm attention: a prompt on the DSA kernels selects the reference's keys and matches its prompt form" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const text = try kernelDimsConfigJson(a);
    defer a.free(text);
    var c = try glm.Config.parse(a, text, null, null);
    defer c.deinit(a);
    try testing.expect(glm.dsaKernelsFit(&c));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!try dsaKernelsRun(a, s, &c)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(53);
    const tw: TestWeights = .{ .rand = prng.random() };
    const h: c_int = @intCast(c.n_heads);
    const hid: c_int = @intCast(c.hidden_size);
    const ql: c_int = @intCast(c.q_lora_rank);
    const kvr: c_int = @intCast(c.kv_lora_rank);
    const nope: c_int = @intCast(c.qk_nope_head_dim);
    const qhd: c_int = @intCast(c.qHeadDim());
    const vd: c_int = @intCast(c.v_head_dim);
    const ih: c_int = @intCast(c.index_n_heads);
    const ihd: c_int = @intCast(c.index_head_dim);
    var layers: [2]Layer = undefined;
    for (&layers, 0..) |*lw, l| {
        lw.* = .{
            .input_norm = try g.ones(&.{hid}, .bfloat16),
            .post_norm = try g.ones(&.{hid}, .bfloat16),
            .q_a = try tw.quant(&g, &.{ ql, hid }, 0.1),
            .q_a_norm = try g.ones(&.{ql}, .bfloat16),
            .q_b = try tw.quant(&g, &.{ h * qhd, ql }, 0.15),
            .kv_a = try tw.quant(&g, &.{ kvr + @as(c_int, @intCast(c.qk_rope_head_dim)), hid }, 0.1),
            .kv_a_norm = try g.ones(&.{kvr}, .bfloat16),
            .embed_q = try tw.quant(&g, &.{ h, kvr, nope }, 0.15),
            .unembed_out = try tw.quant(&g, &.{ h, vd, kvr }, 0.05),
            .o = try tw.quant(&g, &.{ hid, h * vd }, 0.02),
        };
        if (l == 0) lw.indexer = .{
            .wq_b = try tw.dense(&g, &.{ ih * ihd, ql }, 0.15),
            .wk = try tw.dense(&g, &.{ ihd, hid }, 0.1),
            .k_norm_w = try g.ones(&.{ihd}, .bfloat16),
            .k_norm_b = try g.zeros(&.{ihd}, .bfloat16),
            .weights_proj = try tw.dense(&g, &.{ ih, hid }, 0.1),
        };
    }
    // Two prompt calls (200 rows, then 40 more): spans past the top-k, the selection carried to the shared layer.
    const calls = [_][2]u32{ .{ 0, 200 }, .{ 200, 40 } };
    var xs: [calls.len]T = undefined;
    for (&xs, calls) |*x, cl| x.* = try tw.dense(&g, &.{ @intCast(cl[1]), hid }, 1);
    var outs: [2][calls.len * 2][]f32 = undefined;
    var sel: [2][]u32 = undefined;
    for ([_]bool{ false, true }, 0..) |kernels, arm| {
        var cache = try Cache.init(a, &c, 256);
        defer cache.deinit(&g);
        for (calls, xs, 0..) |cl, x, ci| {
            var carry = try Carry.init(a, &c, cl[1], cl[0] + cl[1], kernels);
            defer carry.deinit(&g);
            for (&layers, 0..) |*lw, l| {
                const y = try attention(&g, &c, lw, @intCast(l), x, cl[0], cl[1], &cache, &carry, .auto, kernels);
                outs[arm][ci * 2 + l] = try hostF32(&g, y);
                if (ci == 0 and l == 0) {
                    // The first call's selection, each row's ids sorted (a row that reaches fewer keys than the top-k
                    // also holds keys past it, which neither form attends: those rows are not compared).
                    var ids: std.ArrayList(T) = .empty;
                    defer ids.deinit(a);
                    for (carry.topk) |t| try ids.append(a, try g.reshape(t.?, &.{ -1, @intCast(c.index_topk) }));
                    const all = try g.sort(try g.astype(try g.concat(ids.items, 0), .uint32), -1);
                    sel[arm] = try a.alloc(u32, cl[1] * c.index_topk);
                    _ = try g.hostU32(all, sel[arm]);
                }
            }
            cache.commit(cl[1]);
        }
    }
    defer for (outs) |arm| for (arm) |o| a.free(o);
    defer for (sel) |x| a.free(x);
    const reach = (c.index_topk - 1) * c.index_topk;
    try testing.expectEqualSlices(u32, sel[0][reach..], sel[1][reach..]);
    for (outs[0], outs[1], 0..) |ref, got, i| {
        var dot: f64 = 0;
        var nr: f64 = 0;
        var ng: f64 = 0;
        var maxd: f32 = 0;
        var maxr: f32 = 0;
        for (ref, got) |r, x| {
            dot += r * x;
            nr += r * r;
            ng += x * x;
            maxd = @max(maxd, @abs(r - x));
            maxr = @max(maxr, @abs(r));
        }
        const cos = dot / @sqrt(nr * ng);
        std.debug.print("glm attention kernels vs reference form, call {d} layer {d}: cos {d:.6}, max |d| {d:.4} (output up to {d:.3})\n", .{ i / 2, i % 2, cos, maxd, maxr });
        try testing.expect(cos > 0.999 and maxd <= 0.05 * maxr);
    }
}

fn hostF32(g: *G, x: T) ![]f32 {
    const f = try g.astype(x, .float32);
    try g.evalAll(&.{f});
    const out = try testing.allocator.alloc(f32, @intCast(g.shapeOf(f).numel()));
    errdefer testing.allocator.free(out);
    _ = try g.hostF32(f, out);
    return out;
}
