//! GLM-5.3-Flash's trunk on the GPU stream: the residents bound from the pack's BF16 shards by their original names, the
//! decode state (each DSA layer's latent and indexer lanes, each KDA layer's conv tail and recurrent state), and the
//! forward of rows at positions [start, start + rows) through every layer: each sublayer inside its 4-stream Sinkhorn
//! hyper-connection (`ops.collapse`, the sublayer on the normalized collapse, `ops.expand`), KDA (kda.zig) or DSA
//! (dsa.zig) attention, then the dense MLP or the MoE (the F32 router, the routed experts through MiMo's expert route
//! (experts.zig) and one ungated shared expert), every SwiGLU clamped; the head reads the final norm of the stream mean.
//!
//! A prompt call (at most `prompt_call_rows` rows) runs layer by layer: P1's predicted seed posted before the layer's
//! attention (the router on the attention's collapsed input, MiMo's predictor), the attention in spans of `span_rows`
//! rows with at most `spans_in_flight` of them committed (the KDA state and the DSA lanes carry from one span to the
//! next), the MoE over every row at once (the wide lane: each routed layer's experts read once per call), the layer's
//! output evaluated at its end (MiMo's tail). A decode step is one row: its routed calls take the decode lane, whose
//! routing barrier evaluates the layer up to its router; nothing else is evaluated before the logits.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const expert_policy = sdk_ext.expert.policy;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const ops = @import("ops.zig");
const kda = @import("kda.zig");
const dsa = @import("dsa.zig");
const experts = @import("experts.zig");
const nvfp4 = @import("nvfp4.zig");
/// Profile builds: the prompt pass's stage marks (MiMo's timeline, armed by `GLM5_NEXT_PREFILL_TIMELINE`); every mark
/// compiles to nothing in a served build.
const ptl = @import("../mimo_v2/prefill_timeline.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// The prompt pass's calls: at most this many rows each (MiMo's `prefill_max_tokens`: the expert math's row scratch and
/// the combine's tiles hold one call).
pub const prompt_call_rows: u32 = nvfp4.Format.max_call_tokens;
/// The rows of a prompt call's attention span, and the spans committed at once (the bill's attention wave).
pub const span_rows: u32 = 1024;
pub const spans_in_flight: u32 = 2;
/// The logits dump's rows per chunk (dump.zig): one chunk's head output (bf16) and its F32 copy live at once.
pub const dump_chunk_rows: u32 = 256;
/// A prompt call's dense MLP (layers 0-2: 12,288 wide) in tiles of this many rows, at most `spans_in_flight` committed.
pub const dense_tile_rows: u32 = 2048;

pub const Mlp = struct { gate: T, up: T, down: T };
/// The router: BF16 [experts, hidden] as stored, its transposed F32 copy [hidden, experts] (the routing product and the
/// lookahead's predictor read it) and the F32 correction bias [experts].
pub const Router = struct { w_t: T, bias: T };
/// A sublayer's hyper-connection: f32 [(2 + hc) * hc, hc * hidden], f32 [(2 + hc) * hc] and f32 [3].
pub const Mhc = struct { fn_rows: T, base: T, scale: T };

pub const Layer = struct {
    input_norm: T,
    post_norm: T,
    hc_attn: Mhc,
    hc_ffn: Mhc,
    attn: union(enum) { kda: kda.Weights, dsa: dsa.Weights },
    /// The dense MLP (the first `first_k_dense` layers), else the MoE's router and shared expert (the routed experts
    /// stream: the expert source's absolute layer is this layer's index).
    dense: ?Mlp = null,
    router: ?Router = null,
    shared: ?Mlp = null,
    /// The layer's index among the DSA layers (its lanes) or the KDA layers (its state).
    slot: u32,
};

/// A tensor of the pack the trunk reads. Every other one (the vision tower, the MTP layer, the DSA layers' unused FP8
/// KV scales) is dropped from the weight map before the residents are read (module.zig), and never billed (bill.zig).
pub fn keptResident(c: *const glm.Config, name: []const u8) bool {
    for ([_][]const u8{ "lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight" }) |n|
        if (std.mem.eql(u8, name, n)) return true;
    const pre = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return false;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
    const l = std.fmt.parseInt(u32, rest[0..dot], 10) catch return false;
    if (l >= c.n_layers) return false;
    const suffix = rest[dot + 1 ..];
    return !std.mem.endsWith(u8, suffix, ".k_scale") and !std.mem.endsWith(u8, suffix, ".v_scale");
}

/// The residents. The f32 tables the trunk reads (the hyper-connections, the routers' transposed copies, the indexer's
/// pool gate, the KDA gate and conv) are this struct's own arrays (`owned`), built once at bind; every other array is
/// the host's weight map's.
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

    /// The owned f32 tables' buffers, as MLX allocates them (each past one page rounded up to whole pages): the bill's
    /// `trunk_f32_tables` (the hyper-connections, the KDA tables, the DSA tables) and `router_f32`.
    pub fn ownedBytes(c: *const glm.Config, page: u64) struct { tables: u64, routers: u64 } {
        const hc_mix: u64 = (2 + @as(u64, c.hc_mult)) * c.hc_mult;
        const mhc = 2 * (bufferBytes(hc_mix * c.hc_mult * c.hidden_size * 4, page) + bufferBytes(hc_mix * 4, page) + bufferBytes(3 * 4, page));
        const kda_tables = 2 * bufferBytes(@as(u64, c.kdaDim()) * 4, page) + bufferBytes(@as(u64, c.kdaConvDim()) * c.kda_conv * 4, page);
        const dsa_tables = bufferBytes(@as(u64, c.hidden_size) * c.index_head_dim * 4, page) + bufferBytes(@as(u64, c.index_kpool) * c.index_head_dim * 4, page);
        return .{
            .tables = c.n_layers * mhc + c.nKda() * kda_tables + c.nDsa() * dsa_tables,
            .routers = c.moeLayers() * (bufferBytes(@as(u64, c.hidden_size) * c.n_routed_experts * 4, page)),
        };
    }

    /// The owned arrays' count (the bill's buffer census).
    pub fn ownedArrays(c: *const glm.Config) u64 {
        return c.n_layers * 6 + c.nKda() * 3 + c.nDsa() * 4 + c.moeLayers();
    }

    pub fn bind(a: std.mem.Allocator, s: mlx.mlx_stream, map: *const sdk.Weights, c: *const glm.Config, diag: ?*sdk.Diag) !Weights {
        const layers = try a.alloc(Layer, c.n_layers);
        errdefer a.free(layers);
        const h: c_int = @intCast(c.hidden_size);
        const vocab: c_int = @intCast(c.vocab_size);
        var w: Weights = .{
            .embed = try need(map, "model.language_model.embed_tokens.weight", &.{ vocab, h }, diag),
            .norm = try need(map, "model.language_model.norm.weight", &.{h}, diag),
            .head = try need(map, "lm_head.weight", &.{ vocab, h }, diag),
            .layers = layers,
        };
        errdefer {
            for (w.owned.items) |x| _ = mlx.mlx_array_free(x);
            w.owned.deinit(a);
        }
        const hc_mix: c_int = @intCast((2 + c.hc_mult) * c.hc_mult);
        const hch: c_int = @intCast(c.hc_mult * c.hidden_size);
        var n_dsa: u32 = 0;
        var n_kda: u32 = 0;
        for (layers, 0..) |*lw, li| {
            const l: u32 = @intCast(li);
            var pb: [96]u8 = undefined;
            const pre = std.fmt.bufPrint(&pb, "model.language_model.layers.{d}.", .{l}) catch unreachable;
            const L = struct {
                pre: []const u8,
                map: *const sdk.Weights,
                diag: ?*sdk.Diag,
                fn get(self: @This(), comptime suffix: []const u8, shape: []const c_int) !T {
                    var nb: [160]u8 = undefined;
                    return need(self.map, std.fmt.bufPrint(&nb, "{s}" ++ suffix, .{self.pre}) catch unreachable, shape, self.diag);
                }
            };
            const lg: L = .{ .pre = pre, .map = map, .diag = diag };
            lw.* = .{
                .input_norm = try lg.get("input_layernorm.weight", &.{h}),
                .post_norm = try lg.get("post_attention_layernorm.weight", &.{h}),
                .hc_attn = .{
                    .fn_rows = try w.own(a, try f32Copy(s, try lg.get("hc_attn_fn", &.{ hc_mix, hch }), false)),
                    .base = try w.own(a, try f32Copy(s, try lg.get("hc_attn_base", &.{hc_mix}), false)),
                    .scale = try w.own(a, try f32Copy(s, try lg.get("hc_attn_scale", &.{3}), false)),
                },
                .hc_ffn = .{
                    .fn_rows = try w.own(a, try f32Copy(s, try lg.get("hc_ffn_fn", &.{ hc_mix, hch }), false)),
                    .base = try w.own(a, try f32Copy(s, try lg.get("hc_ffn_base", &.{hc_mix}), false)),
                    .scale = try w.own(a, try f32Copy(s, try lg.get("hc_ffn_scale", &.{3}), false)),
                },
                .attn = undefined,
                .slot = undefined,
            };
            if (c.isDsa(l)) {
                const heads: c_int = @intCast(c.n_heads);
                const hd: c_int = @intCast(c.head_dim);
                const r: c_int = @intCast(c.kv_lora_rank);
                const ql: c_int = @intCast(c.q_lora_rank);
                const ih: c_int = @intCast(c.index_n_heads);
                const ihd: c_int = @intCast(c.index_head_dim);
                const kv_b = try lg.get("self_attn.kv_b_proj.weight", &.{ heads * 2 * hd, r });
                lw.attn = .{ .dsa = .{
                    .q_a = try lg.get("self_attn.q_a_proj.weight", &.{ ql, h }),
                    .q_a_norm = try lg.get("self_attn.q_a_layernorm.weight", &.{ql}),
                    .q_b = try lg.get("self_attn.q_b_proj.weight", &.{ heads * hd, ql }),
                    .kv_a = try lg.get("self_attn.kv_a_proj_with_mqa.weight", &.{ r, h }),
                    .kv_a_norm = try lg.get("self_attn.kv_a_layernorm.weight", &.{r}),
                    .wk = try w.own(a, try view(s, kv_b, &.{ heads, 2 * hd, r }, 0, hd)),
                    .wv = try w.own(a, try view(s, kv_b, &.{ heads, 2 * hd, r }, hd, 2 * hd)),
                    .o = try lg.get("self_attn.o_proj.weight", &.{ h, heads * hd }),
                    .idx_wq_b = try lg.get("self_attn.indexer.wq_b.weight", &.{ ih * ihd, ql }),
                    .idx_wk = try lg.get("self_attn.indexer.wk.weight", &.{ ihd, h }),
                    .idx_k_norm_w = try lg.get("self_attn.indexer.k_norm.weight", &.{ihd}),
                    .idx_k_norm_b = try lg.get("self_attn.indexer.k_norm.bias", &.{ihd}),
                    .idx_wproj = try lg.get("self_attn.indexer.weights_proj.weight", &.{ ih, h }),
                    .idx_gate_t = try w.own(a, try f32Copy(s, try lg.get("self_attn.indexer.index_kpool_compress_gate", &.{ ihd, h }), true)),
                    .idx_ape = try w.own(a, try f32Copy(s, try lg.get("self_attn.indexer.index_kpool_compress_ape", &.{ @intCast(c.index_kpool), ihd }), false)),
                } };
                lw.slot = n_dsa;
                n_dsa += 1;
            } else {
                const hv: c_int = @intCast(c.kda_heads);
                const dk: c_int = @intCast(c.kda_head_dim);
                const dim = hv * dk;
                lw.attn = .{ .kda = .{
                    .q = try lg.get("self_attn.q_proj.weight", &.{ dim, h }),
                    .k = try lg.get("self_attn.k_proj.weight", &.{ dim, h }),
                    .v = try lg.get("self_attn.v_proj.weight", &.{ dim, h }),
                    .f_a = try lg.get("self_attn.f_a_proj.weight", &.{ dk, h }),
                    .f_b = try lg.get("self_attn.f_b_proj.weight", &.{ dim, dk }),
                    .g_a = try lg.get("self_attn.g_a_proj.weight", &.{ dk, h }),
                    .g_b = try lg.get("self_attn.g_b_proj.weight", &.{ dim, dk }),
                    .b = try lg.get("self_attn.b_proj.weight", &.{ hv, h }),
                    .a_exp = try w.own(a, try aExp(s, try lg.get("self_attn.A_log", &.{hv}), hv, dk)),
                    .dt_bias = try w.own(a, try f32Copy(s, try lg.get("self_attn.dt_bias", &.{dim}), false)),
                    .conv_w = try w.own(a, try convLayout(s, try lg.get("self_attn.conv1d.weight", &.{ 3 * dim, 1, @intCast(c.kda_conv) }))),
                    .o_norm = try lg.get("self_attn.o_norm.weight", &.{dk}),
                    .o = try lg.get("self_attn.o_proj.weight", &.{ h, dim }),
                } };
                lw.slot = n_kda;
                n_kda += 1;
            }
            if (c.isMoe(l)) {
                const e: c_int = @intCast(c.n_routed_experts);
                const si: c_int = @intCast(c.moe_intermediate_size * c.n_shared_experts);
                lw.router = .{
                    .w_t = try w.own(a, try f32Copy(s, try lg.get("mlp.gate.weight", &.{ e, h }), true)),
                    .bias = try lg.get("mlp.gate.e_score_correction_bias", &.{e}),
                };
                if (mlx.mlx_array_dtype(lw.router.?.bias) != .float32) return refuse(diag, error.Glm5NextResidentDtype, "layer {d} gate.e_score_correction_bias is not F32", .{l});
                lw.shared = .{
                    .gate = try lg.get("mlp.shared_experts.gate_proj.weight", &.{ si, h }),
                    .up = try lg.get("mlp.shared_experts.up_proj.weight", &.{ si, h }),
                    .down = try lg.get("mlp.shared_experts.down_proj.weight", &.{ h, si }),
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

    /// The gates the expert scheduler's lookahead reads, one per model layer (a dense layer's empty): the router's
    /// transposed F32 copy and its correction bias.
    pub fn gates(w: *const Weights, a: std.mem.Allocator) ![]experts.Gate {
        const out = try a.alloc(experts.Gate, w.layers.len);
        for (out, w.layers) |*gt, lw| gt.* = if (lw.router) |r| .{ .w = r.w_t, .bias = r.bias } else .{ .w = .{ .ctx = null }, .bias = .{ .ctx = null } };
        return out;
    }
};

fn refuse(diag: ?*sdk.Diag, err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    if (diag) |d| d.set("glm5_next: " ++ fmt, args);
    return err;
}

/// An MLX buffer of `n` bytes as the Metal allocator takes it: past one page, rounded up to whole pages.
pub fn bufferBytes(n: u64, page: u64) u64 {
    return if (n > page) std.mem.alignForward(u64, n, page) else n;
}

/// The resident `name`, its shape the served form's (refused by name otherwise: once, at construction).
fn need(map: *const sdk.Weights, name: []const u8, shape: []const c_int, diag: ?*sdk.Diag) !T {
    const x = map.get(name) orelse return refuse(diag, error.Glm5NextResidentMissing, "the residents lack {s}", .{name});
    if (!std.mem.eql(c_int, mlx.getShape(x), shape)) return refuse(diag, error.Glm5NextResidentShape, "{s} is {any}, the served form's {any}", .{ name, mlx.getShape(x), shape });
    return x;
}

/// `x` as f32 in a buffer of its own (transposed when asked), evaluated: never the weight map's buffer.
pub fn f32Copy(s: mlx.mlx_stream, x: T, transposed: bool) !T {
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    if (transposed) try mlx.check(mlx.mlx_transpose(&t, x, s)) else _ = mlx.mlx_array_set(&t, x);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, t, .float32, s));
    // A row-contiguous copy (MLX's `contiguous`): the table never keeps the weight map's array alive.
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, f, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// kv_b_proj [heads * (nope + v), latent] as [heads, nope + v, latent], rows [lo, hi) of each head: a view.
pub fn view(s: mlx.mlx_stream, kv_b: T, shape: []const c_int, lo: c_int, hi: c_int) !T {
    var r3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r3);
    try mlx.check(mlx.mlx_reshape(&r3, kv_b, shape.ptr, shape.len, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, r3, &[_]c_int{ 0, lo, 0 }, 3, &[_]c_int{ shape[0], hi, shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// exp(A_log) [heads] (f32) repeated over each head's key channels: f32 [heads * dk], evaluated (the forget gate's
/// constant factor, once).
fn aExp(s: mlx.mlx_stream, a_log: T, hv: c_int, dk: c_int) !T {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a_log, .float32, s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, f, &[_]c_int{ hv, 1 }, 2, s));
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_broadcast_to(&bc, col, &[_]c_int{ hv, dk }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, bc, &[_]c_int{hv * dk}, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_exp(&out, flat, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The pack's depthwise conv [C, 1, K] (HF) as MLX's conv1d reads it, [C, K, 1], f32, evaluated.
fn convLayout(s: mlx.mlx_stream, w: T) !T {
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_swapaxes(&t, w, 1, 2, s));
    return f32Copy(s, t, false);
}

/// The decode state of one request: each DSA layer's two lanes, bounded at `cap` positions, and each KDA layer's state.
pub const Cache = struct {
    a: std.mem.Allocator,
    latent: []dsa.Lane,
    index: []dsa.Lane,
    kda: []kda.State,
    /// The MTP verify's captures, one per KDA layer (`Head.verify`): released after each round (`rollback`).
    capture: []kda.Capture,
    len: u32 = 0,
    cap: u32,

    pub fn init(a: std.mem.Allocator, c: *const glm.Config, cap: u32) !Cache {
        const latent = try a.alloc(dsa.Lane, c.nDsa());
        errdefer a.free(latent);
        const index = try a.alloc(dsa.Lane, c.nDsa());
        errdefer a.free(index);
        const kda_states = try a.alloc(kda.State, c.nKda());
        errdefer a.free(kda_states);
        const captures = try a.alloc(kda.Capture, c.nKda());
        for (latent, index) |*lt, *ix| {
            lt.* = dsa.Lane.init(cap, cap);
            ix.* = dsa.Lane.init(cap, cap);
        }
        for (kda_states) |*st| st.* = .{};
        for (captures) |*cp| cp.* = .{};
        return .{ .a = a, .latent = latent, .index = index, .kda = kda_states, .capture = captures, .cap = cap };
    }

    pub fn deinit(self: *Cache, g: *G) void {
        for (self.latent, self.index) |*lt, *ix| {
            lt.deinit(g);
            ix.deinit(g);
        }
        for (self.kda) |*st| st.reset(g);
        for (self.capture) |*cp| cp.release(g);
        self.a.free(self.latent);
        self.a.free(self.index);
        self.a.free(self.kda);
        self.a.free(self.capture);
        self.* = undefined;
    }

    /// Every lane allocated at its bound (bf16 zeros), evaluated, once: a request's appends then write in place.
    pub fn preallocate(self: *Cache, g: *G, c: *const glm.Config) !void {
        const m = g.mark();
        defer g.resetTo(m);
        var news: std.ArrayList(T) = .empty;
        defer news.deinit(self.a);
        for (self.latent, self.index) |*lt, *ix| {
            for ([_]*dsa.Lane{ lt, ix }, [_]u32{ c.kv_lora_rank, 2 * c.index_head_dim }) |lane, d| {
                if (lane.buf != null) continue;
                lane.buf = g.keep(try g.zeros(&.{ 1, @intCast(self.cap), @intCast(d) }, .bfloat16));
                lane.cap = self.cap;
                try news.append(self.a, lane.buf.?);
            }
        }
        if (news.items.len > 0) try g.evalAll(news.items);
    }

    /// A new request from position 0: the lanes empty (their buffers kept), every KDA state cleared.
    pub fn restart(self: *Cache, g: *G) void {
        for (self.latent, self.index) |*lt, *ix| {
            lt.truncateTo(0);
            ix.truncateTo(0);
        }
        for (self.kda) |*st| st.reset(g);
        for (self.capture) |*cp| cp.release(g);
        self.len = 0;
    }

    /// The MTP round's rollback after a verify of `rows` rows at `start` that kept `kept` of them (t1 and the accepted
    /// drafts): the DSA lanes truncated to the kept positions; every KDA state the capture's at row `kept - 1` (a partial
    /// accept; a full one keeps the verify's own final state); the captures released; the length the kept positions.
    pub fn rollback(self: *Cache, g: *G, c: *const glm.Config, start: u32, rows: u32, kept: u32) !void {
        std.debug.assert(kept >= 1 and kept <= rows and self.len == start + rows);
        const len = start + kept;
        for (self.latent, self.index) |*lt, *ix| {
            lt.truncateTo(len);
            ix.truncateTo(len);
        }
        if (kept < rows) {
            const m = g.mark();
            defer g.resetTo(m);
            var restored: std.ArrayList(T) = .empty;
            defer restored.deinit(self.a);
            for (self.kda, self.capture) |*st, *cp| {
                try kda.restore(g, c, st, cp, kept - 1);
                try restored.append(self.a, st.conv.?);
                try restored.append(self.a, st.rec.?);
            }
            // The copies evaluated now, so the captures' buffers go with their handles below.
            try g.evalAll(restored.items);
        }
        for (self.capture) |*cp| cp.release(g);
        self.len = len;
    }

    /// The verify's captures (every KDA layer's state rows and conv window): evaluated with the verify.
    pub fn captureArrays(self: *const Cache, out: *std.ArrayList(T), a: std.mem.Allocator) !void {
        for (self.capture) |cp| {
            if (cp.seq) |x| try out.append(a, x);
            if (cp.window) |x| try out.append(a, x);
        }
    }

    /// The layer's arrays an evaluation of its forward covers (lanes and states).
    fn buffers(self: *const Cache, lw: *const Layer, out: *[2]T) []const T {
        var n: usize = 0;
        switch (lw.attn) {
            .dsa => for ([_]?T{ self.latent[lw.slot].buf, self.index[lw.slot].buf }) |b| if (b) |x| {
                out[n] = x;
                n += 1;
            },
            .kda => for ([_]?T{ self.kda[lw.slot].conv, self.kda[lw.slot].rec }) |b| if (b) |x| {
                out[n] = x;
                n += 1;
            },
        }
        return out[0..n];
    }

    /// Every KDA state (conv tail and recurrent state): a step evaluates them with its logits (a conv tail feeds no
    /// output of its own step, so nothing else would).
    pub fn stateArrays(self: *const Cache, out: *std.ArrayList(T), a: std.mem.Allocator) !void {
        for (self.kda) |st| {
            if (st.conv) |x| try out.append(a, x);
            if (st.rec) |x| try out.append(a, x);
        }
    }
};

pub fn linear(g: *G, x: T, w: T) !T {
    return g.matmul(x, try g.transpose(w));
}

pub fn rmsNorm(g: *G, x: T, w: T, eps: f32) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, g.s));
    return g.adopt(o);
}

/// silu(min(gate, limit)) * clip(up, -limit, limit) (the SDK's clamped SwiGLU: mlx-serve's `clampedSwiGLU`), then down.
pub fn mlp(g: *G, c: *const glm.Config, m: Mlp, x: T) !T {
    const act = try sdk_ext.quant.activation(G, g, .{ .swiglu_clamped = c.swiglu_limit }, try linear(g, x, m.gate), try linear(g, x, m.up));
    return linear(g, act, m.down);
}

/// A prompt call's dense MLP over `x` [rows, hidden] in tiles of `dense_tile_rows` rows: each tile committed as built
/// and the tile before it waited (at most `spans_in_flight` tiles' intermediates live), the tiles joined. The same rows
/// through the same graph as `mlp`: each output row depends on its own input row only.
fn mlpTiles(g: *G, a: std.mem.Allocator, c: *const glm.Config, m: Mlp, x: T, rows: u32) !T {
    if (rows <= dense_tile_rows) return mlp(g, c, m, x);
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
        o.* = g.keep(try mlp(g, c, m, try g.slice(x, &.{ lo, 0 }, &.{ hi, hidden }, &.{ 1, 1 })));
        kept += 1;
        g.resetTo(ms);
        try g.asyncEval(&.{o.*});
        if (i > 0) try g.evalAll(&.{outs[i - 1]});
    }
    return g.concat(outs, 0);
}

pub const Routing = struct { indices: T, weights: T };

/// noaux_tc at one group: f32 logits on the transposed F32 router, sigmoid, the correction bias for the choice, the
/// top-k, the chosen sigmoid scores normalized (`norm_topk_prob`, + 1e-20) and scaled. x [n, hidden] -> ids u32 [n, k],
/// weights f32 [n, k].
pub fn route(g: *G, c: *const glm.Config, r: Router, x: T) !Routing {
    const k: c_int = @intCast(c.n_experts_per_tok);
    const sig = try g.sigmoid(try g.matmul(try g.astype(x, .float32), r.w_t));
    const part = try g.argpartition(try g.neg(try g.add(sig, r.bias)), k - 1, -1);
    const n = g.shapeOf(part).dim(0);
    const inds = try g.astype(try g.slice(part, &.{ 0, 0 }, &.{ n, k }, &.{ 1, 1 }), .uint32);
    var w = try g.takeAlongAxis(sig, inds, -1);
    if (c.norm_topk_prob) w = try g.div(w, try g.add(try g.sum(w, -1, true), try g.scalar(1e-20, .float32)));
    w = try g.mul(w, try g.scalar(c.routed_scaling, .float32));
    return .{ .indices = inds, .weights = w };
}

/// The sublayer attention of layer `lw` on `stream` [1, R, hc, hidden] (rows at `start`): the collapse, the norm, KDA or
/// DSA, the expand -> the new stream.
fn attentionSpan(g: *G, c: *const glm.Config, lw: *const Layer, stream: T, start: u32, cache: *Cache, routes: dsa.Routes, comptime head: Head) !T {
    var col = try ops.collapse(g.s, stream, lw.hc_attn.fn_rows, lw.hc_attn.scale, lw.hc_attn.base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
    defer col.deinit();
    const normed = try rmsNorm(g, col.x, lw.input_norm, c.rms_norm_eps);
    const out = switch (lw.attn) {
        // The MTP verify captures every row's state (`Cache.rollback`); every other call advances the state alone.
        .kda => |*kw| if (head == .verify)
            try kda.forwardCapture(g, c, kw, normed, &cache.kda[lw.slot], &cache.capture[lw.slot])
        else
            try kda.forward(g, c, kw, normed, &cache.kda[lw.slot]),
        .dsa => |*dw| try dsa.forward(g, c, dw, normed, start, .{ .latent = &cache.latent[lw.slot], .index = &cache.index[lw.slot] }, routes),
    };
    return g.adopt(try ops.expand(g.s, out, stream, &col));
}

/// P1 (a prompt call of a routed layer): the router on the layer's collapsed attention input (its post-attention norm),
/// the call's predicted top-k counted over every row (one eval), ranked hottest first and posted to the stream, which
/// reads them into the layer's free persistent rows while the attention runs. The prediction only picks reads.
fn readAhead(g: *G, c: *const glm.Config, lw: *const Layer, spans: []const T, src: *experts.Source, layer: u32) !void {
    const m = g.mark();
    defer g.resetTo(m);
    const r = lw.router.?;
    var ids: [prompt_call_rows / span_rows + 1]T = undefined;
    for (spans, ids[0..spans.len]) |sp, *out| {
        var col = try ops.collapse(g.s, sp, lw.hc_attn.fn_rows, lw.hc_attn.scale, lw.hc_attn.base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
        defer col.deinit();
        const sh = g.shapeOf(col.x);
        const x = try g.reshape(try rmsNorm(g, col.x, lw.post_norm, c.rms_norm_eps), &.{ sh.dim(1), sh.dim(2) });
        out.* = (try route(g, c, r, x)).indices;
    }
    try g.evalAll(ids[0..spans.len]);
    const counts = src.ahead_counts[0..c.n_routed_experts];
    @memset(counts, 0);
    for (ids[0..spans.len]) |x| {
        const n: usize = @intCast(g.shapeOf(x).dim(0) * g.shapeOf(x).dim(1));
        for (try g.readU32(x, src.ahead_ids[0..n])) |e| counts[e] += 1;
    }
    try src.readAheadSeed(layer, expert_policy.rankHottest(counts, &src.ahead_ranked));
}

/// What a forward hands back after its last layer (a comptime route per call site):
///   last    the last row's logits f32 [1, vocab] (the served prompt pass and serial steps without the MTP lane)
///   dump    `last`, and every row's logits to the dump route's sink (the KL gate)
///   hidden  `last`, and every row's final-normed hidden [rows, hidden] bf16 into `Taps.hidden` (the MTP lane's pairs:
///           the prompt pass and serial steps with the lane on)
///   verify  every row's logits f32 [rows, vocab] and final-normed hidden (`Taps.hidden`), each KDA layer's per-row
///           states captured (the MTP lane's verify; `Cache.rollback`)
pub const Head = enum { last, dump, hidden, verify };

/// The hidden rows a `hidden` / `verify` forward hands back (a kept handle the caller releases).
pub const Taps = struct { hidden: ?T = null };

/// One forward of `ids` at positions [start, start + ids.len) through every layer (each layer's state advanced in
/// `cache`), the routed experts through `src`: the last row's logits f32 [1, vocab], a kept handle the caller frees.
/// `head` `.dump` (the logits-dump route, chosen at construction) also hands every row's logits, f32 in chunks of
/// `dump_chunk_rows` rows, to `sink.append(g, chunk)` before returning.
pub fn forward(g: *G, a: std.mem.Allocator, c: *const glm.Config, w: *const Weights, ids: []const u32, start: u32, cache: *Cache, src: *experts.Source, routes: dsa.Routes, comptime head: Head, sink: anytype) !T {
    const rows: u32 = @intCast(ids.len);
    if (rows == 0 or rows > prompt_call_rows) return error.Glm5NextCallRows;
    if (start + rows > cache.cap) return error.ContextOverBill;
    // A wide call (the prompt's) or the decode lane's (the source's phase route: one row, or the MTP verify's in decode).
    const prompt = !src.decodeLaneAt(rows);
    const m0 = g.mark();
    defer g.resetTo(m0);
    const hidden: c_int = @intCast(c.hidden_size);
    const hc: c_int = @intCast(c.hc_mult);
    // The stream in spans of `span_rows` rows, each [1, r, hc, hidden] (kept across the layers' resets).
    const n_spans: usize = (rows + span_rows - 1) / span_rows;
    const spans = try a.alloc(T, n_spans);
    defer a.free(spans);
    var live: usize = 0;
    defer for (spans[0..live]) |x| g.release(x);
    {
        const idx = try g.hostArray(std.mem.sliceAsBytes(ids), &.{@intCast(rows)}, .uint32);
        const emb = try g.take(w.embed, idx, 0);
        for (spans, 0..) |*sp, i| {
            const lo: c_int = @intCast(i * span_rows);
            const hi: c_int = @intCast(@min(rows, (i + 1) * span_rows));
            const e = try g.reshape(try g.slice(emb, &.{ lo, 0 }, &.{ hi, hidden }, &.{ 1, 1 }), &.{ 1, hi - lo, 1, hidden });
            sp.* = g.keep(try g.broadcastTo(e, &.{ 1, hi - lo, hc, hidden }));
            live += 1;
        }
    }
    const cols = try a.alloc(ops.Collapsed, n_spans);
    defer a.free(cols);
    const normed = try a.alloc(T, n_spans);
    defer a.free(normed);
    for (w.layers, 0..) |*lw, li| {
        const l: u32 = @intCast(li);
        const m = g.mark();
        defer g.resetTo(m);
        if (comptime ptl.enabled) {
            ptl.layer(l);
            ptl.stage(.predict);
        }
        if (prompt and lw.router != null) try readAhead(g, c, lw, spans, src, l);
        if (comptime ptl.enabled) ptl.stage(.attn);
        // The attention, span by span (the KDA state and the DSA lanes carry from one span to the next); in a prompt
        // call each span is committed as built and the span before it waited, so at most `spans_in_flight` run at once.
        var prev: ?T = null;
        for (spans, 0..) |*sp, i| {
            const ms = g.mark();
            const next = g.keep(try attentionSpan(g, c, lw, sp.*, start + @as(u32, @intCast(i * span_rows)), cache, routes, head));
            g.resetTo(ms);
            g.release(sp.*);
            sp.* = next;
            if (!prompt) continue;
            var bufs: [2]T = undefined;
            const b = cache.buffers(lw, &bufs);
            var commit: [3]T = undefined;
            commit[0] = next;
            @memcpy(commit[1 .. 1 + b.len], b);
            try g.asyncEval(commit[0 .. 1 + b.len]);
            if (prev) |p| try g.evalAll(&.{p});
            prev = next;
        }
        // The MLP over every row at once: each span's collapse, the rows joined, the dense MLP or the MoE (the routed
        // experts' weighted sum and the shared expert), each span's expand. (Profile marks: `dense` the FFN's collapse
        // and norm with the dense MLP tiles or the shared expert committed before the routing barrier, `router` the
        // routed call, `tail` the expand and the layer's output evaluated.)
        if (comptime ptl.enabled) ptl.stage(.dense);
        var n_cols: usize = 0;
        defer for (cols[0..n_cols]) |*cl| cl.deinit();
        for (spans, cols, normed) |sp, *cl, *nx| {
            cl.* = try ops.collapse(g.s, sp, lw.hc_ffn.fn_rows, lw.hc_ffn.scale, lw.hc_ffn.base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
            n_cols += 1;
            nx.* = try rmsNorm(g, cl.x, lw.post_norm, c.rms_norm_eps);
        }
        const x = try g.reshape(if (n_spans == 1) normed[0] else try g.concat(normed, 1), &.{ @intCast(rows), hidden });
        const f = if (lw.dense) |d| (if (prompt) try mlpTiles(g, a, c, d, x, rows) else try mlp(g, c, d, x)) else blk: {
            const r = try route(g, c, lw.router.?, x);
            const shared = try mlp(g, c, lw.shared.?, x);
            // A prompt call commits the shared expert before its routed call (the GPU runs it while the host routes and
            // the first group's reads land); a decode-lane call hands it over as the call's hoist (run during its read
            // wait). The sum is the same either way.
            if (prompt) try g.asyncEval(&.{shared});
            if (comptime ptl.enabled) ptl.stage(.router);
            const routed = try g.adopt(try src.forward(l, x, r.indices, r.weights, if (prompt) &.{} else &.{shared}));
            break :blk try g.add(routed, shared);
        };
        if (comptime ptl.enabled) ptl.stage(.tail);
        for (spans, cols[0..n_spans], 0..) |*sp, *cl, i| {
            const lo: c_int = @intCast(i * span_rows);
            const hi: c_int = @intCast(@min(rows, (i + 1) * span_rows));
            const fi = try g.reshape(try g.slice(f, &.{ lo, 0 }, &.{ hi, hidden }, &.{ 1, 1 }), &.{ 1, hi - lo, hidden });
            const next = g.keep(try g.adopt(try ops.expand(g.s, fi, sp.*, cl)));
            g.release(sp.*);
            sp.* = next;
        }
        // MiMo's tail: a prompt layer's output evaluated before the next layer starts (bounding its graph), then the
        // pooled scratch's purge. A decode step's stays lazy: the next routing barrier evaluates it.
        if (prompt) {
            try g.evalAll(spans);
            src.promptLayerEnd();
        }
    }
    if (comptime ptl.enabled) ptl.leaveLayers();
    cache.len = start + rows;
    if (head == .dump) try dumpRows(g, c, w, spans, sink);
    if (head == .hidden or head == .verify) {
        const final_rows = try normedRows(g, c, w, spans, rows);
        sink.hidden = g.keep(final_rows);
        // The verify's rows: every row's logits from the same normed rows.
        if (head == .verify) return g.keep(try g.astype(try linear(g, final_rows, w.head), .float32));
    }
    // The last row: the stream's mean over its hc streams, the final norm, the head, f32.
    const last = spans[n_spans - 1];
    const lr = g.shapeOf(last).dim(1);
    const row = try g.slice(last, &.{ 0, lr - 1, 0, 0 }, &.{ 1, lr, hc, hidden }, &.{ 1, 1, 1, 1 });
    const mean = try g.reshape(try g.mean(row, 2, false), &.{ 1, hidden });
    return g.keep(try g.astype(try linear(g, try rmsNorm(g, mean, w.norm, c.rms_norm_eps), w.head), .float32));
}

/// Every row's final-normed hidden [rows, hidden] (bf16): the stream's mean over its hc streams, the final norm (the
/// head's input: the MTP lane's pairs take the hidden the lm_head reads).
fn normedRows(g: *G, c: *const glm.Config, w: *const Weights, spans: []const T, rows: u32) !T {
    const hidden: c_int = @intCast(c.hidden_size);
    var means: [prompt_call_rows / span_rows + 1]T = undefined;
    for (spans, means[0..spans.len]) |sp, *mn| mn.* = try g.mean(sp, 2, false);
    const all = if (spans.len == 1) means[0] else try g.concat(means[0..spans.len], 1);
    return rmsNorm(g, try g.reshape(all, &.{ @intCast(rows), hidden }), w.norm, c.rms_norm_eps);
}

/// The dump route's rows: every row of every span through the final norm and the head, f32, `dump_chunk_rows` at a time,
/// each chunk evaluated and handed to `sink` before the next is built (one chunk's head output and its f32 copy live).
fn dumpRows(g: *G, c: *const glm.Config, w: *const Weights, spans: []const T, sink: anytype) !void {
    const hidden: c_int = @intCast(c.hidden_size);
    const hc: c_int = @intCast(c.hc_mult);
    for (spans) |sp| {
        const sr = g.shapeOf(sp).dim(1);
        var lo: c_int = 0;
        while (lo < sr) : (lo += dump_chunk_rows) {
            const hi = @min(sr, lo + @as(c_int, dump_chunk_rows));
            const m = g.mark();
            defer g.resetTo(m);
            const rows = try g.slice(sp, &.{ 0, lo, 0, 0 }, &.{ 1, hi, hc, hidden }, &.{ 1, 1, 1, 1 });
            const mean = try g.reshape(try g.mean(rows, 2, false), &.{ hi - lo, hidden });
            const logits = try g.astype(try linear(g, try rmsNorm(g, mean, w.norm, c.rms_norm_eps), w.head), .float32);
            try g.evalAll(&.{logits});
            try sink.append(logits);
        }
    }
}

const testing = std.testing;

test "glm5_next graph: the kept residents are the trunk's (no vision tower, no MTP layer, no FP8 KV scales)" {
    const c: glm.Config = .{};
    for ([_][]const u8{
        "lm_head.weight",                                                  "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",                                "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.44.mlp.gate.e_score_correction_bias", "model.language_model.layers.3.self_attn.kv_b_proj.weight",
    }) |n| try testing.expect(keptResident(&c, n));
    for ([_][]const u8{
        "model.visual.blocks.0.attn.qkv.weight",                  "model.language_model.layers.45.eh_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.k_scale", "model.language_model.layers.3.self_attn.v_proj.v_scale",
        "model.language_model.layers.x.input_layernorm.weight",
    }) |n| try testing.expect(!keptResident(&c, n));
}

test "glm5_next graph: the owned f32 tables' bytes and the call geometry" {
    const c: glm.Config = .{};
    const page: u64 = 16384;
    const o = Weights.ownedBytes(&c, page);
    // 90 hyper-connections ([24, 16384] + [24] + [3] f32), 34 KDA layers' exp(A_log) and dt_bias [8192] and conv
    // [24576, 4], 11 DSA layers' pool gate [4096, 128] and ape [4, 128]; 42 routers [4096, 288] f32.
    try testing.expectEqual(@as(u64, 90 * (1_572_864 + 96 + 12) + 34 * (2 * 32_768 + 393_216) + 11 * (2_097_152 + 2048)), o.tables);
    try testing.expectEqual(@as(u64, 42 * 4_718_592), o.routers);
    try testing.expectEqual(@as(u32, 16_384), prompt_call_rows);
    try testing.expect(prompt_call_rows % span_rows == 0 and dump_chunk_rows <= span_rows);
}
