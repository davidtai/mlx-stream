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
const fp8 = @import("fp8.zig");
const glue = @import("glue.zig");
const settings = @import("settings.zig");
const swiglu = @import("swiglu.zig");
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

/// A SwiGLU MLP's projections (the dense layers' and the shared experts'): BF16, or the FP8 route's (fp8.zig).
pub const Mlp = struct { gate: fp8.Linear, up: fp8.Linear, down: fp8.Linear };
/// The router: BF16 [experts, hidden] as stored (`w`: the fused glue's router text reads it), its transposed F32 copy
/// [hidden, experts] (the stock routing product and the lookahead's predictor read it) and the F32 correction bias
/// [experts].
pub const Router = struct { w: T, w_t: T, bias: T };
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
    /// `residents_format: fp8_block128`: the FP8 route every FP8 projection is bound to (module-owned; a prompt layer's end
    /// releases its BF16 weights, `forward`); null under `bf16`.
    fp8: ?*fp8.Route = null,

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
        const kda_tables = 2 * bufferBytes(@as(u64, c.kdaDim()) * 4, page) + bufferBytes(@as(u64, c.kda_heads) * 4, page) + bufferBytes(@as(u64, c.kdaConvDim()) * c.kda_conv * 4, page);
        const dsa_tables = bufferBytes(@as(u64, c.hidden_size) * c.index_head_dim * 4, page) + bufferBytes(@as(u64, c.index_kpool) * c.index_head_dim * 4, page);
        return .{
            .tables = c.n_layers * mhc + c.nKda() * kda_tables + c.nDsa() * dsa_tables,
            .routers = c.moeLayers() * (bufferBytes(@as(u64, c.hidden_size) * c.n_routed_experts * 4, page)),
        };
    }

    /// The owned arrays' count (the bill's buffer census).
    pub fn ownedArrays(c: *const glm.Config) u64 {
        return c.n_layers * 6 + c.nKda() * 4 + c.nDsa() * 4 + c.moeLayers();
    }

    /// The residents route per site, by the pack's declarations (`glm.Config.siteIsFp8`; fp8.zig `siteTensors`, one table
    /// with the CPU tests): a trunk site (`glm.Config.fp8Sites` / `fp8MlpSites`) declared FP8 binds as codes U8 [out, in]
    /// with F32 scales [out / 64, in / 128] on `fp8_route` (installed by the module when the pack declares any); every other
    /// one BF16 [out, in] on the stock matmul (all of them under `bf16`). A missing scale, another dtype or an off-grid shape
    /// refuses the load by name.
    pub fn bind(a: std.mem.Allocator, s: mlx.mlx_stream, map: *const sdk.Weights, c: *const glm.Config, fp8_route: ?*fp8.Route, diag: ?*sdk.Diag) !Weights {
        if ((c.fp8Declared() > 0) != (fp8_route != null)) return error.Glm5NextResidentsRoute;
        const layers = try a.alloc(Layer, c.n_layers);
        errdefer a.free(layers);
        const h: c_int = @intCast(c.hidden_size);
        const vocab: c_int = @intCast(c.vocab_size);
        var w: Weights = .{
            .embed = try need(map, "model.language_model.embed_tokens.weight", &.{ vocab, h }, diag),
            .norm = try need(map, "model.language_model.norm.weight", &.{h}, diag),
            .head = try need(map, "lm_head.weight", &.{ vocab, h }, diag),
            .layers = layers,
            .fp8 = fp8_route,
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
                c: *const glm.Config,
                l: u32,
                route: ?*fp8.Route,
                fn get(self: @This(), comptime suffix: []const u8, shape: []const c_int) !T {
                    var nb: [160]u8 = undefined;
                    return need(self.map, std.fmt.bufPrint(&nb, "{s}" ++ suffix, .{self.pre}) catch unreachable, shape, self.diag);
                }
                /// A trunk site [out, in] as the pack declares it (fp8.zig `siteTensors`).
                fn proj(self: @This(), comptime suffix: []const u8, out: c_int, in: c_int) !fp8.Linear {
                    const format: glm.ResidentsFormat = if (self.c.siteIsFp8(self.l, suffix)) .fp8_block128 else .bf16;
                    const want = fp8.siteTensors(format, @intCast(out), @intCast(in));
                    const wt = try self.get(suffix ++ ".weight", &want.weight_shape);
                    const dt: mlx.mlx_dtype = switch (want.weight) {
                        .u8 => .uint8,
                        .bf16 => .bfloat16,
                        .f32 => .float32,
                    };
                    if (mlx.mlx_array_dtype(wt) != dt) return refuse(self.diag, error.Glm5NextResidentDtype, "{s}" ++ suffix ++ ".weight is {s}; declared {s}, it binds {s}", .{ self.pre, @tagName(mlx.mlx_array_dtype(wt)), @tagName(format), @tagName(dt) });
                    const ss = want.scales orelse return .{ .bf16 = wt };
                    const sc = try self.get(suffix ++ ".weight_scale_inv", &ss);
                    if (mlx.mlx_array_dtype(sc) != .float32) return refuse(self.diag, error.Glm5NextResidentDtype, "{s}" ++ suffix ++ ".weight_scale_inv is {s}; declared fp8_block128, it binds float32", .{ self.pre, @tagName(mlx.mlx_array_dtype(sc)) });
                    return .{ .fp8 = fp8.bind(self.route.?, wt, sc, @intCast(out), @intCast(in)) catch |e| return refuse(self.diag, e, "{s}" ++ suffix ++ " [{d}, {d}] has no FP8 GEMV plan", .{ self.pre, out, in }) };
                }
            };
            const lg: L = .{ .pre = pre, .map = map, .diag = diag, .c = c, .l = l, .route = fp8_route };
            lw.* = .{
                .input_norm = try checked(lg, "input_layernorm.weight", &.{h}, l, diag),
                .post_norm = try checked(lg, "post_attention_layernorm.weight", &.{h}, l, diag),
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
                    .q_a = try lg.proj("self_attn.q_a_proj", ql, h),
                    .q_a_norm = try lg.get("self_attn.q_a_layernorm.weight", &.{ql}),
                    .q_b = try lg.proj("self_attn.q_b_proj", heads * hd, ql),
                    .kv_a = try lg.proj("self_attn.kv_a_proj_with_mqa", r, h),
                    .kv_a_norm = try lg.get("self_attn.kv_a_layernorm.weight", &.{r}),
                    .wk = try w.own(a, try view(s, kv_b, &.{ heads, 2 * hd, r }, 0, hd)),
                    .wv = try w.own(a, try view(s, kv_b, &.{ heads, 2 * hd, r }, hd, 2 * hd)),
                    .o = try lg.proj("self_attn.o_proj", h, heads * hd),
                    .idx_wq_b = try lg.proj("self_attn.indexer.wq_b", ih * ihd, ql),
                    .idx_wk = try lg.proj("self_attn.indexer.wk", ihd, h),
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
                const a_log = try lg.get("self_attn.A_log", &.{hv});
                const conv_raw = try lg.get("self_attn.conv1d.weight", &.{ 3 * dim, 1, @intCast(c.kda_conv) });
                try dtypeCheck("self_attn.conv1d.weight", mlx.mlx_array_dtype(conv_raw), l, diag);
                lw.attn = .{ .kda = .{
                    .q = try lg.proj("self_attn.q_proj", dim, h),
                    .k = try lg.proj("self_attn.k_proj", dim, h),
                    .v = try lg.proj("self_attn.v_proj", dim, h),
                    .f_a = try lg.proj("self_attn.f_a_proj", dk, h),
                    .f_b = try checked(lg, "self_attn.f_b_proj.weight", &.{ dim, dk }, l, diag),
                    .g_a = try lg.proj("self_attn.g_a_proj", dk, h),
                    .g_b = try checked(lg, "self_attn.g_b_proj.weight", &.{ dim, dk }, l, diag),
                    .b = try lg.proj("self_attn.b_proj", hv, h),
                    .a_log = try w.own(a, try aLogTable(s, a_log, l, diag)),
                    .a_exp = try w.own(a, try aExp(s, a_log, hv, dk)),
                    .dt_bias = try w.own(a, try f32Copy(s, try checked(lg, "self_attn.dt_bias", &.{dim}, l, diag), false)),
                    .conv_w = try w.own(a, try convLayout(s, conv_raw)),
                    .conv_raw = conv_raw,
                    .o_norm = try checked(lg, "self_attn.o_norm.weight", &.{dk}, l, diag),
                    .o = try lg.proj("self_attn.o_proj", h, dim),
                } };
                lw.slot = n_kda;
                n_kda += 1;
            }
            if (c.isMoe(l)) {
                const e: c_int = @intCast(c.n_routed_experts);
                const si: c_int = @intCast(c.moe_intermediate_size * c.n_shared_experts);
                const gate_w = try lg.get("mlp.gate.weight", &.{ e, h });
                try dtypeCheck("mlp.gate.weight", mlx.mlx_array_dtype(gate_w), l, diag);
                lw.router = .{
                    .w = gate_w,
                    .w_t = try w.own(a, try f32Copy(s, gate_w, true)),
                    .bias = try checked(lg, "mlp.gate.e_score_correction_bias", &.{e}, l, diag),
                };
                lw.shared = .{
                    .gate = try lg.proj("mlp.shared_experts.gate_proj", si, h),
                    .up = try lg.proj("mlp.shared_experts.up_proj", si, h),
                    .down = try lg.proj("mlp.shared_experts.down_proj", h, si),
                };
            } else {
                const inter: c_int = @intCast(c.intermediate_size);
                lw.dense = .{
                    .gate = try lg.proj("mlp.gate_proj", inter, h),
                    .up = try lg.proj("mlp.up_proj", inter, h),
                    .down = try lg.proj("mlp.down_proj", h, inter),
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

    /// The gates the expert scheduler's lookahead reads, one per model layer (a dense layer's empty): the router's weight
    /// the installed predictor reads (`decode_predict`: `bf16` the stored BF16 [experts, hidden], `f32` the transposed
    /// F32 copy) and its correction bias.
    pub fn gates(w: *const Weights, a: std.mem.Allocator, predict: settings.DecodePredict) ![]experts.Gate {
        const out = try a.alloc(experts.Gate, w.layers.len);
        for (out, w.layers) |*gt, lw| gt.* = if (lw.router) |r| .{ .w = if (predict == .bf16) r.w else r.w_t, .bias = r.bias } else .{ .w = .{ .ctx = null }, .bias = .{ .ctx = null } };
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

/// The A_log dtypes the bind takes (the pack of record stores F32 on layers 0-2 and BF16 on the rest): the floats the
/// device converts to f32 exactly.
pub fn aLogAccepted(dt: mlx.mlx_dtype) bool {
    return dt == .float32 or dt == .bfloat16 or dt == .float16;
}

/// `layer`'s A_log dtype as the bind takes it; any other refuses the load by name.
fn aLogCheck(dt: mlx.mlx_dtype, layer: u32, diag: ?*sdk.Diag) !void {
    return dtypeCheck("self_attn.A_log", dt, layer, diag);
}

/// Which layers hold a rule's tensor.
pub const RuleLayers = enum { all, kda, moe };
/// A per-layer tensor a route reads in its stored dtype (or converts once at the bind), the layers that hold it, and the
/// dtypes the bind takes for it. One table: the bind's refusals (`dtypeCheck`) and the pack-header test (GLM5_BANK) read
/// it, so a real pack's dtype is checked against every route before a load, never found by a refusal in a window.
pub const DtypeRule = struct { suffix: []const u8, layers: RuleLayers, accepted: []const mlx.mlx_dtype, desc: []const u8 };
const float_dtypes = [_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 };
const float_desc = "a float (F32, BF16 or F16)";
pub const dtype_rules = [_]DtypeRule{
    // KDA: A_log (its own F32 table) and dt_bias (an F32 copy) converted once; the conv, f_b, g_b and o_norm read as
    // stored by the fused decode text (kda_decode.zig: one compile-checked variant per dtype combination) and the op chain.
    .{ .suffix = "self_attn.A_log", .layers = .kda, .accepted = &float_dtypes, .desc = float_desc },
    .{ .suffix = "self_attn.dt_bias", .layers = .kda, .accepted = &float_dtypes, .desc = float_desc },
    .{ .suffix = "self_attn.conv1d.weight", .layers = .kda, .accepted = &float_dtypes, .desc = float_desc },
    .{ .suffix = "self_attn.f_b_proj.weight", .layers = .kda, .accepted = &float_dtypes, .desc = float_desc },
    .{ .suffix = "self_attn.g_b_proj.weight", .layers = .kda, .accepted = &float_dtypes, .desc = float_desc },
    .{ .suffix = "self_attn.o_norm.weight", .layers = .kda, .accepted = &float_dtypes, .desc = float_desc },
    // The mHC norms: hcPre's plan (ops.HcPlan) launches every sublayer's norm in the stream's dtype (BF16).
    .{ .suffix = "input_layernorm.weight", .layers = .all, .accepted = &.{.bfloat16}, .desc = "BF16 (the stream's dtype)" },
    .{ .suffix = "post_attention_layernorm.weight", .layers = .all, .accepted = &.{.bfloat16}, .desc = "BF16 (the stream's dtype)" },
    // The router: the glue's router text and the BF16 predictor read the weight as BF16; the correction bias is F32.
    .{ .suffix = "mlp.gate.weight", .layers = .moe, .accepted = &.{.bfloat16}, .desc = "BF16" },
    .{ .suffix = "mlp.gate.e_score_correction_bias", .layers = .moe, .accepted = &.{.float32}, .desc = "F32" },
};

fn ruleOf(comptime suffix: []const u8) DtypeRule {
    for (dtype_rules) |r| if (std.mem.eql(u8, r.suffix, suffix)) return r;
    @compileError("glm5_next: no dtype rule for " ++ suffix);
}

pub fn accepts(r: DtypeRule, dt: mlx.mlx_dtype) bool {
    return std.mem.indexOfScalar(mlx.mlx_dtype, r.accepted, dt) != null;
}

/// A layer's tensor through the bind's getter, its dtype checked against its rule (`dtypeCheck`).
fn checked(lg: anytype, comptime suffix: []const u8, shape: []const c_int, layer: u32, diag: ?*sdk.Diag) !T {
    const x = try lg.get(suffix, shape);
    try dtypeCheck(suffix, mlx.mlx_array_dtype(x), layer, diag);
    return x;
}

/// `layer`'s `suffix` tensor dtype against its rule; any other refuses the load by name.
pub fn dtypeCheck(comptime suffix: []const u8, dt: mlx.mlx_dtype, layer: u32, diag: ?*sdk.Diag) !void {
    const r = comptime ruleOf(suffix);
    if (!accepts(r, dt)) return refuse(diag, error.Glm5NextResidentDtype, "layer {d} " ++ suffix ++ " is {t}, not " ++ r.desc, .{ layer, dt });
}

/// A_log [heads] of `layer` as its own F32 table, evaluated, once (the per-core prompt route reads it: kda_percore.zig).
fn aLogTable(s: mlx.mlx_stream, a_log: T, layer: u32, diag: ?*sdk.Diag) !T {
    try aLogCheck(mlx.mlx_array_dtype(a_log), layer, diag);
    return f32Copy(s, a_log, false);
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
    pub fn buffers(self: *const Cache, lw: *const Layer, out: *[2]T) []const T {
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
    const act = try sdk_ext.quant.activation(G, g, .{ .swiglu_clamped = c.swiglu_limit }, try fp8.project(g, m.gate, x), try fp8.project(g, m.up, x));
    return fp8.project(g, m.down, act);
}

/// A decode-lane MLP (x [rows, hidden], rows <= 4): the activation on the fused text where the module installed it
/// (`decode_swiglu: fused`, swiglu.zig), the op chain otherwise (`mlp`). The projections are the residents route's either
/// way (fp8.zig `project`: BF16 on the stock matmul, or under `fp8_block128` MiMo's FP8 GEMV at the decode lane's rows).
fn mlpDecode(g: *G, c: *const glm.Config, m: Mlp, x: T, sw: ?*const swiglu.Kernels, width: swiglu.Width, rows: u32) !T {
    const k = sw orelse return mlp(g, c, m, x);
    return fp8.project(g, m.down, try swiglu.activate(g, k, width, try fp8.project(g, m.gate, x), try fp8.project(g, m.up, x), rows));
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

/// Prompt lever 4 (glm5next/prefill-levers): a prompt call's MoE commits, the routing barrier's arrays first (the routed
/// ids and weights: the router's graph over the FFN's collapse and norm), then the shared expert. The routed call's
/// barrier (its host read of the ids) then waits for the router alone, not behind the shared expert's 16K-row GEMMs
/// (~15 ms a layer in the profile), so the call's plans and reads start that much sooner in a drive-bound MoE stage; the
/// GPU runs the shared expert behind the router, ahead of the first group's tiles. The same graphs and values, only the
/// commit order moved: exact. `Ops`: the module's MLX backend (the CPU tests trace it).
/// Prefill host idle (g21c profile: the tail -> next layer's attention gap, 5.2 ms a layer, GPU idle while the host purged
/// MLX's cache after the tail's evaluation): a prompt layer's purge (`Source.promptLayerEnd`: MLX's cache cleared once
/// half full) runs after the next layer's first attention span is committed, so the GPU starts that span first. The
/// cache stays under its limit (MLX frees on allocation past it), so no bill term moves; no value changes: exact.
pub const DeferredPurge = struct {
    pending: bool = false,

    pub fn layerEnd(p: *DeferredPurge) void {
        p.pending = true;
    }

    pub fn afterFirstCommit(p: *DeferredPurge, src: anytype) void {
        if (!p.pending) return;
        p.pending = false;
        src.promptLayerEnd();
    }
};

/// Prefill host idle (g21c profile: the dense stage's GPU idle, ~5-7 ms a routed layer while the host built the 16 spans'
/// collapse and norm graphs with nothing committed): a prompt call's span collapse and norm committed as built. The same
/// graph and values, only the commit point moved: exact. `Ops`: the module's MLX backend (the CPU tests trace it).
pub fn commitSpanNorm(comptime Ops: type, g: *Ops, normed: Ops.T) !void {
    try g.asyncEval(&.{normed});
}

pub fn commitRoutingFirst(comptime Ops: type, g: *Ops, indices: Ops.T, weights: Ops.T, shared: Ops.T) !void {
    try g.asyncEval(&.{ indices, weights });
    try g.asyncEval(&.{shared});
}

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
pub fn attentionSpan(g: *G, c: *const glm.Config, lw: *const Layer, stream: T, start: u32, cache: *Cache, routes: Routes, decode_lane: bool, comptime head: Head) !T {
    var col = try ops.collapse(g.s, stream, lw.hc_attn.fn_rows, lw.hc_attn.scale, lw.hc_attn.base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
    defer col.deinit();
    const normed = try rmsNorm(g, col.x, lw.input_norm, c.rms_norm_eps);
    const out = switch (lw.attn) {
        // The MTP verify captures every row's state (`Cache.rollback`); every other call advances the state alone.
        .kda => |*kw| if (head == .verify)
            try kda.forwardCapture(g, c, kw, normed, &cache.kda[lw.slot], &cache.capture[lw.slot])
        else if (decode_lane)
            try routes.kda.run(routes.kda.ctx, g, c, kw, normed, &cache.kda[lw.slot])
        else
            try kda.forward(g, c, kw, normed, &cache.kda[lw.slot]),
        .dsa => |*dw| try dsa.forward(g, c, dw, normed, start, .{ .latent = &cache.latent[lw.slot], .index = &cache.index[lw.slot] }, routes.dsa),
    };
    return g.adopt(try ops.expand(g.s, out, stream, &col));
}

/// A prompt call's spans at most.
const max_spans = prompt_call_rows / span_rows + 1;

/// P1's predictor of one routed layer for a prompt call (prompt lever 5): the call's predicted top-k counted per expert
/// on the GPU (u32 [experts]: one small read back in place of every row's ids and the host's tally). Built on the layer's
/// input spans at the previous layer's end, committed behind that layer's tail and evaluated with it (`forward`: no host
/// gap at the boundary), or at the layer's own start for a call's first routed layer that no layer of the call precedes.
pub const Ahead = struct {
    layer: ?u32 = null,
    counts: ?T = null,

    pub fn release(a: *Ahead, g: *G) void {
        if (a.counts) |x| g.release(x);
        a.* = .{};
    }
};

/// Routed layer `l`'s predictor over the call's input `spans` (`Ahead`), unevaluated: the router on each span's collapsed
/// attention input (its post-attention norm), every row's top-k compared with each expert id and summed (exact integer
/// counts: the host's tally of the same ids, so the same ranking and the same read-ahead), the spans' counts added. The
/// counts are kept; the graph's intermediates belong to the caller's scope.
pub fn predictor(g: *G, c: *const glm.Config, lw: *const Layer, l: u32, spans: []const T, ahead: *Ahead) !void {
    std.debug.assert(ahead.layer == null and spans.len <= max_spans);
    const r = lw.router.?;
    const n: c_int = @intCast(c.n_routed_experts);
    const k: c_int = @intCast(c.n_experts_per_tok);
    const ids_row = try g.reshape(try g.arange(0, @floatFromInt(n), 1, .uint32), &.{ 1, n });
    var total: ?T = null;
    for (spans) |sp| {
        var col = try ops.collapse(g.s, sp, lw.hc_attn.fn_rows, lw.hc_attn.scale, lw.hc_attn.base, c.hc_iters, c.hc_eps, c.rms_norm_eps);
        defer col.deinit();
        const sh = g.shapeOf(col.x);
        const x = try g.reshape(try rmsNorm(g, col.x, lw.post_norm, c.rms_norm_eps), &.{ sh.dim(1), sh.dim(2) });
        const span_counts = try expertCounts(G, g, (try route(g, c, r, x)).indices, sh.dim(1), k, ids_row);
        total = if (total) |t| try g.add(t, span_counts) else span_counts;
    }
    ahead.counts = g.keep(try g.astype(total.?, .uint32));
    ahead.layer = l;
}

/// Each expert's routed rows in `ids` (u32 [rows, k]): the ids compared with every expert id (`all_ids`, u32 [1, n]) and
/// summed per expert, int32 [n] (integer sums: exactly the host's tally of the same ids). `Ops`: the module's MLX backend
/// (the CPU tests trace it).
pub fn expertCounts(comptime Ops: type, g: *Ops, ids: Ops.T, rows: c_int, k: c_int, all_ids: Ops.T) !Ops.T {
    return g.sum(try g.equal(try g.reshape(ids, &.{ rows * k, 1 }), all_ids), 0, false);
}

/// P1: routed layer `l`'s predicted counts (evaluated) ranked hottest first and posted to the stream, which reads the
/// seed into the layer's free persistent rows while the attention runs. The prediction only picks reads.
fn postReadAhead(g: *G, c: *const glm.Config, ahead: *const Ahead, src: *experts.Source, l: u32) !void {
    std.debug.assert(ahead.layer.? == l);
    const counts = try g.readU32(ahead.counts.?, src.ahead_counts[0..c.n_routed_experts]);
    try src.readAheadSeed(l, expert_policy.rankHottest(counts, &src.ahead_ranked));
}

/// The forward's construction-time routes (Module.init): the DSA kernels' and, under the fused glue, its installed texts
/// (glue.zig; null under `decode_glue: ops`, whose instance never reads it).
pub const Routes = struct {
    dsa: dsa.Routes = .{},
    glue: ?*const glue.Kernels = null,
    /// The decode lane's KDA layer (`decode_kda`, kda.zig `Step`): the op chain unless the module installed the fused
    /// text (kda_decode.zig). The prompt's wide calls and the MTP verify's capture always run the op chain.
    kda: kda.Step = .{},
    /// The decode lane's MLP activation (`decode_swiglu`): the op chain unless the module installed the fused text.
    swiglu: ?*const swiglu.Kernels = null,
};

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
pub fn forward(g: *G, a: std.mem.Allocator, c: *const glm.Config, w: *const Weights, ids: []const u32, start: u32, cache: *Cache, src: *experts.Source, routes: Routes, comptime head: Head, comptime gl: glue.Route, sink: anytype) !T {
    const rows: u32 = @intCast(ids.len);
    if (rows == 0 or rows > prompt_call_rows) return error.Glm5NextCallRows;
    if (start + rows > cache.cap) return error.ContextOverBill;
    // A wide call (the prompt's) or the decode lane's (the source's phase route: one row, or the MTP verify's in decode).
    const prompt = !src.decodeLaneAt(rows);
    // The fused glue (`decode_glue: fused`, the instance the module installed) serves the decode lane; a wide call keeps
    // the stock glue (the phase route).
    if (comptime gl == .fused) if (!prompt) return forwardFused(g, c, w, ids, start, cache, src, routes, head, sink);
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
    // Prompt lever 5: the next routed layer's predictor, built and evaluated with each layer's output.
    var ahead: Ahead = .{};
    defer ahead.release(g);
    // Prefill host idle: a prompt layer's cache purge runs after the next layer's first span is committed.
    var purge: DeferredPurge = .{};
    for (w.layers, 0..) |*lw, li| {
        const l: u32 = @intCast(li);
        const m = g.mark();
        defer g.resetTo(m);
        if (comptime ptl.enabled) {
            ptl.layer(l);
            ptl.stage(.predict);
        }
        if (prompt and lw.router != null) {
            // Built at the previous layer's end; a call's first routed layer with no earlier layer in the call builds it here.
            if (ahead.layer == null) {
                try predictor(g, c, lw, l, spans, &ahead);
                try g.evalAll(&.{ahead.counts.?});
            }
            try postReadAhead(g, c, &ahead, src, l);
            ahead.release(g);
        }
        if (comptime ptl.enabled) ptl.stage(.attn);
        // The attention, span by span (the KDA state and the DSA lanes carry from one span to the next); in a prompt
        // call each span is committed as built and the span before it waited, so at most `spans_in_flight` run at once.
        var prev: ?T = null;
        for (spans, 0..) |*sp, i| {
            const ms = g.mark();
            const next = g.keep(try attentionSpan(g, c, lw, sp.*, start + @as(u32, @intCast(i * span_rows)), cache, routes, !prompt, head));
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
            if (i == 0) purge.afterFirstCommit(src);
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
            // Prefill host idle: a prompt call commits each span's collapse and norm as built (the GPU runs them while
            // the host builds the next span's), as the attention loop commits its spans; decode stays lazy.
            if (prompt) try commitSpanNorm(G, g, nx.*);
        }
        const x = try g.reshape(if (n_spans == 1) normed[0] else try g.concat(normed, 1), &.{ @intCast(rows), hidden });
        const f = if (lw.dense) |d| (if (prompt) try mlpTiles(g, a, c, d, x, rows) else try mlpDecode(g, c, d, x, routes.swiglu, .dense, rows)) else blk: {
            const r = try route(g, c, lw.router.?, x);
            const shared = if (prompt) try mlp(g, c, lw.shared.?, x) else try mlpDecode(g, c, lw.shared.?, x, routes.swiglu, .shared, rows);
            // Prompt lever 4: a prompt call commits the routing barrier's arrays, then the shared expert (the GPU runs it
            // behind the router while the host routes and the first group's reads land); a decode-lane call hands it over
            // as the call's hoist (run during its read wait). The sum is the same either way.
            if (prompt) try commitRoutingFirst(G, g, r.indices, r.weights, shared);
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
        // The expands hold the collapses they read.
        for (cols[0..n_cols]) |*cl| cl.deinit();
        n_cols = 0;
        // MiMo's tail: a prompt layer's output evaluated before the next layer starts (bounding its graph), then the
        // pooled scratch's purge. A decode step's stays lazy: the next routing barrier evaluates it. Prompt lever 5: the
        // tail is committed, the next routed layer's predictor built on its output while the GPU runs it, then committed
        // behind it and evaluated with it (no host gap at the boundary). The layer's own arrays are released first (the
        // tail's graph holds what it reads): the boundary holds the stream, the tail's inputs and the predictor's arrays,
        // not the MoE stage's (bill.zig `promptWaveBytes`: `boundary`).
        if (prompt) {
            g.resetTo(m);
            try g.asyncEval(spans);
            var outs: [max_spans + 1]T = undefined;
            @memcpy(outs[0..n_spans], spans);
            var n_outs = n_spans;
            if (li + 1 < w.layers.len and w.layers[li + 1].router != null) {
                try predictor(g, c, &w.layers[li + 1], l + 1, spans, &ahead);
                outs[n_outs] = ahead.counts.?;
                n_outs += 1;
            }
            try g.evalAll(outs[0..n_outs]);
            purge.layerEnd();
            // `fp8_block128`: the layer's BF16 weights released (no later layer reads them; fp8.zig `promptBytes`).
            if (w.fp8) |r| r.layerEnd();
        }
    }
    purge.afterFirstCommit(src);
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

/// A sublayer's expand not applied yet (the fused glue): its output over its input stream with its gates, kept across the
/// layer scope's resets until the next sublayer's hcPre folds it in.
const Pending = struct {
    deferred: ops.Deferred,

    fn of(g: *G, y: T, post: T, comb: T, h: T) Pending {
        return .{ .deferred = .{ .y = g.keep(y), .post = g.keep(post), .comb = g.keep(comb), .h = g.keep(h) } };
    }

    fn release(p: *Pending, g: *G) void {
        for ([_]T{ p.deferred.y, p.deferred.post, p.deferred.comb, p.deferred.h }) |x| g.release(x);
    }
};

/// The decode lane under the fused glue (`decode_glue: fused`, glue.zig): `forward`'s layers with each sublayer's collapse
/// and norm in one hcPre dispatch that also applies the previous sublayer's expand (deferred until the next sublayer reads
/// it; the last one is materialized for the head), the router's product and top-8 in two texts, the routed combine with
/// the shared expert's add in one. The decode lane's rows are one span (at most `glue.max_rows`, hcPre's).
fn forwardFused(g: *G, c: *const glm.Config, w: *const Weights, ids: []const u32, start: u32, cache: *Cache, src: *experts.Source, routes: Routes, comptime head: Head, sink: anytype) !T {
    if (comptime head == .dump) @compileError("glm5_next: the logits dump is a prompt pass (the stock glue)");
    const k = routes.glue.?;
    const rows: u32 = @intCast(ids.len);
    const m0 = g.mark();
    defer g.resetTo(m0);
    const hidden: c_int = @intCast(c.hidden_size);
    const hc: c_int = @intCast(c.hc_mult);
    const r: c_int = @intCast(rows);
    // The stream entering the first sublayer, [1, R, hc, hidden]; each deferral's rebuild (hcPre's `h`) replaces it.
    var stream: T = blk: {
        const idx = try g.hostArray(std.mem.sliceAsBytes(ids), &.{r}, .uint32);
        const e = try g.reshape(try g.take(w.embed, idx, 0), &.{ 1, r, 1, hidden });
        break :blk g.keep(try g.broadcastTo(e, &.{ 1, r, hc, hidden }));
    };
    defer g.release(stream);
    var pend: ?Pending = null;
    defer if (pend) |*p| p.release(g);
    for (w.layers, 0..) |*lw, li| {
        const l: u32 = @intCast(li);
        const m = g.mark();
        defer g.resetTo(m);
        // The attention sublayer: its collapse and norm, with the previous sublayer's expand (none before layer 0).
        const pa = if (pend) |*p|
            try glue.pre(g, k, null, &p.deferred, lw.hc_attn.fn_rows, lw.hc_attn.scale, lw.hc_attn.base, lw.input_norm)
        else
            try glue.pre(g, k, stream, null, lw.hc_attn.fn_rows, lw.hc_attn.scale, lw.hc_attn.base, lw.input_norm);
        if (pend) |*p| {
            p.release(g);
            pend = null;
            g.release(stream);
            stream = g.keep(pa.h.?);
        }
        const out = switch (lw.attn) {
            // The MTP verify captures every row's state (`Cache.rollback`); every other call advances the state alone.
            .kda => |*kw| if (head == .verify)
                try kda.forwardCapture(g, c, kw, pa.normed, &cache.kda[lw.slot], &cache.capture[lw.slot])
            else
                try routes.kda.run(routes.kda.ctx, g, c, kw, pa.normed, &cache.kda[lw.slot]),
            .dsa => |*dw| try dsa.forward(g, c, dw, pa.normed, start, .{ .latent = &cache.latent[lw.slot], .index = &cache.index[lw.slot] }, routes.dsa),
        };
        pend = Pending.of(g, out, pa.post, pa.comb, stream);
        // The FFN sublayer: the attention's expand folded into its collapse and norm.
        const pf = try glue.pre(g, k, null, &pend.?.deferred, lw.hc_ffn.fn_rows, lw.hc_ffn.scale, lw.hc_ffn.base, lw.post_norm);
        pend.?.release(g);
        pend = null;
        g.release(stream);
        stream = g.keep(pf.h.?);
        const x = try g.reshape(pf.normed, &.{ r, hidden });
        const f = if (lw.dense) |d| try mlpDecode(g, c, d, x, routes.swiglu, .dense, rows) else blk: {
            const rt = lw.router.?;
            const routing = try glue.route(g, k, rt.w, rt.bias, x, rows);
            // The shared expert is the call's hoist (committed during its read wait); the combine text adds it.
            const shared = try mlpDecode(g, c, lw.shared.?, x, routes.swiglu, .shared, rows);
            break :blk try g.adopt(try src.forwardShared(l, x, routing.ids, routing.weights, shared, k));
        };
        pend = Pending.of(g, try g.reshape(f, &.{ 1, r, hidden }), pf.post, pf.comb, stream);
    }
    cache.len = start + rows;
    // The last sublayer's expand, materialized: the stream the head reads.
    const final = try g.adopt(try ops.materialize(g.s, &k.hc, &pend.?.deferred));
    const spans = [_]T{final};
    if (head == .hidden or head == .verify) {
        const final_rows = try normedRows(g, c, w, &spans, rows);
        sink.hidden = g.keep(final_rows);
        if (head == .verify) return g.keep(try g.astype(try linear(g, final_rows, w.head), .float32));
    }
    const row = try g.slice(final, &.{ 0, r - 1, 0, 0 }, &.{ 1, r, hc, hidden }, &.{ 1, 1, 1, 1 });
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

test {
    // The fused decode texts' CPU tests and their window parity tests (skipped without their switches); the FP8 residents
    // route's likewise.
    _ = @import("glue.zig");
    _ = @import("glue_parity.zig");
    _ = @import("kda_decode.zig");
    _ = @import("kda_decode_parity.zig");
    _ = @import("dsa_decode.zig");
    _ = @import("dsa_decode_parity.zig");
    _ = @import("swiglu.zig");
    _ = @import("fp8.zig");
    _ = @import("fp8_parity.zig");
    _ = @import("determinism_gpu.zig");
}

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

test "glm5_next graph: prompt lever 4 commits the routing barrier's ids and weights, then the shared expert (two commits, in that order; nothing else)" {
    const Trace = ops_be.TraceOps;
    var g = Trace.init(testing.allocator);
    defer g.deinit();
    const ids = try g.input(&.{ 16, 8 }, .uint32);
    const weights = try g.input(&.{ 16, 8 }, .float32);
    const shared = try g.input(&.{ 16, 4096 }, .bfloat16);
    const nodes = g.nodes.items.len;
    try commitRoutingFirst(Trace, &g, ids, weights, shared);
    try testing.expectEqual(@as(usize, 2), g.commit_spans.items.len);
    try testing.expectEqualSlices(Trace.T, &.{ ids, weights }, g.committed.items[g.commit_spans.items[0][0]..g.commit_spans.items[0][1]]);
    try testing.expectEqualSlices(Trace.T, &.{shared}, g.committed.items[g.commit_spans.items[1][0]..g.commit_spans.items[1][1]]);
    // Only the two commit markers: no node of a graph is built or evaluated here.
    try testing.expectEqual(nodes + 2, g.nodes.items.len);
    try testing.expectEqual(@as(usize, 0), g.evals.items.len);
}

test "glm5_next graph: prefill host idle's deferred purge runs once, after the next layer's first commit, and at the pass's end" {
    const Fake = struct {
        purges: u32 = 0,
        pub fn promptLayerEnd(f: *@This()) void {
            f.purges += 1;
        }
    };
    var src: Fake = .{};
    var p: DeferredPurge = .{};
    p.afterFirstCommit(&src); // layer 0's first span: nothing pending
    try testing.expectEqual(@as(u32, 0), src.purges);
    p.layerEnd(); // layer 0's end: deferred
    try testing.expectEqual(@as(u32, 0), src.purges);
    p.afterFirstCommit(&src); // layer 1's first span committed: layer 0's purge
    try testing.expectEqual(@as(u32, 1), src.purges);
    p.afterFirstCommit(&src); // no second purge
    try testing.expectEqual(@as(u32, 1), src.purges);
    p.layerEnd();
    p.afterFirstCommit(&src); // the pass's end
    try testing.expectEqual(@as(u32, 2), src.purges);
}

test "glm5_next graph: prefill host idle commits a span's collapse and norm alone, as built (one commit; nothing evaluated)" {
    const Trace = ops_be.TraceOps;
    var g = Trace.init(testing.allocator);
    defer g.deinit();
    const normed = try g.input(&.{ 1, 1024, 4096 }, .bfloat16);
    const nodes = g.nodes.items.len;
    try commitSpanNorm(Trace, &g, normed);
    try testing.expectEqual(@as(usize, 1), g.commit_spans.items.len);
    try testing.expectEqualSlices(Trace.T, &.{normed}, g.committed.items[g.commit_spans.items[0][0]..g.commit_spans.items[0][1]]);
    try testing.expectEqual(nodes + 1, g.nodes.items.len);
    try testing.expectEqual(@as(usize, 0), g.evals.items.len);
}

test "glm5_next graph: prompt lever 5's expert counts compare every row's top-k with each expert id and sum per expert: int32 [experts] (the trace backend: no device)" {
    const Trace = ops_be.TraceOps;
    var g = Trace.init(testing.allocator);
    defer g.deinit();
    const ids = try g.input(&.{ 1024, 8 }, .uint32);
    const all_ids = try g.input(&.{ 1, 288 }, .uint32);
    const counts = try expertCounts(Trace, &g, ids, 1024, 8, all_ids);
    try testing.expectEqualSlices(c_int, &.{288}, g.shapeOf(counts).slice());
    try testing.expectEqual(mlx.mlx_dtype.int32, g.dtypeOf(counts));
    try testing.expectEqual(@as(usize, 0), g.evals.items.len);
}

test "glm5_next graph: prompt lever 5's expert counts in MLX (two spans added, u32) are the host's tally of the same ids, so the read-ahead's ranking is the control's (guarded window: MLX's CPU stream creates the Metal device)" {
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    const n: u32 = 288;
    const k: u32 = 8;
    const rows: u32 = 96;
    var prng = std.Random.DefaultPrng.init(0x51ed);
    const rng = prng.random();
    var want: [n]u32 = @splat(0);
    const all_ids = try g.reshape(try g.arange(0, @floatFromInt(n), 1, .uint32), &.{ 1, n });
    var total: ?T = null;
    for (0..2) |_| {
        // Every row's top-k distinct, skewed toward low ids (hot experts) so the ranking has ties and gaps.
        var ids: [rows * k]u32 = undefined;
        for (0..rows) |r| {
            var j: usize = 0;
            while (j < k) {
                const u = rng.float(f32);
                const e: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(n - 1)), u * u * @as(f32, @floatFromInt(n))));
                if (std.mem.indexOfScalar(u32, ids[r * k ..][0..j], e) != null) continue;
                ids[r * k + j] = e;
                want[e] += 1;
                j += 1;
            }
        }
        const x = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ rows, k }, .uint32);
        const span_counts = try expertCounts(G, &g, x, rows, k, all_ids);
        total = if (total) |t| try g.add(t, span_counts) else span_counts;
    }
    const counts = try g.astype(total.?, .uint32);
    try g.evalAll(&.{counts});
    var got: [n]u32 = undefined;
    try testing.expectEqualSlices(u32, &want, try g.readU32(counts, &got));
    var rank_want: [n]u16 = undefined;
    var rank_got: [n]u16 = undefined;
    try testing.expectEqualSlices(u16, expert_policy.rankHottest(&want, &rank_want), expert_policy.rankHottest(&got, &rank_got));
}

test "glm5_next graph: the owned f32 tables' bytes and the call geometry" {
    const c: glm.Config = .{};
    const page: u64 = 16384;
    const o = Weights.ownedBytes(&c, page);
    // 90 hyper-connections ([24, 16384] + [24] + [3] f32), 34 KDA layers' exp(A_log) and dt_bias [8192], A_log [64] and
    // conv [24576, 4], 11 DSA layers' pool gate [4096, 128] and ape [4, 128]; 42 routers [4096, 288] f32.
    try testing.expectEqual(@as(u64, 90 * (1_572_864 + 96 + 12) + 34 * (2 * 32_768 + 256 + 393_216) + 11 * (2_097_152 + 2048)), o.tables);
    try testing.expectEqual(@as(u64, 45 * 6 + 34 * 4 + 11 * 4 + 42), Weights.ownedArrays(&c));
    try testing.expectEqual(@as(u64, 42 * 4_718_592), o.routers);
    try testing.expectEqual(@as(u32, 16_384), prompt_call_rows);
    try testing.expect(prompt_call_rows % span_rows == 0 and dump_chunk_rows <= span_rows);
}

test "glm5_next graph: the bind takes A_log in F32 and BF16 (the pack's dtypes) and F16, and refuses any other dtype by name" {
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 }) |dt| try aLogCheck(dt, 4, null);
    for ([_]mlx.mlx_dtype{ .bool_, .uint8, .uint32, .int8, .int32, .int64, .float64, .complex64 }) |dt| {
        var diag: sdk.Diag = .{};
        try testing.expectError(error.Glm5NextResidentDtype, aLogCheck(dt, 7, &diag));
        var buf: [96]u8 = undefined;
        const want = try std.fmt.bufPrint(&buf, "layer 7 self_attn.A_log is {t}, not a float", .{dt});
        try testing.expect(std.mem.indexOf(u8, diag.message(), want) != null);
    }
}

/// A safetensors dtype name as MLX's dtype (null: none).
fn safetensorsDtype(name: []const u8) ?mlx.mlx_dtype {
    const table = [_]struct { []const u8, mlx.mlx_dtype }{
        .{ "BOOL", .bool_ },  .{ "U8", .uint8 },     .{ "U16", .uint16 }, .{ "U32", .uint32 }, .{ "U64", .uint64 },
        .{ "I8", .int8 },     .{ "I16", .int16 },    .{ "I32", .int32 },  .{ "I64", .int64 },  .{ "F16", .float16 },
        .{ "F32", .float32 }, .{ "F64", .float64 },  .{ "BF16", .bfloat16 },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], name)) return e[1];
    return null;
}

/// A shard's safetensors header, read past the page cache (F_NOCACHE) and parsed.
fn shardHeader(a: std.mem.Allocator, dir: []const u8, shard: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, shard }, 0);
    defer a.free(path);
    const fd = try sdk.io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    var prefix: [8]u8 = undefined;
    if (std.c.pread(fd, &prefix, prefix.len, 0) != prefix.len) return error.Glm5NextResidentShard;
    const len = std.mem.readInt(u64, &prefix, .little);
    if (len < 2 or len > 64 << 20) return error.Glm5NextResidentShard;
    const header = try a.alloc(u8, @intCast(len));
    defer a.free(header);
    var done: usize = 0;
    while (done < header.len) {
        const n = std.c.pread(fd, header[done..].ptr, header.len - done, @intCast(8 + done));
        if (n <= 0) return error.Glm5NextResidentShard;
        done += @intCast(n);
    }
    return std.json.parseFromSlice(std.json.Value, a, header, .{ .allocate = .alloc_always });
}

// GLM5_BANK=<pack> (headers only, no payload): every KDA layer's A_log dtype as the pack's shards declare it, each one the
// bind converts; the dtype histogram printed (the pack of record: F32 on layers 0-2, BF16 on the other 31).
test "glm5_next graph: every KDA layer's A_log dtype in the pack's headers is one the bind converts (GLM5_BANK)" {
    const dir = std.mem.span(std.c.getenv("GLM5_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const config_text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(config_text);
    var cdiag: glm.Diag = .{};
    const c = glm.Config.parse(a, config_text, &cdiag) catch |e| {
        std.debug.print("config refused: {s}\n", .{cdiag.message()});
        return e;
    };
    const raw = try d.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20));
    defer a.free(raw);
    const index = try std.json.parseFromSlice(std.json.Value, a, raw, .{ .allocate = .alloc_always });
    defer index.deinit();
    const wm = index.value.object.get("weight_map").?.object;
    var headers: std.StringArrayHashMapUnmanaged(std.json.Parsed(std.json.Value)) = .empty;
    defer {
        for (headers.values()) |*h| h.deinit();
        headers.deinit(a);
    }
    var hist: std.StringArrayHashMapUnmanaged(u32) = .empty;
    defer hist.deinit(a);
    var n: u32 = 0;
    var l: u32 = 0;
    while (l < c.n_layers) : (l += 1) {
        if (c.isDsa(l)) continue;
        var nb: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.self_attn.A_log", .{l});
        const shard = (wm.get(name) orelse return error.Glm5NextResidentMissing).string;
        if (headers.get(shard) == null) {
            const header = try shardHeader(a, dir, shard);
            errdefer header.deinit();
            try headers.put(a, shard, header);
        }
        const desc = headers.get(shard).?.value.object.get(name) orelse return error.Glm5NextResidentMissing;
        const dtype = desc.object.get("dtype").?.string;
        const dt = safetensorsDtype(dtype) orelse {
            std.debug.print("layer {d} self_attn.A_log: dtype {s} has no MLX dtype\n", .{ l, dtype });
            return error.Glm5NextResidentDtype;
        };
        var ldiag: sdk.Diag = .{};
        aLogCheck(dt, l, &ldiag) catch |e| {
            std.debug.print("{s}\n", .{ldiag.message()});
            return e;
        };
        const h = try hist.getOrPut(a, dtype);
        if (!h.found_existing) h.value_ptr.* = 0;
        h.value_ptr.* += 1;
        n += 1;
    }
    try testing.expectEqual(c.nKda(), n);
    std.debug.print("\nGLM5_NEXT A_LOG pack={s} kda_layers={d} dtypes:", .{ dir, n });
    for (hist.keys(), hist.values()) |k, v| std.debug.print(" {s}={d}", .{ k, v });
    std.debug.print("\n", .{});
}

test "glm5_next graph: the dtype rules take each of Flash's float dtypes where a route reads floats, and name the tensor otherwise" {
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 }) |dt| try dtypeCheck("self_attn.conv1d.weight", dt, 0, null);
    var diag: sdk.Diag = .{};
    try testing.expectError(error.Glm5NextResidentDtype, dtypeCheck("self_attn.conv1d.weight", .uint8, 2, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "layer 2 self_attn.conv1d.weight is uint8, not a float (F32, BF16 or F16)") != null);
    try testing.expectError(error.Glm5NextResidentDtype, dtypeCheck("mlp.gate.weight", .float32, 3, &diag));
    try testing.expectError(error.Glm5NextResidentDtype, dtypeCheck("mlp.gate.e_score_correction_bias", .bfloat16, 3, &diag));
}

// GLM5_BANK=<pack> (headers only, no payload): every per-layer tensor of `dtype_rules` (the tensors the fused routes and
// the bind's conversions read as stored) on every layer that holds it, its dtype as the pack's shards declare it, against
// its rule: the check the bind makes at the load, made on the real pack before any window. The histograms printed (the
// pack of record: A_log, dt_bias and conv1d F32 on layers 0-2, BF16 on the other 31).
test "glm5_next graph: every tensor the routes bind has a dtype its rule takes in the pack's headers (GLM5_BANK)" {
    const dir = std.mem.span(std.c.getenv("GLM5_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const config_text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(config_text);
    var cdiag: glm.Diag = .{};
    const c = glm.Config.parse(a, config_text, &cdiag) catch |e| {
        std.debug.print("config refused: {s}\n", .{cdiag.message()});
        return e;
    };
    const raw = try d.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20));
    defer a.free(raw);
    const index = try std.json.parseFromSlice(std.json.Value, a, raw, .{ .allocate = .alloc_always });
    defer index.deinit();
    const wm = index.value.object.get("weight_map").?.object;
    var headers: std.StringArrayHashMapUnmanaged(std.json.Parsed(std.json.Value)) = .empty;
    defer {
        for (headers.values()) |*h| h.deinit();
        headers.deinit(a);
    }
    var bad: u32 = 0;
    std.debug.print("\nGLM5_NEXT DTYPE_RULES pack={s}\n", .{dir});
    for (dtype_rules) |r| {
        var counts: [3]u32 = @splat(0);
        var other: u32 = 0;
        var l: u32 = 0;
        while (l < c.n_layers) : (l += 1) {
            const holds = switch (r.layers) {
                .all => true,
                .kda => !c.isDsa(l),
                .moe => c.isMoe(l),
            };
            if (!holds) continue;
            var nb: [128]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.{s}", .{ l, r.suffix });
            const shard = (wm.get(name) orelse {
                std.debug.print("  {s}: missing from the index\n", .{name});
                return error.Glm5NextResidentMissing;
            }).string;
            if (headers.get(shard) == null) {
                const header = try shardHeader(a, dir, shard);
                errdefer header.deinit();
                try headers.put(a, shard, header);
            }
            const desc = headers.get(shard).?.value.object.get(name) orelse return error.Glm5NextResidentMissing;
            const dtype = desc.object.get("dtype").?.string;
            const dt = safetensorsDtype(dtype) orelse .complex64;
            if (!accepts(r, dt)) {
                std.debug.print("  REFUSED layer {d} {s} is {s}, not {s}\n", .{ l, r.suffix, dtype, r.desc });
                bad += 1;
            }
            switch (dt) {
                .float32 => counts[0] += 1,
                .bfloat16 => counts[1] += 1,
                .float16 => counts[2] += 1,
                else => other += 1,
            }
        }
        std.debug.print("  {s}: F32={d} BF16={d} F16={d} other={d}\n", .{ r.suffix, counts[0], counts[1], counts[2], other });
    }
    try testing.expectEqual(@as(u32, 0), bad);
}
