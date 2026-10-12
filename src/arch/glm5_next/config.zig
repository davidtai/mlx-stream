//! GLM-5.3-Flash (`glm5_next`, Glm5NextForConditionalGeneration's text model) as mlx-stream serves it: the pack's
//! config.json, parsed once at construction and refused by name where it leaves the served form. 45 layers: Kimi Delta
//! Attention (KDA, linear) on 34, DeepSeek sparse attention (DSA: absorbed NoPE MLA over a 512 latent, a pooled indexer)
//! on the rest, every sublayer inside DeepSeek-V4's 4-stream Sinkhorn hyper-connection (mHC); a dense SwiGLU MLP on the
//! first `first_k_dense` layers, then a sigmoid noaux_tc MoE (288 routed experts, top-8, one ungated shared expert); every
//! SwiGLU clamped at `swiglu_limit`. The vision tower is not served.

const std = @import("std");

pub const max_layers = 64;
/// The routed experts' quantization the bank holds (bank.zig): MLX's nvfp4.
pub const default_expert_rows: u32 = 8;

pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,
    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Error = error{ OutOfMemory, Glm5NextConfigSyntax, Glm5NextConfigField, Glm5NextConfigUnsupported, Glm5NextResidentsFormat };

/// The trunk's residents (the pack's top-level `residents_format`, fp8.zig): `bf16` (absent: every trunk projection BF16)
/// or `fp8_block128` (each trunk projection the pack's `quantization` declares FP8 as e4m3fn codes U8 [out, in] with F32
/// [out / 64, in / 128] scales; every other one BF16). The same two names say how one projection is stored.
pub const ResidentsFormat = enum { bf16, fp8_block128 };
/// One FP8 scale per 64 output rows x 128 input columns (the release-style grid MiMo's fp8res texts read).
pub const fp8_scale_rows: u32 = 64;
pub const fp8_scale_cols: u32 = 128;

/// One trunk projection the FP8 route serves: its name after the layer's prefix (without `.weight`) and its [out, in].
pub const Fp8Site = struct { suffix: []const u8, out: u32, in: u32 };

fn refuse(diag: ?*Diag, err: Error, comptime fmt: []const u8, args: anytype) Error {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, "glm5_next config: " ++ fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

pub const Config = struct {
    vocab_size: u32 = 154880,
    hidden_size: u32 = 4096,
    /// The dense MLP's width (the first `first_k_dense` layers).
    intermediate_size: u32 = 12288,
    moe_intermediate_size: u32 = 2048,
    n_layers: u32 = 45,
    /// DSA: query heads, the query's low rank, the latent, the per-head width (qk_nope_head_dim == v_head_dim, NoPE).
    n_heads: u32 = 64,
    q_lora_rank: u32 = 1536,
    kv_lora_rank: u32 = 512,
    head_dim: u32 = 256,
    /// The pooled indexer: heads, width, the tokens a query reads and the pool they are picked in.
    index_n_heads: u32 = 32,
    index_head_dim: u32 = 128,
    index_topk: u32 = 2048,
    index_kpool: u32 = 4,
    /// KDA: heads (key and value alike), head width (dk == dv), the short conv's taps, the gate's lower bound.
    kda_heads: u32 = 64,
    kda_head_dim: u32 = 128,
    kda_conv: u32 = 4,
    kda_lower_bound: f32 = -5.0,
    n_routed_experts: u32 = 288,
    n_experts_per_tok: u32 = 8,
    n_shared_experts: u32 = 1,
    first_k_dense: u32 = 3,
    routed_scaling: f32 = 2.5,
    norm_topk_prob: bool = true,
    swiglu_limit: f32 = 10.0,
    rms_norm_eps: f32 = 1e-5,
    hc_mult: u32 = 4,
    hc_iters: u32 = 20,
    hc_eps: f32 = 1e-6,
    max_position_embeddings: u64 = 1 << 20,
    tie_word_embeddings: bool = false,
    eos: [4]u32 = .{ 154820, 154827, 154829, 0 },
    n_eos: u8 = 3,
    /// Bit l set: layer l is a DSA layer (the rest KDA). Flash: 3, 7, ..., 43.
    dsa_mask: u64 = flash_dsa_mask,
    /// The trunk's residents (`residents_format`).
    residents_format: ResidentsFormat = .bf16,
    /// Bit i of layer l: the layer's site i (`fp8Sites` then `fp8MlpSites`) is declared FP8 (only under `fp8_block128`).
    fp8_mask: [max_layers]u16 = @splat(0),
    /// The MTP layer declares FP8 projections (the draft lane binds BF16 residents: refused with `mtp_lane`).
    mtp_fp8: bool = false,

    pub const flash_dsa_mask: u64 = blk: {
        var m: u64 = 0;
        var l: u6 = 3;
        while (l < 45) : (l += 4) m |= @as(u64, 1) << l;
        break :blk m;
    };

    pub fn isDsa(c: *const Config, layer: u32) bool {
        return (c.dsa_mask >> @intCast(layer)) & 1 == 1;
    }

    pub fn isMoe(c: *const Config, layer: u32) bool {
        return layer >= c.first_k_dense;
    }

    pub fn nDsa(c: *const Config) u32 {
        return @popCount(c.dsa_mask & ((@as(u64, 1) << @intCast(c.n_layers)) - 1));
    }

    pub fn nKda(c: *const Config) u32 {
        return c.n_layers - c.nDsa();
    }

    pub fn moeLayers(c: *const Config) u32 {
        return c.n_layers - c.first_k_dense;
    }

    /// The KDA projections' widths: q, k and v each `kda_heads x kda_head_dim`; the short conv runs over all three.
    pub fn kdaDim(c: *const Config) u32 {
        return c.kda_heads * c.kda_head_dim;
    }

    pub fn kdaConvDim(c: *const Config) u32 {
        return 3 * c.kdaDim();
    }

    /// Layer `l`'s attention projections the FP8 route serves (KDA: q, k, v, o and the gate's and beta's input stages;
    /// DSA: q_a, q_b, kv_a, o and the indexer's wk, wq_b). KDA's f_b / g_b (K = 128), kv_b (used per head), the indexer's
    /// weights_proj (32 rows) and pool gate stay BF16.
    pub fn fp8Sites(c: *const Config, l: u32, out: *[7]Fp8Site) []const Fp8Site {
        const h = c.hidden_size;
        if (c.isDsa(l)) {
            out[0..6].* = .{
                .{ .suffix = "self_attn.q_a_proj", .out = c.q_lora_rank, .in = h },
                .{ .suffix = "self_attn.q_b_proj", .out = c.n_heads * c.head_dim, .in = c.q_lora_rank },
                .{ .suffix = "self_attn.kv_a_proj_with_mqa", .out = c.kv_lora_rank, .in = h },
                .{ .suffix = "self_attn.o_proj", .out = h, .in = c.n_heads * c.head_dim },
                .{ .suffix = "self_attn.indexer.wk", .out = c.index_head_dim, .in = h },
                .{ .suffix = "self_attn.indexer.wq_b", .out = c.index_n_heads * c.index_head_dim, .in = c.q_lora_rank },
            };
            return out[0..6];
        }
        const dim = c.kdaDim();
        out.* = .{
            .{ .suffix = "self_attn.q_proj", .out = dim, .in = h },
            .{ .suffix = "self_attn.k_proj", .out = dim, .in = h },
            .{ .suffix = "self_attn.v_proj", .out = dim, .in = h },
            .{ .suffix = "self_attn.o_proj", .out = h, .in = dim },
            .{ .suffix = "self_attn.f_a_proj", .out = c.kda_head_dim, .in = h },
            .{ .suffix = "self_attn.g_a_proj", .out = c.kda_head_dim, .in = h },
            .{ .suffix = "self_attn.b_proj", .out = c.kda_heads, .in = h },
        };
        return out[0..7];
    }

    /// Layer `l`'s MLP projections the FP8 route serves: the dense MLP's or the shared expert's gate, up and down.
    pub fn fp8MlpSites(c: *const Config, l: u32) [3]Fp8Site {
        const h = c.hidden_size;
        if (c.isMoe(l)) {
            const si = c.moe_intermediate_size * c.n_shared_experts;
            return .{
                .{ .suffix = "mlp.shared_experts.gate_proj", .out = si, .in = h },
                .{ .suffix = "mlp.shared_experts.up_proj", .out = si, .in = h },
                .{ .suffix = "mlp.shared_experts.down_proj", .out = h, .in = si },
            };
        }
        return .{
            .{ .suffix = "mlp.gate_proj", .out = c.intermediate_size, .in = h },
            .{ .suffix = "mlp.up_proj", .out = c.intermediate_size, .in = h },
            .{ .suffix = "mlp.down_proj", .out = h, .in = c.intermediate_size },
        };
    }

    /// Layer `l`'s site index of `suffix` (its `fp8Sites`, then its `fp8MlpSites`), or null when `suffix` names none of
    /// the projections the FP8 route can serve.
    pub fn siteIndex(c: *const Config, l: u32, suffix: []const u8) ?u4 {
        var buf: [7]Fp8Site = undefined;
        const attn = c.fp8Sites(l, &buf);
        for (attn, 0..) |s, i| if (std.mem.eql(u8, s.suffix, suffix)) return @intCast(i);
        for (c.fp8MlpSites(l), 0..) |s, i| if (std.mem.eql(u8, s.suffix, suffix)) return @intCast(attn.len + i);
        return null;
    }

    /// Whether `suffix` names one of layer `l`'s sites the FP8 route can serve.
    pub fn isFp8Site(c: *const Config, l: u32, suffix: []const u8) bool {
        return c.siteIndex(l, suffix) != null;
    }

    /// Whether the pack declares layer `l`'s site `suffix` FP8 (else it is BF16).
    pub fn siteIsFp8(c: *const Config, l: u32, suffix: []const u8) bool {
        const i = c.siteIndex(l, suffix) orelse return false;
        return (c.fp8_mask[l] >> i) & 1 == 1;
    }

    /// The trunk's sites the FP8 route can serve, over every layer (439 on Flash).
    pub fn fp8TrunkSites(c: *const Config) u32 {
        var n: u32 = 0;
        var buf: [7]Fp8Site = undefined;
        for (0..c.n_layers) |l| n += @intCast(c.fp8Sites(@intCast(l), &buf).len + 3);
        return n;
    }

    /// The trunk projections the pack declares FP8 (0 under `bf16`; the full set 439, the hybrid set 190).
    pub fn fp8Declared(c: *const Config) u32 {
        var n: u32 = 0;
        for (c.fp8_mask[0..c.n_layers]) |m| n += @popCount(m);
        return n;
    }

    /// The bytes one position adds to the DSA caches: each layer's latent and its indexer row (key | gate), BF16.
    pub fn kvBytesPerToken(c: *const Config) u64 {
        return @as(u64, c.nDsa()) * (c.kv_lora_rank + 2 * c.index_head_dim) * 2;
    }

    /// The KDA states, fixed per request: each layer's f32 recurrent state [heads, dv, dk] (the reference's dtype) and
    /// its f32 conv tail [taps - 1, conv dim] (the conv's input dtype).
    pub fn kdaStateBytes(c: *const Config) u64 {
        const state: u64 = @as(u64, c.kda_heads) * c.kda_head_dim * c.kda_head_dim * 4;
        const conv: u64 = @as(u64, c.kda_conv - 1) * c.kdaConvDim() * 4;
        return @as(u64, c.nKda()) * (state + conv);
    }

    pub fn isEos(c: *const Config, id: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.n_eos], id) != null;
    }

    /// `text`: the pack's config.json (the text model's fields at the top or in `text_config`).
    pub fn parse(a: std.mem.Allocator, text: []const u8, diag: ?*Diag) Error!Config {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch return refuse(diag, error.Glm5NextConfigSyntax, "config.json does not parse", .{});
        if (root != .object) return refuse(diag, error.Glm5NextConfigSyntax, "config.json is not an object", .{});
        const top_type = str(root.object, "model_type") orelse return refuse(diag, error.Glm5NextConfigField, "model_type missing", .{});
        if (!std.mem.eql(u8, top_type, "glm5_next") and !std.mem.eql(u8, top_type, "glm5_next_text")) return refuse(diag, error.Glm5NextConfigUnsupported, "model_type {s} (glm5_next served)", .{top_type});
        const t = if (root.object.get("text_config")) |tc| (if (tc == .object) tc.object else return refuse(diag, error.Glm5NextConfigField, "text_config is not an object", .{})) else root.object;
        var c: Config = .{};
        inline for (.{
            .{ "vocab_size", "vocab_size" },                 .{ "hidden_size", "hidden_size" },
            .{ "intermediate_size", "intermediate_size" },   .{ "moe_intermediate_size", "moe_intermediate_size" },
            .{ "n_layers", "num_hidden_layers" },            .{ "n_heads", "num_attention_heads" },
            .{ "q_lora_rank", "q_lora_rank" },               .{ "kv_lora_rank", "kv_lora_rank" },
            .{ "head_dim", "qk_nope_head_dim" },             .{ "index_n_heads", "index_n_heads" },
            .{ "index_head_dim", "index_head_dim" },         .{ "index_topk", "index_topk" },
            .{ "index_kpool", "index_kpool" },               .{ "n_routed_experts", "n_routed_experts" },
            .{ "n_experts_per_tok", "num_experts_per_tok" }, .{ "n_shared_experts", "n_shared_experts" },
            .{ "first_k_dense", "first_k_dense_replace" },   .{ "hc_mult", "hc_mult" },
            .{ "hc_iters", "hc_sinkhorn_iters" },
        }) |f| if (t.get(f[1])) |v| {
            @field(c, f[0]) = try u32Of(v, diag, f[1]);
        };
        inline for (.{ .{ "routed_scaling", "routed_scaling_factor" }, .{ "swiglu_limit", "swiglu_limit" }, .{ "rms_norm_eps", "rms_norm_eps" }, .{ "hc_eps", "hc_eps" } }) |f| if (t.get(f[1])) |v| {
            @field(c, f[0]) = try f32Of(v, diag, f[1]);
        };
        if (t.get("max_position_embeddings")) |v| c.max_position_embeddings = try u32Of(v, diag, "max_position_embeddings");
        if (t.get("norm_topk_prob")) |v| c.norm_topk_prob = if (v == .bool) v.bool else return refuse(diag, error.Glm5NextConfigField, "norm_topk_prob is not a bool", .{});
        if (t.get("tie_word_embeddings")) |v| c.tie_word_embeddings = v == .bool and v.bool;
        if (t.get("linear_attn_config")) |lac| {
            if (lac != .object) return refuse(diag, error.Glm5NextConfigField, "linear_attn_config is not an object", .{});
            if (lac.object.get("num_heads")) |v| c.kda_heads = try u32Of(v, diag, "linear_attn_config.num_heads");
            if (lac.object.get("head_dim")) |v| c.kda_head_dim = try u32Of(v, diag, "linear_attn_config.head_dim");
            if (lac.object.get("short_conv_kernel_size")) |v| c.kda_conv = try u32Of(v, diag, "linear_attn_config.short_conv_kernel_size");
            if (lac.object.get("gate_lower_bound")) |v| if (v != .null) {
                c.kda_lower_bound = try f32Of(v, diag, "linear_attn_config.gate_lower_bound");
            };
        }
        if (t.get("eos_token_id")) |v| switch (v) {
            .integer => {
                c.eos[0] = try u32Of(v, diag, "eos_token_id");
                c.n_eos = 1;
            },
            .array => |arr| {
                if (arr.items.len == 0 or arr.items.len > c.eos.len) return refuse(diag, error.Glm5NextConfigField, "eos_token_id holds {d} ids", .{arr.items.len});
                for (arr.items, 0..) |item, i| c.eos[i] = try u32Of(item, diag, "eos_token_id");
                c.n_eos = @intCast(arr.items.len);
            },
            else => return refuse(diag, error.Glm5NextConfigField, "eos_token_id", .{}),
        };
        // The served form, refused by name where a config leaves it.
        if (c.n_layers == 0 or c.n_layers > max_layers) return refuse(diag, error.Glm5NextConfigUnsupported, "num_hidden_layers {d}", .{c.n_layers});
        if (t.get("qk_rope_head_dim")) |v| if (v != .integer or v.integer != 0) return refuse(diag, error.Glm5NextConfigUnsupported, "qk_rope_head_dim (only NoPE sparse attention is served)", .{});
        if (t.get("v_head_dim")) |v| if ((try u32Of(v, diag, "v_head_dim")) != c.head_dim) return refuse(diag, error.Glm5NextConfigUnsupported, "v_head_dim differs from qk_nope_head_dim", .{});
        if (c.hc_mult != 4) return refuse(diag, error.Glm5NextConfigUnsupported, "hc_mult {d} (4 served)", .{c.hc_mult});
        if (c.index_kpool == 0 or c.index_topk % c.index_kpool != 0) return refuse(diag, error.Glm5NextConfigUnsupported, "index_topk {d} is not a multiple of index_kpool {d}", .{ c.index_topk, c.index_kpool });
        if (c.kda_head_dim != 128 or c.kda_conv != 4) return refuse(diag, error.Glm5NextConfigUnsupported, "KDA head_dim {d} / short conv {d} (128 / 4 served)", .{ c.kda_head_dim, c.kda_conv });
        if (c.n_shared_experts != 1) return refuse(diag, error.Glm5NextConfigUnsupported, "n_shared_experts {d} (one served)", .{c.n_shared_experts});
        if (c.first_k_dense > c.n_layers) return refuse(diag, error.Glm5NextConfigField, "first_k_dense_replace {d}", .{c.first_k_dense});
        if (c.tie_word_embeddings) return refuse(diag, error.Glm5NextConfigUnsupported, "tie_word_embeddings (an untied head served)", .{});
        inline for (.{ .{ "scoring_func", "sigmoid" }, .{ "topk_method", "noaux_tc" } }) |kv| if (t.get(kv[0])) |v| {
            if (v != .string or !std.mem.eql(u8, v.string, kv[1])) return refuse(diag, error.Glm5NextConfigUnsupported, kv[0] ++ " (" ++ kv[1] ++ " served)", .{});
        };
        // One expert group: the noaux_tc selection is the top-k of the biased scores (no group limit).
        inline for (.{ "n_group", "topk_group" }) |key| if (t.get(key)) |v| if ((try u32Of(v, diag, key)) != 1) return refuse(diag, error.Glm5NextConfigUnsupported, key ++ " (1 served)", .{});
        inline for (.{ "index_kpool_compress", "index_kpool_always_select_tail", "mla_use_nope" }) |key| if (t.get(key)) |v| {
            if (v == .bool and !v.bool) return refuse(diag, error.Glm5NextConfigUnsupported, key ++ "=false", .{});
        };
        if (t.get("indexer_types")) |v| {
            if (v != .array) return refuse(diag, error.Glm5NextConfigField, "indexer_types", .{});
            for (v.array.items) |item| if (item != .string or !std.mem.eql(u8, item.string, "full")) return refuse(diag, error.Glm5NextConfigUnsupported, "a shared indexer layer", .{});
        }
        if (t.get("layer_types")) |v| {
            if (v != .array or v.array.items.len != c.n_layers) return refuse(diag, error.Glm5NextConfigField, "layer_types does not name every layer", .{});
            c.dsa_mask = 0;
            for (v.array.items, 0..) |item, i| {
                if (item != .string) return refuse(diag, error.Glm5NextConfigField, "layer_types[{d}]", .{i});
                if (std.mem.eql(u8, item.string, "deepseek_sparse_attention")) c.dsa_mask |= @as(u64, 1) << @intCast(i) else if (!std.mem.eql(u8, item.string, "linear_attention"))
                    return refuse(diag, error.Glm5NextConfigUnsupported, "layer_types[{d}] {s}", .{ i, item.string });
            }
        } else if (c.n_layers != 45) return refuse(diag, error.Glm5NextConfigField, "layer_types missing", .{});
        if (t.get("mlp_layer_types")) |v| {
            if (v != .array or v.array.items.len != c.n_layers) return refuse(diag, error.Glm5NextConfigField, "mlp_layer_types does not name every layer", .{});
            for (v.array.items, 0..) |item, i| {
                if (item != .string) return refuse(diag, error.Glm5NextConfigField, "mlp_layer_types[{d}]", .{i});
                const dense = std.mem.eql(u8, item.string, "dense");
                if (!dense and !std.mem.eql(u8, item.string, "sparse")) return refuse(diag, error.Glm5NextConfigUnsupported, "mlp_layer_types[{d}] {s}", .{ i, item.string });
                if (dense != (i < c.first_k_dense)) return refuse(diag, error.Glm5NextConfigUnsupported, "mlp_layer_types: layer {d} off the dense prefix of {d}", .{ i, c.first_k_dense });
            }
        }
        if (c.hidden_size % 64 != 0 or c.moe_intermediate_size % 64 != 0) return refuse(diag, error.Glm5NextConfigUnsupported, "hidden {d} / moe intermediate {d}", .{ c.hidden_size, c.moe_intermediate_size });
        // The routed experts' activation is installed at that limit (experts.zig `expertActivation`).
        if (c.swiglu_limit != 10.0) return refuse(diag, error.Glm5NextConfigUnsupported, "swiglu_limit {d} (10 served)", .{c.swiglu_limit});
        if (c.n_experts_per_tok != 8) return refuse(diag, error.Glm5NextConfigUnsupported, "num_experts_per_tok {d} (8 served)", .{c.n_experts_per_tok});
        try c.parseResidents(root.object, diag);
        return c;
    }

    /// The residents' format and the pack's per-projection declarations (`quantization`, else `quantization_config`),
    /// once: an FP8 declaration ({"mode": "fp8_e4m3", "bits": 8, "block": [64, 128]}) is accepted only under
    /// `residents_format: fp8_block128` and only on a trunk site the route serves (`isFp8Site`: its bit in `fp8_mask`) or
    /// the MTP layer (`mtp_fp8`); every trunk site not declared FP8 stays BF16 (the full set declares all 439, the hybrid
    /// 190); the format with no FP8 declaration, or anything else, refuses by name (`Glm5NextResidentsFormat`).
    fn parseResidents(c: *Config, root: std.json.ObjectMap, diag: ?*Diag) Error!void {
        if (root.get("residents_format")) |v| {
            if (v != .string) return refuse(diag, error.Glm5NextResidentsFormat, "residents_format is not a string", .{});
            c.residents_format = std.meta.stringToEnum(ResidentsFormat, v.string) orelse return refuse(diag, error.Glm5NextResidentsFormat, "residents_format {s} (bf16 / fp8_block128 served)", .{v.string});
        }
        const q = root.get("quantization") orelse root.get("quantization_config");
        var declared: u32 = 0;
        if (q) |qv| if (qv == .object) {
            var it = qv.object.iterator();
            while (it.next()) |e| {
                const v = e.value_ptr.*;
                const key = e.key_ptr.*;
                const fp8 = isFp8Decl(v);
                if (!fp8) {
                    if (isFp8Mode(v)) return refuse(diag, error.Glm5NextResidentsFormat, "{s}: an FP8 declaration other than {{fp8_e4m3, 8 bits, block [64, 128]}}", .{key});
                    continue;
                }
                if (c.residents_format != .fp8_block128) return refuse(diag, error.Glm5NextResidentsFormat, "{s} is FP8 without residents_format fp8_block128", .{key});
                const pre = "model.language_model.layers.";
                const rest = if (std.mem.startsWith(u8, key, pre)) key[pre.len..] else return refuse(diag, error.Glm5NextResidentsFormat, "{s} is FP8, outside the trunk's layers", .{key});
                const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return refuse(diag, error.Glm5NextResidentsFormat, "{s} is FP8, outside the trunk's layers", .{key});
                const l = std.fmt.parseInt(u32, rest[0..dot], 10) catch return refuse(diag, error.Glm5NextResidentsFormat, "{s} is FP8, outside the trunk's layers", .{key});
                if (l == c.n_layers) {
                    c.mtp_fp8 = true;
                    continue;
                }
                const site = if (l < c.n_layers) c.siteIndex(l, rest[dot + 1 ..]) else null;
                const i = site orelse return refuse(diag, error.Glm5NextResidentsFormat, "{s} is FP8, not one of the trunk's FP8 projections", .{key});
                c.fp8_mask[l] |= @as(u16, 1) << i;
                declared += 1;
            }
        };
        if (c.residents_format == .fp8_block128 and declared == 0)
            return refuse(diag, error.Glm5NextResidentsFormat, "residents_format fp8_block128 declares none of the trunk's {d} FP8 projections", .{c.fp8TrunkSites()});
    }

    /// MiMo's FP8 declaration (its `Storage.fromJson` rule): exactly {"mode": "fp8_e4m3", "bits": 8, "block": [64, 128]}.
    fn isFp8Decl(v: std.json.Value) bool {
        if (v != .object or v.object.count() != 3) return false;
        const mode = v.object.get("mode") orelse return false;
        const bits = v.object.get("bits") orelse return false;
        const block = v.object.get("block") orelse return false;
        if (mode != .string or !std.mem.eql(u8, mode.string, "fp8_e4m3") or bits != .integer or bits.integer != 8) return false;
        if (block != .array or block.array.items.len != 2) return false;
        const r = block.array.items[0];
        const col = block.array.items[1];
        return r == .integer and r.integer == fp8_scale_rows and col == .integer and col.integer == fp8_scale_cols;
    }

    fn isFp8Mode(v: std.json.Value) bool {
        if (v != .object) return false;
        const mode = v.object.get("mode") orelse return false;
        return mode == .string and std.mem.startsWith(u8, mode.string, "fp8");
    }

    fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
        const v = o.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    fn u32Of(v: std.json.Value, diag: ?*Diag, comptime key: []const u8) Error!u32 {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return refuse(diag, error.Glm5NextConfigField, "{s} is not a count", .{key});
        return @intCast(v.integer);
    }

    fn f32Of(v: std.json.Value, diag: ?*Diag, comptime key: []const u8) Error!f32 {
        return switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => refuse(diag, error.Glm5NextConfigField, "{s} is not a number", .{key}),
        };
    }
};

const testing = std.testing;

/// The pack's text_config fields (vcruz305/GLM-5.3-Flash-NVFP4 @ b7d3cbf6), the layer lists generated.
pub fn flashConfigJson(a: std.mem.Allocator) ![]u8 {
    var lt: std.ArrayList(u8) = .empty;
    defer lt.deinit(a);
    var mt: std.ArrayList(u8) = .empty;
    defer mt.deinit(a);
    for (0..45) |i| {
        try lt.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", if (i % 4 == 3) "deepseek_sparse_attention" else "linear_attention" });
        try mt.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", if (i < 3) "dense" else "sparse" });
    }
    return std.fmt.allocPrint(a,
        \\{{"model_type":"glm5_next","architectures":["Glm5NextForConditionalGeneration"],"tie_word_embeddings":false,
        \\"text_config":{{"model_type":"glm5_next_text","vocab_size":154880,"hidden_size":4096,"intermediate_size":12288,
        \\"moe_intermediate_size":2048,"num_hidden_layers":45,"num_attention_heads":64,"num_key_value_heads":64,
        \\"q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":256,"qk_rope_head_dim":0,"v_head_dim":256,
        \\"index_n_heads":32,"index_head_dim":128,"index_topk":2048,"index_kpool":4,"index_kpool_compress":true,
        \\"linear_attn_config":{{"gate_lower_bound":-5.0,"head_dim":128,"num_heads":64,"short_conv_kernel_size":4}},
        \\"n_routed_experts":288,"num_experts_per_tok":8,"n_shared_experts":1,"first_k_dense_replace":3,"n_group":1,"topk_group":1,
        \\"routed_scaling_factor":2.5,"norm_topk_prob":true,"scoring_func":"sigmoid","topk_method":"noaux_tc","swiglu_limit":10.0,
        \\"rms_norm_eps":1e-05,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,"max_position_embeddings":1048576,
        \\"eos_token_id":[154820,154827,154829],"tie_word_embeddings":false,"layer_types":[{s}],"mlp_layer_types":[{s}]}}}}
    , .{ lt.items, mt.items });
}

test "glm5_next config: the pack's text_config parses to Flash's served form" {
    const text = try flashConfigJson(testing.allocator);
    defer testing.allocator.free(text);
    var diag: Diag = .{};
    const c = Config.parse(testing.allocator, text, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expectEqual(@as(u32, 11), c.nDsa());
    try testing.expectEqual(@as(u32, 34), c.nKda());
    try testing.expectEqual(@as(u32, 42), c.moeLayers());
    try testing.expectEqual(Config.flash_dsa_mask, c.dsa_mask);
    try testing.expect(c.isDsa(3) and c.isDsa(43) and !c.isDsa(0) and !c.isDsa(44));
    try testing.expect(!c.isMoe(2) and c.isMoe(3));
    try testing.expectEqual(@as(u64, 16_896), c.kvBytesPerToken());
    try testing.expectEqual(@as(u32, 24_576), c.kdaConvDim());
    try testing.expectEqual(@as(u64, 34 * (64 * 128 * 128 * 4 + 3 * 24_576 * 4)), c.kdaStateBytes());
    try testing.expect(c.isEos(154827) and !c.isEos(1));
    try testing.expectEqual(@as(f32, -5.0), c.kda_lower_bound);
}

test "glm5_next config: residents_format fp8_block128 per the map (the full set 439, the hybrid set 190, the rest BF16); the MTP layer's FP8 noted; anything else refused by name" {
    const a = testing.allocator;
    const base = try flashConfigJson(a);
    defer a.free(base);
    const c0 = try Config.parse(a, base, null);
    try testing.expectEqual(ResidentsFormat.bf16, c0.residents_format);
    try testing.expectEqual(@as(u32, 238 + 44 + 22 + 126 + 9), c0.fp8TrunkSites());
    // The pack's declaration: every trunk site FP8, the BF16-kept projections false, the MTP layer's FP8 (the prepared
    // variant's), the experts' nvfp4.
    var q: std.ArrayList(u8) = .empty;
    defer q.deinit(a);
    const fp8_decl = "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}";
    try q.appendSlice(a, "\"quantization\":{\"group_size\":16,\"bits\":4,\"mode\":\"nvfp4\",\"lm_head\":false");
    for (0..45) |l| {
        var buf: [7]Fp8Site = undefined;
        for (c0.fp8Sites(@intCast(l), &buf)) |st| try q.print(a, ",\"model.language_model.layers.{d}.{s}\":{s}", .{ l, st.suffix, fp8_decl });
        for (c0.fp8MlpSites(@intCast(l))) |st| try q.print(a, ",\"model.language_model.layers.{d}.{s}\":{s}", .{ l, st.suffix, fp8_decl });
        try q.print(a, ",\"model.language_model.layers.{d}.hc_attn_fn\":false", .{l});
        if (l >= 3) try q.print(a, ",\"model.language_model.layers.{d}.mlp.switch_mlp.gate_proj\":{{\"group_size\":16,\"bits\":4,\"mode\":\"nvfp4\"}}", .{l});
    }
    try q.print(a, ",\"model.language_model.layers.45.self_attn.q_a_proj\":{s}}}", .{fp8_decl});
    const with = struct {
        fn f(al: std.mem.Allocator, b: []const u8, extra: []const u8) ![]u8 {
            return std.fmt.allocPrint(al, "{{{s},{s}", .{ extra, b[1..] });
        }
    }.f;
    {
        const flag = try std.fmt.allocPrint(a, "\"residents_format\":\"fp8_block128\",{s}", .{q.items});
        defer a.free(flag);
        const text = try with(a, base, flag);
        defer a.free(text);
        var diag: Diag = .{};
        const c = Config.parse(a, text, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        try testing.expectEqual(ResidentsFormat.fp8_block128, c.residents_format);
        try testing.expect(c.mtp_fp8);
        try testing.expectEqual(@as(u32, 439), c.fp8Declared());
        try testing.expect(c.isFp8Site(4, "self_attn.q_proj") and c.isFp8Site(3, "self_attn.indexer.wq_b") and !c.isFp8Site(4, "self_attn.f_b_proj"));
        try testing.expect(c.isFp8Site(0, "mlp.gate_proj") and !c.isFp8Site(3, "mlp.gate_proj") and c.isFp8Site(3, "mlp.shared_experts.down_proj"));
        try testing.expect(c.siteIsFp8(4, "self_attn.q_proj") and c.siteIsFp8(3, "self_attn.indexer.wq_b") and !c.siteIsFp8(4, "self_attn.f_b_proj"));
    }
    {
        // The pack lane's hybrid set (`--groups dsa,indexer_wk,shared,dense`: the projections GLM's own weights hold on an
        // FP8 grid): the KDA layers' sites and the indexer's wq_b declared false, so BF16.
        var hybrid: std.ArrayList(u8) = .empty;
        defer hybrid.deinit(a);
        try hybrid.appendSlice(a, "\"residents_format\":\"fp8_block128\",\"quantization\":{\"lm_head\":false");
        for (0..45) |l| {
            var buf: [7]Fp8Site = undefined;
            const kda_layer = !c0.isDsa(@intCast(l));
            for (c0.fp8Sites(@intCast(l), &buf)) |st| {
                const fp8 = !kda_layer and !std.mem.eql(u8, st.suffix, "self_attn.indexer.wq_b");
                try hybrid.print(a, ",\"model.language_model.layers.{d}.{s}\":{s}", .{ l, st.suffix, if (fp8) fp8_decl else "false" });
            }
            for (c0.fp8MlpSites(@intCast(l))) |st| try hybrid.print(a, ",\"model.language_model.layers.{d}.{s}\":{s}", .{ l, st.suffix, fp8_decl });
        }
        try hybrid.appendSlice(a, "}");
        const text = try with(a, base, hybrid.items);
        defer a.free(text);
        var diag: Diag = .{};
        const c = Config.parse(a, text, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        try testing.expectEqual(@as(u32, 11 * 4 + 11 + 42 * 3 + 3 * 3), c.fp8Declared());
        try testing.expect(!c.mtp_fp8);
        try testing.expect(!c.siteIsFp8(4, "self_attn.q_proj") and c.siteIsFp8(3, "self_attn.o_proj") and c.siteIsFp8(3, "self_attn.indexer.wk") and !c.siteIsFp8(3, "self_attn.indexer.wq_b"));
        try testing.expect(c.siteIsFp8(0, "mlp.down_proj") and c.siteIsFp8(4, "mlp.shared_experts.up_proj"));
    }
    const Case = struct { extra: []const u8, why: []const u8 };
    const no_flag = try std.fmt.allocPrint(a, "{s}", .{q.items});
    defer a.free(no_flag);
    const head_fp8 = try std.mem.replaceOwned(u8, a, q.items, "\"lm_head\":false", "\"lm_head\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}");
    defer a.free(head_fp8);
    const off_site = try std.fmt.allocPrint(a, "\"residents_format\":\"fp8_block128\",{s}", .{head_fp8});
    defer a.free(off_site);
    for ([_]Case{
        .{ .extra = no_flag, .why = "without residents_format fp8_block128" },
        .{ .extra = "\"residents_format\":\"fp8_block128\",\"quantization\":{\"lm_head\":false}", .why = "declares none of the trunk's 439" },
        .{ .extra = "\"residents_format\":\"fp8_block128\",\"quantization\":{\"model.language_model.layers.4.self_attn.f_b_proj\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}}", .why = "f_b_proj is FP8, not one of the trunk's FP8 projections" },
        .{ .extra = off_site, .why = "lm_head is FP8, outside the trunk's layers" },
        .{ .extra = "\"residents_format\":\"fp8\"", .why = "residents_format fp8 (bf16 / fp8_block128 served)" },
        .{ .extra = "\"residents_format\":\"fp8_block128\",\"quantization\":{\"lm_head\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[128,128]}}", .why = "an FP8 declaration other than" },
    }) |cs| {
        const text = try with(a, base, cs.extra);
        defer a.free(text);
        var diag: Diag = .{};
        if (Config.parse(a, text, &diag)) |_| return error.TestUnexpectedResult else |e| try testing.expectEqual(error.Glm5NextResidentsFormat, e);
        testing.expect(std.mem.indexOf(u8, diag.message(), cs.why) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ cs.why, diag.message() });
            return e;
        };
    }
}

test "glm5_next config: what leaves the served form is refused by name" {
    const a = testing.allocator;
    const base = try flashConfigJson(a);
    defer a.free(base);
    const Case = struct { from: []const u8, to: []const u8, why: []const u8 };
    for ([_]Case{
        .{ .from = "\"qk_rope_head_dim\":0", .to = "\"qk_rope_head_dim\":64", .why = "qk_rope_head_dim" },
        .{ .from = "\"hc_mult\":4", .to = "\"hc_mult\":2", .why = "hc_mult 2" },
        .{ .from = "\"n_group\":1", .to = "\"n_group\":8", .why = "n_group" },
        .{ .from = "\"scoring_func\":\"sigmoid\"", .to = "\"scoring_func\":\"softmax\"", .why = "scoring_func" },
        .{ .from = "\"index_kpool\":4", .to = "\"index_kpool\":3", .why = "index_kpool 3" },
        .{ .from = "\"model_type\":\"glm5_next\",", .to = "\"model_type\":\"glm_moe_dsa\",", .why = "model_type glm_moe_dsa" },
    }) |cs| {
        const text = try std.mem.replaceOwned(u8, a, base, cs.from, cs.to);
        defer a.free(text);
        var diag: Diag = .{};
        if (Config.parse(a, text, &diag)) |_| return error.TestUnexpectedResult else |_| {}
        testing.expect(std.mem.indexOf(u8, diag.message(), cs.why) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ cs.why, diag.message() });
            return e;
        };
    }
}
