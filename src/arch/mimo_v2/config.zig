//! Host-only MiMo V2.6 Flash construction contract (affine 4-/8-bit, BF16 or the release's FP8 residents, MXFP4 experts).
const std = @import("std");

pub const max_layers = 48;
pub const moe_layers = 47;
pub const prefill_chunk: u32 = 512;
pub const prefill_max_tokens: u32 = 16_384;
pub const default_expert_rows: u32 = 8;
/// The Q4 pack's residents (`residentPayloadBytes` at 4 bits): the bill fixtures' value.
pub const resident_payload_bytes: u64 = 3_437_227_904;
/// `Storage.bits` of a dense BF16 projection (declared `false` in the pack's `quantization`, mlx-lm's convention).
pub const bf16_bits: u8 = 16;

/// fp8res (`residents_format: "fp8_block128"`): the release's FP8 projections as shipped. One f32 scale covers
/// `fp8_scale_rows` output rows x `fp8_scale_cols` input columns: the release's 128 x 128 tiles, re-indexed on a
/// 64-row grid because the converter's q/k/v regroup of the tensor-parallel rank slabs splits tiles at 64 rows
/// (reports/.../mimo/fp8res-20261009/tools/fp8res.py).
pub const fp8_scale_rows: u32 = 64;
pub const fp8_scale_cols: u32 = 128;

/// One resident projection's storage: affine `bits` 4 or 8 at `group` 32 or 64 (U32 codes [out, in*bits/32], BF16
/// scales and biases [out, in/group]), dense BF16 [out, in] (`bits == bf16_bits`), or the release's FP8 (`fp8`:
/// e4m3fn codes U8 [out, in] `.weight` + F32 [out / 64, in / 128] `.weight_scale_inv`, value = code x scale).
pub const Storage = struct {
    bits: u8 = 4,
    group: u8 = 64,
    fp8: bool = false,
    pub const bf16: Storage = .{ .bits = bf16_bits, .group = 0 };
    pub const fp8_block128: Storage = .{ .bits = 8, .group = 0, .fp8 = true };
    pub fn dense(s: Storage) bool {
        return s.bits == bf16_bits;
    }
    pub fn eql(a: Storage, b: Storage) bool {
        return a.bits == b.bits and a.group == b.group and a.fp8 == b.fp8;
    }
    /// Arrays per projection: the weight, plus scales and biases when affine, plus the f32 scales when FP8.
    pub fn arrays(s: Storage) u64 {
        return if (s.dense()) 1 else if (s.fp8) 2 else 3;
    }
    pub fn bytes(s: Storage, out: u64, in: u64) u64 {
        if (s.dense()) return out * in * 2;
        if (s.fp8) return out * in + 4 * ((out + fp8_scale_rows - 1) / fp8_scale_rows) * ((in + fp8_scale_cols - 1) / fp8_scale_cols);
        return out * in * s.bits / 8 + 2 * 2 * (out * in / s.group);
    }
    /// `{"mode":"affine","group_size":32|64,"bits":4|8}`, `false` (BF16) or `{"mode":"fp8_e4m3","bits":8,"block":[64,128]}`
    /// (the fp8res pack's FP8 projections); anything else is null.
    pub fn fromJson(v: std.json.Value) ?Storage {
        if (v == .bool) return if (v.bool) null else bf16;
        if (v != .object) return null;
        const mode = v.object.get("mode") orelse return null;
        if (mode != .string) return null;
        if (std.mem.eql(u8, mode.string, "fp8_e4m3")) {
            const bits = v.object.get("bits") orelse return null;
            const block = v.object.get("block") orelse return null;
            if (v.object.count() != 3 or bits != .integer or bits.integer != 8 or block != .array or block.array.items.len != 2) return null;
            const r = block.array.items[0];
            const c = block.array.items[1];
            if (r != .integer or r.integer != fp8_scale_rows or c != .integer or c.integer != fp8_scale_cols) return null;
            return fp8_block128;
        }
        const group = v.object.get("group_size") orelse return null;
        const bits = v.object.get("bits") orelse return null;
        if (!std.mem.eql(u8, mode.string, "affine") or group != .integer or bits != .integer) return null;
        if (group.integer != 32 and group.integer != 64) return null;
        if (bits.integer != 4 and bits.integer != 8) return null;
        return .{ .bits = @intCast(bits.integer), .group = @intCast(group.integer) };
    }
};

/// The residents' format, the pack's top-level flag `residents_format` (absent: `.standard`, the per-role affine / BF16
/// storages). `.fp8_block128` is the release's FP8 trunk as shipped: q/k/v and layer 0's dense MLP FP8, embed, head and
/// o_proj BF16 (as the release ships them); it selects the FP8 resident route at construction (module.zig).
pub const ResidentsFormat = enum { standard, fp8_block128 };
pub const fp8_roles = [_]Role{ .q, .k, .v, .mlp };

/// The resident projection roles; every projection of a role shares one storage (e.g. all 48 layers' q_proj).
pub const Role = enum { embed, head, q, k, v, o, mlp };
pub const n_roles = std.enums.values(Role).len;
pub const allocator_cache_bytes: u64 = 256 << 20;
/// P7 `prefill_scratch: "pooled"`: MLX's buffer cache limit through the prompt pass (the decode phase keeps
/// `allocator_cache_bytes`). Holds a layer's freed temporaries for its next tiles of the same shapes -- an attention
/// call's tile (~270 MB at 16K rows), two expert projection tiles in flight (~2 x 134 MB), the combine's per-layer
/// buffers -- with the partial tiles' odd sizes beside them; the pass purges it at each layer's end once it is half
/// full (`Source.promptLayerEnd`). The bill charges the prompt phase this limit (MLX trims the cache to it after every
/// allocation; the one-buffer overshoot is billed apart).
pub const prompt_pool_cache_bytes: u64 = 2 << 30;
pub const full_pattern = [_]u32{ 0, 5, 11, 17, 23, 29, 35, 41, 47 };

pub const Attention = enum { full, sliding };
pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,
    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};
pub const Error = error{ OutOfMemory, MimoConfigSyntax, MimoConfigField, MimoConfigUnsupported };

fn refuse(diag: ?*Diag, err: Error, key: []const u8) Error {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, "MiMo V2.6 Flash config: {s}", .{key})) |m| m.len else |_| d.buf.len;
    return err;
}

pub const Config = struct {
    vocab_size: u32 = 152576,
    hidden_size: u32 = 4096,
    intermediate_size: u32 = 16384,
    moe_intermediate_size: u32 = 2048,
    n_layers: u32 = 48,
    n_heads: u32 = 64,
    n_kv_heads: u32 = 4,
    swa_n_kv_heads: u32 = 8,
    head_dim: u32 = 192,
    value_head_dim: u32 = 128,
    rope_head_dim: u32 = 64,
    n_routed_experts: u32 = 256,
    n_experts_per_tok: u32 = 8,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000000,
    swa_rope_theta: f32 = 10000,
    window: u32 = 128,
    attention_value_scale: f32 = 0.707,
    max_position_embeddings: u64 = 1048576,
    layer_types: [max_layers]Attention = initialLayerTypes(),
    /// Each resident role's storage, read once from the pack's `quantization` block (default: the Q4 pack's affine 4/64).
    residents: [n_roles]Storage = @splat(.{}),
    /// The pack's `residents_format` flag (fp8res): read once here; module.zig installs the FP8 route from it.
    residents_format: ResidentsFormat = .standard,

    pub fn parse(a: std.mem.Allocator, text: []const u8, diag: ?*Diag) Error!Config {
        const parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => refuse(diag, error.MimoConfigSyntax, "invalid JSON"),
        };
        defer parsed.deinit();
        if (parsed.value != .object) return refuse(diag, error.MimoConfigSyntax, "expected object");
        const src = Src{ .o = parsed.value.object, .diag = diag };
        try src.string("model_type", "mimo_v2_flash");
        var c: Config = .{};
        inline for (.{
            .{ "vocab_size", "vocab_size" },               .{ "hidden_size", "hidden_size" },
            .{ "intermediate_size", "intermediate_size" }, .{ "moe_intermediate_size", "moe_intermediate_size" },
            .{ "num_hidden_layers", "n_layers" },          .{ "num_attention_heads", "n_heads" },
            .{ "num_key_value_heads", "n_kv_heads" },      .{ "swa_num_key_value_heads", "swa_n_kv_heads" },
            .{ "head_dim", "head_dim" },                   .{ "v_head_dim", "value_head_dim" },
            .{ "n_routed_experts", "n_routed_experts" },   .{ "num_experts_per_tok", "n_experts_per_tok" },
            .{ "sliding_window", "window" },
        }) |pair| try src.uintExact(pair[0], @field(c, pair[1]));
        try src.uintExact("swa_num_attention_heads", c.n_heads);
        try src.uintExact("swa_head_dim", c.head_dim);
        try src.uintExact("swa_v_head_dim", c.value_head_dim);
        try src.uintExact("n_group", 1);
        try src.uintExact("topk_group", 1);
        try src.floatExact("partial_rotary_factor", 0.334);
        try src.floatExact("layernorm_epsilon", c.rms_norm_eps);
        try src.floatExact("attention_value_scale", c.attention_value_scale);
        try src.floatExact("rope_theta", c.rope_theta);
        try src.floatExact("swa_rope_theta", c.swa_rope_theta);
        try src.string("scoring_func", "sigmoid");
        try src.string("topk_method", "noaux_tc");
        try src.string("hidden_act", "silu");
        try src.boolean("norm_topk_prob", true);
        try src.boolean("attention_bias", false);
        try src.boolean("tie_word_embeddings", false);
        try src.boolean("add_full_attention_sink_bias", false);
        try src.boolean("add_swa_attention_sink_bias", true);
        if (src.o.get("attention_dropout")) |v| if (try number(v, diag, "attention_dropout") != 0)
            return refuse(diag, error.MimoConfigUnsupported, "attention_dropout");
        if (src.o.get("attention_projection_layout")) |v| {
            if (v != .string or (!std.mem.eql(u8, v.string, "split_qkv") and !std.mem.eql(u8, v.string, "fused_qkv")))
                return refuse(diag, error.MimoConfigUnsupported, "attention_projection_layout");
        }
        if (src.o.get("n_shared_experts")) |v| if (v != .null and (v != .integer or v.integer != 0))
            return refuse(diag, error.MimoConfigUnsupported, "shared experts");
        if (src.o.get("routed_scaling_factor")) |v| if (v != .null and try number(v, diag, "routed_scaling_factor") != 1)
            return refuse(diag, error.MimoConfigUnsupported, "routed_scaling_factor");
        const max_pos = try src.get("max_position_embeddings");
        if (max_pos != .integer or max_pos.integer < 1 or max_pos.integer > std.math.maxInt(c_int))
            return refuse(diag, error.MimoConfigField, "max_position_embeddings");
        c.max_position_embeddings = @intCast(max_pos.integer);
        const hybrid = try src.get("hybrid_layer_pattern");
        const moe = try src.get("moe_layer_freq");
        if (hybrid != .array or moe != .array or hybrid.array.items.len != max_layers or moe.array.items.len != max_layers)
            return refuse(diag, error.MimoConfigField, "layer patterns must have 48 entries");
        for (0..max_layers) |i| {
            const h = hybrid.array.items[i];
            const m = moe.array.items[i];
            const want_h: i64 = if (c.layer_types[i] == .sliding) 1 else 0;
            if (h != .integer or h.integer != want_h or m != .integer or m.integer != @as(i64, if (i == 0) 0 else 1))
                return refuse(diag, error.MimoConfigUnsupported, "hybrid/MoE layer topology");
        }
        const quant = try src.get("quantization");
        if (quant != .object) return refuse(diag, error.MimoConfigField, "quantization");
        // The default (top-level) entry is the residents' affine storage (the Q4 pack) or the experts' MXFP4
        // (mlx-lm's convention: then every resident projection is declared on its own, `false` = BF16).
        const default: ?Storage = if (isExpertWidth(quant)) null else Storage.fromJson(quant) orelse
            return refuse(diag, error.MimoConfigUnsupported, "resident quantization");
        var seen: [n_roles]?Storage = @splat(null);
        var counts: [n_roles]u32 = @splat(0);
        var it = quant.object.iterator();
        while (it.next()) |e| {
            const v = e.value_ptr.*;
            if (v != .object and v != .bool) continue;
            if (std.mem.indexOf(u8, e.key_ptr.*, ".switch_mlp.") != null) {
                if (!isExpertWidth(v)) return refuse(diag, error.MimoConfigUnsupported, "expert quantization");
                continue;
            }
            const role = c.roleOf(e.key_ptr.*) orelse return refuse(diag, error.MimoConfigUnsupported, "resident projection name");
            const st = Storage.fromJson(v) orelse return refuse(diag, error.MimoConfigUnsupported, "resident quantization");
            const r = @backingInt(role);
            if (seen[r]) |prev| if (!prev.eql(st)) return refuse(diag, error.MimoConfigUnsupported, "mixed storage within a resident role");
            seen[r] = st;
            counts[r] += 1;
        }
        for (0..n_roles) |r| {
            const declared = seen[r] orelse {
                c.residents[r] = default orelse return refuse(diag, error.MimoConfigUnsupported, "undeclared resident projections");
                continue;
            };
            // A partly declared role must agree with the default that covers the rest.
            if (counts[r] != c.roleCount(@fromBackingInt(@intCast(r))) and !(if (default) |d| d.eql(declared) else false))
                return refuse(diag, error.MimoConfigUnsupported, if (default == null) "undeclared resident projections" else "mixed storage within a resident role");
            c.residents[r] = declared;
        }
        // fp8res: the flag and the roles' storages agree, both ways (an FP8 entry without the flag, or the flag over
        // anything but the release's own split, refuses here; the route never meets a mixed tier).
        if (src.o.get("residents_format")) |v| {
            if (v != .string or !std.mem.eql(u8, v.string, @tagName(ResidentsFormat.fp8_block128)))
                return refuse(diag, error.MimoConfigUnsupported, "residents_format");
            c.residents_format = .fp8_block128;
        }
        for (std.enums.values(Role)) |r| {
            const fp8_role = std.mem.indexOfScalar(Role, &fp8_roles, r) != null;
            const st = c.storage(r);
            const ok = switch (c.residents_format) {
                .standard => !st.fp8,
                .fp8_block128 => if (fp8_role) st.eql(Storage.fp8_block128) else st.dense(),
            };
            if (!ok) return refuse(diag, error.MimoConfigUnsupported, if (c.residents_format == .standard) "FP8 residents without residents_format" else "residents_format fp8_block128 needs q/k/v/mlp FP8 and embed/head/o BF16");
        }
        if (src.o.get("rope_scaling")) |r| if (r != .null) {
            if (r != .object) return refuse(diag, error.MimoConfigUnsupported, "rope_scaling");
            const kind = r.object.get("rope_type") orelse r.object.get("type");
            if (kind) |k| if (k != .string or !std.mem.eql(u8, k.string, "default"))
                return refuse(diag, error.MimoConfigUnsupported, "non-default rope scaling");
        };
        if (src.o.get("rope_parameters")) |r| if (r != .null) {
            if (r != .object) return refuse(diag, error.MimoConfigUnsupported, "rope_parameters");
            const rope = Src{ .o = r.object, .diag = diag };
            try rope.floatExact("rope_theta", c.rope_theta);
            try rope.floatExact("partial_rotary_factor", 0.334);
            if (r.object.contains("rope_type")) try rope.string("rope_type", "default");
            if (r.object.contains("type")) try rope.string("type", "default");
        };
        return c;
    }

    pub fn storage(c: Config, r: Role) Storage {
        return c.residents[@backingInt(r)];
    }

    /// A resident projection name's role (the pack's `quantization` keys), or null.
    pub fn roleOf(c: Config, name: []const u8) ?Role {
        if (std.mem.eql(u8, name, "model.embed_tokens")) return .embed;
        if (std.mem.eql(u8, name, "lm_head")) return .head;
        inline for (.{ .{ "gate_proj", Role.mlp }, .{ "up_proj", Role.mlp }, .{ "down_proj", Role.mlp } }) |m|
            if (std.mem.eql(u8, name, "model.layers.0.mlp." ++ m[0])) return m[1];
        const prefix = "model.layers.";
        if (!std.mem.startsWith(u8, name, prefix)) return null;
        const rest = name[prefix.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
        if (layer >= c.n_layers or (dot > 1 and rest[0] == '0')) return null;
        inline for (.{ .{ "q_proj", Role.q }, .{ "k_proj", Role.k }, .{ "v_proj", Role.v }, .{ "o_proj", Role.o } }) |m|
            if (std.mem.eql(u8, rest[dot..], ".self_attn." ++ m[0])) return m[1];
        return null;
    }

    /// Projections per role.
    pub fn roleCount(c: Config, r: Role) u32 {
        return switch (r) {
            .embed, .head => 1,
            .q, .k, .v, .o => c.n_layers,
            .mlp => 3,
        };
    }

    /// The resident projections' MLX arrays (U2's census): one per BF16 projection, three per quantized one.
    pub fn residentProjectionArrays(c: Config) u64 {
        var n: u64 = 0;
        for (0..n_roles) |r| n += c.roleCount(@fromBackingInt(@intCast(r))) * c.residents[r].arrays();
        return n;
    }

    /// The residents' payload by geometry and storage (the pack's model.safetensors tensors, headers excluded):
    /// embed + head, per layer q/k/v/o and two norms, the sliding layers' sinks, layer 0's dense MLP, each routed
    /// layer's BF16 router and F32 correction, the final norm.
    pub fn residentPayloadBytes(c: Config) u64 {
        const h: u64 = c.hidden_size;
        var total: u64 = c.storage(.embed).bytes(c.vocab_size, h) + c.storage(.head).bytes(c.vocab_size, h) + 2 * h;
        for (0..c.n_layers) |i| {
            const kvh: u64 = c.kvHeads(i);
            total += c.storage(.q).bytes(@as(u64, c.n_heads) * c.head_dim, h) + c.storage(.k).bytes(kvh * c.head_dim, h) +
                c.storage(.v).bytes(kvh * c.value_head_dim, h) + c.storage(.o).bytes(h, @as(u64, c.n_heads) * c.value_head_dim) + 4 * h;
            if (c.layer_types[i] == .sliding) total += 2 * @as(u64, c.n_heads);
            total += if (i == 0)
                2 * c.storage(.mlp).bytes(c.intermediate_size, h) + c.storage(.mlp).bytes(h, c.intermediate_size)
            else
                @as(u64, c.n_routed_experts) * h * 2 + @as(u64, c.n_routed_experts) * 4;
        }
        return total;
    }

    pub fn kvHeads(c: Config, layer: usize) u32 {
        return if (c.layer_types[layer] == .sliding) c.swa_n_kv_heads else c.n_kv_heads;
    }
    pub fn fullLayers(c: Config) u32 {
        var n: u32 = 0;
        for (c.layer_types) |t| n += @intFromBool(t == .full);
        return n;
    }
    /// Retained BF16 keys and values; full layers retain the admitted request capacity.
    pub fn kvBytes(c: Config, reserve: u64) u64 {
        const width: u64 = c.head_dim + c.value_head_dim;
        const n_full: u64 = c.fullLayers();
        return 2 * width * (n_full * c.n_kv_heads * reserve + (c.n_layers - n_full) * c.swa_n_kv_heads * @min(reserve, c.window));
    }
    pub fn routerPromotionBytes(c: Config) u64 {
        return @as(u64, c.n_layers - 1) * c.n_routed_experts * c.hidden_size * 4;
    }
    pub fn reserveTokens(c: Config, seq: u64, max_tokens: u64, host_context: u64) u64 {
        const context = @min(c.max_position_embeddings, host_context);
        return @max(seq, @min(seq +| max_tokens, context));
    }
    /// Conservative chunk-local rows, including dense MLP and FP32 routing. Attention scores are separate.
    pub fn workspaceBytes(c: Config, rows: u64) u64 {
        return rows * (16 * @as(u64, c.hidden_size) * 4 + 8 * @as(u64, c.intermediate_size) * 4 + 4 * @as(u64, c.n_routed_experts) * 4 + @as(u64, c.n_heads) * (c.head_dim * 2 + c.value_head_dim) * 8);
    }
    /// Layer-major residuals/norms/outputs, router F32 input and permutations.
    /// Attention and dense MLP temporaries remain in the bounded tile workspace;
    /// the source separately charges its full routed-output join peak.
    pub fn tokenWorkspaceBytes(c: Config, rows: u64) u64 {
        // @min with a constant bound narrows its integer result. Widen before
        // the first product so ReleaseFast cannot discard this payload.
        const bounded_rows: u64 = @min(rows, @as(u64, prefill_max_tokens));
        return 12 * bounded_rows * @as(u64, c.hidden_size) * 2;
    }
    /// Extra BF16 Q/K/V256, attention output256 and materialized output128.
    /// The native vector phase has no padding; full and sliding prefill do
    /// not run concurrently, so charge their larger live view.
    pub fn paddedAttentionBytes(c: Config, reserve: u64) u64 {
        if (reserve <= 8) return 0;
        const rows: u64 = @min(reserve, @as(u64, prefill_chunk));
        const outputs = @as(u64, c.n_heads) * rows * (256 + 256 + c.value_head_dim);
        const full = @as(u64, c.n_kv_heads) * reserve * (256 + 256);
        const sliding = @as(u64, c.swa_n_kv_heads) * @min(reserve, c.window + rows - 1) * (256 + 256);
        return 2 * (outputs + @max(full, sliding));
    }
    /// Qualified prefill uses forced fused attention; only the native phase
    /// with at most eight query rows can allocate unfused scores. Masks,
    /// padding and cache updates still use the real prefill tile geometry.
    pub fn requestBytes(c: Config, reserve: u64) u64 {
        const rows: u64 = @min(reserve, @as(u64, prefill_chunk));
        const native_rows: u64 = @min(reserve, @as(u64, 8));
        const score_bytes = 3 * @as(u64, c.n_heads) * native_rows * reserve * 4;
        const update_bytes = 2 * @as(u64, c.n_kv_heads) * (c.head_dim + c.value_head_dim) * reserve;
        const local_bytes = 4 * @as(u64, c.swa_n_kv_heads) * (c.head_dim + c.value_head_dim) * (c.window + rows);
        return c.kvBytes(reserve) + c.workspaceBytes(rows) + c.tokenWorkspaceBytes(reserve) + score_bytes + update_bytes + local_bytes + 2 * rows * reserve + c.paddedAttentionBytes(reserve);
    }
};

/// `{"mode":"mxfp4","group_size":32,"bits":4}`: the experts' storage.
fn isExpertWidth(v: std.json.Value) bool {
    if (v != .object) return false;
    const mode = v.object.get("mode") orelse return false;
    const group = v.object.get("group_size") orelse return false;
    const bits = v.object.get("bits") orelse return false;
    return mode == .string and std.mem.eql(u8, mode.string, "mxfp4") and group == .integer and group.integer == 32 and bits == .integer and bits.integer == 4;
}

fn initialLayerTypes() [max_layers]Attention {
    var out: [max_layers]Attention = @splat(.sliding);
    for (full_pattern) |i| out[i] = .full;
    return out;
}
const Src = struct {
    o: std.json.ObjectMap,
    diag: ?*Diag,
    fn get(s: Src, k: []const u8) Error!std.json.Value {
        return s.o.get(k) orelse refuse(s.diag, error.MimoConfigField, k);
    }
    fn uintExact(s: Src, k: []const u8, expected: u32) Error!void {
        const v = try s.get(k);
        if (v != .integer or v.integer != expected) return refuse(s.diag, error.MimoConfigUnsupported, k);
    }
    fn floatExact(s: Src, k: []const u8, expected: f32) Error!void {
        if (@as(f32, @floatCast(try number(try s.get(k), s.diag, k))) != expected) return refuse(s.diag, error.MimoConfigUnsupported, k);
    }
    fn string(s: Src, k: []const u8, expected: []const u8) Error!void {
        const v = try s.get(k);
        if (v != .string or !std.mem.eql(u8, v.string, expected)) return refuse(s.diag, error.MimoConfigUnsupported, k);
    }
    fn boolean(s: Src, k: []const u8, expected: bool) Error!void {
        const v = try s.get(k);
        if (v != .bool or v.bool != expected) return refuse(s.diag, error.MimoConfigUnsupported, k);
    }
};
fn number(v: std.json.Value, d: ?*Diag, key: []const u8) Error!f64 {
    return switch (v) {
        .integer => @floatFromInt(v.integer),
        .float => if (std.math.isFinite(v.float)) v.float else refuse(d, error.MimoConfigField, key),
        else => refuse(d, error.MimoConfigField, key),
    };
}

pub const WindowView = struct { first_key: u64, key_count: u32 };
/// The first query reaches back window-1; later queries also need every new key.
pub fn windowView(offset: u64, rows: u32, window: u32) WindowView {
    const first = offset -| (window - 1);
    return .{ .first_key = first, .key_count = @intCast(offset + rows - first) };
}
pub fn attends(query: u64, key: u64, window: u32) bool {
    return key <= query and (window == 0 or query - key < window);
}

/// Independent scalar test oracle. The live router runs projection, sigmoid and selection in MLX F32.
pub fn routeReference(logits: []const f32, correction: []const f32, ids: []u32, weights: []f32) !void {
    if (logits.len != correction.len or ids.len != weights.len or ids.len == 0 or ids.len > logits.len) return error.MimoRouteShape;
    var sum: f32 = 0;
    for (ids, 0..) |*id, j| {
        var best: f32 = -std.math.inf(f32);
        var selected: u32 = 0;
        for (logits, 0..) |logit, i| {
            if (std.mem.indexOfScalar(u32, ids[0..j], @intCast(i)) != null) continue;
            const score = 1 / (1 + @exp(-logit)) + correction[i];
            if (score > best) {
                best = score;
                selected = @intCast(i);
            }
        }
        id.* = selected;
        weights[j] = 1 / (1 + @exp(-logits[selected]));
        sum += weights[j];
    }
    for (weights) |*w| w.* /= sum + 1e-20;
}

pub const fixture_json =
    \\{"model_type":"mimo_v2_flash","vocab_size":152576,"hidden_size":4096,"intermediate_size":16384,"moe_intermediate_size":2048,
    \\"num_hidden_layers":48,"num_attention_heads":64,"num_key_value_heads":4,"swa_num_attention_heads":64,"swa_num_key_value_heads":8,
    \\"head_dim":192,"v_head_dim":128,"swa_head_dim":192,"swa_v_head_dim":128,"partial_rotary_factor":0.334,
    \\"layernorm_epsilon":1e-6,"attention_value_scale":0.707,"rope_theta":10000000,"swa_rope_theta":10000,"sliding_window":128,
    \\"n_routed_experts":256,"num_experts_per_tok":8,"n_group":1,"topk_group":1,"n_shared_experts":null,"routed_scaling_factor":null,
    \\"scoring_func":"sigmoid","topk_method":"noaux_tc","hidden_act":"silu","norm_topk_prob":true,"attention_bias":false,
    \\"tie_word_embeddings":false,"add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,"max_position_embeddings":1048576,
    \\"attention_projection_layout":"fused_qkv","hybrid_layer_pattern":[0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
    \\"moe_layer_freq":[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    \\"quantization":{"group_size":64,"bits":4,"mode":"affine"}}
;

test "mlx-stream MiMo config accepts exact text topology and ignores stale fused metadata" {
    const c = try Config.parse(std.testing.allocator, fixture_json, null);
    try std.testing.expectEqual(@as(u32, 48), c.n_layers);
    try std.testing.expectEqual(@as(u32, 64), c.rope_head_dim);
    try std.testing.expectEqual(@as(u32, 9), c.fullLayers());
    try std.testing.expectEqual(@as(u32, 8), c.kvHeads(1));
    try std.testing.expectEqual(@as(u32, 4), c.kvHeads(0));
    try std.testing.expectEqual(@as(f32, 0.707), c.attention_value_scale);
}

test "mlx-stream MiMo construction refuses changed routing and missing value scale" {
    const a = std.testing.allocator;
    const changed = try std.mem.replaceOwned(u8, a, fixture_json, "\"sigmoid\"", "\"softmax\"");
    defer a.free(changed);
    var diag: Diag = .{};
    try std.testing.expectError(error.MimoConfigUnsupported, Config.parse(a, changed, &diag));
    try std.testing.expect(diag.message().len > 0);
    const missing = try std.mem.replaceOwned(u8, a, fixture_json, "\"attention_value_scale\":0.707,", "");
    defer a.free(missing);
    try std.testing.expectError(error.MimoConfigField, Config.parse(a, missing, null));
}

test "mlx-stream MiMo accepts the repacker's split_qkv declaration" {
    const a = std.testing.allocator;
    const prepared = try std.mem.replaceOwned(u8, a, fixture_json, "\"fused_qkv\"", "\"split_qkv\"");
    defer a.free(prepared);
    const c = try Config.parse(a, prepared, null);
    try std.testing.expectEqual(@as(u32, 48), c.n_layers);
}

test "mlx-stream MiMo sliding window preserves first query reach at the 128 boundary" {
    const v = windowView(127, 32, 128);
    try std.testing.expectEqual(@as(u64, 0), v.first_key);
    try std.testing.expectEqual(@as(u32, 159), v.key_count);
    try std.testing.expect(attends(127, 0, 128));
    try std.testing.expect(!attends(128, 0, 128));
    try std.testing.expect(!attends(127, 128, 128));
    const next = windowView(159, 32, 128);
    try std.testing.expectEqual(@as(u64, 32), next.first_key);
    try std.testing.expectEqual(@as(u32, 159), next.key_count);
}

test "mlx-stream MiMo routing correction changes selection but never probability weights" {
    const logits = [_]f32{ 0, 2, -2, 1 };
    const bias = [_]f32{ 0, 0, 10, 0 };
    var ids: [2]u32 = undefined;
    var scores: [2]f32 = undefined;
    try routeReference(&logits, &bias, &ids, &scores);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, &ids);
    const low = 1 / (1 + @exp(@as(f32, 2)));
    const high = 1 / (1 + @exp(@as(f32, -2)));
    try std.testing.expectApproxEqAbs(low / (low + high), scores[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scores[0] + scores[1], 1e-7);
}

test "mlx-stream MiMo BF16 KV bill separates full history from sliding storage" {
    const c = try Config.parse(std.testing.allocator, fixture_json, null);
    try std.testing.expectEqual(@as(u64, 23040 * 1024 + 199680 * 128), c.kvBytes(1024));
    try std.testing.expectEqual(@as(u64, 47 * 256 * 4096 * 4), c.routerPromotionBytes());
}

test "mlx-stream MiMo request capacity includes decode budget at small prompts and clamps to context" {
    const c: Config = .{};
    try std.testing.expectEqual(@as(u64, 132), c.reserveTokens(4, 128, 32768));
    try std.testing.expectEqual(@as(u64, 32768), c.reserveTokens(16384, 32768, 32768));
    try std.testing.expectEqual(c.max_position_embeddings, c.reserveTokens(4, std.math.maxInt(u64), 2097152));
    try std.testing.expectEqual(@as(u64, 4), c.reserveTokens(4, 0, 32768));
}

test "mlx-stream MiMo rejects contradictory nested rotary metadata before arrays exist" {
    const a = std.testing.allocator;
    const changed = try std.mem.replaceOwned(u8, a, fixture_json, "\"rope_theta\":10000000,", "\"rope_theta\":10000000,\"rope_parameters\":{\"rope_theta\":500000,\"partial_rotary_factor\":0.334},");
    defer a.free(changed);
    try std.testing.expectError(error.MimoConfigUnsupported, Config.parse(a, changed, null));
}

fn residentBlock(a: std.mem.Allocator, c: Config, by_role: [n_roles][]const u8) ![]u8 {
    var block: std.ArrayList(u8) = .empty;
    errdefer block.deinit(a);
    try block.appendSlice(a, "\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"");
    try block.print(a, ",\"model.embed_tokens\":{s},\"lm_head\":{s}", .{ by_role[0], by_role[1] });
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |n| try block.print(a, ",\"model.layers.0.mlp.{s}\":{s}", .{ n, by_role[6] });
    for (0..c.n_layers) |l| for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, 2..) |p, r|
        try block.print(a, ",\"model.layers.{d}.self_attn.{s}\":{s}", .{ l, p, by_role[r] });
    try block.appendSlice(a, ",\"model.layers.1.mlp.switch_mlp.down_proj\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}}");
    return block.toOwnedSlice(a);
}

const top_q4 = "\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}";
const q8g64 = "{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"}";
const q8g32 = "{\"group_size\":32,\"bits\":8,\"mode\":\"affine\"}";

fn parseTier(a: std.mem.Allocator, by_role: [n_roles][]const u8, diag: ?*Diag) !Config {
    const block = try residentBlock(a, .{}, by_role);
    defer a.free(block);
    const text = try std.mem.replaceOwned(u8, a, fixture_json, top_q4, block);
    defer a.free(text);
    return Config.parse(a, text, diag);
}

test "mlx-stream MiMo resident storage: the Q4, Q8, BF16 and DSV-recipe tiers parse once per role" {
    const a = std.testing.allocator;
    const q4 = try Config.parse(a, fixture_json, null);
    for (q4.residents) |st| try std.testing.expect(st.eql(.{ .bits = 4, .group = 64 }));
    try std.testing.expectEqual(resident_payload_bytes, q4.residentPayloadBytes());
    // Q8 with the residents' width as the default entry (the Q4 pack's style) and one explicit agreeing entry.
    const q8_default = try std.mem.replaceOwned(u8, a, fixture_json, top_q4, "\"quantization\":{\"group_size\":64,\"bits\":8,\"mode\":\"affine\",\"lm_head\":" ++ q8g64 ++ ",\"model.layers.1.mlp.switch_mlp.gate_proj\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}}");
    defer a.free(q8_default);
    const q8 = try Config.parse(a, q8_default, null);
    try std.testing.expectEqual(@as(u64, 6_404_173_696), q8.residentPayloadBytes());
    // mlx-lm style: the converter's `--bits 8` and `--bits 16` blocks, and the DSV recipe (8-bit g32 q/k/v + dense MLP,
    // BF16 embed/head/o_proj).
    const bf16 = try parseTier(a, @splat("false"), null);
    for (bf16.residents) |st| try std.testing.expect(st.dense());
    try std.testing.expectEqual(@as(u64, 11_967_197_056), bf16.residentPayloadBytes());
    try std.testing.expectEqual(@as(u64, 197), bf16.residentProjectionArrays());
    try std.testing.expectEqual(@as(u64, 6_404_173_696), (try parseTier(a, @splat(q8g64), null)).residentPayloadBytes());
    const dsv = try parseTier(a, .{ "false", "false", q8g32, q8g32, q8g32, "false", q8g32 }, null);
    try std.testing.expect(dsv.storage(.o).dense() and dsv.storage(.q).eql(.{ .bits = 8, .group = 32 }));
    try std.testing.expectEqual(@as(u64, 9_277_992_832), dsv.residentPayloadBytes());
    try std.testing.expectEqual(@as(u64, 2 + 48 + 3 * (3 * 48 + 3)), dsv.residentProjectionArrays());
}

test "mlx-stream MiMo resident storage refusals: widths, groups, mixed roles, undeclared and unknown projections" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "\"bits\":6,\"mode\":\"affine\"}", "\"bits\":16,\"mode\":\"affine\"}", "\"bits\":4,\"mode\":\"mxfp8\"}" }) |bad| {
        const t = try std.mem.replaceOwned(u8, a, fixture_json, "\"bits\":4,\"mode\":\"affine\"}", bad);
        defer a.free(t);
        var diag: Diag = .{};
        try std.testing.expectError(error.MimoConfigUnsupported, Config.parse(a, t, &diag));
        try std.testing.expectEqualStrings("MiMo V2.6 Flash config: resident quantization", diag.message());
    }
    // An unsupported width on one projection under the converter's block.
    var diag: Diag = .{};
    try std.testing.expectError(error.MimoConfigUnsupported, parseTier(a, .{ "false", "false", "{\"group_size\":128,\"bits\":8,\"mode\":\"affine\"}", "false", "false", "false", "false" }, &diag));
    try std.testing.expectEqualStrings("MiMo V2.6 Flash config: resident quantization", diag.message());
    // Mixed storage inside one role (layer 3's q_proj differs from the other 47).
    const block = try residentBlock(a, .{}, @splat("false"));
    defer a.free(block);
    const mixed_text = try std.mem.replaceOwned(u8, a, block, "\"model.layers.3.self_attn.q_proj\":false", "\"model.layers.3.self_attn.q_proj\":" ++ q8g64);
    defer a.free(mixed_text);
    const mixed = try std.mem.replaceOwned(u8, a, fixture_json, top_q4, mixed_text);
    defer a.free(mixed);
    try std.testing.expectError(error.MimoConfigUnsupported, Config.parse(a, mixed, &diag));
    try std.testing.expectEqualStrings("MiMo V2.6 Flash config: mixed storage within a resident role", diag.message());
    // One projection left undeclared under the MXFP4 default, and an unknown resident name.
    for ([_][2][]const u8{ .{ "\"lm_head\":false", "\"model.norm\":false" }, .{ "\"model.layers.47.self_attn.o_proj\":false", "\"model.layers.48.self_attn.o_proj\":false" } }, [_][]const u8{ "resident projection name", "resident projection name" }) |swap, want| {
        const t1 = try std.mem.replaceOwned(u8, a, block, swap[0], swap[1]);
        defer a.free(t1);
        const t = try std.mem.replaceOwned(u8, a, fixture_json, top_q4, t1);
        defer a.free(t);
        try std.testing.expectError(error.MimoConfigUnsupported, Config.parse(a, t, &diag));
        try std.testing.expectEqualStrings(want, diag.message()["MiMo V2.6 Flash config: ".len..]);
    }
    const dropped = try std.mem.replaceOwned(u8, a, block, ",\"lm_head\":false", "");
    defer a.free(dropped);
    const t = try std.mem.replaceOwned(u8, a, fixture_json, top_q4, dropped);
    defer a.free(t);
    try std.testing.expectError(error.MimoConfigUnsupported, Config.parse(a, t, &diag));
    try std.testing.expectEqualStrings("MiMo V2.6 Flash config: undeclared resident projections", diag.message());
}

// ---- fp8res (`residents_format: "fp8_block128"`): begin ----

const fp8_entry = "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}";
const fp8_flag = ",\"residents_format\":\"fp8_block128\"";
/// The fp8res pack's split: embed, head, q, k, v, o, mlp.
const fp8_split = [n_roles][]const u8{ "false", "false", fp8_entry, fp8_entry, fp8_entry, "false", fp8_entry };

fn parseFp8(a: std.mem.Allocator, by_role: [n_roles][]const u8, flag: []const u8, diag: ?*Diag) !Config {
    const block = try residentBlock(a, .{}, by_role);
    defer a.free(block);
    const with_flag = try std.mem.concat(a, u8, &.{ block, flag });
    defer a.free(with_flag);
    const text = try std.mem.replaceOwned(u8, a, fixture_json, top_q4, with_flag);
    defer a.free(text);
    return Config.parse(a, text, diag);
}

test "mlx-stream MiMo fp8res storage: the release's FP8 split parses once per role under the flag; payload and census" {
    const a = std.testing.allocator;
    const c = try parseFp8(a, fp8_split, fp8_flag, null);
    try std.testing.expectEqual(ResidentsFormat.fp8_block128, c.residents_format);
    for (fp8_roles) |r| try std.testing.expect(c.storage(r).eql(Storage.fp8_block128) and !c.storage(r).dense());
    for ([_]Role{ .embed, .head, .o }) |r| try std.testing.expect(c.storage(r).dense());
    // The fp8res pack's model.safetensors payload (convert_mimo_v2_residents.py --bits fp8, 574 tensors): 8,895,321,472 B,
    // the BF16 tier's 11,967,197,056 less 3,073,376,256 code bytes plus 1,500,672 scale bytes.
    try std.testing.expectEqual(@as(u64, 8_895_321_472), c.residentPayloadBytes());
    try std.testing.expectEqual(@as(u64, 11_967_197_056 - 3_073_376_256 + 1_500_672), c.residentPayloadBytes());
    // Two arrays per FP8 projection (codes + scales), one per BF16 one: 147 x 2 + 50.
    try std.testing.expectEqual(@as(u64, 147 * 2 + 50), c.residentProjectionArrays());
    // The scale grid: q [12288, 4096] -> [192, 32]; a full layer's k [768, 4096] -> [12, 32] (four 192-row rank slabs).
    try std.testing.expectEqual(@as(u64, 12288 * 4096 + 192 * 32 * 4), Storage.fp8_block128.bytes(12288, 4096));
    try std.testing.expectEqual(@as(u64, 768 * 4096 + 12 * 32 * 4), Storage.fp8_block128.bytes(768, 4096));
    try std.testing.expectEqual(@as(u64, 4096 * 16384 + 64 * 128 * 4), Storage.fp8_block128.bytes(4096, 16384));
    // The control tiers are untouched by the field: the BF16 pack still parses as the standard format.
    const bf16 = try parseTier(a, @splat("false"), null);
    try std.testing.expectEqual(ResidentsFormat.standard, bf16.residents_format);
    try std.testing.expectEqual(@as(u64, 11_967_197_056), bf16.residentPayloadBytes());
}

test "mlx-stream MiMo fp8res refusals: FP8 without the flag, the flag over another split, unknown flags and blocks" {
    const a = std.testing.allocator;
    var diag: Diag = .{};
    // FP8 entries without the flag (a pack the route was not asked for).
    try std.testing.expectError(error.MimoConfigUnsupported, parseFp8(a, fp8_split, "", &diag));
    try std.testing.expectEqualStrings("MiMo V2.6 Flash config: FP8 residents without residents_format", diag.message());
    // The flag over BF16 q (a half-converted pack) and over an FP8 o_proj (the release ships it BF16).
    var bf16_q = fp8_split;
    bf16_q[@backingInt(Role.q)] = "false";
    try std.testing.expectError(error.MimoConfigUnsupported, parseFp8(a, bf16_q, fp8_flag, &diag));
    try std.testing.expectEqualStrings("MiMo V2.6 Flash config: residents_format fp8_block128 needs q/k/v/mlp FP8 and embed/head/o BF16", diag.message());
    var fp8_o = fp8_split;
    fp8_o[@backingInt(Role.o)] = fp8_entry;
    try std.testing.expectError(error.MimoConfigUnsupported, parseFp8(a, fp8_o, fp8_flag, &diag));
    try std.testing.expectError(error.MimoConfigUnsupported, parseFp8(a, @splat("false"), fp8_flag, &diag));
    // A flag this binary has no route for, and a non-string flag.
    for ([_][]const u8{ ",\"residents_format\":\"fp8_block64\"", ",\"residents_format\":\"bf16\"", ",\"residents_format\":true" }) |flag| {
        try std.testing.expectError(error.MimoConfigUnsupported, parseFp8(a, fp8_split, flag, &diag));
        try std.testing.expectEqualStrings("MiMo V2.6 Flash config: residents_format", diag.message());
    }
    // Another scale grid, another width, a stray key: not the fp8res storage.
    for ([_][]const u8{
        "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[128,128]}",
        "{\"mode\":\"fp8_e4m3\",\"bits\":4,\"block\":[64,128]}",
        "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128],\"group_size\":128}",
        "{\"mode\":\"fp8_e5m2\",\"bits\":8,\"block\":[64,128]}",
    }) |bad| {
        var split = fp8_split;
        split[@backingInt(Role.k)] = bad;
        try std.testing.expectError(error.MimoConfigUnsupported, parseFp8(a, split, fp8_flag, &diag));
        try std.testing.expectEqualStrings("MiMo V2.6 Flash config: resident quantization", diag.message());
    }
}
// ---- fp8res: end ----
