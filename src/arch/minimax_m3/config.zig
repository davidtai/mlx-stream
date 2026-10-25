//! MiniMax M3.1 Flash text geometry, checked against MiniMaxAI/MiniMax-M3.1-Flash
//! config.json at 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e.
const std = @import("std");

pub const max_layers = 60;
pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,
    pub fn message(d: *const Diag) []const u8 {
        return d.buf[0..d.len];
    }
};
pub const Error = error{ OutOfMemory, MiniMaxConfigSyntax, MiniMaxConfigField, MiniMaxConfigUnsupported };
fn refuse(d: ?*Diag, e: Error, key: []const u8) Error {
    if (d) |p| p.len = if (std.fmt.bufPrint(&p.buf, "MiniMax M3.1 Flash config: {s}", .{key})) |s| s.len else |_| p.buf.len;
    return e;
}

pub const Config = struct {
    vocab_size: u32 = 200064,
    hidden_size: u32 = 6144,
    intermediate_size: u32 = 3072,
    dense_intermediate_size: u32 = 12288,
    shared_intermediate_size: u32 = 3072,
    n_layers: u32 = 60,
    n_heads: u32 = 64,
    n_kv_heads: u32 = 4,
    head_dim: u32 = 128,
    n_routed_experts: u32 = 128,
    n_experts_per_tok: u32 = 4,
    n_shared_experts: u32 = 1,
    rotary_dim: u32 = 64,
    rope_theta: f32 = 10000000,
    rms_norm_eps: f32 = 1e-6,
    swiglu_alpha: f32 = 1.702,
    swiglu_beta: f32 = 1,
    swiglu_limit: f32 = 7,
    routed_scaling_factor: f32 = 2,
    max_position_embeddings: u32 = 1048576,
    index_n_heads: u32 = 4,
    index_head_dim: u32 = 128,
    index_block_size: u32 = 128,
    index_topk_blocks: u32 = 16,
    index_local_blocks: u32 = 1,
    index_init_blocks: u32 = 0,

    pub fn isMoe(_: Config, layer: usize) bool {
        return layer >= 3;
    }
    pub fn isSparse(_: Config, _: usize) bool {
        return true;
    }
    pub fn moeLayers(c: Config) u32 {
        return c.n_layers - 3;
    }
    /// BF16 K/V plus one BF16 index key per token, for every layer.
    pub fn kvBytes(c: Config, tokens: u64) u64 {
        return @as(u64, c.n_layers) * tokens * (2 * @as(u64, c.n_kv_heads) * c.head_dim + c.index_head_dim) * 2;
    }

    /// Construction is intentionally restricted to the released text architecture.
    /// Unknown metadata is allowed; changed operations and layer topology are refused.
    pub fn parse(a: std.mem.Allocator, text: []const u8, diag: ?*Diag) Error!Config {
        const p = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => refuse(diag, error.MiniMaxConfigSyntax, "invalid JSON"),
        };
        defer p.deinit();
        if (p.value != .object) return refuse(diag, error.MiniMaxConfigSyntax, "expected object");
        const root = p.value.object;
        const nested = root.get("text_config");
        if (nested) |v| if (v != .object) return refuse(diag, error.MiniMaxConfigField, "text_config");
        const s = Src{ .o = if (nested) |v| v.object else root, .diag = diag };
        const mt = root.get("model_type") orelse return refuse(diag, error.MiniMaxConfigField, "model_type");
        if (mt != .string or !(std.mem.eql(u8, mt.string, "minimax_m3_vl") or (nested == null and (std.mem.eql(u8, mt.string, "minimax_m3_text") or std.mem.eql(u8, mt.string, "minimax_m3")))))
            return refuse(diag, error.MiniMaxConfigUnsupported, "model_type");
        var c: Config = .{};
        inline for (.{
            .{ "vocab_size", "vocab_size" },                             .{ "hidden_size", "hidden_size" },
            .{ "intermediate_size", "intermediate_size" },               .{ "dense_intermediate_size", "dense_intermediate_size" },
            .{ "shared_intermediate_size", "shared_intermediate_size" }, .{ "num_hidden_layers", "n_layers" },
            .{ "num_attention_heads", "n_heads" },                       .{ "num_key_value_heads", "n_kv_heads" },
            .{ "head_dim", "head_dim" },                                 .{ "num_local_experts", "n_routed_experts" },
            .{ "num_experts_per_tok", "n_experts_per_tok" },             .{ "n_shared_experts", "n_shared_experts" },
            .{ "rotary_dim", "rotary_dim" },
        }) |pair| try s.uintExact(pair[0], @field(c, pair[1]));
        inline for (.{ "rope_theta", "rms_norm_eps", "swiglu_alpha", "swiglu_limit", "routed_scaling_factor" }) |key|
            try s.floatExact(key, @field(c, key));
        if (s.o.contains("swiglu_beta")) try s.floatExact("swiglu_beta", 1);
        try s.floatExact("partial_rotary_factor", 0.5);
        try s.boolean("use_gemma_norm", true);
        try s.boolean("use_qk_norm", true);
        try s.boolean("use_routing_bias", true);
        try s.boolean("attention_output_gate", false);
        try s.boolean("tie_word_embeddings", false);
        try s.string("hidden_act", "swigluoai");
        try s.string("qk_norm_type", "per_head");
        try s.string("scoring_func", "sigmoid");
        const pos = try s.get("max_position_embeddings");
        if (pos != .integer or pos.integer < 1 or pos.integer > std.math.maxInt(c_int)) return refuse(diag, error.MiniMaxConfigField, "max_position_embeddings");
        c.max_position_embeddings = @intCast(pos.integer);
        if (s.o.get("moe_layer_freq")) |v| try s.frequency(v, "moe_layer_freq", true) else if (s.o.get("mlp_layer_types")) |v| {
            try s.layerTypes(v, "mlp_layer_types", true);
        } else return refuse(diag, error.MiniMaxConfigField, "moe_layer_freq / mlp_layer_types");
        if (s.o.get("mlp_layer_types")) |v| try s.layerTypes(v, "mlp_layer_types", true);
        if (s.o.get("layer_types")) |v| try s.layerTypes(v, "layer_types", false);
        const sparse = try s.get("sparse_attention_config");
        if (sparse != .object) return refuse(diag, error.MiniMaxConfigField, "sparse_attention_config");
        const sp = Src{ .o = sparse.object, .diag = diag };
        try sp.boolean("use_sparse_attention", true);
        try sp.uintExact("sparse_index_dim", c.index_head_dim);
        try sp.uintExact("sparse_num_index_heads", c.index_n_heads);
        try sp.uintExact("sparse_topk_blocks", c.index_topk_blocks);
        try sp.uintExact("sparse_block_size", c.index_block_size);
        try sp.uintExact("sparse_init_block", c.index_init_blocks);
        try sp.uintExact("sparse_local_block", c.index_local_blocks);
        try sp.string("sparse_score_type", "max");
        try sp.frequency(try sp.get("sparse_attention_freq"), "sparse_attention_freq", false);
        try sp.frequency(try sp.get("sparse_disable_index_value"), "sparse_disable_index_value", false);
        // The graph implements unscaled split-half partial RoPE.
        inline for (.{ "rope_scaling", "rope_parameters" }) |key| if (s.o.get(key)) |v| if (v != .null) {
            if (v != .object) return refuse(diag, error.MiniMaxConfigUnsupported, key);
            const r = Src{ .o = v.object, .diag = diag };
            var it = r.o.iterator();
            while (it.next()) |entry| {
                const allowed = [_][]const u8{ "rope_type", "type", "factor", "rope_theta", "partial_rotary_factor" };
                var known = false;
                for (allowed) |name| if (std.mem.eql(u8, entry.key_ptr.*, name)) {
                    known = true;
                    break;
                };
                if (!known) return refuse(diag, error.MiniMaxConfigUnsupported, key);
            }
            if (r.o.contains("rope_type")) try r.string("rope_type", "default");
            if (r.o.contains("type")) try r.string("type", "default");
            if (r.o.contains("factor")) try r.floatExact("factor", 1);
            if (r.o.contains("rope_theta")) try r.floatExact("rope_theta", c.rope_theta);
            if (r.o.contains("partial_rotary_factor")) try r.floatExact("partial_rotary_factor", 0.5);
        };
        return c;
    }
};

const Src = struct {
    o: std.json.ObjectMap,
    diag: ?*Diag,
    fn get(s: Src, k: []const u8) Error!std.json.Value {
        return s.o.get(k) orelse refuse(s.diag, error.MiniMaxConfigField, k);
    }
    fn uintExact(s: Src, k: []const u8, want: u32) Error!void {
        const v = try s.get(k);
        if (v != .integer) return refuse(s.diag, error.MiniMaxConfigField, k);
        if (v.integer != want) return refuse(s.diag, error.MiniMaxConfigUnsupported, k);
    }
    fn floatExact(s: Src, k: []const u8, want: f32) Error!void {
        const v = try s.get(k);
        const f: f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => return refuse(s.diag, error.MiniMaxConfigField, k),
        };
        if (!std.math.isFinite(f) or f != want) return refuse(s.diag, error.MiniMaxConfigUnsupported, k);
    }
    fn boolean(s: Src, k: []const u8, want: bool) Error!void {
        const v = try s.get(k);
        if (v != .bool) return refuse(s.diag, error.MiniMaxConfigField, k);
        if (v.bool != want) return refuse(s.diag, error.MiniMaxConfigUnsupported, k);
    }
    fn string(s: Src, k: []const u8, want: []const u8) Error!void {
        const v = try s.get(k);
        if (v != .string) return refuse(s.diag, error.MiniMaxConfigField, k);
        if (!std.mem.eql(u8, v.string, want)) return refuse(s.diag, error.MiniMaxConfigUnsupported, k);
    }
    fn frequency(s: Src, v: std.json.Value, k: []const u8, moe: bool) Error!void {
        if (v != .array or v.array.items.len != max_layers) return refuse(s.diag, error.MiniMaxConfigField, k);
        for (v.array.items, 0..) |item, i| if (item != .integer or item.integer != @as(i64, if (moe and i < 3) 0 else 1)) return refuse(s.diag, error.MiniMaxConfigUnsupported, k);
    }
    fn layerTypes(s: Src, v: std.json.Value, k: []const u8, moe: bool) Error!void {
        if (v != .array or v.array.items.len != max_layers) return refuse(s.diag, error.MiniMaxConfigField, k);
        for (v.array.items, 0..) |item, i| {
            const want = if (moe) (if (i < 3) "dense" else "sparse") else "minimax_m3_sparse";
            if (item != .string or !std.mem.eql(u8, item.string, want)) return refuse(s.diag, error.MiniMaxConfigUnsupported, k);
        }
    }
};

test "released Flash config has sparse attention in all sixty layers and three dense MLPs" {
    const c = try Config.parse(std.testing.allocator, @embedFile("config_fixture.json"), null);
    try std.testing.expect(!c.isMoe(2) and c.isMoe(3) and c.isSparse(0) and c.isSparse(59));
    try std.testing.expectEqual(@as(u32, 57), c.moeLayers());
    try std.testing.expectEqual(@as(f32, 10000000), c.rope_theta);
    try std.testing.expectEqual(@as(u64, 138240), c.kvBytes(1));
}

test "config refuses changed math and topology by field name" {
    const a = std.testing.allocator;
    const base = @embedFile("config_fixture.json");
    const Case = struct { from: []const u8, to: []const u8, key: []const u8 };
    for ([_]Case{
        .{ .from = "\"rope_theta\":10000000", .to = "\"rope_theta\":5000000", .key = "rope_theta" },
        .{ .from = "\"use_gemma_norm\":true", .to = "\"use_gemma_norm\":false", .key = "use_gemma_norm" },
        .{ .from = "\"sparse_score_type\":\"max\"", .to = "\"sparse_score_type\":\"lse\"", .key = "sparse_score_type" },
        .{ .from = "\"moe_layer_freq\":[0,0,0,1", .to = "\"moe_layer_freq\":[0,0,1,1", .key = "moe_layer_freq" },
        .{ .from = "\"sparse_attention_freq\":[1,1", .to = "\"sparse_attention_freq\":[0,1", .key = "sparse_attention_freq" },
    }) |cs| {
        const text = try std.mem.replaceOwned(u8, a, base, cs.from, cs.to);
        defer a.free(text);
        var d: Diag = .{};
        try std.testing.expectError(error.MiniMaxConfigUnsupported, Config.parse(a, text, &d));
        try std.testing.expect(std.mem.indexOf(u8, d.message(), cs.key) != null);
    }
}
