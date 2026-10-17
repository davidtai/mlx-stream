//! Beam (`beam`, BeamForCausalLM: Reflection's Beam 1.0 501B-A23B) as mlx-stream serves it: the pack's config.json,
//! parsed once at construction and refused by name where it leaves the served form. Every layer is a sandwich-normed
//! block with depth-scaled residual adds: Qwen3-MoE attention (q/k/v/o, per-head q/k RMSNorm, GQA) with a per-head
//! sigmoid output gate, on sliding-window layers with RoPE or on full-attention layers without positions and with the
//! scalable-softmax query scale; then a dense SwiGLU MLP on the leading dense layers or a routed MoE (DeepSeek-V3's
//! noaux_tc group top-k, sigmoid, normalized, the weighted sum times `routed_scaling_factor`). Every SwiGLU runs in f32.
//! The head is untied and soft-capped. The served geometry is the config's own: the pack's and the test fixture's.

const std = @import("std");

pub const max_layers = 64;
/// The fill's floor (bill.zig `min_rows`).
pub const default_expert_rows: u32 = 8;
/// The scalable-softmax query scale of a full-attention layer (modeling_beam.py `compute_attention_scaling`):
/// (1 + (position + 1) / ssmax_base) ^ ssmax_power.
pub const ssmax_base: f32 = 2048;
pub const ssmax_power: f32 = 0.1;

pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,
    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Error = error{ OutOfMemory, BeamConfigSyntax, BeamConfigField, BeamConfigUnsupported };

fn refuse(diag: ?*Diag, err: Error, comptime fmt: []const u8, args: anytype) Error {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, "beam config: " ++ fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

pub const Config = struct {
    vocab_size: u32 = 151_936,
    hidden_size: u32 = 5120,
    /// The dense MLP's width (the leading `first_k_dense` layers).
    intermediate_size: u32 = 20_480,
    moe_intermediate_size: u32 = 2560,
    n_layers: u32 = 52,
    n_heads: u32 = 48,
    n_kv_heads: u32 = 8,
    head_dim: u32 = 128,
    sliding_window: u32 = 512,
    rope_theta: f32 = 10_000,
    rms_norm_eps: f32 = 1e-6,
    n_routed_experts: u32 = 256,
    n_experts_per_tok: u32 = 8,
    n_group: u32 = 8,
    topk_group: u32 = 4,
    routed_scaling: f32 = 2.5,
    norm_topk_prob: bool = true,
    final_logit_softcapping: f32 = 30,
    max_position_embeddings: u64 = 1 << 20,
    /// The leading dense layers (`mlp_layer_types`: "dense" then "sparse").
    first_k_dense: u32 = 3,
    /// Bit l set: layer l runs full attention (no RoPE, the query scale); the rest sliding attention.
    full_mask: u64 = beam_full_mask,

    /// Beam 1.0: every sixth layer and the last run full attention (5, 11, ..., 47, 51).
    pub const beam_full_mask: u64 = blk: {
        var m: u64 = 0;
        for (0..52) |i| {
            if ((i + 1) % 6 == 0 or i == 51) m |= @as(u64, 1) << i;
        }
        break :blk m;
    };

    pub fn isFull(c: *const Config, layer: u32) bool {
        return (c.full_mask >> @intCast(layer)) & 1 == 1;
    }

    pub fn isMoe(c: *const Config, layer: u32) bool {
        return layer >= c.first_k_dense;
    }

    pub fn nFull(c: *const Config) u32 {
        return @popCount(c.full_mask & layerMask(c.n_layers));
    }

    /// The bits of layers [0, n): every bit at n = 64 (a shift by 64 is outside u64's).
    pub fn layerMask(n: u32) u64 {
        return if (n >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1;
    }

    pub fn nSliding(c: *const Config) u32 {
        return c.n_layers - c.nFull();
    }

    pub fn moeLayers(c: *const Config) u32 {
        return c.n_layers - c.first_k_dense;
    }

    /// The experts of one routing group.
    pub fn groupExperts(c: *const Config) u32 {
        return c.n_routed_experts / c.n_group;
    }

    /// The depth-scaled residual add's index of layer `l`'s attention (2 (l + 1)) and MLP (2 (l + 1) + 1) sublayers:
    /// y = x_sub / sqrt(i) + residual * sqrt((i - 1) / i) (modeling_beam.py `depth_scaled_residual_add`).
    pub fn residualIndex(layer: u32, mlp: bool) u32 {
        return 2 * (layer + 1) + @intFromBool(mlp);
    }

    /// `text`: the pack's config.json.
    pub fn parse(a: std.mem.Allocator, text: []const u8, diag: ?*Diag) Error!Config {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch return refuse(diag, error.BeamConfigSyntax, "config.json does not parse", .{});
        if (root != .object) return refuse(diag, error.BeamConfigSyntax, "config.json is not an object", .{});
        const t = root.object;
        const mt = str(t, "model_type") orelse return refuse(diag, error.BeamConfigField, "model_type missing", .{});
        if (!std.mem.eql(u8, mt, "beam")) return refuse(diag, error.BeamConfigUnsupported, "model_type {s} (beam served)", .{mt});
        var c: Config = .{};
        inline for (.{
            .{ "vocab_size", "vocab_size" },                 .{ "hidden_size", "hidden_size" },
            .{ "intermediate_size", "intermediate_size" },   .{ "moe_intermediate_size", "moe_intermediate_size" },
            .{ "n_layers", "num_hidden_layers" },            .{ "n_heads", "num_attention_heads" },
            .{ "n_kv_heads", "num_key_value_heads" },        .{ "head_dim", "head_dim" },
            .{ "sliding_window", "sliding_window" },         .{ "n_routed_experts", "num_experts" },
            .{ "n_experts_per_tok", "num_experts_per_tok" }, .{ "n_group", "n_group" },
            .{ "topk_group", "topk_group" },
        }) |f| if (t.get(f[1])) |v| {
            @field(c, f[0]) = try u32Of(v, diag, f[1]);
        };
        inline for (.{ .{ "routed_scaling", "routed_scaling_factor" }, .{ "rms_norm_eps", "rms_norm_eps" }, .{ "final_logit_softcapping", "final_logit_softcapping" } }) |f| if (t.get(f[1])) |v| {
            @field(c, f[0]) = try f32Of(v, diag, f[1]);
        };
        if (t.get("max_position_embeddings")) |v| c.max_position_embeddings = try u32Of(v, diag, "max_position_embeddings");
        if (t.get("norm_topk_prob")) |v| c.norm_topk_prob = if (v == .bool) v.bool else return refuse(diag, error.BeamConfigField, "norm_topk_prob is not a bool", .{});
        // The served form, refused by name where a config leaves it.
        if (c.n_layers == 0 or c.n_layers > max_layers) return refuse(diag, error.BeamConfigUnsupported, "num_hidden_layers {d} (1 .. {d} served)", .{ c.n_layers, max_layers });
        if (t.get("tie_word_embeddings")) |v| if (v == .bool and v.bool) return refuse(diag, error.BeamConfigUnsupported, "tie_word_embeddings (an untied head served)", .{});
        if (t.get("attention_bias")) |v| if (v == .bool and v.bool) return refuse(diag, error.BeamConfigUnsupported, "attention_bias (no bias served)", .{});
        if (t.get("hidden_act")) |v| if (v != .string or !std.mem.eql(u8, v.string, "silu")) return refuse(diag, error.BeamConfigUnsupported, "hidden_act (silu served)", .{});
        if (c.n_kv_heads == 0 or c.n_heads % c.n_kv_heads != 0) return refuse(diag, error.BeamConfigUnsupported, "num_attention_heads {d} is not a multiple of num_key_value_heads {d}", .{ c.n_heads, c.n_kv_heads });
        // MLX's fused attention and the routed experts' NVFP4 groups.
        if (c.head_dim != 64 and c.head_dim != 128) return refuse(diag, error.BeamConfigUnsupported, "head_dim {d} (64 / 128 served)", .{c.head_dim});
        // The NVFP4 bank's 16-input groups (nvfp4_bank.zig `Geometry.init`).
        if (c.hidden_size % 16 != 0 or c.moe_intermediate_size % 16 != 0) return refuse(diag, error.BeamConfigUnsupported, "hidden {d} / moe intermediate {d} (multiples of 16 served)", .{ c.hidden_size, c.moe_intermediate_size });
        // The window: a prompt tile's keys are its rows and the window less one (graph.zig `windowMask`; the ring's base).
        if (c.sliding_window < 2 or c.sliding_window > 512) return refuse(diag, error.BeamConfigUnsupported, "sliding_window {d} (2 .. 512 served)", .{c.sliding_window});
        // The routed experts: at most the scheduler's top-8 rows a token, at most 256 experts (the route trace's u8 ids).
        if (c.n_experts_per_tok == 0 or c.n_experts_per_tok > 8) return refuse(diag, error.BeamConfigUnsupported, "num_experts_per_tok {d} (1 .. 8 served)", .{c.n_experts_per_tok});
        if (c.n_routed_experts < c.n_experts_per_tok or c.n_routed_experts > 256) return refuse(diag, error.BeamConfigUnsupported, "num_experts {d} (top-k .. 256 served)", .{c.n_routed_experts});
        if (c.n_group == 0 or c.n_routed_experts % c.n_group != 0 or c.topk_group == 0 or c.topk_group > c.n_group) return refuse(diag, error.BeamConfigUnsupported, "n_group {d} / topk_group {d} over {d} experts", .{ c.n_group, c.topk_group, c.n_routed_experts });
        if (c.groupExperts() < 2 or c.topk_group * c.groupExperts() < c.n_experts_per_tok) return refuse(diag, error.BeamConfigUnsupported, "{d} groups of {d} experts, {d} kept: fewer than the top-{d}", .{ c.n_group, c.groupExperts(), c.topk_group, c.n_experts_per_tok });
        if (!(c.final_logit_softcapping > 0)) return refuse(diag, error.BeamConfigUnsupported, "final_logit_softcapping {d} (a positive cap served)", .{c.final_logit_softcapping});
        try c.parseLayers(t, diag);
        try c.parseRope(t, diag);
        return c;
    }

    fn parseLayers(c: *Config, t: std.json.ObjectMap, diag: ?*Diag) Error!void {
        const lt = t.get("layer_types") orelse return refuse(diag, error.BeamConfigField, "layer_types missing", .{});
        if (lt != .array or lt.array.items.len != c.n_layers) return refuse(diag, error.BeamConfigField, "layer_types does not name every layer", .{});
        c.full_mask = 0;
        for (lt.array.items, 0..) |item, i| {
            if (item != .string) return refuse(diag, error.BeamConfigField, "layer_types[{d}]", .{i});
            if (std.mem.eql(u8, item.string, "full_attention")) c.full_mask |= @as(u64, 1) << @intCast(i) else if (!std.mem.eql(u8, item.string, "sliding_attention"))
                return refuse(diag, error.BeamConfigUnsupported, "layer_types[{d}] {s}", .{ i, item.string });
        }
        const mt = t.get("mlp_layer_types") orelse return refuse(diag, error.BeamConfigField, "mlp_layer_types missing", .{});
        if (mt != .array or mt.array.items.len != c.n_layers) return refuse(diag, error.BeamConfigField, "mlp_layer_types does not name every layer", .{});
        var dense: u32 = 0;
        for (mt.array.items, 0..) |item, i| {
            if (item != .string) return refuse(diag, error.BeamConfigField, "mlp_layer_types[{d}]", .{i});
            const is_dense = std.mem.eql(u8, item.string, "dense");
            if (!is_dense and !std.mem.eql(u8, item.string, "sparse")) return refuse(diag, error.BeamConfigUnsupported, "mlp_layer_types[{d}] {s}", .{ i, item.string });
            if (is_dense) {
                if (dense != i) return refuse(diag, error.BeamConfigUnsupported, "mlp_layer_types: dense layer {d} after a sparse one (a dense prefix served)", .{i});
                dense += 1;
            }
        }
        if (dense == c.n_layers) return refuse(diag, error.BeamConfigUnsupported, "mlp_layer_types: no sparse layer", .{});
        c.first_k_dense = dense;
    }

    /// `rope_parameters`: the sliding layers' default RoPE at `rope_theta`; the full layers' entry null (no positions).
    fn parseRope(c: *Config, t: std.json.ObjectMap, diag: ?*Diag) Error!void {
        const rp = t.get("rope_parameters") orelse return refuse(diag, error.BeamConfigField, "rope_parameters missing", .{});
        if (rp != .object) return refuse(diag, error.BeamConfigField, "rope_parameters is not an object", .{});
        const sl = rp.object.get("sliding_attention") orelse return refuse(diag, error.BeamConfigField, "rope_parameters.sliding_attention missing", .{});
        if (sl != .object) return refuse(diag, error.BeamConfigField, "rope_parameters.sliding_attention is not an object", .{});
        const rt = str(sl.object, "rope_type") orelse "default";
        if (!std.mem.eql(u8, rt, "default")) return refuse(diag, error.BeamConfigUnsupported, "rope_parameters.sliding_attention.rope_type {s} (default served)", .{rt});
        if (sl.object.get("rope_theta")) |v| c.rope_theta = try f32Of(v, diag, "rope_parameters.sliding_attention.rope_theta");
        if (rp.object.get("full_attention")) |v| if (v != .null) return refuse(diag, error.BeamConfigUnsupported, "rope_parameters.full_attention (no positions on full attention served)", .{});
        if (rp.object.get("partial_rotary_factor") != null or sl.object.get("partial_rotary_factor") != null) return refuse(diag, error.BeamConfigUnsupported, "partial_rotary_factor (full rotary served)", .{});
    }

    fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
        const v = o.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    fn u32Of(v: std.json.Value, diag: ?*Diag, comptime key: []const u8) Error!u32 {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return refuse(diag, error.BeamConfigField, "{s} is not a count", .{key});
        return @intCast(v.integer);
    }

    fn f32Of(v: std.json.Value, diag: ?*Diag, comptime key: []const u8) Error!f32 {
        return switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => refuse(diag, error.BeamConfigField, "{s} is not a number", .{key}),
        };
    }
};

const testing = std.testing;

/// The release's config.json (reflection/Beam-1.0-501B-A23B-NVFP4 @ 3bd1dccf, the quantization block left out), the
/// layer lists generated, with `extra` (a JSON fragment ending in a comma, or empty) after `model_type`.
pub fn beamConfigJson(a: std.mem.Allocator, extra: []const u8) ![]u8 {
    var lt: std.ArrayList(u8) = .empty;
    defer lt.deinit(a);
    var mt: std.ArrayList(u8) = .empty;
    defer mt.deinit(a);
    for (0..52) |i| {
        try lt.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", if ((i + 1) % 6 == 0 or i == 51) "full_attention" else "sliding_attention" });
        try mt.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", if (i < 3) "dense" else "sparse" });
    }
    return std.fmt.allocPrint(a,
        \\{{"architectures":["BeamForCausalLM"],"model_type":"beam",{s}"attention_bias":false,"attention_dropout":0.0,
        \\"dtype":"bfloat16","final_logit_softcapping":30.0,"head_dim":128,"hidden_act":"silu","hidden_size":5120,
        \\"intermediate_size":20480,"layer_types":[{s}],"max_position_embeddings":1048576,"mlp_layer_types":[{s}],
        \\"moe_intermediate_size":2560,"n_group":8,"norm_topk_prob":true,"num_attention_heads":48,"num_experts_per_tok":8,
        \\"num_hidden_layers":52,"num_key_value_heads":8,"rms_norm_eps":1e-06,
        \\"rope_parameters":{{"sliding_attention":{{"rope_type":"default","rope_theta":10000.0}},"full_attention":null}},
        \\"routed_scaling_factor":2.5,"sliding_window":512,"tie_word_embeddings":false,"topk_group":4,"vocab_size":151936,
        \\"num_experts":256}}
    , .{ extra, lt.items, mt.items });
}

test "beam config: the release's config.json parses to Beam 1.0's served form" {
    const text = try beamConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    var diag: Diag = .{};
    const c = try Config.parse(testing.allocator, text, &diag);
    try testing.expectEqual(@as(u32, 52), c.n_layers);
    try testing.expectEqual(@as(u32, 9), c.nFull());
    try testing.expectEqual(@as(u32, 43), c.nSliding());
    try testing.expectEqual(Config.beam_full_mask, c.full_mask);
    for ([_]u32{ 5, 11, 17, 23, 29, 35, 41, 47, 51 }) |l| try testing.expect(c.isFull(l));
    try testing.expect(!c.isFull(48) and !c.isFull(0));
    try testing.expectEqual(@as(u32, 3), c.first_k_dense);
    try testing.expectEqual(@as(u32, 49), c.moeLayers());
    try testing.expectEqual(@as(u32, 32), c.groupExperts());
    try testing.expectEqual(@as(f32, 10_000), c.rope_theta);
    try testing.expectEqual(@as(f32, 30), c.final_logit_softcapping);
    try testing.expectEqual(@as(u32, 2), Config.residualIndex(0, false));
    try testing.expectEqual(@as(u32, 105), Config.residualIndex(51, true));
}

test "beam config: the layer mask at every served depth, 64 layers included" {
    try testing.expectEqual(@as(u64, 0), Config.layerMask(0));
    try testing.expectEqual(@as(u64, 0x7f), Config.layerMask(7));
    try testing.expectEqual(~@as(u64, 0) >> 1, Config.layerMask(63));
    try testing.expectEqual(~@as(u64, 0), Config.layerMask(max_layers));
    const c: Config = .{ .n_layers = max_layers, .full_mask = ~@as(u64, 0) };
    try testing.expectEqual(@as(u32, 64), c.nFull());
    try testing.expectEqual(@as(u32, 0), c.nSliding());
}

test "beam config: a config off the served form refuses by name" {
    const a = testing.allocator;
    const base = try beamConfigJson(a, "");
    defer a.free(base);
    const Case = struct { from: []const u8, to: []const u8, err: Error };
    for ([_]Case{
        .{ .from = "\"model_type\":\"beam\"", .to = "\"model_type\":\"qwen3_moe\"", .err = error.BeamConfigUnsupported },
        .{ .from = "\"tie_word_embeddings\":false", .to = "\"tie_word_embeddings\":true", .err = error.BeamConfigUnsupported },
        .{ .from = "\"full_attention\":null", .to = "\"full_attention\":{\"rope_type\":\"default\"}", .err = error.BeamConfigUnsupported },
        .{ .from = "\"num_experts_per_tok\":8", .to = "\"num_experts_per_tok\":9", .err = error.BeamConfigUnsupported },
        .{ .from = "\"n_group\":8", .to = "\"n_group\":7", .err = error.BeamConfigUnsupported },
        .{ .from = "\"head_dim\":128", .to = "\"head_dim\":96", .err = error.BeamConfigUnsupported },
        .{ .from = "\"hidden_act\":\"silu\"", .to = "\"hidden_act\":\"gelu\"", .err = error.BeamConfigUnsupported },
        .{ .from = "\"num_hidden_layers\":52", .to = "\"num_hidden_layers\":51", .err = error.BeamConfigField },
    }) |c| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, base, c.from));
        const text = try std.mem.replaceOwned(u8, a, base, c.from, c.to);
        defer a.free(text);
        var diag: Diag = .{};
        try testing.expectError(c.err, Config.parse(a, text, &diag));
        try testing.expect(diag.message().len > 0);
    }
}
