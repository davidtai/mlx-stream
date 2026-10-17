//! beam's construction config: the pack's model (config.zig), the pack's `expert_streaming.settings` (read once at
//! construction; a key that is not a setting refuses the load by name) and the host's load facts. MiMo's contract: the
//! operator pins the rows with `expert_prefill_rows` / `expert_rows` (either alone forces both); `expert_rows_by_layer`
//! splits the decode rows between the routed layers (`uniform`, the default, or offsets summing to 0); the rate tally's
//! keys (`expert_resident_ratio`, `expert_tally_window`, `expert_tally_file`: sdk_ext/expert/tally.zig, every streamed
//! arch's), the usage seed and record (`resident_seed`, `expert_usage_record`, `expert_usage_label`: usage.zig),
//! `prompt_residents` (keep, the default, or forget), MiMo's X12 keep-warm (`gpu_keep_warm`: waits, the default, or
//! off) and the decode lane's glue (`decode_glue`: fused, the default, or ops). Beam has no draft lane: there is no lane
//! key.

const std = @import("std");
const sdk = @import("sdk");
const beam = @import("config.zig");
const sdk_settings = @import("../../sdk_ext/settings.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const expert_tally = @import("../../sdk_ext.zig").expert.tally;
const expert_usage = @import("../../sdk_ext.zig").expert.usage;
const glue = @import("glue.zig");

/// The context a load bills when the model sets none: the standard request (16,384 prompt tokens and 1,024 generated).
pub const fill_reserve_tokens: u64 = 16_384 + 1_024;
/// The host side of the process the bill charges (MiMo's declared host runtime and MLX's compiled kernels): no Beam
/// reading yet, so both phases bill the declared pair.
pub const host_bytes: u64 = 1_000_000_000;
pub const host_construction_bytes: u64 = 400_000_000;
pub const compilation_bytes: u64 = 512 << 20;
/// The reader's staging per worker: at least the bank's widest span, page-rounded, plus a page. Beam's gate/up span
/// (2 x (2,560 x 5,120 / 2 + 2,560 x 5,120 / 16) + 8 = 14,745,608 B) and its alignment: 16 MiB.
pub const reader_staging_bytes: u64 = 16 << 20;
/// The rate tally's default ratio on Beam: 0 (off), glm5_next's (no Beam measurement yet).
pub const default_tally_ratio: f64 = 0;

/// Every key `applySettings` reads; `expert_streaming.settings` may name these and nothing else (`unknownSetting`).
pub const setting_keys = [_][]const u8{ "expert_rows", "expert_prefill_rows", "expert_rows_by_layer", "expert_resident_ratio", "expert_tally_window", "expert_tally_file", "prompt_residents", "gpu_keep_warm", "decode_glue" } ++ expert_usage.keys;

/// The decode lane's glue, installed once at construction: `fused` (the default) the router's and the predictor's texts
/// (glue.zig: the router's product and its tail in two launches, the predictor's in one) and each layer's q, k, v and
/// gate projections in one buffer and one GEMV (graph.zig `fuseProjections`), each bit for bit the stock ops; `ops` the
/// stock op chain (the router's input cast, GEMV and 17 ops; the predictor's four; four projection GEMVs). The default
/// resolves to `ops` where the texts do not register the model's router (another routed geometry) or on a CPU stream
/// (`decodeGlueRoute`); `fused` named there refuses the load by name.
pub const DecodeGlue = enum { fused, ops };

pub const PromptResidents = expert_usage.PromptResidents;
pub const LayerRows = sdk_settings.LayerRows;

pub fn unknownSetting(o: std.json.ObjectMap) ?[]const u8 {
    return sdk_settings.unknownSetting(o, &setting_keys);
}

/// The per-layer rows of both phases (sdk_ext/expert/layer_rows.zig), refused by beam's names.
pub fn layerRows(split: LayerRows, rows: sdk.Rows, n_experts: u32, prompt: []u32, decode: []u32) !void {
    sdk_settings.layerRows(split, rows, n_experts, prompt, decode) catch |e| return switch (e) {
        error.ExpertRowsByLayer => error.BeamExpertRowsByLayer,
        error.ExpertRowsByLayerLength => error.BeamExpertRowsByLayerLength,
        error.ExpertRowsByLayerRange => error.BeamExpertRowsByLayerRange,
    };
}

pub const Config = struct {
    model: beam.Config = .{},
    /// The pack's directory: config.json, the resident shards, the bank (experts.bin and its manifest).
    model_dir: ?[]const u8 = null,
    /// The operator's rows (`expert_rows` the decode rows, `expert_prefill_rows` the prompt rows); null = the bill's fill.
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    expert_rows_by_layer: LayerRows = .uniform,
    prompt_residents: PromptResidents = .keep,
    residency: expert_usage.Settings = .{},
    tally: expert_tally.Settings = .{ .ratio = default_tally_ratio },
    gpu_keep_warm: gpu_warm.Route = .waits,
    gpu_keep_warm_requested: bool = false,
    decode_glue: DecodeGlue = .fused,
    decode_glue_requested: bool = false,
    malformed: Malformed = .{},
    /// The load's facts.
    memory_baseline_bytes: ?u64 = null,
    wired_margin_bytes: u64 = 0,
    effective_context: ?u64 = null,
    kv_quant_bits: ?u8 = null,

    pub const Malformed = struct { rows: bool = false, prompt_residents: bool = false, gpu_keep_warm: bool = false, decode_glue: bool = false };

    /// The pack's `expert_streaming.settings` object (after `unknownSetting`), or a host's per-model settings object.
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        if (raw != .object) return;
        const o = raw.object;
        sdk_settings.rows(o, "expert_rows", &c.expert_rows, &c.malformed.rows);
        sdk_settings.rows(o, "expert_prefill_rows", &c.expert_prefill_rows, &c.malformed.rows);
        if (o.get("expert_rows_by_layer")) |v| c.expert_rows_by_layer = LayerRows.parse(v);
        _ = sdk_settings.route(PromptResidents, o, "prompt_residents", &c.prompt_residents, &c.malformed.prompt_residents);
        if (o.get("gpu_keep_warm")) |v| {
            const r: gpu_warm.Route = if (v == .string) gpu_warm.parseRoute(v.string) else .invalid;
            if (r == .invalid) c.malformed.gpu_keep_warm = true else c.gpu_keep_warm = r;
            c.gpu_keep_warm_requested = true;
        }
        if (sdk_settings.route(DecodeGlue, o, "decode_glue", &c.decode_glue, &c.malformed.decode_glue)) c.decode_glue_requested = true;
        _ = c.tally.apply(o);
        _ = c.residency.apply(o);
    }

    /// The decode glue the construction installs on a stream (`gpu`: the stream runs on the GPU): the setting's, except
    /// that the DEFAULT resolves to `ops` where the texts do not register the model's router or on a CPU stream; `fused`
    /// named on a CPU stream refuses by name (`validate` refuses it on another router).
    pub fn decodeGlueRoute(c: *const Config, gpu: bool) !DecodeGlue {
        if (c.decode_glue == .ops or !glue.registered(&c.model)) return .ops;
        if (!gpu) return if (c.decode_glue_requested) error.BeamDecodeGlueNeedsGpu else .ops;
        return .fused;
    }

    /// The glue the bill charges (`decodeGlueRoute` on a GPU stream: an upper bound on a CPU stream).
    pub fn decodeGlueBilled(c: *const Config) DecodeGlue {
        return if (c.decode_glue == .fused and glue.registered(&c.model)) .fused else .ops;
    }

    /// The keep-warm route the construction installs on a stream (`gpu`: the stream runs on the GPU): the setting's,
    /// except that the DEFAULT on a CPU stream resolves to `off` (its waits are host waits; MiMo's rule).
    pub fn warmRoute(c: *const Config, gpu: bool) gpu_warm.Route {
        if (c.gpu_keep_warm != .off and !c.gpu_keep_warm_requested and !gpu) return .off;
        return c.gpu_keep_warm;
    }

    pub fn withFacts(c: Config, facts: *const sdk.LoadFacts) Config {
        var out = c;
        out.memory_baseline_bytes = facts.memory_baseline_bytes;
        out.wired_margin_bytes = facts.wired_margin_bytes;
        // A host-provided count wins; else the setting's.
        out.expert_rows = facts.expert_rows orelse c.expert_rows;
        out.expert_prefill_rows = facts.expert_prefill_rows orelse c.expert_prefill_rows;
        out.effective_context = facts.effective_context;
        out.kv_quant_bits = facts.kv_quant_bits;
        return out;
    }

    /// The positions a load bills: the served context (prompt and generation), else the standard request.
    pub fn reserveTokens(c: Config) u64 {
        return @min(c.model.max_position_embeddings, c.effective_context orelse fill_reserve_tokens);
    }

    /// Once, before the bill and any array: every refusal by name.
    pub fn validate(c: Config) !void {
        if (c.malformed.rows) return error.BeamExpertRowsSetting;
        if (c.malformed.prompt_residents) return error.BeamPromptResidentsSetting;
        if (c.malformed.gpu_keep_warm) return error.BeamGpuKeepWarmSetting;
        if (c.malformed.decode_glue) return error.BeamDecodeGlueSetting;
        if (c.decode_glue_requested and c.decode_glue == .fused and !glue.registered(&c.model)) return error.BeamDecodeGlueUnregistered;
        try c.tally.check("beam");
        try c.residency.check("beam");
        c.expert_rows_by_layer.check(c.model.moeLayers()) catch |e| return switch (e) {
            error.ExpertRowsByLayer => error.BeamExpertRowsByLayer,
            error.ExpertRowsByLayerLength => error.BeamExpertRowsByLayerLength,
            error.ExpertRowsByLayerSum => error.BeamExpertRowsByLayerSum,
        };
        if (c.kv_quant_bits) |b| if (b != 0 and b != 16) return error.BeamKvQuantUnsupported;
        if (c.expert_rows) |d| if (d > c.model.n_routed_experts) return error.BeamExpertRows;
        if (c.expert_prefill_rows) |p| if (p > c.model.n_routed_experts) return error.BeamExpertRows;
        if (c.expert_rows) |d| if (c.expert_prefill_rows) |p| if (p > d) return error.BeamPhaseRows;
        if (c.reserveTokens() == 0) return error.BeamContext;
    }
};

const testing = std.testing;

fn parseSettings(text: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
    defer parsed.deinit();
    if (unknownSetting(parsed.value.object)) |k| {
        std.debug.print("unknown {s}\n", .{k});
        return error.BeamSettingUnknown;
    }
    var c: Config = .{};
    c.applySettings(parsed.value);
    return c;
}

test "beam settings: the defaults, the operator's rows, the split and the shared groups' keys" {
    const d: Config = .{};
    try d.validate();
    try testing.expectEqual(LayerRows.uniform, d.expert_rows_by_layer);
    try testing.expectEqual(PromptResidents.keep, d.prompt_residents);
    try testing.expectEqual(@as(f64, 0), d.tally.ratio);
    try testing.expectEqual(gpu_warm.Route.off, d.warmRoute(false));
    try testing.expectEqual(gpu_warm.Route.waits, d.warmRoute(true));
    const c = try parseSettings("{\"expert_prefill_rows\":100,\"expert_rows\":120,\"expert_resident_ratio\":0.25,\"prompt_residents\":\"forget\",\"resident_seed\":\"none\"}");
    try c.validate();
    try testing.expectEqual(@as(?u32, 120), c.expert_rows);
    try testing.expectEqual(@as(?u32, 100), c.expert_prefill_rows);
    try testing.expectEqual(@as(f64, 0.25), c.tally.ratio);
    try testing.expectEqual(PromptResidents.forget, c.prompt_residents);
}

test "beam settings: every bad value refuses by name; an unknown key (a lane, a route) is not a beam setting" {
    const Case = struct { text: []const u8, err: anyerror };
    for ([_]Case{
        .{ .text = "{\"expert_rows\":0}", .err = error.BeamExpertRowsSetting },
        .{ .text = "{\"expert_rows\":300}", .err = error.BeamExpertRows },
        .{ .text = "{\"expert_rows\":100,\"expert_prefill_rows\":101}", .err = error.BeamPhaseRows },
        .{ .text = "{\"prompt_residents\":\"maybe\"}", .err = error.BeamPromptResidentsSetting },
        .{ .text = "{\"gpu_keep_warm\":\"always\"}", .err = error.BeamGpuKeepWarmSetting },
        .{ .text = "{\"decode_glue\":\"maybe\"}", .err = error.BeamDecodeGlueSetting },
        .{ .text = "{\"expert_rows_by_layer\":\"derived\"}", .err = error.BeamExpertRowsByLayer },
        .{ .text = "{\"expert_rows_by_layer\":[1,-1]}", .err = error.BeamExpertRowsByLayerLength },
        .{ .text = "{\"expert_resident_ratio\":2}", .err = error.ResidentRatioOutOfRange },
        .{ .text = "{\"resident_seed\":\"some\"}", .err = error.ResidentSeedSetting },
    }) |case| try testing.expectError(case.err, (try parseSettings(case.text)).validate());
    for ([_][]const u8{ "{\"mtp_lane\":true}", "{\"decode_moe\":\"fused\"}", "{\"draft_depth\":2}" }) |text|
        try testing.expectError(error.BeamSettingUnknown, parseSettings(text));
    var c: Config = .{};
    c.kv_quant_bits = 8;
    try testing.expectError(error.BeamKvQuantUnsupported, c.validate());
}

test "beam settings: the decode glue is fused by default on Beam's router and a GPU stream, ops elsewhere; fused named off its router refuses" {
    const d: Config = .{};
    try testing.expectEqual(DecodeGlue.fused, try d.decodeGlueRoute(true));
    try testing.expectEqual(DecodeGlue.ops, try d.decodeGlueRoute(false));
    try testing.expectEqual(DecodeGlue.fused, d.decodeGlueBilled());
    const ops_c = try parseSettings("{\"decode_glue\":\"ops\"}");
    try ops_c.validate();
    try testing.expectEqual(DecodeGlue.ops, try ops_c.decodeGlueRoute(true));
    try testing.expectEqual(DecodeGlue.ops, ops_c.decodeGlueBilled());
    var named = try parseSettings("{\"decode_glue\":\"fused\"}");
    try named.validate();
    try testing.expectError(error.BeamDecodeGlueNeedsGpu, named.decodeGlueRoute(false));
    // Another router (6 experts a token): the default resolves to ops; fused named refuses the load by name.
    var other: Config = .{ .model = .{ .n_experts_per_tok = 6 } };
    try other.validate();
    try testing.expectEqual(DecodeGlue.ops, try other.decodeGlueRoute(true));
    named.model = other.model;
    try testing.expectError(error.BeamDecodeGlueUnregistered, named.validate());
}
