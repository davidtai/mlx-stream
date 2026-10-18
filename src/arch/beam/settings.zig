//! beam's construction config: the pack's model (config.zig), the pack's `expert_streaming.settings` (read once at
//! construction; a key that is not a setting refuses the load by name) and the host's load facts. MiMo's contract: the
//! operator pins the rows with `expert_prefill_rows` / `expert_rows` (either alone forces both); `expert_rows_by_layer`
//! splits the decode rows between the routed layers (`uniform`, or offsets summing to 0; by default the calibrated
//! offsets on Beam 1.0's 49 routed layers, `uniform` on any other geometry: `layerSplit`); the rate tally's
//! keys (`expert_resident_ratio`, `expert_tally_window`, `expert_tally_file`: sdk_ext/expert/tally.zig, every streamed
//! arch's), the usage seed and record (`resident_seed`, `expert_usage_record`, `expert_usage_label`: usage.zig),
//! `prompt_residents` (keep, the default, or forget), MiMo's X12 keep-warm (`gpu_keep_warm`: waits, the default, or
//! off), the decode lane's expert route (`decode_moe`: hitpass, the default, fused or stock) and its glue
//! (`decode_glue`: fused, the default, or ops). The draft lane (lane.zig, Reflection's DFlash2 drafter from the pack's
//! sidecar, drafter.zig) takes glm5_next's keys: `mtp_lane` installs it (off, the serial control, by default),
//! `draft_depth` is its drafts a round (1 .. the drafter's block less one, default `default_draft_depth`), `draft_gate`
//! its confidence gate (theta in [0, 1), default 0: off), `mtp_acceptance` its rule (`exact`, the default, or `typical`:
//! lossy, an opt-in), `mtp_typical_delta` typical's delta ((0, 1], default `default_typical_delta`, named only with
//! `typical`) and `draft_path` the drafter's path on a sampled request (`greedy`, the default, or `sampled`: `DraftPath`).
//! A draft key without `mtp_lane: true` refuses: an inert knob is a lie.

const std = @import("std");
const sdk = @import("sdk");
const beam = @import("config.zig");
const acc = @import("../../sdk_ext/acceptance.zig");
const sdk_settings = @import("../../sdk_ext/settings.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const expert_tally = @import("../../sdk_ext.zig").expert.tally;
const expert_usage = @import("../../sdk_ext.zig").expert.usage;
const decode_kernels = @import("decode_kernels.zig");
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
pub const setting_keys = [_][]const u8{ "expert_rows", "expert_prefill_rows", "expert_rows_by_layer", "expert_resident_ratio", "expert_tally_window", "expert_tally_file", "prompt_residents", "gpu_keep_warm", "decode_moe", "decode_glue" } ++ draft_keys ++ expert_usage.keys;

/// The decode lane's routed-expert route, installed once at construction: `hitpass` (the default: glm5_next's, the
/// shared scheduler's DEVROUTE) the fused texts with each call's hit wave run from the layer's resident slot LUT on the
/// device, committed behind the routing barrier before the host reads the ids (the host routes and builds only the miss
/// parts; exact: the same rows as `fused`); `fused` the qmv texts (decode_kernels.zig: gate + up + the f32 SwiGLU, then
/// down, one launch of each per wave over the three banks; bit for bit the stock route's rows) with the hit wave built
/// by the host after the ids; `stock` MLX's gather_qmm nvfp4 per bank (two gathers and the f32 SwiGLU, then a gather per
/// down). The default resolves to `stock` where the set is not registered (another routed geometry: a test fixture's)
/// or on a CPU stream (`decodeMoeRoute`); `hitpass` or `fused` named there refuses the load by name.
pub const DecodeMoe = enum { hitpass, fused, stock };

/// The decode lane's glue, installed once at construction: `fused` (the default) the router's and the predictor's texts
/// (glue.zig: the router's product and its tail in two launches, the predictor's in one), each layer's q, k, v and gate
/// projections in one buffer and one GEMV (graph.zig `fuseProjections`), its q / k norms with RoPE or the query scale
/// in one launch (glue.zig `qkNorm`) and each sublayer boundary (the sublayer's norm, the depth-scaled add, the next
/// sublayer's norm) in one launch (glue.zig `addNorm`), each bit for bit the stock ops; `ops` the stock op chain (the
/// router's input cast, GEMV and 17 ops; the predictor's four; four projection GEMVs; two norms and two ropes, or two
/// norms and a scale; a boundary's two norms and its add). The default resolves to `ops` where the texts do not
/// register the model's router (another routed geometry) or on a CPU stream (`decodeGlueRoute`); `fused` named there
/// refuses the load by name.
pub const DecodeGlue = enum { fused, ops };
/// The draft lane's keys (glm5_next's names).
pub const draft_keys = [_][]const u8{ "mtp_lane", "draft_depth", "draft_gate", "mtp_acceptance", "mtp_typical_delta", "draft_path" };

/// The lane's default drafts a round: 3, the depth measured and kept (docs/beam.md, the draft lane: exact depth 3 is
/// the fastest exact depth on the stack's fold 1, and its ABBA on fold 4 gives +19.8 % decode).
pub const default_draft_depth: u32 = 3;
/// The most drafts a round names: the acceptance graphs' rows less the bonus row and one (`acceptance.max_rows` 17), the
/// trained block's 15. The lane refuses a depth past its drafter's own block less one (`BeamDraftDepthOverBlock`).
pub const max_draft_depth: u32 = acc.max_rows - 2;
/// Typical acceptance's default delta (glm5_next's).
pub const default_typical_delta: f32 = 0.5;
/// The lane's acceptance rule (`mtp_acceptance`), installed at construction.
pub const MtpAcceptance = enum { exact, typical };
/// The drafter's path on a sampled request (`draft_path`), installed at construction: `greedy` (the default) the
/// selector's greedy walk, a point mass; `sampled` each draft drawn from the selector's distribution over its candidates
/// (softmax of the scores at the request's temperature) at the target's own uniform for that token, by the inverse CDF
/// in token-id order (drafter.zig `sampledPathOf`: glm5_next's `boundedDraw` pattern). The verify is the same under
/// both: each emitted token is the target's draw at its position, so the path only steers acceptance. A greedy request
/// takes the greedy walk under both.
pub const DraftPath = enum { greedy, sampled };

pub const PromptResidents = expert_usage.PromptResidents;
pub const LayerRows = sdk_settings.LayerRows;

pub fn unknownSetting(o: std.json.ObjectMap) ?[]const u8 {
    return sdk_settings.unknownSetting(o, &setting_keys);
}

/// Beam 1.0's default decode split (`expert_rows_by_layer`): 49 offsets (routed layer 0, model layer 3, first) that sum to
/// 0, each at least -10, so that every layer keeps its prompt rows at the pinned 173 / 183. They come from two prompts
/// that are not the benchmark's (`explainer.json`, `prose.json`: 16,695 and 16,456 prompt tokens, 1,024 generated, seed
/// 1234, uniform 173 / 183 rows), never from the benchmark prompt's own trace. Each routed layer's decode is replayed from
/// its recorded handover under the stream's decode policy at each decode row count (the replay matches the recorded
/// hits on each step of a fresh request). The 490 rows above the prompt rows then go one at a time to the layer whose
/// next row removes the most gated steps (a step with at least one miss), over the two prompts' mean.
pub const default_layer_offsets = [_]i16{ 58, 70, 70, 50, 58, 8, 34, 10, -2, -6, -6, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -6, -6, -10, -6, -6, -6, -6, -2, -10, -10, -6, -10 };

comptime {
    var sum: i32 = 0;
    for (default_layer_offsets) |x| sum += x;
    const model: beam.Config = .{};
    if (sum != 0 or default_layer_offsets.len != model.moeLayers()) @compileError("beam: the default per-layer offsets must be one a routed layer, summing to 0");
}

pub fn defaultLayerRows() LayerRows {
    var o: LayerRows.Offsets = .{ .len = default_layer_offsets.len };
    @memcpy(o.d[0..default_layer_offsets.len], &default_layer_offsets);
    return .{ .offsets = o };
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
    /// The pack's or the host's `expert_rows_by_layer`; null when neither names one (`layerSplit` gives the default).
    expert_rows_by_layer: ?LayerRows = null,
    prompt_residents: PromptResidents = .keep,
    residency: expert_usage.Settings = .{},
    tally: expert_tally.Settings = .{ .ratio = default_tally_ratio },
    gpu_keep_warm: gpu_warm.Route = .waits,
    gpu_keep_warm_requested: bool = false,
    decode_moe: DecodeMoe = .hitpass,
    decode_moe_requested: bool = false,
    decode_glue: DecodeGlue = .fused,
    decode_glue_requested: bool = false,
    /// The draft lane (`mtp_lane`): installed at construction from the pack's sidecar, or absent (the serial control).
    mtp_lane: bool = false,
    /// Drafts a round (`draft_depth`); 0 marks a value outside 1 .. `max_draft_depth` or not an integer.
    draft_depth: u32 = default_draft_depth,
    /// The draft gate's theta (`draft_gate`); NaN marks a value outside [0, 1) or not a number.
    draft_gate: f32 = 0,
    mtp_acceptance: MtpAcceptance = .exact,
    mtp_typical_delta: f32 = default_typical_delta,
    mtp_typical_delta_set: bool = false,
    draft_path: DraftPath = .greedy,
    /// A draft key other than `mtp_lane` was named.
    draft_named: bool = false,
    malformed: Malformed = .{},
    /// The load's facts.
    memory_baseline_bytes: ?u64 = null,
    wired_margin_bytes: u64 = 0,
    effective_context: ?u64 = null,
    kv_quant_bits: ?u8 = null,

    pub const Malformed = struct { rows: bool = false, prompt_residents: bool = false, gpu_keep_warm: bool = false, decode_moe: bool = false, decode_glue: bool = false, mtp_lane: bool = false, mtp_acceptance: bool = false, mtp_typical_delta: bool = false, draft_path: bool = false };

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
        if (sdk_settings.route(DecodeMoe, o, "decode_moe", &c.decode_moe, &c.malformed.decode_moe)) c.decode_moe_requested = true;
        if (sdk_settings.route(DecodeGlue, o, "decode_glue", &c.decode_glue, &c.malformed.decode_glue)) c.decode_glue_requested = true;
        _ = c.tally.apply(o);
        _ = c.residency.apply(o);
        c.applyDraft(o);
    }

    /// The draft lane's keys (glm5_next's parse): a malformed value is marked, refused by name in `validate`.
    fn applyDraft(c: *Config, o: std.json.ObjectMap) void {
        if (o.get("mtp_lane")) |v| {
            if (v == .bool) c.mtp_lane = v.bool else c.malformed.mtp_lane = true;
        }
        if (o.get("draft_depth")) |v| {
            c.draft_named = true;
            c.draft_depth = if (v == .integer and v.integer >= 1 and v.integer <= max_draft_depth) @intCast(v.integer) else 0;
        }
        if (o.get("draft_gate")) |v| {
            c.draft_named = true;
            c.draft_gate = switch (v) {
                .float => |f| @floatCast(f),
                .integer => |i| @floatFromInt(i),
                else => std.math.nan(f32),
            };
        }
        if (o.get("mtp_acceptance")) |v| {
            c.draft_named = true;
            const m: ?MtpAcceptance = if (v == .string) std.meta.stringToEnum(MtpAcceptance, v.string) else null;
            if (m) |x| c.mtp_acceptance = x else c.malformed.mtp_acceptance = true;
        }
        if (o.get("mtp_typical_delta")) |v| {
            c.draft_named = true;
            c.mtp_typical_delta_set = true;
            const x: f64 = switch (v) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => std.math.nan(f64),
            };
            if (std.math.isFinite(x) and x > 0 and x <= 1) c.mtp_typical_delta = @floatCast(x) else c.malformed.mtp_typical_delta = true;
        }
        if (o.get("draft_path")) |v| {
            c.draft_named = true;
            const m: ?DraftPath = if (v == .string) std.meta.stringToEnum(DraftPath, v.string) else null;
            if (m) |x| c.draft_path = x else c.malformed.draft_path = true;
        }
    }

    /// The lane's rule (construction): exact, or typical at the setting's delta.
    pub fn draftRule(c: Config) acc.Rule {
        return switch (c.mtp_acceptance) {
            .exact => .exact,
            .typical => .{ .typical = .{ .delta = c.mtp_typical_delta } },
        };
    }

    /// The model's routed experts are the fused set's (decode_kernels.zig `registeredAt`).
    pub fn fusedRegistered(c: *const Config) bool {
        return decode_kernels.registeredAt(c.model.hidden_size, c.model.moe_intermediate_size, c.model.n_experts_per_tok);
    }

    /// The decode route the construction installs on a stream (`gpu`: the stream runs on the GPU): the setting's, except
    /// that the DEFAULT resolves to `stock` where the fused set is not registered or on a CPU stream (the texts need a
    /// GPU stream); `hitpass` or `fused` named on a CPU stream refuses by name (`validate` refuses it on another
    /// geometry).
    pub fn decodeMoeRoute(c: *const Config, gpu: bool) !DecodeMoe {
        if (c.decode_moe == .stock) return .stock;
        if (!c.fusedRegistered()) return .stock;
        if (!gpu) return if (c.decode_moe_requested) error.BeamDecodeMoeNeedsGpu else .stock;
        return c.decode_moe;
    }

    /// The decode route the bill charges (`decodeMoeRoute` on a GPU stream: an upper bound on a CPU stream).
    pub fn decodeMoeBilled(c: *const Config) DecodeMoe {
        return if (c.decode_moe != .stock and c.fusedRegistered()) c.decode_moe else .stock;
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

    /// The split the construction installs: the setting when one is named; else `default_layer_offsets` on a model of
    /// their 49 routed layers (Beam 1.0) and `uniform` on any other (the fixtures' geometries).
    pub fn layerSplit(c: *const Config) LayerRows {
        if (c.expert_rows_by_layer) |r| return r;
        return if (c.model.moeLayers() == default_layer_offsets.len) defaultLayerRows() else .uniform;
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
        if (c.malformed.decode_moe) return error.BeamDecodeMoeSetting;
        if (c.decode_moe_requested and c.decode_moe != .stock and !c.fusedRegistered()) return error.BeamDecodeMoeUnregistered;
        if (c.malformed.decode_glue) return error.BeamDecodeGlueSetting;
        if (c.decode_glue_requested and c.decode_glue == .fused and !glue.registered(&c.model)) return error.BeamDecodeGlueUnregistered;
        if (c.malformed.mtp_lane) return error.BeamMtpLaneSetting;
        if (c.malformed.mtp_acceptance) return error.BeamMtpAcceptanceSetting;
        if (c.malformed.mtp_typical_delta) return error.BeamMtpTypicalDeltaSetting;
        if (c.malformed.draft_path) return error.BeamDraftPathSetting;
        if (c.draft_depth == 0) return error.BeamDraftDepthRange;
        if (!(c.draft_gate >= 0 and c.draft_gate < 1)) return error.BeamDraftGateRange;
        if (c.draft_named and !c.mtp_lane) return error.BeamDraftSettingWithoutLane;
        if (c.mtp_typical_delta_set and c.mtp_acceptance != .typical) return error.BeamMtpTypicalDeltaWithoutTypical;
        try c.tally.check("beam");
        try c.residency.check("beam");
        c.layerSplit().check(c.model.moeLayers()) catch |e| return switch (e) {
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
    try testing.expect(d.expert_rows_by_layer == null);
    const split = d.layerSplit();
    try testing.expectEqualSlices(i16, &default_layer_offsets, split.offsets.slice());
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

test "beam settings: expert_rows_by_layer: the calibrated offsets by default on Beam 1.0, uniform on another geometry and by name; the rows keep the plan's totals" {
    var prompt: [49]u32 = undefined;
    var decode: [49]u32 = undefined;
    const d: Config = .{};
    // The default at the pinned rows: every layer keeps its 173 prompt rows, the decode rows are 183 + the offset.
    try layerRows(d.layerSplit(), .{ .prompt = 173, .decode = 183 }, 256, &prompt, &decode);
    for (prompt, decode, default_layer_offsets) |p, r, off| {
        try testing.expectEqual(@as(u32, 173), p);
        try testing.expectEqual(@as(i64, 183 + off), @as(i64, r));
    }
    // At other plan rows the totals stay the plan's: prompt rows uniform, decode rows >= prompt rows, at most 256.
    for ([_]sdk.Rows{ .{ .prompt = 178, .decode = 189 }, .{ .prompt = 180, .decode = 183 }, .{ .prompt = 120, .decode = 140 } }) |rows| {
        try layerRows(d.layerSplit(), rows, 256, &prompt, &decode);
        var sum_d: u64 = 0;
        for (prompt, decode) |p, r| {
            try testing.expectEqual(rows.prompt, p);
            try testing.expect(r >= p and r <= 256);
            sum_d += r;
        }
        try testing.expectEqual(@as(u64, 49) * rows.decode, sum_d);
    }
    // Uniform by name.
    const uni = try parseSettings("{\"expert_rows_by_layer\":\"uniform\"}");
    try uni.validate();
    try testing.expect(uni.layerSplit() == .uniform);
    // Another geometry (a fixture's 3 routed layers): uniform by default, and it validates.
    var small: Config = .{};
    small.model.n_layers = 6;
    try small.validate();
    try testing.expect(small.layerSplit() == .uniform);
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
        .{ .text = "{\"decode_moe\":\"banked\"}", .err = error.BeamDecodeMoeSetting },
        .{ .text = "{\"decode_moe\":1}", .err = error.BeamDecodeMoeSetting },
    }) |case| try testing.expectError(case.err, (try parseSettings(case.text)).validate());
    for ([_][]const u8{ "{\"mtp_heads\":true}", "{\"decode_dsa\":\"fused\"}", "{\"draft_head\":\"q4\"}" }) |text|
        try testing.expectError(error.BeamSettingUnknown, parseSettings(text));
    var c: Config = .{};
    c.kv_quant_bits = 8;
    try testing.expectError(error.BeamKvQuantUnsupported, c.validate());
}

test "beam settings: decode_moe: hitpass by default where the set is registered, stock elsewhere or on a CPU stream; named hitpass or fused refuses there" {
    const d: Config = .{};
    try d.validate();
    try testing.expectEqual(DecodeMoe.hitpass, try d.decodeMoeRoute(true));
    try testing.expectEqual(DecodeMoe.stock, try d.decodeMoeRoute(false));
    try testing.expectEqual(DecodeMoe.hitpass, d.decodeMoeBilled());
    const hp = try parseSettings("{\"decode_moe\":\"hitpass\"}");
    try hp.validate();
    try testing.expectEqual(DecodeMoe.hitpass, try hp.decodeMoeRoute(true));
    try testing.expectError(error.BeamDecodeMoeNeedsGpu, hp.decodeMoeRoute(false));
    const st = try parseSettings("{\"decode_moe\":\"stock\"}");
    try st.validate();
    try testing.expectEqual(DecodeMoe.stock, try st.decodeMoeRoute(true));
    try testing.expectEqual(DecodeMoe.stock, st.decodeMoeBilled());
    const named = try parseSettings("{\"decode_moe\":\"fused\"}");
    try named.validate();
    try testing.expectEqual(DecodeMoe.fused, try named.decodeMoeRoute(true));
    try testing.expectError(error.BeamDecodeMoeNeedsGpu, named.decodeMoeRoute(false));
    // A fixture's narrow trunk: the default resolves to stock; fused named is refused.
    var fixture: Config = .{};
    fixture.model.hidden_size = 256;
    fixture.model.moe_intermediate_size = 128;
    fixture.model.n_experts_per_tok = 2;
    try fixture.validate();
    try testing.expectEqual(DecodeMoe.stock, try fixture.decodeMoeRoute(true));
    try testing.expectEqual(DecodeMoe.stock, fixture.decodeMoeBilled());
    var fixture_named = named;
    fixture_named.model = fixture.model;
    try testing.expectError(error.BeamDecodeMoeUnregistered, fixture_named.validate());
    var fixture_hp = hp;
    fixture_hp.model = fixture.model;
    try testing.expectError(error.BeamDecodeMoeUnregistered, fixture_hp.validate());
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

test "beam settings: the draft lane's keys (off by default; exact the default rule; typical opt-in), each bad value refused by name" {
    const d: Config = .{};
    try testing.expect(!d.mtp_lane);
    try testing.expectEqual(acc.Rule.exact, d.draftRule());
    try testing.expectEqual(@as(u32, 3), d.draft_depth);
    try testing.expectEqual(DraftPath.greedy, d.draft_path);
    const sampled = try parseSettings("{\"mtp_lane\":true,\"draft_path\":\"sampled\"}");
    try sampled.validate();
    try testing.expectEqual(DraftPath.sampled, sampled.draft_path);
    try testing.expectEqual(acc.Rule.exact, sampled.draftRule());
    const on = try parseSettings("{\"mtp_lane\":true,\"draft_depth\":15,\"draft_gate\":0.5,\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":0.2}");
    try on.validate();
    try testing.expectEqual(@as(u32, 15), on.draft_depth);
    try testing.expectEqual(@as(f32, 0.5), on.draft_gate);
    try testing.expectEqual(acc.Rule{ .typical = .{ .delta = 0.2 } }, on.draftRule());
    const exact = try parseSettings("{\"mtp_lane\":true,\"mtp_acceptance\":\"exact\",\"draft_depth\":1}");
    try exact.validate();
    try testing.expectEqual(acc.Rule.exact, exact.draftRule());
    const Case = struct { text: []const u8, err: anyerror };
    for ([_]Case{
        .{ .text = "{\"mtp_lane\":1}", .err = error.BeamMtpLaneSetting },
        .{ .text = "{\"mtp_lane\":true,\"draft_depth\":0}", .err = error.BeamDraftDepthRange },
        .{ .text = "{\"mtp_lane\":true,\"draft_depth\":16}", .err = error.BeamDraftDepthRange },
        .{ .text = "{\"mtp_lane\":true,\"draft_depth\":2.5}", .err = error.BeamDraftDepthRange },
        .{ .text = "{\"mtp_lane\":true,\"draft_gate\":1}", .err = error.BeamDraftGateRange },
        .{ .text = "{\"mtp_lane\":true,\"draft_gate\":\"high\"}", .err = error.BeamDraftGateRange },
        .{ .text = "{\"mtp_lane\":true,\"mtp_acceptance\":\"tokenv3\"}", .err = error.BeamMtpAcceptanceSetting },
        .{ .text = "{\"mtp_lane\":true,\"draft_path\":\"beam\"}", .err = error.BeamDraftPathSetting },
        .{ .text = "{\"mtp_lane\":true,\"draft_path\":true}", .err = error.BeamDraftPathSetting },
        .{ .text = "{\"draft_path\":\"sampled\"}", .err = error.BeamDraftSettingWithoutLane },
        .{ .text = "{\"mtp_lane\":true,\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":0}", .err = error.BeamMtpTypicalDeltaSetting },
        .{ .text = "{\"mtp_lane\":true,\"mtp_typical_delta\":0.3}", .err = error.BeamMtpTypicalDeltaWithoutTypical },
        .{ .text = "{\"draft_depth\":3}", .err = error.BeamDraftSettingWithoutLane },
        .{ .text = "{\"mtp_lane\":false,\"mtp_acceptance\":\"typical\"}", .err = error.BeamDraftSettingWithoutLane },
    }) |case| try testing.expectError(case.err, (try parseSettings(case.text)).validate());
}
