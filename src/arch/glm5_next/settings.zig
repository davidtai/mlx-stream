//! glm5_next's construction config: the pack's model (config.zig), the pack's `expert_streaming.settings` (read once at
//! construction; a key that is not a setting refuses the load by name) and the host's load facts. MiMo's contract: the
//! operator pins the rows with `expert_prefill_rows` / `expert_rows` (either alone forces both); `mtp_lane` installs the
//! MTP draft lane (mtp.zig; off, the serial control, by default); `decode_moe` the decode lane's expert route (`hitpass`,
//! the default: the NVFP4 decode texts with the device hit pass; `fused`, the texts with the host-routed hit wave, its
//! control; `stock`, MLX's gathers, the texts' control); `decode_glue` the decode lane's glue (`fused`, the default: the
//! mHC hcPre chain, the router and top-8 texts, the combine with the shared expert's add, glue.zig; `ops`, the stock op
//! chains, the control); `decode_kda` the decode lane's KDA layer (`fused`, the default: one text a layer, kda_decode.zig;
//! `ops`, the op chain, the control); `decode_dsa` the decode lane's sparse latent attention (`fused`, the default: two
//! texts, dsa_decode.zig; `ops`, the op chain, the control); `decode_predict` the lookahead predictor's router weight
//! (`bf16`, the default: the stored BF16 weight; `f32`, the transposed F32 copy, the control); `decode_swiglu` the decode lane's MLP activation
//! (`fused`, the default: one text, swiglu.zig; `ops`, the op chain, the control); `logits_dump_dir` (an absolute directory) installs the logits dump route for the KL gate
//! (dump.zig): absent, nothing of it is built.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("config.zig");

/// The context a load bills when the model sets none: the standard request (16,384 prompt tokens and 1,024 generated).
pub const fill_reserve_tokens: u64 = 16_384 + 1_024;
/// The host side of the process the bill charges (MiMo's declared host runtime and MLX's compiled kernels).
pub const host_bytes: u64 = 1_000_000_000;
pub const host_construction_bytes: u64 = 400_000_000;
pub const compilation_bytes: u64 = 512 << 20;
/// MLX's buffer cache limit through the request (the decode phase's and the prompt pass's).
pub const allocator_cache_bytes: u64 = 256 << 20;
/// The reader's staging per worker: Flash's gate/up span (9,437,192 B) plus its page alignment and a page (bank.zig).
pub const reader_staging_bytes: u64 = 10 << 20;

/// Every key `applySettings` reads; `expert_streaming.settings` may name these and nothing else (`unknownSetting`).
pub const setting_keys = [_][]const u8{ "expert_rows", "expert_prefill_rows", "mtp_lane", "decode_moe", "decode_glue", "decode_kda", "decode_dsa", "decode_predict", "decode_swiglu", "logits_dump_dir" };

/// The decode lane's routed-expert route, installed once at construction: `stock` MLX's gather_qmm nvfp4 (two gathers
/// and the clamped SwiGLU, then a gather per down), `fused` the NVFP4 decode texts (decode_kernels.zig: MiMo's k1 family
/// over the bank; one launch of each per wave, rounding-class against the stock route), `hitpass` (the default: MiMo's
/// HITPASS, the shared scheduler's DEVROUTE) the fused texts with each decode-lane call's hit wave dispatched from the
/// layer's device-resident slot LUT behind the routing barrier's arrays, before the host reads the ids (the host then
/// routes and builds only the miss parts, whose rows replace their pairs in the join): exact against `fused`.
pub const DecodeMoe = enum { stock, fused, hitpass };
/// The decode lane's glue, installed once at construction (glue.zig): `fused` the hcPre chain (each sublayer's collapse
/// and norm with the previous expand in one dispatch), the router and top-8 texts and the combine with the shared expert's
/// add; `ops` the stock op chains (the control). Rounding-class against `ops`.
pub const DecodeGlue = enum { ops, fused };
/// The decode lane's KDA layer, installed once at construction (kda_decode.zig): `fused` the host's oMLX KDA decode text
/// in its BF16-gate variant (the conv, the gate stages, the norms, the gate, the delta rule and the gated norm in one
/// dispatch; the projections stay GEMVs); `ops` the op chain (the control). Rounding-class against `ops`.
pub const DecodeKda = enum { ops, fused };
/// The decode lane's DSA sparse latent attention, installed once at construction (dsa_decode.zig): `fused` a split text
/// (MiMo's dec_attn split pattern for MLA over the selected latent rows) and MiMo's merge text; `ops` the op chain (the
/// control). Rounding-class against `ops`.
pub const DecodeDsa = enum { ops, fused };
/// The lookahead predictor's scores (the next routed layer's router on this layer's routed input; they pick reads only,
/// never an expert or a weight of a forward), installed once at construction: `bf16` the router's BF16 weight as stored
/// [experts, hidden] (MiMo X19 `predict_router: bf16`: half the bytes of the F32 copy, no input cast); `f32` the
/// transposed F32 copy (the control). Token-exact: the forward's routing never reads them.
pub const DecodePredict = enum { f32, bf16 };
/// The decode lane's MLP activation (the dense MLPs' and the shared experts' clamped SwiGLU), installed once at
/// construction (swiglu.zig): `fused` one text between the gate / up GEMVs and the down GEMV; `ops` the op chain (the
/// control). Rounding-class against `ops`.
pub const DecodeSwiglu = enum { ops, fused };
/// The longest dump directory a setting may name.
pub const max_dump_dir: usize = 480;

pub fn unknownSetting(o: std.json.ObjectMap) ?[]const u8 {
    var it = o.iterator();
    while (it.next()) |kv| {
        for (setting_keys) |k| {
            if (std.mem.eql(u8, k, kv.key_ptr.*)) break;
        } else return kv.key_ptr.*;
    }
    return null;
}

pub const Config = struct {
    model: glm.Config = .{},
    /// The pack's directory: config.json, the resident shards, the bank (experts.bin and its manifest).
    model_dir: ?[]const u8 = null,
    /// The operator's rows (`expert_rows` the decode rows, `expert_prefill_rows` the prompt rows); null = the bill's fill.
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    mtp_lane: bool = false,
    decode_moe: DecodeMoe = .hitpass,
    /// The decode lane's glue (glue.zig `Route`): `fused` (the default) or `ops` (the stock op chains, the control).
    decode_glue: DecodeGlue = .fused,
    /// The decode lane's KDA layer (kda_decode.zig): `fused` (the default) or `ops` (the op chain, the control).
    decode_kda: DecodeKda = .fused,
    /// The decode lane's DSA sparse latent attention (dsa_decode.zig): `fused` (the default) or `ops` (the control).
    decode_dsa: DecodeDsa = .fused,
    /// The lookahead predictor's router weight: `bf16` (the default) or `f32` (the control).
    decode_predict: DecodePredict = .bf16,
    /// The decode lane's MLP activation (swiglu.zig): `fused` (the default) or `ops` (the control).
    decode_swiglu: DecodeSwiglu = .fused,
    /// `logits_dump_dir`: the dump route's directory (`dumpDir`), held in the config itself (no allocation).
    dump_dir_buf: [max_dump_dir]u8 = undefined,
    dump_dir_len: u16 = 0,
    /// A setting given a value it does not take (refused by name in `validate`).
    malformed: Malformed = .{},
    /// The load's facts.
    memory_baseline_bytes: ?u64 = null,
    wired_margin_bytes: u64 = 0,
    effective_context: ?u64 = null,
    kv_quant_bits: ?u8 = null,

    pub const Malformed = struct { rows: bool = false, mtp_lane: bool = false, decode_moe: bool = false, decode_glue: bool = false, decode_kda: bool = false, decode_dsa: bool = false, decode_predict: bool = false, decode_swiglu: bool = false, dump_dir: bool = false };

    /// The pack's `expert_streaming.settings` object (after `unknownSetting`), or a host's per-model settings object.
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        if (raw != .object) return;
        if (raw.object.get("expert_rows")) |v| c.expert_rows = c.rowsSetting(v);
        if (raw.object.get("expert_prefill_rows")) |v| c.expert_prefill_rows = c.rowsSetting(v);
        if (raw.object.get("mtp_lane")) |v| {
            if (v == .bool) c.mtp_lane = v.bool else c.malformed.mtp_lane = true;
        }
        if (raw.object.get("decode_moe")) |v| {
            const route: ?DecodeMoe = if (v == .string) std.meta.stringToEnum(DecodeMoe, v.string) else null;
            if (route) |r| c.decode_moe = r else c.malformed.decode_moe = true;
        }
        if (raw.object.get("decode_glue")) |v| {
            const route: ?DecodeGlue = if (v == .string) std.meta.stringToEnum(DecodeGlue, v.string) else null;
            if (route) |r| c.decode_glue = r else c.malformed.decode_glue = true;
        }
        if (raw.object.get("decode_kda")) |v| {
            const route: ?DecodeKda = if (v == .string) std.meta.stringToEnum(DecodeKda, v.string) else null;
            if (route) |r| c.decode_kda = r else c.malformed.decode_kda = true;
        }
        if (raw.object.get("decode_dsa")) |v| {
            const route: ?DecodeDsa = if (v == .string) std.meta.stringToEnum(DecodeDsa, v.string) else null;
            if (route) |r| c.decode_dsa = r else c.malformed.decode_dsa = true;
        }
        if (raw.object.get("decode_predict")) |v| {
            const route: ?DecodePredict = if (v == .string) std.meta.stringToEnum(DecodePredict, v.string) else null;
            if (route) |r| c.decode_predict = r else c.malformed.decode_predict = true;
        }
        if (raw.object.get("decode_swiglu")) |v| {
            const route: ?DecodeSwiglu = if (v == .string) std.meta.stringToEnum(DecodeSwiglu, v.string) else null;
            if (route) |r| c.decode_swiglu = r else c.malformed.decode_swiglu = true;
        }
        if (raw.object.get("logits_dump_dir")) |v| {
            const ok = v == .string and v.string.len > 1 and v.string.len <= max_dump_dir and v.string[0] == '/' and std.mem.indexOfScalar(u8, v.string, 0) == null;
            if (ok) {
                @memcpy(c.dump_dir_buf[0..v.string.len], v.string);
                c.dump_dir_len = @intCast(v.string.len);
            } else c.malformed.dump_dir = true;
        }
    }

    /// The logits dump route's directory (`logits_dump_dir`), null when the setting is absent.
    pub fn dumpDir(c: *const Config) ?[]const u8 {
        return if (c.dump_dir_len == 0) null else c.dump_dir_buf[0..c.dump_dir_len];
    }

    fn rowsSetting(c: *Config, v: std.json.Value) ?u32 {
        if (v == .integer and v.integer > 0 and v.integer <= std.math.maxInt(u32)) return @intCast(v.integer);
        c.malformed.rows = true;
        return null;
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
        if (c.malformed.rows) return error.Glm5NextExpertRowsSetting;
        if (c.malformed.mtp_lane) return error.Glm5NextMtpLaneSetting;
        if (c.malformed.dump_dir) return error.Glm5NextLogitsDumpDir;
        if (c.malformed.decode_moe) return error.Glm5NextDecodeMoeSetting;
        // The draft lane binds the MTP layer's residents BF16 (its spec); a pack declaring them FP8 cannot serve the lane.
        if (c.mtp_lane and c.model.mtp_fp8) return error.Glm5NextMtpResidentsFp8;
        if (c.malformed.decode_glue) return error.Glm5NextDecodeGlueSetting;
        if (c.malformed.decode_kda) return error.Glm5NextDecodeKdaSetting;
        if (c.malformed.decode_dsa) return error.Glm5NextDecodeDsaSetting;
        if (c.malformed.decode_predict) return error.Glm5NextDecodePredictSetting;
        if (c.malformed.decode_swiglu) return error.Glm5NextDecodeSwigluSetting;
        if (c.kv_quant_bits) |b| if (b != 0 and b != 16) return error.Glm5NextKvQuantUnsupported;
        if (c.expert_rows) |d| if (d > c.model.n_routed_experts) return error.Glm5NextExpertRows;
        if (c.expert_prefill_rows) |p| if (p > c.model.n_routed_experts) return error.Glm5NextExpertRows;
        if (c.expert_rows) |d| if (c.expert_prefill_rows) |p| if (p > d) return error.Glm5NextPhaseRows;
        if (c.reserveTokens() == 0) return error.Glm5NextContext;
    }
};

const testing = std.testing;

fn parseSettings(text: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
    defer parsed.deinit();
    if (unknownSetting(parsed.value.object)) |_| return error.Unknown;
    var c: Config = .{};
    c.applySettings(parsed.value);
    return c;
}

test "glm5_next settings: the rows pin, the lane, unknown keys and malformed values" {
    const pinned = try parseSettings("{\"expert_prefill_rows\":120,\"expert_rows\":130}");
    try testing.expectEqual(@as(?u32, 130), pinned.expert_rows);
    try testing.expectEqual(@as(?u32, 120), pinned.expert_prefill_rows);
    try pinned.validate();
    try testing.expectError(error.Unknown, parseSettings("{\"expert_rows\":1,\"decode_window\":2}"));
    try testing.expectError(error.Glm5NextExpertRowsSetting, (try parseSettings("{\"expert_rows\":0}")).validate());
    try testing.expectError(error.Glm5NextExpertRowsSetting, (try parseSettings("{\"expert_rows\":\"130\"}")).validate());
    try testing.expectError(error.Glm5NextPhaseRows, (try parseSettings("{\"expert_rows\":120,\"expert_prefill_rows\":121}")).validate());
    try testing.expectError(error.Glm5NextExpertRows, (try parseSettings("{\"expert_rows\":289}")).validate());
    // The lane: off by default, on by name; anything but a bool refused.
    try testing.expect(!(try parseSettings("{}")).mtp_lane);
    const on = try parseSettings("{\"mtp_lane\":true}");
    try on.validate();
    try testing.expect(on.mtp_lane);
    try testing.expectError(error.Glm5NextMtpLaneSetting, (try parseSettings("{\"mtp_lane\":1}")).validate());
    try (try parseSettings("{\"mtp_lane\":false}")).validate();
    // The dump route: an absolute directory, else refused by name; absent, none.
    const dumped = try parseSettings("{\"logits_dump_dir\":\"/tmp/kl/incoming\"}");
    try dumped.validate();
    try testing.expectEqualStrings("/tmp/kl/incoming", dumped.dumpDir().?);
    try testing.expect(pinned.dumpDir() == null);
    try testing.expectError(error.Glm5NextLogitsDumpDir, (try parseSettings("{\"logits_dump_dir\":\"relative/dir\"}")).validate());
    try testing.expectError(error.Glm5NextLogitsDumpDir, (try parseSettings("{\"logits_dump_dir\":7}")).validate());
    // The decode route: hitpass unless named; fused and stock (the controls) by name; anything else refused by name.
    try testing.expectEqual(DecodeMoe.hitpass, pinned.decode_moe);
    try testing.expectEqual(DecodeMoe.fused, (try parseSettings("{\"decode_moe\":\"fused\"}")).decode_moe);
    try testing.expectEqual(DecodeMoe.stock, (try parseSettings("{\"decode_moe\":\"stock\"}")).decode_moe);
    try testing.expectEqual(DecodeMoe.hitpass, (try parseSettings("{\"decode_moe\":\"hitpass\"}")).decode_moe);
    try (try parseSettings("{\"decode_moe\":\"stock\"}")).validate();
    try testing.expectError(error.Glm5NextDecodeMoeSetting, (try parseSettings("{\"decode_moe\":\"k1\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeMoeSetting, (try parseSettings("{\"decode_moe\":1}")).validate());
    // An FP8-declared MTP layer (config.zig `mtp_fp8`) refuses the lane by name; the serial control loads.
    {
        var fp8_mtp = try parseSettings("{\"mtp_lane\":true}");
        fp8_mtp.model.mtp_fp8 = true;
        try testing.expectError(error.Glm5NextMtpResidentsFp8, fp8_mtp.validate());
        fp8_mtp.mtp_lane = false;
        try fp8_mtp.validate();
    }
    // The decode KDA layer: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeKda.fused, pinned.decode_kda);
    try testing.expectEqual(DecodeKda.ops, (try parseSettings("{\"decode_kda\":\"ops\"}")).decode_kda);
    try testing.expectError(error.Glm5NextDecodeKdaSetting, (try parseSettings("{\"decode_kda\":\"percore\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeKdaSetting, (try parseSettings("{\"decode_kda\":0}")).validate());
    // The decode DSA attention: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeDsa.fused, pinned.decode_dsa);
    try testing.expectEqual(DecodeDsa.ops, (try parseSettings("{\"decode_dsa\":\"ops\"}")).decode_dsa);
    try testing.expectError(error.Glm5NextDecodeDsaSetting, (try parseSettings("{\"decode_dsa\":\"nax\"}")).validate());
    // The lookahead predictor: bf16 unless named; f32 (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodePredict.bf16, pinned.decode_predict);
    try testing.expectEqual(DecodePredict.f32, (try parseSettings("{\"decode_predict\":\"f32\"}")).decode_predict);
    try testing.expectError(error.Glm5NextDecodePredictSetting, (try parseSettings("{\"decode_predict\":\"fp8\"}")).validate());
    // The decode MLP activation: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeSwiglu.fused, pinned.decode_swiglu);
    try testing.expectEqual(DecodeSwiglu.ops, (try parseSettings("{\"decode_swiglu\":\"ops\"}")).decode_swiglu);
    try testing.expectError(error.Glm5NextDecodeSwigluSetting, (try parseSettings("{\"decode_swiglu\":true}")).validate());
    // The decode glue: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeGlue.fused, pinned.decode_glue);
    try testing.expectEqual(DecodeGlue.ops, (try parseSettings("{\"decode_glue\":\"ops\"}")).decode_glue);
    try (try parseSettings("{\"decode_glue\":\"fused\"}")).validate();
    try testing.expectError(error.Glm5NextDecodeGlueSetting, (try parseSettings("{\"decode_glue\":\"stock\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeGlueSetting, (try parseSettings("{\"decode_glue\":true}")).validate());
    // The host's facts: its rows win; its context is the reserve.
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10, .expert_rows = 90, .effective_context = 32_768, .wired_margin_bytes = 2 << 30 };
    const built = pinned.withFacts(&facts);
    try testing.expectEqual(@as(?u32, 90), built.expert_rows);
    try testing.expectEqual(@as(?u32, 120), built.expert_prefill_rows);
    try testing.expectEqual(@as(u64, 32_768), built.reserveTokens());
    try testing.expectEqual(fill_reserve_tokens, (Config{}).reserveTokens());
}
