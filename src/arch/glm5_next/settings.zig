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
//! (`fused`, the default: one text, swiglu.zig; `ops`, the op chain, the control); `gpu_keep_warm` MiMo's X12 route
//! (`waits`, the default: a one-simdgroup spin on its own Metal queue while the stream's gated parts and read-ahead wait
//! for their bytes, mimo_v2/gpu_warm.zig; `off`, the control); `logits_dump_dir` (an absolute directory) installs the
//! logits dump route for the KL gate (dump.zig): absent, nothing of it is built.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("config.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");

/// The context a load bills when the model sets none: the standard request (16,384 prompt tokens and 1,024 generated).
pub const fill_reserve_tokens: u64 = 16_384 + 1_024;
/// The host side of the process the bill charges (MiMo's declared host runtime and MLX's compiled kernels).
pub const host_bytes: u64 = 1_000_000_000;
pub const host_construction_bytes: u64 = 400_000_000;
pub const compilation_bytes: u64 = 512 << 20;

/// MiMo's M1 (`host_runtime`) + M2 (`compiler_runtime`) for glm5_next, the DECODE phase only (bill.zig
/// `decodeHostTerms`): one measured bound for the process's host side, its footprint less MLX's active and cache (the
/// host's allocations, the compiled graphs and Metal pipelines: what neither MLX nor a geometric term counts), the
/// reading `Module.requestEnd` prints in its `glm5_next decode memory:` line. The rule is MiMo's and DeepSeek's: the
/// max-shape reading + 0.02 GB for a long decode, rounded up to the next 50 MB. Readings: all 112 of the served and
/// profile cells of 2026-10-12 (reports/.../mimo/cells/glm5next-*/**/server.log, stub runs excluded): BF16 and FP8
/// packs, serial and the MTP lane, 1, 1,024, 10,403 and 16,384 decode tokens, reserves 4,096 and 32,768; host side
/// 494.8-833.3 MB, the largest 833,268,278 B (g21b-vs-g21c 03-B: FP8, lane on, 115/131, MLX's cache read 0 at the end);
/// the 16K-token decodes read 494.8 and 550.2 MB (no growth with decode length). Hence 900 MB in decode for host_runtime
/// and compiler_runtime TOGETHER (the reading cannot split them): compiler_runtime's decode bytes are 0 and
/// host_runtime's are this. The prompt phase keeps both declared terms (no prompt-phase reading), and so does every
/// reserve the readings do not cover (`decode_host_measured_reserve`: 32,768, the largest read). MLX's cache term stays
/// declared (bounded by its limit and the overshoot term, not by a snapshot). Releases 636,870,912 B in decode.
pub const decode_host_reading_bytes: u64 = 833_268_278;
pub const decode_host_long_decode_bytes: u64 = 20_000_000;
pub const decode_host_bytes: u64 = (std.math.divCeil(u64, decode_host_reading_bytes + decode_host_long_decode_bytes, 50_000_000) catch unreachable) * 50_000_000;
/// The largest construction reserve (tokens) the readings cover; above it the decode phase bills the declared terms.
pub const decode_host_measured_reserve: u64 = 32_768;
/// The no-under-bill fixture (bill.zig test "M1-M3"): each reading's footprint, MLX active, MLX cache, its cell's process
/// bound and reserve: the largest host side of each pack and lane, the largest cache, the own-fill cells, the long
/// decodes and the smallest reserve.
pub const decode_host_readings = [_][5]u64{
    .{ 95_305_026_376, 94_471_758_098, 0, 98_472_578_980, 32_768 }, // g21b-vs-g21c 03-B (FP8, lane; the largest host side)
    .{ 95_617_616_664, 94_528_430_319, 276_389_478, 98_467_622_820, 32_768 }, // g19m-vs-g21b 02-B (FP8, lane; the largest cache)
    .{ 96_932_302_648, 95_861_683_071, 268_727_143, 99_189_569_332, 32_768 }, // g19m-serial-vs-lane 02-B (BF16, lane)
    .{ 97_995_707_200, 97_264_520_570, 8_586_644, 100_176_321_724, 32_768 }, // g19f-vs-g19f-stack3 03-B (BF16, serial)
    .{ 103_666_213_752, 103_047_808_854, 8_586_644, 105_510_501_940, 32_768 }, // g19q-vs-g19q3 02-B (FP8, serial, own fill)
    .{ 102_456_024_728, 101_606_242_482, 268_732_608, 105_024_025_756, 32_768 }, // g19-vs-g21c-ownfill 02-B (FP8, lane, own fill)
    .{ 103_207_182_840, 102_615_296_446, 8_586_644, 105_492_585_324, 32_768 }, // g19-vs-g21c-ownfill 01-A (BF16, serial, own fill)
    .{ 103_223_075_320, 102_615_296_446, 57_622_863, 105_492_585_324, 32_768 }, // glm5next-16k r05-16k (10,403 tokens)
    .{ 103_118_643_680, 102_615_296_446, 8_586_644, 105_492_589_484, 32_768 }, // profile16k p16k (16,384 tokens)
    .{ 86_480_924_336, 85_916_160_054, 28_309_072, 88_319_536_788, 4_096 }, // KL on the FP8 pack (reserve 4,096)
};
comptime {
    // The decode phase holds the constructed module's host side too.
    if (decode_host_bytes < host_construction_bytes) @compileError("decode host side under its construction bound");
    if (decode_host_bytes > host_bytes + compilation_bytes) @compileError("decode host side over the declared terms");
}
/// MLX's buffer cache limit through the request (the decode phase's and the prompt pass's).
pub const allocator_cache_bytes: u64 = 256 << 20;
/// The reader's staging per worker: Flash's gate/up span (9,437,192 B) plus its page alignment and a page (bank.zig).
pub const reader_staging_bytes: u64 = 10 << 20;

/// Every key `applySettings` reads; `expert_streaming.settings` may name these and nothing else (`unknownSetting`).
pub const setting_keys = [_][]const u8{ "expert_rows", "expert_prefill_rows", "mtp_lane", "decode_moe", "decode_glue", "decode_kda", "decode_dsa", "decode_predict", "decode_swiglu", "gpu_keep_warm", "logits_dump_dir" };

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
    /// MiMo's X12 keep-warm (mimo_v2/gpu_warm.zig, adopted there as the default: round -4.7 ms, decode +2.4 %, ids
    /// identical): `waits` (the default) warms the GPU while the stream's event-gated parts (decode miss parts, a prompt
    /// group's reads) and the host's read-ahead wait for their bytes; `off` (the control) installs nothing. Exact by
    /// construction (the engine touches no model array). On a CPU stream the default resolves to `off` (`warmRoute`).
    gpu_keep_warm: gpu_warm.Route = .waits,
    /// The settings named `gpu_keep_warm` (a named route refuses where the default resolves to the control).
    gpu_keep_warm_requested: bool = false,
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

    pub const Malformed = struct { rows: bool = false, mtp_lane: bool = false, decode_moe: bool = false, decode_glue: bool = false, decode_kda: bool = false, decode_dsa: bool = false, decode_predict: bool = false, decode_swiglu: bool = false, gpu_keep_warm: bool = false, dump_dir: bool = false };

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
        if (raw.object.get("gpu_keep_warm")) |v| {
            const route: gpu_warm.Route = if (v == .string) gpu_warm.parseRoute(v.string) else .invalid;
            if (route == .invalid) c.malformed.gpu_keep_warm = true else c.gpu_keep_warm = route;
            c.gpu_keep_warm_requested = true;
        }
        if (raw.object.get("logits_dump_dir")) |v| {
            const ok = v == .string and v.string.len > 1 and v.string.len <= max_dump_dir and v.string[0] == '/' and std.mem.indexOfScalar(u8, v.string, 0) == null;
            if (ok) {
                @memcpy(c.dump_dir_buf[0..v.string.len], v.string);
                c.dump_dir_len = @intCast(v.string.len);
            } else c.malformed.dump_dir = true;
        }
    }

    /// The keep-warm route the construction installs on a stream (`gpu`: the stream runs on the GPU): the setting's,
    /// except that the DEFAULT on a CPU stream resolves to `off` (its waits are host waits; MiMo's rule); a named `waits`
    /// stays and refuses by name at the install.
    pub fn warmRoute(c: *const Config, gpu: bool) gpu_warm.Route {
        if (c.gpu_keep_warm != .off and !c.gpu_keep_warm_requested and !gpu) return .off;
        return c.gpu_keep_warm;
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
        if (c.malformed.gpu_keep_warm) return error.Glm5NextGpuKeepWarmSetting;
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
    // The keep-warm route: waits unless named; off (the control) by name; anything else refused by name. The default on a
    // CPU stream resolves to off; a named waits stays (the install refuses it there by name).
    try testing.expectEqual(gpu_warm.Route.waits, pinned.gpu_keep_warm);
    try testing.expect(!pinned.gpu_keep_warm_requested);
    try testing.expectEqual(gpu_warm.Route.waits, pinned.warmRoute(true));
    try testing.expectEqual(gpu_warm.Route.off, pinned.warmRoute(false));
    const warm_off = try parseSettings("{\"gpu_keep_warm\":\"off\"}");
    try warm_off.validate();
    try testing.expectEqual(gpu_warm.Route.off, warm_off.warmRoute(true));
    const warm_named = try parseSettings("{\"gpu_keep_warm\":\"waits\"}");
    try warm_named.validate();
    try testing.expectEqual(gpu_warm.Route.waits, warm_named.warmRoute(false));
    try testing.expectError(error.Glm5NextGpuKeepWarmSetting, (try parseSettings("{\"gpu_keep_warm\":\"all\"}")).validate());
    try testing.expectError(error.Glm5NextGpuKeepWarmSetting, (try parseSettings("{\"gpu_keep_warm\":true}")).validate());
    // The host's facts: its rows win; its context is the reserve.
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10, .expert_rows = 90, .effective_context = 32_768, .wired_margin_bytes = 2 << 30 };
    const built = pinned.withFacts(&facts);
    try testing.expectEqual(@as(?u32, 90), built.expert_rows);
    try testing.expectEqual(@as(?u32, 120), built.expert_prefill_rows);
    try testing.expectEqual(@as(u64, 32_768), built.reserveTokens());
    try testing.expectEqual(fill_reserve_tokens, (Config{}).reserveTokens());
}
