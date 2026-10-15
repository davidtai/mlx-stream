//! glm5_next's construction config: the pack's model (config.zig), the pack's `expert_streaming.settings` (read once at
//! construction; a key that is not a setting refuses the load by name) and the host's load facts. MiMo's contract: the
//! operator pins the rows with `expert_prefill_rows` / `expert_rows` (either alone forces both); `mtp_lane` installs the
//! MTP draft lane (mtp.zig; off, the serial control, by default); `decode_moe` the decode lane's expert route (`stock`, MLX's
//! gathers, the default; `fused`, the NVFP4 decode texts); `logits_dump_dir` (an absolute directory) installs the logits
//! dump route for the KL gate (dump.zig): absent, nothing of it is built.

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
pub const setting_keys = [_][]const u8{ "expert_rows", "expert_prefill_rows", "mtp_lane", "decode_moe", "logits_dump_dir" };

/// The decode lane's routed-expert route, installed once at construction: `stock` MLX's gather_qmm nvfp4 (two gathers
/// and the clamped SwiGLU, then a gather per down), `fused` the NVFP4 decode texts (decode_kernels.zig: MiMo's k1 family
/// over the bank; one launch of each per wave, rounding-class against the stock route).
pub const DecodeMoe = enum { stock, fused };
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
    decode_moe: DecodeMoe = .stock,
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

    pub const Malformed = struct { rows: bool = false, mtp_lane: bool = false, decode_moe: bool = false, dump_dir: bool = false };

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
    // The decode route: stock unless named; fused by name; anything else refused by name.
    try testing.expectEqual(DecodeMoe.stock, pinned.decode_moe);
    try testing.expectEqual(DecodeMoe.fused, (try parseSettings("{\"decode_moe\":\"fused\"}")).decode_moe);
    try (try parseSettings("{\"decode_moe\":\"stock\"}")).validate();
    try testing.expectError(error.Glm5NextDecodeMoeSetting, (try parseSettings("{\"decode_moe\":\"k1\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeMoeSetting, (try parseSettings("{\"decode_moe\":1}")).validate());
    // The host's facts: its rows win; its context is the reserve.
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10, .expert_rows = 90, .effective_context = 32_768, .wired_margin_bytes = 2 << 30 };
    const built = pinned.withFacts(&facts);
    try testing.expectEqual(@as(?u32, 90), built.expert_rows);
    try testing.expectEqual(@as(?u32, 120), built.expert_prefill_rows);
    try testing.expectEqual(@as(u64, 32_768), built.reserveTokens());
    try testing.expectEqual(fill_reserve_tokens, (Config{}).reserveTokens());
}
