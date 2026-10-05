//! DeepSeek-V4.1's own config on the host side: its sidecar paths, its prompt-pass bill, the load's facts, and the
//! model settings it takes with the tier defaults they fall back to. The fields keep the names the host's
//! ModelConfig gave them, so every reader in the package reads them unchanged.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const log = @import("sdk").log;
const v41 = @import("deepseek_v41.zig");

/// The arch's numerics, chosen at construction (`numeric_tier`): "stock" (the exact reference math, the prompt in
/// decode-width forwards) or "served" (the tier of record, its rounding-class prefill).
pub const NumericTier = enum { stock, served };

pub const Config = struct {
    /// The model directory (config.json, the resident shards, the expert bank) and the exported Engram token map
    /// beside it; owned by the host's config.
    expert_bank_dir: ?[]const u8 = null,
    engram_token_map_path: ?[]const u8 = null,
    /// The prompt pass's bill for the prefill admission.
    dsv41_prefill: ?v41.PrefillBill = null,
    /// The shell's facts: routed experts and layers.
    num_experts: u32 = 0,
    num_hidden_layers: u32 = 0,
    /// The load's facts: the memory in use before the load (`--memory-baseline-gb`, else the preflight's reading),
    /// the decode and prompt slot rows per layer (`--expert-rows`, a harness's; null = the admission's fill), the
    /// residents past the page cache (null = on).
    memory_baseline_bytes: ?u64 = null,
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    nocache_weights: ?bool = null,
    /// The routed waves wait on the reads' events instead of the host (null = the tier's default).
    expert_event_gates: ?bool = null,
    /// The read pool threads' scheduling (null = off).
    expert_reader_sched: ?sdk_ext.expert.Sched = null,
    /// The numerics, chosen at construction (null = served).
    numeric_tier: ?NumericTier = null,
    /// The prompt pass layer by layer (null = the tier's default).
    layer_major_prefill: ?bool = null,
    /// The longest request the served module admits (prompt tokens; null: the standard request's, `fill_prompt_tokens`).
    /// Set, the construction bills every prompt length up to it (`bill.billCovering`); a longer request is refused by name
    /// before its prompt pass (`error.ContextOverBill`).
    max_context_tokens: ?u32 = null,
    /// A `ctx_size` above the model's limit (`Config.max_ctx_size`): refused by name at load and construction
    /// (`error.CtxSizeOverModelLimit`), never a silent fall back to the standard request's context.
    ctx_size_over_limit: ?i64 = null,
    /// The wide prefill read schedule (null = the tier's defaults below).
    expert_wide_feed: ?bool = null,
    expert_wide_seed: ?bool = null,
    expert_wide_hot_first: ?bool = null,
    expert_wide_depth: ?u8 = null,
    /// Wide-call experts of at most this many rows on the decode GEMV (null = none).
    expert_wide_cold_rows: ?u8 = null,
    /// The wide call's base-bank rows as one deferred call.
    expert_wide_defer_base: ?bool = null,
    /// P1: each layer's predicted seed read ahead during its attention.
    expert_wide_read_ahead: ?bool = null,
    /// P1b: the seed's deferred base call run as soon as the seed has landed.
    expert_wide_base_at_seed: ?bool = null,
    /// The decode read-ahead's speculative records per layer call (1..4; null = 2).
    expert_lookahead_budget: ?u8 = null,
    /// P1c: the seed's ranks grouped apart from the stream's, the base call after the last seed group.
    expert_wide_seed_aligned: ?bool = null,
    /// P1d: the base rows resident at the barrier drain first, in their own call (null = off).
    expert_wide_resident_first: ?bool = null,
    /// The input embedding read from its host rows from construction (null = on).
    embedding_host_rows: ?bool = null,

    /// The host's load facts onto this config (the module and the bill read them under these names).
    pub fn withFacts(c: Config, facts: *const sdk.LoadFacts) Config {
        var out = c;
        out.memory_baseline_bytes = facts.memory_baseline_bytes;
        out.expert_rows = facts.expert_rows;
        out.expert_prefill_rows = facts.expert_prefill_rows;
        out.nocache_weights = facts.nocache_weights;
        return out;
    }

    /// This model's model-settings.json object: each key this arch takes, set when present and valid (anything
    /// else is unset), and one `[model-settings]` line when any was.
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        const obj = switch (raw) {
            .object => |o| o,
            else => return,
        };
        var any = false;
        inline for (.{ "expert_event_gates", "layer_major_prefill", "expert_wide_feed", "expert_wide_seed", "expert_wide_hot_first", "embedding_host_rows" }) |key| {
            if (obj.get(key)) |v| if (v == .bool) {
                @field(c, key) = v.bool;
                any = true;
            };
        }
        if (obj.get("numeric_tier")) |v| if (v == .string) {
            if (std.meta.stringToEnum(NumericTier, v.string)) |t| {
                c.numeric_tier = t;
                any = true;
            }
        };
        if (obj.get("expert_reader_sched")) |v| if (v == .string) {
            if (sdk_ext.expert.Sched.parse(v.string)) |rs| {
                c.expert_reader_sched = rs;
                any = true;
            }
        };
        if (obj.get("expert_wide_cold_rows")) |v| if (v == .integer and v.integer >= 1 and v.integer <= 8) {
            c.expert_wide_cold_rows = @intCast(v.integer);
            any = true;
        };
        if (obj.get("expert_wide_depth")) |v| if (v == .integer and v.integer >= 1 and v.integer <= 2) {
            c.expert_wide_depth = @intCast(v.integer);
            any = true;
        };
        // The model's context (`ctx_size`, the host's own key: the server refuses a longer prompt with a 400): the
        // construction bills every prompt up to it (`max_context_tokens`). Unset, the standard request's 16,384.
        if (obj.get("ctx_size")) |v| if (v == .integer and v.integer >= 1) {
            if (v.integer <= max_ctx_size) c.max_context_tokens = @intCast(v.integer) else c.ctx_size_over_limit = v.integer;
            any = true;
        };
        if (any) log.info("[model-settings] deepseek_v41: event_gates={s} numeric_tier={s} layer_major_prefill={s} wide_feed={s} wide_seed={s} wide_hot_first={s} wide_depth={d} wide_cold_rows={d} embedding_host_rows={s} billed_context={d}\n", .{
            onOff(c.expert_event_gates),  if (c.numeric_tier) |t| @tagName(t) else "default",
            onOff(c.layer_major_prefill), onOff(c.expert_wide_feed),
            onOff(c.expert_wide_seed),    onOff(c.expert_wide_hot_first),
            c.expert_wide_depth orelse 0, c.expert_wide_cold_rows orelse 0,
            onOff(c.embedding_host_rows),     c.max_context_tokens orelse 0,
        });
    }

    /// The longest `ctx_size` the bill takes (the bank's max_position_embeddings).
    pub const max_ctx_size: i64 = 1 << 20;

    fn onOff(v: ?bool) []const u8 {
        return if (v) |b| (if (b) "on" else "off") else "default";
    }

    /// The prefill routes as the module builds them: a setting when given, else the tier's default (the served
    /// tier: K16 layer-major, two wide groups in flight, the wide feed; the stock tier, whose prompt forwards are
    /// decode-width: none).
    pub fn dsv41LayerMajor(self: *const Config) bool {
        return self.layer_major_prefill orelse self.dsv41ServedTier();
    }

    /// The decode read-ahead's speculative records per layer call: 2 unless set.
    pub fn dsv41LookaheadBudget(self: *const Config) u8 {
        return self.expert_lookahead_budget orelse 2;
    }

    /// The served tier reads 3 groups ahead (P1's v1b: the SSD kept busy through the routed stage's drains).
    pub fn dsv41WideDepth(self: *const Config) u8 {
        return self.expert_wide_depth orelse if (self.dsv41ServedTier()) 5 else 1;
    }

    pub fn dsv41WideFeed(self: *const Config) bool {
        return self.expert_wide_feed orelse self.dsv41ServedTier();
    }

    /// The feed's halves: each its own setting, else the feed's value (the feed = seed + hot-first).
    pub fn dsv41WideSeed(self: *const Config) bool {
        return self.expert_wide_seed orelse self.dsv41WideFeed();
    }

    pub fn dsv41WideHotFirst(self: *const Config) bool {
        return self.expert_wide_hot_first orelse self.dsv41WideFeed();
    }

    /// The deferred base-bank call: the setting, else on for the served tier without cold rows.
    pub fn dsv41WideDeferBase(self: *const Config) bool {
        return self.expert_wide_defer_base orelse (self.dsv41ServedTier() and (self.expert_wide_cold_rows orelse 0) == 0);
    }

    /// P1's read-ahead: the setting, else on wherever the prompt pass is layer-major with the wide seed.
    pub fn dsv41WideReadAhead(self: *const Config) bool {
        return self.expert_wide_read_ahead orelse (self.dsv41LayerMajor() and self.dsv41WideSeed());
    }

    /// P1b's base call at the seed: the setting, else on wherever the wide seed and the deferred base call both are.
    pub fn dsv41WideBaseAtSeed(self: *const Config) bool {
        return self.expert_wide_base_at_seed orelse (self.dsv41WideSeed() and self.dsv41WideDeferBase());
    }

    /// P1c's seed-aligned groups: the setting, else on wherever the base call at the seed and the hottest-first order are.
    pub fn dsv41WideSeedAligned(self: *const Config) bool {
        return self.expert_wide_seed_aligned orelse (self.dsv41WideBaseAtSeed() and self.dsv41WideHotFirst());
    }

    /// P1d's resident-first base call: the setting, else off.
    pub fn dsv41WideResidentFirst(self: *const Config) bool {
        return self.expert_wide_resident_first orelse false;
    }

    fn dsv41ServedTier(self: *const Config) bool {
        return (self.numeric_tier orelse .served) == .served;
    }
};

const testing = std.testing;

fn settingsOf(json: []const u8) !Config {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var c: Config = .{};
    c.applySettings(try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}));
    return c;
}

test "dsv41 settings: numeric_tier names stock or served, anything else is unset" {
    try testing.expectEqual(@as(?NumericTier, .stock), (try settingsOf("{\"numeric_tier\": \"stock\"}")).numeric_tier);
    try testing.expectEqual(@as(?NumericTier, .served), (try settingsOf("{\"numeric_tier\": \"served\"}")).numeric_tier);
    try testing.expectEqual(@as(?NumericTier, null), (try settingsOf("{\"numeric_tier\": \"fast\"}")).numeric_tier);
}

test "dsv41 settings: expert_event_gates and embedding_host_rows are bools, anything else is unset" {
    try testing.expectEqual(@as(?bool, true), (try settingsOf("{\"expert_event_gates\": true}")).expert_event_gates);
    try testing.expectEqual(@as(?bool, null), (try settingsOf("{\"expert_event_gates\": 1}")).expert_event_gates);
    try testing.expectEqual(@as(?bool, false), (try settingsOf("{\"embedding_host_rows\": false}")).embedding_host_rows);
}

test "dsv41 settings: the prefill routes are a bool, a bool and a depth of 1 or 2; cold rows 1 to 8; anything else is unset" {
    const a = try settingsOf("{\"layer_major_prefill\": true, \"expert_wide_feed\": false, \"expert_wide_depth\": 2, \"expert_wide_cold_rows\": 8}");
    try testing.expectEqual(@as(?bool, true), a.layer_major_prefill);
    try testing.expectEqual(@as(?bool, false), a.expert_wide_feed);
    try testing.expectEqual(@as(?u8, 2), a.expert_wide_depth);
    try testing.expectEqual(@as(?u8, 8), a.expert_wide_cold_rows);
    const c = try settingsOf("{\"layer_major_prefill\": 1, \"expert_wide_depth\": 3, \"expert_wide_cold_rows\": 9, \"expert_wide_seed\": \"on\"}");
    try testing.expectEqual(@as(?bool, null), c.layer_major_prefill);
    try testing.expectEqual(@as(?u8, null), c.expert_wide_depth);
    try testing.expectEqual(@as(?u8, null), c.expert_wide_cold_rows);
    try testing.expectEqual(@as(?bool, null), c.expert_wide_seed);
    // Not an object: nothing set.
    try testing.expectEqual(Config{}, try settingsOf("[1]"));
}

test "dsv41 settings: ctx_size bills every prompt up to it; anything else leaves the standard request's" {
    try testing.expectEqual(@as(?u32, 132096), (try settingsOf("{\"ctx_size\": 132096}")).max_context_tokens);
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"ctx_size\": 0}")).max_context_tokens);
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"ctx_size\": \"131072\"}")).max_context_tokens);
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"ctx_size\": 2097152}")).max_context_tokens);
    // Above the model's 1,048,576: kept, so the load refuses it by name.
    try testing.expectEqual(@as(?i64, 2097152), (try settingsOf("{\"ctx_size\": 2097152}")).ctx_size_over_limit);
    try testing.expectEqual(@as(?i64, null), (try settingsOf("{\"ctx_size\": 1048576}")).ctx_size_over_limit);
}

test "dsv41 settings: the reader schedule parses its knob list, a conflict is unset; the load facts replace the fill's inputs" {
    try testing.expectEqual(@as(?sdk_ext.expert.Sched, .{ .qos = true, .spin = true }), (try settingsOf("{\"expert_reader_sched\": \"qos,spin\"}")).expert_reader_sched);
    try testing.expectEqual(@as(?sdk_ext.expert.Sched, .{}), (try settingsOf("{\"expert_reader_sched\": \"off\"}")).expert_reader_sched);
    try testing.expectEqual(@as(?sdk_ext.expert.Sched, null), (try settingsOf("{\"expert_reader_sched\": \"spin\"}")).expert_reader_sched);
    try testing.expectEqual(@as(?sdk_ext.expert.Sched, null), (try settingsOf("{\"expert_reader_sched\": true}")).expert_reader_sched);
    // The host's facts win over the settings' (rows, baseline, the page-cache setting); the routes stay the settings'.
    const s = try settingsOf("{\"numeric_tier\": \"stock\"}");
    var set = s;
    set.expert_rows = 99;
    const c = set.withFacts(&.{ .memory_baseline_bytes = 9_200_000_000, .expert_prefill_rows = 130, .nocache_weights = true, .wired_margin_bytes = 2_000_000_000 });
    try testing.expectEqual(@as(?u64, 9_200_000_000), c.memory_baseline_bytes);
    try testing.expectEqual(@as(?u32, null), c.expert_rows);
    try testing.expectEqual(@as(?u32, 130), c.expert_prefill_rows);
    try testing.expectEqual(@as(?bool, true), c.nocache_weights);
    try testing.expectEqual(@as(?NumericTier, .stock), c.numeric_tier);
}
