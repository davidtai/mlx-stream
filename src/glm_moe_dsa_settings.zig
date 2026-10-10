//! GLM-5.3's config on the host side: the model directory, the parsed model config, the load's facts and the model
//! settings it takes (`model-settings.json`) with the defaults they fall back to.

const std = @import("std");
const sdk = @import("sdk");
const log = @import("sdk").log;
const glm = @import("glm_moe_dsa.zig");

/// The arch's numerics: "stock" (the reference's op chain) is the only tier; any other name is unset.
pub const NumericTier = enum { stock };

pub const Config = struct {
    /// The pack's directory (config.json, the resident shards, the expert bank); owned by the host's config.
    model_dir: ?[]const u8 = null,
    /// The parsed config.json (owned: `deinit`).
    model: ?glm.Config = null,
    /// The load's facts: the memory in use before the load, the decode and prompt slot rows per layer (null = the
    /// fill's), the residents past the page cache (null = on).
    memory_baseline_bytes: ?u64 = null,
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    nocache_weights: ?bool = null,
    /// The longest prompt the module admits (`ctx_size`; null: the standard request's `default_context`). The
    /// construction bills every prompt up to it; a longer one is refused by name before its pass.
    max_context_tokens: ?u32 = null,
    /// A `ctx_size` above the model's limit: refused by name at load (`error.CtxSizeOverModelLimit`).
    ctx_size_over_limit: ?i64 = null,
    numeric_tier: ?NumericTier = null,
    /// The routed waves wait on the reads' events instead of the host (null = on).
    expert_event_gates: ?bool = null,
    /// The prompt pass layer by layer over the whole prompt (null = on); off, chunk by chunk.
    layer_major_prefill: ?bool = null,
    /// Prompt routes live at once in one layer (null = 2).
    expert_wide_depth: ?u8 = null,

    /// The longest `ctx_size` the bill takes (GLM-5.3's max_position_embeddings).
    pub const max_ctx_size: i64 = 1 << 20;
    /// The prompt routes one layer may hold live at once (`sdk_ext.expert.stream.max_wide_depth`).
    pub const max_wide_depth: i64 = 5;

    pub fn deinit(c: *Config, gpa: std.mem.Allocator) void {
        if (c.model) |*m| m.deinit(gpa);
        c.model = null;
    }

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
        inline for (.{ "expert_event_gates", "layer_major_prefill" }) |key| {
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
        if (obj.get("expert_wide_depth")) |v| if (v == .integer and v.integer >= 1 and v.integer <= max_wide_depth) {
            c.expert_wide_depth = @intCast(v.integer);
            any = true;
        };
        if (obj.get("ctx_size")) |v| if (v == .integer and v.integer >= 1) {
            if (v.integer <= max_ctx_size) c.max_context_tokens = @intCast(v.integer) else c.ctx_size_over_limit = v.integer;
            any = true;
        };
        if (any) log.info("[model-settings] glm_moe_dsa: numeric_tier={s} event_gates={s} layer_major_prefill={s} wide_depth={d} billed_context={d}\n", .{
            if (c.numeric_tier) |t| @tagName(t) else "default", onOff(c.expert_event_gates), onOff(c.layer_major_prefill), c.expert_wide_depth orelse 0, c.max_context_tokens orelse 0,
        });
    }

    fn onOff(v: ?bool) []const u8 {
        return if (v) |b| (if (b) "on" else "off") else "default";
    }

    pub fn eventGates(c: *const Config) bool {
        return c.expert_event_gates orelse true;
    }

    pub fn layerMajor(c: *const Config) bool {
        return c.layer_major_prefill orelse true;
    }

    pub fn wideDepth(c: *const Config) u8 {
        return c.expert_wide_depth orelse 2;
    }

    /// A `ctx_size` over the model's limit, refused by name (never a silent fall back to the standard context).
    pub fn checkCtxSize(c: *const Config) error{CtxSizeOverModelLimit}!void {
        const v = c.ctx_size_over_limit orelse return;
        log.warn("glm_moe_dsa: load refused: ctx_size {d} is over the model's limit of {d} tokens (CtxSizeOverModelLimit)\n", .{ v, max_ctx_size });
        return error.CtxSizeOverModelLimit;
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

test "glm settings: numeric_tier names stock only; the gates and the prompt order are bools; anything else is unset" {
    try testing.expectEqual(@as(?NumericTier, .stock), (try settingsOf("{\"numeric_tier\": \"stock\"}")).numeric_tier);
    try testing.expectEqual(@as(?NumericTier, null), (try settingsOf("{\"numeric_tier\": \"served\"}")).numeric_tier);
    try testing.expectEqual(@as(?bool, false), (try settingsOf("{\"expert_event_gates\": false}")).expert_event_gates);
    try testing.expectEqual(@as(?bool, null), (try settingsOf("{\"expert_event_gates\": 0}")).expert_event_gates);
    try testing.expectEqual(@as(?bool, false), (try settingsOf("{\"layer_major_prefill\": false}")).layer_major_prefill);
    const d: Config = .{};
    try testing.expect(d.eventGates() and d.layerMajor() and d.wideDepth() == 2);
    try testing.expectEqual(Config{}, try settingsOf("[1]"));
}

test "glm settings: expert_wide_depth is 1 to 5; ctx_size bills every prompt up to it, over the limit it is kept for the refusal" {
    try testing.expectEqual(@as(?u8, 5), (try settingsOf("{\"expert_wide_depth\": 5}")).expert_wide_depth);
    try testing.expectEqual(@as(?u8, null), (try settingsOf("{\"expert_wide_depth\": 6}")).expert_wide_depth);
    try testing.expectEqual(@as(?u8, null), (try settingsOf("{\"expert_wide_depth\": 0}")).expert_wide_depth);
    try testing.expectEqual(@as(?u32, 131072), (try settingsOf("{\"ctx_size\": 131072}")).max_context_tokens);
    try testing.expectEqual(@as(?u32, null), (try settingsOf("{\"ctx_size\": \"131072\"}")).max_context_tokens);
    const over = try settingsOf("{\"ctx_size\": 2097152}");
    try testing.expectEqual(@as(?u32, null), over.max_context_tokens);
    try testing.expectError(error.CtxSizeOverModelLimit, over.checkCtxSize());
    try (Config{}).checkCtxSize();
}

test "glm settings: the load facts replace the fill's inputs, the settings stay" {
    var s = try settingsOf("{\"expert_wide_depth\": 3}");
    s.expert_rows = 99;
    const c = s.withFacts(&.{ .memory_baseline_bytes = 9_000_000_000, .expert_prefill_rows = 120, .nocache_weights = false, .wired_margin_bytes = 2 << 30 });
    try testing.expectEqual(@as(?u64, 9_000_000_000), c.memory_baseline_bytes);
    try testing.expectEqual(@as(?u32, null), c.expert_rows);
    try testing.expectEqual(@as(?u32, 120), c.expert_prefill_rows);
    try testing.expectEqual(@as(?u8, 3), c.expert_wide_depth);
}
