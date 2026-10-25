//! MiniMax M3.1 Flash text-only SDK architecture and bounded NVFP4 demand reader.
const std = @import("std");
const sdk = @import("sdk");
const model_config = @import("config.zig");
const bank = @import("bank.zig");
const prepared = @import("../mimo_v2/prepared.zig");
const usage = @import("../../sdk_ext.zig").expert.usage;
const resident_layout = @import("resident_layout.zig");
const cache_layout = @import("cache.zig");
pub const name = "minimax_m3";
pub const default_context: u32 = 4096;
pub const generation_headroom: u32 = 0;
pub const Module = @import("module.zig").Module;
pub const caps: sdk.Caps = .{ .owns_decode_state = true, .prefill_whole_prompt = true, .prefill_yields_last_logits = true, .residents_past_page_cache = true };
pub const claimProcess = @import("../../sdk_ext.zig").expert.takeReader;
pub const releaseProcess = @import("../../sdk_ext.zig").expert.giveReader;
pub const Config = struct {
    model: model_config.Config = .{},
    model_dir: ?[]const u8 = null,
    context: u32 = default_context,
    kv_quant_bits: u8 = 0,
    expert_rows: u32 = 4,
    expert_rows_forced: bool = false,
    settings_error: bool = false,
    residents_format: resident_layout.Format = .bf16,
    expert_format: bank.Kind = .nvfp4,
    residency: usage.Settings = .{},
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        if (raw == .null) return;
        if (raw != .object) {
            c.settings_error = true;
            return;
        }
        _ = c.residency.apply(raw.object);
        // Speculation and multimodal execution are deliberately refused.
        for (raw.object.keys(), raw.object.values()) |key, val| {
            if (std.mem.eql(u8, key, "expert_rows")) {
                if (val == .integer and val.integer >= 4 and val.integer <= 128) {
                    c.expert_rows = @intCast(val.integer);
                    c.expert_rows_forced = true;
                } else c.settings_error = true;
            } else if (std.mem.eql(u8, key, "mtp_lane") or std.mem.eql(u8, key, "mtp") or std.mem.eql(u8, key, "vision")) {
                if (val != .bool or val.bool) c.settings_error = true;
            }
        }
    }
};
pub fn declared(v: std.json.Value) bool {
    if (v != .object) return false;
    inline for (.{ .{ "family", "mlx-stream" }, .{ "manifest", "expert-manifest.json" } }) |pair| {
        const value = v.object.get(pair[0]) orelse return false;
        if (value != .string or !std.mem.eql(u8, value.string, pair[1])) return false;
    }
    const format = v.object.get("format") orelse return false;
    if (format != .string) return false;
    _ = bank.parseFormat(format.string) catch return false;
    return true;
}
pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    const mt = p.modelType() orelse return null;
    if (!std.mem.eql(u8, mt, "minimax_m3_vl") and !std.mem.eql(u8, mt, "minimax_m3") and !std.mem.eql(u8, mt, "minimax_m3_text")) return null;
    return if (declared(p.root.get("expert_streaming") orelse .null)) .native else null;
}
pub fn parse(a: std.mem.Allocator, p: *const sdk.ConfigPeek, diag: *sdk.Diag) !*Config {
    if (!declared(p.root.get("expert_streaming") orelse .null)) {
        diag.set("minimax_m3 requires a prepared expert bank (expert_streaming)", .{});
        return error.MiniMaxM3NeedsExpertBank;
    }
    var md: model_config.Diag = .{};
    const model = model_config.Config.parse(a, p.text, &md) catch |e| {
        diag.set("{s}", .{md.message()});
        return e;
    };
    const c = try a.create(Config);
    errdefer a.destroy(c);
    const residents_format = resident_layout.Format.parse(p.root.get("residents_format") orelse .{ .string = "bf16" }) catch |e| {
        diag.set("minimax_m3: residents_format must be bf16, mxfp8 or mixed", .{});
        return e;
    };
    c.* = .{ .model = model, .residents_format = residents_format, .expert_format = try bank.parseFormat(p.root.get("expert_streaming").?.object.get("format").?.string), .model_dir = if (p.model_dir.len > 0) try a.dupe(u8, p.model_dir) else null };
    errdefer if (c.model_dir) |dir| a.free(dir);
    if (p.root.get("expert_streaming").?.object.get("settings")) |raw| c.applySettings(raw);
    try c.residency.check(name);
    if (c.settings_error) {
        diag.set("minimax_m3: unsupported settings; text-only serial execution requires expert_rows=4..128, mtp_lane=false, vision=false", .{});
        return error.MiniMaxUnsupportedSettings;
    }
    return c;
}
pub fn freeConfig(a: std.mem.Allocator, c: *Config) void {
    if (c.model_dir) |d| a.free(d);
    a.destroy(c);
}
pub fn shell(c: *const Config) sdk.Shell {
    return .{ .num_experts = c.model.n_routed_experts, .num_layers = c.model.n_layers, .kv_quant_bits = c.kv_quant_bits };
}
pub fn applySettings(c: *Config, raw: std.json.Value) void {
    c.applySettings(raw);
}
pub fn applyLoadFacts(c: *Config, facts: *const sdk.LoadFacts) !void {
    if (c.settings_error) return error.MiniMaxUnsupportedSettings;
    try c.residency.check(name);
    if (facts.kv_quant_bits) |bits| {
        if (bits != 0 and bits != 8) return error.MiniMaxKvPrecisionUnsupported;
        c.kv_quant_bits = @intCast(bits);
    }
    if (facts.effective_context) |ctx| {
        if (ctx == 0 or ctx > c.model.max_position_embeddings) return error.MiniMaxContextUnsupported;
        c.context = @intCast(ctx);
    }
    if (facts.expert_rows) |rows| {
        c.expert_rows = rows;
        c.expert_rows_forced = true;
    }
    if (facts.expert_prefill_rows) |rows| if (rows != c.expert_rows) return error.MiniMaxPhaseRowsUnsupported;
    if (c.expert_rows < 4 or c.expert_rows > 128) return error.MiniMaxExpertRows;
}
fn memoryBill(a: std.mem.Allocator, io: std.Io, c: *const Config) !sdk.MemoryBill {
    const dir = c.model_dir orelse return error.MiniMaxModelDirectoryRequired;
    const geometry = try bank.inspect(a, io, dir);
    if (geometry.kind != c.expert_format) return error.MiniMaxExpertFormatMismatch;
    try bank.requireFlash(&geometry);
    const residents = try prepared.residentBytesBound(a, io, dir, 64 << 30);
    const slots = geometry.slotRowBytes();
    // Capacity is rounded, and a growing layer can briefly retain its old
    // buffers beside the replacement. Other layers retain one capacity each.
    const kv = if (c.kv_quant_bits == 8)
        try @import("q8_reference.zig").cacheBytes(c.model.n_layers, c.model.n_kv_heads, c.model.head_dim, c.model.index_head_dim, cache_layout.paddedCapacity(c.context))
    else
        c.model.kvBytes(cache_layout.paddedCapacity(c.context));
    const kv_copy = kv / c.model.n_layers;
    const reader = readerStagingBytes(&geometry);
    const scratch = 32 * slots + 256 * @as(u64, c.model.hidden_size) + 8 * @as(u64, c.model.n_heads) * c.context;
    const terms = try a.dupe(sdk.MemoryBill.Term, &.{
        .{ .name = "residents", .bytes = .{ residents, residents }, .at_construction = true },
        .{ .name = "kv_and_index", .bytes = .{ kv, kv }, .at_construction = false },
        .{ .name = "kv_append", .bytes = .{ kv_copy, kv_copy }, .at_construction = false },
        .{ .name = "reader_staging", .bytes = .{ reader, reader }, .at_construction = true },
        .{ .name = "serial_scratch", .bytes = .{ scratch, scratch }, .at_construction = false },
        .{ .name = "mlx_runtime_allowance", .bytes = .{ 1 << 30, 1 << 30 }, .at_construction = false },
    });
    return .{ .terms = terms, .per_row = slots * geometry.num_moe_layers };
}

fn readerStagingBytes(geometry: anytype) u64 {
    const page: u64 = std.heap.pageSize();
    const widest = @max(geometry.gate_up_bytes, geometry.logical_bytes - geometry.gate_up_bytes);
    return 2 * (std.mem.alignForward(u64, widest, page) + page);
}
pub fn bill(a: std.mem.Allocator, io: std.Io, req: *const sdk.BillRequest) !sdk.MemoryBill {
    const c: *const Config = @ptrCast(@alignCast(req.cfg));
    return memoryBill(a, io, c);
}
pub fn loadBytes(a: std.mem.Allocator, io: std.Io, c: *const Config, facts: *const sdk.LoadFacts, ceiling: u64) !u64 {
    _ = ceiling;
    var built = c.*;
    try applyLoadFacts(&built, facts);
    const b = try memoryBill(a, io, &built);
    defer b.free(a);
    return b.processBound(.{ .prompt = built.expert_rows, .decode = built.expert_rows });
}
pub fn promptBytes(_: *const Config, _: u64, _: u32) u64 {
    return 0;
}
pub fn init(load: *const sdk.LoadCtx, c: *const Config) !*Module {
    var built = c.*;
    try applyLoadFacts(&built, &load.facts);
    const b = try memoryBill(load.gpa, load.io, &built);
    defer b.free(load.gpa);
    const baseline = load.facts.memory_baseline_bytes orelse 0;
    const target = load.ceiling -| load.facts.wired_margin_bytes;
    if (!built.expert_rows_forced) {
        const rows = try sdk.fill(b, baseline, target, built.model.n_routed_experts, 4);
        built.expert_rows = rows.decode;
    }
    try sdk.admit(b, baseline, .{ .prompt = built.expert_rows, .decode = built.expert_rows }, target);
    sdk.log.info("minimax_m3 expert rows: {d} per routed layer ({s})\n", .{ built.expert_rows, if (built.expert_rows_forced) "setting" else "memory fill" });
    return Module.init(load, &built);
}
pub fn deinit(m: *Module) void {
    m.deinit();
}
pub fn prefill(m: *Module, ids: []const u32, req: sdk.RequestShape) !sdk.mlx.mlx_array {
    return m.prefill(ids, req);
}
pub fn step(m: *Module, ids: []const u32) !sdk.mlx.mlx_array {
    return m.step(ids);
}
pub fn position(m: *const Module) u64 {
    return m.position();
}
pub fn requestEnd(m: *Module) void {
    m.requestEnd();
}
pub const draft_lane = struct {
    pub fn blockSize(_: *const Module) u32 {
        return 0;
    }
    pub fn laneName(_: *const Module) []const u8 {
        return "serial";
    }
    pub fn arm(_: *Module, _: sdk.ArmRequest) sdk.DraftArm {
        return .off;
    }
    pub fn round(_: *Module, _: std.mem.Allocator, _: u32, _: u32, _: sdk.SamplingParams) !sdk.DraftRound {
        return error.MiniMaxSpeculationUnsupported;
    }
    pub fn stats(_: *const Module) sdk.DraftStats {
        return .{};
    }
    pub fn receipt(_: *Module, _: []u8) []const u8 {
        return "";
    }
};
test "MiniMax claims only declared NVFP4 packs and exposes text serial architecture" {
    try sdk.testing.expectClaims(claims, &.{
        .{ .config = "{\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"}}", .want = .native },
        .{ .config = "{\"model_type\":\"minimax_m3_vl\"}", .want = null },
        .{ .config = "{\"model_type\":\"glm5_next\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"}}", .want = null },
    });
    const vt = comptime sdk.Arch.of(@This());
    try std.testing.expect(vt.caps.owns_decode_state);
    try std.testing.expect(vt.request_end != null);
}

test "MiniMax load facts refuse unsupported KV, context and phase rows" {
    var c: Config = .{};
    try applyLoadFacts(&c, &.{ .effective_context = 8192, .expert_rows = 8, .wired_margin_bytes = 0 });
    try std.testing.expectEqual(@as(u32, 8192), c.context);
    try std.testing.expectEqual(@as(u32, 8), c.expert_rows);
    try applyLoadFacts(&c, &.{ .kv_quant_bits = 8, .wired_margin_bytes = 0 });
    try std.testing.expectEqual(@as(u8, 8), c.kv_quant_bits);
    try std.testing.expectEqual(@as(u8, 8), shell(&c).kv_quant_bits);
    try std.testing.expectError(error.MiniMaxKvPrecisionUnsupported, applyLoadFacts(&c, &.{ .kv_quant_bits = 4, .wired_margin_bytes = 0 }));
    try std.testing.expectError(error.MiniMaxContextUnsupported, applyLoadFacts(&c, &.{ .effective_context = 1048577, .wired_margin_bytes = 0 }));
    try std.testing.expectError(error.MiniMaxPhaseRowsUnsupported, applyLoadFacts(&c, &.{ .expert_prefill_rows = 4, .wired_margin_bytes = 0 }));
}

test "MiniMax reader bill matches shared stream staging at the actual host page size" {
    const geometry = try bank.Geometry.init(6144, 3072, 128, 3, 57);
    var layers = [_]bank.Layer{.{ .logical_bytes = geometry.logical_bytes, .base_offset = 0, .segments = geometry.segments }};
    const view: bank.SDKBank = .{ .allocator = std.testing.allocator, .layers = &layers, .n_experts = 128, .first_moe_layer = 3, .record_bytes = geometry.record_bytes, .sidecar = .{ .fd = -1, .size = 0 } };
    const shared = @import("experts.zig").S.stagingBytes(&view);
    try std.testing.expectEqual(2 * shared, readerStagingBytes(&geometry));
    const model: model_config.Config = .{};
    try std.testing.expectEqual(@as(u64, 138240 * 768), model.kvBytes(cache_layout.paddedCapacity(513)));
}
test "MiniMax parse binds source model config and refuses speculative settings" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const source = @embedFile("config_fixture.json");
    const text = try std.mem.replaceOwned(u8, arena.allocator(), source, "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},");
    const p = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", text);
    var diag: sdk.Diag = .{};
    const c = try parse(a, &p, &diag);
    defer freeConfig(a, c);
    try std.testing.expectEqual(@as(u32, 60), shell(c).num_layers);
    const raw = try std.json.parseFromSlice(std.json.Value, a, "{\"mtp_lane\":true}", .{});
    defer raw.deinit();
    applySettings(c, raw.value);
    try std.testing.expectError(error.MiniMaxUnsupportedSettings, applyLoadFacts(c, &.{ .wired_margin_bytes = 0 }));
}

test "MiniMax parse admits MXFP8 residents and rejects unrecognized resident formats" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(resident_layout.Format.bf16, (Config{}).residents_format);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const base = try std.mem.replaceOwned(u8, arena.allocator(), @embedFile("config_fixture.json"), "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},\"residents_format\":\"mxfp8\",");
    const p = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", base);
    var diag: sdk.Diag = .{};
    const c = try parse(a, &p, &diag);
    defer freeConfig(a, c);
    try std.testing.expectEqual(resident_layout.Format.mxfp8, c.residents_format);
    const invalid = try std.mem.replaceOwned(u8, arena.allocator(), base, "\"residents_format\":\"mxfp8\"", "\"residents_format\":\"fp8\"");
    const bad = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", invalid);
    try std.testing.expectError(error.MiniMaxResidentFormatUnsupported, parse(a, &bad, &diag));
}

test "MiniMax forwards shared usage settings and rejects malformed usage labels" {
    const a = std.testing.allocator;
    const raw = try std.json.parseFromSlice(std.json.Value, a, "{\"resident_seed\":\"none\",\"expert_usage_record\":\"/tmp/minimax-usage-fixture.json\",\"expert_usage_label\":\"fixture\"}", .{});
    defer raw.deinit();
    var c: Config = .{};
    applySettings(&c, raw.value);
    try applyLoadFacts(&c, &.{ .wired_margin_bytes = 0 });
    try std.testing.expectEqual(usage.Seed.none, c.residency.seed.?);
    try std.testing.expectEqualStrings("fixture", c.residency.label());
    try std.testing.expectEqualStrings("/tmp/minimax-usage-fixture.json", c.residency.recordPath().?);
    const bad = try std.json.parseFromSlice(std.json.Value, a, "{\"expert_usage_label\":\"fixture\"}", .{});
    defer bad.deinit();
    var b: Config = .{};
    applySettings(&b, bad.value);
    try std.testing.expectError(error.ExpertUsageLabelWithoutRecord, applyLoadFacts(&b, &.{ .wired_margin_bytes = 0 }));
}
