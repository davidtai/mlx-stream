//! Reflection Beam 1.0 (`beam`) behind the `arch` kind (sdk.Arch.of): the pack mlx-stream serves (config.json, the BF16
//! residents, the NVFP4 expert bank `experts.bin` beside `expert-manifest.json`, declared by the config's
//! `expert_streaming` block; scripts/prepare_nvfp4_pack.py --family beam writes it). Every entry wraps the module's own;
//! the host reaches the module once per prompt, serial step or request end, never per layer. No draft lane: every
//! request decodes in serial steps, the plugin's sampler drawing each token.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../../sdk_ext.zig");
const beam = @import("config.zig");
const settings = @import("settings.zig");
const bank = @import("../../expert/nvfp4_bank.zig");
const bills = @import("bill.zig");
const module = @import("module.zig");

pub const name = "beam";
pub const family = "mlx-stream";
pub const manifest_file = "expert-manifest.json";

pub const caps: sdk.Caps = .{
    .owns_decode_state = true,
    .prefill_whole_prompt = true,
    .prefill_yields_last_logits = true,
    .residents_past_page_cache = true,
};

pub const Config = settings.Config;
pub const Module = module.Module;

/// G6: the process's one expert reader, claimed at the load claim, released when the loaded model goes.
pub const claimProcess = sdk_ext.expert.takeReader;
pub const releaseProcess = sdk_ext.expert.giveReader;

/// The pack's `expert_streaming` block names mlx-stream's NVFP4 bank and its manifest.
pub fn declared(v: std.json.Value) bool {
    if (v != .object) return false;
    inline for (.{ .{ "family", family }, .{ "format", bank.format }, .{ "manifest", manifest_file } }) |pair| {
        const field = v.object.get(pair[0]) orelse return false;
        if (field != .string or !std.mem.eql(u8, field.string, pair[1])) return false;
    }
    return true;
}

pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    const t = p.modelType() orelse return null;
    if (!std.mem.eql(u8, t, "beam")) return null;
    return if (declared(p.root.get("expert_streaming") orelse .null)) .native else null;
}

/// The arch's own parse of config.json (refusals by name), the pack's directory, and the pack's construction settings
/// (`expert_streaming.settings`: an object of `settings.setting_keys` only).
pub fn parse(a: std.mem.Allocator, p: *const sdk.ConfigPeek, diag: *sdk.Diag) !*Config {
    const block = p.root.get("expert_streaming") orelse .null;
    if (!declared(block)) {
        diag.set("beam: the pack declares no mlx-stream NVFP4 bank (expert_streaming {{family, format, manifest}})", .{});
        return error.BeamPackRequired;
    }
    var md: beam.Diag = .{};
    const model = beam.Config.parse(a, p.text, &md) catch |e| {
        diag.set("{s}", .{md.message()});
        return e;
    };
    const c = try a.create(Config);
    errdefer a.destroy(c);
    c.* = .{ .model = model, .model_dir = if (p.model_dir.len > 0) try a.dupe(u8, p.model_dir) else null };
    errdefer if (c.model_dir) |d| a.free(d);
    if (block.object.get("settings")) |raw| {
        if (raw != .object) {
            diag.set("beam: expert_streaming.settings is not a JSON object", .{});
            return error.BeamSettingsBlock;
        }
        if (settings.unknownSetting(raw.object)) |key| {
            diag.set("beam: expert_streaming.settings: \"{s}\" is not a beam setting", .{key});
            return error.BeamSettingUnknown;
        }
        c.applySettings(raw);
    }
    return c;
}

pub fn freeConfig(a: std.mem.Allocator, c: *Config) void {
    if (c.model_dir) |d| a.free(d);
    a.destroy(c);
}

pub fn shell(c: *const Config) sdk.Shell {
    return .{ .num_experts = c.model.n_routed_experts, .num_layers = c.model.n_layers, .kv_quant_bits = 0 };
}

/// The host's per-model settings object (the same keys as the pack's block).
pub fn applySettings(c: *Config, raw: std.json.Value) void {
    c.applySettings(raw);
}

/// The load's facts, validated once (every refusal by name before any bill or array).
pub fn applyLoadFacts(c: *Config, facts: *const sdk.LoadFacts) !void {
    const built = c.withFacts(facts);
    try built.validate();
    c.* = built;
}

/// The load preflight's requirement: the process bound at the fill's floor rows (forced rows load as named: an
/// operator pin, never a preflight refusal).
pub fn loadBytes(a: std.mem.Allocator, io: std.Io, c: *const Config, facts: *const sdk.LoadFacts, ceiling: u64) !u64 {
    _ = ceiling;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const built = c.withFacts(facts);
    try built.validate();
    const b = try bills.completeBill(arena.allocator(), io, &built, built.reserveTokens());
    return b.processBound(.{ .prompt = bills.min_rows, .decode = bills.min_rows });
}

/// The construction billed every prompt up to the load's context (the lanes are allocated at it); a longer prompt is
/// the module's refusal by name (`ContextOverBill`).
pub fn promptBytes(c: *const Config, seq: u64, max_tokens: u32) u64 {
    _ = .{ c, seq, max_tokens };
    return 0;
}

/// G4: the bill's terms at the load's context (pure host: the shards' headers and the bank's manifest).
pub fn bill(a: std.mem.Allocator, io: std.Io, req: *const sdk.BillRequest) !sdk.MemoryBill {
    const c: *const Config = @ptrCast(@alignCast(req.cfg));
    return bills.completeBill(a, io, c, c.reserveTokens());
}

pub fn init(load: *const sdk.LoadCtx, c: *const Config) !*Module {
    var built = try module.admitted(load.io, c, &load.facts);
    try built.validate();
    return Module.init(load.gpa, load.io, &built, load.weights, load.stream, load.ceiling, load.facts.wired_margin_bytes);
}

pub fn deinit(m: *Module) void {
    m.deinit();
}

/// The prompt pass: the request's sampler bound (`draft_lane.arm`), its last row the plugin's first token when bound.
pub fn prefill(m: *Module, ids: []const u32, req: sdk.RequestShape) !sdk.mlx.mlx_array {
    m.sampler.begin(draft_lane.laneName(m));
    return m.sampler.chosen(try m.prefill(ids, req), m.position());
}

/// A serial step: its row the plugin's token at the next position when the request is bound.
pub fn step(m: *Module, ids: []const u32) !sdk.mlx.mlx_array {
    return m.sampler.chosen(try m.step(ids), m.position());
}

pub fn position(m: *const Module) u64 {
    return m.position();
}

pub fn requestEnd(m: *Module) void {
    m.requestEnd();
}

/// Serial: no lane. The host arms every clean request (blockSize 1: the sampler's serial steps), so the plugin binds
/// its sampler and draws every token (`sdk_ext.sampler`); `arm` answers `.off` and no round runs.
pub const draft_lane = struct {
    pub fn blockSize(_: *const Module) u32 {
        return 1;
    }

    pub fn laneName(_: *const Module) []const u8 {
        return "serial";
    }

    pub fn arm(m: *Module, req: sdk.ArmRequest) sdk.DraftArm {
        m.sampler.arm(req);
        return .off;
    }

    /// Never called: `arm` answers `.off` for every request (the host runs a round only on an armed lane).
    pub fn round(_: *Module, _: std.mem.Allocator, _: u32, _: u32, _: sdk.SamplingParams) !sdk.DraftRound {
        return error.BeamNoDraftLane;
    }

    pub fn stats(_: *const Module) sdk.DraftStats {
        return .{};
    }

    pub fn receipt(_: *Module, buf: []u8) []const u8 {
        return buf[0..0];
    }
};

const testing = std.testing;

const nvfp4_block = "\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"";

test "beam claims: the pack's NVFP4 declaration claims it; other packs and models decline" {
    try sdk.testing.expectClaims(claims, &.{
        .{ .config = "{\"model_type\":\"beam\"," ++ nvfp4_block ++ "}}", .want = .native },
        .{ .config = "{\"model_type\":\"beam\"}", .want = null },
        .{ .config =
        \\{"model_type":"beam","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
        , .want = null },
        .{ .config = "{\"model_type\":\"glm5_next\"," ++ nvfp4_block ++ "}}", .want = null },
        .{ .config =
        \\{"model_type":"beam","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-nvfp4-expert-bank-v1","manifest":"other.json"}}
        , .want = null },
    });
}

test "beam parse: the settings block is applied at parse; a non-object block or an unknown key refuses by name" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const Case = struct { block: []const u8, err: ?anyerror };
    for ([_]Case{
        .{ .block = nvfp4_block ++ ",\"settings\":{\"expert_prefill_rows\":100,\"expert_rows\":120}},", .err = null },
        .{ .block = nvfp4_block ++ ",\"settings\":[1]},", .err = error.BeamSettingsBlock },
        .{ .block = nvfp4_block ++ ",\"settings\":{\"mtp_lane\":true}},", .err = error.BeamSettingUnknown },
        .{ .block = "", .err = error.BeamPackRequired },
    }) |case| {
        const text = try beam.beamConfigJson(arena.allocator(), case.block);
        const peek = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", text);
        var diag: sdk.Diag = .{};
        if (case.err) |want| {
            try testing.expectError(want, parse(a, &peek, &diag));
            try testing.expect(diag.message().len > 0);
            continue;
        }
        const c = try parse(a, &peek, &diag);
        defer freeConfig(a, c);
        try testing.expectEqual(@as(?u32, 120), c.expert_rows);
        try testing.expectEqual(@as(?u32, 100), c.expert_prefill_rows);
        try testing.expectEqualStrings("/fixture", c.model_dir.?);
        try testing.expectEqual(@as(u32, 52), shell(c).num_layers);
        try testing.expectEqual(@as(u32, 256), shell(c).num_experts);
    }
}

test "beam arch table: the registry builds it (the draft lane surface, serial; no prefix restore, no handover)" {
    const vt = comptime sdk.Arch.of(@This());
    try testing.expect(vt.request_end != null);
    try testing.expect(vt.restore_prefix == null and vt.handover == null);
    try testing.expect(vt.spec == .draft_lane);
    var m: Module = undefined;
    m.sampler.pending = null;
    m.sampler.active = null;
    try testing.expectEqual(@as(u32, 1), draft_lane.blockSize(&m));
    try testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(&m, .{ .greedy = false, .clean = true, .sampling = .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 } }));
    try testing.expectEqual(@as(u64, 1), m.sampler.pending.?.seed);
    try testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expect(m.sampler.pending == null);
    try testing.expectEqualStrings("serial", draft_lane.laneName(&m));
    try testing.expectEqual(@as(u64, 0), draft_lane.stats(&m).rounds);
}
