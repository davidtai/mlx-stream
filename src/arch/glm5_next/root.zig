//! GLM-5.3-Flash (`glm5_next`) behind the `arch` kind (sdk.Arch.of): the pack mlx-stream serves (config.json with the
//! text model's dims in `text_config`, the BF16 residents, the NVFP4 expert bank `experts.bin` beside
//! `expert-manifest.json`, declared by the config's `expert_streaming` block). Every entry wraps the module's own; the
//! host reaches the module once per prompt, serial step or request end, never per layer. On a host without the DSA
//! kernels (`mlx_host.dsa_nax`) the parse and the construction refuse by name and no module exists.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../../sdk_ext.zig");
const host_caps = @import("../../host_caps.zig");
const glm = @import("config.zig");
const settings = @import("settings.zig");
const bank = @import("bank.zig");
const bills = @import("bill.zig");
const module = @import("module.zig");
const mtp = @import("mtp.zig");

pub const name = "glm5_next";
pub const family = "mlx-stream";
pub const manifest_file = "expert-manifest.json";

const host_surface_missing = "glm5_next: this host exports no DSA kernels (mlx_host.dsa_nax): GLM-5.3-Flash needs a host built with them (Glm5NextHostSurfaceMissing)";

pub const caps: sdk.Caps = .{
    .owns_decode_state = true,
    .prefill_whole_prompt = true,
    .prefill_yields_last_logits = true,
    .residents_past_page_cache = true,
};

pub const Config = settings.Config;
pub const Module = if (host_caps.dsa_nax) module.Module else struct {};

/// G6: the process's one expert reader, claimed at the load claim, released when the loaded model goes.
pub const claimProcess = sdk_ext.expert.takeReader;
pub const releaseProcess = sdk_ext.expert.giveReader;

/// The pack's `expert_streaming` block: mlx-stream's NVFP4 bank and its manifest.
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
    if (!std.mem.eql(u8, t, "glm5_next")) return null;
    return if (declared(p.root.get("expert_streaming") orelse .null)) .native else null;
}

/// The arch's own parse of config.json (refusals by name), the pack's directory, and the pack's construction settings
/// (`expert_streaming.settings`: an object of `settings.setting_keys` only).
pub fn parse(a: std.mem.Allocator, p: *const sdk.ConfigPeek, diag: *sdk.Diag) !*Config {
    if (comptime !host_caps.dsa_nax) {
        diag.set("{s}", .{host_surface_missing});
        return error.Glm5NextHostSurfaceMissing;
    }
    const block = p.root.get("expert_streaming") orelse .null;
    if (!declared(block)) {
        diag.set("glm5_next: the pack declares no mlx-stream NVFP4 bank (expert_streaming {{family, format, manifest}})", .{});
        return error.Glm5NextPackRequired;
    }
    var md: glm.Diag = .{};
    const model = glm.Config.parse(a, p.text, &md) catch |e| {
        diag.set("{s}", .{md.message()});
        return e;
    };
    const c = try a.create(Config);
    errdefer a.destroy(c);
    c.* = .{ .model = model, .model_dir = if (p.model_dir.len > 0) try a.dupe(u8, p.model_dir) else null };
    errdefer if (c.model_dir) |d| a.free(d);
    if (block.object.get("settings")) |raw| {
        if (raw != .object) {
            diag.set("glm5_next: expert_streaming.settings is not a JSON object", .{});
            return error.Glm5NextSettingsBlock;
        }
        if (settings.unknownSetting(raw.object)) |key| {
            diag.set("glm5_next: expert_streaming.settings: \"{s}\" is not a glm5_next setting", .{key});
            return error.Glm5NextSettingUnknown;
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
    if (comptime !host_caps.dsa_nax) {
        sdk.log.err("{s}\n", .{host_surface_missing});
        return error.Glm5NextHostSurfaceMissing;
    }
    var built = try module.admitted(load.io, c, &load.facts);
    try built.validate();
    return Module.init(load.gpa, load.io, &built, load.weights, load.stream, load.ceiling, load.facts.wired_margin_bytes);
}

pub fn deinit(m: *Module) void {
    if (comptime !host_caps.dsa_nax) unreachable;
    m.deinit();
}

pub fn prefill(m: *Module, ids: []const u32, req: sdk.RequestShape) !sdk.mlx.mlx_array {
    if (comptime !host_caps.dsa_nax) unreachable;
    return m.prefill(ids, req);
}

pub fn step(m: *Module, ids: []const u32) !sdk.mlx.mlx_array {
    if (comptime !host_caps.dsa_nax) unreachable;
    return m.step(ids);
}

pub fn position(m: *const Module) u64 {
    if (comptime !host_caps.dsa_nax) unreachable;
    return m.position();
}

pub fn requestEnd(m: *Module) void {
    if (comptime !host_caps.dsa_nax) unreachable;
    m.requestEnd();
}

/// The draft lane: the MTP lane (`mtp_lane`, mtp.zig, MiMo's draft-lane seam) when the construction installed it,
/// else serial (blockSize 0: the host arms nothing and every request stays serial).
pub const draft_lane = struct {
    pub fn blockSize(m: *const Module) u32 {
        if (comptime !host_caps.dsa_nax) unreachable;
        return if (m.mtp != null) mtp.verify_rows else 0;
    }

    pub fn laneName(m: *const Module) []const u8 {
        if (comptime !host_caps.dsa_nax) unreachable;
        return if (m.mtp != null) mtp.lane_name else "serial";
    }

    /// The exact rule's arms (MiMo's): a clean greedy request `.greedy`, a clean sampled one `.stochastic`; logprobs,
    /// grammar and penalties stay serial.
    pub fn arm(m: *const Module, req: sdk.ArmRequest) sdk.DraftArm {
        if (comptime !host_caps.dsa_nax) unreachable;
        if (m.mtp == null) return .off;
        return mtp.armOf(req);
    }

    /// The host runs a round only on an armed lane (`blockSize` > 0: the construction installed it) after a prompt pass:
    /// the host contract's invariant (MiMo's), not re-checked per round.
    pub fn round(m: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
        if (comptime !host_caps.dsa_nax) unreachable;
        std.debug.assert(m.mtp != null);
        return m.mtpRound(a, t1, accepted_cap, sampling);
    }

    pub fn stats(m: *const Module) sdk.DraftStats {
        if (comptime !host_caps.dsa_nax) unreachable;
        const lane = m.mtp orelse return .{};
        return .{ .rounds = lane.stats.rounds, .drafted = lane.stats.drafted, .accepted = lane.stats.accepted, .generated = lane.stats.generated };
    }

    /// The request's receipt line (MiMo's: rounds, accepted per round, misses and bytes per round); the host logs it
    /// with `[spec-stats]`.
    pub fn receipt(m: *Module, buf: []u8) []const u8 {
        if (comptime !host_caps.dsa_nax) unreachable;
        return m.laneReceipt(buf);
    }
};

const testing = std.testing;

test "glm5_next claims: the pack's NVFP4 declaration claims it; other packs and models decline" {
    try sdk.testing.expectClaims(claims, &.{
        .{ .config =
        \\{"model_type":"glm5_next","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-nvfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
        , .want = .native },
        .{ .config = "{\"model_type\":\"glm5_next\"}", .want = null },
        .{ .config =
        \\{"model_type":"glm5_next","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
        , .want = null },
        .{ .config =
        \\{"model_type":"glm_moe_dsa","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-nvfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
        , .want = null },
        .{ .config =
        \\{"model_type":"glm5_next","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-nvfp4-expert-bank-v1","manifest":"other.json"}}
        , .want = null },
    });
}

test "glm5_next parse: the settings block is applied at parse; a non-object block or an unknown key refuses by name" {
    if (comptime !host_caps.dsa_nax) return error.SkipZigTest;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const base = try glm.flashConfigJson(arena.allocator());
    const marker = "\"model_type\":\"glm5_next\",";
    const Case = struct { block: []const u8, err: ?anyerror };
    for ([_]Case{
        .{ .block = "\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\",\"settings\":{\"expert_prefill_rows\":100,\"expert_rows\":120,\"logits_dump_dir\":\"/tmp/kl\"}},", .err = null },
        .{ .block = "\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\",\"settings\":[1]},", .err = error.Glm5NextSettingsBlock },
        .{ .block = "\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\",\"settings\":{\"decode_glue\":\"ops\"}},", .err = error.Glm5NextSettingUnknown },
        .{ .block = "", .err = error.Glm5NextPackRequired },
    }) |case| {
        const text = try std.mem.replaceOwned(u8, arena.allocator(), base, marker, try std.mem.concat(arena.allocator(), u8, &.{ marker, case.block }));
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
        try testing.expectEqualStrings("/tmp/kl", c.dumpDir().?);
        try testing.expectEqualStrings("/fixture", c.model_dir.?);
        try testing.expectEqual(@as(u32, 45), shell(c).num_layers);
    }
}

test "glm5_next arch table: the registry builds it (the draft lane surface, no prefix restore, no handover); no lane, serial" {
    const vt = comptime sdk.Arch.of(@This());
    try testing.expect(vt.request_end != null);
    try testing.expect(vt.restore_prefix == null and vt.handover == null);
    try testing.expect(vt.spec == .draft_lane);
    if (comptime !host_caps.dsa_nax) return error.SkipZigTest;
    var m: Module = undefined;
    m.mtp = null;
    try testing.expectEqual(@as(u32, 0), draft_lane.blockSize(&m));
    try testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expectEqualStrings("serial", draft_lane.laneName(&m));
    try testing.expectEqual(@as(u64, 0), draft_lane.stats(&m).rounds);
    var rbuf: [8]u8 = undefined;
    try testing.expectEqual(@as(usize, 0), draft_lane.receipt(&m, &rbuf).len);
    // With the lane installed the arms are the exact rule's (mtp.armOf): greedy, stochastic; an unclean request serial.
    try testing.expectEqual(sdk.DraftArm.greedy, mtp.armOf(.{ .greedy = true, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.stochastic, mtp.armOf(.{ .greedy = false, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, mtp.armOf(.{ .greedy = true, .clean = false }));
    // The installed lane through the host's table: its verify rows, its name, its arms and its counters.
    var lane: mtp.Lane = undefined;
    lane.stats = .{};
    m.mtp = &lane;
    const tbl = vt.spec.draft_lane;
    try testing.expectEqual(@as(u32, 4), tbl.block_size(&m));
    try testing.expectEqualStrings("glm5_next-mtp exact", tbl.lane_name(&m));
    try testing.expectEqual(sdk.DraftArm.greedy, tbl.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.stochastic, tbl.arm(&m, .{ .greedy = false, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, tbl.arm(&m, .{ .greedy = false, .clean = false }));
    lane.stats.add(3, 2);
    try testing.expectEqual(sdk.DraftStats{ .rounds = 1, .drafted = 3, .accepted = 2, .generated = 3 }, tbl.stats(&m));
}
