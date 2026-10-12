//! MiMo V2.6 Flash Q4 SDK2 architecture contracts.
const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../../sdk_ext.zig");
const mimo = @import("config.zig");
const settings = @import("settings.zig");
const prepared = @import("prepared.zig");
const modules = @import("module.zig");
const bills = @import("bill.zig");
const draft = @import("draft.zig");
const pagecache = @import("pagecache.zig");

pub const name = "mimo_v2_flash";
pub const Config = settings.Config;
pub const Module = modules.Module;
pub const fixture_json = @embedFile("../../fixtures/mimo_v2_flash_q4_config.json");
pub const caps: sdk.Caps = .{
    .owns_decode_state = true,
    .prefill_whole_prompt = true,
    .prefill_yields_last_logits = true,
    .residents_past_page_cache = true,
};
pub const claimProcess = sdk_ext.expert.takeReader;
pub const releaseProcess = sdk_ext.expert.giveReader;

pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    if (!std.mem.eql(u8, p.modelType() orelse return null, name)) return null;
    return if (prepared.declared(p.root.get("expert_streaming") orelse .null)) .native else null;
}

pub fn parse(a: std.mem.Allocator, p: *const sdk.ConfigPeek, diag: *sdk.Diag) !*Config {
    if (!prepared.declared(p.root.get("expert_streaming") orelse .null)) {
        diag.set("MiMo Q4 streaming requires the prepared MXFP4 expert bank and expert-manifest.json", .{});
        return error.MimoRepackRequired;
    }
    var md: mimo.Diag = .{};
    const parsed = mimo.Config.parse(a, p.text, &md) catch |err| {
        diag.set("{s}", .{md.message()});
        return err;
    };
    const c = try a.create(Config);
    errdefer a.destroy(c);
    c.* = .{ .mimo = parsed, .expert_bank_dir = if (p.model_dir.len > 0) try a.dupe(u8, p.model_dir) else null };
    errdefer if (c.expert_bank_dir) |dir| a.free(dir);
    // The pack's construction settings (`expert_streaming.settings`), read once here and validated with the load's
    // facts (`applyLoadFacts`): a block that is not an object or a key that is not a setting refuses the load by name.
    if (p.root.get("expert_streaming").?.object.get("settings")) |raw| {
        if (raw != .object) {
            diag.set("expert_streaming.settings is not a JSON object", .{});
            return error.MimoSettingsBlock;
        }
        if (settings.unknownSetting(raw.object)) |key| {
            diag.set("expert_streaming.settings: \"{s}\" is not a MiMo setting", .{key});
            return error.MimoSettingUnknown;
        }
        c.applySettings(raw);
    }
    return c;
}

pub fn freeConfig(a: std.mem.Allocator, c: *Config) void {
    if (c.expert_bank_dir) |dir| a.free(dir);
    a.destroy(c);
}

pub fn shell(c: *const Config) sdk.Shell {
    return .{ .num_experts = c.mimo.n_routed_experts, .num_layers = c.mimo.n_layers, .kv_quant_bits = if (c.kv_mode == .bf16) 0 else 8 };
}

/// The SDK's per-model settings seam: a host that hands this model's settings object applies it after the pack's
/// block (the same keys, `settings.setting_keys`; a host key it does not read is left to the host).
pub fn applySettings(c: *Config, raw: std.json.Value) void {
    c.applySettings(raw);
}

pub fn applyLoadFacts(c: *Config, facts: *const sdk.LoadFacts) !void {
    const built = c.withFacts(facts);
    try built.validate();
    c.* = built;
}

pub fn loadBytes(a: std.mem.Allocator, io: std.Io, c: *const Config, facts: *const sdk.LoadFacts, ceiling: u64) !u64 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var floor = c.withFacts(facts);
    // The load preflight's requirement: the process bound at the fill's floor rows, forced rows or not (DeepSeek's
    // `loadBytes`, deepseek_v41_plugin.zig). Forced rows are an operator pin the construction honours (`bills.choose`
    // reports how far they sit over the fill), never a preflight refusal.
    floor.expert_rows = mimo.default_expert_rows;
    floor.expert_prefill_rows = mimo.default_expert_rows;
    const plan = try bills.plan(arena.allocator(), io, floor, ceiling, facts.wired_margin_bytes);
    return plan.bill.processBound(plan.rows);
}

pub fn promptBytes(c: *const Config, seq: u64, max_tokens: u32) u64 {
    return bills.requestForConfig(c.*, c.mimo.reserveTokens(seq, max_tokens, c.reserveTokens()));
}

pub fn bill(a: std.mem.Allocator, io: std.Io, req: *const sdk.BillRequest) !sdk.MemoryBill {
    const c: *const Config = @ptrCast(@alignCast(req.cfg));
    const reserve = c.mimo.reserveTokens(req.prompt_tokens, req.max_tokens, c.reserveTokens());
    return bills.completeBill(a, io, c.*, reserve);
}

pub fn init(load: *const sdk.LoadCtx, c: *const Config) !*Module {
    const built = try admitted(load.io, c, &load.facts);
    return Module.init(load.gpa, load.io, &built, load.weights, load.stream, load.ceiling, load.facts.wired_margin_bytes);
}

/// The construction config: the host's facts, with the fill's baseline the box as the window guard counts it
/// (`pagecache.admissionBaseline`: physical used less the pack's own evicted page cache, floored at the host's non-file
/// baseline). The load preflight (`loadBytes`) keeps the host's non-file baseline: its rows, and so its bound, are
/// never below construction's.
fn admitted(io: std.Io, c: *const Config, facts: *const sdk.LoadFacts) !Config {
    var built = c.withFacts(facts);
    if (facts.memory_baseline_bytes) |nonfile| if (c.expert_bank_dir) |dir| {
        built.memory_baseline_bytes = try pagecache.admissionBaseline(io, dir, nonfile);
    };
    return built;
}

pub fn deinit(m: *Module) void {
    m.deinit();
}

pub fn prefill(m: *Module, ids: []const u32, req: sdk.RequestShape) !sdk.mlx.mlx_array {
    return m.prefill(ids, m.c.reserveTokens(req.prompt_tokens, req.max_tokens, req.host_context));
}

pub fn step(m: *Module, ids: []const u32) !sdk.mlx.mlx_array {
    return m.extend(ids);
}

pub fn position(m: *const Module) u64 {
    return m.position();
}

/// The MTP draft lane (`draft.zig`), installed at construction when the setting `mtp_lane` sets `mtp_heads` (blockSize 0 otherwise: the host
/// arms nothing and every request stays serial).
pub const draft_lane = struct {
    pub fn blockSize(m: *const Module) u32 {
        return if (m.lane != null) draft.verify_rows else 0;
    }

    pub fn laneName(m: *const Module) []const u8 {
        return if (m.lane) |l| l.name() else "serial";
    }

    /// A clean request drafts under the construction tier, greedy or sampled: `typical` labels every request `typical`;
    /// `exact` is argmax equality on a greedy request (`greedy`) and exact speculative sampling on a sampled one
    /// (`stochastic`). Logprobs, grammar and penalties consume logits the lane never shapes and stay serial.
    pub fn arm(m: *const Module, req: sdk.ArmRequest) sdk.DraftArm {
        const lane = m.lane orelse return .off;
        if (!req.clean) return .off;
        return switch (lane.rule) {
            .exact => if (req.greedy) .greedy else .stochastic,
            .typical => .typical,
        };
    }

    pub fn round(m: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
        return draft.Lane.round(m, a, t1, accepted_cap, sampling);
    }

    pub fn stats(m: *const Module) sdk.DraftStats {
        const l = m.lane orelse return .{};
        return .{ .rounds = l.stats.rounds, .drafted = l.stats.drafted, .accepted = l.stats.accepted, .generated = l.stats.generated };
    }

    /// The request's receipt line (rounds, accepted/round, misses/round, bytes/round); the host logs it with
    /// `[spec-stats]`.
    pub fn receipt(m: *Module, buf: []u8) []const u8 {
        const l = m.lane orelse return buf[0..0];
        // Profile builds: the request's `[mtp-profile]` summary (and its files) at the receipt, on the inference thread.
        if (comptime @import("profile.zig").enabled) m.mtpProfileFinish();
        // Rows audit: the decode phase's memory, once per request.
        m.logDecodeMemory();
        return l.stats.line(buf, l.name(), m.source.stream_source.stats());
    }
};

test "mimo SDK2 prepared parse and load facts expose the same installed shell" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", fixture_json);
    var diag: sdk.Diag = .{};
    const c = try parse(a, &peek, &diag);
    defer freeConfig(a, c);
    try std.testing.expectEqualStrings("/fixture", c.expert_bank_dir.?);
    try std.testing.expectEqual(@as(u32, 48), shell(c).num_layers);
    try std.testing.expectEqual(@as(?u8, 0), shell(c).kv_quant_bits);
    try std.testing.expectEqual(bills.requestBytesBF16(c.mimo, 33), promptBytes(c, 1, 32));
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10_000_000_000, .wired_margin_bytes = 2_000_000_000, .effective_context = 32_768, .kv_quant_bits = 8 };
    try applyLoadFacts(c, &facts);
    try std.testing.expectEqual(@as(?u8, 8), shell(c).kv_quant_bits);
    try std.testing.expectEqual(@as(u64, 32_768), c.reserveTokens());
    try std.testing.expectEqual(bills.requestBytes(c.mimo, 33), promptBytes(c, 1, 32));
    const resident = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", "{\"model_type\":\"mimo_v2_flash\"}");
    try std.testing.expectError(error.MimoRepackRequired, parse(a, &resident, &diag));
    try std.testing.expect(diag.message().len > 0);
}

test "mimo SDK2 KV choice defaults to BF16 and retains an explicit Q8 construction selection" {
    if (comptime @hasField(settings.Config, "kv_mode") and @hasDecl(settings.Config, "validate")) {
        var c: settings.Config = .{};
        try c.validate();
        try std.testing.expectEqual(settings.KVMode.bf16, c.kv_mode);
        const off = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"kv_quant\":\"off\"}", .{});
        defer off.deinit();
        c.applySettings(off.value);
        try c.validate();
        try std.testing.expectEqual(settings.KVMode.bf16, c.kv_mode);
        const q8 = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"kv_quant\":8}", .{});
        defer q8.deinit();
        c.applySettings(q8.value);
        try c.validate();
        try std.testing.expectEqual(settings.KVMode.q8, c.kv_mode);
        const unsupported = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"kv_quant\":4}", .{});
        defer unsupported.deinit();
        c.applySettings(unsupported.value);
        try std.testing.expectError(error.MimoKVModeUnsupported, c.validate());
    } else return error.MimoSDK2KVChoiceMissing;
}

test "mimo SDK2 claim accepts the prepared MXFP4 format and declines adjacent models" {
    if (comptime @hasDecl(@This(), "claims")) {
        try sdk.testing.expectClaims(@field(@This(), "claims"), &.{
            .{ .config =
            \\{"model_type":"mimo_v2_flash","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
            , .want = .native },
            .{ .config = "{\"model_type\":\"mimo_v2_flash\"}", .want = null },
            .{ .config =
            \\{"model_type":"mimo_v2_flash","quantization_config":{"quant_method":"fp8"}}
            , .want = null },
            .{ .config =
            \\{"model_type":"deepseek_v41","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
            , .want = null },
            .{ .config =
            \\{"model_type":"mimo_v2_flash","expert_streaming":{"family":"mlx-stream","format":"exl3","manifest":"expert-manifest.json"}}
            , .want = null },
            .{ .config =
            \\{"model_type":"mimo_v2_flash","expert_streaming":{"family":"mlx-stream","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"other.json"}}
            , .want = null },
            .{ .config =
            \\{"model_type":"mimo_v2_flash","expert_streaming":{"family":"other","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"expert-manifest.json"}}
            , .want = null },
            .{ .config = "{\"model_type\":\"mimo_v2_flash\",\"expert_streaming\":null}", .want = null },
            .{ .config = "{\"model_type\":\"mimo_v2_flash\",\"expert_streaming\":true}", .want = null },
            .{ .config = "{\"model_type\":\"mimo_v2_flash\",\"expert_streaming\":{\"family\":7}}", .want = null },
        });
    } else return error.MimoSDK2ClaimsMissing;
}

test "mimo SDK2 complete bill retains architecture source and row-following wiring terms" {
    if (comptime @hasDecl(@This(), "bill")) {
        const a = std.testing.allocator;
        const io = std.testing.io;
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeBillFixture(a, io, tmp.dir);
        const path = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(path);
        // The serial control (the fixture pack carries no MTP sidecar; the lane's terms: bill.zig "mimo mtp bill").
        const c: @This().Config = .{ .expert_bank_dir = path, .mtp_heads = false };
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const peek = try sdk.ConfigPeek.parse(arena.allocator(), path, "{}");
        const req: sdk.BillRequest = .{
            .peek = &peek,
            .cfg = &c,
            .routes = undefined,
            .prompt_tokens = 16_384,
            .max_tokens = 1_024,
            .ceiling = 120_000_000_000,
            .stop = 2_000_000_000,
        };
        const b = try @field(@This(), "bill")(a, io, &req);
        defer b.free(a);
        const fixture = try @import("bill.zig").fixtureBill(a);
        defer fixture.free(a);
        const names = [_][]const u8{
            "residents",           "mtp_heads",        "router_f32",         "mlx_cache",
            "mlx_cache_overshoot", "compiler_runtime", "host_runtime",       "resident_wiring",
            "buffer_allowance",    "request",          "kv_packed",          "kv_scales",
            "kv_biases",           "kv_dense",         "kv_dequant",         "kv_update",
            "kv_incoming",         "request_wiring",   "slot_transient",     "reader_staging",
            "source_metadata",     "expert_workspace", "event_runtime",      "event_graph",
            "spec_staging",        "spec_runtime",     "predictor_metadata", "predictor_workspace",
            "predictor_runtime",   "slot_wiring",
            "kv_sliding_ring",     "mtp_workspace",      "fp8_prompt_weights", "gpu_warm",
        };
        try std.testing.expectEqual(names.len, b.terms.len);
        for (names) |term_name| {
            var count: usize = 0;
            for (b.terms) |term| count += @intFromBool(std.mem.eql(u8, term_name, term.name));
            try std.testing.expectEqual(@as(usize, 1), count);
        }
        try std.testing.expectEqual(@as(u64, 47 * 13_369_344), b.per_row);
        try std.testing.expect(b.row_terms != null);
        try std.testing.expectEqualSlices(u64, &fixture.row_terms.?.data, &b.row_terms.?.data);
        // The source's twelve terms, after the architecture's.
        for (b.terms[b.terms.len - 12 ..], fixture.terms[fixture.terms.len - 12 ..]) |actual, expected| {
            try std.testing.expectEqual(expected, actual);
        }
        var row_terms: usize = 0;
        for (b.terms) |term| row_terms += @intFromBool(term.with_rows);
        try std.testing.expectEqual(@as(usize, 1), row_terms);
        for ([_]sdk.MemoryBill.Phase{ .prompt, .decode }) |phase| {
            try std.testing.expect(b.fixed(phase) > 4);
            for ([_]u32{ 1, 8, 128 }) |rows| {
                try std.testing.expect(b.total(phase, 0, rows) > b.fixed(phase) + rows * b.per_row);
            }
        }
        // The host's box: the 112 GiB ceiling less the pinned 2 GiB wired margin.
        const target: u64 = 120_259_084_288 - (2 << 30);
        const filled = try sdk.fill(b, 10_000_000_000, target, 256, 8);
        try std.testing.expect(b.total(.prompt, 10_000_000_000, filled.prompt) <= target);
        try std.testing.expect(b.total(.prompt, 10_000_000_000, filled.prompt + 1) > target);
    } else return error.MimoSDK2CompleteBillMissing;
}

/// G4 (`sdk.testing.expectBillBoundsLoad`): the load preflight (`loadBytes`) and the itemized bill (`bill`) read one bill
/// at the fill's floor rows, at the host's ceiling and pinned 2 GiB stop, for the standard 16K + 1K request.
fn expectBillBoundsLoadAt(dir: []const u8, kv_bits: u8) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), dir, "{}");
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10_000_000_000, .wired_margin_bytes = 2 << 30, .effective_context = 17_408, .kv_quant_bits = kv_bits };
    var c: Config = .{ .expert_bank_dir = dir, .mtp_heads = false }; // the serial control: the fixture pack has no MTP sidecar
    try applyLoadFacts(&c, &facts);
    const req: sdk.BillRequest = .{ .peek = &peek, .cfg = &c, .routes = undefined, .prompt_tokens = 16_384, .max_tokens = 1_024, .ceiling = 120_259_084_288, .stop = facts.wired_margin_bytes };
    const floor: sdk.Rows = .{ .prompt = mimo.default_expert_rows, .decode = mimo.default_expert_rows };
    try sdk.testing.expectBillBoundsLoad(@This(), a, io, &c, &facts, &req, floor);
}

test "mimo SDK2 the itemized bill bounds the process exactly as the load preflight bills it" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeBillFixture(a, io, tmp.dir);
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try expectBillBoundsLoadAt(path, 0);
    try expectBillBoundsLoadAt(path, 8);
}

// MIMO_BANK=<prepared pack> (headers only): the same on the real pack's residents and manifest.
test "mimo SDK2 the itemized bill bounds the process exactly as the load preflight bills it (bank)" {
    const dir = std.mem.span(std.c.getenv("MIMO_BANK") orelse return error.SkipZigTest);
    try expectBillBoundsLoadAt(dir, 0);
    try expectBillBoundsLoadAt(dir, 8);
}

/// Metadata only: the bill's manifest reader never opens an expert payload.
fn writeBillFixture(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const bank = @import("bank.zig");
    const g = try bank.Geometry.init(4096, 2048, 256, 1, 47);
    const Segment = struct { name: []const u8, dtype: []const u8, shape: [2]u64, offset: u64, length: u64 };
    var segments: [bank.n_components]Segment = undefined;
    for (&segments, g.segments, 0..) |*s, source, i| {
        const component: bank.Component = @fromBackingInt(@intCast(i));
        s.* = .{ .name = component.name(), .dtype = @tagName(source.dtype), .shape = source.shape, .offset = source.offset, .length = source.length };
    }
    const Record = struct { layer: u32, expert: u32, offset: u64, length: u64, sha256: []const u8 };
    const records = try a.alloc(Record, @as(usize, g.num_experts) * g.num_moe_layers);
    defer a.free(records);
    const digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    for (records, 0..) |*r, i| r.* = .{
        .layer = @intCast(g.first_moe_layer + i / g.num_experts),
        .expert = @intCast(i % g.num_experts),
        .offset = i * g.record_bytes,
        .length = g.record_bytes,
        .sha256 = digest,
    };
    const manifest = try std.json.Stringify.valueAlloc(a, .{
        .format = bank.format,
        .data_file = "experts.bin",
        .hidden_size = g.hidden_size,
        .intermediate_size = g.intermediate_size,
        .num_experts = g.num_experts,
        .first_moe_layer = g.first_moe_layer,
        .num_moe_layers = g.num_moe_layers,
        .record_bytes = g.record_bytes,
        .gate_up_bytes = g.gate_up_bytes,
        .quantization = .{ .mode = "mxfp4", .bits = 4, .group_size = 32 },
        .source = .{ .repository = "mimo-sdk2-bill-fixture", .revision = "0123456789012345678901234567890123456789" },
        .segments = segments,
        .records = records,
        .output_files = .{ .@"experts.bin" = .{ .size = g.total_bytes, .sha256 = digest } },
    }, .{});
    defer a.free(manifest);
    try dir.writeFile(io, .{ .sub_path = "expert-manifest.json", .data = manifest });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"model.norm.weight":"resident.safetensors"}}
    });
    const header =
        \\{"model.norm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
    ;
    var shard: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    try dir.writeFile(io, .{ .sub_path = "resident.safetensors", .data = &shard });
}

test "mimo draft lane: the table the registry builds; no lane = serial (block 0, off); a lane arms clean requests by tier" {
    const vt = comptime sdk.Arch.of(@This());
    try std.testing.expect(vt.spec == .draft_lane);
    var m: Module = undefined;
    m.lane = null;
    try std.testing.expectEqual(@as(u32, 0), draft_lane.blockSize(&m));
    try std.testing.expectEqualStrings("serial", draft_lane.laneName(&m));
    try std.testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try std.testing.expectEqual(sdk.DraftStats{}, draft_lane.stats(&m));
    var lane: draft.Lane = undefined;
    lane.rule = .exact;
    lane.name_len = 0;
    lane.stats = .{};
    m.lane = &lane;
    try std.testing.expectEqual(@as(u32, 4), draft_lane.blockSize(&m));
    try std.testing.expectEqual(sdk.DraftArm.greedy, draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try std.testing.expectEqual(sdk.DraftArm.stochastic, draft_lane.arm(&m, .{ .greedy = false, .clean = true }));
    try std.testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(&m, .{ .greedy = true, .clean = false }));
    lane.rule = .{ .typical = .{ .delta = 0.3 } };
    try std.testing.expectEqual(sdk.DraftArm.typical, draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try std.testing.expectEqual(sdk.DraftArm.typical, draft_lane.arm(&m, .{ .greedy = false, .clean = true }));
    try std.testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(&m, .{ .greedy = false, .clean = false }));
    lane.stats.add(3, 2);
    try std.testing.expectEqual(sdk.DraftStats{ .rounds = 1, .drafted = 3, .accepted = 2, .generated = 3 }, draft_lane.stats(&m));
}

test "mimo filepart: construction fills on the box baseline, the preflight and a dirless config on the host's facts" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    const huge: u64 = 1 << 60;
    // No pack directory, or no host baseline: the host's facts, as before.
    const dirless: Config = .{};
    try std.testing.expectEqual(@as(?u64, 9_000_000_000), (try admitted(io, &dirless, &.{ .memory_baseline_bytes = 9_000_000_000, .wired_margin_bytes = 0 })).memory_baseline_bytes);
    const packed_cfg: Config = .{ .expert_bank_dir = dir };
    try std.testing.expectEqual(@as(?u64, null), (try admitted(io, &packed_cfg, &.{ .wired_margin_bytes = 0 })).memory_baseline_bytes);
    // A pack: the box's baseline, never below the host's non-file one (a larger host value wins exactly).
    try std.testing.expectEqual(@as(?u64, huge), (try admitted(io, &packed_cfg, &.{ .memory_baseline_bytes = huge, .wired_margin_bytes = 0 })).memory_baseline_bytes);
    const box = (try admitted(io, &packed_cfg, &.{ .memory_baseline_bytes = 1, .wired_margin_bytes = 0 })).memory_baseline_bytes.?;
    try std.testing.expect(box >= 1);
    if (comptime @import("builtin").os.tag.isDarwin()) try std.testing.expect(box > 1_000_000_000);
    // withFacts (applyLoadFacts, loadBytes) is unchanged: the host's value.
    try std.testing.expectEqual(@as(?u64, 9_000_000_000), packed_cfg.withFacts(&.{ .memory_baseline_bytes = 9_000_000_000, .wired_margin_bytes = 0 }).memory_baseline_bytes);
}

test "mimo pack settings: expert_streaming.settings is applied at parse; a non-object block or an unknown key refuses by name" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const marker = "\"manifest\": \"expert-manifest.json\"";
    std.debug.assert(std.mem.indexOf(u8, fixture_json, marker) != null);
    const Case = struct { block: []const u8, err: ?anyerror };
    for ([_]Case{
        .{ .block = ", \"settings\": {\"mtp_lane\": false, \"decode_glue\": \"ops\", \"prefill_read_schedule\": \"current\"}", .err = null },
        .{ .block = ", \"settings\": [1]", .err = error.MimoSettingsBlock },
        .{ .block = ", \"settings\": {\"embedding\": \"rows\"}", .err = error.MimoSettingUnknown },
    }) |case| {
        const text = try std.mem.replaceOwned(u8, arena.allocator(), fixture_json, marker, try std.mem.concat(arena.allocator(), u8, &.{ marker, case.block }));
        const peek = try sdk.ConfigPeek.parse(arena.allocator(), "/fixture", text);
        var diag: sdk.Diag = .{};
        if (case.err) |want| {
            try std.testing.expectError(want, parse(a, &peek, &diag));
            try std.testing.expect(diag.message().len > 0);
            continue;
        }
        const c = try parse(a, &peek, &diag);
        defer freeConfig(a, c);
        try std.testing.expect(!c.mtp_heads);
        try std.testing.expectEqual(@import("glue.zig").Route.ops, c.decode_glue);
        try std.testing.expectEqual(@import("stream.zig").ReadSchedule.current, c.prefill_read_schedule);
    }
}
