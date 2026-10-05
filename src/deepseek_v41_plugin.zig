//! DeepSeek-V4.1 behind the `arch` kind (sdk.Arch.of): every declaration wraps the arch's own entry. The host
//! reaches the module through these, once per prompt, serial step, handover or draft round; nothing per layer.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const v41 = @import("deepseek_v41.zig");
const settings = @import("deepseek_v41_settings.zig");
const module = @import("deepseek_v41_module.zig");
const bill_mod = @import("deepseek_v41_bill.zig");

pub const name = "deepseek_v41";

pub const caps: sdk.Caps = .{
    .owns_decode_state = true,
    .prefill_whole_prompt = true,
    .prefill_yields_last_logits = true,
    .residents_past_page_cache = true,
};

pub const Config = settings.Config;
pub const Module = module.Module;

/// G6: the process's one expert reader. The host claims it at the load claim, before the preflight and the weights,
/// when the stream reads through it, and releases it when the loaded model goes (or the load fails). A second load
/// that needs it is refused by name (`error.ExpertReaderInUse`).
pub const claimProcess = if (@import("expert_stream.zig").uses_reader) sdk_ext.expert.takeReader else {};
pub const releaseProcess = if (@import("expert_stream.zig").uses_reader) sdk_ext.expert.giveReader else {};

pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    const t = p.modelType() orelse return null;
    return if (std.mem.eql(u8, t, name)) .native else null;
}

/// The arch's own parse of config.json (refusals by name), its prompt-pass bill, and the sidecars beside it in the
/// model directory (none without one).
pub fn parse(gpa: std.mem.Allocator, p: *const sdk.ConfigPeek, diag: *sdk.Diag) !*Config {
    var vd: v41.Diag = .{};
    const c41 = v41.Config.parse(gpa, p.text, &vd) catch |e| {
        diag.set("{s}", .{vd.message()});
        return e;
    };
    const c = try gpa.create(Config);
    errdefer gpa.destroy(c);
    // The prompt admission bills the rings only on the served tier (`promptBytes`; stock bills its full history), so its
    // bill rows them at the served tier's states' geometry (`module.kvGeometry` of a served config).
    c.* = .{ .dsv41_prefill = .of(&c41, module.numericTier(.served).kv), .num_experts = c41.n_routed_experts, .num_hidden_layers = c41.n_layers };
    if (p.model_dir.len > 0) {
        c.expert_bank_dir = try gpa.dupe(u8, p.model_dir);
        errdefer gpa.free(c.expert_bank_dir.?);
        c.engram_token_map_path = try std.fmt.allocPrint(gpa, "{s}/" ++ module.engram_token_map_file, .{p.model_dir});
    }
    return c;
}

pub fn freeConfig(gpa: std.mem.Allocator, c: *Config) void {
    if (c.expert_bank_dir) |d| gpa.free(d);
    if (c.engram_token_map_path) |m| gpa.free(m);
    gpa.destroy(c);
}

pub fn shell(c: *const Config) sdk.Shell {
    return .{ .num_experts = c.num_experts, .num_layers = c.num_hidden_layers };
}

pub fn applySettings(c: *Config, raw: std.json.Value) void {
    c.applySettings(raw);
}

/// The load preflight's requirement: the native bill's process bound at the fill's floor rows.
pub fn loadBytes(gpa: std.mem.Allocator, io: std.Io, c: *const Config, facts: *const sdk.LoadFacts, ceiling: u64) !u64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    return bill_mod.loadRequirementBytes(arena.allocator(), io, c.withFacts(facts), ceiling);
}

/// G4: the bill's terms at the request's shape, rows-free (taken at the fill's floor rows), through the routes the
/// host passes (the served path's: none). The load preflight's inputs (`loadRequirementBytes`): no baseline in the
/// terms, the bank's headers through `io`, no device.
pub fn bill(gpa: std.mem.Allocator, io: std.Io, req: *const sdk.BillRequest) !sdk.MemoryBill {
    const ov: *const module.RouteOverrides = @ptrCast(@alignCast(req.routes));
    var c = @as(*const Config, @ptrCast(@alignCast(req.cfg))).*;
    c.memory_baseline_bytes = 0;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const b = try bill_mod.billAtFloor(arena.allocator(), io, c, req.prompt_tokens, req.max_tokens, null, req.ceiling, ov.*);
    return bill_mod.memoryBill(gpa, b);
}

/// The prompt admission's bytes: the tier's bill for the pass the module builds (K16 layer-major, or chunk-major).
pub fn promptBytes(c: *const Config, seq: u64, max_tokens: u32) u64 {
    _ = c;
    _ = seq;
    _ = max_tokens;
    return prompt_bytes_beyond_admission;
}

/// What a prompt needs beyond what the Module already holds, for the host's per-request guard (it compares this with the
/// box's free memory NOW, which after a request still counts the Module's decode-phase rows): nothing. The construction
/// admitted every prompt up to the billed context (`bill.servedBill`: the covering wave, both phases under the target),
/// and each prompt pass starts with the reverse phase change, which returns the previous request's decode rows before
/// anything of the prompt allocates (its settle checked by name). A prompt over the billed context is the Module's own
/// refusal by name before its pass (`ContextOverBill`), never a memory guess here.
pub const prompt_bytes_beyond_admission: u64 = 0;

pub fn init(load: *const sdk.LoadCtx, c: *const Config) !*Module {
    const built = c.withFacts(&load.facts);
    return Module.init(load.gpa, load.io, &built, load.weights, load.stream, .{ .ceiling = load.ceiling, .wired_margin = load.facts.wired_margin_bytes, .loader = load.loader });
}

pub fn deinit(m: *Module) void {
    m.deinit();
}

/// The served prompt reserves no positions beyond the module's own bound (the prompt and the generation headroom).
pub fn reservedTokens(_: sdk.RequestShape) u64 {
    return 0;
}

/// The prompt pass: from position 0, or after the positions `restorePrefix` kept (`req.prompt_tokens` counts them).
pub fn prefill(m: *Module, ids: []const u32, req: sdk.RequestShape) !sdk.mlx.mlx_array {
    return m.prefillAt(req.prompt_tokens - ids.len, ids, reservedTokens(req), true);
}

/// Multi-turn over the host's prefix cache: the positions of the host's match the Module's kept boundary honours.
pub fn restorePrefix(m: *Module, prefix: []const u32) u64 {
    return m.restorePrefix(prefix);
}

pub fn step(m: *Module, ids: []const u32) !sdk.mlx.mlx_array {
    return m.extend(ids);
}

pub fn position(m: *const Module) u64 {
    return m.position();
}

pub fn handover(m: *Module, h: sdk.DecodeHandover) !void {
    return m.decodeHandover(h);
}

/// The DSpark draft head's lane: typical acceptance at the tier's delta, inside the module.
pub const draft_lane = struct {
    pub fn blockSize(m: *const Module) u32 {
        return m.draftBlockSize();
    }

    pub fn laneName(m: *const Module) []const u8 {
        return m.decodeLane();
    }

    /// A clean greedy request takes typical acceptance (with the greedy correction); sampled and shaped ones stay
    /// serial.
    pub fn arm(_: *const Module, req: sdk.ArmRequest) sdk.DraftArm {
        return if (req.greedy and req.clean) .typical else .off;
    }

    pub fn round(m: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32) !sdk.DraftRound {
        const r = try m.dsparkRound(a, t1, accepted_cap);
        return .{ .tokens = r.tokens, .accepted = r.accepted, .next_token = r.next_token };
    }

    pub fn stats(m: *const Module) sdk.DraftStats {
        const s = m.dsparkStats() orelse return .{};
        return .{ .rounds = s.cycles, .drafted = s.drafted_tokens, .accepted = s.accepted_drafts, .generated = s.generated_tokens };
    }
};

const testing = std.testing;

test "dsv41 plugin: claims its own model_type at native priority and declines the rest" {
    const real = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(real);
    try sdk.testing.expectClaims(claims, &.{
        .{ .config = real, .want = .native },
        .{ .config = "{\"model_type\": \"deepseek_v4\"}", .want = null },
        .{ .config = "{\"architectures\": [\"DeepseekV4ForCausalLM\"]}", .want = null },
    });
}

test "dsv41 plugin: parse builds the arch's config and its sidecar paths, and refuses by the arch's own names" {
    const real = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(real);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag: sdk.Diag = .{};
    const in_dir = try parse(testing.allocator, &try sdk.ConfigPeek.parse(arena.allocator(), "/m", real), &diag);
    defer freeConfig(testing.allocator, in_dir);
    try testing.expectEqualStrings("/m", in_dir.expert_bank_dir.?);
    try testing.expectEqualStrings("/m/engram-token-map.u32", in_dir.engram_token_map_path.?);
    try testing.expect(in_dir.dsv41_prefill != null);
    try testing.expectEqual(sdk.Shell{ .num_experts = 384, .num_layers = 40 }, shell(in_dir));
    const no_dir = try parse(testing.allocator, &try sdk.ConfigPeek.parse(arena.allocator(), "", real), &diag);
    defer freeConfig(testing.allocator, no_dir);
    try testing.expect(no_dir.expert_bank_dir == null and no_dir.engram_token_map_path == null);
    const bad = try std.mem.replaceOwned(u8, arena.allocator(), real, "sqrtsoftplus", "softmax");
    try testing.expectError(error.NotImplemented, parse(testing.allocator, &try sdk.ConfigPeek.parse(arena.allocator(), "/m", bad), &diag));
    try testing.expect(diag.message().len > 0);
}

test "dsv41 plugin: the draft lane arms only clean greedy requests, and the served prompt reserves nothing" {
    const m: *const Module = undefined;
    try testing.expectEqual(sdk.DraftArm.typical, draft_lane.arm(m, .{ .greedy = true, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(m, .{ .greedy = true, .clean = false }));
    try testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(m, .{ .greedy = false, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, draft_lane.arm(m, .{ .greedy = false, .clean = false }));
    for ([_]sdk.RequestShape{ .{ .prompt_tokens = 1, .max_tokens = 0, .host_context = 0 }, .{ .prompt_tokens = 16384, .max_tokens = 1024, .host_context = 1 << 20 } }) |req|
        try testing.expectEqual(@as(u64, 0), reservedTokens(req));
}

test "dsv41 plugin: the table the registry builds (owns its decode state, a handover, a draft lane, its bills)" {
    const vt = comptime sdk.Arch.of(@This());
    try testing.expect(vt.caps.owns_decode_state and !vt.caps.batches_decode);
    try testing.expect(vt.handover != null and vt.prompt_bytes != null and vt.bill != null and vt.spec == .draft_lane);
    // The host's per-request guard: a prompt needs nothing beyond the Module's own admission (the covering bill), at
    // any length; the context is the Module's refusal by name.
    var pc: Config = .{};
    for ([_]u64{ 1, 2047, 4096, 16384, 131072, 1 << 20 }) |n| try testing.expectEqual(@as(u64, 0), vt.prompt_bytes.?(@ptrCast(&pc), n, 64));
}

// DSV41_BANK=<bank> (host): the term-wise bill's process bound at the fill's floor rows is the load preflight's number.
test "dsv41 plugin: the term-wise bill bounds the process exactly as the load preflight bills it (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const host = try @import("deepseek_v41_host.zig").model.parseConfig(testing.io, a, bank_dir);
    const cfg: *const Config = @ptrCast(@alignCast(host.arch_cfg.?));
    const ceiling: u64 = 120_259_084_288;
    const ov: module.RouteOverrides = .{};
    const p = try sdk.ConfigPeek.parse(a, bank_dir, "{}");
    const req: sdk.BillRequest = .{ .peek = &p, .cfg = cfg, .routes = &ov, .prompt_tokens = bill_mod.fill_prompt_tokens, .max_tokens = bill_mod.fill_max_tokens, .ceiling = ceiling, .stop = module.ceiling_stop_bytes };
    const floor: sdk.Rows = .{ .prompt = bill_mod.min_fill_rows, .decode = bill_mod.min_fill_rows };
    try sdk.testing.expectBillBoundsLoad(@This(), testing.allocator, testing.io, cfg, &host.loadFacts(), &req, floor);
}
