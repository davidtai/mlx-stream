//! The MTP draft lane against its reference: the tiny model of `scripts/glm_moe_dsa_mtp_goldens.py` (the parity
//! fixture's trunk with an MTP layer of random BF16 weights, its experts dense) through the served module's code on
//! MLX's CPU stream, the lane's bank kind `.dense`. Per case (two prompts x depth 1..5 x the MTP's own drafts or
//! forced ones) every round's t1, drafts, draft-step logits, accepted count and next token against the reference's,
//! and the rounds' tokens against the serial greedy decode; after the rounds, both KV states against a serial run's.
//! Window inputs: GLM53_MTP_PARITY=<the fixture's pack (mtp-goldens.json and mtp/ beside it)> with the MLX device
//! allowed (DSV41_PHASE0B_MLX=1); skipped otherwise. The refusals below need neither.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const quant = sdk_ext.quant;
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const module = @import("glm_moe_dsa_module.zig");
const bank_mod = @import("glm_moe_dsa_bank.zig");
const graph = @import("glm_moe_dsa_graph.zig");
const mtp = @import("glm_moe_dsa_mtp.zig");
const plugin = @import("glm_moe_dsa_plugin.zig");

const testing = std.testing;
const M = module.ModuleOf(bank_mod, quant.FromGatherMatmul(quant.GatherQmm), .dense);
const G = graph.G;

const Round = struct { t1: u32, forced: u32, drafts: []const u32, draft_logits: []const []const f32, accepted: u32, next: u32 };
const Case = struct { name: []const u8, schedule: []const u8, prompt: []const u32, depth: u32, serial: []const u32, rounds: []const Round };
const Goldens = struct { tokens: u32, cases: []const Case };

/// A module of the fixture at `depth` (exact acceptance), its weights beside it.
const Rig = struct {
    cfg: settings.Config,
    weights: sdk.Weights,
    m: *M,
    s: mlx.mlx_stream,

    fn open(a: std.mem.Allocator, dir: []const u8, depth: u32) !*Rig {
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        // The module copies the config and borrows its model's storage: the rig keeps it.
        r.cfg = .{ .model_dir = dir, .model = try glm.Config.load(a, testing.io, dir, null, null), .max_context_tokens = 64, .expert_rows = 6, .expert_prefill_rows = 4, .mtp_depth = depth };
        errdefer r.cfg.deinit(a);
        r.weights = try sdk.loader.dir(testing.io, a, dir, .{});
        errdefer r.weights.deinit();
        r.s = mlx.mlx_default_cpu_stream_new();
        r.m = try M.initWith(a, testing.io, &r.cfg, &r.weights, r.s, .{ .ceiling = 64 << 30, .wired_margin = 0 }, .{});
        return r;
    }

    fn close(r: *Rig, a: std.mem.Allocator) void {
        r.m.deinit();
        r.weights.deinit();
        _ = mlx.mlx_stream_free(r.s);
        r.cfg.deinit(a);
        a.destroy(r);
    }

    fn argmaxOf(r: *Rig, x: mlx.mlx_array) !u32 {
        defer _ = mlx.mlx_array_free(x);
        return r.m.g.hostArgmax(x);
    }

    /// The prompt pass and the handover: the first token.
    fn begin(r: *Rig, prompt: []const u32, max_tokens: u32) !u32 {
        _ = r.m.restorePrefix(&.{});
        const t1 = try r.argmaxOf(try r.m.prefillAt(0, prompt));
        try r.m.decodeHandover(.{ .prompt_tokens = @intCast(prompt.len), .reserved_tokens = prompt.len + max_tokens, .native_draft = true });
        return t1;
    }
};

fn fixtureDir() ?[]const u8 {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return null;
    return std.mem.span(std.c.getenv("GLM53_MTP_PARITY") orelse return null);
}

fn loadGoldens(a: std.mem.Allocator, dir: []const u8) !std.json.Parsed(Goldens) {
    const path = try std.fmt.allocPrint(a, "{s}/mtp-goldens.json", .{dir});
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    return std.json.parseFromSlice(Goldens, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

const greedy: sdk.SamplingParams = .{ .temperature = 0 };

/// Every round of `cs` through the lane against the reference; returns the largest draft-logit delta. The rig is left
/// after the case's last round.
fn runCase(a: std.mem.Allocator, r: *Rig, cs: *const Case, vocab: usize) !f32 {
    var t1 = try r.begin(cs.prompt, @intCast(cs.serial.len + 8));
    try testing.expectEqual(cs.rounds[0].t1, t1);
    const lg = try a.alloc(f32, cs.depth * vocab);
    defer a.free(lg);
    var emitted: std.ArrayList(u32) = .empty;
    defer emitted.deinit(a);
    var worst: f32 = 0;
    for (cs.rounds, 0..) |rd, i| {
        testing.expectEqual(rd.t1, t1) catch |e| {
            std.debug.print("{s} {s} depth {d} round {d}: t1 {d}, reference {d}\n", .{ cs.name, cs.schedule, cs.depth, i, t1, rd.t1 });
            return e;
        };
        var probe: M.Probe = .{ .force = rd.drafts[0..rd.forced], .logits = lg };
        var out = try r.m.roundWith(a, t1, std.math.maxInt(u32), greedy, &probe);
        defer out.deinit(a);
        try testing.expectEqual(cs.depth, probe.depth);
        testing.expectEqualSlices(u32, rd.drafts, probe.drafts[0..probe.depth]) catch |e| {
            std.debug.print("{s} {s} depth {d} round {d}: drafts differ\n", .{ cs.name, cs.schedule, cs.depth, i });
            return e;
        };
        for (rd.draft_logits, 0..) |row, k| for (row, lg[k * vocab ..][0..vocab]) |x, y| {
            worst = @max(worst, @abs(x - y));
        };
        try testing.expectEqual(rd.accepted, out.accepted);
        try testing.expectEqual(rd.next, out.next_token);
        try emitted.appendSlice(a, out.tokens);
        t1 = out.next_token;
    }
    try testing.expectEqualSlices(u32, cs.serial, emitted.items[0..cs.serial.len]);
    return worst;
}

test "glm mtp parity: every round's drafts, draft logits, accepted count and next token equal the reference's; the rounds emit the serial decode's tokens at every depth" {
    const dir = fixtureDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const gold = try loadGoldens(a, dir);
    defer gold.deinit();
    var worst: f32 = 0;
    var accepted: u64 = 0;
    for (1..mtp.max_depth + 1) |depth| {
        const r = try Rig.open(a, dir, @intCast(depth));
        defer r.close(a);
        const vocab = r.m.model.vocab_size;
        for (gold.value.cases) |*cs| if (cs.depth == depth) {
            const w = try runCase(a, r, cs, vocab);
            for (cs.rounds) |rd| accepted += rd.accepted;
            std.debug.print("glm mtp parity {s} {s} depth {d}: {d} rounds, draft logits max |delta| {d:.6}\n", .{ cs.name, cs.schedule, depth, cs.rounds.len, w });
            worst = @max(worst, w);
        };
    }
    try testing.expect(accepted > 0);
    // The CPU stream runs the reference's own kernels over the same ops: exact.
    try testing.expectEqual(@as(f32, 0), worst);
}

/// Every lane's first rows as host f32 (the target's layers, then the MTP layer's), and the lengths.
const Snapshot = struct {
    rows: std.ArrayList(f32) = .empty,
    history: []u32 = &.{},
    pairs: u32 = 0,
    pending: u32 = 0,

    fn take(a: std.mem.Allocator, r: *Rig) !Snapshot {
        var out: Snapshot = .{ .history = try a.dupe(u32, r.m.history.items) };
        errdefer out.deinit(a);
        const g = &r.m.g;
        const n: c_int = @intCast(r.m.cache.len);
        for (0..r.m.model.n_layers) |li| {
            const l: u32 = @intCast(li);
            try out.lane(a, g, try r.m.cache.latentView(g, l), n);
            try out.lane(a, g, try r.m.cache.ropeView(g, l), n);
            if (r.m.model.isFull(l)) try out.lane(a, g, try r.m.cache.indexView(g, l), n);
        }
        // The MTP layer: every pair whose next token is known appended, the last position pending.
        const ln = r.m.mtp.?;
        try ln.append(g, &r.m.w, out.history[ln.cache.len + 1 ..]);
        const p: c_int = @intCast(ln.cache.len);
        try out.lane(a, g, try ln.cache.latentView(g, 0), p);
        try out.lane(a, g, try ln.cache.ropeView(g, 0), p);
        try out.lane(a, g, try ln.cache.indexView(g, 0), p);
        out.pairs = ln.cache.len;
        out.pending = ln.n_pending;
        return out;
    }

    fn lane(self: *Snapshot, a: std.mem.Allocator, g: *G, x: mlx.mlx_array, n: c_int) !void {
        const m = g.mark();
        defer g.resetTo(m);
        const v = try g.astype(try graph.rowSlice(g, x, 0, n), .float32);
        const at = self.rows.items.len;
        try self.rows.resize(a, at + @as(usize, @intCast(g.shapeOf(v).numel())));
        _ = try g.hostF32(v, self.rows.items[at..]);
    }

    fn deinit(self: *Snapshot, a: std.mem.Allocator) void {
        self.rows.deinit(a);
        a.free(self.history);
    }
};

test "glm mtp parity: after the rounds the target's lanes and the MTP layer's equal a serial run's over the same tokens, bit for bit, and hold exactly the committed positions" {
    const dir = fixtureDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const gold = try loadGoldens(a, dir);
    defer gold.deinit();
    for (gold.value.cases) |*cs| {
        if (!std.mem.eql(u8, cs.schedule, "forced") or (cs.depth != 2 and cs.depth != 5)) continue;
        // One module at a time: the process has one expert reader.
        var rounds = blk: {
            const r = try Rig.open(a, dir, cs.depth);
            defer r.close(a);
            _ = try runCase(a, r, cs, r.m.model.vocab_size);
            try testing.expectEqual(r.m.history.items.len, r.m.cache.len);
            break :blk try Snapshot.take(a, r);
        };
        defer rounds.deinit(a);
        // The serial run: the prompt pass, then each committed token as a serial step.
        var serial = blk: {
            const s = try Rig.open(a, dir, cs.depth);
            defer s.close(a);
            _ = try s.begin(cs.prompt, 64);
            for (rounds.history[cs.prompt.len..]) |t| _ = try s.argmaxOf(try s.m.extend(&.{t}));
            break :blk try Snapshot.take(a, s);
        };
        defer serial.deinit(a);
        try testing.expectEqualSlices(u32, rounds.history, serial.history);
        try testing.expectEqual(rounds.history.len - 1, rounds.pairs);
        try testing.expectEqual(rounds.pairs, serial.pairs);
        try testing.expectEqual(@as(u32, 1), rounds.pending);
        try testing.expectEqual(rounds.rows.items.len, serial.rows.items.len);
        try testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(rounds.rows.items), std.mem.sliceAsBytes(serial.rows.items)));
        std.debug.print("glm mtp parity {s} depth {d}: {d} target positions and {d} MTP pairs equal the serial run's ({d} values)\n", .{ cs.name, cs.depth, rounds.history.len, rounds.pairs, rounds.rows.items.len });
    }
}

test "glm mtp parity: a later prompt keeps the prefix the lane tracks and drafts the reference's tokens after it" {
    const dir = fixtureDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const gold = try loadGoldens(a, dir);
    defer gold.deinit();
    for (gold.value.cases) |*cs| {
        if (!std.mem.eql(u8, cs.schedule, "mtp") or cs.depth != 3 or !std.mem.eql(u8, cs.name, "long")) continue;
        const r = try Rig.open(a, dir, cs.depth);
        defer r.close(a);
        _ = try runCase(a, r, cs, r.m.model.vocab_size);
        // The same prompt again: the host restores prompt[0 .. n - 1], whose last pair the lane formed with the token
        // after the match (here the prompt's own, in general another): the state keeps one position less.
        const kept = r.m.restorePrefix(cs.prompt[0 .. cs.prompt.len - 1]);
        try testing.expectEqual(cs.prompt.len - 2, kept);
        try testing.expectEqual(kept, r.m.mtp.?.tracked());
        const t1 = try r.argmaxOf(try r.m.prefillAt(kept, cs.prompt[kept..]));
        try r.m.decodeHandover(.{ .prompt_tokens = @intCast(cs.prompt.len), .reserved_tokens = cs.prompt.len + 32, .native_draft = true });
        try testing.expectEqual(cs.rounds[0].t1, t1);
        var probe: M.Probe = .{};
        var out = try r.m.roundWith(a, t1, std.math.maxInt(u32), greedy, &probe);
        defer out.deinit(a);
        try testing.expectEqualSlices(u32, cs.rounds[0].drafts, probe.drafts[0..probe.depth]);
        try testing.expectEqual(cs.rounds[0].next, out.next_token);
    }
}

test "glm mtp parity: a round's verify reads, each expert's by the first row that routes it, sum to the stream's demand bytes; with its first draft rejected the kept row read exactly a serial step's bytes from the same state" {
    const dir = fixtureDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const gold = try loadGoldens(a, dir);
    defer gold.deinit();
    const cs = for (gold.value.cases) |*c| {
        if (c.depth == 3 and std.mem.eql(u8, c.schedule, "mtp")) break c;
    } else return error.SkipZigTest;
    // One module at a time: the serial step's demand bytes from the prompt's state first.
    const serial_bytes = blk: {
        const r = try Rig.open(a, dir, cs.depth);
        defer r.close(a);
        const t1 = try r.begin(cs.prompt, 8);
        const b0 = r.m.stats().expert_bytes_read;
        _ = try r.argmaxOf(try r.m.extend(&.{t1}));
        break :blk r.m.stats().expert_bytes_read - b0;
    };
    const r = try Rig.open(a, dir, cs.depth);
    defer r.close(a);
    const t1 = try r.begin(cs.prompt, 8);
    // The greedy target's token after t1 is the serial decode's second: any other first draft is rejected.
    const wrong = (cs.serial[1] + 1) % r.m.model.vocab_size;
    var probe: M.Probe = .{ .force = &.{wrong} };
    const ln = r.m.mtp.?;
    const b0 = r.m.stats().expert_bytes_read;
    var out = try r.m.roundWith(a, t1, std.math.maxInt(u32), greedy, &probe);
    defer out.deinit(a);
    const round_bytes = r.m.stats().expert_bytes_read - b0;
    try testing.expectEqual(@as(u32, 0), out.accepted);
    try testing.expectEqual(cs.serial[1], out.next_token);
    try testing.expect(serial_bytes > 0);
    try testing.expectEqual(round_bytes, ln.counts.verify_bytes);
    try testing.expectEqual(serial_bytes, ln.counts.verify_bytes - ln.counts.rejected_bytes);
    try testing.expect(ln.counts.rejected_bytes > 0);
    // The decision reached the first draft only, rejected.
    var decided: u64 = 0;
    for (ln.counts.decided) |d| decided += d;
    try testing.expectEqual(@as(u64, 1), decided);
    for (ln.counts.accepted_by_p) |acc| try testing.expectEqual(@as(u64, 0), acc);
    std.debug.print("glm mtp parity: a rejected round read {d} B, {d} B of it for the rejected rows; a serial step {d} B\n", .{ round_bytes, ln.counts.rejected_bytes, serial_bytes });
}

// ── refusals and decisions (no fixture) ──

/// A synthetic pack of the tiny model with the release's MTP fields in its config (`index_share` as given).
fn synthPack(a: std.mem.Allocator, tmp: *std.testing.TmpDir, buf: []u8, index_share: bool) !struct { dir: []const u8, text: []u8, model: glm.Config } {
    const base = try glm.tinyConfigJson(a, glm.tiny_quant);
    defer a.free(base);
    const text = try std.mem.replaceOwned(u8, a, base, "\"num_hidden_layers\":5,", if (index_share) "\"num_hidden_layers\":5,\"num_nextn_predict_layers\":1,\"index_share_for_mtp_iteration\":true," else "\"num_hidden_layers\":5,\"num_nextn_predict_layers\":1,");
    errdefer a.free(text);
    var model = try glm.Config.parse(a, text, null, null);
    errdefer model.deinit(a);
    const bin = try bank_mod.writeSynth(a, testing.io, tmp.dir, &model, .{});
    a.free(bin);
    try glm.writeResidents(a, testing.io, tmp.dir, &model);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = text });
    return .{ .dir = try bank_mod.tmpRoot(tmp, buf), .text = text, .model = model };
}

test "glm mtp: a depth past the route limit, a pack without mtp/ and a config without a shared selection are refused by name at load, before any device array" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const p = try synthPack(a, &tmp, &rbuf, true);
    defer a.free(p.text);
    var cfg: settings.Config = .{ .model = p.model, .model_dir = p.dir };
    defer cfg.deinit(a);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const facts: sdk.LoadFacts = .{ .wired_margin_bytes = 2 << 30 };
    // mtp_depth 6: six drafts verify in seven rows of top-8 routes, 56 routed ids over the decode lane's 48.
    const json = try std.json.parseFromSlice(std.json.Value, a, "{\"mtp_depth\": 6}", .{});
    defer json.deinit();
    plugin.applySettings(&cfg, json.value);
    try testing.expectEqual(@as(?i64, 6), cfg.mtp_depth_over_limit);
    try testing.expectError(error.MtpDepthOverRouteLimit, plugin.loadBytes(a, testing.io, &cfg, &facts, 64 << 30));
    var weights = sdk.Weights.init(a);
    defer weights.deinit();
    try testing.expectError(error.MtpDepthOverRouteLimit, M.initWith(a, testing.io, &cfg, &weights, undefined, .{ .ceiling = 64 << 30, .wired_margin = 0 }, .{}));
    // mtp_depth 5 on a pack without its MTP directory.
    cfg.mtp_depth_over_limit = null;
    cfg.mtp_depth = 5;
    try testing.expectError(error.MtpPackMissing, plugin.loadBytes(a, testing.io, &cfg, &facts, 64 << 30));
    var diag: glm.Diag = .{};
    try testing.expectError(error.MtpPackMissing, @import("glm_moe_dsa_bill.zig").billOf(arena.allocator(), testing.io, &cfg, 1024, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "mtp/mtp-residents.safetensors") != null);
    try testing.expectError(error.MtpPackMissing, M.initWith(a, testing.io, &cfg, &weights, undefined, .{ .ceiling = 64 << 30, .wired_margin = 0 }, .{}));
    // Off, the same pack loads its bill with no MTP term.
    cfg.mtp_depth = 0;
    const mb = try @import("glm_moe_dsa_bill.zig").billOf(arena.allocator(), testing.io, &cfg, 1024, null);
    for (mb.terms) |t| try testing.expect(std.mem.indexOf(u8, t.name, "MTP") == null);
    // A config without index_share_for_mtp_iteration drafts one token at most.
    var tmp2 = testing.tmpDir(.{});
    defer tmp2.cleanup();
    var rbuf2: [512]u8 = undefined;
    const q = try synthPack(a, &tmp2, &rbuf2, false);
    defer a.free(q.text);
    var cfg2: settings.Config = .{ .model = q.model, .model_dir = q.dir, .mtp_depth = 2 };
    defer cfg2.deinit(a);
    try testing.expectError(error.MtpConfig, plugin.loadBytes(a, testing.io, &cfg2, &facts, 64 << 30));
}

test "glm mtp: the lane arms clean requests only, exact as greedy or stochastic, typical only when the setting names it" {
    var lane: module.Module.Lane = undefined;
    lane.mode = .exact;
    var m: module.Module = undefined;
    m.mtp = null;
    try testing.expectEqual(sdk.DraftArm.off, plugin.draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    m.mtp = &lane;
    try testing.expectEqual(sdk.DraftArm.greedy, plugin.draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.stochastic, plugin.draft_lane.arm(&m, .{ .greedy = false, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, plugin.draft_lane.arm(&m, .{ .greedy = true, .clean = false }));
    lane.mode = .{ .typical = .{ .delta = 0.3 } };
    try testing.expectEqual(sdk.DraftArm.typical, plugin.draft_lane.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.typical, plugin.draft_lane.arm(&m, .{ .greedy = false, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, plugin.draft_lane.arm(&m, .{ .greedy = false, .clean = false }));
}

/// The decisions on host logits `rows` (f32 [w, vocab]) through the lane's device graph.
fn decideHost(g: *G, rows: []const f32, w: usize, drafts: []const u32, mode: sdk.acceptance.Mode, sp: sdk.SamplingParams, base: u64) !mtp.Decision {
    const m = g.mark();
    defer g.resetTo(m);
    const lg = try g.hostArray(std.mem.sliceAsBytes(rows), &.{ @intCast(w), @intCast(rows.len / w) }, .float32);
    return mtp.decide(g, lg, drafts, mode, sp, base);
}

test "glm mtp: the decisions equal the host references: argmax, typical at temperature 1, the sampled point-mass rule, sampled typical" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const ds = @import("deepseek_v41_dspark.zig");
    const acc = sdk.acceptance;
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    // Three verify rows over a 6-token vocabulary: row 0 argmax 2 with 4 close, row 1 argmax 4 with 0 rare, row 2 argmax 1.
    const rows = [_]f32{ 0.1, 0.4, 3.0, 0.2, 2.6, -1.0, -2.0, 0.6, 0.4, 0.3, 3.0, 0.2, -0.5, 2.0, 0.1, 0.0, 0.3, 0.2 };
    const vocab = 6;
    // Greedy exact: the argmax row by row.
    try testing.expectEqual(mtp.Decision{ .accepted = 1, .next = 4 }, try decideHost(&g, &rows, 3, &.{ 2, 3 }, .exact, .{ .temperature = 0 }, 0));
    try testing.expectEqual(mtp.Decision{ .accepted = 2, .next = 1 }, try decideHost(&g, &rows, 3, &.{ 2, 4 }, .exact, .{ .temperature = 0 }, 0));
    try testing.expectEqual(mtp.Decision{ .accepted = 0, .next = 2 }, try decideHost(&g, &rows, 3, &.{ 4, 4 }, .exact, .{ .temperature = 0 }, 0));
    // Greedy typical: draft 4 at row 0 (not the argmax) passes at delta 0.3, draft 0 at row 1 does not.
    const typ: acc.Mode = .{ .typical = .{ .delta = 0.3 } };
    var p: [3][vocab]f64 = undefined;
    for (0..3) |r| ds.refDistribution(rows[r * vocab ..][0..vocab], .{ .temperature = 1 }, &p[r]);
    try testing.expect(p[0][4] > acc.typicalThreshold(&p[0], 0.3, 1.0) and p[1][0] <= acc.typicalThreshold(&p[1], 0.3, 1.0));
    try testing.expectEqual(mtp.Decision{ .accepted = 1, .next = 4 }, try decideHost(&g, &rows, 3, &.{ 4, 0 }, typ, .{ .temperature = 0 }, 0));
    // Sampled exact (the point-mass rule) and sampled typical against the host references, over many seeds.
    const sp_base: sdk.SamplingParams = .{ .temperature = 0.8, .top_p = 0.95 };
    var accepted_any = false;
    var rejected_any = false;
    for (0..64) |seed| {
        var sp = sp_base;
        sp.seed = seed;
        const sm: ds.Sampling = .{ .temperature = sp.temperature, .top_p = sp.top_p, .seed = sp.seed };
        const base: u64 = 40;
        const drafts = [_]u32{ 2, 4 };
        var want: mtp.Decision = .{ .accepted = 0, .next = 0 };
        var q: [3][vocab]f64 = undefined;
        for (0..3) |r| ds.refDistribution(rows[r * vocab ..][0..vocab], sm, &q[r]);
        for (0..3) |r| {
            const pos = base + r + 1;
            const step = ds.refSpec(&q[r], if (r < drafts.len) drafts[r] else null, ds.drawU(sp.seed, pos), ds.drawV(sp.seed, pos));
            if (step.accept) {
                want.accepted += 1;
                continue;
            }
            want.next = step.tok;
            break;
        }
        const got = try decideHost(&g, &rows, 3, &drafts, .exact, sp, base);
        try testing.expectEqual(want, got);
        accepted_any = accepted_any or got.accepted > 0;
        rejected_any = rejected_any or got.accepted < 2;
        // Sampled typical: the same p against its own entropy; the correction and the bonus drawn from p.
        var wt: mtp.Decision = .{ .accepted = 0, .next = 0 };
        for (0..3) |r| {
            if (r < drafts.len and q[r][drafts[r]] > acc.typicalThreshold(&q[r], 0.3, 1.0)) {
                wt.accepted += 1;
                continue;
            }
            wt.next = ds.refDraw(&q[r], ds.drawV(sp.seed, base + r + 1));
            break;
        }
        try testing.expectEqual(wt, try decideHost(&g, &rows, 3, &drafts, typ, sp, base));
    }
    try testing.expect(accepted_any and rejected_any);
}
