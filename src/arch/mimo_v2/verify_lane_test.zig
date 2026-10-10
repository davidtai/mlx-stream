//! The MTP verify's decode lane (`Source.decodeLane`): a decode-phase call of M <= 4 rows takes the scheduler's decode
//! lane (one route over the rows' union, the hit pass, the decode admissions, the lookahead and its pre-read); the
//! prompt's M >= 2 calls and any M >= 5 call keep the wide lane. CPU only: a host- or MLX-CPU-memory stream over a tmp
//! sidecar, the trace backend or the MLX CPU stream. No device, no lock.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const ops = @import("../deepseek_v41/ops.zig");
const shared = @import("../deepseek_v41/experts.zig");
const bank_mod = @import("bank.zig");
const stream_mod = @import("stream.zig");
const source_mod = @import("source.zig");
const prefill_math = @import("math.zig");
const moe = @import("moe.zig");
const decode_barrier = @import("decode_barrier.zig");
const quant = sdk_ext.quant;

const testing = std.testing;

test {
    // `MIMO_DIAG=missfree` reuses this fixture (src/tests.zig lists this file).
    _ = @import("diag_test.zig");
}
const StreamSource = stream_mod.StreamSource;
pub const top_k: u32 = 8;
pub const n_experts: u32 = 64;

/// A MiMo sidecar over [layer 0 dense, layers 1, 2 routed] x 64 experts: E8M0 scales near 2^-4..2^0 (finite outputs),
/// random FP4 words; the stream over it with `memory`'s slot rows (and the lookahead class when `lookahead`).
pub const Fixture = struct {
    geom: bank_mod.Geometry,
    tmp: std.testing.TmpDir,
    bytes: []u8,
    fd: sdk_ext.expert.io.UncachedFd,
    owner: bank_mod.Bank,
    bank: bank_mod.SDKBank,
    stream: *stream_mod.Stream,

    pub fn init(f: *Fixture, a: std.mem.Allocator, hidden: u32, inter: u32, rows: []const u32, memory: stream_mod.SlotMemory, lookahead: bool) !void {
        return f.initWith(a, hidden, inter, rows, memory, lookahead, false);
    }

    /// `gated`: the stream arms a host event (MiMo's gated route: the scheduler's miss parts wait on event values, never
    /// on the host; the trace backend's waits are aliases).
    pub fn initWith(f: *Fixture, a: std.mem.Allocator, hidden: u32, inter: u32, rows: []const u32, memory: stream_mod.SlotMemory, lookahead: bool, gated: bool) !void {
        f.geom = try bank_mod.Geometry.init(hidden, inter, n_experts, 1, 2);
        f.bytes = try a.alloc(u8, f.geom.total_bytes);
        errdefer a.free(f.bytes);
        var state: u32 = 0x5eed_1a7e;
        for (1..3) |layer| for (0..n_experts) |expert| for (f.geom.segments) |seg| {
            const at = f.bytes[f.geom.recordOffset(@intCast(layer), @intCast(expert)) + seg.offset ..][0..seg.length];
            for (at) |*b| b.* = switch (seg.dtype) {
                .U32 => @truncate(next(&state)),
                .U8 => @intCast(123 + next(&state) % 5),
            };
        };
        f.tmp = std.testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        try f.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts.bin", .data = f.bytes });
        var dir_buf: [512]u8 = undefined;
        var path_buf: [600]u8 = undefined;
        const dir = dir_buf[0..try f.tmp.dir.realPath(std.testing.io, &dir_buf)];
        const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/experts.bin", .{dir}, 0);
        f.fd = try sdk_ext.expert.openUncached(path, null);
        errdefer f.fd.close();
        f.owner = .{
            .allocator = a,
            .geometry = f.geom,
            .digests = &.{},
            .source_repository = "synthetic",
            .source_revision = "0000000000000000000000000000000000000000",
            .sidecar_digest = @splat(0),
            .sidecar_fd = f.fd.fd,
            .sidecar_file_size = f.fd.size,
        };
        f.bank = try bank_mod.SDKBank.init(a, &f.owner);
        errdefer f.bank.deinit();
        var opts = stream_mod.options(rows, memory);
        if (lookahead) opts.lookahead = stream_mod.lookahead;
        if (gated) opts.event = .{ .backend = .host, .watchdog_ms = 10_000 };
        f.stream = try stream_mod.Stream.init(a, &f.bank, opts);
    }

    pub fn deinit(f: *Fixture, a: std.mem.Allocator) void {
        f.stream.deinit();
        f.bank.deinit();
        f.fd.close();
        f.tmp.cleanup();
        a.free(f.bytes);
    }

    /// Expert `e`'s component `c` of absolute layer `layer` as the stock gather's dense bank row bytes.
    fn segment(f: *const Fixture, layer: u32, e: u32, c: usize) []const u8 {
        const seg = f.geom.segments[c];
        return f.bytes[f.geom.recordOffset(layer, e) + seg.offset ..][0..seg.length];
    }
};

fn next(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

/// Four verify rows' top-8 over 64 experts: rows overlap (the union is smaller than 32), each row's ids distinct.
pub const verify_ids = [4][top_k]u16{
    .{ 3, 9, 17, 22, 40, 41, 5, 60 },
    .{ 9, 3, 18, 22, 44, 41, 6, 61 },
    .{ 3, 10, 17, 23, 40, 45, 5, 62 },
    .{ 11, 9, 19, 22, 46, 41, 7, 63 },
};

fn unionCount(rows: []const [top_k]u16) u32 {
    var seen: [n_experts]bool = @splat(false);
    var n: u32 = 0;
    for (rows) |r| for (r) |e| {
        n += @intFromBool(!seen[e]);
        seen[e] = true;
    };
    return n;
}

test "mimo verify lane: the phase route: one row either phase, M <= 4 in decode the decode lane; the prompt's M >= 2 and M >= 5 the wide lane" {
    try testing.expectEqual(@as(u32, 4), source_mod.decode_lane_rows);
    try testing.expectEqual(@as(u32, 4), prefill_math.Math.decode_forward_rows);
    try testing.expectEqual(@as(u32, 5), prefill_math.Math.prefill_min_tokens);
    try testing.expectEqual(@as(u32, 4), @import("mtp.zig").verify_rows);
    for (1..9) |n| {
        const nt: u32 = @intCast(n);
        try testing.expectEqual(nt == 1, source_mod.decodeLane(nt, false));
        try testing.expectEqual(nt <= 4, source_mod.decodeLane(nt, true));
    }
}

// ── Trace backend over MiMo's real stream source (host slot rows, the lookahead class on) ──

/// MiMo's decode-lane constants and predictor on the trace backend; the banked waves are one kernel each (the fused
/// texts); the wide lane's math emits MiMo's `projectTiles` graph per bank (take, three sorted gathers, the
/// activation's four ops, one drain) so the census can count it.
pub const LaneMath = struct {
    pub const has_banked = true;
    pub const has_parts = true;
    pub const decode_forward_rows = prefill_math.Math.decode_forward_rows;
    pub const prefill_min_tokens = prefill_math.Math.prefill_min_tokens;
    pub const predictor_decode_only = prefill_math.Math.predictor_decode_only;
    pub const predictorScores = prefill_math.Math.predictorScores;
    pub const Profile = LaneProfile;
    const Tr = ops.TraceOps;
    pub fn Projection(comptime A: type) type {
        return quant.GatherQmm.Arrays(A);
    }
    /// MiMo's event projection (the gated route's waits alias the packed words and scales).
    pub const EventProjection = prefill_math.Math.EventProjection;
    hidden: c_int,
    inter: c_int,
    barrier: decode_barrier.Routes(Tr) = decode_barrier.Routes(Tr).control,
    pub fn gateUp(self: *const LaneMath, g: *Tr, x: u32, ids: u32, gate: Projection(u32), up: Projection(u32)) !u32 {
        const n = g.shapeOf(x).d[0];
        const r = try g.reshape(x, &.{ n, 1, self.hidden });
        const gg = try g.gatherQmm(r, gate.w, gate.scales.?, ids, .mxfp4);
        const uu = try g.gatherQmm(r, up.w, up.scales.?, ids, .mxfp4);
        return activation(g, try g.reshape(gg, &.{ n, self.inter }), try g.reshape(uu, &.{ n, self.inter }));
    }
    pub fn down(self: *const LaneMath, g: *Tr, h: u32, ids: u32, d: Projection(u32)) !u32 {
        const n = g.shapeOf(h).d[0];
        return g.reshape(try g.gatherQmm(try g.reshape(h, &.{ n, 1, self.inter }), d.w, d.scales.?, ids, .mxfp4), &.{ n, self.hidden });
    }
    pub fn gateUpBanked(self: *const LaneMath, g: *Tr, x: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
    pub fn downBanked(self: *const LaneMath, g: *Tr, h: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .bfloat16);
    }
    fn activation(g: *Tr, gate: u32, up: u32) !u32 {
        return g.mul(try g.astype(try g.silu(try g.astype(gate, .float32)), .bfloat16), up);
    }
    /// K6 / K7 (decode_barrier.zig): MiMo's decode-barrier seams on the trace backend, through the routes' own code (the
    /// control unless a test binds another route: `barrier`).
    pub fn decodeCommit(self: *const LaneMath, g: *Tr, bar: []const u32, dev: ?u32) !void {
        return self.barrier.commit(g, bar, dev);
    }
    pub fn decodeRead(self: *const LaneMath, g: *Tr, bar: []const u32, indices: u32, sc: ?u32, ids_out: []u16, score_out: []f32, scores: *[]const f32) ![]const u16 {
        return self.barrier.read(g, bar, indices, sc, ids_out, score_out, scores);
    }
    pub fn decodeWave(self: *const LaneMath, g: *Tr, outs: []const u32) !void {
        return self.barrier.wave(g, outs);
    }
    /// MiMo's `projectTiles` (one tile: a verify has <= 32 rows), its drain included.
    fn tile(self: *const LaneMath, g: *Tr, x: u32, rows: quant.PrefillRows, bank: quant.BankArrays(Projection(u32))) !u32 {
        const n: c_int = @intCast(rows.slot.len);
        const act = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(rows.act_row.?), &.{n}, .uint32), 0);
        const rhs = try g.hostArray(std.mem.sliceAsBytes(rows.slot), &.{n}, .uint32);
        // MiMo's sorted gather is [1, n, K] x [n] rows; the trace's gather takes one row per index (the same launches).
        const r = try g.reshape(act, &.{ n, 1, self.hidden });
        const gg = try g.reshape(try g.gatherQmm(r, bank.gate.w, bank.gate.scales.?, rhs, .mxfp4), &.{ n, self.inter });
        const uu = try g.reshape(try g.gatherQmm(r, bank.up.w, bank.up.scales.?, rhs, .mxfp4), &.{ n, self.inter });
        const h = try g.reshape(try activation(g, gg, uu), &.{ n, 1, self.inter });
        const y = try g.reshape(try g.gatherQmm(h, bank.down.w, bank.down.scales.?, rhs, .mxfp4), &.{ n, self.hidden });
        try g.evalAll(&.{y});
        return g.keep(y);
    }
    pub fn prefill(self: *LaneMath, g: *Tr, _: u32, x: u32, rows: quant.PrefillRows, bank: quant.BankArrays(Projection(u32))) !u32 {
        return self.tile(g, x, rows, bank);
    }
    pub fn prefillParts(self: *LaneMath, g: *Tr, _: u32, x: u32, rows: quant.PrefillRows, bank: quant.BankArrays(Projection(u32)), alloc: std.mem.Allocator, outs: *std.ArrayList(u32), pos: *std.ArrayList(u32)) !void {
        try outs.append(alloc, try self.tile(g, x, rows, bank));
        for (0..rows.slot.len) |i| try pos.append(alloc, @intCast(i));
    }
    pub fn finishPrefill(_: *LaneMath, _: *Tr) !void {}
};

/// The scheduler's wide-call counter (`runWideCore` counts each entry); the rest of the profile surface is inert.
pub const LaneProfile = struct {
    pub const enabled = false;
    pub const Mode = @import("../deepseek_v41/prefill_timers.zig").Mode;
    var wide_calls: u32 = 0;
    pub fn now() u64 {
        return 0;
    }
    pub fn charge(_: anytype, _: u64) void {}
    pub fn setMode(_: Mode) void {}
    pub fn countWideCall() void {
        wide_calls += 1;
    }
};

/// Scripted host reads: each barrier's ids (the call's rows' routes), each predictor read's scores.
pub const Script = struct {
    calls: []const []const u16,
    next_call: usize = 0,
    pub fn values(self: *Script) ops.TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax, .f32s = f32s };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *Script = @ptrCast(@alignCast(ctx));
        if (self.next_call >= self.calls.len) return error.ScriptExhausted;
        const c = self.calls[self.next_call];
        self.next_call += 1;
        if (c.len != out.len) return error.ScriptShape;
        @memcpy(out, c);
    }
    fn argmax(_: *anyopaque) anyerror!u32 {
        return error.NoPicks;
    }
    /// The next layer's gate scores: per row, experts 24..31 hottest (the lookahead's candidates).
    fn f32s(_: *anyopaque, out: []f32) anyerror!void {
        for (out, 0..) |*v, i| v.* = if (i % n_experts >= 24 and i % n_experts < 32) 0.9 else 0.01;
    }
};

const Census = struct { dispatches: usize = 0, host_reads: usize = 0, evals: usize = 0, commits: usize = 0 };

/// Device work in the trace between `from` and the end: every node but views, markers and host-array inputs.
fn census(g: *const ops.TraceOps, from: usize, evals_from: usize) Census {
    var c: Census = .{ .evals = g.evals.items.len - evals_from };
    for (g.nodes.items[from..]) |nd| switch (nd.op) {
        .input, .host, .scalar, .reshape, .transpose_axes, .broadcast_to, .expand_dims, .slice, .tape_begin, .tape_end => {},
        .host_read => c.host_reads += 1,
        .async_eval => c.commits += 1,
        .event_wait => {},
        else => c.dispatches += 1,
    };
    return c;
}

pub fn flat(rows: []const [top_k]u16, out: []u16) []const u16 {
    for (rows, 0..) |r, i| @memcpy(out[i * top_k ..][0..top_k], &r);
    return out[0 .. rows.len * top_k];
}

const Arm = enum { hitpass, fused };

fn laneRun(a: std.mem.Allocator, arm: Arm, report: bool) !void {
    const Tr = ops.TraceOps;
    const rows = [_]u32{ 8, 8 };
    const f = try a.create(Fixture);
    defer a.destroy(f);
    try f.init(a, 32, 32, &rows, .host, true);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    var g = Tr.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = shared.ExpertsWith(Tr, StreamSource, LaneMath, .{ .prefill = true, .lookahead = true });
    const c: @import("config.zig").Config = .{ .hidden_size = 32, .moe_intermediate_size = 32, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 32, n_experts }, .float32), .bias = try g.input(&.{n_experts}, .float32) };
    var ex = try Ex.initWith(a, &g, &ss, .{ .hidden = 32, .inter = 32 }, &c, .{ .gates = &gates, .banked = true, .devroute = arm == .hitpass, .wide = .{ .seed = true, .hot_first = true, .depth = stream_mod.wide_depth } });
    defer ex.deinit();

    var buf: [8 * top_k]u16 = undefined;
    var buf5: [5 * top_k]u16 = undefined;
    const prompt = flat(&verify_ids, &buf);
    const five_rows = [5][top_k]u16{ verify_ids[0], verify_ids[1], verify_ids[2], verify_ids[3], .{ 0, 1, 2, 12, 13, 14, 15, 16 } };
    const five = flat(&five_rows, &buf5);
    const one = [_]u16{ 3, 9, 17, 22, 40, 41, 5, 60 };
    var script: Script = .{ .calls = &.{ prompt, &one, prompt, five, prompt } };
    g.host_values = script.values();
    LaneProfile.wide_calls = 0;

    // The prompt's M = 4 chunk (Source.forward's phase route: `routedParts`): the wide lane, as before.
    const x4 = try g.input(&.{ 4, 32 }, .bfloat16);
    const idx4 = try g.input(&.{ 4, 8 }, .int32);
    try testing.expect(!source_mod.decodeLane(4, ss.decoding()));
    _ = try ex.at(1).routedParts(&g, x4, idx4);
    ex.releaseParts(&g);
    try ex.flush();
    try testing.expectEqual(@as(u32, 1), LaneProfile.wide_calls);

    // The phase change, then a one-token tick.
    try ex.grow(&g, &rows);
    try testing.expect(ss.decoding());
    const x1 = try g.input(&.{ 1, 32 }, .bfloat16);
    const idx1 = try g.input(&.{ 1, 8 }, .int32);
    const n0 = g.nodes.items.len;
    const e0 = g.evals.items.len;
    const st_a = ss.stats();
    _ = try ex.at(1).routed(&g, x1, idx1);
    const c1 = census(&g, n0, e0);
    const st_b = ss.stats();
    try ex.flush();

    // The round's verify: M = 4 in the decode phase. `Source.forward` sends it to `routed` (the decode lane).
    try testing.expect(source_mod.decodeLane(4, ss.decoding()));
    var resident: [n_experts]bool = undefined;
    for (&resident, 0..) |*r, e| r.* = ss.isResident(1, @intCast(e));
    var want_misses: u32 = 0;
    var seen: [n_experts]bool = @splat(false);
    for (verify_ids) |r| for (r) |e| {
        if (!seen[e] and !resident[e]) want_misses += 1;
        seen[e] = true;
    };
    const st0 = ss.stats();
    const n1 = g.nodes.items.len;
    const e1 = g.evals.items.len;
    const p0 = if (arm == .hitpass) ex.dev_parity[1] else 0;
    const out = try ex.at(1).routed(&g, x4, idx4);
    const c4 = census(&g, n1, e1);
    const st1 = ss.stats();
    try ex.flush();
    try testing.expectEqualSlices(c_int, &.{ 4, 8, 32 }, g.shapeOf(out).slice());
    // Never the wide lane (no `runWideCore`, so no `seedPrefill`: the NotPrefill class cannot be reached).
    try testing.expectEqual(@as(u32, 1), LaneProfile.wide_calls);
    // One route for the four rows; each of the union's misses loaded (read) once.
    try testing.expectEqual(@as(u64, 1), st1.route_calls - st0.route_calls);
    try testing.expectEqual(@as(u64, unionCount(&verify_ids)), (st1.expert_cache_hits - st0.expert_cache_hits) + (st1.expert_cache_misses - st0.expert_cache_misses));
    try testing.expectEqual(@as(u64, want_misses), st1.expert_cache_misses - st0.expert_cache_misses);
    try testing.expectEqual(@as(u64, want_misses), (st1.persistent_loads - st0.persistent_loads) + (st1.transient_loads - st0.transient_loads));
    // The lookahead class stays on: the route's certain misses pre-read, and the four rows' next-layer scores read
    // (one host read [4, 64] f32 beside the ids') and handed to the route (the speculation's input).
    try testing.expect(st1.pre_issued > st0.pre_issued);
    try testing.expectEqual(@as(usize, 2), c4.host_reads);
    if (arm == .hitpass) {
        // The hit pass: one device wave over the 32 pairs; the layer's LUT rewritten after the route.
        try testing.expect(ex.dev_parity[1] != p0);
        var lut: [n_experts]u32 = undefined;
        ss.residentLut(1, &lut);
        try testing.expectEqualSlices(u32, &lut, std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[1][ex.dev_parity[1]])))));
    }

    // A decode-phase M = 5 call (no decode caller makes one) still takes the wide lane.
    const x5 = try g.input(&.{ 5, 32 }, .bfloat16);
    const idx5 = try g.input(&.{ 5, 8 }, .int32);
    const n5 = g.nodes.items.len;
    const e5 = g.evals.items.len;
    _ = try ex.at(1).routed(&g, x5, idx5);
    const c5 = census(&g, n5, e5);
    try ex.flush();
    try testing.expectEqual(@as(u32, 2), LaneProfile.wide_calls);

    // The route before this change: the verify's decode-phase M = 4 through the wide lane (`routedParts`, as
    // `Source.forward` sent every M >= 2 call), for the census.
    const nb = g.nodes.items.len;
    const eb = g.evals.items.len;
    _ = try ex.at(1).routedParts(&g, x4, idx4);
    const cb = census(&g, nb, eb);
    ex.releaseParts(&g);
    try ex.flush();
    try testing.expectEqual(@as(u32, 3), LaneProfile.wide_calls);

    if (report) {
        std.debug.print("MIMO_VERIFY_CENSUS arm={s} lane=wide(before) M=4 dispatches={d} host_reads={d} evals={d} commits={d} (scheduler part only; Source's joinless combine adds its own)\n", .{ @tagName(arm), cb.dispatches, cb.host_reads, cb.evals, cb.commits });
        std.debug.print("MIMO_VERIFY_CENSUS arm={s} lane=decode M=1 dispatches={d} host_reads={d} evals={d} commits={d} misses={d}\n", .{ @tagName(arm), c1.dispatches, c1.host_reads, c1.evals, c1.commits, st_b.expert_cache_misses - st_a.expert_cache_misses });
        std.debug.print("MIMO_VERIFY_CENSUS arm={s} lane=decode M=4 dispatches={d} host_reads={d} evals={d} commits={d} misses={d} union={d}\n", .{ @tagName(arm), c4.dispatches, c4.host_reads, c4.evals, c4.commits, want_misses, unionCount(&verify_ids) });
        std.debug.print("MIMO_VERIFY_CENSUS arm={s} lane=wide(before) M=5 dispatches={d} host_reads={d} evals={d} commits={d} (scheduler part only; Source's joinless combine adds its own)\n", .{ @tagName(arm), c5.dispatches, c5.host_reads, c5.evals, c5.commits });
    }
}

test "mimo verify lane: a decode-phase M=4 call takes the decode lane on the host-memory stream (never runWideCore / seedPrefill): one route, each union miss read once, the lookahead on, the hit pass's LUT rewritten; the prompt's M=4 and a decode M=5 keep the wide lane" {
    try laneRun(testing.allocator, .hitpass, true);
    try laneRun(testing.allocator, .fused, true);
}

// ── mimo/layer-rows: a per-layer rows split (settings `expert_rows_by_layer`) on the trace backend ──

test "mimo layer rows (trace backend): each routed layer plans at its own rows through the prompt, the phase change and 12 verify rounds; each union miss read once, residency never past the layer's rows" {
    const a = testing.allocator;
    const Tr = ops.TraceOps;
    // A split of the uniform 8 / 11 rows (prompt / decode): offsets -3, +3 (sum 0), the same growth (3) on both layers.
    const prompt_rows = [_]u32{ 5, 11 };
    const decode_rows = [_]u32{ 8, 14 };
    const f = try a.create(Fixture);
    defer a.destroy(f);
    try f.init(a, 32, 32, &prompt_rows, .host, true);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    var g = Tr.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = shared.ExpertsWith(Tr, StreamSource, LaneMath, .{ .prefill = true, .lookahead = true });
    const c: @import("config.zig").Config = .{ .hidden_size = 32, .moe_intermediate_size = 32, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 32, n_experts }, .float32), .bias = try g.input(&.{n_experts}, .float32) };
    var ex = try Ex.initWith(a, &g, &ss, .{ .hidden = 32, .inter = 32 }, &c, .{ .gates = &gates, .banked = true, .devroute = true, .wide = .{ .seed = true, .hot_first = true, .depth = stream_mod.wide_depth } });
    defer ex.deinit();
    // The script: each layer's prompt chunk (M = 4, the wide lane), then per round layer 1's and layer 2's verify
    // (M = 4): the four rows' ids shifted by 5 x round (mod 64: still distinct per row), so every round brings misses.
    const rounds = 12;
    var bufs: [2 + 2 * rounds][4][top_k]u16 = undefined;
    var flats: [2 + 2 * rounds][4 * top_k]u16 = undefined;
    var calls: [2 + 2 * rounds][]const u16 = undefined;
    for (&bufs, &flats, &calls, 0..) |*b, *fl, *cl, i| {
        const shift: u16 = if (i < 2) 0 else @intCast(5 * ((i - 2) / 2));
        for (b, verify_ids) |*row, src| for (row, src) |*e, id| {
            e.* = (id + shift) % @as(u16, n_experts);
        };
        cl.* = flat(b, fl);
    }
    var script: Script = .{ .calls = &calls };
    g.host_values = script.values();
    const x4 = try g.input(&.{ 4, 32 }, .bfloat16);
    const idx4 = try g.input(&.{ 4, 8 }, .int32);
    // The prompt: each layer seeds and plans at its own prompt rows.
    for (1..3) |layer| {
        _ = try ex.at(@intCast(layer)).routedParts(&g, x4, idx4);
        ex.releaseParts(&g);
        try ex.flush();
        const l: u32 = @intCast(layer);
        try testing.expectEqual(prompt_rows[layer - 1], ss.slotCapacity(l));
        try testing.expect(ss.policy(l).occupancy <= prompt_rows[layer - 1]);
    }
    // The phase change: each layer grows by the same 3 rows to its own decode rows.
    try ex.grow(&g, &decode_rows);
    try testing.expect(ss.decoding());
    var total_misses: u64 = 0;
    for (0..rounds) |_| for (1..3) |layer| {
        const l: u32 = @intCast(layer);
        const ids = calls[script.next_call];
        var want: u64 = 0;
        var seen: [n_experts]bool = @splat(false);
        for (ids) |e| {
            if (!seen[e] and !ss.isResident(l, e)) want += 1;
            seen[e] = true;
        }
        const st0 = ss.stats();
        _ = try ex.at(l).routed(&g, x4, idx4);
        try ex.flush();
        const st1 = ss.stats();
        try testing.expectEqual(want, st1.expert_cache_misses - st0.expert_cache_misses);
        try testing.expectEqual(want, (st1.persistent_loads - st0.persistent_loads) + (st1.transient_loads - st0.transient_loads));
        try testing.expectEqual(decode_rows[layer - 1], ss.slotCapacity(l));
        try testing.expect(ss.policy(l).occupancy <= decode_rows[layer - 1]);
        total_misses += want;
    };
    try testing.expectEqual(script.calls.len, script.next_call);
    try testing.expect(total_misses > 0);
    // Both layers filled to their own rows (the small layer cannot hold what the large one holds).
    try testing.expectEqual(decode_rows[0], ss.policy(1).occupancy);
    try testing.expectEqual(decode_rows[1], ss.policy(2).occupancy);
}

// ── MLX CPU: the decode lane's stock math (grouped gather_qmm over the slot banks) on the real stream ──

/// The dense reference bank of `layer` (every expert's rows, the stock gather's [E, out, in/8] / [E, out, in/32]).
fn denseBank(a: std.mem.Allocator, g: *ops.MlxOps, f: *const Fixture, layer: u32) !bank_mod.BankArrays {
    var parts: [bank_mod.n_components]mlx.mlx_array = undefined;
    for (&parts, f.geom.segments, 0..) |*p, seg, ci| {
        const bytes = try a.alloc(u8, n_experts * seg.length);
        defer a.free(bytes);
        for (0..n_experts) |e| @memcpy(bytes[e * seg.length ..][0..seg.length], f.segment(layer, @intCast(e), ci));
        const x = try g.hostArray(bytes, &.{ @intCast(n_experts), @intCast(seg.shape[0]), @intCast(seg.shape[1]) }, bank_mod.mlxDtype(seg.dtype));
        try g.evalAll(&.{x});
        p.* = g.keep(x);
    }
    return bank_mod.bankArraysOf(parts);
}

fn hostBits(a: std.mem.Allocator, g: *ops.MlxOps, x: mlx.mlx_array) ![]u32 {
    const n: usize = @intCast(g.shapeOf(x).numel());
    const out = try a.alloc(f32, n);
    defer a.free(out);
    _ = try g.hostF32(try g.astype(x, .float32), out);
    const bits = try a.alloc(u32, n);
    for (out, bits) |v, *b| b.* = @bitCast(v);
    return bits;
}

test "mimo verify lane CPU: the decode lane's M = 2, 3, 4 rows are bit-equal to M = 1 per row and to the stock gather_qmm per pair; the combine per row" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const H: u32 = 64;
    const I: u32 = 64;
    const rows = [_]u32{ 8, 8 };
    const f = try a.create(Fixture);
    defer a.destroy(f);
    try f.init(a, H, I, &rows, .{ .mlx = s }, false);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    const Ex = shared.ExpertsWith(ops.MlxOps, StreamSource, prefill_math.Math, .{ .prefill = true });
    const c: @import("config.zig").Config = .{ .hidden_size = H, .moe_intermediate_size = I, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    const math = try prefill_math.Math.init(a, f.geom, source_mod.expertActivation);
    var ex = try Ex.initWith(a, &g, &ss, math, &c, .{});
    defer {
        ex.math.deinit(&g);
        ex.deinit();
    }
    try ex.grow(&g, &rows);
    try testing.expect(ss.decoding());
    const dense = try denseBank(a, &g, f, 1);
    defer for ([_]quant.GatherQmm.Arrays(mlx.mlx_array){ dense.gate, dense.up, dense.down }) |p| {
        g.release(p.w);
        g.release(p.scales.?);
    };
    g.reset();

    // Inputs: four rows of x (bf16), their ids (int32) and router weights (f32, normalized per row).
    var state: u32 = 0x7e51_f00d;
    var xin: [4 * H]f32 = undefined;
    for (&xin) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(next(&state) % 2001)) - 1000)) / 4000;
    var idsin: [4 * top_k]i32 = undefined;
    for (verify_ids, 0..) |r, t| for (r, 0..) |e, j| {
        idsin[t * top_k + j] = e;
    };
    var win: [4 * top_k]f32 = undefined;
    for (0..4) |t| {
        var sum: f32 = 0;
        for (win[t * top_k ..][0..top_k]) |*v| {
            v.* = moe.sigmoidRef(@as(f32, @floatFromInt(@as(i32, @intCast(next(&state) % 2001)) - 1000)) / 250);
            sum += v.*;
        }
        for (win[t * top_k ..][0..top_k]) |*v| v.* /= sum + 1e-20;
    }
    const Inputs = struct {
        fn at(gg: *ops.MlxOps, xs: []const f32, ids: []const i32, ws: []const f32, r0: usize, m: usize) ![3]mlx.mlx_array {
            const h: usize = xs.len / 4;
            const x = try gg.astype(try gg.hostArray(std.mem.sliceAsBytes(xs[r0 * h ..][0 .. m * h]), &.{ @intCast(m), @intCast(h) }, .float32), .bfloat16);
            const id = try gg.hostArray(std.mem.sliceAsBytes(ids[r0 * top_k ..][0 .. m * top_k]), &.{ @intCast(m), top_k }, .int32);
            const w = try gg.hostArray(std.mem.sliceAsBytes(ws[r0 * top_k ..][0 .. m * top_k]), &.{ @intCast(m), top_k }, .float32);
            return .{ x, id, w };
        }
    };
    const kh = top_k * H;
    // M = 1, each row alone (the serial path): its routed rows and its combine.
    var one_rows: [4][]u32 = undefined;
    var one_comb: [4][]u32 = undefined;
    for (0..4) |r| {
        const in = try Inputs.at(&g, &xin, &idsin, &win, r, 1);
        const routed = try ex.at(1).routed(&g, in[0], in[1]);
        const comb = try source_mod.combineStock(undefined, &g, routed, in[2]);
        try g.evalAll(&.{ routed, comb });
        try ex.flush();
        one_rows[r] = try hostBits(a, &g, routed);
        one_comb[r] = try hostBits(a, &g, comb);
        g.reset();
    }
    defer for (one_rows, one_comb) |x, y| {
        a.free(x);
        a.free(y);
    };
    // The stock gather_qmm per pair on the dense bank (the reference the slot banks must reproduce).
    for (0..4) |r| for (0..top_k) |j| {
        const e: u32 = verify_ids[r][j];
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin[r * H ..][0..H]), &.{ 1, 1, @intCast(H) }, .float32), .bfloat16);
        const idx = try g.hostArray(std.mem.sliceAsBytes(&[_]u32{e}), &.{1}, .uint32);
        const gg = try g.reshape(try g.gatherQmm(x, dense.gate.w, dense.gate.scales.?, idx, .mxfp4), &.{ 1, @intCast(I) });
        const uu = try g.reshape(try g.gatherQmm(x, dense.up.w, dense.up.scales.?, idx, .mxfp4), &.{ 1, @intCast(I) });
        const h = try source_mod.expertActivation(&g, gg, uu);
        const y = try g.gatherQmm(try g.reshape(h, &.{ 1, 1, @intCast(I) }), dense.down.w, dense.down.scales.?, idx, .mxfp4);
        const bits = try hostBits(a, &g, y);
        defer a.free(bits);
        try testing.expectEqualSlices(u32, bits, one_rows[r][j * H ..][0..H]);
        g.reset();
    };
    // M = 2, 3, 4 in one decode-lane call: every row bit-equal to its M = 1 call (row independence).
    for ([_]usize{ 2, 3, 4 }) |m| {
        const st0 = ss.stats();
        const in = try Inputs.at(&g, &xin, &idsin, &win, 0, m);
        const routed = try ex.at(1).routed(&g, in[0], in[1]);
        const comb = try source_mod.combineStock(undefined, &g, routed, in[2]);
        try g.evalAll(&.{ routed, comb });
        try ex.flush();
        const st1 = ss.stats();
        try testing.expectEqual(@as(u64, 1), st1.route_calls - st0.route_calls);
        try testing.expectEqual(@as(u64, unionCount(verify_ids[0..m])), (st1.expert_cache_hits - st0.expert_cache_hits) + (st1.expert_cache_misses - st0.expert_cache_misses));
        const got = try hostBits(a, &g, routed);
        defer a.free(got);
        const got_c = try hostBits(a, &g, comb);
        defer a.free(got_c);
        for (0..m) |r| {
            try testing.expectEqualSlices(u32, one_rows[r], got[r * kh ..][0..kh]);
            try testing.expectEqualSlices(u32, one_comb[r], got_c[r * H ..][0..H]);
        }
        std.debug.print("MIMO_VERIFY_CPU M={d} pairs={d} union={d} misses={d} rows_bit_equal_to_M1=true combine_bit_equal_to_M1=true\n", .{ m, m * top_k, unionCount(verify_ids[0..m]), st1.expert_cache_misses - st0.expert_cache_misses });
        g.reset();
    }
    // The fused combine text's host reference (`moe.combineRef`) is row-independent at n = 1..4 (bit for bit).
    var rows_f: [4 * top_k * H]f32 = undefined;
    for (&rows_f) |*v| v.* = moe.bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast(next(&state) % 2001)) - 1000)) / 512);
    var all: [4 * H]f32 = undefined;
    moe.combineRef(&rows_f, &win, 4, H, &all);
    for (1..5) |m| {
        var part: [4 * H]f32 = undefined;
        moe.combineRef(rows_f[0 .. m * top_k * H], win[0 .. m * top_k], m, H, part[0 .. m * H]);
        try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(all[0 .. m * H])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(part[0 .. m * H])));
    }
}
