//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages; `Options.slot_memory` picks one). `Stream`: per-layer slot pools,
//! routes, deferred release, growth, lookahead and event gates.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
/// PROFILE builds only (`-Ddsv41-prefill-timers=true`): P1's read-ahead record; every call compiles to nothing otherwise.
const bo = @import("build_flags.zig");
const mlx = @import("sdk").mlx;
const expert_bank = @import("expert_bank.zig");
const expert_io = sdk_ext.expert.io;
const expert_policy = @import("sdk_ext.zig").expert.policy;
const expert_lookahead = @import("expert_lookahead.zig");
const exl3_quant = @import("exl3_quant.zig");

const S = sdk_ext.expert.stream.StreamOf(expert_bank, if (@hasDecl(bo, "dsv41_prefill_timers")) bo.dsv41_prefill_timers else false);
const n_components = S.n_components;
const gu_components = S.gu_components;
const Component = S.Component;
const Layer = S.Layer;
const LayerPolicy = S.LayerPolicy;
const Phase = S.Phase;
const Plan = S.Plan;
const max_route_ids = S.max_route_ids;
pub const HostSlotRows = S.HostSlotRows;
pub const LayerSlotBank = S.LayerSlotBank;
const mlxActive = S.mlxActive;
const evalArrays = S.evalArrays;
pub const SlotMemory = S.SlotMemory;
const evalRows = S.evalRows;
const Rows = S.Rows;
pub const source_caps = S.source_caps;
pub const uses_reader = S.uses_reader;
pub const BankKind = S.BankKind;
pub const SlotRef = S.SlotRef;
pub const Options = S.Options;
pub const read_ahead_probed = S.read_ahead_probed;
pub const ReadAheadProbe = S.ReadAheadProbe;
pub const GrowFill = S.GrowFill;
pub const DecodePool = S.DecodePool;
pub const PoolCand = S.PoolCand;
pub const poolRows = S.poolRows;
const ProbeSlot = S.ProbeSlot;
const no_probe = S.no_probe;
pub const FirstVerifyWarm = S.FirstVerifyWarm;
pub const Lookahead = S.Lookahead;
pub const Event = S.Event;
pub const Gates = S.Gates;
pub const Stats = S.Stats;
pub const Error = S.Error;
const SlotState = S.SlotState;
const SlotMeta = S.SlotMeta;
pub const Part = S.Part;
pub const Route = S.Route;
const route_capacity = S.route_capacity;
pub const max_wide_depth = S.max_wide_depth;
pub const phase_change_releases_wide_windows = S.phase_change_releases_wide_windows;
pub const transient_release_default = S.transient_release_default;
pub const decode_staging_rows = S.decode_staging_rows;
const wait_timeout_ns = S.wait_timeout_ns;
pub const Stream = S.Stream;
/// The EXL3 bank's slot arrays by projection.
pub const ProjArrays = expert_bank.ProjArrays;
pub const BankArrays = expert_bank.BankArrays;

/// A bank's arrays by projection, as the quant binds them (`sdk_ext.quant.BankArrays` of the EXL3 quant's `Arrays`).
pub fn BankArraysOf(comptime T: type) type {
    return sdk_ext.quant.BankArrays(exl3_quant.Arrays(T));
}

/// Per routed id of `r`, its wave: 0 for a slot resident at the call (a hit),
/// p + 1 for a slot part p loads.
pub fn wavesOf(plan: *const Plan, hit_slots: []const u32, part_loads: []const []const expert_policy.Load, out: []u8) void {
    var keys: [max_route_ids]u32 = undefined;
    var vals: [max_route_ids]u8 = undefined;
    var n: usize = 0;
    for (hit_slots) |s| {
        keys[n] = s;
        vals[n] = 0;
        n += 1;
    }
    for (part_loads, 1..) |loads, w| for (loads) |l| {
        keys[n] = l.slot;
        vals[n] = @intCast(w);
        n += 1;
    };
    for (plan.slotsOf(), out) |s, *w| w.* = vals[std.mem.indexOfScalar(u32, keys[0..n], s).?];
}

/// Trace arrays in one record's geometry, `rows` rows (a trace backend's `input`).
pub fn traceBank(g: anytype, geom: *const Layer, rows: u32) !BankArraysOf(@TypeOf(g.*).T) {
    var a: [n_components]@TypeOf(g.*).T = undefined;
    for (&a, geom.segments) |*x, seg| {
        var shape: [4]c_int = undefined;
        shape[0] = @intCast(rows);
        for (seg.shape[0..seg.rank], 1..) |d, i| shape[i] = @intCast(d);
        x.* = try g.input(shape[0 .. seg.rank + 1], switch (seg.dtype) {
            .I16 => .int16,
            .F16 => .float16,
        });
    }
    return .{
        .gate = .{ .code = a[0], .rout = a[1], .rin = a[2] },
        .up = .{ .code = a[3], .rout = a[4], .rin = a[5] },
        .down = .{ .code = a[6], .rout = a[7], .rin = a[8] },
    };
}

/// DEVROUTE's resident map of one layer from its policy: each resident expert's persistent slot as its packed
/// (bank << 24 | row) through `src.slotRef`; non-resident experts 0.
pub fn lutOf(pol: *const expert_policy.LayerPolicy, src: anytype, layer: u32, out: []u32) void {
    for (out, 0..) |*o, e| {
        o.* = 0;
        if (e >= pol.n_experts) continue;
        const s = pol.slotOf(@intCast(e)) orelse continue;
        if (s >= pol.capacity) continue;
        const ref = src.slotRef(layer, s);
        o.* = (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
    }
}

// ── StreamSource: the stream as an expert source (the EXL3 source's side of the contract) ──

/// The streamer as an expert source. Routes, waits, release, flush and growth
/// are the Stream's own; the call view comes from `refsOf` and the parts'
/// loads (`Route.partLoads`); the MLX arrays from `bankArrays` (slot memory
/// `.mlx`: with host rows the MLX executor refuses at construction).
pub const StreamSource = struct {
    stream: *Stream,
    calls: [n_routes]Call = @splat(.{}),

    /// What the stream supports; an arm installs a subset at construction.
    pub const caps = source_caps;
    /// The slot arrays it fills are the EXL3 quant's (`sdk_ext.expert`: a source's arrays are its quant's).
    pub const Arrays = exl3_quant.Arrays;

    /// One call per live route of the Stream's ring.
    pub const n_routes = @typeInfo(@FieldType(Stream, "routes")).array.len;

    pub const Call = struct {
        route: ?*Route = null,
        n_ids: u32 = 0,
        refs: [max_route_ids]SlotRef = undefined,
        waves: [max_route_ids]u8 = undefined,
    };

    pub fn init(stream: *Stream) StreamSource {
        return .{ .stream = stream };
    }

    /// Calls mirror the Stream's route ring one to one.
    fn callOf(self: *StreamSource, r: *Route) *Call {
        const i = (@intFromPtr(r) - @intFromPtr(&self.stream.routes[0])) / @sizeOf(Route);
        return &self.calls[i];
    }

    pub fn route(self: *StreamSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
        const r = try self.stream.route(layer, ids, scores);
        const call = self.callOf(r);
        call.* = .{ .route = r, .n_ids = @intCast(ids.len) };
        _ = self.stream.refsOf(r, &call.refs);
        var bufs: [max_route_ids][max_route_ids]expert_policy.Load = undefined;
        var parts: [max_route_ids][]const expert_policy.Load = undefined;
        for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
        wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], call.waves[0..ids.len]);
        return call;
    }

    pub fn served(_: *StreamSource, call: *const Call) sdk_ext.expert.Served {
        return .{ .refs = call.refs[0..call.n_ids], .waves = call.waves[0..call.n_ids], .n_parts = call.route.?.n_parts };
    }

    /// DEVROUTE: layer `layer`'s resident map as the device reads it: `out[e]` = the packed (bank << 24 | row) of
    /// expert e's persistent slot, else 0 (a miss: the device row is computed from base row 0 and never joined).
    pub fn residentLut(self: *const StreamSource, layer: u32, out: []u32) void {
        lutOf(&self.stream.layers[layer].policy, self.stream, layer, out);
    }

    /// The decode phase (the grown banks; DEVROUTE's LUTs exist).
    pub fn decoding(self: *const StreamSource) bool {
        return self.stream.phase == .decode;
    }

    pub fn waitGu(self: *StreamSource, call: *Call, part: u32) Error!void {
        return self.stream.waitGu(call.route.?, part);
    }

    pub fn waitDown(self: *StreamSource, call: *Call, part: u32) Error!void {
        return self.stream.waitDown(call.route.?, part);
    }

    pub fn release(self: *StreamSource, call: *Call) void {
        self.stream.release(call.route.?);
    }

    pub fn flush(self: *StreamSource) Error!void {
        return self.stream.flush();
    }

    pub fn grow(self: *StreamSource, decode_rows: []const u32) !void {
        return self.stream.grow(decode_rows);
    }

    /// The phase change's first free (`Stream.releaseTransient`): the bytes freed.
    pub fn releaseTransient(self: *StreamSource) !u64 {
        return self.stream.releaseTransient();
    }

    /// The reverse phase change's free (`Stream.shrink`): the bytes freed.
    pub fn shrink(self: *StreamSource, prompt_rows: []const u32) !u64 {
        return self.stream.shrink(prompt_rows);
    }

    /// The reverse phase change's allocation (`Stream.regrowTransient`): the bytes allocated.
    pub fn regrowTransient(self: *StreamSource) !u64 {
        return self.stream.regrowTransient();
    }

    pub fn seedPrefill(self: *StreamSource, layer: u32, ids: []const u16) !void {
        return self.stream.seedPrefill(layer, ids);
    }

    /// The last seedPrefill's seed ranks (`LayerPolicy.seed_ranks`).
    pub fn seedRanks(self: *const StreamSource, layer: u32) u32 {
        return self.stream.seedRanks(layer);
    }

    /// P1: the layer's predicted seed read ahead of its routes (`Stream.readAheadSeed`).
    pub fn readAheadSeed(self: *StreamSource, layer: u32, experts: []const u16) !void {
        return self.stream.readAheadSeed(layer, experts);
    }

    /// P1: the layer's read-ahead landed (`Stream.awaitReadAhead`).
    pub fn awaitReadAhead(self: *StreamSource, layer: u32) Error!void {
        return self.stream.awaitReadAhead(layer);
    }

    /// P1's construction self-check over `n` experts of `layer` not resident there (the highest ids):
    /// read ahead == demand read, bit for bit (`Stream.checkReadAhead`). Returns how many were checked.
    pub fn checkReadAhead(self: *StreamSource, layer: u32, n: u32) !u32 {
        const policy = &self.stream.layers[layer].policy;
        var experts: [max_route_ids]u16 = undefined;
        var k: u32 = 0;
        var e: u32 = policy.n_experts;
        while (e > 0 and k < @min(n, max_route_ids)) {
            e -= 1;
            if (policy.slotOf(@intCast(e)) != null) continue;
            experts[k] = @intCast(e);
            k += 1;
        }
        try self.stream.checkReadAhead(layer, experts[0..k]);
        return k;
    }

    /// Whether `expert` holds one of `layer`'s persistent slots now (the profile builds' residency query, before a route).
    pub fn isResident(self: *const StreamSource, layer: u32, expert: u16) bool {
        return self.stream.layers[layer].policy.slotOf(expert) != null;
    }

    /// The pool's demand read gauge now: wall ns with a demand (or pre-) read in flight (A0's per-layer split).
    pub fn readWallNs(self: *StreamSource) u64 {
        return @intCast(@max(self.stream.pool.readGauge()[4], 0));
    }

    /// A live prefill call's persistent slots kept pinned and held past its release (`Wide.defer_base`).
    pub fn holdBase(self: *StreamSource, call: *Call) !void {
        return self.stream.holdBase(call.route.?);
    }

    pub fn releaseHeld(self: *StreamSource) void {
        self.stream.releaseHeld();
    }

    /// Prefill routes one layer may hold live at once (`Stream.Options.wide_depth`).
    pub fn wideDepth(self: *const StreamSource) u8 {
        return self.stream.wide_depth;
    }

    /// A0 (a) (profile builds read it): the ns `layer`'s first decode route waited for its started warm jobs.
    pub fn warmWaitNs(self: *const StreamSource, layer: u32) u64 {
        return self.stream.warmWaitNs(layer);
    }

    pub fn stats(self: *StreamSource) Stats {
        return self.stream.stats();
    }

    /// Event gates of the call's reads (a stream built with `event`).
    pub fn gate(self: *StreamSource, call: *Call) Error!?Gates {
        return self.stream.gate(call.route.?);
    }

    /// The end of a decode cycle (option (b)'s clock; `Stream.cycleEnd`).
    pub fn cycleEnd(self: *StreamSource) !void {
        return self.stream.cycleEnd();
    }

    pub fn bankRows(self: *StreamSource, layer: u32, kind: BankKind) u32 {
        const ls = &self.stream.layers[layer];
        return switch (kind) {
            .base => ls.base.rows,
            .ext => if (ls.ext) |e| e.rows else 0,
            .transient => self.stream.transient.rows,
        };
    }

    /// MLX backends: the stream's own arrays (never freed here); null for host rows or a bank without rows.
    /// Trace backends: inputs in the bank's geometry.
    pub fn bankArrays(self: *StreamSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
        const G = @TypeOf(g.*);
        if (G.T == mlx.mlx_array) {
            const b = self.stream.bankArrays(layer, kind) orelse return null;
            return .{
                .gate = .{ .code = b.gate.code, .rout = b.gate.rout, .rin = b.gate.rin },
                .up = .{ .code = b.up.code, .rout = b.up.rout, .rin = b.up.rin },
                .down = .{ .code = b.down.code, .rout = b.down.rout, .rin = b.down.rin },
            };
        } else {
            if (comptime !@hasDecl(G, "input")) @compileError("StreamSource binds MLX arrays or a trace backend's inputs; " ++ @typeName(G) ++ " has neither");
            const rows = self.bankRows(layer, kind);
            if (rows == 0) return null;
            return try traceBank(g, &self.stream.bank.layers[layer], rows);
        }
    }
};

// ── Tests ──

const testing = std.testing;

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

/// Only creating a Metal device maps a GPU driver bundle (AGXMetal*), so its
/// absence proves this process did no Metal work.
fn metalDriverLoaded() bool {
    for (0.._dyld_image_count()) |i| {
        const name = _dyld_get_image_name(@intCast(i)) orelse continue;
        if (std.mem.indexOf(u8, std.mem.span(name), "AGXMetal") != null) return true;
    }
    return false;
}

const FixSeg = struct { component: []const u8, offset: u64, length: u64, sha256: []const u8, head16: []const u8 };
const FixRec = struct {
    layer: u32,
    expert: u32,
    sidecar_offset: u64,
    record_bytes: u64,
    logical_bytes: u64,
    v2_sha256: []const u8,
    v1_sha256: []const u8,
    segments: []const FixSeg,
};
const Fixture = struct { layer_set: []const FixRec, pick_set: []const FixRec };

const Env = struct {
    bank: expert_bank.Bank,
    text: []u8,
    parsed: std.json.Parsed(Fixture),

    /// DSV41_BANK + DSV41_PHASE0_FIXTURE, else the test is skipped.
    fn open() !Env {
        const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
        const fixture = std.mem.span(std.c.getenv("DSV41_PHASE0_FIXTURE") orelse return error.SkipZigTest);
        var diag: expert_bank.Diag = .{};
        var bank = expert_bank.Bank.open(testing.allocator, std.testing.io, dir, expert_bank.dsv41, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        errdefer bank.deinit();
        const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, fixture, testing.allocator, .limited(4 << 20));
        errdefer testing.allocator.free(text);
        const parsed = try std.json.parseFromSlice(Fixture, testing.allocator, text, .{ .ignore_unknown_fields = true });
        return .{ .bank = bank, .text = text, .parsed = parsed };
    }

    fn close(self: *Env) void {
        self.parsed.deinit();
        testing.allocator.free(self.text);
        self.bank.deinit();
    }
};

fn hexEq(hex: []const u8, bytes: []const u8) bool {
    var buf: [64]u8 = undefined;
    if (hex.len != 2 * bytes.len or hex.len > 2 * buf.len) return false;
    const got = std.fmt.hexToBytes(&buf, hex) catch return false;
    return std.mem.eql(u8, got, bytes);
}

/// Reads `set` (one job, <= 8 records of one geometry) through the pool into
/// `rows`, then checks every record against a direct pread of the full record,
/// both manifest digests and the Python reader's per-component sha256.
fn checkSet(name: []const u8, bank: *const expert_bank.Bank, set: []const FixRec, rows: anytype) !u64 {
    const n = set.len;
    var refs: [expert_io.max_items]expert_bank.RecordRef = undefined;
    var dests: [expert_io.max_items][n_components]u64 = undefined;
    for (set, 0..) |r, i| {
        try testing.expectEqual(bank.recordOffset(r.layer, r.expert), r.sidecar_offset);
        refs[i] = .{ .layer = r.layer, .expert = r.expert };
        dests[i] = rows.rowDest(@intCast(i));
    }
    // Stops (drains + joins) before the caller frees `rows`.
    var pool = try expert_io.Pool.start(testing.allocator, .{});
    defer pool.stop();
    const first = try expert_bank.submitRecords(pool, bank, refs[0..n], dests[0..n]);
    try pool.wait(first, @intCast(2 * n), 60 * std.time.ns_per_s);
    var calls: i64 = 0;
    var payload: i64 = 0;
    var returned: i64 = 0;
    var t0: i64 = std.math.maxInt(i64);
    var t1: i64 = 0;
    for (0..2 * n) |t| {
        const res = pool.result(first + @as(u32, @intCast(t)));
        try testing.expectEqual(expert_io.Status.ok, res.status);
        calls += res.preadv_calls;
        payload += res.payload;
        returned += res.bytes_returned;
        t0 = @min(t0, res.t_start_ns);
        t1 = @max(t1, res.t_end_ns);
    }
    std.debug.print("{s}: {d} records, pool payload {d} B in {d} preadv ({d} B returned) over {d} us; verify pread {d} B\n", .{ name, n, payload, calls, returned, @divTrunc(t1 - t0, 1000), n * set[0].record_bytes });

    const Sha256 = std.crypto.hash.sha2.Sha256;
    const whole = try testing.allocator.alloc(u8, @intCast(set[0].record_bytes));
    defer testing.allocator.free(whole);
    for (set, 0..) |r, i| {
        const layer = &bank.layers[r.layer];
        try testing.expectEqual(layer.record_bytes, r.record_bytes);
        var got: usize = 0;
        while (got < whole.len) {
            const k = std.c.pread(bank.sidecar.fd, whole[got..].ptr, whole.len - got, @intCast(r.sidecar_offset + got));
            if (k <= 0) return error.ShortRead;
            got += @intCast(k);
        }
        var d: [32]u8 = undefined;
        Sha256.hash(whole, &d, .{});
        try testing.expectEqualSlices(u8, &bank.digest(r.layer, r.expert).padded, &d);
        try testing.expect(hexEq(r.v2_sha256, &d));
        Sha256.hash(whole[0..@intCast(layer.logical_bytes)], &d, .{});
        try testing.expectEqualSlices(u8, &bank.digest(r.layer, r.expert).logical, &d);
        try testing.expect(hexEq(r.v1_sha256, &d));
        try testing.expectEqual(@as(usize, n_components), r.segments.len);
        for (layer.segments, r.segments, 0..) |seg, fs, c| {
            const comp: Component = @enumFromInt(c);
            try testing.expectEqualStrings(comp.name(), fs.component);
            try testing.expectEqual(r.sidecar_offset + seg.offset, fs.offset);
            try testing.expectEqual(seg.length, fs.length);
            const slot = rows.row(comp, @intCast(i))[0..@intCast(seg.length)];
            try testing.expectEqualSlices(u8, whole[@intCast(seg.offset)..][0..@intCast(seg.length)], slot);
            Sha256.hash(slot, &d, .{});
            try testing.expect(hexEq(fs.sha256, &d));
            try testing.expect(hexEq(fs.head16, slot[0..16]));
        }
    }
    return @intCast(calls);
}

test "dsv41 slots: layer 13, 8 rows through the pool == pread == v2 sha == Python fixture" {
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.layer_set;
    try testing.expectEqual(@as(usize, 8), set.len);
    var rows = try HostSlotRows.init(&env.bank.layers[set[0].layer], @intCast(set.len));
    defer rows.deinit();
    const calls = try checkSet("layer_set", &env.bank, set, &rows);
    try testing.expect(calls >= 2 * set.len);
    // With DSV41_PHASE0B_MLX the 0b tests run earlier in this process and made the Metal device.
    if (std.c.getenv("DSV41_PHASE0B_MLX") == null) try testing.expect(!metalDriverLoaded());
}

test "dsv41 slots: cross-layer PICK set" {
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.pick_set;
    try testing.expectEqual(@as(usize, 8), set.len);
    var rows = try HostSlotRows.init(&env.bank.layers[set[0].layer], @intCast(set.len));
    defer rows.deinit();
    _ = try checkSet("pick_set", &env.bank, set, &rows);
    // With DSV41_PHASE0B_MLX the 0b tests run earlier in this process and made the Metal device.
    if (std.c.getenv("DSV41_PHASE0B_MLX") == null) try testing.expect(!metalDriverLoaded());
}

// Phase 0b, inside a guarded window (GPU lock held): DSV41_PHASE0B_MLX=1.
test "dsv41 slots 0b: an MLX LayerSlotBank on the CPU stream fills like the host rows" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    var env = try Env.open();
    defer env.close();
    const set = env.parsed.value.layer_set;
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var slots = try LayerSlotBank.init(&env.bank.layers[set[0].layer], @intCast(set.len), stream);
    defer slots.deinit();
    try testing.expectEqual(mlx.mlx_dtype.int16, mlx.mlx_array_dtype(slots.arrays[0]));
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(slots.arrays[1]));
    try testing.expectEqual(@as(usize, 4), mlx.mlx_array_ndim(slots.arrays[0]));
    try testing.expectEqualSlices(c_int, &.{ @intCast(set.len), 320, 144, 48 }, mlx.mlx_array_shape(slots.arrays[0])[0..4]);
    _ = try checkSet("layer_set (MLX LayerSlotBank)", &env.bank, set, &slots);
}

test "dsv41 slots: no Metal device in this process" {
    if (std.c.getenv("DSV41_PHASE0B_MLX") != null) return error.SkipZigTest;
    try testing.expect(!metalDriverLoaded());
}

/// A 2-layer synthetic bank on disk (hidden 64, inter 32: 2,880-byte records
/// in 4 KiB slots), opened, with its experts.bin image.
const SynthBank = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    bank: expert_bank.Bank,

    fn open(n_experts: u32) !SynthBank {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try expert_bank.writeSynth(testing.allocator, &tmp, .{ .n_experts = n_experts });
        errdefer testing.allocator.free(image);
        var rbuf: [512]u8 = undefined;
        const implemented: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = n_experts, .n_layers = 2 };
        const bank = try expert_bank.Bank.open(testing.allocator, std.testing.io, try expert_bank.tmpRoot(&tmp, &rbuf), implemented, null);
        return .{ .tmp = tmp, .image = image, .bank = bank };
    }

    fn close(self: *SynthBank) void {
        self.bank.deinit();
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }
};

const test_pool: expert_io.Options = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 };

/// Routes `ids` and waits for every part (gate/up first, as the kernels do).
fn serve(s: *Stream, layer: u32, ids: []const u16) !*Route {
    const r = try s.route(layer, ids, &.{});
    for (0..r.n_parts) |p| {
        try s.waitGu(r, @intCast(p));
        try s.waitDown(r, @intCast(p));
    }
    return r;
}

/// Every routed id's slot holds its record's bytes.
fn expectServed(s: *Stream, sb: *const SynthBank, r: *const Route, ids: []const u16) !void {
    const geom = &sb.bank.layers[r.layer];
    for (ids, r.plan.slotsOf()) |e, slot| {
        const off = sb.bank.recordOffset(r.layer, e);
        for (geom.segments, 0..) |seg, c| {
            const want = sb.image[off + seg.offset ..][0..seg.length];
            try testing.expectEqualSlices(u8, want, s.slotRow(r.layer, slot, @enumFromInt(c))[0..seg.length]);
        }
    }
}

fn readsOf(r: *const Route) u64 {
    var n: u64 = 0;
    for (r.partsOf()) |p| n += p.n_reads;
    return n;
}

test "dsv41 stream: every routed id is served from a slot holding its record" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var unique: u64 = 0;
    var reads: u64 = 0;
    // Prefill: layer 0 seeded, one wave per layer.
    try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
    const waves = [_]struct { layer: u32, ids: []const u16 }{
        .{ .layer = 0, .ids = &.{ 1, 2, 3, 5, 9 } },
        .{ .layer = 1, .ids = &.{ 7, 8, 9 } },
    };
    for (waves) |w| {
        const r = try serve(s, w.layer, w.ids);
        try expectServed(s, &sb, r, w.ids);
        unique += r.plan.n_hits + r.plan.n_misses;
        reads += readsOf(r);
        s.release(r);
    }
    try s.grow(&.{ 6, 4 });
    // Decode: a deterministic pseudo-random trace over both layers.
    var rng = std.Random.DefaultPrng.init(7);
    const rand = rng.random();
    var ids: [12]u16 = undefined;
    for (0..80) |step| {
        const layer: u32 = @intCast(step % 2);
        const n = rand.intRangeAtMost(usize, 1, 12);
        const span: u16 = if (step % 5 == 0) 32 else 10;
        for (ids[0..n]) |*e| e.* = rand.intRangeLessThan(u16, 0, span);
        const r = try serve(s, layer, ids[0..n]);
        try expectServed(s, &sb, r, ids[0..n]);
        // Parts: at most three records each, in file order.
        var prev: ?u16 = null;
        for (r.partsOf()) |p| {
            try testing.expect(p.n >= 1 and p.n <= 3);
            for (r.order[p.first..][0..p.n]) |li| {
                const e = r.plan.loads[li].expert;
                if (prev) |pe| try testing.expect(pe < e);
                prev = e;
            }
        }
        unique += r.plan.n_hits + r.plan.n_misses;
        reads += readsOf(r);
        s.release(r);
    }
    try s.flush();
    for (0..2) |l| for (0..s.layers[l].policy.capacity + 12) |slot| {
        try testing.expectEqual(@as(u16, 0), s.pinsOf(@intCast(l), @intCast(slot)));
    };
    const st = s.stats();
    try testing.expectEqual(@as(u64, 82), st.route_calls);
    try testing.expectEqual(unique, st.expert_cache_hits + st.expert_cache_misses);
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    try testing.expectEqual(st.expert_cache_misses, reads + st.loads_skipped);
    try testing.expectEqual(reads * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
    try testing.expect(st.expert_cache_hits > 0 and st.expert_cache_evictions > 0 and st.transient_loads > 0);
    try testing.expect(st.preadv_calls >= 2 * reads);
}

test "dsv41 stream: a released route keeps its slots pinned until the next route" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 0 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    // 1, 2 fill layer 0's two slots; 3 is served from transient row 0 (slot 2).
    const r = try serve(s, 0, &.{ 1, 2, 3 });
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, r.plan.slotsOf());
    s.release(r);
    try testing.expectEqual(@as(u16, 1), s.pinsOf(0, 0));
    try testing.expectEqual(@as(u16, 1), s.pinsOf(0, 2));
    // The next route flushes first: layer 1 then reuses transient row 0.
    const r2 = try serve(s, 1, &.{4});
    try testing.expectEqual(@as(u16, 0), s.pinsOf(0, 0));
    try testing.expectEqual(@as(u16, 0), s.pinsOf(0, 1));
    try testing.expectEqualSlices(u32, &.{0}, r2.plan.slotsOf());
    try testing.expectEqual(@as(u16, 1), s.pinsOf(1, 0));
    try expectServed(s, &sb, r2, &.{4});
    s.release(r2);
}

test "dsv41 stream: a row an unreleased route serves from is never refilled" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 0, 0 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    _ = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.SlotStillPinned, s.route(1, &.{3}, &.{}));
    try testing.expectError(error.StreamFailed, s.route(1, &.{3}, &.{}));
}

test "dsv41 stream: routes the caller never releases run out, by name" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    for (0..route_capacity) |_| _ = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesExhausted, s.route(0, &.{ 1, 2 }, &.{}));
}

test "dsv41 stream: a transient row still holding the record is not read again" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 0, 0 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    const steps = [_]struct { layer: u32, ids: []const u16, reads: u64 }{
        .{ .layer = 0, .ids = &.{ 5, 6 }, .reads = 2 },
        .{ .layer = 0, .ids = &.{ 5, 6 }, .reads = 0 },
        .{ .layer = 1, .ids = &.{5}, .reads = 1 },
        .{ .layer = 0, .ids = &.{5}, .reads = 1 },
    };
    for (steps) |st| {
        const r = try serve(s, st.layer, st.ids);
        try testing.expectEqual(st.reads, readsOf(r));
        try expectServed(s, &sb, r, st.ids);
        s.release(r);
    }
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2), st.loads_skipped);
    try testing.expectEqual(4 * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
}

test "dsv41 stream: growth is the one phase change" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var r = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesLive, s.grow(&.{ 4, 4 }));
    s.release(r);
    // Off its route (the default) the release is refused by name and the scratch stays whole through decode.
    try testing.expectError(error.TransientReleaseNotInstalled, s.releaseTransient());
    try testing.expectError(error.InvalidRows, s.grow(&.{ 1, 4 }));
    try testing.expectError(error.InvalidRows, s.grow(&.{4}));
    try s.grow(&.{ 4, 3 });
    try testing.expectEqual(@as(u32, 12), s.transient.rows);
    try testing.expectError(error.AlreadyGrown, s.grow(&.{ 4, 4 }));
    try testing.expectError(error.NotPrefill, s.seedPrefill(0, &.{1}));
    // Residents keep their slots and bytes; the added rows fill before any eviction.
    r = try serve(s, 0, &.{ 1, 2, 3, 4 });
    try testing.expectEqual(@as(u32, 2), r.plan.n_hits);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, r.plan.slotsOf());
    try testing.expectEqual(@as(u32, 0), r.plan.n_evictions);
    try expectServed(s, &sb, r, &.{ 1, 2, 3, 4 });
    s.release(r);
}

test "dsv41 stream: A0 (a): the grow's warm reads fill empty rows below demand; a layer's first decode route lands the started, cancels the queued (served on demand) and counts its hits" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const page = std.heap.pageSize();
    // Off its route the class is refused by name and counts nothing.
    {
        const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
        defer s.deinit();
        try s.grow(&.{ 4, 4 });
        try testing.expectError(error.WarmNotInstalled, s.warmIssue(0, &.{3}));
        try testing.expectEqual(@as(u64, 0), s.stats().warm_issued);
    }
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .first_verify_warm = .{ .max_records = 8, .busy_max = 1 } });
    defer s.deinit();
    defer expert_io.clearFaults();
    var r = try serve(s, 0, &.{ 1, 2 });
    s.release(r);
    r = try serve(s, 1, &.{ 1, 2 });
    s.release(r);
    try testing.expectError(error.WarmBeforeGrow, s.warmIssue(0, &.{3}));
    try s.grow(&.{ 6, 6 });
    // Layer 0's first warm record (expert 3) holds the reader 80 ms, so the rest wait in the warm ring (busy limit 1).
    const off3 = sb.bank.recordOffset(0, 3);
    expert_io.injectFault(off3 - off3 % page, 5, 80 * std.time.ns_per_ms);
    // Layer 0: 1 is resident (skipped), 3, 4 and 5 issued; layer 1: 7; a layer issues once.
    try testing.expectEqual(@as(u32, 3), try s.warmIssue(0, &.{ 1, 3, 4, 5 }));
    try testing.expectEqual(@as(u32, 1), try s.warmIssue(1, &.{7}));
    try testing.expectError(error.InvalidWarm, s.warmIssue(0, &.{6}));
    while (s.pool.counter(.warm_started) == 0) std.Thread.yield() catch {};
    // Layer 0's first decode route: 3 lands (waited for), 4 and 5 are cancelled; 3 is a hit, 4 and 6 are read on demand.
    r = try serve(s, 0, &.{ 3, 4, 6 });
    try testing.expectEqual(@as(u32, 1), r.plan.n_hits);
    try testing.expectEqualSlices(u16, &.{3}, r.plan.hitsOf());
    try expectServed(s, &sb, r, &.{ 3, 4, 6 });
    try testing.expect(s.warmWaitNs(0) > 0);
    s.release(r);
    // Layer 1's warm record lands below demand; its first route serves it as a hit.
    const w = &s.warm.?;
    const t7 = w.loads[w.layers[1].lo].ticket;
    try s.pool.wait(t7, 2, 10 * std.time.ns_per_s);
    r = try serve(s, 1, &.{ 7, 8 });
    try testing.expectEqualSlices(u16, &.{7}, r.plan.hitsOf());
    try expectServed(s, &sb, r, &.{ 7, 8 });
    s.release(r);
    // Settled once: a later route of layer 0 counts no warm hit.
    r = try serve(s, 0, &.{3});
    s.release(r);
    const st = s.stats();
    try testing.expectEqual(@as(u64, 4), st.warm_issued);
    try testing.expectEqual(@as(u64, 2), st.warm_landed);
    try testing.expectEqual(@as(u64, 2), st.warm_cancelled);
    try testing.expectEqual(@as(u64, 2), st.warm_hits);
    try testing.expectEqual(st.warm_issued, st.warm_landed + st.warm_cancelled);
    try testing.expect(!s.warm_live);
}

test "dsv41 stream: slots held for a deferred call are never refilled until released" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var r = try serve(s, 0, &.{ 1, 2 });
    const kept = [2]u32{ s.layers[0].policy.expert_to_slot[1], s.layers[0].policy.expert_to_slot[2] };
    try testing.expect(kept[0] < 2 and kept[1] < 2);
    try s.holdBase(r);
    s.release(r);
    // One layer at a time.
    const other = try serve(s, 1, &.{7});
    try testing.expectError(error.HeldOtherLayer, s.holdBase(other));
    s.release(other);
    // A later route of the layer: the held slots are neither evicted nor refilled; it is served right.
    r = try serve(s, 0, &.{ 3, 4, 5 });
    for (r.plan.slotsOf()) |sl| try testing.expect(sl != kept[0] and sl != kept[1]);
    try expectServed(s, &sb, r, &.{ 3, 4, 5 });
    try testing.expectEqual(kept[0], s.layers[0].policy.expert_to_slot[1]);
    try testing.expectEqual(kept[1], s.layers[0].policy.expert_to_slot[2]);
    s.release(r);
    // Released: the slots take loads again.
    s.releaseHeld();
    r = try serve(s, 0, &.{ 6, 7, 8, 9 });
    try expectServed(s, &sb, r, &.{ 6, 7, 8, 9 });
    s.release(r);
    try s.flush();
    for (0..2) |sl| try testing.expectEqual(@as(u16, 0), s.layers[0].meta[sl].pins);
}

test "dsv41 stream: growth from any thread but the one that built the stream is refused" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    const Helper = struct {
        fn run(st: *Stream, out: *?anyerror) void {
            st.grow(&.{ 4, 4 }) catch |e| {
                out.* = e;
                return;
            };
            out.* = null;
        }
    };
    var got: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, Helper.run, .{ s, &got });
    t.join();
    try testing.expectEqual(@as(?anyerror, error.NotInferenceThread), got);
    try s.grow(&.{ 4, 4 });
}

test "dsv41 stream: a failed read fails the route and every later one" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    defer expert_io.clearFaults();
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 1).gu_offset / page * page, 2, 0);
    const r = try s.route(0, &.{1}, &.{});
    try testing.expectError(error.ReadFailed, s.waitDown(r, 0));
    try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(1));
    try testing.expectError(error.StreamFailed, s.route(0, &.{1}, &.{}));
}

/// One layer's prompt call on a fresh stream (the pool is the process's: one stream at a time), its predicted
/// seed read ahead first when given: every routed id served from its record; the call's hits and stats.
fn p1Call(sb: *const SynthBank, predicted: ?[]const u16) !struct { hits: u64, st: Stats } {
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 6, 6 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    if (predicted) |p| {
        try s.readAheadSeed(0, p);
        try testing.expect(s.ahead.live and s.ahead.n == p.len);
        try s.awaitReadAhead(0);
        for (p) |e| {
            const slot = s.layers[0].policy.slotOf(e).?;
            try testing.expectEqual(SlotState.ready, s.locate(0, slot).meta.state);
            try testing.expectEqual(@as(u16, 0), s.pinsOf(0, slot));
        }
    }
    // The layer's routed ids at its barrier, then its distinct experts in two groups.
    try s.seedPrefill(0, &.{ 1, 2, 3, 1, 2, 1, 4, 5, 6, 9, 10, 4, 1, 2 });
    var hits: u64 = 0;
    for ([_][]const u16{ &.{ 1, 2, 4, 3, 5 }, &.{ 6, 9, 10 } }) |grp| {
        const r = try serve(s, 0, grp);
        try expectServed(s, sb, r, grp);
        hits += r.plan.n_hits;
        s.release(r);
    }
    try s.flush();
    return .{ .hits = hits, .st = s.stats() };
}

test "dsv41 stream: P1: a read-ahead's records land before the layer's routes, which serve them as hits; every served byte is a stream's without it" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // The predicted seed: three of the call's four hottest, and 7 (mispredicted).
    const on = try p1Call(&sb, &.{ 1, 2, 7, 4 });
    const off = try p1Call(&sb, null);
    // 1, 2 and 4: hits, read during the attention; 7: read and never served (the waste); the rest as without.
    try testing.expectEqual(off.hits + 3, on.hits);
    try testing.expectEqual(off.st.expert_cache_misses - 3, on.st.expert_cache_misses);
    const rec = sb.bank.layers[0].logical_bytes;
    try testing.expectEqual(off.st.expert_bytes_read + (4 - 3) * rec, on.st.expert_bytes_read);
    // Its engagement at the barrier: 4 posted, 3 routed by the call (7 not), the seed's 3, 5 and 6 on demand.
    try testing.expectEqual(@as(u64, 4), on.st.ahead_posted);
    try testing.expectEqual(@as(u64, 3), on.st.ahead_hits);
    try testing.expectEqual(@as(u64, 3), on.st.ahead_demand);
    try testing.expectEqual(4 * rec, on.st.ahead_bytes);
    try testing.expectEqual(@as(u64, 0), off.st.ahead_posted + off.st.ahead_hits + off.st.ahead_demand + off.st.ahead_bytes);
}

test "dsv41 stream: P1: a route lands its layer's read-ahead first, another layer's lands the live one, full rows admit none, the phase change lands one" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 3, 3 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    // Three rows: 4, 5 and 6 admitted, 7 does not fit.
    try s.readAheadSeed(0, &.{ 4, 5, 6, 7 });
    try testing.expectEqual(@as(u32, 3), s.ahead.n);
    try s.readAheadSeed(1, &.{8});
    try testing.expect(s.ahead.live and s.ahead.layer == 1);
    for ([_]u16{ 4, 5, 6 }) |e| try testing.expectEqual(SlotState.ready, s.locate(0, s.layers[0].policy.slotOf(e).?).meta.state);
    try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(7));
    // A route of layer 1 before its barrier's await: the read-ahead lands first, 8 is a hit.
    const r = try serve(s, 1, &.{ 8, 9 });
    try testing.expect(!s.ahead.live);
    try testing.expectEqual(@as(u32, 1), r.plan.n_hits);
    try expectServed(s, &sb, r, &.{ 8, 9 });
    s.release(r);
    try s.readAheadSeed(0, &.{ 10, 11 });
    try testing.expect(!s.ahead.live and s.ahead.n == 0);
    try s.readAheadSeed(1, &.{12});
    try testing.expect(s.ahead.live);
    try s.grow(&.{ 4, 4 });
    try testing.expect(!s.ahead.live);
    try testing.expectEqual(SlotState.ready, s.locate(1, s.layers[1].policy.slotOf(12).?).meta.state);
    try testing.expectError(error.NotPrefill, s.readAheadSeed(1, &.{13}));
}

test "dsv41 stream: P1: a read-ahead whose record fails to land forgets its job's records and fails the stream" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    defer expert_io.clearFaults();
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 2).gu_offset / page * page, 2, 0);
    try s.readAheadSeed(0, &.{ 1, 2, 3 });
    try testing.expectError(error.ReadFailed, s.awaitReadAhead(0));
    for ([_]u16{ 1, 2, 3 }) |e| try testing.expectEqual(@as(?u32, null), s.layers[0].policy.slotOf(e));
    try testing.expectError(error.StreamFailed, s.route(0, &.{1}, &.{}));
}

test "dsv41 stream: P1's construction self-check: records read ahead equal their demand reads over two jobs; the layer is left as found" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 12, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    const experts = [_]u16{ 31, 30, 29, 28, 27, 26, 25, 24, 23 };
    try s.checkReadAhead(0, &experts);
    try testing.expectEqual(@as(u32, 0), s.layers[0].policy.occupancy);
    for (0..12) |slot| {
        try testing.expectEqual(@as(u16, 0), s.pinsOf(0, @intCast(slot)));
        try testing.expectEqual(SlotState.empty, s.locate(0, @intCast(slot)).meta.state);
    }
    try testing.expect(!s.ahead.live and !s.failed);
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2 * experts.len) * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
    // Refused shapes: a resident expert; more experts than free rows.
    const r = try serve(s, 0, &.{5});
    s.release(r);
    try testing.expectError(error.ReadAheadCheckShape, s.checkReadAhead(0, &.{5}));
    try testing.expectError(error.ReadAheadCheckShape, s.checkReadAhead(0, &(experts ++ [_]u16{ 22, 21, 20 })));
}

/// sha256 of a served slot's logical record (its nine component rows).
fn slotDigest(s: *Stream, layer: u32, slot: u32, geom: *const Layer) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (geom.segments, 0..) |seg, c| h.update(s.slotRow(layer, slot, @enumFromInt(c))[0..seg.length]);
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

// DSV41_BANK=<bank dir> DSV41_PHASE1_ROUTE_FIXTURE=<json from R/exl3/runtime/dump_phase1_route_fixture.py>
test "dsv41 stream: a recorded trace on the real bank serves every slot's bytes" {
    try realBankTrace(.host);
}

/// The phase-1 one-layer recorded trace on the real bank, the slot rows in `memory`.
fn realBankTrace(memory: SlotMemory) !void {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_PHASE1_ROUTE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(64 << 20));
    defer a.free(text);
    const policy_fixtures = @import("expert_policy_test.zig");
    const parsed = try std.json.parseFromSlice(struct { bank_trace: policy_fixtures.BankTrace }, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const bt = parsed.value.bank_trace;
    const L = bt.layer;
    var rows: [40]u32 = @splat(0);
    rows[L] = bt.prefill_rows;
    const s = try Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = bt.transient, .transient_rows = bt.transient, .slot_memory = memory });
    defer s.deinit();
    const geom = &bank.layers[L];
    var served: u64 = 0;
    try s.seedPrefill(L, bt.seed);
    for ([_][]const policy_fixtures.FixPlan{ bt.prefill, bt.routes }, 0..) |plans, phase| {
        if (phase == 1) {
            rows[L] = bt.decode_rows;
            try s.grow(&rows);
        }
        for (plans) |want| {
            const r = try serve(s, L, want.ids);
            try policy_fixtures.expectPlan(&r.plan, want);
            // Every served expert's slot holds its record: sha256 == the runtime manifest's.
            for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |e, slot| {
                const d = slotDigest(s, L, slot, geom);
                try testing.expectEqualSlices(u8, &bank.digest(L, e).logical, &d);
                served += 1;
            }
            for (r.plan.loadsOf()) |l| {
                const d = slotDigest(s, L, l.slot, geom);
                try testing.expectEqualSlices(u8, &bank.digest(L, l.expert).logical, &d);
                served += 1;
            }
            s.release(r);
        }
    }
    try s.flush();
    const st = s.stats();
    const ru = std.posix.getrusage(std.c.rusage.SELF);
    std.debug.print(
        "real bank layer {d} ({s} rows): {d} routes, {d} served slots sha256-checked; hits {d} misses {d} evictions {d} persistent {d} transient {d} skipped {d}; {d} B read in {d} preadv ({d:.3} s read, {d} ms wall); {d} ms total; peak RSS {d} B\n",
        .{ L, @tagName(memory), st.route_calls, served, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions, st.persistent_loads, st.transient_loads, st.loads_skipped, st.expert_bytes_read, st.preadv_calls, st.expert_read_seconds, @divTrunc(st.read_wall_ns, std.time.ns_per_ms), @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms), ru.maxrss },
    );
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped) * geom.logical_bytes, st.expert_bytes_read);
}

// Inside a guarded window: DSV41_PHASE0B_MLX=1 + the bank env above.
test "dsv41 stream 0b: the recorded trace on the real bank fills MLX slot banks" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    try realBankTrace(.{ .mlx = stream });
}

test "dsv41 stream: slot refs name each served slot's bank and row" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 2, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    var refs: [max_route_ids]SlotRef = undefined;
    // Prefill: 1 and 2 fill the two rows, 3 lands in the shared transient scratch.
    var r = try serve(s, 0, &.{ 1, 2, 3 });
    try testing.expectEqualSlices(SlotRef, &.{ .{ .bank = .base, .row = 0 }, .{ .bank = .base, .row = 1 }, .{ .bank = .transient, .row = 0 } }, s.refsOf(r, &refs));
    s.release(r);
    try s.grow(&.{ 4, 2 });
    // Decode: the grown rows of layer 0 are its ext bank.
    r = try serve(s, 0, &.{ 1, 4, 5 });
    try testing.expectEqualSlices(SlotRef, &.{ .{ .bank = .base, .row = 0 }, .{ .bank = .ext, .row = 0 }, .{ .bank = .ext, .row = 1 } }, s.refsOf(r, &refs));
    try expectServed(s, &sb, r, &.{ 1, 4, 5 });
    // A part's loads are the rows its waves read, in file order.
    var loads: [max_route_ids]expert_policy.Load = undefined;
    try testing.expectEqual(@as(u32, 1), r.n_parts);
    const pl = r.partLoads(0, &loads);
    try testing.expectEqual(@as(usize, 2), pl.len);
    try testing.expectEqual(@as(u16, 4), pl[0].expert);
    try testing.expectEqual(@as(u16, 5), pl[1].expert);
    // Host rows have no MLX arrays to bind.
    try testing.expectEqual(@as(?BankArrays, null), s.bankArrays(0, .base));
    s.release(r);
}

fn expectShape(arr: mlx.mlx_array, dtype: mlx.mlx_dtype, want: []const c_int) !void {
    try testing.expectEqual(dtype, mlx.mlx_array_dtype(arr));
    try testing.expectEqual(want.len, mlx.mlx_array_ndim(arr));
    try testing.expectEqualSlices(c_int, want, mlx.mlx_array_shape(arr)[0..want.len]);
}

// DSV41_PHASE0B_MLX=1, inside a guarded window.
test "dsv41 stream 0b: MLX slot memory is filled by the pool like host rows" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .slot_memory = .{ .mlx = stream } });
    defer s.deinit();
    // Hidden 64 / inter 32 synthetic geometry: code [rows, 4, 2, 48] (gate/up), [rows, 2, 4, 48] (down).
    const base = s.bankArrays(0, .base).?;
    try expectShape(base.gate.code, .int16, &.{ 4, 4, 2, 48 });
    try expectShape(base.gate.rout, .float16, &.{ 4, 32 });
    try expectShape(base.down.code, .int16, &.{ 4, 2, 4, 48 });
    try expectShape(base.down.rin, .float16, &.{ 4, 32 });
    try testing.expectEqual(s.bankArrays(0, .transient).?.up.code.ctx, s.bankArrays(1, .transient).?.up.code.ctx);
    try testing.expectEqual(@as(?BankArrays, null), s.bankArrays(0, .ext));
    s.release(try serve(s, 0, &.{ 1, 2, 3, 5, 9 }));
    s.release(try serve(s, 1, &.{ 7, 8, 9 }));
    try s.grow(&.{ 6, 4 });
    try expectShape(s.bankArrays(0, .ext).?.up.rin, .float16, &.{ 2, 64 });
    var rng = std.Random.DefaultPrng.init(7);
    const rand = rng.random();
    var ids: [12]u16 = undefined;
    for (0..40) |step| {
        const layer: u32 = @intCast(step % 2);
        const n = rand.intRangeAtMost(usize, 1, 12);
        for (ids[0..n]) |*e| e.* = rand.intRangeLessThan(u16, 0, if (step % 5 == 0) 32 else 10);
        const r = try serve(s, layer, ids[0..n]);
        try expectServed(s, &sb, r, ids[0..n]);
        s.release(r);
    }
    try s.flush();
}

// DSV41_PHASE0B_MLX=1 only (lock-held; 2 GB at a time, freed between). Growth-overlap step 2's premise, measured
// before it is built: the inference thread's cost of 2 GB of new slot memory under the server's wired policy, (a) as
// zeros + one eval (step 1), (b) as a page-aligned mapping wrapped no-copy while untouched, then its first GPU use,
// (c) the same after a helper thread touched every page (the helper's time printed apart). Asserts only no-copy.
test "dsv41 growth 0b: new slot memory's cost on the inference thread, zeros vs a no-copy wrap before and after a helper's touch" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const io = testing.io;
    _ = mlx.applyWiredPolicy();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bytes: usize = 2 << 30;
    const elems: c_int = @intCast(bytes / 2);
    const page = std.heap.pageSize();
    const Probe = struct {
        fn ms(t: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(t.untilNow(testing.io, .boot).nanoseconds)) / 1e6;
        }
        const Payload = struct { m: []align(std.heap.page_size_min) u8 };
        fn dtor(ctx: ?*anyopaque) callconv(.c) void {
            const pl: *Payload = @ptrCast(@alignCast(ctx.?));
            std.posix.munmap(pl.m);
            std.heap.c_allocator.destroy(pl);
        }
        fn map(len: usize) ![]align(std.heap.page_size_min) u8 {
            return std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        }
        /// The mapping as an int16 array, no copy (MLX calls `dtor` at once when it had to copy).
        fn wrap(m: []align(std.heap.page_size_min) u8, n: c_int) !struct { arr: mlx.mlx_array, no_copy: bool } {
            const pl = try std.heap.c_allocator.create(Payload);
            pl.* = .{ .m = m };
            const base = m.ptr;
            const arr = mlx.mlx_array_new_data_managed_payload(@ptrCast(base), &[_]c_int{n}, 1, .int16, pl, dtor);
            const d = mlx.mlx_array_data_uint8(arr);
            return .{ .arr = arr, .no_copy = if (d) |p| @intFromPtr(p) == @intFromPtr(base) else false };
        }
        /// One GPU command that reads the array (a sum over its first 1024 elements).
        fn use(arr: mlx.mlx_array, st: mlx.mlx_stream) !void {
            var sl = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sl);
            try mlx.check(mlx.mlx_slice(&sl, arr, &[_]c_int{0}, 1, &[_]c_int{1024}, 1, &[_]c_int{1}, 1, st));
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_sum(&r, sl, false, st));
            try mlx.check(mlx.mlx_array_eval(r));
        }
        fn touch(m: []u8, step: usize) void {
            var i: usize = 0;
            while (i < m.len) : (i += step) m[i] = 0;
        }
    };
    // (a) step 1: zeros and one eval.
    var t = std.Io.Timestamp.now(io, .boot);
    var z = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{elems}, 1, .int16, s));
    try evalArrays(&.{z});
    const zeros_ms = Probe.ms(t);
    _ = mlx.mlx_array_free(z);
    _ = mlx.mlx_clear_cache();
    // (b) an untouched mapping, wrapped, then its first use.
    const mb = try Probe.map(bytes);
    t = std.Io.Timestamp.now(io, .boot);
    const wb = try Probe.wrap(mb, elems);
    const wrap_untouched_ms = Probe.ms(t);
    t = std.Io.Timestamp.now(io, .boot);
    try Probe.use(wb.arr, s);
    const use_untouched_ms = Probe.ms(t);
    _ = mlx.mlx_array_free(wb.arr);
    // (c) a helper touches every page first (off the timed thread), then the wrap and its first use.
    const mc = try Probe.map(bytes);
    t = std.Io.Timestamp.now(io, .boot);
    const helper = try std.Thread.spawn(.{}, Probe.touch, .{ @as([]u8, mc), page });
    helper.join();
    const helper_ms = Probe.ms(t);
    t = std.Io.Timestamp.now(io, .boot);
    const wc = try Probe.wrap(mc, elems);
    const wrap_touched_ms = Probe.ms(t);
    t = std.Io.Timestamp.now(io, .boot);
    try Probe.use(wc.arr, s);
    const use_touched_ms = Probe.ms(t);
    _ = mlx.mlx_array_free(wc.arr);
    _ = mlx.mlx_clear_cache();
    std.debug.print("\nGROWTH_OVERLAP_PROBE {{\"bytes\": {d}, \"zeros_eval_ms\": {d:.2}, \"wrap_untouched_ms\": {d:.2}, \"first_use_untouched_ms\": {d:.2}, \"helper_touch_ms\": {d:.2}, \"wrap_touched_ms\": {d:.2}, \"first_use_touched_ms\": {d:.2}, \"no_copy\": [{}, {}]}}\n", .{
        bytes, zeros_ms, wrap_untouched_ms, use_untouched_ms, helper_ms, wrap_touched_ms, use_touched_ms, wb.no_copy, wc.no_copy,
    });
    try testing.expect(wb.no_copy and wc.no_copy);
}

test "dsv41 stream: the construction's forget: the warm-up's seeded residents cleared, the first prompt's seed takes every row and its read-ahead fills them" {
    var sb = try SynthBank.open(64);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 8, 8 }, .pool = test_pool });
    defer s.deinit();
    // The warm-up's wide call on layer 0: its 6 experts seeded (protected) and read into the rows.
    try s.seedPrefill(0, &.{ 40, 41, 42, 43, 44, 45, 40, 41 });
    s.release(try serve(s, 0, &.{ 40, 41, 42, 43, 44, 45 }));
    try s.flush();
    try testing.expectEqual(@as(u32, 6), s.layers[0].policy.occupancy);
    try testing.expectEqual(@as(usize, 6), s.layers[0].policy.protected.count());
    // Kept, they would hold 6 of layer 0's 8 rows through a prompt (a seed of 8 - 6 = 2). Forgotten once:
    try testing.expectEqual(@as(u32, 6), try s.forgetResidents());
    for (s.layers) |*ls| {
        try testing.expectEqual(@as(u32, 0), ls.policy.occupancy);
        try testing.expectEqual(@as(usize, 0), ls.policy.protected.count());
    }
    // The first prompt: each layer's read-ahead posts every row and each seed takes every row (1..8 twice, 9 and 10 once).
    const prompt = [_]u16{ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10 };
    for (0..2) |l| {
        try s.readAheadSeed(@intCast(l), &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        try s.seedPrefill(@intCast(l), &prompt);
        try testing.expectEqual(@as(u32, 8), s.seedRanks(@intCast(l)));
    }
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2 * 8), st.ahead_posted);
    try testing.expectEqual(@as(u64, 0), st.ahead_demand);
    // Refused while anything is live.
    const r = try serve(s, 0, &.{ 1, 2 });
    try testing.expectError(error.RoutesLive, s.forgetResidents());
    s.release(r);
}

/// The 0b box probes' readings (the growth probe's and the transient release's): the footprint on either side of a
/// posix_spawn'd vm_stat, the settles, the report lines.
const ProbeBox = struct {
    const ar = @import("deepseek_v41_ar.zig");

    pages: ar.VmStatPages,
    fp: u64,
    pm: sdk.memory.ProcessMemory,

    const Self = @This();
    const Settled = struct { b: Self, ms: ?u64 };

    /// One moment's reading: the footprint on either side of a posix_spawn'd vm_stat, within the harnesses' bound.
    fn mark(buf: []u8) !@This() {
        var n: u32 = 0;
        while (n < ar.box_mark_attempts) : (n += 1) {
            if (n > 0) std.Io.sleep(testing.io, .fromMilliseconds(ar.box_mark_retry_ms), .awake) catch {};
            const f0 = sdk.memory.footprint().now;
            const pages = try ar.vmStatPages(try ar.readVmStat(buf));
            const f1 = sdk.memory.footprint().now;
            if (@max(f0, f1) - @min(f0, f1) <= ar.box_mark_stable_bytes) return .{ .pages = pages, .fp = @max(f0, f1), .pm = sdk.memory.processMemory() };
        }
        return error.BoxMarkUnstable;
    }
    fn d(x: u64, y: u64) i64 {
        return @as(i64, @intCast(y)) - @as(i64, @intCast(x));
    }
    /// Physical growth outside the footprint since `b0`, the file-backed pages excluded (the guard credits the cache).
    fn outside(b0: @This(), b: @This()) i64 {
        return d(b0.pages.physical(), b.pages.physical()) - d(b0.pages.file_backed, b.pages.file_backed) - d(b0.fp, b.fp);
    }
    /// Marks every 50 ms until the growth outside the footprint since `b0` is within `limit`, at most `bound_ms`
    /// (ms: null when it never is).
    fn settle(b0: Self, buf: []u8, bound_ms: u64, limit: i64) !Settled {
        const t0 = std.Io.Timestamp.now(testing.io, .boot);
        while (true) {
            const b = try mark(buf);
            const ms: u64 = @intCast(@divTrunc(t0.untilNow(testing.io, .boot).nanoseconds, std.time.ns_per_ms));
            if (outside(b0, b) <= limit) return .{ .b = b, .ms = ms };
            if (ms >= bound_ms) return .{ .b = b, .ms = null };
            std.Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
        }
    }
    /// Marks every 50 ms until the footprint is at most `limit` above `ref`'s (a release measured by the process's own
    /// ledger, not only by the growth outside it), at most `bound_ms` (ms: null when it never is).
    fn settleFootprint(ref: Self, buf: []u8, bound_ms: u64, limit: i64) !Settled {
        const t0 = std.Io.Timestamp.now(testing.io, .boot);
        while (true) {
            const b = try mark(buf);
            const ms: u64 = @intCast(@divTrunc(t0.untilNow(testing.io, .boot).nanoseconds, std.time.ns_per_ms));
            if (d(ref.fp, b.fp) <= limit) return .{ .b = b, .ms = ms };
            if (ms >= bound_ms) return .{ .b = b, .ms = null };
            std.Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
        }
    }
    /// A baseline once wired and physical hold still across two marks 100 ms apart (an earlier step's pages still being
    /// retired would land inside the measured steps), at most `bound_ms` (ms: null when they never did).
    fn settled(buf: []u8, bound_ms: u64) !Settled {
        const t0 = std.Io.Timestamp.now(testing.io, .boot);
        var prev = try mark(buf);
        const stable: u64 = ar.box_mark_stable_bytes;
        while (true) {
            std.Io.sleep(testing.io, .fromMilliseconds(100), .awake) catch {};
            const b = try mark(buf);
            const ms: u64 = @intCast(@divTrunc(t0.untilNow(testing.io, .boot).nanoseconds, std.time.ns_per_ms));
            if (@abs(d(prev.pages.wired, b.pages.wired)) <= stable and @abs(d(prev.pages.physical(), b.pages.physical())) <= stable) return .{ .b = b, .ms = ms };
            if (ms >= bound_ms) return .{ .b = b, .ms = null };
            prev = b;
        }
    }
    /// One more GPU command (a release the driver retires only at a later submission).
    fn nextCommand(s: mlx.mlx_stream) !void {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{1}, 1, .float32, s));
        try evalArrays(&.{z});
        _ = mlx.mlx_array_free(z);
    }
    fn msOf(x: ?u64, out: []u8) []const u8 {
        return if (x) |v| std.fmt.bufPrint(out, "{d}", .{v}) catch out[0..0] else "null";
    }
    fn line(b0: @This(), b: @This(), out: []u8) []const u8 {
        return std.fmt.bufPrint(out, "{{\"d_footprint\": {d}, \"d_physical\": {d}, \"d_wired\": {d}, \"d_file_backed\": {d}, \"d_graphics_nofootprint\": {d}, \"d_internal\": {d}, \"outside\": {d}}}", .{
            d(b0.fp, b.fp), d(b0.pages.physical(), b.pages.physical()), d(b0.pages.wired, b.pages.wired), d(b0.pages.file_backed, b.pages.file_backed),
            d(b0.pm.graphics_nofootprint, b.pm.graphics_nofootprint), d(b0.pm.internal, b.pm.internal), outside(b0, b),
        }) catch out[0..0];
    }
    const Payload = struct { m: []align(std.heap.page_size_min) u8 };
    fn dtor(ctx: ?*anyopaque) callconv(.c) void {
        const pl: *Payload = @ptrCast(@alignCast(ctx.?));
        std.posix.munmap(pl.m);
        std.heap.c_allocator.destroy(pl);
    }
    fn touch(m: []u8, step: usize) void {
        var i: usize = 0;
        while (i < m.len) : (i += step) @as(*volatile u8, &m[i]).* = 0;
    }
};

// DSV41_PHASE0B_MLX=1, inside a guarded window: SERVED13's kill (12 GB outside the footprint late in decode; the grow's
// wrapped rows its one new mechanism) at 2 GB: the box's pages around each step of the overlapped grow's mechanics,
// the release included (SERVED14's probe: the release left 2.15 GB wired outside the footprint).
test "dsv41 growth 0b: box probe: a no-copy wrap of 2 GB of touched anonymous pages, its first GPU read of every page and its release stay inside the footprint" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    _ = mlx.applyWiredPolicy();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bytes: usize = 2 << 30;
    const elems: c_int = @intCast(bytes / 2);
    const page = std.heap.pageSize();
    const Box = ProbeBox;
    var buf: [1 << 16]u8 = undefined;
    const base = try Box.settled(&buf, 3000);
    const b0 = base.b;
    // The overlapped grow's mechanics: an untouched private anonymous mapping, a helper's touch of every page.
    const m = try std.posix.mmap(null, bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    const helper = try std.Thread.spawn(.{}, Box.touch, .{ @as([]u8, m), page });
    helper.join();
    const b1 = try Box.mark(&buf);
    // The no-copy wrap (its deleter unmaps), a view of it and one eval.
    const pl = try std.heap.c_allocator.create(Box.Payload);
    pl.* = .{ .m = m };
    const arr = mlx.mlx_array_new_data_managed_payload(@ptrCast(m.ptr), &[_]c_int{elems}, 1, .int16, pl, Box.dtor);
    const no_copy = if (mlx.mlx_array_data_uint8(arr)) |p| @intFromPtr(p) == @intFromPtr(m.ptr) else false;
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&view, arr, &[_]c_int{ 1024, @divExact(elems, 1024) }, 2, s));
    try evalArrays(&.{view});
    const b2 = try Box.mark(&buf);
    // The first GPU read of every page: a sum over the whole array.
    var sum = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum(&sum, view, false, s));
    try evalArrays(&.{sum});
    const b3 = try Box.mark(&buf);
    // Released: the arrays freed (the wrap's deleter unmaps), synchronize, MLX's cache cleared, the box settled from the
    // post-touch state (2 s bound). Else one more GPU command and a second settle (1 s): a release the driver retires only
    // at a later submission. The first GPU read is judged from the post-touch state too.
    _ = mlx.mlx_array_free(sum);
    _ = mlx.mlx_array_free(view);
    _ = mlx.mlx_array_free(arr);
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    const limit: i64 = @intCast(bytes / 10);
    const r1 = try Box.settle(b1, &buf, 2000, limit);
    var r2: ?Box.Settled = null;
    if (r1.ms == null) {
        try Box.nextCommand(s);
        r2 = try Box.settle(b1, &buf, 1000, limit);
    }
    const released = r1.ms != null or (r2 != null and r2.?.ms != null);
    const out3 = Box.outside(b1, b3);
    const verdict = if (out3 > limit) "GrowWrapOutsideFootprint" else if (!released) "GrowWrapReleaseOutsideFootprint" else "inside";
    var l: [6][320]u8 = undefined;
    var ms: [3][24]u8 = undefined;
    std.debug.print("\nGROWTH_BOX_PROBE {{\"bytes\": {d}, \"no_copy\": {}, \"baseline_settle_ms\": {s}, \"touch\": {s}, \"wrap_eval\": {s}, \"first_gpu_read\": {s}, \"first_gpu_read_from_touch\": {s}, \"release_from_touch\": {s}, \"release_settle_ms\": {s}, \"after_next_command_from_touch\": {s}, \"after_next_command_settle_ms\": {s}, \"outside_limit\": {d}, \"verdict\": \"{s}\"}}\n", .{
        bytes, no_copy, Box.msOf(base.ms, &ms[2]), Box.line(b0, b1, &l[0]), Box.line(b0, b2, &l[1]), Box.line(b0, b3, &l[2]), Box.line(b1, b3, &l[5]), Box.line(b1, r1.b, &l[3]), Box.msOf(r1.ms, &ms[0]),
        if (r2) |x| Box.line(b1, x.b, &l[4]) else "null", if (r2) |x| Box.msOf(x.ms, &ms[1]) else "null", limit, verdict,
    });
    // Control (the growth's way back): an MLX-allocated array of the same bytes, written and read in full on the GPU,
    // released through the allocator (synchronize, cache cleared) from its own settled baseline; its release is the
    // footprint back within the limit of that baseline (reported beside the wrap's, not judged).
    const cbase = try Box.settled(&buf, 3000);
    const c0 = cbase.b;
    var za = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&za, &[_]c_int{elems}, 1, .int16, s));
    var zs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum(&zs, za, false, s));
    try evalArrays(&.{ za, zs });
    const c1 = try Box.mark(&buf);
    _ = mlx.mlx_array_free(zs);
    _ = mlx.mlx_array_free(za);
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    const cr = try Box.settleFootprint(c0, &buf, 2000, limit);
    var cr2: ?Box.Settled = null;
    if (cr.ms == null) {
        try Box.nextCommand(s);
        cr2 = try Box.settleFootprint(c0, &buf, 1000, limit);
    }
    const c_released = cr.ms != null or (cr2 != null and cr2.?.ms != null);
    var cl: [3][320]u8 = undefined;
    var cms: [3][24]u8 = undefined;
    std.debug.print("\nGROWTH_BOX_PROBE_CONTROL {{\"bytes\": {d}, \"baseline_settle_ms\": {s}, \"written_read\": {s}, \"release\": {s}, \"release_footprint_settle_ms\": {s}, \"after_next_command\": {s}, \"after_next_command_footprint_settle_ms\": {s}, \"footprint_limit\": {d}, \"verdict\": \"{s}\"}}\n", .{
        bytes, Box.msOf(cbase.ms, &cms[2]), Box.line(c0, c1, &cl[0]), Box.line(c0, cr.b, &cl[1]), Box.msOf(cr.ms, &cms[0]),
        if (cr2) |x| Box.line(c0, x.b, &cl[2]) else "null", if (cr2) |x| Box.msOf(x.ms, &cms[1]) else "null", limit, if (c_released) "released" else "ControlReleaseKeptFootprint",
    });
    try testing.expect(no_copy);
    if (out3 > limit) return error.GrowWrapOutsideFootprint;
    if (!released) return error.GrowWrapReleaseOutsideFootprint;
}

// DSV41_PHASE0B_MLX=1 and DSV41_BANK=<bank dir>, inside a guarded window (SERVED16): the 240-row MLX scratch filled with
// records and read on the GPU, released (the allocator check) and cache-cleared back inside the footprint (the box
// probe's after-release rule); then decode's window 0 serves a route's records.
test "dsv41 stream 0b: the transient release frees the 240-row MLX scratch back inside the footprint, and window 0 serves decode" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    _ = mlx.applyWiredPolicy();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, testing.io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const depth = max_wide_depth;
    if (bank.n_experts < depth * max_route_ids) return error.TooFewExperts;
    const L: u32 = 0;
    const rows = try a.alloc(u32, bank.layers.len);
    defer a.free(rows);
    @memset(rows, 0);
    var buf: [1 << 16]u8 = undefined;
    const base = try ProbeBox.settled(&buf, 3000);
    const b0 = base.b;
    const s = try Stream.init(a, &bank, .{ .rows = rows, .transient_rows = depth * max_route_ids, .wide_depth = depth, .slot_memory = .{ .mlx = stream }, .transient_release = true });
    defer s.deinit();
    // The prompt's wide reads: five live routes of layer L fill every window with records (no persistent rows).
    var live: [depth]*Route = undefined;
    for (&live, 0..) |*r, w| {
        var ids: [max_route_ids]u16 = undefined;
        for (&ids, 0..) |*e, i| e.* = @intCast(w * max_route_ids + i);
        r.* = try serve(s, L, &ids);
        try testing.expectEqual(@as(u8, @intCast(w)), r.*.window);
    }
    // The GPU reads every row, as the prompt's waves do.
    var sums: [n_components]mlx.mlx_array = @splat(.{});
    for (&sums, s.transient.backing.mlx.arrays) |*x, arr| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum(x, arr, false, stream));
    }
    try evalArrays(&sums);
    for (sums) |x| _ = mlx.mlx_array_free(x);
    for (live) |r| s.release(r);
    _ = mlx.mlx_synchronize(stream);
    const b1 = try ProbeBox.mark(&buf);
    // The release (its allocator check), synchronize, MLX's cache cleared; from the post-fill state (2 s bounds; else one
    // more GPU command and 1 s): the footprint falls by the freed scratch (to within 10 %), and the growth outside the
    // footprint does not rise past 10 % of it.
    var active: [2]usize = .{ 0, 0 };
    _ = mlx.mlx_get_active_memory(&active[0]);
    const freed = try s.releaseTransient();
    _ = mlx.mlx_get_active_memory(&active[1]);
    try testing.expectEqual(transientBytes(s, depth * max_route_ids), freed);
    _ = mlx.mlx_synchronize(stream);
    _ = mlx.mlx_clear_cache();
    const limit: i64 = @intCast(freed / 10);
    const fp_limit: i64 = limit - @as(i64, @intCast(freed));
    const f1 = try ProbeBox.settleFootprint(b1, &buf, 2000, fp_limit);
    const r1 = try ProbeBox.settle(b1, &buf, 2000, limit);
    var r2: ?ProbeBox.Settled = null;
    var f2: ?ProbeBox.Settled = null;
    if (f1.ms == null or r1.ms == null) {
        try ProbeBox.nextCommand(stream);
        f2 = try ProbeBox.settleFootprint(b1, &buf, 1000, fp_limit);
        r2 = try ProbeBox.settle(b1, &buf, 1000, limit);
    }
    const released = r1.ms != null or (r2 != null and r2.?.ms != null);
    const kept = !(f1.ms != null or (f2 != null and f2.?.ms != null));
    // Decode: window 0 and layer L's four rows; the route's misses past them land in window 0 and hold their records.
    rows[L] = 4;
    try s.grow(rows);
    const b2 = try ProbeBox.mark(&buf);
    var ids: [max_route_ids]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(bank.n_experts - 1 - i);
    const r = try serve(s, L, &ids);
    try testing.expectEqual(@as(u8, 0), r.window);
    const geom = &bank.layers[L];
    var in_window0 = true;
    for (r.plan.loadsOf()) |ld| {
        in_window0 = in_window0 and ld.slot < s.layers[L].policy.capacity + max_route_ids;
        const dg = slotDigest(s, L, ld.slot, geom);
        try testing.expectEqualSlices(u8, &bank.digest(L, ld.expert).logical, &dg);
    }
    s.release(r);
    try s.flush();
    const verdict = if (kept) "TransientReleaseKeptFootprint" else if (!released) "TransientReleaseOutsideFootprint" else "inside";
    var l: [5][320]u8 = undefined;
    var ms: [5][24]u8 = undefined;
    std.debug.print("\nTRANSIENT_RELEASE_PROBE {{\"transient_rows\": {d}, \"freed_bytes\": {d}, \"d_active\": {d}, \"window0_rows\": {d}, \"baseline_settle_ms\": {s}, \"filled\": {s}, \"release_from_fill\": {s}, \"footprint_settle_ms\": {s}, \"outside_settle_ms\": {s}, \"after_next_command_from_fill\": {s}, \"after_next_command_footprint_settle_ms\": {s}, \"after_next_command_outside_settle_ms\": {s}, \"grown\": {s}, \"limit\": {d}, \"verdict\": \"{s}\"}}\n", .{
        depth * max_route_ids, freed, active[0] -| active[1], s.transient.rows, ProbeBox.msOf(base.ms, &ms[4]), ProbeBox.line(b0, b1, &l[0]), ProbeBox.line(b1, r1.b, &l[1]), ProbeBox.msOf(f1.ms, &ms[0]), ProbeBox.msOf(r1.ms, &ms[1]),
        if (r2) |x| ProbeBox.line(b1, x.b, &l[2]) else "null", if (f2) |x| ProbeBox.msOf(x.ms, &ms[2]) else "null", if (r2) |x| ProbeBox.msOf(x.ms, &ms[3]) else "null", ProbeBox.line(b0, b2, &l[3]), limit, verdict,
    });
    try testing.expect(in_window0);
    if (kept) return error.TransientReleaseKeptFootprint;
    if (!released) return error.TransientReleaseOutsideFootprint;
}

// DSV41_PHASE0B_MLX=1, inside a guarded window: the GPU reads the slot arrays behind the event gate.
test "dsv41 stream 0b: gated waves over the MLX slot arrays read the landed bytes on the GPU" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const expert_event = sdk_ext.expert.event;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var sb = try SynthBank.open(32);
    defer sb.close();
    const ev = try expert_event.createMetal();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 8, 8 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .slot_memory = .{ .mlx = stream }, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false }, .event = .{ .backend = .{ .metal = ev.object }, .watchdog_ms = 10_000 } });
    defer s.deinit();
    defer expert_io.clearFaults();
    try s.grow(&.{ 8, 8 });
    // Expert 4's read is held 300 ms: the GPU, not the host, waits for it.
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 4).gu_offset / page * page, 5, 300 * std.time.ns_per_ms);
    const ids = [_]u16{ 3, 1, 4, 5, 9 };
    const r = try s.route(0, &ids, &.{});
    const g = (try s.gate(r)).?;
    var refs: [max_route_ids]SlotRef = undefined;
    const rf = s.refsOf(r, &refs);
    var rows_i: [ids.len]i32 = undefined;
    for (rf, &rows_i) |ref, *ri| {
        try testing.expectEqual(BankKind.base, ref.bank);
        ri.* = @intCast(ref.row);
    }
    const bank_arrays = s.bankArrays(0, .base).?;
    const src = [n_components]mlx.mlx_array{ bank_arrays.gate.code, bank_arrays.gate.rout, bank_arrays.gate.rin, bank_arrays.up.code, bank_arrays.up.rout, bank_arrays.up.rin, bank_arrays.down.code, bank_arrays.down.rout, bank_arrays.down.rin };
    var gated: [n_components]mlx.mlx_array = @splat(.{});
    defer for (gated) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // Every wave of the call has landed at the last down value.
    try expert_event.wait(&src, ev, g.down_first + g.n_parts - 1, &.{}, false, stream, &gated);
    const idx = mlx.mlx_array_new_data(&rows_i, &[_]c_int{ids.len}, 1, .int32);
    defer _ = mlx.mlx_array_free(idx);
    var taken: [n_components]mlx.mlx_array = undefined;
    for (&taken, gated) |*t, x| {
        t.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(t, x, idx, 0, stream));
    }
    defer for (taken) |t| {
        _ = mlx.mlx_array_free(t);
    };
    const t0 = std.Io.Timestamp.now(std.testing.io, .boot);
    const vec = mlx.mlx_vector_array_new_data(&taken, taken.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    const ms = @divTrunc(t0.untilNow(std.testing.io, .boot).nanoseconds, std.time.ns_per_ms);
    const geom = &sb.bank.layers[0];
    for (taken, geom.segments) |t, seg| {
        const got = (mlx.mlx_array_data_uint8(t) orelse return error.MlxNoData)[0 .. ids.len * seg.length];
        for (ids, 0..) |e, i| {
            const off = sb.bank.recordOffset(0, e) + seg.offset;
            try testing.expectEqualSlices(u8, sb.image[off..][0..seg.length], got[i * seg.length ..][0..seg.length]);
        }
    }
    std.debug.print("gated MLX slot arrays: GPU gather evaluated after {d} ms, {d} rows x 9 components equal the records\n", .{ ms, ids.len });
    try testing.expect(ms >= 250);
    s.release(r);
    try s.flush();
    try testing.expectEqual(@as(u64, 0), s.stats().gates_forced);
}

// ── Lookahead, pre-read and event gates ──

const la_pool: expert_io.Options = .{ .workers = 2, .staging_bytes = 16384, .tickets = 512 };

/// One score row over 32 experts: `top` in descending order, the rest 0.
fn scoresFor(top: []const u16) [32]f32 {
    var s: [32]f32 = @splat(0);
    for (top, 0..) |e, i| s[e] = @floatFromInt(top.len - i);
    return s;
}

fn waitCounter(s: *Stream, which: expert_io.Counter, at_least: i64) !void {
    var t: u32 = 0;
    while (s.pool.counter(which) < at_least) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn waitWord(s: *const Stream, value: u64) !void {
    const w = s.eventWord().?;
    var t: u32 = 0;
    while (@as(u64, @intCast(@atomicLoad(i64, w, .acquire))) < value) : (t += 1) {
        if (t > 20_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMicroseconds(500), .awake) catch {};
    }
}

/// Routes a call and, as the GPU would, waits on its gates instead of the pool. A gate also opens when its read fails
/// or the watchdog forces it, so before anyone reads the slots: a forced gate is refused by name (GateForced), then
/// every part is settled (no wait once its gate opened; a failed read raises ReadFailed by name).
fn serveGated(s: *Stream, layer: u32, ids: []const u16, scores: []const f32) !*Route {
    const r = try s.route(layer, ids, scores);
    if (try s.gate(r)) |g| {
        try waitWord(s, g.down_first + g.n_parts - 1);
        if (s.pool.counter(.ev_wd_forced) != 0) return error.GateForced;
        for (0..r.n_parts) |i| try s.waitDown(r, @intCast(i));
    }
    return r;
}

test "dsv41 stream: the next layer's predicted records are read ahead and claimed by its route" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false } });
    defer s.deinit();
    try s.grow(&.{ 4, 4 });
    // Layer 0's call predicts layer 1's experts 20 and 21.
    const pred = scoresFor(&.{ 20, 21, 3, 4, 5, 6 });
    const r0 = try s.route(0, &.{ 1, 2 }, &pred);
    try s.waitDown(r0, 0);
    s.release(r0);
    try waitCounter(s, .landed, 2);
    const r1 = try serve(s, 1, &.{ 20, 9, 21 });
    try expectServed(s, &sb, r1, &.{ 20, 9, 21 });
    s.release(r1);
    try s.flush();
    const st = s.stats();
    const rec = sb.bank.layers[1].logical_bytes;
    try testing.expectEqual(@as(u64, 2), st.spec_issued);
    try testing.expectEqual(@as(u64, 2), st.claimed);
    try testing.expectEqual(@as(u64, 4), st.adopt_ranges);
    try testing.expectEqual(2 * rec, st.adopt_bytes);
    // Every record still lands in its slot: read, or copied out of the speculative staging.
    try testing.expectEqual(5 * rec, st.expert_bytes_read);
    // The adopted ranges' copy time is its own part of the read time.
    try testing.expect(st.adopt_copy_seconds > 0 and st.adopt_copy_seconds < st.expert_read_seconds);
}

test "dsv41 stream: pre-read ranges carry a decode call's certain misses to its reads" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 1, .chunks = 1 } });
    defer s.deinit();
    // Prefill routes never pre-read.
    s.release(try serve(s, 0, &.{ 7, 8 }));
    try testing.expectEqual(@as(i64, 0), s.pool.counter(.pre_calls));
    try s.grow(&.{ 4, 4 });
    const ids = [_]u16{ 1, 7, 2, 3, 1, 8 };
    const r = try serve(s, 0, &ids);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    try s.flush();
    // Certain misses 1, 2, 3: one gate/up and one down range each, bound by the submit or read by it.
    try testing.expectEqual(@as(i64, 1), s.pool.counter(.pre_calls));
    try testing.expectEqual(@as(i64, 6), s.pool.counter(.pre_issued));
    try testing.expectEqual(@as(i64, 6), s.pool.counter(.pre_bound) + s.pool.counter(.pre_cancelled));
    try testing.expectEqual(s.pool.counter(.pre_bound), s.pool.counter(.pre_served));
    // A call whose experts are all resident pre-reads nothing.
    s.release(try serve(s, 0, &.{ 7, 8 }));
    try testing.expectEqual(@as(i64, 1), s.pool.counter(.pre_calls));
    try s.flush();
    try testing.expectEqual(5 * sb.bank.layers[0].logical_bytes, s.stats().expert_bytes_read);
}

test "dsv41 stream: gated routes register the gate/up wave, then each part's down wave" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 8, 8 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1 }, .event = .{ .watchdog_ms = 10_000 } });
    defer s.deinit();
    try s.grow(&.{ 8, 8 });
    // Five misses in file order: parts of three and two.
    const ids = [_]u16{ 3, 1, 4, 5, 9 };
    const r = try s.route(0, &ids, &.{});
    try testing.expectEqual(@as(u32, 2), r.n_parts);
    const g = (try s.gate(r)).?;
    try testing.expectEqual(Gates{ .gu = 1, .down_first = 2, .n_parts = 2 }, g);
    try waitWord(s, g.gu);
    for (r.partsOf()) |p| for (0..p.n_reads) |i| try testing.expect(s.pool.result(p.ticket + @as(u32, @intCast(i))).status != .pending);
    try waitWord(s, 3);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    // Values continue across calls; a call that reads nothing is not gated.
    const r2 = try s.route(1, &.{ 2, 6 }, &.{});
    try testing.expectEqual(Gates{ .gu = 4, .down_first = 5, .n_parts = 1 }, (try s.gate(r2)).?);
    try waitWord(s, 5);
    s.release(r2);
    const r3 = try s.route(0, &.{ 3, 9 }, &.{});
    try testing.expectEqual(@as(?Gates, null), try s.gate(r3));
    s.release(r3);
    try s.flush();
    const st = s.stats();
    try testing.expectEqual(@as(u64, 5), st.gates);
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
}

test "dsv41 stream: a gate the watchdog forces fails the stream at the next flush" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false }, .event = .{ .watchdog_ms = 50 } });
    defer s.deinit();
    defer expert_io.clearFaults();
    try s.grow(&.{ 4, 4 });
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 11).gu_offset / page * page, 5, 400 * std.time.ns_per_ms);
    const r = try s.route(0, &.{11}, &.{});
    const g = (try s.gate(r)).?;
    try waitWord(s, g.down_first);
    try testing.expect(s.stats().gates_forced >= 1);
    s.release(r);
    try testing.expectError(error.GateForced, s.route(1, &.{2}, &.{}));
    try testing.expectError(error.StreamFailed, s.route(1, &.{2}, &.{}));
}

test "dsv41 stream: a forced gate fails the flush every round runs before it returns its tokens (an unfilled row read under it never reaches a client)" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 1, .preread = false }, .event = .{ .watchdog_ms = 50 }, .grow_fill = .unfilled });
    defer s.deinit();
    defer expert_io.clearFaults();
    try s.grow(&.{ 4, 4 });
    const page = std.heap.pageSize();
    expert_io.injectFault(sb.bank.spans(0, 11).gu_offset / page * page, 5, 400 * std.time.ns_per_ms);
    const r = try s.route(0, &.{11}, &.{});
    const g = (try s.gate(r)).?;
    try waitWord(s, g.down_first);
    s.release(r);
    // The round's own flush (dspark_loop round: `ex.flush()` before `return .{ .tokens ...}`) raises it by name.
    try testing.expectError(error.GateForced, s.flush());
    try testing.expectError(error.StreamFailed, s.route(1, &.{2}, &.{}));
}

test "dsv41 stream: lookahead options outside the lane's ranges are refused at construction" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const a = testing.allocator;
    const base: Options = .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = la_pool };
    var o = base;
    o.lookahead = .{ .k = 5 };
    try testing.expectError(error.InvalidSelector, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .budget = 5 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .chunks = 3 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .idle_busy = 2 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{ .tau = std.math.nan(f32) };
    try testing.expectError(error.InvalidSelector, Stream.init(a, &sb.bank, o));
    o = base;
    o.event = .{};
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    o.lookahead = .{};
    o.event = .{ .watchdog_ms = 10 };
    try testing.expectError(error.InvalidOptions, Stream.init(a, &sb.bank, o));
    // The tier's values construct.
    o.event = .{};
    const s = try Stream.init(a, &sb.bank, o);
    s.deinit();
}

test "dsv41 stream: a verify trace of 1-8 rows with lookahead, pre-read and gates serves every slot's bytes" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // Verify widths up to 8 rows x top-6 = 48 ids, the transient scratch sized for them.
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 2 }, .event = .{ .watchdog_ms = 10_000 } });
    defer s.deinit();
    try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
    s.release(try serve(s, 0, &.{ 1, 2, 3, 5, 9 }));
    s.release(try serve(s, 1, &.{ 7, 8, 9 }));
    try s.grow(&.{ 6, 4 });
    // Verify forwards of 1..8 rows x top-6 over both layers; layer 0 predicts layer 1.
    var rng = std.Random.DefaultPrng.init(21);
    const rand = rng.random();
    var ids: [48]u16 = undefined;
    var scores: [8 * 32]f32 = undefined;
    var gates: u64 = 0;
    var reads: u64 = 0;
    for (0..60) |step| {
        const layer: u32 = @intCast(step % 2);
        const m = rand.intRangeAtMost(usize, 1, 8);
        const span: u16 = if (step % 7 == 0) 32 else 12;
        for (ids[0 .. 6 * m]) |*e| e.* = rand.intRangeLessThan(u16, 0, span);
        for (scores[0 .. 32 * m]) |*v| v.* = rand.float(f32);
        const pred: []const f32 = if (layer == 0) scores[0 .. 32 * m] else &.{};
        const r = try serveGated(s, layer, ids[0 .. 6 * m], pred);
        try expectServed(s, &sb, r, ids[0 .. 6 * m]);
        if (r.n_parts > 0) gates += r.n_parts + 1;
        reads += readsOf(r);
        s.release(r);
    }
    try s.flush();
    for (0..2) |l| for (0..s.layers[l].policy.capacity + 48) |slot| {
        try testing.expectEqual(@as(u16, 0), s.pinsOf(@intCast(l), @intCast(slot)));
    };
    const st = s.stats();
    try testing.expectEqual(gates, st.gates);
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
    try testing.expectEqual(reads * sb.bank.layers[0].logical_bytes + 8 * sb.bank.layers[0].logical_bytes, st.expert_bytes_read);
    try testing.expect(st.spec_issued > 0 and st.pre_issued > 0);
}

/// The bytes of `rows` rows of the widest layer's record (the transient scratch's row).
fn transientBytes(s: *const Stream, rows: u64) u64 {
    var n: u64 = 0;
    for (s.bank.layers[s.transient_layer].segments) |seg| n += seg.length;
    return rows * n;
}

test "dsv41 stream: the transient release frees the whole scratch with nothing live or held, and the grow allocates decode's window 0" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 5 * 12, .wide_depth = 5, .pool = test_pool, .transient_release = true });
    defer s.deinit();
    // The prompt: two live routes of layer 0 (windows 0 and 1), the first one's persistent slots held for a deferred call.
    const r0 = try serve(s, 0, &.{ 1, 2, 3, 4, 5, 6 });
    const r1 = try serve(s, 0, &.{ 7, 8, 9, 10, 11, 12 });
    try testing.expectEqual(@as(u8, 1), r1.window);
    try testing.expectError(error.RoutesLive, s.releaseTransient());
    try s.holdBase(r0);
    s.release(r0);
    s.release(r1);
    try testing.expectError(error.RoutesLive, s.releaseTransient());
    s.releaseHeld();
    // The grow needs the release first (the bill counts it).
    try testing.expectError(error.TransientNotReleased, s.grow(&.{ 6, 6 }));
    try testing.expectEqual(transientBytes(s, 5 * 12), try s.releaseTransient());
    try testing.expectEqual(@as(u32, 0), s.transient.rows);
    try testing.expectEqual(@as(usize, 0), s.transient_meta.len);
    try testing.expectEqual(@as(u8, 1), s.wide_depth);
    try testing.expectError(error.TransientAlreadyReleased, s.releaseTransient());
    const Off = struct {
        fn release(st: *Stream, out: *?anyerror) void {
            _ = st.releaseTransient() catch |e| {
                out.* = e;
                return;
            };
            out.* = null;
        }
    };
    var got: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, Off.release, .{ s, &got });
    t.join();
    try testing.expectEqual(@as(?anyerror, error.NotInferenceThread), got);
    try s.grow(&.{ 6, 6 });
    try testing.expect(!s.transient_released);
    try testing.expectEqual(@as(u32, 12 + decode_staging_rows), s.transient.rows);
    try testing.expectEqual(@as(usize, 12 + decode_staging_rows), s.transient_meta.len);
    for (s.transient_meta) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
    try testing.expectError(error.AlreadyGrown, s.releaseTransient());
    // Decode: layer 1's misses past its six rows land in window 0 and serve their records.
    const ids = [_]u16{ 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };
    const r = try serve(s, 1, &ids);
    try testing.expectEqual(@as(u8, 0), r.window);
    for (r.plan.loadsOf()) |l| try testing.expect(l.slot < s.layers[1].policy.capacity + 12);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    try s.flush();
}

test "dsv41 stream 0b: REVERSE: two requests through MLX slot rows: the reverse change's frees leave the footprint and the box, the scratch comes back, and request 2 serves request 1's bytes" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    _ = mlx.applyWiredPolicy();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, testing.io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const depth = max_wide_depth;
    if (bank.n_experts < depth * max_route_ids + max_route_ids) return error.TooFewExperts;
    const L: u32 = 0;
    const prompt_rows = try a.alloc(u32, bank.layers.len);
    defer a.free(prompt_rows);
    @memset(prompt_rows, 0);
    prompt_rows[L] = 4;
    const decode_rows = try a.dupe(u32, prompt_rows);
    defer a.free(decode_rows);
    decode_rows[L] = 4 + 24;
    var buf: [1 << 16]u8 = undefined;
    const s = try Stream.init(a, &bank, .{ .rows = prompt_rows, .transient_rows = depth * max_route_ids, .wide_depth = depth, .slot_memory = .{ .mlx = stream }, .transient_release = true });
    defer s.deinit();
    const geom = &bank.layers[L];
    var served: [2][max_route_ids][32]u8 = undefined;
    var l: [3][320]u8 = undefined;
    var ms: [2][24]u8 = undefined;
    for (0..2) |req| {
        // The prompt: every window of layer L filled, the GPU reads every scratch row, then the routes end.
        var live: [depth]*Route = undefined;
        for (&live, 0..) |*r, w| {
            var ids: [max_route_ids]u16 = undefined;
            for (&ids, 0..) |*e, i| e.* = @intCast(w * max_route_ids + i);
            r.* = try serve(s, L, &ids);
        }
        var sums: [n_components]mlx.mlx_array = @splat(.{});
        for (&sums, s.transient.backing.mlx.arrays) |*x, arr| {
            x.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sum(x, arr, false, stream));
        }
        try evalArrays(&sums);
        for (sums) |x| _ = mlx.mlx_array_free(x);
        for (live) |r| s.release(r);
        _ = mlx.mlx_synchronize(stream);
        _ = try s.releaseTransient();
        _ = mlx.mlx_clear_cache();
        try s.grow(decode_rows);
        // Decode: the same misses every request, served into the grown rows and window 0.
        var ids: [max_route_ids]u16 = undefined;
        for (&ids, 0..) |*e, i| e.* = @intCast(bank.n_experts - 1 - i);
        const r = try serve(s, L, &ids);
        for (ids, r.plan.slotsOf(), 0..) |e, slot, i| {
            served[req][i] = slotDigest(s, L, slot, geom);
            try testing.expectEqualSlices(u8, &bank.digest(L, e).logical, &served[req][i]);
        }
        s.release(r);
        try s.flush();
        if (req == 1) break;
        // The reverse change: synchronize, the frees, the cache clear; the footprint falls by them (within 10 %) and the
        // pages outside it do not rise (10 %); only then the scratch.
        _ = mlx.mlx_synchronize(stream);
        const b1 = try ProbeBox.mark(&buf);
        const freed = try s.shrink(prompt_rows);
        try testing.expectEqual(transientBytes(s, 24 + max_route_ids + decode_staging_rows), freed);
        _ = mlx.mlx_synchronize(stream);
        _ = mlx.mlx_clear_cache();
        const limit: i64 = @intCast(freed / 10);
        const f1 = try ProbeBox.settleFootprint(b1, &buf, 2000, limit - @as(i64, @intCast(freed)));
        const r1 = try ProbeBox.settle(b1, &buf, 2000, limit);
        const regrown = try s.regrowTransient();
        const b2 = try ProbeBox.mark(&buf);
        std.debug.print("\nREVERSE_PHASE_PROBE {{\"freed_bytes\": {d}, \"regrown_bytes\": {d}, \"released\": {s}, \"footprint_settle_ms\": {s}, \"outside_settle_ms\": {s}, \"regrown\": {s}, \"limit\": {d}}}\n", .{
            freed, regrown, ProbeBox.line(b1, r1.b, &l[0]), ProbeBox.msOf(f1.ms, &ms[0]), ProbeBox.msOf(r1.ms, &ms[1]), ProbeBox.line(b1, b2, &l[1]), limit,
        });
        if (f1.ms == null) return error.ReverseKeptFootprint;
        if (r1.ms == null) return error.ReverseOutsideFootprint;
        try testing.expectEqual(transientBytes(s, depth * max_route_ids), regrown);
    }
    for (served[0], served[1]) |x, y| try testing.expectEqualSlices(u8, &x, &y);
}

test "dsv41 stream: after the transient release, decode with the served lookahead, pre-reads and gates loads only window 0" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // The served decode configuration (Lookahead{}: k 8, tau inf, budget 2, 4 chunks, pre-read; event gates) at depth 5.
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .transient_rows = 5 * max_route_ids, .wide_depth = 5, .pool = la_pool, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = true });
    defer s.deinit();
    // The prompt fills all five windows of layer 0 (five live routes), then releases them.
    var live: [5]*Route = undefined;
    for (&live, 0..) |*r, w| {
        var pids: [6]u16 = undefined;
        for (&pids, 0..) |*e, i| e.* = @intCast(w * 6 + i);
        r.* = try serve(s, 0, &pids);
        try testing.expectEqual(@as(u8, @intCast(w)), r.*.window);
    }
    for (live) |r| s.release(r);
    _ = try s.releaseTransient();
    try s.grow(&.{ 6, 4 });
    try testing.expectEqual(@as(u32, max_route_ids + decode_staging_rows), s.transient.rows);
    // Verify forwards of 1..8 rows x top-6; layer 0's scores favour layer 1's next ids, so its reads are claimed.
    var rng = std.Random.DefaultPrng.init(1616);
    const rand = rng.random();
    var ids: [2][48]u16 = undefined;
    var scores: [8 * 32]f32 = undefined;
    for (0..40) |_| {
        const m = [2]usize{ rand.intRangeAtMost(usize, 1, 8), rand.intRangeAtMost(usize, 1, 8) };
        for (0..2) |layer| for (ids[layer][0 .. 6 * m[layer]]) |*e| {
            e.* = rand.intRangeLessThan(u16, 0, 32);
        };
        for (0..m[0]) |row| for (scores[row * 32 ..][0..32], 0..) |*v, e| {
            v.* = if (std.mem.indexOfScalar(u16, ids[1][0 .. 6 * m[1]], @intCast(e)) != null) 1 + rand.float(f32) else rand.float(f32) / 2;
        };
        for (0..2) |layer| {
            const n = 6 * m[layer];
            const pred: []const f32 = if (layer == 0) scores[0 .. 32 * m[0]] else &.{};
            const r = try serveGated(s, @intCast(layer), ids[layer][0..n], pred);
            // No other route is live or awaiting its flush, so every load (demand, pre-read bound at submit, or
            // adopted from the speculative staging) lands in a persistent row or window 0.
            try testing.expectEqual(@as(u8, 0), r.window);
            try testing.expectEqual(@as(u8, 0), s.n_released);
            var n_live: u32 = 0;
            for (&s.routes) |*o| n_live += @intFromBool(o.state != .free);
            try testing.expectEqual(@as(u32, 1), n_live);
            const cap = s.layers[layer].policy.capacity;
            for (r.plan.loadsOf()) |l| try testing.expect(l.slot < cap + max_route_ids);
            try expectServed(s, &sb, r, ids[layer][0..n]);
            s.release(r);
        }
    }
    try s.flush();
    const st = s.stats();
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
    try testing.expect(st.spec_issued > 0 and st.pre_issued > 0 and (st.claimed > 0 or st.adopt_ranges > 0));
}

test "dsv41 stream: three requests: each prompt, phase change, decode and reverse change serve every slot's bytes; the reverse change restores the prompt configuration (a cancelled request's routes included)" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .transient_rows = 5 * max_route_ids, .wide_depth = 5, .pool = la_pool, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = true });
    defer s.deinit();
    const prompt_bytes = s.promptTransientBytes();
    try testing.expectEqual(transientBytes(s, 5 * max_route_ids), prompt_bytes);
    var rng = std.Random.DefaultPrng.init(777);
    const rand = rng.random();
    for (0..3) |req| {
        // The prompt: five live routes of layer 0 in their windows (the same prompt every request), layer 1 after.
        try testing.expectEqual(Phase.prefill, s.phase);
        try testing.expectEqual(@as(u8, 5), s.wide_depth);
        var live: [5]*Route = undefined;
        for (&live, 0..) |*r, w| {
            var pids: [6]u16 = undefined;
            for (&pids, 0..) |*e, i| e.* = @intCast(w * 6 + i);
            r.* = try serve(s, 0, &pids);
            try testing.expectEqual(@as(u8, @intCast(w)), r.*.window);
            try expectServed(s, &sb, r.*, &pids);
        }
        for (live) |r| s.release(r);
        _ = try s.releaseTransient();
        try s.grow(&.{ 10, 9 });
        var ids: [48]u16 = undefined;
        for (0..12) |_| for (0..2) |layer| {
            const n = 6 * rand.intRangeAtMost(usize, 1, 8);
            for (ids[0..n]) |*e| e.* = rand.intRangeLessThan(u16, 0, 32);
            const r = try serveGated(s, @intCast(layer), ids[0..n], &.{});
            try expectServed(s, &sb, r, ids[0..n]);
            s.release(r);
        };
        // Request 1 is cancelled mid-forward: a route stays live (never released) into the reverse change.
        if (req == 1) _ = try serveGated(s, 1, &.{ 3, 4, 5, 6, 7, 8 }, &.{});
        // Freed: the grown rows (6 + 5) and decode's window 0.
        const record = prompt_bytes / (5 * max_route_ids);
        try testing.expectEqual((6 + 5 + max_route_ids + decode_staging_rows) * record, try s.shrink(&.{ 4, 4 }));
        try testing.expectEqual(Phase.prefill, s.phase);
        try testing.expect(s.transient_released);
        for (s.layers) |*ls| {
            try testing.expectEqual(@as(u32, 4), ls.policy.capacity);
            // Residents forgotten (the next prompt's schedule equals the first's).
            try testing.expectEqual(@as(u32, 0), ls.policy.occupancy);
            try testing.expect(ls.ext == null);
            for (ls.meta[4..]) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
            for (ls.meta[0..4]) |m| try testing.expectEqual(@as(u16, 0), m.pins);
        }
        for (&s.routes) |*r| try testing.expect(r.state == .free);
        try testing.expectError(error.NotGrown, s.shrink(&.{ 4, 4 }));
        try testing.expectEqual(prompt_bytes, try s.regrowTransient());
        try testing.expectEqual(@as(u32, 5 * max_route_ids), s.transient.rows);
        for (s.transient_meta) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
        try testing.expectError(error.TransientNotReleased, s.regrowTransient());
    }
    try testing.expectEqual(@as(u64, 0), s.stats().gates_forced);
}

test "dsv41 stream: a request cancelled in its prompt phase: its live and held routes are settled at its end, and the next prompt routes" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 2 * 12, .wide_depth = 2, .pool = test_pool, .transient_release = true });
    defer s.deinit();
    const r0 = try serve(s, 0, &.{ 1, 2, 3, 4, 5, 6 });
    try s.holdBase(r0);
    s.release(r0);
    _ = try serve(s, 0, &.{ 7, 8, 9, 10, 11, 12 }); // never released: the cancel
    try s.settleRoutes();
    for (&s.routes) |*r| try testing.expect(r.state == .free);
    try testing.expectEqual(@as(usize, 0), s.held_base.items.len);
    for (s.layers[0].meta) |m| try testing.expectEqual(@as(u16, 0), m.pins);
    const ids = [_]u16{ 1, 7, 13, 14, 15, 16 };
    const r = try serve(s, 0, &ids);
    try expectServed(s, &sb, r, &ids);
    s.release(r);
    try s.flush();
}

test "dsv41 stream: grow fill unfilled: no route reads a grown row before its record lands in it (rows poisoned after the grow)" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    // The served decode configuration (lookahead, pre-reads, gates) at depth 5, the transient release on, unfilled grow.
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 2 }, .transient_rows = 5 * max_route_ids, .wide_depth = 5, .pool = la_pool, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = true, .grow_fill = .unfilled });
    defer s.deinit();
    try testing.expectEqual(GrowFill.unfilled, s.grow_fill);
    for (0..5) |w| {
        var pids: [6]u16 = undefined;
        for (&pids, 0..) |*e, i| e.* = @intCast(w * 6 + i);
        s.release(try serve(s, 0, &pids));
    }
    _ = try s.releaseTransient();
    try s.grow(&.{ 12, 10 });
    // Every row the grow added (each layer's ext and decode's window 0) holds garbage, as an unfilled buffer may; the
    // grown slots are empty by the state machine, so a kernel can reach a row only after a load wrote its record.
    for (s.layers, 0..) |*ls, l| {
        const e = &ls.ext.?;
        for (0..e.rows) |r| for (0..n_components) |c| @memset(e.row(@enumFromInt(c), @intCast(r)), 0xA5);
        for (ls.meta[ls.base.rows..ls.policy.capacity]) |m| try testing.expect(m.state != .ready);
        _ = l;
    }
    for (0..s.transient.rows) |r| for (0..n_components) |c| @memset(s.transient.row(@enumFromInt(c), @intCast(r)), 0xA5);
    for (s.transient_meta) |m| try testing.expect(m.state != .ready);
    var rng = std.Random.DefaultPrng.init(4242);
    const rand = rng.random();
    var ids: [2][48]u16 = undefined;
    var scores: [8 * 32]f32 = undefined;
    for (0..60) |_| {
        const m = [2]usize{ rand.intRangeAtMost(usize, 1, 8), rand.intRangeAtMost(usize, 1, 8) };
        for (0..2) |layer| for (ids[layer][0 .. 6 * m[layer]]) |*e| {
            e.* = rand.intRangeLessThan(u16, 0, 32);
        };
        for (0..m[0]) |row| for (scores[row * 32 ..][0..32], 0..) |*v, e| {
            v.* = if (std.mem.indexOfScalar(u16, ids[1][0 .. 6 * m[1]], @intCast(e)) != null) 1 + rand.float(f32) else rand.float(f32) / 2;
        };
        for (0..2) |layer| {
            const n = 6 * m[layer];
            const pred: []const f32 = if (layer == 0) scores[0 .. 32 * m[0]] else &.{};
            const r = try serveGated(s, @intCast(layer), ids[layer][0..n], pred);
            // Every slot the route hands the kernels (hits, demand loads, pre-reads, adopted reads) holds its record.
            try expectServed(s, &sb, r, ids[layer][0..n]);
            s.release(r);
        }
    }
    try s.flush();
    // The grown rows were used: loads landed past the prompt rows.
    try testing.expect(s.stats().spec_issued > 0);
}

/// One replay of the phase-2 fixture's layers 13 and 14 (served lookahead 8:inf:2, 4 chunks, pre-read, gates) at the
/// served depth 5, with the transient release on its route or not: each decode route's signature (layer, window, hits,
/// loads with their slots and reads) into `sigs`, every served slot sha256-checked; returns the stream's Stats.
fn replayLookahead(a: std.mem.Allocator, bank: *const expert_bank.Bank, f: anytype, sfd: std.c.fd_t, release: bool, sigs: *std.ArrayList(u64)) !Stats {
    const L: u32 = 13;
    var rows: [40]u32 = @splat(0);
    rows[L] = 3;
    rows[L + 1] = 3;
    const s = try Stream.init(a, bank, .{ .rows = &rows, .max_route_ids = 6, .transient_rows = 5 * 6, .wide_depth = 5, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = release });
    defer s.deinit();
    const geom = &bank.layers[L];
    for ([_]u32{ L, L + 1 }) |l| {
        var seed: [3]u16 = undefined;
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        @memcpy(&seed, sorted[0..3]);
        try s.seedPrefill(l, &seed);
        s.release(try serve(s, l, &seed));
    }
    rows[L] = 4;
    rows[L + 1] = 4;
    if (release) _ = try s.releaseTransient();
    try s.grow(&rows);
    var routes: u64 = 0;
    var score_row: [384]f32 = undefined;
    var raw: [384 * 4]u8 = undefined;
    var at: u64 = 0;
    var cycle: usize = 0;
    outer: while (cycle < f.rows.len) : (cycle += 1) {
        const m = f.rows[cycle];
        const c13 = f.calls[cycle * f.layers + L];
        const c14 = f.calls[cycle * f.layers + L + 1];
        for (0..m) |r| {
            if (routes >= 120) break :outer;
            const off = (at + L * m + r) * 384 * 4;
            if (std.c.pread(sfd, &raw, raw.len, @intCast(off)) != raw.len) return error.ShortRead;
            for (&score_row, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
            for ([_]struct { l: u32, ids: []const u16, scores: []const f32 }{
                .{ .l = L, .ids = c13.ids[6 * r ..][0..6], .scores = &score_row },
                .{ .l = L + 1, .ids = c14.ids[6 * r ..][0..6], .scores = &.{} },
            }) |call| {
                const rt = try serveGated(s, call.l, call.ids, call.scores);
                var h = std.hash.Wyhash.init(call.l);
                h.update(std.mem.asBytes(&rt.window));
                h.update(std.mem.asBytes(&rt.plan.n_hits));
                for (rt.plan.loadsOf(), rt.reads[0..rt.plan.n_loads]) |ld, rd| {
                    h.update(std.mem.asBytes(&ld.expert));
                    h.update(std.mem.asBytes(&ld.slot));
                    h.update(std.mem.asBytes(&ld.persistent));
                    h.update(std.mem.asBytes(&rd));
                }
                try sigs.append(a, h.final());
                for (rt.plan.slotsOf(), call.ids) |slot, e| {
                    const d = slotDigest(s, call.l, slot, geom);
                    try testing.expectEqualSlices(u8, &bank.digest(call.l, e).logical, &d);
                }
                routes += 1;
                s.release(rt);
            }
        }
        at += @as(u64, m) * (f.layers - 1);
    }
    try s.flush();
    return s.stats();
}

// DSV41_BANK=<bank dir> DSV41_PHASE2_FIXTURE=<json from R/exl3/runtime/dump_phase2_lookahead_fixture.py>: SERVED16's decode
// read regression against the release. The release frees only the transient scratch, so the decode routes and reads
// must be the same with and without it: every route's plan and reads, and the read pool's lookahead and pre-read counts.
test "dsv41 stream: the recorded lookahead trace on the real bank routes and reads the same with and without the transient release" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_PHASE2_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(16 << 20));
    defer a.free(text);
    const Fix = struct { layers: u32, experts: u32, rows: []const u32, resident0: []const []const u16, scores_file: []const u8, calls: []const TraceCall };
    const parsed = try std.json.parseFromSlice(Fix, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var pbuf: [1024]u8 = undefined;
    const spath = try std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}", .{ std.fs.path.dirname(fixture) orelse ".", parsed.value.scores_file }, 0);
    const sfd = std.c.open(spath.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (sfd < 0) return error.OpenFailed;
    defer _ = std.c.close(sfd);
    var sig: [2]std.ArrayList(u64) = .{ .empty, .empty };
    defer for (&sig) |*x| x.deinit(a);
    var st: [2]Stats = undefined;
    var ms: [2]i64 = undefined;
    for (0..2) |i| {
        const t0 = std.Io.Timestamp.now(io, .boot);
        st[i] = try replayLookahead(a, &bank, parsed.value, sfd, i == 1, &sig[i]);
        ms[i] = @intCast(@divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms));
    }
    for (st, ms, [_][]const u8{ "release off", "release on" }) |x, t, name| std.debug.print(
        "replay ({s}): {d} routes; misses {d} hits {d} evictions {d} persistent {d} transient {d} skipped {d}; {d} B read in {d} preadv; lookahead: issued {d} landed {d} claimed {d} adopted {d} ranges / {d} B, spec {d} B; pre-read issued {d} served {d} expired {d}; gates {d} forced {d}; {d} ms\n",
        .{ name, x.route_calls, x.expert_cache_misses, x.expert_cache_hits, x.expert_cache_evictions, x.persistent_loads, x.transient_loads, x.loads_skipped, x.expert_bytes_read, x.preadv_calls, x.spec_issued, x.spec_landed, x.claimed, x.adopt_ranges, x.adopt_bytes, x.spec_bytes, x.pre_issued, x.pre_served, x.pre_expired, x.gates, x.gates_forced, t },
    );
    // The same routes, plans and reads (window 0 throughout), and the same reads issued, claimed and adopted.
    try testing.expectEqualSlices(u64, sig[0].items, sig[1].items);
    inline for (.{ "route_calls", "expert_cache_misses", "expert_cache_hits", "expert_cache_evictions", "persistent_loads", "transient_loads", "loads_skipped", "expert_bytes_read", "preadv_calls", "spec_issued", "claimed", "pre_issued", "gates" }) |k|
        try testing.expectEqual(@field(st[0], k), @field(st[1], k));
    try testing.expectEqual(@as(u64, 0), st[0].gates_forced + st[1].gates_forced);
}

/// The phase-2 fixture's calls for one layer as M = 1 routes, and the P1 scores of each row.
const TraceCall = struct { ids: []const u16, pre: []const u16, sel: []const []const u16, cand: []const []const u16 };

// DSV41_BANK=<bank dir> DSV41_PHASE2_FIXTURE=<json from R/exl3/runtime/dump_phase2_lookahead_fixture.py>
test "dsv41 stream: a two-layer recorded trace with lookahead and gates on the real bank serves every slot's bytes" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_PHASE2_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(16 << 20));
    defer a.free(text);
    const Fix = struct { layers: u32, experts: u32, rows: []const u32, resident0: []const []const u16, scores_file: []const u8, calls: []const TraceCall };
    const parsed = try std.json.parseFromSlice(Fix, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var pbuf: [1024]u8 = undefined;
    const spath = try std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}", .{ std.fs.path.dirname(fixture) orelse ".", f.scores_file }, 0);
    const sfd = std.c.open(spath.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (sfd < 0) return error.OpenFailed;
    defer _ = std.c.close(sfd);

    // Layers 13 and 14 at 3 -> 4 rows, 6 transient rows, the tier's lookahead (8:inf:2, 4 chunks) + event gates.
    const L: u32 = 13;
    var rows: [40]u32 = @splat(0);
    rows[L] = 3;
    rows[L + 1] = 3;
    const s = try Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = 6, .transient_rows = 6, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 } });
    defer s.deinit();
    const geom = &bank.layers[L];
    for ([_]u32{ L, L + 1 }) |l| {
        var seed: [3]u16 = undefined;
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        @memcpy(&seed, sorted[0..3]);
        try s.seedPrefill(l, &seed);
        s.release(try serve(s, l, &seed));
    }
    rows[L] = 4;
    rows[L + 1] = 4;
    try s.grow(&rows);

    var served: u64 = 0;
    var routes: u64 = 0;
    var score_row: [384]f32 = undefined;
    var raw: [384 * 4]u8 = undefined;
    var at: u64 = 0; // scored rows before this cycle
    var cycle: usize = 0;
    outer: while (cycle < f.rows.len) : (cycle += 1) {
        const m = f.rows[cycle];
        const c13 = f.calls[cycle * f.layers + L];
        const c14 = f.calls[cycle * f.layers + L + 1];
        for (0..m) |r| {
            if (routes >= 40) break :outer;
            // Row r of layer 13's call predicts layer 14 (P1 scores of that row).
            const off = (at + L * m + r) * 384 * 4;
            if (std.c.pread(sfd, &raw, raw.len, @intCast(off)) != raw.len) return error.ShortRead;
            for (&score_row, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
            for ([_]struct { l: u32, ids: []const u16, scores: []const f32 }{
                .{ .l = L, .ids = c13.ids[6 * r ..][0..6], .scores = &score_row },
                .{ .l = L + 1, .ids = c14.ids[6 * r ..][0..6], .scores = &.{} },
            }) |call| {
                const rt = try serveGated(s, call.l, call.ids, call.scores);
                for (rt.plan.slotsOf(), call.ids) |slot, e| {
                    const d = slotDigest(s, call.l, slot, geom);
                    try testing.expectEqualSlices(u8, &bank.digest(call.l, e).logical, &d);
                    served += 1;
                }
                routes += 1;
                s.release(rt);
            }
        }
        at += @as(u64, m) * (f.layers - 1);
    }
    try s.flush();
    const st = s.stats();
    const ru = std.posix.getrusage(std.c.rusage.SELF);
    std.debug.print(
        "real bank layers {d}+{d}: {d} gated routes, {d} served slots sha256-checked; misses {d} skipped {d}; {d} B landed ({d} preadv); lookahead: issued {d} landed {d} claimed {d} adopted {d} ranges / {d} B, spec {d} B; pre-read issued {d} served {d} expired {d}; gates {d} forced {d}; {d} ms total; peak RSS {d} B\n",
        .{ L, L + 1, st.route_calls, served, st.expert_cache_misses, st.loads_skipped, st.expert_bytes_read, st.preadv_calls, st.spec_issued, st.spec_landed, st.claimed, st.adopt_ranges, st.adopt_bytes, st.spec_bytes, st.pre_issued, st.pre_served, st.pre_expired, st.gates, st.gates_forced, @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms), ru.maxrss },
    );
    try testing.expectEqual(@as(u64, 0), st.gates_forced);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped) * geom.logical_bytes, st.expert_bytes_read);
    try testing.expect(st.spec_issued > 0 and st.pre_issued > 0 and st.gates > 0);
}

test "dsv41 stream: wide depth 2 holds two prefill routes of a layer in disjoint slots and transient windows" {
    var sb = try SynthBank.open(128);
    defer sb.close();
    // The transient rows must hold both windows.
    try testing.expectError(error.InvalidOptions, Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = test_pool, .wide_depth = 2 }));
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = test_pool, .wide_depth = 2, .transient_rows = 2 * max_route_ids });
    defer s.deinit();
    var a_ids: [max_route_ids]u16 = undefined;
    var b_ids: [max_route_ids]u16 = undefined;
    for (&a_ids, &b_ids, 0..) |*x, *y, i| {
        x.* = @intCast(i);
        y.* = @intCast(max_route_ids + i);
    }
    // Group A fills the 16 persistent rows and 32 transient rows of window 0.
    const ra = try serve(s, 0, &a_ids);
    try testing.expectEqual(@as(u8, 0), ra.window);
    // Group B while A is live: no victim among A's slots (they are held), window 1's rows.
    const rb = try serve(s, 0, &b_ids);
    try testing.expectEqual(@as(u8, 1), rb.window);
    try testing.expectEqual(@as(u32, 0), rb.plan.n_persistent);
    for (rb.plan.loadsOf()) |l| {
        try testing.expect(!l.persistent);
        try testing.expect(l.slot >= 16 + max_route_ids and l.slot < 16 + 2 * max_route_ids);
    }
    try expectServed(s, &sb, ra, &a_ids);
    try expectServed(s, &sb, rb, &b_ids);
    s.release(ra);
    s.release(rb);
    try s.flush();
    // After both went back, a route takes window 0 again and may evict their persistent rows.
    var c_ids: [8]u16 = .{ 100, 101, 102, 103, 104, 105, 106, 107 };
    const rc = try serve(s, 0, &c_ids);
    try testing.expectEqual(@as(u8, 0), rc.window);
    try testing.expectEqual(@as(u32, 8), rc.plan.n_persistent);
    try expectServed(s, &sb, rc, &c_ids);
    const st = s.stats();
    try testing.expectEqual(@as(u64, (2 * max_route_ids + 8) * sb.bank.layers[0].logical_bytes), st.expert_bytes_read);
    s.release(rc);
    try s.flush();
}

test "dsv41 stream: wide depth 1 plans as before (one window, a live route's slots are not held)" {
    var sb = try SynthBank.open(128);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 16, 16 }, .pool = test_pool });
    defer s.deinit();
    var ids: [8]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const r = try serve(s, 0, &ids);
    try testing.expectEqual(@as(u8, 0), r.window);
    for (r.plan.loadsOf()) |l| try testing.expect(l.persistent and l.slot < 16);
    s.release(r);
    try s.flush();
}

// #23: per-layer decode rows change only which records stay resident. The same prompt and decode trace through a
// stream grown uniform and one grown per layer (same total): every routed id is served from a slot holding its record
// in both, so the routed math reads the same bytes; only the hit / miss split moves.
test "dsv41 stream: per-layer decode rows serve every routed id its record, as uniform rows do, on the same trace" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    var st: [2]Stats = undefined;
    for ([_][2]u32{ .{ 8, 8 }, .{ 12, 4 } }, 0..) |grown, arm| {
        const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .pool = la_pool, .lookahead = .{ .k = 6, .budget = 2, .chunks = 2 }, .event = .{ .watchdog_ms = 10_000 } });
        defer s.deinit();
        try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
        s.release(try serve(s, 0, &.{ 1, 2, 3, 5, 9 }));
        s.release(try serve(s, 1, &.{ 7, 8, 9 }));
        try s.grow(&grown);
        try testing.expectEqual(grown[0], s.layers[0].policy.capacity);
        try testing.expectEqual(grown[1], s.layers[1].policy.capacity);
        var rng = std.Random.DefaultPrng.init(23);
        const rand = rng.random();
        var ids: [48]u16 = undefined;
        var scores: [8 * 32]f32 = undefined;
        for (0..80) |step| {
            const layer: u32 = @intCast(step % 2);
            const m = rand.intRangeAtMost(usize, 1, 8);
            // Layer 0 routes over all 32 experts, layer 1 over 10: the wide layer is the one that gains rows.
            const span: u16 = if (layer == 0) 32 else 10;
            for (ids[0 .. 6 * m]) |*e| e.* = rand.intRangeLessThan(u16, 0, span);
            for (scores[0 .. 32 * m]) |*v| v.* = rand.float(f32);
            const pred: []const f32 = if (layer == 0) scores[0 .. 32 * m] else &.{};
            const r = try serveGated(s, layer, ids[0 .. 6 * m], pred);
            try expectServed(s, &sb, r, ids[0 .. 6 * m]);
            s.release(r);
        }
        try s.flush();
        st[arm] = s.stats();
        try testing.expectEqual(@as(u64, 0), st[arm].gates_forced);
    }
    try testing.expectEqual(st[0].route_calls, st[1].route_calls);
    try testing.expectEqual(st[0].expert_cache_hits + st[0].expert_cache_misses, st[1].expert_cache_hits + st[1].expert_cache_misses);
    std.debug.print("\nper-layer rows: misses uniform {d}, per-layer {d}\n", .{ st[0].expert_cache_misses, st[1].expert_cache_misses });
}

test "dsv41 stream: a request's start zeroes every layer's prompt counts, and only them" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
    try s.seedPrefill(1, &.{7});
    s.release(try serve(s, 0, &.{ 1, 2, 3 }));
    try testing.expectEqual(@as(u32, 2), s.promptCounts(0)[1]);
    const resident = s.layers[0].policy.occupancy;
    s.resetPromptCounts();
    for (0..2) |l| for (s.promptCounts(@intCast(l))) |c| try testing.expectEqual(@as(u32, 0), c);
    try testing.expectEqual(resident, s.layers[0].policy.occupancy);
}

test "dsv41 stream: option (b)'s rule keeps the total, the floor and the cap, and equal misses keep every layer uniform" {
    var cands: [4 * 2 * 3]PoolCand = undefined;
    var out: [4]u32 = undefined;
    const floor = [_]u32{ 7, 7, 7, 7 };
    poolRows(&cands, &.{ 5, 5, 5, 5 }, &floor, 10, 3, 32, &out);
    try testing.expectEqualSlices(u32, &.{ 10, 10, 10, 10 }, &out);
    poolRows(&cands, &.{ 0, 0, 0, 0 }, &floor, 10, 3, 32, &out);
    try testing.expectEqualSlices(u32, &.{ 10, 10, 10, 10 }, &out);
    // One layer misses: it takes the cap (13), the rest give rows by row then layer.
    poolRows(&cands, &.{ 0, 90, 0, 0 }, &floor, 10, 3, 32, &out);
    try testing.expectEqual(@as(u32, 13), out[1]);
    var total: u32 = 0;
    for (out) |r| {
        total += r;
        try testing.expect(r >= 7 and r <= 13);
    }
    try testing.expectEqual(@as(u32, 40), total);
    // m / r^3: twice the misses is worth 2^(1/3) more rows, never all of them.
    poolRows(&cands, &.{ 10, 20, 10, 10 }, &floor, 10, 3, 32, &out);
    try testing.expect(out[1] > 10 and out[1] < 13);
}

// Option (b) on the stream: window 0 carries the pool rows (the same rows the uniform grow puts in the ext banks), each
// routed id is served its record before and after the re-plan, the re-plan moves rows to the layer that missed, and the
// rows allocated are the uniform grow's: the bill's (layers x U + window 0) by construction.
test "dsv41 stream: decode_first16 re-owns pool rows of window 0 at its cycle and serves every id its record" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    var st: [2]Stats = undefined;
    for ([_]?DecodePool{ null, .{ .per_layer = 2, .from_cycle = 2, .at_cycle = 4 } }, 0..) |dp, arm| {
        const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .transient_release = true, .decode_pool = dp });
        defer s.deinit();
        try s.seedPrefill(0, &.{ 1, 2, 3, 1 });
        s.release(try serve(s, 0, &.{ 1, 2, 3, 5 }));
        s.release(try serve(s, 1, &.{ 7, 8, 9 }));
        _ = try s.releaseTransient();
        try s.grow(&.{ 8, 8 });
        // The same rows either way: ext + window 0 == layers x (U - P) + the scratch window.
        var allocated: u32 = s.transient.rows;
        for (s.layers) |ls| allocated += if (ls.ext) |e| e.rows else 0;
        try testing.expectEqual(@as(u32, 2 * (8 - 4) + 12 + decode_staging_rows), allocated);
        var rng = std.Random.DefaultPrng.init(17);
        const rand = rng.random();
        var ids: [12]u16 = undefined;
        for (1..9) |cycle| {
            for (0..2) |l| {
                const layer: u32 = @intCast(l);
                const n = rand.intRangeAtMost(usize, 2, 8);
                // Layer 0 reuses 6 experts, layer 1 roams 20.
                for (ids[0..n]) |*e| e.* = if (layer == 0) rand.intRangeLessThan(u16, 0, 6) else rand.intRangeLessThan(u16, 10, 30);
                const r = try serve(s, layer, ids[0..n]);
                try expectServed(s, &sb, r, ids[0..n]);
                s.release(r);
            }
            try s.flush();
            try s.cycleEnd();
            if (dp != null and cycle < 4) try testing.expect(s.poolReplan() == null);
        }
        st[arm] = s.stats();
        for (0..2) |l| for (0..s.layers[l].policy.capacity + 12) |slot| {
            try testing.expectEqual(@as(u16, 0), s.pinsOf(@intCast(l), @intCast(slot)));
        };
        if (dp) |_| {
            const rp = s.poolReplan().?;
            try testing.expectEqual(@as(u32, 16), rp.rows[0] + rp.rows[1]);
            try testing.expect(rp.rows[1] > rp.rows[0]);
            try testing.expect(rp.misses[1] > rp.misses[0]);
            for (rp.rows, 0..) |r, l| try testing.expectEqual(r, s.layers[l].policy.capacity);
            // Every pool row has exactly one owner.
            var seen: [4]bool = @splat(false);
            for (s.layers) |ls| for (ls.pool[0 .. ls.policy.capacity - ls.pool_lo]) |pr| {
                try testing.expect(!seen[pr]);
                seen[pr] = true;
            };
            for (seen) |x| try testing.expect(x);
        }
    }
    try testing.expectEqual(st[0].route_calls, st[1].route_calls);
    try testing.expectEqual(st[0].expert_cache_hits + st[0].expert_cache_misses, st[1].expert_cache_hits + st[1].expert_cache_misses);
    std.debug.print("\ndecode_first16 misses: uniform {d}, pool {d}\n", .{ st[0].expert_cache_misses, st[1].expert_cache_misses });
}

test "dsv41 stream: decode_first16 is refused without the transient release or with a zero pool" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    try testing.expectError(error.InvalidOptions, Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .decode_pool = .{} }));
    try testing.expectError(error.InvalidOptions, Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .transient_release = true, .decode_pool = .{ .per_layer = 0 } }));
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .transient_release = true, .decode_pool = .{ .per_layer = 2 } });
    defer s.deinit();
    _ = try s.releaseTransient();
    // Decode rows must hold the pool above the prompt rows.
    try testing.expectError(error.InvalidRows, s.grow(&.{ 5, 8 }));
}

// The re-plan's compaction: a donor's surviving resident in its top pool slot is relabelled below its new capacity and
// is still served its own record from its own row (the pool table moves with it).
test "dsv41 stream: decode_first16's re-plan relabels a donor's surviving pool resident with its row" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .transient_release = true, .decode_pool = .{ .per_layer = 2, .from_cycle = 2, .at_cycle = 4 } });
    defer s.deinit();
    s.release(try serve(s, 0, &.{ 1, 2, 3, 4 }));
    s.release(try serve(s, 1, &.{ 1, 2, 3, 4 }));
    _ = try s.releaseTransient();
    try s.grow(&.{ 8, 8 });
    // Layer 0: every decode row filled, then its top pool slot's expert used last.
    s.release(try serve(s, 0, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }));
    try s.flush();
    const ls = &s.layers[0];
    try testing.expectEqual(@as(u32, 8), ls.policy.occupancy);
    const top = ls.policy.slot_to_expert[7];
    s.release(try serve(s, 0, &.{top}));
    try s.flush();
    // Layer 1 missed twice as often: rows 7 / 9 (row 9 of layer 1 beats row 8 of layer 0; row 7 of layer 0 beats row 10).
    const d = &s.dpool.?;
    d.first[0] = 10;
    d.first[1] = 20;
    try s.replanPool();
    try testing.expectEqualSlices(u32, &.{ 7, 9 }, d.rows);
    try testing.expectEqual(@as(u32, 7), ls.policy.capacity);
    try testing.expectEqual(@as(u32, 6), ls.policy.expert_to_slot[top]);
    const r = try serve(s, 0, &.{top});
    try testing.expectEqual(@as(u32, 1), r.plan.n_hits);
    try expectServed(s, &sb, r, &.{top});
    s.release(r);
    // Layer 1's new slot is empty and fills by a read.
    const r1 = try serve(s, 1, &.{ 20, 21, 22, 23, 24 });
    try expectServed(s, &sb, r1, &.{ 20, 21, 22, 23, 24 });
    s.release(r1);
    try s.flush();
}

// decode_first16 across requests: the reverse phase change frees the pool with window 0 and resets its clock, so the
// second request's grow re-arms 20 (here 2) pool rows per layer, re-plans once more at its own cycle, and serves every
// id its record.
test "dsv41 stream: decode_first16 re-arms its pool and its clock for the next request" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool, .transient_release = true, .decode_pool = .{ .per_layer = 2, .from_cycle = 2, .at_cycle = 3 } });
    defer s.deinit();
    var rng = std.Random.DefaultPrng.init(5);
    const rand = rng.random();
    var ids: [12]u16 = undefined;
    for (0..2) |_| {
        s.release(try serve(s, 0, &.{ 1, 2, 3 }));
        _ = try s.releaseTransient();
        try s.grow(&.{ 8, 8 });
        for (s.layers) |ls| try testing.expectEqual(@as(u32, 6), ls.pool_lo);
        for (1..6) |cycle| {
            for (0..2) |l| {
                const n = rand.intRangeAtMost(usize, 2, 8);
                for (ids[0..n]) |*e| e.* = if (l == 0) rand.intRangeLessThan(u16, 0, 5) else rand.intRangeLessThan(u16, 8, 30);
                const r = try serve(s, @intCast(l), ids[0..n]);
                try expectServed(s, &sb, r, ids[0..n]);
                s.release(r);
            }
            try s.flush();
            try s.cycleEnd();
            try testing.expectEqual(cycle >= 3, s.poolReplan() != null);
        }
        const rp = s.poolReplan().?;
        try testing.expectEqual(@as(u32, 16), rp.rows[0] + rp.rows[1]);
        _ = try s.shrink(&.{ 4, 4 });
        for (s.layers) |ls| {
            try testing.expectEqual(@as(u32, 4), ls.pool_lo);
            try testing.expectEqual(@as(usize, 0), ls.pool.len);
        }
        try testing.expect(s.poolReplan() == null);
        _ = try s.regrowTransient();
    }
}

fn streamInitDeinit(a: std.mem.Allocator, sb: *const SynthBank, opt: Options) !void {
    const s = try Stream.init(a, &sb.bank, opt);
    s.deinit();
}

// The construction's every allocation (the stream's, its layers', the selector's, the warm and decode-pool state's and
// the read pool's) failed in turn: each failure unwinds to error.OutOfMemory with nothing leaked and the pool stopped.
test "dsv41 stream: every allocation failure of the construction unwinds, the decode pool installed" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    try std.testing.checkAllAllocationFailures(testing.allocator, streamInitDeinit, .{ &sb, Options{
        .rows = &.{ 4, 2 },
        .max_route_ids = 12,
        .transient_rows = 2 * 12,
        .wide_depth = 2,
        .pool = la_pool,
        .lookahead = .{ .k = 6, .budget = 2, .chunks = 1 },
        .event = .{ .watchdog_ms = 10_000 },
        .transient_release = true,
        .first_verify_warm = .{ .max_records = 8, .busy_max = 1 },
        .decode_pool = .{ .per_layer = 2, .from_cycle = 2, .at_cycle = 4 },
    } });
}

test "dsv41 stream: every allocation failure of the construction unwinds (lookahead, gates, wide depth, warm, transient release)" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    try std.testing.checkAllAllocationFailures(testing.allocator, streamInitDeinit, .{ &sb, Options{
        .rows = &.{ 4, 2 },
        .max_route_ids = 12,
        .transient_rows = 2 * 12,
        .wide_depth = 2,
        .pool = la_pool,
        .lookahead = .{ .k = 6, .budget = 2, .chunks = 1 },
        .event = .{ .watchdog_ms = 10_000 },
        .transient_release = true,
        .first_verify_warm = .{ .max_records = 8, .busy_max = 1 },
    } });
}

test "dsv41 stream: no decode plan runs beside held slots: the phase change refuses a held base and leaves decode one window" {
    var sb = try SynthBank.open(32);
    defer sb.close();
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 4, 4 }, .max_route_ids = 12, .transient_rows = 2 * 12, .wide_depth = 2, .pool = test_pool });
    defer s.deinit();
    const r = try serve(s, 0, &.{ 1, 2, 3 });
    try s.holdBase(r);
    s.release(r);
    try testing.expectError(error.RoutesLive, s.grow(&.{ 6, 6 }));
    s.releaseHeld();
    try s.grow(&.{ 6, 6 });
    try testing.expectEqual(@as(u8, 1), s.wide_depth);
    const d = try serve(s, 0, &.{ 4, 5, 1 });
    try testing.expectEqual(@as(u8, 0), d.window);
    try expectServed(s, &sb, d, &.{ 4, 5, 1 });
    s.release(d);
}
