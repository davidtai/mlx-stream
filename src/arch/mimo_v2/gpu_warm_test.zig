//! X12 `gpu_keep_warm` CPU proof over MiMo's real gated stream source (no device, no lock): the trace backend runs the
//! shared scheduler's decode lane (hit pass, M = 1 and M = 4, layers 1 and 2) twice on identical scripts: the control
//! (no warm arm) and the "waits" binding (a trace warm engine armed from `StreamSource.gate`, reading the stream's host
//! event word, its thread running beside the calls). Every route, plan, slot, next LUT, stream counter and trace node
//! must be identical; the engine must be armed once per gated decode call with that call's last gate value (the
//! stream's newest), never in the prompt phase, and must park once the pool's reads have handed the word every value.
const std = @import("std");
const ops = @import("../deepseek_v41/ops.zig");
const shared = @import("../deepseek_v41/experts.zig");
const stream_mod = @import("stream.zig");
const gpu_warm = @import("gpu_warm.zig");
const vl = @import("verify_lane_test.zig");

const testing = std.testing;
const Tr = ops.TraceOps;
const StreamSource = stream_mod.StreamSource;
const top_k = vl.top_k;
const n_experts = vl.n_experts;

const n_calls = 30;

fn xorshift(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

/// Each row's top-8 over 64 experts (a hot band half the time: hits, misses and evictions on most calls).
fn drawRows(state: *u32, m: usize, out: []u16) void {
    for (0..m) |r| {
        var seen: [n_experts]bool = @splat(false);
        var j: usize = 0;
        while (j < top_k) {
            const e: u16 = @intCast(if (xorshift(state) % 2 == 0) xorshift(state) % 16 else xorshift(state) % n_experts);
            if (seen[e]) continue;
            seen[e] = true;
            out[r * top_k + j] = e;
            j += 1;
        }
    }
}

const Facts = std.ArrayList(u64);

fn put(f: *Facts, a: std.mem.Allocator, v: anytype) !void {
    try f.append(a, @as(u64, @intCast(v)));
}

/// The stream's host counters, every live route's plan (slots, hits, misses, loads, evictions, parts) and both routed
/// layers' slot maps.
fn recordFacts(f: *Facts, a: std.mem.Allocator, ss: *StreamSource) !void {
    const st = ss.stats();
    inline for (.{ "route_calls", "expert_cache_hits", "expert_cache_misses", "expert_cache_evictions", "persistent_loads", "transient_loads", "loads_skipped", "gates" }) |name| try put(f, a, @field(st, name));
    for (&ss.stream.routes) |*r| {
        if (r.state == .free) continue;
        try put(f, a, r.layer);
        try put(f, a, r.plan.n_ids);
        for (r.plan.slots[0..r.plan.n_ids]) |s| try put(f, a, s);
        for (r.plan.hitsOf()) |e| try put(f, a, e);
        try put(f, a, @as(u64, 0xffff));
        for (r.plan.missesOf()) |e| try put(f, a, e);
        try put(f, a, @as(u64, 0xffff));
        for (r.plan.loadsOf()) |l| {
            try put(f, a, l.expert);
            try put(f, a, l.slot);
            try put(f, a, @intFromBool(l.persistent));
        }
        for (r.plan.evictionsOf()) |ev| {
            try put(f, a, ev.slot);
            try put(f, a, ev.previous);
            try put(f, a, ev.next);
        }
        try put(f, a, r.n_parts);
        for (r.partsOf()) |p| {
            try put(f, a, p.first);
            try put(f, a, p.n);
            try put(f, a, p.n_reads);
        }
    }
    for (1..3) |l| {
        const pol = ss.policy(@intCast(l));
        for (pol.slot_to_expert[0..pol.capacity]) |e| try put(f, a, e);
    }
}

/// The arm the stream source calls: records (value, the stream's newest gate value, decoding) and forwards to the
/// engine.
const Recorder = struct {
    engine: ?*gpu_warm.TraceEngine,
    stream: *const stream_mod.Stream,
    ss: *const StreamSource,
    values: [256]u64 = undefined,
    newest: [256]u64 = undefined,
    n: usize = 0,
    prompt_arms: usize = 0,

    pub fn arm(self: *Recorder, v: u64) void {
        if (!self.ss.decoding()) self.prompt_arms += 1;
        self.values[self.n] = v;
        self.newest[self.n] = self.stream.gate_value;
        self.n += 1;
        if (self.engine) |e| e.arm(v);
    }

    pub fn hold(self: *Recorder, on: bool) void {
        if (self.engine) |e| e.hold(on);
    }
};

fn hostWord(p: ?*const anyopaque) u64 {
    const w: *const i64 = @ptrCast(@alignCast(p.?));
    return @intCast(@atomicLoad(i64, w, .acquire));
}

fn parkedNow(e: *const gpu_warm.TraceEngine) bool {
    return e.isParked();
}

fn wordAt(w: *const i64, v: u64) bool {
    return @as(u64, @intCast(@atomicLoad(i64, w, .acquire))) >= v;
}

fn waitFor(comptime f: anytype, args: anytype, timeout_ms: u64) !void {
    const t0 = gpu_warm.nowNs();
    while (!@call(.auto, f, args)) {
        if (gpu_warm.nowNs() - t0 > timeout_ms * std.time.ns_per_ms) return error.TestTimeout;
        const ts = std.c.timespec{ .sec = 0, .nsec = 200 * std.time.ns_per_us };
        _ = std.c.nanosleep(&ts, null);
    }
}

const Run = struct {
    facts: Facts = .empty,
    nodes: std.ArrayList(Tr.Node) = .empty,
    arms: usize = 0,
    /// Decode calls that registered gates (the stream's gate value moved across the call).
    gated_calls: usize = 0,
    stats: gpu_warm.Stats = .{},

    fn deinit(r: *Run, a: std.mem.Allocator) void {
        r.facts.deinit(a);
        r.nodes.deinit(a);
    }
};

fn run(a: std.mem.Allocator, warm: bool, out: *Run) !void {
    const rows = [_]u32{ 8, 8 };
    const f = try a.create(vl.Fixture);
    defer a.destroy(f);
    try f.initWith(a, 32, 32, &rows, .host, true, true);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    const word = f.stream.eventWord() orelse return error.NoHostEvent;
    var log: gpu_warm.Trace.Log = .{};
    const engine: ?*gpu_warm.TraceEngine = if (warm) try gpu_warm.TraceEngine.create(a, .{ .value_fn = hostWord, .ctx = word, .log = &log }, .{ .target_chunk_ns = 50 * std.time.ns_per_us, .first_iters = 25_000 }) else null;
    defer if (engine) |e| e.destroy();
    var rec: Recorder = .{ .engine = engine, .stream = f.stream, .ss = &ss };
    // The "waits" binding (Source.initWith binds the served engine's arm the same way); the control binds nothing.
    if (warm) ss.warm_arm = gpu_warm.Arm.of(Recorder, &rec);
    var g = Tr.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = shared.ExpertsWith(Tr, StreamSource, vl.LaneMath, .{ .prefill = true, .gated = true, .lookahead = true });
    const c: @import("config.zig").Config = .{ .hidden_size = 32, .moe_intermediate_size = 32, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 32, n_experts }, .float32), .bias = try g.input(&.{n_experts}, .float32) };
    var ex = try Ex.initWith(a, &g, &ss, .{ .hidden = 32, .inter = 32 }, &c, .{ .gates = &gates, .banked = true, .devroute = true, .wide = .{ .seed = true, .hot_first = true, .depth = stream_mod.wide_depth } });
    defer ex.deinit();

    var bufs: [n_calls + 1][4 * top_k]u16 = undefined;
    var calls: [n_calls + 1][]const u16 = undefined;
    var state: u32 = 0x9a7d_3a11;
    drawRows(&state, 4, &bufs[0]);
    calls[0] = bufs[0][0 .. 4 * top_k];
    for (0..n_calls) |ci| {
        const m: usize = if ((ci / 2) % 3 == 0) 4 else 1;
        drawRows(&state, m, &bufs[ci + 1]);
        calls[ci + 1] = bufs[ci + 1][0 .. m * top_k];
    }
    var script: vl.Script = .{ .calls = calls[0 .. n_calls + 1] };
    g.host_values = script.values();

    // The prompt's M = 4 chunk (the wide lane), then the phase change.
    const x4 = try g.input(&.{ 4, 32 }, .bfloat16);
    const idx4 = try g.input(&.{ 4, 8 }, .int32);
    _ = try ex.at(1).routedParts(&g, x4, idx4);
    ex.releaseParts(&g);
    try ex.flush();
    try ex.grow(&g, &rows);
    try testing.expect(ss.decoding());
    for (0..n_calls) |ci| {
        const layer: u32 = if (ci % 2 == 0) 1 else 2;
        const m: u32 = if ((ci / 2) % 3 == 0) 4 else 1;
        const x = try g.input(&.{ @intCast(m), 32 }, .bfloat16);
        const idx = try g.input(&.{ @intCast(m), 8 }, .int32);
        const gv0 = f.stream.gate_value;
        _ = try ex.at(layer).routed(&g, x, idx);
        out.gated_calls += @intFromBool(f.stream.gate_value != gv0);
        try recordFacts(&out.facts, a, &ss);
        if (layer == 2) try ex.flush();
    }
    try ex.flush();
    try out.nodes.appendSlice(a, g.nodes.items);
    if (engine) |e| {
        // the pool's reads land: the word reaches the newest value and the engine parks
        try waitFor(wordAt, .{ word, f.stream.gate_value }, 5000);
        try waitFor(parkedNow, .{e}, 5000);
        out.stats = e.stats();
        out.arms = rec.n;
        try testing.expectEqual(@as(usize, 0), rec.prompt_arms);
        var prev: u64 = 0;
        for (rec.values[0..rec.n], rec.newest[0..rec.n]) |v, nw| {
            // the call's last gate value is the stream's newest; the values only grow
            try testing.expectEqual(nw, v);
            try testing.expect(v > prev);
            prev = v;
        }
        try testing.expect(out.stats.chunks - out.stats.retired <= gpu_warm.max_inflight);
        try testing.expect(!out.stats.failed);
        try testing.expectEqual(@as(u64, 0), out.stats.stuck);
    }
}

test "mimo gpu warm route CPU: the decode lane's routes, plans, slots, counters and trace nodes are identical with the warm engine armed from StreamSource.gate; one arm per gated decode call at its last gate value, none in the prompt; the engine parks when the reads land (trace backend, MiMo's gated stream source)" {
    const a = testing.allocator;
    var off: Run = .{};
    defer off.deinit(a);
    var waits: Run = .{};
    defer waits.deinit(a);
    try run(a, false, &off);
    try run(a, true, &waits);
    try testing.expectEqualSlices(u64, off.facts.items, waits.facts.items);
    try testing.expectEqual(off.nodes.items.len, waits.nodes.items.len);
    for (off.nodes.items, waits.nodes.items) |x, y| {
        try testing.expectEqual(x.op, y.op);
        try testing.expectEqual(x.dtype, y.dtype);
        try testing.expectEqualSlices(c_int, x.shape.slice(), y.shape.slice());
    }
    try testing.expect(waits.gated_calls > 0);
    try testing.expectEqual(waits.gated_calls, waits.arms);
    try testing.expectEqual(off.gated_calls, waits.gated_calls);
    std.debug.print("MIMO_GPU_WARM_CPU calls={d} facts={d} nodes={d} decode_gated_calls={d} arms={d} episodes={d} chunks={d} identical=true\n", .{ n_calls, off.facts.items.len, off.nodes.items.len, waits.gated_calls, waits.arms, waits.stats.episodes, waits.stats.chunks });
}
