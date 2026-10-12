//! `MIMO_DIAG=missfree` on the trace backend over MiMo's real stream source (host slot rows, the lookahead class on),
//! for the hit pass and the fused route: after the prompt pass, every decode-phase miss (serial M = 1 and the MTP
//! verify's M = 4) resolves to a resident slot of its layer; no load is planned or read, no pre-read and no
//! speculation is posted, the resident set is unchanged; the prompt pass is the stock route. CPU only.
const std = @import("std");
const ops = @import("../deepseek_v41/ops.zig");
const shared = @import("../deepseek_v41/experts.zig");
const stream_mod = @import("stream.zig");
const source_mod = @import("source.zig");
const diag = @import("diag.zig");
const vl = @import("verify_lane_test.zig");

const testing = std.testing;
const StreamSource = stream_mod.StreamSource;
const top_k = vl.top_k;
const n_experts = vl.n_experts;
const Arm = enum { hitpass, fused };

/// Layer 2's prompt routes (layer 1's are `vl.verify_ids`): every routed layer holds residents at the decode boundary.
const layer2_ids = [4][top_k]u16{
    .{ 1, 2, 4, 8, 16, 32, 33, 34 },
    .{ 2, 4, 8, 16, 35, 36, 37, 38 },
    .{ 1, 4, 8, 16, 39, 48, 49, 50 },
    .{ 2, 8, 16, 32, 51, 52, 53, 54 },
};
/// The serial tick's ids on layer 1: mostly experts the prompt never routed (certain misses without missfree).
const one = [_]u16{ 3, 0, 12, 13, 14, 15, 30, 31 };
/// The verify's four rows on layer 1: most of the union misses.
const verify4 = [4][top_k]u16{
    .{ 3, 9, 50, 51, 52, 53, 54, 55 },
    .{ 9, 3, 56, 57, 58, 59, 60, 61 },
    .{ 0, 1, 2, 4, 6, 7, 8, 10 },
    .{ 11, 12, 13, 14, 15, 16, 62, 63 },
};

fn residentSet(ss: *const StreamSource, layer: u32) [n_experts]bool {
    var out: [n_experts]bool = undefined;
    for (&out, 0..) |*r, e| r.* = ss.isResident(layer, @intCast(e));
    return out;
}

fn missesOf(res: *const [n_experts]bool, ids: []const u16) u32 {
    var seen: [n_experts]bool = @splat(false);
    var n: u32 = 0;
    for (ids) |e| {
        if (!seen[e] and !res[e]) n += 1;
        seen[e] = true;
    }
    return n;
}

/// The I/O counters a missfree decode call must leave unchanged.
fn expectNoIo(a: stream_mod.Stats, b: stream_mod.Stats) !void {
    try testing.expectEqual(a.expert_cache_misses, b.expert_cache_misses);
    try testing.expectEqual(a.persistent_loads + a.transient_loads, b.persistent_loads + b.transient_loads);
    try testing.expectEqual(a.expert_bytes_read, b.expert_bytes_read);
    try testing.expectEqual(a.expert_cache_evictions, b.expert_cache_evictions);
    try testing.expectEqual(a.pre_issued, b.pre_issued);
    try testing.expectEqual(a.spec_issued, b.spec_issued);
    try testing.expectEqual(a.ahead_posted, b.ahead_posted);
    try testing.expectEqual(a.gates, b.gates);
}

fn run(a: std.mem.Allocator, arm: Arm) !void {
    const Tr = ops.TraceOps;
    const rows = [_]u32{ 8, 8 };
    const f = try a.create(vl.Fixture);
    defer a.destroy(f);
    try f.init(a, 32, 32, &rows, .host, true);
    defer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    var mf: diag.MissFree = .{};
    var g = Tr.init(a);
    defer g.deinit();
    g.record_host = true;
    const Ex = shared.ExpertsWith(Tr, StreamSource, vl.LaneMath, .{ .prefill = true, .lookahead = true });
    const c: @import("config.zig").Config = .{ .hidden_size = 32, .moe_intermediate_size = 32, .n_layers = 3, .n_routed_experts = n_experts, .n_experts_per_tok = top_k };
    var gates: [3]Ex.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 32, n_experts }, .float32), .bias = try g.input(&.{n_experts}, .float32) };
    var ex = try Ex.initWith(a, &g, &ss, .{ .hidden = 32, .inter = 32 }, &c, .{ .gates = &gates, .banked = true, .devroute = arm == .hitpass, .wide = .{ .seed = true, .hot_first = true, .depth = stream_mod.wide_depth } });
    defer ex.deinit();

    var b1: [4 * top_k]u16 = undefined;
    var b2: [4 * top_k]u16 = undefined;
    var b4: [4 * top_k]u16 = undefined;
    const p1 = vl.flat(&vl.verify_ids, &b1);
    const p2 = vl.flat(&layer2_ids, &b2);
    const v4 = vl.flat(&verify4, &b4);
    var script: vl.Script = .{ .calls = &.{ p1, p2, &one, v4 } };
    g.host_values = script.values();

    // Armed before the prompt (as the profile build arms it at construction): the prompt pass is the stock route.
    ss.missfree = &mf;
    const x4 = try g.input(&.{ 4, 32 }, .bfloat16);
    const idx4 = try g.input(&.{ 4, 8 }, .int32);
    const st_p0 = ss.stats();
    for ([_]u32{ 1, 2 }) |layer| {
        _ = try ex.at(layer).routedParts(&g, x4, idx4);
        ex.releaseParts(&g);
        try ex.flush();
    }
    const st_p1 = ss.stats();
    try testing.expect(st_p1.expert_cache_misses > st_p0.expert_cache_misses);
    try testing.expect(st_p1.expert_bytes_read > st_p0.expert_bytes_read);
    try testing.expectEqual(@as(u64, 0), mf.routes);

    // The phase change fills the substitutes from the prompt's residents.
    try ex.grow(&g, &rows);
    try testing.expect(ss.decoding());
    for ([_]u32{ 1, 2 }) |layer| {
        const res = residentSet(&ss, layer);
        const rel = layer - 1;
        try testing.expect(mf.residents[rel] > 0);
        for (0..n_experts) |e| {
            const s = mf.sub(rel, @intCast(e));
            try testing.expect(res[s]);
            if (res[e]) try testing.expectEqual(@as(u16, @intCast(e)), s);
        }
    }
    const res0 = residentSet(&ss, 1);
    const res0_l2 = residentSet(&ss, 2);
    // Without missfree these calls would miss (the test exercises misses).
    try testing.expect(missesOf(&res0, &one) >= 4);
    try testing.expect(missesOf(&res0, v4) >= 16);

    // Serial M = 1 through the scheduler's decode lane.
    const x1 = try g.input(&.{ 1, 32 }, .bfloat16);
    const idx1 = try g.input(&.{ 1, 8 }, .int32);
    const s0 = ss.stats();
    const out1 = try ex.at(1).routed(&g, x1, idx1);
    try ex.flush();
    const s1 = ss.stats();
    try testing.expectEqualSlices(c_int, &.{ 1, 8, 32 }, g.shapeOf(out1).slice());
    try expectNoIo(s0, s1);
    try testing.expectEqual(@as(u64, 1), s1.route_calls - s0.route_calls);
    try testing.expectEqual(@as(u64, 1), mf.routes);

    // The MTP verify: M = 4 in the decode phase (`Source.decodeLane`), one route over the substituted union.
    try testing.expect(source_mod.decodeLane(4, ss.decoding()));
    const out4 = try ex.at(1).routed(&g, x4, idx4);
    try ex.flush();
    const s2 = ss.stats();
    try testing.expectEqualSlices(c_int, &.{ 4, 8, 32 }, g.shapeOf(out4).slice());
    try expectNoIo(s1, s2);
    try testing.expectEqual(@as(u64, 1), s2.route_calls - s1.route_calls);
    try testing.expectEqual(@as(u64, countSubs(&res0, &one) + countSubs(&res0, v4)), mf.substituted);

    if (arm == .hitpass) {
        // The device LUT after the route: every expert maps to its substitute's resident slot (no miss sentinel).
        const lut = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[1][ex.dev_parity[1]]))));
        var want: [n_experts]u32 = undefined;
        ss.residentLut(1, &want, ex.math.devrouteMissingId());
        try testing.expectEqualSlices(u32, &want, lut);
        for (0..n_experts) |e| {
            const slot = ss.policy(1).slotOf(mf.sub(0, @intCast(e))).?;
            const ref = f.stream.slotRef(0, slot);
            try testing.expectEqual((@as(u32, @backingInt(ref.bank)) << 24) | ref.row, want[e]);
        }
    }

    // The source's own call view: each routed id served from its substitute's resident slot, one wave (hits only).
    for ([_][]const u16{ &one, v4 }) |ids| {
        const before = ss.stats();
        const call = try ss.route(1, ids, &.{});
        const sv = ss.served(call);
        try testing.expectEqual(@as(u32, 0), sv.n_parts);
        for (ids, sv.refs, sv.waves) |e, ref, wave| {
            const slot = ss.policy(1).slotOf(mf.sub(0, e)).?;
            try testing.expect(slot < ss.slotCapacity(1));
            try testing.expectEqual(f.stream.slotRef(0, slot), ref);
            try testing.expectEqual(@as(u8, 0), wave);
        }
        ss.release(call);
        try ss.flush();
        try expectNoIo(before, ss.stats());
    }

    // The resident set is the prompt's, unchanged by the decode calls.
    try testing.expectEqualSlices(bool, &res0, &residentSet(&ss, 1));
    try testing.expectEqualSlices(bool, &res0_l2, &residentSet(&ss, 2));
}

fn countSubs(res: *const [n_experts]bool, ids: []const u16) u64 {
    var n: u64 = 0;
    for (ids) |e| n += @intFromBool(!res[e]);
    return n;
}

test "mimo diag missfree: decode misses (serial M=1 and the MTP verify M=4) resolve to resident slots; zero loads, reads, pre-reads and speculation; residency unchanged; the prompt pass stock (hitpass and fused)" {
    try run(testing.allocator, .hitpass);
    try run(testing.allocator, .fused);
}

test {
    _ = diag;
}
