//! A1's recall check, profile builds only (`-Ddsv41-decode-timers=true`, the decode timers' flag). At every verify
//! layer, P1's predictor (the layer's own router over its input, attention left out) is compared with the experts
//! the layer routes after its attention. Counts accumulate per layer over the decode and print once
//! (`DECODE_RECALL`). No read is issued and no output changes. In every other build `enabled` is false and no
//! call site compiles.

const std = @import("std");
const dt = @import("dsv41_decode_timers.zig");

pub const enabled = dt.enabled;
pub const max_layers = 64;
/// Expert ids below this (the widest bank's 384, with room).
pub const max_experts = 512;

/// Per layer, summed over its decode-width calls; each count is of distinct experts per call.
pub const Counts = struct {
    calls: u64 = 0,
    /// Routed experts, and those not resident at the call (its misses).
    routed: u64 = 0,
    missed: u64 = 0,
    /// Predicted experts: all, those routed, those missed (a read the predictor would have started early).
    predicted: u64 = 0,
    predicted_routed: u64 = 0,
    predicted_missed: u64 = 0,
    /// Predicted, not routed, and not resident before the call: a read the predictor would have started for nothing.
    wasted: u64 = 0,

    pub fn add(self: *Counts, o: Counts) void {
        inline for (@typeInfo(Counts).@"struct".field_names) |name| @field(self, name) += @field(o, name);
    }
};

pub var layers: [max_layers]Counts = @splat(.{});
/// Set by the run that wants the check (the served cell's decode profile), outside the timed spans.
pub var active: bool = false;

pub fn reset() void {
    layers = @splat(.{});
}

const Set = std.StaticBitSet(max_experts);

fn setOf(ids: []const u16) Set {
    var s: Set = .empty;
    for (ids) |e| s.set(e);
    return s;
}

/// Predicted experts neither routed nor resident before the call (`resident(ctx, e)`): query it before the route, which
/// admits and evicts.
pub fn wastedOf(predicted: []const u16, routed: []const u16, ctx: anytype, comptime resident: fn (@TypeOf(ctx), u16) bool) u64 {
    const p = setOf(predicted).differenceWith(setOf(routed));
    var n: u64 = 0;
    var it = p.iterator(.{});
    while (it.next()) |e| n += @intFromBool(!resident(ctx, @intCast(e)));
    return n;
}

/// One call: `routed` holds the ids in the router's order and `waves` each id's wave in the call's plan (0: resident at
/// the call); `wasted` is `wastedOf`'s count.
pub fn countCall(predicted: []const u16, routed: []const u16, waves: []const u8, wasted: u64) Counts {
    std.debug.assert(routed.len == waves.len);
    const p = setOf(predicted);
    const r = setOf(routed);
    var m: Set = .empty;
    for (routed, waves) |e, w| if (w > 0) m.set(e);
    return .{
        .calls = 1,
        .routed = r.count(),
        .missed = m.count(),
        .predicted = p.count(),
        .predicted_routed = p.intersectWith(r).count(),
        .predicted_missed = p.intersectWith(m).count(),
        .wasted = wasted,
    };
}

pub fn record(layer: u32, c: Counts) void {
    layers[layer].add(c);
}

fn ratio(a: u64, b: u64) f64 {
    return if (b == 0) 0 else @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
}

/// The receipt's record: the totals, the four ratios, and every layer's counts.
pub const Summary = struct {
    total: Counts,
    /// predicted_routed / routed, predicted_missed / missed, predicted_routed / predicted,
    /// predicted_missed / (predicted_missed + wasted): the share of the predictor's reads that were needed.
    recall: f64,
    recall_missed: f64,
    precision: f64,
    read_precision: f64,
    per_layer: []const Counts,
};

pub fn summary(n_layers: usize) Summary {
    var t: Counts = .{};
    for (layers[0..n_layers]) |c| t.add(c);
    return .{
        .total = t,
        .recall = ratio(t.predicted_routed, t.routed),
        .recall_missed = ratio(t.predicted_missed, t.missed),
        .precision = ratio(t.predicted_routed, t.predicted),
        .read_precision = ratio(t.predicted_missed, t.predicted_missed + t.wasted),
        .per_layer = layers[0..n_layers],
    };
}

/// One line: the summary as JSON, each layer as [calls, routed, missed, predicted, predicted_routed, predicted_missed,
/// wasted].
pub fn line(buf: []u8, n_layers: usize) []const u8 {
    const s = summary(n_layers);
    var w: std.Io.Writer = .fixed(buf);
    const t = s.total;
    w.print("DECODE_RECALL {{\"calls\": {d}, \"routed\": {d}, \"missed\": {d}, \"predicted\": {d}, \"predicted_routed\": {d}, \"predicted_missed\": {d}, \"wasted\": {d}, \"recall\": {d:.4}, \"recall_missed\": {d:.4}, \"precision\": {d:.4}, \"read_precision\": {d:.4}, \"per_layer\": [", .{
        t.calls, t.routed, t.missed, t.predicted, t.predicted_routed, t.predicted_missed, t.wasted, s.recall, s.recall_missed, s.precision, s.read_precision,
    }) catch return buf[0..0];
    for (s.per_layer, 0..) |c, i| {
        w.print("{s}[{d}, {d}, {d}, {d}, {d}, {d}, {d}]", .{ if (i == 0) "" else ", ", c.calls, c.routed, c.missed, c.predicted, c.predicted_routed, c.predicted_missed, c.wasted }) catch return buf[0..0];
    }
    w.writeAll("]}") catch return buf[0..0];
    return w.buffered();
}

test "dsv41 decode recall: a call's distinct experts, its misses, and the predictor's hits and wasted reads" {
    // Two rows of top-3: experts 4 and 9 appear twice; 7 and 9 are misses (waves 1, 2).
    const routed = [_]u16{ 4, 7, 9, 4, 9, 11 };
    const waves = [_]u8{ 0, 1, 2, 0, 2, 0 };
    // Predicted 4, 9, 20, 21 (twice 9): 4 and 9 routed, 9 missed; 20 not resident, 21 resident.
    const predicted = [_]u16{ 9, 4, 20, 9, 21, 4 };
    const Res = struct {
        fn f(_: void, e: u16) bool {
            return e == 21;
        }
    };
    const wasted = wastedOf(&predicted, &routed, {}, Res.f);
    try std.testing.expectEqual(@as(u64, 1), wasted);
    const c = countCall(&predicted, &routed, &waves, wasted);
    try std.testing.expectEqual(Counts{ .calls = 1, .routed = 4, .missed = 2, .predicted = 4, .predicted_routed = 2, .predicted_missed = 1, .wasted = 1 }, c);
    reset();
    defer reset();
    record(1, c);
    record(1, c);
    const s = summary(2);
    try std.testing.expectEqual(@as(u64, 2), s.total.calls);
    try std.testing.expectEqual(@as(u64, 0), s.per_layer[0].calls);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), s.recall, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), s.recall_missed, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), s.read_precision, 1e-12);
    var buf: [1024]u8 = undefined;
    const l = line(&buf, 2);
    try std.testing.expect(std.mem.startsWith(u8, l, "DECODE_RECALL {\"calls\": 2, \"routed\": 8, \"missed\": 4,"));
    try std.testing.expect(std.mem.endsWith(u8, l, "\"per_layer\": [[0, 0, 0, 0, 0, 0, 0], [2, 8, 4, 8, 4, 2, 2]]}"));
}
