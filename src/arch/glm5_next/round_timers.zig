//! glm5_next's decode round anatomy, for PROFILE builds only (glm5_next/profile.zig owns the rows and stamps them; in a
//! served build every stamp compiles to nothing). Per decode forward (a cycle: the index of DECODE_TIMERS and of the verify
//! GPU timeline, `profile.zig` `state.steps`) the host stamps fixed points of the MTP round around that forward, from the
//! round's entry to its return, on one clock (`verify_timeline.now`: CLOCK_UPTIME_RAW, the GPU timeline's own clock, so a
//! stamp joins the timeline's command buffers). The host's time between two rounds (its token emission, its scheduler tick)
//! is the next round's `host` part. Every part is the interval between two consecutive stamps, so a round's parts add up to
//! its span and the request's rows to the decode's span (the first round's entry to the last one's return) by
//! construction; the cell's report checks that sum against the decode walls the plugin and the host print.
//! This file is pure over the rows (the tests feed it by hand): the summary line and the per-round TSV.

const std = @import("std");

/// Rows kept per request (a 16,384-token decode emits at least one token a forward); later forwards are counted, not kept.
pub const max_rounds: u32 = 16384;

/// The stamps of a round, in time order. module.zig `roundWith` stamps them all (mtp.zig `Lane.append` and `chainDrafts`
/// the draft's, profile.zig `stepBegin` / `stepEnd` the verify's ends); a serial step (module.zig `stepWith`) stamps
/// `enter`, its forward's and `exit`.
pub const Point = enum(u8) {
    /// The round's entry (roundWith's wall start; a serial step's: stepWith's).
    enter,
    /// The pending pairs' catch-up: their key chunks evaluated (mtp.zig Lane.append's loop).
    appended,
    /// The last pending row copied out and evaluated (Lane.append's end).
    copied,
    /// The draft chain's graph built: `depth` steps of the MTP layer, the lm_head and the draw (chainDrafts, before its eval).
    chain_built,
    /// The chain evaluated: the host blocked on the chain's GPU work.
    chain_done,
    /// The drafts and their top-1 probabilities read to the host.
    chain_read,
    /// The verify forward begins (profile.stepBegin: DECODE_TIMERS' verify span, the GPU timeline's verify begin).
    verify_begin,
    /// graph.forward returned: the verify's graph encoded, its routed calls' barriers, routes and reads inside it.
    fwd_built,
    /// The decision's graph (the position-keyed draws, sample-then-verify) and the eval list built.
    dec_built,
    /// The verify's eval returned: the GPU's last layers, the head and the decision.
    evaluated,
    /// The expert stream flushed.
    flushed,
    /// The decision read to the host and the commit planned.
    decided,
    /// The rollback to the kept rows done: the verify forward ends (profile.stepEnd).
    verify_end,
    /// The kept rows' hidden pended for the next round (mtp.zig Lane.pend: its eval).
    pended,
    /// The round returns: the history, the lane's counters, the tokens, the trace, the decode wall.
    exit,
};
pub const n_points = @typeInfo(Point).@"enum".field_names.len;

/// A point stamped after its cycle's verify ended (profile.stepEnd counted the cycle): its row is the cycle just ended.
pub fn afterVerify(p: Point) bool {
    return p == .pended or p == .exit;
}

/// A round's parts in time order: `host` is the time from the previous round's `exit` to this round's `enter` (0 in the
/// first round), and part i > 0 spans point i - 1 to point i.
pub const Part = enum(u8) {
    host,
    append,
    pending_copy,
    chain_encode,
    chain_wait,
    chain_read,
    gate,
    fwd_encode,
    decision_build,
    verify_wait,
    flush,
    decision_read,
    rollback,
    pend,
    bookkeep,
};
pub const n_parts = @typeInfo(Part).@"enum".field_names.len;

comptime {
    if (n_parts != n_points or @intFromEnum(Part.fwd_encode) != @intFromEnum(Point.fwd_built) or @intFromEnum(Part.bookkeep) != @intFromEnum(Point.exit))
        @compileError("glm5_next round timers: part i must span point i - 1 to point i");
}

/// The verify forward's parts (verify_begin .. verify_end: DECODE_TIMERS' verify span).
pub fn inVerify(p: Part) bool {
    return @intFromEnum(p) >= @intFromEnum(Part.fwd_encode) and @intFromEnum(p) <= @intFromEnum(Part.rollback);
}

/// One forward's stamps (CLOCK_UPTIME_RAW ns; 0 = not stamped) and its counts.
pub const Row = struct {
    t: [n_points]u64 = @splat(0),
    /// A serial step (module.zig stepWith), not an MTP round.
    serial: bool = false,
    /// The chain's drafts, the drafts the draft gate verified, the drafts accepted.
    drafted: u8 = 0,
    verified: u8 = 0,
    kept: u8 = 0,
    /// The tokens the forward emitted.
    emitted: u32 = 0,
};

/// The row's stamps, the unstamped points filled: a point not stamped takes the previous point's time (its part is 0), and
/// leading unstamped points the first stamped one.
pub fn filled(r: *const Row) [n_points]u64 {
    var t = r.t;
    var prev: u64 = 0;
    for (t) |x| if (x != 0) {
        prev = x;
        break;
    };
    for (&t) |*x| {
        if (x.* == 0) x.* = prev else prev = x.*;
    }
    return t;
}

pub const Parts = struct { ns: [n_parts]u64, disorder: u32 };

/// Row `r`'s parts in ns: `host` from `prev`'s exit (none: 0), then each consecutive pair of its filled stamps. A pair out of
/// order gives 0 and counts in `disorder`.
pub fn partsOf(prev: ?*const Row, r: *const Row) Parts {
    const t = filled(r);
    var out: Parts = .{ .ns = @splat(0), .disorder = 0 };
    if (prev) |p| {
        const pe = filled(p)[@intFromEnum(Point.exit)];
        if (t[0] >= pe) out.ns[0] = t[0] - pe else out.disorder += 1;
    }
    for (1..n_points) |i| {
        if (t[i] >= t[i - 1]) out.ns[i] = t[i] - t[i - 1] else out.disorder += 1;
    }
    return out;
}

pub const Summary = struct {
    rows: u32 = 0,
    serial: u32 = 0,
    drafted_rounds: u32 = 0,
    disorder: u32 = 0,
    /// Per part, summed over the rows (ns).
    total: [n_parts]u64 = @splat(0),
    /// The first row's entry to the last row's return (ns): the sum of every part.
    span: u64 = 0,
};

pub fn summarize(rows: []const Row) Summary {
    var s: Summary = .{ .rows = @intCast(rows.len) };
    for (rows, 0..) |*r, i| {
        const p = partsOf(if (i == 0) null else &rows[i - 1], r);
        for (&s.total, p.ns) |*a, b| a.* += b;
        s.disorder += p.disorder;
        s.serial += @intFromBool(r.serial);
        s.drafted_rounds += @intFromBool(r.drafted > 0);
    }
    if (rows.len > 0) s.span = filled(&rows[rows.len - 1])[n_points - 1] -| filled(&rows[0])[0];
    return s;
}

pub const line_name = "GLM5_NEXT_DECODE_ROUND";

fn ms(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e6;
}

fn median(xs: []f64) f64 {
    if (xs.len == 0) return 0;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

/// `GLM5_NEXT_DECODE_ROUND {...}` over the kept rows: the forwards counted (`rounds`) and kept, the serial steps, the rounds
/// that drafted, the out-of-order pairs, whether the TSV is written, the span and its in-round share (the span less the
/// host's parts: the plugin's decode wall), then per part the total, the mean per kept row and the median (ms). `scratch`
/// holds at least `rows.len` values.
pub fn writeLine(w: *std.Io.Writer, rows: []const Row, rounds: u64, scratch: []f64, tsv: bool) !void {
    std.debug.assert(scratch.len >= rows.len);
    const s = summarize(rows);
    try w.print(line_name ++ " {{\"rounds\": {d}, \"stored\": {d}, \"cap\": {d}, \"serial_steps\": {d}, \"drafted_rounds\": {d}, \"disorder\": {d}, \"tsv\": {}, \"clock\": \"CLOCK_UPTIME_RAW ns (the verify GPU timeline's)\", \"span_ms\": {d:.3}, \"in_round_ms\": {d:.3}, \"parts\": [", .{
        rounds, s.rows, max_rounds, s.serial, s.drafted_rounds, s.disorder, tsv, ms(s.span), ms(s.span -| s.total[@intFromEnum(Part.host)]),
    });
    inline for (@typeInfo(Part).@"enum".field_names, 0..) |name, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ", ", name });
    try w.writeAll("], \"total_ms\": {");
    inline for (@typeInfo(Part).@"enum".field_names, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, ms(s.total[i]) });
    try w.writeAll("}, \"mean_ms\": {");
    const n: f64 = @floatFromInt(@max(s.rows, 1));
    inline for (@typeInfo(Part).@"enum".field_names, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, ms(s.total[i]) / n });
    try w.writeAll("}, \"median_ms\": {");
    inline for (@typeInfo(Part).@"enum".field_names, 0..) |name, i| {
        for (rows, 0..) |*r, j| scratch[j] = ms(partsOf(if (j == 0) null else &rows[j - 1], r).ns[i]);
        try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, median(scratch[0..rows.len]) });
    }
    try w.writeAll("}}");
}

/// The kept rows, one a line: `cycle serial drafted verified kept emitted` then every point's stamp (`<point>_ns`, raw:
/// 0 = not stamped; readers apply `filled`'s rule).
pub fn writeTsv(w: *std.Io.Writer, rows: []const Row) !void {
    try w.writeAll("cycle\tserial\tdrafted\tverified\tkept\temitted");
    inline for (@typeInfo(Point).@"enum".field_names) |name| try w.writeAll("\t" ++ name ++ "_ns");
    try w.writeAll("\n");
    for (rows, 0..) |r, i| {
        try w.print("{d}\t{d}\t{d}\t{d}\t{d}\t{d}", .{ i, @intFromBool(r.serial), r.drafted, r.verified, r.kept, r.emitted });
        for (r.t) |x| try w.print("\t{d}", .{x});
        try w.writeAll("\n");
    }
}

const testing = std.testing;

/// Three forwards by hand (CLOCK_UPTIME_RAW ns): a round with drafts; a round whose append found nothing pending (its
/// catch-up and copy unstamped); a serial step (no draft points, no decision read, no pend).
const hand_rows = [3]Row{
    .{ .t = .{ 1000000000, 1000300000, 1000400000, 1001300000, 1013300000, 1013350000, 1013500000, 1073500000, 1073700000, 1093700000, 1093710000, 1093750000, 1094250000, 1094850000, 1094880000 }, .serial = false, .drafted = 3, .verified = 2, .kept = 1, .emitted = 2 },
    .{ .t = .{ 1095130000, 0, 0, 1096130000, 1108630000, 1108690000, 1108830000, 1163830000, 1164040000, 1185040000, 1185052000, 1185090000, 1185490000, 1186040000, 1186065000 }, .serial = false, .drafted = 3, .verified = 3, .kept = 3, .emitted = 4 },
    .{ .t = .{ 1186365000, 0, 0, 0, 0, 0, 1186445000, 1226445000, 1226545000, 1241545000, 1241555000, 0, 1241560000, 0, 1241580000 }, .serial = true, .drafted = 0, .verified = 0, .kept = 0, .emitted = 1 },
};

test "glm5_next round timers: the parts are the consecutive stamps; a missing point takes the previous one (0 ms); host = the time since the previous exit" {
    const rows = hand_rows;
    const p0 = partsOf(null, &rows[0]);
    try testing.expectEqual(@as(u64, 0), p0.ns[@intFromEnum(Part.host)]);
    try testing.expectEqual(@as(u64, 300_000), p0.ns[@intFromEnum(Part.append)]);
    try testing.expectEqual(@as(u64, 12_000_000), p0.ns[@intFromEnum(Part.chain_wait)]);
    try testing.expectEqual(@as(u64, 60_000_000), p0.ns[@intFromEnum(Part.fwd_encode)]);
    try testing.expectEqual(@as(u64, 600_000), p0.ns[@intFromEnum(Part.pend)]);
    const p1 = partsOf(&rows[0], &rows[1]);
    try testing.expectEqual(@as(u64, 250_000), p1.ns[@intFromEnum(Part.host)]);
    try testing.expectEqual(@as(u64, 0), p1.ns[@intFromEnum(Part.append)]);
    try testing.expectEqual(@as(u64, 0), p1.ns[@intFromEnum(Part.pending_copy)]);
    // copied (filled: enter) -> chain_built
    try testing.expectEqual(@as(u64, 1_000_000), p1.ns[@intFromEnum(Part.chain_encode)]);
    const p2 = partsOf(&rows[1], &rows[2]);
    try testing.expectEqual(@as(u64, 300_000), p2.ns[@intFromEnum(Part.host)]);
    // the serial step: enter -> verify_begin is its gate, flushed -> verify_end its rollback (decided filled), the rest of the
    // round's points 0
    try testing.expectEqual(@as(u64, 80_000), p2.ns[@intFromEnum(Part.gate)]);
    try testing.expectEqual(@as(u64, 0), p2.ns[@intFromEnum(Part.chain_wait)]);
    try testing.expectEqual(@as(u64, 0), p2.ns[@intFromEnum(Part.decision_read)]);
    try testing.expectEqual(@as(u64, 5_000), p2.ns[@intFromEnum(Part.rollback)]);
    try testing.expectEqual(@as(u64, 20_000), p2.ns[@intFromEnum(Part.bookkeep)]);
    var verify: u64 = 0;
    for (p2.ns, 0..) |x, i| verify += if (inVerify(@enumFromInt(@as(u8, @intCast(i))))) x else 0;
    try testing.expectEqual(rows[2].t[@intFromEnum(Point.verify_end)] - rows[2].t[@intFromEnum(Point.verify_begin)], verify);
    // the parts add up to the span (the first entry to the last return) by construction
    const s = summarize(&rows);
    var total: u64 = 0;
    for (s.total) |x| total += x;
    try testing.expectEqual(s.span, total);
    try testing.expectEqual(@as(u64, 241_580_000), s.span);
    try testing.expectEqual(@as(u32, 0), s.disorder);
    try testing.expectEqual(@as(u32, 1), s.serial);
    try testing.expectEqual(@as(u32, 2), s.drafted_rounds);
    // a stamp out of order gives a 0 part and is counted
    var bad = rows;
    bad[1].t[@intFromEnum(Point.chain_done)] = 1;
    const pb = partsOf(&bad[0], &bad[1]);
    try testing.expectEqual(@as(u32, 1), pb.disorder);
    try testing.expectEqual(@as(u64, 0), pb.ns[@intFromEnum(Part.chain_wait)]);
    try testing.expect(afterVerify(.pended) and afterVerify(.exit) and !afterVerify(.verify_end));
}

test "glm5_next round timers: the line and the TSV, byte for byte (the profile cell's timeline_report.py self-test parses these bytes)" {
    var scratch: [3]f64 = undefined;
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeLine(&w, &hand_rows, 3, &scratch, true);
    try testing.expectEqualStrings(expected_line, w.buffered());
    var tb: [4096]u8 = undefined;
    var tw: std.Io.Writer = .fixed(&tb);
    try writeTsv(&tw, &hand_rows);
    try testing.expectEqualStrings(expected_tsv, tw.buffered());
}

/// The exact output over `hand_rows`.
const expected_line = "GLM5_NEXT_DECODE_ROUND {\"rounds\": 3, \"stored\": 3, \"cap\": 16384, \"serial_steps\": 1, \"drafted_rounds\": 2, \"disorder\": 0, \"tsv\": true, \"clock\": \"CLOCK_UPTIME_RAW ns (the verify GPU timeline's)\", \"span_ms\": 241.580, \"in_round_ms\": 241.030, \"parts\": [\"host\", \"append\", \"pending_copy\", \"chain_encode\", \"chain_wait\", \"chain_read\", \"gate\", \"fwd_encode\", \"decision_build\", \"verify_wait\", \"flush\", \"decision_read\", \"rollback\", \"pend\", \"bookkeep\"], \"total_ms\": {\"host\": 0.550, \"append\": 0.300, \"pending_copy\": 0.100, \"chain_encode\": 1.900, \"chain_wait\": 24.500, \"chain_read\": 0.110, \"gate\": 0.370, \"fwd_encode\": 155.000, \"decision_build\": 0.510, \"verify_wait\": 56.000, \"flush\": 0.032, \"decision_read\": 0.078, \"rollback\": 0.905, \"pend\": 1.150, \"bookkeep\": 0.075}, \"mean_ms\": {\"host\": 0.183, \"append\": 0.100, \"pending_copy\": 0.033, \"chain_encode\": 0.633, \"chain_wait\": 8.167, \"chain_read\": 0.037, \"gate\": 0.123, \"fwd_encode\": 51.667, \"decision_build\": 0.170, \"verify_wait\": 18.667, \"flush\": 0.011, \"decision_read\": 0.026, \"rollback\": 0.302, \"pend\": 0.383, \"bookkeep\": 0.025}, \"median_ms\": {\"host\": 0.250, \"append\": 0.000, \"pending_copy\": 0.000, \"chain_encode\": 0.900, \"chain_wait\": 12.000, \"chain_read\": 0.050, \"gate\": 0.140, \"fwd_encode\": 55.000, \"decision_build\": 0.200, \"verify_wait\": 20.000, \"flush\": 0.010, \"decision_read\": 0.038, \"rollback\": 0.400, \"pend\": 0.550, \"bookkeep\": 0.025}}";

const expected_tsv = "cycle\tserial\tdrafted\tverified\tkept\temitted\tenter_ns\tappended_ns\tcopied_ns\tchain_built_ns\tchain_done_ns\tchain_read_ns\tverify_begin_ns\tfwd_built_ns\tdec_built_ns\tevaluated_ns\tflushed_ns\tdecided_ns\tverify_end_ns\tpended_ns\texit_ns\n" ++
    "0\t0\t3\t2\t1\t2\t1000000000\t1000300000\t1000400000\t1001300000\t1013300000\t1013350000\t1013500000\t1073500000\t1073700000\t1093700000\t1093710000\t1093750000\t1094250000\t1094850000\t1094880000\n" ++
    "1\t0\t3\t3\t3\t4\t1095130000\t0\t0\t1096130000\t1108630000\t1108690000\t1108830000\t1163830000\t1164040000\t1185040000\t1185052000\t1185090000\t1185490000\t1186040000\t1186065000\n" ++
    "2\t1\t0\t0\t0\t1\t1186365000\t0\t0\t0\t0\t0\t1186445000\t1226445000\t1226545000\t1241545000\t1241555000\t0\t1241560000\t0\t1241580000\n";
