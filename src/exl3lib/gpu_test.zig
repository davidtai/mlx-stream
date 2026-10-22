//! The decode chain's GPU parity and timing on whatever GPU runs the tests (EXL3LIB_GPU=1, else skipped): synthetic
//! banks (`ref.synthBank`) at several rates, codebooks and windows, the two-launch chain against the host reference
//! (`ref.moeRowDec`), then the chain's time per layer. Lines: EXL3LIB_PARITY per case, EXL3LIB_TIME per timed case.
//! Parity is a tolerance (max |err| over the reference's rms): the kernels reduce in a different order than the
//! reference and the f16 rounding points can flip on a last bit.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const G = @import("../arch/deepseek_v41/ops.zig").MlxOps;
const format = @import("format.zig");
const decode = @import("decode.zig");
const ref = @import("ref.zig");

const testing = std.testing;

const Case = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rows: u32,
    experts: u32,
    n: u32,
    dec: format.Decode,
    timed: bool = false,
};

const cases = [_]Case{
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 32, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 40, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 48, .dec = .{ .codebook = .mcg, .window = .w12 } },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 64, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 80, .dec = .mul1 },
    .{ .hidden = 512, .inter = 256, .topk = 4, .rows = 2, .experts = 6, .n = 128, .dec = .mcg },
    // Qwen3.8-Flash-Next's geometry at the served rate
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
};

const tolerance: f64 = 5e-3;

fn ci(x: u32) c_int {
    return @intCast(x);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn runCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, c: Case) !void {
    var prng = std.Random.DefaultPrng.init(0xE7135EED ^ c.n);
    const rnd = prng.random();
    const rate = format.Rate.fromN(c.n).?;
    var bank = try ref.synthBank(a, rnd, c.experts, c.hidden, c.inter, rate);
    defer bank.deinit(a);
    const nslots = c.rows * c.topk;
    const x = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, nslots);
    defer a.free(slots);
    const scores = try a.alloc(f32, nslots);
    defer a.free(scores);
    for (0..c.rows) |r| {
        var sum: f32 = 0;
        for (0..c.topk) |k| {
            // distinct experts per row, two rows sharing some
            slots[r * c.topk + k] = @intCast((k * 2 + r) % c.experts);
            scores[r * c.topk + k] = 0.2 + rnd.float(f32);
            sum += scores[r * c.topk + k];
        }
        for (0..c.topk) |k| scores[r * c.topk + k] /= sum;
    }
    // host reference
    const y_ref = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(y_ref);
    for (0..c.rows) |r| try ref.moeRowDec(a, &bank, c.dec, x[r * c.hidden ..][0..c.hidden], slots[r * c.topk ..][0..c.topk], scores[r * c.topk ..][0..c.topk], y_ref[r * c.hidden ..][0..c.hidden]);
    // device
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(c.hidden / 16);
    const it = ci(c.inter / 16);
    const arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(c.experts), ci(c.inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(c.experts), ci(c.inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(c.experts), it, ht, ci(c.n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(c.experts), ci(c.inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(c.experts), ci(c.hidden) }, .float16),
    };
    const xa = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(c.rows), ci(c.hidden) }, .float32);
    const sa = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(nslots)}, .uint32);
    const wa = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ci(nslots)}, .float32);
    var k = try decode.Kernels.init(.{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rate = rate, .dec = c.dec });
    defer k.deinit();
    const out = try k.moe(s, xa, sa, wa, arrays, c.rows);
    _ = try g.adopt(out.h);
    const y = try g.adopt(out.y);
    const got = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(got);
    _ = try g.hostF32(y, got);
    const e = ref.compare(got, y_ref);
    var wbuf: [8]u8 = undefined;
    std.debug.print("EXL3LIB_PARITY n={d} cb={s} w={d} geo={d}x{d} rows={d} topk={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{
        c.n,                                              @tagName(c.dec.codebook), c.dec.window.bits(), c.hidden, c.inter, c.rows, c.topk, e.max_abs, e.rms_ref, e.max_ratio,
        if (e.max_ratio <= tolerance) "PASS" else "FAIL",
    });
    _ = &wbuf;
    try testing.expect(std.math.isFinite(e.max_abs));
    try testing.expect(e.max_ratio <= tolerance);
    if (!c.timed) return;
    // timing: the two launches, evaluated, median of 30 after 5 warm
    var times: [30]u64 = undefined;
    var i: usize = 0;
    while (i < 35) : (i += 1) {
        const t0 = nowNs();
        const o = try k.moe(s, xa, sa, wa, arrays, c.rows);
        try mlx.check(mlx.mlx_array_eval(o.y));
        const dt = nowNs() - t0;
        _ = mlx.mlx_array_free(o.h);
        _ = mlx.mlx_array_free(o.y);
        if (i >= 5) times[i - 5] = dt;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const bytes = @as(f64, @floatFromInt(c.topk)) * 3.0 * @as(f64, @floatFromInt(c.hidden)) * @as(f64, @floatFromInt(c.inter)) * @as(f64, @floatFromInt(c.n)) / 128.0;
    const med_us = @as(f64, @floatFromInt(times[15])) / 1000.0;
    std.debug.print("EXL3LIB_TIME n={d} geo={d}x{d} rows={d} topk={d} median_us={d:.1} p10_us={d:.1} p90_us={d:.1} expert_gbps={d:.1}\n", .{
        c.n, c.hidden, c.inter, c.rows, c.topk, med_us, @as(f64, @floatFromInt(times[3])) / 1000.0, @as(f64, @floatFromInt(times[27])) / 1000.0, bytes / (med_us * 1e-6) / 1e9,
    });
}

test "exl3lib gpu: the two-launch decode chain equals the host reference at every rate and codebook (EXL3LIB_GPU=1)" {
    _ = std.c.getenv("EXL3LIB_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    for (cases) |c| try runCase(a, &g, s, c);
}
