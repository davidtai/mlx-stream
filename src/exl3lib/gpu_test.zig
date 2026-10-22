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
const root = @import("root.zig");

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
    // Qwen3.8-Flash-Next's geometry at the served rate: 1 row (decode), 4 and 8 rows (MTP verify widths)
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 4, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
    .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 8, .experts = 12, .n = 80, .dec = .mul1, .timed = true },
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
    // the launches apart: gate/up alone, then down alone on a fixed h
    var t_gu: [30]u64 = undefined;
    var t_dn: [30]u64 = undefined;
    const h_fixed = try k.gateUp(s, xa, sa, arrays, c.rows);
    defer _ = mlx.mlx_array_free(h_fixed);
    try mlx.check(mlx.mlx_array_eval(h_fixed));
    i = 0;
    while (i < 35) : (i += 1) {
        var t0 = nowNs();
        const hh = try k.gateUp(s, xa, sa, arrays, c.rows);
        try mlx.check(mlx.mlx_array_eval(hh));
        const d1 = nowNs() - t0;
        _ = mlx.mlx_array_free(hh);
        t0 = nowNs();
        const yy = try k.downReduce(s, h_fixed, sa, wa, arrays, c.rows);
        try mlx.check(mlx.mlx_array_eval(yy));
        const d2 = nowNs() - t0;
        _ = mlx.mlx_array_free(yy);
        if (i >= 5) {
            t_gu[i - 5] = d1;
            t_dn[i - 5] = d2;
        }
    }
    std.mem.sort(u64, &t_gu, {}, std.sort.asc(u64));
    std.mem.sort(u64, &t_dn, {}, std.sort.asc(u64));
    const dist = @as(f64, @floatFromInt(c.rows)) * bytes; // every row its own experts here: bytes scale with rows
    std.debug.print("EXL3LIB_TIME n={d} geo={d}x{d} rows={d} topk={d} median_us={d:.1} p10_us={d:.1} p90_us={d:.1} gu_us={d:.1} down_us={d:.1} expert_gbps={d:.1}\n", .{
        c.n,                                        c.hidden,                                   c.inter,                      c.rows, c.topk, med_us, @as(f64, @floatFromInt(times[3])) / 1000.0, @as(f64, @floatFromInt(times[27])) / 1000.0,
        @as(f64, @floatFromInt(t_gu[15])) / 1000.0, @as(f64, @floatFromInt(t_dn[15])) / 1000.0, dist / (med_us * 1e-6) / 1e9,
    });
    // the ALU shares: the same chain without the codebook hash (probe 1) and without the funnel either (probe 2)
    for ([_]u8{ 1, 2 }) |probe| {
        var kp = try decode.Kernels.init(.{ .hidden = c.hidden, .inter = c.inter, .topk = c.topk, .rate = rate, .dec = c.dec, .probe = probe });
        defer kp.deinit();
        var tp: [30]u64 = undefined;
        i = 0;
        while (i < 35) : (i += 1) {
            const t0 = nowNs();
            const o = try kp.moe(s, xa, sa, wa, arrays, c.rows);
            try mlx.check(mlx.mlx_array_eval(o.y));
            const dt = nowNs() - t0;
            _ = mlx.mlx_array_free(o.h);
            _ = mlx.mlx_array_free(o.y);
            if (i >= 5) tp[i - 5] = dt;
        }
        std.mem.sort(u64, &tp, {}, std.sort.asc(u64));
        std.debug.print("EXL3LIB_PROBE n={d} rows={d} probe={d} median_us={d:.1} p10_us={d:.1}\n", .{ c.n, c.rows, probe, @as(f64, @floatFromInt(tp[15])) / 1000.0, @as(f64, @floatFromInt(tp[3])) / 1000.0 });
    }
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

// The host's path: `root.moe` on a bf16 x [1, rows, D], i32 inds [1, rows, K] and bf16 scores, output in x's dtype.
test "exl3lib gpu: root.moe (the sushi_exl3-shaped entry) equals the reference through the bf16 host operands (EXL3LIB_GPU=1)" {
    _ = std.c.getenv("EXL3LIB_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: Case = .{ .hidden = 2560, .inter = 640, .topk = 10, .rows = 3, .experts = 12, .n = 80, .dec = .mul1 };
    var prng = std.Random.DefaultPrng.init(0xB16);
    const rnd = prng.random();
    const rate = format.Rate.fromN(c.n).?;
    var bank = try ref.synthBank(a, rnd, c.experts, c.hidden, c.inter, rate);
    defer bank.deinit(a);
    const nslots = c.rows * c.topk;
    const x = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(x);
    const bf = @import("../arch/deepseek_v41/ops.zig").bf16Bits;
    for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16); // bf16-exact inputs
    const slots = try a.alloc(u32, nslots);
    defer a.free(slots);
    const scores = try a.alloc(f32, nslots);
    defer a.free(scores);
    for (0..c.rows) |r| for (0..c.topk) |k| {
        slots[r * c.topk + k] = @intCast((k * 2 + r) % c.experts);
        scores[r * c.topk + k] = @bitCast(@as(u32, bf(0.05 + 0.1 * rnd.float(f32))) << 16); // bf16-exact scores
    };
    const y_ref = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(y_ref);
    for (0..c.rows) |r| try ref.moeRowDec(a, &bank, c.dec, x[r * c.hidden ..][0..c.hidden], slots[r * c.topk ..][0..c.topk], scores[r * c.topk ..][0..c.topk], y_ref[r * c.hidden ..][0..c.hidden]);
    const ht = ci(c.hidden / 16);
    const it = ci(c.inter / 16);
    const bank_arrays: root.Bank = .{
        .gate = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(c.experts), ci(c.inter) }, .float16) },
        .up = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(c.experts), ht, it, ci(c.n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(c.experts), ci(c.hidden) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(c.experts), ci(c.inter) }, .float16) },
        .down = .{ .trellis = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(c.experts), it, ht, ci(c.n) }, .uint16), .suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(c.experts), ci(c.inter) }, .float16), .svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(c.experts), ci(c.hidden) }, .float16) },
    };
    const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(c.rows), ci(c.hidden) }, .float32), .bfloat16);
    const inds_i32 = try a.alloc(i32, nslots);
    defer a.free(inds_i32);
    for (slots, inds_i32) |sl, *d| d.* = @intCast(sl);
    const inds3 = try g.hostArray(std.mem.sliceAsBytes(inds_i32), &.{ 1, ci(c.rows), ci(c.topk) }, .int32);
    const sc3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(c.rows), ci(c.topk) }, .float32), .bfloat16);
    const y = try g.adopt(try root.moe(s, x3, bank_arrays, inds3, sc3, c.dec, false));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
    const got = try a.alloc(f32, c.rows * c.hidden);
    defer a.free(got);
    _ = try g.hostF32(try g.astype(y, .float32), got);
    const e = ref.compare(got, y_ref);
    std.debug.print("EXL3LIB_PARITY root.moe bf16 geo={d}x{d} rows={d} topk={d} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{ c.hidden, c.inter, c.rows, c.topk, e.max_abs, e.rms_ref, e.max_ratio, if (e.max_ratio <= 1.5e-2) "PASS" else "FAIL" });
    try testing.expect(e.max_ratio <= 1.5e-2); // the bf16 output rounding (2^-8) on top of the chain's tolerance
    // a prefill-width call refuses by name (the host keeps its arm)
    const xw = try g.zeros(&.{ 1, 32, ci(c.hidden) }, .bfloat16);
    const iw = try g.zeros(&.{ 1, 32, ci(c.topk) }, .int32);
    const sw = try g.zeros(&.{ 1, 32, ci(c.topk) }, .bfloat16);
    try testing.expectError(error.Exl3EngineNotImplemented, root.moe(s, xw, bank_arrays, iw, sw, c.dec, false));
}

// The dense linear (the full-EXL3 trunk's path): rows 1, 4 and 16 through the GEMV kernels, 40 through dequantize-and-matmul,
// at K7 (the attention and GDN projections' rate), K6 (lm_head, MTP fc) and K5, both directions of a 2560 x 640 shape.
test "exl3lib gpu: dense.linear equals format.project per row at the trunk's rates, GEMV and prefill paths (EXL3LIB_GPU=1)" {
    _ = std.c.getenv("EXL3LIB_GPU") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const Shape = struct { in: u32, out: u32, n: u32, dec: format.Decode };
    const shapes = [_]Shape{
        .{ .in = 2560, .out = 640, .n = 112, .dec = .mul1 },
        .{ .in = 640, .out = 2560, .n = 112, .dec = .mul1 },
        .{ .in = 2560, .out = 512, .n = 96, .dec = .mul1 },
        .{ .in = 2560, .out = 640, .n = 80, .dec = .mul1 },
        .{ .in = 512, .out = 256, .n = 48, .dec = .{ .codebook = .mcg, .window = .w12 } },
    };
    const row_counts = [_]u32{ 1, 4, 16, 40 };
    for (shapes) |sh| {
        var prng = std.Random.DefaultPrng.init(0xD3A5E ^ sh.n ^ sh.out);
        const rnd = prng.random();
        const rate = format.Rate.fromN(sh.n).?;
        // one expert of a synthetic bank: its gate projection is a dense in -> out linear
        var bank = try ref.synthBank(a, rnd, 1, sh.in, sh.out, rate);
        defer bank.deinit(a);
        const ht = ci(sh.in / 16);
        const ot = ci(sh.out / 16);
        const mark = g.mark();
        defer g.resetTo(mark);
        const trellis = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ht, ot, ci(sh.n) }, .uint16);
        const suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ci(sh.in)}, .float16);
        const svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ci(sh.out)}, .float16);
        for (row_counts) |rows| {
            const x = try a.alloc(f32, rows * sh.in);
            defer a.free(x);
            const bf = @import("../arch/deepseek_v41/ops.zig").bf16Bits;
            for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16);
            const y_ref = try a.alloc(f32, rows * sh.out);
            defer a.free(y_ref);
            const transformed = try a.alloc(f32, sh.in);
            defer a.free(transformed);
            const inner = try a.alloc(f32, sh.out);
            defer a.free(inner);
            for (0..rows) |r| format.project(x[r * sh.in ..][0..sh.in], bank.gate_t, bank.gate_suh, bank.gate_svh, sh.in, sh.out, rate, sh.dec, transformed, inner, y_ref[r * sh.out ..][0..sh.out]);
            // f32 x: the numeric bar is the decode chain's (5e-3 of the reference rms); the output is f32 too
            const x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(sh.in) }, .float32);
            const y = try g.adopt(try root.linear(s, x3, trellis, suh, svh, sh.dec));
            try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(y));
            const ysh = mlx.getShape(y);
            try testing.expect(ysh.len == 3 and ysh[1] == ci(rows) and ysh[2] == ci(sh.out));
            const got = try a.alloc(f32, rows * sh.out);
            defer a.free(got);
            _ = try g.hostF32(y, got);
            const e = ref.compare(got, y_ref);
            std.debug.print("EXL3LIB_PARITY dense in={d} out={d} n={d} cb={s} rows={d} path={s} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{
                sh.in, sh.out, sh.n, @tagName(sh.dec.codebook), rows, if (rows <= 16) "gemv" else "matmul", e.max_abs, e.rms_ref, e.max_ratio, if (e.max_ratio <= 5e-3) "PASS" else "FAIL",
            });
            try testing.expect(e.max_ratio <= 5e-3);
            // bf16 x: the served dtype in and out (one bf16 ulp of rounding on top)
            if (rows == 4 or rows == 40) {
                const xb = try g.astype(x3, .bfloat16);
                const yb = try g.adopt(try root.linear(s, xb, trellis, suh, svh, sh.dec));
                try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(yb));
                _ = try g.hostF32(try g.astype(yb, .float32), got);
                const eb = ref.compare(got, y_ref);
                try testing.expect(eb.max_ratio <= 2.5e-2);
            }
        }
    }
}

// Real tensors: EXL3LIB_REAL_SHARD=<trunk-exl3-NN.safetensors of the full Qwen3.8 pack> runs root.linear on two of its trellis
// triples (a GDN in-projection and a shared-expert down-projection) against format.project on the same halfwords, bf16 x,
// rows 3 (GEMV path) and 40 (matmul path). Pins the pack-side interpretation (orientation, dtypes, suh/svh) on turboderp's data.
test "exl3lib gpu: root.linear on real trunk trellises from the full pack equals format.project (EXL3LIB_REAL_SHARD)" {
    _ = std.c.getenv("EXL3LIB_GPU") orelse return error.SkipZigTest;
    const shard_c = std.c.getenv("EXL3LIB_REAL_SHARD") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    const path = std.mem.span(shard_c);
    // the safetensors header
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var hl_buf: [8]u8 = undefined;
    try testing.expectEqual(@as(isize, 8), std.c.pread(fd, &hl_buf, 8, 0));
    const hlen: usize = @intCast(std.mem.readInt(u64, &hl_buf, .little));
    const hdr = try a.alloc(u8, hlen);
    defer a.free(hdr);
    try testing.expectEqual(@as(isize, @intCast(hlen)), std.c.pread(fd, hdr.ptr, hlen, 8));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, hdr, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const data_off: usize = 8 + hlen;
    const Tensor = struct { bytes: []u8, shape: [3]c_int, ndim: usize };
    const readT = struct {
        fn run(alloc: std.mem.Allocator, o: std.json.ObjectMap, f: c_int, base: usize, name: []const u8) !Tensor {
            const t = (o.get(name) orelse return error.MissingWeight).object;
            const offs = t.get("data_offsets").?.array.items;
            const o0: usize = @intCast(offs[0].integer);
            const o1: usize = @intCast(offs[1].integer);
            const bytes = try alloc.alloc(u8, o1 - o0);
            if (std.c.pread(f, bytes.ptr, bytes.len, @intCast(base + o0)) != @as(isize, @intCast(bytes.len))) return error.ShortRead;
            const dims = t.get("shape").?.array.items;
            var sh: [3]c_int = .{ 1, 1, 1 };
            for (dims, 0..) |d, i| sh[i] = @intCast(d.integer);
            return .{ .bytes = bytes, .shape = sh, .ndim = dims.len };
        }
    }.run;
    const bases = [_][]const u8{ "language_model.model.layers.4.linear_attn.in_proj_qkv", "language_model.model.layers.4.mlp.shared_expert.down_proj" };
    for (bases) |base| {
        var nb: [200]u8 = undefined;
        const w = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
        defer a.free(w.bytes);
        var nb2: [200]u8 = undefined;
        const suh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb2, "{s}.scales", .{base}));
        defer a.free(suh.bytes);
        var nb3: [200]u8 = undefined;
        const svh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb3, "{s}.biases", .{base}));
        defer a.free(svh.bytes);
        try testing.expectEqual(@as(usize, 3), w.ndim);
        const in_dim: u32 = @intCast(w.shape[0] * 16);
        const out_dim: u32 = @intCast(w.shape[1] * 16);
        const n: u32 = @intCast(w.shape[2]);
        const rate = format.Rate.fromN(n).?;
        try testing.expectEqual(in_dim, @as(u32, @intCast(suh.shape[0])));
        try testing.expectEqual(out_dim, @as(u32, @intCast(svh.shape[0])));
        const trellis = std.mem.bytesAsSlice(u16, w.bytes);
        const suh16 = std.mem.bytesAsSlice(u16, suh.bytes);
        const svh16 = std.mem.bytesAsSlice(u16, svh.bytes);
        // suh/svh sanity: finite, O(1) scales
        var smin: f32 = 1e9;
        var smax: f32 = -1e9;
        for (suh16) |v| {
            const f: f32 = @floatCast(@as(f16, @bitCast(v)));
            smin = @min(smin, f);
            smax = @max(smax, f);
        }
        std.debug.print("EXL3LIB_REAL {s}: in={d} out={d} n={d} suh[min,max]=[{d:.4}, {d:.4}]\n", .{ base, in_dim, out_dim, n, smin, smax });
        const mark = g.mark();
        defer g.resetTo(mark);
        const ta = try g.hostArray(w.bytes, &.{ w.shape[0], w.shape[1], w.shape[2] }, .uint16);
        const sa = try g.hostArray(suh.bytes, &.{ci(in_dim)}, .float16);
        const va = try g.hostArray(svh.bytes, &.{ci(out_dim)}, .float16);
        var prng = std.Random.DefaultPrng.init(0xBEEF ^ n);
        const rnd = prng.random();
        for ([_]u32{ 3, 40 }) |rows| {
            const x = try a.alloc(f32, rows * in_dim);
            defer a.free(x);
            const bf = @import("../arch/deepseek_v41/ops.zig").bf16Bits;
            for (x) |*v| v.* = @bitCast(@as(u32, bf(rnd.floatNorm(f32))) << 16);
            const y_ref = try a.alloc(f32, rows * out_dim);
            defer a.free(y_ref);
            const transformed = try a.alloc(f32, in_dim);
            defer a.free(transformed);
            const inner = try a.alloc(f32, out_dim);
            defer a.free(inner);
            for (0..rows) |r| format.project(x[r * in_dim ..][0..in_dim], @alignCast(trellis), @alignCast(suh16), @alignCast(svh16), in_dim, out_dim, rate, .mul1, transformed, inner, y_ref[r * out_dim ..][0..out_dim]);
            const x3 = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(in_dim) }, .float32), .bfloat16);
            const y = try g.adopt(try root.linear(s, x3, ta, sa, va, .mul1));
            const got = try a.alloc(f32, rows * out_dim);
            defer a.free(got);
            _ = try g.hostF32(try g.astype(y, .float32), got);
            const e = ref.compare(got, y_ref);
            std.debug.print("EXL3LIB_REAL_PARITY {s} rows={d} path={s} max_abs={e:.3} rms={e:.3} ratio={e:.3} {s}\n", .{ base, rows, if (rows <= 16) "gemv" else "matmul", e.max_abs, e.rms_ref, e.max_ratio, if (e.max_ratio <= 2.5e-2) "PASS" else "FAIL" });
            try testing.expect(e.max_ratio <= 2.5e-2);
        }
    }
}

// CPU-only tool. EXL3LIB_PERM_FILE names a small safetensors file holding, per trunk module <base>, the EXL3 triple as
// "exl3:<base>.{weight,scales,biases}" and the mixed pack's affine 8-bit triple as "aff:<base>.{weight,scales,biases}"
// (weight U32 [out, in/4], scales/biases BF16 [out, in/group]). It reconstructs the EXL3 public weight [in, out]
// (format.reconstructPublic) and the affine weight [out, in], then reports how the EXL3 output columns map onto the
// affine output rows: per 128-block identity agreement and, for a sample of columns, the best-matching affine row.
// A permutation here means turboderp's export orders a projection's outputs differently from mlx-serve's converter.
test "exl3lib tool: trunk column order, EXL3 vs the affine pack (EXL3LIB_PERM_FILE)" {
    const path_c = std.c.getenv("EXL3LIB_PERM_FILE") orelse return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const path = std.mem.span(path_c);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var hl_buf: [8]u8 = undefined;
    try testing.expectEqual(@as(isize, 8), std.c.pread(fd, &hl_buf, 8, 0));
    const hlen: usize = @intCast(std.mem.readInt(u64, &hl_buf, .little));
    const hdr = try a.alloc(u8, hlen);
    defer a.free(hdr);
    try testing.expectEqual(@as(isize, @intCast(hlen)), std.c.pread(fd, hdr.ptr, hlen, 8));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, hdr, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const data_off: usize = 8 + hlen;
    const Tensor = struct { bytes: []u8, shape: [3]usize, ndim: usize };
    const readT = struct {
        fn run(alloc: std.mem.Allocator, o: std.json.ObjectMap, f: c_int, base: usize, name: []const u8) !Tensor {
            const t = (o.get(name) orelse return error.MissingWeight).object;
            const offs = t.get("data_offsets").?.array.items;
            const o0: usize = @intCast(offs[0].integer);
            const o1: usize = @intCast(offs[1].integer);
            const bytes = try alloc.alloc(u8, o1 - o0);
            if (std.c.pread(f, bytes.ptr, bytes.len, @intCast(base + o0)) != @as(isize, @intCast(bytes.len))) return error.ShortRead;
            const dims = t.get("shape").?.array.items;
            var sh: [3]usize = .{ 1, 1, 1 };
            for (dims, 0..) |d, i| sh[i] = @intCast(d.integer);
            return .{ .bytes = bytes, .shape = sh, .ndim = dims.len };
        }
    }.run;
    var it = obj.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.startsWith(u8, k, "exl3:") or !std.mem.endsWith(u8, k, ".weight")) continue;
        const base = k[5 .. k.len - 7];
        var nb: [6][256]u8 = undefined;
        const w = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[0], "exl3:{s}.weight", .{base}));
        defer a.free(w.bytes);
        const suh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[1], "exl3:{s}.scales", .{base}));
        defer a.free(suh.bytes);
        const svh = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[2], "exl3:{s}.biases", .{base}));
        defer a.free(svh.bytes);
        const aw = readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[3], "aff:{s}.weight", .{base})) catch {
            std.debug.print("EXL3LIB_PERM {s}: no affine counterpart\n", .{base});
            continue;
        };
        defer a.free(aw.bytes);
        const asc = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[4], "aff:{s}.scales", .{base}));
        defer a.free(asc.bytes);
        const abi = try readT(a, obj, fd, data_off, try std.fmt.bufPrint(&nb[5], "aff:{s}.biases", .{base}));
        defer a.free(abi.bytes);
        const in_dim = w.shape[0] * 16;
        const out_dim = w.shape[1] * 16;
        const n: u32 = @intCast(w.shape[2]);
        const rate = format.Rate.fromN(n) orelse return error.BadExl3Shape;
        try testing.expectEqual(out_dim, aw.shape[0]);
        try testing.expectEqual(in_dim / 4, aw.shape[1]);
        const gs = in_dim / asc.shape[1];
        // EXL3 public weight [in, out]
        const we16 = try a.alloc(u16, in_dim * out_dim);
        defer a.free(we16);
        const trellis: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, w.bytes));
        const suh16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, suh.bytes));
        const svh16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, svh.bytes));
        try format.reconstructPublic(a, trellis, suh16, svh16, in_dim, out_dim, rate, .mul1, we16);
        // normalized EXL3 columns ce[c][i], with their norms
        const ce = try a.alloc(f32, in_dim * out_dim);
        defer a.free(ce);
        const ne = try a.alloc(f32, out_dim);
        defer a.free(ne);
        for (0..out_dim) |c| {
            var ss: f64 = 0;
            for (0..in_dim) |i| {
                const v = format.f16BitsToF32(we16[i * out_dim + c]);
                ce[c * in_dim + i] = v;
                ss += @as(f64, v) * v;
            }
            const nrm: f32 = @floatCast(@sqrt(ss) + 1e-20);
            ne[c] = nrm;
            for (0..in_dim) |i| ce[c * in_dim + i] /= nrm;
        }
        // normalized affine rows ra[o][i] = (scale * q + bias) / norm
        const ra = try a.alloc(f32, in_dim * out_dim);
        defer a.free(ra);
        const na = try a.alloc(f32, out_dim);
        defer a.free(na);
        const aw32: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, aw.bytes));
        const asc16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, asc.bytes));
        const abi16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, abi.bytes));
        const groups = in_dim / gs;
        for (0..out_dim) |o| {
            var ss: f64 = 0;
            for (0..in_dim) |i| {
                const q: f32 = @floatFromInt((aw32[o * (in_dim / 4) + i / 4] >> @intCast(8 * (i % 4))) & 0xff);
                const s: f32 = @bitCast(@as(u32, asc16[o * groups + i / gs]) << 16);
                const b: f32 = @bitCast(@as(u32, abi16[o * groups + i / gs]) << 16);
                const v = s * q + b;
                ra[o * in_dim + i] = v;
                ss += @as(f64, v) * v;
            }
            const nrm: f32 = @floatCast(@sqrt(ss) + 1e-20);
            na[o] = nrm;
            for (0..in_dim) |i| ra[o * in_dim + i] /= nrm;
        }
        const dot = struct {
            fn run(x: []const f32, y: []const f32) f32 {
                var s: f32 = 0;
                for (x, y) |p, q| s += p * q;
                return s;
            }
        }.run;
        // identity agreement per 128-block
        const nblk = out_dim / 128;
        var blocks: [512]u8 = undefined;
        var ident: usize = 0;
        for (0..@min(nblk, blocks.len)) |bk| {
            var m: usize = 0;
            for (0..128) |j| {
                const c = bk * 128 + j;
                if (dot(ce[c * in_dim ..][0..in_dim], ra[c * in_dim ..][0..in_dim]) > 0.9) m += 1;
            }
            ident += m;
            blocks[bk] = if (m == 128) 'X' else if (m > 0) 'x' else '.';
        }
        std.debug.print("EXL3LIB_PERM {s} in={d} out={d} n={d} gs={d} identity_cols={d}/{d} blocks(128)={s}\n", .{ base, in_dim, out_dim, n, gs, ident, out_dim, blocks[0..@min(nblk, blocks.len)] });
        // a sample of columns: where does each one's best match sit?
        const samples: usize = 24;
        for (0..samples) |j| {
            const c = (j * out_dim) / samples + 5;
            var best: usize = 0;
            var best_v: f32 = -2;
            for (0..out_dim) |o| {
                const v = dot(ce[c * in_dim ..][0..in_dim], ra[o * in_dim ..][0..in_dim]);
                if (v > best_v) {
                    best_v = v;
                    best = o;
                }
            }
            const id_v = dot(ce[c * in_dim ..][0..in_dim], ra[c * in_dim ..][0..in_dim]);
            std.debug.print("  col {d:>6} -> row {d:>6} corr {d:.3} (identity corr {d:.3}) norm_ratio {d:.3}\n", .{ c, best, best_v, id_v, ne[c] / na[best] });
        }
    }
}
