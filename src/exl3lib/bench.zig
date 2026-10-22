//! The decode-chain benchmark (EXL3LIB_GPU=1 EXL3LIB_BENCH=1, else skipped): three arms on the same synthetic banks at
//! every integer rate K2..K8 (n 32..128) plus the fractional n40, Qwen3.8-Flash-Next's geometry (2560 x 640, top-10),
//! rows 1 / 4 / 8 (decode and MTP verify widths), 48 synthetic experts with rows sharing few of them (Flash-Next's
//! 512-expert routing shares few slots across verify rows, so weight reuse between rows is not what the bench rewards):
//!   bNkM    - exl3lib's form 3 (split-K planes in both GEMVs, finish launches fuse SwiGLU, the down prepare and the
//!             reduce) with N output tiles per GEMV threadgroup (2N f32 accumulators per lane) and up to M K slices
//!   sushi   - the host's Sushi module chain (`sushi_exl3.moe`, reached through `mlx_host`), the shipped control
//! Parity of every arm against the host reference first (an arm failing parity is reported, not timed); then rounds
//! interleaved over the arms with a rotated start, 3 warm, `rounds` timed; median / p10 us per layer and the effective
//! GB/s over the slots' trellis bytes. Lines: EXL3LIB_BENCH_PARITY, EXL3LIB_BENCH per (n, rows, arm). Relative numbers
//! only on a laptop GPU; the box's window gives the ones that count.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const host = @import("mlx_host");
const G = @import("../arch/deepseek_v41/ops.zig").MlxOps;
const format = @import("format.zig");
const decode = @import("decode.zig");
const unfused = @import("unfused.zig");
const ref = @import("ref.zig");

const testing = std.testing;
const has_sushi = @hasDecl(host, "sushi_exl3");
const sushi = if (has_sushi) host.sushi_exl3 else struct {};

fn ci(x: u32) c_int {
    return @intCast(x);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Arm = enum { b2k4g, b2k4, b1k4g, sushi };
const arms = [_]Arm{ .b2k4g, .b2k4, .b1k4g, .sushi };
/// form-3 variants: bN = N output tiles per GEMV threadgroup (2N accumulators per lane), kM = up to M K slices, g =
/// expert-grouped row-slots (two slots that chose the same expert share one lead threadgroup and one decode)
const rates = [_]u32{ 32, 40, 48, 64, 80, 96, 112, 128 };
const row_counts = [_]u32{ 1, 4, 8 };
const hidden: u32 = 2560;
const inter: u32 = 640;
const topk: u32 = 10;
const experts: u32 = 48;
const rounds: usize = 20;
const tolerance: f64 = 5e-3;

const Ctx = struct {
    s: mlx.mlx_stream,
    rows: u32,
    x2: mlx.mlx_array, // [rows, hidden] f32
    x3: mlx.mlx_array, // [1, rows, hidden] f32
    slots1: mlx.mlx_array, // [rows*topk] u32
    slots3: mlx.mlx_array, // [1, rows, topk] u32
    sc1: mlx.mlx_array,
    sc3: mlx.mlx_array,
    bank: decode.Kernels.Bank,
    forms: []*decode.Kernels3,
    dec: format.Decode,

    fn run(c: *const Ctx, arm: Arm) !mlx.mlx_array {
        switch (arm) {
            .b2k4g, .b2k4, .b1k4g => return c.forms[@backingInt(arm)].moe(c.s, c.x2, c.slots1, c.sc1, c.bank, c.rows),
            .sushi => {
                if (!has_sushi) return error.SkipZigTest;
                const b: sushi.Bank = .{
                    .gate = .{ .trellis = c.bank.gate_t, .suh = c.bank.gate_suh, .svh = c.bank.gate_svh },
                    .up = .{ .trellis = c.bank.up_t, .suh = c.bank.up_suh, .svh = c.bank.up_svh },
                    .down = .{ .trellis = c.bank.down_t, .suh = c.bank.down_suh, .svh = c.bank.down_svh },
                };
                const dec: sushi.format.Decode = .{ .codebook = @fromBackingInt(@intCast(@backingInt(c.dec.codebook))), .window = @fromBackingInt(@intCast(@backingInt(c.dec.window))) };
                return sushi.moe(c.s, c.x3, b, c.slots3, c.sc3, dec, false);
            },
        }
    }
};

fn benchCase(a: std.mem.Allocator, g: *G, s: mlx.mlx_stream, n: u32, rows: u32) !void {
    var prng = std.Random.DefaultPrng.init(0xBE4C ^ n ^ (rows << 8));
    const rnd = prng.random();
    const rate = format.Rate.fromN(n).?;
    const dec: format.Decode = .mul1;
    var bank = try ref.synthBank(a, rnd, experts, hidden, inter, rate);
    defer bank.deinit(a);
    const nslots = rows * topk;
    const x = try a.alloc(f32, rows * hidden);
    defer a.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const slots = try a.alloc(u32, nslots);
    defer a.free(slots);
    const scores = try a.alloc(f32, nslots);
    defer a.free(scores);
    for (0..rows) |r| {
        var sum: f32 = 0;
        for (0..topk) |k| {
            slots[r * topk + k] = @intCast((k * 5 + r * 11) % experts); // rows share few experts, as Flash-Next's 512-expert routing does
            scores[r * topk + k] = 0.2 + rnd.float(f32);
            sum += scores[r * topk + k];
        }
        for (0..topk) |k| scores[r * topk + k] /= sum;
    }
    const y_ref = try a.alloc(f32, rows * hidden);
    defer a.free(y_ref);
    for (0..rows) |r| try ref.moeRowDec(a, &bank, dec, x[r * hidden ..][0..hidden], slots[r * topk ..][0..topk], scores[r * topk ..][0..topk], y_ref[r * hidden ..][0..hidden]);
    const mark = g.mark();
    defer g.resetTo(mark);
    const ht = ci(hidden / 16);
    const it = ci(inter / 16);
    const bank_arrays: decode.Kernels.Bank = .{
        .gate_t = try g.hostArray(std.mem.sliceAsBytes(bank.gate_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .gate_suh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .gate_svh = try g.hostArray(std.mem.sliceAsBytes(bank.gate_svh), &.{ ci(experts), ci(inter) }, .float16),
        .up_t = try g.hostArray(std.mem.sliceAsBytes(bank.up_t), &.{ ci(experts), ht, it, ci(n) }, .uint16),
        .up_suh = try g.hostArray(std.mem.sliceAsBytes(bank.up_suh), &.{ ci(experts), ci(hidden) }, .float16),
        .up_svh = try g.hostArray(std.mem.sliceAsBytes(bank.up_svh), &.{ ci(experts), ci(inter) }, .float16),
        .down_t = try g.hostArray(std.mem.sliceAsBytes(bank.down_t), &.{ ci(experts), it, ht, ci(n) }, .uint16),
        .down_suh = try g.hostArray(std.mem.sliceAsBytes(bank.down_suh), &.{ ci(experts), ci(inter) }, .float16),
        .down_svh = try g.hostArray(std.mem.sliceAsBytes(bank.down_svh), &.{ ci(experts), ci(hidden) }, .float16),
    };
    const geo: decode.Geometry = .{ .hidden = hidden, .inter = inter, .topk = topk, .rate = rate, .dec = dec };
    const variants = [_][3]u32{ .{ 2, 4, 2 }, .{ 2, 4, 1 }, .{ 1, 4, 2 } };
    var forms_store: [variants.len]decode.Kernels3 = undefined;
    var forms: [variants.len]*decode.Kernels3 = undefined;
    for (variants, 0..) |v, i| {
        forms_store[i] = try decode.Kernels3.initWith(geo, v[0], v[1], v[2]);
        forms[i] = &forms_store[i];
    }
    defer for (&forms_store) |*f| f.deinit();
    const c: Ctx = .{
        .s = s,
        .rows = rows,
        .x2 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ ci(rows), ci(hidden) }, .float32),
        .x3 = try g.hostArray(std.mem.sliceAsBytes(x), &.{ 1, ci(rows), ci(hidden) }, .float32),
        .slots1 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ci(nslots)}, .uint32),
        .slots3 = try g.hostArray(std.mem.sliceAsBytes(slots), &.{ 1, ci(rows), ci(topk) }, .uint32),
        .sc1 = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ci(nslots)}, .float32),
        .sc3 = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, ci(rows), ci(topk) }, .float32),
        .bank = bank_arrays,
        .forms = &forms,
        .dec = dec,
    };
    // parity
    var timed: [arms.len]bool = @splat(false);
    const got = try a.alloc(f32, rows * hidden);
    defer a.free(got);
    for (arms, 0..) |arm, ai| {
        const y = c.run(arm) catch |err| {
            if (err == error.SkipZigTest) continue;
            std.debug.print("EXL3LIB_BENCH_PARITY n={d} rows={d} arm={s} ERROR {s}\n", .{ n, rows, @tagName(arm), @errorName(err) });
            continue;
        };
        const y2 = try g.adopt(y);
        var yr = y2;
        if (mlx.mlx_array_dtype(y2) != .float32) yr = try g.astype(y2, .float32);
        var flat = yr;
        if (mlx.mlx_array_ndim(yr) != 2) flat = try g.reshape(yr, &.{ ci(rows), ci(hidden) });
        _ = try g.hostF32(flat, got);
        const e = ref.compare(got, y_ref);
        timed[ai] = e.max_ratio <= tolerance;
        std.debug.print("EXL3LIB_BENCH_PARITY n={d} rows={d} arm={s} max_abs={e:.3} ratio={e:.3} {s}\n", .{ n, rows, @tagName(arm), e.max_abs, e.max_ratio, if (timed[ai]) "PASS" else "FAIL" });
    }
    // timing: interleaved rounds, rotated start
    var samples: [arms.len][rounds]u64 = undefined;
    var r: usize = 0;
    while (r < rounds + 3) : (r += 1) {
        var ai: usize = 0;
        while (ai < arms.len) : (ai += 1) {
            const i = (ai + r) % arms.len;
            if (!timed[i]) continue;
            const t0 = nowNs();
            const y = try c.run(arms[i]);
            try mlx.check(mlx.mlx_array_eval(y));
            const dt = nowNs() - t0;
            _ = mlx.mlx_array_free(y);
            if (r >= 3) samples[i][r - 3] = dt;
        }
        _ = mlx.mlx_clear_cache();
    }
    const bytes = @as(f64, @floatFromInt(topk)) * 3.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter)) * @as(f64, @floatFromInt(n)) / 128.0 * @as(f64, @floatFromInt(rows));
    for (arms, 0..) |arm, ai| {
        if (!timed[ai]) continue;
        std.mem.sort(u64, &samples[ai], {}, std.sort.asc(u64));
        const med = @as(f64, @floatFromInt(samples[ai][rounds / 2])) / 1000.0;
        const p10 = @as(f64, @floatFromInt(samples[ai][rounds / 10])) / 1000.0;
        std.debug.print("EXL3LIB_BENCH n={d} k={d}.{d:0>2} rows={d} arm={s} median_us={d:.1} p10_us={d:.1} gbps={d:.1}\n", .{ n, n / 16, (n % 16) * 100 / 16, rows, @tagName(arm), med, p10, bytes / (med * 1e-6) / 1e9 });
    }
}

test "exl3lib bench: fused vs separately launched steps vs Sushi's chain, K2..K8 (+ n40), Qwen geometry, rows 1/4/8 (EXL3LIB_BENCH=1)" {
    _ = std.c.getenv("EXL3LIB_GPU") orelse return error.SkipZigTest;
    _ = std.c.getenv("EXL3LIB_BENCH") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const a = testing.allocator;
    var g = try G.init(a, s);
    defer g.deinit();
    std.debug.print("EXL3LIB_BENCH_START geo={d}x{d} topk={d} experts={d} rounds={d} sushi={}\n", .{ hidden, inter, topk, experts, rounds, has_sushi });
    for (rates) |n| for (row_counts) |rows| {
        try benchCase(a, &g, s, n, rows);
    };
    std.debug.print("EXL3LIB_BENCH_END\n", .{});
}
