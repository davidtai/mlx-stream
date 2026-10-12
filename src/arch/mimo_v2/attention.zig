//! MiMo prefill SDPA at the existing post-RoPE/post-value-scale boundary.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");
const mimo = @import("config.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

/// Installed prefill lane for logical M > 8, BF16 Q/K192 and V128. The
/// caller establishes these invariants once and keeps the native KV cache.
/// Zero coordinates preserve QK and the first 128 coordinates of PV; scale,
/// masks and sink logits retain their original meaning at width 192.
pub fn prefill(q: Array, k: Array, v: Array, scale: f32, mask: Array, sinks: Array, s: Stream) !Array {
    return prefillMode(q, k, v, scale, "array", mask, sinks, s);
}

/// `+mask` (exact): a global layer (no sinks, bottom-right causal over the
/// view [0, kL)) through MLX's built-in causal mode. No host mask array; the
/// fused kernel skips the masked blocks itself. MLX's causal predicate
/// (q_idx = kL - qL + r >= k_idx) is `config.attends` at window 0.
pub fn prefillCausal(q: Array, k: Array, v: Array, scale: f32, s: Stream) !Array {
    return prefillMode(q, k, v, scale, "causal", .{}, .{}, s);
}

fn prefillMode(q: Array, k: Array, v: Array, scale: f32, mode: [*:0]const u8, mask: Array, sinks: Array, s: Stream) !Array {
    const zero = mlx.mlx_array_new_float(0);
    defer drop(zero);
    const qp = try padLast(q, 64, zero, s);
    defer drop(qp);
    const kp = try padLast(k, 64, zero, s);
    defer drop(kp);
    const vp = try padLast(v, 128, zero, s);
    defer drop(vp);
    var fused: Array = .{};
    defer drop(fused);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&fused, qp, kp, vp, scale, mode, mask, sinks, true, s));
    const shape = mlx.getShape(q);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice(&out, fused, &.{ 0, 0, 0, 0 }, 4, &.{ shape[0], shape[1], shape[2], 128 }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

fn padLast(x: Array, comptime high: c_int, zero: Array, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_pad(&out, x, &.{3}, 1, &.{0}, 1, &.{high}, 1, zero, "constant", s));
    return out;
}

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

const Probe = struct {
    name: []const u8,
    rows: u32,
    keys: u32,
    kv_heads: u32,
    offset: u64,
    first_key: u64,
    window: u32,
    sinks: bool,
};
const probes = [_]Probe{
    .{ .name = "late_global_128", .rows = 128, .keys = 16384, .kv_heads = 4, .offset = 16256, .first_key = 0, .window = 0, .sinks = false },
    .{ .name = "sliding_128_sinks", .rows = 128, .keys = 255, .kv_heads = 8, .offset = 16384, .first_key = 16257, .window = 128, .sinks = true },
    .{ .name = "global_tail_9", .rows = 9, .keys = 16384, .kv_heads = 4, .offset = 16375, .first_key = 0, .window = 0, .sinks = false },
};
const heads: u64 = 64;
const cache_bytes: u64 = 256 << 20;
const compiler_bytes: u64 = 512 << 20;
const host_bytes: u64 = 128 << 20;
const ceiling: u64 = 3 << 30;

fn probeBound(p: Probe) u64 {
    const rows: u64 = p.rows;
    const keys: u64 = p.keys;
    const kv: u64 = p.kv_heads;
    // The cache fixture has another 128 backing rows beyond its visible view.
    // Four native payloads cover originals, host construction, scaled Q and
    // native contiguous/cast copies. Both attention graphs remain live.
    const native = 2 * (heads * rows * 192 + kv * (keys + 128) * (192 + 128));
    const scores = 3 * 4 * heads * rows * (keys + 1);
    const padded = 2 * 256 * (heads * rows + 2 * kv * keys);
    // Control, full padded output and the sliced output's materialization.
    const outputs = 2 * heads * rows * (128 + 256 + 128);
    // Two F32 casts plus two host arrays for numerical agreement.
    const metrics = 4 * 4 * heads * rows * 128;
    // Input mask plus a possible expanded bool-mask copy for each graph.
    const masks = rows * keys + 2 * heads * rows * keys;
    const sinks = 4 * 2 * heads;
    return 4 * native + scores + padded + outputs + metrics + masks + sinks + cache_bytes + compiler_bytes + host_bytes;
}

fn probeNext(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

fn bf16(value: f32) u16 {
    const word: u32 = @bitCast(value);
    return @intCast((word + 0x7fff + ((word >> 16) & 1)) >> 16);
}

fn probeInput(a: std.mem.Allocator, g: *ops.MlxOps, shape: []const c_int, seed: u32, amplitude: f32) !Array {
    var count: usize = 1;
    for (shape) |d| count *= @intCast(d);
    const values = try a.alloc(u16, count);
    defer a.free(values);
    var state = seed;
    for (values) |*value| {
        const bits = probeNext(&state);
        const exponent = 125 + ((bits >> 21) & 1);
        const word = (bits & 0x80000000) | (exponent << 23) | (((bits >> 10) & 127) << 16);
        value.* = bf16(@as(f32, @bitCast(word)) * amplitude);
    }
    return g.hostArray(std.mem.sliceAsBytes(values), shape, .bfloat16);
}

fn probeMask(a: std.mem.Allocator, g: *ops.MlxOps, p: Probe) !Array {
    const values = try a.alloc(bool, @as(usize, p.rows) * p.keys);
    defer a.free(values);
    for (0..p.rows) |r| for (0..p.keys) |k| {
        values[r * p.keys + k] = mimo.attends(p.offset + r, p.first_key + k, p.window);
    };
    return g.hostArray(std.mem.sliceAsBytes(values), &.{ @intCast(p.rows), @intCast(p.keys) }, .bool_);
}

fn probeDigest(a: std.mem.Allocator, g: *ops.MlxOps, inputs: []const Array) ![32]u8 {
    const mark = g.mark();
    defer g.resetTo(mark);
    try g.evalAll(inputs);
    var hash = std.crypto.hash.Blake3.init(.{});
    for (inputs) |x| {
        const n = mlx.mlx_array_size(x);
        hash.update(std.mem.sliceAsBytes(mlx.getShape(x)));
        if (mlx.mlx_array_dtype(x) == .bool_) {
            const values = try a.alloc(bool, n);
            defer a.free(values);
            _ = try g.hostBool(x, values);
            hash.update(std.mem.sliceAsBytes(values));
        } else {
            try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(x));
            const values = try a.alloc(f32, n);
            defer a.free(values);
            _ = try g.hostF32(try g.astype(x, .float32), values);
            hash.update(std.mem.sliceAsBytes(values));
        }
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}

fn probeAgreement(a: std.mem.Allocator, g: *ops.MlxOps, name: []const u8, actual: Array, reference: Array) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(reference), mlx.getShape(actual));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(actual));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference));
    const n = mlx.mlx_array_size(actual);
    const got = try a.alloc(f32, n);
    defer a.free(got);
    const want = try a.alloc(f32, n);
    defer a.free(want);
    _ = try g.hostF32(try g.astype(actual, .float32), got);
    _ = try g.hostF32(try g.astype(reference, .float32), want);
    var max_abs: f64 = 0;
    var diff_squared: f64 = 0;
    var reference_squared: f64 = 0;
    var actual_squared: f64 = 0;
    var dot: f64 = 0;
    for (got, want) |left, right| {
        if (!std.math.isFinite(left) or !std.math.isFinite(right)) return error.MimoAttentionNonFinite;
        const l: f64 = left;
        const r: f64 = right;
        const diff = l - r;
        max_abs = @max(max_abs, @abs(diff));
        diff_squared += diff * diff;
        reference_squared += r * r;
        actual_squared += l * l;
        dot += l * r;
    }
    if (reference_squared == 0 or actual_squared == 0) return error.MimoAttentionDegenerateProbe;
    const relative_l2 = @sqrt(diff_squared / reference_squared);
    const cosine = dot / @sqrt(actual_squared * reference_squared);
    std.debug.print("MIMO_PADDED_ATTENTION numerics: stage={s} values={d} max_abs={d:.9} relative_l2={d:.9} cosine={d:.9}\n", .{ name, n, max_abs, relative_l2, cosine });
    if (!std.math.isFinite(relative_l2) or !std.math.isFinite(cosine) or relative_l2 > 0.01 or cosine < 0.9999) return error.MimoAttentionAgreement;
}

// Pinned MLX full-kernel rejection for this negative probe. A different
// failure must stay latched rather than being cleared as an expected probe.
const probe_raw_rejection = "[scaled_dot_product_attention] force_fused=True but no fused " ++
    "kernel is available: the full attention kernel supports head dims " ++
    "{64, 72, 80, 96, 128, 192, 256} with matching query/value dims, " ++
    "or (query, value) head dims (96, 64), " ++
    "plus head dim 512 on NAX GPUs (float32 also requires TF32); got " ++
    "query head dim 192 and value head dim 128.";

fn probeRawRejectionMatches(msg: []const u8) bool {
    return std.mem.indexOf(u8, msg, probe_raw_rejection) != null;
}

test "mimo attention rejection diagnostic accepts pinned 192 128 force fused reason only" {
    // Pinned MLX scaled_dot_product_attention.cpp:757 and:853; string-only
    // regression for the negative probe, with no MLX/device/latch calls.
    const pinned = "[scaled_dot_product_attention] force_fused=True but no fused " ++
        "kernel is available: the full attention kernel supports head dims " ++
        "{64, 72, 80, 96, 128, 192, 256} with matching query/value dims, " ++
        "or (query, value) head dims (96, 64), " ++
        "plus head dim 512 on NAX GPUs (float32 also requires TF32); got " ++
        "query head dim 192 and value head dim 128.";
    try std.testing.expect(probeRawRejectionMatches(pinned));
    try std.testing.expect(!probeRawRejectionMatches("[METAL] Command buffer execution failed: Insufficient Memory"));
    try std.testing.expect(!probeRawRejectionMatches("[scaled_dot_product_attention] force_fused=True but no fused kernel is available: Resource limit exceeded"));
    const wrong_dims = try std.mem.replaceOwned(u8, std.testing.allocator, pinned, "query head dim 192 and value head dim 128", "query head dim 128 and value head dim 192");
    defer std.testing.allocator.free(wrong_dims);
    try std.testing.expect(!probeRawRejectionMatches(wrong_dims));
    const unforced = try std.mem.replaceOwned(u8, std.testing.allocator, pinned, "force_fused=True", "force_fused=False");
    defer std.testing.allocator.free(unforced);
    try std.testing.expect(!probeRawRejectionMatches(unforced));
}

// `+mask` on the CPU: MLX's causal mode and the host bool mask (`config.attends`, window 0) select the same scores on
// the real 64/4 heads and 192/128 widths, so the unfused CPU path returns the same bits. Offsets cross the 128-key
// window boundary and the 512-row tile boundary; the first case is a cold prompt tile.
test "mimo +mask CPU: causal mode equals the host array mask bit for bit across the window and tile boundaries" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const kv = @import("kv.zig");
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cases = [_]struct { rows: u32, offset: u32 }{ .{ .rows = 9, .offset = 0 }, .{ .rows = 32, .offset = 96 }, .{ .rows = 17, .offset = 127 }, .{ .rows = 24, .offset = 500 } };
    const scale = 1 / @sqrt(@as(f32, 192));
    for (cases) |c| {
        const keys = c.offset + c.rows;
        const q = try kv.fixtureInput(a, 64, c.rows, 192, c.offset + 3);
        defer drop(q);
        const k = try kv.fixtureInput(a, 4, keys, 192, 7);
        defer drop(k);
        const v = try kv.fixtureInput(a, 4, keys, 128, 13);
        defer drop(v);
        const values = try a.alloc(bool, @as(usize, c.rows) * keys);
        defer a.free(values);
        for (0..c.rows) |r| for (0..keys) |key| {
            values[r * keys + key] = mimo.attends(c.offset + r, key, 0);
        };
        const mask = mlx.mlx_array_new_data(values.ptr, &.{ @intCast(c.rows), @intCast(keys) }, 2, .bool_);
        defer drop(mask);
        var by_array: Array = .{};
        defer drop(by_array);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&by_array, q, k, v, scale, "array", mask, .{}, false, s));
        var by_mode: Array = .{};
        defer drop(by_mode);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&by_mode, q, k, v, scale, "causal", .{}, .{}, false, s));
        try kv.compareBits(u16, a, by_mode, by_array, s);
        std.debug.print("MIMO_MASK cpu rows={d} offset={d} keys={d} bit_equal=true\n", .{ c.rows, c.offset, keys });
    }
}

// `+mask` on the device (pre-ship gate, guarded): the served padded fused route in causal mode against the same route
// with the host array mask, byte for byte, on the real late-tile shape (512 rows over 16,384 keys) and a short tail.
test "mimo sdk2 GPU: +mask causal mode equals the host array mask byte for byte at 512 x 16384" {
    const enabled = std.c.getenv("MIMO_Q4_PADDED_ATTENTION") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const cases = [_]Probe{
        .{ .name = "global_tile_512", .rows = 512, .keys = 16384, .kv_heads = 4, .offset = 15872, .first_key = 0, .window = 0, .sinks = false },
        .{ .name = "global_tail_9", .rows = 9, .keys = 16384, .kv_heads = 4, .offset = 16375, .first_key = 0, .window = 0, .sinks = false },
    };
    // Inputs, both padded graphs (Q/K/V256 + output256 + slice), the mask and two expanded copies.
    const p = cases[0];
    const rows: u64 = p.rows;
    const keys: u64 = p.keys;
    const bound = 2 * (heads * rows * 192 + 4 * keys * 320) + 2 * 2 * 256 * (heads * rows + 2 * 4 * keys) + 2 * 2 * heads * rows * (256 + 128) + rows * keys + 2 * heads * rows * keys + cache_bytes + compiler_bytes + host_bytes;
    std.debug.print("MIMO_MASK static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoAttentionAdmission;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    for (cases) |c| {
        const mark = g.mark();
        defer {
            _ = mlx.mlx_synchronize(g.s);
            g.resetTo(mark);
        }
        const qt = try probeInput(a, &g, &.{ 1, @intCast(c.rows), 64, 192 }, 0x732eb179, 1);
        const q = try g.transposeAxes(qt, &.{ 0, 2, 1, 3 });
        const k = try probeInput(a, &g, &.{ 1, 4, @intCast(c.keys), 192 }, 0x61f953c7, 1);
        const v = try probeInput(a, &g, &.{ 1, 4, @intCast(c.keys), 128 }, 0xa18de563, 0.707);
        const mask = try probeMask(a, &g, c);
        const scale = 1 / @sqrt(@as(f32, 192));
        const by_array = try prefill(q, k, v, scale, mask, .{}, s);
        defer drop(by_array);
        const by_mode = try prefillCausal(q, k, v, scale, s);
        defer drop(by_mode);
        try @import("kv.zig").compareBits(u16, a, by_mode, by_array, s);
        std.debug.print("MIMO_MASK gpu stage={s} rows={d} keys={d} byte_equal=true\n", .{ c.name, c.rows, c.keys });
    }
}

fn probeRawRejected(q: Array, k: Array, v: Array, scale: f32, mask: Array, sinks: Array, s: Stream) !void {
    if (mlx.errorPending()) {
        std.debug.print("MIMO_PADDED_ATTENTION error before raw rejection remains latched: {s}\n", .{mlx.peekErrorName() orelse "missing latch"});
        return error.MimoAttentionPreexistingError;
    }
    var out: Array = .{};
    defer drop(out);
    try std.testing.expectError(error.MlxError, mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "array", mask, sinks, true, s)));
    if (!mlx.errorPending()) return error.MimoAttentionMissingRejection;
    if (!mlx.takeErrorIf(probe_raw_rejection)) {
        std.debug.print("MIMO_PADDED_ATTENTION unexpected raw rejection remains latched: {s}\n", .{mlx.peekErrorName() orelse "missing latch"});
        return error.MimoAttentionUnexpectedRejection;
    }
    std.debug.print("MIMO_PADDED_ATTENTION raw_192_128_rejected: {s}\n", .{probe_raw_rejection});
}

fn probeCase(a: std.mem.Allocator, g: *ops.MlxOps, p: Probe, reject_raw: bool) !void {
    const mark = g.mark();
    defer {
        _ = mlx.mlx_synchronize(g.s);
        g.resetTo(mark);
    }
    const rows: c_int = @intCast(p.rows);
    const keys: c_int = @intCast(p.keys);
    const kv: c_int = @intCast(p.kv_heads);
    const qt = try probeInput(a, g, &.{ 1, rows, 64, 192 }, 0x732eb179, 1);
    const q = try g.transposeAxes(qt, &.{ 0, 2, 1, 3 });
    const kb = try probeInput(a, g, &.{ 1, kv, keys + 128, 192 }, 0x61f953c7, 1);
    const vb = try probeInput(a, g, &.{ 1, kv, keys + 128, 128 }, 0xa18de563, 0.707);
    const k = try g.slice(kb, &.{ 0, 0, 0, 0 }, &.{ 1, kv, keys, 192 }, &.{ 1, 1, 1, 1 });
    const v = try g.slice(vb, &.{ 0, 0, 0, 0 }, &.{ 1, kv, keys, 128 }, &.{ 1, 1, 1, 1 });
    const mask = try probeMask(a, g, p);
    var sink: Array = .{};
    if (p.sinks) {
        var values: [64]u16 = undefined;
        for (&values, 0..) |*value, h| value.* = bf16(2 * @as(f32, @floatFromInt(h % 5)) - 2);
        sink = try g.hostArray(std.mem.sliceAsBytes(&values), &.{64}, .bfloat16);
    }
    try std.testing.expectEqualSlices(c_int, &.{ 1, 64, rows, 192 }, mlx.getShape(q));
    try std.testing.expectEqualSlices(c_int, &.{ 1, kv, keys, 192 }, mlx.getShape(k));
    try std.testing.expectEqualSlices(c_int, &.{ 1, kv, keys, 128 }, mlx.getShape(v));
    const inputs = [_]Array{ q, k, v, mask, sink };
    const live_inputs = inputs[0..if (p.sinks) @as(usize, 5) else 4];
    const before = try probeDigest(a, g, live_inputs);
    // Scaling remains tied to the original Q/K width, after partial RoPE.
    // V's fixture already includes the native 0.707 scale rounded to BF16.
    const scale = 1 / @sqrt(@as(f32, 192));
    if (reject_raw) try probeRawRejected(q, k, v, scale, mask, sink, g.s);
    var reference: Array = .{};
    defer drop(reference);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&reference, q, k, v, scale, "array", mask, sink, false, g.s));
    const actual = try prefill(q, k, v, scale, mask, sink, g.s);
    defer drop(actual);
    try g.evalAll(&.{ reference, actual });
    try std.testing.expectEqualSlices(c_int, &.{ 1, 64, rows, 128 }, mlx.getShape(actual));
    try probeAgreement(a, g, p.name, actual, reference);
    const after = try probeDigest(a, g, live_inputs);
    try std.testing.expectEqualSlices(u8, &before, &after);
}

test "mimo sdk2 GPU: MiMo padded prefill attention agrees at real BF16 global sliding and tail shapes" {
    const enabled = std.c.getenv("MIMO_Q4_PADDED_ATTENTION") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    var bound: u64 = 0;
    for (probes) |p| bound = @max(bound, probeBound(p));
    std.debug.print("MIMO_PADDED_ATTENTION static: heads={d} native_copies=4 fp32_scores=3 masks=2 cache={d} compiler={d} host={d} bound={d} ceiling={d}\n", .{ heads, cache_bytes, compiler_bytes, host_bytes, bound, ceiling });
    if (bound > ceiling) return error.MimoAttentionAdmission;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, cache_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    @import("sdk").memory.startFootprintInterval();
    {
        var g = try ops.MlxOps.init(std.testing.allocator, s);
        defer g.deinit();
        for (probes, 0..) |p, i| try probeCase(std.testing.allocator, &g, p, i == 0);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = @import("sdk").memory.processMemory();
    std.debug.print("MIMO_PADDED_ATTENTION peak: mlx={d} footprint_interval={d} bound={d} ceiling={d}\n", .{ peak, process.footprint_interval_peak, bound, ceiling });
    if (peak > ceiling or process.footprint_interval_peak > ceiling) return error.MimoAttentionPeakExceeded;
}
