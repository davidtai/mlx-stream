//! Device-only attribution of installed Q8 cache updates, without attention.
//! Production arithmetic and evaluation boundaries remain in kv.zig.
const std = @import("std");
const mlx = @import("sdk").mlx;
const kv = @import("kv.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

fn stamp() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn settle(cache: *const kv.Cache, s: Stream) !void {
    const arrays = [_]Array{ cache.k.q, cache.k.scales, cache.k.biases, cache.v.q, cache.v.scales, cache.v.biases };
    const roots = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(roots);
    try mlx.check(mlx.mlx_eval(roots));
    try mlx.check(mlx.mlx_synchronize(s));
}

fn scenario(comptime full: bool, comptime chunk: u32, s: Stream) !void {
    const heads: u32 = if (full) 4 else 8;
    const initial: u32 = if (full) 16_384 else 128;
    var cache: kv.Cache = .{};
    defer cache.deinit();
    if (full) try cache.allocate(heads, 17_408, 192, 128, s);
    {
        const k = try kv.fixtureInput(std.testing.allocator, heads, initial, 192, 7);
        defer drop(k);
        const v = try kv.fixtureInput(std.testing.allocator, heads, initial, 128, 13);
        defer drop(v);
        var view = if (full) try cache.updateFullPacked(k, v, 0, initial, s) else try cache.updateSlidingPacked(k, v, 0, initial, 128, s);
        defer view.deinit();
        try settle(&cache, s);
    }
    const k = try kv.fixtureInput(std.testing.allocator, heads, chunk, 192, 17);
    defer drop(k);
    const v = try kv.fixtureInput(std.testing.allocator, heads, chunk, 128, 23);
    defer drop(v);
    try mlx.check(mlx.mlx_array_eval(k));
    try mlx.check(mlx.mlx_array_eval(v));
    const samples = 16;
    var encode_ns: [samples]u64 = undefined;
    var complete_ns: [samples]u64 = undefined;
    var pos: u64 = initial;
    for (0..samples + 4) |i| {
        const start = stamp();
        var view = if (full) try cache.updateFullPacked(k, v, pos, chunk, s) else try cache.updateSlidingPacked(k, v, pos, chunk, 128, s);
        defer view.deinit();
        const encoded = stamp();
        // Full updates are lazy; materialize all six stored results once.
        // Sliding updates have already evaluated their compaction internally.
        try settle(&cache, s);
        const finished = stamp();
        if (i >= 4) {
            encode_ns[i - 4] = encoded - start;
            complete_ns[i - 4] = finished - start;
        }
        pos += chunk;
    }
    std.sort.heap(u64, &encode_ns, {}, std.sort.asc(u64));
    std.sort.heap(u64, &complete_ns, {}, std.sort.asc(u64));
    const encode_ms = @as(f64, @floatFromInt(encode_ns[7] + encode_ns[8])) / 2e6;
    const complete_ms = @as(f64, @floatFromInt(complete_ns[7] + complete_ns[8])) / 2e6;
    std.debug.print("MIMO_Q8_UPDATE_PROFILE mode={s} chunk={d} samples={d} encode_median_ms={d:.6} complete_median_ms={d:.6} complete_min_ns={d} complete_max_ns={d}\n", .{ if (full) "full" else "sliding", chunk, samples, encode_ms, complete_ms, complete_ns[0], complete_ns[15] });
    try std.testing.expectEqual(if (full) @as(u32, @intCast(pos)) else @as(u32, 128), cache.rows);
}

test "mimo GPU KV8: update profiling full decode sliding decode and prefill" {
    const enabled = std.c.getenv("MIMO_Q8_UPDATE_PROFILE") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const locked = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(locked), "1")) return error.GpuLockRequired;
    // Serial scenarios: twelve full BF16 payloads and eight triple banks cover
    // incoming inputs, quantization, old/new storage and evaluated views. The
    // same 256 MiB cache/512 MiB compiler/128 MiB host allowances as the gate.
    const arrays: u64 = 12 * (2 * 4 * 17_408 * (192 + 128)) + 8 * (4 * 17_408 * 340) + (8 << 20);
    const bound: u64 = arrays + (256 << 20) + (512 << 20) + (128 << 20);
    const ceiling: usize = 3 << 30;
    if (bound > ceiling) return error.MimoKVProfileAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, 256 << 20));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    try scenario(true, 1, s);
    try scenario(false, 1, s);
    try scenario(false, 512, s);
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = @import("sdk").memory.processMemory();
    std.debug.print("MIMO_Q8_UPDATE_PROFILE peak: mlx={d} footprint_interval={d} bound={d} ceiling={d}\n", .{ peak, process.footprint_interval_peak, bound, ceiling });
    if (peak > bound or process.footprint_interval_peak > ceiling) return error.MimoKVProfilePeak;
}
