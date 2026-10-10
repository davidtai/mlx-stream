//! Real-shape M1 fence screen. The parent owns the lease and paired binaries.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const event = @import("../../sdk_ext.zig").expert.event;
const bank_mod = @import("bank.zig");
const stream_mod = @import("stream.zig");
const source_mod = @import("source.zig");
const H: u32 = 4096;
const I: u32 = 2048;
const E: u32 = 16;
const K: u32 = 8;
const ceiling: u64 = 4 << 30;
const cache_bytes: u64 = 256 << 20;
const compiler_bytes: u64 = 512 << 20;
const host_bytes: u64 = 128 << 20;

const Bound = struct {
    geometry: bank_mod.Geometry,
    source: source_mod.MemoryBill,
    oracle: u64,
    controls: u64,
    gates: u64,
    scratch: u64,
    total: u64,
    fn init() !Bound {
        const geometry = try bank_mod.Geometry.init(H, I, E, 1, 2);
        const source = try source_mod.memoryBillForTokens(geometry, &.{ 8, 8 }, 1);
        const oracle: u64 = E * geometry.record_bytes;
        // Independent gather/cast/SiLU/product/reduction arrays, inputs and
        // four simultaneously held outputs, with 2 MiB graph/handle slack.
        const controls: u64 = K * (20 * H + 24 * I + 64) + 24 * H + (2 << 20);
        const gates: u64 = 4 * (H * E + E);
        const scratch = geometry.record_bytes;
        return .{ .geometry = geometry, .source = source, .oracle = oracle, .controls = controls, .gates = gates, .scratch = scratch, .total = source.total_bytes + oracle + controls + gates + scratch + cache_bytes + compiler_bytes + host_bytes };
    }
    fn report(b: Bound) void {
        std.debug.print("MIMO_SYNC_STATIC source_abi={d} memory_bill_abi={d} H={d} I={d} experts={d} layers=2 resident_rows=16 transient_rows={d} record={d} file={d} source={d} oracle={d} controls={d} gates={d} scratch={d} cache={d} compiler={d} host={d} bound={d} ceiling={d}\n", .{
            @sizeOf(source_mod.Source), @sizeOf(source_mod.MemoryBill), H,          I,       E,         source_mod.transient_rows, b.geometry.record_bytes, b.geometry.total_bytes,
            b.source.total_bytes,       b.oracle,                       b.controls, b.gates, b.scratch, cache_bytes,               compiler_bytes,          host_bytes,
            b.total,                    ceiling,
        });
    }
};

test "mimo sync CPU: real-shape M1 fixture fits the declared envelope" {
    const bound = try Bound.init();
    bound.report();
    try std.testing.expect(bound.total < ceiling);
}

test "mimo sync GPU: native M1 all-hit and mixed slot reuse" {
    const enabled = std.c.getenv("MIMO_SYNC_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const locked = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(locked), "1")) return error.GpuLockRequired;
    const bound = try Bound.init();
    bound.report();
    if (bound.total >= ceiling) return error.MimoSyncAdmission;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeBank(a, tmp.dir, bound.geometry);
    var dir_buf: [512]u8 = undefined;
    var path_buf: [600]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(std.testing.io, &dir_buf)];
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/experts.bin", .{dir}, 0);
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
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
    try mlx.check(mlx.mlx_synchronize(stream));
    try mlx.check(mlx.mlx_clear_cache());
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    if (before + bound.total >= ceiling) return error.MimoSyncAdmission;
    try mlx.check(mlx.mlx_reset_peak_memory());
    sdk.memory.startFootprintInterval();
    const outcome = run(a, stream, path, bound.geometry);
    try mlx.check(mlx.mlx_synchronize(stream));
    try mlx.check(mlx.mlx_clear_cache());
    var active: usize = 0;
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = sdk.memory.processMemory();
    std.debug.print("MIMO_SYNC_PEAK before={d} after_drop={d} mlx_peak={d} footprint_peak={d} bound={d} ceiling={d}\n", .{ before, active, peak, process.footprint_interval_peak, bound.total, ceiling });
    if (active != before) return error.MimoSyncRetained;
    if (peak > ceiling or process.footprint_interval_peak > ceiling) return error.MimoSyncPeakExceeded;
    try outcome;
}

fn run(a: std.mem.Allocator, stream: mlx.mlx_stream, path: [:0]const u8, geometry: bank_mod.Geometry) !void {
    var g = try ops.MlxOps.init(a, stream);
    defer {
        _ = mlx.mlx_synchronize(stream);
        g.deinit();
    }
    const layer: bank_mod.Layer = .{ .logical_bytes = geometry.record_bytes, .base_offset = 0, .segments = geometry.segments };
    var resident = try stream_mod.LayerSlotBank.init(&layer, E, stream);
    defer resident.deinit();
    {
        const record = try a.alloc(u8, @intCast(geometry.record_bytes));
        defer a.free(record);
        for (0..E) |id| {
            fill(record, geometry, @intCast(id));
            for (resident.rowDest(@intCast(id)), geometry.segments) |address, segment| {
                const dest: [*]u8 = @ptrFromInt(address);
                @memcpy(dest[0..segment.length], record[segment.offset..][0..segment.length]);
            }
        }
    }
    const oracle = bank_mod.bankArraysOf(resident.arrays);
    const gates = try source_mod.fixtureGates(a, &g, geometry);
    defer a.free(gates);
    const event_before = event.stats();
    const source = try source_mod.Source.initFixtureBank(a, try bank(a, path, geometry), &.{ 8, 8 }, gates, stream);
    defer source.deinit();
    // The last routed layer cannot invoke a next-layer predictor.
    try std.testing.expect(source.ex.predictorGate(2) == null);
    try source.beginDecode();
    var values: [H]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&values), &.{ 1, H }, .float32), .bfloat16);
    const ids_a = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const ids_b = [_]u32{ 4, 5, 6, 7, 8, 9, 10, 11 };
    const ids_c = [_]u32{ 8, 9, 10, 11, 12, 13, 14, 15 };
    const ids_d = [_]u32{ 12, 13, 14, 15, 0, 1, 2, 3 };
    const route_a = try g.hostArray(std.mem.sliceAsBytes(&ids_a), &.{ 1, K }, .uint32);
    const route_b = try g.hostArray(std.mem.sliceAsBytes(&ids_b), &.{ 1, K }, .uint32);
    const route_c = try g.hostArray(std.mem.sliceAsBytes(&ids_c), &.{ 1, K }, .uint32);
    const route_d = try g.hostArray(std.mem.sliceAsBytes(&ids_d), &.{ 1, K }, .uint32);
    const route_weights = [_]f32{ 1.0 / 36.0, 2.0 / 36.0, 3.0 / 36.0, 4.0 / 36.0, 5.0 / 36.0, 6.0 / 36.0, 7.0 / 36.0, 8.0 / 36.0 };
    const weights = try g.hostArray(std.mem.sliceAsBytes(&route_weights), &.{ 1, K }, .float32);
    // Prime the actual hit/miss wave shapes and the independent oracle once.
    for ([_]mlx.mlx_array{ route_a, route_b, route_c, route_d }) |route| {
        const output = try source.forward(2, x, route, weights);
        defer g.release(output);
        var snapshot: [H]f32 = undefined;
        try check(&g, oracle, x, route, weights, output, &snapshot);
    }
    try source.beginPrefill();
    try source.beginDecode();
    const held = try source.forward(2, x, route_a, weights);
    defer g.release(held);
    var held_snapshot: [H]f32 = undefined;
    try check(&g, oracle, x, route_a, weights, held, &held_snapshot);
    const hits_before = source.stream_source.stats();
    const hit_start = std.Io.Timestamp.now(std.testing.io, .boot);
    // Each call is evaluated before the next: these ids do not derive from
    // the prior one-token output, so no routing barrier evaluates it (L1).
    for (0..64) |_| {
        const output = try source.forward(2, x, route_a, weights);
        try g.evalAll(&.{output});
        g.release(output);
    }
    const hit_ns = hit_start.untilNow(std.testing.io, .boot).nanoseconds;
    const hits_after = source.stream_source.stats();
    try std.testing.expectEqual(hits_before.expert_bytes_read, hits_after.expert_bytes_read);
    try std.testing.expectEqual(hits_before.expert_cache_misses, hits_after.expert_cache_misses);
    try std.testing.expectEqual(hits_before.expert_cache_hits + 64 * K, hits_after.expert_cache_hits);
    try heldExact(&g, held, &held_snapshot);
    std.debug.print("MIMO_SYNC_TIMING case=all_hit calls=64 elapsed_ns={d} mean_us={d:.6} bytes=0 misses=0 hits={d}\n", .{ hit_ns, @as(f64, @floatFromInt(hit_ns)) / 64_000, 64 * K });

    try source.beginPrefill();
    try source.beginDecode();
    const mixed_held = try source.forward(2, x, route_a, weights);
    defer g.release(mixed_held);
    var mixed_snapshot: [H]f32 = undefined;
    try check(&g, oracle, x, route_a, weights, mixed_held, &mixed_snapshot);
    const mixed_before = source.stream_source.stats();
    const waits_before = event.stats().gpu_waits;
    const mixed_start = std.Io.Timestamp.now(std.testing.io, .boot);
    const routes = [_]mlx.mlx_array{ route_b, route_c, route_d, route_a };
    for (0..32) |i| {
        const output = try source.forward(2, x, routes[i % routes.len], weights);
        try g.evalAll(&.{output});
        g.release(output);
    }
    const mixed_ns = mixed_start.untilNow(std.testing.io, .boot).nanoseconds;
    const mixed_after = source.stream_source.stats();
    const mixed_waits = event.stats().gpu_waits - waits_before;
    try std.testing.expect(mixed_after.expert_cache_hits > mixed_before.expert_cache_hits);
    try std.testing.expect(mixed_after.expert_cache_misses > mixed_before.expert_cache_misses);
    try std.testing.expect(mixed_after.expert_bytes_read > mixed_before.expert_bytes_read);
    try std.testing.expect(mixed_after.gates > mixed_before.gates);
    try std.testing.expect(mixed_waits > 0);
    const last = try source.forward(2, x, route_a, weights);
    defer g.release(last);
    var last_snapshot: [H]f32 = undefined;
    try check(&g, oracle, x, route_a, weights, last, &last_snapshot);
    try heldExact(&g, held, &held_snapshot);
    try heldExact(&g, mixed_held, &mixed_snapshot);
    std.debug.print("MIMO_SYNC_TIMING case=mixed calls=32 elapsed_ns={d} mean_us={d:.6} bytes={d} misses={d} hits={d} skipped={d} gates={d} gpu_waits={d}\n", .{
        mixed_ns,                                                           @as(f64, @floatFromInt(mixed_ns)) / 32_000,                     mixed_after.expert_bytes_read - mixed_before.expert_bytes_read,
        mixed_after.expert_cache_misses - mixed_before.expert_cache_misses, mixed_after.expert_cache_hits - mixed_before.expert_cache_hits, mixed_after.loads_skipped - mixed_before.loads_skipped,
        mixed_after.gates - mixed_before.gates,                             mixed_waits,
    });
    try source.beginPrefill();
    try heldExact(&g, held, &held_snapshot);
    try heldExact(&g, mixed_held, &mixed_snapshot);
    const stats = source.stream_source.stats();
    const events_after = event.stats();
    try std.testing.expectEqual(@as(u64, 0), stats.gates_forced);
    try std.testing.expectEqual(@as(u64, 0), stats.spec_issued);
    try std.testing.expectEqual(event_before.host_timeouts, events_after.host_timeouts);
    std.debug.print("MIMO_SYNC_CHECK exact=true held_exact=true forced={d} host_timeouts={d} spec_issued={d}\n", .{ stats.gates_forced, events_after.host_timeouts - event_before.host_timeouts, stats.spec_issued });
}

fn check(g: *ops.MlxOps, resident: bank_mod.BankArrays, x: mlx.mlx_array, ids: mlx.mlx_array, weights: mlx.mlx_array, output: mlx.mlx_array, snapshot: []f32) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    const rhs = try g.reshape(ids, &.{K});
    const zeros: [K]u32 = @splat(0);
    const rows = try g.hostArray(std.mem.sliceAsBytes(&zeros), &.{K}, .uint32);
    const repeated = try g.reshape(try g.take(x, rows, 0), &.{ K, 1, H });
    const gate = try g.gatherQmm(repeated, resident.gate.w, resident.gate.scales.?, rhs, .mxfp4);
    const up = try g.gatherQmm(repeated, resident.up.w, resident.up.scales.?, rhs, .mxfp4);
    var gate_host: [K * I]f32 = undefined;
    _ = try g.hostF32(try g.astype(gate, .float32), &gate_host);
    for (&gate_host) |*v| v.* = v.* / (1 + @exp(-v.*));
    const silu = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&gate_host), &.{ K, 1, I }, .float32), .bfloat16);
    const down = try g.gatherQmm(try g.mul(silu, up), resident.down.w, resident.down.scales.?, rhs, .mxfp4);
    const joined = try g.reshape(try g.astype(down, .float32), &.{ 1, K, H });
    const wanted = try g.astype(try g.sum(try g.mul(joined, try g.reshape(weights, &.{ 1, K, 1 })), 1, false), .bfloat16);
    var expected: [H]f32 = undefined;
    _ = try g.hostF32(try g.astype(wanted, .float32), &expected);
    _ = try g.hostF32(try g.astype(output, .float32), snapshot);
    var nonzero = false;
    for (expected, snapshot) |reference, actual| {
        if (!std.math.isFinite(reference) or !std.math.isFinite(actual)) return error.MimoSyncNonFinite;
        nonzero = nonzero or reference != 0;
    }
    try std.testing.expect(nonzero);
    try std.testing.expectEqualSlices(f32, &expected, snapshot);
}

fn heldExact(g: *ops.MlxOps, output: mlx.mlx_array, snapshot: []const f32) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    var actual: [H]f32 = undefined;
    _ = try g.hostF32(try g.astype(output, .float32), &actual);
    try std.testing.expectEqualSlices(f32, snapshot, &actual);
}

fn fill(record: []u8, geometry: bank_mod.Geometry, id: u32) void {
    @memset(record, 0);
    for (0..3) |projection| {
        const words = geometry.segments[projection * 2];
        const scales = geometry.segments[projection * 2 + 1];
        const input: u32 = if (projection == 2) I else H;
        const width: u32 = if (projection == 2) H else I;
        const code: u32 = 2 + (id + @as(u32, @intCast(projection))) % 4;
        for (0..width) |row| {
            const column = (row * 17 + id * 13 + projection * 5) % input;
            const offset = words.offset + (row * (input / 8) + column / 8) * 4;
            std.mem.writeInt(u32, record[offset..][0..4], code << @as(u5, @intCast((column % 8) * 4)), .little);
        }
        @memset(record[scales.offset..][0..scales.length], if (projection == 2) 125 else 126);
    }
}

fn writeBank(a: std.mem.Allocator, dir: std.Io.Dir, geometry: bank_mod.Geometry) !void {
    const file = try dir.createFile(std.testing.io, "experts.bin", .{ .read = true });
    defer file.close(std.testing.io);
    try sdk.io_util.noCache(file.handle, .{ .read_ahead = true });
    const record = try a.alloc(u8, @intCast(geometry.record_bytes));
    defer a.free(record);
    for (0..geometry.num_moe_layers) |layer| for (0..E) |id| {
        fill(record, geometry, @intCast(id));
        try file.writePositionalAll(std.testing.io, record, (layer * E + id) * geometry.record_bytes);
    };
}

fn bank(a: std.mem.Allocator, path: [:0]const u8, geometry: bank_mod.Geometry) !bank_mod.Bank {
    const digests = try a.alloc([32]u8, E * geometry.num_moe_layers);
    errdefer a.free(digests);
    @memset(digests, @splat(0));
    const repository = try a.dupe(u8, "mimo-sync-native-fixture");
    errdefer a.free(repository);
    const revision = try a.dupe(u8, "0000000000000000000000000000000000000000");
    errdefer a.free(revision);
    const fd = try sdk.io_util.openNoCache(path, .{ .follow_symlinks = false });
    return .{ .allocator = a, .geometry = geometry, .digests = digests, .source_repository = repository, .source_revision = revision, .sidecar_digest = @splat(0), .sidecar_fd = fd, .sidecar_file_size = geometry.total_bytes };
}
