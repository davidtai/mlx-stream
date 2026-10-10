//! Bounded synthetic native MXFP4 event/slot-lifetime gate. No model artifact
//! is opened; the parent owns the GPU lease and exact service restoration.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const event = @import("../../sdk_ext.zig").expert.event;
const bank_mod = @import("bank.zig");
const stream_mod = @import("stream.zig");
const source_mod = @import("source.zig");
const expert_sdk = @import("../../sdk_ext.zig").expert;
const math = @import("math.zig").Math;

const tokens: u32 = 7;
const top_k = source_mod.top_k;
const ceiling: u64 = 4 << 30;
const cache_bytes: u64 = 256 << 20;
const compiler_bytes: u64 = 512 << 20;
const host_bytes: u64 = 128 << 20;

test "mimo event GPU: real-shape native gates, two windows and held outputs" {
    const enabled = std.c.getenv("MIMO_EVENT_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const geometry = try bank_mod.Geometry.init(4096, 2048, 64, 1, 2);
    // Source's host scratch is charged at its production capacity even though
    // this fixture's GPU arrays contain only seven logical tokens.
    const bill = try source_mod.memoryBillForTokens(geometry, &.{ 1, 1 }, tokens);
    const oracle_bytes = @as(u64, geometry.num_experts) * geometry.record_bytes;
    const nr = @as(u64, tokens) * top_k;
    // Independent gather/SiLU/reduction graphs, input copies, comparison casts,
    // and all simultaneously held returned outputs; aliases add no payload.
    const control_bytes = nr * (20 * @as(u64, geometry.hidden_size) + 24 * @as(u64, geometry.intermediate_size) + 64) +
        tokens * (64 * @as(u64, geometry.hidden_size) + top_k * 32) + (4 << 20) +
        4 * (@as(u64, geometry.hidden_size) * geometry.num_experts + geometry.num_experts);
    const bound = bill.total_bytes + oracle_bytes + control_bytes + cache_bytes + compiler_bytes + host_bytes;
    std.debug.print("MIMO_EVENT static: H={d} I={d} experts={d} tokens={d} top_k={d} record={d} file={d} persistent_rows={d} transient_rows={d} persistent={d} transient={d} staging={d} policy={d} reader_metadata={d} manifest={d} workspace={d} event_runtime={d} event_graph={d} source_total={d} oracle={d} controls={d} cache={d} compiler={d} host={d} bound={d} ceiling={d}\n", .{
        geometry.hidden_size,  geometry.intermediate_size, geometry.num_experts,   tokens,                     top_k,
        geometry.record_bytes, geometry.total_bytes,       bill.persistent_rows,   source_mod.transient_rows,  bill.persistent_bytes,
        bill.transient_bytes,  bill.staging_bytes,         bill.policy_bytes,      bill.reader_metadata_bytes, bill.manifest_bytes,
        bill.workspace_bytes,  bill.event_runtime_bytes,   bill.event_graph_bytes, bill.total_bytes,           oracle_bytes,
        control_bytes,         cache_bytes,                compiler_bytes,         host_bytes,                 bound,
        ceiling,
    });
    std.debug.print("MIMO_EVENT lookahead_static: moe_layers={d} spec_slots={d} spec_staging={d} spec_runtime={d} predictor_metadata={d} predictor_workspace={d} predictor_runtime={d}\n", .{ geometry.num_moe_layers, stream_mod.spec_slots, bill.spec_staging_bytes, bill.spec_runtime_bytes, bill.predictor_metadata_bytes, bill.predictor_workspace_bytes, bill.predictor_runtime_bytes });
    if (bound > ceiling) return error.MimoEventAdmission;
    try std.testing.expectEqual(@as(u64, 8 << 20), bill.event_runtime_bytes);
    try std.testing.expectEqual(@as(u64, 16 << 20), bill.event_graph_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeBank(a, tmp.dir, geometry);
    var dir_buf: [512]u8 = undefined;
    var path_buf: [600]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(std.testing.io, &dir_buf)];
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/experts.bin", .{dir}, 0);

    // Every MLX/device operation occurs after both environment guards and the
    // static admission. The parent is the actual lock-owning process.
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
    if (before + bound > ceiling) return error.MimoEventAdmission;
    try mlx.check(mlx.mlx_reset_peak_memory());
    sdk.memory.startFootprintInterval();
    const outcome = runFixture(a, stream, path, geometry);
    // runFixture has released its outputs, synchronized, stopped/joined the
    // reader and dropped both slot banks before the stream may be freed.
    try mlx.check(mlx.mlx_synchronize(stream));
    try mlx.check(mlx.mlx_clear_cache());
    var active: usize = 0;
    var cached: usize = 0;
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    try mlx.check(mlx.mlx_get_cache_memory(&cached));
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = sdk.memory.processMemory();
    std.debug.print("MIMO_EVENT peak: mlx_before={d} mlx_after_drop={d} mlx_peak={d} mlx_cache={d} footprint={d} footprint_interval_peak={d} bound={d} ceiling={d}\n", .{ before, active, peak, cached, process.footprint, process.footprint_interval_peak, bound, ceiling });
    if (peak > ceiling or process.footprint_interval_peak > ceiling) return error.MimoEventPeakExceeded;
    // The process-lifetime event record retains no MLX array payload.
    if (active != before) return error.MimoEventGraphRetained;
    try outcome;
}

fn runFixture(a: std.mem.Allocator, stream: mlx.mlx_stream, path: [:0]const u8, geometry: bank_mod.Geometry) !void {
    var g = try ops.MlxOps.init(a, stream);
    defer {
        _ = mlx.mlx_synchronize(stream);
        g.deinit();
    }
    const layer: bank_mod.Layer = .{ .logical_bytes = geometry.record_bytes, .base_offset = 0, .segments = geometry.segments };
    var resident = try stream_mod.LayerSlotBank.init(&layer, geometry.num_experts, stream);
    defer resident.deinit();
    // Regenerate one record at a time directly into the independent resident
    // oracle. Neither side retains a second whole-bank host payload.
    {
        const record = try a.alloc(u8, @intCast(geometry.record_bytes));
        defer a.free(record);
        for (0..geometry.num_experts) |expert| {
            fillRecord(record, geometry, @intCast(expert));
            for (resident.rowDest(@intCast(expert)), geometry.segments) |destination, seg| {
                const dest: [*]u8 = @ptrFromInt(destination);
                @memcpy(dest[0..seg.length], record[seg.offset..][0..seg.length]);
            }
        }
    }
    const resident_arrays = bank_mod.bankArraysOf(resident.arrays);
    const gates = try predictorGates(a, &g, geometry);
    defer a.free(gates);
    const event_before = event.stats();
    // initFixtureBank transfers ownership and performs the exact native M=1
    // event-versus-waited self-check on cold layer 1 before resetting policy.
    const source = try source_mod.Source.initFixtureBank(a, try fixtureBank(a, path, geometry), &.{ 1, 1 }, gates, stream);
    defer source.deinit();
    try std.testing.expectEqual(gates[2].w.ctx, source.ex.predictorGate(1).?.w.ctx);
    try std.testing.expect(source.ex.predictorGate(2) == null);
    try std.testing.expect(event.stats().gpu_waits > event_before.gpu_waits);
    for (0..geometry.num_experts) |expert| {
        try std.testing.expectEqual(@as(?u32, null), source.stream_source.policy(1).slotOf(@intCast(expert)));
        try std.testing.expectEqual(@as(u32, 0), source.stream_source.policy(1).prefill_freq[expert]);
    }
    try std.testing.expectEqual(@as(u32, 96), source.sdk_stream.transient.rows);
    try std.testing.expectEqual(@as(u8, 2), source.stream_source.wideDepth());

    var input: [tokens * 4096]f32 = undefined;
    var state: u32 = 0x38abd247;
    for (&input) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(next(&state) % 513)) - 256)) / 256;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input), &.{ tokens, 4096 }, .float32), .bfloat16);
    var ids: [tokens * top_k]u32 = undefined;
    var weights: [tokens * top_k]f32 = undefined;
    for (&ids, &weights, 0..) |*id, *weight, i| {
        id.* = @intCast((i * 17) % ids.len);
        weight.* = @as(f32, @floatFromInt(1 + i % top_k)) / 36;
    }
    const route = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ tokens, top_k }, .uint32);
    const routing = try g.hostArray(std.mem.sliceAsBytes(&weights), &.{ tokens, top_k }, .float32);
    const wide_before = source.stream_source.stats();
    const wide_waits = event.stats().gpu_waits;
    const wide = try source.forward(1, x, route, routing);
    defer g.release(wide);
    const wide_after = source.stream_source.stats();
    try std.testing.expectEqual(@as(u64, 2), wide_after.route_calls - wide_before.route_calls);
    try std.testing.expectEqual(@as(u64, 56), wide_after.expert_cache_misses - wide_before.expert_cache_misses);
    try std.testing.expect(wide_after.gates > wide_before.gates);
    try std.testing.expect(event.stats().gpu_waits > wide_waits);
    var wide_snapshot: [tokens * 4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "wide_48_plus_8", x, route, routing, wide, &wide_snapshot);
    try std.testing.expectEqual(@as(?u32, 0), source.stream_source.policy(1).slotOf(0));

    const decode_x = try g.slice(x, &.{ 0, 0 }, &.{ 1, 4096 }, &.{ 1, 1 });
    const decode_weights = try g.slice(routing, &.{ 0, 0 }, &.{ 1, top_k }, &.{ 1, 1 });
    const decode_ids = [_]u32{ 0, 56, 3, 57, 9, 58, 55, 59 };
    const decode_route = try g.hostArray(std.mem.sliceAsBytes(&decode_ids), &.{ 1, top_k }, .uint32);
    try predictorNumerics(&g, source, gates[2], decode_x, input[0..4096]);
    // A logical M1 prompt tail has a next router, but genuine prompt phase
    // must not issue predictions or pre-read. Its authoritative output holds.
    const prompt_before = source.stream_source.stats();
    const tail = try source.forward(1, decode_x, decode_route, decode_weights);
    defer g.release(tail);
    var tail_host: [4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "prefill_m1_no_prediction", decode_x, decode_route, decode_weights, tail, &tail_host);
    const prompt_after = source.stream_source.stats();
    try std.testing.expectEqual(prompt_before.spec_issued, prompt_after.spec_issued);
    try std.testing.expectEqual(prompt_before.pre_issued, prompt_after.pre_issued);
    try source.beginDecode();
    const landed_before = source.sdk_stream.pool.counter(.landed);
    const decode_before = source.stream_source.stats();
    const decode = try source.forward(1, decode_x, decode_route, decode_weights);
    defer g.release(decode);
    const decode_after = source.stream_source.stats();
    try std.testing.expect(decode_after.expert_cache_hits > decode_before.expert_cache_hits);
    try std.testing.expect(decode_after.expert_cache_misses > decode_before.expert_cache_misses);
    try std.testing.expectEqual(decode_before.spec_issued + 2, decode_after.spec_issued);
    var decode_snapshot: [4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "decode_mixed", decode_x, decode_route, decode_weights, decode, &decode_snapshot);

    // Outside the production path: wait for both useful records to land so
    // adoption is deterministic rather than dependent on SSD/GPU timing.
    try waitLanded(source, landed_before + 2);
    const layer2_ids = [_]u32{ 56, 3, 57, 9, 58, 55, 59, 0 };
    const layer2_route = try g.hostArray(std.mem.sliceAsBytes(&layer2_ids), &.{ 1, top_k }, .uint32);
    const adoption_before = source.stream_source.stats();
    const adopted = try source.forward(2, decode_x, layer2_route, decode_weights);
    defer g.release(adopted);
    var adopted_host: [4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "decode_adopted_six_components", decode_x, layer2_route, decode_weights, adopted, &adopted_host);
    const adoption_after = source.stream_source.stats();
    try std.testing.expectEqual(adoption_before.claimed + 2, adoption_after.claimed);
    try std.testing.expectEqual(adoption_before.adopt_ranges + 4, adoption_after.adopt_ranges);
    try std.testing.expectEqual(adoption_before.adopt_bytes + 2 * geometry.record_bytes, adoption_after.adopt_bytes);
    try std.testing.expect(adoption_after.pre_issued > adoption_before.pre_issued);

    const changed_ids = [_]u32{ 0, 60, 5, 61, 17, 62, 53, 63 };
    const changed_route = try g.hostArray(std.mem.sliceAsBytes(&changed_ids), &.{ 1, top_k }, .uint32);
    const changed = try source.forward(1, decode_x, changed_route, decode_weights);
    defer g.release(changed);
    var changed_host: [4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "decode_changed", decode_x, changed_route, decode_weights, changed, &changed_host);
    // Predictions favour 56/57 (or cold tie 0 when one is resident). These
    // authoritative ids deliberately exclude every possible predicted id.
    const wrong_ids = [_]u32{ 40, 41, 42, 43, 44, 45, 46, 47 };
    const wrong_route = try g.hostArray(std.mem.sliceAsBytes(&wrong_ids), &.{ 1, top_k }, .uint32);
    const wrong_before = source.stream_source.stats();
    const wrong = try source.forward(2, decode_x, wrong_route, decode_weights);
    defer g.release(wrong);
    var wrong_host: [4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "decode_wrong_prediction", decode_x, wrong_route, decode_weights, wrong, &wrong_host);
    try std.testing.expectEqual(wrong_before.claimed, source.stream_source.stats().claimed);
    try expectHeld(&g, wide, &wide_snapshot);
    try expectHeld(&g, decode, &decode_snapshot);

    const retained_slot = source.stream_source.policy(1).slotOf(0);
    const retained_count = source.stream_source.policy(1).prefill_freq[0];
    // Leave a live speculative horizon at a mid-request reverse boundary.
    const pending = try source.forward(1, decode_x, decode_route, decode_weights);
    defer g.release(pending);
    // A one-token output is lazy (L1): evaluate it before the phase boundary.
    try g.evalAll(&.{pending});
    const clock_before = source.sdk_stream.clock;
    const gate_before = source.sdk_stream.gate_value;
    try source.resumePrefill();
    try std.testing.expect(!source.sdk_stream.route_lookahead and !source.sdk_stream.route_preread);
    try std.testing.expectEqual(clock_before, source.sdk_stream.clock);
    try std.testing.expectEqual(gate_before, source.sdk_stream.gate_value);
    try std.testing.expectEqual(retained_slot, source.stream_source.policy(1).slotOf(0));
    try std.testing.expectEqual(retained_count, source.stream_source.policy(1).prefill_freq[0]);
    try source.beginDecode();
    try std.testing.expect(source.sdk_stream.route_lookahead and source.sdk_stream.route_preread);
    const redecoded = try source.forward(1, decode_x, changed_route, decode_weights);
    defer g.release(redecoded);
    var redecoded_host: [4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "decode_after_resume", decode_x, changed_route, decode_weights, redecoded, &redecoded_host);
    try std.testing.expect(source.sdk_stream.clock > clock_before);
    try std.testing.expect(source.sdk_stream.gate_value >= gate_before);
    try source.beginPrefill();
    try std.testing.expectEqual(@as(u8, 2), source.stream_source.wideDepth());
    // A fresh 56-expert prompt replaces protected seed 0 with expert 8 and
    // rewrites both windows while every preceding output remains held.
    for (&ids, 0..) |*id, i| id.* = @intCast(8 + (i * 17) % ids.len);
    const next_route = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ tokens, top_k }, .uint32);
    const next_output = try source.forward(1, x, next_route, routing);
    defer g.release(next_output);
    var next_host: [tokens * 4096]f32 = undefined;
    try compare(&g, geometry, resident_arrays, "new_request", x, next_route, routing, next_output, &next_host);
    try std.testing.expectEqual(@as(?u32, null), source.stream_source.policy(1).slotOf(0));
    try std.testing.expectEqual(@as(?u32, 0), source.stream_source.policy(1).slotOf(8));
    try std.testing.expect(source.stream_source.policy(1).protected.isSet(8));
    try expectHeld(&g, wide, &wide_snapshot);
    try expectHeld(&g, decode, &decode_snapshot);
    try expectHeld(&g, tail, &tail_host);
    try expectHeld(&g, adopted, &adopted_host);
    const stats = source.stream_source.stats();
    const event_after = event.stats();
    std.debug.print("MIMO_EVENT stats: routes={d} hits={d} misses={d} read_bytes={d} skipped={d} gates={d} forced={d} gpu_waits={d} host_timeouts={d}\n", .{ stats.route_calls, stats.expert_cache_hits, stats.expert_cache_misses, stats.expert_bytes_read, stats.loads_skipped, stats.gates, stats.gates_forced, event_after.gpu_waits - event_before.gpu_waits, event_after.host_timeouts - event_before.host_timeouts });
    std.debug.print("MIMO_EVENT lookahead_stats: issued={d} landed={d} claimed={d} adopt_ranges={d} adopt_bytes={d} spec_bytes={d} preread_issued={d} preread_served={d} preread_expired={d} clock={d} gate_value={d}\n", .{ stats.spec_issued, stats.spec_landed, stats.claimed, stats.adopt_ranges, stats.adopt_bytes, stats.spec_bytes, stats.pre_issued, stats.pre_served, stats.pre_expired, source.sdk_stream.clock, source.sdk_stream.gate_value });
    try std.testing.expectEqual(@as(u64, 0), stats.gates_forced);
    try std.testing.expectEqual(event_before.host_timeouts, event_after.host_timeouts);
}

/// Independent resident gathered MXFP4 arithmetic: CPU F32 SiLU, BF16 round,
/// BF16 multiply by up, BF16 down, then F32 weights/reduction and BF16 round.
fn reference(g: *ops.MlxOps, geometry: bank_mod.Geometry, resident: bank_mod.BankArrays, x: mlx.mlx_array, ids: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    const nt = g.shapeOf(x).d[0];
    const nr: c_int = nt * @as(c_int, top_k);
    const H: c_int = @intCast(geometry.hidden_size);
    const I: c_int = @intCast(geometry.intermediate_size);
    var rows: [tokens * top_k]u32 = undefined;
    for (rows[0..@intCast(nr)], 0..) |*row, i| row.* = @intCast(i / top_k);
    const rhs = try g.reshape(ids, &.{nr});
    const src = try g.hostArray(std.mem.sliceAsBytes(rows[0..@intCast(nr)]), &.{nr}, .uint32);
    const repeated = try g.reshape(try g.take(x, src, 0), &.{ nr, 1, H });
    const gate = try g.gatherQmm(repeated, resident.gate.w, resident.gate.scales.?, rhs, .mxfp4);
    const up = try g.gatherQmm(repeated, resident.up.w, resident.up.scales.?, rhs, .mxfp4);
    const gate_host = try g.gpa.alloc(f32, @intCast(nr * I));
    defer g.gpa.free(gate_host);
    _ = try g.hostF32(try g.astype(gate, .float32), gate_host);
    for (gate_host) |*v| v.* = v.* / (1 + @exp(-v.*));
    const silu = try g.astype(try g.hostArray(std.mem.sliceAsBytes(gate_host), &.{ nr, 1, I }, .float32), .bfloat16);
    const down = try g.gatherQmm(try g.mul(silu, up), resident.down.w, resident.down.scales.?, rhs, .mxfp4);
    const weighted = try g.mul(try g.reshape(try g.astype(down, .float32), &.{ nt, top_k, H }), try g.reshape(weights, &.{ nt, top_k, 1 }));
    return g.astype(try g.sum(weighted, 1, false), .bfloat16);
}

fn compare(g: *ops.MlxOps, geometry: bank_mod.Geometry, resident: bank_mod.BankArrays, name: []const u8, x: mlx.mlx_array, ids: mlx.mlx_array, weights: mlx.mlx_array, got: mlx.mlx_array, snapshot: []f32) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    const want = try reference(g, geometry, resident, x, ids, weights);
    const expected = try g.gpa.alloc(f32, snapshot.len);
    defer g.gpa.free(expected);
    _ = try g.hostF32(try g.astype(want, .float32), expected);
    _ = try g.hostF32(try g.astype(got, .float32), snapshot);
    var diff_squared: f64 = 0;
    var reference_squared: f64 = 0;
    var max_abs: f64 = 0;
    for (expected, snapshot) |left, right| {
        if (!std.math.isFinite(left) or !std.math.isFinite(right)) return error.MimoEventNonFinite;
        const diff = @as(f64, left) - right;
        diff_squared += diff * diff;
        reference_squared += @as(f64, left) * left;
        max_abs = @max(max_abs, @abs(diff));
    }
    if (reference_squared == 0) return error.MimoEventDegenerateProbe;
    const relative_l2 = @sqrt(diff_squared / reference_squared);
    std.debug.print("MIMO_EVENT numerics: stage={s} values={d} max_abs={d:.9} relative_l2={d:.9}\n", .{ name, snapshot.len, max_abs, relative_l2 });
    if (relative_l2 > 0.01) return error.MimoEventAgreement;
}

fn expectHeld(g: *ops.MlxOps, output: mlx.mlx_array, snapshot: []const f32) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    const actual = try g.gpa.alloc(f32, snapshot.len);
    defer g.gpa.free(actual);
    _ = try g.hostF32(try g.astype(output, .float32), actual);
    try std.testing.expectEqualSlices(f32, snapshot, actual);
}

fn next(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

fn fillRecord(record: []u8, geometry: bank_mod.Geometry, expert: u32) void {
    for (geometry.segments, 0..) |seg, component| {
        var state = @as(u32, 0x7a02d195) ^ ((expert + 1) *% 0x9e3779b9) ^ (@as(u32, @intCast(component + 1)) *% 0xc58143af);
        if (state == 0) state = 1;
        const bytes = record[seg.offset..][0..seg.length];
        if (seg.dtype == .U32) {
            var offset: usize = 0;
            while (offset < bytes.len) : (offset += 4) std.mem.writeInt(u32, bytes[offset..][0..4], next(&state), .little);
        } else for (bytes) |*scale| scale.* = @intCast(117 + next(&state) % 5);
    }
}

fn writeBank(a: std.mem.Allocator, dir: std.Io.Dir, geometry: bank_mod.Geometry) !void {
    const file = try dir.createFile(std.testing.io, "experts.bin", .{ .read = true });
    defer file.close(std.testing.io);
    // The synthetic writer also bypasses the file cache; the disk bank is not
    // an uncharged 855-MB host payload alongside the admitted MLX banks.
    try sdk.io_util.noCache(file.handle, .{ .read_ahead = true });
    const record = try a.alloc(u8, @intCast(geometry.record_bytes));
    defer a.free(record);
    for (0..geometry.num_moe_layers) |layer| for (0..geometry.num_experts) |expert_id| {
        fillRecord(record, geometry, @intCast(expert_id));
        try file.writePositionalAll(std.testing.io, record, (layer * geometry.num_experts + expert_id) * geometry.record_bytes);
    };
    try std.testing.expectEqual(geometry.total_bytes, (try file.stat(std.testing.io)).size);
}

fn fixtureBank(a: std.mem.Allocator, path: [:0]const u8, geometry: bank_mod.Geometry) !bank_mod.Bank {
    const digests = try a.alloc([32]u8, geometry.num_experts * geometry.num_moe_layers);
    errdefer a.free(digests);
    @memset(digests, @splat(0));
    const repository = try a.dupe(u8, "mimo-native-event-fixture");
    errdefer a.free(repository);
    const revision = try a.dupe(u8, "0000000000000000000000000000000000000000");
    errdefer a.free(revision);
    const fd = try sdk.io_util.openNoCache(path, .{ .follow_symlinks = false });
    return .{ .allocator = a, .geometry = geometry, .digests = digests, .source_repository = repository, .source_revision = revision, .sidecar_digest = @splat(0), .sidecar_fd = fd, .sidecar_file_size = geometry.total_bytes };
}

fn predictorGates(a: std.mem.Allocator, g: *ops.MlxOps, geometry: bank_mod.Geometry) ![]source_mod.Gate {
    const gates = try a.alloc(source_mod.Gate, geometry.first_moe_layer + geometry.num_moe_layers);
    errdefer a.free(gates);
    @memset(gates, .{ .w = .{}, .bias = .{} });
    const values = try a.alloc(f32, geometry.hidden_size * geometry.num_experts);
    defer a.free(values);
    @memset(values, 0);
    values[56] = 0.5;
    values[geometry.num_experts + 56] = -0.25;
    values[57] = -0.125;
    values[geometry.num_experts + 57] = 0.25;
    const w = try g.hostArray(std.mem.sliceAsBytes(values), &.{ @intCast(geometry.hidden_size), @intCast(geometry.num_experts) }, .float32);
    var corrections: [64]f32 = @splat(0);
    corrections[56] = 3;
    corrections[57] = 2;
    const correction = try g.hostArray(std.mem.sliceAsBytes(&corrections), &.{64}, .float32);
    for (gates[geometry.first_moe_layer..]) |*gate| gate.* = .{ .w = w, .bias = correction };
    return gates;
}

fn predictorNumerics(g: *ops.MlxOps, source: *source_mod.Source, gate: source_mod.Gate, x: mlx.mlx_array, input: []const f32) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    const scores = try math.predictorScores(g, x, gate.w, gate.bias);
    var actual: [64]f32 = undefined;
    _ = try g.hostF32(scores, &actual);
    var max_abs: f32 = 0;
    for (actual, 0..) |value, i| {
        const z: f32 = if (i == 56) input[0] * 0.5 - input[1] * 0.25 else if (i == 57) -input[0] * 0.125 + input[1] * 0.25 else 0;
        const correction: f32 = if (i == 56) 3 else if (i == 57) 2 else 0;
        const expected: f32 = 1 / (1 + @exp(-z)) + correction;
        max_abs = @max(max_abs, @abs(expected - value));
        try std.testing.expectApproxEqAbs(expected, value, 0.000001);
    }
    var selector = try expert_sdk.lookahead.SelectorOf(8).init(g.gpa, 64, 8, std.math.inf(f32), 2);
    defer selector.deinit(g.gpa);
    var ids: [2]u16 = undefined;
    try std.testing.expectEqualSlices(u16, &.{ 56, 57 }, selector.select(&actual, source.stream_source.policy(2), &ids));
    std.debug.print("MIMO_EVENT predictor_numerics: values=64 max_abs={d:.9} selected=56,57\n", .{max_abs});
}

fn waitLanded(source: *source_mod.Source, target: i64) !void {
    var polls: u32 = 0;
    while (source.sdk_stream.pool.counter(.landed) < target) : (polls += 1) {
        if (polls >= 10_000) return error.MimoPredictorFixtureTimeout;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
}
