//! Reference fixtures for DeepSeek-V4.1; no plugin output is used as an oracle. The Engram hashes are the official
//! `NgramHashState.forward` run on the CPU (hermetic, every suite). The stored FP8 / FP4 values are independent Torch
//! casts and a transcription of the official FP4 table (opt-in: an MLX-array test, so a GPU-lock-held run under
//! `DSV41_PHASE0B_MLX` on the CPU stream or `_GPU_WINDOW_LOCKED` on the GPU stream, as src/arch/deepseek_v41/ops.zig's).
//! `scripts/dump_dsv41_reference_fixtures.py` regenerates both files; README, "Independent reference fixtures".
const std = @import("std");
const engram = @import("engram.zig");
const ops = @import("ops.zig");
const graph = @import("graph.zig");
const mlx = @import("sdk").mlx;
const testing = std.testing;

const HashFixture = struct {
    pad_id: u32,
    compressed_vocab_size: u32,
    vocab_size: usize,
    token_map: []const struct { id: u32, compressed: u32 },
    multipliers: [2][4]i64,
    primes: [2][24]i64,
    offsets: [2][24]i64,
    cases: []const struct {
        input_ids: []const u32,
        token_mask: []const bool,
        expected_rows: []const [2][24]i64,
    },
};

test "dsv41 reference: Engram hashes match the official forward across every split and rollback" {
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(HashFixture, a, @embedFile("../../fixtures/dsv41_official_engram.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var h: engram.Hashing = .{ .max_ngram = 4, .n_heads = 8, .n_layers = 2, .pad_id = f.pad_id, .compressed_vocab = f.compressed_vocab_size };
    for (0..2) |l| {
        @memcpy(h.multipliers[l][0..4], &f.multipliers[l]);
        @memcpy(h.flat_offsets[l][0..24], &f.offsets[l]);
        for (0..3) |k| @memcpy(h.primes[l][k][0..8], f.primes[l][k * 8 ..][0..8]);
    }
    const map = try a.alloc(u32, f.vocab_size);
    defer a.free(map);
    @memset(map, std.math.maxInt(u32));
    for (f.token_map) |entry| map[entry.id] = entry.compressed;
    try testing.expect(map[f.pad_id] < f.compressed_vocab_size);
    for (f.cases) |c| {
        try testing.expectEqual(c.input_ids.len, c.token_mask.len);
        try testing.expectEqual(c.input_ids.len, c.expected_rows.len);
        const masked = try a.alloc(bool, c.token_mask.len);
        defer a.free(masked);
        for (c.token_mask, masked) |live, *dead| dead.* = !live;
        for (c.input_ids) |id| try testing.expect(map[id] < f.compressed_vocab_size);
        const out = try a.alloc(i64, c.input_ids.len * 48);
        defer a.free(out);
        const expected = std.mem.bytesAsSlice(i64, std.mem.sliceAsBytes(c.expected_rows));
        for (0..c.input_ids.len + 1) |split| {
            var state: engram.HashState = .{};
            defer state.deinit(a);
            try state.advance(a, &h, map, c.input_ids[0..split], masked[0..split], out[0 .. split * 48]);
            try state.advance(a, &h, map, c.input_ids[split..], masked[split..], out[split * 48 ..]);
            try testing.expectEqualSlices(i64, expected, out);
            state.trim(c.input_ids.len - split);
            // A rejected draft may contain different tokens and masked positions.
            const draft = [_]u32{ 42, 0, 1 };
            var scratch: [3 * 48]i64 = undefined;
            try state.advance(a, &h, map, &draft, &.{ false, true, false }, &scratch);
            state.trim(draft.len);
            try state.advance(a, &h, map, c.input_ids[split..], masked[split..], out[split * 48 ..]);
            try testing.expectEqualSlices(i64, expected, out);
        }
        var state: engram.HashState = .{};
        defer state.deinit(a);
        for (c.input_ids, 0..) |_, i| try state.advance(a, &h, map, c.input_ids[i..][0..1], masked[i..][0..1], out[i * 48 ..][0..48]);
        try testing.expectEqualSlices(i64, expected, out);
    }
}

const QuantFixture = struct {
    cases: []const struct {
        format: []const u8,
        n: usize,
        k: usize,
        weight: []const u8,
        scales: []const u8,
        dequant: []const u16,
        engram_shape: ?[3]c_int = null,
    },
};

test "dsv41 reference: stored FP8 Engram rows and FP4 decoding match independent casts (MLX-array test)" {
    // The repo's contract for MLX-array tests (ops.zig): GPU under _GPU_WINDOW_LOCKED (the GPU lock's holder sets it),
    // CPU stream under DSV41_PHASE0B_MLX (a GPU-lock-held run: any MLX array creates the Metal device), else skip.
    const s = if (std.c.getenv("_GPU_WINDOW_LOCKED") != null) mlx.mlx_default_gpu_stream_new() else if (std.c.getenv("DSV41_PHASE0B_MLX") != null) mlx.mlx_default_cpu_stream_new() else return error.SkipZigTest;
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(QuantFixture, a, @embedFile("../../fixtures/dsv41_independent_dequant.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    var saw_engram_shape = false;
    for (parsed.value.cases) |c| {
        defer g.reset();
        const fp8 = std.mem.eql(u8, c.format, "fp8");
        try testing.expect(fp8 or std.mem.eql(u8, c.format, "fp4"));
        const words_per_row = c.k / (if (fp8) @as(usize, 4) else 8);
        try testing.expectEqual(c.n * words_per_row * 4, c.weight.len);
        try testing.expectEqual(c.n * c.k, c.dequant.len);
        const words = try a.alloc(u32, c.weight.len / 4);
        defer a.free(words);
        for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, c.weight[i * 4 ..][0..4], .little);
        // Official linear FP8 scales span 32 output rows; Engram scales belong to each row. The linear FP8 and FP4
        // cases are MLX format checks of `dequantize` (production reads them through qmm / gatherQmm); the Engram case
        // is the production path (`Trunk.engramRows`).
        const groups = c.k / 32;
        const block_scales = fp8 and c.engram_shape == null;
        try testing.expectEqual((if (block_scales) (c.n + 31) / 32 else c.n) * groups, c.scales.len);
        const scales = try a.alloc(u8, c.n * groups);
        defer a.free(scales);
        for (0..c.n) |r| @memcpy(scales[r * groups ..][0..groups], c.scales[(if (block_scales) r / 32 else r) * groups ..][0..groups]);
        const w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(c.n), @intCast(words_per_row) }, .uint32);
        const sc = try g.hostArray(scales, &.{ @intCast(c.n), @intCast(groups) }, .uint8);
        const shape = c.engram_shape orelse [3]c_int{ 1, 1, @intCast(c.n) };
        const y = if (fp8) try graph.Trunk(ops.MlxOps).engramRows(&g, w, sc, shape[0], shape[1], shape[2]) else try g.dequantize(w, sc, .mxfp4);
        if (c.engram_shape != null) {
            saw_engram_shape = true;
            try testing.expect(fp8);
            try testing.expectEqual(@as(usize, 256), c.k);
            try testing.expectEqual(@as(c_int, 24), shape[2]);
            try testing.expect(g.shapeOf(y).eql(ops.Shape.of(&.{ shape[0], shape[1], shape[2], @intCast(c.k) })));
        }
        try testing.expectEqual(ops.Dtype.bfloat16, g.dtypeOf(y));
        const got = try a.alloc(f32, c.dequant.len);
        defer a.free(got);
        _ = try g.hostF32(try g.astype(y, .float32), got);
        for (c.dequant, got) |bits, value| {
            const expected: f32 = @bitCast(@as(u32, bits) << 16);
            // Bit for bit, except zeros by value: the fixture follows OCP E2M1 (nibble 0x8 is -0) while the official
            // FP4_TABLE maps it to +0; either sign of zero is the same stored weight.
            if (expected == 0.0 and value == 0.0) continue;
            try testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(value)));
        }
    }
    try testing.expect(saw_engram_shape);
}
