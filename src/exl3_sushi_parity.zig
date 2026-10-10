//! Independent exllamav3 fixture expectations consumed through this plugin's host decoder.
//! K2/K3/K4 compare inner weights and complete normalized H128 + suh/svh public f16
//! weights; these 128-wide fixtures do not certify production GPU projection paths.
//! Defaults live under lib/sushi/src/exl3/fixtures in the host checkout. An explicit
//! SUSHI_EXL3_K{2,3,4}_FIXTURE must exist; an absent default skips the corresponding test.

const std = @import("std");
const kernels = @import("exl3_kernels.zig");
const nocache_io = @import("sdk").io_util;

const fixture_default = "lib/sushi/src/exl3/fixtures/exl3_k3_linear.safetensors";

const Tensor = struct { dtype: []const u8, shape: [4]u64, rank: usize, data_offsets: [2]u64 };

fn tensorOf(header: std.json.Value, name: []const u8) !Tensor {
    const t = header.object.get(name) orelse return error.FixtureTensorMissing;
    const shape = t.object.get("shape").?.array.items;
    var dims: [4]u64 = @splat(0);
    for (shape, 0..) |d, i| dims[i] = @intCast(d.integer);
    const offs = t.object.get("data_offsets").?.array.items;
    return .{
        .dtype = t.object.get("dtype").?.string,
        .shape = dims,
        .rank = shape.len,
        .data_offsets = .{ @intCast(offs[0].integer), @intCast(offs[1].integer) },
    };
}

test "exl3 sushi parity: sushi's K3 mul1 fixture trellis reconstructs to its inner weights bit for bit through this plugin's decode" {
    const a = std.testing.allocator;
    const env = std.c.getenv("SUSHI_EXL3_K3_FIXTURE");
    const path = if (env) |p| std.mem.span(p) else fixture_default;
    const bytes = nocache_io.readAllNoCache(a, path, 1 << 20) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer a.free(bytes);

    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    defer parsed.deinit();
    const data = bytes[8 + n ..];

    const trellis = try tensorOf(parsed.value, "trellis");
    const inner = try tensorOf(parsed.value, "inner");
    try std.testing.expectEqualStrings("U16", trellis.dtype);
    try std.testing.expectEqualStrings("F16", inner.dtype);
    try std.testing.expect(trellis.rank == 3 and inner.rank == 2);
    const n_i: usize = @intCast(trellis.shape[0]);
    const n_j: usize = @intCast(trellis.shape[1]);
    const K: usize = @intCast(trellis.shape[2] / 16);
    try std.testing.expectEqual(@as(usize, 3), K);
    try std.testing.expectEqual(@as(u64, 16 * n_i), inner.shape[0]);
    try std.testing.expectEqual(@as(u64, 16 * n_j), inner.shape[1]);

    const code_bytes = data[trellis.data_offsets[0]..trellis.data_offsets[1]];
    const code = try a.alloc(i16, code_bytes.len / 2);
    defer a.free(code);
    for (code, 0..) |*c, i| c.* = std.mem.readInt(i16, code_bytes[2 * i ..][0..2], .little);
    const want_bytes = data[inner.data_offsets[0]..inner.data_offsets[1]];

    const table = try a.create([65536]u16);
    defer a.destroy(table);
    kernels.mul1Table(table);
    const got = try a.alloc(u16, 256 * n_i * n_j);
    defer a.free(got);
    kernels.reconstruct(code, n_i, n_j, K, table, got, null);

    var mismatches: usize = 0;
    var distinct = std.AutoHashMap(u16, void).init(a);
    defer distinct.deinit();
    for (got, 0..) |g, i| {
        if (g != std.mem.readInt(u16, want_bytes[2 * i ..][0..2], .little)) mismatches += 1;
        try distinct.put(g, {});
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
    // A real weight, not a constant fill (mul1 decodes to at most 1021 values): the comparison cannot pass vacuously.
    try std.testing.expect(distinct.count() > 100);
}

test "exl3 sushi parity: K2 full H128 scaled public weights" {
    try fullPublicParity(2, .exact_words);
}

test "exl3 sushi parity: K3 full H128 scaled public weights" {
    try fullPublicParity(3, .exact_words);
}

test "exl3 sushi parity: K4 full H128 scaled public weights" {
    try fullPublicParity(4, .exact_words);
}

test "exl3 mixed K5: independent CPU library full H128 scaled public weights" {
    if (std.c.getenv("SUSHI_EXL3_K5_FIXTURE") == null) {
        if (std.c.getenv("DSV41_KERNELS_GPU")) |enabled| {
            if (!std.mem.eql(u8, std.mem.span(enabled), "0")) return error.PublicK5FixtureRequired;
        }
        return error.SkipZigTest;
    }
    try fullPublicParity(5, .pony_numpy_f16_v1);
}

fn halfAt(bytes: []const u8, index: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little))));
}

// Dense normalized H128, matching the public fixture's reconstruction order rather
// than a fast butterfly's different f32 rounding. Expected weights are never derived here.
fn publicHadamard(values: *[128]f32) void {
    var out: [128]f32 = undefined;
    for (0..128) |r| {
        var sum: f32 = 0;
        for (values, 0..) |v, c| {
            const h: f32 = if (@popCount(r & c) % 2 == 0) 0.08838834764831845 else -0.08838834764831845;
            sum += h * v;
        }
        out[r] = sum;
    }
    values.* = out;
}

fn fixtureData(header: std.json.Value, data: []const u8, name: []const u8, dtype: []const u8, shape: []const u64) ![]const u8 {
    const tensor = try tensorOf(header, name);
    try std.testing.expectEqualStrings(dtype, tensor.dtype);
    try std.testing.expectEqual(shape.len, tensor.rank);
    try std.testing.expectEqualSlices(u64, shape, tensor.shape[0..tensor.rank]);
    var elements: u64 = 1;
    for (shape) |dim| elements *= dim;
    try std.testing.expect(tensor.data_offsets[0] <= tensor.data_offsets[1] and tensor.data_offsets[1] <= data.len);
    try std.testing.expectEqual(elements * 2, tensor.data_offsets[1] - tensor.data_offsets[0]);
    return data[tensor.data_offsets[0]..tensor.data_offsets[1]];
}

const ComparisonPolicy = enum { exact_words, pony_numpy_f16_v1 };

fn orderedHalf(word: u16) i32 {
    const magnitude: i32 = word & 0x7fff;
    return if (word & 0x8000 != 0) 0x8000 - magnitude else 0x8000 + magnitude;
}

// Frozen empirical CPU-oracle policy, not a universal BLAS error bound.
fn comparePublic(policy: ComparisonPolicy, actual: []const u16, reference: []const u8) !void {
    if (actual.len == 0 or reference.len != actual.len * 2) return error.PublicComparisonShape;
    var reference_square: f64 = 0;
    var error_square: f64 = 0;
    for (actual, 0..) |word, i| {
        const expected = std.mem.readInt(u16, reference[i * 2 ..][0..2], .little);
        const want: f64 = @floatCast(@as(f16, @bitCast(expected)));
        const got: f64 = @floatCast(@as(f16, @bitCast(word)));
        if (!std.math.isFinite(want) or !std.math.isFinite(got)) return error.PublicComparisonNonfinite;
        if (policy == .exact_words and word != expected) return error.PublicComparisonMismatch;
        reference_square += want * want;
        error_square += (got - want) * (got - want);
    }
    if (policy == .exact_words) return;
    if (reference_square == 0) {
        if (error_square != 0) return error.PublicComparisonMismatch;
        return;
    }
    const reference_rms = @sqrt(reference_square / @as(f64, @floatFromInt(actual.len)));
    const absolute_floor = 8 * 0x1p-23 * reference_rms;
    for (actual, 0..) |word, i| {
        const expected = std.mem.readInt(u16, reference[i * 2 ..][0..2], .little);
        const want: f64 = @floatCast(@as(f16, @bitCast(expected)));
        const got: f64 = @floatCast(@as(f16, @bitCast(word)));
        if (@abs(orderedHalf(word) - orderedHalf(expected)) > 1 and @abs(got - want) > absolute_floor)
            return error.PublicComparisonMismatch;
    }
    if (@sqrt(error_square / reference_square) > 0x1p-14) return error.PublicComparisonMismatch;
}

fn fullPublicParity(comptime rate: usize, policy: ComparisonPolicy) !void {
    const a = std.testing.allocator;
    const suffix = std.fmt.comptimePrint("K{d}", .{rate});
    const env = std.c.getenv("SUSHI_EXL3_" ++ suffix ++ "_FIXTURE");
    const path = if (env) |p| std.mem.span(p) else std.fmt.comptimePrint("lib/sushi/src/exl3/fixtures/exl3_k{d}_linear.safetensors", .{rate});
    const bytes = nocache_io.readAllNoCache(a, path, 1 << 20) catch |e| switch (e) {
        error.FileNotFound => if (env == null) return error.SkipZigTest else return e,
        else => return e,
    };
    defer a.free(bytes);
    try std.testing.expect(bytes.len >= 8);
    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    try std.testing.expect(n <= bytes.len - 8);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    defer parsed.deinit();
    if (policy == .pony_numpy_f16_v1) {
        const metadata = parsed.value.object.get("__metadata__") orelse return error.ReferenceProvenanceMissing;
        const revision = metadata.object.get("reference_revision") orelse return error.ReferenceProvenanceMissing;
        const declared = metadata.object.get("comparison_policy") orelse return error.ReferenceProvenanceMissing;
        const sample = metadata.object.get("sample_sha256") orelse return error.ReferenceProvenanceMissing;
        const codebook = metadata.object.get("codebook_sha256") orelse return error.ReferenceProvenanceMissing;
        try std.testing.expectEqualStrings("8e7fa6b1556f59fc669e25087903b279b9b0346f", revision.string);
        try std.testing.expectEqualStrings(@tagName(policy), declared.string);
        try std.testing.expectEqualStrings("d7db53973044421de435e62901243205ba0e1868fb0056bd5da19673db0af014", sample.string);
        try std.testing.expectEqualStrings("bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8", codebook.string);
    }
    const data = bytes[8 + n ..];
    const code_bytes = try fixtureData(parsed.value, data, "trellis", "U16", &.{ 8, 8, 16 * rate });
    const inner_bytes = try fixtureData(parsed.value, data, "inner", "F16", &.{ 128, 128 });
    const public_bytes = try fixtureData(parsed.value, data, "public", "F16", &.{ 128, 128 });
    const suh = try fixtureData(parsed.value, data, "suh", "F16", &.{128});
    const svh = try fixtureData(parsed.value, data, "svh", "F16", &.{128});
    const code = try a.alloc(i16, code_bytes.len / 2);
    defer a.free(code);
    for (code, 0..) |*c, i| c.* = std.mem.readInt(i16, code_bytes[i * 2 ..][0..2], .little);
    const table = try a.create([65536]u16);
    defer a.destroy(table);
    kernels.mul1Table(table);
    const inner = try a.alloc(u16, 128 * 128);
    defer a.free(inner);
    kernels.reconstruct(code, 8, 8, rate, table, inner, null);
    for (inner, 0..) |bits, i| {
        try std.testing.expectEqual(std.mem.readInt(u16, inner_bytes[i * 2 ..][0..2], .little), bits);
    }
    const weights = try a.alloc(f32, inner.len);
    defer a.free(weights);
    for (inner, weights) |bits, *w| w.* = @floatCast(@as(f16, @bitCast(bits)));
    for (0..128) |c| {
        var column: [128]f32 = undefined;
        for (0..128) |r| column[r] = weights[r * 128 + c];
        publicHadamard(&column);
        for (0..128) |r| weights[r * 128 + c] = column[r];
    }
    for (0..128) |r| {
        const scale = halfAt(suh, r);
        for (weights[r * 128 ..][0..128]) |*w| w.* *= scale;
    }
    for (0..128) |r| publicHadamard(weights[r * 128 ..][0..128]);
    var distinct = std.AutoHashMap(u16, void).init(a);
    defer distinct.deinit();
    for (weights, 0..) |w, i| {
        const got: u16 = @bitCast(@as(f16, @floatCast(w * halfAt(svh, i % 128))));
        const want = std.mem.readInt(u16, public_bytes[i * 2 ..][0..2], .little);
        inner[i] = got;
        try distinct.put(want, {});
    }
    try comparePublic(policy, inner, public_bytes);
    try std.testing.expect(distinct.count() > 100);
}

test "exl3 mixed oracle policy: local adjacency absolute floor and global drift" {
    const t = std.testing;
    var reference: [16384]u16 = @splat(0x3c00);
    var actual = reference;
    const policy: ComparisonPolicy = .pony_numpy_f16_v1;
    const bytes = std.mem.sliceAsBytes(&reference);
    try comparePublic(policy, &actual, bytes);
    actual[0] = 0x3c01;
    try comparePublic(policy, &actual, bytes);
    try t.expectError(error.PublicComparisonMismatch, comparePublic(.exact_words, &actual, bytes));
    @memset(&actual, 0x3c01);
    try t.expectError(error.PublicComparisonMismatch, comparePublic(policy, &actual, bytes));
    actual = reference;
    actual[0] = 0x3c02;
    try t.expectError(error.PublicComparisonMismatch, comparePublic(policy, &actual, bytes));
    actual[0] = 0x3bfe;
    try t.expectError(error.PublicComparisonMismatch, comparePublic(policy, &actual, bytes));
    actual[0] = 0x3bff;
    try comparePublic(policy, &actual, bytes);
    reference[0] = 0;
    actual[0] = 4;
    try comparePublic(policy, &actual, bytes);
    actual[0] = 32;
    try t.expectError(error.PublicComparisonMismatch, comparePublic(policy, &actual, bytes));
}

test "exl3 mixed oracle policy: zeros finite boundaries and malformed comparisons" {
    const t = std.testing;
    const policy: ComparisonPolicy = .pony_numpy_f16_v1;
    var reference: [16384]u16 = @splat(0x3c00);
    var actual = reference;
    const bytes = std.mem.sliceAsBytes(&reference);
    for ([_][2]u16{ .{ 1, 2 }, .{ 0x8001, 0x8002 }, .{ 0x8001, 0 }, .{ 0x03ff, 0x0400 }, .{ 0x83ff, 0x8400 }, .{ 0xbc00, 0xbbff } }) |pair| {
        reference[0] = pair[0];
        actual[0] = pair[1];
        try comparePublic(policy, &actual, bytes);
    }
    for ([_][2]u16{ .{ 0x7bff, 0x7bfe }, .{ 0xfbff, 0xfbfe } }) |pair| {
        try t.expectEqual(@as(u32, 1), @abs(orderedHalf(pair[0]) - orderedHalf(pair[1])));
        reference[0] = pair[0];
        actual[0] = pair[1];
        try t.expectError(error.PublicComparisonMismatch, comparePublic(policy, &actual, bytes));
    }
    @memset(&reference, 0);
    @memset(&actual, 0x8000);
    try comparePublic(policy, &actual, bytes);
    try t.expectError(error.PublicComparisonMismatch, comparePublic(.exact_words, &actual, bytes));
    actual[0] = 1;
    try t.expectError(error.PublicComparisonMismatch, comparePublic(policy, &actual, bytes));
    for ([_]u16{ 0x7c00, 0xfc00, 0x7e00 }) |bad| {
        actual[0] = bad;
        try t.expectError(error.PublicComparisonNonfinite, comparePublic(policy, &actual, bytes));
        actual[0] = 0;
        reference[0] = bad;
        try t.expectError(error.PublicComparisonNonfinite, comparePublic(policy, &actual, bytes));
        reference[0] = 0;
    }
    try t.expectError(error.PublicComparisonShape, comparePublic(policy, &.{}, &.{}));
    try t.expectError(error.PublicComparisonShape, comparePublic(policy, &.{0}, &.{}));
}
