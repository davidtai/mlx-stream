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
    try fullPublicParity(2);
}

test "exl3 sushi parity: K3 full H128 scaled public weights" {
    try fullPublicParity(3);
}

test "exl3 sushi parity: K4 full H128 scaled public weights" {
    try fullPublicParity(4);
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

fn fullPublicParity(comptime rate: usize) !void {
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
        try std.testing.expectEqual(want, got);
        try distinct.put(want, {});
    }
    try std.testing.expect(distinct.count() > 100);
}
