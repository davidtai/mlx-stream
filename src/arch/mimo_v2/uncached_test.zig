//! The residents' load past the page cache (`caps.residents_past_page_cache` -> the host's `sdk.loader.dir(.nocache)`):
//! the uncached loader's arrays equal the cached loader's byte for byte, and the uncached load leaves none of the
//! file in the page cache. The census (a file's cached bytes by mmap + mincore, its eviction by msync MS_INVALIDATE, the
//! box's vm_stat "File-backed pages") lives in `pagecache.zig`, which the admission baseline reads at construction.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const module = @import("module.zig");
const root = @import("root.zig");
const testing = std.testing;

const pagecache = @import("pagecache.zig");
const fileBackedBytes = pagecache.fileBackedBytes;
const fileCachedBytes = pagecache.fileCachedBytes;

/// A safetensors image with resident-like tensors (BF16 norms, U32 packed weights) at unaligned offsets.
fn writeShard(a: std.mem.Allocator, dir: std.Io.Dir, name: []const u8) !void {
    const specs = [_]struct { name: []const u8, dtype: []const u8, elems: usize, width: usize }{
        .{ .name = "model.norm.weight", .dtype = "BF16", .elems = 4097, .width = 2 },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .dtype = "U32", .elems = 300_001, .width = 4 },
        .{ .name = "model.layers.0.self_attn.q_proj.scales", .dtype = "BF16", .elems = 70_003, .width = 2 },
        .{ .name = "lm_head.weight", .dtype = "U32", .elems = 600_007, .width = 4 },
    };
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    try header.writer.writeAll("{");
    var off: usize = 0;
    for (specs, 0..) |t, i| {
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ if (i > 0) "," else "", t.name, t.dtype, t.elems, off, off + t.elems * t.width });
        off += t.elems * t.width;
    }
    try header.writer.writeAll("}");
    const hlen = header.written().len;
    const image = try a.alloc(u8, 8 + hlen + off);
    defer a.free(image);
    std.mem.writeInt(u64, image[0..8], hlen, .little);
    @memcpy(image[8..][0..hlen], header.written());
    // Payload bytes: finite BF16 patterns only matter for none of this (raw bytes are compared), any pattern does.
    for (image[8 + hlen ..], 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 11);
    try dir.writeFile(testing.io, .{ .sub_path = name, .data = image });
}

fn bytesOf(x: mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(x));
    const p = mlx.mlx_array_data_uint8(x) orelse return error.MlxNoData;
    return p[0 .. mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x)];
}

test "mimo residents uncached: the arch loads its residents past the page cache (the host's loader.dir(.nocache))" {
    try testing.expect(root.caps.residents_past_page_cache);
}

test "mimo residents uncached: the page-cache proof's bound is DSV's 500 MB" {
    try testing.expect(module.residentPageCache(10_000_000_000, 10_499_999_999).within);
    try testing.expect(module.residentPageCache(10_000_000_000, 10_500_000_000).within);
    try testing.expect(!module.residentPageCache(10_000_000_000, 10_500_000_001).within);
    try testing.expectEqual(@as(i64, -4096), module.residentPageCache(8192, 4096).created);
    // The bf16res load the guard killed (loop1-bf16-20261008): 11.97 GB resident shard through the cache.
    try testing.expect(!module.residentPageCache(18_067_963_904, 18_067_963_904 + 11_967_246_874).within);
}

test "mimo residents uncached: the nocache loader's arrays equal the cached loader's byte for byte, and leave the file uncached (MLX CPU)" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, tmp.dir, "model.safetensors");
    var root_buf: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/model.safetensors", .{root_buf[0..try tmp.dir.realPath(testing.io, &root_buf)]}, 0);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);

    // Uncached first, from an evicted file: every tensor materialized, the file still uncached.
    try testing.expectEqual(@as(u64, 0), try fileCachedBytes(path, true));
    const box_before = fileBackedBytes();
    var uncached = sdk.Weights.init(a);
    defer uncached.deinit();
    try sdk.loader.file(a, &uncached, path.ptr, cpu, .{ .nocache = true });
    var it = uncached.map.valueIterator();
    var total: usize = 0;
    while (it.next()) |v| total += (try bytesOf(v.*)).len;
    const box_after = fileBackedBytes();
    const left = try fileCachedBytes(path, false);
    std.debug.print("\nmimo residents uncached: {d} tensors {d} B read; file cached after the uncached load {d} B; box file-backed {d} -> {d} B ({d} B)\n", .{ uncached.count(), total, left, box_before, box_after, @as(i64, @intCast(box_after)) - @as(i64, @intCast(box_before)) });
    try testing.expectEqual(@as(u32, 4), uncached.count());
    try testing.expectEqual(@as(u64, 0), left);

    // The cached loader (MLX's own reader): the same names, the same bytes.
    var cached = sdk.Weights.init(a);
    defer cached.deinit();
    try sdk.loader.file(a, &cached, path.ptr, cpu, .{});
    try testing.expectEqual(uncached.count(), cached.count());
    var kit = uncached.map.iterator();
    while (kit.next()) |e| {
        const other = cached.get(e.key_ptr.*) orelse return error.MissingTensor;
        try testing.expectEqual(mlx.mlx_array_dtype(other), mlx.mlx_array_dtype(e.value_ptr.*));
        try testing.expectEqualSlices(u8, try bytesOf(other), try bytesOf(e.value_ptr.*));
    }
    // The control: the cached loader does leave the file's pages cached (the census sees what the guard counts).
    try testing.expect(try fileCachedBytes(path, false) > 0);
    _ = try fileCachedBytes(path, true);
}
