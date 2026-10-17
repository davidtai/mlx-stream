//! A pack's verified sidecar: one safetensors file that a manifest states tensor by tensor (dtype, shape, offset in the
//! payload, length, sha256), with the file's size and sha256. `verify` reads it once, past the page cache: the file's
//! size is the manifest's, the header's entries are the manifest's (the same names, dtypes, shapes and offsets, and no
//! other tensor), the tensors tile the payload in offset order, and each tensor's sha256 and the file's are the stated
//! ones. The sink takes a tensor's bytes only after its sha256 matched (the buffer is reused after `take` returns), so the
//! arrays a caller makes hold exactly the verified bytes. The draft lanes' weights: MiMo's MTP heads (mimo_v2/mtp.zig)
//! and Beam's DFlash2 drafter (beam/drafter.zig). A caller maps the refusals to its own names.

const std = @import("std");
const sdk = @import("sdk");
const io_util = sdk.io_util;

pub const Error = error{ SidecarMissing, SidecarSize, SidecarHeader, SidecarLayout, SidecarTensorSha, SidecarSha };

/// The widest tensor rank a manifest states.
pub const max_rank = 4;

/// One tensor as the manifest states it: its safetensors name and dtype ("BF16", "U32", ...), its shape, its bytes at
/// `offset` from the payload's start, their sha256.
pub const Entry = struct {
    name: []const u8,
    dtype: []const u8,
    rank: u8,
    dims: [max_rank]u64 = @splat(0),
    offset: u64,
    length: u64,
    sha256: [32]u8,
};

/// Reads `dir/data_file` past the page cache and checks it against `entries` (the manifest's tensors), `data_size`,
/// `data_sha256` and `payload_bytes`; `sink.take(index, bytes)` receives entry `index`'s bytes once their sha256 matched,
/// in file order. A header longer than `header_limit` refuses.
pub fn verify(a: std.mem.Allocator, dir: []const u8, data_file: []const u8, entries: []const Entry, data_size: u64, data_sha256: [32]u8, payload_bytes: u64, header_limit: u64, sink: anytype) !void {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, data_file }, 0);
    defer a.free(path);
    const fd = io_util.openNoCache(path.ptr, .{}) catch |e| return if (e == error.FileNotFound) error.SidecarMissing else e;
    defer _ = std.c.close(fd);
    const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size_i < 8 or @as(u64, @intCast(size_i)) != data_size) return error.SidecarSize;
    var prefix: [8]u8 = undefined;
    try io_util.readAligned(fd, &prefix, 0);
    const hlen = std.mem.readInt(u64, &prefix, .little);
    if (hlen < 2 or hlen > header_limit or 8 + hlen + payload_bytes != data_size) return error.SidecarHeader;
    const header = try a.alloc(u8, @intCast(hlen));
    defer a.free(header);
    try io_util.readAligned(fd, header, 8);
    try checkHeader(a, header, entries);
    var file_hash = std.crypto.hash.sha2.Sha256.init(.{});
    file_hash.update(&prefix);
    file_hash.update(header);
    var widest: u64 = 0;
    for (entries) |e| widest = @max(widest, e.length);
    const buf = try std.heap.page_allocator.alloc(u8, @intCast(widest));
    defer std.heap.page_allocator.free(buf);
    const order = try a.alloc(usize, entries.len);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, entries, struct {
        fn lt(es: []const Entry, x: usize, y: usize) bool {
            return es[x].offset < es[y].offset;
        }
    }.lt);
    var next: u64 = 0;
    for (order) |i| {
        const e = entries[i];
        if (e.offset != next) return error.SidecarLayout;
        const bytes = buf[0..@intCast(e.length)];
        try io_util.readAligned(fd, bytes, 8 + hlen + e.offset);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        if (!std.mem.eql(u8, &digest, &e.sha256)) return error.SidecarTensorSha;
        file_hash.update(bytes);
        try sink.take(i, bytes);
        next = e.offset + e.length;
    }
    if (next != payload_bytes) return error.SidecarLayout;
    var whole: [32]u8 = undefined;
    file_hash.final(&whole);
    if (!std.mem.eql(u8, &whole, &data_sha256)) return error.SidecarSha;
}

/// The header's tensors are exactly `entries` (`__metadata__` aside): each one's dtype, shape and data_offsets.
fn checkHeader(a: std.mem.Allocator, header: []const u8, entries: []const Entry) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.SidecarHeader;
    defer parsed.deinit();
    if (parsed.value != .object) return error.SidecarHeader;
    var count: usize = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| count += @intFromBool(!std.mem.eql(u8, kv.key_ptr.*, "__metadata__"));
    if (count != entries.len) return error.SidecarHeader;
    for (entries) |e| {
        const t = parsed.value.object.get(e.name) orelse return error.SidecarHeader;
        if (t != .object) return error.SidecarHeader;
        const dt = t.object.get("dtype") orelse return error.SidecarHeader;
        if (dt != .string or !std.mem.eql(u8, dt.string, e.dtype)) return error.SidecarHeader;
        const shape = t.object.get("shape") orelse return error.SidecarHeader;
        if (shape != .array or shape.array.items.len != e.rank) return error.SidecarHeader;
        for (shape.array.items, e.dims[0..e.rank]) |d, want| if (d != .integer or d.integer != want) return error.SidecarHeader;
        const off = t.object.get("data_offsets") orelse return error.SidecarHeader;
        if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer or
            off.array.items[0].integer != e.offset or off.array.items[1].integer != e.offset + e.length) return error.SidecarHeader;
    }
}

/// A 64-character lowercase or uppercase hex string as 32 bytes (a manifest's sha256 fields); null otherwise.
pub fn hex32(v: ?std.json.Value) ?[32]u8 {
    const s = v orelse return null;
    if (s != .string or s.string.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s.string) catch return null;
    return out;
}

const testing = std.testing;

/// A two-tensor sidecar in `dir` (U32 [2, 3] then BF16 [4]) and its entries; `flip` changes a payload byte after the
/// hashes.
fn writeTiny(io: std.Io, dir: std.Io.Dir, flip: bool, entries: *[2]Entry, whole: *[32]u8) !u64 {
    var payload: [2 * 3 * 4 + 4 * 2]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @intCast((i * 37 + 11) % 251);
    const header = "{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"U32\",\"shape\":[2,3],\"data_offsets\":[0,24]},\"b\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[24,32]}}";
    entries[0] = .{ .name = "a", .dtype = "U32", .rank = 2, .dims = .{ 2, 3, 0, 0 }, .offset = 0, .length = 24, .sha256 = undefined };
    entries[1] = .{ .name = "b", .dtype = "BF16", .rank = 1, .dims = .{ 4, 0, 0, 0 }, .offset = 24, .length = 8, .sha256 = undefined };
    std.crypto.hash.sha2.Sha256.hash(payload[0..24], &entries[0].sha256, .{});
    std.crypto.hash.sha2.Sha256.hash(payload[24..32], &entries[1].sha256, .{});
    var lenb: [8]u8 = undefined;
    std.mem.writeInt(u64, &lenb, header.len, .little);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(&lenb);
    h.update(header);
    h.update(&payload);
    h.final(whole);
    if (flip) payload[25] ^= 1;
    var file: [8 + header.len + payload.len]u8 = undefined;
    @memcpy(file[0..8], &lenb);
    @memcpy(file[8..][0..header.len], header);
    @memcpy(file[8 + header.len ..], &payload);
    try dir.writeFile(io, .{ .sub_path = "tiny.safetensors", .data = &file });
    return file.len;
}

test "sdk_ext sidecar: a tiny file verifies tensor by tensor in file order; a flipped byte, a missing or renamed entry, a wrong size or file hash refuse by name" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(io, &dir_buf)];
    var entries: [2]Entry = undefined;
    var whole: [32]u8 = undefined;
    const size = try writeTiny(io, tmp.dir, false, &entries, &whole);
    const Sink = struct {
        seen: *[2]usize,
        n: *usize,
        pub fn take(self: @This(), i: usize, bytes: []const u8) !void {
            self.seen[self.n.*] = i * 1000 + bytes.len;
            self.n.* += 1;
        }
    };
    var seen: [2]usize = undefined;
    var n: usize = 0;
    try verify(a, dir, "tiny.safetensors", &entries, size, whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n });
    try testing.expectEqualSlices(usize, &.{ 24, 1008 }, &seen);
    n = 0;
    try testing.expectError(error.SidecarSize, verify(a, dir, "tiny.safetensors", &entries, size + 1, whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n }));
    var bad_whole = whole;
    bad_whole[0] ^= 0xff;
    n = 0;
    try testing.expectError(error.SidecarSha, verify(a, dir, "tiny.safetensors", &entries, size, bad_whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n }));
    var renamed = entries;
    renamed[1].name = "c";
    n = 0;
    try testing.expectError(error.SidecarHeader, verify(a, dir, "tiny.safetensors", &renamed, size, whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n }));
    n = 0;
    try testing.expectError(error.SidecarHeader, verify(a, dir, "tiny.safetensors", entries[0..1], size, whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n }));
    n = 0;
    try testing.expectError(error.SidecarMissing, verify(a, dir, "none.safetensors", &entries, size, whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n }));
    _ = try writeTiny(io, tmp.dir, true, &entries, &whole);
    n = 0;
    try testing.expectError(error.SidecarTensorSha, verify(a, dir, "tiny.safetensors", &entries, size, whole, 32, 1 << 20, Sink{ .seen = &seen, .n = &n }));
    // The first tensor's bytes reached the sink before the second's hash refused.
    try testing.expectEqual(@as(usize, 1), n);
}
