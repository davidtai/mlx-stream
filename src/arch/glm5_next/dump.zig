//! The logits dump route (`expert_streaming.settings.logits_dump_dir`, installed at construction; absent, nothing of it
//! is built): each prompt pass writes ONE safetensors file into the directory, complete before the pass returns (the
//! request's response follows it), for the KL gate against the BF16 teacher (cells/glm5next-kl-20261012):
//!   tensor `logits`    F32 [prompt rows, vocab]: row t the native head's logits after the final norm at position t (it
//!                      predicts token t + 1), the f32 copy of the head's bf16 output;
//!   tensor `token_ids` I32 [prompt rows]: the ids as sent;
//!   metadata `format`, `rows`, `vocab`, `ids_int32_sha256` (sha256 of the ids as little-endian int32: the KL cell's
//!   `token_ids_sha256` is the panel's .npy file's, so it is left out).
//! The file is written as `.<name>.partial` (a dotfile the cell skips) and renamed to `<name>.safetensors` once whole; the
//! rows arrive in chunks of `graph.dump_chunk_rows` (`append`), written past the page cache (F_NOCACHE).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const io_util = sdk.io_util;

pub const format = "glm5_next-logits-dump-v1";

pub const Writer = struct {
    a: std.mem.Allocator,
    fd: std.c.fd_t,
    tmp: [:0]u8,
    final: [:0]u8,
    rows: u32,
    vocab: u32,
    written: u32 = 0,
    data_offset: u64,

    /// The file of a prompt pass of `ids` (every row of the pass), named by this process and `seq` (the load's count of
    /// dumps): its header and nothing else yet.
    pub fn open(a: std.mem.Allocator, dir: []const u8, seq: u64, ids: []const u32, vocab: u32) !Writer {
        const n: u64 = ids.len;
        const logits_bytes = n * vocab * 4;
        var digest: [32]u8 = undefined;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (ids) |id| {
            var le: [4]u8 = undefined;
            std.mem.writeInt(i32, &le, @intCast(id), .little);
            h.update(&le);
        }
        h.final(&digest);
        const hex = std.fmt.bytesToHex(digest, .lower);
        var header = try std.fmt.allocPrint(a,
            \\{{"__metadata__":{{"format":"{s}","rows":"{d}","vocab":"{d}","ids_int32_sha256":"{s}"}},"logits":{{"dtype":"F32","shape":[{d},{d}],"data_offsets":[0,{d}]}},"token_ids":{{"dtype":"I32","shape":[{d}],"data_offsets":[{d},{d}]}}}}
        , .{ format, n, vocab, &hex, n, vocab, logits_bytes, n, logits_bytes, logits_bytes + 4 * n });
        defer a.free(header);
        const raw_len = header.len;
        const padded = std.mem.alignForward(usize, raw_len, 8);
        header = try a.realloc(header, padded);
        @memset(header[raw_len..], ' ');
        const pid = std.c.getpid();
        const tmp = try std.fmt.allocPrintSentinel(a, "{s}/.glm5next-logits-{d}-{d:0>4}.partial", .{ dir, pid, seq }, 0);
        errdefer a.free(tmp);
        const final = try std.fmt.allocPrintSentinel(a, "{s}/glm5next-logits-{d}-{d:0>4}.safetensors", .{ dir, pid, seq }, 0);
        errdefer a.free(final);
        const fd = std.c.open(tmp.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.Glm5NextDumpOpen;
        errdefer {
            _ = std.c.close(fd);
            _ = std.c.unlink(tmp.ptr);
        }
        try io_util.noCache(fd, .{});
        var len_le: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_le, padded, .little);
        try writeAt(fd, &len_le, 0);
        try writeAt(fd, header, 8);
        return .{ .a = a, .fd = fd, .tmp = tmp, .final = final, .rows = @intCast(n), .vocab = vocab, .data_offset = 8 + padded };
    }

    /// The next rows: `x` f32 [r, vocab], evaluated and row-contiguous (the head's output cast), written in place.
    pub fn append(self: *Writer, x: mlx.mlx_array) !void {
        const shape = mlx.getShape(x);
        if (shape.len != 2 or shape[1] != @as(c_int, @intCast(self.vocab)) or mlx.mlx_array_dtype(x) != .float32) return error.Glm5NextDumpLayout;
        const r: u32 = @intCast(shape[0]);
        if (self.written + r > self.rows) return error.Glm5NextDumpRows;
        const strides = mlx.mlx_array_strides(x);
        if (strides[0] != self.vocab or strides[1] != 1) return error.Glm5NextDumpLayout;
        const data = mlx.mlx_array_data_float32(x) orelse return error.Glm5NextDumpLayout;
        const bytes = std.mem.sliceAsBytes(data[0 .. @as(usize, r) * self.vocab]);
        try writeAt(self.fd, bytes, self.data_offset + @as(u64, self.written) * self.vocab * 4);
        self.written += r;
    }

    /// Every row written: the ids after the logits, the file closed and renamed to its final name.
    pub fn finish(self: *Writer, ids: []const u32) !void {
        defer self.free();
        errdefer _ = std.c.unlink(self.tmp.ptr);
        if (self.written != self.rows or ids.len != self.rows) {
            _ = std.c.close(self.fd);
            return error.Glm5NextDumpRows;
        }
        const raw = try self.a.alloc(i32, ids.len);
        defer self.a.free(raw);
        for (raw, ids) |*o, id| o.* = std.mem.nativeToLittle(i32, @intCast(id));
        writeAt(self.fd, std.mem.sliceAsBytes(raw), self.data_offset + @as(u64, self.rows) * self.vocab * 4) catch |e| {
            _ = std.c.close(self.fd);
            return e;
        };
        if (std.c.close(self.fd) != 0) return error.Glm5NextDumpWrite;
        if (std.c.rename(self.tmp.ptr, self.final.ptr) != 0) return error.Glm5NextDumpRename;
    }

    /// A failed pass: the partial file removed.
    pub fn abort(self: *Writer) void {
        _ = std.c.close(self.fd);
        _ = std.c.unlink(self.tmp.ptr);
        self.free();
    }

    fn free(self: *Writer) void {
        self.a.free(self.tmp);
        self.a.free(self.final);
    }
};

fn writeAt(fd: std.c.fd_t, bytes: []const u8, offset: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pwrite(fd, bytes[done..].ptr, bytes.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
        if (n <= 0) return error.Glm5NextDumpWrite;
        done += @intCast(n);
    }
}

/// The resident bytes of one dump chunk: `rows` of the head's bf16 output and their f32 copy (the final norm's rows and
/// the hc mean beside them).
pub fn chunkBytes(rows: u64, vocab: u64, hidden: u64) u64 {
    return rows * vocab * (2 + 4) + 2 * rows * hidden * 2;
}

const testing = std.testing;

test "glm5_next dump: the header names logits F32 [rows, vocab] and token_ids I32 [rows] in place, the ids' int32 sha, no token_ids_sha256" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);
    const ids = [_]u32{ 5, 154_842, 0 };
    var w = try Writer.open(a, dir, 7, &ids, 4);
    // The rows as the pass hands them: written in place through the file's own bytes (no MLX array here).
    const rows = [_]f32{ 0.5, -1, 2, 3, 4, 5, 6, 7.25, -8, 9, 10, 11 };
    try writeAt(w.fd, std.mem.sliceAsBytes(&rows), w.data_offset);
    w.written = 3;
    try w.finish(&ids);
    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "glm5next-logits-{d}-0007.safetensors", .{std.c.getpid()});
    const file = try tmp.dir.readFileAlloc(testing.io, name, a, .limited(1 << 20));
    defer a.free(file);
    const n = std.mem.readInt(u64, file[0..8], .little);
    try testing.expect(n % 8 == 0);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, std.mem.trimEnd(u8, file[8..][0..n], " "), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const meta = root.get("__metadata__").?.object;
    try testing.expect(meta.get("token_ids_sha256") == null);
    try testing.expectEqualStrings("3", meta.get("rows").?.string);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&[_]i32{ 5, 154_842, 0 }), &digest, .{});
    try testing.expectEqualStrings(&std.fmt.bytesToHex(digest, .lower), meta.get("ids_int32_sha256").?.string);
    const logits = root.get("logits").?.object;
    try testing.expectEqualStrings("F32", logits.get("dtype").?.string);
    try testing.expectEqual(@as(i64, 48), logits.get("data_offsets").?.array.items[1].integer);
    const body = file[8 + n ..];
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&rows), body[0..48]);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]i32{ 5, 154_842, 0 }), body[48..60]);
    try testing.expectEqual(@as(usize, 8 + n + 60), file.len);
    // No partial file is left beside it.
    var it = tmp.dir.iterate();
    var count: usize = 0;
    while (try it.next(testing.io)) |_| count += 1;
    try testing.expectEqual(@as(usize, 1), count);
    // The resident chunk bound the README states (256 rows of Flash's vocab, hidden 4096).
    try testing.expectEqual(@as(u64, 242_089_984), chunkBytes(256, 154_880, 4096));
}
