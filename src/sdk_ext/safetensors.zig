//! A pack's safetensors index and shard headers, read past the page cache with no payload: the residents' bytes as MLX
//! allocates them (the bills of the streamed arches). A caller maps the refusals to its own names.

const std = @import("std");
const io_util = @import("sdk").io_util;

/// An MLX buffer of `n` bytes as the Metal allocator takes it: past one page, rounded up to whole pages.
pub fn bufferBytes(n: u64, page: u64) u64 {
    return if (n > page) std.mem.alignForward(u64, n, page) else n;
}

pub const Residents = struct { bytes: u64, tensors: u64 };

/// The residents `kept(ctx, name)` keeps of the pack's indexed shards, each in its own MLX buffer (page-rounded past one
/// page). Every shard header is bounded and checked against its file; a tensor outside its file refuses
/// (`ResidentShard`); an index that is not a weight map of `.safetensors` files refuses (`ResidentIndex`).
pub fn residentBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, ctx: anytype, comptime kept: fn (@TypeOf(ctx), []const u8) bool, page: u64) !Residents {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const raw = d.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20)) catch return error.ResidentIndex;
    defer a.free(raw);
    const index = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return error.ResidentIndex;
    defer index.deinit();
    const wm = if (index.value == .object) index.value.object.get("weight_map") orelse return error.ResidentIndex else return error.ResidentIndex;
    if (wm != .object or wm.object.count() == 0) return error.ResidentIndex;
    var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer shards.deinit(a);
    var it = wm.object.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.*;
        if (v != .string or !std.mem.endsWith(u8, v.string, ".safetensors") or std.mem.indexOfAny(u8, v.string, "/\\") != null) return error.ResidentIndex;
        try shards.put(a, v.string, {});
    }
    var out: Residents = .{ .bytes = 0, .tensors = 0 };
    for (shards.keys()) |shard| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, shard }, 0);
        defer a.free(path);
        const fd = try io_util.openNoCache(path.ptr, .{});
        defer _ = std.c.close(fd);
        const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size_i < 8) return error.ResidentShard;
        const size: u64 = @intCast(size_i);
        var prefix: [8]u8 = undefined;
        try readAt(fd, &prefix, 0);
        const len = std.mem.readInt(u64, &prefix, .little);
        if (len < 2 or len > 64 << 20 or len > size - 8) return error.ResidentShard;
        const header = try a.alloc(u8, @intCast(len));
        defer a.free(header);
        try readAt(fd, header, 8);
        const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.ResidentShard;
        defer parsed.deinit();
        if (parsed.value != .object) return error.ResidentShard;
        var tensors = parsed.value.object.iterator();
        while (tensors.next()) |t| {
            const name = t.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__") or !kept(ctx, name)) continue;
            const desc = t.value_ptr.*;
            const offsets = if (desc == .object) desc.object.get("data_offsets") orelse return error.ResidentShard else return error.ResidentShard;
            if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[0] != .integer or offsets.array.items[1] != .integer) return error.ResidentShard;
            const lo = offsets.array.items[0].integer;
            const hi = offsets.array.items[1].integer;
            if (lo < 0 or hi < lo or @as(u64, @intCast(hi)) > size - 8 - len) return error.ResidentShard;
            out.bytes += bufferBytes(@intCast(hi - lo), page);
            out.tensors += 1;
        }
    }
    return out;
}

fn readAt(fd: std.c.fd_t, bytes: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes[done..].ptr, bytes.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
        if (n <= 0) return error.ResidentShard;
        done += @intCast(n);
    }
}

