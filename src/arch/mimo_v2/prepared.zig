//! mlx-stream load boundary. Inspect only headers before the host loads resident tensors.
const std = @import("std");
const io_util = @import("sdk").io_util;

pub const family = "mlx-stream";
pub const format = "mlx-stream-mxfp4-expert-bank-v1";
const mimo = @import("config.zig");
/// Header admission bound: the widest supported residents' geometric payload (all BF16: 11,967,197,056 B). The exact
/// payload is checked against the parsed storage at the bill (bill.zig `archBill`).
pub const resident_limit: u64 = (mimo.Config{ .residents = @splat(mimo.Storage.bf16) }).residentPayloadBytes();
const header_limit = 8 << 20;

pub fn declared(v: std.json.Value) bool {
    if (v != .object) return false;
    inline for (.{ .{ "family", family }, .{ "format", format }, .{ "manifest", "expert-manifest.json" } }) |pair| {
        const field = v.object.get(pair[0]) orelse return false;
        if (field != .string or !std.mem.eql(u8, field.string, pair[1])) return false;
    }
    return true;
}

fn residentName(name: []const u8) bool {
    if (std.mem.indexOf(u8, name, ".switch_mlp.") != null or std.mem.indexOf(u8, name, ".experts.") != null) return false;
    return std.mem.startsWith(u8, name, "model.layers.") or
        std.mem.startsWith(u8, name, "model.embed_tokens.") or
        std.mem.startsWith(u8, name, "model.norm.") or
        std.mem.startsWith(u8, name, "lm_head.");
}

/// Validates the actual files selected by the host loader, not just the index's tensor list.
/// No tensor payload is read. All header metadata is bounded and rejected before MLX sees it.
pub fn residentBytes(a: std.mem.Allocator, io: std.Io, model_dir: []const u8) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(header_limit)) catch return error.MimoResidentIndexInvalid;
    defer a.free(raw);
    const index = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return error.MimoResidentIndexInvalid;
    defer index.deinit();
    if (index.value != .object) return error.MimoResidentIndexInvalid;
    const wm = index.value.object.get("weight_map") orelse return error.MimoResidentIndexInvalid;
    if (wm != .object or wm.object.count() == 0) return error.MimoResidentIndexInvalid;
    var shards: std.StringHashMapUnmanaged(void) = .empty;
    defer shards.deinit(a);
    var it = wm.object.iterator();
    while (it.next()) |entry| {
        const v = entry.value_ptr.*;
        if (!residentName(entry.key_ptr.*) or v != .string or
            !std.mem.endsWith(u8, v.string, ".safetensors") or
            std.mem.indexOfAny(u8, v.string, "/\\") != null or std.mem.indexOfScalar(u8, v.string, 0) != null)
            return error.MimoResidentIndexInvalid;
        try shards.put(a, v.string, {});
    }
    var total: u64 = 0;
    var found: usize = 0;
    var names = shards.keyIterator();
    while (names.next()) |shard| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ model_dir, shard.* }, 0);
        defer a.free(path);
        const fd = try io_util.openNoCache(path.ptr, .{});
        defer _ = std.c.close(fd);
        const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size_i < 8 or size_i > resident_limit + header_limit + 8) return error.MimoResidentShardInvalid;
        const size: u64 = @intCast(size_i);
        var prefix: [8]u8 = undefined;
        try readAt(fd, &prefix, 0);
        const len = std.mem.readInt(u64, &prefix, .little);
        if (len < 2 or len > header_limit or len > size - 8) return error.MimoResidentShardInvalid;
        const header = try a.alloc(u8, @intCast(len));
        defer a.free(header);
        try readAt(fd, header, 8);
        const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.MimoResidentShardInvalid;
        defer parsed.deinit();
        if (parsed.value != .object) return error.MimoResidentShardInvalid;
        var tensors = parsed.value.object.iterator();
        var payload: u64 = 0;
        while (tensors.next()) |tensor| {
            const name = tensor.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            const indexed = wm.object.get(name) orelse return error.MimoResidentShardInvalid;
            if (!residentName(name) or indexed != .string or !std.mem.eql(u8, indexed.string, shard.*)) return error.MimoResidentShardInvalid;
            const desc = tensor.value_ptr.*;
            if (desc != .object) return error.MimoResidentShardInvalid;
            const dtype = desc.object.get("dtype") orelse return error.MimoResidentShardInvalid;
            const shape = desc.object.get("shape") orelse return error.MimoResidentShardInvalid;
            const offsets = desc.object.get("data_offsets") orelse return error.MimoResidentShardInvalid;
            if (dtype != .string or shape != .array or offsets != .array or offsets.array.items.len != 2) return error.MimoResidentShardInvalid;
            // U8: the fp8res pack's e4m3 codes (config.zig `Storage.fp8`); every tensor's role and dtype are checked again
            // where it loads (module.zig `QLinear.load`).
            const width: u64 = if (std.mem.eql(u8, dtype.string, "BF16")) 2 else if (std.mem.eql(u8, dtype.string, "F32") or std.mem.eql(u8, dtype.string, "U32")) 4 else if (std.mem.eql(u8, dtype.string, "U8")) 1 else return error.MimoResidentShardInvalid;
            var bytes = width;
            for (shape.array.items) |dim| {
                if (dim != .integer or dim.integer < 1) return error.MimoResidentShardInvalid;
                bytes = std.math.mul(u64, bytes, @intCast(dim.integer)) catch return error.MimoResidentShardInvalid;
            }
            const start = offsets.array.items[0];
            const end = offsets.array.items[1];
            if (start != .integer or end != .integer or start.integer < 0 or end.integer < start.integer) return error.MimoResidentShardInvalid;
            if (@as(u64, @intCast(end.integer)) > size - 8 - len or end.integer - start.integer != bytes) return error.MimoResidentShardInvalid;
            payload = std.math.add(u64, payload, bytes) catch return error.MimoResidentShardInvalid;
            found += 1;
        }
        if (payload != size - 8 - len) return error.MimoResidentShardInvalid;
        total = std.math.add(u64, total, payload) catch return error.MimoResidentShardInvalid;
        if (total > resident_limit) return error.MimoResidentShardInvalid;
    }
    if (found != wm.object.count()) return error.MimoResidentIndexInvalid;
    return total;
}

fn readAt(fd: std.c.fd_t, bytes: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes[done..].ptr, bytes.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
        if (n <= 0) return error.MimoResidentShardInvalid;
        done += @intCast(n);
    }
}

test "mlx-stream mimo prepared: only the mlx-stream MXFP4 declaration claims the source" {
    const a = std.testing.allocator;
    const good = try std.json.parseFromSlice(std.json.Value, a,
        \\{"family":"mlx-stream","format":"mlx-stream-mxfp4-expert-bank-v1","manifest":"expert-manifest.json"}
    , .{});
    defer good.deinit();
    try std.testing.expect(declared(good.value));
    try std.testing.expect(!declared(.null));
    const bad = try std.json.parseFromSlice(std.json.Value, a,
        \\{"family":"mlx-stream","format":"exl3","manifest":"expert-manifest.json"}
    , .{});
    defer bad.deinit();
    try std.testing.expect(!declared(bad.value));
}

test "mlx-stream mimo prepared: actual shard headers prevent index-hidden expert loads" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"model.norm.weight":"resident.safetensors"}}
    });
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try writeShard(io, tmp.dir,
        \\{"model.norm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
    );
    try std.testing.expectEqual(@as(u64, 4), try residentBytes(a, io, path));
    try writeShard(io, tmp.dir,
        \\{"model.norm.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"model.layers.1.mlp.switch_mlp.gate_proj.scales":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}
    );
    try std.testing.expectError(error.MimoResidentShardInvalid, residentBytes(a, io, path));
}

test "mlx-stream mimo prepared fp8res: U8 codes and F32 scales count by their bytes; other dtypes still refuse" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"model.layers.0.self_attn.q_proj.weight":"resident.safetensors","model.layers.0.self_attn.q_proj.weight_scale_inv":"resident.safetensors"}}
    });
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try writeShardPayload(io, tmp.dir,
        \\{"model.layers.0.self_attn.q_proj.weight":{"dtype":"U8","shape":[2,4],"data_offsets":[0,8]},"model.layers.0.self_attn.q_proj.weight_scale_inv":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}
    , 12);
    try std.testing.expectEqual(@as(u64, 12), try residentBytes(a, io, path));
    try writeShardPayload(io, tmp.dir,
        \\{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F8_E4M3","shape":[2,4],"data_offsets":[0,8]},"model.layers.0.self_attn.q_proj.weight_scale_inv":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}
    , 12);
    try std.testing.expectError(error.MimoResidentShardInvalid, residentBytes(a, io, path));
}

fn writeShardPayload(io: std.Io, dir: std.Io.Dir, header: []const u8, payload: usize) !void {
    var bytes: [2048]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try dir.writeFile(io, .{ .sub_path = "resident.safetensors", .data = bytes[0 .. 8 + header.len + payload] });
}

fn writeShard(io: std.Io, dir: std.Io.Dir, header: []const u8) !void {
    var bytes: [2048]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try dir.writeFile(io, .{ .sub_path = "resident.safetensors", .data = bytes[0 .. 8 + header.len + 4] });
}
