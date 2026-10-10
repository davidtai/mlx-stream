//! R1 (mimo/decode-rows-audit, settings `"embedding": "rows"`): the input embedding's rows read on demand from the pack's
//! resident shard, past the page cache, instead of the resident BF16 table (152,576 x 4,096 x 2 = 1,249,902,592 B on
//! MiMo-V2.6-Flash: two resident expert rows in either phase). DeepSeek-V4.1's `embedding_rows` reader, reused as is
//! (`ngram_table.NgramTable.openTensor` + `gatherRaw`: F_NOCACHE with read-ahead off, whole aligned pages only, each
//! distinct row read once, 15 helpers at prompt widths, the caller alone below 32 ids). A lookup is the table's rows byte
//! for byte (a copy, no arithmetic): `[n, hidden]` BF16, what `take(table, ids)` returns on the table route. Only a dense
//! BF16 table qualifies (settings `validate` refuses the route on an affine one).
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const io_util = sdk.io_util;
const ngram = @import("../../ngram_table.zig");
const mimo = @import("config.zig");
const Array = mlx.mlx_array;

/// The table's tensor in the pack's resident shards (`model.safetensors.index.json` names its shard).
pub const tensor_name = "model.embed_tokens.weight";

/// The resident table's bytes the route never loads (the bill's `residents` less these).
pub fn tableBytes(c: mimo.Config) u64 {
    return @as(u64, c.vocab_size) * c.hidden_size * 2;
}

/// Decode-width lookups (at most `cache_max_ids` ids: the verify's 1-4 rows, a head's rows) go through a host cache of
/// `cache_rows` table rows by token id, so a round reads only its new tokens (t1, the drafts) from the drive: the heads
/// re-embed the committed tokens of the last rounds every round. Wider lookups (a prompt call) bypass it.
pub const cache_rows: u32 = 1024;
pub const cache_max_ids: usize = 256;
const no_token = std.math.maxInt(u32);

/// The route's host bytes for the table's life: the gather's stages and scratch (`RowGather.persistentBytes`) and the
/// decode-width cache (its rows, slot tokens and clock bits, and the token -> slot map over the vocabulary).
pub fn persistentBytes(c: mimo.Config) u64 {
    return io_util.RowGather.persistentBytes(ngram.NgramTable.raw_gather_helpers, ngram.NgramTable.raw_gather_max_ids) +
        @as(u64, cache_rows) * (@as(u64, c.hidden_size) * 2 + @sizeOf(u32) + @sizeOf(bool)) + @as(u64, c.vocab_size) * @sizeOf(u32);
}

/// A lookup's host staging at `rows` rows (freed when the lookup's array is made): a cached lookup stages its output and
/// its misses, so the bill passes twice its widest decode lookup (bill.zig `embedRowsBytes`).
pub fn stagingBytes(c: mimo.Config, rows: u64) u64 {
    return rows * c.hidden_size * 2;
}

pub const Rows = struct {
    table: ngram.NgramTable,
    hidden: u32,
    /// The decode-width cache: `cache_rows` rows (each a copy of the table's bytes, made when the row was first read),
    /// the token in each slot, its clock bit, and the token -> slot + 1 map (0: not cached). Clock replacement.
    cache: []u8 = &.{},
    slot_token: []u32 = &.{},
    ref: []bool = &.{},
    where: []u32 = &.{},
    hand: u32 = 0,

    /// Construction: the table's shard from the resident index, its header checked against the config's geometry
    /// (BF16 [vocab, hidden]); the gather's helpers start here, once.
    pub fn open(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, c: mimo.Config) !*Rows {
        const path = try shardPath(a, io, model_dir);
        defer a.free(path);
        var t = ngram.NgramTable.openTensor(path, tensor_name) catch |e| {
            std.log.scoped(.mimo).err("embedding rows: {s} in {s}: {s}", .{ tensor_name, path, @errorName(e) });
            return error.MimoEmbeddingRows;
        };
        errdefer t.close();
        if (t.rows != c.vocab_size or t.dim != c.hidden_size or t.bits != 16) return error.MimoEmbeddingRowsGeometry;
        const self = try a.create(Rows);
        errdefer a.destroy(self);
        self.* = .{ .table = t, .hidden = c.hidden_size };
        self.cache = try a.alloc(u8, @as(usize, cache_rows) * self.rowBytes());
        errdefer a.free(self.cache);
        self.slot_token = try a.alloc(u32, cache_rows);
        errdefer a.free(self.slot_token);
        self.ref = try a.alloc(bool, cache_rows);
        errdefer a.free(self.ref);
        self.where = try a.alloc(u32, c.vocab_size);
        @memset(self.slot_token, no_token);
        @memset(self.ref, false);
        @memset(self.where, 0);
        return self;
    }

    pub fn close(self: *Rows, a: std.mem.Allocator) void {
        self.table.close();
        a.free(self.cache);
        a.free(self.slot_token);
        a.free(self.ref);
        a.free(self.where);
        a.destroy(self);
    }

    fn rowBytes(self: *const Rows) usize {
        return @as(usize, self.hidden) * 2;
    }

    /// The rows of `ids` into `out`: a decode width (<= `cache_max_ids` ids) through the cache (hits copied from it, the
    /// misses read from the table in one gather, then cached), a wider call straight from the table. The output is
    /// complete before any insertion, so an eviction never touches this call's rows. The table's bytes either way.
    pub fn gatherCached(self: *Rows, a: std.mem.Allocator, ids: []const u32, out: []u8) !void {
        if (ids.len > cache_max_ids) return self.gather(ids, out);
        const rb = self.rowBytes();
        if (out.len != ids.len * rb) return error.NgramTableRegion;
        var miss_ids: [cache_max_ids]u32 = undefined;
        var miss_at: [cache_max_ids]u32 = undefined;
        var n_miss: usize = 0;
        for (ids, 0..) |id, i| {
            if (id >= self.where.len) return error.NgramTableRegion;
            const w = self.where[id];
            if (w != 0) {
                @memcpy(out[i * rb ..][0..rb], self.cache[(w - 1) * rb ..][0..rb]);
                self.ref[w - 1] = true;
                continue;
            }
            miss_ids[n_miss] = id;
            miss_at[n_miss] = @intCast(i);
            n_miss += 1;
        }
        if (n_miss == 0) return;
        const tmp = try a.alloc(u8, n_miss * rb);
        defer a.free(tmp);
        try self.gather(miss_ids[0..n_miss], tmp);
        for (miss_at[0..n_miss], 0..) |at, k| @memcpy(out[@as(usize, at) * rb ..][0..rb], tmp[k * rb ..][0..rb]);
        for (miss_ids[0..n_miss], 0..) |id, k| {
            if (self.where[id] != 0) continue; // a repeat within this call: cached by its first occurrence
            const slot = self.victim();
            const old = self.slot_token[slot];
            if (old != no_token) self.where[old] = 0;
            self.slot_token[slot] = id;
            self.where[id] = slot + 1;
            self.ref[slot] = true;
            @memcpy(self.cache[@as(usize, slot) * rb ..][0..rb], tmp[k * rb ..][0..rb]);
        }
    }

    /// The clock's next victim: an empty slot, else the first slot whose bit is clear (bits cleared as the hand passes).
    fn victim(self: *Rows) u32 {
        while (true) {
            const s = self.hand;
            self.hand = (self.hand + 1) % cache_rows;
            if (self.slot_token[s] == no_token or !self.ref[s]) return s;
            self.ref[s] = false;
        }
    }

    /// The rows of `ids`, in order, into `out` (`ids.len * hidden * 2` bytes): the table's bytes.
    pub fn gather(self: *const Rows, ids: []const u32, out: []u8) !void {
        return self.table.gatherRaw(ids, out);
    }

    /// `ids` embedded: [n, hidden] BF16 (MLX-owned: the staging is copied in and freed here).
    pub fn lookup(self: *Rows, a: std.mem.Allocator, ids: []const u32) !Array {
        const buf = try a.alloc(u8, ids.len * self.rowBytes());
        defer a.free(buf);
        try self.gatherCached(a, ids, buf);
        const out = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ @intCast(ids.len), @intCast(self.hidden) }, 2, .bfloat16);
        if (out.ctx == null) return error.MimoEmbeddingRowsArray;
        return out;
    }
};

/// The resident shard holding `tensor_name` (`model.safetensors.index.json`'s weight map; a bare file name only).
fn shardPath(a: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![:0]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(8 << 20)) catch return error.MimoEmbeddingRowsIndex;
    defer a.free(raw);
    const index = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return error.MimoEmbeddingRowsIndex;
    defer index.deinit();
    if (index.value != .object) return error.MimoEmbeddingRowsIndex;
    const wm = index.value.object.get("weight_map") orelse return error.MimoEmbeddingRowsIndex;
    if (wm != .object) return error.MimoEmbeddingRowsIndex;
    const shard = wm.object.get(tensor_name) orelse return error.MimoEmbeddingRowsIndex;
    if (shard != .string or !std.mem.endsWith(u8, shard.string, ".safetensors") or std.mem.indexOfAny(u8, shard.string, "/\\\x00") != null)
        return error.MimoEmbeddingRowsIndex;
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ model_dir, shard.string }, 0);
}

const testing = std.testing;

/// A tiny pack: `model.safetensors` with a BF16 [rows, dim] table (row r's bytes a function of r) and another tensor
/// before it, and its index.
fn writeTinyPack(dir: std.Io.Dir, rows: u32, dim: u32) !void {
    const io = testing.io;
    const a = testing.allocator;
    const pad: u64 = 3 * 2; // a 3-element BF16 tensor ahead of the table: the table's offset is not page-aligned
    const table: u64 = @as(u64, rows) * dim * 2;
    const header = try std.fmt.allocPrint(a, "{{\"model.norm.weight\":{{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[0,{d}]}},\"" ++ tensor_name ++ "\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ pad, rows, dim, pad, pad + table });
    defer a.free(header);
    const file = try a.alloc(u8, 8 + header.len + pad + table);
    defer a.free(file);
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8..][0..header.len], header);
    const data = file[8 + header.len ..];
    @memset(data[0..pad], 0xab);
    for (0..rows) |r| for (0..dim) |k| {
        const v: u16 = @truncate(r *% 2654435761 +% k * 40503 +% 17);
        std.mem.writeInt(u16, data[pad + (r * dim + k) * 2 ..][0..2], v, .little);
    };
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.norm.weight\":\"model.safetensors\",\"" ++ tensor_name ++ "\":\"model.safetensors\"}}" });
}

test "mimo rows audit R1: the embedding rows are the table's bytes (any order, repeats, first and last rows, a prompt-width gather) and the geometry is refused by name" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const rows: u32 = 4096;
    const dim: u32 = 64;
    try writeTinyPack(tmp.dir, rows, dim);
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    var c: mimo.Config = .{};
    c.vocab_size = rows;
    c.hidden_size = dim;
    const r = try Rows.open(a, io, path, c);
    defer r.close(a);
    // The table as written, read back whole through an independent read of the file.
    const file = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .unlimited);
    defer a.free(file);
    const hlen: usize = @intCast(std.mem.readInt(u64, file[0..8], .little));
    const table = file[8 + hlen + 6 ..][0 .. @as(usize, rows) * dim * 2];
    // Decode widths (the caller alone) and a prompt width (the helpers), every id kind.
    var ids: [2048]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast((i * 2654435761 + 7) % rows);
    ids[0..6].* = .{ 0, rows - 1, 5, 5, rows - 1, 0 };
    for ([_]usize{ 1, 4, 7, 31, 32, 131, 2048 }) |n| {
        const out = try a.alloc(u8, n * dim * 2);
        defer a.free(out);
        try r.gather(ids[0..n], out);
        for (ids[0..n], 0..) |id, i| try testing.expectEqualSlices(u8, table[@as(usize, id) * dim * 2 ..][0 .. dim * 2], out[i * dim * 2 ..][0 .. dim * 2]);
    }
    // The decode-width cache: lookups of overlapping ids, repeats within a call, more distinct rows than the cache holds
    // (every slot recycled several times) and a prompt-width call (bypassing it) all return the table's bytes; a looked-up
    // row is cached until the clock recycles it.
    var rng: std.Random.DefaultPrng = .init(5);
    var call: [cache_max_ids + 1]u32 = undefined;
    const out = try a.alloc(u8, call.len * dim * 2);
    defer a.free(out);
    for (0..400) |round| {
        const n: usize = if (round % 97 == 96) cache_max_ids + 1 else 1 + rng.random().uintLessThan(usize, 7);
        for (call[0..n]) |*x| x.* = if (rng.random().boolean()) rng.random().uintLessThan(u32, 64) else rng.random().uintLessThan(u32, rows);
        if (n > 2) call[n - 1] = call[0];
        try r.gatherCached(a, call[0..n], out[0 .. n * dim * 2]);
        for (call[0..n], 0..) |id, i| try testing.expectEqualSlices(u8, table[@as(usize, id) * dim * 2 ..][0 .. dim * 2], out[i * dim * 2 ..][0 .. dim * 2]);
        if (n <= cache_max_ids) for (call[0..n]) |id| try testing.expect(r.where[id] != 0);
    }
    var cached: u32 = 0;
    for (r.slot_token, 0..) |id, slot| if (id != no_token) {
        cached += 1;
        try testing.expectEqual(@as(u32, @intCast(slot + 1)), r.where[id]);
        try testing.expectEqualSlices(u8, table[@as(usize, id) * dim * 2 ..][0 .. dim * 2], r.cache[slot * dim * 2 ..][0 .. dim * 2]);
    };
    try testing.expect(cached > cache_rows / 2);
    // An id past the table, the wrong geometry: refused by name.
    var one: [dim * 2]u8 = undefined;
    try testing.expectError(error.NgramTableRegion, r.gather(&.{rows}, &one));
    try testing.expectError(error.NgramTableRegion, r.gatherCached(a, &.{rows}, &one));
    var wrong = c;
    wrong.hidden_size = dim * 2;
    try testing.expectError(error.MimoEmbeddingRowsGeometry, Rows.open(a, io, path, wrong));
    wrong = c;
    wrong.vocab_size = rows + 1;
    try testing.expectError(error.MimoEmbeddingRowsGeometry, Rows.open(a, io, path, wrong));
}

// MIMO_PACK=<prepared pack> (CPU, F_NOCACHE reads, no MLX): the real pack's table rows through the route equal the
// tensor's bytes read independently at their header offsets (the shard, the header and the row arithmetic proven on the
// file the served table route loads).
test "mimo rows audit R1: the real pack's embedding rows equal the tensor's bytes at its header offsets (pack)" {
    const dir = std.mem.span(std.c.getenv("MIMO_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const config_text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(config_text);
    const c = try mimo.Config.parse(a, config_text, null);
    const r = try Rows.open(a, io, dir, c);
    defer r.close(a);
    const path = try shardPath(a, io, dir);
    defer a.free(path);
    const fd = try io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    var len_bytes: [8]u8 = undefined;
    try io_util.readAligned(fd, &len_bytes, 0);
    const hlen: u64 = std.mem.readInt(u64, &len_bytes, .little);
    const header = try a.alloc(u8, @intCast(hlen));
    defer a.free(header);
    try io_util.readAligned(fd, header, 8);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    const region = parsed.value.object.get(tensor_name).?.object.get("data_offsets").?.array.items;
    const base = 8 + hlen + @as(u64, @intCast(region[0].integer));
    const row_bytes: usize = @as(usize, c.hidden_size) * 2;
    try testing.expectEqual(tableBytes(c), @as(u64, @intCast(region[1].integer - region[0].integer)));
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast((i * 40503 + 17) % c.vocab_size);
    ids[0..4].* = .{ 0, 1, c.vocab_size - 1, c.vocab_size / 2 };
    const out = try a.alloc(u8, ids.len * row_bytes);
    defer a.free(out);
    try r.gather(&ids, out);
    // Independent: each row read on its own at its header offset (`io_util.readAligned`, past the cache).
    const want = try a.alloc(u8, row_bytes);
    defer a.free(want);
    for (ids, 0..) |id, i| {
        try io_util.readAligned(fd, want, base + @as(u64, id) * row_bytes);
        try testing.expectEqualSlices(u8, want, out[i * row_bytes ..][0..row_bytes]);
    }
    std.debug.print("MIMO_EMBED_ROWS pack={s} shard={s} rows={d} dim={d} table_bytes={d} checked_ids={d} equal=yes\n", .{ dir, path, c.vocab_size, c.hidden_size, tableBytes(c), ids.len });
}
