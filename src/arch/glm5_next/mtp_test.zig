//! The MTP draft lane's CPU tests (no MLX array is made: each ends on `sdk.testing.expectNoDevice` outside a window):
//! the draft bank through the routed bank's loader and reader (a synthetic bank always; the pack's when `GLM5_BANK`
//! names it, record 0 against the reference reader's check-mtp vector, `testdata/mtp-vector-e000.json`), the MTP
//! layer's residents from the shards' headers, the exact rule's rounds on synthetic targets, and the bill's terms. The
//! lane's device code is compiled here (its declarations referenced); it runs in the arch's window cells (MTP.md).

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../../sdk_ext.zig");
const acc = sdk_ext.acceptance;
const glm = @import("config.zig");
const bank = @import("bank.zig");
const graph = @import("graph.zig");
const mtp = @import("mtp.zig");
const settings = @import("settings.zig");
const mimo_draft = @import("../mimo_v2/draft.zig");
const testing = std.testing;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Component = bank.Component;

/// The projections' components: (weight, block scales, global scale).
const projections = [3][3]Component{
    .{ .gate_weight, .gate_scales, .gate_global_scale },
    .{ .up_weight, .up_scales, .up_global_scale },
    .{ .down_weight, .down_scales, .down_global_scale },
};

fn noDevice() !void {
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) try sdk.testing.expectNoDevice();
}

/// A small model whose MTP layer is layer 45 (the release's index) at toy projections.
fn toyConfig() glm.Config {
    return .{ .hidden_size = 64, .moe_intermediate_size = 32, .n_routed_experts = 3, .n_layers = 45, .n_heads = 2, .head_dim = 16, .q_lora_rank = 32, .kv_lora_rank = 16, .index_n_heads = 2, .index_head_dim = 16, .index_kpool = 4, .vocab_size = 128 };
}

fn dirPath(io: std.Io, dir: std.Io.Dir, buf: []u8) ![]const u8 {
    return buf[0..try dir.realPath(io, buf)];
}

/// A draft bank in the pack's form (`bank.mtp_files`) for `c`'s MTP layer: every E2M1 code and every E4M3 byte but
/// the two NaNs from a seeded generator, positive global scales, the padding zero. `data` is the whole data file.
const Synth = struct {
    geom: bank.Geometry,
    data: []u8,

    fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, seed: u64, layout: []const u8) !Synth {
        const g = try bank.Geometry.init(c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.n_layers, 1);
        const data = try a.alloc(u8, @intCast(g.total_bytes));
        errdefer a.free(data);
        @memset(data, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        for (0..g.num_experts) |i| {
            const rec = data[i * g.record_bytes ..][0..@intCast(g.logical_bytes)];
            for (projections) |p| {
                const w = g.segments[@backingInt(p[0])];
                rand.bytes(rec[@intCast(w.offset)..][0..@intCast(w.length)]);
                const s = g.segments[@backingInt(p[1])];
                for (rec[@intCast(s.offset)..][0..@intCast(s.length)]) |*b| {
                    b.* = rand.int(u8);
                    if (b.* & 0x7f == 0x7f) b.* -= 1;
                }
                const gs = g.segments[@backingInt(p[2])];
                const scale: f32 = 0.01 + rand.float(f32) * 4.0;
                std.mem.writeInt(u32, rec[@intCast(gs.offset)..][0..4], @bitCast(scale), .little);
            }
        }
        const digests = try a.alloc([32]u8, g.num_experts);
        defer a.free(digests);
        for (digests, 0..) |*d, i| Sha256.hash(data[i * g.record_bytes ..][0..@intCast(g.logical_bytes)], d, .{});
        var whole: [32]u8 = undefined;
        Sha256.hash(data, &whole, .{});
        const text = try manifestText(a, &g, layout, digests, whole);
        defer a.free(text);
        try dir.writeFile(io, .{ .sub_path = bank.mtp_files.data, .data = data });
        try dir.writeFile(io, .{ .sub_path = bank.mtp_files.manifest, .data = text });
        return .{ .geom = g, .data = data };
    }

    fn deinit(self: *Synth, a: std.mem.Allocator) void {
        a.free(self.data);
        self.* = undefined;
    }

    fn record(self: *const Synth, expert: u32) []const u8 {
        return self.data[@as(usize, expert) * @as(usize, @intCast(self.geom.record_bytes)) ..][0..@intCast(self.geom.logical_bytes)];
    }
};

/// A draft bank's manifest in the pack's form for `g` (`layout` its record layout), record i's sha256 `digests[i]`, the
/// data file's `whole` (opening never hashes the bank: only its form is checked).
fn manifestText(a: std.mem.Allocator, g: *const bank.Geometry, layout: []const u8, digests: []const [32]u8, whole: [32]u8) ![]u8 {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(a);
    try text.print(a, "{{\"format\":\"{s}\",\"data_file\":\"{s}\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_experts\":{d},\"first_moe_layer\":{d},\"num_moe_layers\":1,", .{ bank.format, bank.mtp_files.data, g.hidden_size, g.intermediate_size, g.num_experts, g.first_moe_layer });
    try text.print(a, "\"layout\":\"{s}\",\"alignment\":{d},\"record_bytes\":{d},\"record_stride\":{d},\"gate_up_bytes\":{d},\"down_bytes\":{d},\"quantization\":{{\"mode\":\"nvfp4\",\"bits\":4,\"group_size\":16}},", .{ layout, bank.page, g.logical_bytes, g.record_bytes, g.gate_up_bytes, g.logical_bytes - g.gate_up_bytes });
    try text.appendSlice(a, "\"source\":{\"repository\":\"synthetic\",\"revision\":\"0123456789abcdef0123456789abcdef01234567\"},\"parts\":[");
    for (g.segments, 0..) |s, ci| {
        try text.print(a, "{s}{{\"name\":\"{s}\",\"dtype\":\"{t}\",\"shape\":[", .{ if (ci == 0) "" else ",", @as(Component, @fromBackingInt(@intCast(ci))).name(), s.dtype });
        for (s.shape[0..s.rank], 0..) |d, k| try text.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try text.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ s.offset, s.length });
    }
    try text.appendSlice(a, "],\"records\":[");
    for (digests, 0..) |d, i| try text.print(a, "{s}{{\"layer\":{d},\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{x}\"}}", .{ if (i == 0) "" else ",", g.first_moe_layer, i, i * g.record_bytes, g.logical_bytes, d });
    try text.print(a, "],\"output_files\":{{\"{s}\":{{\"size\":{d},\"sha256\":\"{x}\"}}}}}}", .{ bank.mtp_files.data, g.total_bytes, whole });
    return text.toOwnedSlice(a);
}

/// A draft bank in the pack's form at `g` whose data file is sparse (sized by `ftruncate`: a record reads as zeros) but
/// for the records `written`, each filled from a seeded generator at its place in the file. Returns their bytes, in
/// `written`'s order (the caller frees each and the list).
fn writeSparse(a: std.mem.Allocator, io: std.Io, tmp: std.Io.Dir, dir: []const u8, g: *const bank.Geometry, written: []const u32, seed: u64) ![][]u8 {
    const rec_len: usize = @intCast(g.logical_bytes);
    const zeros = try a.alloc(u8, rec_len);
    defer a.free(zeros);
    @memset(zeros, 0);
    var zero_digest: [32]u8 = undefined;
    Sha256.hash(zeros, &zero_digest, .{});
    const digests = try a.alloc([32]u8, g.num_experts);
    defer a.free(digests);
    @memset(digests, zero_digest);
    const recs = try a.alloc([]u8, written.len);
    var n_recs: usize = 0;
    errdefer {
        for (recs[0..n_recs]) |r| a.free(r);
        a.free(recs);
    }
    var prng = std.Random.DefaultPrng.init(seed);
    for (written) |e| {
        recs[n_recs] = try a.alloc(u8, rec_len);
        n_recs += 1;
        prng.random().bytes(recs[n_recs - 1]);
        Sha256.hash(recs[n_recs - 1], &digests[e], .{});
    }
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, bank.mtp_files.data }, 0);
    defer a.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.TestUnexpectedResult;
    defer _ = std.c.close(fd);
    if (std.c.ftruncate(fd, @intCast(g.total_bytes)) != 0) return error.TestUnexpectedResult;
    for (written, recs) |e, r| {
        var done: usize = 0;
        while (done < r.len) {
            const n = std.c.pwrite(fd, r[done..].ptr, r.len - done, @intCast(@as(u64, e) * g.record_bytes + done));
            if (n <= 0) return error.TestUnexpectedResult;
            done += @intCast(n);
        }
    }
    const text = try manifestText(a, g, bank.layout, digests, @splat(0));
    defer a.free(text);
    try tmp.writeFile(io, .{ .sub_path = bank.mtp_files.manifest, .data = text });
    return recs;
}

/// The MTP layer's residents of `c` as one shard (`model-00001-of-00001.safetensors`, zero payload) and its index; `skip`
/// leaves that spec entry out, `bad_shape` gives it one more row.
fn writeResidents(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, skip: ?usize, bad_shape: ?usize) !u64 {
    var pb: [64]u8 = undefined;
    const pre = mtp.prefixOf(&pb, c);
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try header.appendSlice(a, "{\"__metadata__\":{\"format\":\"pt\"}");
    try index.appendSlice(a, "{\"metadata\":{},\"weight_map\":{\"model.language_model.norm.weight\":\"model-00001-of-00001.safetensors\"");
    var off: u64 = 0;
    for (mtp.residentSpec(c), 0..) |r, i| {
        if (skip == i) continue;
        var shape = r.shape;
        if (bad_shape == i) shape[0] += 1;
        var elems: u64 = 1;
        for (shape[0..r.rank]) |x| elems *= x;
        const bytes = elems * @as(u64, if (r.dtype == .F32) 4 else 2);
        try header.print(a, ",\"{s}{s}\":{{\"dtype\":\"{t}\",\"shape\":[", .{ pre, r.suffix, r.dtype });
        for (shape[0..r.rank], 0..) |d, k| try header.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + bytes });
        try index.print(a, ",\"{s}{s}\":\"model-00001-of-00001.safetensors\"", .{ pre, r.suffix });
        off += bytes;
    }
    try header.append(a, '}');
    try index.appendSlice(a, "}}");
    const file = try a.alloc(u8, 8 + header.items.len + off);
    defer a.free(file);
    @memset(file, 0);
    std.mem.writeInt(u64, file[0..8], header.items.len, .little);
    @memcpy(file[8..][0..header.items.len], header.items);
    try dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    return off;
}

test "glm5_next mtp bank: a synthetic draft bank opens through the routed bank's loader, reads through its reader into host slot rows (every record's sha256 checked) as written; absent, off its form, off the model, short or corrupt, it is refused by name" {
    const a = testing.allocator;
    const io = testing.io;
    const c = toyConfig();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try dirPath(io, tmp.dir, &buf);
    var diag: sdk.Diag = .{};
    // No bank beside the pack: refused by name, the file named.
    try testing.expectError(error.Glm5NextMtpBankMissing, mtp.DraftBank.open(a, io, dir, &c, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), bank.mtp_files.manifest) != null);
    var synth = try Synth.write(a, io, tmp.dir, &c, 0x5eed, bank.layout);
    defer synth.deinit(a);
    {
        var db = try mtp.DraftBank.open(a, io, dir, &c, &diag);
        defer db.deinit();
        try testing.expectEqual(@as(u32, 45), db.owner.geometry.first_moe_layer);
        var rows = try mtp.Slots.HostSlotRows.init(db.layer(), 3);
        defer rows.deinit();
        // Every record, in another order than the file's: each lands in its row as the writer laid it out.
        try db.read(a, &rows, &.{ 2, 0, 1 }, &.{ 0, 1, 2 }, &diag);
        for ([_]u32{ 2, 0, 1 }, 0..) |e, r| {
            const rec = synth.record(e);
            for (synth.geom.segments, 0..) |s, ci| try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], rows.row(@fromBackingInt(@intCast(ci)), @intCast(r))[0..@intCast(s.length)]);
        }
        try testing.expectEqual(synth.geom.total_bytes, db.view.sidecar.size);
    }
    // Off the model: the bank is not this model's MTP layer.
    var other = c;
    other.n_layers = 44;
    try testing.expectError(error.Glm5NextMtpGeometry, mtp.DraftBank.open(a, io, dir, &other, &diag));
    other = c;
    other.moe_intermediate_size = 64;
    try testing.expectError(error.Glm5NextMtpGeometry, mtp.DraftBank.open(a, io, dir, &other, &diag));
    // A corrupt record: refused by its digest after the reads.
    {
        const bad = try a.dupe(u8, synth.data);
        defer a.free(bad);
        bad[@intCast(synth.geom.record_bytes + 17)] ^= 0x40;
        try tmp.dir.writeFile(io, .{ .sub_path = bank.mtp_files.data, .data = bad });
        var db = try mtp.DraftBank.open(a, io, dir, &c, &diag);
        defer db.deinit();
        var rows = try mtp.Slots.HostSlotRows.init(db.layer(), 3);
        defer rows.deinit();
        try testing.expectError(error.Glm5NextMtpBankDigest, db.read(a, &rows, &.{ 0, 1, 2 }, &.{ 0, 1, 2 }, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "expert 1") != null);
    }
    // A short data file, then a manifest off the routed bank's form.
    try tmp.dir.writeFile(io, .{ .sub_path = bank.mtp_files.data, .data = synth.data[0 .. synth.data.len - 1] });
    try testing.expectError(error.Glm5NextMtpBankFile, mtp.DraftBank.open(a, io, dir, &c, &diag));
    var off_form = try Synth.write(a, io, tmp.dir, &c, 7, "nvfp4-gud-3part");
    defer off_form.deinit(a);
    try testing.expectError(error.Glm5NextMtpBankManifest, mtp.DraftBank.open(a, io, dir, &c, &diag));
    try noDevice();
}

test "glm5_next mtp bank: a read of more records than one submit takes (17: submits of 8, 8 and 1) lands every record in its row, byte for byte" {
    // The lane's open reads all 288 records: every submit but the last is full (8 records, 16 tickets). A full submit's
    // ticket count once overflowed the type `@min` narrowed it to (u4) and the read waited on no ticket.
    const a = testing.allocator;
    const io = testing.io;
    var c = toyConfig();
    c.n_routed_experts = 17;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try dirPath(io, tmp.dir, &buf);
    var synth = try Synth.write(a, io, tmp.dir, &c, 0x17, bank.layout);
    defer synth.deinit(a);
    var diag: sdk.Diag = .{};
    var db = try mtp.DraftBank.open(a, io, dir, &c, &diag);
    defer db.deinit();
    var rows = try mtp.Slots.HostSlotRows.init(db.layer(), 17);
    defer rows.deinit();
    var ids: [17]u32 = undefined;
    var at: [17]u32 = undefined;
    for (&ids, &at, 0..) |*e, *r, i| {
        e.* = @intCast((i * 5) % 17);
        r.* = @intCast(i);
    }
    try db.read(a, &rows, &ids, &at, &diag);
    for (ids, at) |e, r| {
        const rec = synth.record(e);
        for (synth.geom.segments, 0..) |s, ci| try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], rows.row(@fromBackingInt(@intCast(ci)), r)[0..@intCast(s.length)]);
    }
    try noDevice();
}

test "glm5_next mtp bank: one read spanning past 2 GiB and 4 GiB of the data file (a sparse bank) lands every record at its 64-bit offset, byte for byte, in submits of 8" {
    // The reader preads each range in staging-sized steps at 64-bit offsets: no single read spans the bank, none takes an
    // offset past INT_MAX or UINT32_MAX wrong. 1,300 records of 3,543,040 B: 4,605,952,000 B, sparse but for four.
    const a = testing.allocator;
    const io = testing.io;
    var c = toyConfig();
    c.hidden_size = 2048;
    c.moe_intermediate_size = 1024;
    c.n_routed_experts = 1300;
    const g = try bank.Geometry.init(c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.n_layers, 1);
    try testing.expect(g.total_bytes > 1 << 32 and 607 * g.record_bytes > 1 << 31 and 1213 * g.record_bytes > 1 << 32);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try dirPath(io, tmp.dir, &buf);
    const written = [_]u32{ 0, 607, 1213, 1299 };
    const recs = try writeSparse(a, io, tmp.dir, dir, &g, &written, 0x2617);
    defer {
        for (recs) |r| a.free(r);
        a.free(recs);
    }
    var diag: sdk.Diag = .{};
    var db = mtp.DraftBank.open(a, io, dir, &c, &diag) catch |e| {
        std.debug.print("{s}\n", .{diag.message()});
        return e;
    };
    defer db.deinit();
    try testing.expectEqual(g.total_bytes, db.view.sidecar.size);
    // Nine records (a full submit, then one) across the file, the written ones among them; every digest checked.
    const ids = [_]u32{ 1299, 0, 150, 300, 607, 900, 1100, 1213, 450 };
    const at = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    var rows = try mtp.Slots.HostSlotRows.init(db.layer(), ids.len);
    defer rows.deinit();
    db.read(a, &rows, &ids, &at, &diag) catch |e| {
        std.debug.print("{s}\n", .{diag.message()});
        return e;
    };
    for (written, recs) |e, rec| {
        const r = std.mem.indexOfScalar(u32, &ids, e).?;
        for (g.segments, 0..) |s, ci| try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], rows.row(@fromBackingInt(@intCast(ci)), @intCast(r))[0..@intCast(s.length)]);
    }
    try noDevice();
}

test "glm5_next mtp bank: a data file that does not open read only past the page cache (unreadable, a symlink) is refused by name with its errno, never read" {
    const a = testing.allocator;
    const io = testing.io;
    const c = toyConfig();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try dirPath(io, tmp.dir, &buf);
    var synth = try Synth.write(a, io, tmp.dir, &c, 0x0ef, bank.layout);
    defer synth.deinit(a);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, bank.mtp_files.data }, 0);
    defer a.free(path);
    var diag: sdk.Diag = .{};
    // Unreadable (mode 0): the uncached open fails (EACCES); the refusal names the file, the open's error and errno.
    try testing.expectEqual(@as(c_int, 0), std.c.chmod(path.ptr, 0));
    if (std.c.access(path.ptr, std.c.R_OK) != 0) {
        try testing.expectError(error.Glm5NextMtpBankFile, mtp.DraftBank.open(a, io, dir, &c, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), bank.mtp_files.data) != null);
        try testing.expect(std.mem.indexOf(u8, diag.message(), "OpenFailed (errno 13)") != null);
    }
    try testing.expectEqual(@as(c_int, 0), std.c.chmod(path.ptr, 0o644));
    // A symlink to the bank: the reader never follows one (ELOOP), as the routed bank's experts.bin.
    const real = try std.fmt.allocPrintSentinel(a, "{s}/real.bin", .{dir}, 0);
    defer a.free(real);
    try testing.expectEqual(@as(c_int, 0), std.c.rename(path.ptr, real.ptr));
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(real.ptr, path.ptr));
    try testing.expectError(error.Glm5NextMtpBankFile, mtp.DraftBank.open(a, io, dir, &c, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "OpenFailed (errno 62)") != null);
    try noDevice();
}

/// The vector's file (the routed vectors' reference.json shape plus the record's sha256).
const Vector = struct {
    records: []const struct {
        layer: u32,
        expert: u32,
        record_sha256: []const u8,
        gate: Sums,
        up: Sums,
        down: Sums,
    },
    const Sums = struct { f32_sha256: []const u8, fp16_sha256: []const u8 };
};

fn hexEq(hex: []const u8, d: *const [32]u8) bool {
    var got: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&got, hex) catch return false;
    return hex.len == 64 and std.mem.eql(u8, &got, d);
}

/// One projection of row `r` decoded by the routed bank's host decoder (`bank.dequantRow`), as f32 and f16 sha256.
fn projectionSums(a: std.mem.Allocator, rows: *const mtp.Slots.HostSlotRows, r: u32, g: *const bank.Geometry, p: [3]Component) !struct { f32: [32]u8, f16: [32]u8, n: usize } {
    const w = g.segments[@backingInt(p[0])];
    const out_rows: usize = @intCast(w.shape[0]);
    const words_per_row: usize = @intCast(w.shape[1]);
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(rows.row(p[0], r))));
    const scales = rows.row(p[1], r);
    const scale_g: f32 = @bitCast(std.mem.readInt(u32, rows.row(p[2], r)[0..4], .little));
    const row = try a.alloc(f32, words_per_row * 8);
    defer a.free(row);
    var h32 = Sha256.init(.{});
    var h16 = Sha256.init(.{});
    for (0..out_rows) |o| {
        bank.dequantRow(words[o * words_per_row ..][0..words_per_row], scales[o * words_per_row / 2 ..][0 .. words_per_row / 2], scale_g, row);
        h32.update(std.mem.sliceAsBytes(row));
        for (row) |v| {
            var b16: [2]u8 = undefined;
            std.mem.writeInt(u16, &b16, @bitCast(bank.toHalf(v)), .little);
            h16.update(&b16);
        }
    }
    return .{ .f32 = h32.finalResult(), .f16 = h16.finalResult(), .n = out_rows * words_per_row * 8 };
}

fn packConfig(a: std.mem.Allocator, io: std.Io, dir: []const u8) !glm.Config {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    var diag: glm.Diag = .{};
    return glm.Config.parse(a, text, &diag);
}

test "glm5_next mtp bank: the pack's draft bank (GLM5_BANK) reads records 0, 143 and 287 through the routed bank's reader at their sha256, and record 0 decodes to the reference reader's values (check-mtp vector: f32 and fp16 sha256 of gate, up and down)" {
    const dir = std.mem.span(std.c.getenv("GLM5_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const c = try packConfig(a, io, dir);
    var diag: sdk.Diag = .{};
    var db = mtp.DraftBank.open(a, io, dir, &c, &diag) catch |e| {
        std.debug.print("{s}\n", .{diag.message()});
        return e;
    };
    defer db.deinit();
    const parsed = try std.json.parseFromSlice(Vector, a, @embedFile("testdata/mtp-vector-e000.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const v = parsed.value.records[0];
    try testing.expectEqual(c.n_layers, v.layer);
    try testing.expect(hexEq(v.record_sha256, &db.owner.digests[v.expert]));
    var rows = try mtp.Slots.HostSlotRows.init(db.layer(), 3);
    defer rows.deinit();
    try db.read(a, &rows, &.{ v.expert, 143, 287 }, &.{ 0, 1, 2 }, &diag);
    var decoded: usize = 0;
    for (projections, [_]Vector.Sums{ v.gate, v.up, v.down }, [_][]const u8{ "gate", "up", "down" }) |p, want, which| {
        const got = try projectionSums(a, &rows, 0, &db.owner.geometry, p);
        testing.expect(hexEq(want.f32_sha256, &got.f32) and hexEq(want.fp16_sha256, &got.f16)) catch |e| {
            std.debug.print("expert {d} {s}: f32 {x} fp16 {x}\n", .{ v.expert, which, got.f32, got.f16 });
            return e;
        };
        decoded += got.n;
    }
    try testing.expectEqual(@as(usize, 3 * 2048 * 4096), decoded);
    std.debug.print("GLM5N_MTP_BANK records=3 sha256=ok vector=expert {d} f32+fp16 sha256 equal (gate, up, down) arrays={d} B\n", .{ v.expert, mtp.DraftBank.arrayBytes(&db.owner.geometry, 16_384) });
    try noDevice();
}

test "glm5_next mtp residents: the MTP layer's 25 tensors from the shards' headers at the served shapes; one missing or off its shape is refused by name" {
    const a = testing.allocator;
    const io = testing.io;
    const c = toyConfig();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try dirPath(io, tmp.dir, &buf);
    var diag: sdk.Diag = .{};
    const want = try writeResidents(a, io, tmp.dir, &c, null, null);
    const page: u64 = 16_384;
    const r = try mtp.residents(a, io, dir, &c, page, &diag);
    try testing.expectEqual(want, r.bytes);
    try testing.expect(r.buffer_bytes >= r.bytes and r.buffer_bytes < r.bytes + mtp.n_residents * page);
    var pb: [64]u8 = undefined;
    const pre = mtp.prefixOf(&pb, &c);
    try testing.expectEqualStrings("model.language_model.layers.45.", pre);
    try testing.expect(mtp.isResident(&c, "model.language_model.layers.45.self_attn.indexer.index_kpool_compress_ape"));
    try testing.expect(!mtp.isResident(&c, "model.language_model.layers.44.eh_proj.weight") and !mtp.isResident(&c, "model.language_model.layers.45.mlp.experts.0.up_proj.weight"));
    _ = try writeResidents(a, io, tmp.dir, &c, 13, null);
    try testing.expectError(error.Glm5NextMtpResidentMissing, mtp.residents(a, io, dir, &c, page, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q_b_proj") != null);
    _ = try writeResidents(a, io, tmp.dir, &c, null, 16);
    try testing.expectError(error.Glm5NextMtpResidentShape, mtp.residents(a, io, dir, &c, page, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "kv_b_proj") != null);
    // The pack's (GLM5_BANK): 369,670,784 B (docs/glm5-next-pack.md, the MTP group).
    if (std.c.getenv("GLM5_BANK")) |p| {
        const pc = try packConfig(a, io, std.mem.span(p));
        const pr = try mtp.residents(a, io, std.mem.span(p), &pc, page, &diag);
        try testing.expectEqual(@as(u64, 369_670_784), pr.bytes);
        std.debug.print("GLM5N_MTP_RESIDENTS tensors={d} bytes={d} buffer_bytes={d}\n", .{ mtp.n_residents, pr.bytes, pr.buffer_bytes });
    }
    try noDevice();
}

/// A toy target: the logits of the token at `pos` after `prev` (seeded, a leader per row).
fn targetRow(prev: u32, pos: u64, out: []f32) void {
    var rng = std.Random.DefaultPrng.init(0x9e37_79b9_7f4a_7c15 ^ (@as(u64, prev) *% 0x1_0000_01b3) ^ (pos *% 0x5851_f42d_4c95_7f2d));
    for (out) |*x| x.* = rng.random().floatNorm(f32);
    out[rng.random().uintLessThan(usize, out.len)] += 2.5;
}

fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return @intCast(best);
}

test "glm5_next mtp acceptance: the exact rule at depth 1..3 emits the target's greedy tokens bit for bit whatever was drafted, rounds stay inside the verify rows; a sampled request's next token follows its target; the arms are greedy, stochastic or off" {
    const V = 64;
    // Greedy: rounds over the toy target with a drafter right about 70 % of the time, against the serial argmax chain.
    for ([_]u32{ 1, 2, 3 }) |cap| for (0..8) |seed| {
        const n = 300;
        var serial: [n + 1]u32 = undefined;
        serial[0] = @intCast(seed % V);
        var row: [V]f32 = undefined;
        for (1..n + 1) |i| {
            targetRow(serial[i - 1], i, &row);
            serial[i] = argmax(&row);
        }
        var drafter = std.Random.DefaultPrng.init(0xd4af7 + seed);
        var emitted: [n + 8]u32 = undefined;
        emitted[0] = serial[0];
        var len: u64 = 1; // the trunk holds position 0; t1 (position 1) is the prompt's argmax
        var t1 = serial[1];
        var rounds: u32 = 0;
        var accepted_total: u64 = 0;
        while (len < n - 4) {
            const k = mtp.draftsAt(cap, len, 1, len, 1000, 1 << 20);
            try testing.expectEqual(cap, k);
            var drafts: [mtp.max_depth]u32 = undefined;
            var logits: [(mtp.max_depth + 1) * V]f32 = undefined;
            var prev = t1;
            targetRow(t1, len + 1, logits[0..V]);
            for (0..k) |i| {
                const right = argmax(logits[i * V ..][0..V]);
                drafts[i] = if (drafter.random().float(f32) < 0.7) right else drafter.random().uintLessThan(u32, V);
                prev = drafts[i];
                targetRow(prev, len + 2 + i, logits[(i + 1) * V ..][0..V]);
            }
            const r = acc.accept(mtp.rule, null, logits[0 .. (k + 1) * V], drafts[0..k], null, len);
            const plan = mimo_draft.commitOf(k, .{ .accepted = r.n_accepted, .next = r.next_token });
            try testing.expect(plan.emitted <= mtp.verify_rows and plan.trim == k - r.n_accepted);
            emitted[len] = t1;
            for (drafts[0..r.n_accepted], 0..) |d, i| emitted[len + 1 + i] = d;
            len += plan.emitted;
            t1 = plan.next;
            rounds += 1;
            accepted_total += r.n_accepted;
        }
        emitted[len] = t1;
        try testing.expectEqualSlices(u32, serial[0 .. len + 1], emitted[0 .. len + 1]);
        try testing.expect(rounds < len); // the drafts were accepted at times: fewer rounds than tokens
        if (seed == 0) std.debug.print("GLM5N_MTP_EXACT depth={d} tokens={d} rounds={d} accepted={d} tokens_per_round={d:.3}\n", .{ cap, len, rounds, accepted_total, @as(f64, @floatFromInt(len - 1)) / @as(f64, @floatFromInt(rounds)) });
    };
    // Untracked (a lane behind the trunk) or no room: no drafts; the lane's depth, the host's cap and the billed
    // positions bound the rest.
    try testing.expectEqual(@as(u32, 0), mtp.draftsAt(3, 9, 1, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 0), mtp.draftsAt(3, 10, 0, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 2), mtp.draftsAt(3, 10, 1, 10, 9, 13));
    try testing.expectEqual(@as(u32, 1), mtp.draftsAt(3, 10, 1, 10, 1, 1000));
    try testing.expectEqual(@as(u32, 3), mtp.draftsAt(3, 10, 1, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 2), mtp.draftsAt(2, 10, 1, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 1), mtp.draftsAt(1, 10, 1, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 1), mtp.draftsAt(2, 10, 1, 10, 1, 1000));
    // Sampled (temperature 1, top_p 0.95): at one context, the token after t1 over seeded rounds with the lane's
    // point-mass draft (the head's argmax) follows the request's own target p (exact speculative sampling).
    {
        const t1: u32 = 5;
        const pos: u64 = 40;
        var logits: [2 * V]f32 = undefined;
        targetRow(t1, pos + 1, logits[0..V]);
        const d1 = argmax(logits[0..V]);
        targetRow(d1, pos + 2, logits[V..]);
        const rounds = 20_000;
        var counts: [V]u32 = @splat(0);
        var accepted: u32 = 0;
        for (0..rounds) |s| {
            const params: acc.Params = .{ .temperature = 1.0, .top_p = 0.95, .seed = s };
            const r = acc.accept(mtp.rule, params, &logits, &.{d1}, null, pos);
            counts[if (r.n_accepted == 1) d1 else r.next_token] += 1;
            accepted += r.n_accepted;
        }
        var p: [V]f64 = undefined;
        acc.distribution(logits[0..V], .{ .temperature = 1.0, .top_p = 0.95, .seed = 0 }, &p);
        for (counts, p) |cnt, pi| {
            const f = @as(f64, @floatFromInt(cnt)) / rounds;
            try testing.expect(@abs(f - pi) <= 5 * @sqrt(pi * (1 - pi) / rounds) + 1e-9);
        }
        // Accepted with probability p(d1).
        const pa = @as(f64, @floatFromInt(accepted)) / rounds;
        try testing.expect(@abs(pa - p[d1]) <= 5 * @sqrt(p[d1] * (1 - p[d1]) / rounds));
        std.debug.print("GLM5N_MTP_SAMPLED rounds={d} accepted={d:.4} p(d1)={d:.4}\n", .{ rounds, pa, p[d1] });
    }
    try testing.expectEqual(sdk.DraftArm.greedy, mtp.armOf(.{ .greedy = true, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.stochastic, mtp.armOf(.{ .greedy = false, .clean = true }));
    try testing.expectEqual(sdk.DraftArm.off, mtp.armOf(.{ .greedy = true, .clean = false }));
    try testing.expectEqual(acc.Rule.exact, mtp.rule);
}

test "glm5_next mtp bill: the lane's terms by construction (the bank's nine arrays, the residents, the head's f32 tables, its lanes bf16 at the billed positions plus a round's drafts, the pending rows bf16, the waves); the pack's (GLM5_BANK)" {
    const a = testing.allocator;
    const io = testing.io;
    const page: u64 = 16_384;
    const c = toyConfig();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = try dirPath(io, tmp.dir, &buf);
    var diag: sdk.Diag = .{};
    var synth = try Synth.write(a, io, tmp.dir, &c, 11, bank.layout);
    defer synth.deinit(a);
    const res_bytes = try writeResidents(a, io, tmp.dir, &c, null, null);
    const f = try mtp.facts(a, io, dir, &c, page, &diag);
    var arrays: u64 = 0;
    // Line-isolated global scales: each [32, 32] f32 (3 experts floored at a page of 128-byte lines), the rest [3, ...].
    for (synth.geom.segments) |s| arrays += graph.bufferBytes(if (s.rank == 0) bank.scale_array_min_bytes else 3 * s.length, page);
    try testing.expectEqual(arrays, f.bank_bytes);
    try testing.expect(f.resident_bytes >= res_bytes);
    try testing.expectEqual(mtp.Head.tableBytes(&c, page), f.table_bytes);
    const positions: u64 = 17_408;
    const t = mtp.terms(f, &c, positions, page);
    const names = [_][]const u8{ "mtp_bank", "mtp_residents", "mtp_tables", "mtp_kv", "mtp_pending", "mtp_wave" };
    for (t, names) |term, name| try testing.expectEqualStrings(name, term.name);
    try testing.expectEqual([2]u64{ f.bank_bytes, f.bank_bytes }, t[0].bytes);
    // The lanes: latent and indexer rows, bf16, at the positions plus a round's draft rows.
    const lanes = (positions + mtp.max_depth) * (c.kv_lora_rank + 2 * c.index_head_dim) * 2;
    try testing.expect(t[3].bytes[0] >= lanes and t[3].bytes[0] < lanes + 2 * page);
    try testing.expect(t[0].at_construction and t[3].at_construction and !t[4].at_construction and !t[5].at_construction);
    // Pending: the decode's serial rows (bf16, twice for a pend's concat); the prompt's adds a call's hidden rows.
    try testing.expectEqual(2 * graph.bufferBytes(mtp.max_pending * c.hidden_size * 2, page), t[4].bytes[1]);
    try testing.expect(t[4].bytes[0] >= @as(u64, graph.prompt_call_rows) * c.hidden_size * 2);
    // The pack's terms at the standard request's positions.
    if (std.c.getenv("GLM5_BANK")) |p| {
        const pc = try packConfig(a, io, std.mem.span(p));
        const pf = try mtp.facts(a, io, std.mem.span(p), &pc, page, &diag);
        // 3 x (288 x 4,194,304 words + 288 x 524,288 scales + 288 x 4 global scales).
        try testing.expectEqual(@as(u64, 4_076_866_944), pf.bank_bytes);
        const pt = mtp.terms(pf, &pc, positions, page);
        var total: [2]u64 = .{ 0, 0 };
        for (pt) |term| {
            total[0] += term.bytes[0];
            total[1] += term.bytes[1];
            std.debug.print("GLM5N_MTP_TERM {s} prompt={d} decode={d} construction={}\n", .{ term.name, term.bytes[0], term.bytes[1], term.at_construction });
        }
        std.debug.print("GLM5N_MTP_TERMS positions={d} prompt={d} decode={d}\n", .{ positions, total[0], total[1] });
        const n = positions + mtp.max_depth;
        try testing.expectEqual(graph.bufferBytes(n * 512 * 2, page) + graph.bufferBytes(n * 256 * 2, page), pt[3].bytes[0]);
    }
    try noDevice();
}

test "glm5_next mtp positions: a round decides drafts 1 .. min(accepted + 1, drafted) and keeps 1 .. accepted; the receipt's tail names the depth and every position" {
    var p: mtp.Positions = .{};
    // (drafted, accepted): a full accept, a first-draft reject, a reject at 2, a reject at 3, a capped round fully
    // accepted, a round without drafts.
    for ([_][2]u32{ .{ 3, 3 }, .{ 3, 0 }, .{ 3, 1 }, .{ 3, 2 }, .{ 2, 2 }, .{ 0, 0 } }) |r| p.add(r[0], r[1]);
    try testing.expectEqualSlices(u64, &.{ 5, 4, 2 }, &p.tested);
    try testing.expectEqualSlices(u64, &.{ 4, 3, 1 }, &p.kept);
    // Every kept draft is counted once (MiMo's `accepted`), every decided one once.
    var st: mimo_draft.Stats = .{};
    for ([_][2]u32{ .{ 3, 3 }, .{ 3, 0 }, .{ 3, 1 }, .{ 3, 2 }, .{ 2, 2 }, .{ 0, 0 } }) |r| st.add(r[0], r[1]);
    try testing.expectEqual(st.accepted, p.kept[0] + p.kept[1] + p.kept[2]);
    var buf: [96]u8 = undefined;
    try testing.expectEqualStrings(" depth=3 pos_tested=5,4,2 pos_kept=4,3,1", p.write(&buf, 3));
    try testing.expectEqualStrings(" depth=2 pos_tested=0,0,0 pos_kept=0,0,0", (mtp.Positions{}).write(&buf, 2));
    // A buffer too short for the tail: nothing (the receipt keeps MiMo's line).
    try testing.expectEqual(@as(usize, 0), p.write(buf[0..12], 3).len);
    try noDevice();
}

/// A host backend for the draft chain's ops (CPU): every array a host vector of f32 (an id array holds its ids as exact
/// integers), every op computed when built; `evals` / `reads` count the host's waits and reads, and a read of an array
/// no eval named is refused (on MLX it would be one more sync).
const ChainOps = struct {
    pub const T = u32;
    a: std.mem.Allocator,
    vals: std.ArrayList([]f32) = .empty,
    evaluated: std.ArrayList(T) = .empty,
    evals: u32 = 0,
    reads: u32 = 0,

    fn deinit(g: *ChainOps) void {
        for (g.vals.items) |v| g.a.free(v);
        g.vals.deinit(g.a);
        g.evaluated.deinit(g.a);
    }

    fn readable(g: *const ChainOps, x: T) !void {
        if (std.mem.indexOfScalar(T, g.evaluated.items, x) == null) return error.ImplicitSync;
    }

    pub fn max(g: *ChainOps, x: T, axis: c_int, keepdims: bool) !T {
        std.debug.assert(axis == 0 and !keepdims);
        return g.new(&.{std.mem.max(f32, g.vals.items[x])});
    }

    pub fn logsumexp(g: *ChainOps, x: T, axis: c_int, keepdims: bool) !T {
        std.debug.assert(axis == 0 and !keepdims);
        const v = g.vals.items[x];
        const m = std.mem.max(f32, v);
        var s: f32 = 0;
        for (v) |f| s += @exp(f - m);
        return g.new(&.{m + @log(s)});
    }

    pub fn sub(g: *ChainOps, x: T, y: T) !T {
        return g.new(&.{g.vals.items[x][0] - g.vals.items[y][0]});
    }

    pub fn exp(g: *ChainOps, x: T) !T {
        return g.new(&.{@exp(g.vals.items[x][0])});
    }

    pub fn hostF32(g: *ChainOps, x: T, out: []f32) ![]const f32 {
        try g.readable(x);
        g.reads += 1;
        const v = g.vals.items[x];
        if (v.len != out.len) return error.HostReadSize;
        @memcpy(out, v);
        return out;
    }

    fn new(g: *ChainOps, v: []const f32) !T {
        try g.vals.append(g.a, try g.a.dupe(f32, v));
        return @intCast(g.vals.items.len - 1);
    }

    pub fn reshape(g: *ChainOps, x: T, _: []const c_int) !T {
        return g.new(g.vals.items[x]);
    }

    /// The first maximum (MLX's argmax), as an id.
    pub fn argmax(g: *ChainOps, x: T, axis: c_int) !T {
        std.debug.assert(axis == 0);
        return g.new(&.{@floatFromInt(firstMax(g.vals.items[x]))});
    }

    pub fn concat(g: *ChainOps, xs: []const T, _: c_int) !T {
        var all: std.ArrayList(f32) = .empty;
        defer all.deinit(g.a);
        for (xs) |x| try all.appendSlice(g.a, g.vals.items[x]);
        return g.new(all.items);
    }

    pub fn evalAll(g: *ChainOps, xs: []const T) !void {
        g.evals += 1;
        try g.evaluated.appendSlice(g.a, xs);
    }

    pub fn hostU32(g: *ChainOps, x: T, out: []u32) ![]const u32 {
        try g.readable(x);
        g.reads += 1;
        const v = g.vals.items[x];
        if (v.len != out.len) return error.HostReadSize;
        for (out, v) |*o, f| o.* = @intFromFloat(f);
        return out;
    }

    pub fn hostArray(g: *ChainOps, bytes: []const u8, _: []const c_int, _: sdk.mlx.mlx_dtype) !T {
        const ids = std.mem.bytesAsSlice(u32, bytes);
        var v: [8]f32 = undefined;
        for (v[0..ids.len], ids) |*f, i| f.* = @floatFromInt(i);
        return g.new(v[0..ids.len]);
    }
};

const firstMax = argmax;

/// A toy draft head over `ChainOps`: a step's logits [V] are `targetRow` of (the pair's token, its position), its normed
/// row carries the token and the position forward; a pair's input is (the token taken from the id array, the normed row).
const ToyHead = struct {
    V: u32,

    pub fn step(h: ToyHead, g: *ChainOps, x: u32, pos: u32) !struct { normed: u32, logits: u32 } {
        const tok: u32 = @intFromFloat(g.vals.items[x][0]);
        var row: [64]f32 = undefined;
        targetRow(tok, pos, row[0..h.V]);
        return .{ .normed = try g.new(&.{ @floatFromInt(tok), @floatFromInt(pos) }), .logits = try g.new(row[0..h.V]) };
    }

    pub fn draw(_: ToyHead, g: *ChainOps, logits: u32, _: u32) !u32 {
        return g.reshape(try g.argmax(logits, 0), &.{1});
    }

    pub fn input(_: ToyHead, g: *ChainOps, idx: u32, hidden: u32) !u32 {
        return g.new(&.{ g.vals.items[idx][0], g.vals.items[hidden][1] });
    }
};

/// The stepwise chain (the lane's draft before the device chain): each step evaluated and its draft read on its own,
/// the next pair's input built from the host's id.
fn stepwiseDrafts(g: *ChainOps, head: ToyHead, x0: u32, start: u32, depth: u32, out: []u32) !void {
    var x = x0;
    for (0..depth) |s| {
        const r = try head.step(g, x, start + @as(u32, @intCast(s)));
        const am = try g.argmax(r.logits, 0);
        try g.evalAll(&.{am});
        var tok: [1]u32 = undefined;
        _ = try g.hostU32(am, &tok);
        out[s] = tok[0];
        if (s + 1 < depth) x = try head.input(g, try g.hostArray(std.mem.sliceAsBytes(out[s..][0..1]), &.{1}, .uint32), r.normed);
    }
}

test "glm5_next mtp draft chain: the device chain (each draft's argmax feeding the next pair on the device, one eval) drafts the stepwise chain's ids at depth 1..3 and reads each draft's top-1 probability in the same sync; one sync a round instead of one a step" {
    const head: ToyHead = .{ .V = 64 };
    for ([_]u32{ 1, 2, 3 }) |depth| for (0..32) |seed| {
        var g: ChainOps = .{ .a = testing.allocator };
        defer g.deinit();
        const t1: u32 = @intCast(seed % head.V);
        const start: u32 = 100 + @as(u32, @intCast(seed)) * 7;
        const x0 = try g.new(&.{ @floatFromInt(t1), 0 });
        var want: [mtp.max_depth]u32 = undefined;
        try stepwiseDrafts(&g, head, x0, start, depth, want[0..depth]);
        try testing.expectEqual(depth, g.evals);
        try testing.expectEqual(depth, g.reads);
        g.evals = 0;
        g.reads = 0;
        var got: [mtp.max_depth]u32 = undefined;
        var q: [mtp.max_depth]f32 = undefined;
        try mtp.chainDrafts(ChainOps, &g, head, x0, start, depth, got[0..depth], q[0..depth]);
        try testing.expectEqualSlices(u32, want[0..depth], got[0..depth]);
        // One eval; the drafts and their probabilities are read from it (no read of an array no eval named).
        try testing.expectEqual(@as(u32, 1), g.evals);
        try testing.expectEqual(@as(u32, 2), g.reads);
        // Each draft is the target's argmax after the previous one, its q the target row's softmax at it.
        var prev = t1;
        for (got[0..depth], q[0..depth], 0..) |d, p, s| {
            var row: [64]f32 = undefined;
            targetRow(prev, start + s, &row);
            try testing.expectEqual(argmax(&row), d);
            const m = std.mem.max(f32, &row);
            var z: f64 = 0;
            for (row) |f| z += @exp(@as(f64, f - m));
            try testing.expectApproxEqRel(@as(f32, @floatCast(1.0 / z)), p, 1e-5);
            prev = d;
        }
    };
    try noDevice();
}

test "glm5_next mtp draft gate: of a round's drafts the leading ones whose cumulative top-1 probability stays at or above theta are verified; theta 0 (off) keeps every draft" {
    const Row = struct { q: []const f32, theta: f32, k: u32 };
    for ([_]Row{
        .{ .q = &.{ 0.9, 0.8, 0.7 }, .theta = 0, .k = 3 }, // off
        .{ .q = &.{ 0.2, 0.1, 0.05 }, .theta = 0, .k = 3 }, // off keeps even unlikely drafts
        .{ .q = &.{ 0.9, 0.8, 0.7 }, .theta = 0.55, .k = 2 }, // 0.9, 0.72, 0.504
        .{ .q = &.{ 0.5, 0.99, 0.99 }, .theta = 0.55, .k = 0 }, // the first draft under theta: a serial step
        .{ .q = &.{ 0.99, 0.6, 0.99 }, .theta = 0.6, .k = 1 }, // 0.99, 0.594
        .{ .q = &.{ 1.0, 1.0, 1.0 }, .theta = 0.99, .k = 3 },
        .{ .q = &.{ 0.81, 0.74, 0.74 }, .theta = 0.55, .k = 2 }, // g21b's mean acceptance: 0.81, 0.599, 0.444
        .{ .q = &.{0.95}, .theta = 0.55, .k = 1 }, // a depth-1 round
        .{ .q = &.{}, .theta = 0.55, .k = 0 }, // no drafts
    }) |r| try testing.expectEqual(r.k, mtp.gatedDrafts(r.q, r.theta));
    try noDevice();
}

test "glm5_next mtp calibration: a synthetic acceptance table (each draft kept with its own q) bins every decided position by q, sums to the per-position counts, and a gate verifies only its prefix" {
    // Rounds of three drafts with q drawn from a seeded table; the target keeps draft i with probability q_i (the exact
    // rule keeps a prefix). Gate off: every position is counted in its q bin, the bins summing to `Positions`.
    var prng = std.Random.DefaultPrng.init(0xca1b);
    const rand = prng.random();
    for ([_]f32{ 0, 0.55 }) |theta| {
        var pos: mtp.Positions = .{};
        var cal: mtp.Calib = .{};
        var dropped: u64 = 0;
        for (0..2000) |_| {
            var q: [3]f32 = undefined;
            for (&q) |*p| p.* = 0.3 + 0.7 * rand.float(f32);
            const k = mtp.gatedDrafts(&q, theta);
            dropped += 3 - k;
            var kept: u32 = 0;
            while (kept < k and rand.float(f32) < q[kept]) kept += 1;
            pos.add(k, kept);
            cal.add(k, kept, q[0..k]);
        }
        for (0..mtp.max_depth) |i| {
            var t: u64 = 0;
            var kk: u64 = 0;
            for (cal.tested[i], cal.kept[i]) |tb, kb| {
                t += tb;
                kk += kb;
                try testing.expect(kb <= tb);
            }
            try testing.expectEqual(pos.tested[i], t);
            try testing.expectEqual(pos.kept[i], kk);
        }
        // Off: no draft dropped; on: some, and a bin's acceptance tracks its q (a kept draft is a q-coin's heads).
        if (theta == 0) try testing.expectEqual(@as(u64, 0), dropped) else try testing.expect(dropped > 0);
        if (theta == 0) {
            const lo = @as(f64, @floatFromInt(cal.kept[0][3] + cal.kept[0][4])) / @as(f64, @floatFromInt(cal.tested[0][3] + cal.tested[0][4]));
            const hi = @as(f64, @floatFromInt(cal.kept[0][9])) / @as(f64, @floatFromInt(cal.tested[0][9]));
            try testing.expect(lo < hi);
        }
    }
    var buf: [512]u8 = undefined;
    var c: mtp.Calib = .{};
    c.add(3, 1, &.{ 0.95, 0.42, 0.05 });
    c.add(2, 2, &.{ 1.0, 0.0 });
    try testing.expectEqualStrings("depth=3 q_bins=10 pos1=0/0,0/0,0/0,0/0,0/0,0/0,0/0,0/0,0/0,2/2 pos2=1/1,0/0,0/0,0/0,1/0,0/0,0/0,0/0,0/0,0/0 pos3=0/0,0/0,0/0,0/0,0/0,0/0,0/0,0/0,0/0,0/0", c.write(&buf, 3));
    try testing.expectEqual(@as(usize, 0), c.write(buf[0..20], 3).len);
    try noDevice();
}

test "glm5_next mtp: the lane's device code compiles against the arch's seams (declarations referenced, nothing run)" {
    std.testing.refAllDecls(mtp.Lane);
    std.testing.refAllDecls(mtp.Head);
    std.testing.refAllDecls(mtp.DraftBank);
    try noDevice();
}

/// The bytes of the arrays the lane's open allocates (`Lane.open`: the bank's nine [experts, ...], the head's f32 router
/// [hidden, experts], pool gate [hidden, index_head_dim] and ape [kpool, index_head_dim], the two bf16 lanes at the
/// positions plus a round's drafts), from their shapes and dtypes.
fn openArrayBytes(g: *const bank.Geometry, c: *const glm.Config, positions: u64) [bank.n_components + 5]u64 {
    var out: [bank.n_components + 5]u64 = undefined;
    const l: bank.Layer = .{ .logical_bytes = g.logical_bytes, .base_offset = 0, .segments = g.segments };
    for (out[0..bank.n_components], 0..) |*o, ci| o.* = mtp.Slots.arrayBytes(&l, ci, g.num_experts);
    const h: u64 = c.hidden_size;
    const ihd: u64 = c.index_head_dim;
    const lanes = positions + mtp.max_depth;
    out[bank.n_components..].* = .{ h * c.n_routed_experts * 4, h * ihd * 4, @as(u64, c.index_kpool) * ihd * 4, lanes * c.kv_lora_rank * 2, lanes * 2 * ihd * 2 };
    return out;
}

test "glm5_next mtp hold: what the open's arrays hold equals the bill's mtp_bank + mtp_tables + mtp_kv by construction (the pack's geometry with GLM5_BANK); a hold over the bill is refused by name" {
    const page: u64 = 16_384;
    // The served model's MTP layer over a bank of its geometry (288 experts at Flash's projections).
    const c: glm.Config = .{};
    const g = try bank.Geometry.init(c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.n_layers, 1);
    for ([_]u64{ 17_408, 32_768, 1_048_576 }) |positions| {
        const arrays = openArrayBytes(&g, &c, positions);
        const held = mtp.heldOf(&arrays, page);
        try testing.expectEqual(mtp.heldBytes(&g, &c, positions, page), held);
        const f: mtp.Facts = .{ .bank_bytes = mtp.DraftBank.arrayBytes(&g, page), .resident_bytes = 0, .table_bytes = mtp.Head.tableBytes(&c, page) };
        const t = mtp.terms(f, &c, positions, page);
        try testing.expectEqual(t[0].bytes[0] + t[2].bytes[0] + t[3].bytes[0], held);
        try mtp.holdCheck(held, held);
        try testing.expectError(error.ConstructionOverBill, mtp.holdCheck(held + 4, held));
    }
    // The window's 32,768-position load billed 4,134,049,152 B for the pack with packed global scales; line-isolated
    // ([288, 32] f32, 36,864 B -> 49,152 B at 16 KiB pages, were 1,152 B) the three arrays add 144,000 B.
    try testing.expectEqual(@as(u64, 4_134_049_152 + 144_000), mtp.heldOf(&openArrayBytes(&g, &c, 32_768), page));
    // The pack's own manifest, when given: its geometry is the served model's.
    const dir = std.mem.span(std.c.getenv("GLM5_BANK") orelse return);
    const a = testing.allocator;
    const io = testing.io;
    const pc = try packConfig(a, io, dir);
    var diag: sdk.Diag = .{};
    var db = try mtp.DraftBank.open(a, io, dir, &pc, &diag);
    defer db.deinit();
    const facts = try mtp.facts(a, io, dir, &pc, page, &diag);
    const t = mtp.terms(facts, &pc, 32_768, page);
    try testing.expectEqual(mtp.heldOf(&openArrayBytes(&db.owner.geometry, &pc, 32_768), page), t[0].bytes[0] + t[2].bytes[0] + t[3].bytes[0]);
}

/// A toy draft head near the toy target: `targetRow` with seeded noise (the head's q against the target's p).
fn draftRow(prev: u32, pos: u64, out: []f32) void {
    targetRow(prev, pos, out);
    var rng = std.Random.DefaultPrng.init(0x77 ^ (@as(u64, prev) *% 0x9e37_79b9) ^ (pos *% 0x5851_f42d));
    for (out) |*x| x.* += rng.random().floatNorm(f32) * 0.6;
}

test "glm5_next mtp coupled drafts: a sampled draft drawn from the head's q at the target's own uniform keeps the emitted ids the serial chain's (exact), and is accepted more often than the head's argmax" {
    const V = 64;
    const sm: acc.Params = .{ .temperature = 1.0, .top_p = 0.95, .seed = 7 };
    var dp: [V]f64 = undefined;
    var dq: [V]f64 = undefined;
    var row: [V]f32 = undefined;
    var drafted: [2]u64 = @splat(0);
    var kept: [2]u64 = @splat(0);
    for ([_]bool{ true, false }) |coupled| for (0..16) |seed| {
        const s: acc.Params = .{ .temperature = sm.temperature, .top_p = sm.top_p, .seed = sm.seed + seed };
        const n = 200;
        // The serial chain: token j is p's draw at position j.
        var serial: [n + 8]u32 = undefined;
        serial[0] = @intCast(seed % V);
        for (1..n + 8) |j| {
            targetRow(serial[j - 1], j, &row);
            acc.distribution(&row, s, &dp);
            serial[j] = acc.sample(&dp, s.seed, j);
        }
        var len: u64 = 1;
        var t1 = serial[1];
        var emitted: [n + 8]u32 = undefined;
        emitted[0] = serial[0];
        while (len < n) {
            var drafts: [3]u32 = undefined;
            var logits: [4 * V]f32 = undefined;
            var prev = t1;
            targetRow(t1, len + 1, logits[0..V]);
            for (0..3) |i| {
                const pos = len + 1 + i;
                draftRow(prev, pos, &row);
                if (coupled) {
                    acc.distribution(&row, s, &dq);
                    drafts[i] = acc.sample(&dq, s.seed, pos);
                } else drafts[i] = argmax(&row);
                prev = drafts[i];
                targetRow(prev, pos + 1, logits[(i + 1) * V ..][0..V]);
            }
            const r = acc.accept(mtp.rule, s, &logits, &drafts, null, len);
            emitted[len] = t1;
            for (drafts[0..r.n_accepted], 0..) |d, i| emitted[len + 1 + i] = d;
            len += r.n_accepted + 1;
            t1 = r.next_token;
            drafted[@intFromBool(coupled)] += 3;
            kept[@intFromBool(coupled)] += r.n_accepted;
        }
        // Exact: either drafter emits the serial chain's ids.
        try testing.expectEqualSlices(u32, serial[0..len], emitted[0..len]);
    };
    const rate_argmax = @as(f64, @floatFromInt(kept[0])) / @as(f64, @floatFromInt(drafted[0]));
    const rate_coupled = @as(f64, @floatFromInt(kept[1])) / @as(f64, @floatFromInt(drafted[1]));
    std.debug.print("GLM5N_MTP_COUPLED acceptance argmax={d:.3} coupled={d:.3}\n", .{ rate_argmax, rate_coupled });
    try testing.expect(rate_coupled > rate_argmax);
    try noDevice();
}

test "glm5_next mtp acceptance setting: exact is today's rule; typical (delta 0.5) keeps more of a noisy head's drafts than exact on a synthetic (p, q) pair, its emitted tokens still drawn from the target" {
    try testing.expectEqual(acc.Rule.exact, mtp.ruleOf(.exact, 0.3));
    try testing.expectEqual(@as(f32, 0.5), mtp.ruleOf(.typical, settings.typical_delta).typical.delta);
    var nb: [mtp.name_cap]u8 = undefined;
    try testing.expectEqualStrings("glm5_next-mtp exact", mtp.nameOf(&nb, mtp.ruleOf(.exact, 0.7)));
    for ([_]f32{ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }, [_][]const u8{ "glm5_next-mtp typical 0.2", "glm5_next-mtp typical 0.3", "glm5_next-mtp typical 0.4", "glm5_next-mtp typical 0.5", "glm5_next-mtp typical 0.6", "glm5_next-mtp typical 0.7", "glm5_next-mtp typical 0.8" }) |d, want| {
        try testing.expectEqual(d, mtp.ruleOf(.typical, d).typical.delta);
        try testing.expectEqualStrings(want, mtp.nameOf(&nb, mtp.ruleOf(.typical, d)));
    }
    const V = 64;
    var row: [V]f32 = undefined;
    var kept: [2]u64 = @splat(0);
    for ([_]acc.Rule{ mtp.ruleOf(.exact, settings.typical_delta), mtp.ruleOf(.typical, settings.typical_delta) }, 0..) |rule, ri| for (0..16) |seed| {
        const s: acc.Params = .{ .temperature = 1.0, .top_p = 0.95, .seed = 11 + seed };
        var len: u64 = 1;
        var t1: u32 = @intCast(seed % V);
        while (len < 200) {
            var drafts: [3]u32 = undefined;
            var logits: [4 * V]f32 = undefined;
            var prev = t1;
            targetRow(t1, len + 1, logits[0..V]);
            for (0..3) |i| {
                draftRow(prev, len + 1 + i, &row);
                drafts[i] = argmax(&row);
                prev = drafts[i];
                targetRow(prev, len + 2 + i, logits[(i + 1) * V ..][0..V]);
            }
            const r = acc.accept(rule, s, &logits, &drafts, null, len);
            kept[ri] += r.n_accepted;
            len += r.n_accepted + 1;
            t1 = r.next_token;
        }
    };
    std.debug.print("GLM5N_MTP_ACCEPTANCE kept exact={d} typical={d}\n", .{ kept[0], kept[1] });
    try testing.expect(kept[1] >= kept[0]);
    try noDevice();
}
