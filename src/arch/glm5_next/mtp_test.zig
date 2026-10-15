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
            for (synth.geom.segments, 0..) |s, ci| try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], rows.row(@fromBackingInt(@intCast(ci)), @intCast(r)));
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
        for (synth.geom.segments, 0..) |s, ci| try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], rows.row(@fromBackingInt(@intCast(ci)), r));
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
        for (g.segments, 0..) |s, ci| try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], rows.row(@fromBackingInt(@intCast(ci)), @intCast(r)));
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
            const k = mtp.draftsAt(len, 1, len, cap, 1 << 20);
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
    // Untracked (a lane behind the trunk) or no room: no drafts; the host's cap and the billed positions bound the rest.
    try testing.expectEqual(@as(u32, 0), mtp.draftsAt(9, 1, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 0), mtp.draftsAt(10, 0, 10, 9, 1000));
    try testing.expectEqual(@as(u32, 2), mtp.draftsAt(10, 1, 10, 9, 13));
    try testing.expectEqual(@as(u32, 1), mtp.draftsAt(10, 1, 10, 1, 1000));
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
    for (synth.geom.segments) |s| arrays += graph.bufferBytes(3 * s.length, page);
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

test "glm5_next mtp: the lane's device code compiles against the arch's seams (declarations referenced, nothing run)" {
    std.testing.refAllDecls(mtp.Lane);
    std.testing.refAllDecls(mtp.Head);
    std.testing.refAllDecls(mtp.DraftBank);
    try noDevice();
}
