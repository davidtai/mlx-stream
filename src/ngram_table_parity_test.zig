//! The plugin's n-gram table (ngram_table.zig, a copy of the table mlx-serve's qwen4_exp serves) against mlx-serve
//! upstream main af34af04's own code (ngram_oracle_af34af04.zig) on the same inputs: the hash, the rows and gathers
//! at every bit width (serial and pooled), the header refusals by name, the bf16 helpers, the prefetch knobs and the
//! warm progress. Test-only.

const std = @import("std");
const testing = std.testing;
const ngram = @import("ngram_table.zig");
const ngram_oracle = @import("ngram_oracle_af34af04.zig");

const MAX_NGRAM_SIZE = ngram.MAX_NGRAM_SIZE;
const MAX_HEADS = ngram.MAX_HEADS;
const NgramHash = ngram.NgramHash;
const NgramTable = ngram.NgramTable;
const bf16ToF32 = ngram.bf16ToF32;
const bf16Rne = ngram.bf16Rne;
const plePrefillPrefetchModeFromEnv = ngram.plePrefillPrefetchModeFromEnv;
const plePrefillPrefetchMinKvFromEnv = ngram.plePrefillPrefetchMinKvFromEnv;
const plePrefillPrefetchWanted = ngram.plePrefillPrefetchWanted;
const PrefillPrefetchMode = ngram.PrefillPrefetchMode;
const PREFILL_PREFETCH_MIN_KV = ngram.PREFILL_PREFETCH_MIN_KV;
const PREFILL_SAY_MIN_ROWS = ngram.PREFILL_SAY_MIN_ROWS;
const WARM_LOG_BYTES = ngram.WARM_LOG_BYTES;
const WARM_LOG_NS = ngram.WARM_LOG_NS;
const WarmProgress = ngram.WarmProgress;

/// A table file of `rows` x `dim` at `bits` (16 = raw BF16) in a temp dir, opened through this plugin's table
/// (mlx-serve's ple_gpu.writeFixture, the writer the host's own n-gram tests use).
const Fixture = struct {
    td: std.testing.TmpDir,
    table: NgramTable,

    fn deinit(self: *Fixture) void {
        self.table.close();
        self.td.cleanup();
    }
};

fn randBf16(r: std.Random) u16 {
    // Normal values only (exponent 2^-17 .. 2^13): real tables carry no NaN, Inf or subnormal.
    const sign: u16 = @as(u16, r.int(u1)) << 15;
    const exp: u16 = r.intRangeAtMost(u16, 110, 140);
    return sign | (exp << 7) | r.int(u7);
}

fn writeFixture(bits: u32, rows: u64, dim: u32, gs: u32, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const a = testing.allocator;
    const hlen: usize = 509;
    const raw = bits == 16;
    const wbytes: u64 = if (raw) rows * dim * 2 else rows * (dim * bits / 32) * 4;
    const sbytes: u64 = if (raw) 0 else rows * (dim / gs) * 2;
    var hbuf: [hlen]u8 = @splat(' ');
    if (raw)
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}", .{ gs, rows, dim, wbytes })
    else
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"{d}\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"U32\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}},\"scales\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}},\"biases\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ bits, gs, rows, dim * bits / 32, wbytes, rows, dim / gs, wbytes, wbytes + sbytes, rows, dim / gs, wbytes + sbytes, wbytes + 2 * sbytes });
    const total: usize = @intCast(8 + hlen + wbytes + 2 * sbytes);
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    std.mem.writeInt(u64, buf[0..8], hlen, .little);
    @memcpy(buf[8 .. 8 + hlen], &hbuf);
    const data = buf[8 + hlen ..];
    if (raw) {
        var i: usize = 0;
        while (i < wbytes) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    } else {
        r.bytes(data[0..@intCast(wbytes)]);
        var i: usize = @intCast(wbytes);
        while (i < data.len) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    }
    var td = std.testing.tmpDir(.{});
    errdefer td.cleanup();
    try td.dir.writeFile(testing.io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(testing.io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});
    ngram.warm_override = false;
    defer ngram.warm_override = null;
    return .{ .td = td, .table = try NgramTable.open(path) };
}

test "ngram table parity: hash init and row ids equal upstream's code on random configs and histories" {
    var prng = std.Random.DefaultPrng.init(0x4e6772);
    const r = prng.random();
    var cases: usize = 0;
    while (cases < 400) : (cases += 1) {
        const n = 2 + r.uintLessThan(u32, MAX_NGRAM_SIZE - 1);
        const heads = 1 + r.uintLessThan(u32, @intCast(MAX_HEADS / (n - 1)));
        const vocab = 2 + r.uintLessThan(u32, 300_000);
        const base = 2 + r.uintLessThan(u64, 40_000);
        const div = 1 + r.uintLessThan(u64, 256);
        const seed = r.int(u64);
        const layer = r.uintLessThan(u32, 3);
        const eos = r.uintLessThan(u32, 64);
        const new = try NgramHash.init(vocab, n, heads, base, div, seed, layer, eos);
        const old = try ngram_oracle.NgramHash.init(vocab, n, heads, base, div, seed, layer, eos);
        try testing.expectEqual(old.n_heads, new.n_heads);
        try testing.expectEqual(old.total_rows, new.total_rows);
        try testing.expectEqualSlices(i64, &old.multipliers, &new.multipliers);
        try testing.expectEqualSlices(i64, &old.vocab, &new.vocab);
        try testing.expectEqualSlices(i64, &old.offsets, &new.offsets);
        // A history with eos resets: tokens from a small alphabet that includes eos.
        var prev: [MAX_NGRAM_SIZE]u32 = undefined;
        for (prev[0 .. n - 1]) |*p| p.* = if (r.boolean()) eos else r.uintLessThan(u32, 64);
        var ids: [24]u32 = undefined;
        const len = 1 + r.uintLessThan(usize, ids.len);
        for (ids[0..len]) |*t| t.* = r.uintLessThan(u32, 64);
        var out_new: [24 * MAX_HEADS]i64 = undefined;
        var out_old: [24 * MAX_HEADS]i64 = undefined;
        const m = len * new.n_heads;
        new.rowIds(prev[0 .. n - 1], ids[0..len], out_new[0..m]);
        old.rowIds(prev[0 .. n - 1], ids[0..len], out_old[0..m]);
        try testing.expectEqualSlices(i64, out_old[0..m], out_new[0..m]);
    }
    // The refusals are the same, by name.
    const bad = [_][8]u64{
        .{ 1000, 1, 8, 100, 1, 0, 0, 0 }, // ngram_size < 2
        .{ 1000, 9, 1, 100, 1, 0, 0, 0 }, // ngram_size > 8
        .{ 1000, 3, 0, 100, 1, 0, 0, 0 }, // no heads
        .{ 1000, 3, 17, 100, 1, 0, 0, 0 }, // heads past MAX_HEADS
        .{ 1000, 3, 8, 100, 0, 0, 0, 0 }, // divisor 0
        .{ 1000, 3, 8, 1, 1, 0, 0, 0 }, // vocab_base < 2
    };
    for (bad) |b| {
        const e_new = NgramHash.init(@intCast(b[0]), @intCast(b[1]), @intCast(b[2]), b[3], b[4], b[5], @intCast(b[6]), @intCast(b[7]));
        const e_old = ngram_oracle.NgramHash.init(@intCast(b[0]), @intCast(b[1]), @intCast(b[2]), b[3], b[4], b[5], @intCast(b[6]), @intCast(b[7]));
        if (e_old) |_| return error.OracleAcceptedABadConfig else |e| try testing.expectError(e, e_new);
    }
}

test "ngram table parity: table rows and gathers (serial and pooled) equal upstream's code at every bit width" {
    const widths = [_][3]u32{ .{ 2, 64, 32 }, .{ 3, 64, 32 }, .{ 4, 64, 32 }, .{ 5, 64, 64 }, .{ 6, 64, 32 }, .{ 8, 64, 64 }, .{ 16, 48, 16 } };
    for (widths, 0..) |w, k| {
        const rows: u64 = 300;
        var fx = try writeFixture(w[0], rows, w[1], w[2], 0xC0FE + k);
        defer fx.deinit();
        var root: [std.fs.max_path_bytes]u8 = undefined;
        var full: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{root[0..try fx.td.dir.realPath(testing.io, &root)]});
        ngram_oracle.warm_override = false;
        defer ngram_oracle.warm_override = null;
        var old = try ngram_oracle.NgramTable.open(path);
        defer old.close();
        const new = &fx.table;
        try testing.expectEqual(old.rows, new.rows);
        try testing.expectEqual(old.dim, new.dim);
        try testing.expectEqual(old.bits, new.bits);
        try testing.expectEqual(old.group_size, new.group_size);
        try testing.expect(!new.nocache and new.cache == null and new.poster == null and new.row_gather == null);
        const dim: usize = new.dim;
        const a = testing.allocator;
        const ro = try a.alloc(f32, dim);
        defer a.free(ro);
        const rn = try a.alloc(f32, dim);
        defer a.free(rn);
        for (0..rows) |r| {
            old.row(r, ro);
            new.row(r, rn);
            try testing.expectEqualSlices(u32, @ptrCast(ro), @ptrCast(rn));
        }
        // A decode-width gather and a prompt-width one (past PrefetchPool.MAX_ROWS), each on both arms.
        var prng = std.Random.DefaultPrng.init(k);
        var ids: [200]i64 = undefined;
        for (&ids) |*id| id.* = @intCast(prng.random().uintLessThan(u64, rows));
        const go = try a.alloc(f32, ids.len * dim);
        defer a.free(go);
        const gn = try a.alloc(f32, ids.len * dim);
        defer a.free(gn);
        for ([_]bool{ false, true }) |pooled| {
            ngram_oracle.ple_prefill_prefetch_override = pooled;
            ngram.ple_prefill_prefetch_override = pooled;
            defer ngram_oracle.ple_prefill_prefetch_override = null;
            defer ngram.ple_prefill_prefetch_override = null;
            for ([_]usize{ 16, ids.len }) |len| {
                old.gather(ids[0..len], go[0 .. len * dim], 1 << 20);
                new.gather(ids[0..len], gn[0 .. len * dim], 1 << 20);
                try testing.expectEqualSlices(u32, @ptrCast(go[0 .. len * dim]), @ptrCast(gn[0 .. len * dim]));
            }
        }
    }
}

test "ngram table parity: table header refusals equal upstream's code, by name" {
    const page = std.heap.page_size_min;
    var map: [2 * 4096]u8 align(page) = @splat(0);
    const headers = [_][]const u8{
        "not json",
        "[]",
        "{}",
        "{\"__metadata__\":[]}",
        "{\"__metadata__\":{\"format\":\"other\",\"bits\":\"4\",\"group_size\":\"32\"}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"group_size\":\"32\"}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"7\",\"group_size\":\"32\"}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[100,116]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,9000]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"BF16\",\"shape\":[4,8],\"data_offsets\":[0,64]}}",
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}}",
        "{\"__metadata__\":{\"bits\":\"8\",\"group_size\":\"64\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[2,16],\"data_offsets\":[0,128]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[128,132]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[132,136]}}",
    };
    for (headers) |h| {
        const r_old = ngram_oracle.NgramTable.parse(&map, h, 1024);
        const r_new = NgramTable.parse(&map, h, 1024);
        if (r_old) |o| {
            const n = try r_new;
            inline for (.{ "rows", "dim", "bits", "group_size", "w_off", "s_off", "b_off", "wcols", "scols" }) |f|
                try testing.expectEqual(@field(o, f), @field(n, f));
        } else |e| try testing.expectError(e, r_new);
    }
}

test "ngram table parity: bf16 helpers, prefetch knobs and warm progress equal upstream's code" {
    var u: u32 = 0;
    while (u <= 0xFFFF) : (u += 1) {
        const x: u16 = @intCast(u);
        try testing.expectEqual(@as(u32, @bitCast(ngram_oracle.bf16ToF32(x))), @as(u32, @bitCast(bf16ToF32(x))));
    }
    var prng = std.Random.DefaultPrng.init(0xBF16);
    for (0..200_000) |_| {
        const f: f32 = @bitCast(prng.random().int(u32));
        try testing.expectEqual(ngram_oracle.bf16Rne(f), bf16Rne(f));
    }
    const raws = [_]?[]const u8{ null, "", "0", "1", "2", "on", "10", " 1", "0x" };
    for (raws) |raw| {
        try testing.expectEqualStrings(@tagName(ngram_oracle.plePrefillPrefetchModeFromEnv(raw)), @tagName(plePrefillPrefetchModeFromEnv(raw)));
        try testing.expectEqual(ngram_oracle.plePrefillPrefetchMinKvFromEnv(raw), plePrefillPrefetchMinKvFromEnv(raw));
    }
    for ([_]?[]const u8{ "65536", " 4096\t", "-1", "18446744073709551615", "1e6" }) |raw|
        try testing.expectEqual(ngram_oracle.plePrefillPrefetchMinKvFromEnv(raw), plePrefillPrefetchMinKvFromEnv(raw));
    for ([_]PrefillPrefetchMode{ .off, .kv_gated, .on }) |m| for ([_]u64{ 0, 1, 262143, 262144, 1 << 30 }) |kv| for ([_]u64{ 0, 262144 }) |min| {
        const om: ngram_oracle.PrefillPrefetchMode = @enumFromInt(@intFromEnum(m));
        try testing.expectEqual(ngram_oracle.plePrefillPrefetchWanted(om, kv, min), plePrefillPrefetchWanted(m, kv, min));
    };
    try testing.expectEqual(ngram_oracle.PREFILL_PREFETCH_MIN_KV, PREFILL_PREFETCH_MIN_KV);
    try testing.expectEqual(ngram_oracle.PREFILL_SAY_MIN_ROWS, PREFILL_SAY_MIN_ROWS);
    try testing.expectEqual(ngram_oracle.WARM_LOG_BYTES, WARM_LOG_BYTES);
    try testing.expectEqual(ngram_oracle.WARM_LOG_NS, WARM_LOG_NS);
    var po: ngram_oracle.WarmProgress = .{};
    var pn: WarmProgress = .{};
    var bytes: u64 = 0;
    var ns: u64 = 0;
    for (0..2000) |_| {
        bytes += prng.random().uintLessThan(u64, 2 << 30);
        ns += prng.random().uintLessThan(u64, 3_000_000_000);
        try testing.expectEqual(po.should(bytes, ns), pn.should(bytes, ns));
    }
}
