//! The hashed n-gram embedding tables both Qwen3.8-Flash-Next (`qwen4_exp`, its PLE) and mlx-stream's DeepSeek-V4.1
//! Engram read: the n-gram id math, the mmapped quantized table (or its raw BF16 rows past the page cache) and the
//! fixed-record row cache. A shared named module (like io_util): the host model and the plugin import it by name.
//!
//! Format: `ngram_table.bin` is a safetensors-format file holding one merged
//! affine table (`weight` U32 [R, dim*bits/32], `scales`/`biases` BF16
//! [R, dim/gs]) written by `tests/convert_qwen38_flash_next.py`, or one
//! merged RAW BF16 table (`weight` BF16 [R, dim], no scales/biases,
//! `"bits":"16"`) for bit-exact PLE lookups.

const std = @import("std");
const log = @import("sdk").log;
const io_util = @import("nocache_io.zig");

const MASK64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_M1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_M2: u64 = 0x94D049BB133111EB;
const PRIME_1: u64 = 10007;

fn splitmix64(v0: u64) u64 {
    var v = v0 +% SPLITMIX_GAMMA;
    v = (v ^ (v >> 30)) *% SPLITMIX_M1;
    v = (v ^ (v >> 27)) *% SPLITMIX_M2;
    return v ^ (v >> 31);
}

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    var d: u64 = 3;
    while (d * d <= v) : (d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

fn nthPrimeAfter(start: u64, count: u32) u64 {
    var p = start;
    for (0..count) |_| {
        p += 1;
        while (!isPrime(p)) p += 1;
    }
    return p;
}

/// Config-driven bounds; `model.validateQwen4Config` refuses a checkpoint past them at load.
pub const MAX_HEADS = 32;
pub const MAX_NGRAM_SIZE = 8;

/// Everything `Qwen4ExpTextNGramEmbedding.__init__` derives from the config.
pub const NgramHash = struct {
    ngram_size: u32,
    heads_per_ngram: u32,
    n_heads: u32,
    eos: u32,
    multipliers: [8]i64,
    vocab: [MAX_HEADS]i64,
    offsets: [MAX_HEADS]i64,
    total_rows: u64,

    /// `ple_layer_index` is the PLE's ordinal among the config's injection points (0 for the
    /// one we support). Fallible: every bound writes a fixed array.
    pub fn init(unigram_vocab: u32, ngram_size: u32, heads_per_ngram: u32, vocab_base: u64, divisor: u64, seed: u64, ple_layer_index: u32, eos: u32) !NgramHash {
        if (ngram_size < 2 or ngram_size > MAX_NGRAM_SIZE) return error.InvalidQwen4NgramSize;
        if (heads_per_ngram == 0 or (ngram_size - 1) * heads_per_ngram > MAX_HEADS) {
            return error.InvalidQwen4NgramHeads;
        }
        if (divisor == 0 or vocab_base < 2) return error.InvalidQwen4NgramVocab;
        var h: NgramHash = .{
            .ngram_size = ngram_size,
            .heads_per_ngram = heads_per_ngram,
            .n_heads = (ngram_size - 1) * heads_per_ngram,
            .eos = eos,
            .multipliers = @splat(0),
            .vocab = @splat(0),
            .offsets = @splat(0),
            .total_rows = 0,
        };
        const max_long: u64 = (1 << 63) - 1;
        const half_bound: u64 = @max(1, (max_long / @max(unigram_vocab, 1)) / 2);
        const base_seed: u64 = seed +% PRIME_1 *% ple_layer_index;
        for (0..ngram_size) |i| {
            const v = base_seed +% SPLITMIX_GAMMA *% (@as(u64, i) + 1);
            h.multipliers[i] = @intCast(2 * (splitmix64(v) % half_bound) + 1);
        }
        var total: u64 = 0;
        for (0..h.n_heads) |i| {
            const global = ple_layer_index * h.n_heads + @as(u32, @intCast(i));
            const size = nthPrimeAfter(vocab_base - 1, global + 1);
            h.vocab[i] = @intCast(size);
            h.offsets[i] = @intCast(total);
            total += size;
        }
        h.total_rows = (total + divisor - 1) / divisor * divisor;
        return h;
    }

    /// Row ids for `ids`, given the (ngram_size-1) tokens that precede them
    /// (`eos` for a fresh sequence). `out` is `[ids.len][n_heads]` row-major.
    /// Mirrors `_shift_right_ignore_eos` + the mixed-id hash: a shifted token
    /// is `eos` when the shift crosses the most recent eos before it.
    pub fn rowIds(self: *const NgramHash, prev: []const u32, ids: []const u32, out: []i64) void {
        const ctx: usize = self.ngram_size - 1;
        std.debug.assert(prev.len == ctx and out.len == ids.len * self.n_heads);
        var last_eos: i64 = -1;
        var t: usize = 0;
        while (t < ctx + ids.len) : (t += 1) {
            const tok = tokAt(prev, ids, t);
            if (t >= ctx) {
                const seg_pos: i64 = @as(i64, @intCast(t)) - (last_eos + 1);
                var mixed: i64 = @as(i64, tok) *% self.multipliers[0];
                const row = out[(t - ctx) * self.n_heads ..][0..self.n_heads];
                var n: usize = 2;
                var pos: usize = 1;
                while (n <= self.ngram_size) : (n += 1) {
                    while (pos < n) : (pos += 1) {
                        const shifted: u32 = if (seg_pos >= @as(i64, @intCast(pos)) and t >= pos) tokAt(prev, ids, t - pos) else self.eos;
                        mixed ^= @as(i64, shifted) *% self.multipliers[pos];
                    }
                    const h0 = (n - 2) * self.heads_per_ngram;
                    for (h0..h0 + self.heads_per_ngram) |h| {
                        row[h] = @mod(mixed, self.vocab[h]) + self.offsets[h];
                    }
                }
            }
            if (tok == self.eos) last_eos = @intCast(t);
        }
    }

    fn tokAt(prev: []const u32, ids: []const u32, i: usize) u32 {
        return if (i < prev.len) prev[i] else ids[i - prev.len];
    }
};

/// The merged quantized table, mmapped read-only.
var warm_env_cached: ?bool = null;
pub var warm_override: ?bool = null;

fn warmEnabled() bool {
    if (warm_override) |v| return v;
    if (warm_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_NGRAM_WARM") orelse break :blk true;
        break :blk raw[0] != '0';
    };
    warm_env_cached = v;
    return v;
}

/// What the background page-cache warm has read so far, and the table's total size. Published
/// by the warm thread, read lock-free by metrics and `/props`; zero when nothing is warming.
pub var live_warm_bytes = std.atomic.Value(u64).init(0);
pub var live_warm_total = std.atomic.Value(u64).init(0);

/// A progress line at each 8 GB step or after 10 s of silence, never twice per step. Pure.
pub const WARM_LOG_BYTES: u64 = 8 << 30;
pub const WARM_LOG_NS: u64 = 10_000_000_000;

pub const WarmProgress = struct {
    next_bytes: u64 = WARM_LOG_BYTES,
    last_ns: u64 = 0,

    pub fn should(self: *WarmProgress, bytes: u64, elapsed_ns: u64) bool {
        const by_bytes = bytes >= self.next_bytes;
        const by_time = elapsed_ns -| self.last_ns >= WARM_LOG_NS;
        if (!by_bytes and !by_time) return false;
        while (self.next_bytes <= bytes) self.next_bytes += WARM_LOG_BYTES;
        self.last_ns = elapsed_ns;
        return true;
    }
};

fn asGb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

pub const NgramTable = struct {
    map: []align(std.heap.page_size_min) const u8,
    rows: u64,
    dim: u32,
    bits: u32,
    group_size: u32,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    wcols: u32,
    scols: u32,
    /// Kept open for the pool's `pread` gather (page faults on one mapping
    /// serialize on the VM map lock; preads run in parallel).
    fd: std.c.fd_t = -1,
    pool: ?*PrefetchPool = null,
    /// Boot-time page-cache warm: the weights load evicts this file, and the
    /// first long prompt then faults 48 rows/token from SSD (38k: 174 s vs
    /// 55 s warm). preads through the kept fd, never the mapping.
    warm_thread: ?std.Thread = null,
    warm_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    warm_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Set once `ple_gpu.wrap` hands the mapping to a no-copy Metal buffer: MLX unmaps it
    /// when its last reference drops, so a kernel still in flight never reads freed pages.
    gpu_owns_map: bool = false,
    /// Rows come through `pread` on `fd`, opened past the page cache
    /// (F_NOCACHE, read-ahead off), never through a mapping (`openTensor`).
    nocache: bool = false,
    /// A record table's resident rows (`attachCache`); without it every row is read.
    cache: ?*RowCache = null,
    /// The table's posted gathers (`enablePosting`): one thread runs them in post order.
    poster: ?*Poster = null,
    /// A raw BF16 table's aligned parallel row reads (`openTensor`; `gatherRaw`): its stages and readers, built once.
    row_gather: ?*io_util.RowGather = null,

    /// `gatherRaw`'s readers: 15 helpers and the caller; pieces of 1,024 ids; the caller alone below 32 ids (a decode
    /// or verify block). Built once per table (`RowGather.persistentBytes`: the bill's named term).
    pub const raw_gather_helpers: usize = 15;
    pub const raw_gather_max_ids: usize = 1024;
    pub const raw_gather_parallel_min: usize = 32;

    /// A gather handed to the table's poster thread: exactly `gatherRecords(rows, out)`, run in post
    /// order (the cache included), ready once `wait` returns. The caller keeps `rows` and `out` alive and
    /// makes no other gather on the table until every posted one is waited.
    pub const Posted = struct {
        rows: []const i64,
        out: []u8,
        state: enum { queued, done, failed } = .queued,
        err: anyerror = error.RecordRead,
        next: ?*Posted = null,
    };

    /// `bits` of a record table: raw fixed-width records, never dequantized here.
    pub const records_bits: u32 = 0;

    /// `rows` records of `record_bytes` at `data_offset` of `path` (an Engram bank, an
    /// embedding tensor inside a safetensors shard), read past the page cache, not mapped.
    pub fn openRecords(path: [:0]const u8, record_bytes: u32, rows: u64, data_offset: u64) !NgramTable {
        if (record_bytes == 0) return error.NgramTableRegion;
        const fd = try io_util.openNoCache(path.ptr, .{});
        errdefer _ = std.c.close(fd);
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        const need = std.math.add(u64, data_offset, std.math.mul(u64, rows, record_bytes) catch return error.NgramTableRegion) catch return error.NgramTableRegion;
        if (size < 0 or @as(u64, @intCast(size)) < need) return error.NgramTableTruncated;
        var t: NgramTable = .{ .map = &empty_map, .rows = rows, .dim = record_bytes, .bits = records_bits, .group_size = 0, .w_off = @intCast(data_offset), .s_off = 0, .b_off = 0, .wcols = 0, .scols = 0, .fd = fd, .nocache = true };
        if (record_bytes <= PrefetchPool.ROW_BUF) t.pool = try PrefetchPool.create();
        return t;
    }

    /// The records of `row_ids` read past the cache (a check's independent read).
    pub fn readUncached(self: *const NgramTable, row_ids: []const i64, out: []u8) !void {
        std.debug.assert(self.bits == records_bits and out.len == row_ids.len * self.dim);
        for (row_ids) |r| if (r < 0 or @as(u64, @intCast(r)) >= self.rows) return error.RowOutOfRange;
        return self.readRecords(row_ids, out);
    }

    /// Start the table's poster thread (`post` / `wait`). Call once the table sits at its final address.
    pub fn enablePosting(self: *NgramTable) !void {
        std.debug.assert(self.poster == null);
        self.poster = try Poster.create(self);
    }

    /// Queue `job` on the poster thread; the gather starts once the jobs before it finish.
    pub fn post(self: *NgramTable, job: *Posted) void {
        self.poster.?.push(job);
    }

    /// Block until `job` ran; its gather's error, if any.
    pub fn wait(self: *NgramTable, job: *Posted) !void {
        return self.poster.?.await(job);
    }

    /// Keep up to `budget_bytes` of this record table's rows resident (`RowCache`).
    pub fn attachCache(self: *NgramTable, gpa: std.mem.Allocator, budget_bytes: u64) !void {
        std.debug.assert(self.bits == records_bits and self.cache == null);
        const c = try gpa.create(RowCache);
        errdefer gpa.destroy(c);
        c.* = try RowCache.init(gpa, self.dim, budget_bytes);
        self.cache = c;
    }

    /// Row reads per row: the three quantized regions, or the one record.
    fn regions(self: *const NgramTable) usize {
        return if (self.bits == records_bits) 1 else 3;
    }

    /// The records of `row_ids`, in order, into `out` (`row_ids.len * record_bytes`),
    /// through the cache when one is attached; reads ride the pool.
    pub fn gatherRecords(self: *const NgramTable, row_ids: []const i64, out: []u8) !void {
        std.debug.assert(self.bits == records_bits and out.len == row_ids.len * self.dim);
        for (row_ids) |r| if (r < 0 or @as(u64, @intCast(r)) >= self.rows) return error.RowOutOfRange;
        if (self.cache) |c| return c.gather(self, row_ids, out);
        return self.readRecords(row_ids, out);
    }

    /// Read `rows`' records into `out`, the pool's rounds when it has one.
    fn readRecords(self: *const NgramTable, rows: []const i64, out: []u8) !void {
        const rb: usize = self.dim;
        if (self.pool) |p| {
            var start: usize = 0;
            while (start < rows.len) : (start += PrefetchPool.MAX_ROWS) {
                const end = @min(start + PrefetchPool.MAX_ROWS, rows.len);
                if (!p.run(self, rows[start..end])) return error.RecordRead;
                for (start..end) |i| @memcpy(out[i * rb ..][0..rb], p.bufs[i - start][0..rb]);
            }
            return;
        }
        for (rows, 0..) |r, i| if (!self.preadSite(@intCast(r), 0, out[i * rb ..][0..rb])) return error.RecordRead;
    }

    pub fn open(path: []const u8) !NgramTable {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (path.len >= pbuf.len) return error.NameTooLong;
        @memcpy(pbuf[0..path.len], path);
        pbuf[path.len] = 0;
        const fd = std.c.open(pbuf[0..path.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        errdefer _ = std.c.close(fd);
        // File size via lseek-to-end: 0.17 has no portable fstat wrapper on
        // Linux (std.c.Stat is void there) and this loader has no `std.Io`.
        const size: usize = @intCast(@max(std.c.lseek(fd, 0, std.c.SEEK.END), 0));
        if (size == 0) return error.StatFailed;
        const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        errdefer std.posix.munmap(map);
        if (size < 8) return error.NgramTableTruncated;
        const hlen: usize = @intCast(std.mem.readInt(u64, map[0..8], .little));
        if (hlen > size - 8) return error.NgramTableTruncated;
        var t = try parse(map, map[8 .. 8 + hlen], 8 + hlen);
        t.fd = fd;
        if (plePrefetchEnabled()) t.pool = PrefetchPool.create() catch null;
        return t;
    }

    /// A raw BF16 `[rows, dim]` tensor named `name` inside any safetensors file
    /// (a checkpoint shard, no `mlx-serve-ngram` metadata), read past the page
    /// cache: the descriptor is F_NOCACHE with read-ahead off, rows come
    /// through `pread` (`gatherRaw`), nothing is mapped and no warm thread runs.
    pub fn openTensor(path: [:0]const u8, name: []const u8) !NgramTable {
        const fd = io_util.openNoCache(path.ptr, .{}) catch |e| return switch (e) {
            error.NoCacheFcntl, error.NoCacheUnsupported => error.NgramTableNoCache,
            else => error.FileNotFound,
        };
        errdefer _ = std.c.close(fd);
        const size: usize = @intCast(@max(std.c.lseek(fd, 0, std.c.SEEK.END), 0));
        if (size < 8) return error.NgramTableTruncated;
        var len_bytes: [8]u8 = undefined;
        try io_util.readAligned(fd, &len_bytes, 0);
        const hlen: usize = @intCast(std.mem.readInt(u64, &len_bytes, .little));
        if (hlen > size - 8) return error.NgramTableTruncated;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const header = try a.alloc(u8, hlen);
        try io_util.readAligned(fd, header, 8);
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return error.NgramTableHeader;
        if (parsed != .object) return error.NgramTableHeader;
        const w = try headerRegion(parsed.object, name, "BF16", 2, size, 8 + hlen);
        var t: NgramTable = .{
            .map = &empty_map,
            .rows = w.rows,
            .dim = @intCast(w.cols),
            .bits = 16,
            .group_size = 0,
            .w_off = 8 + hlen + @as(usize, @intCast(w.start)),
            .s_off = 0,
            .b_off = 0,
            .wcols = 0,
            .scols = 0,
            .fd = fd,
            .nocache = true,
        };
        t.row_gather = try io_util.RowGather.init(fd, t.w_off, @as(usize, t.dim) * 2, t.rows, raw_gather_helpers, raw_gather_max_ids, raw_gather_parallel_min);
        return t;
    }

    const empty_map: [0]u8 align(std.heap.page_size_min) = .{};

    /// Raw BF16 rows `row_ids`, in order, into `out` (`row_ids.len * dim * 2`
    /// bytes): the table's bytes, no conversion (a bits-16 table). A no-cache table reads through its row gather:
    /// whole aligned pages only (an unaligned read would leave its pages in the page cache), each distinct row once,
    /// in parallel at prompt widths.
    pub fn gatherRaw(self: *const NgramTable, row_ids: []const u32, out: []u8) !void {
        std.debug.assert(self.bits == 16);
        const rb: usize = @as(usize, self.dim) * 2;
        if (out.len != row_ids.len * rb) return error.NgramTableRegion;
        for (row_ids) |r| if (r >= self.rows) return error.NgramTableRegion;
        if (self.nocache) return self.row_gather.?.gather(row_ids, out);
        for (row_ids, 0..) |r, i| {
            const off = self.w_off + @as(usize, r) * rb;
            @memcpy(out[i * rb ..][0..rb], self.map[off..][0..rb]);
        }
    }

    /// Widths `mx.quantize` packs and `dequantRow` unpacks, plus 16 = raw
    /// BF16 rows (no scales/biases; `row` copies them out converted to f32).
    fn bitsSupported(bits: u32) bool {
        return switch (bits) {
            2, 3, 4, 5, 6, 8, 16 => true,
            else => false,
        };
    }

    const HeaderRegion = struct {
        rows: u64,
        cols: u64,
        start: u64, // relative to the data section, as the header spells it
        end: u64,

        fn overlaps(a: HeaderRegion, b: HeaderRegion) bool {
            return a.start < b.end and b.start < a.end;
        }
    };

    /// One header entry, every access checked and the region proven to hold exactly
    /// `rows x cols x elem` bytes inside the mapping.
    fn headerRegion(
        obj: std.json.ObjectMap,
        key: []const u8,
        dtype: []const u8,
        elem: u64,
        map_len: usize,
        data_off: usize,
    ) !HeaderRegion {
        const v = obj.get(key) orelse return error.NgramTableHeader;
        if (v != .object) return error.NgramTableHeader;
        const o = v.object;
        const dt = o.get("dtype") orelse return error.NgramTableHeader;
        if (dt != .string or !std.mem.eql(u8, dt.string, dtype)) return error.NgramTableHeader;
        const shape = o.get("shape") orelse return error.NgramTableHeader;
        if (shape != .array or shape.array.items.len != 2) return error.NgramTableHeader;
        if (shape.array.items[0] != .integer or shape.array.items[1] != .integer) return error.NgramTableHeader;
        const dofs = o.get("data_offsets") orelse return error.NgramTableHeader;
        if (dofs != .array or dofs.array.items.len != 2) return error.NgramTableHeader;
        if (dofs.array.items[0] != .integer or dofs.array.items[1] != .integer) return error.NgramTableHeader;

        const rows_i = shape.array.items[0].integer;
        const cols_i = shape.array.items[1].integer;
        const start_i = dofs.array.items[0].integer;
        const end_i = dofs.array.items[1].integer;
        if (rows_i <= 0 or cols_i <= 0 or start_i < 0 or end_i < start_i) return error.NgramTableRegion;
        const r: HeaderRegion = .{
            .rows = @intCast(rows_i),
            .cols = @intCast(cols_i),
            .start = @intCast(start_i),
            .end = @intCast(end_i),
        };
        if (r.cols > std.math.maxInt(u32) or r.rows > std.math.maxInt(u32)) return error.NgramTableRegion;
        const need = std.math.mul(u64, r.rows, r.cols * elem) catch return error.NgramTableRegion;
        if (r.end - r.start != need) return error.NgramTableRegion;
        const abs_end = std.math.add(u64, data_off, r.end) catch return error.NgramTableTruncated;
        if (abs_end > map_len) return error.NgramTableTruncated;
        return r;
    }

    /// Absent `format` stamp is accepted once, loudly; a different format is a refusal.
    var stamp_warned: bool = false;

    pub fn parse(map: []align(std.heap.page_size_min) const u8, header: []const u8, data_off: usize) !NgramTable {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return error.NgramTableHeader;
        if (parsed != .object) return error.NgramTableHeader;
        const obj = parsed.object;
        const meta_v = obj.get("__metadata__") orelse return error.NgramTableHeader;
        if (meta_v != .object) return error.NgramTableHeader;
        const meta = meta_v.object;
        if (meta.get("format")) |f| {
            if (f != .string or !std.mem.eql(u8, f.string, "mlx-serve-ngram")) return error.NgramTableHeader;
        } else if (!stamp_warned) {
            stamp_warned = true;
            log.info("[qwen4] ngram table has no \"format\" stamp (written before the converter added it); accepting\n", .{});
        }
        const bits_v = meta.get("bits") orelse return error.NgramTableHeader;
        const gs_v = meta.get("group_size") orelse return error.NgramTableHeader;
        if (bits_v != .string or gs_v != .string) return error.NgramTableHeader;
        const bits: u32 = std.fmt.parseInt(u32, bits_v.string, 10) catch return error.NgramTableHeader;
        const gs: u32 = std.fmt.parseInt(u32, gs_v.string, 10) catch return error.NgramTableHeader;
        if (!bitsSupported(bits)) return error.NgramTableBits;

        // Raw BF16 mode: one `weight` BF16 [rows, dim] region, no
        // scales/biases. `wcols`/`scols` stay 0; only `row`'s raw arm reads it.
        if (bits == 16) {
            const w = try headerRegion(obj, "weight", "BF16", 2, map.len, data_off);
            if (w.cols > std.math.maxInt(u32)) return error.NgramTableRegion;
            return .{
                .map = map,
                .rows = w.rows,
                .dim = @intCast(w.cols),
                .bits = bits,
                .group_size = gs,
                .w_off = data_off + @as(usize, @intCast(w.start)),
                .s_off = 0,
                .b_off = 0,
                .wcols = 0,
                .scols = 0,
            };
        }

        if (gs == 0 or gs > 1024) return error.NgramTableBits;

        const w = try headerRegion(obj, "weight", "U32", 4, map.len, data_off);
        const sc = try headerRegion(obj, "scales", "BF16", 2, map.len, data_off);
        const bi = try headerRegion(obj, "biases", "BF16", 2, map.len, data_off);
        // The three regions describe the same rows and may not overlap.
        if (sc.rows != w.rows or bi.rows != w.rows or sc.cols != bi.cols) return error.NgramTableRegion;
        if (w.overlaps(sc) or w.overlaps(bi) or sc.overlaps(bi)) return error.NgramTableRegion;

        const dim: u64 = sc.cols * gs;
        if (dim > std.math.maxInt(u32)) return error.NgramTableRegion;
        if (dim * bits != w.cols * 32) return error.NgramTableHeader;

        return .{
            .map = map,
            .rows = w.rows,
            .dim = @intCast(dim),
            .bits = bits,
            .group_size = gs,
            .w_off = data_off + @as(usize, @intCast(w.start)),
            .s_off = data_off + @as(usize, @intCast(sc.start)),
            .b_off = data_off + @as(usize, @intCast(bi.start)),
            .wcols = @intCast(w.cols),
            .scols = @intCast(sc.cols),
        };
    }

    pub fn close(self: *NgramTable) void {
        if (self.poster) |p| p.destroy();
        self.poster = null;
        if (self.warm_thread) |th| {
            self.warm_stop.store(true, .release);
            th.join();
            self.warm_thread = null;
        }
        live_warm_bytes.store(0, .release);
        live_warm_total.store(0, .release);
        if (self.pool) |p| p.destroy();
        self.pool = null;
        if (self.row_gather) |g| g.deinit();
        self.row_gather = null;
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
        if (self.cache) |c| {
            const gpa = c.gpa;
            c.deinit();
            gpa.destroy(c);
            self.cache = null;
        }
        if (!self.gpu_owns_map and !self.nocache) std.posix.munmap(self.map);
    }

    const WARM_CHUNK: usize = 8 << 20;

    /// Read the whole table through the fd once, in the background, so the
    /// first prompt's PLE gathers hit a warm page cache. Call only once the
    /// table sits at its final address (the thread holds `self`). Off via
    /// MLX_SERVE_NGRAM_WARM=0.
    pub fn startWarm(self: *NgramTable) void {
        if (self.fd < 0 or self.warm_thread != null or self.nocache) return;
        // The off arm says so: a cold first request faults rows off the SSD (38k prompt: 174 s vs 55 s).
        if (!warmEnabled()) {
            log.info("[qwen4] ngram table warm: disabled (MLX_SERVE_NGRAM_WARM=0) - the first long prompt faults the table in from SSD\n", .{});
            return;
        }
        self.warm_stop.store(false, .release);
        self.warm_bytes.store(0, .release);
        live_warm_bytes.store(0, .release);
        live_warm_total.store(self.map.len, .release);
        log.info("[qwen4] ngram table warm: started, {d:.1} GB in the background (page cache; MLX_SERVE_NGRAM_WARM=0 disables)\n", .{asGb(self.map.len)});
        self.warm_thread = std.Thread.spawn(.{}, warmMain, .{self}) catch null;
    }

    fn warmMain(self: *NgramTable) void {
        var scratch: [WARM_CHUNK]u8 align(16) = undefined;
        const wio = std.Io.Threaded.global_single_threaded.io();
        const t0 = std.Io.Timestamp.now(wio, .boot);
        var off: u64 = 0;
        const total: u64 = self.map.len;
        var prog: WarmProgress = .{};
        while (off < total) {
            if (self.warm_stop.load(.acquire)) return;
            const want: usize = @intCast(@min(total - off, WARM_CHUNK));
            const got = std.c.pread(self.fd, &scratch, want, @intCast(off));
            if (got <= 0) return;
            off += @intCast(got);
            self.warm_bytes.store(off, .release);
            live_warm_bytes.store(off, .release);
            // One clock read per 8 MB pread is free next to the read itself.
            const el: u64 = @intCast(t0.untilNow(wio, .boot).nanoseconds);
            if (prog.should(off, el)) log.info("[qwen4] ngram table warm: {d:.1}/{d:.1} GB after {d:.0} s\n", .{ asGb(off), asGb(total), @as(f64, @floatFromInt(el)) / 1e9 });
        }
        const secs: f64 = @as(f64, @floatFromInt(t0.untilNow(wio, .boot).nanoseconds)) / 1e9;
        log.info("[qwen4] ngram table warm: done, {d:.1} GB in {d:.1} s (page cache; MLX_SERVE_NGRAM_WARM=0 disables)\n", .{ asGb(total), secs });
    }

    /// Dequantize one row into `out[0..dim]` (mx.quantize packing: element i
    /// sits at bit offset i * bits of the little-endian u32 stream and may
    /// straddle a word boundary at 3/5/6 bits).
    pub fn row(self: *const NgramTable, r: u64, out: []f32) void {
        std.debug.assert(r < self.rows and out.len >= self.dim and !self.nocache);
        // Raw BF16 arm: straight convert, no scales/biases.
        if (self.bits == 16) {
            const raw = self.map[self.w_off + r * self.dim * 2 ..][0 .. self.dim * 2];
            var i: u32 = 0;
            while (i < self.dim) : (i += 1) {
                out[i] = bf16ToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            }
            return;
        }
        const words = self.map[self.w_off + r * self.wcols * 4 ..][0 .. self.wcols * 4];
        const scales = self.map[self.s_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        const biases = self.map[self.b_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        self.dequantRow(words, scales, biases, out);
    }

    fn dequantRow(self: *const NgramTable, words: []const u8, scales: []const u8, biases: []const u8, out: []f32) void {
        const mask: u32 = (@as(u32, 1) << @intCast(self.bits)) - 1;
        var i: u32 = 0;
        while (i < self.dim) : (i += 1) {
            const off = i * self.bits;
            const w = off / 32;
            const shift = off % 32;
            var v: u64 = std.mem.readInt(u32, words[w * 4 ..][0..4], .little);
            if (shift + self.bits > 32) v |= @as(u64, std.mem.readInt(u32, words[w * 4 + 4 ..][0..4], .little)) << 32;
            const q: u32 = @truncate((v >> @intCast(shift)) & mask);
            const g = i / self.group_size;
            const sc = bf16ToF32(std.mem.readInt(u16, scales[g * 2 ..][0..2], .little));
            const bi = bf16ToF32(std.mem.readInt(u16, biases[g * 2 ..][0..2], .little));
            out[i] = @as(f32, @floatFromInt(q)) * sc + bi;
        }
    }

    /// Gather + concatenate the `n_heads` rows of each token: `out` is
    /// `[ids.len / n_heads][n_heads * dim]` row-major. `kv_len` is the context position this
    /// gather runs at and picks the wide arm; the output is byte-identical either way.
    pub fn gather(self: *const NgramTable, row_ids: []const i64, out: []f32, kv_len: u64) void {
        // Raw BF16 arm rides no pool (single region, no dequant): serial walk.
        if (self.bits == 16) {
            for (row_ids, 0..) |r, i| self.row(@intCast(r), out[i * self.dim ..][0..self.dim]);
            return;
        }
        const need: usize = self.wcols * 4 + self.scols * 4;
        // Prefill-width gathers ride the pool only past `PREFILL_PREFETCH_MIN_KV`: a resident
        // table loses 2-7% to the wake rounds, an evicted one (weights pushed the 32 GB
        // mapping out) went 67.7 -> 267.9 ms per 1000 tokens on the serial walk.
        const wide = row_ids.len > PrefetchPool.MAX_ROWS;
        const wide_ok = !wide or plePrefillPrefetchEnabled(kv_len);
        // Announce the arm that actually runs, not the lever that permits it.
        const pooled = wide_ok and self.pool != null and self.fd >= 0 and need <= PrefetchPool.ROW_BUF;
        if (wide) notePrefillGatherArm(pooled, row_ids.len);
        if (self.pool) |p| if (self.fd >= 0 and need <= PrefetchPool.ROW_BUF and wide_ok) {
            const wl: usize = self.wcols * 4;
            const sl: usize = self.scols * 2;
            var start: usize = 0;
            while (start < row_ids.len) : (start += PrefetchPool.MAX_ROWS) {
                const end = @min(start + PrefetchPool.MAX_ROWS, row_ids.len);
                if (!p.run(self, row_ids[start..end])) break;
                for (start..end) |i| {
                    const b = &p.bufs[i - start];
                    self.dequantRow(b[0..wl], b[wl .. wl + sl], b[wl + sl .. wl + 2 * sl], out[i * self.dim ..][0..self.dim]);
                }
            }
            if (start >= row_ids.len) return;
        };
        for (row_ids, 0..) |r, i| self.row(@intCast(r), out[i * self.dim ..][0..self.dim]);
    }

    /// One (row, region) pread into the pool's row buffer. False on a short read.
    fn preadSite(self: *const NgramTable, r: u64, region: usize, buf: []u8) bool {
        const wl: usize = if (self.bits == records_bits) self.dim else self.wcols * 4;
        const sl: usize = self.scols * 2;
        const off: usize, const dst: []u8 = switch (region) {
            0 => .{ self.w_off + r * wl, buf[0..wl] },
            1 => .{ self.s_off + r * sl, buf[wl .. wl + sl] },
            else => .{ self.b_off + r * sl, buf[wl + sl .. wl + 2 * sl] },
        };
        return std.c.pread(self.fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }

};

/// A byte-budgeted LRU of a record table's rows, Python's `NGramRowCache` rule for rule: a
/// gather's distinct rows in first-appearance order, a hit made newest, the misses read and
/// entered sorted (sub-runs capped at the slot count), each taking the highest free slot or the oldest row's.
pub const RowCache = struct {
    pub const Stats = struct { hits: u64 = 0, misses: u64 = 0, evictions: u64 = 0, reads: u64 = 0, rows_read: u64 = 0, gathers: u64 = 0 };
    pub const nil: u32 = std.math.maxInt(u32);
    const Miss = struct { row: u64, d: u32 };

    gpa: std.mem.Allocator,
    record_bytes: u32,
    slot_count: u32,
    arena: []u8,
    slot_row: []u64,
    prev: []u32,
    next: []u32,
    free: []u32,
    n_free: u32,
    oldest: u32 = nil,
    newest: u32 = nil,
    index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// Removals since the index was last rehashed (its tombstones).
    removed: u32 = 0,
    stats: Stats = .{},
    // One gather's scratch, kept across gathers.
    first: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    distinct: std.ArrayList(u64) = .empty,
    chain_head: std.ArrayList(u32) = .empty,
    chain_tail: std.ArrayList(u32) = .empty,
    chain_next: std.ArrayList(u32) = .empty,
    misses: std.ArrayList(Miss) = .empty,
    miss_rows: std.ArrayList(i64) = .empty,
    miss_recs: std.ArrayList(u8) = .empty,

    /// Python's slot rule: `max(1, max(record_bytes, budget) // record_bytes)`.
    pub fn slotCount(record_bytes: u32, budget_bytes: u64) u32 {
        return @intCast(@max(1, @max(record_bytes, budget_bytes) / record_bytes));
    }

    /// The row index's capacity for `slots` rows (the std map's 80 % load rule, minimum 8).
    pub fn indexCapacity(slots: u32) u32 {
        return @max(8, std.math.ceilPowerOfTwo(u32, @intCast(@as(u64, slots) * 100 / 80 + 1)) catch unreachable);
    }

    /// Host bytes the cache holds for its whole life (a memory bill's term): the slot
    /// arena, the per-slot links and the row index.
    pub fn hostBytes(record_bytes: u32, budget_bytes: u64) u64 {
        const slots = slotCount(record_bytes, budget_bytes);
        return @as(u64, slots) * (record_bytes + @sizeOf(u64) + 3 * @sizeOf(u32)) + @as(u64, indexCapacity(slots)) * (@sizeOf(u64) + @sizeOf(u32) + 1);
    }

    pub fn init(gpa: std.mem.Allocator, record_bytes: u32, budget_bytes: u64) !RowCache {
        const slots = slotCount(record_bytes, budget_bytes);
        const arena = try gpa.alloc(u8, @as(usize, slots) * record_bytes);
        errdefer gpa.free(arena);
        const slot_row = try gpa.alloc(u64, slots);
        errdefer gpa.free(slot_row);
        const prev = try gpa.alloc(u32, slots);
        errdefer gpa.free(prev);
        const next = try gpa.alloc(u32, slots);
        errdefer gpa.free(next);
        const free = try gpa.alloc(u32, slots);
        errdefer gpa.free(free);
        for (free, 0..) |*f, i| f.* = @intCast(i);
        var self: RowCache = .{ .gpa = gpa, .record_bytes = record_bytes, .slot_count = slots, .arena = arena, .slot_row = slot_row, .prev = prev, .next = next, .free = free, .n_free = slots };
        try self.index.ensureTotalCapacity(gpa, slots);
        return self;
    }

    pub fn deinit(self: *RowCache) void {
        const a = self.gpa;
        a.free(self.arena);
        a.free(self.slot_row);
        a.free(self.prev);
        a.free(self.next);
        a.free(self.free);
        self.index.deinit(a);
        self.first.deinit(a);
        self.distinct.deinit(a);
        self.chain_head.deinit(a);
        self.chain_tail.deinit(a);
        self.chain_next.deinit(a);
        self.misses.deinit(a);
        self.miss_rows.deinit(a);
        self.miss_recs.deinit(a);
    }

    fn unlink(self: *RowCache, s: u32) void {
        const p = self.prev[s];
        const n = self.next[s];
        if (p != nil) self.next[p] = n else self.oldest = n;
        if (n != nil) self.prev[n] = p else self.newest = p;
    }

    fn linkNewest(self: *RowCache, s: u32) void {
        self.prev[s] = self.newest;
        self.next[s] = nil;
        if (self.newest != nil) self.next[self.newest] = s else self.oldest = s;
        self.newest = s;
    }

    /// Python `_alloc_slot`: `list.pop()` of the free list, else the oldest row is evicted.
    fn allocSlot(self: *RowCache) u32 {
        if (self.n_free > 0) {
            self.n_free -= 1;
            return self.free[self.n_free];
        }
        const s = self.oldest;
        _ = self.index.remove(self.slot_row[s]);
        self.removed += 1;
        self.unlink(s);
        self.stats.evictions += 1;
        return s;
    }

    fn lessRow(_: void, x: Miss, y: Miss) bool {
        return x.row < y.row;
    }

    /// `rec` into every position of distinct row `d` in `out`.
    fn put(self: *const RowCache, out: []u8, d: u32, rec: []const u8) void {
        const rb: usize = self.record_bytes;
        var i = self.chain_head.items[d];
        while (i != nil) : (i = self.chain_next.items[i]) @memcpy(out[@as(usize, i) * rb ..][0..rb], rec);
    }

    /// Python `gather_bytes` over `table` (rows already range-checked).
    fn gather(self: *RowCache, table: *const NgramTable, rows: []const i64, out: []u8) !void {
        const a = self.gpa;
        const rb: usize = self.record_bytes;
        self.first.clearRetainingCapacity();
        self.distinct.clearRetainingCapacity();
        self.chain_head.clearRetainingCapacity();
        self.chain_tail.clearRetainingCapacity();
        try self.chain_next.resize(a, rows.len);
        for (rows, 0..) |r, i| {
            const gop = try self.first.getOrPut(a, @intCast(r));
            self.chain_next.items[i] = nil;
            if (gop.found_existing) {
                const d = gop.value_ptr.*;
                self.chain_next.items[self.chain_tail.items[d]] = @intCast(i);
                self.chain_tail.items[d] = @intCast(i);
            } else {
                gop.value_ptr.* = @intCast(self.distinct.items.len);
                try self.distinct.append(a, @intCast(r));
                try self.chain_head.append(a, @intCast(i));
                try self.chain_tail.append(a, @intCast(i));
            }
        }
        self.misses.clearRetainingCapacity();
        for (self.distinct.items, 0..) |row, d| {
            if (self.index.get(row)) |slot| {
                self.unlink(slot);
                self.linkNewest(slot);
                self.stats.hits += 1;
                self.put(out, @intCast(d), self.arena[@as(usize, slot) * rb ..][0..rb]);
            } else try self.misses.append(a, .{ .row = row, .d = @intCast(d) });
        }
        const ms = self.misses.items;
        if (ms.len > 0) {
            std.mem.sort(Miss, ms, {}, lessRow);
            try self.miss_rows.resize(a, ms.len);
            for (ms, self.miss_rows.items) |m, *r| r.* = @intCast(m.row);
            try self.miss_recs.resize(a, ms.len * rb);
            try table.readRecords(self.miss_rows.items, self.miss_recs.items);
        }
        var i: usize = 0;
        while (i < ms.len) {
            var j = i + 1;
            while (j < ms.len and ms[j].row == ms[j - 1].row + 1) j += 1;
            var off = i;
            while (off < j) {
                const n: usize = @min(j - off, self.slot_count);
                self.stats.reads += 1;
                self.stats.rows_read += n;
                self.stats.misses += n;
                for (ms[off..][0..n], off..) |m, k| {
                    const slot = self.allocSlot();
                    const rec = self.miss_recs.items[k * rb ..][0..rb];
                    @memcpy(self.arena[@as(usize, slot) * rb ..][0..rb], rec);
                    self.slot_row[slot] = m.row;
                    self.index.putAssumeCapacity(m.row, slot);
                    self.linkNewest(slot);
                    self.put(out, m.d, rec);
                }
                off += n;
            }
            i = j;
        }
        self.stats.gathers += 1;
        // Evictions leave tombstones in the index: clear them once they reach half the slots.
        if (self.removed >= self.slot_count / 2 + 1) {
            self.index.rehash(std.hash_map.AutoContext(u64){});
            self.removed = 0;
        }
    }
};

/// One table's posted gathers: a thread that runs them in post order (each is `gatherRecords`, so the
/// pool's rounds and the cache's order are the blocking path's). Jobs are drained before it stops.
const Poster = struct {
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    head: ?*NgramTable.Posted = null,
    tail: ?*NgramTable.Posted = null,
    quit: bool = false,
    table: *NgramTable,
    thread: std.Thread = undefined,

    fn create(table: *NgramTable) !*Poster {
        const a = std.heap.page_allocator;
        const p = try a.create(Poster);
        errdefer a.destroy(p);
        p.* = .{ .table = table };
        p.thread = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, run, .{p});
        return p;
    }

    fn destroy(self: *Poster) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        self.thread.join();
        std.heap.page_allocator.destroy(self);
    }

    fn push(self: *Poster, job: *NgramTable.Posted) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        job.state = .queued;
        job.next = null;
        self.mu.lockUncancelable(io);
        if (self.tail) |t| t.next = job else self.head = job;
        self.tail = job;
        self.cv.broadcast(io);
        self.mu.unlock(io);
    }

    fn await(self: *Poster, job: *NgramTable.Posted) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        while (job.state == .queued) self.cv.wait(io, &self.mu) catch {};
        const failed = job.state == .failed;
        self.mu.unlock(io);
        if (failed) return job.err;
    }

    fn run(self: *Poster) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.head == null and !self.quit) self.cv.wait(io, &self.mu) catch {};
            const job = self.head orelse {
                self.mu.unlock(io);
                return;
            };
            self.head = job.next;
            if (self.head == null) self.tail = null;
            self.mu.unlock(io);
            const r = self.table.gatherRecords(job.rows, job.out);
            self.mu.lockUncancelable(io);
            if (r) |_| job.state = .done else |e| {
                job.err = e;
                job.state = .failed;
            }
            self.cv.broadcast(io);
            self.mu.unlock(io);
        }
    }
};

/// Persistent gather workers. Every row's three regions are one SSD read on
/// the cold 32 GB table (~100 us), 48 per token: serial mmap faults were ~5 ms
/// of every decode step, 16 fault threads ~0.7 ms, and more threads got SLOWER
/// (faults on one mapping serialize on the VM map lock), so workers `pread`
/// instead and dequantize their rows in place. Workers wake on a generation
/// bump and count themselves down; the caller spins (the job is ~100 us).
pub const PrefetchPool = struct {
    const N = 48;
    pub const MAX_ROWS = 64;
    const ROW_BUF = 512;
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    gen: u64 = 0,
    quit: bool = false,
    table: ?*const NgramTable = null,
    rows: []const i64 = &.{},
    bufs: [MAX_ROWS][ROW_BUF]u8 = undefined,
    pending: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    /// Fan-out rounds issued; the engagement counter the prefill test reads.
    runs: std.atomic.Value(u64) = .init(0),
    threads: [N]std.Thread = undefined,

    fn create() !*PrefetchPool {
        const a = std.heap.page_allocator;
        const p = try a.create(PrefetchPool);
        p.* = .{};
        var started: usize = 0;
        errdefer {
            p.shutdown(started);
            a.destroy(p);
        }
        for (0..N) |i| {
            p.threads[i] = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, worker, .{ p, i });
            started += 1;
        }
        return p;
    }

    pub fn destroy(self: *PrefetchPool) void {
        self.shutdown(N);
        std.heap.page_allocator.destroy(self);
    }

    fn shutdown(self: *PrefetchPool, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |t| t.join();
    }

    /// Fan the `regions * rows.len` preads over the workers; rows land in `bufs`.
    fn run(self: *PrefetchPool, table: *const NgramTable, rows: []const i64) bool {
        _ = self.runs.fetchAdd(1, .monotonic);
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.table = table;
        self.rows = rows;
        self.failed.store(0, .release);
        self.pending.store(N, .release);
        self.gen += 1;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
        return self.failed.load(.acquire) == 0;
    }

    fn worker(self: *PrefetchPool, idx: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        var seen: u64 = 0;
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
            if (self.quit) {
                self.mu.unlock(io);
                return;
            }
            seen = self.gen;
            const table = self.table.?;
            const rows = self.rows;
            self.mu.unlock(io);
            const k = table.regions();
            var i = idx;
            while (i < rows.len * k) : (i += N) {
                if (!table.preadSite(@intCast(rows[i / k]), i % k, &self.bufs[i / k])) _ = self.failed.fetchAdd(1, .acq_rel);
            }
            _ = self.pending.fetchSub(1, .acq_rel);
        }
    }
};

fn plePrefetchEnabled() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v) |v| return v;
    const raw = std.c.getenv("QWEN4_PLE_PREFETCH");
    const v = raw == null or raw.?[0] != '0';
    S.v = v;
    return v;
}

/// One-shot engagement lines per arm; both arms say something.
/// Narrowest gather that is a genuine prefill chunk rather than a warmup forward.
pub const PREFILL_SAY_MIN_ROWS: usize = 1024;

/// [arm][bucket]: arm 0/1 = serial/pooled, bucket 0/1 = warmup/prefill width.
pub var ple_prefill_arm_said: [2][2]std.atomic.Value(bool) =
    .{ .{ .init(false), .init(false) }, .{ .init(false), .init(false) } };

fn notePrefillGatherArm(pooled: bool, rows: usize) void {
    const arm: usize = if (pooled) 1 else 0;
    const bucket: usize = if (rows >= PREFILL_SAY_MIN_ROWS) 1 else 0;
    if (ple_prefill_arm_said[arm][bucket].swap(true, .monotonic)) return;
    const width: []const u8 = if (bucket == 1) "prefill width" else "warmup width";
    if (pooled) {
        const batches = (rows + PrefetchPool.MAX_ROWS - 1) / PrefetchPool.MAX_ROWS;
        log.info("[qwen4] PLE prefill gather: POOLED ({s}: {d} rows, {d} batches of {d}; past kv {d}, QWEN4_PLE_PREFETCH_PREFILL=0 forces the serial walk)\n", .{ width, rows, batches, PrefetchPool.MAX_ROWS, plePrefillPrefetchMinKv() });
    } else {
        log.info("[qwen4] PLE prefill gather: SERIAL mmap walk ({s}: {d} rows; the pool engages past kv {d}, QWEN4_PLE_PREFETCH_PREFILL_MIN_KV overrides)\n", .{ width, rows, plePrefillPrefetchMinKv() });
    }
}

/// Test seams for the prefill gate below (both envs are read once per process).
pub var ple_prefill_prefetch_override: ?bool = null;
pub var ple_prefill_min_kv_override: ?u64 = null;

/// The kv length past which a wide prefill gather takes the pool: the top of the measured
/// cost range (the pool loses at every rung to 256k on a resident table; the only win is the
/// evicted table on the 374k ladder). `QWEN4_PLE_PREFETCH_PREFILL_MIN_KV` overrides.
pub const PREFILL_PREFETCH_MIN_KV: u64 = 262144;

pub const PrefillPrefetchMode = enum { off, kv_gated, on };

/// `QWEN4_PLE_PREFETCH_PREFILL`: absent = the kv gate, `0` = serial walk, `1` = pool.
pub fn plePrefillPrefetchModeFromEnv(raw: ?[]const u8) PrefillPrefetchMode {
    const r = raw orelse return .kv_gated;
    if (r.len == 0) return .kv_gated;
    if (r[0] == '0') return .off;
    if (r[0] == '1') return .on;
    return .kv_gated;
}

/// The threshold, or the constant when the override is absent or unparsable.
pub fn plePrefillPrefetchMinKvFromEnv(raw: ?[]const u8) u64 {
    const r = raw orelse return PREFILL_PREFETCH_MIN_KV;
    const t = std.mem.trim(u8, r, " \t");
    if (t.len == 0) return PREFILL_PREFETCH_MIN_KV;
    return std.fmt.parseInt(u64, t, 10) catch PREFILL_PREFETCH_MIN_KV;
}

pub fn plePrefillPrefetchWanted(mode: PrefillPrefetchMode, kv_len: u64, min_kv: u64) bool {
    return switch (mode) {
        .off => false,
        .on => true,
        .kv_gated => kv_len >= min_kv,
    };
}

fn plePrefillPrefetchMinKv() u64 {
    if (ple_prefill_min_kv_override) |v| return v;
    const S = struct {
        var v: ?u64 = null;
    };
    if (S.v) |v| return v;
    const raw = std.c.getenv("QWEN4_PLE_PREFETCH_PREFILL_MIN_KV");
    const v = plePrefillPrefetchMinKvFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    S.v = v;
    return v;
}

fn plePrefillPrefetchEnabled(kv_len: u64) bool {
    if (ple_prefill_prefetch_override) |v| return v;
    const S = struct {
        var v: ?PrefillPrefetchMode = null;
    };
    const mode = S.v orelse blk: {
        const raw = std.c.getenv("QWEN4_PLE_PREFETCH_PREFILL");
        const m = plePrefillPrefetchModeFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
        S.v = m;
        break :blk m;
    };
    return plePrefillPrefetchWanted(mode, kv_len, plePrefillPrefetchMinKv());
}

pub fn bf16ToF32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

/// Round-to-nearest-even f32 -> bf16 bits, the PLE rows' upload format.
pub fn bf16Rne(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}
