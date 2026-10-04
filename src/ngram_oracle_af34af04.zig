//! TEST ORACLE ONLY (never imported outside tests): mlx-serve upstream main af34af04's n-gram code from
//! `src/qwen4_exp.zig` lines 14-702, verbatim except: `log` comes through `sdk`, the unused `ple_gpu.zig` import
//! is dropped, and `NgramTable.parse` is `pub`. ngram_table_parity_test.zig runs it beside this plugin's copy
//! (ngram_table.zig) on the same inputs and requires the same results.

const std = @import("std");
const log = @import("sdk").log;

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
        if (self.warm_thread) |th| {
            self.warm_stop.store(true, .release);
            th.join();
            self.warm_thread = null;
        }
        live_warm_bytes.store(0, .release);
        live_warm_total.store(0, .release);
        if (self.pool) |p| p.destroy();
        self.pool = null;
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
        if (!self.gpu_owns_map) std.posix.munmap(self.map);
    }

    const WARM_CHUNK: usize = 8 << 20;

    /// Read the whole table through the fd once, in the background, so the
    /// first prompt's PLE gathers hit a warm page cache. Call only once the
    /// table sits at its final address (the thread holds `self`). Off via
    /// MLX_SERVE_NGRAM_WARM=0.
    pub fn startWarm(self: *NgramTable) void {
        if (self.fd < 0 or self.warm_thread != null) return;
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
        std.debug.assert(r < self.rows and out.len >= self.dim);
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
        const wl: usize = self.wcols * 4;
        const sl: usize = self.scols * 2;
        const off: usize, const dst: []u8 = switch (region) {
            0 => .{ self.w_off + r * wl, buf[0..wl] },
            1 => .{ self.s_off + r * sl, buf[wl .. wl + sl] },
            else => .{ self.b_off + r * sl, buf[wl + sl .. wl + 2 * sl] },
        };
        return std.c.pread(self.fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }

};

/// Persistent gather workers. Every row's three regions are one SSD read on
/// the cold 32 GB table (~100 us), 48 per token: serial mmap faults were ~5 ms
/// of every decode step, 16 fault threads ~0.7 ms, and more threads got SLOWER
/// (faults on one mapping serialize on the VM map lock), so workers `pread`
/// instead and dequantize their rows in place. Workers wake on a generation
/// bump and count themselves down; the caller spins (the job is ~100 us).
const PrefetchPool = struct {
    const N = 48;
    const MAX_ROWS = 64;
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

    fn destroy(self: *PrefetchPool) void {
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

    /// Fan the `3 * rows.len` preads over the workers; rows land in `bufs`.
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
            var i = idx;
            while (i < rows.len * 3) : (i += N) {
                if (!table.preadSite(@intCast(rows[i / 3]), i % 3, &self.bufs[i / 3])) _ = self.failed.fetchAdd(1, .acq_rel);
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

