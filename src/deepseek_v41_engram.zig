//! Engram row ids and bank rows for DeepSeek-V4.1 (layers 1 and 14), host
//! side: the manifest's hashing recipe (Python `NgramHashState`: rolling XOR
//! of the n-gram's compressed ids times per-layer multipliers, mod per-head
//! primes, plus flat offsets) and the 264-byte mxfp8 records it indexes. The
//! compressed token map is a converter output (tokenizer normalisation stays
//! Python); the rows' dequantize and the gated add are MLX (`engramApply`).

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const ngram = @import("ngram_table.zig");
const io_util = @import("sdk").io_util;

pub const max_ngram = 8;
pub const max_heads = 16;
pub const max_cols = (max_ngram - 1) * max_heads;
pub const max_layers = 8;
/// A masked position (an image span): no n-gram may span it.
pub const dead: i64 = -1;

pub const Hashing = struct {
    max_ngram: u32,
    n_heads: u32,
    n_layers: u32,
    layer_ids: [max_layers]u32 = @splat(0),
    multipliers: [max_layers][max_ngram]i64 = @splat(@splat(0)),
    primes: [max_layers][max_ngram - 1][max_heads]i64 = @splat(@splat(@splat(0))),
    flat_offsets: [max_layers][max_cols]i64 = @splat(@splat(0)),
    total_rows: [max_layers]u64 = @splat(0),
    pad_id: u32,
    compressed_vocab: u32,

    pub fn cols(self: *const Hashing) u32 {
        return (self.max_ngram - 1) * self.n_heads;
    }
};

pub const Bank = struct {
    head_dim: u32,
    record_bytes: u32,
    files: [max_layers][]const u8,
    /// Records per layer file (the manifest's `rows`).
    rows: [max_layers]u64 = @splat(0),
    /// The manifest's own identity field; a token map names the manifest it was built for.
    manifest_sha256: []const u8 = "",
};

/// `engram-manifest.json` checked against the config's Engram fields; any
/// codec, geometry or table this build does not implement is refused.
pub fn parseManifest(a: std.mem.Allocator, text: []const u8, c: *const v41.Config, diag: ?*v41.Diag) !struct { hashing: Hashing, bank: Bank } {
    const Layer = struct {
        layer_id: u32,
        file: []const u8,
        rows: u64,
        record_bytes: u32,
        quant: struct { bits: u32, group_size: u32, mode: []const u8, head_dim: u32 },
    };
    const PerLayer = struct { layer_id: u32, primes: []const []const i64, flat_offsets: []const i64, total_rows: u64 };
    const M = struct {
        format: []const u8,
        manifest_sha256: []const u8 = "",
        layers: []const Layer,
        hashing: struct {
            layer_ids: []const u32,
            max_ngram_size: u32,
            n_heads: u32,
            head_dim: u32,
            compressed_vocab_size: u32,
            pad_id: u32,
            n_hash_cols: u32,
            hash_multipliers: []const []const i64,
            per_layer: []const PerLayer,
        },
    };
    const m = std.json.parseFromSliceLeaky(M, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return fail(diag, "engram manifest: {s}", .{@errorName(e)}),
    };
    if (!std.mem.eql(u8, m.format, "mtplx-engram-manifest-v1")) return fail(diag, "engram manifest: format {s}", .{m.format});
    const h = m.hashing;
    const e = c.engram;
    if (h.max_ngram_size != e.max_ngram_size or h.n_heads != e.n_heads or h.head_dim != e.head_dim or
        h.compressed_vocab_size != e.compressed_vocab_size or h.pad_id != e.pad_token_id or h.n_hash_cols != e.hashCols())
        return fail(diag, "engram manifest: hashing geometry differs from config.json", .{});
    if (h.max_ngram_size < 2 or h.max_ngram_size > max_ngram or h.n_heads > max_heads) return fail(diag, "engram manifest: {d}-grams x {d} heads not implemented", .{ h.max_ngram_size, h.n_heads });
    if (h.layer_ids.len != e.n_layers or h.per_layer.len != e.n_layers or h.hash_multipliers.len != e.n_layers or m.layers.len != e.n_layers)
        return fail(diag, "engram manifest: {d} layers, config has {d}", .{ h.layer_ids.len, e.n_layers });
    var hs: Hashing = .{ .max_ngram = h.max_ngram_size, .n_heads = h.n_heads, .n_layers = e.n_layers, .pad_id = h.pad_id, .compressed_vocab = h.compressed_vocab_size };
    var bank: Bank = .{ .head_dim = h.head_dim, .record_bytes = h.head_dim + h.head_dim / 32, .files = undefined, .manifest_sha256 = m.manifest_sha256 };
    for (0..e.n_layers) |i| {
        const lid = h.layer_ids[i];
        const pl = h.per_layer[i];
        const ly = m.layers[i];
        if (lid != e.layer_ids[i] or pl.layer_id != lid or ly.layer_id != lid) return fail(diag, "engram manifest: layer order differs from config.json at {d}", .{i});
        if (pl.total_rows != e.num_embeddings[i] or ly.rows != pl.total_rows) return fail(diag, "engram layer {d}: {d} rows, config has {d}", .{ lid, pl.total_rows, e.num_embeddings[i] });
        if (!std.mem.eql(u8, ly.quant.mode, "mxfp8") or ly.quant.bits != 8 or ly.quant.group_size != 32 or ly.quant.head_dim != h.head_dim or ly.record_bytes != bank.record_bytes)
            return fail(diag, "engram layer {d}: codec {s} bits {d} group {d} record {d} not implemented", .{ lid, ly.quant.mode, ly.quant.bits, ly.quant.group_size, ly.record_bytes });
        if (h.hash_multipliers[i].len != h.max_ngram_size or pl.primes.len != h.max_ngram_size - 1 or pl.flat_offsets.len != hs.cols())
            return fail(diag, "engram layer {d}: hash tables have the wrong shape", .{lid});
        hs.layer_ids[i] = lid;
        hs.total_rows[i] = pl.total_rows;
        for (h.hash_multipliers[i], 0..) |v, k| hs.multipliers[i][k] = v;
        for (pl.primes, 0..) |row, k| {
            if (row.len != h.n_heads) return fail(diag, "engram layer {d}: primes row {d} has {d} heads", .{ lid, k, row.len });
            for (row, 0..) |p, hd| {
                if (p <= 0) return fail(diag, "engram layer {d}: prime {d} is not positive", .{ lid, p });
                hs.primes[i][k][hd] = p;
            }
        }
        for (pl.flat_offsets, 0..) |o, k| hs.flat_offsets[i][k] = o;
        bank.files[i] = ly.file;
        bank.rows[i] = ly.rows;
    }
    return .{ .hashing = hs, .bank = bank };
}

fn fail(diag: ?*v41.Diag, comptime fmt: []const u8, args: anytype) error{EngramManifest} {
    if (diag) |d| d.set(fmt, args);
    return error.EngramManifest;
}

/// The per-sequence compressed-id history (Python `NgramHashState._buf`):
/// lookbacks cross the prefill / decode boundary; `trim` rewinds a rejected
/// draft so re-fed tokens hash identically.
pub const HashState = struct {
    hist: std.ArrayList(i64) = .empty,

    pub fn deinit(self: *HashState, gpa: std.mem.Allocator) void {
        self.hist.deinit(gpa);
    }

    pub fn trim(self: *HashState, n: usize) void {
        self.hist.shrinkRetainingCapacity(self.hist.items.len - n);
    }

    /// Feed `ids` (compressed through `map`; a masked position is `dead`) and
    /// write their row ids, `[len(ids)][n_layers][cols]`, into `out`.
    pub fn advance(self: *HashState, gpa: std.mem.Allocator, h: *const Hashing, map: []const u32, ids: []const u32, masked: ?[]const bool, out: []i64) !void {
        const cols = h.cols();
        if (out.len != ids.len * h.n_layers * cols) return error.RowsShape;
        const start = self.hist.items.len;
        for (ids, 0..) |id, i| {
            if (id >= map.len) return error.TokenOutOfMap;
            const dead_here = if (masked) |mk| mk[i] else false;
            try self.hist.append(gpa, if (dead_here) dead else map[id]);
        }
        const pad: i64 = map[h.pad_id];
        for (0..ids.len) |i| {
            const pos = start + i;
            var toks: [max_ngram]i64 = undefined;
            var blocked = false;
            for (0..h.max_ngram) |shift| {
                const src = self.hist.items[if (pos >= shift) pos - shift else 0];
                blocked = blocked or pos < shift or src == dead;
                toks[shift] = if (blocked) pad else src;
            }
            for (0..h.n_layers) |l| {
                var rolling: i64 = toks[0] *% h.multipliers[l][0];
                for (1..h.max_ngram) |k| {
                    rolling ^= toks[k] *% h.multipliers[l][k];
                    for (0..h.n_heads) |hd| {
                        const col = (k - 1) * h.n_heads + hd;
                        out[(i * h.n_layers + l) * cols + col] = @mod(rolling, h.primes[l][k - 1][hd]) + h.flat_offsets[l][col];
                    }
                }
            }
        }
    }
};

/// Read the records of `rows` from one layer's bank: `codes` gets the E4M3
/// words (`head_dim` bytes per row), `scales` the E8M0 bytes (`head_dim / 32`).
pub fn readRows(fd: std.c.fd_t, bank: *const Bank, rows: []const i64, codes: []u8, scales: []u8) !void {
    const rb: usize = bank.record_bytes;
    const hd: usize = bank.head_dim;
    var rec: [4096]u8 = undefined;
    if (rb > rec.len) return error.RecordTooLarge;
    for (rows, 0..) |r, i| {
        if (r < 0) return error.RowOutOfRange;
        const off: u64 = @as(u64, @intCast(r)) * rb;
        var done: usize = 0;
        while (done < rb) {
            const n = std.c.pread(fd, rec[done..].ptr, rb - done, @intCast(off + done));
            if (n < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                return error.ReadFailed;
            }
            if (n == 0) return error.ShortRead;
            done += @intCast(n);
        }
        @memcpy(codes[i * hd ..][0..hd], rec[0..hd]);
        @memcpy(scales[i * (hd / 32) ..][0 .. hd / 32], rec[hd..rb]);
    }
}

/// The lane's Engram row budget per bank (its MTPLX_ENGRAM_CACHE_LIMIT): the one charged constant.
pub const row_cache_bytes_per_bank: u64 = 64 << 20;

/// The row caches' host bytes (a memory bill's term): two banks of 264-byte records.
pub const row_cache_host_bytes: u64 = 2 * ngram.RowCache.hostBytes(264, row_cache_bytes_per_bank);

/// Construction-time refusals of the row source (one named error each; the
/// message says which field or file).
pub const Refusal = error{ EngramManifest, EngramTokenMap, EngramBankFile };

fn refuse(diag: ?*v41.Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// The compressed token map: `R/exl3/runtime/export_dsv41_engram_token_map.py`
/// runs our Python `build_compressed_token_map` (tokenizer normalisation stays
/// Python) and writes one little-endian u32 per vocab id plus a JSON sidecar
/// naming the tokenizer.json and the manifest it was built for. Checked once.
pub const TokenMap = struct {
    ids: []u32,
    pad_compressed: u32,

    pub const format = "mtplx-dsv41-engram-token-map-v1";

    /// `map_path` + `map_path.json`, against the bank's tokenizer.json, the
    /// manifest identity and the config / hashing geometry. `a` owns the result.
    pub fn load(a: std.mem.Allocator, io: std.Io, map_path: []const u8, bank_dir: []const u8, c: *const v41.Config, h: *const Hashing, bank: *const Bank, diag: ?*v41.Diag) !TokenMap {
        const Meta = struct {
            format: []const u8,
            vocab: u64,
            compressed_vocab_size: u64,
            pad_id: u64,
            pad_compressed: u64,
            map_sha256: []const u8,
            tokenizer_sha256: []const u8,
            manifest_sha256: []const u8,
        };
        const meta_path = try std.fmt.allocPrint(a, "{s}.json", .{map_path});
        const meta_text = std.Io.Dir.cwd().readFileAlloc(io, meta_path, a, .limited(1 << 16)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s} (the converter writes the map and its sidecar together)", .{ meta_path, @errorName(e) }),
        };
        const meta = std.json.parseFromSliceLeaky(Meta, a, meta_text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s}", .{ meta_path, @errorName(e) }),
        };
        if (!std.mem.eql(u8, meta.format, format)) return refuse(diag, error.EngramTokenMap, "{s}: format {s}", .{ meta_path, meta.format });
        if (meta.vocab != c.vocab_size) return refuse(diag, error.EngramTokenMap, "token map covers {d} ids, config vocab_size is {d}", .{ meta.vocab, c.vocab_size });
        if (meta.compressed_vocab_size != h.compressed_vocab) return refuse(diag, error.EngramTokenMap, "token map compresses to {d} ids, the manifest to {d}", .{ meta.compressed_vocab_size, h.compressed_vocab });
        if (meta.pad_id != h.pad_id) return refuse(diag, error.EngramTokenMap, "token map pad id {d}, the manifest's {d}", .{ meta.pad_id, h.pad_id });
        if (!std.mem.eql(u8, meta.manifest_sha256, bank.manifest_sha256)) return refuse(diag, error.EngramTokenMap, "token map built for manifest {s}, this bank's is {s}", .{ meta.manifest_sha256, bank.manifest_sha256 });
        const tok_path = try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{bank_dir});
        const tok = io_util.readAllNoCache(a, tok_path, 256 << 20) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s}", .{ tok_path, @errorName(e) }),
        };
        if (!std.mem.eql(u8, &sha256Hex(tok), meta.tokenizer_sha256)) return refuse(diag, error.EngramTokenMap, "token map built from tokenizer {s}, the bank's tokenizer.json is {s}", .{ meta.tokenizer_sha256, &sha256Hex(tok) });
        const raw = io_util.readAllNoCache(a, map_path, 64 << 20) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramTokenMap, "{s}: {s}", .{ map_path, @errorName(e) }),
        };
        if (raw.len != meta.vocab * 4) return refuse(diag, error.EngramTokenMap, "{s}: {d} bytes, want {d} (u32 per id)", .{ map_path, raw.len, meta.vocab * 4 });
        if (!std.mem.eql(u8, &sha256Hex(raw), meta.map_sha256)) return refuse(diag, error.EngramTokenMap, "{s}: sha256 differs from its sidecar", .{map_path});
        const ids = try a.alloc(u32, raw.len / 4);
        var top: u32 = 0;
        for (ids, 0..) |*v, i| {
            v.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
            top = @max(top, v.*);
        }
        // Python keys the map by distinct normalised forms: ids 0 .. size-1, all used.
        if (@as(u64, top) + 1 != h.compressed_vocab) return refuse(diag, error.EngramTokenMap, "token map ids reach {d}, want 0 .. {d}", .{ top, h.compressed_vocab - 1 });
        if (ids[h.pad_id] != meta.pad_compressed) return refuse(diag, error.EngramTokenMap, "token map pads to {d}, its sidecar says {d}", .{ ids[h.pad_id], meta.pad_compressed });
        return .{ .ids = ids, .pad_compressed = ids[h.pad_id] };
    }
};

fn sha256Hex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// The Engram row source (Python `EngramV41`'s row fetch: `NgramHashState` ids,
/// `NGramRowCache` records): the manifest, the token map and one record table
/// per Engram layer (qwen4_exp's row table behind the lane's row cache), all
/// checked at `open`; each sequence owns a `HashState`. The dequantize of the
/// fetched records is MLX (`Trunk.engramRows`), the gated add `Trunk.engramApply`.
pub const RowSource = struct {
    arena: std.heap.ArenaAllocator,
    hashing: Hashing,
    bank: Bank,
    map: TokenMap,
    tables: [max_layers]?*ngram.NgramTable = @splat(null),
    /// The tables' descriptors (they own them).
    fds: [max_layers]std.c.fd_t = @splat(-1),
    /// A gather's records before the codes / scales split (heap: `read` takes a const source).
    recs: ?*std.ArrayList(u8) = null,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, bank_dir: []const u8, map_path: []const u8, c: *const v41.Config, diag: ?*v41.Diag) !RowSource {
        var self: RowSource = .{ .arena = std.heap.ArenaAllocator.init(gpa), .hashing = undefined, .bank = undefined, .map = undefined };
        errdefer self.deinit();
        const a = self.arena.allocator();
        const mpath = try std.fmt.allocPrint(a, "{s}/engram/engram-manifest.json", .{bank_dir});
        const mtext = std.Io.Dir.cwd().readFileAlloc(io, mpath, a, .limited(1 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.EngramManifest, "{s}: {s}", .{ mpath, @errorName(e) }),
        };
        const m = try parseManifest(a, mtext, c, diag);
        self.hashing = m.hashing;
        self.bank = m.bank;
        self.map = try TokenMap.load(a, io, map_path, bank_dir, c, &self.hashing, &self.bank, diag);
        const recs = try gpa.create(std.ArrayList(u8));
        recs.* = .empty;
        self.recs = recs;
        for (0..self.hashing.n_layers) |i| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}/engram/{s}", .{ bank_dir, self.bank.files[i] }, 0);
            const t = try gpa.create(ngram.NgramTable);
            t.* = ngram.NgramTable.openRecords(path, self.bank.record_bytes, self.bank.rows[i], 0) catch |e| {
                gpa.destroy(t);
                return refuse(diag, error.EngramBankFile, "{s}: {s} (the manifest's {d} rows x {d} B)", .{ path, @errorName(e), self.bank.rows[i], self.bank.record_bytes });
            };
            self.tables[i] = t;
            self.fds[i] = t.fd;
            const size = std.c.lseek(t.fd, 0, std.c.SEEK.END);
            const want = self.bank.rows[i] * self.bank.record_bytes;
            if (size < 0 or @as(u64, @intCast(size)) != want) return refuse(diag, error.EngramBankFile, "{s}: {d} bytes, the manifest's {d} rows x {d} need {d}", .{ path, size, self.bank.rows[i], self.bank.record_bytes, want });
            try t.attachCache(gpa, row_cache_bytes_per_bank);
        }
        return self;
    }

    pub fn deinit(self: *RowSource) void {
        const gpa = self.arena.child_allocator;
        for (self.tables) |t| if (t) |x| {
            x.close();
            gpa.destroy(x);
        };
        if (self.recs) |r| {
            r.deinit(gpa);
            gpa.destroy(r);
        }
        self.arena.deinit();
    }

    /// Layer slot `li`'s row-cache statistics (Python's `cache.stats`).
    pub fn cacheStats(self: *const RowSource, li: usize) ngram.RowCache.Stats {
        return self.tables[li].?.cache.?.stats;
    }

    /// Row ids per position: `[n_layers][cols]`.
    pub fn perToken(self: *const RowSource) usize {
        return self.hashing.n_layers * self.hashing.cols();
    }

    /// `NgramHashState.advance` for one sequence: `out` gets `[ids][n_layers][cols]`.
    pub fn advance(self: *const RowSource, gpa: std.mem.Allocator, st: *HashState, ids: []const u32, out: []i64) !void {
        return st.advance(gpa, &self.hashing, self.map.ids, ids, null, out);
    }

    /// Layer slot `li`'s records for the `n` positions of `rows` (`advance`'s
    /// output): `codes` gets `[n * cols][head_dim]` E4M3 bytes, `scales`
    /// `[n * cols][head_dim / 32]` E8M0 bytes, positions then columns.
    pub fn read(self: *const RowSource, li: usize, rows: []const i64, n: usize, ids_buf: []i64, codes: []u8, scales: []u8) !void {
        const cols = self.hashing.cols();
        const per = self.perToken();
        for (0..n) |t| @memcpy(ids_buf[t * cols ..][0..cols], rows[t * per + li * cols ..][0..cols]);
        return self.readIds(li, ids_buf[0 .. n * cols], codes, scales);
    }

    /// Start every layer slot's poster thread (`post` / `take`): the prompt pass's gathers run ahead of it.
    pub fn enablePosting(self: *RowSource) !void {
        for (self.tables[0..self.hashing.n_layers]) |t| try t.?.enablePosting();
    }

    /// Layer slot `li`'s gather for the `n` positions of `rows`, posted: `read`'s ids now, its records
    /// when `take` returns (the table's poster runs `read`'s gather, cache included, in post order).
    pub const Posted = struct { job: ngram.NgramTable.Posted, li: usize };

    /// Post layer slot `li`'s gather for `n` positions of `rows` (`advance`'s output); `a` holds the ids and
    /// records until `take` (and every later wave that reads them) is done.
    pub fn post(self: *const RowSource, a: std.mem.Allocator, li: usize, rows: []const i64, n: usize) !*Posted {
        const cols = self.hashing.cols();
        const per = self.perToken();
        const ids = try a.alloc(i64, n * cols);
        for (0..n) |t| @memcpy(ids[t * cols ..][0..cols], rows[t * per + li * cols ..][0..cols]);
        const p = try a.create(Posted);
        p.* = .{ .job = .{ .rows = ids, .out = try a.alloc(u8, ids.len * self.bank.record_bytes) }, .li = li };
        self.tables[li].?.post(&p.job);
        return p;
    }

    /// A posted gather's records, split as `read` splits them (`codes` `[n * cols][head_dim]`, `scales`
    /// `[n * cols][head_dim / 32]`).
    pub fn take(self: *const RowSource, p: *Posted, codes: []u8, scales: []u8) !void {
        try self.tables[p.li].?.wait(&p.job);
        self.split(p.job.out, p.job.rows.len, codes, scales);
    }

    /// Wait a posted gather without its records (an aborted pass drains its posts before its memory goes).
    pub fn drain(self: *const RowSource, p: *Posted) void {
        self.tables[p.li].?.wait(&p.job) catch {};
    }

    /// Free a taken (or drained) posted gather's ids, records and itself (`a` = `post`'s allocator).
    pub fn release(_: *const RowSource, a: std.mem.Allocator, p: *Posted) void {
        a.free(p.job.rows);
        a.free(p.job.out);
        a.destroy(p);
    }

    fn split(self: *const RowSource, recs: []const u8, n_rows: usize, codes: []u8, scales: []u8) void {
        const rb: usize = self.bank.record_bytes;
        const hd: usize = self.bank.head_dim;
        for (0..n_rows) |i| {
            const rec = recs[i * rb ..][0..rb];
            @memcpy(codes[i * hd ..][0..hd], rec[0..hd]);
            @memcpy(scales[i * (hd / 32) ..][0 .. hd / 32], rec[hd..rb]);
        }
    }

    /// Every layer slot's posted gather against an independent read past the cache, bitwise, for `n`
    /// positions of `rows`: the construction check of the posted route (refused by name otherwise).
    pub fn checkPosted(self: *const RowSource, gpa: std.mem.Allocator, rows: []const i64, n: usize) !void {
        const cols = self.hashing.cols();
        const hd: usize = self.bank.head_dim;
        const rb: usize = self.bank.record_bytes;
        const per = self.perToken();
        const codes = try gpa.alloc(u8, n * cols * hd);
        defer gpa.free(codes);
        const scales = try gpa.alloc(u8, n * cols * (hd / 32));
        defer gpa.free(scales);
        const ids = try gpa.alloc(i64, n * cols);
        defer gpa.free(ids);
        const recs = try gpa.alloc(u8, n * cols * rb);
        defer gpa.free(recs);
        for (0..self.hashing.n_layers) |li| {
            const p = try self.post(gpa, li, rows, n);
            defer self.release(gpa, p);
            self.take(p, codes, scales) catch |e| {
                self.drain(p);
                return e;
            };
            for (0..n) |t| @memcpy(ids[t * cols ..][0..cols], rows[t * per + li * cols ..][0..cols]);
            try self.tables[li].?.readUncached(ids, recs);
            for (0..ids.len) |i| {
                const rec = recs[i * rb ..][0..rb];
                if (!std.mem.eql(u8, codes[i * hd ..][0..hd], rec[0..hd]) or !std.mem.eql(u8, scales[i * (hd / 32) ..][0 .. hd / 32], rec[hd..rb]))
                    return error.EngramPostedSelfCheck;
            }
        }
    }

    /// The records of row ids `ids` of layer slot `li`, through the bank's row cache.
    pub fn readIds(self: *const RowSource, li: usize, ids: []const i64, codes: []u8, scales: []u8) !void {
        const rb: usize = self.bank.record_bytes;
        const recs = self.recs.?;
        try recs.resize(self.arena.child_allocator, ids.len * rb);
        try self.tables[li].?.gatherRecords(ids, recs.items);
        self.split(recs.items, ids.len, codes, scales);
    }
};

const testing = std.testing;

/// A small table with hand-checkable arithmetic: 3-grams, 2 heads, 1 layer.
fn tinyHashing() Hashing {
    var h: Hashing = .{ .max_ngram = 3, .n_heads = 2, .n_layers = 1, .pad_id = 0, .compressed_vocab = 10 };
    h.multipliers[0][0] = 3;
    h.multipliers[0][1] = 5;
    h.multipliers[0][2] = 7;
    h.primes[0][0][0] = 11;
    h.primes[0][0][1] = 13;
    h.primes[0][1][0] = 17;
    h.primes[0][1][1] = 19;
    h.flat_offsets[0][0] = 0;
    h.flat_offsets[0][1] = 11;
    h.flat_offsets[0][2] = 24;
    h.flat_offsets[0][3] = 41;
    return h;
}

test "dsv41 engram: the n-gram hash follows the manifest recipe" {
    const h = tinyHashing();
    const map = [_]u32{ 9, 2, 4, 6 }; // pad id 0 -> compressed 9
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var rows: [2 * 4]i64 = undefined;
    try st.advance(testing.allocator, &h, &map, &.{ 1, 2 }, null, &rows);
    // pos 0: toks (2, pad 9, pad 9); pos 1: toks (4, 2, pad 9).
    const r0a: i64 = (2 * 3) ^ (9 * 5);
    const r0b: i64 = r0a ^ (9 * 7);
    const r1a: i64 = (4 * 3) ^ (2 * 5);
    const r1b: i64 = r1a ^ (9 * 7);
    const want = [_]i64{
        @mod(r0a, 11), @mod(r0a, 13) + 11, @mod(r0b, 17) + 24, @mod(r0b, 19) + 41,
        @mod(r1a, 11), @mod(r1a, 13) + 11, @mod(r1b, 17) + 24, @mod(r1b, 19) + 41,
    };
    try testing.expectEqualSlices(i64, &want, &rows);
}

test "dsv41 engram: streaming, masking and trim hash like one pass" {
    const h = tinyHashing();
    const map = [_]u32{ 9, 2, 4, 6, 1, 3 };
    const ids = [_]u32{ 1, 2, 3, 4, 5, 1 };
    var one: HashState = .{};
    defer one.deinit(testing.allocator);
    var all: [6 * 4]i64 = undefined;
    try one.advance(testing.allocator, &h, &map, &ids, null, &all);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var part: [6 * 4]i64 = undefined;
    try st.advance(testing.allocator, &h, &map, ids[0..4], null, part[0 .. 4 * 4]);
    // a rejected draft: feed two wrong tokens, trim them, re-feed the real ones
    var junk: [2 * 4]i64 = undefined;
    try st.advance(testing.allocator, &h, &map, &.{ 3, 3 }, null, &junk);
    st.trim(2);
    try st.advance(testing.allocator, &h, &map, ids[4..6], null, part[4 * 4 ..]);
    try testing.expectEqualSlices(i64, &all, &part);
    // a masked position pads every n-gram that spans it
    var m: HashState = .{};
    defer m.deinit(testing.allocator);
    var masked: [3 * 4]i64 = undefined;
    try m.advance(testing.allocator, &h, &map, &.{ 1, 2, 3 }, &.{ false, true, false }, &masked);
    var p: HashState = .{};
    defer p.deinit(testing.allocator);
    var fresh: [1 * 4]i64 = undefined;
    try p.advance(testing.allocator, &h, &map, &.{3}, null, &fresh);
    try testing.expectEqualSlices(i64, &fresh, masked[2 * 4 ..]); // pos 2 sees (tok, pad, pad)
    try testing.expectError(error.TokenOutOfMap, p.advance(testing.allocator, &h, &map, &.{6}, null, &fresh));
}

// DSV41_BANK=<bank> DSV41_ENGRAM_FIXTURE=<json from R/exl3/runtime/dump_dsv41_engram_fixture.py>
test "dsv41 engram: the real manifest and token map hash the prompt to the Python oracle's rows" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_ENGRAM_FIXTURE") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank_dir, &diag);
    const mtext = try std.Io.Dir.cwd().readFileAlloc(testing.io, try std.fmt.allocPrint(a, "{s}/engram/engram-manifest.json", .{bank_dir}), a, .limited(1 << 20));
    const parsed = try parseManifest(a, mtext, &c, &diag);
    const Row = struct { layer: u32, row: i64, sha256: []const u8 };
    const Fixture = struct { token_map: []const u8, ids: []const u32, prompt: u32, rows: []const i64, pad_compressed: i64, bytes: []const Row };
    const fx = try std.json.parseFromSliceLeaky(Fixture, a, try std.Io.Dir.cwd().readFileAlloc(testing.io, fixture, a, .limited(64 << 20)), .{ .ignore_unknown_fields = true });
    const map_bytes = try std.Io.Dir.cwd().readFileAlloc(testing.io, fx.token_map, a, .limited(16 << 20));
    const map = try a.alloc(u32, map_bytes.len / 4);
    for (map, 0..) |*v, i| v.* = std.mem.readInt(u32, map_bytes[i * 4 ..][0..4], .little);
    try testing.expectEqual(fx.pad_compressed, @as(i64, map[parsed.hashing.pad_id]));
    const per = parsed.hashing.n_layers * parsed.hashing.cols();
    const got = try a.alloc(i64, fx.ids.len * per);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    // the prompt in one span, then the decode tokens one by one (the oracle fed them the same way)
    try st.advance(testing.allocator, &parsed.hashing, map, fx.ids[0..fx.prompt], null, got[0 .. fx.prompt * per]);
    for (fx.prompt..fx.ids.len) |t| try st.advance(testing.allocator, &parsed.hashing, map, fx.ids[t .. t + 1], null, got[t * per ..][0..per]);
    try testing.expectEqualSlices(i64, fx.rows, got);
    for (fx.bytes) |r| {
        const slot = std.mem.indexOfScalar(u32, parsed.hashing.layer_ids[0..parsed.hashing.n_layers], r.layer) orelse return error.TestUnexpectedResult;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/engram/{s}", .{ bank_dir, parsed.bank.files[slot] }, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);
        var codes: [256]u8 = undefined;
        var scales: [8]u8 = undefined;
        try readRows(fd, &parsed.bank, &.{r.row}, &codes, &scales);
        var d = std.crypto.hash.sha2.Sha256.init(.{});
        d.update(&codes);
        d.update(&scales);
        try testing.expectEqualStrings(r.sha256, &std.fmt.bytesToHex(d.finalResult(), .lower));
    }
    std.debug.print("dsv41 engram: {d} positions x {d} rows equal the oracle; {d} records byte-equal\n", .{ fx.ids.len, per, fx.bytes.len });
}

// ── the row source on a synthetic mini bank (hermetic) ──

pub const MiniBank = struct {
    vocab: u32 = 64,
    map_mod: u32 = 50,
    sidecar_vocab: ?u64 = null,
    tokenizer_differs: bool = false,
    manifest_sha: []const u8 = "mini-manifest",
    sidecar_manifest_sha: []const u8 = "mini-manifest",
    flip_map_byte: bool = false,
    drop_sidecar: bool = false,
    short_bank: bool = false,
    codec: []const u8 = "mxfp8",
};

/// Record r byte i of the mini bank: `(r * 7 + i) mod 251`.
fn miniRecordByte(r: u64, i: u64) u8 {
    return @intCast((r * 7 + i) % 251);
}

/// A bank dir for the mini config (Engram layer 1: 97 rows, 3-grams x 2 heads,
/// head_dim 32 -> 33-byte records) with its token map + sidecar; returns the map path.
pub fn writeMiniBank(a: std.mem.Allocator, tmp: *std.testing.TmpDir, root: []const u8, f: MiniBank) ![]const u8 {
    const io = testing.io;
    try tmp.dir.createDirPath(io, "engram");
    const tok = if (f.tokenizer_differs) "{\"model\":\"other\"}" else "{\"model\":\"mini\"}";
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tok });
    const manifest = try std.fmt.allocPrint(a,
        \\{{"format":"mtplx-engram-manifest-v1","manifest_sha256":"{s}",
        \\"layers":[{{"layer_id":1,"file":"engram-L1.bin","rows":97,"record_bytes":33,
        \\"quant":{{"bits":8,"group_size":32,"mode":"{s}","head_dim":32}}}}],
        \\"hashing":{{"layer_ids":[1],"max_ngram_size":3,"n_heads":2,"head_dim":32,"compressed_vocab_size":50,
        \\"pad_id":2,"n_hash_cols":4,"hash_multipliers":[[3,5,7]],
        \\"per_layer":[{{"layer_id":1,"primes":[[11,13],[17,19]],"flat_offsets":[0,11,24,41],"total_rows":97}}]}}}}
    , .{ f.manifest_sha, f.codec });
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/engram-manifest.json", .data = manifest });
    const n_rec: u64 = if (f.short_bank) 96 else 97;
    const bank = try a.alloc(u8, @intCast(n_rec * 33));
    for (0..n_rec) |r| for (0..33) |i| {
        bank[r * 33 + i] = miniRecordByte(r, i);
    };
    try tmp.dir.writeFile(io, .{ .sub_path = "engram/engram-L1.bin", .data = bank });
    const map = try a.alloc(u8, f.vocab * 4);
    for (0..f.vocab) |i| std.mem.writeInt(u32, map[i * 4 ..][0..4], @intCast(i % f.map_mod), .little);
    const map_sha = sha256Hex(map);
    if (f.flip_map_byte) map[5] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "map.u32", .data = map });
    if (!f.drop_sidecar) {
        const meta = try std.fmt.allocPrint(a,
            \\{{"format":"{s}","vocab":{d},"compressed_vocab_size":50,"pad_id":2,"pad_compressed":2,
            \\"map_sha256":"{s}","tokenizer_sha256":"{s}","manifest_sha256":"{s}"}}
        , .{ TokenMap.format, f.sidecar_vocab orelse f.vocab, &map_sha, &sha256Hex("{\"model\":\"mini\"}"), f.sidecar_manifest_sha });
        try tmp.dir.writeFile(io, .{ .sub_path = "map.u32.json", .data = meta });
    }
    return std.fmt.allocPrint(a, "{s}/map.u32", .{root});
}

fn miniConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .mini);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

test "dsv41 engram: a synthetic bank opens and its rows come back through the row source" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const map_path = try writeMiniBank(a, &tmp, root, .{});
    const c = try miniConfig();
    var diag: v41.Diag = .{};
    var src = RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer src.deinit();
    try testing.expectEqual(@as(usize, 4), src.perToken());
    try testing.expectEqual(@as(u32, 2), src.map.pad_compressed);
    // The source hashes like a bare HashState over the same table and map.
    const ids = [_]u32{ 5, 7, 9 };
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var rows: [3 * 4]i64 = undefined;
    try src.advance(testing.allocator, &st, &ids, &rows);
    var ref: HashState = .{};
    defer ref.deinit(testing.allocator);
    var want: [3 * 4]i64 = undefined;
    try ref.advance(testing.allocator, &src.hashing, src.map.ids, &ids, null, &want);
    try testing.expectEqualSlices(i64, &want, &rows);
    // Records land positions-then-columns, code bytes and scale bytes split.
    var idb: [12]i64 = undefined;
    var codes: [12 * 32]u8 = undefined;
    var scales: [12]u8 = undefined;
    try src.read(0, &rows, 3, &idb, &codes, &scales);
    for (0..12) |k| {
        const r: u64 = @intCast(rows[k]);
        for (0..32) |i| try testing.expectEqual(miniRecordByte(r, i), codes[k * 32 + i]);
        try testing.expectEqual(miniRecordByte(r, 32), scales[k]);
    }
    try testing.expectError(error.RowOutOfRange, src.readIds(0, &.{97}, codes[0..32], scales[0..1]));
}

test "dsv41 engram: the posted route's construction check passes on the mini bank, and a direct read is the record" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const map_path = try writeMiniBank(a, &tmp, root, .{});
    const c = try miniConfig();
    var diag: v41.Diag = .{};
    var src = try RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag);
    defer src.deinit();
    try src.enablePosting();
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var rows: [5 * 4]i64 = undefined;
    try src.advance(testing.allocator, &st, &.{ 3, 5, 7, 9, 11 }, &rows);
    try src.checkPosted(testing.allocator, &rows, 5);
    // readRows (the parity harness's direct read): each record's code bytes then its scale byte.
    var codes: [3 * 32]u8 = undefined;
    var scales: [3]u8 = undefined;
    try readRows(src.fds[0], &src.bank, &.{ 0, 41, 96 }, &codes, &scales);
    for ([_]u64{ 0, 41, 96 }, 0..) |r, k| {
        for (0..32) |i| try testing.expectEqual(miniRecordByte(r, i), codes[k * 32 + i]);
        try testing.expectEqual(miniRecordByte(r, 32), scales[k]);
    }
    try testing.expectError(error.RowOutOfRange, readRows(src.fds[0], &src.bank, &.{-1}, codes[0..32], scales[0..1]));
    try testing.expectError(error.ShortRead, readRows(src.fds[0], &src.bank, &.{97}, codes[0..32], scales[0..1]));
}

test "dsv41 engram: posted gathers return the blocking reads' bytes in post order and leave the cache as they do" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const map_path = try writeMiniBank(a, &tmp, root, .{});
    const c = try miniConfig();
    var diag: v41.Diag = .{};
    var blocking = try RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag);
    defer blocking.deinit();
    var posted = try RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag);
    defer posted.deinit();
    try posted.enablePosting();
    // Two spans' rows (the second repeats some of the first's: cache hits on both paths).
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    var rows1: [3 * 4]i64 = undefined;
    var rows2: [4 * 4]i64 = undefined;
    try blocking.advance(testing.allocator, &st, &.{ 5, 7, 9 }, &rows1);
    try blocking.advance(testing.allocator, &st, &.{ 5, 7, 9, 11 }, &rows2);
    var idb: [16]i64 = undefined;
    var want1_c: [12 * 32]u8 = undefined;
    var want1_s: [12]u8 = undefined;
    var want2_c: [16 * 32]u8 = undefined;
    var want2_s: [16]u8 = undefined;
    try blocking.read(0, &rows1, 3, &idb, &want1_c, &want1_s);
    try blocking.read(0, &rows2, 4, &idb, &want2_c, &want2_s);
    // Both posted before either is taken: the poster runs them in post order.
    const p1 = try posted.post(a, 0, &rows1, 3);
    const p2 = try posted.post(a, 0, &rows2, 4);
    var got2_c: [16 * 32]u8 = undefined;
    var got2_s: [16]u8 = undefined;
    var got1_c: [12 * 32]u8 = undefined;
    var got1_s: [12]u8 = undefined;
    try posted.take(p2, &got2_c, &got2_s);
    try posted.take(p1, &got1_c, &got1_s);
    try testing.expectEqualSlices(u8, &want1_c, &got1_c);
    try testing.expectEqualSlices(u8, &want1_s, &got1_s);
    try testing.expectEqualSlices(u8, &want2_c, &got2_c);
    try testing.expectEqualSlices(u8, &want2_s, &got2_s);
    try testing.expectEqual(blocking.cacheStats(0), posted.cacheStats(0));
    // A row out of the bank fails its own job, by name; the poster keeps serving.
    const bad = try posted.post(a, 0, &.{ 97, 97, 97, 97 }, 1);
    try testing.expectError(error.RowOutOfRange, posted.take(bad, got1_c[0..128], got1_s[0..4]));
    const again = try posted.post(a, 0, &rows1, 3);
    try posted.take(again, &got1_c, &got1_s);
    try testing.expectEqualSlices(u8, &want1_c, &got1_c);
}

test "dsv41 engram: the row source refuses a map or bank built for something else, by name" {
    const Case = struct { f: MiniBank, err: anyerror };
    const cases = [_]Case{
        .{ .f = .{ .sidecar_vocab = 63 }, .err = error.EngramTokenMap },
        .{ .f = .{ .tokenizer_differs = true }, .err = error.EngramTokenMap },
        .{ .f = .{ .sidecar_manifest_sha = "another-manifest" }, .err = error.EngramTokenMap },
        .{ .f = .{ .flip_map_byte = true }, .err = error.EngramTokenMap },
        .{ .f = .{ .map_mod = 49 }, .err = error.EngramTokenMap },
        .{ .f = .{ .drop_sidecar = true }, .err = error.EngramTokenMap },
        .{ .f = .{ .short_bank = true }, .err = error.EngramBankFile },
        .{ .f = .{ .codec = "affine" }, .err = error.EngramManifest },
    };
    for (cases, 0..) |cs, i| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var rbuf: [512]u8 = undefined;
        const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
        const map_path = try writeMiniBank(a, &tmp, root, cs.f);
        const c = try miniConfig();
        var diag: v41.Diag = .{};
        if (RowSource.open(testing.allocator, testing.io, root, map_path, &c, &diag)) |opened| {
            var o = opened;
            o.deinit();
            std.debug.print("case {d}: opened, wanted {s}\n", .{ i, @errorName(cs.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            testing.expectEqual(cs.err, e) catch |x| {
                std.debug.print("case {d}: {s}\n", .{ i, diag.message() });
                return x;
            };
            try testing.expect(diag.message().len > 0);
        }
    }
}

// DSV41_BANK=<bank> DSV41_ENGRAM_TOKEN_MAP=<converter output> DSV41_ENGRAM_FIXTURE=<m0 fixture json>
// DSV41_BANK + DSV41_ENGRAM_TOKEN_MAP (host only, F_NOCACHE reads): the module's construction check of the
// posted route on the real tables, over one 2048-position chunk; prints the posted gathers' wall time.
test "dsv41 engram: the real bank's posted gathers equal a read past the cache on a prefill chunk" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank_dir, &diag);
    var src = try RowSource.open(testing.allocator, testing.io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    try src.enablePosting();
    const n = 2048;
    const ids = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*x, i| x.* = @intCast((i * 7919 + 13) % c.vocab_size);
    const rows = try testing.allocator.alloc(i64, n * src.perToken());
    defer testing.allocator.free(rows);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    try src.advance(testing.allocator, &st, ids, rows);
    const t0 = std.Io.Timestamp.now(testing.io, .awake);
    try src.checkPosted(testing.allocator, rows, n);
    const ms = @divTrunc(t0.untilNow(testing.io, .awake).nanoseconds, std.time.ns_per_ms);
    std.debug.print("dsv41 engram posted: {d} positions x {d} slots x {d} cols gathered, posted and checked in {d} ms\n", .{ n, src.hashing.n_layers, src.hashing.cols(), ms });
}

test "dsv41 engram: the real bank's row source hashes and reads like the Python oracle" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("DSV41_ENGRAM_FIXTURE") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank_dir, &diag);
    var src = try RowSource.open(testing.allocator, testing.io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    const Row = struct { layer: u32, row: i64, sha256: []const u8 };
    const Fixture = struct { ids: []const u32, prompt: u32, rows: []const i64, bytes: []const Row };
    const fx = try std.json.parseFromSliceLeaky(Fixture, a, try std.Io.Dir.cwd().readFileAlloc(testing.io, fixture, a, .limited(64 << 20)), .{ .ignore_unknown_fields = true });
    const per = src.perToken();
    const got = try a.alloc(i64, fx.ids.len * per);
    var st: HashState = .{};
    defer st.deinit(testing.allocator);
    try src.advance(testing.allocator, &st, fx.ids[0..fx.prompt], got[0 .. fx.prompt * per]);
    for (fx.prompt..fx.ids.len) |t| try src.advance(testing.allocator, &st, fx.ids[t .. t + 1], got[t * per ..][0..per]);
    try testing.expectEqualSlices(i64, fx.rows, got);
    for (fx.bytes) |r| {
        const li = std.mem.indexOfScalar(u32, src.hashing.layer_ids[0..src.hashing.n_layers], r.layer) orelse return error.TestUnexpectedResult;
        var codes: [256]u8 = undefined;
        var scales: [8]u8 = undefined;
        try src.readIds(li, &.{r.row}, &codes, &scales);
        var d = std.crypto.hash.sha2.Sha256.init(.{});
        d.update(&codes);
        d.update(&scales);
        try testing.expectEqualStrings(r.sha256, &std.fmt.bytesToHex(d.finalResult(), .lower));
    }
    std.debug.print("dsv41 engram: row source over {d} rows x {d} layers; {d} positions and {d} records equal the oracle\n", .{ src.bank.rows[0], src.hashing.n_layers, fx.ids.len, fx.bytes.len });
}

fn msSince(t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.untilNow(testing.io, .boot).nanoseconds)) / 1e6;
}

/// The Engram files' page-cache residency (mincore).
fn engramResident(src: *const RowSource, bank_dir: []const u8) !u64 {
    var sum: u64 = 0;
    var buf: [1024]u8 = undefined;
    for (0..src.hashing.n_layers) |i| sum += try io_util.residentBytes(try std.fmt.bufPrintSentinel(&buf, "{s}/engram/{s}", .{ bank_dir, src.bank.files[i] }, 0));
    return sum;
}

/// Little-endian reader over the replay's binary forwards file.
const ReplayBin = struct {
    b: []const u8,
    at: usize = 0,
    fn int(r: *ReplayBin, comptime T: type) T {
        const v = std.mem.readInt(T, r.b[r.at..][0..@sizeOf(T)], .little);
        r.at += @sizeOf(T);
        return v;
    }
};

// DSV41_BANK=<bank> DSV41_ENGRAM_TOKEN_MAP=<map> DSV41_ENGRAM_REPLAY=<engram_replay.py json, --schedule unsplit>:
// Python's hash and cache on the lane's forwards, replayed through the row source.
test "dsv41 engram: the lane's gather sequence replays through the row source with Python's stats and bytes" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    const replay = std.mem.span(std.c.getenv("DSV41_ENGRAM_REPLAY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, testing.io, bank_dir, &diag);
    const Py = struct { stats: []const ngram.RowCache.Stats, bytes_read: u64, sha256: []const u8 };
    const Seq = struct { name: []const u8, bin: []const u8, forwards: u32, gathers: u32, lookups: u64, python: Py };
    const doc = try std.json.parseFromSliceLeaky(struct { format: []const u8, schedule: []const u8, sequences: []const Seq }, a, try std.Io.Dir.cwd().readFileAlloc(testing.io, replay, a, .limited(64 << 20)), .{ .ignore_unknown_fields = true });
    try testing.expectEqualStrings("unsplit", doc.schedule);
    for (doc.sequences) |seq| {
        var src = try RowSource.open(gpa, testing.io, bank_dir, map_path, &c, &diag);
        defer src.deinit();
        const resident0 = try engramResident(&src, bank_dir);
        const nl = src.hashing.n_layers;
        const cols = src.hashing.cols();
        const hd: usize = src.bank.head_dim;
        var rb: ReplayBin = .{ .b = try std.Io.Dir.cwd().readFileAlloc(testing.io, seq.bin, a, .limited(1 << 30)) };
        const n_fw = rb.int(u32);
        try testing.expectEqual(seq.forwards, n_fw);
        var st: HashState = .{};
        defer st.deinit(gpa);
        var sha = std.crypto.hash.sha2.Sha256.init(.{});
        var ms: [2]f64 = .{ 0, 0 };
        var max_ms: [2]f64 = .{ 0, 0 };
        var count: [2]u32 = .{ 0, 0 };
        for (0..n_fw) |_| {
            const f: usize = rb.int(u8);
            const n = rb.int(u32);
            const trim = rb.int(u32);
            const ids = try a.alloc(u32, n);
            for (ids) |*t| t.* = rb.int(u32);
            try testing.expectEqual(@as(u32, @intCast(nl)), rb.int(u32));
            const want = try a.alloc(i64, n * cols * nl);
            for (0..nl) |li| {
                try testing.expectEqual(@as(u32, @intCast(li)), rb.int(u32));
                try testing.expectEqual(@as(u32, @intCast(n * cols)), rb.int(u32));
                for (want[li * n * cols ..][0 .. n * cols]) |*w| w.* = rb.int(i64);
            }
            const out = try a.alloc(i64, n * nl * cols);
            const ids_buf = try a.alloc(i64, n * cols);
            const codes = try a.alloc(u8, n * cols * hd);
            const scales = try a.alloc(u8, n * cols * (hd / 32));
            const t0 = std.Io.Timestamp.now(testing.io, .boot);
            try src.advance(gpa, &st, ids, out);
            for (0..nl) |li| {
                try src.read(li, out, n, ids_buf, codes, scales);
                try testing.expectEqualSlices(i64, want[li * n * cols ..][0 .. n * cols], ids_buf[0 .. n * cols]);
                for (0..n * cols) |r| {
                    sha.update(codes[r * hd ..][0..hd]);
                    sha.update(scales[r * (hd / 32) ..][0 .. hd / 32]);
                }
            }
            const dt = msSince(t0);
            ms[f] += dt;
            max_ms[f] = @max(max_ms[f], dt);
            count[f] += 1;
            if (trim > 0) st.trim(trim);
        }
        try testing.expectEqual(rb.b.len, rb.at);
        var bytes: u64 = 0;
        for (0..nl) |li| {
            const s = src.cacheStats(li);
            testing.expectEqual(seq.python.stats[li], s) catch |e| {
                std.debug.print("{s} bank {d}: python {any}\n  port {any}\n", .{ seq.name, li, seq.python.stats[li], s });
                return e;
            };
            bytes += s.rows_read * src.bank.record_bytes;
        }
        try testing.expectEqual(seq.python.bytes_read, bytes);
        const hex = std.fmt.bytesToHex(sha.finalResult(), .lower);
        try testing.expectEqualStrings(seq.python.sha256, &hex);
        const resident1 = try engramResident(&src, bank_dir);
        std.debug.print("\nDSV41_ENGRAM_REPLAY {s}: {d} forwards, {d} lookups; bank 1 {any}; bank 14 {any}; {d} B read; sha256 {s} (= Python); prefill {d} forwards {d:.1} ms (max {d:.2}); decode {d} forwards {d:.1} ms (max {d:.2}, mean {d:.3}); Engram page cache {d} -> {d} B\n", .{
            seq.name, seq.forwards, seq.lookups, src.cacheStats(0), src.cacheStats(1), bytes, hex[0..16], count[0], ms[0], max_ms[0], count[1], ms[1], max_ms[1], ms[1] / @as(f64, @floatFromInt(@max(count[1], 1))), resident0, resident1,
        });
        try testing.expectEqual(resident0, resident1);
    }
}
