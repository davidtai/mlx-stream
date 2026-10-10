//! MiMo-V2.6-Flash's three trained MTP heads (`model.mtp.layers.{0,1,2}`), the draft half of the MTP lane
//! (`draft.zig` verifies and commits).
//!
//! The heads live in the pack's sidecar `mtp-heads.safetensors`, described by `mtp-manifest.json`
//! (`mlx-stream-mtp-heads-v1`: per-tensor dtype, shape, offset, length and sha256; each projection role's storage
//! from the manifest's `quantization` (affine bits/group, `false` = BF16 with no scales/biases, or the release's FP8
//! under the fp8res flag `residents_format: "fp8_block128"`: U8 codes + F32 `.weight_scale_inv`), BF16 norms and
//! sinks, q/k/v split from the source's fused qkv). Construction checks the manifest against the
//! trunk's geometry, every header entry against the manifest, and hashes every tensor's bytes (and the whole file)
//! before the bytes become arrays: the arrays hold exactly the bytes whose sha256 matched. Nothing is read through the
//! page cache.
//!
//! Each head (upstream mlx-serve `origin/mimo-v2` `src/mimo_mtp.zig` `layerForward` :408-459, SGLang's MiMo-V2 MTP):
//!
//!   x  = eh_proj(cat[rms(embed(token), enorm), rms(trunk_hidden, hnorm)])
//!   x2 = x + o(attn(rms(x, input_layernorm)))     one sliding-window layer with sinks (the trunk's sliding geometry)
//!   x3 = x2 + down(silu(gate(m)) * up(m)),  m = rms(x2, pre_mlp_layernorm)     dense SwiGLU, width 16,384
//!   out = rms(x3, final_layernorm)  -> the trunk's shared lm_head
//!
//! Head k's row p pairs the trunk's FINAL-NORMED hidden at p (upstream `mtpReadsFinalNorm`) with token x_{p+k+1} at
//! rope position p and predicts x_{p+k+2}. Every head reads the trunk's hidden, never the previous head's output; only
//! the token shifts. A round at last committed position q drafts d1 = head 0 at q, d_{i+1} = head i at q over the
//! drafts so far (depth <= 3: up to 4 verify rows). Rows of head i past q - i carry drafts and are dropped before the
//! next round (`committed_end`). The arithmetic boundaries are the trunk's own (`module.zig`): BF16-rounded RMS norm
//! before the weight, F32 SiLU rounded to BF16, F32 value scale, the trunk's rotary tables at the sliding theta.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const io_util = sdk.io_util;
const mimo = @import("config.zig");
const tm = @import("module.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const QLinear = tm.QLinear;
const fp8_linear = @import("fp8_linear.zig");
const embed_rows = @import("embed_rows.zig");
const drop = tm.drop;

pub const manifest_file = "mtp-manifest.json";
pub const data_file = "mtp-heads.safetensors";
pub const family = "mlx-stream";
pub const format = "mlx-stream-mtp-heads-v1";
pub const n_heads = 3;
/// Drafts per round (upstream generate.zig: depth cap 3): verify rows = 1 + drafts.
pub const max_drafts: u32 = 3;
pub const verify_rows: u32 = 1 + max_drafts;
/// Committed hiddens and tokens the lane keeps: every head's window, the rows a lagging head catches up, a round.
pub const ring_rows: u32 = 256;
pub const tensors_per_head = suffixes.len;
const manifest_limit = 1 << 20;
const header_limit = 1 << 20;

/// The tensors of one head (names `model.mtp.layers.{k}.<suffix>`), the loader's names (upstream `loadLayer`).
const Kind = enum { norm, sink, weight, scales, biases, scale_inv };
const Proj = enum { eh, q, k, v, o, gate, up, down };
const Suffix = struct { name: []const u8, kind: Kind, proj: ?Proj = null };
const suffixes = [_]Suffix{
    .{ .name = "enorm.weight", .kind = .norm },
    .{ .name = "hnorm.weight", .kind = .norm },
    .{ .name = "input_layernorm.weight", .kind = .norm },
    .{ .name = "pre_mlp_layernorm.weight", .kind = .norm },
    .{ .name = "final_layernorm.weight", .kind = .norm },
    .{ .name = "self_attn.attention_sink_bias", .kind = .sink },
} ++ projections("eh_proj", .eh) ++ projections("self_attn.q_proj", .q) ++ projections("self_attn.k_proj", .k) ++
    projections("self_attn.v_proj", .v) ++ projections("self_attn.o_proj", .o) ++ projections("mlp.gate_proj", .gate) ++
    projections("mlp.up_proj", .up) ++ projections("mlp.down_proj", .down);

fn projections(comptime base: []const u8, comptime p: Proj) [4]Suffix {
    return .{
        .{ .name = base ++ ".weight", .kind = .weight, .proj = p },
        .{ .name = base ++ ".scales", .kind = .scales, .proj = p },
        .{ .name = base ++ ".biases", .kind = .biases, .proj = p },
        // fp8res: the release's f32 block scales of an FP8 projection (config.zig `Storage.fp8`).
        .{ .name = base ++ ".weight_scale_inv", .kind = .scale_inv, .proj = p },
    };
}

/// The manifest's projection names (per head), in `Proj` order.
const proj_names = [_][]const u8{ "eh_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
pub const n_proj = proj_names.len;
/// Each projection role's storage (shared by the three heads), from the manifest.
pub const Storages = [n_proj]mimo.Storage;
/// The Q4 sidecar's storage (affine 4/64 everywhere).
pub const q4_storages: Storages = @splat(.{});

/// Whether a suffix is stored under `st`: BF16 projections carry no scales or biases, FP8 ones their f32 block scales
/// only.
fn present(s: Suffix, st: Storages) bool {
    return switch (s.kind) {
        .scales, .biases => !st[@backingInt(s.proj.?)].dense() and !st[@backingInt(s.proj.?)].fp8,
        .scale_inv => st[@backingInt(s.proj.?)].fp8,
        else => true,
    };
}

/// Tensors per head under `st` (30 for a quantized sidecar, 14 for BF16 projections, 20 for the fp8res sidecar).
pub fn headTensors(st: Storages) usize {
    var n: usize = 0;
    for (suffixes) |s| n += @intFromBool(present(s, st));
    return n;
}

/// A projection's (input, output) features from the trunk's config (the sliding layer's geometry).
fn projDims(c: mimo.Config, p: Proj) [2]u32 {
    return switch (p) {
        .eh => .{ 2 * c.hidden_size, c.hidden_size },
        .q => .{ c.hidden_size, c.n_heads * c.head_dim },
        .k => .{ c.hidden_size, c.swa_n_kv_heads * c.head_dim },
        .v => .{ c.hidden_size, c.swa_n_kv_heads * c.value_head_dim },
        .o => .{ c.n_heads * c.value_head_dim, c.hidden_size },
        .gate, .up => .{ c.hidden_size, c.intermediate_size },
        .down => .{ c.intermediate_size, c.hidden_size },
    };
}

pub const Dtype = enum { BF16, U32, U8, F32 };
pub const Shape = struct { dtype: Dtype, rank: u8, dims: [2]u64 };

/// The dtype and shape a suffix must have under the trunk's config and the projection's storage: affine (U32
/// [out, in*bits/32], BF16 scales and biases [out, in/group]), BF16 [out, in], or FP8 (U8 codes [out, in], F32 block
/// scales [out / 64, in / 128]); BF16 norms [H] and sinks [heads].
pub fn expectedShape(c: mimo.Config, s: Suffix, st: Storages) Shape {
    return switch (s.kind) {
        .norm => .{ .dtype = .BF16, .rank = 1, .dims = .{ c.hidden_size, 0 } },
        .sink => .{ .dtype = .BF16, .rank = 1, .dims = .{ c.n_heads, 0 } },
        .weight => blk: {
            const d = projDims(c, s.proj.?);
            const p = st[@backingInt(s.proj.?)];
            break :blk if (p.dense()) .{ .dtype = .BF16, .rank = 2, .dims = .{ d[1], d[0] } } else if (p.fp8) .{ .dtype = .U8, .rank = 2, .dims = .{ d[1], d[0] } } else .{ .dtype = .U32, .rank = 2, .dims = .{ d[1], d[0] * p.bits / 32 } };
        },
        .scales, .biases => .{ .dtype = .BF16, .rank = 2, .dims = .{ projDims(c, s.proj.?)[1], projDims(c, s.proj.?)[0] / st[@backingInt(s.proj.?)].group } },
        .scale_inv => .{ .dtype = .F32, .rank = 2, .dims = .{ projDims(c, s.proj.?)[1] / mimo.fp8_scale_rows, projDims(c, s.proj.?)[0] / mimo.fp8_scale_cols } },
    };
}

fn bytesOf(sh: Shape) u64 {
    var n: u64 = switch (sh.dtype) {
        .U32, .F32 => 4,
        .BF16 => 2,
        .U8 => 1,
    };
    for (sh.dims[0..sh.rank]) |d| n *= d;
    return n;
}

/// One head's payload under `c` and `st` (the bill's term per head; the manifest must agree).
pub fn headPayloadBytes(c: mimo.Config, st: Storages) u64 {
    var n: u64 = 0;
    for (suffixes) |s| if (present(s, st)) {
        n += bytesOf(expectedShape(c, s, st));
    };
    return n;
}

/// One tensor of the manifest, checked against the trunk's geometry.
pub const Entry = struct {
    head: u8,
    suffix: u8,
    offset: u64,
    length: u64,
    sha256: [32]u8,
};

/// The manifest, checked: geometry against the trunk, every present tensor (90 quantized, 42 BF16) where the
/// trunk's config and the projection's storage put it.
pub const Manifest = struct {
    entries: [n_heads * tensors_per_head]Entry,
    count: usize = 0,
    storage: Storages = q4_storages,
    data_size: u64,
    data_sha256: [32]u8,
    payload_bytes: u64,

    /// The manifest's statement checked against `c` (no tensor bytes read).
    pub fn parse(a: std.mem.Allocator, text: []const u8, c: mimo.Config) !Manifest {
        const parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return error.MimoMtpManifestSyntax;
        defer parsed.deinit();
        const root = parsed.value;
        if (root != .object) return error.MimoMtpManifestSyntax;
        const o = root.object;
        try expectString(o, "family", family);
        try expectString(o, "format", format);
        try expectString(o, "data_file", data_file);
        if (try uint(o, "head_count") != n_heads) return error.MimoMtpManifestGeometry;
        const cfg = o.get("config") orelse return error.MimoMtpManifestField;
        if (cfg != .object) return error.MimoMtpManifestField;
        const want = [_]struct { []const u8, u64 }{
            .{ "num_nextn_predict_layers", n_heads },      .{ "hidden_size", c.hidden_size },
            .{ "intermediate_size", c.intermediate_size }, .{ "num_attention_heads", c.n_heads },
            .{ "num_key_value_heads", c.swa_n_kv_heads },  .{ "head_dim", c.head_dim },
            .{ "v_head_dim", c.value_head_dim },           .{ "sliding_window", c.window },
            .{ "vocab_size", c.vocab_size },
        };
        for (want) |w| if (try uint(cfg.object, w[0]) != w[1]) return error.MimoMtpManifestGeometry;
        const fwant = [_]struct { []const u8, f32 }{
            .{ "rope_theta", c.swa_rope_theta }, .{ "attention_value_scale", c.attention_value_scale }, .{ "layernorm_epsilon", c.rms_norm_eps },
        };
        for (fwant) |w| if (@as(f32, @floatCast(try float(cfg.object, w[0]))) != w[1]) return error.MimoMtpManifestGeometry;
        const sink = cfg.object.get("attention_sink_bias") orelse return error.MimoMtpManifestField;
        if (sink != .bool or !sink.bool) return error.MimoMtpManifestGeometry;
        const quant = o.get("quantization") orelse return error.MimoMtpManifestField;
        if (quant != .object) return error.MimoMtpManifestField;
        var out: Manifest = .{ .entries = undefined, .data_size = try uint(o, "data_size"), .data_sha256 = try hex32(o.get("data_sha256")), .payload_bytes = try uint(o, "payload_bytes") };
        var name_buf: [128]u8 = undefined;
        // Each projection role's storage, once: the three heads must agree.
        for (0..n_heads) |h| for (proj_names, 0..) |base, p| {
            const prefix = try std.fmt.bufPrint(&name_buf, "model.mtp.layers.{d}.{s}", .{ h, base });
            const q = quant.object.get(prefix) orelse return error.MimoMtpManifestQuant;
            const st = mimo.Storage.fromJson(q) orelse return error.MimoMtpManifestQuant;
            if (h == 0) out.storage[p] = st else if (!out.storage[p].eql(st)) return error.MimoMtpManifestQuant;
        };
        // fp8res: the sidecar's flag is the trunk's (one resident route per module), and under it the heads are the
        // release's split: q/k/v/gate/up/down FP8, eh_proj and o_proj BF16; FP8 never appears without the flag.
        const residents: mimo.ResidentsFormat = if (o.get("residents_format")) |v| blk: {
            if (v != .string or !std.mem.eql(u8, v.string, @tagName(mimo.ResidentsFormat.fp8_block128))) return error.MimoMtpManifestFormat;
            break :blk .fp8_block128;
        } else .standard;
        if (residents != c.residents_format) return error.MimoMtpResidentsFormat;
        for (out.storage, 0..) |st, p| {
            const fp8_proj = p != @backingInt(Proj.eh) and p != @backingInt(Proj.o);
            const ok = switch (residents) {
                .standard => !st.fp8,
                .fp8_block128 => if (fp8_proj) st.eql(mimo.Storage.fp8_block128) else st.dense(),
            };
            if (!ok) return error.MimoMtpManifestQuant;
        }
        const tensors = o.get("tensors") orelse return error.MimoMtpManifestField;
        if (tensors != .object or tensors.object.count() != n_heads * headTensors(out.storage)) return error.MimoMtpManifestTensors;
        var total: u64 = 0;
        for (0..n_heads) |h| for (suffixes, 0..) |s, si| {
            if (!present(s, out.storage)) continue;
            const name = try std.fmt.bufPrint(&name_buf, "model.mtp.layers.{d}.{s}", .{ h, s.name });
            const t = tensors.object.get(name) orelse return error.MimoMtpManifestTensors;
            if (t != .object) return error.MimoMtpManifestTensors;
            const sh = expectedShape(c, s, out.storage);
            const dt = t.object.get("dtype") orelse return error.MimoMtpManifestTensors;
            if (dt != .string or !std.mem.eql(u8, dt.string, @tagName(sh.dtype))) return error.MimoMtpTensorShape;
            try expectDims(t.object.get("shape"), sh);
            const len = try uint(t.object, "length");
            if (len != bytesOf(sh)) return error.MimoMtpTensorShape;
            out.entries[out.count] = .{ .head = @intCast(h), .suffix = @intCast(si), .offset = try uint(t.object, "offset"), .length = len, .sha256 = try hex32(t.object.get("sha256")) };
            out.count += 1;
            total += len;
        };
        if (total != out.payload_bytes or out.payload_bytes != n_heads * headPayloadBytes(c, out.storage)) return error.MimoMtpManifestGeometry;
        return out;
    }

    /// Entries in file order (the payload is contiguous from offset 0: no gaps, no overlaps).
    fn sorted(m: *const Manifest) [n_heads * tensors_per_head]Entry {
        var e = m.entries;
        std.mem.sort(Entry, e[0..m.count], {}, struct {
            fn lt(_: void, x: Entry, y: Entry) bool {
                return x.offset < y.offset;
            }
        }.lt);
        return e;
    }
};

fn expectString(o: std.json.ObjectMap, k: []const u8, want: []const u8) !void {
    const v = o.get(k) orelse return error.MimoMtpManifestField;
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.MimoMtpManifestFormat;
}
fn uint(o: std.json.ObjectMap, k: []const u8) !u64 {
    const v = o.get(k) orelse return error.MimoMtpManifestField;
    if (v != .integer or v.integer < 0) return error.MimoMtpManifestField;
    return @intCast(v.integer);
}
fn float(o: std.json.ObjectMap, k: []const u8) !f64 {
    const v = o.get(k) orelse return error.MimoMtpManifestField;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.MimoMtpManifestField,
    };
}
fn hex32(v: ?std.json.Value) ![32]u8 {
    const s = v orelse return error.MimoMtpManifestField;
    if (s != .string or s.string.len != 64) return error.MimoMtpManifestField;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s.string) catch return error.MimoMtpManifestField;
    return out;
}
fn expectDims(v: ?std.json.Value, sh: Shape) !void {
    const a = v orelse return error.MimoMtpTensorShape;
    if (a != .array or a.array.items.len != sh.rank) return error.MimoMtpTensorShape;
    for (a.array.items, sh.dims[0..sh.rank]) |d, want| if (d != .integer or d.integer != want) return error.MimoMtpTensorShape;
}

/// The manifest's payload (the bill's resident term), checked against `c`; no tensor bytes read.
pub fn payloadBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: mimo.Config) !u64 {
    const text = try readManifest(a, io, dir);
    defer a.free(text);
    return (try Manifest.parse(a, text, c)).payload_bytes;
}

fn readManifest(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    return d.readFileAlloc(io, manifest_file, a, .limited(manifest_limit)) catch |e| switch (e) {
        error.FileNotFound => error.MimoMtpSidecarMissing,
        else => e,
    };
}

/// Every tensor's bytes, verified, in file order: the header's entries must be the manifest's (dtype, shape, offsets),
/// the file exactly header + payload, each tensor's sha256 and the file's sha256 as stated. `sink.take(head, suffix,
/// shape, bytes)` receives each tensor only after its hash matched (the bytes are reused after it returns).
pub fn verifySidecar(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: mimo.Config, sink: anytype) !Manifest {
    const text = try readManifest(a, io, dir);
    defer a.free(text);
    const m = try Manifest.parse(a, text, c);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, data_file }, 0);
    defer a.free(path);
    const fd = io_util.openNoCache(path.ptr, .{}) catch |e| return if (e == error.FileNotFound) error.MimoMtpSidecarMissing else e;
    defer _ = std.c.close(fd);
    const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size_i < 8 or @as(u64, @intCast(size_i)) != m.data_size) return error.MimoMtpSidecarSize;
    var prefix: [8]u8 = undefined;
    try io_util.readAligned(fd, &prefix, 0);
    const hlen = std.mem.readInt(u64, &prefix, .little);
    if (hlen < 2 or hlen > header_limit or 8 + hlen + m.payload_bytes != m.data_size) return error.MimoMtpSidecarHeader;
    const header = try a.alloc(u8, @intCast(hlen));
    defer a.free(header);
    try io_util.readAligned(fd, header, 8);
    try checkHeader(a, header, &m, c);
    var file_hash = std.crypto.hash.sha2.Sha256.init(.{});
    file_hash.update(&prefix);
    file_hash.update(header);
    var widest: u64 = 0;
    for (m.entries[0..m.count]) |e| widest = @max(widest, e.length);
    const buf = try std.heap.page_allocator.alloc(u8, @intCast(widest));
    defer std.heap.page_allocator.free(buf);
    var next: u64 = 0;
    const ordered = m.sorted();
    for (ordered[0..m.count]) |e| {
        if (e.offset != next) return error.MimoMtpSidecarLayout;
        const bytes = buf[0..@intCast(e.length)];
        try io_util.readAligned(fd, bytes, 8 + hlen + e.offset);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        if (!std.mem.eql(u8, &digest, &e.sha256)) return error.MimoMtpTensorSha;
        file_hash.update(bytes);
        try sink.take(e.head, e.suffix, expectedShape(c, suffixes[e.suffix], m.storage), bytes);
        next = e.offset + e.length;
    }
    var whole: [32]u8 = undefined;
    file_hash.final(&whole);
    if (!std.mem.eql(u8, &whole, &m.data_sha256)) return error.MimoMtpSidecarSha;
    return m;
}

fn checkHeader(a: std.mem.Allocator, header: []const u8, m: *const Manifest, c: mimo.Config) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.MimoMtpSidecarHeader;
    defer parsed.deinit();
    if (parsed.value != .object) return error.MimoMtpSidecarHeader;
    var count: usize = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| count += @intFromBool(!std.mem.eql(u8, kv.key_ptr.*, "__metadata__"));
    if (count != m.count) return error.MimoMtpSidecarHeader;
    var name_buf: [128]u8 = undefined;
    for (m.entries[0..m.count]) |e| {
        const s = suffixes[e.suffix];
        const name = try std.fmt.bufPrint(&name_buf, "model.mtp.layers.{d}.{s}", .{ e.head, s.name });
        const t = parsed.value.object.get(name) orelse return error.MimoMtpSidecarHeader;
        if (t != .object) return error.MimoMtpSidecarHeader;
        const sh = expectedShape(c, s, m.storage);
        const dt = t.object.get("dtype") orelse return error.MimoMtpSidecarHeader;
        if (dt != .string or !std.mem.eql(u8, dt.string, @tagName(sh.dtype))) return error.MimoMtpSidecarHeader;
        expectDims(t.object.get("shape"), sh) catch return error.MimoMtpSidecarHeader;
        const off = t.object.get("data_offsets") orelse return error.MimoMtpSidecarHeader;
        if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer or
            off.array.items[0].integer != e.offset or off.array.items[1].integer != e.offset + e.length) return error.MimoMtpSidecarHeader;
    }
}

/// One head's arrays (owned).
pub const Head = struct {
    enorm: Array = .{},
    hnorm: Array = .{},
    input_norm: Array = .{},
    pre_mlp_norm: Array = .{},
    final_norm: Array = .{},
    sinks: Array = .{},
    eh: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    q: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    k: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    v: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    o: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    gate: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    up: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },
    down: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} },

    fn slot(h: *Head, s: Suffix) *Array {
        return switch (s.kind) {
            .norm => if (std.mem.startsWith(u8, s.name, "enorm")) &h.enorm else if (std.mem.startsWith(u8, s.name, "hnorm")) &h.hnorm else if (std.mem.startsWith(u8, s.name, "input")) &h.input_norm else if (std.mem.startsWith(u8, s.name, "pre_mlp")) &h.pre_mlp_norm else &h.final_norm,
            .sink => &h.sinks,
            .weight, .scales, .biases, .scale_inv => blk: {
                const p = h.projection(s.proj.?);
                break :blk switch (s.kind) {
                    .weight => &p.w,
                    // An FP8 projection's f32 block scales live where an affine one's scales do (module.zig QLinear).
                    .scales, .scale_inv => &p.scales,
                    else => &p.biases,
                };
            },
        };
    }

    fn projection(h: *Head, p: Proj) *QLinear {
        return switch (p) {
            .eh => &h.eh,
            .q => &h.q,
            .k => &h.k,
            .v => &h.v,
            .o => &h.o,
            .gate => &h.gate,
            .up => &h.up,
            .down => &h.down,
        };
    }

    fn deinit(h: *Head) void {
        for (suffixes) |s| {
            const x = h.slot(s);
            drop(x.*);
            x.* = .{};
        }
    }
};

/// The three heads, loaded from the verified sidecar.
pub const Heads = struct {
    heads: [n_heads]Head = @splat(.{}),
    payload_bytes: u64 = 0,

    /// Construction only: the manifest and every tensor verified (`verifySidecar`), then the arrays; the FP8 projections
    /// of an fp8res sidecar bind to the module's FP8 route (`fp8`, installed from the trunk's flag; refused without it).
    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: mimo.Config, fp8: ?*const fp8_linear.Kernels) !Heads {
        var out: Heads = .{};
        errdefer out.deinit();
        const Sink = struct {
            h: *Heads,
            pub fn take(self: @This(), head: u8, suffix: u8, sh: Shape, bytes: []const u8) !void {
                var dims: [2]c_int = undefined;
                for (dims[0..sh.rank], sh.dims[0..sh.rank]) |*d, v| d.* = @intCast(v);
                const arr = mlx.mlx_array_new_data(bytes.ptr, &dims, sh.rank, switch (sh.dtype) {
                    .U32 => .uint32,
                    .BF16 => .bfloat16,
                    .U8 => .uint8,
                    .F32 => .float32,
                });
                if (arr.ctx == null) return error.MimoMtpArray;
                self.h.heads[head].slot(suffixes[suffix]).* = arr;
            }
        };
        const m = try verifySidecar(a, io, dir, c, Sink{ .h = &out });
        for (&out.heads) |*h| for (m.storage, 0..) |st, p| {
            const q = h.projection(@fromBackingInt(@intCast(p)));
            q.storage = st;
            if (st.fp8) try q.installFp8(fp8 orelse return error.MimoFp8RouteUnbound);
        };
        out.payload_bytes = m.payload_bytes;
        return out;
    }

    pub fn deinit(self: *Heads) void {
        for (&self.heads) |*h| h.deinit();
    }
};

/// The trunk arrays a head borrows: the shared embedding and lm_head, the sliding rotary frequencies, the value scale
/// and the unit norm weight (the trunk's own), and its geometry.
pub const Trunk = struct {
    a: std.mem.Allocator,
    s: Stream,
    embed: QLinear,
    lm_head: QLinear,
    rope_freq: Array,
    value_scale: Array,
    ones: Array,
    eps: f32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    v_dim: u32,
    rope_dim: u32,
    window: u32,
    /// R1 (rows audit, `"embedding": "rows"`): the trunk's table rows on the host; null = the resident table (`embed`).
    /// With rows, a head's ids are host values: `draft` reads each draft id back before the next head embeds it.
    rows: ?*embed_rows.Rows = null,
};

/// Rows [base, base + len) of one head's keys and values (BF16 [1, kv, len, d]), the newest last (upstream `RowCache`).
pub const HeadCache = struct {
    k: Array = .{},
    v: Array = .{},
    base: u64 = 0,
    len: u32 = 0,

    pub fn end(c: *const HeadCache) u64 {
        return c.base + c.len;
    }

    pub fn reset(c: *HeadCache, at: u64) void {
        drop(c.k);
        drop(c.v);
        c.* = .{ .base = at };
    }

    /// Drop every row at or past `at` (a draft row); a cut below the kept rows starts over at `at`.
    pub fn truncate(c: *HeadCache, s: Stream, at: u64) !void {
        if (at >= c.end()) return;
        if (at <= c.base) return c.reset(at);
        const keep: c_int = @intCast(at - c.base);
        try replace(&c.k, try tm.sliceTime(c.k, 0, keep, s));
        try replace(&c.v, try tm.sliceTime(c.v, 0, keep, s));
        c.len = @intCast(at - c.base);
    }

    /// Append rows from `pos0 == end()`, keeping the `window - 1` rows before them that the first new row attends.
    pub fn append(c: *HeadCache, s: Stream, k: Array, v: Array, pos0: u64, window: u32) !void {
        if (c.len == 0) c.base = pos0;
        if (pos0 != c.end()) return error.MimoMtpRowGap;
        const rows: u32 = @intCast(mlx.getShape(k)[2]);
        const keep = @min(c.len, window - 1);
        if (keep == 0) {
            drop(c.k);
            drop(c.v);
            c.k = try tm.retain(k);
            c.v = try tm.retain(v);
        } else {
            const old_k = try tm.sliceTime(c.k, @intCast(c.len - keep), @intCast(c.len), s);
            defer drop(old_k);
            const old_v = try tm.sliceTime(c.v, @intCast(c.len - keep), @intCast(c.len), s);
            defer drop(old_v);
            try replace(&c.k, try tm.concatenate(old_k, k, 2, s));
            try replace(&c.v, try tm.concatenate(old_v, v, 2, s));
        }
        c.base = pos0 - keep;
        c.len = keep + rows;
    }
};

fn replace(slot: *Array, next: Array) !void {
    drop(slot.*);
    slot.* = next;
}

/// One request's head state: committed final-normed trunk hiddens h_p and tokens x_p by absolute position, each
/// head's rows, and how far each head's rows carry committed tokens.
pub const State = struct {
    /// Final-normed trunk hiddens BF16 [n, H] at positions [hid_base, hid_base + hid_len).
    hid: Array = .{},
    hid_base: u64 = 0,
    hid_len: u32 = 0,
    /// Committed tokens x_p at positions [tok_base, tok_base + tok_len).
    toks: [2 * ring_rows]u32 = undefined,
    tok_base: u64 = 0,
    tok_len: u32 = 0,
    caches: [n_heads]HeadCache = @splat(.{}),
    /// Head k's rows past this carry a draft token; a round drops them first.
    committed_end: [n_heads]u64 = @splat(0),

    pub fn deinit(st: *State) void {
        drop(st.hid);
        for (&st.caches) |*c| c.reset(0);
        st.* = .{};
    }

    pub fn hidEnd(st: *const State) u64 {
        return st.hid_base + st.hid_len;
    }

    pub fn tokEnd(st: *const State) u64 {
        return st.tok_base + st.tok_len;
    }

    /// Committed hiddens h_{p0..p0+n} (`hidden` [n, H]) and tokens x_{p0..p0+ids.len}: a committed forward's tail.
    pub fn record(st: *State, s: Stream, p0: u64, hidden: Array, ids: []const u32) !void {
        try st.recordHidden(s, p0, hidden);
        try st.pushTokens(p0, ids);
    }

    /// Committed hiddens h_{p0..p0+n} (`hidden` [n, H], evaluated here). A gap (a position the lane never saw)
    /// restarts the ring at p0; the ring keeps the newest `ring_rows`.
    pub fn recordHidden(st: *State, s: Stream, p0: u64, hidden: Array) !void {
        try mlx.check(mlx.mlx_array_eval(hidden));
        const n: u32 = @intCast(mlx.getShape(hidden)[0]);
        if (st.hid_len == 0 or p0 != st.hidEnd()) {
            drop(st.hid);
            st.hid = try tm.retain(hidden);
            st.hid_base = p0;
            st.hid_len = n;
        } else {
            try replace(&st.hid, try tm.concatenate(st.hid, hidden, 0, s));
            st.hid_len += n;
        }
        if (st.hid_len > ring_rows) {
            const cut = st.hid_len - ring_rows;
            const view = try tm.slice2(st.hid, @intCast(cut), @intCast(st.hid_len), 0, mlx.getShape(st.hid)[1], s);
            defer drop(view);
            var owned: Array = .{};
            try mlx.check(mlx.mlx_contiguous(&owned, view, false, s));
            errdefer drop(owned);
            try mlx.check(mlx.mlx_array_eval(owned));
            try replace(&st.hid, owned);
            st.hid_base += cut;
            st.hid_len = ring_rows;
        }
    }

    /// Whether a round at position `p` (t1 = x_p pushed) can draft: the ring holds h_{p-1}, the tokens run through
    /// x_p, and every head's catch-up row (from the ring's first hidden) has its token.
    pub fn ready(st: *const State, p: u64) bool {
        return st.hid_len > 0 and st.hidEnd() == p and st.tokEnd() == p + 1 and st.tok_base <= st.hid_base + 1;
    }

    /// Tokens x_{p0..}: appended at the ring's end, or (a gap) the ring restarted at p0.
    pub fn pushTokens(st: *State, p0: u64, ids: []const u32) !void {
        const tail = ids[ids.len -| ring_rows..];
        const p = p0 + (ids.len - tail.len);
        if (st.tok_len == 0 or p != st.tokEnd()) {
            st.tok_base = p;
            st.tok_len = 0;
        }
        if (st.tok_len + tail.len > st.toks.len) {
            const cut = st.tok_len + @as(u32, @intCast(tail.len)) - ring_rows;
            std.mem.copyForwards(u32, st.toks[0 .. st.tok_len - cut], st.toks[cut..st.tok_len]);
            st.tok_len -= cut;
            st.tok_base += cut;
        }
        @memcpy(st.toks[st.tok_len..][0..tail.len], tail);
        st.tok_len += @intCast(tail.len);
    }

    fn token(st: *const State, p: u64) !u32 {
        if (p < st.tok_base or p >= st.tokEnd()) return error.MimoMtpTokenOutOfRing;
        return st.toks[@intCast(p - st.tok_base)];
    }

    fn hiddenRows(st: *const State, s: Stream, from: u64, to: u64) !Array {
        if (from < st.hid_base or to > st.hidEnd() or from >= to) return error.MimoMtpHiddenOutOfRing;
        return tm.slice2(st.hid, @intCast(from - st.hid_base), @intCast(to - st.hid_base), 0, mlx.getShape(st.hid)[1], s);
    }
};

/// Head `li` over rows [pos0, pos0 + rows): `ids` uint32 [rows] (row p's token x_{p+li+1}), `hid` BF16 [rows, H] the
/// trunk's final-normed hiddens. Appends the rows' keys and values; returns the final-normed rows [rows, H].
pub fn layerForward(t: *const Trunk, h: *const Head, cache: *HeadCache, ids: Array, hid: Array, pos0: u64) !Array {
    const emb = try t.embed.lookup(ids, t.s);
    defer drop(emb);
    return layerForwardEmb(t, h, cache, emb, hid, pos0);
}

/// `layerForward` over the rows' embeddings `emb` BF16 [rows, H] (the table's rows, from either source).
pub fn layerForwardEmb(t: *const Trunk, h: *const Head, cache: *HeadCache, emb: Array, hid: Array, pos0: u64) !Array {
    const s = t.s;
    const rows: u32 = @intCast(mlx.getShape(emb)[0]);
    const en = try tm.rmsNorm(emb, h.enorm, t.ones, t.eps, s);
    defer drop(en);
    const hn = try tm.rmsNorm(hid, h.hnorm, t.ones, t.eps, s);
    defer drop(hn);
    const cat = try tm.concatenate(en, hn, 1, s);
    defer drop(cat);
    const x = try h.eh.forward(cat, s);
    defer drop(x);
    const a_in = try tm.rmsNorm(x, h.input_norm, t.ones, t.eps, s);
    defer drop(a_in);
    const attn = try attention(t, h, cache, a_in, rows, pos0);
    defer drop(attn);
    const x2 = try tm.binary(mlx.mlx_add, x, attn, s);
    defer drop(x2);
    const m_in = try tm.rmsNorm(x2, h.pre_mlp_norm, t.ones, t.eps, s);
    defer drop(m_in);
    const gate = try h.gate.forward(m_in, s);
    defer drop(gate);
    const up = try h.up.forward(m_in, s);
    defer drop(up);
    const silu = try tm.siluBf16(gate, s);
    defer drop(silu);
    const mid = try tm.binary(mlx.mlx_multiply, silu, up, s);
    defer drop(mid);
    const mlp = try h.down.forward(mid, s);
    defer drop(mlp);
    const x3 = try tm.binary(mlx.mlx_add, x2, mlp, s);
    defer drop(x3);
    return tm.rmsNorm(x3, h.final_norm, t.ones, t.eps, s);
}

fn attention(t: *const Trunk, h: *const Head, cache: *HeadCache, x: Array, rows: u32, pos0: u64) !Array {
    const s = t.s;
    const n: c_int = @intCast(rows);
    const qp = try h.q.forward(x, s);
    defer drop(qp);
    const kp = try h.k.forward(x, s);
    defer drop(kp);
    const vp = try h.v.forward(x, s);
    defer drop(vp);
    const qt = try tm.headsView(qp, n, t.heads, t.head_dim, s);
    defer drop(qt);
    const kt = try tm.headsView(kp, n, t.kv_heads, t.head_dim, s);
    defer drop(kt);
    const vt = try tm.headsView(vp, n, t.kv_heads, t.v_dim, s);
    defer drop(vt);
    var rotary = try tm.rotaryTables(t.rope_freq, pos0, rows, s);
    defer rotary.deinit();
    const q = try tm.partialRope(qt, rotary, t.rope_dim, s);
    defer drop(q);
    const k = try tm.partialRope(kt, rotary, t.rope_dim, s);
    defer drop(k);
    const v = try tm.scaleValues(vt, t.value_scale, s);
    defer drop(v);
    try cache.append(s, k, v, pos0, t.window);
    const mask = try tm.causalMask(t.a, pos0, rows, cache.base, cache.len, t.window);
    defer drop(mask);
    var out: Array = .{};
    defer drop(out);
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(t.head_dim)));
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, cache.k, cache.v, scale, "array", mask, h.sinks, false, s));
    const transposed = try tm.transpose4(out, s);
    defer drop(transposed);
    const flat = try tm.reshape(transposed, &.{ n, @intCast(t.heads * t.v_dim) }, s);
    defer drop(flat);
    return h.o.forward(flat, s);
}

/// Head `li`'s rows from wherever its rows end through `q` (the round's last committed position), the last `drafted`
/// of them carrying this round's drafts (`drafts`, uint32 [1] device arrays) and the rest committed tokens. Returns
/// the last row, final-normed [1, H].
fn forwardRows(t: *const Trunk, h: *const Head, st: *State, li: usize, q: u64, drafts: []const Array, host_drafts: []const u32) !Array {
    const s = t.s;
    const cache = &st.caches[li];
    // Rows before the last query's window never reach a later query either.
    const floor = (q + 1) -| t.window;
    const start = @max(@max(cache.end(), floor), st.hid_base);
    if (cache.end() < start) cache.reset(start);
    if (start > q) return error.MimoMtpNothingToForward;
    const rows: u32 = @intCast(q + 1 - start);
    const tentative: u32 = @min(@as(u32, @intCast(drafts.len)), rows);
    var host: [ring_rows + mimo.prefill_chunk]u32 = undefined;
    const committed = rows - tentative;
    if (committed > host.len) return error.MimoMtpHiddenOutOfRing;
    for (host[0..committed], 0..) |*x, i| x.* = try st.token(start + i + li + 1);
    const hid = try st.hiddenRows(s, start, q + 1);
    defer drop(hid);
    const out = if (t.rows) |er| blk: {
        // R1: every row's token on the host (the drafts read back by `draft`), its embedding from the table's rows.
        std.debug.assert(host_drafts.len == drafts.len);
        if (committed + tentative > host.len) return error.MimoMtpHiddenOutOfRing;
        @memcpy(host[committed..][0..tentative], host_drafts[host_drafts.len - tentative ..]);
        const emb = try er.lookup(t.a, host[0 .. committed + tentative]);
        defer drop(emb);
        break :blk try layerForwardEmb(t, h, cache, emb, hid, start);
    } else blk: {
        var parts: [max_drafts + 1]Array = undefined;
        var np: usize = 0;
        const committed_ids = mlx.mlx_array_new_data(&host, &[_]c_int{@intCast(committed)}, 1, .uint32);
        defer drop(committed_ids);
        if (committed > 0) {
            parts[np] = committed_ids;
            np += 1;
        }
        // Row p's token x_{p+li+1} is draft d_{p+li-q}.
        for (drafts[drafts.len - tentative ..]) |d| {
            parts[np] = d;
            np += 1;
        }
        const vec = mlx.mlx_vector_array_new_data(&parts, np);
        defer _ = mlx.mlx_vector_array_free(vec);
        var ids: Array = .{};
        defer drop(ids);
        try mlx.check(mlx.mlx_concatenate_axis(&ids, vec, 0, s));
        break :blk try layerForward(t, h, cache, ids, hid, start);
    };
    defer drop(out);
    st.committed_end[li] = q + 1 - tentative;
    return tm.slice2(out, @intCast(rows - 1), @intCast(rows), 0, mlx.getShape(out)[1], s);
}

/// The round's drafts at last committed position q: d1 from head 0, d_{i+1} from head i over d1..d_i, `k` drafts
/// (k <= max_drafts), lazy uint32 [1] each in `out` (owned by the caller). Each head first drops its draft rows of the
/// previous round. The lm_head readout is the trunk's (no trunk final norm: the head's final_layernorm precedes it).
pub fn draft(t: *const Trunk, heads: *const Heads, st: *State, q: u64, k: u32, out: []Array) !void {
    std.debug.assert(k <= max_drafts and k <= n_heads and out.len >= k);
    var made: usize = 0;
    errdefer for (out[0..made]) |d| drop(d);
    // R1: on the rows route the next head embeds this head's draft from the host rows: its id is read back first.
    var host_drafts: [max_drafts]u32 = undefined;
    for (0..k) |i| {
        try st.caches[i].truncate(t.s, st.committed_end[i]);
        const last = try forwardRows(t, &heads.heads[i], st, i, q, out[0..i], host_drafts[0..i]);
        defer drop(last);
        const logits = try t.lm_head.forward(last, t.s);
        defer drop(logits);
        var arg: Array = .{};
        defer drop(arg);
        try mlx.check(mlx.mlx_argmax_axis(&arg, logits, -1, false, t.s));
        out[i] = try tm.cast(arg, .uint32, t.s);
        made += 1;
        if (t.rows != null and i + 1 < k) {
            try mlx.check(mlx.mlx_array_eval(out[i]));
            host_drafts[i] = (mlx.mlx_array_data_uint32(out[i]) orelse return error.MimoDraftRead)[0];
        }
    }
}

/// The lane's verify-row and head workspace (the bill's `mtp_workspace`): the decision graph over the verify rows (F32
/// logits and up to eight row-sized intermediates: log-softmax, p, entropy products / the sampled path's tempered,
/// sorted, cumulative rows), a head's catch-up forward at its widest (window + drafts rows: embedding, norms, the cat,
/// q/k/v/o, the F32 SiLU and the SwiGLU, F32 scores against window + rows keys), the heads' row caches with their
/// concatenation transient, the hidden ring with its concatenation, and the drafts' F32 lm_head rows.
pub fn workspaceBytes(c: mimo.Config) u64 {
    const V: u64 = c.vocab_size;
    const H: u64 = c.hidden_size;
    const I: u64 = c.intermediate_size;
    const R: u64 = c.window + max_drafts;
    const kv_row: u64 = @as(u64, c.swa_n_kv_heads) * (c.head_dim + c.value_head_dim) * 2;
    const decision = 9 * @as(u64, verify_rows) * V * 4;
    const head_rows = R * (8 * H * 2 + 2 * H * 2 + (@as(u64, c.n_heads) * (c.head_dim + c.value_head_dim) + kv_row / 2) * 2 + I * (2 + 2 + 4 + 4 + 2)) +
        @as(u64, c.n_heads) * R * (c.window + R) * 4 + R * (c.window + R);
    const caches = n_heads * 2 * (c.window + R) * kv_row;
    const ring = 2 * @as(u64, ring_rows) * H * 2 + @as(u64, ring_rows) * H * 2;
    const draft_logits = @as(u64, max_drafts) * V * (2 + 4);
    return decision + head_rows + caches + ring + draft_logits;
}

const testing = std.testing;

/// The MLX tests' gate: any MLX array constructs the Metal device (the CPU stream's too), so they run only inside a
/// guarded window (`_GPU_WINDOW_LOCKED`) and when asked (`MIMO_MTP_MLX`).
pub fn requireMlxWindow() !void {
    _ = std.c.getenv("MIMO_MTP_MLX") orelse return error.SkipZigTest;
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GpuLockRequired;
}

test "mimo mtp: the head's tensor table under the trunk's geometry (30 per head, 329,273,408 params)" {
    const c: mimo.Config = .{};
    // The table holds every storage's suffixes (fp8res added the FP8 block scales); a Q4 head stores 30 of them.
    try testing.expectEqual(@as(usize, 38), tensors_per_head);
    try testing.expectEqual(@as(usize, 30), headTensors(q4_storages));
    var by_kind: [6]usize = @splat(0);
    for (suffixes) |s| by_kind[@backingInt(s.kind)] += 1;
    try testing.expectEqualSlices(usize, &.{ 5, 1, 8, 8, 8, 8 }, &by_kind);
    const want = [_]struct { []const u8, Shape }{
        .{ "eh_proj.weight", .{ .dtype = .U32, .rank = 2, .dims = .{ 4096, 1024 } } },
        .{ "self_attn.q_proj.weight", .{ .dtype = .U32, .rank = 2, .dims = .{ 12288, 512 } } },
        .{ "self_attn.k_proj.scales", .{ .dtype = .BF16, .rank = 2, .dims = .{ 1536, 64 } } },
        .{ "self_attn.v_proj.biases", .{ .dtype = .BF16, .rank = 2, .dims = .{ 1024, 64 } } },
        .{ "self_attn.o_proj.weight", .{ .dtype = .U32, .rank = 2, .dims = .{ 4096, 1024 } } },
        .{ "mlp.down_proj.weight", .{ .dtype = .U32, .rank = 2, .dims = .{ 4096, 2048 } } },
        .{ "mlp.gate_proj.scales", .{ .dtype = .BF16, .rank = 2, .dims = .{ 16384, 64 } } },
        .{ "self_attn.attention_sink_bias", .{ .dtype = .BF16, .rank = 1, .dims = .{ 64, 0 } } },
        .{ "hnorm.weight", .{ .dtype = .BF16, .rank = 1, .dims = .{ 4096, 0 } } },
    };
    for (want) |w| {
        const s = for (suffixes) |s| {
            if (std.mem.eql(u8, s.name, w[0])) break s;
        } else return error.TestUnexpectedResult;
        try testing.expectEqual(w[1], expectedShape(c, s, q4_storages));
    }
    // The receipt's per-head payload (repack-v2: 185,245,824 B; 555,737,472 for three heads).
    try testing.expectEqual(@as(u64, 185_245_824), headPayloadBytes(c, q4_storages));
    // BF16 projections: 14 tensors per head, the weights [out, in] BF16.
    const bf16: Storages = @splat(mimo.Storage.bf16);
    try testing.expectEqual(@as(usize, 14), headTensors(bf16));
    const eh = for (suffixes) |s| {
        if (std.mem.eql(u8, s.name, "eh_proj.weight")) break s;
    } else return error.TestUnexpectedResult;
    try testing.expectEqual(Shape{ .dtype = .BF16, .rank = 2, .dims = .{ 4096, 8192 } }, expectedShape(c, eh, bf16));
    try testing.expectEqual(@as(u64, 2 * (4096 * 8192 + 4096 * 12288 + 4096 * 1536 + 4096 * 1024 + 8192 * 4096 + 3 * 16384 * 4096) + 2 * (5 * 4096 + 64)), headPayloadBytes(c, bf16));
    try testing.expectEqual(@as(u32, 4), verify_rows);
}

test "mimo mtp: a synthetic sidecar verifies tensor by tensor, and a flipped byte, a renamed tensor or wrong geometry refuse by name" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = tinyConfig();
    try writeSidecar(a, io, tmp.dir, c, .none, q4_storages);
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    const Count = struct {
        n: *usize,
        bytes: *u64,
        pub fn take(self: @This(), _: u8, _: u8, sh: Shape, b: []const u8) !void {
            try testing.expectEqual(bytesOf(sh), b.len);
            self.n.* += 1;
            self.bytes.* += b.len;
        }
    };
    var n: usize = 0;
    var bytes: u64 = 0;
    const m = try verifySidecar(a, io, path, c, Count{ .n = &n, .bytes = &bytes });
    try testing.expectEqual(@as(usize, 90), n);
    try testing.expectEqual(m.payload_bytes, bytes);
    try testing.expectEqual(3 * headPayloadBytes(c, q4_storages), try payloadBytes(a, io, path, c));
    inline for (.{ .{ Corrupt.flip, error.MimoMtpTensorSha }, .{ Corrupt.rename, error.MimoMtpManifestTensors }, .{ Corrupt.file_sha, error.MimoMtpSidecarSha } }) |case| {
        try writeSidecar(a, io, tmp.dir, c, case[0], q4_storages);
        try testing.expectError(case[1], verifySidecar(a, io, path, c, Count{ .n = &n, .bytes = &bytes }));
    }
    // The BF16 tier's sidecar (14 tensors per head) and the DSV-recipe tier's (eh BF16, the rest affine 8/32).
    var dsv: Storages = @splat(.{ .bits = 8, .group = 32 });
    dsv[@backingInt(Proj.eh)] = mimo.Storage.bf16;
    for ([_]Storages{ @splat(mimo.Storage.bf16), dsv }) |st| {
        try writeSidecar(a, io, tmp.dir, c, .none, st);
        n = 0;
        bytes = 0;
        const tier = try verifySidecar(a, io, path, c, Count{ .n = &n, .bytes = &bytes });
        try testing.expectEqual(n_heads * headTensors(st), n);
        try testing.expectEqual(3 * headPayloadBytes(c, st), bytes);
        for (tier.storage, st) |got, want| try testing.expect(got.eql(want));
    }
    try writeSidecar(a, io, tmp.dir, c, .none, q4_storages);
    var other = c;
    other.window = c.window * 2;
    try testing.expectError(error.MimoMtpManifestGeometry, verifySidecar(a, io, path, other, Count{ .n = &n, .bytes = &bytes }));
    try tmp.dir.deleteFile(io, data_file);
    try testing.expectError(error.MimoMtpSidecarMissing, verifySidecar(a, io, path, c, Count{ .n = &n, .bytes = &bytes }));
}

// MIMO_PACK=<the repacked pack>: the real sidecar's 90 tensors and the file hash verify against the real config
// (555,747,768 B read past the page cache; no MLX).
test "mimo mtp: the pack's sidecar verifies every tensor's sha256 and the file's (MIMO_PACK)" {
    const dir = std.mem.span(std.c.getenv("MIMO_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", a, .limited(16 << 20));
    defer a.free(text);
    const c = try mimo.Config.parse(a, text, null);
    const Count = struct {
        n: *usize,
        pub fn take(self: @This(), head: u8, _: u8, sh: Shape, b: []const u8) !void {
            try testing.expect(head < n_heads);
            try testing.expectEqual(bytesOf(sh), b.len);
            self.n.* += 1;
        }
    };
    var n: usize = 0;
    const m = try verifySidecar(a, io, dir, c, Count{ .n = &n });
    try testing.expectEqual(n_heads * headTensors(m.storage), n);
    try testing.expectEqual(m.payload_bytes, 3 * headPayloadBytes(c, m.storage));
    std.debug.print("MIMO_MTP_PACK tensors={d} payload={d} data_size={d}\n", .{ n, m.payload_bytes, m.data_size });
}

/// A tiny head geometry (every projection a multiple of the Q4 group; rotary dims 64 as the trunk's tables).
pub fn tinyConfig() mimo.Config {
    var c: mimo.Config = .{};
    c.vocab_size = 96;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.n_heads = 2;
    c.swa_n_kv_heads = 1;
    c.head_dim = 64;
    c.value_head_dim = 64;
    c.rope_head_dim = 64;
    c.window = 6;
    return c;
}

const Corrupt = enum { none, flip, rename, file_sha };

/// The tiny sidecar: seeded bytes per tensor (affine Q4 payloads as random words; scales small, biases centred),
/// a manifest with each tensor's sha256 and the file's.
pub fn writeSidecar(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: mimo.Config, corrupt: Corrupt, st: Storages) !void {
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(a);
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var tensors: std.ArrayList(u8) = .empty;
    defer tensors.deinit(a);
    var quant: std.ArrayList(u8) = .empty;
    defer quant.deinit(a);
    var rng: std.Random.DefaultPrng = .init(0x6d74_7031);
    try header.appendSlice(a, "{\"__metadata__\":{\"format\":\"mlx\"}");
    var name_buf: [128]u8 = undefined;
    for (0..n_heads) |h| for (suffixes) |s| {
        if (!present(s, st)) continue;
        const sh = expectedShape(c, s, st);
        const len = bytesOf(sh);
        const off = payload.items.len;
        const bytes = try payload.addManyAsSlice(a, @intCast(len));
        switch (sh.dtype) {
            .U32, .U8 => rng.random().bytes(bytes),
            .F32 => for (std.mem.bytesAsSlice(f32, bytes)) |*x| {
                x.* = 0.001 * (1 + rng.random().float(f32));
            },
            .BF16 => for (std.mem.bytesAsSlice(u16, bytes)) |*x| {
                const v: f32 = switch (s.kind) {
                    .scales => 0.002 + 0.004 * rng.random().float(f32),
                    .biases => -0.015 + 0.002 * rng.random().floatNorm(f32),
                    .norm => 1.0 + 0.1 * rng.random().floatNorm(f32),
                    else => 0.5 * rng.random().floatNorm(f32),
                };
                x.* = @truncate(@as(u32, @bitCast(v)) >> 16);
            },
        }
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        if (corrupt == .flip and h == 1 and s.kind == .sink) bytes[0] ^= 1;
        const name = try std.fmt.bufPrint(&name_buf, "model.mtp.layers.{d}.{s}", .{ h, s.name });
        const listed = if (corrupt == .rename and h == 2 and s.kind == .sink) "model.mtp.layers.2.self_attn.sinks" else name;
        const dims = if (sh.rank == 1) try std.fmt.allocPrint(a, "[{d}]", .{sh.dims[0]}) else try std.fmt.allocPrint(a, "[{d},{d}]", .{ sh.dims[0], sh.dims[1] });
        defer a.free(dims);
        try header.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ name, @tagName(sh.dtype), dims, off, off + len });
        try tensors.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"offset\":{d},\"length\":{d},\"sha256\":\"{s}\"}}", .{ if (tensors.items.len == 0) "" else ",", listed, @tagName(sh.dtype), dims, off, len, &std.fmt.bytesToHex(digest, .lower) });
        if (s.kind == .weight) {
            const p = st[@backingInt(s.proj.?)];
            try quant.print(a, "{s}\"{s}\":", .{ if (quant.items.len == 0) "" else ",", name[0 .. name.len - ".weight".len] });
            if (p.dense()) try quant.appendSlice(a, "false") else if (p.fp8) try quant.appendSlice(a, "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}") else try quant.print(a, "{{\"mode\":\"affine\",\"bits\":{d},\"group_size\":{d}}}", .{ p.bits, p.group });
        }
    };
    try header.append(a, '}');
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var lenb: [8]u8 = undefined;
    std.mem.writeInt(u64, &lenb, header.items.len, .little);
    try file.appendSlice(a, &lenb);
    try file.appendSlice(a, header.items);
    try file.appendSlice(a, payload.items);
    var whole: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(file.items, &whole, .{});
    if (corrupt == .file_sha) whole[0] ^= 0xff;
    try dir.writeFile(io, .{ .sub_path = data_file, .data = file.items });
    const manifest = try std.fmt.allocPrint(a,
        \\{{"family":"mlx-stream","format":"mlx-stream-mtp-heads-v1","data_file":"mtp-heads.safetensors","data_size":{d},"data_sha256":"{s}","payload_bytes":{d},"head_count":3,
        \\"config":{{"num_nextn_predict_layers":3,"hidden_size":{d},"intermediate_size":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"v_head_dim":{d},"sliding_window":{d},"attention_sink_bias":true,"rope_theta":{d},"attention_value_scale":{d},"layernorm_epsilon":{d},"vocab_size":{d}}},
        \\"quantization":{{{s}}},"tensors":{{{s}}}}}
    , .{ file.items.len, &std.fmt.bytesToHex(whole, .lower), payload.items.len, c.hidden_size, c.intermediate_size, c.n_heads, c.swa_n_kv_heads, c.head_dim, c.value_head_dim, c.window, c.swa_rope_theta, c.attention_value_scale, c.rms_norm_eps, c.vocab_size, quant.items, tensors.items });
    defer a.free(manifest);
    try dir.writeFile(io, .{ .sub_path = manifest_file, .data = manifest });
}

fn bf16Bits(v: f32) u16 {
    const r: u32 = @bitCast(v);
    return @truncate((r +% 0x7fff +% ((r >> 16) & 1)) >> 16);
}

/// A seeded affine Q4 g64 projection [out, in] (random codes, small scales, centred biases).
fn tinyLinear(rng: *std.Random.DefaultPrng, out: u32, in: u32) QLinear {
    var w: [96 * 64 / 8]u32 = undefined;
    var sc: [96]u16 = undefined;
    var bi: [96]u16 = undefined;
    const n = out * in / 8;
    for (w[0..n]) |*x| x.* = rng.random().int(u32);
    for (sc[0 .. out * in / 64], bi[0 .. out * in / 64]) |*x, *y| {
        x.* = bf16Bits(0.01 + 0.02 * rng.random().float(f32));
        y.* = bf16Bits(-0.12 + 0.02 * rng.random().floatNorm(f32));
    }
    return .{
        .w = mlx.mlx_array_new_data(&w, &.{ @intCast(out), @intCast(in / 8) }, 2, .uint32),
        .scales = mlx.mlx_array_new_data(&sc, &.{ @intCast(out), @intCast(in / 64) }, 2, .bfloat16),
        .biases = mlx.mlx_array_new_data(&bi, &.{ @intCast(out), @intCast(in / 64) }, 2, .bfloat16),
    };
}

fn freeLinear(q: QLinear) void {
    drop(q.w);
    drop(q.scales);
    drop(q.biases);
}

/// Two rounds of three chained drafts over a recorded history; returns the six drafts. Checks the head rows' shape
/// and commitment after the first round, and that the second round's rows trimmed back to the first round's commit
/// are bit-equal to them.
fn tinyRounds(t: *const Trunk, heads: *const Heads, hid: Array, toks: []const u32, out: *[6]u32) !void {
    const s = t.s;
    var st: State = .{};
    defer st.deinit();
    const first = try tm.slice2(hid, 0, 10, 0, mlx.getShape(hid)[1], s);
    defer drop(first);
    try st.record(s, 0, first, toks[0..10]);
    // Round 1 at P = 10 (t1 = x_10), q = 9.
    try st.pushTokens(10, toks[10..11]);
    try testing.expect(st.ready(10));
    var d: [max_drafts]Array = undefined;
    try draft(t, heads, &st, 9, 3, &d);
    {
        defer for (d) |x| drop(x);
        for (d, 0..) |x, i| {
            try mlx.check(mlx.mlx_array_eval(x));
            try testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(x));
            out[i] = (mlx.mlx_array_data_uint32(x) orelse return error.TestUnexpectedResult)[0];
            try testing.expect(out[i] < 96);
        }
    }
    // Head i's rows: positions [q + 1 - window, q] (window 6), the last i carrying drafts.
    for (st.caches, 0..) |c, i| {
        try testing.expectEqual(@as(u64, 4), c.base);
        try testing.expectEqual(@as(u32, 6), c.len);
        try testing.expectEqualSlices(c_int, &.{ 1, 1, 6, 64 }, mlx.getShape(c.k));
        try testing.expectEqual(@as(u64, 10 - i), st.committed_end[i]);
    }
    // Commit one accepted draft: h_10, h_11; x_11 = d1; then round 2 at P = 12 (t1 = x_12), q = 11.
    const kept = try tm.slice2(hid, 10, 12, 0, mlx.getShape(hid)[1], s);
    defer drop(kept);
    try st.recordHidden(s, 10, kept);
    try st.pushTokens(11, out[0..1]);
    try st.pushTokens(12, toks[12..13]);
    try testing.expect(st.ready(12));
    var snap: [n_heads]struct { k: Array, v: Array, end: u64, len: u32 } = undefined;
    for (&st.caches, st.committed_end, &snap) |*c, ce, *sn| {
        try c.truncate(s, ce);
        try mlx.check(mlx.mlx_array_eval(c.k));
        sn.* = .{ .k = try tm.retain(c.k), .v = try tm.retain(c.v), .end = c.end(), .len = c.len };
    }
    defer for (snap) |sn| {
        drop(sn.k);
        drop(sn.v);
    };
    try draft(t, heads, &st, 11, 3, &d);
    defer for (d) |x| drop(x);
    for (d, 3..) |x, i| {
        try mlx.check(mlx.mlx_array_eval(x));
        out[i] = (mlx.mlx_array_data_uint32(x) orelse return error.TestUnexpectedResult)[0];
    }
    // The trim keeps every row a later query can reach (the window's last window - 1); the round's append let only
    // older ones go. Those kept rows are the pre-round rows, bit for bit.
    for (&st.caches, snap) |*c, sn| {
        try c.truncate(s, sn.end);
        try testing.expectEqual(sn.end, c.end());
        try testing.expect(c.len >= @min(sn.len, t.window - 1) and c.len <= sn.len);
        const k_tail = try tm.sliceTime(sn.k, @intCast(sn.len - c.len), @intCast(sn.len), s);
        defer drop(k_tail);
        const v_tail = try tm.sliceTime(sn.v, @intCast(sn.len - c.len), @intCast(sn.len), s);
        defer drop(v_tail);
        try testing.expect(try tm.bitEqual(c.k, k_tail, s) and try tm.bitEqual(c.v, v_tail, s));
    }
}

test "mimo mtp: three chained drafts on a tiny head (shapes, committed rows), a draft-row trim restores each head's rows bit-equal, a replay drafts the same ids (MIMO_MTP_MLX, guarded window, MLX CPU stream)" {
    try requireMlxWindow();
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = tinyConfig();
    try writeSidecar(a, io, tmp.dir, c, .none, q4_storages);
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    var heads = try Heads.load(a, io, path, c, null);
    defer heads.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    var rng: std.Random.DefaultPrng = .init(0x7469_6e79);
    const embed = tinyLinear(&rng, c.vocab_size, c.hidden_size);
    defer freeLinear(embed);
    const lm_head = tinyLinear(&rng, c.vocab_size, c.hidden_size);
    defer freeLinear(lm_head);
    const freq = tm.inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
    defer drop(freq);
    const scale = mlx.mlx_array_new_float(c.attention_value_scale);
    defer drop(scale);
    var ones: Array = .{};
    try mlx.check(mlx.mlx_ones(&ones, &.{@intCast(c.hidden_size)}, 1, .bfloat16, s));
    defer drop(ones);
    const t: Trunk = .{ .a = a, .s = s, .embed = embed, .lm_head = lm_head, .rope_freq = freq, .value_scale = scale, .ones = ones, .eps = c.rms_norm_eps, .heads = c.n_heads, .kv_heads = c.swa_n_kv_heads, .head_dim = c.head_dim, .v_dim = c.value_head_dim, .rope_dim = c.rope_head_dim, .window = c.window };
    var hb: [16 * 64]u16 = undefined;
    for (&hb) |*x| x.* = bf16Bits(rng.random().floatNorm(f32));
    const hid = mlx.mlx_array_new_data(&hb, &.{ 16, 64 }, 2, .bfloat16);
    defer drop(hid);
    var toks: [16]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast((i * 7 + 3) % 96);
    var first: [6]u32 = undefined;
    var replay: [6]u32 = undefined;
    try tinyRounds(&t, &heads, hid, &toks, &first);
    try tinyRounds(&t, &heads, hid, &toks, &replay);
    try testing.expectEqualSlices(u32, &first, &replay);
}

test "mimo mtp: the token ring appends by position, restarts at a gap, keeps the newest rows; a round is ready only with h_{p-1} and x_p (host)" {
    var st: State = .{};
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i);
    try st.pushTokens(0, ids[0..10]);
    try testing.expectEqual(@as(u64, 10), st.tokEnd());
    try st.pushTokens(10, ids[10..11]);
    try testing.expectEqual(@as(u32, 10), try st.token(10));
    // A gap restarts the ring at the new position.
    try st.pushTokens(20, ids[20..22]);
    try testing.expectEqual(@as(u64, 20), st.tok_base);
    try testing.expectError(error.MimoMtpTokenOutOfRing, st.token(10));
    // Past the ring: the newest ring_rows stay.
    try st.pushTokens(22, ids[22..300]);
    try testing.expectEqual(@as(u64, 300), st.tokEnd());
    try testing.expect(st.tok_len <= st.toks.len and st.tok_len >= ring_rows);
    try testing.expectEqual(@as(u32, 299), try st.token(299));
    try testing.expectEqual(@as(u32, 300 - st.tok_len), try st.token(300 - st.tok_len));
    // ready(p): the hiddens end at p, the tokens at p + 1, and reach back to the first hidden's successor.
    st.hid_base = 200;
    st.hid_len = 100;
    try testing.expect(!st.ready(300)); // x_300 not pushed yet
    try st.pushTokens(300, &.{7});
    try testing.expect(st.ready(300));
    st.hid_len = 99;
    try testing.expect(!st.ready(300)); // h_299 missing
    st.hid_base = 10;
    st.hid_len = 290;
    try testing.expect(!st.ready(300)); // tokens no longer reach x_11
}

// ---- fp8res (`residents_format: "fp8_block128"`): begin ----

/// The fp8res sidecar's storages: the release's split (q/k/v/gate/up/down FP8, eh_proj and o_proj BF16).
pub const fp8_storages: Storages = blk: {
    var st: Storages = @splat(mimo.Storage.fp8_block128);
    st[@backingInt(Proj.eh)] = mimo.Storage.bf16;
    st[@backingInt(Proj.o)] = mimo.Storage.bf16;
    break :blk st;
};

const zero_sha = "0000000000000000000000000000000000000000000000000000000000000000";
/// A manifest (no data file) for `st` at the trunk's geometry: every present tensor listed contiguously, `flag` the
/// residents_format line (or empty), `bad_scale` a wrong grid on head 1's q scales.
fn manifestText(a: std.mem.Allocator, c: mimo.Config, st: Storages, flag: []const u8, bad_scale: bool) ![]u8 {
    var tensors: std.ArrayList(u8) = .empty;
    defer tensors.deinit(a);
    var quant: std.ArrayList(u8) = .empty;
    defer quant.deinit(a);
    var off: u64 = 0;
    var name_buf: [128]u8 = undefined;
    for (0..n_heads) |h| for (suffixes) |s| {
        if (!present(s, st)) continue;
        var sh = expectedShape(c, s, st);
        if (bad_scale and h == 1 and s.kind == .scale_inv and s.proj.? == .q) sh.dims[0] *= 2;
        const len = bytesOf(sh);
        const name = try std.fmt.bufPrint(&name_buf, "model.mtp.layers.{d}.{s}", .{ h, s.name });
        const dims = if (sh.rank == 1) try std.fmt.allocPrint(a, "[{d}]", .{sh.dims[0]}) else try std.fmt.allocPrint(a, "[{d},{d}]", .{ sh.dims[0], sh.dims[1] });
        defer a.free(dims);
        try tensors.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"offset\":{d},\"length\":{d},\"sha256\":\"{s}\"}}", .{ if (tensors.items.len == 0) "" else ",", name, @tagName(sh.dtype), dims, off, len, zero_sha });
        off += len;
        if (s.kind == .weight) {
            const p = st[@backingInt(s.proj.?)];
            try quant.print(a, "{s}\"{s}\":", .{ if (quant.items.len == 0) "" else ",", name[0 .. name.len - ".weight".len] });
            if (p.dense()) try quant.appendSlice(a, "false") else if (p.fp8) try quant.appendSlice(a, "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}") else try quant.print(a, "{{\"mode\":\"affine\",\"bits\":{d},\"group_size\":{d}}}", .{ p.bits, p.group });
        }
    };
    return std.fmt.allocPrint(a,
        \\{{{s}"family":"mlx-stream","format":"mlx-stream-mtp-heads-v1","data_file":"mtp-heads.safetensors","data_size":{d},"data_sha256":"{s}","payload_bytes":{d},"head_count":3,
        \\"config":{{"num_nextn_predict_layers":3,"hidden_size":{d},"intermediate_size":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"v_head_dim":{d},"sliding_window":{d},"attention_sink_bias":true,"rope_theta":{d},"attention_value_scale":{d},"layernorm_epsilon":{d},"vocab_size":{d}}},
        \\"quantization":{{{s}}},"tensors":{{{s}}}}}
    , .{ flag, off + 1024, zero_sha, off, c.hidden_size, c.intermediate_size, c.n_heads, c.swa_n_kv_heads, c.head_dim, c.value_head_dim, c.window, c.swa_rope_theta, c.attention_value_scale, c.rms_norm_eps, c.vocab_size, quant.items, tensors.items });
}

test "mimo mtp fp8res: the sidecar's FP8 projections parse under the trunk's flag (20 tensors and 396,530,816 B per head); mismatched flags, another split and a wrong scale grid refuse by name" {
    const a = testing.allocator;
    var c: mimo.Config = .{};
    c.residents_format = .fp8_block128;
    const flag = "\"residents_format\":\"fp8_block128\",";
    {
        const text = try manifestText(a, c, fp8_storages, flag, false);
        defer a.free(text);
        const m = try Manifest.parse(a, text, c);
        for (m.storage, fp8_storages) |got, want| try testing.expect(got.eql(want));
        try testing.expectEqual(@as(usize, 20), headTensors(fp8_storages));
        try testing.expectEqual(@as(usize, n_heads * 20), m.count);
        // The pack's sidecar: 1,189,592,448 B for the three heads (the BF16 sidecar 1,975,640,448).
        try testing.expectEqual(@as(u64, 396_530_816), headPayloadBytes(c, fp8_storages));
        try testing.expectEqual(@as(u64, 1_189_592_448), m.payload_bytes);
        const q = for (suffixes) |s| {
            if (s.kind == .scale_inv and s.proj.? == .q) break s;
        } else return error.TestUnexpectedResult;
        try testing.expectEqual(Shape{ .dtype = .F32, .rank = 2, .dims = .{ 192, 32 } }, expectedShape(c, q, fp8_storages));
    }
    // The sidecar's flag must be the trunk's, both ways.
    {
        const text = try manifestText(a, c, fp8_storages, "", false);
        defer a.free(text);
        try testing.expectError(error.MimoMtpResidentsFormat, Manifest.parse(a, text, c));
        const std_text = try manifestText(a, c, fp8_storages, flag, false);
        defer a.free(std_text);
        try testing.expectError(error.MimoMtpResidentsFormat, Manifest.parse(a, std_text, .{}));
        const unknown = try manifestText(a, c, fp8_storages, "\"residents_format\":\"fp8_block64\",", false);
        defer a.free(unknown);
        try testing.expectError(error.MimoMtpManifestFormat, Manifest.parse(a, unknown, c));
    }
    // Under the flag the heads are the release's split: an FP8 eh_proj, or BF16 q, refuse.
    for ([_]Proj{ .eh, .q }) |p| {
        var st = fp8_storages;
        st[@backingInt(p)] = if (p == .eh) mimo.Storage.fp8_block128 else mimo.Storage.bf16;
        const text = try manifestText(a, c, st, flag, false);
        defer a.free(text);
        try testing.expectError(error.MimoMtpManifestQuant, Manifest.parse(a, text, c));
    }
    // FP8 storage on a standard (BF16) trunk refuses even with every tensor consistent.
    {
        const text = try manifestText(a, .{}, fp8_storages, "", false);
        defer a.free(text);
        try testing.expectError(error.MimoMtpManifestQuant, Manifest.parse(a, text, .{}));
    }
    // A wrong scale grid (head 1's q scales twice the rows).
    {
        const text = try manifestText(a, c, fp8_storages, flag, true);
        defer a.free(text);
        try testing.expectError(error.MimoMtpTensorShape, Manifest.parse(a, text, c));
    }
}
// ---- fp8res: end ----
