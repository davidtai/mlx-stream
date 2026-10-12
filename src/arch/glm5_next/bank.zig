//! GLM-5.3-Flash's routed experts as an NVFP4 expert bank: `experts.bin` beside `expert-manifest.json`, the pack's
//! ModelOpt NVFP4 bytes copied as they are into MLX's nvfp4 layout (group 16, bits 4).
//!
//! One record per (MoE layer, expert) in the `nvfp4-gud-9part` layout (the converter's, docs/glm5-next-pack.md): nine
//! parts back to back, per projection (gate, up, down) its words, its scales and its global scale:
//!   gate/up span:  gate_proj.weight  U32 [inter, hidden / 8]   E2M1 codes, element j of a word in bits 4 * (j % 8)
//!                  gate_proj.scales  U8  [inter, hidden / 16]  E4M3 block scales, one per 16 inputs
//!                  gate_proj.global_scale  F32 []
//!                  up_proj.weight, up_proj.scales, up_proj.global_scale  as gate's
//!   down span:     down_proj.weight  U32 [hidden, inter / 8], down_proj.scales U8 [hidden, inter / 16],
//!                  down_proj.global_scale F32 []
//! A global scale is the f32 MLX's nvfp4 takes (`g`, the tensor's amax: ModelOpt's weight_scale_2 x 2688, so that
//! g / 2688 == weight_scale_2 in f32). Each projection's global scale sits in the span its weights do, so a slot's
//! scale lands with the bytes its wave waits for (the stream's event gates cover a span, not a record). Records start
//! every `record_bytes` here (the manifest's `record_stride`: the parts' bytes, here `logical_bytes` and the manifest's
//! `record_bytes`, rounded up to a page); the padding is never read. The stream
//! scatters each component into its own slot array: a global scale's is f32 [rows], the array MLX's gather_qmm takes
//! as its per-expert global scale.
//!
//! A value of row r: e2m1(code) * (e4m3(scale) * (g / 2688)), in f32 in that order (MLX's fp_dequantize and its
//! Metal loaders: the group scale times g / 2688 first, then the code; `dequantRow`).

const std = @import("std");
const io_util = @import("sdk").io_util;
const sdk_ext = @import("../../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const Module = @This();

pub const format = "mlx-stream-nvfp4-expert-bank-v1";
pub const n_components = 9;
pub const gu_components = 6;
pub const routed_top_k = 8;
/// MLX's nvfp4: 16 inputs per E4M3 block scale, 4-bit E2M1 codes, 8 per U32 word.
pub const group_size = 16;
pub const bits = 4;
/// The records' alignment in `experts.bin`.
pub const page = 4096;
pub const Records = sdk_ext.expert.Records(n_components, gu_components);
pub const Dtype = enum { U32, U8, F32 };
/// The record's layout name in the manifest (`layout`).
pub const layout = "nvfp4-gud-9part";
pub const Component = enum(u8) {
    gate_weight,
    gate_scales,
    gate_global_scale,
    up_weight,
    up_scales,
    up_global_scale,
    down_weight,
    down_scales,
    down_global_scale,
    pub fn name(c: Component) []const u8 {
        return names[@backingInt(c)];
    }
    const names = [_][]const u8{
        "gate_proj.weight",     "gate_proj.scales", "gate_proj.global_scale", "up_proj.weight",         "up_proj.scales",
        "up_proj.global_scale", "down_proj.weight", "down_proj.scales",       "down_proj.global_scale",
    };
};
/// A component's place in the record; `rank` 0 is a scalar (its slot array is [rows]).
pub const Segment = struct { dtype: Dtype, rank: u8 = 2, shape: [2]u64, offset: u64, length: u64 };

pub fn mlxDtype(dtype: Dtype) mlx.mlx_dtype {
    return switch (dtype) {
        .U32 => .uint32,
        .U8 => .uint8,
        .F32 => .float32,
    };
}

/// One projection's slot arrays: the E2M1 words [slots, out, in / 8], the E4M3 scales [slots, out, in / 16] and the
/// f32 global scales [slots], as MLX's gather_qmm takes them in nvfp4 mode.
pub fn Arrays(comptime T: type) type {
    return struct { w: T, scales: T, global_scale: T };
}

pub fn BankArraysOf(comptime T: type) type {
    return sdk_ext.quant.BankArrays(Arrays(T));
}

pub const BankArrays = BankArraysOf(mlx.mlx_array);

pub fn bankArraysOf(arrays: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .w = arrays[0], .scales = arrays[1], .global_scale = arrays[2] },
        .up = .{ .w = arrays[3], .scales = arrays[4], .global_scale = arrays[5] },
        .down = .{ .w = arrays[6], .scales = arrays[7], .global_scale = arrays[8] },
    };
}

/// One SDK routed layer: the record's components and where the layer's records start.
pub const Layer = struct {
    logical_bytes: u64,
    base_offset: u64,
    segments: [n_components]Segment,
};

fn layerOf(g: *const Geometry, routed_layer: u32) Layer {
    return .{
        .logical_bytes = g.logical_bytes,
        .base_offset = @as(u64, routed_layer) * g.num_experts * g.record_bytes,
        .segments = g.segments,
    };
}

/// SDK view of an open Bank: owns only its layer table. The Bank keeps the sidecar fd and must outlive this view and
/// its Stream (destroy the Stream, then this view, then the Bank).
pub const SDKBank = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    n_experts: u32,
    first_moe_layer: u32,
    record_bytes: u64,
    sidecar: sdk_ext.expert.UncachedFd,

    pub fn init(a: std.mem.Allocator, owner: *const Bank) !SDKBank {
        if (owner.sidecar_fd < 0 or owner.sidecar_file_size != owner.geometry.total_bytes) return error.SidecarNotOpen;
        const g = &owner.geometry;
        const layers = try a.alloc(Layer, g.num_moe_layers);
        for (layers, 0..) |*layer, i| layer.* = layerOf(g, @intCast(i));
        return .{
            .allocator = a,
            .layers = layers,
            .n_experts = g.num_experts,
            .first_moe_layer = g.first_moe_layer,
            .record_bytes = g.record_bytes,
            .sidecar = .{ .fd = owner.sidecar_fd, .size = owner.sidecar_file_size },
        };
    }

    /// The reader has already stopped. The borrowed fd stays open.
    pub fn deinit(b: *SDKBank) void {
        b.allocator.free(b.layers);
        b.* = undefined;
    }

    /// SDK layer 0 is the manifest's first_moe_layer, not model layer 0.
    pub fn recordOffset(b: *const SDKBank, layer: u32, expert: u32) u64 {
        return b.layers[layer].base_offset + @as(u64, expert) * b.record_bytes;
    }

    pub fn spans(b: *const SDKBank, layer: u32, expert: u32) struct { gu_offset: u64, down_offset: u64 } {
        const off = b.recordOffset(layer, expert);
        return .{ .gu_offset = off, .down_offset = off + b.layers[layer].segments[gu_components].offset };
    }
};

/// StreamOf's bank module, the borrowed SDKBank as its Bank. The owning Bank (parse, open, deinit) is the manifest and
/// file API.
pub const SDK = struct {
    pub const n_components = Module.n_components;
    pub const gu_components = Module.gu_components;
    pub const Records = Module.Records;
    pub const Component = Module.Component;
    pub const Layer = Module.Layer;
    pub const Bank = SDKBank;
    pub const mlxDtype = Module.mlxDtype;
    pub const BankArrays = Module.BankArrays;
    pub const bankArraysOf = Module.bankArraysOf;
    pub const routed_top_k = Module.routed_top_k;
};

/// The record geometry of (hidden, inter) projections: no MLX, no file.
pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    /// The components' bytes (what a record holds and the reader reads).
    logical_bytes: u64,
    /// The stride of records in `experts.bin`: `logical_bytes` rounded up to a page.
    record_bytes: u64,
    /// The gate/up span: the first `gu_components` components.
    gate_up_bytes: u64,
    total_bytes: u64,
    segments: [n_components]Segment,

    pub fn init(hidden: u64, inter: u64, experts: u64, first: u64, layers: u64) !Geometry {
        if (hidden == 0 or inter == 0 or hidden > 1 << 16 or inter > 1 << 16 or hidden % group_size != 0 or inter % group_size != 0 or
            experts == 0 or experts >= std.math.maxInt(u16) or layers == 0 or layers > 4096 or first > 4096 or first + layers > 4096 or experts * layers > 1 << 20)
            return error.InvalidGeometry;
        var g: Geometry = .{
            .hidden_size = @intCast(hidden),
            .intermediate_size = @intCast(inter),
            .num_experts = @intCast(experts),
            .first_moe_layer = @intCast(first),
            .num_moe_layers = @intCast(layers),
            .logical_bytes = 0,
            .record_bytes = 0,
            .gate_up_bytes = 0,
            .total_bytes = 0,
            .segments = undefined,
        };
        var off: u64 = 0;
        const place = struct {
            fn at(seg: *Segment, o: *u64, dtype: Dtype, rank: u8, shape: [2]u64, length: u64) void {
                seg.* = .{ .dtype = dtype, .rank = rank, .shape = shape, .offset = o.*, .length = length };
                o.* += length;
            }
        }.at;
        // gate, up: [inter, hidden]; down: [hidden, inter].
        for ([_]Component{ .gate_weight, .up_weight }) |w| {
            const c = @backingInt(w);
            place(&g.segments[c], &off, .U32, 2, .{ inter, hidden / 8 }, inter * hidden / 2);
            place(&g.segments[c + 1], &off, .U8, 2, .{ inter, hidden / group_size }, inter * hidden / group_size);
            place(&g.segments[c + 2], &off, .F32, 0, .{ 0, 0 }, 4);
        }
        g.gate_up_bytes = off;
        place(&g.segments[@backingInt(Component.down_weight)], &off, .U32, 2, .{ hidden, inter / 8 }, hidden * inter / 2);
        place(&g.segments[@backingInt(Component.down_scales)], &off, .U8, 2, .{ hidden, inter / group_size }, hidden * inter / group_size);
        place(&g.segments[@backingInt(Component.down_global_scale)], &off, .F32, 0, .{ 0, 0 }, 4);
        g.logical_bytes = off;
        g.record_bytes = std.mem.alignForward(u64, off, page);
        g.total_bytes = std.math.mul(u64, g.record_bytes, experts * layers) catch return error.InvalidGeometry;
        if (g.total_bytes > std.math.maxInt(i64)) return error.InvalidGeometry;
        return g;
    }

    /// Model layer `layer` (its range was fixed at installation).
    pub fn recordOffset(g: *const Geometry, layer: u32, expert: u32) u64 {
        return (@as(u64, layer - g.first_moe_layer) * g.num_experts + expert) * g.record_bytes;
    }

    /// GLM-5.3-Flash (glm5_next): hidden 4096, moe_intermediate 2048, 288 routed experts, MoE layers 3..44.
    pub fn requireFlash(g: *const Geometry) !void {
        if (g.hidden_size != 4096 or g.intermediate_size != 2048 or g.num_experts != 288 or g.first_moe_layer != 3 or g.num_moe_layers != 42)
            return error.UnsupportedTopology;
    }
};

/// The manifest the converter writes (scripts/convert_glm5next_nvfp4.py, docs/glm5-next-pack.md): `record_bytes` the
/// parts' bytes (`Geometry.logical_bytes`), `record_stride` the records' stride (`Geometry.record_bytes`), `parts` the
/// nine parts in record order.
const Manifest = struct {
    format: []const u8,
    data_file: []const u8,
    layout: []const u8,
    alignment: u64,
    hidden_size: u64,
    intermediate_size: u64,
    num_experts: u64,
    first_moe_layer: u64,
    num_moe_layers: u64,
    record_bytes: u64,
    record_stride: u64,
    gate_up_bytes: u64,
    down_bytes: u64,
    quantization: struct { mode: []const u8, bits: u32, group_size: u32 },
    source: struct { repository: []const u8, revision: []const u8 },
    parts: []const struct { name: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 },
    records: []const struct { layer: u64, expert: u64, offset: u64, length: u64, sha256: []const u8 },
    output_files: std.json.Value,
};

pub const Bank = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    digests: [][32]u8,
    source_repository: []const u8,
    source_revision: []const u8,
    sidecar_fd: std.c.fd_t = -1,
    sidecar_file_size: u64 = 0,

    /// The manifest against the geometry it declares, every record's place and digest declared. The payload is the
    /// converter's to verify: opening a model never reads or hashes the bank.
    pub fn parse(a: std.mem.Allocator, text: []const u8) !Bank {
        const parsed = try std.json.parseFromSlice(Manifest, a, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const m = parsed.value;
        if (!std.mem.eql(u8, m.format, format) or !std.mem.eql(u8, m.data_file, "experts.bin") or !std.mem.eql(u8, m.layout, layout) or m.alignment != page) return error.InvalidFormat;
        if (!std.mem.eql(u8, m.quantization.mode, "nvfp4") or m.quantization.bits != bits or m.quantization.group_size != group_size) return error.InvalidQuantization;
        const g = try Geometry.init(m.hidden_size, m.intermediate_size, m.num_experts, m.first_moe_layer, m.num_moe_layers);
        if (m.record_stride != g.record_bytes or m.record_bytes != g.logical_bytes or m.gate_up_bytes != g.gate_up_bytes or m.down_bytes != g.logical_bytes - g.gate_up_bytes) return error.InvalidGeometry;
        if (m.parts.len != n_components) return error.InvalidSegment;
        for (m.parts, g.segments, 0..) |got, want, i| {
            const component: Component = @fromBackingInt(@intCast(i));
            if (!std.mem.eql(u8, got.name, component.name()) or !std.mem.eql(u8, got.dtype, @tagName(want.dtype)) or
                !std.mem.eql(u64, got.shape, want.shape[0..want.rank]) or got.offset != want.offset or got.length != want.length)
                return error.InvalidSegment;
        }
        if (m.source.repository.len == 0 or m.source.revision.len != 40) return error.InvalidSource;
        for (m.source.revision) |c| if (!std.ascii.isHex(c)) return error.InvalidSource;
        const count = @as(usize, g.num_experts) * g.num_moe_layers;
        if (m.records.len != count) return error.InvalidRecord;
        const digests = try a.alloc([32]u8, count);
        errdefer a.free(digests);
        for (m.records, digests, 0..) |r, *digest, i| {
            const layer = g.first_moe_layer + i / g.num_experts;
            const expert = i % g.num_experts;
            if (r.layer != layer or r.expert != expert or r.offset != i * g.record_bytes or r.length != g.logical_bytes)
                return error.InvalidRecord;
            digest.* = try parseDigest(r.sha256);
        }
        if (m.output_files != .object) return error.InvalidOutputFile;
        const out = m.output_files.object.get("experts.bin") orelse return error.InvalidOutputFile;
        if (out != .object) return error.InvalidOutputFile;
        const size = out.object.get("size") orelse return error.InvalidOutputFile;
        const digest = out.object.get("sha256") orelse return error.InvalidOutputFile;
        if (size != .integer or size.integer < 0 or size.integer != g.total_bytes or digest != .string) return error.InvalidOutputFile;
        _ = try parseDigest(digest.string);
        const repository = try a.dupe(u8, m.source.repository);
        errdefer a.free(repository);
        const revision = try a.dupe(u8, m.source.revision);
        return .{
            .allocator = a,
            .geometry = g,
            .digests = digests,
            .source_repository = repository,
            .source_revision = revision,
        };
    }

    /// Construction only: the F_NOCACHE reader's fd; never follows an experts.bin symlink, and refuses a truncated or
    /// extended file.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
        var b = try readManifest(a, io, dir);
        errdefer b.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
        defer a.free(path);
        b.sidecar_fd = try io_util.openNoCache(path, .{ .follow_symlinks = false });
        var st: std.c.Stat = undefined;
        if (std.c.fstat(b.sidecar_fd, &st) != 0) return error.SidecarOpen;
        if (st.mode & std.c.S.IFMT != std.c.S.IFREG or st.size < 0 or @as(u64, @intCast(st.size)) != b.geometry.total_bytes)
            return error.SidecarGeometry;
        b.sidecar_file_size = @intCast(st.size);
        return b;
    }

    pub fn deinit(b: *Bank) void {
        if (b.sidecar_fd >= 0) _ = std.c.close(b.sidecar_fd);
        b.allocator.free(b.digests);
        b.allocator.free(b.source_repository);
        b.allocator.free(b.source_revision);
        b.* = undefined;
    }
};

fn parseDigest(hex: []const u8) ![32]u8 {
    if (hex.len != 64) return error.InvalidDigest;
    var digest: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&digest, hex) catch return error.InvalidDigest;
    return digest;
}

fn readManifest(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
    if (!std.fs.path.isAbsolute(dir)) return error.BankDirNotAbsolute;
    const path = try std.fs.path.join(a, &.{ dir, "expert-manifest.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    defer a.free(text);
    return Bank.parse(a, text);
}

/// The manifest's geometry, validated, for admission and configuration: no sidecar, no MLX, no reader.
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    var bank = try readManifest(a, io, dir);
    defer bank.deinit();
    return bank.geometry;
}

// ── The host-side decoder: a record's values as MLX's nvfp4 dequantizes them ──

/// E2M1 codes (sign in bit 3), MLX's nvfp4 table.
pub const e2m1 = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

/// An E4M3 (fn: no infinities, S.1111.111 NaN) byte as f32, exactly.
pub fn e4m3(b: u8) f32 {
    const sign: f32 = if (b & 0x80 != 0) -1.0 else 1.0;
    const e: i32 = (b >> 3) & 0xf;
    const m: f32 = @floatFromInt(b & 0x7);
    if (e == 0xf and b & 0x7 == 0x7) return std.math.nan(f32);
    if (e == 0) return sign * std.math.ldexp(m / 8.0, -6);
    return sign * std.math.ldexp(1.0 + m / 8.0, e - 7);
}

/// The f32 MLX folds into every group scale: g / (448 * 6).
pub fn invScaleEnc(g: f32) f32 {
    return g / (448.0 * 6.0);
}

/// One row of a projection: `words` its in / 8 E2M1 words, `scales` its in / 16 E4M3 bytes, `g` its global scale;
/// out[j] = e2m1(code j) * (e4m3(scale j / 16) * (g / 2688)), each product rounded to f32.
pub fn dequantRow(words: []const u32, scales: []const u8, g: f32, out: []f32) void {
    std.debug.assert(out.len == words.len * 8 and scales.len * group_size == out.len);
    const inv = invScaleEnc(g);
    for (scales, 0..) |s, q| {
        const scale = e4m3(s) * inv;
        for (0..group_size) |k| {
            const j = q * group_size + k;
            const code: u4 = @truncate(words[j / 8] >> @intCast(4 * (j % 8)));
            out[j] = scale * e2m1[code];
        }
    }
}

/// `x` as IEEE half, round to nearest even (MLX's astype to float16).
pub fn toHalf(x: f32) f16 {
    return @floatCast(x);
}

// ── Tests ──

const testing = std.testing;

test "glm5_next nvfp4 bank: Flash's record keeps every projection's global scale in its own span, records on pages" {
    const g = try Geometry.init(4096, 2048, 288, 3, 42);
    try g.requireFlash();
    // gate/up: 4,194,304 B of words + 524,288 B of scales each, then their two global scales.
    try testing.expectEqual(@as(u64, 9_437_192), g.gate_up_bytes);
    try testing.expectEqual(@as(u64, 14_155_788), g.logical_bytes);
    try testing.expectEqual(@as(u64, 14_159_872), g.record_bytes);
    try testing.expectEqual(@as(u64, 0), g.record_bytes % page);
    try testing.expectEqual(@as(u64, 42 * 288 * 14_159_872), g.total_bytes);
    try testing.expectEqualSlices(u64, &.{ 2048, 512 }, &g.segments[0].shape);
    try testing.expectEqualSlices(u64, &.{ 2048, 256 }, &g.segments[1].shape);
    try testing.expectEqualSlices(u64, &.{ 4096, 256 }, &g.segments[6].shape);
    try testing.expectEqualSlices(u64, &.{ 4096, 128 }, &g.segments[7].shape);
    for ([_]Component{ .gate_global_scale, .up_global_scale, .down_global_scale }) |c| {
        const s = g.segments[@backingInt(c)];
        try testing.expectEqual(Dtype.F32, s.dtype);
        try testing.expectEqual(@as(u8, 0), s.rank);
        try testing.expectEqual(@as(u64, 4), s.length);
    }
    // The gate and up global scales lie in the gate/up span (each after its projection's scales); down's in the down span.
    try testing.expect(g.segments[2].offset + 4 <= g.gate_up_bytes and g.segments[5].offset + 4 == g.gate_up_bytes);
    try testing.expectEqual(@as(u64, 4_718_592), g.segments[2].offset);
    try testing.expectEqual(@as(u64, 4_718_596), g.segments[3].offset);
    try testing.expect(g.segments[8].offset >= g.gate_up_bytes and g.segments[8].offset + 4 == g.logical_bytes);
    // The components are back to back (the reader's scatter).
    var off: u64 = 0;
    for (g.segments) |s| {
        try testing.expectEqual(off, s.offset);
        off += s.length;
    }
    try testing.expectEqual(@as(u64, 288 * 14_159_872 + 5 * 14_159_872), g.recordOffset(4, 5));
}

test "glm5_next nvfp4 bank: invalid geometry is refused before arithmetic" {
    try testing.expectError(error.InvalidGeometry, Geometry.init(40, 32, 2, 1, 1));
    try testing.expectError(error.InvalidGeometry, Geometry.init(32, 32, 0, 1, 1));
    try testing.expectError(error.InvalidGeometry, Geometry.init(32, 32, 2, 1, 0));
    try testing.expectError(error.InvalidGeometry, Geometry.init(1 << 31, 1 << 31, 288, 3, 42));
    const tiny = try Geometry.init(64, 32, 2, 3, 1);
    try testing.expectError(error.UnsupportedTopology, tiny.requireFlash());
}

test "glm5_next nvfp4 SDK bank: nine ranked components, records at the page stride, spans past the gate/up span" {
    sdk_ext.expert.assertBank(SDK);
    try testing.expect(Records == sdk_ext.expert.Records(9, 6));
    const g = try Geometry.init(4096, 2048, 288, 3, 42);
    var layers = [_]Layer{ layerOf(&g, 0), layerOf(&g, 41) };
    const b: SDKBank = .{
        .allocator = testing.allocator,
        .layers = &layers,
        .n_experts = g.num_experts,
        .first_moe_layer = g.first_moe_layer,
        .record_bytes = g.record_bytes,
        .sidecar = .{ .fd = -1, .size = g.total_bytes },
    };
    try testing.expectEqual(g.logical_bytes, layers[0].logical_bytes);
    try testing.expectEqual(g.recordOffset(3, 7), b.recordOffset(0, 7));
    try testing.expectEqual(g.recordOffset(44, 287), b.recordOffset(1, 287));
    try testing.expectEqual(g.recordOffset(44, 287) + g.gate_up_bytes, b.spans(1, 287).down_offset);
    try testing.expectEqual(mlx.mlx_dtype.float32, mlxDtype(layers[0].segments[2].dtype));
    var owner = try Bank.parse(testing.allocator, fixture);
    defer owner.deinit();
    try testing.expectError(error.SidecarNotOpen, SDKBank.init(testing.allocator, &owner));
}

test "glm5_next nvfp4 bank: the manifest's topology, components and record places are construction invariants" {
    const a = testing.allocator;
    var bank = try Bank.parse(a, fixture);
    defer bank.deinit();
    try testing.expectEqual(@as(u64, 2 * 4096), bank.geometry.total_bytes);
    try testing.expectEqual(@as(u64, 3468), bank.geometry.logical_bytes);
    try testing.expectEqual(@as(u64, 2312), bank.geometry.gate_up_bytes);
    try testing.expectEqual(@as(usize, 2), bank.digests.len);
    const Case = struct { from: []const u8, to: []const u8, err: anyerror };
    for ([_]Case{
        .{ .from = "\"nvfp4\"", .to = "\"mxfp4\"", .err = error.InvalidQuantization },
        .{ .from = "\"group_size\":16", .to = "\"group_size\":32", .err = error.InvalidQuantization },
        .{ .from = "\"record_stride\":4096", .to = "\"record_stride\":3468", .err = error.InvalidGeometry },
        .{ .from = "\"layout\":\"nvfp4-gud-9part\"", .to = "\"layout\":\"nvfp4-9part\"", .err = error.InvalidFormat },
        .{ .from = "\"name\":\"up_proj.global_scale\"", .to = "\"name\":\"down_proj.global_scale\"", .err = error.InvalidSegment },
        .{ .from = "\"dtype\":\"F32\",\"shape\":[],\"offset\":1152", .to = "\"dtype\":\"F16\",\"shape\":[],\"offset\":1152", .err = error.InvalidSegment },
        .{ .from = "\"offset\":4096,\"length\":3468", .to = "\"offset\":3468,\"length\":3468", .err = error.InvalidRecord },
        .{ .from = "\"format\":\"mlx-stream-nvfp4-expert-bank-v1\"", .to = "\"format\":\"mlx-stream-mxfp4-expert-bank-v1\"", .err = error.InvalidFormat },
    }) |c| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, fixture, c.from));
        const text = try std.mem.replaceOwned(u8, a, fixture, c.from, c.to);
        defer a.free(text);
        try testing.expectError(c.err, Bank.parse(a, text));
    }
}

test "glm5_next nvfp4 decoder: E4M3 and E2M1 as MLX reads them, the group scale before the code" {
    try testing.expectEqual(@as(f32, 448.0), e4m3(0x7e));
    try testing.expectEqual(@as(f32, -448.0), e4m3(0xfe));
    try testing.expect(std.math.isNan(e4m3(0x7f)));
    try testing.expectEqual(@as(f32, 1.0), e4m3(0x38));
    try testing.expectEqual(std.math.ldexp(@as(f32, 1.0), -9), e4m3(0x01));
    try testing.expectEqual(std.math.ldexp(@as(f32, 0.875), -6), e4m3(0x07));
    try testing.expectEqual(std.math.ldexp(@as(f32, 1.0), -6), e4m3(0x08));
    // The slice the research checked: g = w2 * 2688 = 0.15625, and g / 2688 is w2 in f32.
    try testing.expectEqual(@as(f32, 5.8128720e-05), invScaleEnc(0.15625));
    // A row of 16: codes 0..15, one scale byte.
    var words = [_]u32{ 0x76543210, 0xfedcba98 };
    const sc = [_]u8{0x40}; // 2.0
    var out: [16]f32 = undefined;
    dequantRow(&words, &sc, 2688.0, &out);
    for (out, 0..) |v, j| try testing.expectEqual(e2m1[j] * 2.0, v);
    try testing.expectEqual(@as(f16, 1.0), toHalf(1.0));
    try testing.expectEqual(@as(f16, 65504.0), toHalf(65504.0));
}

/// A tiny bank's manifest: hidden 64, inter 32, two experts of MoE layer 3; logical 3,468 B, records every page.
const fixture =
    \\{"format":"mlx-stream-nvfp4-expert-bank-v1","data_file":"experts.bin","layout":"nvfp4-gud-9part","alignment":4096,
    \\"hidden_size":64,"intermediate_size":32,"num_experts":2,"first_moe_layer":3,"num_moe_layers":1,
    \\"record_bytes":3468,"record_stride":4096,"gate_up_bytes":2312,"down_bytes":1156,"quantization":{"mode":"nvfp4","bits":4,"group_size":16},
    \\"source":{"repository":"fixture","revision":"0123456789012345678901234567890123456789"},
    \\"parts":[
    \\{"name":"gate_proj.weight","dtype":"U32","shape":[32,8],"offset":0,"length":1024},
    \\{"name":"gate_proj.scales","dtype":"U8","shape":[32,4],"offset":1024,"length":128},
    \\{"name":"gate_proj.global_scale","dtype":"F32","shape":[],"offset":1152,"length":4},
    \\{"name":"up_proj.weight","dtype":"U32","shape":[32,8],"offset":1156,"length":1024},
    \\{"name":"up_proj.scales","dtype":"U8","shape":[32,4],"offset":2180,"length":128},
    \\{"name":"up_proj.global_scale","dtype":"F32","shape":[],"offset":2308,"length":4},
    \\{"name":"down_proj.weight","dtype":"U32","shape":[64,4],"offset":2312,"length":1024},
    \\{"name":"down_proj.scales","dtype":"U8","shape":[64,2],"offset":3336,"length":128},
    \\{"name":"down_proj.global_scale","dtype":"F32","shape":[],"offset":3464,"length":4}],
    \\"records":[
    \\{"layer":3,"expert":0,"offset":0,"length":3468,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    \\{"layer":3,"expert":1,"offset":4096,"length":3468,"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],
    \\"output_files":{"experts.bin":{"size":8192,"sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}}}
;
