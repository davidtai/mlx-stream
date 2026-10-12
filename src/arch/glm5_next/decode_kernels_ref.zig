//! The host reference of the NVFP4 decode texts' weight arithmetic (kernels/glm5_next_nvfp4/header_glm5n_nvfp4.metal), line
//! for line: `e4m3` is glm5n_e4m3, `decode8` glm5n_decode8, `nvScale` glm5n_nv_scale (with `invOf`, the texts' g / 2688),
//! and `decodeProjection` walks a projection as a lane does (per row, per block of 16 codes: the uint2 of words 2q and
//! 2q + 1 and scale byte q, decoded, each code's half times the lane's scale). The tests pin it to MLX's dequant order
//! (inv = g / 2688, s = e4m3 * inv, w = s * e2m1, f32): every scale byte and code at the pack's global scales, and the
//! three test vectors' fp16 decodes bit for bit (GLM5NEXT_NVFP4_VECTORS=<dir of the 3 npz and vectors.json>, the pack's
//! `scripts/glm5next_nvfp4_reader.py vectors`). Host only: no MLX array, no device.

const std = @import("std");
const io_util = @import("sdk").io_util;

/// `(c & 7) << 9 | (c & 8) << 12`: e2m1(c) x 2^-14 as half bits (mimo_e2m1's).
pub fn halfBits(c: u4) u16 {
    return (@as(u16, c & 7) << 9) | (@as(u16, c & 8) << 12);
}

pub fn halfF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// glm5n_decode8: half j of the result holds codes j (lo) and j + 4 (hi) of word `w`.
pub fn decode8(w: u32) [4]u32 {
    return .{
        ((w << 9) & 0x0E000E00) | ((w << 12) & 0x80008000),
        ((w << 5) & 0x0E000E00) | ((w << 8) & 0x80008000),
        ((w << 1) & 0x0E000E00) | ((w << 4) & 0x80008000),
        ((w >> 3) & 0x0E000E00) | (w & 0x80008000),
    };
}

/// glm5n_e4m3: MLX's fp8_e4m3 -> float16 bits, then times 256.
pub fn e4m3(b: u8) f32 {
    const v: u32 = b & 127;
    const u: u16 = @intCast((v << 7) | (((v + 1) >> 7) << 14) | ((@as(u32, b) & 128) << 8));
    return halfF32(u) * 256.0;
}

/// The texts' and fp_dequantize's `*global_scale / (F8E4M3_MAX * F4E2M1_MAX)`.
pub fn invOf(g: f32) f32 {
    return g / 2688.0;
}

/// glm5n_nv_scale: the group scale times the folded global scale (one rounding), times 2^14.
pub fn nvScale(b: u8, inv: f32) f32 {
    return (e4m3(b) * inv) * 16384.0;
}

/// The texts' weights of one projection, row-major [rows, cols] f32: per row, per lane block of 16 codes (words 2q and
/// 2q + 1, scale byte q), glm5n_decode16, then each code's half times the lane's scale.
pub fn decodeProjection(words: []const u32, scales: []const u8, g: f32, rows: usize, cols: usize, out: []f32) void {
    std.debug.assert(words.len == rows * cols / 8 and scales.len == rows * cols / 16 and out.len == rows * cols);
    const inv = invOf(g);
    for (0..rows) |r| for (0..cols / 16) |q| {
        const s = nvScale(scales[r * (cols / 16) + q], inv);
        const hx = decode8(words[r * (cols / 8) + 2 * q]);
        const hy = decode8(words[r * (cols / 8) + 2 * q + 1]);
        const o = out[r * cols + 16 * q ..][0..16];
        for (0..4) |j| {
            o[j] = s * halfF32(@truncate(hx[j]));
            o[j + 4] = s * halfF32(@truncate(hx[j] >> 16));
            o[8 + j] = s * halfF32(@truncate(hy[j]));
            o[12 + j] = s * halfF32(@truncate(hy[j] >> 16));
        }
    };
}

// ── Independent definitions (the tests' side) ──

/// OCP E4M3FN: bias 7, subnormals at exponent 0, NaN at S.1111.111, no infinity.
pub fn e4m3Value(b: u8) f32 {
    const e: i32 = (b >> 3) & 0xF;
    const m: u8 = b & 7;
    const mag: f32 = if (e == 15 and m == 7) std.math.nan(f32) else if (e == 0) std.math.ldexp(@as(f32, @floatFromInt(m)), -9) else std.math.ldexp(@as(f32, @floatFromInt(8 + @as(u32, m))), e - 10);
    return if (b & 0x80 != 0) -mag else mag;
}

/// E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (code 8 is -0).
pub fn e2m1Value(c: u4) f32 {
    const mag = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = mag[c & 7];
    return if (c & 8 != 0) -v else v;
}

/// MLX's fp_dequantize of one weight: s = e4m3 * (g / 2688), then s * e2m1, f32.
pub fn mlxWeight(b: u8, c: u4, g: f32) f32 {
    return (e4m3Value(b) * invOf(g)) * e2m1Value(c);
}

// ── The test vectors (.npz: a zip of stored .npy members, zip64 sizes) ──

pub const NpyArray = struct { name: []const u8, descr: []const u8, shape: [4]u64, rank: usize, data: []const u8 };

/// The members of an uncompressed .npz image (`numpy.savez`): each a C-order .npy array; refused by name otherwise.
pub fn npzMembers(a: std.mem.Allocator, img: []const u8) ![]NpyArray {
    var out: std.ArrayList(NpyArray) = .empty;
    errdefer out.deinit(a);
    var at: usize = 0;
    while (at + 30 <= img.len) {
        const sig = std.mem.readInt(u32, img[at..][0..4], .little);
        if (sig == 0x02014b50 or sig == 0x06054b50 or sig == 0x06064b50) break;
        if (sig != 0x04034b50) return error.NpzNotZip;
        const method = std.mem.readInt(u16, img[at + 8 ..][0..2], .little);
        var size: u64 = std.mem.readInt(u32, img[at + 22 ..][0..4], .little);
        const name_len = std.mem.readInt(u16, img[at + 26 ..][0..2], .little);
        const extra_len = std.mem.readInt(u16, img[at + 28 ..][0..2], .little);
        if (method != 0) return error.NpzCompressed;
        const name = img[at + 30 ..][0..name_len];
        const extra = img[at + 30 + name_len ..][0..extra_len];
        if (size == 0xFFFFFFFF) {
            // zip64: header 0x0001, then the uncompressed size first
            var e: usize = 0;
            size = while (e + 4 <= extra.len) {
                const id = std.mem.readInt(u16, extra[e..][0..2], .little);
                const len = std.mem.readInt(u16, extra[e + 2 ..][0..2], .little);
                if (id == 0x0001 and len >= 8) break std.mem.readInt(u64, extra[e + 4 ..][0..8], .little);
                e += 4 + @as(usize, len);
            } else return error.NpzZip64;
        }
        const data_at = at + 30 + name_len + extra_len;
        if (data_at + size > img.len) return error.NpzTruncated;
        try out.append(a, try npyArray(name, img[data_at..][0..size]));
        at = data_at + size;
    }
    return out.toOwnedSlice(a);
}

fn npyArray(member_name: []const u8, npy: []const u8) !NpyArray {
    if (npy.len < 10 or !std.mem.eql(u8, npy[0..6], "\x93NUMPY")) return error.NpyMagic;
    const hlen: usize = if (npy[6] == 1) std.mem.readInt(u16, npy[8..10], .little) else std.mem.readInt(u32, npy[8..12], .little);
    const hstart: usize = if (npy[6] == 1) 10 else 12;
    const header = npy[hstart..][0..hlen];
    if (std.mem.indexOf(u8, header, "'fortran_order': False") == null) return error.NpyOrder;
    const d0 = (std.mem.indexOf(u8, header, "'descr': '") orelse return error.NpyHeader) + 10;
    const d1 = std.mem.indexOfScalarPos(u8, header, d0, '\'') orelse return error.NpyHeader;
    const s0 = (std.mem.indexOf(u8, header, "'shape': (") orelse return error.NpyHeader) + 10;
    const s1 = std.mem.indexOfScalarPos(u8, header, s0, ')') orelse return error.NpyHeader;
    var arr: NpyArray = .{ .name = if (std.mem.endsWith(u8, member_name, ".npy")) member_name[0 .. member_name.len - 4] else member_name, .descr = header[d0..d1], .shape = @splat(1), .rank = 0, .data = npy[hstart + hlen ..] };
    var it = std.mem.tokenizeAny(u8, header[s0..s1], ", ");
    while (it.next()) |dim| {
        if (arr.rank == arr.shape.len) return error.NpyHeader;
        arr.shape[arr.rank] = try std.fmt.parseInt(u64, dim, 10);
        arr.rank += 1;
    }
    return arr;
}

pub fn member(arrays: []const NpyArray, name: []const u8) ?NpyArray {
    for (arrays) |x| if (std.mem.eql(u8, x.name, name)) return x;
    return null;
}

const JPart = struct { name: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 };
const JProj = struct { shape: []const u64, g_f32_hex: []const u8, f16_sha256: []const u8 };
const JVector = struct { layer: u32, expert: u32, record_sha256: []const u8, projections: struct { gate_proj: JProj, up_proj: JProj, down_proj: JProj } };
const JVectors = struct { layout: []const u8, arithmetic: []const u8, record_bytes: u64, parts: []const JPart, vectors: []const JVector };

fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

fn part(parts: []const JPart, name: []const u8) !JPart {
    for (parts) |p| if (std.mem.eql(u8, p.name, name)) return p;
    return error.VectorsPartMissing;
}

const testing = std.testing;

test "glm5next nvfp4 ref: the texts' E4M3 decode is the format's at every byte; their two-codes-per-op decode is mimo_e2m1's half at every code and position" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const got = e4m3(b);
        const want = e4m3Value(b);
        if (std.math.isNan(want)) {
            try testing.expect(std.math.isNan(got));
        } else {
            try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
        }
    }
    try testing.expectEqual(@as(f32, 448), e4m3(0x7E));
    try testing.expectEqual(@as(f32, 0x1p-9), e4m3(0x01));
    try testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(e4m3(0x80))));
    for (0..16) |c| try testing.expectEqual(@as(u32, @bitCast(e2m1Value(@intCast(c)))), @as(u32, @bitCast(halfF32(halfBits(@intCast(c))) * 16384.0)));
    var state: u32 = 0x67356e6e;
    for (0..8) |pos| for (0..16) |c| {
        state = state *% 1664525 +% 1013904223;
        const w = (state & ~(@as(u32, 0xF) << @intCast(4 * pos))) | (@as(u32, @intCast(c)) << @intCast(4 * pos));
        const h = decode8(w);
        const half: u16 = if (pos < 4) @truncate(h[pos]) else @truncate(h[pos - 4] >> 16);
        try testing.expectEqual(halfBits(@intCast(c)), half);
    };
}

test "glm5next nvfp4 ref: the texts' weight (the lane's scale x 2^14 times the code's half) is MLX's s * e2m1 bit for bit at every scale byte, every code and the pack's global scales" {
    // the test vectors' g (gate / up / down of L3 E0, L23 E144, L44 E287) and the pack's range ends (weight_scale_2
    // 3.49e-05 .. 3.72e-03: g 0.094 .. 10.0), then a sweep across it
    var gs: [24]f32 = undefined;
    const fixed = [_]u32{ 0x3e200000, 0x3df00000, 0x3de00000, 0x3e000000, 0x3e900000, 0x3dc00000, 0x41200000, 0x3f800000 };
    for (fixed, gs[0..fixed.len]) |h, *g| g.* = @bitCast(h);
    for (gs[fixed.len..], 0..) |*g, i| g.* = 0.09375 * std.math.pow(f32, 1.36, @floatFromInt(i));
    var n: usize = 0;
    for (gs) |g| for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7F == 0x7F) continue;
        const s = nvScale(b, invOf(g));
        for (0..16) |c| {
            const got = s * halfF32(halfBits(@intCast(c)));
            const want = mlxWeight(b, @intCast(c), g);
            testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got))) catch |e| {
                std.debug.print("g {e} byte 0x{x:0>2} code {d}: kernel {e} mlx {e}\n", .{ g, b, c, got, want });
                return e;
            };
            n += 1;
        }
    };
    try testing.expectEqual(@as(usize, 24 * 254 * 16), n);
}

test "glm5next nvfp4 vectors: the texts' arithmetic decodes the three test vectors (L3 E0, L23 E144, L44 E287) to their fp16 decodes bit for bit" {
    const dir = std.mem.span(std.c.getenv("GLM5NEXT_NVFP4_VECTORS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const jpath = try std.fmt.allocPrint(a, "{s}/vectors.json", .{dir});
    defer a.free(jpath);
    const jtext = try io_util.readAllNoCache(a, jpath, 1 << 20);
    defer a.free(jtext);
    const parsed = try std.json.parseFromSlice(JVectors, a, jtext, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const v = parsed.value;
    try testing.expectEqualStrings("nvfp4-gud-9part", v.layout);
    try testing.expectEqualStrings("inv = f32(g / 2688); s = f32(e4m3 * inv); w = f32(s * e2m1); float16 = RNE(w)", v.arithmetic);
    try testing.expectEqual(@as(usize, 3), v.vectors.len);
    const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var total: usize = 0;
    for (v.vectors) |vec| {
        const npath = try std.fmt.allocPrint(a, "{s}/glm5next-nvfp4-L{d:0>2}-E{d:0>3}.npz", .{ dir, vec.layer, vec.expert });
        defer a.free(npath);
        const img = try io_util.readAllNoCache(a, npath, 128 << 20);
        defer a.free(img);
        const arrays = try npzMembers(a, img);
        defer a.free(arrays);
        const record = member(arrays, "record") orelse return error.VectorsRecordMissing;
        try testing.expectEqualStrings("|u1", record.descr);
        try testing.expectEqual(v.record_bytes, record.data.len);
        try testing.expectEqualStrings(vec.record_sha256, &shaHex(record.data));
        const gnpz = member(arrays, "g") orelse return error.VectorsGMissing;
        try testing.expectEqualStrings("<f4", gnpz.descr);
        for (projs, 0..) |name, pi| {
            const jp = switch (pi) {
                0 => vec.projections.gate_proj,
                1 => vec.projections.up_proj,
                else => vec.projections.down_proj,
            };
            const rows: usize = @intCast(jp.shape[0]);
            const cols: usize = @intCast(jp.shape[1]);
            var nb: [64]u8 = undefined;
            const pw = try part(v.parts, try std.fmt.bufPrint(&nb, "{s}.weight", .{name}));
            const ps = try part(v.parts, try std.fmt.bufPrint(&nb, "{s}.scales", .{name}));
            const pg = try part(v.parts, try std.fmt.bufPrint(&nb, "{s}.global_scale", .{name}));
            try testing.expect(pw.length == rows * cols / 2 and ps.length == rows * cols / 16 and pg.length == 4);
            // the record's global scale is the npz's and the manifest's g, bit for bit
            const g_bits = std.mem.readInt(u32, record.data[@intCast(pg.offset)..][0..4], .little);
            try testing.expectEqual(g_bits, std.mem.readInt(u32, gnpz.data[4 * pi ..][0..4], .little));
            try testing.expectEqual(g_bits, try std.fmt.parseInt(u32, jp.g_f32_hex, 16));
            const words = try a.alloc(u32, rows * cols / 8);
            defer a.free(words);
            @memcpy(std.mem.sliceAsBytes(words), record.data[@intCast(pw.offset)..][0..@intCast(pw.length)]);
            const scales = record.data[@intCast(ps.offset)..][0..@intCast(ps.length)];
            const want = member(arrays, name) orelse return error.VectorsProjectionMissing;
            try testing.expectEqualStrings("<f2", want.descr);
            try testing.expect(want.rank == 2 and want.shape[0] == rows and want.shape[1] == cols and want.data.len == 2 * rows * cols);
            try testing.expectEqualStrings(jp.f16_sha256, &shaHex(want.data));
            const w = try a.alloc(f32, rows * cols);
            defer a.free(w);
            decodeProjection(words, scales, @bitCast(g_bits), rows, cols, w);
            var bad: usize = 0;
            var first: ?usize = null;
            for (w, 0..) |x, i| {
                const got: u16 = @bitCast(@as(f16, @floatCast(x)));
                const ref = std.mem.readInt(u16, want.data[2 * i ..][0..2], .little);
                if (got != ref) {
                    bad += 1;
                    if (first == null) first = i;
                }
            }
            std.debug.print("GLM5N_NVFP4_VECTORS L{d} E{d} {s} [{d}, {d}] g 0x{x:0>8} f16_bitwise={d}/{d}\n", .{ vec.layer, vec.expert, name, rows, cols, g_bits, rows * cols - bad, rows * cols });
            if (first) |i| std.debug.print("  first mismatch at {d}: kernel f32 {e}\n", .{ i, w[i] });
            try testing.expectEqual(@as(usize, 0), bad);
            total += rows * cols;
        }
    }
    std.debug.print("GLM5N_NVFP4_VECTORS PASS values={d}\n", .{total});
}
