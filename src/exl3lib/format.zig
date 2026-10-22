//! EXL3 trellis format, the host reference: rates (any even n in [16, 128], K = n/16 bits per weight), codeword
//! windows (8..16), the MUL1 and MCG codebooks, the tensor-core tile permutation, the H128 Hadamard and the
//! suh/svh scales, and a scalar reference projection (`project`) every kernel of this library is checked against.
//!
//! Ported from Sushi's `src/exl3/expert_exl3.zig` (MIT, Copyright (c) 2026 Theinruj Toranavikrai, David Dalcu) so
//! that a pack decodes to the same bits on either engine; the pins below are its pins. The bitstream itself is
//! exllamav3's (pack.cu: 16-bit emission, MSB-first, SWAP16 per uint32; quantize.py's tensor-core permutation).
const std = @import("std");

pub const TILE: usize = 16;
pub const TILE_VALUES: usize = TILE * TILE;
pub const HAD_DIM: usize = 128;
/// 1 / sqrt(128)
pub const HAD_SCALE: f32 = 0.08838834764831845;
pub const MCG_MULT: u32 = 0xCBAC1FED;
pub const MUL1_MULT: u32 = 0x83DCD12D;

pub const Codebook = enum(u8) {
    mul1 = 0,
    mcg = 1,

    pub const count = 2;

    pub fn fromName(name: []const u8) ?Codebook {
        if (std.mem.eql(u8, name, "mcg")) return .mcg;
        if (std.mem.eql(u8, name, "mul1")) return .mul1;
        return null;
    }

    pub fn multiplier(self: Codebook) u32 {
        return switch (self) {
            .mul1 => MUL1_MULT,
            .mcg => MCG_MULT,
        };
    }
};

/// The codeword width a pack's search hashed: a decoder takes the low `w` bits of the 16-bit sliding window.
/// The bitstream and the bits per weight are the same at every width; only the value a window decodes to
/// changes. A pack naming no width is w16 (masks nothing).
pub const Window = enum(u8) {
    w16 = 0,
    w8,
    w9,
    w10,
    w11,
    w12,
    w13,
    w14,
    w15,

    pub const min_bits: u8 = 8;
    pub const count = 16 - min_bits + 1;

    pub fn bits(self: Window) u8 {
        if (self == .w16) return 16;
        return @backingInt(self) + min_bits - 1;
    }

    pub fn mask(self: Window) u16 {
        return @truncate((@as(u32, 1) << @intCast(self.bits())) - 1);
    }

    pub fn fromBits(w: i64) ?Window {
        if (w == 16) return .w16;
        if (w < min_bits or w > 15) return null;
        return @fromBackingInt(@intCast(w - min_bits + 1));
    }
};

/// How a pack's trellis decodes: codebook and window travel together.
pub const Decode = struct {
    codebook: Codebook,
    window: Window = .w16,

    pub const mul1: Decode = .{ .codebook = .mul1 };
    pub const mcg: Decode = .{ .codebook = .mcg };
};

/// A trellis rate K = n/16 bits per weight, n the halfwords a packed 256-weight tile carries (even, 16..128).
/// Weight t's codeword is the 16-bit window of the tile's circular bitstream ending at `bitEnd(t)`.
pub const Rate = struct {
    n: u32,

    pub const min_n: u32 = 16;
    pub const max_n: u32 = 128;

    pub fn fromK(k: u32) Rate {
        return .{ .n = k * @as(u32, TILE) };
    }

    /// Any even n in [16, 128]; null otherwise (an odd n leaves the tile short of a whole uint32 word).
    pub fn fromN(n: u64) ?Rate {
        if (n % 2 != 0 or n < min_n or n > max_n) return null;
        return .{ .n = @intCast(n) };
    }

    pub fn halfwords(self: Rate) usize {
        return self.n;
    }

    pub fn words(self: Rate) usize {
        return self.n / 2;
    }

    pub fn totalBits(self: Rate) usize {
        return TILE * @as(usize, self.n);
    }

    /// One past the last bit of weight t's codeword window: floor((t + 1) * K).
    pub fn bitEnd(self: Rate, t: usize) usize {
        return ((t + 1) * @as(usize, self.n)) >> 4;
    }

    pub fn freshBits(self: Rate, t: usize) usize {
        return self.bitEnd(t) - if (t == 0) 0 else self.bitEnd(t - 1);
    }

    pub fn isInteger(self: Rate) bool {
        return self.n % TILE == 0;
    }

    /// The rate as a human reads it ("2.25"), never the halfword count.
    pub fn kText(self: Rate, buf: []u8) []const u8 {
        const k = @as(f64, @floatFromInt(self.n)) / 16.0;
        return std.fmt.bufPrint(buf, "{d}", .{k}) catch "?";
    }
};

/// Sushi's name for `Rate.fromN`: the rate a packed halfword count carries, null when not admitted.
pub fn kFromPackedDim(packed_hw: usize) ?Rate {
    return Rate.fromN(packed_hw);
}

pub fn f16BitsToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub fn f32ToF16Bits(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

/// MCG: the hashed codeword's two f16 halves (sign and mantissa bits kept, exponents fixed) summed.
pub fn decodeMcg(codeword: u16) u16 {
    const mixed = @as(u32, codeword) *% MCG_MULT;
    const pair = 0x3B603B60 ^ (mixed & 0x8FFF8FFF);
    const lo = f16BitsToF32(@truncate(pair));
    const hi = f16BitsToF32(@truncate(pair >> 16));
    return f32ToF16Bits(lo + hi);
}

/// MUL1: the byte sum of the hashed codeword as an f16 in [1024, 2048), affinely mapped.
pub fn decodeMul1(codeword: u16) u16 {
    const mixed = @as(u32, codeword) *% MUL1_MULT;
    const bytes = std.mem.toBytes(mixed);
    var byte_sum: u32 = 0;
    for (bytes) |b| byte_sum += b;
    const h = f16BitsToF32(@truncate(0x6400 + byte_sum));
    const inverse = f16BitsToF32(0x1EEE);
    const bias = f16BitsToF32(0xC931);
    return f32ToF16Bits(@mulAdd(f32, h, inverse, bias));
}

pub fn decodeCodeword(codeword: u16, codebook: Codebook) u16 {
    return switch (codebook) {
        .mcg => decodeMcg(codeword),
        .mul1 => decodeMul1(codeword),
    };
}

fn wordU32(words: []const u16, index: usize) u32 {
    return @as(u32, words[index * 2]) | (@as(u32, words[index * 2 + 1]) << 16);
}

/// The 256 codewords of one packed tile, in code-position order (the tensor-core order; `decodeTile` permutes).
/// The tile's bitstream is circular (tail-biting): position 0's window takes its leading bits from the stream's end.
pub fn unpackTile(words: []const u16, rate: Rate, out: *[TILE_VALUES]u16) void {
    const word_count = rate.words();
    const total = rate.totalBits();
    var thread: usize = 0;
    while (thread < 128) : (thread += 1) {
        const e0 = rate.bitEnd(thread * 2);
        const e1 = rate.bitEnd(thread * 2 + 1);
        const bit0 = e0 + total - 16;
        const bit2 = e1 + total;
        const index0 = bit0 / 32;
        const index1 = (bit2 - 1) / 32;
        const shift: u6 = @intCast((index1 + 1) * 32 - bit2);
        const merged = (@as(u64, wordU32(words, index0 % word_count)) << 32) | @as(u64, wordU32(words, index1 % word_count));
        const funnel: u32 = @truncate(merged >> shift);
        const fresh: u5 = @intCast(e1 - e0);
        out[thread * 2] = @truncate((funnel >> fresh) & 0xFFFF);
        out[thread * 2 + 1] = @truncate(funnel & 0xFFFF);
    }
}

/// The stream bit at circular position `bit` (0 = the first emitted bit): stream bit j sits at bit 31 - (j % 32)
/// of uint32 word j / 32, the words being the stored halfword pairs (little-endian).
pub fn streamBit(words: []const u16, rate: Rate, bit: usize) u1 {
    const total = rate.totalBits();
    const j = bit % total;
    const w = wordU32(words, j / 32);
    return @truncate(w >> @intCast(31 - (j % 32)));
}

/// Weight t's codeword read bit by bit from the stream: the plain definition `unpackTile` must agree with.
pub fn codewordByDefinition(words: []const u16, rate: Rate, t: usize) u16 {
    const total = rate.totalBits();
    const end = rate.bitEnd(t);
    var cw: u16 = 0;
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        const pos = (end + total - 16 + i) % total;
        cw = (cw << 1) | @as(u16, streamBit(words, rate, pos));
    }
    return cw;
}

/// One tile's 256 weights as f16 bits in row-major [16][16] order.
pub fn decodeTile(words: []const u16, rate: Rate, dec: Decode, out: *[TILE_VALUES]u16) void {
    var codewords: [TILE_VALUES]u16 = undefined;
    unpackTile(words, rate, &codewords);
    var perm: [TILE_VALUES]usize = undefined;
    tensorCorePerm(&perm);
    const mask = dec.window.mask();
    for (codewords, 0..) |cw, i| {
        out[perm[i]] = decodeCodeword(cw & mask, dec.codebook);
    }
}

/// quantize.py's layout: code position p holds the row-major tile index perm[p].
pub fn tensorCorePerm(out: *[TILE_VALUES]usize) void {
    var thread: usize = 0;
    while (thread < 32) : (thread += 1) {
        const row0 = (thread % 4) * 2;
        const row1 = row0 + 1;
        const row2 = row0 + 8;
        const row3 = row0 + 9;
        const col0 = thread / 4;
        const col1 = col0 + 8;
        const base = thread * 8;
        out[base + 0] = row0 * 16 + col0;
        out[base + 1] = row1 * 16 + col0;
        out[base + 2] = row2 * 16 + col0;
        out[base + 3] = row3 * 16 + col0;
        out[base + 4] = row0 * 16 + col1;
        out[base + 5] = row1 * 16 + col1;
        out[base + 6] = row2 * 16 + col1;
        out[base + 7] = row3 * 16 + col1;
    }
}

fn hadamardEntry(row: usize, col: usize) f32 {
    const bits = @popCount(row & col);
    const sign: f32 = if (bits % 2 == 0) 1.0 else -1.0;
    return sign * HAD_SCALE;
}

/// The Sylvester Hadamard in natural order, scaled 1/sqrt(128), applied in place (f32 reference order).
pub fn hadamard128(values: *[HAD_DIM]f32) void {
    var tmp: [HAD_DIM]f32 = undefined;
    for (0..HAD_DIM) |r| {
        var acc: f32 = 0;
        for (0..HAD_DIM) |k| acc += hadamardEntry(r, k) * values[k];
        tmp[r] = acc;
    }
    values.* = tmp;
}

/// The inner (pre-Hadamard, unscaled) weight [in, out] as f16 bits.
pub fn reconstructInner(trellis: []const u16, in_features: usize, out_features: usize, rate: Rate, dec: Decode, out: []u16) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = rate.halfwords();
    var tile_out: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], rate, dec, &tile_out);
            for (0..TILE) |r| {
                const dst = (tk * TILE + r) * out_features + tn * TILE;
                @memcpy(out[dst .. dst + TILE], tile_out[r * TILE ..][0..TILE]);
            }
        }
    }
}

/// The public weight [in, out] as f16 bits: H128 over input blocks, suh per input row, H128 over output blocks,
/// svh per output column (exllamav3's dequant order).
pub fn reconstructPublic(allocator: std.mem.Allocator, trellis: []const u16, suh: []const u16, svh: []const u16, in_features: usize, out_features: usize, rate: Rate, dec: Decode, out: []u16) !void {
    const inner = try allocator.alloc(u16, in_features * out_features);
    defer allocator.free(inner);
    reconstructInner(trellis, in_features, out_features, rate, dec, inner);
    const w = try allocator.alloc(f32, in_features * out_features);
    defer allocator.free(w);
    for (inner, 0..) |bits, i| w[i] = f16BitsToF32(bits);
    var row_block: usize = 0;
    while (row_block < in_features) : (row_block += HAD_DIM) {
        for (0..out_features) |col| {
            var vec: [HAD_DIM]f32 = undefined;
            for (0..HAD_DIM) |r| vec[r] = w[(row_block + r) * out_features + col];
            hadamard128(&vec);
            for (0..HAD_DIM) |r| w[(row_block + r) * out_features + col] = vec[r];
        }
    }
    for (0..in_features) |r| {
        const s = f16BitsToF32(suh[r]);
        const row = w[r * out_features ..][0..out_features];
        for (row) |*v| v.* *= s;
    }
    var col_block: usize = 0;
    while (col_block < out_features) : (col_block += HAD_DIM) {
        for (0..in_features) |r| {
            var vec: [HAD_DIM]f32 = undefined;
            for (0..HAD_DIM) |c| vec[c] = w[r * out_features + col_block + c];
            hadamard128(&vec);
            for (0..HAD_DIM) |c| w[r * out_features + col_block + c] = vec[c];
        }
    }
    for (0..out_features) |c| {
        const s = f16BitsToF32(svh[c]);
        var r: usize = 0;
        while (r < in_features) : (r += 1) w[r * out_features + c] *= s;
    }
    for (w, 0..) |v, i| out[i] = f32ToF16Bits(v);
}

/// x (f16-rounded) * suh, then H128 per 128-block, rounded to f16: the GEMM's left operand.
pub fn prepareInput(x: []const f32, suh: []const u16, out: []f32) void {
    for (x, suh, out) |xv, s, *d| {
        d.* = f16BitsToF32(f32ToF16Bits(xv)) * f16BitsToF32(s);
    }
    var block: usize = 0;
    while (block < out.len) : (block += HAD_DIM) {
        var vec: [HAD_DIM]f32 = undefined;
        @memcpy(&vec, out[block..][0..HAD_DIM]);
        hadamard128(&vec);
        for (0..HAD_DIM) |i| out[block + i] = f16BitsToF32(f32ToF16Bits(vec[i]));
    }
}

/// transformed [in] times the inner weight, f32 accumulation, the result rounded to f16.
pub fn innerGemv(trellis: []const u16, transformed: []const f32, in_features: usize, out_features: usize, rate: Rate, dec: Decode, out: []f32) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = rate.halfwords();
    @memset(out, 0);
    var tile_w: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], rate, dec, &tile_w);
            const xbase = tk * TILE;
            const ybase = tn * TILE;
            for (0..TILE) |r| {
                const xv = transformed[xbase + r];
                for (0..TILE) |c| out[ybase + c] += xv * f16BitsToF32(tile_w[r * TILE + c]);
            }
        }
    }
    for (out) |*v| v.* = f16BitsToF32(f32ToF16Bits(v.*));
}

/// H128 per 128-block of the inner product, then * svh, rounded to f16.
pub fn finishOutput(inner: []const f32, svh: []const u16, out: []f32) void {
    @memcpy(out, inner);
    var block: usize = 0;
    while (block < out.len) : (block += HAD_DIM) {
        var vec: [HAD_DIM]f32 = undefined;
        @memcpy(&vec, out[block..][0..HAD_DIM]);
        hadamard128(&vec);
        @memcpy(out[block..][0..HAD_DIM], &vec);
    }
    for (out, svh) |*v, s| v.* = f16BitsToF32(f32ToF16Bits(v.* * f16BitsToF32(s)));
}

/// One row through one EXL3 linear: the reference every kernel is held to.
pub fn project(x: []const f32, trellis: []const u16, suh: []const u16, svh: []const u16, in_features: usize, out_features: usize, rate: Rate, dec: Decode, transformed: []f32, inner: []f32, out: []f32) void {
    std.debug.assert(x.len == in_features and suh.len == in_features and transformed.len == in_features);
    std.debug.assert(svh.len == out_features and inner.len == out_features and out.len == out_features);
    prepareInput(x, suh, transformed);
    innerGemv(trellis, transformed, in_features, out_features, rate, dec, inner);
    finishOutput(inner, svh, out);
}

// ── the packer (exllamav3 pack.cu), for tests: 256 K-bit symbols -> the tile's halfwords ──

/// Packs 256 symbols of `k` bits (integer rates) as pack.cu writes them: 16-bit emission MSB-first, SWAP16 per
/// uint32, i.e. stream bit j at bit 31 - (j % 32) of word j / 32. The halfwords are the stored (little-endian) ones.
pub fn packSymbols(symbols: []const u16, k: u32, out: []u16) void {
    std.debug.assert(symbols.len == TILE_VALUES and out.len == TILE_VALUES * k / 16);
    var out16: [TILE_VALUES]u16 = undefined;
    var i: usize = 0;
    var j: usize = 0;
    var row: usize = 0;
    while (row < 16) : (row += 1) {
        var buf: u32 = 0;
        var kk: i32 = 32;
        var col: usize = 0;
        while (col < 16) : (col += 1) {
            kk -= @intCast(k);
            buf |= (@as(u32, symbols[i]) & ((@as(u32, 1) << @intCast(k)) - 1)) << @intCast(kk);
            if (kk <= 16) {
                out16[j] = @truncate(buf >> 16);
                buf = buf << 16;
                kk += 16;
                j += 1;
            }
            i += 1;
        }
    }
    // SWAP16: within each uint32 the two halfwords swap, so the little-endian halfword pair reads as the stream
    var w: usize = 0;
    while (w < out.len / 2) : (w += 1) {
        out[2 * w] = out16[2 * w + 1];
        out[2 * w + 1] = out16[2 * w];
    }
}

const testing = std.testing;

test "exl3lib MUL1 codebook pins known codewords (Sushi's pins)" {
    try testing.expectEqual(@as(u16, 49896), decodeMul1(0));
    try testing.expectEqual(@as(u16, 14625), decodeMul1(1));
    try testing.expectEqual(@as(u16, 47511), decodeMul1(7));
}

test "exl3lib codebook names resolve; an unknown name is null; multipliers" {
    try testing.expectEqual(Codebook.mul1, Codebook.fromName("mul1").?);
    try testing.expectEqual(Codebook.mcg, Codebook.fromName("mcg").?);
    try testing.expectEqual(@as(?Codebook, null), Codebook.fromName("mul2"));
    try testing.expectEqual(MUL1_MULT, Codebook.mul1.multiplier());
    try testing.expectEqual(MCG_MULT, Codebook.mcg.multiplier());
}

test "exl3lib MCG zero codeword maps to the finite half pair" {
    const pair: u32 = 0x3B603B60;
    const want = f32ToF16Bits(f16BitsToF32(@truncate(pair)) + f16BitsToF32(@truncate(pair >> 16)));
    try testing.expectEqual(want, decodeMcg(0));
}

test "exl3lib windows: bits, masks, the config widths" {
    try testing.expectEqual(@as(u8, 16), Window.w16.bits());
    try testing.expectEqual(@as(u16, 0xFFFF), Window.w16.mask());
    try testing.expectEqual(@as(u8, 12), Window.w12.bits());
    try testing.expectEqual(@as(u16, 0x0FFF), Window.w12.mask());
    try testing.expectEqual(Window.w8, Window.fromBits(8).?);
    try testing.expectEqual(Window.w15, Window.fromBits(15).?);
    try testing.expectEqual(Window.w16, Window.fromBits(16).?);
    try testing.expectEqual(@as(?Window, null), Window.fromBits(7));
    try testing.expectEqual(@as(?Window, null), Window.fromBits(17));
}

test "exl3lib rates: every even n in [16, 128] admits, the window ends follow floor((t+1)K)" {
    try testing.expectEqual(@as(?Rate, null), Rate.fromN(15));
    try testing.expectEqual(@as(?Rate, null), Rate.fromN(130));
    try testing.expectEqual(@as(?Rate, null), Rate.fromN(49));
    var n: u64 = 16;
    while (n <= 128) : (n += 2) {
        const r = Rate.fromN(n).?;
        try testing.expectEqual(r.totalBits(), r.bitEnd(TILE_VALUES - 1));
        var sum: usize = 0;
        for (0..TILE_VALUES) |t| sum += r.freshBits(t);
        try testing.expectEqual(r.totalBits(), sum);
    }
    try testing.expectEqual(@as(usize, 4), Rate.fromK(4).bitEnd(0));
    try testing.expectEqual(@as(usize, 80), Rate.fromN(80).?.bitEnd(15));
    try testing.expectEqual(@as(usize, 2), Rate.fromN(36).?.bitEnd(0)); // 2.25 bits: 2, 4, 6, 9, ...
    try testing.expectEqual(@as(usize, 9), Rate.fromN(36).?.bitEnd(3));
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("2.25", Rate.fromN(36).?.kText(&buf));
    try testing.expectEqualStrings("5", Rate.fromN(80).?.kText(&buf));
}

test "exl3lib unpackTile agrees with the bit-by-bit definition at every admitted rate" {
    var prng = std.Random.DefaultPrng.init(0xE7135EED);
    const rnd = prng.random();
    var words: [Rate.max_n]u16 = undefined;
    var got: [TILE_VALUES]u16 = undefined;
    var n: u64 = 16;
    while (n <= 128) : (n += 2) {
        const rate = Rate.fromN(n).?;
        for (words[0..rate.n]) |*w| w.* = rnd.int(u16);
        unpackTile(words[0..rate.n], rate, &got);
        for (0..TILE_VALUES) |t| try testing.expectEqual(codewordByDefinition(words[0..rate.n], rate, t), got[t]);
    }
}

test "exl3lib the packer round-trips integer rates: each window's fresh bits are the symbol, K1 to K8" {
    var prng = std.Random.DefaultPrng.init(0x5EED5EED);
    const rnd = prng.random();
    var symbols: [TILE_VALUES]u16 = undefined;
    var words: [Rate.max_n]u16 = undefined;
    var got: [TILE_VALUES]u16 = undefined;
    var k: u32 = 1;
    while (k <= 8) : (k += 1) {
        const rate = Rate.fromK(k);
        for (&symbols) |*s| s.* = rnd.int(u16) & @as(u16, @truncate((@as(u32, 1) << @intCast(k)) - 1));
        packSymbols(&symbols, k, words[0..rate.n]);
        unpackTile(words[0..rate.n], rate, &got);
        // weight t's window ends at bit (t+1)K: its low K bits are symbol t
        const kmask: u16 = @truncate((@as(u32, 1) << @intCast(k)) - 1);
        for (0..TILE_VALUES) |t| try testing.expectEqual(symbols[t], got[t] & kmask);
    }
}

test "exl3lib the tensor-core permutation is a permutation of the 256 tile positions" {
    var perm: [TILE_VALUES]usize = undefined;
    tensorCorePerm(&perm);
    var seen: [TILE_VALUES]bool = @splat(false);
    for (perm) |p| {
        try testing.expect(p < TILE_VALUES and !seen[p]);
        seen[p] = true;
    }
    try testing.expectEqual(@as(usize, 0), perm[0]);
    try testing.expectEqual(@as(usize, 16), perm[1]);
    try testing.expectEqual(@as(usize, 8), perm[4]);
}

test "exl3lib the Hadamard is orthonormal: applied twice it is the identity (within f32 rounding)" {
    var v: [HAD_DIM]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 17)) - 8.0;
    var w = v;
    hadamard128(&w);
    hadamard128(&w);
    for (v, w) |a, b| try testing.expect(@abs(a - b) < 1e-4);
}

test "exl3lib project: a 128x128 K4 linear with identity-like scales reproduces reconstructPublic's matvec" {
    var prng = std.Random.DefaultPrng.init(0x7E57);
    const rnd = prng.random();
    const in_f: usize = 128;
    const out_f: usize = 128;
    const rate = Rate.fromK(4);
    const n_tiles = (in_f / TILE) * (out_f / TILE);
    const a = testing.allocator;
    const trellis = try a.alloc(u16, n_tiles * rate.n);
    defer a.free(trellis);
    for (trellis) |*w| w.* = rnd.int(u16);
    const suh = try a.alloc(u16, in_f);
    defer a.free(suh);
    const svh = try a.alloc(u16, out_f);
    defer a.free(svh);
    for (suh) |*s| s.* = f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh) |*s| s.* = f32ToF16Bits(0.5 + rnd.float(f32));
    const pub_w = try a.alloc(u16, in_f * out_f);
    defer a.free(pub_w);
    try reconstructPublic(a, trellis, suh, svh, in_f, out_f, rate, .mul1, pub_w);
    var x: [128]f32 = undefined;
    for (&x) |*v| v.* = rnd.float(f32) - 0.5;
    var transformed: [128]f32 = undefined;
    var inner: [128]f32 = undefined;
    var y: [128]f32 = undefined;
    project(&x, trellis, suh, svh, in_f, out_f, rate, .mul1, &transformed, &inner, &y);
    // the public weight's matvec on the f16-rounded x, to a loose tolerance: the two paths round at different points
    var max_ref: f32 = 0;
    var max_err: f32 = 0;
    for (0..out_f) |c| {
        var acc: f32 = 0;
        for (0..in_f) |r| acc += f16BitsToF32(f32ToF16Bits(x[r])) * f16BitsToF32(pub_w[r * out_f + c]);
        max_ref = @max(max_ref, @abs(acc));
        max_err = @max(max_err, @abs(acc - y[c]));
    }
    try testing.expect(max_ref > 0);
    try testing.expect(max_err <= 0.02 * max_ref);
}
