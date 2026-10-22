//! The lane plan every exl3lib kernel reads tiles by, at any admitted rate: lane L of a simdgroup owns code positions
//! 8L..8L+7 of a 256-weight tile (the tensor-core order `format.tensorCorePerm` maps to tile rows and columns). Its eight
//! codewords end at stream bits e_m = ((8L + m + 1) * n) >> 4 and span 7n/16 + 16 bits (<= 72 at n = 128) from the first
//! window's start, which lands anywhere in its word: at most four consecutive 32-bit words of the tile's circular bitstream
//! (two at n <= 36, three at n <= 108). A lane loads words b..b+3 (mod n/2) once per tile and extracts each window with a
//! loop-invariant (word, shift) pair: `lanePlan` computes those pairs on the host exactly
//! as the Metal helper text (`metal_helpers`) computes them from its `NHW` constant, and the test below holds both to
//! `format.unpackTile` at every even n in [16, 128].
//!
//! Bit order: stream bit j is bit 31 - (j % 32) of word j / 32 (exllamav3's 16-bit MSB-first emission with SWAP16), so
//! two consecutive words (lo = word q-1, hi = word q) read as one 64-bit MSB-first stream and the 16 bits ending at stream
//! bit e are `((lo << 32) | hi) >> (32 (q + 1) - e)` masked to 16 bits.
const std = @import("std");
const format = @import("format.zig");

pub const Window = struct {
    /// which of the lane's four loaded words holds the window's last bit (0..3)
    q: u8,
    /// right shift of the (word q-1 : word q) pair that lands the window's last bit at bit 0 (0..31)
    sh: u8,
};

pub const Plan = struct {
    /// the first word the lane loads (tile word index, before the modulo)
    base: u32,
    windows: [8]Window,
    /// how many of the four words the lane needs (2..4)
    words: u8,
};

/// Lane L's plan at rate n: the Metal helper's arithmetic, on the host.
pub fn lanePlan(rate: format.Rate, lane: u32) Plan {
    const n = rate.n;
    const e0 = ((8 * lane + 1) * n) >> 4;
    const start = e0 + 16 * n - 16; // the first bit the lane needs, kept positive by one period
    const base = start >> 5;
    var plan: Plan = .{ .base = base, .windows = undefined, .words = 2 };
    var m: u32 = 0;
    while (m < 8) : (m += 1) {
        const e = ((8 * lane + m + 1) * n) >> 4;
        const off = e + 16 * n - 32 * base; // the end bit relative to word `base`: 16 <= off <= 128
        std.debug.assert(off >= 16 and off <= 128);
        const q: u8 = @intCast((off - 1) >> 5);
        const sh: u8 = @intCast(32 * (@as(u32, q) + 1) - off);
        plan.windows[m] = .{ .q = q, .sh = sh };
        plan.words = @max(plan.words, q + 1);
    }
    return plan;
}

/// A tile word by index modulo the tile (the circular stream).
fn tileWord(words: []const u16, rate: format.Rate, index: u32) u32 {
    const nw: u32 = @intCast(rate.words());
    const i = index % nw;
    return @as(u32, words[i * 2]) | (@as(u32, words[i * 2 + 1]) << 16);
}

/// The Metal funnel: the 16 bits ending at bit 0 of (lo : hi) >> sh.
pub fn funnel(lo: u32, hi: u32, sh: u8) u16 {
    const v: u32 = if (sh == 0) hi else (hi >> @intCast(sh)) | (lo << @intCast(32 - @as(u32, sh)));
    return @truncate(v & 0xFFFF);
}

/// Lane L's eight codewords (code positions 8L + m) read with its plan: four loads, eight funnels.
pub fn laneCodewords(words: []const u16, rate: format.Rate, lane: u32, out: *[8]u16) void {
    const p = lanePlan(rate, lane);
    const w = [4]u32{ tileWord(words, rate, p.base), tileWord(words, rate, p.base + 1), tileWord(words, rate, p.base + 2), tileWord(words, rate, p.base + 3) };
    for (p.windows, 0..) |win, m| {
        const hi = w[win.q];
        const lo = if (win.q == 0) tileWord(words, rate, p.base + @as(u32, @intCast(rate.words())) - 1) else w[win.q - 1];
        out[m] = funnel(lo, hi, win.sh);
    }
}

/// The integer-rate fast path (n a multiple of 16, K = n / 16): lane L's first window starts at stream bit
/// 8 L K + K - 16 (+ one period). The lane-dependent part 8 L K is a multiple of 8, so relative to the word holding the
/// start the position is 16 a + 8 b + c0 with a, b in {0, 1} per lane and c0 = K mod 8 a constant. Taking halfword a of word
/// qb as the lane's origin leaves start = 8 b + c0 + K m for window m: its 32-bit funnel word is H_j,
/// j = (8 b + cm) >> 4 = (cm >> 4) + carry with cm = c0 + K m and carry = b and (cm mod 16) >= 8, and the window is the
/// 16 bits of H_j starting at bit position (8 b + cm) & 15 from its top. H_j are the 32-bit values at halfword offsets
/// a + j of word qb (even offsets: a word; odd: the low half of one and the high half of the next). Per weight: one select
/// (carry), one shift, one mask. This mirrors the Metal fast path; the test holds it to `format.unpackTile`.
pub const IntPlan = struct {
    /// the word holding the first window's start (before the modulo), the halfword within it, and the 8-bit offset
    qb: u32,
    a: u32,
    b: u32,
};

pub fn intPlan(rate: format.Rate, l: u32) IntPlan {
    std.debug.assert(rate.n % 16 == 0);
    const k = rate.n / 16;
    const start = 8 * l * k + k + 16 * rate.n - 16; // the first window's start bit, one period in
    const qb = start >> 5;
    const lp = start - 32 * qb; // 0, 8, 16, 24
    const a = lp >> 4;
    const b = (lp >> 3) & 1;
    return .{ .qb = qb, .a = a, .b = b };
}

/// The 32 bits at halfword offset j (0..7) of the words starting at `qb` (stream order: MSB first).
fn hword(words: []const u16, rate: format.Rate, qb: u32, j: u32) u32 {
    const w0 = tileWord(words, rate, qb + (j >> 1));
    if (j & 1 == 0) return w0;
    const w1 = tileWord(words, rate, qb + (j >> 1) + 1);
    return (w0 << 16) | (w1 >> 16);
}

pub fn laneCodewordsInt(words: []const u16, rate: format.Rate, l: u32, out: *[8]u16) void {
    const k = rate.n / 16;
    const p = intPlan(rate, l);
    var m: u32 = 0;
    while (m < 8) : (m += 1) {
        const cm = (k % 8) + k * m;
        const carry: u32 = if (p.b == 1 and (cm & 15) >= 8) 1 else 0;
        const j = (cm >> 4) + carry;
        const r = (8 * p.b + cm) & 15;
        const h = hword(words, rate, p.qb, j + p.a);
        out[m] = @truncate((h >> @intCast(16 - r)) & 0xFFFF);
    }
}

/// The general fast path (every even n): 8 L n is a multiple of 16 when n is even, so window m of lane L ends at
/// e_m = (8 L n) / 16 + (((m + 1) n) >> 4): a lane constant plus the compile-time constant C_m = (((m + 1) n) >> 4) - (n >> 4)
/// past the lane's first window. With s = e_0 - 16 (+ one period) the lane's start bit, qb = s >> 5 its word, a = (s >> 4) & 1
/// its halfword and lp = s & 15, window m starts lp + C_m bits past halfword a of word qb, so it is the 16 bits at offset
/// (lp + C_m) & 15 from the top of H_j, j = (lp + C_m) >> 4, which is J_m = C_m >> 4 or J_m + 1 (one select). The integer
/// path (`intPlan`) is the special case lp in {0, 8}.
pub const GenPlan = struct { qb: u32, a: u32, lp: u32 };

pub fn genPlan(rate: format.Rate, l: u32) GenPlan {
    const n = rate.n;
    std.debug.assert(n % 2 == 0);
    const e0 = ((8 * l + 1) * n) >> 4;
    const s = e0 + 16 * n - 16;
    return .{ .qb = s >> 5, .a = (s >> 4) & 1, .lp = s & 15 };
}

pub fn genOffset(rate: format.Rate, m: u32) u32 {
    return (((m + 1) * rate.n) >> 4) - (rate.n >> 4);
}

pub fn laneCodewordsGen(words: []const u16, rate: format.Rate, l: u32, out: *[8]u16) void {
    const p = genPlan(rate, l);
    var m: u32 = 0;
    while (m < 8) : (m += 1) {
        const pos = p.lp + genOffset(rate, m);
        const j = pos >> 4;
        std.debug.assert(j == (genOffset(rate, m) >> 4) or j == (genOffset(rate, m) >> 4) + 1);
        std.debug.assert(j + p.a <= 5); // the Metal IntWords holds halfword offsets 0..5
        const r = pos & 15;
        const h = hword(words, rate, p.qb, j + p.a);
        out[m] = @truncate((h >> @intCast(16 - r)) & 0xFFFF);
    }
}

/// The Metal text of the helpers every exl3lib kernel includes, each templated on what the kernel body binds from its
/// template ints (`NHW` the rate's halfwords, `WBITS` the window, `CODEBOOK` 0 MUL1 / 1 MCG): `exl3_lane_words<N>` loads
/// a lane's four words, `exl3_lane_cw<WMASK>` extracts window m, `exl3_decode<CB>` turns a codeword into the f16 weight.
pub const metal_helpers =
    \\// exl3lib helpers: template parameters are only visible inside a kernel body, so every helper takes the rate's
    \\// halfwords N (and the window mask / codebook) as its own template argument; the body binds them from NHW, WBITS,
    \\// CODEBOOK once.
    \\struct LaneWords { uint w0, w1, w2, w3; uint wprev; };
    \\// lane L's first needed word: base = ((8L + 1) N >> 4) + 16 N - 16) >> 5; the lane's windows end off_m bits past it
    \\template <uint N> inline uint exl3_lane_base(uint L) { return ((((8u * L + 1u) * N) >> 4u) + 16u * N - 16u) >> 5u; }
    \\template <uint N> inline uint exl3_lane_off(uint L, uint m, uint base) { return (((8u * L + m + 1u) * N) >> 4u) + 16u * N - 32u * base; }
    \\template <uint N> inline LaneWords exl3_lane_words(const device uint* tile, uint base) {
    \\  constexpr uint NW = N / 2u;
    \\  LaneWords r; const uint b = base % NW;
    \\  r.w0 = tile[b]; r.w1 = tile[(b + 1u) % NW]; r.w2 = tile[(b + 2u) % NW]; r.w3 = tile[(b + 3u) % NW]; r.wprev = tile[(b + NW - 1u) % NW];
    \\  return r;
    \\}
    \\inline uint exl3_funnel(uint lo, uint hi, uint sh) { return ((sh == 0u) ? hi : ((hi >> sh) | (lo << (32u - sh)))) & 0xffffu; }
    \\// window m of lane L from its loaded words: off = end bit past word base, q = its word, sh lands its last bit at bit 0
    \\template <uint WMASK> inline uint exl3_lane_cw(LaneWords w, uint off) {
    \\  const uint q = (off - 1u) >> 5u; const uint sh = 32u * (q + 1u) - off;
    \\  const uint hi = (q == 0u) ? w.w0 : ((q == 1u) ? w.w1 : ((q == 2u) ? w.w2 : w.w3));
    \\  const uint lo = (q == 0u) ? w.wprev : ((q == 1u) ? w.w0 : ((q == 2u) ? w.w1 : w.w2));
    \\  return exl3_funnel(lo, hi, sh) & WMASK;
    \\}
    \\// the integer-rate fast path (N % 16 == 0): the lane's base word qb' and b (see lane.zig IntPlan)
    \\template <uint N> inline uint exl3_int_qb(uint L) { constexpr uint K = N / 16u; return (8u * L * K + K + 16u * N - 16u) >> 5u; }
    \\template <uint N> inline uint exl3_int_a(uint L) { constexpr uint K = N / 16u; return ((8u * L * K + K + 16u * N - 16u) >> 4u) & 1u; }
    \\template <uint N> inline uint exl3_int_b(uint L) { constexpr uint K = N / 16u; return ((8u * L * K + K + 16u * N - 16u) >> 3u) & 1u; }
    \\// the 32-bit values at halfword offsets a + 0..5 of the lane's word qb: a lane's eight windows span at most
    \\// 15 + 7 N / 16 + 16 <= 87 bits past that halfword at N <= 128, so four words cover them (h[6], h[7] unreachable)
    \\struct IntWords { uint h[6]; };
    \\template <uint N> inline IntWords exl3_int_words(const device uint* tile, uint qb, uint a) {
    \\  constexpr uint NW = N / 2u;
    \\  const uint b0 = qb % NW;
    \\  const uint w0 = tile[b0], w1 = tile[(b0 + 1u) % NW], w2 = tile[(b0 + 2u) % NW], w3 = tile[(b0 + 3u) % NW];
    \\  const uint o1 = (w0 << 16u) | (w1 >> 16u), o3 = (w1 << 16u) | (w2 >> 16u), o5 = (w2 << 16u) | (w3 >> 16u);
    \\  IntWords r;
    \\  if (a == 0u) { r.h[0] = w0; r.h[1] = o1; r.h[2] = w1; r.h[3] = o3; r.h[4] = w2; r.h[5] = o5; }
    \\  else { r.h[0] = o1; r.h[1] = w1; r.h[2] = o3; r.h[3] = w2; r.h[4] = o5; r.h[5] = w3; }
    \\  return r;
    \\}
    \\// window m: cm = (K mod 8) + K m, j = (cm >> 4) + carry, the 16 bits of h[j] at (8 b + cm) & 15 from the top; M is the unrolled m
    \\template <uint N, uint WMASK, uint M> inline uint exl3_int_cw(IntWords w, uint b) {
    \\  constexpr uint K = N / 16u; constexpr uint CM = (K % 8u) + K * M; constexpr uint JC = CM >> 4u; constexpr bool CARRY_IF_B = (CM & 15u) >= 8u;
    \\  const uint hsel = (CARRY_IF_B && b != 0u) ? w.h[JC + 1u] : w.h[JC];
    \\  const uint cw = metal::extract_bits(hsel, 16u - ((8u * b + CM) & 15u), 16u);
    \\  return (WMASK == 0xffffu) ? cw : (cw & WMASK);
    \\}
    \\// the general fast path (every even N; lane.zig GenPlan): the lane's start bit s, its word qb, halfword a, offset lp
    \\template <uint N> inline uint exl3_gen_s(uint L) { return (((8u * L + 1u) * N) >> 4u) + 16u * N - 16u; }
    \\template <uint N> inline uint exl3_gen_qb(uint L) { return exl3_gen_s<N>(L) >> 5u; }
    \\template <uint N> inline uint exl3_gen_a(uint L) { return (exl3_gen_s<N>(L) >> 4u) & 1u; }
    \\template <uint N> inline uint exl3_gen_lp(uint L) { return exl3_gen_s<N>(L) & 15u; }
    \\// window m: C_m = ((m + 1) N >> 4) - (N >> 4) past the lane's start; j = (lp + C_m) >> 4 is J or J + 1
    \\template <uint N, uint WMASK, uint M> inline uint exl3_gen_cw(IntWords w, uint lp) {
    \\  constexpr uint CM = (((M + 1u) * N) >> 4u) - (N >> 4u); constexpr uint J = CM >> 4u;
    \\  const uint pos = lp + CM;
    \\  const uint hsel = ((pos >> 4u) == J) ? w.h[J] : w.h[J + 1u];
    \\  // the 16 bits at offset (pos & 15) from the top: one bit-field extract (bits 16 - r .. 31 - r)
    \\  const uint cw = metal::extract_bits(hsel, 16u - (pos & 15u), 16u);
    \\  return (WMASK == 0xffffu) ? cw : (cw & WMASK);
    \\}
    \\template <int CB> inline half exl3_decode(uint cw) {
    \\  if (CB == 1) {
    \\    const uint mixed = cw * 0xCBAC1FEDu; const uint pair = 0x3B603B60u ^ (mixed & 0x8FFF8FFFu);
    \\    return half(float(as_type<half>(ushort(pair & 0xffffu))) + float(as_type<half>(ushort(pair >> 16u))));
    \\  }
    \\  const uint rr = cw * 0x83DCD12Du;
    \\  const uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
    \\  const half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
    \\  return fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u)));
    \\}
    \\
;

const testing = std.testing;

test "exl3lib lane plan: every lane's eight windows lie in its four words at every admitted rate (two at n <= 36)" {
    var n: u64 = 16;
    while (n <= 128) : (n += 2) {
        const rate = format.Rate.fromN(n).?;
        var lane: u32 = 0;
        while (lane < 32) : (lane += 1) {
            const p = lanePlan(rate, lane);
            try testing.expect(p.words >= 2 and p.words <= 4);
            if (n <= 36) try testing.expectEqual(@as(u8, 2), p.words); // 7n/16 + 16 <= 31.75 bits: never a third word
            if (n <= 108) try testing.expect(p.words <= 3);
        }
    }
}

test "exl3lib lane plan: the three-load funnel read equals unpackTile at every admitted rate (the Metal arithmetic)" {
    var prng = std.Random.DefaultPrng.init(0x1A4E);
    const rnd = prng.random();
    var words: [format.Rate.max_n]u16 = undefined;
    var ref: [format.TILE_VALUES]u16 = undefined;
    var got: [8]u16 = undefined;
    var n: u64 = 16;
    while (n <= 128) : (n += 2) {
        const rate = format.Rate.fromN(n).?;
        var trial: u32 = 0;
        while (trial < 4) : (trial += 1) {
            for (words[0..rate.n]) |*w| w.* = rnd.int(u16);
            format.unpackTile(words[0..rate.n], rate, &ref);
            var lane: u32 = 0;
            while (lane < 32) : (lane += 1) {
                laneCodewords(words[0..rate.n], rate, lane, &got);
                for (got, 0..) |cw, m| try testing.expectEqual(ref[8 * lane + m], cw);
            }
        }
    }
}

test "exl3lib lane plan: the integer-rate fast path equals unpackTile at K1..K8 (the Metal select-shift-mask arithmetic)" {
    var prng = std.Random.DefaultPrng.init(0x1A4E1);
    const rnd = prng.random();
    var words: [format.Rate.max_n]u16 = undefined;
    var ref: [format.TILE_VALUES]u16 = undefined;
    var got: [8]u16 = undefined;
    var k: u32 = 1;
    while (k <= 8) : (k += 1) {
        const rate = format.Rate.fromK(k);
        var trial: u32 = 0;
        while (trial < 6) : (trial += 1) {
            for (words[0..rate.n]) |*w| w.* = rnd.int(u16);
            format.unpackTile(words[0..rate.n], rate, &ref);
            var l: u32 = 0;
            while (l < 32) : (l += 1) {
                laneCodewordsInt(words[0..rate.n], rate, l, &got);
                for (got, 0..) |cw, m| try testing.expectEqual(ref[8 * l + m], cw);
            }
        }
    }
}

test "exl3lib lane plan: the general fast path equals unpackTile at every even n in [16, 128] (the Metal arithmetic)" {
    var prng = std.Random.DefaultPrng.init(0x6E4E);
    const rnd = prng.random();
    var words: [format.Rate.max_n]u16 = undefined;
    var ref: [format.TILE_VALUES]u16 = undefined;
    var got: [8]u16 = undefined;
    var n: u64 = 16;
    while (n <= 128) : (n += 2) {
        const rate = format.Rate.fromN(n).?;
        var trial: u32 = 0;
        while (trial < 4) : (trial += 1) {
            for (words[0..rate.n]) |*w| w.* = rnd.int(u16);
            format.unpackTile(words[0..rate.n], rate, &ref);
            var l: u32 = 0;
            while (l < 32) : (l += 1) {
                laneCodewordsGen(words[0..rate.n], rate, l, &got);
                for (got, 0..) |cw, m| try testing.expectEqual(ref[8 * l + m], cw);
            }
        }
    }
}

test "exl3lib lane plan: the MUL1 decode in the Metal text is the reference's bit arithmetic (host mirror on the pins)" {
    // the text's integer path: rr = cw * MUL1; bs = byte pairs summed; hs = f16 bits (bs*0x10001 + 0x64000000) >> 16
    for ([_]u16{ 0, 1, 7, 0x1234, 0xFFFF }) |cw| {
        const rr = @as(u32, cw) *% format.MUL1_MULT;
        const bs = (rr & 0x00FF00FF) + ((rr >> 8) & 0x00FF00FF);
        const hs_bits: u16 = @truncate((bs *% 0x00010001 +% 0x64000000) >> 16);
        const hs = format.f16BitsToF32(hs_bits);
        const w = format.f32ToF16Bits(@mulAdd(f32, hs, format.f16BitsToF32(0x1EEE), format.f16BitsToF32(0xC931)));
        try testing.expectEqual(format.decodeMul1(cw), w);
    }
}
