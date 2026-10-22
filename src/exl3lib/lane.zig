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

/// The Metal text of the helpers every exl3lib kernel includes: `NHW` (the rate's halfwords), `CODEBOOK` (0 MUL1,
/// 1 MCG) and `WBITS` (the codeword window, 8..16) are template ints. `exl3_lane_words` loads a lane's four words,
/// `exl3_lane_cw` extracts window m, `exl3_decode` turns a codeword into the f16 weight (Sushi's / exllamav3's bits).
pub const metal_helpers =
    \\constexpr uint N = uint(NHW);
    \\constexpr uint NW = N / 2u;
    \\constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\struct LaneWords { uint w0, w1, w2, w3; uint wprev; };
    \\// lane L's first needed word: base = ((8L + 1) N >> 4) - 16 + 16 N) >> 5; the lane's windows end at off_m bits past it
    \\inline uint exl3_lane_base(uint L) { return ((((8u * L + 1u) * N) >> 4u) + 16u * N - 16u) >> 5u; }
    \\inline uint exl3_lane_off(uint L, uint m, uint base) { return (((8u * L + m + 1u) * N) >> 4u) + 16u * N - 32u * base; }
    \\inline LaneWords exl3_lane_words(const device uint* tile, uint base) {
    \\  LaneWords r; const uint b = base % NW;
    \\  r.w0 = tile[b]; r.w1 = tile[(b + 1u) % NW]; r.w2 = tile[(b + 2u) % NW]; r.w3 = tile[(b + 3u) % NW]; r.wprev = tile[(b + NW - 1u) % NW];
    \\  return r;
    \\}
    \\inline uint exl3_funnel(uint lo, uint hi, uint sh) { return ((sh == 0u) ? hi : ((hi >> sh) | (lo << (32u - sh)))) & 0xffffu; }
    \\// window m of lane L from its loaded words: off = end bit past word base, q = its word, sh lands its last bit at bit 0
    \\inline uint exl3_lane_cw(LaneWords w, uint off) {
    \\  const uint q = (off - 1u) >> 5u; const uint sh = 32u * (q + 1u) - off;
    \\  const uint hi = (q == 0u) ? w.w0 : ((q == 1u) ? w.w1 : ((q == 2u) ? w.w2 : w.w3));
    \\  const uint lo = (q == 0u) ? w.wprev : ((q == 1u) ? w.w0 : ((q == 2u) ? w.w1 : w.w2));
    \\  return exl3_funnel(lo, hi, sh) & WMASK;
    \\}
    \\inline half exl3_decode(uint cw) {
    \\  if (CODEBOOK == 1) {
    \\    const uint mixed = cw * 0xCBAC1FEDu; const uint pair = 0x3B603B60u ^ (mixed & 0x8FFF8FFFu);
    \\    return half(float(as_type<half>(ushort(pair & 0xffffu))) + float(as_type<half>(ushort(pair >> 16u))));
    \\  }
    \\  const uint rr = cw * 0x83DCD12Du;
    \\  const uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
    \\  const half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));
    \\  return half(fma(float(hs), float(as_type<half>(ushort(0x1EEEu))), float(as_type<half>(ushort(0xC931u)))));
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
