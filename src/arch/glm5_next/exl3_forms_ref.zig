//! The host truth of the GLM-5.3-Flash EXL3 K=3 decode forms (`exl3_forms.zig`) and the kernel bench
//! (`exl3_kbench.zig`): synthetic records in the pack's layout, the EXL3 decode (exl3/kernels.zig `reconstruct`, mul1
//! and mcg), the f64 expert forward of exl3_ref.py (y = t128(t128(x suh) W_hat) svh, t128 the natural-order Sylvester
//! Hadamard over 128-blocks scaled 1/sqrt(128); the clamped SwiGLU at 10), an f32 emulation of the seam chain's
//! arithmetic per output word (the order every form keeps), the union rule of the fused forms, and their column-sum
//! identity, and the gate / up texts g5k3_guone and g5k3_guone_pair per lane as written (the two-row rule's parity). CPU
//! only: nothing here creates an MLX array.
//! The CPU tests pin this module to exl3_ref.py (testdata/exl3-ref, written by scripts/glm5next_exl3_ref_fixture.py).

const std = @import("std");
const ek = @import("../../exl3/kernels.zig");

pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
pub const top_k: u32 = 8;

// ── The pack's record (R/mimo/glm5next-exl3-pack-contract.md sec. 2: bank-v2, 9 segments, stride 9,474,048 B) ──

pub const Component = enum { gate_code, gate_rout, gate_rin, up_code, up_rout, up_rin, down_code, down_rout, down_rin };
pub const Segment = struct { off: u32, len: u32 };
pub const segments = blk: {
    const lens = [_]u32{
        hidden / 16 * (inter / 16) * 48 * 2, inter * 2, hidden * 2,
        hidden / 16 * (inter / 16) * 48 * 2, inter * 2, hidden * 2,
        inter / 16 * (hidden / 16) * 48 * 2, hidden * 2, inter * 2,
    };
    var out: [9]Segment = undefined;
    var at: u32 = 0;
    for (lens, 0..) |l, i| {
        out[i] = .{ .off = at, .len = l };
        at += l;
    }
    break :blk out;
};
pub const record_bytes: u32 = segments[8].off + segments[8].len;

pub fn segment(c: Component) Segment {
    return segments[@intFromEnum(c)];
}

// ── Synthetic inputs (the same streams as scripts/glm5next_exl3_ref_fixture.py) ──

fn unit(z: u64) f64 {
    return @as(f64, @floatFromInt(z >> 11)) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
}

/// A projection's code: the little-endian int16 view of splitmix64(seed) words.
pub fn fillCode(seed: u64, out: []i16) void {
    var s = seed;
    var i: usize = 0;
    while (i < out.len) : (i += 4) {
        const z = ek.splitmix64(&s);
        inline for (0..4) |j| {
            if (i + j < out.len) out[i + j] = @bitCast(@as(u16, @truncate(z >> (16 * j))));
        }
    }
}

/// suh / svh: f16 bits of sign (0.5 + u) scale.
pub fn fillScales(seed: u64, scale: f64, out: []u16) void {
    var s = seed;
    for (out) |*o| {
        const z = ek.splitmix64(&s);
        const sign: f64 = if (z & 1 != 0) -1.0 else 1.0;
        const v: f16 = @floatCast(sign * (0.5 + unit(z)) * scale);
        o.* = @bitCast(v);
    }
}

pub fn bf16Round(x: f32) f32 {
    const b: u32 = @bitCast(x);
    const r: u32 = ((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16) << 16;
    return @bitCast(r);
}

/// x: bf16 values of f32(2 u - 1).
pub fn fillX(seed: u64, out: []f32) void {
    var s = seed;
    for (out) |*o| o.* = bf16Round(@floatCast(2.0 * unit(ek.splitmix64(&s)) - 1.0));
}

/// A record image in the pack's layout from one seed (code seeds seed + 1, 4, 7; suh seed + 2, 5, 8 at 1.0; svh seed
/// + 3, 6, 9 at 0.01: the fixture's expert at seed 0x676c6d35).
pub fn synthRecord(seed: u64, rec: []u8) void {
    std.debug.assert(rec.len == record_bytes);
    const comps = [_]Component{ .gate_code, .up_code, .down_code };
    for (comps, 0..) |c, p| {
        const sg = segment(c);
        fillCode(seed + 1 + 3 * p, @alignCast(std.mem.bytesAsSlice(i16, rec[sg.off..][0..sg.len])));
        const rin = segment(@enumFromInt(@intFromEnum(c) + 2));
        const rout = segment(@enumFromInt(@intFromEnum(c) + 1));
        fillScales(seed + 2 + 3 * p, 1.0, @alignCast(std.mem.bytesAsSlice(u16, rec[rin.off..][0..rin.len])));
        fillScales(seed + 3 + 3 * p, 0.01, @alignCast(std.mem.bytesAsSlice(u16, rec[rout.off..][0..rout.len])));
    }
}

// ── Decode ──

pub const Codebook = enum { mul1, mcg };

/// exllamav3 codebook mcg (codebook.cuh L67-75): x *= 0xCBAC1FED; x = (x & 0x8FFF8FFF) ^ 0x3B603B60; the f16 sum of its
/// two halves (__hadd: exact in f64, one rounding).
pub fn mcgDecode(state: u32) u16 {
    const x = ((state *% 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60;
    const lo: f64 = @as(f16, @bitCast(@as(u16, @truncate(x))));
    const hi: f64 = @as(f16, @bitCast(@as(u16, @truncate(x >> 16))));
    const v: f16 = @floatCast(lo + hi);
    return @bitCast(v);
}

pub fn table(cb: Codebook, out: *[65536]u16) void {
    switch (cb) {
        .mul1 => ek.mul1Table(out),
        .mcg => for (out, 0..) |*o, s| {
            o.* = mcgDecode(@intCast(s));
        },
    }
}

/// W_hat f16 bits [in, out] of a projection's code [in / 16, out / 16, 48].
pub fn decodeW(code: []const i16, in: u32, out: u32, tbl: *const [65536]u16, w: []u16) void {
    ek.reconstruct(code, in / 16, out / 16, 3, tbl, w, null);
}

fn h16(b: u16) f64 {
    return @as(f16, @bitCast(b));
}

// ── The f64 forward (exl3_ref.py forward) ──

pub fn fwht128(v: []f64) void {
    var b: usize = 0;
    while (b < v.len) : (b += 128) {
        const x = v[b..][0..128];
        var h: usize = 1;
        while (h < 128) : (h <<= 1) {
            var i: usize = 0;
            while (i < 128) : (i += 2 * h) {
                for (i..i + h) |j| {
                    const a = x[j];
                    const c = x[j + h];
                    x[j] = a + c;
                    x[j + h] = a - c;
                }
            }
        }
        for (x) |*e| e.* /= @sqrt(128.0);
    }
}

/// y [out] = t128(t128(x suh) W_hat) svh, all f64.
pub fn forward(a: std.mem.Allocator, x: []const f64, w: []const u16, suh: []const u16, svh: []const u16, y: []f64) !void {
    const in = x.len;
    const out = y.len;
    const xh = try a.alloc(f64, in);
    defer a.free(xh);
    for (xh, x, suh) |*o, xv, s| o.* = xv * h16(s);
    fwht128(xh);
    @memset(y, 0);
    for (0..in) |i| {
        const xi = xh[i];
        const row = w[i * out ..][0..out];
        for (y, row) |*o, wb| o.* += xi * h16(wb);
    }
    fwht128(y);
    for (y, svh) |*o, s| o.* *= h16(s);
}

pub fn swiglu(g: f64, u: f64) f64 {
    const gc = @min(g, 10.0);
    return gc / (1.0 + @exp(-gc)) * std.math.clamp(u, -10.0, 10.0);
}

/// One expert's decoded projections (host), from a record image.
pub const Expert = struct {
    wg: []u16,
    wu: []u16,
    wd: []u16,
    rec: []const u8,

    pub fn init(a: std.mem.Allocator, rec: []const u8, tbl: *const [65536]u16) !Expert {
        const wg = try a.alloc(u16, hidden * inter);
        errdefer a.free(wg);
        const wu = try a.alloc(u16, hidden * inter);
        errdefer a.free(wu);
        const wd = try a.alloc(u16, hidden * inter);
        decodeW(codeOf(rec, .gate_code), hidden, inter, tbl, wg);
        decodeW(codeOf(rec, .up_code), hidden, inter, tbl, wu);
        decodeW(codeOf(rec, .down_code), inter, hidden, tbl, wd);
        return .{ .wg = wg, .wu = wu, .wd = wd, .rec = rec };
    }

    pub fn deinit(e: *Expert, a: std.mem.Allocator) void {
        a.free(e.wg);
        a.free(e.wu);
        a.free(e.wd);
    }

    /// h [inter] the clamped SwiGLU, y [hidden] the down output, f64.
    pub fn run(e: *const Expert, a: std.mem.Allocator, x: []const f64, h: []f64, y: []f64) !void {
        const g = try a.alloc(f64, inter);
        defer a.free(g);
        const u = try a.alloc(f64, inter);
        defer a.free(u);
        try forward(a, x, e.wg, halvesOf(e.rec, .gate_rin), halvesOf(e.rec, .gate_rout), g);
        try forward(a, x, e.wu, halvesOf(e.rec, .up_rin), halvesOf(e.rec, .up_rout), u);
        for (h, g, u) |*o, gv, uv| o.* = swiglu(gv, uv);
        try forward(a, h, e.wd, halvesOf(e.rec, .down_rin), halvesOf(e.rec, .down_rout), y);
    }
};

pub fn codeOf(rec: []const u8, c: Component) []const i16 {
    const s = segment(c);
    return @alignCast(std.mem.bytesAsSlice(i16, rec[s.off..][0..s.len]));
}
pub fn halvesOf(rec: []const u8, c: Component) []const u16 {
    const s = segment(c);
    return @alignCast(std.mem.bytesAsSlice(u16, rec[s.off..][0..s.len]));
}

/// max |got - ref| / rms(ref) (DeepSeek-V4.1's EXL3 f64 check, exl3/selfcheck.zig maxErrOverRowRms, one row).
pub fn maxErrOverRms(got: []const f64, ref: []const f64) f64 {
    var ss: f64 = 0;
    var mx: f64 = 0;
    for (ref, got) |r, v| {
        ss += r * r;
        mx = @max(mx, @abs(r - v));
    }
    return mx / (@sqrt(ss / @as(f64, @floatFromInt(ref.len))) + 1e-30);
}

/// DeepSeek-V4.1's per-stage tolerance (exl3/selfcheck.zig: recordTol(.., 1e-4)).
pub const tol_f32: f64 = 1e-4;

/// A bf16 output against the f64 truth: max over words of |got - ref| / (2^-8 |ref| + tol_f32 rms(ref)); <= 1 passes
/// (the store's rounding, at most half a bf16 ulp = 2^-8 relative, on top of the f32 tolerance).
pub fn bf16ErrRatio(got: []const f64, ref: []const f64) f64 {
    var ss: f64 = 0;
    for (ref) |r| ss += r * r;
    const rms = @sqrt(ss / @as(f64, @floatFromInt(ref.len)));
    var worst: f64 = 0;
    for (ref, got) |r, v| worst = @max(worst, @abs(r - v) / (@abs(r) / 256.0 + tol_f32 * rms + 1e-30));
    return worst;
}

// ── The seam chain's arithmetic per output word, in f32 (the order every form keeps) ──

const SCALE: f32 = @bitCast(@as(u32, 1035273459));

/// b3 in_rin / din_rin: t128 of (x * r) per 128-block in the 64-thread stage order, times SCALE, f32.
pub fn emuRotIn(x: []const f32, r: []const u16, out: []f32) void {
    for (out, x, r) |*o, xv, rv| o.* = xv * @as(f32, @floatCast(h16(rv)));
    var b: usize = 0;
    while (b < out.len) : (b += 128) {
        emuButterfly(out[b..][0..128]);
        for (out[b..][0..128]) |*e| e.* = e.* * SCALE;
    }
}

/// The b3 texts' butterflies: stage h, thread t: j = ((t - k) << 1) + k, k = t & (h - 1).
pub fn emuButterfly(B: *[128]f32) void {
    var h: u32 = 1;
    while (h < 128) : (h <<= 1) {
        for (0..64) |ti| {
            const t: u32 = @intCast(ti);
            const k = t & (h - 1);
            const j = ((t - k) << 1) + k;
            const a = B[j];
            const c = B[j + h];
            B[j] = a + c;
            B[j + h] = a - c;
        }
    }
}

/// A GEMV tile's column sums: `stock` the threadgroup red array (the seam's, gu_one's, down_pair's and the u8 GEMV's),
/// `registers` the fused forms' g5k3_colsum (g5k3_gu_fused, g5k3_dn_fused, g5k3_dn_pro).
pub const ColSum = enum { stock, registers };

pub fn colsumOf(kind: ColSum, acc: *const [32][8]f32, z: *[16]f32) void {
    switch (kind) {
        .stock => colsumStock(acc, z),
        .registers => colsumRegisters(acc, z),
    }
}

/// The stock GEMV (glm5n_exl3_gemv) of output tile tn: per lane l and weight m an fma chain over the k-tiles, then
/// red[c][l] = 0 + acc[4..7] (c = l >> 2) / 0 + acc[0..3] (c = l / 4 + 8), column c = 0 + red[c][0..31] in lane order.
pub fn emuGemvTile(xh: []const f32, w: []const u16, in: u32, out: u32, tn: u32, z: *[16]f32) void {
    emuGemvTileSum(xh, w, in, out, tn, .stock, z);
}

/// `emuGemvTile` with the column sums of `sum` (the per-lane fma chains are every form's).
pub fn emuGemvTileSum(xh: []const f32, w: []const u16, in: u32, out: u32, tn: u32, sum: ColSum, z: *[16]f32) void {
    var accs: [32][8]f32 = undefined;
    for (0..32) |li| {
        const l: u32 = @intCast(li);
        var acc: [8]f32 = @splat(0);
        for (0..in / 16) |tk| {
            for (0..8) |mi| {
                const m: u32 = @intCast(mi);
                const q = m & 3;
                const dr = 2 * (l & 3) + (if (q < 2) @as(u32, 9) else 3) - q;
                const col = (l >> 2) + (if (m < 4) @as(u32, 8) else 0);
                const wv: f32 = @floatCast(h16(w[(tk * 16 + dr) * out + tn * 16 + col]));
                acc[mi] = @mulAdd(f32, xh[tk * 16 + dr], wv, acc[mi]);
            }
        }
        accs[li] = acc;
    }
    colsumOf(sum, &accs, z);
}

/// The fused forms' column sums (header_g5k3.metal g5k3_colsum): column c = 0 + the four lanes' partials in lane order.
pub fn colsumRegisters(acc: *const [32][8]f32, z: *[16]f32) void {
    for (0..16) |c| {
        const b = (c & 7) * 4;
        var s: f32 = 0;
        for (0..4) |k| {
            const l = b + k;
            var hi: f32 = 0;
            hi += acc[l][4];
            hi += acc[l][5];
            hi += acc[l][6];
            hi += acc[l][7];
            var lo: f32 = 0;
            lo += acc[l][0];
            lo += acc[l][1];
            lo += acc[l][2];
            lo += acc[l][3];
            s += if (c < 8) hi else lo;
        }
        z[c] = s;
    }
}

/// The stock column sums from per-lane accumulators (the threadgroup red array), for the identity test.
pub fn colsumStock(acc: *const [32][8]f32, z: *[16]f32) void {
    var red: [16][32]f32 = @splat(@splat(0));
    for (0..32) |l| for (0..8) |m| {
        const c = (l >> 2) + (if (m < 4) @as(usize, 8) else 0);
        red[c][l] += acc[l][m];
    };
    for (0..16) |c| {
        var s: f32 = 0;
        for (0..32) |l| s += red[c][l];
        z[c] = s;
    }
}

/// The seam chain of one pair in f32 (std exp for gu_epi's): x [hidden] (bf16 values) -> h [inter] f32, and the down
/// output words of hidden blocks [blk_lo, blk_hi) (f32, before dpost's store rounding).
pub fn emuExpert(a: std.mem.Allocator, e: *const Expert, x: []const f32, h: []f32, y: []f32, blk_lo: u32, blk_hi: u32) !void {
    return emuExpertSums(a, e, x, h, y, blk_lo, blk_hi, .stock, .stock);
}

/// `emuExpert` with the gate / up GEMVs' column sums `gu` and the down GEMV's `dn`.
pub fn emuExpertSums(a: std.mem.Allocator, e: *const Expert, x: []const f32, h: []f32, y: []f32, blk_lo: u32, blk_hi: u32, gu: ColSum, dn: ColSum) !void {
    const xg = try a.alloc(f32, hidden);
    defer a.free(xg);
    const xu = try a.alloc(f32, hidden);
    defer a.free(xu);
    emuRotIn(x, halvesOf(e.rec, .gate_rin), xg);
    emuRotIn(x, halvesOf(e.rec, .up_rin), xu);
    const og = halvesOf(e.rec, .gate_rout);
    const ou = halvesOf(e.rec, .up_rout);
    var blk: u32 = 0;
    while (blk < inter / 128) : (blk += 1) {
        var bg: [128]f32 = undefined;
        var bu: [128]f32 = undefined;
        for (0..8) |t| {
            emuGemvTileSum(xg, e.wg, hidden, inter, blk * 8 + @as(u32, @intCast(t)), gu, bg[t * 16 ..][0..16]);
            emuGemvTileSum(xu, e.wu, hidden, inter, blk * 8 + @as(u32, @intCast(t)), gu, bu[t * 16 ..][0..16]);
        }
        emuButterfly(&bg);
        emuButterfly(&bu);
        for (0..128) |ee| {
            const col = blk * 128 + ee;
            const g = bg[ee] * SCALE;
            const gm = g * @as(f32, @floatCast(h16(og[col])));
            const gate = if (std.math.isNan(gm)) gm else (if (gm < 10.0) gm else 10.0);
            const u = bu[ee] * SCALE;
            const um = u * @as(f32, @floatCast(h16(ou[col])));
            const up0 = if (std.math.isNan(um)) um else (if (um > -10.0) um else -10.0);
            const up = if (std.math.isNan(up0)) up0 else (if (up0 < 10.0) up0 else 10.0);
            const yv = 1 / (1 + @exp(@abs(gate)));
            const sg = if (gate < 0) yv else 1 - yv;
            h[col] = (gate * sg) * up;
        }
    }
    const xd = try a.alloc(f32, inter);
    defer a.free(xd);
    emuRotIn(h, halvesOf(e.rec, .down_rin), xd);
    const od = halvesOf(e.rec, .down_rout);
    var hb = blk_lo;
    while (hb < blk_hi) : (hb += 1) {
        var bd: [128]f32 = undefined;
        for (0..8) |t| emuGemvTileSum(xd, e.wd, inter, hidden, hb * 8 + @as(u32, @intCast(t)), dn, bd[t * 16 ..][0..16]);
        emuButterfly(&bd);
        for (0..128) |c| y[hb * 128 + c] = (bd[c] * SCALE) * @as(f32, @floatCast(h16(od[hb * 128 + c])));
    }
}

// ── The union rule of the fused forms (header_g5k3.metal g5k3_members) ──

/// Pair e's members (itself and the next maxm - 1 pairs naming its slot, in pair order) when its rank among them is a
/// multiple of maxm; 0 otherwise.
pub fn members(ids: []const u32, e: usize, maxm: u32, mem: []u32) u32 {
    if (maxm == 1) {
        mem[0] = @intCast(e);
        return 1;
    }
    var rank: u32 = 0;
    for (ids[0..e]) |v| rank += @intFromBool(v == ids[e]);
    if (rank % maxm != 0) return 0;
    var n: u32 = 0;
    for (ids[e..], e..) |v, j| {
        if (n == maxm) break;
        if (v == ids[e]) {
            mem[n] = @intCast(j);
            n += 1;
        }
    }
    return n;
}

/// Case `it` of the column-sum identity's 2,000 (the CPU test's generator, the forms' parity test reuses it): random
/// accumulators with signed zeros, small integers (cancellation) and values over 2^-15 .. 2^16; every fifth case all -0.
pub fn colsumCase(s: *u64, it: usize, acc: *[32][8]f32) void {
    for (acc) |*row| for (row) |*v| {
        const z = ek.splitmix64(s);
        v.* = switch (z % 7) {
            0 => -0.0,
            1 => 0.0,
            2 => @as(f32, @floatFromInt(@as(i32, @intCast(z >> 40 & 0xff)) - 128)),
            else => @as(f32, @floatCast((unit(z) - 0.5) * std.math.pow(f64, 2.0, @as(f64, @floatFromInt(@as(i32, @intCast(z >> 56 & 31)) - 15))))),
        };
    };
    if (it % 5 == 0) for (acc) |*row| for (row) |*v| {
        v.* = -0.0;
    };
}

pub const colsum_cases: usize = 2000;

// ── The gate / up texts g5k3_guone and g5k3_guone_pair, per lane as written (the two-row rule's host parity) ──

/// g5k3_guone's decode words (g5k3_guone.metal:33-36): a 16-bit window -> the f16 weight bits.
pub fn decodeGuoneWords(window: u32) u16 {
    const rr = window *% 0x83DCD12D;
    const bs = (rr & 0x00FF00FF) +% ((rr >> 8) & 0x00FF00FF);
    return hfmaWeight(@truncate((bs *% 0x00010001 +% 0x64000000) >> 16));
}

/// g5k3_pair's decode words (g5k3_pair.metal:48-51, g5k3_guone_pair's): the high half of z.
pub fn decodePairWords(window: u32) u16 {
    const rr = window *% 0x83DCD12D;
    const q = (rr >> 8) & 0x00FF00FF;
    const z = (rr -% q *% 255) *% 0x00010001 +% 0x64000000;
    return hfmaWeight(@truncate(z >> 16));
}

/// fma(hs, 0x1EEE, 0xC931) in half: the exact value (hs = 1024 + a byte sum, exact in f64) rounded once.
fn hfmaWeight(hs: u16) u16 {
    const v: f16 = @floatCast(h16(hs) * h16(0x1EEE) + h16(0xC931));
    return @bitCast(v);
}

/// The 8 windows of lane `lane` in one 16 x 16 tile's 48 code words (the texts' wa / wb / lp funnel shifts).
pub fn laneWindows(tile: []const i16, lane: u32) [8]u32 {
    const wa: u32 = 2 * ((3 * lane + 2) >> 2);
    const wb: u32 = (wa + 46) % 48;
    const lp: u32 = 8 * ((lane + 1) & 3);
    const w32 = struct {
        fn at(t: []const i16, i: u32) u32 {
            return @as(u32, @as(u16, @bitCast(t[2 * i]))) | (@as(u32, @as(u16, @bitCast(t[2 * i + 1]))) << 16);
        }
    };
    const rd8 = w32.at(tile, wa >> 1);
    const rd9 = w32.at(tile, wb >> 1);
    const v0: u32 = if (lp == 0) rd8 else ((rd8 >> @intCast(lp)) | (rd9 << @intCast(32 - lp)));
    const v1: u32 = if (lp < 16) ((rd8 >> @intCast(lp + 16)) | (rd9 << @intCast(16 - lp))) else (rd9 >> @intCast(lp - 16));
    var out: [8]u32 = undefined;
    for (0..6) |m| out[m] = (v0 >> @intCast(3 * m)) & 0xffff;
    for (6..8) |m| out[m] = (v1 >> @intCast(3 * m - 16)) & 0xffff;
    return out;
}

pub const GuText = enum { guone, guone_pair };

/// A gate / up launch's code banks: per bank (0..2) and projection (0 gate, 1 up) the slots' codes [slots, IT, OT, 48].
pub const GuBanks = struct {
    code: [3][2][]const i16,

    fn slotCode(b: *const GuBanks, id: u32, proj: u32) []const i16 {
        const per = (hidden / 16) * (inter / 16) * 48;
        return b.code[id >> 24][proj][(id & 0xFFFFFF) * per ..][0..per];
    }
};

/// One launch of `text` over P = ids.len rows (grid (inter / 16 x 32, 2 P)), threadgroups at the output tiles `tiles`:
/// xg / xu [P, hidden] f32, out_g / out_u [P, inter] f32 (only the threadgroups' words are written), `writes` counts the
/// stores per (projection, row, tile index) and `decodes` the trellis decodes. Per threadgroup as the text runs it: the
/// row (and for guone_pair the rank rule, the odd rank's exit, the leader's mate), each lane's fma chains over the k-tiles
/// with one decode per weight per threadgroup, then the threadgroup column sums (red[c][l], in lane order) per row.
pub fn emuGuLaunch(text: GuText, xg: []const f32, xu: []const f32, ids: []const u32, banks: *const GuBanks, tiles: []const u32, out_g: []f32, out_u: []f32, writes: []u8, decodes: *u64) void {
    const nr: u32 = @intCast(ids.len);
    const it = hidden / 16;
    const ot = inter / 16;
    for (0..2 * nr) |yi| for (tiles, 0..) |tn, ti| {
        const y: u32 = @intCast(yi);
        const half_up = y / nr;
        const row = y - half_up * nr;
        const slot = ids[row];
        var rank: u32 = 0;
        var cnt: u32 = 0;
        var mate = row;
        if (text == .guone_pair) {
            for (0..nr) |ji| {
                const j: u32 = @intCast(ji);
                const same: u32 = @intFromBool(ids[j] == slot);
                rank += if (j < row) same else 0;
                cnt += same;
                mate = if (mate == row and j > row and same == 1) j else mate;
            }
            if (rank & 1 != 0) continue;
        }
        const two = text == .guone_pair and rank + 1 < cnt;
        const code = banks.slotCode(slot, half_up);
        const xh = if (half_up != 0) xu else xg;
        const out = if (half_up != 0) out_u else out_g;
        var acc: [32][8]f32 = @splat(@splat(0));
        var accb: [32][8]f32 = @splat(@splat(0));
        for (0..32) |li| {
            const lane: u32 = @intCast(li);
            var dr: [8]u32 = undefined;
            for (0..8) |mi| {
                const m: u32 = @intCast(mi);
                const q = m & 3;
                dr[mi] = 2 * (lane & 3) + (if (q < 2) @as(u32, 9) else 3) - q;
            }
            for (0..it) |tki| {
                const tk: u32 = @intCast(tki);
                const win = laneWindows(code[(tk * ot + tn) * 48 ..][0..48], lane);
                for (0..8) |m| {
                    const wbits = switch (text) {
                        .guone => decodeGuoneWords(win[m]),
                        .guone_pair => decodePairWords(win[m]),
                    };
                    decodes.* += 1;
                    const w: f32 = @floatCast(h16(wbits));
                    acc[li][m] = @mulAdd(f32, xh[row * hidden + tk * 16 + dr[m]], w, acc[li][m]);
                    if (two) accb[li][m] = @mulAdd(f32, xh[mate * hidden + tk * 16 + dr[m]], w, accb[li][m]);
                }
            }
        }
        var z: [16]f32 = undefined;
        colsumStock(&acc, &z);
        @memcpy(out[row * inter + tn * 16 ..][0..16], &z);
        writes[(half_up * nr + row) * tiles.len + ti] += 1;
        if (two) {
            colsumStock(&accb, &z);
            @memcpy(out[mate * inter + tn * 16 ..][0..16], &z);
            writes[(half_up * nr + mate) * tiles.len + ti] += 1;
        }
    };
}

/// The pairs the two-row rule shares: the leaders (even ranks with a partner) of a wave's ids.
pub fn pairLeaders(ids: []const u32) u32 {
    var n: u32 = 0;
    for (ids, 0..) |v, i| {
        var rank: u32 = 0;
        var cnt: u32 = 0;
        for (ids, 0..) |u, j| if (u == v) {
            cnt += 1;
            if (j < i) rank += 1;
        };
        if (rank & 1 == 0 and rank + 1 < cnt) n += 1;
    }
    return n;
}

// ── Tests ──

const testing = std.testing;

test "glm5_next exl3 ref: the record is the pack contract's (9 segments, 9,474,048 B, a whole number of pages)" {
    try testing.expectEqual(@as(u32, 9_474_048), record_bytes);
    try testing.expectEqual(@as(u32, 0), record_bytes % 4096);
    const want = [_][2]u32{ .{ 0, 3_145_728 }, .{ 3_145_728, 4_096 }, .{ 3_149_824, 8_192 }, .{ 3_158_016, 3_145_728 }, .{ 6_303_744, 4_096 }, .{ 6_307_840, 8_192 }, .{ 6_316_032, 3_145_728 }, .{ 9_461_760, 8_192 }, .{ 9_469_952, 4_096 } };
    for (segments, want) |s, w| {
        try testing.expectEqual(w[0], s.off);
        try testing.expectEqual(w[1], s.len);
    }
}

/// A little-endian f32 file's values as f64.
fn f32File(bytes: []const u8, out: []f64) void {
    std.debug.assert(bytes.len == 4 * out.len);
    for (out, 0..) |*o, i| o.* = @as(f32, @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)));
}

test "glm5_next exl3 ref: decode and forward equal exl3_ref.py on the fixture expert (W_hat sha256 per projection and codebook, h and y)" {
    const a = testing.allocator;
    const dir = "testdata/exl3-ref/";
    const meta = @embedFile(dir ++ "meta.json");
    const seed: u64 = 0x676C6D35;
    const rec = try a.alignedAlloc(u8, .@"16", record_bytes);
    defer a.free(rec);
    synthRecord(seed, rec);
    var x32: [hidden]f32 = undefined;
    fillX(seed + 10, &x32);
    var x: [hidden]f64 = undefined;
    for (&x, x32) |*o, v| o.* = v;
    const tbl = try a.create([65536]u16);
    defer a.destroy(tbl);
    for ([_]Codebook{ .mul1, .mcg }) |cb| {
        table(cb, tbl);
        var e = try Expert.init(a, rec, tbl);
        defer e.deinit(a);
        for ([_][]const u16{ e.wg, e.wu, e.wd }) |w| {
            var d: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(w), &d, .{});
            const hex = std.fmt.bytesToHex(d, .lower);
            try testing.expect(std.mem.indexOf(u8, meta, &hex) != null);
        }
        const h = try a.alloc(f64, inter);
        defer a.free(h);
        const y = try a.alloc(f64, hidden);
        defer a.free(y);
        try e.run(a, &x, h, y);
        var ry: [hidden]f64 = undefined;
        f32File(if (cb == .mul1) @embedFile(dir ++ "y_mul1.f32") else @embedFile(dir ++ "y_mcg.f32"), &ry);
        try testing.expect(maxErrOverRms(y, &ry) < 1e-6);
        if (cb == .mul1) {
            var rh: [inter]f64 = undefined;
            f32File(@embedFile(dir ++ "h_mul1.f32"), &rh);
            try testing.expect(maxErrOverRms(h, &rh) < 1e-6);
        }
    }
}

test "glm5_next exl3 ref: the fused forms' register column sums equal the stock threadgroup sums bit for bit (signed zeros, cancellation, random)" {
    var s: u64 = 7;
    var acc: [32][8]f32 = undefined;
    for (0..colsum_cases) |it| {
        colsumCase(&s, it, &acc);
        var p: [16]f32 = undefined;
        var q: [16]f32 = undefined;
        colsumStock(&acc, &p);
        colsumRegisters(&acc, &q);
        for (p, q) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
    }
}

test "glm5_next exl3 ref: the union rule covers every pair of a wave exactly once (P 1..64, slot repeats up to 9, hits and misses mixed, all one bank)" {
    var s: u64 = 11;
    var ids: [64]u32 = undefined;
    var mem: [4]u32 = undefined;
    for (1..65) |p| for (0..20) |pat| {
        for (ids[0..p], 0..) |*v, i| {
            const z = ek.splitmix64(&s);
            v.* = switch (pat % 4) {
                0 => @intCast(i), // all distinct
                1 => @as(u32, @intCast(z % 3)), // heavy repeats
                2 => (@as(u32, @intCast(z % 3)) << 24) | @as(u32, @intCast(z % 7)), // three banks
                else => (2 << 24) | @as(u32, @intCast(z % 10)), // a transient part only (0 hits)
            };
        }
        for ([_]u32{ 1, 4 }) |maxm| {
            var seen: [64]u8 = @splat(0);
            for (0..p) |e| {
                const n = members(ids[0..p], e, maxm, &mem);
                for (mem[0..n]) |m| {
                    try testing.expectEqual(ids[e], ids[m]);
                    seen[m] += 1;
                }
            }
            for (seen[0..p]) |c| try testing.expectEqual(@as(u8, 1), c);
        }
    };
}

test "glm5_next exl3 ref: the seam chain's f32 arithmetic meets DeepSeek-V4.1's 1e-4 against the f64 truth at GLM's shape (waves of M = 1..3 tokens: hits only, misses only, mixed)" {
    const a = testing.allocator;
    const tbl = try a.create([65536]u16);
    defer a.destroy(tbl);
    table(.mul1, tbl);
    // three records (two "resident", one "transient"); waves name them per token (M tokens, two pairs a token)
    var recs: [3][]align(16) u8 = undefined;
    var exps: [3]Expert = undefined;
    for (&recs, &exps, 0..) |*r, *e, i| {
        r.* = try a.alignedAlloc(u8, .@"16", record_bytes);
        synthRecord(0x5eed0000 + 17 * i, r.*);
        e.* = try Expert.init(a, r.*, tbl);
    }
    defer for (&recs, &exps) |r, *e| {
        e.deinit(a);
        a.free(r);
    };
    // M = 1: one token, two resident pairs (no miss); M = 2: two tokens, the transient record each (no hit); M = 3: three
    // tokens, mixed. A pair's arithmetic does not depend on its wave (the forms' bench checks that on the device).
    const waves = [_][]const u8{ &.{ 0, 1 }, &.{ 2, 2 }, &.{ 0, 2, 1 } };
    const tok_of = [_][]const u8{ &.{ 0, 0 }, &.{ 0, 1 }, &.{ 0, 1, 2 } };
    var worst_h: f64 = 0;
    var worst_y: f64 = 0;
    for (waves, tok_of) |wave, toks| for (wave, toks) |ri, t| {
        var x32: [hidden]f32 = undefined;
        fillX(0x7700 + @as(u64, t), &x32);
        var x: [hidden]f64 = undefined;
        for (&x, x32) |*o, v| o.* = v;
        const h = try a.alloc(f32, inter);
        defer a.free(h);
        const y = try a.alloc(f32, hidden);
        defer a.free(y);
        try emuExpert(a, &exps[ri], &x32, h, y, 0, 2);
        const rh = try a.alloc(f64, inter);
        defer a.free(rh);
        const ry = try a.alloc(f64, hidden);
        defer a.free(ry);
        try exps[ri].run(a, &x, rh, ry);
        var gh: [inter]f64 = undefined;
        for (&gh, h) |*o, v| o.* = v;
        var gy: [256]f64 = undefined;
        for (&gy, y[0..256]) |*o, v| o.* = v;
        worst_h = @max(worst_h, maxErrOverRms(&gh, rh));
        worst_y = @max(worst_y, maxErrOverRms(&gy, ry[0..256]));
    };
    try testing.expect(worst_h < tol_f32);
    try testing.expect(worst_y < tol_f32);
}

test "glm5_next exl3 ref: a bf16-rounded truth passes the bf16 ratio, an error of 2^-7 relative does not" {
    var r: [512]f64 = undefined;
    var b: [512]f64 = undefined;
    var s: u64 = 3;
    for (&r, &b) |*x, *y| {
        x.* = (unit(ek.splitmix64(&s)) - 0.5) * 7.0;
        y.* = bf16Round(@floatCast(x.*));
    }
    try testing.expect(bf16ErrRatio(&b, &r) <= 1.0);
    b[100] = r[100] * (1.0 + 1.0 / 128.0) + 1.0;
    try testing.expect(bf16ErrRatio(&b, &r) > 1.0);
}

test "glm5_next exl3 ref: guone_pair's decode words (g5k3_pair's) equal g5k3_guone's and the mul1 table for every 16-bit window" {
    const tbl = try testing.allocator.create([65536]u16);
    defer testing.allocator.destroy(tbl);
    table(.mul1, tbl);
    for (0..65536) |wi| {
        const w: u32 = @intCast(wi);
        try testing.expectEqual(tbl[wi], decodeGuoneWords(w));
        try testing.expectEqual(tbl[wi], decodePairWords(w));
    }
}

/// The guone_pair parity's banks: bank 0 two slots, bank 1 one, bank 2 two; gate and up codes per slot from fillCode.
const GpFixture = struct {
    store: [3][2][]i16,
    banks: GuBanks,

    fn init(a: std.mem.Allocator) !GpFixture {
        const per = (hidden / 16) * (inter / 16) * 48;
        const slots = [3]u32{ 2, 1, 2 };
        var f: GpFixture = undefined;
        var made: usize = 0;
        errdefer for (0..made) |k| a.free(f.store[k / 2][k % 2]);
        for (0..3) |b| for (0..2) |p| {
            f.store[b][p] = try a.alloc(i16, slots[b] * per);
            made += 1;
            for (0..slots[b]) |s| fillCode(0x67705f00 + 64 * b + 8 * p + s, f.store[b][p][s * per ..][0..per]);
            f.banks.code[b][p] = f.store[b][p];
        };
        return f;
    }
    fn deinit(f: *GpFixture, a: std.mem.Allocator) void {
        for (f.store) |bp| for (bp) |s| a.free(s);
    }
};

/// Rows' rotated inputs [P, hidden]: f32 values over 2^-13 .. 2^20 with signs (in_rin's output is any f32).
fn gpX(seed: u64, out: []f32) void {
    var s = seed;
    for (out) |*o| {
        const z = ek.splitmix64(&s);
        const mag: f64 = (0.5 + unit(z)) * std.math.pow(f64, 2.0, @as(f64, @floatFromInt(@as(i32, @intCast(z >> 59)) - 12)));
        o.* = @floatCast(if (z & 1 != 0) -mag else mag);
    }
}

/// guone_pair against guone on one wave: every word of both outputs bit for bit, each (projection, row, tile) written
/// exactly once, and the trellis decodes = guone's x (P - leaders) / P.
fn gpCase(a: std.mem.Allocator, fx: *const GpFixture, ids: []const u32, tiles: []const u32, seed: u64) !void {
    const p = ids.len;
    const xg = try a.alloc(f32, p * hidden);
    defer a.free(xg);
    const xu = try a.alloc(f32, p * hidden);
    defer a.free(xu);
    gpX(seed, xg);
    gpX(seed ^ 0x7570, xu);
    var outs: [2][2][]f32 = undefined; // [text][gate, up]
    var writes: [2][]u8 = undefined;
    var decodes: [2]u64 = .{ 0, 0 };
    for (0..2) |t| {
        for (0..2) |k| {
            outs[t][k] = try a.alloc(f32, p * inter);
            @memset(outs[t][k], @bitCast(@as(u32, 0x7FBADBAD)));
        }
        writes[t] = try a.alloc(u8, 2 * p * tiles.len);
        @memset(writes[t], 0);
    }
    defer for (0..2) |t| {
        for (0..2) |k| a.free(outs[t][k]);
        a.free(writes[t]);
    };
    for ([_]GuText{ .guone, .guone_pair }, 0..) |text, t| emuGuLaunch(text, xg, xu, ids, &fx.banks, tiles, outs[t][0], outs[t][1], writes[t], &decodes[t]);
    for (0..2) |t| for (writes[t]) |c| try testing.expectEqual(@as(u8, 1), c);
    for (0..2) |k| for (0..p) |r| for (tiles) |tn| for (0..16) |c| {
        const i = r * inter + tn * 16 + c;
        try testing.expectEqual(@as(u32, @bitCast(outs[0][k][i])), @as(u32, @bitCast(outs[1][k][i])));
        try testing.expect(@as(u32, @bitCast(outs[1][k][i])) != 0x7FBADBAD);
    };
    const leaders = pairLeaders(ids);
    try testing.expectEqual(decodes[0] / p * (p - leaders), decodes[1]);
}

test "glm5_next exl3 ref: g5k3_guone as written equals the seam GEMV over the decoded W_hat (the windows' layout)" {
    const a = testing.allocator;
    var fx = try GpFixture.init(a);
    defer fx.deinit(a);
    const tbl = try a.create([65536]u16);
    defer a.destroy(tbl);
    table(.mul1, tbl);
    const ids = [_]u32{ (2 << 24) | 1, 1 << 24 };
    const tiles = [_]u32{ 0, 5, 64, 127 };
    const xg = try a.alloc(f32, ids.len * hidden);
    defer a.free(xg);
    const xu = try a.alloc(f32, ids.len * hidden);
    defer a.free(xu);
    gpX(0x616e63, xg);
    gpX(0x616e64, xu);
    const og = try a.alloc(f32, ids.len * inter);
    defer a.free(og);
    const ou = try a.alloc(f32, ids.len * inter);
    defer a.free(ou);
    var writes: [2 * ids.len * tiles.len]u8 = @splat(0);
    var decodes: u64 = 0;
    emuGuLaunch(.guone, xg, xu, &ids, &fx.banks, &tiles, og, ou, &writes, &decodes);
    const w = try a.alloc(u16, hidden * inter);
    defer a.free(w);
    for (ids, 0..) |id, r| for ([_][]const f32{ xg, xu }, [_][]const f32{ og, ou }, 0..) |x, o, proj| {
        decodeW(fx.banks.slotCode(id, @intCast(proj)), hidden, inter, tbl, w);
        for (tiles) |tn| {
            var z: [16]f32 = undefined;
            emuGemvTile(x[r * hidden ..][0..hidden], w, hidden, inter, tn, &z);
            for (z, o[r * inter + tn * 16 ..][0..16]) |e, g| try testing.expectEqual(@as(u32, @bitCast(e)), @as(u32, @bitCast(g)));
        }
    };
}

test "glm5_next exl3 ref: g5k3_guone_pair equals g5k3_guone bit for bit at Flash's gate / up shape (P 1..4, with and without slot sharing, three banks), one decode per weight for two rows of a slot" {
    const a = testing.allocator;
    var fx = try GpFixture.init(a);
    defer fx.deinit(a);
    const A: u32 = 0;
    const B: u32 = 1;
    const C: u32 = 1 << 24;
    const D: u32 = (2 << 24) | 1;
    const E: u32 = 2 << 24;
    const waves = [_][]const u32{
        &.{A},
        &.{ A, B },          &.{ D, D },
        &.{ A, C, E },       &.{ C, C, C }, &.{ A, B, A },
        &.{ A, C, D, E },    &.{ A, A, A, A }, &.{ D, B, D, B }, &.{ E, E, E, A }, &.{ B, A, C, A },
    };
    const tiles = [_]u32{ 0, 1, 7, 8, 63, 64, 120, 127 };
    var shared: u32 = 0;
    for (waves, 0..) |ids, i| {
        try gpCase(a, &fx, ids, &tiles, 0x6770_0000 + i);
        shared += pairLeaders(ids);
    }
    try testing.expectEqual(@as(u32, 9), shared);
    // one wave over every output tile of both projections (the whole [P, inter] outputs)
    var all: [inter / 16]u32 = undefined;
    for (&all, 0..) |*t, i| t.* = @intCast(i);
    try gpCase(a, &fx, &.{ B, E, B, C }, &all, 0x67706675);
}
