//! CPU parity of the glm5_next exl3 decode GEMV (`kernels/glm5_next_exl3/glm5n_exl3_gemv.metal`: DeepSeek-V4.1's banked
//! mul1h text at GLM's shapes) against R/exl3/exl3_ref.py, on random K = 3 codes at GLM-5.3-Flash's expert shapes. No
//! device: the text's arithmetic emulated on the host, as written (each lane's two words and funnel-shifted windows, the
//! mul1h decode in half with one rounding, its f32 accumulation over the k-tiles, the 32-lane reduction in order).
//!   - W_hat: DeepSeek's host decoder (`exl3/kernels.zig reconstruct`, exl3_ref.reconstruct transliterated) over the
//!     seeded codes equals exl3_ref.py's W_hat bit for bit (sha256 pinned from scripts/glm5next_exl3_parity_golden.py);
//!   - the text's decode: every weight each lane decodes, at the (row, column) the text accumulates it into, equals W_hat
//!     bit for bit (DeepSeek's `decode_table` check, here over every weight of both shapes on the host);
//!   - the text's GEMV of random rows against the float64 product xh @ W_hat: within DeepSeek's f64 tolerance class
//!     (f32 accumulation, relative error at most 2^-16 of the row's scale).
//! The mcg codebook (an mcg pack's texts: glm5n_exl3_gemv_mcg, g5k3_guone_mcg, g5k3_pair_mcg, the DIG-X header's mcg
//! dig_cb; the kernel bench's mcg words), against the kernel bench's reference (exl3_forms_ref.zig `mcgDecode` / `table(.mcg)`,
//! pinned to exl3_ref.py's W_hat on the bench's fixture expert):
//!   - the words, emulated as written, for every 16-bit window: the GEMV words and the DIG-X dig_cb equal the bench's
//!     reference decode bit for bit;
//!   - the text's decode of every weight of the fixture expert's three projections, at the (row, column) the text
//!     accumulates it into, equals the bench's W_hat (whose sha256 is exl3_ref.py's) bit for bit;
//!   - the text's GEMV over random rows against the float64 product, the mul1 tolerance class;
//!   - the mcg texts are the mul1 texts with only the decode words replaced (texts_mcg below).

const std = @import("std");
const xk = @import("../../exl3/kernels.zig");
const kernels = @import("exl3_kernels.zig");
const fref = @import("exl3_forms_ref.zig");

const testing = std.testing;

/// The goldens (scripts/glm5next_exl3_parity_golden.py over R/exl3/exl3_ref.py, numpy 2.5.4; build/g22-exl3/logs/parity-golden.log).
const Plane = struct { in: u32, out: u32, seed: u64, code_sha256: []const u8, w_hat_sha256: []const u8 };
const planes = [_]Plane{
    .{ .in = 4096, .out = 2048, .seed = 0x474c4d35e3010001, .code_sha256 = "334910c3086f12bd760d4d0cd974f2be78ef4d544c9608c8ea94b391c7aa976e", .w_hat_sha256 = "f22a9f4828c4d119e33ca76201e578d5d0759a6de3f312be7300dd0aec934ba5" },
    .{ .in = 2048, .out = 4096, .seed = 0x474c4d35e3010002, .code_sha256 = "8a2ae8214525ae1b7d8fa2b6b9c288387309fd96a99e3ac1fd1bae7d33549b41", .w_hat_sha256 = "e2d4731fcb4a17a5a59dab8e99195d51215c1b884d158106b385d43dbc3f4265" },
};

/// The text's decode of one window (dsv41_exl3_mul1h_k3: rr = window * 0x83DCD12D, the two byte-pair sums folded into
/// 0x6400 + bytesum's half bits, then fma(h, 0x1EEE, 0xC931) in half: one rounding of an exact f64 value).
fn textDecode(window: u32) f32 {
    const rr: u32 = window *% 0x83DCD12D;
    const bs: u32 = (rr & 0x00FF00FF) +% ((rr >> 8) & 0x00FF00FF);
    const hs: f16 = @bitCast(@as(u16, @truncate((bs *% 0x00010001 +% 0x64000000) >> 16)));
    const k_inv: f64 = @as(f16, @bitCast(@as(u16, 0x1EEE)));
    const k_bias: f64 = @as(f16, @bitCast(@as(u16, 0xC931)));
    const h: f16 = @floatCast(@as(f64, hs) * k_inv + k_bias);
    return h;
}

/// The mcg words of the kernel bench (g5k3_gemv_mcg; glm5n_exl3_gemv_mcg, g5k3_guone_mcg, g5k3_pair_mcg): rr = ((window *
/// 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60; hh = as_type<half2>(rr) (x the low halfword); float(hh.x + hh.y), the half sum
/// rounded once (IEEE half addition: the exact f64 sum rounded to nearest even).
fn textDecodeMcg(window: u32) f32 {
    const rr: u32 = ((window *% 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60;
    const hx: f64 = @as(f16, @bitCast(@as(u16, @truncate(rr))));
    const hy: f64 = @as(f16, @bitCast(@as(u16, @truncate(rr >> 16))));
    const h: f16 = @floatCast(hx + hy);
    return h;
}

/// The mcg header's dig_cb (header_glm5n_exl3_dig_mcg.metal): x = ((w * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60; the half
/// hh.x + hh.y returned (the GEMM's B stage stores it, f16 as is).
fn digCbMcg(w: u32) u16 {
    const x: u32 = ((w *% 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60;
    const hx: f64 = @as(f16, @bitCast(@as(u16, @truncate(x))));
    const hy: f64 = @as(f16, @bitCast(@as(u16, @truncate(x >> 16))));
    const h: f16 = @floatCast(hx + hy);
    return @bitCast(h);
}

/// Lane `lane`'s eight windows of the tile at `tile` (48 halfwords): wa / wb / lp as the text computes them.
fn laneWindows(tile: []const i16, lane: u32, out: *[8]u32) void {
    const wa: u32 = 2 * ((3 * lane + 2) >> 2);
    const wb: u32 = (wa + 46) % 48;
    const w32 = struct {
        fn at(t: []const i16, i: u32) u32 {
            return @as(u32, @as(u16, @bitCast(t[2 * i]))) | (@as(u32, @as(u16, @bitCast(t[2 * i + 1]))) << 16);
        }
    }.at;
    const rd8: u64 = w32(tile, wa >> 1);
    const rd9: u64 = w32(tile, wb >> 1);
    const lp: u6 = @intCast(8 * ((lane + 1) & 3));
    const v: u64 = ((rd9 << 32) | rd8) >> lp;
    for (out, 0..) |*w, m| w.* = @truncate((v >> @intCast(3 * m)) & 0xffff);
}

/// The text's row (within the 16-row k-tile) and column (within the 16-column tile) of lane `lane`'s window m.
fn laneRow(lane: u32, m: u32) u32 {
    const q = m & 3;
    return 2 * (lane & 3) + (if (q < 2) @as(u32, 9) else 3) - q;
}

fn laneCol(lane: u32, m: u32) u32 {
    return (lane >> 2) + (if (m < 4) @as(u32, 8) else 0);
}

/// The text's GEMV of one row xh [in] against `code` [in/16, out/16, 48]: z [out] in its accumulation order.
fn textGemv(code: []const i16, in: u32, out: u32, xh: []const f32, z: []f32) void {
    textGemvWith(textDecode, code, in, out, xh, z);
}

fn textGemvWith(comptime decode: fn (u32) f32, code: []const i16, in: u32, out: u32, xh: []const f32, z: []f32) void {
    const it = in / 16;
    const ot = out / 16;
    var win: [8]u32 = undefined;
    for (0..ot) |tn_| {
        const tn: u32 = @intCast(tn_);
        var red: [16][32]f32 = @splat(@splat(0));
        for (0..32) |l_| {
            const lane: u32 = @intCast(l_);
            var acc: [8]f32 = @splat(0);
            for (0..it) |tk_| {
                const tk: u32 = @intCast(tk_);
                const tile = code[(@as(usize, tk) * ot + tn) * 48 ..][0..48];
                laneWindows(tile, lane, &win);
                for (0..8) |m| acc[m] = @mulAdd(f32, xh[tk * 16 + laneRow(lane, @intCast(m))], decode(win[m]), acc[m]);
            }
            for (0..8) |m| red[laneCol(lane, @intCast(m))][lane] += acc[m];
        }
        for (0..16) |c| {
            var s: f32 = 0;
            for (0..32) |l| s += red[c][l];
            z[tn * 16 + c] = s;
        }
    }
}

fn hexOf(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

test "glm5_next exl3 parity: W_hat of random K=3 mul1 codes at GLM's shapes is exl3_ref.py's, and the GEMV text decodes every weight to it bit for bit" {
    const a = testing.allocator;
    const table = try a.create([65536]u16);
    defer a.destroy(table);
    xk.mul1Table(table);
    for (planes) |p| {
        const g: xk.GoldenPlane = .{ .in_dim = p.in, .out_dim = p.out, .seed = p.seed, .code_sha256 = undefined, .w_hat_sha256 = undefined, .onehot_rows = 0, .onehot_rows_sha256 = undefined };
        try testing.expect(p.in == kernels.hidden or p.in == kernels.inter);
        var hp = try xk.HostPlane.init(a, &g, table);
        defer hp.deinit(a);
        try testing.expectEqualStrings(p.code_sha256, &hexOf(std.mem.sliceAsBytes(hp.code)));
        try testing.expectEqualStrings(p.w_hat_sha256, &hexOf(std.mem.sliceAsBytes(hp.w)));
        // Every weight the text decodes, at the (row, column) it accumulates into.
        const it = p.in / 16;
        const ot = p.out / 16;
        var win: [8]u32 = undefined;
        var bad: u64 = 0;
        var n: u64 = 0;
        for (0..it) |tk| for (0..ot) |tn| {
            const tile = hp.code[(tk * ot + tn) * 48 ..][0..48];
            for (0..32) |l| {
                laneWindows(tile, @intCast(l), &win);
                for (0..8) |m| {
                    const row = tk * 16 + laneRow(@intCast(l), @intCast(m));
                    const col = tn * 16 + laneCol(@intCast(l), @intCast(m));
                    const want: f32 = @as(f16, @bitCast(hp.w[row * p.out + col]));
                    bad += @intFromBool(@as(u32, @bitCast(textDecode(win[m]))) != @as(u32, @bitCast(want)));
                    n += 1;
                }
            }
        };
        try testing.expectEqual(@as(u64, p.in) * p.out, n);
        try testing.expectEqual(@as(u64, 0), bad);
    }
}

test "glm5_next exl3 parity: the GEMV text over random rows at GLM's shapes against the float64 product (f32 accumulation class)" {
    const a = testing.allocator;
    const table = try a.create([65536]u16);
    defer a.destroy(table);
    xk.mul1Table(table);
    var prng = std.Random.DefaultPrng.init(0x474c4d35);
    const r = prng.random();
    for (planes) |p| {
        const g: xk.GoldenPlane = .{ .in_dim = p.in, .out_dim = p.out, .seed = p.seed, .code_sha256 = undefined, .w_hat_sha256 = undefined, .onehot_rows = 0, .onehot_rows_sha256 = undefined };
        var hp = try xk.HostPlane.init(a, &g, table);
        defer hp.deinit(a);
        const xh = try a.alloc(f32, p.in);
        defer a.free(xh);
        const z = try a.alloc(f32, p.out);
        defer a.free(z);
        const ref = try a.alloc(f64, p.out);
        defer a.free(ref);
        for (0..2) |_| {
            for (xh) |*v| v.* = r.floatNorm(f32);
            textGemv(hp.code, p.in, p.out, xh, z);
            @memset(ref, 0);
            for (0..p.in) |i| {
                const xi: f64 = xh[i];
                const row = hp.w[i * p.out ..][0..p.out];
                for (ref, row) |*o, w| o.* += xi * @as(f64, @as(f16, @bitCast(w)));
            }
            var ss: f64 = 0;
            for (ref) |v| ss += v * v;
            const scale = @sqrt(ss / @as(f64, @floatFromInt(p.out)));
            var worst: f64 = 0;
            for (z, ref) |got, want| worst = @max(worst, @abs(@as(f64, got) - want) / scale);
            try testing.expect(worst <= 1.0 / 65536.0);
        }
    }
}

// ── The mcg codebook ──

test "glm5_next exl3 parity mcg: the mcg words (GEMV / guone / pair) and the DIG-X header's mcg dig_cb equal the kernel bench's reference decode for every 16-bit window" {
    const tbl = try testing.allocator.create([65536]u16);
    defer testing.allocator.destroy(tbl);
    fref.table(.mcg, tbl);
    for (0..65536) |w| {
        const want: f32 = @as(f16, @bitCast(tbl[w]));
        try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(textDecodeMcg(@intCast(w)))));
        try testing.expectEqual(tbl[w], digCbMcg(@intCast(w)));
        try testing.expectEqual(fref.mcgDecode(@intCast(w)), tbl[w]);
    }
}

test "glm5_next exl3 parity mcg: the text decodes every weight of the bench's fixture expert (gate, up, down) to its mcg W_hat (exl3_ref.py's sha256) bit for bit" {
    const a = testing.allocator;
    const meta = @embedFile("testdata/exl3-ref/meta.json");
    const rec = try a.alignedAlloc(u8, .@"16", fref.record_bytes);
    defer a.free(rec);
    fref.synthRecord(0x676C6D35, rec);
    const tbl = try a.create([65536]u16);
    defer a.destroy(tbl);
    fref.table(.mcg, tbl);
    const Proj = struct { c: fref.Component, in: u32, out: u32 };
    for ([_]Proj{ .{ .c = .gate_code, .in = fref.hidden, .out = fref.inter }, .{ .c = .up_code, .in = fref.hidden, .out = fref.inter }, .{ .c = .down_code, .in = fref.inter, .out = fref.hidden } }) |p| {
        const sg = fref.segment(p.c);
        const code: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, rec[sg.off..][0..sg.len]));
        const w = try a.alloc(u16, @as(usize, p.in) * p.out);
        defer a.free(w);
        fref.decodeW(code, p.in, p.out, tbl, w);
        try testing.expect(std.mem.indexOf(u8, meta, &hexOf(std.mem.sliceAsBytes(w))) != null);
        const it = p.in / 16;
        const ot = p.out / 16;
        var win: [8]u32 = undefined;
        var bad: u64 = 0;
        var n: u64 = 0;
        for (0..it) |tk| for (0..ot) |tn| {
            const tile = code[(tk * ot + tn) * 48 ..][0..48];
            for (0..32) |l| {
                laneWindows(tile, @intCast(l), &win);
                for (0..8) |m| {
                    const row = tk * 16 + laneRow(@intCast(l), @intCast(m));
                    const col = tn * 16 + laneCol(@intCast(l), @intCast(m));
                    const want: f32 = @as(f16, @bitCast(w[row * p.out + col]));
                    bad += @intFromBool(@as(u32, @bitCast(textDecodeMcg(win[m]))) != @as(u32, @bitCast(want)));
                    n += 1;
                }
            }
        };
        try testing.expectEqual(@as(u64, p.in) * p.out, n);
        try testing.expectEqual(@as(u64, 0), bad);
    }
}

test "glm5_next exl3 parity mcg: the mcg GEMV text over random rows of the fixture expert against the float64 product (the mul1 tolerance class)" {
    const a = testing.allocator;
    const rec = try a.alignedAlloc(u8, .@"16", fref.record_bytes);
    defer a.free(rec);
    fref.synthRecord(0x676C6D35, rec);
    const tbl = try a.create([65536]u16);
    defer a.destroy(tbl);
    fref.table(.mcg, tbl);
    var prng = std.Random.DefaultPrng.init(0x6d6367);
    const r = prng.random();
    const Proj = struct { c: fref.Component, in: u32, out: u32 };
    for ([_]Proj{ .{ .c = .gate_code, .in = fref.hidden, .out = fref.inter }, .{ .c = .down_code, .in = fref.inter, .out = fref.hidden } }) |p| {
        const sg = fref.segment(p.c);
        const code: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, rec[sg.off..][0..sg.len]));
        const w = try a.alloc(u16, @as(usize, p.in) * p.out);
        defer a.free(w);
        fref.decodeW(code, p.in, p.out, tbl, w);
        const xh = try a.alloc(f32, p.in);
        defer a.free(xh);
        const z = try a.alloc(f32, p.out);
        defer a.free(z);
        const want = try a.alloc(f64, p.out);
        defer a.free(want);
        for (xh) |*v| v.* = r.floatNorm(f32);
        textGemvWith(textDecodeMcg, code, p.in, p.out, xh, z);
        @memset(want, 0);
        for (0..p.in) |i| {
            const xi: f64 = xh[i];
            for (want, w[i * p.out ..][0..p.out]) |*o, wv| o.* += xi * @as(f64, @as(f16, @bitCast(wv)));
        }
        var ss: f64 = 0;
        for (want) |v| ss += v * v;
        const scale = @sqrt(ss / @as(f64, @floatFromInt(p.out)));
        var worst: f64 = 0;
        for (z, want) |got, wv| worst = @max(worst, @abs(@as(f64, got) - wv) / scale);
        try testing.expect(worst <= 1.0 / 65536.0);
    }
}

/// The mul1 texts' decode words and the mcg words that replace them (scripts/glm5next_exl3_kernels_gen.py / _forms_gen.py).
const mul1_words = "                uint rr = window * 0x83DCD12Du;\n" ++
    "                uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);\n" ++
    "                half hs = as_type<half>(ushort((bs * 0x00010001u + 0x64000000u) >> 16u));\n" ++
    "                float w = float(fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));\n";
const mcg_words = "                uint rr = ((window * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;\n" ++
    "                half2 hh = as_type<half2>(rr);\n" ++
    "                float w = float(hh.x + hh.y);\n";
const pair_mul1_words = "                    uint rr = window * 0x83DCD12Du;\n" ++
    "                    uint q = (rr >> 8u) & 0x00FF00FFu;\n" ++
    "                    uint z = (rr - q * 255u) * 0x00010001u + 0x64000000u;\n" ++
    "                    float w = float(fma(as_type<half2>(z).y, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u))));\n";
const pair_mcg_words = "                    uint rr = ((window * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;\n" ++
    "                    half2 hh = as_type<half2>(rr);\n" ++
    "                    float w = float(hh.x + hh.y);\n";
const dig_mul1 = "  const uint x = w * 2212286765u;\n" ++
    "  const uint t = (x & 0x00ff00ffu) + ((x >> 8u) & 0x00ff00ffu);\n" ++
    "  const half hs = as_type<half>(ushort((t * 0x00010001u + 0x64000000u) >> 16u));\n" ++
    "  return metal::fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u)));\n";
const dig_mcg = "  const uint x = ((w * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;\n" ++
    "  const half2 hh = as_type<half2>(x);\n" ++
    "  return hh.x + hh.y;\n";

/// A text without its comment lines.
fn stripComments(al: std.mem.Allocator, t: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(al);
    var it = std.mem.splitScalar(u8, t, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, std.mem.trimStart(u8, line, " "), "//")) continue;
        try out.appendSlice(al, line);
        try out.append(al, '\n');
    }
    return out.toOwnedSlice(al);
}

/// `mcg` minus its comment lines equals `mul1` minus its comment lines with `count` x `from` replaced by `to`.
fn onlyWordsDiffer(a: std.mem.Allocator, mul1: []const u8, mcg: []const u8, from: []const u8, to: []const u8, count: usize) !void {
    try testing.expectEqual(count, std.mem.count(u8, mul1, from));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, mcg, from));
    try testing.expectEqual(count, std.mem.count(u8, mcg, to));
    const want = try std.mem.replaceOwned(u8, a, mul1, from, to);
    defer a.free(want);
    const strip = stripComments;
    const x = try strip(a, want);
    defer a.free(x);
    const y = try strip(a, mcg);
    defer a.free(y);
    try testing.expectEqualStrings(x, y);
}

test "glm5_next exl3 parity mcg: texts_mcg: each mcg text is its mul1 text with only the decode words replaced (the GEMV, guone, pair, the DIG-X header); the GEMV's mcg words are the kernel bench's g5k3_gemv_mcg" {
    const a = testing.allocator;
    const xf = @import("exl3_forms.zig");
    try onlyWordsDiffer(a, kernels.textOf(.mul1, .gemv), kernels.gemv_mcg, mul1_words, mcg_words, 2);
    try onlyWordsDiffer(a, xf.textOf(.g5k3_guone), xf.textOf(.g5k3_guone_mcg), mul1_words, mcg_words, 2);
    try onlyWordsDiffer(a, xf.textOf(.g5k3_pair), xf.textOf(.g5k3_pair_mcg), pair_mul1_words, pair_mcg_words, 4);
    try onlyWordsDiffer(a, kernels.header_gemm, kernels.header_gemm_mcg, dig_mul1, dig_mcg, 1);
    // The kernel bench's mcg text (g5k3_gemv_mcg) and the served mcg GEMV are one text below their heads.
    {
        const x = try stripComments(a, xf.textOf(.g5k3_gemv_mcg));
        defer a.free(x);
        const y = try stripComments(a, kernels.gemv_mcg);
        defer a.free(y);
        try testing.expectEqualStrings(x, y);
    }
    // The mcg route: the texts and headers each kernel binds, and their MLX names.
    try testing.expectEqual(kernels.gemv_mcg.ptr, kernels.textOf(.mcg, .gemv).ptr);
    try testing.expectEqual(kernels.header_gemm_mcg.ptr, kernels.headerOf(.mcg, .gemm_gu).ptr);
    try testing.expectEqual(kernels.header_gemm_mcg.ptr, kernels.headerOf(.mcg, .gemm_dn).ptr);
    try testing.expectEqual(kernels.header_gemm.ptr, kernels.headerOf(.mul1, .gemm_gu).ptr);
    for (kernels.texts, std.enums.values(kernels.Kernel)) |t, k| try testing.expectEqual(t.ptr, kernels.textOf(.mul1, k).ptr);
    inline for (comptime std.enums.values(kernels.Kernel)) |k| {
        if (k != .gemv) try testing.expectEqual(kernels.textOf(.mul1, k).ptr, kernels.textOf(.mcg, k).ptr);
        if (!kernels.decodesInHeader(k)) try testing.expectEqual(kernels.headerOf(.mul1, k).ptr, kernels.headerOf(.mcg, k).ptr);
        const n1 = comptime kernels.kernelName(.mul1, k);
        const n2 = comptime kernels.kernelName(.mcg, k);
        try testing.expectEqual(k == .gemv or kernels.decodesInHeader(k), !std.mem.eql(u8, n1, n2));
    }
    try testing.expectEqualStrings("glm5n_exl3_gemm_gu_mcg", comptime kernels.kernelName(.mcg, .gemm_gu));
    // Only the mul1 texts carry the mul1 constant; no mcg text does.
    for ([_][]const u8{ kernels.gemv_mcg, xf.textOf(.g5k3_guone_mcg), xf.textOf(.g5k3_pair_mcg) }) |t| {
        try testing.expect(std.mem.indexOf(u8, t, "0x83DCD12D") == null);
    }
    try testing.expect(std.mem.indexOf(u8, kernels.header_gemm_mcg, "2212286765u") == null);
    try kernels.checkPins();
    try xf.checkPins();
}

test "glm5_next exl3 parity mcg: the codebook route: an mcg pack installs the seam / guone / pair forms only (the u8, mcg-price, fused and prologue forms have no mcg text)" {
    const xf = @import("exl3_forms.zig");
    const m = xf.textsFor(.mcg, .{ .gu = .guone, .dn = .pair }).?;
    try testing.expectEqual(xf.Kernel.g5k3_guone_mcg, m.guone);
    try testing.expectEqual(xf.Kernel.g5k3_pair_mcg, m.pair);
    try testing.expect(xf.textsFor(.mcg, .{}) != null);
    const u = xf.textsFor(.mul1, .{ .gu = .guone, .dn = .pair }).?;
    try testing.expectEqual(xf.Kernel.g5k3_guone, u.guone);
    try testing.expectEqual(xf.Kernel.g5k3_pair, u.pair);
    for ([_]xf.Forms{ .{ .gu = .u8 }, .{ .gu = .mcg, .dn = .mcg }, .{ .gu = .fused16, .dn = .fused1 }, .{ .gu = .guone, .dn = .pro1 }, .{ .gu = .fused8, .dn = .pair, .maxm = 4 } }) |f| {
        try testing.expect(xf.textsFor(.mcg, f) == null);
        try testing.expect(xf.textsFor(.mul1, f) != null);
    }
    try testing.expectEqual(@as(u64, 0xCBAC1FED), kernels.Codebook.mcg.multiplier());
    try testing.expectEqual(@as(u64, 0x83DCD12D), kernels.Codebook.mul1.multiplier());
    try testing.expectEqual(kernels.Codebook.mcg, kernels.Codebook.ofName("mcg").?);
    try testing.expect(kernels.Codebook.ofName("3inst") == null);
}
