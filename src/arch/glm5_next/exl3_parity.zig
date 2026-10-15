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

const std = @import("std");
const xk = @import("../../exl3/kernels.zig");
const kernels = @import("exl3_kernels.zig");

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
                for (0..8) |m| acc[m] = @mulAdd(f32, xh[tk * 16 + laneRow(lane, @intCast(m))], textDecode(win[m]), acc[m]);
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
