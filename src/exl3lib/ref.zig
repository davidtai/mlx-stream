//! The host reference of one routed MoE layer and the synthetic banks the GPU tests and benches use. The reference is
//! Sushi's `moeSwigluHost` over this library's `format.project`: gate and up through the full projection (f16 rounding at
//! the input, the inner product and the output), SwiGLU in f32, the down projection on that f32 h (its prepare rounds h
//! to f16), and the score-weighted sum over the slots in slot order. Every kernel is held to this within a tolerance.
const std = @import("std");
const format = @import("format.zig");

pub const Bank = struct {
    experts: usize,
    hidden: usize,
    inter: usize,
    rate: format.Rate,
    gate_t: []u16,
    gate_suh: []u16,
    gate_svh: []u16,
    up_t: []u16,
    up_suh: []u16,
    up_svh: []u16,
    down_t: []u16,
    down_suh: []u16,
    down_svh: []u16,

    pub fn deinit(self: *Bank, a: std.mem.Allocator) void {
        a.free(self.gate_t);
        a.free(self.gate_suh);
        a.free(self.gate_svh);
        a.free(self.up_t);
        a.free(self.up_suh);
        a.free(self.up_svh);
        a.free(self.down_t);
        a.free(self.down_suh);
        a.free(self.down_svh);
    }

    /// halfwords of one expert's gate or up trellis
    pub fn guStride(self: Bank) usize {
        return (self.hidden / 16) * (self.inter / 16) * self.rate.n;
    }
};

/// Random trellis halfwords (every 16-bit pattern is a valid codeword), suh in [0.5, 1.5] and svh in [0.5, 1.5] times a
/// scale that keeps the outputs O(1): the decoded weights are O(1), a 128-rotated input is O(1), so an inner product over
/// `in` terms is O(sqrt(in)).
pub fn synthBank(a: std.mem.Allocator, rnd: std.Random, experts: usize, hidden: usize, inter: usize, rate: format.Rate) !Bank {
    const n = rate.n;
    const gu = experts * (hidden / 16) * (inter / 16) * n;
    const b: Bank = .{
        .experts = experts,
        .hidden = hidden,
        .inter = inter,
        .rate = rate,
        .gate_t = try a.alloc(u16, gu),
        .gate_suh = try a.alloc(u16, experts * hidden),
        .gate_svh = try a.alloc(u16, experts * inter),
        .up_t = try a.alloc(u16, gu),
        .up_suh = try a.alloc(u16, experts * hidden),
        .up_svh = try a.alloc(u16, experts * inter),
        .down_t = try a.alloc(u16, gu),
        .down_suh = try a.alloc(u16, experts * inter),
        .down_svh = try a.alloc(u16, experts * hidden),
    };
    for (b.gate_t) |*w| w.* = rnd.int(u16);
    for (b.up_t) |*w| w.* = rnd.int(u16);
    for (b.down_t) |*w| w.* = rnd.int(u16);
    const out_scale_gu: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
    const out_scale_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(inter)));
    for (b.gate_suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (b.up_suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (b.down_suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
    for (b.gate_svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale_gu);
    for (b.up_svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale_gu);
    for (b.down_svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * out_scale_d);
    return b;
}

/// y [hidden] for one row x [hidden] over its slots (expert ids) and scores, in slot order.
pub fn moeRow(a: std.mem.Allocator, b: *const Bank, x: []const f32, slots: []const u32, scores: []const f32, y: []f32) !void {
    const hidden = b.hidden;
    const inter = b.inter;
    @memset(y, 0);
    const transformed = try a.alloc(f32, hidden);
    defer a.free(transformed);
    const inner = try a.alloc(f32, @max(hidden, inter));
    defer a.free(inner);
    const gate_y = try a.alloc(f32, inter);
    defer a.free(gate_y);
    const up_y = try a.alloc(f32, inter);
    defer a.free(up_y);
    const h = try a.alloc(f32, inter);
    defer a.free(h);
    const down_y = try a.alloc(f32, hidden);
    defer a.free(down_y);
    const stride = b.guStride();
    for (slots, scores) |e_u, w| {
        const e: usize = e_u;
        const off = e * stride;
        format.project(x, b.gate_t[off..][0..stride], b.gate_suh[e * hidden ..][0..hidden], b.gate_svh[e * inter ..][0..inter], hidden, inter, b.rate, .{ .codebook = .mul1 }, transformed, inner[0..inter], gate_y);
        _ = &gate_y;
        format.project(x, b.up_t[off..][0..stride], b.up_suh[e * hidden ..][0..hidden], b.up_svh[e * inter ..][0..inter], hidden, inter, b.rate, .{ .codebook = .mul1 }, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        format.project(h, b.down_t[off..][0..stride], b.down_suh[e * inter ..][0..inter], b.down_svh[e * hidden ..][0..hidden], inter, hidden, b.rate, .{ .codebook = .mul1 }, inner[0..inter], transformed, down_y);
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
}

/// `moeRow` under a decode (codebook, window) other than MUL1 w16.
pub fn moeRowDec(a: std.mem.Allocator, b: *const Bank, dec: format.Decode, x: []const f32, slots: []const u32, scores: []const f32, y: []f32) !void {
    const hidden = b.hidden;
    const inter = b.inter;
    @memset(y, 0);
    const transformed = try a.alloc(f32, hidden);
    defer a.free(transformed);
    const inner = try a.alloc(f32, @max(hidden, inter));
    defer a.free(inner);
    const gate_y = try a.alloc(f32, inter);
    defer a.free(gate_y);
    const up_y = try a.alloc(f32, inter);
    defer a.free(up_y);
    const h = try a.alloc(f32, inter);
    defer a.free(h);
    const down_y = try a.alloc(f32, hidden);
    defer a.free(down_y);
    const stride = b.guStride();
    for (slots, scores) |e_u, w| {
        const e: usize = e_u;
        const off = e * stride;
        format.project(x, b.gate_t[off..][0..stride], b.gate_suh[e * hidden ..][0..hidden], b.gate_svh[e * inter ..][0..inter], hidden, inter, b.rate, dec, transformed, inner[0..inter], gate_y);
        format.project(x, b.up_t[off..][0..stride], b.up_suh[e * hidden ..][0..hidden], b.up_svh[e * inter ..][0..inter], hidden, inter, b.rate, dec, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        format.project(h, b.down_t[off..][0..stride], b.down_suh[e * inter ..][0..inter], b.down_svh[e * hidden ..][0..hidden], inter, hidden, b.rate, dec, inner[0..inter], transformed, down_y);
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
}

/// max |a - b| and the rms of b.
pub const Err = struct { max_abs: f64, rms_ref: f64, max_ratio: f64 };

pub fn compare(got: []const f32, ref: []const f32) Err {
    var max_abs: f64 = 0;
    var ss: f64 = 0;
    for (got, ref) |g, r| {
        max_abs = @max(max_abs, @abs(@as(f64, g) - @as(f64, r)));
        ss += @as(f64, r) * @as(f64, r);
    }
    const rms = @sqrt(ss / @as(f64, @floatFromInt(ref.len)));
    return .{ .max_abs = max_abs, .rms_ref = rms, .max_ratio = if (rms > 0) max_abs / rms else max_abs };
}

const testing = std.testing;

test "exl3lib ref: a 256x128 K4 bank's row through the reference is finite and O(1); slot order is the summation order" {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();
    const a = testing.allocator;
    var b = try synthBank(a, rnd, 4, 256, 128, format.Rate.fromK(4));
    defer b.deinit(a);
    var x: [256]f32 = undefined;
    for (&x) |*v| v.* = rnd.floatNorm(f32);
    var y: [256]f32 = undefined;
    try moeRow(a, &b, &x, &.{ 1, 3 }, &.{ 0.7, 0.3 }, &y);
    var ss: f64 = 0;
    for (y) |v| {
        try testing.expect(std.math.isFinite(v));
        ss += v * v;
    }
    const rms = @sqrt(ss / 256.0);
    try testing.expect(rms > 1e-3 and rms < 100.0);
    // the same slots in the other order differ only by f32 summation order
    var y2: [256]f32 = undefined;
    try moeRow(a, &b, &x, &.{ 3, 1 }, &.{ 0.3, 0.7 }, &y2);
    const e = compare(&y2, &y);
    try testing.expect(e.max_ratio < 1e-5);
}
