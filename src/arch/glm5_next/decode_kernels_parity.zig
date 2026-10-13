//! The window's parity test of the NVFP4 decode kernels (`decode_kernels`): GLM5NEXT_NVFP4_GPU=1 (else skipped),
//! _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child) and GLM5NEXT_NVFP4_VECTORS=<dir of the three test
//! vector .npz files and vectors.json>. The three real records (L3 E0, L23 E144, L44 E287) sit in the slot banks beside
//! synthetic NVFP4 records (8 slots a bank, three banks: the `device` passing of every array); waves of 1 to 8 tokens of
//! top-8 (P = 8 .. 64 pairs) and two partial waves (3 and 13 pairs) run the texts, MLX's gather_qmm in nvfp4 mode (the
//! stock route) and the f32 truth, and the test gates:
//!   - parity (MiMo's design-2c tolerance, src/arch/mimo_v2/moe.zig "mimo moe GPU"): each pair's SwiGLU row and down row,
//!     relative RMS against the f32 truth <= 2 x the stock route's + 1e-6;
//!   - M-invariance: every pair's rows in a wave are bit for bit its rows alone (a one-pair wave), so the union texts'
//!     per-member arithmetic is the one-token texts';
//!   - decode probe: the down text over one-hot activations returns bf16(w) of the real records' weights, bit for bit
//!     against the host reference (`decode_kernels_ref.decodeProjection`; a zero's sign aside);
//!   - MLX's device dequant (nvfp4, global scale, f32) of the real records equals the host reference bit for bit (the
//!     device's g / 2688 is the reference's: the texts use the same expression).
//! The stock route: the mlx-c binding of gather_qmm (56b2d39) takes no global_scale; MLX's own nvfp4 GEMV accumulates the
//! scaled codes in f32 and multiplies by g / 2688 last, so the route runs gather_qmm without it on f32 activations (an f32
//! output: the accumulation unrounded), multiplies by g / 2688 (computed on the device by MLX's divide) and rounds to bf16:
//! the same arithmetic, then the SDK's clamped SwiGLU on bf16 (`sdk_ext.quant.activation`).
//! Static bound (declared before any run): 3 banks x 8 slots x 14,155,788 B = 340 MB of slot arrays (host copies freed
//! after each MLX copy: <= 2 x 14 MB transient); per pair the truth dequantizes three f32 projections (3 x 32 MiB) and the
//! device dequant check three more, evaluated and released per pair; the waves' outputs < 3 MiB. Bound: 340 MB + 192 MiB
//! + MLX JIT (two texts, four instantiations, two passings) <= 0.8 GB; declared 1.2 GB.

const std = @import("std");
const mlx = @import("sdk").mlx;
const io_util = @import("sdk").io_util;
const xk = @import("decode_kernels.zig");
const ref = @import("decode_kernels_ref.zig");
const ops = @import("../deepseek_v41/ops.zig");
const quant = @import("../../sdk_ext.zig").quant;

const testing = std.testing;
const T = mlx.mlx_array;

const parity_bound_bytes: u64 = 1_200_000_000;
const slots: u32 = 8;
/// f32 per global-scale line (bank.zig `scale_line_floats`).
const gl: c_int = @import("bank.zig").scale_line_floats;
const gl_n: usize = @import("bank.zig").scale_line_floats;
const H: u32 = xk.hidden;
const I: u32 = xk.inter;
const record_bytes: usize = 14_155_788;

/// A record's parts (the pack's nvfp4-gud-9part layout: docs/glm5-next-pack.md on glm5next/pack).
const Part = struct { off: usize, len: usize };
const parts = struct {
    const gate_w: Part = .{ .off = 0, .len = 4_194_304 };
    const gate_s: Part = .{ .off = 4_194_304, .len = 524_288 };
    const gate_g: Part = .{ .off = 4_718_592, .len = 4 };
    const up_w: Part = .{ .off = 4_718_596, .len = 4_194_304 };
    const up_s: Part = .{ .off = 8_912_900, .len = 524_288 };
    const up_g: Part = .{ .off = 9_437_188, .len = 4 };
    const down_w: Part = .{ .off = 9_437_192, .len = 4_194_304 };
    const down_s: Part = .{ .off = 13_631_496, .len = 524_288 };
    const down_g: Part = .{ .off = 14_155_784, .len = 4 };
};

fn nextU32(state: *u32) u32 {
    state.* = state.* *% 1664525 +% 1013904223;
    return state.*;
}

/// A synthetic record: random codes, E4M3 scale bytes 0x28 .. 0x57 (0.25 .. 15: the real banks' spread around their g),
/// and the given global scales.
fn synthRecord(rec: []u8, g: [3]f32, state: *u32) void {
    for (0..record_bytes / 4) |i| std.mem.writeInt(u32, rec[4 * i ..][0..4], nextU32(state), .little);
    for ([_]Part{ parts.gate_s, parts.up_s, parts.down_s }) |p| for (rec[p.off..][0..p.len]) |*b| {
        b.* = @intCast(0x28 + nextU32(state) % 0x30);
    };
    for ([_]Part{ parts.gate_g, parts.up_g, parts.down_g }, g) |p, v| std.mem.writeInt(u32, rec[p.off..][0..4], @bitCast(v), .little);
}

fn gOf(rec: []const u8, p: Part) f32 {
    return @bitCast(std.mem.readInt(u32, rec[p.off..][0..4], .little));
}

/// bf16 bits of `x`, rounded to nearest even (MLX's cast; no NaN here).
fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn relRms(got: []const f32, truth: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, truth) |x, t| {
        num += (@as(f64, x) - t) * (@as(f64, x) - t);
        den += @as(f64, t) * t;
    }
    return @sqrt(num / @max(den, 1e-30));
}

/// One projection's three slot arrays from `recs` (one record a slot): words, scales, global scales.
fn projArrays(a: std.mem.Allocator, g: *ops.MlxOps, recs: []const []const u8, w: Part, s: Part, gp: Part, rows: u32, cols: u32) !xk.ProjArrays {
    const n: usize = recs.len;
    const words = try a.alloc(u8, n * w.len);
    defer a.free(words);
    const scales = try a.alloc(u8, n * s.len);
    defer a.free(scales);
    // Line-isolated global scales (bank.zig `scale_line_bytes`): slot i's word at column 0 of row i, the rest zero.
    var gs: [slots * gl_n]f32 = @splat(0);
    for (recs, 0..) |r, i| {
        @memcpy(words[i * w.len ..][0..w.len], r[w.off..][0..w.len]);
        @memcpy(scales[i * s.len ..][0..s.len], r[s.off..][0..s.len]);
        gs[i * gl_n] = gOf(r, gp);
    }
    const wa = try g.hostArray(words, &.{ @intCast(n), @intCast(rows), @intCast(cols / 8) }, .uint32);
    const sa = try g.hostArray(scales, &.{ @intCast(n), @intCast(rows), @intCast(cols / 16) }, .uint8);
    const ga = try g.hostArray(std.mem.sliceAsBytes(gs[0 .. n * gl_n]), &.{ @intCast(n), gl }, .float32);
    try g.evalAll(&.{ wa, sa, ga });
    return .{ .w = g.keep(wa), .scales = g.keep(sa), .global_scale = g.keep(ga) };
}

/// MLX's dequant of one slot's projection (nvfp4, group 16, its global scale, f32): [rows, cols].
fn dequantF32(g: *ops.MlxOps, p: xk.ProjArrays, slot: u32, rows: u32, cols: u32) !T {
    const sl: c_int = @intCast(slot);
    const w = try g.reshape(try g.slice(p.w, &.{ sl, 0, 0 }, &.{ sl + 1, @intCast(rows), @intCast(cols / 8) }, &.{ 1, 1, 1 }), &.{ @intCast(rows), @intCast(cols / 8) });
    const s = try g.reshape(try g.slice(p.scales, &.{ sl, 0, 0 }, &.{ sl + 1, @intCast(rows), @intCast(cols / 16) }, &.{ 1, 1, 1 }), &.{ @intCast(rows), @intCast(cols / 16) });
    const gs = try g.reshape(try g.slice(p.global_scale, &.{ sl, 0 }, &.{ sl + 1, 1 }, &.{ 1, 1 }), &.{1});
    var r = mlx.mlx_array_new();
    mlx.check(mlx.mlx_dequantize(&r, w, s, .{}, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", gs, .{ .value = .float32, .has_value = true }, g.s)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

/// The stock route's projection of one pair: MLX's gather_qmm in nvfp4 mode at the pair's slot, on f32 activations (an
/// f32 output), times g / 2688 of the slot (MLX's divide), rounded to bf16: [1, rows].
fn stockProj(g: *ops.MlxOps, xf: T, p: xk.ProjArrays, slot: T, rows: u32) !T {
    const y = try g.gatherQmm(xf, p.w, p.scales, slot, .nvfp4);
    const n = g.shapeOf(p.global_scale).d[0];
    const col0 = try g.reshape(try g.slice(p.global_scale, &.{ 0, 0 }, &.{ n, 1 }, &.{ 1, 1 }), &.{n});
    const inv = try g.take(try g.div(col0, try g.scalar(2688.0, .float32)), slot, 0);
    return g.astype(try g.mul(try g.reshape(y, &.{ 1, @intCast(rows) }), try g.reshape(inv, &.{ 1, 1 })), .bfloat16);
}

/// The f32 truth of one pair: the dequantized slot views, f32 GEMVs, the clamped SwiGLU in f32, the f32 down.
fn truthRows(g: *ops.MlxOps, bank: xk.BankArrays, slot: u32, x: T) !struct { act: T, rows: T } {
    const xf = try g.astype(x, .float32);
    const gt = try g.matmul(xf, try g.transpose(try dequantF32(g, bank.gate, slot, I, H)));
    const ut = try g.matmul(xf, try g.transpose(try dequantF32(g, bank.up, slot, I, H)));
    const lim: f64 = @floatFromInt(xk.swiglu_limit);
    const gc = try g.minimum(gt, try g.scalar(lim, .float32));
    const uc = try g.clip(ut, try g.scalar(-lim, .float32), try g.scalar(lim, .float32));
    const act = try g.mul(try g.silu(gc), uc);
    return .{ .act = act, .rows = try g.matmul(act, try g.transpose(try dequantF32(g, bank.down, slot, H, I))) };
}

fn hostOf(a: std.mem.Allocator, g: *ops.MlxOps, x: T) ![]f32 {
    const n: usize = @intCast(g.shapeOf(x).numel());
    const out = try a.alloc(f32, n);
    errdefer a.free(out);
    _ = try g.hostF32(try g.astype(x, .float32), out);
    return out;
}

fn bitwiseEqual(p: []const f32, q: []const f32) usize {
    var eq: usize = 0;
    for (p, q) |x, y| eq += @intFromBool(@as(u32, @bitCast(x)) == @as(u32, @bitCast(y)));
    return eq;
}

test "glm5next nvfp4 GPU: the fused texts vs MLX's gather_qmm nvfp4 and the f32 truth on the test vectors' records (waves of 1 to 8 tokens); M-invariance, the decode probe and MLX's device dequant bit for bit" {
    _ = std.c.getenv("GLM5NEXT_NVFP4_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const dir = std.mem.span(std.c.getenv("GLM5NEXT_NVFP4_VECTORS") orelse return error.Glm5nVectorsRequired);
    std.debug.print("GLM5N_NVFP4_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    if (!mlx.streamIsGpu(stream)) return error.SkipZigTest;
    var diag: xk.Diag = .{};
    try xk.checkModel(H, I, xk.top_k, 10.0, &diag);
    var inst = xk.install(a, stream, &diag) catch |e| {
        std.debug.print("glm5next nvfp4 install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer inst.deinit();
    var g = try ops.MlxOps.init(a, stream);
    defer g.deinit();

    // The records: the three test vectors (record sha256 checked against vectors.json by `decode_kernels_ref`'s CPU test;
    // here their bytes and global scales), and 21 synthetic ones.
    var real: [3][]u8 = undefined;
    var n_real: usize = 0;
    defer for (real[0..n_real]) |r| a.free(r);
    for ([_][2]u32{ .{ 3, 0 }, .{ 23, 144 }, .{ 44, 287 } }) |le| {
        const path = try std.fmt.allocPrint(a, "{s}/glm5next-nvfp4-L{d:0>2}-E{d:0>3}.npz", .{ dir, le[0], le[1] });
        defer a.free(path);
        const img = try io_util.readAllNoCache(a, path, 128 << 20);
        defer a.free(img);
        const arrays = try ref.npzMembers(a, img);
        defer a.free(arrays);
        const rec = ref.member(arrays, "record") orelse return error.VectorsRecordMissing;
        if (rec.data.len != record_bytes) return error.VectorsRecordSize;
        real[n_real] = try a.dupe(u8, rec.data);
        n_real += 1;
    }
    var state: u32 = 0x676c6d35;
    const synth = try a.alloc(u8, 21 * record_bytes);
    defer a.free(synth);
    for (0..21) |i| synthRecord(synth[i * record_bytes ..][0..record_bytes], .{ gOf(real[i % 3], parts.gate_g), gOf(real[(i + 1) % 3], parts.up_g), gOf(real[(i + 2) % 3], parts.down_g) }, &state);
    // Bank b slot s: a real record at (0, 0), (1, 3), (2, 7), synthetic elsewhere.
    var recs: [xk.n_banks][slots][]const u8 = undefined;
    var k: usize = 0;
    for (&recs, 0..) |*bank, b| for (bank, 0..) |*r, s| {
        r.* = if (b == 0 and s == 0) real[0] else if (b == 1 and s == 3) real[1] else if (b == 2 and s == 7) real[2] else blk: {
            defer k += 1;
            break :blk synth[k * record_bytes ..][0..record_bytes];
        };
    };
    var banks: [xk.n_banks]xk.BankArrays = undefined;
    for (&banks, recs) |*bk, rs| bk.* = .{
        .gate = try projArrays(a, &g, &rs, parts.gate_w, parts.gate_s, parts.gate_g, I, H),
        .up = try projArrays(a, &g, &rs, parts.up_w, parts.up_s, parts.up_g, I, H),
        .down = try projArrays(a, &g, &rs, parts.down_w, parts.down_s, parts.down_g, H, I),
    };
    defer for (banks) |bk| for ([_]xk.ProjArrays{ bk.gate, bk.up, bk.down }) |p| {
        g.release(p.w);
        g.release(p.scales);
        g.release(p.global_scale);
    };
    g.reset();

    // 1. MLX's device dequant of the real records = the host reference, bit for bit.
    const real_at = [_][2]u32{ .{ 0, 0 }, .{ 1, 3 }, .{ 2, 7 } };
    var dq_words: usize = 0;
    for (real_at, 0..) |at, ri| {
        const bk = banks[at[0]];
        for ([_]xk.ProjArrays{ bk.gate, bk.up, bk.down }, [_][3]Part{ .{ parts.gate_w, parts.gate_s, parts.gate_g }, .{ parts.up_w, parts.up_s, parts.up_g }, .{ parts.down_w, parts.down_s, parts.down_g } }, 0..) |p, pp, pi| {
            const rows: u32 = if (pi == 2) H else I;
            const cols: u32 = if (pi == 2) I else H;
            const mark = g.mark();
            const dev = try hostOf(a, &g, try dequantF32(&g, p, at[1], rows, cols));
            defer a.free(dev);
            g.resetTo(mark);
            const words = try a.alloc(u32, rows * cols / 8);
            defer a.free(words);
            @memcpy(std.mem.sliceAsBytes(words), real[ri][pp[0].off..][0..pp[0].len]);
            const host = try a.alloc(f32, rows * cols);
            defer a.free(host);
            ref.decodeProjection(words, real[ri][pp[1].off..][0..pp[1].len], gOf(real[ri], pp[2]), rows, cols, host);
            const eq = bitwiseEqual(dev, host);
            std.debug.print("GLM5N_NVFP4_DEQUANT record={d} proj={d} bitwise={d}/{d}\n", .{ ri, pi, eq, host.len });
            try testing.expectEqual(host.len, eq);
            dq_words += eq;
        }
    }

    // 2. The decode probe: one-hot activations through the down text (27 pairs: the union texts), bf16(w) bit for bit.
    {
        const cols_probe = [_]u32{ 0, 1, 7, 8, 15, 16, 511, 512, I - 1 };
        const n_probe = cols_probe.len * real_at.len;
        var ids: [n_probe]u32 = undefined;
        const act = try a.alloc(u16, n_probe * I);
        defer a.free(act);
        @memset(act, 0);
        for (real_at, 0..) |at, ri| for (cols_probe, 0..) |c, ci| {
            const p = ri * cols_probe.len + ci;
            ids[p] = (at[0] << xk.row_bits) | at[1];
            act[p * I + c] = 0x3F80; // bf16 1.0
        };
        const act_a = try g.hostArray(std.mem.sliceAsBytes(act), &.{ @intCast(n_probe), @intCast(I) }, .bfloat16);
        const id_a = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{@intCast(n_probe)}, .uint32);
        const y = try g.adopt(try inst.fused.down(act_a, id_a, &banks));
        const got = try hostOf(a, &g, y);
        defer a.free(got);
        var eq: usize = 0;
        for (real_at, 0..) |_, ri| {
            const words = try a.alloc(u32, H * I / 8);
            defer a.free(words);
            @memcpy(std.mem.sliceAsBytes(words), real[ri][parts.down_w.off..][0..parts.down_w.len]);
            const w = try a.alloc(f32, H * I);
            defer a.free(w);
            ref.decodeProjection(words, real[ri][parts.down_s.off..][0..parts.down_s.len], gOf(real[ri], parts.down_g), H, I, w);
            for (cols_probe, 0..) |c, ci| {
                const p = ri * cols_probe.len + ci;
                for (0..H) |r| {
                    // bf16 bits, a zero's sign aside (a sum of a -0 weight and +0 products is +0)
                    const wb = bf16Bits(w[r * I + c]);
                    const want: u16 = if (wb & 0x7FFF == 0) 0 else wb;
                    const hb: u16 = @truncate(@as(u32, @bitCast(got[p * H + r])) >> 16);
                    const have: u16 = if (hb & 0x7FFF == 0) 0 else hb;
                    eq += @intFromBool(have == want);
                }
            }
        }
        std.debug.print("GLM5N_NVFP4_PROBE pairs={d} bitwise={d}/{d}\n", .{ n_probe, eq, n_probe * H });
        try testing.expectEqual(@as(usize, n_probe * H), eq);
        g.reset();
    }

    // 3. Waves: M = 1 .. 8 tokens of top-8 (ids drawn over the 24 slots, so the tokens of a wave share slots: the union
    // texts from 9 pairs), and two partial waves (3, 13 pairs). Per wave: parity and M-invariance.
    var worst_rows: f64 = 0;
    var worst_act: f64 = 0;
    var inv_words: usize = 0;
    const widths = [_]u32{ 3, 8, 13, 16, 24, 32, 40, 48, 56, 64 };
    for (widths) |pairs| {
        const m: u32 = (pairs + xk.top_k - 1) / xk.top_k;
        var ids: [xk.max_pairs]u32 = undefined;
        var tok: [xk.max_pairs]i32 = undefined;
        for (0..pairs) |p| {
            const t: u32 = @intCast(p / xk.top_k);
            // within a token distinct slots (a router's top-8); across tokens shared, every real record named
            const pick: u32 = @intCast((p % xk.top_k) * 3 + (t * 5 + nextU32(&state) % 2) % 3);
            ids[p] = ((pick / slots) << xk.row_bits) | (pick % slots);
            tok[p] = @intCast(t);
        }
        var xin: [xk.max_tokens * H]f32 = undefined;
        for (xin[0 .. m * H]) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(nextU32(&state) % 2001)) - 1000)) / 1000;
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin[0 .. m * H]), &.{ @intCast(m), @intCast(H) }, .float32), .bfloat16);
        const xp = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..pairs]), &.{@intCast(pairs)}, .int32), 0);
        const id_a = try g.hostArray(std.mem.sliceAsBytes(ids[0..pairs]), &.{@intCast(pairs)}, .uint32);
        const kh = try g.adopt(try inst.fused.gateUp(xp, id_a, &banks));
        const kr = try g.adopt(try inst.fused.down(kh, id_a, &banks));
        try g.evalAll(&.{ kh, kr });
        const kh_h = try hostOf(a, &g, kh);
        defer a.free(kh_h);
        const kr_h = try hostOf(a, &g, kr);
        defer a.free(kr_h);
        const sh_h = try a.alloc(f32, pairs * I);
        defer a.free(sh_h);
        const sr_h = try a.alloc(f32, pairs * H);
        defer a.free(sr_h);
        const th_h = try a.alloc(f32, pairs * I);
        defer a.free(th_h);
        const tr_h = try a.alloc(f32, pairs * H);
        defer a.free(tr_h);
        var alone_eq: usize = 0;
        for (0..pairs) |p| {
            const mark = g.mark();
            const slot_v: u32 = ids[p] & ((@as(u32, 1) << xk.row_bits) - 1);
            const bk = banks[ids[p] >> xk.row_bits];
            const xr = try g.slice(xp, &.{ @intCast(p), 0 }, &.{ @intCast(p + 1), @intCast(H) }, &.{ 1, 1 });
            // stock: MLX's gather_qmm nvfp4 per projection, the SDK's clamped SwiGLU, down
            const slot = try g.hostArray(std.mem.sliceAsBytes(&[_]u32{slot_v}), &.{1}, .uint32);
            const xf = try g.reshape(try g.astype(xr, .float32), &.{ 1, 1, @intCast(H) });
            const sg = try stockProj(&g, xf, bk.gate, slot, I);
            const su = try stockProj(&g, xf, bk.up, slot, I);
            const sh = try quant.activation(ops.MlxOps, &g, .{ .swiglu_clamped = @floatFromInt(xk.swiglu_limit) }, sg, su);
            const sr = try stockProj(&g, try g.reshape(try g.astype(sh, .float32), &.{ 1, 1, @intCast(I) }), bk.down, slot, H);
            _ = try g.hostF32(try g.astype(sh, .float32), sh_h[p * I ..][0..I]);
            _ = try g.hostF32(try g.astype(sr, .float32), sr_h[p * H ..][0..H]);
            // the f32 truth
            const t = try truthRows(&g, bk, slot_v, xr);
            _ = try g.hostF32(t.act, th_h[p * I ..][0..I]);
            _ = try g.hostF32(t.rows, tr_h[p * H ..][0..H]);
            // the pair alone (a one-pair wave): its rows bit for bit
            const one = try g.hostArray(std.mem.sliceAsBytes(ids[p .. p + 1]), &.{1}, .uint32);
            const ah = try g.adopt(try inst.fused.gateUp(xr, one, &banks));
            const ar = try g.adopt(try inst.fused.down(ah, one, &banks));
            const ah_h = try hostOf(a, &g, ah);
            defer a.free(ah_h);
            const ar_h = try hostOf(a, &g, ar);
            defer a.free(ar_h);
            alone_eq += bitwiseEqual(ah_h, kh_h[p * I ..][0..I]) + bitwiseEqual(ar_h, kr_h[p * H ..][0..H]);
            g.resetTo(mark);
        }
        const e_act_k = relRms(kh_h, th_h);
        const e_act_s = relRms(sh_h, th_h);
        const e_rows_k = relRms(kr_h, tr_h);
        const e_rows_s = relRms(sr_h, tr_h);
        std.debug.print("GLM5N_NVFP4_PARITY M={d} pairs={d} act_relrms fused={e:.3} stock={e:.3} rows_relrms fused={e:.3} stock={e:.3} m_invariant={d}/{d}\n", .{ m, pairs, e_act_k, e_act_s, e_rows_k, e_rows_s, alone_eq, pairs * (I + H) });
        try testing.expect(e_act_k <= 2 * e_act_s + 1e-6);
        try testing.expect(e_rows_k <= 2 * e_rows_s + 1e-6);
        try testing.expectEqual(@as(usize, pairs * (I + H)), alone_eq);
        worst_act = @max(worst_act, e_act_k / @max(e_act_s, 1e-12));
        worst_rows = @max(worst_rows, e_rows_k / @max(e_rows_s, 1e-12));
        inv_words += alone_eq;
        g.reset();
    }
    std.debug.print("GLM5N_NVFP4_PARITY PASS worst_act_ratio={d:.3} worst_rows_ratio={d:.3} m_invariant_words={d} dequant_bitwise_words={d}\n", .{ worst_act, worst_rows, inv_words, dq_words });
}
