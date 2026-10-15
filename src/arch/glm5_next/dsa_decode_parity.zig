//! The window's parity test of the fused DSA decode texts (dsa_decode.zig): GLM5NEXT_DSA_DECODE_GPU=1 (else skipped) and
//! _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child). Synthetic absorbed queries [1, 64, S, 512], a
//! latent lane of 8,192 rows and selections of 2,051 slots per row (2,048 distinct rows in increasing order and the
//! partial pool's tail, the unused tail slots -1, as `ops.selectTokens` lays them out), S = 1..4:
//!   - the texts' output within 1e-3 relative RMS of the f64 reference (softmax over the selected rows at scale
//!     1 / sqrt(256)) and within 2e-2 of the op chain (`ops.sparseLatentAttention`: bf16 scores and probabilities);
//!   - a selection whose every slot is one row: the output is that row (the softmax of one key), bit for bit in bf16.
//! Static bound (declared before any run): the lane 8 MiB, queries 0.5 MiB, the op chain's gathered rows and scores
//! < 16 MiB, the texts' partials 2 MiB: < 64 MiB with the JIT; declared 0.2 GB.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const ops = @import("ops.zig");
const dsa_decode = @import("dsa_decode.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops_be.MlxOps;

const parity_bound_bytes: u64 = 200_000_000;
const R: usize = dsa_decode.latent;
const H: usize = 64;
const lanes: usize = 8192;
const width: usize = 2051;

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn bf16F32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn hostF32(g: *G, a: std.mem.Allocator, x: T) ![]f32 {
    const f = try g.astype(x, .float32);
    try g.evalAll(&.{f});
    const out = try a.alloc(f32, mlx.mlx_array_size(f));
    _ = try g.hostF32(f, out);
    return out;
}

fn relRms(got: []const f32, want: []const f64) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, want) |x, t| {
        num += (@as(f64, x) - t) * (@as(f64, x) - t);
        den += t * t;
    }
    return @sqrt(num / @max(den, 1e-30));
}

test "glm5next dsa decode GPU: the split / merge texts against the f64 reference and the op chain at S = 1..4; one repeated row" {
    _ = std.c.getenv("GLM5NEXT_DSA_DECODE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_DSA_DECODE_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: glm.Config = .{};
    var diag: @import("sdk").Diag = .{};
    var k = dsa_decode.install(a, s, &c, .current, &diag) catch |e| {
        std.debug.print("glm5next dsa decode install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer k.deinit();
    // `dsa_split_form: lean`: the same outputs bit for bit (the same operations in the same order per output).
    var kl = dsa_decode.install(a, s, &c, .lean, &diag) catch |e| {
        std.debug.print("glm5next dsa decode install (lean) refused: {s}\n", .{diag.message()});
        return e;
    };
    defer kl.deinit();
    var prng = std.Random.DefaultPrng.init(0x647361);
    const r = prng.random();
    const lat_h = try a.alloc(u16, lanes * R);
    defer a.free(lat_h);
    for (lat_h) |*x| x.* = bf16Bits(r.floatNorm(f32));
    const lat = try g.hostArray(std.mem.sliceAsBytes(lat_h), &.{ 1, 1, @intCast(lanes), @intCast(R) }, .bfloat16);
    const scale: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim)));
    for (1..dsa_decode.max_rows + 1) |rows| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const q_h = try a.alloc(u16, H * rows * R);
        defer a.free(q_h);
        for (q_h) |*x| x.* = bf16Bits(0.5 * r.floatNorm(f32));
        const sel_h = try a.alloc(i32, rows * width);
        defer a.free(sel_h);
        for (0..rows) |row| {
            // 2,048 distinct rows in increasing order (a random sample of the lane), then the tail: `row` used slots.
            var picked: usize = 0;
            var i: usize = 0;
            while (picked < 2048) : (i += 1) {
                if (r.uintLessThan(usize, lanes - i) < 2048 - picked) {
                    sel_h[row * width + picked] = @intCast(i);
                    picked += 1;
                }
            }
            for (0..3) |t| sel_h[row * width + 2048 + t] = if (t < row) @intCast(lanes - 3 + t) else -1;
        }
        const q = try g.hostArray(std.mem.sliceAsBytes(q_h), &.{ 1, @intCast(H), @intCast(rows), @intCast(R) }, .bfloat16);
        const sel = try g.hostArray(std.mem.sliceAsBytes(sel_h), &.{ 1, @intCast(rows), @intCast(width) }, .int32);
        const got = try hostF32(&g, a, try dsa_decode.attend(&g, &k, q, lat, sel, @intCast(rows)));
        defer a.free(got);
        const got_lean = try hostF32(&g, a, try dsa_decode.attend(&g, &kl, q, lat, sel, @intCast(rows)));
        defer a.free(got_lean);
        try testing.expectEqualSlices(u32, @as([*]const u32, @ptrCast(got.ptr))[0..got.len], @as([*]const u32, @ptrCast(got_lean.ptr))[0..got_lean.len]);
        std.debug.print("GLM5N_DSA_SPLIT_LEAN rows={d} identical=yes\n", .{rows});
        if (rows == 1) {
            // The side text (the MTP head's draft steps): the lane's first lanes - 3 rows and its last three as side rows.
            const lane_view = try g.slice(lat, &.{ 0, 0, 0, 0 }, &.{ 1, 1, @intCast(lanes - 3), @intCast(R) }, &.{ 1, 1, 1, 1 });
            const side = try g.hostArray(std.mem.sliceAsBytes(lat_h[(lanes - 3) * R ..]), &.{ 1, 1, 3, @intCast(R) }, .bfloat16);
            const got_side = try hostF32(&g, a, try dsa_decode.attendSide(&g, &kl, q, lane_view, side, sel, @intCast(lanes - 3)));
            defer a.free(got_side);
            try testing.expectEqualSlices(u32, @as([*]const u32, @ptrCast(got.ptr))[0..got.len], @as([*]const u32, @ptrCast(got_side.ptr))[0..got_side.len]);
            std.debug.print("GLM5N_DSA_SPLIT_SIDE rows=1 identical=yes\n", .{});
        }
        const chain = try hostF32(&g, a, try g.adopt(try ops.sparseLatentAttention(s, q, lat, sel, @floatCast(scale), 64)));
        defer a.free(chain);
        // The f64 reference, [1, H, rows, R].
        const want = try a.alloc(f64, H * rows * R);
        defer a.free(want);
        const sc = try a.alloc(f64, width);
        defer a.free(sc);
        for (0..H) |h| for (0..rows) |row| {
            const qr = q_h[(h * rows + row) * R ..][0..R];
            var mx: f64 = -std.math.inf(f64);
            for (0..width) |j| {
                const id = sel_h[row * width + j];
                if (id < 0) {
                    sc[j] = -std.math.inf(f64);
                    continue;
                }
                const kr = lat_h[@as(usize, @intCast(id)) * R ..][0..R];
                var d: f64 = 0;
                for (qr, kr) |x, y| d += @as(f64, bf16F32(x)) * bf16F32(y);
                sc[j] = d * scale;
                mx = @max(mx, sc[j]);
            }
            var z: f64 = 0;
            for (sc) |*v| {
                v.* = if (v.* == -std.math.inf(f64)) 0 else @exp(v.* - mx);
                z += v.*;
            }
            const o = want[(h * rows + row) * R ..][0..R];
            @memset(o, 0);
            for (0..width) |j| if (sc[j] > 0) {
                const kr = lat_h[@as(usize, @intCast(sel_h[row * width + j])) * R ..][0..R];
                for (o, kr) |*x, y| x.* += sc[j] / z * bf16F32(y);
            };
        };
        const rel = relRms(got, want);
        const rel_chain = relRms(chain, want);
        var gc: f64 = 0;
        var gd: f64 = 0;
        for (got, chain) |x, y| {
            gc += (@as(f64, x) - y) * (@as(f64, x) - y);
            gd += @as(f64, y) * y;
        }
        const vs_chain = @sqrt(gc / @max(gd, 1e-30));
        std.debug.print("GLM5N_DSA_DECODE_PARITY rows={d} texts_vs_f64={e} chain_vs_f64={e} texts_vs_chain={e}\n", .{ rows, rel, rel_chain, vs_chain });
        try testing.expect(rel <= 1e-3);
        try testing.expect(vs_chain <= 2e-2);
    }
    // One repeated row: the softmax of one key, the output that row (bf16) bit for bit.
    {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const sel_h = try a.alloc(i32, width);
        defer a.free(sel_h);
        @memset(sel_h, 77);
        sel_h[width - 1] = -1;
        const q_h = try a.alloc(u16, H * R);
        defer a.free(q_h);
        for (q_h) |*x| x.* = bf16Bits(r.floatNorm(f32));
        const q = try g.hostArray(std.mem.sliceAsBytes(q_h), &.{ 1, @intCast(H), 1, @intCast(R) }, .bfloat16);
        const sel = try g.hostArray(std.mem.sliceAsBytes(sel_h), &.{ 1, 1, @intCast(width) }, .int32);
        const got = try hostF32(&g, a, try dsa_decode.attend(&g, &k, q, lat, sel, 1));
        defer a.free(got);
        for (0..H) |h| for (0..R) |d| try testing.expectEqual(bf16F32(lat_h[77 * R + d]), got[h * R + d]);
    }
}
