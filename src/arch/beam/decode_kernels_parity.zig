//! The window's parity tests of beam's fused NVFP4 decode texts (decode_kernels.zig): BEAM_PARITY_SLICE=<the real-weight
//! slice pack> and _GPU_WINDOW_LOCKED=1 (else skipped / refused: the window orchestrator's child).
//!
//! - The self-check: the construction's compile check on the device at every registered instantiation and check width,
//!   then `install` (the launches prepared per wave width).
//! - Parity, word for word: real records of the slice's three routed layers in the three slot banks (base 16 slots, ext 8,
//!   transient 8; the module's line-isolated global scales), waves of 1, 3, 8 (one bank and three), 13, 16, 24, 32 and 64
//!   pairs (1 to 8 tokens; the later tokens share half of their slots with the first's, so the union texts read slots
//!   for up to 4 members). Each wave runs the stock decode route as the shared scheduler builds it (per bank group the
//!   pairs' rows, `Math.gateUp`: MLX's gather_qmm nvfp4 for gate and up and the f32 SwiGLU, then `Math.down`) and the
//!   fused texts over every bank (`Math.gateUpBanked`, `downBanked`), and gates on 0 differing words in each pair's
//!   SwiGLU row, its down row over the stock route's SwiGLU rows (the down text alone) and its down row over the fused
//!   SwiGLU rows (the chain).
//! Static bound (declared before any run): 32 records x 22,118,412 B = 0.71 GB of slot arrays (each record read into its
//! slot in place), a wave's rows < 3 MiB, MLX JIT (two texts, four instantiations) and the binary: 1.5 GB declared.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const io_util = sdk.io_util;
const experts = @import("experts.zig");
const decode_kernels = @import("decode_kernels.zig");
const bank = experts.bank;

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops.MlxOps;
const Kernels = decode_kernels.Kernels;

pub const parity_bound_bytes: u64 = 1_500_000_000;
const H: u32 = Kernels.hidden;
const I: u32 = Kernels.inter;
const slots = [3]u32{ 16, 8, 8 };

fn ci(x: anytype) c_int {
    return @intCast(x);
}

fn hostZeros(g: *G, shape: []const c_int, dt: mlx.mlx_dtype) !struct { arr: T, bytes: []u8 } {
    const x = g.keep(try g.zeros(shape, dt));
    try g.evalAll(&.{x});
    return .{ .arr = x, .bytes = try g.hostBytes(x) };
}

/// A bank of `n` slots as the stream lays its slot arrays out (the module's line-isolated global scales, a page at
/// least), slot s holding the slice's record (routed layer `routed`, expert `first + s`), read in place from `fd` (the
/// slice's experts.bin). Kept arrays: the caller releases them (`releaseBank`).
pub fn buildBank(g: *G, fd: std.c.fd_t, geo: *const bank.Geometry, routed: u32, first: u32, n: u32) !bank.BankArrays {
    var out: [3]bank.Arrays(T) = undefined;
    const lines: u32 = @max(n, @as(u32, @intCast(bank.scale_array_min_bytes / bank.scale_line_bytes)));
    for (0..3) |pi| {
        const rows: u32 = if (pi == 2) geo.hidden_size else geo.intermediate_size;
        const cols: u32 = if (pi == 2) geo.intermediate_size else geo.hidden_size;
        const w = try hostZeros(g, &.{ ci(n), ci(rows), ci(cols / 8) }, .uint32);
        const sc = try hostZeros(g, &.{ ci(n), ci(rows), ci(cols / bank.group_size) }, .uint8);
        const gs = try hostZeros(g, &.{ ci(lines), ci(bank.scale_line_floats) }, .float32);
        const segw = geo.segments[3 * pi];
        const segs = geo.segments[3 * pi + 1];
        const segg = geo.segments[3 * pi + 2];
        for (0..n) |s| {
            const off = geo.recordOffset(geo.first_moe_layer + routed, first + @as(u32, @intCast(s)));
            try io_util.readAligned(fd, w.bytes[s * segw.length ..][0..segw.length], off + segw.offset);
            try io_util.readAligned(fd, sc.bytes[s * segs.length ..][0..segs.length], off + segs.offset);
            var gb: [4]u8 = undefined;
            try io_util.readAligned(fd, &gb, off + segg.offset);
            @memcpy(gs.bytes[s * bank.scale_line_bytes ..][0..4], &gb);
        }
        out[pi] = .{ .w = w.arr, .scales = sc.arr, .global_scale = gs.arr };
    }
    return .{ .gate = out[0], .up = out[1], .down = out[2] };
}

pub fn releaseBank(g: *G, b: bank.BankArrays) void {
    for ([_]bank.Arrays(T){ b.gate, b.up, b.down }) |p| {
        g.release(p.w);
        g.release(p.scales);
        g.release(p.global_scale);
    }
}

fn hostOf(a: std.mem.Allocator, g: *G, x: T) ![]f32 {
    const n: usize = @intCast(g.shapeOf(x).numel());
    const out = try a.alloc(f32, n);
    errdefer a.free(out);
    _ = try g.hostF32(try g.astype(x, .float32), out);
    return out;
}

fn differ(p: []const f32, q: []const f32) usize {
    var n: usize = 0;
    for (p, q) |x, y| n += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
    return n;
}

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u32 {
        r.s = r.s *% 6364136223846793005 +% 1442695040888963407;
        return @truncate(r.s >> 33);
    }
};

fn openSlice(a: std.mem.Allocator, dir: []const u8) !struct { geo: bank.Geometry, fd: std.c.fd_t } {
    const geo = try bank.inspect(a, testing.io, dir);
    if (geo.hidden_size != H or geo.intermediate_size != I or geo.num_moe_layers < 3) return error.BeamSliceGeometry;
    const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
    defer a.free(path);
    return .{ .geo = geo, .fd = try io_util.openNoCache(path.ptr, .{}) };
}

fn windowLocked() !bool {
    _ = std.c.getenv("BEAM_PARITY_SLICE") orelse return false;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    return true;
}

test "beam nvfp4 selfcheck: every qmv kernel builds and completes a launch at every registered instantiation and check width on the GPU; the module's install" {
    if (!try windowLocked()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const set = try Kernels.openForTest(a, s);
    defer set.deinit();
    var report: Kernels.selfcheck.Report = .{};
    defer report.deinit(a);
    try Kernels.selfcheck.runSubset(a, &set.reg, &set.bound, std.enums.values(Kernels.Kernel), .full, &report);
    try Kernels.selfcheck.judge(&report, null);
    var launches: usize = 0;
    for (std.enums.values(Kernels.Kernel)) |k| for (Kernels.instancesOf(k)) |inst| {
        launches += Kernels.selfcheck.checkWidths(inst).len;
    };
    try testing.expectEqual(launches, report.results.items.len);
    var diag: sdk.Diag = .{};
    var f = decode_kernels.FusedDecode.install(a, s, H, I, Kernels.top_k, .swiglu_f32, &diag) catch |e| {
        std.debug.print("beam nvfp4 install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer f.deinit();
    try testing.expectEqual(Kernels.max_pairs, f.installed.fused.n_prepared);
    std.debug.print("BEAM_NVFP4_SELFCHECK PASS launches={d}\n", .{report.results.items.len});
}

test "beam nvfp4 GPU: the fused qmv texts give the stock decode route's SwiGLU and down rows word for word on the slice's records (waves of 1 to 8 tokens over the three banks)" {
    if (!try windowLocked()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE").?);
    std.debug.print("BEAM_NVFP4_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    const a = std.heap.c_allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    if (!mlx.streamIsGpu(stream)) return error.SkipZigTest;
    var g = try G.init(a, stream);
    defer g.deinit();
    const sl = try openSlice(a, dir);
    defer _ = std.c.close(sl.fd);
    var banks: [3]bank.BankArrays = undefined;
    for (&banks, 0..) |*bk, b| bk.* = try buildBank(&g, sl.fd, &sl.geo, @intCast(b), 0, slots[b]);
    defer for (banks) |bk| releaseBank(&g, bk);
    g.reset();
    var math = try experts.Math.init(a, sl.geo, experts.expertActivation);
    defer math.deinit(&g);
    var diag: sdk.Diag = .{};
    math.fused = decode_kernels.FusedDecode.install(a, stream, H, I, Kernels.top_k, .swiglu_f32, &diag) catch |e| {
        std.debug.print("beam nvfp4 install refused: {s}\n", .{diag.message()});
        return e;
    };

    const Wave = struct { name: []const u8, pairs: u32, base_only: bool };
    const waves = [_]Wave{
        .{ .name = "p1", .pairs = 1, .base_only = true },
        .{ .name = "p3", .pairs = 3, .base_only = false },
        .{ .name = "m1_base", .pairs = 8, .base_only = true },
        .{ .name = "m1_banks", .pairs = 8, .base_only = false },
        .{ .name = "p13", .pairs = 13, .base_only = false },
        .{ .name = "m2", .pairs = 16, .base_only = false },
        .{ .name = "m3", .pairs = 24, .base_only = false },
        .{ .name = "m4", .pairs = 32, .base_only = false },
        .{ .name = "m8", .pairs = 64, .base_only = false },
    };
    var r: Rng = .{ .s = 0x62656164 };
    var words: usize = 0;
    var bad: usize = 0;
    for (waves) |wv| {
        const k = Kernels.top_k;
        const tokens: u32 = (wv.pairs + k - 1) / k;
        var ids: [64]u32 = undefined;
        var tok: [64]i32 = undefined;
        for (0..tokens) |t| {
            const n_t: usize = @min(k, wv.pairs - t * k);
            var row: [8]u32 = undefined;
            var n: usize = 0;
            while (n < n_t) {
                const v = r.next();
                const b: u32 = if (wv.base_only) 0 else v % 3;
                const id: u32 = if (t > 0 and (v >> 8) % 2 == 0) ids[(v >> 12) % k] else (b << 24) | ((v >> 4) % slots[b]);
                if (std.mem.indexOfScalar(u32, row[0..n], id) != null) continue;
                row[n] = id;
                n += 1;
            }
            for (row[0..n_t], 0..) |id, j| {
                ids[t * k + j] = id;
                tok[t * k + j] = @intCast(t);
            }
        }
        const xin = try a.alloc(f32, tokens * H);
        defer a.free(xin);
        for (xin) |*v| v.* = (@as(f32, @floatFromInt(r.next() % 4001)) - 2000.0) / 1000.0;
        const mark = g.mark();
        const xt = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin), &.{ ci(tokens), ci(H) }, .float32), .bfloat16);
        const xp = try g.take(xt, try g.hostArray(std.mem.sliceAsBytes(tok[0..wv.pairs]), &.{ci(wv.pairs)}, .int32), 0);
        const ida = try g.hostArray(std.mem.sliceAsBytes(ids[0..wv.pairs]), &.{ci(wv.pairs)}, .uint32);
        // The stock route: per bank, its pairs in routed order (the scheduler's grouped waves), each row placed back.
        const sh = try a.alloc(f32, wv.pairs * I);
        defer a.free(sh);
        const sy = try a.alloc(f32, wv.pairs * H);
        defer a.free(sy);
        for (0..3) |b| {
            var pos: [64]i32 = undefined;
            var rows: [64]u32 = undefined;
            var n: usize = 0;
            for (ids[0..wv.pairs], 0..) |id, p| if (id >> 24 == b) {
                pos[n] = @intCast(p);
                rows[n] = id & 0xFFFFFF;
                n += 1;
            };
            if (n == 0) continue;
            const xb = try g.take(xp, try g.hostArray(std.mem.sliceAsBytes(pos[0..n]), &.{ci(n)}, .int32), 0);
            const rb = try g.hostArray(std.mem.sliceAsBytes(rows[0..n]), &.{ci(n)}, .uint32);
            const h = try math.gateUp(&g, xb, rb, banks[b].gate, banks[b].up);
            const y = try math.down(&g, h, rb, banks[b].down);
            const hh = try hostOf(a, &g, h);
            defer a.free(hh);
            const yy = try hostOf(a, &g, y);
            defer a.free(yy);
            for (pos[0..n], 0..) |p, j| {
                @memcpy(sh[@as(usize, @intCast(p)) * I ..][0..I], hh[j * I ..][0..I]);
                @memcpy(sy[@as(usize, @intCast(p)) * H ..][0..H], yy[j * H ..][0..H]);
            }
        }
        // The fused route over every bank: the SwiGLU rows, the down text over the stock SwiGLU rows, the chain.
        const sh_bf = try g.astype(try g.hostArray(std.mem.sliceAsBytes(sh), &.{ ci(wv.pairs), ci(I) }, .float32), .bfloat16);
        const fh = try math.gateUpBanked(&g, xp, ida, &banks);
        const fy_alone = try math.downBanked(&g, sh_bf, ida, &banks);
        const fy = try math.downBanked(&g, fh, ida, &banks);
        const fhh = try hostOf(a, &g, fh);
        defer a.free(fhh);
        const fya = try hostOf(a, &g, fy_alone);
        defer a.free(fya);
        const fyy = try hostOf(a, &g, fy);
        defer a.free(fyy);
        const dh = differ(fhh, sh);
        const dya = differ(fya, sy);
        const dy = differ(fyy, sy);
        std.debug.print("BEAM_NVFP4_PARITY wave={s} pairs={d} tokens={d} swiglu_differ={d}/{d} down_alone_differ={d}/{d} chain_differ={d}/{d}\n", .{ wv.name, wv.pairs, tokens, dh, sh.len, dya, sy.len, dy, sy.len });
        words += sh.len + 2 * sy.len;
        bad += dh + dya + dy;
        g.resetTo(mark);
    }
    std.debug.print("BEAM_NVFP4_PARITY {s} words={d} differ={d}\n", .{ if (bad == 0) "PASS" else "FAIL", words, bad });
    try testing.expectEqual(@as(usize, 0), bad);
}
