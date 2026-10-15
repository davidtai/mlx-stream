//! The window's parity test of the FP8 residents route (fp8.zig) at GLM-5.3-Flash's sites, in the pattern of
//! `decode_kernels_parity.zig`: GLM5NEXT_FP8_GPU=1 (else skipped) and _GPU_WINDOW_LOCKED=1 (else refused: the window
//! orchestrator's child). Never run outside a window. MiMo's texts' own parity is MiMo's (`mimo fp8res GPU check`); this
//! runs GLM's route as the trunk calls it (`fp8.project` over [1, M, in] rows, the GEMV's plan per width from `fp8.bind`)
//! on every distinct FP8 site shape of the trunk (13), synthetic codes and scales in the release's range (MiMo's
//! `Fixture`):
//!   - the GEMV at M = 1..8 (decode, the MTP verify's 4, a short prompt call): every sampled output bit for bit MiMo's
//!     kernel-order host emulation (`hostGemvDot` at the site's K partitions, rounded to BF16 once), so every row is its
//!     M = 1 row;
//!   - the prompt route at M = 9: the stock matmul over the host's bf16(code x scale) weight, bit for bit (the BF16 path's
//!     prompt arithmetic), through the route's LRU.
//! Static bound (declared before any run), at the widest site (the DSA o_proj [4096, 16384]), everything freed per site:
//! the fixture's codes on the host and their MLX copy (2 x 67.1 MB; the scales 0.03 MB), the site's BF16 weight three
//! times (the route's cache, the host reference and its MLX copy: 3 x 134.2 MB), rows of at most 9 x 16,384 bf16 and
//! outputs of 9 x 4,096 f32 (< 1 MB): 537 MB, plus the JIT (two texts); declared 0.8 GB.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const fp8 = @import("fp8.zig");
const mimo_fp8 = @import("../mimo_v2/fp8_linear.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops_be.MlxOps;

const use_bound_bytes: u64 = 800_000_000;

test "glm5next fp8 GPU: the residents route at every trunk FP8 site shape: the GEMV at M = 1..8 bit for bit the kernel-order emulation; the prompt route at M = 9 the stock matmul over bf16(code x scale)" {
    _ = std.c.getenv("GLM5NEXT_FP8_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_FP8_USE bound_bytes={d}\n", .{use_bound_bytes});
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const route = try fp8.Route.install(a);
    defer route.destroy(a);
    route.open();
    // The trunk's distinct FP8 site shapes.
    const c: glm.Config = .{};
    var shapes: [32][2]u32 = undefined;
    var n_shapes: usize = 0;
    for (0..c.n_layers) |l| {
        var buf: [7]glm.Fp8Site = undefined;
        const mlp = c.fp8MlpSites(@intCast(l));
        for ([_][]const glm.Fp8Site{ c.fp8Sites(@intCast(l), &buf), &mlp }) |set| for (set) |st| {
            const sh = [2]u32{ st.out, st.in };
            for (shapes[0..n_shapes]) |x| {
                if (x[0] == sh[0] and x[1] == sh[1]) break;
            } else {
                shapes[n_shapes] = sh;
                n_shapes += 1;
            }
        };
    }
    var failed: usize = 0;
    for (shapes[0..n_shapes], 0..) |sh, si| {
        const n = sh[0];
        const k = sh[1];
        const f = try mimo_fp8.Fixture.init(a, n, k, 9, 0x676c_f800 + si);
        defer f.deinit(a);
        const m0 = g.mark();
        defer g.resetTo(m0);
        const w = try g.hostArray(f.codes, &.{ @intCast(n), @intCast(k) }, .uint8);
        const sc = try g.hostArray(std.mem.sliceAsBytes(f.scales), &fp8.scaleShape(n, k), .float32);
        try g.evalAll(&.{ w, sc });
        const lin: fp8.Linear = .{ .fp8 = try fp8.bind(route, w, sc, n, k) };
        const plan = mimo_fp8.gemvPlan(n, k).?;
        // The GEMV at M = 1..8, [1, M, k] rows as the trunk passes them.
        for (1..@as(usize, fp8.gemv_max_rows) + 1) |m| {
            const x = try g.hostArray(std.mem.sliceAsBytes(f.x[0 .. m * k]), &.{ 1, @intCast(m), @intCast(k) }, .bfloat16);
            const y = try g.astype(try fp8.project(&g, lin, x), .float32);
            try g.evalAll(&.{y});
            const got = mlx.mlx_array_data_float32(y) orelse return error.MlxNoData;
            var j: usize = 0;
            while (j < n) : (j += if (j + 7 < n) 7 else 1) {
                for (0..m) |row| {
                    const want = mimo_fp8.f32ToBf16(mimo_fp8.hostGemvDot(plan, f.x[row * k ..][0..k], f.codes[j * k ..][0..k], f.srow(j, k)));
                    const have = mimo_fp8.f32ToBf16(got[row * n + j]);
                    if (want != have) {
                        failed += 1;
                        if (failed <= 8) std.debug.print("GLM5N_FP8_USE gemv [{d}, {d}] M={d} row={d} out={d}: got {x} want {x}\n", .{ n, k, m, row, j, have, want });
                    }
                }
            }
        }
        // The prompt route at M = 9: the stock matmul over the host's bf16(code x scale).
        {
            const wb_host = try a.alloc(u16, @as(usize, n) * k);
            defer a.free(wb_host);
            for (0..n) |r| {
                const srow = f.srow(r, k);
                for (0..k) |i| wb_host[r * k + i] = mimo_fp8.f32ToBf16(mimo_fp8.weight32(f.codes[r * k + i], srow[i / fp8.scale_cols]));
            }
            const wb = try g.hostArray(std.mem.sliceAsBytes(wb_host), &.{ @intCast(n), @intCast(k) }, .bfloat16);
            const x = try g.hostArray(std.mem.sliceAsBytes(f.x[0 .. 9 * k]), &.{ 1, 9, @intCast(k) }, .bfloat16);
            const want = try g.astype(try g.matmul(x, try g.transpose(wb)), .float32);
            const got = try g.astype(try fp8.project(&g, lin, x), .float32);
            try g.evalAll(&.{ want, got });
            const wv = mlx.mlx_array_data_float32(want) orelse return error.MlxNoData;
            const gv = mlx.mlx_array_data_float32(got) orelse return error.MlxNoData;
            for (0..9 * @as(usize, n)) |i| if (@as(u32, @bitCast(wv[i])) != @as(u32, @bitCast(gv[i]))) {
                failed += 1;
                if (failed <= 8) std.debug.print("GLM5N_FP8_USE prompt [{d}, {d}] element {d}: got {e} want {e}\n", .{ n, k, i, gv[i], wv[i] });
            };
            route.cache.release();
        }
        std.debug.print("GLM5N_FP8_USE site [{d}, {d}] ks={d} done\n", .{ n, k, plan.ks });
    }
    std.debug.print("GLM5N_FP8_USE shapes={d} failed={d}\n", .{ n_shapes, failed });
    try testing.expectEqual(@as(usize, 0), failed);
}
