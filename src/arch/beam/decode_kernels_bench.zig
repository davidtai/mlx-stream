//! beam's decode experts bench (window only): BEAM_DECODE_BENCH=<the real-weight slice pack> and _GPU_WINDOW_LOCKED=1
//! (else skipped / refused). No model: real records of the slice's three routed layers scattered
//! into the three slot banks (base 128 slots, ext 8, transient 16), and per arm one routed layer of the decode lane as
//! the shared scheduler builds it, without its routing barrier:
//!   - `stock`: per bank group the pairs' token rows taken, MLX's gather_qmm nvfp4 for gate and up (`Math.gateUp`: the
//!     f32 SwiGLU), for down (`Math.down`); the join (the groups' rows in routed order); the stock combine;
//!   - `fused`: the wave's token rows taken, the qmv texts' gate/up and down over the three banks (`Math.gateUpBanked`,
//!     `downBanked`); the same join and combine.
//! A round chains 49 layers (Beam's routed layers; layer l + 1's x is layer l's output, each layer its own slots), one
//! eval a round, arms interleaved per round (start rotated), 2 warm rounds and 9 timed. Per (case, arm): the median /
//! p10 / p90 us a layer, the step's ms (x 49), the dispatches a layer (by construction) and the effective GB/s (the
//! wave's distinct records' bytes over its time). Cases: one token over the base bank (8 slots), one token with two
//! pairs in the transient bank (two bank groups), 4 tokens (the decode lane's widest call) and 8 tokens (a draft
//! verify), the later tokens sharing half of their slots with the first token's. Then the read probe: one text that
//! reads 4 GiB once (uint4 loads, every lane strided over the buffer), the box's read bandwidth, the floor a wave's bytes
//! are judged against. Lines: BEAM_DECODE_BENCH per (case, arm), BEAM_DECODE_READ, `overall`. The outputs of the two
//! arms are compared bit for bit on each case's last round (BEAM_DECODE_BENCH_EQUAL). Bound: 152 records x 22.1 MB =
//! 3.36 GB of slot arrays, the 4 GiB probe buffer, a round's chain < 0.1 GB, JIT and the binary ~0.4 GB: 8 GB declared.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");
const mtl = @import("../../sdk_ext/metal_launch.zig");
const io_util = sdk.io_util;
const experts = @import("experts.zig");
const route = @import("../mimo_v2/route.zig");
const decode_kernels = @import("decode_kernels.zig");
const bank = experts.bank;

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops.MlxOps;

pub const bound_bytes: u64 = 8_000_000_000;
const H: u32 = 5120;
const I: u32 = 2560;
const K: u32 = 8;
const layers: u32 = 49;
const layers_f: f64 = @floatFromInt(layers);
const rounds: u32 = 9;
const warm: u32 = 2;
const caps = [3]u32{ 128, 8, 16 };
const record_bytes: f64 = 22_118_412;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ci(x: anytype) c_int {
    return @intCast(x);
}

const Banks = [3]bank.BankArrays;

const parity = @import("decode_kernels_parity.zig");

const Arm = enum { stock, fused };

/// A layer's wave: its pairs' ids (`bank << 24 | slot`), each pair's token, the routed order's join.
const Wave = struct { ids: [64]u32, tok: [64]i32, pairs: u32, tokens: u32 };

/// The scheduler's join of one call (`ExpertsWith.join`): the outputs' rows in routed order, [n, k, H].
fn join(g: *G, outs: []const T, pos: []const u32, n: u32) !T {
    const joined = try g.concat(outs, 0);
    var order: [64]u32 = undefined;
    for (pos, 0..) |p, j| order[p] = @intCast(j);
    const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..pos.len]), &.{ci(pos.len)}, .uint32);
    return g.reshape(try g.take(joined, ord, 0), &.{ ci(n), ci(K), ci(H) });
}

/// One routed layer of `arm` over `w`: x [tokens, H] bf16 -> the combined rows [tokens, H] bf16.
fn layer(g: *G, math: *const experts.Math, banks: *const Banks, arm: Arm, x: T, w: *const Wave, weights: T) !T {
    var outs: [3]T = undefined;
    var pos: [64]u32 = undefined;
    var n_outs: usize = 0;
    var n_pos: usize = 0;
    switch (arm) {
        .stock => for (0..3) |b| {
            var tok: [64]i32 = undefined;
            var rows: [64]u32 = undefined;
            var n: usize = 0;
            for (w.ids[0..w.pairs], 0..) |id, p| if (id >> 24 == b) {
                tok[n] = w.tok[p];
                rows[n] = id & 0xFFFFFF;
                pos[n_pos] = @intCast(p);
                n_pos += 1;
                n += 1;
            };
            if (n == 0) continue;
            const xb = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..n]), &.{ci(n)}, .int32), 0);
            const rb = try g.hostArray(std.mem.sliceAsBytes(rows[0..n]), &.{ci(n)}, .uint32);
            const h = try math.gateUp(g, xb, rb, banks[b].gate, banks[b].up);
            outs[n_outs] = try math.down(g, h, rb, banks[b].down);
            n_outs += 1;
        },
        .fused => {
            const xb = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(w.tok[0..w.pairs]), &.{ci(w.pairs)}, .int32), 0);
            const ids = try g.hostArray(std.mem.sliceAsBytes(w.ids[0..w.pairs]), &.{ci(w.pairs)}, .uint32);
            outs[0] = try math.downBanked(g, try math.gateUpBanked(g, xb, ids, banks), ids, banks);
            n_outs = 1;
            for (0..w.pairs) |p| pos[p] = @intCast(p);
            n_pos = w.pairs;
        },
    }
    const routed = try join(g, outs[0..n_outs], pos[0..n_pos], w.tokens);
    return route.combineStock(g, routed, weights, K);
}

/// Dispatches a layer launches, by construction: per bank group (stock) the take, then for each of gate, up and down the
/// gather's global-scale copy (MLX's GatherQMM makes the strided per-slot scales contiguous) and its gather_qmv_fast, and
/// the f32 SwiGLU's five (two widenings, the compiled SiLU, the product, the narrowing); per wave (fused) the take and
/// the two texts. The join (one take, plus a concatenate over more than one output) and the combine (widening,
/// product, sum, narrowing) follow either.
fn dispatches(arm: Arm, groups: u32) u32 {
    const tail: u32 = 1 + @as(u32, @intFromBool(groups > 1 and arm == .stock)) + 4;
    return switch (arm) {
        .stock => groups * (1 + 3 * 2 + 5) + tail,
        .fused => 3 + tail,
    };
}

const Case = struct { name: []const u8, tokens: u32, transient: u32 };
const cases = [_]Case{
    .{ .name = "m1_base", .tokens = 1, .transient = 0 },
    .{ .name = "m1_two_misses", .tokens = 1, .transient = 2 },
    .{ .name = "m4_base", .tokens = 4, .transient = 0 },
    .{ .name = "m8_base", .tokens = 8, .transient = 0 },
};

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u32 {
        r.s = r.s *% 6364136223846793005 +% 1442695040888963407;
        return @truncate(r.s >> 33);
    }
};

/// Layer l's wave of case `cs`: token 0 takes 8 distinct base slots from layer l's window of the base bank (so a
/// round's layers read different records), its last `transient` pairs from the transient bank; each later token keeps
/// half of token 0's slots and draws the rest from the window.
fn waveOf(cs: Case, l: u32, r: *Rng) Wave {
    var w: Wave = .{ .ids = undefined, .tok = undefined, .pairs = cs.tokens * K, .tokens = cs.tokens };
    const window: u32 = 24;
    const base0 = (l * 16) % (caps[0] - window);
    for (0..cs.tokens) |t| {
        var row: [8]u32 = undefined;
        var n: usize = 0;
        while (n < 8) {
            const v = r.next();
            const id: u32 = if (t == 0 and n >= 8 - cs.transient)
                (2 << 24) | (v % caps[2])
            else if (t > 0 and n < 4)
                w.ids[(v >> 8) % 8]
            else
                base0 + v % window;
            if (std.mem.indexOfScalar(u32, row[0..n], id) != null) continue;
            row[n] = id;
            n += 1;
        }
        for (row, 0..) |id, j| {
            w.ids[t * 8 + j] = id;
            w.tok[t * 8 + j] = @intCast(t);
        }
    }
    return w;
}

fn distinctRecords(w: *const Wave) u32 {
    var n: u32 = 0;
    for (w.ids[0..w.pairs], 0..) |v, i| n += @intFromBool(std.mem.indexOfScalar(u32, w.ids[0..i], v) == null);
    return n;
}

fn groupsOf(w: *const Wave) u32 {
    var seen = [3]bool{ false, false, false };
    for (w.ids[0..w.pairs]) |id| seen[id >> 24] = true;
    return @as(u32, @intFromBool(seen[0])) + @intFromBool(seen[1]) + @intFromBool(seen[2]);
}

fn pct(times: []f64, q: f64) f64 {
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    const i: usize = @intFromFloat(@round(q * @as(f64, @floatFromInt(times.len - 1))));
    return times[i];
}

const read_src =
    \\const uint i = thread_position_in_grid.x;
    \\const uint n = threads_per_grid.x;
    \\const device uint4* s = (const device uint4*)src;
    \\uint4 acc = uint4(0);
    \\for (size_t j = i; j < size_t(N4); j += n) acc ^= s[j];
    \\out[i] = acc.x ^ acc.y ^ acc.z ^ acc.w;
;

test "beam decode bench GPU: the fused qmv texts against the stock gathers per routed layer (M 1, 4, 8; one and two bank groups) on the slice's records, and the box's read bandwidth" {
    const dir = std.mem.span(std.c.getenv("BEAM_DECODE_BENCH") orelse return error.SkipZigTest);
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("BEAM_DECODE_BENCH bound_bytes={d} layers={d} rounds={d} warm={d}\n", .{ bound_bytes, layers, rounds, warm });
    const a = std.heap.c_allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    if (!mlx.streamIsGpu(stream)) return error.SkipZigTest;
    var old_limit: usize = 0;
    _ = mlx.mlx_set_cache_limit(&old_limit, 512 << 20);
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_limit);
    }
    var g = try G.init(a, stream);
    defer g.deinit();
    const geo = try bank.inspect(a, testing.io, dir);
    try testing.expect(geo.hidden_size == H and geo.intermediate_size == I and geo.num_moe_layers >= 3 and geo.num_experts >= caps[0]);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
    defer a.free(path);
    const fd = try io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    const t_build = nowNs();
    var banks: Banks = undefined;
    for (&banks, 0..) |*bk, b| bk.* = try parity.buildBank(&g, fd, &geo, @intCast(b), 0, caps[b]);
    defer for (banks) |bk| parity.releaseBank(&g, bk);
    g.reset();
    std.debug.print("BEAM_DECODE_BENCH banks built slots={d} ms={d}\n", .{ caps[0] + caps[1] + caps[2], (nowNs() - t_build) / std.time.ns_per_ms });
    var math = try experts.Math.init(a, geo, experts.expertActivation);
    defer math.deinit(&g);
    var diag: sdk.Diag = .{};
    math.fused = decode_kernels.FusedDecode.install(a, stream, H, I, K, .swiglu_f32, &diag) catch |e| {
        std.debug.print("BEAM_DECODE_BENCH install refused: {s}\n", .{diag.message()});
        return e;
    };

    var all_equal = true;
    for (cases) |cs| {
        _ = mlx.mlx_clear_cache();
        var r: Rng = .{ .s = 0x62656e63 ^ (@as(u64, cs.tokens) << 8) ^ cs.transient };
        var waves: [layers]Wave = undefined;
        var bytes: f64 = 0;
        for (&waves, 0..) |*w, l| {
            w.* = waveOf(cs, @intCast(l), &r);
            bytes += @as(f64, @floatFromInt(distinctRecords(w))) * record_bytes;
        }
        const groups = groupsOf(&waves[0]);
        var xin: [8 * H]f32 = undefined;
        for (xin[0 .. cs.tokens * H]) |*v| v.* = (@as(f32, @floatFromInt(r.next() % 4001)) - 2000.0) / 1000.0;
        const x0 = g.keep(try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin[0 .. cs.tokens * H]), &.{ ci(cs.tokens), ci(H) }, .float32), .bfloat16));
        defer g.release(x0);
        const wts: [8 * K]f32 = @splat(0.125);
        const weights = g.keep(try g.hostArray(std.mem.sliceAsBytes(wts[0 .. cs.tokens * K]), &.{ ci(cs.tokens), ci(K) }, .float32));
        defer g.release(weights);
        try g.evalAll(&.{ x0, weights });
        var times: [2][rounds]f64 = undefined;
        var last: [2]?T = .{ null, null };
        defer for (last) |x| if (x) |v| g.release(v);
        for (0..warm + rounds) |ri| {
            for (0..2) |ai0| {
                const ai = (ai0 + ri) % 2;
                const arm: Arm = @fromBackingInt(@intCast(ai));
                const mark = g.mark();
                var x = x0;
                for (&waves) |*w| x = try layer(&g, &math, &banks, arm, x, w, weights);
                const t0 = nowNs();
                try g.evalAll(&.{x});
                const dt = @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0 / layers_f;
                if (ri >= warm) times[ai][ri - warm] = dt;
                if (ri == warm + rounds - 1) {
                    if (last[ai]) |v| g.release(v);
                    last[ai] = g.keep(x);
                }
                g.resetTo(mark);
            }
        }
        for (0..2) |ai| {
            const arm: Arm = @fromBackingInt(@intCast(ai));
            var ts = times[ai];
            const med = pct(&ts, 0.5);
            std.debug.print("BEAM_DECODE_BENCH case={s} arm={s} tokens={d} pairs={d} groups={d} us_per_layer median={d:.1} p10={d:.1} p90={d:.1} ms_per_step={d:.3} dispatches_per_layer={d} distinct_mb_per_layer={d:.1} gb_s={d:.1}\n", .{ cs.name, @tagName(arm), cs.tokens, cs.tokens * K, groups, med, pct(&ts, 0.1), pct(&ts, 0.9), med * layers_f / 1000.0, dispatches(arm, groups), bytes / layers_f / 1e6, bytes / layers_f / (med * 1e3) });
        }
        // The last round's outputs of both arms, bit for bit (the chain of 49 layers through both routes).
        const n_out: usize = cs.tokens * H;
        const hs = try a.alloc(f32, 2 * n_out);
        defer a.free(hs);
        for (0..2) |ai| _ = try g.hostF32(try g.astype(last[ai].?, .float32), hs[ai * n_out ..][0..n_out]);
        g.reset();
        var differ: usize = 0;
        for (hs[0..n_out], hs[n_out..]) |p, q| differ += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
        std.debug.print("BEAM_DECODE_BENCH_EQUAL case={s} words={d} differ={d}\n", .{ cs.name, n_out, differ });
        if (differ != 0) all_equal = false;
    }

    // ── The read probe: 4 GiB read once per launch ──
    const n4: u64 = (4 << 30) / 16;
    const buf = g.keep(try g.zeros(&.{ci(n4 * 4)}, .uint32));
    defer g.release(buf);
    try g.evalAll(&.{buf});
    const in_names = [_][*:0]const u8{"src"};
    const out_names = [_][*:0]const u8{"out"};
    const k = kset.newKernel("beam_decode_read_probe", &in_names, &out_names, read_src, "", true) orelse return error.BeamBenchKernel;
    defer mtl.mlx_fast_metal_kernel_free(k);
    const threads: u32 = 1 << 18;
    const oshape = [_]c_int{ci(threads)};
    const oshapes = [_][]const c_int{&oshape};
    const tmpl = [_]kset.TemplateArg{.{ .name = "N4", .value = .{ .int = @intCast(n4) } }};
    const cfg = try kset.buildConfig(.{ .grid = .{ threads, 1, 1 }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &oshapes, .out_dtypes = &.{.uint32}, .template = &tmpl });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var best: f64 = 0;
    var rt: [7]f64 = undefined;
    for (0..8) |ri| {
        var outs: [1]T = undefined;
        try kset.applyConfig(k, cfg, &.{buf}, &outs, stream);
        const o = try g.adopt(outs[0]);
        const t0 = nowNs();
        try g.evalAll(&.{o});
        const s = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
        if (ri > 0) rt[ri - 1] = @as(f64, @floatFromInt(n4 * 16)) / s / 1e9;
        g.reset();
    }
    for (rt) |v| best = @max(best, v);
    std.debug.print("BEAM_DECODE_READ bytes={d} gb_s median={d:.1} best={d:.1}\n", .{ n4 * 16, pct(&rt, 0.5), best });
    std.debug.print("BEAM_DECODE_BENCH overall {s}\n", .{if (all_equal) "PASS" else "FAIL"});
    try testing.expect(all_equal);
}
