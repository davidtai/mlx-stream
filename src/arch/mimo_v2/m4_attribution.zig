//! M4 attribution (mimo/m4-attribution): where the MTP verify's 4-row forward spends its GPU time against the serial
//! one-row forward, per component, on the real shapes. The device family `m4attr` (pre-ship form: its own ReleaseFast
//! test binary, `MIMO_M4ATTR_GPU=1` + `_GPU_WINDOW_LOCKED=1`, a declared memory bound) times, with Metal's
//! command-buffer GPUStartTime / GPUEndTime (csrc/mimo_gpu_probe.inc, test builds only), at M = 1 and M = 4:
//!   (a) the resident BF16 trunk projections of a full layer (5) and a sliding layer (1), the dense layer 0's MLP and
//!       the head, read from the pack's model.safetensors (F_NOCACHE);
//!   (b) decode attention over a 17,408-key full-layer KV and a sliding view (127 + M keys), kernels route and stock;
//!   (c) the fused decode MoE (host ids `fused` and the device-LUT `hitpass` wave) on layer 5's real expert records,
//!       8 resident pairs (M = 1) vs 32 pairs over 32 / 24 / 12 distinct slots (M = 4), the per-pair banked texts
//!       (the control's) and the union texts (lever 1);
//!   (d) the router, the join + combine, the per-layer glue (norms, rope, value scale, residuals, KV writes) and the
//!       dispatch floor (an empty command buffer; one tiny dispatch inside a chain).
//! Each sample: a 256 MiB scrub read (evicts the SLC) outside the probe, then one eval of the component's graph with
//! the probe armed; the sample is the GPU busy union of the buffers it committed (wait-only rows started at the
//! queue's arrival, as `dsv41_verify_timeline.repairRows`). Median of `reps` after `warm` unrecorded runs. One table
//! (`MIMO_M4ATTR row=...`) and the sums against the measured barriers (42.8 ms serial, 102.6 ms verify; missfree
//! BF16 cells, decode_attention stock, hitpass), the residual stated.
const std = @import("std");
const mlx = @import("sdk").mlx;
const io_util = @import("sdk").io_util;
const ops = @import("../deepseek_v41/ops.zig");
const mimo = @import("config.zig");
const module = @import("module.zig");
const moe = @import("moe.zig");
const bank_mod = @import("bank.zig");
const decode_attention = @import("decode_attention.zig");
const math_mod = @import("math.zig");
const decode_barrier = @import("decode_barrier.zig");
const source_mod = @import("source.zig");

const T = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const testing = std.testing;

/// The measured barriers the table sums against (prof-mtp-bf16-20261008, missfree cells: serial cycle barrier, MTP
/// verify barrier).
pub const measured_serial_barrier_ms: f64 = 42.820;
pub const measured_verify_barrier_ms: f64 = 102.626;
/// The served shapes: a 17,408-key context (the cells' MIMO_CTX), the sliding window, the verify rows.
pub const keys: u32 = 17_408;
pub const verify_rows: u32 = 4;
const full_layer: u32 = 5;
const sliding_layer: u32 = 1;
const scrub_bytes: u64 = 256 << 20;
const probe_cap: u32 = 4096;
const reps: u32 = 9;
const warm: u32 = 2;

// ── the command-buffer probe (csrc/mimo_gpu_probe.inc; linked into test binaries only) ──

const probe = struct {
    extern fn mimo_gpu_probe_install() i32;
    extern fn mimo_gpu_probe_arm(rows: [*]f64, cap: u32) void;
    extern fn mimo_gpu_probe_disarm() void;
    extern fn mimo_gpu_probe_counts(out: *[4]u32) void;
    extern fn mimo_gpu_probe_selftest() i32;
};

/// One recorded buffer (s, mach-absolute host clock): host commit, GPU start (0 when Metal gave none), GPU end.
pub const Row = [3]f64;

/// The GPU busy time (s) of `rows` (commit order): the union of [start, end]; a row with no start but an end ran no
/// kernel (an event wait) and starts when the queue reached it: the later of its commit and the previous row's end.
pub fn busyUnion(rows: []const Row, spans: [][2]f64) f64 {
    var n: usize = 0;
    var prev_end: f64 = 0;
    for (rows) |r| {
        if (r[2] <= 0) continue;
        const start = if (r[1] > 0) r[1] else @max(r[0], prev_end);
        prev_end = @max(prev_end, r[2]);
        if (r[2] <= start or n == spans.len) continue;
        spans[n] = .{ start, r[2] };
        n += 1;
    }
    std.mem.sort([2]f64, spans[0..n], {}, struct {
        fn lt(_: void, x: [2]f64, y: [2]f64) bool {
            return x[0] < y[0];
        }
    }.lt);
    var busy: f64 = 0;
    var cur: ?[2]f64 = null;
    for (spans[0..n]) |sp| {
        if (cur) |c| {
            if (sp[0] <= c[1]) {
                cur = .{ c[0], @max(c[1], sp[1]) };
                continue;
            }
            busy += c[1] - c[0];
        }
        cur = sp;
    }
    if (cur) |c| busy += c[1] - c[0];
    return busy;
}

test "mimo m4 attribution CPU: the probe hooks a fake buffer hierarchy (no Metal object) and the busy union repairs wait-only rows" {
    try testing.expectEqual(@as(i32, 0), probe.mimo_gpu_probe_selftest());
    var spans: [8][2]f64 = undefined;
    // two overlapping kernels, a wait-only row (start 0) queued behind them, a kernel after it, an empty row
    const rows = [_]Row{ .{ 1.0, 1.0, 2.0 }, .{ 1.1, 1.5, 3.0 }, .{ 1.2, 0, 4.0 }, .{ 1.3, 4.0, 4.5 }, .{ 1.4, 5.0, 5.0 } };
    try testing.expectApproxEqAbs(@as(f64, 3.5), busyUnion(&rows, &spans), 1e-12);
    // a wait-only row whose queue arrival is its own commit (an idle queue)
    const idle = [_]Row{ .{ 1.0, 1.0, 1.5 }, .{ 2.0, 0, 2.25 } };
    try testing.expectApproxEqAbs(@as(f64, 0.75), busyUnion(&idle, &spans), 1e-12);
}

// ── the device family ──

/// Static bound (declared before any device work): device arrays = head 1,250 MB + dense layer 0 MLP 403 MB + the two
/// layers' projections 367 MB + router f32/bf16 copies 6.3 MB + 24 expert records 321 MB + the 17,408-key slab 44.6 MB
/// + the sliding buffer 3.4 MB + the scrub 268 MB + outputs and MoE / attention transients < 64 MB; host staging peaks
/// at the head (1,250 MB, freed before the next tensor). JIT and allocator slack 512 MiB.
pub fn bound() u64 {
    const c: mimo.Config = .{};
    const H: u64 = c.hidden_size;
    const head = @as(u64, c.vocab_size) * H * 2;
    const dense = 3 * @as(u64, c.intermediate_size) * H * 2;
    const proj = struct {
        fn f(cc: mimo.Config, kv: u64) u64 {
            const h: u64 = cc.hidden_size;
            return 2 * (h * cc.n_heads * cc.head_dim + 2 * h * kv * cc.head_dim / 2 + h * kv * cc.value_head_dim + h * cc.n_heads * cc.value_head_dim);
        }
    }.f;
    const projections = proj(c, c.n_kv_heads) + proj(c, c.swa_n_kv_heads);
    const routers = @as(u64, c.n_routed_experts) * H * (4 + 2);
    const experts = 24 * @as(u64, 13_369_344);
    const slab = @as(u64, c.n_kv_heads) * keys * (c.head_dim + c.value_head_dim) * 2;
    const sliding = @as(u64, c.swa_n_kv_heads) * 656 * (c.head_dim + c.value_head_dim) * 2;
    // (e)'s chains: should a slab update not donate, `chain_layers` copies of the full slab live until the sample's eval.
    const transient: u64 = (64 << 20) + chain_layers * slab;
    const device = head + dense + projections + routers + experts + slab + sliding + scrub_bytes + transient;
    return device + head + (512 << 20);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Sample = struct { gpu_ms: f64, wall_ms: f64, buffers: u32 };

const Bench = struct {
    a: std.mem.Allocator,
    g: *ops.MlxOps,
    s: Stream,
    scrub: T,
    rows: []Row,
    spans: [][2]f64,

    fn scrubCache(b: *Bench) !void {
        const m = b.g.mark();
        defer b.g.resetTo(m);
        try b.g.evalAll(&.{try b.g.sum(b.scrub, 0, false)});
        try mlx.check(mlx.mlx_synchronize(b.s));
    }

    /// One eval of `outs` with the probe armed (the stream idle before and after).
    fn sample(b: *Bench, outs: []const T) !Sample {
        try mlx.check(mlx.mlx_synchronize(b.s));
        probe.mimo_gpu_probe_arm(@ptrCast(b.rows.ptr), @intCast(b.rows.len));
        const t0 = nowNs();
        try b.g.evalAll(outs);
        try mlx.check(mlx.mlx_synchronize(b.s));
        const t1 = nowNs();
        probe.mimo_gpu_probe_disarm();
        var k: [4]u32 = undefined;
        var spins: u32 = 0;
        while (true) {
            probe.mimo_gpu_probe_counts(&k);
            if (k[1] >= @min(k[0], b.rows.len)) break;
            spins += 1;
            if (spins > 2_000_000) return error.MimoM4AttrProbePending;
            std.Thread.yield() catch {};
        }
        if (k[0] == 0) return error.MimoM4AttrProbeSawNoCommit;
        if (k[2] != 0) return error.MimoM4AttrProbeDropped;
        const busy = busyUnion(b.rows[0..k[0]], b.spans);
        return .{ .gpu_ms = busy * 1e3, .wall_ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6, .buffers = k[0] };
    }

    /// Median over `reps` samples of `comp.graph` (each built fresh, after a scrub), after `warm` unrecorded runs.
    fn measure(b: *Bench, name: []const u8, m: u32, comp: anytype) !Sample {
        return b.measureIn(comp) catch |e| {
            var buf: [1024]u8 = undefined;
            const msg = mlx.takeError(&buf) orelse "";
            std.debug.print("MIMO_M4ATTR FAIL component={s} M={d} err={s} mlx=\"{s}\"\n", .{ name, m, @errorName(e), msg });
            return e;
        };
    }

    fn measureIn(b: *Bench, comp: anytype) !Sample {
        var gpu: [reps]f64 = undefined;
        var wall: [reps]f64 = undefined;
        var bufs: u32 = 0;
        for (0..warm + reps) |i| {
            try b.scrubCache();
            const m = b.g.mark();
            var outs: [8]T = undefined;
            const n = try comp.graph(b.g, &outs);
            const smp = try b.sample(outs[0..n]);
            b.g.resetTo(m);
            if (i < warm) continue;
            gpu[i - warm] = smp.gpu_ms;
            wall[i - warm] = smp.wall_ms;
            bufs = smp.buffers;
        }
        std.mem.sort(f64, &gpu, {}, std.sort.asc(f64));
        std.mem.sort(f64, &wall, {}, std.sort.asc(f64));
        return .{ .gpu_ms = gpu[reps / 2], .wall_ms = wall[reps / 2], .buffers = bufs };
    }
};

// ── loading the pack's tensors (F_NOCACHE: no page cache left behind) ──

const Pack = struct {
    a: std.mem.Allocator,
    fd: std.c.fd_t,
    data_start: u64,
    header: std.json.Parsed(std.json.Value),

    fn open(a: std.mem.Allocator, dir: []const u8) !Pack {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
        defer a.free(path);
        const fd = try io_util.openNoCache(path.ptr, .{});
        errdefer _ = std.c.close(fd);
        var prefix: [8]u8 = undefined;
        try readAt(fd, &prefix, 0);
        const len = std.mem.readInt(u64, &prefix, .little);
        if (len < 2 or len > (64 << 20)) return error.MimoM4AttrPackHeader;
        const text = try a.alloc(u8, @intCast(len));
        defer a.free(text);
        try readAt(fd, text, 8);
        const header = try std.json.parseFromSlice(std.json.Value, a, text, .{ .allocate = .alloc_always });
        return .{ .a = a, .fd = fd, .data_start = 8 + len, .header = header };
    }

    fn close(p: *Pack) void {
        p.header.deinit();
        _ = std.c.close(p.fd);
    }

    /// A resident tensor as a kept MLX array (dtype and shape checked against the expectation).
    fn tensor(p: *Pack, g: *ops.MlxOps, name: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !T {
        const d = p.header.value.object.get(name) orelse return error.MimoM4AttrTensorMissing;
        const offs = d.object.get("data_offsets").?.array.items;
        const start: u64 = @intCast(offs[0].integer);
        const end: u64 = @intCast(offs[1].integer);
        var want: u64 = switch (dt) {
            .bfloat16 => 2,
            .float32 => 4,
            else => return error.MimoM4AttrDtype,
        };
        for (shape) |x| want *= @intCast(x);
        if (end - start != want) return error.MimoM4AttrTensorShape;
        const host = try p.a.alloc(u8, @intCast(want));
        defer p.a.free(host);
        try readAt(p.fd, host, p.data_start + start);
        const arr = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), dt);
        if (arr.ctx == null) return error.MlxError;
        defer _ = mlx.mlx_array_free(arr);
        const kept = g.keep(arr);
        try g.evalAll(&.{kept});
        return kept;
    }
};

fn readAt(fd: std.c.fd_t, bytes: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes[done..].ptr, bytes.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
        if (n <= 0) return error.MimoM4AttrRead;
        done += @intCast(n);
    }
}

/// 24 of layer 5's expert records (experts 0..23) as one bank's slot arrays (slot i = expert i), read F_NOCACHE.
fn expertBank(a: std.mem.Allocator, g: *ops.MlxOps, dir: []const u8, slots: u32) !moe.Arrays {
    const geometry = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
    defer a.free(path);
    const fd = try io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    var parts: [bank_mod.n_components][]u8 = undefined;
    var n_parts: usize = 0;
    defer for (parts[0..n_parts]) |x| a.free(x);
    for (geometry.segments, 0..) |seg, c| {
        parts[c] = try a.alloc(u8, @as(usize, @intCast(seg.length)) * slots);
        n_parts += 1;
    }
    const record = try a.alloc(u8, @intCast(geometry.record_bytes));
    defer a.free(record);
    for (0..slots) |e| {
        try readAt(fd, record, geometry.recordOffset(full_layer, @intCast(e)));
        for (geometry.segments, 0..) |seg, c| {
            const len: usize = @intCast(seg.length);
            const off: usize = @intCast(seg.offset);
            @memcpy(parts[c][e * len ..][0..len], record[off..][0..len]);
        }
    }
    var arrays: [bank_mod.n_components]T = undefined;
    for (geometry.segments, 0..) |seg, c| {
        const shape = [_]c_int{ @intCast(slots), @intCast(seg.shape[0]), @intCast(seg.shape[1]) };
        const arr = mlx.mlx_array_new_data(parts[c].ptr, &shape, 3, bank_mod.mlxDtype(seg.dtype));
        if (arr.ctx == null) return error.MlxError;
        defer _ = mlx.mlx_array_free(arr);
        arrays[c] = g.keep(arr);
    }
    try g.evalAll(&arrays);
    return .{ .gate = .{ .w = arrays[0], .scales = arrays[1] }, .up = .{ .w = arrays[2], .scales = arrays[3] }, .down = .{ .w = arrays[4], .scales = arrays[5] } };
}

fn randomBf16(a: std.mem.Allocator, g: *ops.MlxOps, shape: []const c_int, scale: f32, seed: u32) !T {
    var n: usize = 1;
    for (shape) |x| n *= @intCast(x);
    const v = try a.alloc(f32, n);
    defer a.free(v);
    var st: u32 = seed | 1;
    for (v) |*x| {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        x.* = (@as(f32, @floatFromInt(st % 2001)) - 1000) / 1000 * scale;
    }
    const f = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    const m = g.mark();
    const b = g.keep(try g.astype(f, .bfloat16));
    try g.evalAll(&.{b});
    g.resetTo(m);
    return b;
}

// ── the components ──

const Proj = struct { q: T, k: T, v: T, o: T };

/// (a) a layer's four projections at M rows (the dense route: `x @ w.T`, MLX's gemv at M = 1, wide gemv at M = 4).
const Projections = struct {
    w: Proj,
    x: T,
    xo: T,
    fn graph(c: *const Projections, g: *ops.MlxOps, outs: *[8]T) !usize {
        outs[0] = try g.matmul(c.x, try g.transpose(c.w.q));
        outs[1] = try g.matmul(c.x, try g.transpose(c.w.k));
        outs[2] = try g.matmul(c.x, try g.transpose(c.w.v));
        outs[3] = try g.matmul(c.xo, try g.transpose(c.w.o));
        return 4;
    }
};

/// (a) one dense matrix (the head) at M rows.
const Matmul = struct {
    w: T,
    x: T,
    fn graph(c: *const Matmul, g: *ops.MlxOps, outs: *[8]T) !usize {
        outs[0] = try g.matmul(c.x, try g.transpose(c.w));
        return 1;
    }
};

/// (a) the dense layer 0's MLP (module `Dense.forward`).
const DenseMlp = struct {
    gate: T,
    up: T,
    down: T,
    x: T,
    s: Stream,
    fn graph(c: *const DenseMlp, g: *ops.MlxOps, outs: *[8]T) !usize {
        const gt = try g.matmul(c.x, try g.transpose(c.gate));
        const up = try g.matmul(c.x, try g.transpose(c.up));
        const silu = try g.adopt(try module.siluBf16(gt, c.s));
        const mid = try g.mul(silu, up);
        outs[0] = try g.matmul(mid, try g.transpose(c.down));
        return 1;
    }
};

/// (b) one layer's M <= 8 attention through an installed route (`Route.run`, as `Attention.forward` calls it).
const Attn = struct {
    route: decode_attention.Route,
    q: T,
    k: T,
    v: T,
    mask: T,
    sinks: T,
    s: Stream,
    fn graph(c: *const Attn, g: *ops.MlxOps, outs: *[8]T) !usize {
        const scale = 1 / @sqrt(@as(f32, 192));
        outs[0] = try g.adopt(try c.route.run(&c.route, c.q, c.k, c.v, scale, c.mask, c.sinks, c.s));
        return 1;
    }
};

/// (c) one routed layer's fused decode MoE call: ids (host-built `fused`, or the device LUT take `hitpass`), the
/// pairs' rows, gate/up + down texts, the join's take and the combine.
const Moe = struct {
    fused: *const moe.Fused,
    banks: *const [moe.n_banks]moe.Arrays,
    xf: T,
    weights: T,
    ids_host: []const u32,
    experts_host: []const u32,
    lut: T,
    hitpass: bool,
    m: u32,
    fn graph(c: *const Moe, g: *ops.MlxOps, outs: *[8]T) !usize {
        const p: c_int = @intCast(c.ids_host.len);
        var tok: [moe.max_pairs]i32 = undefined;
        var ord: [moe.max_pairs]u32 = undefined;
        for (0..c.ids_host.len) |i| {
            tok[i] = @intCast(i / moe.top_k);
            ord[i] = @intCast(i);
        }
        const ids = if (c.hitpass)
            try g.take(c.lut, try g.hostArray(std.mem.sliceAsBytes(c.experts_host), &.{p}, .uint32), 0)
        else
            try g.hostArray(std.mem.sliceAsBytes(c.ids_host), &.{p}, .uint32);
        const x = try g.take(c.xf, try g.hostArray(std.mem.sliceAsBytes(tok[0..c.ids_host.len]), &.{p}, .int32), 0);
        const rows = try c.fused.down(g, try c.fused.gateUp(g, x, ids, c.banks), ids, c.banks);
        const joined = try g.take(rows, try g.hostArray(std.mem.sliceAsBytes(ord[0..c.ids_host.len]), &.{p}, .uint32), 0);
        outs[0] = try c.fused.combineRows(g, try g.reshape(joined, &.{ @intCast(c.m), @intCast(moe.top_k), 4096 }), c.weights);
        return 1;
    }
};

/// (c) K1 (mimo/k1-moe-bw): the wave's two texts alone (gate/up + SiLU, down) on prebuilt device inputs (each pair's
/// token row, the packed ids): the kernels' GPU time against the bytes of the wave's distinct slots.
const MoeKernels = struct {
    fused: *const moe.Fused,
    banks: *const [moe.n_banks]moe.Arrays,
    x: T,
    ids: T,
    fn graph(c: *const MoeKernels, g: *ops.MlxOps, outs: *[8]T) !usize {
        outs[0] = try c.fused.down(g, try c.fused.gateUp(g, c.x, c.ids, c.banks), c.ids, c.banks);
        return 1;
    }
};

/// K1: one slot record's bytes (gate + up + down codes and scales, MXFP4 g32 at 4096 x 2048).
pub const record_bytes: f64 = 13_369_344;

/// (c) K1's third column, MLX's own kernels on the same waves (David: no duplicate of MLX without a measured reason):
/// `mlx_gather_qmm` (mode mxfp4, group 32: the `"decode_moe": "stock"` route's kernels; one x row per pair, M = 1 per
/// gather, MLX's gather_qmv path at these shapes) over one bank (every slot of the family's waves is in the base bank:
/// MLX's best case, one gather per projection, no bank grouping). `act`: gate + up, the arch's activation
/// (`source.expertActivation`, what a stock route runs between them), down: the full equivalent of the two texts. `!act`:
/// the three gathers alone (down reads gate's rows), MLX's weight stream without the elementwise dispatches.
const MoeStock = struct {
    bank: *const moe.Arrays,
    x: T,
    rows: T,
    act: bool,
    fn graph(c: *const MoeStock, g: *ops.MlxOps, outs: *[8]T) !usize {
        const p = g.shapeOf(c.x).d[0];
        const hid: c_int = @intCast(moe.hidden);
        const inter_c: c_int = @intCast(moe.inter);
        const xr = try g.reshape(c.x, &.{ p, 1, hid });
        const gg = try g.reshape(try g.gatherQmm(xr, c.bank.gate.w, c.bank.gate.scales.?, c.rows, .mxfp4), &.{ p, inter_c });
        const uu = try g.reshape(try g.gatherQmm(xr, c.bank.up.w, c.bank.up.scales.?, c.rows, .mxfp4), &.{ p, inter_c });
        const h = if (c.act) try source_mod.expertActivation(g, gg, uu) else gg;
        outs[0] = try g.reshape(try g.gatherQmm(try g.reshape(h, &.{ p, 1, inter_c }), c.bank.down.w, c.bank.down.scales.?, c.rows, .mxfp4), &.{ p, hid });
        if (c.act) return 1;
        outs[1] = uu;
        return 2;
    }
};

/// (d) the router (module `route`): f32 projection, sigmoid, corrected top-8, normalized weights.
const Router = struct {
    x: T,
    w_t: T,
    bias: T,
    s: Stream,
    fn graph(c: *const Router, g: *ops.MlxOps, outs: *[8]T) !usize {
        const r = try module.route(c.x, c.w_t, c.bias, 8, c.s);
        outs[0] = try g.adopt(r.ids);
        outs[1] = try g.adopt(r.scores);
        return 2;
    }
};

/// (d) a layer's glue outside the projections, attention and MLP (`Layer.forward` / `Attention.forward`): pre norm,
/// heads views, partial rope on q and k, value scale, the KV write (the full layer's slab update, donated as the
/// module's; the sliding layer's ring write), the output transpose, residual add, post norm, the MLP residual add.
/// `attn` is the attention's heads output [1, 64, M, 128] (transposed and flattened here, the o projection's input);
/// `attn_o` the o projection's output [M, H] (the residual's addend; the projection itself is component a).
const Glue = struct {
    x: T,
    attn_o: T,
    qp: T,
    kp: T,
    vp: T,
    attn: T,
    mlp: T,
    pre: T,
    post: T,
    ones: T,
    value_scale: T,
    rope: module.RotaryTables,
    kv_heads: u32,
    slab_k: *T,
    slab_v: *T,
    pos: u32,
    m: u32,
    s: Stream,
    fn graph(c: *const Glue, g: *ops.MlxOps, outs: *[8]T) !usize {
        const n: c_int = @intCast(c.m);
        const normed = try g.adopt(try module.rmsNorm(c.x, c.pre, c.ones, 1e-6, c.s));
        const qt = try g.adopt(try module.headsView(c.qp, n, 64, 192, c.s));
        const kt = try g.adopt(try module.headsView(c.kp, n, c.kv_heads, 192, c.s));
        const vt = try g.adopt(try module.headsView(c.vp, n, c.kv_heads, 128, c.s));
        const q = try g.adopt(try module.partialRope(qt, c.rope, 64, c.s));
        const k = try g.adopt(try module.partialRope(kt, c.rope, 64, c.s));
        const v = try g.adopt(try module.scaleValues(vt, c.value_scale, c.s));
        var nk: T = .{};
        var nv: T = .{};
        try mlx.check(mlx.mlx_slice_update(&nk, c.slab_k.*, k, &.{ 0, 0, @intCast(c.pos), 0 }, 4, &.{ 1, @intCast(c.kv_heads), @intCast(c.pos + c.m), 192 }, 4, &.{ 1, 1, 1, 1 }, 4, c.s));
        try mlx.check(mlx.mlx_slice_update(&nv, c.slab_v.*, v, &.{ 0, 0, @intCast(c.pos), 0 }, 4, &.{ 1, @intCast(c.kv_heads), @intCast(c.pos + c.m), 128 }, 4, &.{ 1, 1, 1, 1 }, 4, c.s));
        // the module drops the cache's previous handle before the eval (its update donates the slab)
        _ = mlx.mlx_array_free(c.slab_k.*);
        _ = mlx.mlx_array_free(c.slab_v.*);
        c.slab_k.* = nk;
        c.slab_v.* = nv;
        const tr = try g.adopt(try module.transpose4(c.attn, c.s));
        const flat = try g.reshape(tr, &.{ n, 64 * 128 });
        const residual = try g.add(c.x, c.attn_o);
        const mlp_in = try g.adopt(try module.rmsNorm(residual, c.post, c.ones, 1e-6, c.s));
        outs[0] = normed;
        outs[1] = q;
        outs[2] = mlp_in;
        outs[3] = try g.add(residual, c.mlp);
        outs[4] = nk;
        outs[5] = nv;
        outs[6] = try g.astype(flat, .bfloat16); // the o projection's contiguous input
        return 7;
    }
};

/// (d) the dispatch floor: one tiny dispatch alone (an otherwise empty command buffer), or `n` chained in one eval.
const Tiny = struct {
    x: T,
    n: u32,
    fn graph(c: *const Tiny, g: *ops.MlxOps, outs: *[8]T) !usize {
        var y = c.x;
        for (0..c.n) |_| y = try g.add(y, c.x);
        outs[0] = y;
        return 1;
    }
};

// ── (e) the routed layer as the server runs it (mimo/barrier-residual) ──

/// Layers per chained sample: each layer's input is the previous one's output, so the host's per-layer round trip (the
/// routing barrier's wait and reads, the route's host work, the next layer's graph build) sits between the GPU's work
/// as in the served verify.
pub const chain_layers: u32 = 6;

/// The lookahead predictor's graph: the control's (`gemm`: the [M, 4096] x [4096, 256] F32 product, op for op as
/// `Math.predictorScores` before mimo/barrier-residual) or the installed one (`rows`: `Math.predictorScores` now).
pub const Predictor = enum { gemm, rows };

fn predictorOf(g: *ops.MlxOps, which: Predictor, x: T, router_t: T, bias: T) !T {
    return switch (which) {
        .gemm => g.add(try g.sigmoid(try g.matmul(try g.astype(x, .float32), router_t)), bias),
        .rows => math_mod.Math.predictorScores(g, x, router_t, bias),
    };
}

/// The 8 largest of `v` in descending order (ties to the lower index).
fn topEight(v: []const f32, out: *[8]u16) void {
    var used: [256]bool = @splat(false);
    for (out) |*o| {
        var best: u16 = 0;
        var bval: f32 = -std.math.inf(f32);
        for (v, 0..) |e, i| if (!used[i] and e > bval) {
            bval = e;
            best = @intCast(i);
        };
        used[best] = true;
        o.* = best;
    }
}

fn nowS() f64 {
    return @as(f64, @floatFromInt(nowNs())) / 1e9;
}

/// (d) the predictor alone (one graph).
const PredictorGraph = struct {
    which: Predictor,
    x: T,
    router_t: T,
    bias: T,
    fn graph(c: *const PredictorGraph, g: *ops.MlxOps, outs: *[8]T) !usize {
        outs[0] = try predictorOf(g, c.which, c.x, c.router_t, c.bias);
        return 1;
    }
};

/// (e) One routed layer as the server runs it on the measured cells' route (decode_attention stock, HITPASS, the fused
/// texts of the served build): `Layer.forward` (pre norm, the dense q/k/v projections, heads views, partial rope,
/// value scale, the KV write and its view, MLX SDPA with the trunk's array mask, the o projection, residual, post norm),
/// the router (`module.route`) and the lookahead predictor (the NEXT layer's router: layer 5's again here), then
/// `ExpertsWith.run`'s order: the barrier set [ids, scores] committed, the device hit wave (the layer's LUT, the fused
/// gate/up + down) committed behind it, the wait, the host reads of the ids and the scores, the route's host work (the
/// rows' slot union, the other LUT rewritten), the join's take and the fused combine, the residual add (lazy: the next
/// layer's barrier evaluates it). Full layers write a 17,408-row slab at `pos` and read its [0, pos + M) view (strided
/// across heads, as the served cache's); sliding layers the ring buffer's window view (the family's (b) shapes).
const ServerChain = struct {
    w: Proj,
    kv_heads: u32,
    pre: T,
    post: T,
    ones: T,
    value_scale: T,
    sinks: T,
    rope: module.RotaryTables,
    mask: T,
    cache_k: *T,
    cache_v: *T,
    /// The KV write row and the attention view's [first, end) rows.
    pos: u32,
    view_first: u32,
    view_end: u32,
    router_t: T,
    bias: T,
    lut: [2]T,
    lut_host: *const [256]u32,
    fused: *const moe.Fused,
    banks: *const [moe.n_banks]moe.Arrays,
    predictor: Predictor,
    m: u32,
    s: Stream,
    /// K6 / K7 (mimo/k6-commits, decode_barrier.zig): the chain's commits (`.layer`: the device's commit limits raised
    /// around the barrier set's and the hit wave's evals, as the served route) and its barrier wait (`.event`: the poll
    /// of the set's eval event and the in-place reads). The control: MLX's cuts and the blocking wait.
    commits: decode_barrier.Commits = .mlx,
    issue: decode_barrier.MissIssue = .host,
    /// Per layer of the sample in progress (s, UPTIME_RAW: the probe's clock): the barrier set's commit window and the
    /// wait's return.
    bar_lo: [chain_layers]f64 = undefined,
    bar_hi: [chain_layers]f64 = undefined,
    ret: [chain_layers]f64 = undefined,
    union_sum: u64 = 0,

    fn layer(c: *ServerChain, g: *ops.MlxOps, li: usize, x: T) !T {
        const n: c_int = @intCast(c.m);
        const H: c_int = 4096;
        const kv: c_int = @intCast(c.kv_heads);
        const normed = try g.adopt(try module.rmsNorm(x, c.pre, c.ones, 1e-6, c.s));
        const qp = try g.matmul(normed, try g.transpose(c.w.q));
        const kp = try g.matmul(normed, try g.transpose(c.w.k));
        const vp = try g.matmul(normed, try g.transpose(c.w.v));
        const q = try g.adopt(try module.partialRope(try g.adopt(try module.headsView(qp, n, 64, 192, c.s)), c.rope, 64, c.s));
        const k = try g.adopt(try module.partialRope(try g.adopt(try module.headsView(kp, n, c.kv_heads, 192, c.s)), c.rope, 64, c.s));
        const v = try g.adopt(try module.scaleValues(try g.adopt(try module.headsView(vp, n, c.kv_heads, 128, c.s)), c.value_scale, c.s));
        var nk: T = .{};
        var nv: T = .{};
        try mlx.check(mlx.mlx_slice_update(&nk, c.cache_k.*, k, &.{ 0, 0, @intCast(c.pos), 0 }, 4, &.{ 1, kv, @intCast(c.pos + c.m), 192 }, 4, &.{ 1, 1, 1, 1 }, 4, c.s));
        try mlx.check(mlx.mlx_slice_update(&nv, c.cache_v.*, v, &.{ 0, 0, @intCast(c.pos), 0 }, 4, &.{ 1, kv, @intCast(c.pos + c.m), 128 }, 4, &.{ 1, 1, 1, 1 }, 4, c.s));
        // the cache drops its previous handle (its update donates the buffer), as the module's
        _ = mlx.mlx_array_free(c.cache_k.*);
        _ = mlx.mlx_array_free(c.cache_v.*);
        c.cache_k.* = nk;
        c.cache_v.* = nv;
        const vk = try g.slice(nk, &.{ 0, 0, @intCast(c.view_first), 0 }, &.{ 1, kv, @intCast(c.view_end), 192 }, &.{ 1, 1, 1, 1 });
        const vv = try g.slice(nv, &.{ 0, 0, @intCast(c.view_first), 0 }, &.{ 1, kv, @intCast(c.view_end), 128 }, &.{ 1, 1, 1, 1 });
        const attn = try g.adopt(try decode_attention.stock.run(&decode_attention.stock, q, vk, vv, 1 / @sqrt(@as(f32, 192)), c.mask, c.sinks, c.s));
        const flat = try g.reshape(try g.adopt(try module.transpose4(attn, c.s)), &.{ n, 64 * 128 });
        const residual = try g.add(x, try g.matmul(flat, try g.transpose(c.w.o)));
        const mlp_in = try g.adopt(try module.rmsNorm(residual, c.post, c.ones, 1e-6, c.s));
        const routes = try module.route(mlp_in, c.router_t, c.bias, 8, c.s);
        const ids = try g.adopt(routes.ids);
        const weights = try g.adopt(routes.scores);
        const idx = try g.astype(ids, .int32);
        const sc = try predictorOf(g, c.predictor, mlp_in, c.router_t, c.bias);
        // HITPASS: the barrier set, then the device hit wave behind it, both before the wait (K6 "layer": each of the two
        // evals committed once, the limits restored right after the wave's commit)
        const limits_prev: ?decode_barrier.Limits = if (c.commits == .layer) try decode_barrier.swapLimits(decode_barrier.raised) else null;
        errdefer if (limits_prev) |l| {
            if (decode_barrier.swapLimits(l)) |_| {} else |_| {}
        };
        c.bar_lo[li] = nowS();
        try g.asyncEval(&.{ idx, sc });
        c.bar_hi[li] = nowS();
        const p: c_int = n * 8;
        const parity = li % 2;
        const slots = try g.take(c.lut[parity], try g.reshape(idx, &.{p}), 0);
        var tok: [moe.max_pairs]i32 = undefined;
        var ord: [moe.max_pairs]u32 = undefined;
        for (0..@intCast(p)) |i| {
            tok[i] = @intCast(i / 8);
            ord[i] = @intCast(i);
        }
        const xr = try g.take(mlp_in, try g.hostArray(std.mem.sliceAsBytes(tok[0..@intCast(p)]), &.{p}, .int32), 0);
        const rows = try c.fused.down(g, try c.fused.gateUp(g, xr, slots, c.banks), slots, c.banks);
        try g.asyncEval(&.{rows});
        if (limits_prev) |l| _ = try decode_barrier.swapLimits(l);
        var host_ids: [moe.max_pairs]u16 = undefined;
        var host_sc: [verify_rows * 256]f32 = undefined;
        const got = switch (c.issue) {
            .event => read: {
                // K7: poll the set's eval event, then read in place (no eval call)
                try decode_barrier.pollReady(idx);
                try decode_barrier.pollReady(sc);
                c.ret[li] = nowS();
                _ = try decode_barrier.readF32InPlace(sc, host_sc[0 .. c.m * 256]);
                break :read try decode_barrier.readIdsInPlace(idx, host_ids[0..@intCast(p)]);
            },
            else => read: {
                try g.evalAll(&.{ idx, sc });
                c.ret[li] = nowS();
                const read_ids = try g.hostIds(idx, host_ids[0..@intCast(p)]);
                _ = try g.hostF32(sc, host_sc[0 .. c.m * 256]);
                break :read read_ids;
            },
        };
        // the route's host work: the rows' union of slots; the layer's other LUT rewritten (the same map: no hit
        // changes, so a wave still reading it reads the same words)
        var seen: [256]bool = @splat(false);
        for (got) |e| {
            const sl = c.lut_host[e];
            c.union_sum += @intFromBool(!seen[sl]);
            seen[sl] = true;
        }
        @memcpy(std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(c.lut[1 - parity])))), c.lut_host);
        const joined = try g.take(rows, try g.hostArray(std.mem.sliceAsBytes(ord[0..@intCast(p)]), &.{p}, .uint32), 0);
        const out = try c.fused.combineRows(g, try g.reshape(joined, &.{ n, 8, H }), weights);
        return g.add(residual, out);
    }
};

const ChainSample = struct { gpu_ms: f64, wall_ms: f64, buffers: f64, starved_ms: f64, queued_ms: f64, wake_ms: f64, union_per_layer: f64 };

/// Per layer (medians over `reps` chained samples after `warm`, a scrub before each): GPU busy, host wall, committed
/// buffers, the GPU's idle split into host-starved (the next buffer to run not yet committed) and queued (committed,
/// not yet started), and the barrier's wake (its set's last GPU end to the wait's return).
fn measureChain(b: *Bench, name: []const u8, c: *ServerChain, x0: T) !ChainSample {
    return measureChainIn(b, c, x0) catch |e| {
        var buf: [1024]u8 = undefined;
        const msg = mlx.takeError(&buf) orelse "";
        std.debug.print("MIMO_M4ATTR FAIL component={s} M={d} err={s} mlx=\"{s}\"\n", .{ name, c.m, @errorName(e), msg });
        return e;
    };
}

fn measureChainIn(b: *Bench, c: *ServerChain, x0: T) !ChainSample {
    var cols: [6][reps]f64 = undefined;
    var unions: u64 = 0;
    var layers_seen: u64 = 0;
    const per: f64 = @floatFromInt(chain_layers);
    for (0..warm + reps) |i| {
        try b.scrubCache();
        const mark = b.g.mark();
        c.union_sum = 0;
        try mlx.check(mlx.mlx_synchronize(b.s));
        probe.mimo_gpu_probe_arm(@ptrCast(b.rows.ptr), @intCast(b.rows.len));
        const t0 = nowNs();
        var x = x0;
        for (0..chain_layers) |li| x = try c.layer(b.g, li, x);
        try b.g.evalAll(&.{x});
        try mlx.check(mlx.mlx_synchronize(b.s));
        const t1 = nowNs();
        probe.mimo_gpu_probe_disarm();
        var k: [4]u32 = undefined;
        var spins: u32 = 0;
        while (true) {
            probe.mimo_gpu_probe_counts(&k);
            if (k[1] >= @min(k[0], b.rows.len)) break;
            spins += 1;
            if (spins > 2_000_000) return error.MimoM4AttrProbePending;
            std.Thread.yield() catch {};
        }
        if (k[0] == 0) return error.MimoM4AttrProbeSawNoCommit;
        if (k[2] != 0) return error.MimoM4AttrProbeDropped;
        const rows = b.rows[0..k[0]];
        b.g.resetTo(mark);
        if (i < warm) continue;
        const busy = busyUnion(rows, b.spans);
        // idle gaps in queue order (rows are in commit order: MLX's queue runs them so)
        var starved: f64 = 0;
        var queued: f64 = 0;
        var prev_end: f64 = 0;
        for (rows) |r| {
            if (r[2] <= 0) continue;
            const start = if (r[1] > 0) r[1] else @max(r[0], prev_end);
            if (prev_end > 0 and start > prev_end) {
                const s_part = @max(@min(r[0], start) - prev_end, 0);
                starved += s_part;
                queued += (start - prev_end) - s_part;
            }
            prev_end = @max(prev_end, r[2]);
        }
        var wake: f64 = 0;
        for (0..chain_layers) |li| {
            var bar_end: f64 = 0;
            for (rows) |r| if (r[0] >= c.bar_lo[li] and r[0] <= c.bar_hi[li]) {
                bar_end = @max(bar_end, r[2]);
            };
            if (bar_end > 0) wake += @max(c.ret[li] - bar_end, 0);
        }
        const j = i - warm;
        cols[0][j] = busy * 1e3 / per;
        cols[1][j] = @as(f64, @floatFromInt(t1 - t0)) / 1e6 / per;
        cols[2][j] = @as(f64, @floatFromInt(k[0])) / per;
        cols[3][j] = starved * 1e3 / per;
        cols[4][j] = queued * 1e3 / per;
        cols[5][j] = wake * 1e3 / per;
        unions += c.union_sum;
        layers_seen += chain_layers;
    }
    for (&cols) |*col| std.mem.sort(f64, col, {}, std.sort.asc(f64));
    return .{ .gpu_ms = cols[0][reps / 2], .wall_ms = cols[1][reps / 2], .buffers = cols[2][reps / 2], .starved_ms = cols[3][reps / 2], .queued_ms = cols[4][reps / 2], .wake_ms = cols[5][reps / 2], .union_per_layer = @as(f64, @floatFromInt(unions)) / @as(f64, @floatFromInt(@max(layers_seen, 1))) };
}

const Line = struct { name: []const u8, layers: u32, m1: f64, m4: f64, note: []const u8 };

fn printRow(l: Line) void {
    std.debug.print("MIMO_M4ATTR row={s} layers={d} m1_ms_per_layer={d:.4} m4_ms_per_layer={d:.4} m1_ms_total={d:.3} m4_ms_total={d:.3} m4_over_m1={d:.2} note=\"{s}\"\n", .{ l.name, l.layers, l.m1, l.m4, l.m1 * @as(f64, @floatFromInt(l.layers)), l.m4 * @as(f64, @floatFromInt(l.layers)), l.m4 / @max(l.m1, 1e-9), l.note });
}

test "mimo GPU M4 attribution: per-component GPU ms at M = 1 and M = 4 on the installed weights and a 17,408-key KV" {
    const enabled = std.c.getenv("MIMO_M4ATTR_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const pack_dir = std.mem.span(std.c.getenv("MIMO_M4ATTR_PACK") orelse return error.MimoM4AttrPackRequired);
    const ceiling: u64 = 5 << 30;
    const b_bytes = bound();
    std.debug.print("MIMO_M4ATTR static: bound={d} ceiling={d} pack={s}\n", .{ b_bytes, ceiling, pack_dir });
    if (b_bytes > ceiling) return error.MimoM4AttrAdmission;
    const a = testing.allocator;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    const rc = probe.mimo_gpu_probe_install();
    var hk: [4]u32 = undefined;
    probe.mimo_gpu_probe_counts(&hk);
    std.debug.print("MIMO_M4ATTR probe: install={d} hooked_classes={d}\n", .{ rc, hk[3] });
    if (rc != 0) return error.MimoM4AttrProbeInstall;

    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    var kept: std.ArrayList(T) = .empty;
    defer {
        for (kept.items) |x| g.release(x);
        kept.deinit(a);
    }
    const K = struct {
        fn keep(list: *std.ArrayList(T), al: std.mem.Allocator, x: T) !T {
            try list.append(al, x);
            return x;
        }
    };

    // ── weights (head first: its host staging is the largest transient) ──
    var pack = try Pack.open(a, pack_dir);
    defer pack.close();
    const c: mimo.Config = .{};
    const H: c_int = @intCast(c.hidden_size);
    const head = try K.keep(&kept, a, try pack.tensor(&g, "lm_head.weight", &.{ @intCast(c.vocab_size), H }, .bfloat16));
    var buf: [128]u8 = undefined;
    const P = struct {
        fn load(pk: *Pack, gg: *ops.MlxOps, list: *std.ArrayList(T), al: std.mem.Allocator, b: []u8, layer: u32, kv: u32) !Proj {
            const cc: mimo.Config = .{};
            const h: c_int = @intCast(cc.hidden_size);
            const names = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" };
            const shapes = [_][2]c_int{ .{ 64 * 192, h }, .{ @intCast(kv * 192), h }, .{ @intCast(kv * 128), h }, .{ h, 64 * 128 } };
            var out: [4]T = undefined;
            for (names, shapes, &out) |nm, sh, *o| {
                const name = try std.fmt.bufPrint(b, "model.layers.{d}.self_attn.{s}.weight", .{ layer, nm });
                o.* = try K.keep(list, al, try pk.tensor(gg, name, &sh, .bfloat16));
            }
            return .{ .q = out[0], .k = out[1], .v = out[2], .o = out[3] };
        }
    };
    const pf = try P.load(&pack, &g, &kept, a, &buf, full_layer, c.n_kv_heads);
    const ps = try P.load(&pack, &g, &kept, a, &buf, sliding_layer, c.swa_n_kv_heads);
    const dense_gate = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.0.mlp.gate_proj.weight", &.{ @intCast(c.intermediate_size), H }, .bfloat16));
    const dense_up = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.0.mlp.up_proj.weight", &.{ @intCast(c.intermediate_size), H }, .bfloat16));
    const dense_down = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.0.mlp.down_proj.weight", &.{ H, @intCast(c.intermediate_size) }, .bfloat16));
    const router_bf16 = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.mlp.gate.weight", &.{ @intCast(c.n_routed_experts), H }, .bfloat16));
    const router_bias = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.mlp.gate.e_score_correction_bias", &.{@intCast(c.n_routed_experts)}, .float32));
    const sinks = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.1.self_attn.attention_sink_bias", &.{@intCast(c.n_heads)}, .bfloat16));
    const norm_pre = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.input_layernorm.weight", &.{H}, .bfloat16));
    const norm_post = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.post_attention_layernorm.weight", &.{H}, .bfloat16));
    const router_t = blk: {
        const m = g.mark();
        const t = g.keep(try g.transpose(try g.astype(router_bf16, .float32)));
        try g.evalAll(&.{t});
        g.resetTo(m);
        break :blk try K.keep(&kept, a, t);
    };
    const bank = try expertBank(a, &g, pack_dir, 24);
    defer for ([_]T{ bank.gate.w, bank.gate.scales.?, bank.up.w, bank.up.scales.?, bank.down.w, bank.down.scales.? }) |x| g.release(x);
    const banks = [moe.n_banks]moe.Arrays{ bank, bank, bank };
    const scrub = blk: {
        const z = g.keep(try g.zeros(&.{@intCast(scrub_bytes / 2)}, .bfloat16));
        try g.evalAll(&.{z});
        g.reset();
        break :blk try K.keep(&kept, a, z);
    };
    const ones = blk: {
        var o: T = .{};
        try mlx.check(mlx.mlx_ones(&o, &.{H}, 1, .bfloat16, s));
        break :blk try K.keep(&kept, a, o);
    };
    const value_scale = try K.keep(&kept, a, mlx.mlx_array_new_float(0.707));
    const geometry = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    var fused_union = try moe.Fused.init(geometry, 24);
    defer fused_union.deinit();
    var fused_pair = try moe.Fused.initTexts(geometry, 24, .pair);
    defer fused_pair.deinit();
    // K1: the served k1 plan and three geometry alternatives (template arguments of the same texts), measured on the
    // same waves as the banked family: verify waves with gate/up 4 rows per simdgroup; gate/up 1 row and down 2; and
    // one-token waves with 2 rows for both (half the work per simdgroup, a shorter tail).
    var fused_k1 = try moe.Fused.initFamily(geometry, 24, .k1);
    defer fused_k1.deinit();
    const k1_alt_plans = [_]moe.K1Plan{
        .{ .union_gate_up = .{ .rows = 4, .maxm = moe.max_tokens } },
        .{ .union_gate_up = .{ .rows = 1, .maxm = moe.max_tokens }, .union_down = .{ .rows = 2, .maxm = moe.max_tokens } },
        .{ .serial_gate_up = .{ .rows = 2, .maxm = 1 }, .serial_down = .{ .rows = 2, .maxm = 1 } },
        // the dot as one FMA per weight, block scale folded once per lane block (rounding-class, never routed): what the
        // banked order's extra FP ops cost at the served geometry, every wave
        moe.k1_plan_fma,
    };
    var fused_k1_alt: [k1_alt_plans.len]moe.Fused = undefined;
    var n_alt: usize = 0;
    defer for (fused_k1_alt[0..n_alt]) |*f| f.deinit();
    for (k1_alt_plans) |plan| {
        fused_k1_alt[n_alt] = try moe.Fused.initTextsWith(geometry, 24, .k1, plan);
        n_alt += 1;
    }
    var kernels = try decode_attention.Kernels.init(c);
    defer kernels.deinit();

    // KV: the full layer's slab at the cells' reserve, `keys - verify_rows` rows written; the sliding view's buffer.
    var slab_k = try randomBf16(a, &g, &.{ 1, @intCast(c.n_kv_heads), @intCast(keys), 192 }, 1, 0x11);
    var slab_v = try randomBf16(a, &g, &.{ 1, @intCast(c.n_kv_heads), @intCast(keys), 128 }, 1, 0x22);
    defer g.release(slab_k);
    defer g.release(slab_v);
    var ring_k = try randomBf16(a, &g, &.{ 1, @intCast(c.swa_n_kv_heads), 656, 192 }, 1, 0x33);
    var ring_v = try randomBf16(a, &g, &.{ 1, @intCast(c.swa_n_kv_heads), 656, 128 }, 1, 0x44);
    defer g.release(ring_k);
    defer g.release(ring_v);
    g.reset();
    if (mlx.takeError(&buf)) |msg| std.debug.print("MIMO_M4ATTR load latched mlx=\"{s}\"\n", .{msg});
    var peak_load: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak_load));
    std.debug.print("MIMO_M4ATTR loaded: mlx_peak={d}\n", .{peak_load});

    const rows_buf = try a.alloc(Row, probe_cap);
    defer a.free(rows_buf);
    const spans = try a.alloc([2]f64, probe_cap);
    defer a.free(spans);
    var bench: Bench = .{ .a = a, .g = &g, .s = s, .scrub = scrub, .rows = rows_buf, .spans = spans };

    var res: [2]struct {
        proj_full: f64 = 0,
        proj_sliding: f64 = 0,
        head: f64 = 0,
        dense: f64 = 0,
        attn_full_stock: f64 = 0,
        attn_full_kernels: f64 = 0,
        attn_sliding_stock: f64 = 0,
        attn_sliding_kernels: f64 = 0,
        router: f64 = 0,
        glue_full: f64 = 0,
        glue_sliding: f64 = 0,
        moe: [2][2][3]f64 = @splat(@splat(@splat(0))), // [hitpass][union][shape: u32/u8, u24, u12]
        moe_k1: [2][3]f64 = @splat(@splat(0)), // K1: [hitpass][shape: u32/u8, u24, u12], the k1 texts
        combine: f64 = 0,
    } = .{ .{}, .{} };

    // M = 4 routed experts per row: union 24 (rows t*6 .. t*6+7 mod 24), 32 (all distinct), 12 (t*4.. mod 12).
    const Shapes = struct {
        fn experts(m: u32, union_size: u32, out: []u32) void {
            for (0..m) |t| for (0..moe.top_k) |j| {
                out[t * moe.top_k + j] = switch (union_size) {
                    32 => @intCast(t * moe.top_k + j),
                    24 => @intCast((t * 6 + j) % 24),
                    12 => @intCast((t * 4 + j) % 12),
                    else => @intCast(j),
                };
            };
        }
    };
    var lut_host: [256]u32 = undefined;
    for (&lut_host, 0..) |*v, e| v.* = @intCast(e % 24); // base bank (0 << 24), slot = expert for 0..23
    const lut = try K.keep(&kept, a, g.keep(try g.hostArray(std.mem.sliceAsBytes(&lut_host), &.{256}, .uint32)));
    g.reset();

    for ([_]u32{ 1, verify_rows }, 0..) |m, mi| {
        const r = &res[mi];
        const n: c_int = @intCast(m);
        const x = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 1, 0x100 + m));
        const xo = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, 64 * 128 }, 1, 0x200 + m));
        r.proj_full = (try bench.measure("proj_full", m, &Projections{ .w = pf, .x = x, .xo = xo })).gpu_ms;
        r.proj_sliding = (try bench.measure("proj_sliding", m, &Projections{ .w = ps, .x = x, .xo = xo })).gpu_ms;
        r.head = (try bench.measure("head", m, &Matmul{ .w = head, .x = x })).gpu_ms;
        r.dense = (try bench.measure("dense_mlp_l0", m, &DenseMlp{ .gate = dense_gate, .up = dense_up, .down = dense_down, .x = x, .s = s })).gpu_ms;
        r.router = (try bench.measure("router", m, &Router{ .x = x, .w_t = router_t, .bias = router_bias, .s = s })).gpu_ms;
        // attention: q [1, 64, M, 192] (the heads view of the q projection), the full view [0, keys) and the sliding
        // view (127 + M keys of the 656-row slot buffer), masks as the trunk builds them
        const qv = try K.keep(&kept, a, try randomBf16(a, &g, &.{ 1, 64, n, 192 }, 1, 0x300 + m));
        {
            const kv_k = try g.slice(slab_k, &.{ 0, 0, 0, 0 }, &.{ 1, 4, @intCast(keys), 192 }, &.{ 1, 1, 1, 1 });
            const kv_v = try g.slice(slab_v, &.{ 0, 0, 0, 0 }, &.{ 1, 4, @intCast(keys), 128 }, &.{ 1, 1, 1, 1 });
            const fmask = try g.adopt(try module.causalMask(a, keys - m, m, 0, keys, 0));
            const kr = decode_attention.install(.kernels, .full, keys, &kernels);
            r.attn_full_stock = (try bench.measure("attn_full_stock", m, &Attn{ .route = decode_attention.stock, .q = qv, .k = kv_k, .v = kv_v, .mask = fmask, .sinks = .{}, .s = s })).gpu_ms;
            r.attn_full_kernels = (try bench.measure("attn_full_kernels", m, &Attn{ .route = kr, .q = qv, .k = kv_k, .v = kv_v, .mask = fmask, .sinks = .{}, .s = s })).gpu_ms;
            const view = mimo.windowView(keys - m, m, c.window);
            const end: c_int = 400;
            const count: c_int = @intCast(view.key_count);
            const sk = try g.slice(ring_k, &.{ 0, 0, end - count, 0 }, &.{ 1, 8, end, 192 }, &.{ 1, 1, 1, 1 });
            const sv = try g.slice(ring_v, &.{ 0, 0, end - count, 0 }, &.{ 1, 8, end, 128 }, &.{ 1, 1, 1, 1 });
            const smask = try g.adopt(try module.causalMask(a, keys - m, m, view.first_key, view.key_count, c.window));
            const ks = decode_attention.install(.kernels, .sliding, keys, &kernels);
            r.attn_sliding_stock = (try bench.measure("attn_sliding_stock", m, &Attn{ .route = decode_attention.stock, .q = qv, .k = sk, .v = sv, .mask = smask, .sinks = sinks, .s = s })).gpu_ms;
            r.attn_sliding_kernels = (try bench.measure("attn_sliding_kernels", m, &Attn{ .route = ks, .q = qv, .k = sk, .v = sv, .mask = smask, .sinks = sinks, .s = s })).gpu_ms;
        }
        g.reset();
        // glue: full (slab write at keys - M) and sliding (ring buffer write); projections' outputs as inputs
        {
            const qp = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, 64 * 192 }, 1, 0x400 + m));
            const kpf = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, 4 * 192 }, 1, 0x500 + m));
            const vpf = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, 4 * 128 }, 1, 0x600 + m));
            const kps = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, 8 * 192 }, 1, 0x700 + m));
            const vps = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, 8 * 128 }, 1, 0x800 + m));
            const attn_out = try K.keep(&kept, a, try randomBf16(a, &g, &.{ 1, 64, n, 128 }, 1, 0x900 + m));
            const mlp_out = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 1, 0xa00 + m));
            const freq_full = module.inverseFrequencies(c.rope_theta, c.rope_head_dim);
            defer _ = mlx.mlx_array_free(freq_full);
            const freq_sliding = module.inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
            defer _ = mlx.mlx_array_free(freq_sliding);
            var rope_full = try module.rotaryTables(freq_full, keys - m, m, s);
            defer rope_full.deinit();
            var rope_sliding = try module.rotaryTables(freq_sliding, keys - m, m, s);
            defer rope_sliding.deinit();
            r.glue_full = (try bench.measure("glue_full", m, &Glue{ .x = x, .attn_o = mlp_out, .qp = qp, .kp = kpf, .vp = vpf, .attn = attn_out, .mlp = mlp_out, .pre = norm_pre, .post = norm_post, .ones = ones, .value_scale = value_scale, .rope = rope_full, .kv_heads = 4, .slab_k = &slab_k, .slab_v = &slab_v, .pos = keys - m, .m = m, .s = s })).gpu_ms;
            r.glue_sliding = (try bench.measure("glue_sliding", m, &Glue{ .x = x, .attn_o = mlp_out, .qp = qp, .kp = kps, .vp = vps, .attn = attn_out, .mlp = mlp_out, .pre = norm_pre, .post = norm_post, .ones = ones, .value_scale = value_scale, .rope = rope_sliding, .kv_heads = 8, .slab_k = &ring_k, .slab_v = &ring_v, .pos = 400 - m, .m = m, .s = s })).gpu_ms;
        }
        g.reset();
        // MoE: routed weights as the router normalises them (8 equal positive weights per row: the combine's cost does
        // not depend on their values)
        {
            const w: [verify_rows * moe.top_k]f32 = @splat(0.125);
            const weights = try K.keep(&kept, a, g.keep(try g.hostArray(std.mem.sliceAsBytes(w[0 .. m * moe.top_k]), &.{ n, @intCast(moe.top_k) }, .float32)));
            const shapes = if (m == 1) [_]u32{ 8, 8, 8 } else [_]u32{ 32, 24, 12 };
            for (shapes, 0..) |u, si| {
                if (m == 1 and si > 0) continue;
                var experts: [moe.max_pairs]u32 = undefined;
                Shapes.experts(m, u, experts[0 .. m * moe.top_k]);
                var ids: [moe.max_pairs]u32 = undefined;
                for (experts[0 .. m * moe.top_k], ids[0 .. m * moe.top_k]) |e, *id| id.* = lut_host[e];
                for ([_]bool{ false, true }, 0..) |hp, hi| for ([_]*const moe.Fused{ &fused_pair, &fused_union }, 0..) |f, ui| {
                    r.moe[hi][ui][si] = (try bench.measure(if (hp) (if (ui == 1) "moe_hitpass_union" else "moe_hitpass_pair") else (if (ui == 1) "moe_fused_union" else "moe_fused_pair"), m, &Moe{ .fused = f, .banks = &banks, .xf = x, .weights = weights, .ids_host = ids[0 .. m * moe.top_k], .experts_host = experts[0 .. m * moe.top_k], .lut = lut, .hitpass = hp, .m = m })).gpu_ms;
                    std.debug.print("MIMO_M4ATTR moe M={d} union={d} route={s} texts={s} gpu_ms={d:.4}\n", .{ m, u, if (hp) "hitpass" else "fused", if (ui == 1) "union" else "pair", r.moe[hi][ui][si] });
                };
                // K1: the same layer graph through the k1 texts
                for ([_]bool{ false, true }, 0..) |hp, hi| {
                    r.moe_k1[hi][si] = (try bench.measure(if (hp) "moe_hitpass_k1" else "moe_fused_k1", m, &Moe{ .fused = &fused_k1, .banks = &banks, .xf = x, .weights = weights, .ids_host = ids[0 .. m * moe.top_k], .experts_host = experts[0 .. m * moe.top_k], .lut = lut, .hitpass = hp, .m = m })).gpu_ms;
                    std.debug.print("MIMO_M4ATTR moe M={d} union={d} route={s} texts=k1 gpu_ms={d:.4}\n", .{ m, u, if (hp) "hitpass" else "fused", r.moe_k1[hi][si] });
                }
            }
            const routed = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, @intCast(moe.top_k), H }, 1, 0xb00 + m));
            const Comb = struct {
                f: *const moe.Fused,
                routed: T,
                w: T,
                fn graph(cc: *const @This(), gg: *ops.MlxOps, outs: *[8]T) !usize {
                    outs[0] = try cc.f.combineRows(gg, cc.routed, cc.w);
                    return 1;
                }
                        };
            r.combine = (try bench.measure("combine", m, &Comb{ .f = &fused_union, .routed = routed, .w = weights })).gpu_ms;
        }
        g.reset();
    }

    // ── K1: the decode expert texts alone (gate/up + SiLU, down), the banked family (the served control: banked texts
    // at P <= 8, union texts beyond) vs the k1 texts, at M = 1..4 over the union-24 routing (token t's k-th pair names
    // expert (6 t + k) % 24: 8, 14, 20, 24 distinct slots) and M = 4 over the union-12 routing (12 distinct); GB/s
    // against the distinct slots' bytes (what the wave must read). The k1 geometry alternatives run the waves their
    // plan changes (verify plans at M = 2..4, the one-token plan at M = 1). ──
    const k1_waves = 5;
    const k1_fams = 2 + k1_alt_plans.len;
    var k1_ms: [k1_waves][k1_fams]f64 = @splat(@splat(0));
    var k1_distinct: [k1_waves]u32 = undefined;
    var k1_stock: [k1_waves][2]f64 = @splat(@splat(0)); // [wave][stock_mlx (with the activation), stock_mlx_qmm]
    var k1_pairs: [k1_waves]u32 = undefined;
    var k1_fs: [k1_fams]*const moe.Fused = undefined;
    k1_fs[0] = &fused_union;
    k1_fs[1] = &fused_k1;
    for (0..n_alt) |i| k1_fs[2 + i] = &fused_k1_alt[i];
    const k1_names = [k1_fams][]const u8{ "banked", "k1", "k1_alt_gu4", "k1_alt_gu1_dn2", "k1_alt_serial_r2", "k1_alt_fma" };
    // which waves an alternative changes: the verify plans at M >= 2 (P >= 9), the one-token plan at M = 1, null = all
    const k1_alt_verify = [k1_alt_plans.len]?bool{ true, true, false, null };
    for (0..k1_waves) |wi| {
        const m: u32 = if (wi < 4) @intCast(wi + 1) else verify_rows;
        const u: u32 = if (wi < 4) 24 else 12;
        const pairs: u32 = m * moe.top_k;
        var ids_w: [moe.max_pairs]u32 = undefined;
        var tok_w: [moe.max_pairs]i32 = undefined;
        for (0..pairs) |q| {
            const t = q / moe.top_k;
            const j = q % moe.top_k;
            const ex: usize = if (u == 24) (t * 6 + j) % 24 else (t * 4 + j) % 12;
            ids_w[q] = lut_host[ex];
            tok_w[q] = @intCast(t);
        }
        var distinct: u32 = 0;
        for (0..pairs) |q| distinct += @intFromBool(std.mem.indexOfScalar(u32, ids_w[0..q], ids_w[q]) == null);
        k1_distinct[wi] = distinct;
        const xm = try K.keep(&kept, a, try randomBf16(a, &g, &.{ @intCast(m), H }, 1, 0xe00 + @as(u32, @intCast(wi))));
        const xp = blk: {
            const mk = g.mark();
            const v = g.keep(try g.take(xm, try g.hostArray(std.mem.sliceAsBytes(tok_w[0..pairs]), &.{@intCast(pairs)}, .int32), 0));
            try g.evalAll(&.{v});
            g.resetTo(mk);
            break :blk try K.keep(&kept, a, v);
        };
        const idv = try K.keep(&kept, a, g.keep(try g.hostArray(std.mem.sliceAsBytes(ids_w[0..pairs]), &.{@intCast(pairs)}, .uint32)));
        try g.evalAll(&.{idv});
        g.reset();
        for (k1_fs, 0..) |f, fi| {
            if (fi >= 2) {
                if (k1_alt_verify[fi - 2]) |v| {
                    if (v != (m >= 2)) continue;
                }
            }
            const smp = try bench.measure(k1_names[fi], m, &MoeKernels{ .fused = f, .banks = &banks, .x = xp, .ids = idv });
            k1_ms[wi][fi] = smp.gpu_ms;
            std.debug.print("MIMO_M4ATTR k1 M={d} pairs={d} distinct={d} union={d} family={s} kernels_ms={d:.4} gb_s={d:.1}\n", .{ m, pairs, distinct, u, k1_names[fi], smp.gpu_ms, @as(f64, @floatFromInt(distinct)) * record_bytes / (smp.gpu_ms * 1e6) });
        }
        // the third column: MLX's gather_qmm on the same pairs (the family's slots are all base-bank rows: id = row)
        for (ids_w[0..pairs]) |id| if (moe.slotOf(id).bank != 0) return error.MimoM4AttrStockBank;
        k1_pairs[wi] = pairs;
        for ([_]bool{ true, false }, 0..) |act, ai| {
            const name: []const u8 = if (act) "stock_mlx" else "stock_mlx_qmm";
            const smp = try bench.measure(name, m, &MoeStock{ .bank = &banks[0], .x = xp, .rows = idv, .act = act });
            k1_stock[wi][ai] = smp.gpu_ms;
            std.debug.print("MIMO_M4ATTR k1 M={d} pairs={d} distinct={d} union={d} family={s} kernels_ms={d:.4} gb_s={d:.1} pair_gb_s={d:.1}\n", .{ m, pairs, distinct, u, name, smp.gpu_ms, @as(f64, @floatFromInt(distinct)) * record_bytes / (smp.gpu_ms * 1e6), @as(f64, @floatFromInt(pairs)) * record_bytes / (smp.gpu_ms * 1e6) });
        }
        g.reset();
    }

    // dispatch floor: one tiny dispatch alone (an otherwise empty command buffer), and per dispatch in a 64-chain
    const tiny_x = try K.keep(&kept, a, try randomBf16(a, &g, &.{1}, 1, 0xc00));
    const one = try bench.measure("dispatch_one", 0, &Tiny{ .x = tiny_x, .n = 1 });
    const chain = try bench.measure("dispatch_chain64", 0, &Tiny{ .x = tiny_x, .n = 64 });
    const per_dispatch_ms = (chain.gpu_ms - one.gpu_ms) / 63;
    std.debug.print("MIMO_M4ATTR dispatch: empty_buffer_gpu_ms={d:.4} empty_buffer_wall_ms={d:.4} per_dispatch_gpu_ms={d:.5} chain64_buffers={d}\n", .{ one.gpu_ms, one.wall_ms, per_dispatch_ms, chain.buffers });

    // ── the table: ms per layer per component, summed over the forward ──
    const full: u32 = 9;
    const sliding: u32 = 39;
    const routed_layers: u32 = 47;
    const r1 = res[0];
    const r4 = res[1];
    const lines = [_]Line{
        .{ .name = "a.proj_full", .layers = full, .m1 = r1.proj_full, .m4 = r4.proj_full, .note = "q/k/v/o BF16, layer 5" },
        .{ .name = "a.proj_sliding", .layers = sliding, .m1 = r1.proj_sliding, .m4 = r4.proj_sliding, .note = "q/k/v/o BF16, layer 1" },
        .{ .name = "a.dense_mlp_l0", .layers = 1, .m1 = r1.dense, .m4 = r4.dense, .note = "gate/up/down 16384" },
        .{ .name = "a.head", .layers = 1, .m1 = r1.head, .m4 = r4.head, .note = "lm_head 152576 (serial: 1 row; verify: 4 rows)" },
        .{ .name = "b.attn_full_stock", .layers = full, .m1 = r1.attn_full_stock, .m4 = r4.attn_full_stock, .note = "MLX SDPA + array mask, 17,408 keys (the measured cells' route)" },
        .{ .name = "b.attn_full_kernels", .layers = full, .m1 = r1.attn_full_kernels, .m4 = r4.attn_full_kernels, .note = "split + merge" },
        .{ .name = "b.attn_sliding_stock", .layers = sliding, .m1 = r1.attn_sliding_stock, .m4 = r4.attn_sliding_stock, .note = "MLX SDPA + mask + sinks, 127 + M keys" },
        .{ .name = "b.attn_sliding_kernels", .layers = sliding, .m1 = r1.attn_sliding_kernels, .m4 = r4.attn_sliding_kernels, .note = "window" },
        .{ .name = "c.moe_hitpass_pair", .layers = routed_layers, .m1 = r1.moe[1][0][0], .m4 = r4.moe[1][0][1], .note = "control texts; M4 = 32 pairs over 24 slots" },
        .{ .name = "c.moe_hitpass_union", .layers = routed_layers, .m1 = r1.moe[1][1][0], .m4 = r4.moe[1][1][1], .note = "lever 1 texts; M4 = 32 pairs over 24 slots" },
        .{ .name = "c.moe_fused_pair", .layers = routed_layers, .m1 = r1.moe[0][0][0], .m4 = r4.moe[0][0][1], .note = "host ids" },
        .{ .name = "c.moe_fused_union", .layers = routed_layers, .m1 = r1.moe[0][1][0], .m4 = r4.moe[0][1][1], .note = "host ids" },
        .{ .name = "c.moe_hitpass_pair_u32", .layers = routed_layers, .m1 = r1.moe[1][0][0], .m4 = r4.moe[1][0][0], .note = "M4 all 32 distinct" },
        .{ .name = "c.moe_hitpass_union_u32", .layers = routed_layers, .m1 = r1.moe[1][1][0], .m4 = r4.moe[1][1][0], .note = "M4 all 32 distinct" },
        .{ .name = "c.moe_hitpass_pair_u12", .layers = routed_layers, .m1 = r1.moe[1][0][0], .m4 = r4.moe[1][0][2], .note = "M4 12 distinct (the missfree cell's union)" },
        .{ .name = "c.moe_hitpass_union_u12", .layers = routed_layers, .m1 = r1.moe[1][1][0], .m4 = r4.moe[1][1][2], .note = "M4 12 distinct" },
        .{ .name = "d.router", .layers = routed_layers, .m1 = r1.router, .m4 = r4.router, .note = "f32 matmul + sigmoid + top-8" },
        .{ .name = "d.combine", .layers = routed_layers, .m1 = r1.combine, .m4 = r4.combine, .note = "fused combine" },
        .{ .name = "d.glue_full", .layers = full, .m1 = r1.glue_full, .m4 = r4.glue_full, .note = "norms, rope, v scale, slab write, residuals" },
        .{ .name = "d.glue_sliding", .layers = sliding, .m1 = r1.glue_sliding, .m4 = r4.glue_sliding, .note = "norms, rope, v scale, ring write, residuals" },
        .{ .name = "d.barrier_buffer", .layers = routed_layers, .m1 = one.gpu_ms, .m4 = one.gpu_ms, .note = "one near-empty command buffer per routing barrier" },
    };
    for (lines) |l| printRow(l);
    // K1 (mimo/k1-moe-bw): the layer graph (ids, x take, the two texts, join, combine) through the k1 texts, the
    // controls above's shapes; then the texts alone per M with GB/s, and the per-forward deltas (47 routed layers).
    const k1_lines = [_]Line{
        .{ .name = "c.moe_k1_hitpass", .layers = routed_layers, .m1 = r1.moe_k1[1][0], .m4 = r4.moe_k1[1][1], .note = "k1 texts; M4 = 32 pairs over 24 slots (vs c.moe_hitpass_union)" },
        .{ .name = "c.moe_k1_fused", .layers = routed_layers, .m1 = r1.moe_k1[0][0], .m4 = r4.moe_k1[0][1], .note = "k1 texts, host ids (vs c.moe_fused_union)" },
        .{ .name = "c.moe_k1_hitpass_u32", .layers = routed_layers, .m1 = r1.moe_k1[1][0], .m4 = r4.moe_k1[1][0], .note = "k1 texts; M4 the u32 routing (vs c.moe_hitpass_union_u32)" },
        .{ .name = "c.moe_k1_hitpass_u12", .layers = routed_layers, .m1 = r1.moe_k1[1][0], .m4 = r4.moe_k1[1][2], .note = "k1 texts; M4 12 distinct (vs c.moe_hitpass_union_u12)" },
    };
    for (k1_lines) |l| printRow(l);
    for (0..k1_waves) |wi| {
        const m: u32 = if (wi < 4) @intCast(wi + 1) else verify_rows;
        const bytes = @as(f64, @floatFromInt(k1_distinct[wi])) * record_bytes;
        const b_ms = k1_ms[wi][0];
        const k_ms = k1_ms[wi][1];
        std.debug.print("MIMO_M4ATTR_K1 M={d} union={d} distinct={d} banked_ms={d:.4} k1_ms={d:.4} banked_gb_s={d:.1} k1_gb_s={d:.1} delta_ms_per_layer={d:.4} delta_ms_per_forward={d:.3}\n", .{ m, if (wi < 4) @as(u32, 24) else 12, k1_distinct[wi], b_ms, k_ms, bytes / (b_ms * 1e6), bytes / (k_ms * 1e6), k_ms - b_ms, (k_ms - b_ms) * @as(f64, @floatFromInt(routed_layers)) });
    }
    // David's rule (no duplicate of MLX without a measured reason): the k1 texts stay only if they beat MLX's own route
    // (stock_mlx: gather_qmm + the arch's activation, the full equivalent of the two texts) by more than 5 % in GB/s of
    // the wave's weight bytes at the deciding waves, M = 1 (8 pairs) and M = 4 over union 24 (32 pairs); otherwise the
    // route of record is the stock path and K1 is dropped. The other waves are printed, not deciding.
    var rule_k1 = true;
    var rule_fma = true;
    for (0..k1_waves) |wi| {
        const m: u32 = if (wi < 4) @intCast(wi + 1) else verify_rows;
        const bytes = @as(f64, @floatFromInt(k1_distinct[wi])) * record_bytes;
        const gb = struct {
            fn f(b: f64, ms: f64) f64 {
                return b / (ms * 1e6);
            }
        }.f;
        const k_gb = gb(bytes, k1_ms[wi][1]);
        const s_gb = gb(bytes, k1_stock[wi][0]);
        const f_gb = gb(bytes, k1_ms[wi][k1_fams - 1]);
        const within = s_gb >= 0.95 * k_gb;
        const decides = wi == 0 or wi == 3;
        if (decides and within) rule_k1 = false;
        if (decides and s_gb >= 0.95 * f_gb) rule_fma = false;
        std.debug.print("MIMO_M4ATTR_K1_RULE5 M={d} union={d} pairs={d} distinct={d} banked_gb_s={d:.1} k1_gb_s={d:.1} k1_fma_gb_s={d:.1} stock_mlx_gb_s={d:.1} stock_mlx_qmm_gb_s={d:.1} k1_over_stock_mlx_pct={d:.1} stock_within_5pct={s} deciding={s}\n", .{ m, if (wi < 4) @as(u32, 24) else 12, k1_pairs[wi], k1_distinct[wi], gb(bytes, k1_ms[wi][0]), k_gb, f_gb, s_gb, gb(bytes, k1_stock[wi][1]), (k_gb / s_gb - 1) * 100, if (within) "yes" else "no", if (decides) "yes" else "no" });
    }
    std.debug.print("MIMO_M4ATTR_K1_RULE5 verdict={s} fma_form={s} (k1 is kept only if it beats stock_mlx by more than 5 % at M = 1 and at M = 4 union 24; fma_form: the same rule for the rounding-class FMA alternative)\n", .{ if (rule_k1) "k1" else "stock_mlx", if (rule_fma) "beats_stock" else "stock_mlx" });
    std.debug.print("MIMO_M4ATTR_SUM k1 graph: m1_ms_per_token control={d:.3} k1={d:.3} delta={d:.3}; m4_ms_per_round u24 control={d:.3} k1={d:.3} delta={d:.3}; u12 control={d:.3} k1={d:.3} delta={d:.3}\n", .{
        47 * r1.moe[1][0][0], 47 * r1.moe_k1[1][0], 47 * (r1.moe_k1[1][0] - r1.moe[1][0][0]),
        47 * r4.moe[1][1][1], 47 * r4.moe_k1[1][1], 47 * (r4.moe_k1[1][1] - r4.moe[1][1][1]),
        47 * r4.moe[1][1][2], 47 * r4.moe_k1[1][2], 47 * (r4.moe_k1[1][2] - r4.moe[1][1][2]),
    });
    // The measured cells' route (decode_attention stock, hitpass, banked texts) summed per forward.
    const sum = struct {
        fn f(r: anytype, moe_ms: f64, stock: bool, floor: f64) f64 {
            const attn_f = if (stock) r.attn_full_stock else r.attn_full_kernels;
            const attn_s = if (stock) r.attn_sliding_stock else r.attn_sliding_kernels;
            return 9 * (r.proj_full + attn_f + r.glue_full) + 39 * (r.proj_sliding + attn_s + r.glue_sliding) + r.dense + r.head + 47 * (r.router + moe_ms + r.combine + floor);
        }
    }.f;
    const s1 = sum(r1, r1.moe[1][0][0], true, one.gpu_ms);
    const s4 = sum(r4, r4.moe[1][0][2], true, one.gpu_ms);
    std.debug.print("MIMO_M4ATTR_SUM route=control(stock attention, hitpass, pair texts; M4 union 12 = the missfree cell) m1_ms={d:.3} measured={d:.3} residual={d:.3} m4_ms={d:.3} measured={d:.3} residual={d:.3}\n", .{ s1, measured_serial_barrier_ms, measured_serial_barrier_ms - s1, s4, measured_verify_barrier_ms, measured_verify_barrier_ms - s4 });
    const s4u = sum(r4, r4.moe[1][1][2], true, one.gpu_ms);
    const s4k = sum(r4, r4.moe[1][1][2], false, one.gpu_ms);
    const s4n = sum(r4, r4.moe[1][1][1], false, one.gpu_ms);
    const s4p = sum(r4, r4.moe[1][0][1], true, one.gpu_ms);
    std.debug.print("MIMO_M4ATTR_SUM levers m4_ms: control_u12={d:.3} union_u12={d:.3} union_u12+kernels={d:.3} control_u24={d:.3} union_u24+kernels={d:.3}\n", .{ s4, s4u, s4k, s4p, s4n });
    // ── (d) the lookahead predictor alone and (e) the routed layer as the server runs it (mimo/barrier-residual) ──
    var pred: [2][2]f64 = undefined; // [M = 1, 4][gemm, rows]
    var chains: [2][2][2]ChainSample = undefined; // [M = 1, 4][sliding, full][gemm, rows]
    // K6 / K7 (mimo/k6-commits): the installed predictor's chains under the three other route pairs.
    const k6_pairs = [_]struct { commits: decode_barrier.Commits, issue: decode_barrier.MissIssue }{ .{ .commits = .layer, .issue = .host }, .{ .commits = .mlx, .issue = .event }, .{ .commits = .layer, .issue = .event } };
    var k6chains: [2][2][k6_pairs.len]ChainSample = undefined; // [M = 1, 4][sliding, full][pair]
    // K6's construction check on this GPU stream (the served route runs it at the source's construction): the commit
    // limits it writes are the ones libmlx's compiled `needs_commit` reads; the limits are the device's defaults again.
    const k6_check = try decode_barrier.selfCheck(s);
    const k6_limits = try decode_barrier.limits();
    std.debug.print("MIMO_M4ATTR_K6 selfcheck={s} limits_ops={d} limits_mb={d}\n", .{ @tagName(k6_check), k6_limits.ops, k6_limits.mb });
    if (k6_check != .proven) return error.MimoDecodeCommitsSelfCheck;
    var lut_pair: [2]T = undefined;
    for (&lut_pair) |*l| l.* = try K.keep(&kept, a, g.keep(try g.hostArray(std.mem.sliceAsBytes(&lut_host), &.{256}, .uint32)));
    try g.evalAll(&lut_pair);
    g.reset();
    const full_pos: u32 = keys - 512; // mid-decode: the slab's [0, pos + M) view is strided across heads, as served
    for ([_]u32{ 1, verify_rows }, 0..) |m, mi| {
        const n: c_int = @intCast(m);
        const x = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 1, 0xd00 + m));
        for ([_]Predictor{ .gemm, .rows }, 0..) |pw, pi| {
            pred[mi][pi] = (try bench.measure(if (pi == 0) "predictor_gemm" else "predictor_rows", m, &PredictorGraph{ .which = pw, .x = x, .router_t = router_t, .bias = router_bias })).gpu_ms;
        }
        {
            // agreement: the installed rows against the control's GEMM, and per row against the serial (M = 1) GEMV
            const mark = g.mark();
            defer g.resetTo(mark);
            var vg: [verify_rows * 256]f32 = undefined;
            var vr: [verify_rows * 256]f32 = undefined;
            _ = try g.hostF32(try predictorOf(&g, .gemm, x, router_t, router_bias), vg[0 .. m * 256]);
            _ = try g.hostF32(try predictorOf(&g, .rows, x, router_t, router_bias), vr[0 .. m * 256]);
            var max_abs: f32 = 0;
            for (vg[0 .. m * 256], vr[0 .. m * 256]) |p, q| max_abs = @max(max_abs, @abs(p - q));
            var top_equal: u32 = 0;
            var serial_bitwise: u32 = 0;
            for (0..m) |r| {
                var tg: [8]u16 = undefined;
                var tr: [8]u16 = undefined;
                topEight(vg[r * 256 ..][0..256], &tg);
                topEight(vr[r * 256 ..][0..256], &tr);
                top_equal += @intFromBool(std.mem.eql(u16, &tg, &tr));
                var serial_row: [256]f32 = undefined;
                const xr = try g.slice(x, &.{ @intCast(r), 0 }, &.{ @intCast(r + 1), H }, &.{ 1, 1 });
                _ = try g.hostF32(try predictorOf(&g, .gemm, xr, router_t, router_bias), &serial_row);
                serial_bitwise += @intFromBool(std.mem.eql(u8, std.mem.sliceAsBytes(&serial_row), std.mem.sliceAsBytes(vr[r * 256 ..][0..256])));
            }
            std.debug.print("MIMO_M4ATTR predictor M={d} gemm_ms={d:.4} rows_ms={d:.4} max_abs={e} top8_equal_rows={d}/{d} rows_bitwise_vs_serial={d}/{d}\n", .{ m, pred[mi][0], pred[mi][1], max_abs, top_equal, m, serial_bitwise, m });
        }
        g.reset();
        const freq_full = module.inverseFrequencies(c.rope_theta, c.rope_head_dim);
        defer _ = mlx.mlx_array_free(freq_full);
        const freq_sliding = module.inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
        defer _ = mlx.mlx_array_free(freq_sliding);
        var rope_full = try module.rotaryTables(freq_full, full_pos, m, s);
        defer rope_full.deinit();
        var rope_sliding = try module.rotaryTables(freq_sliding, keys - m, m, s);
        defer rope_sliding.deinit();
        const fmask = try K.keep(&kept, a, g.keep(try g.adopt(try module.causalMask(a, full_pos, m, 0, full_pos + m, 0))));
        const view = mimo.windowView(keys - m, m, c.window);
        const ring_end: u32 = 400;
        const smask = try K.keep(&kept, a, g.keep(try g.adopt(try module.causalMask(a, keys - m, m, view.first_key, view.key_count, c.window))));
        for ([_]mimo.Attention{ .sliding, .full }, 0..) |kind, ki| for ([_]Predictor{ .gemm, .rows }, 0..) |pw, pi| {
            var sc_chain: ServerChain = if (kind == .full) .{
                .w = pf, .kv_heads = c.n_kv_heads, .pre = norm_pre, .post = norm_post, .ones = ones, .value_scale = value_scale, .sinks = .{}, .rope = rope_full, .mask = fmask,
                .cache_k = &slab_k, .cache_v = &slab_v, .pos = full_pos, .view_first = 0, .view_end = full_pos + m,
                .router_t = router_t, .bias = router_bias, .lut = lut_pair, .lut_host = &lut_host, .fused = &fused_union, .banks = &banks, .predictor = pw, .m = m, .s = s,
            } else .{
                .w = ps, .kv_heads = c.swa_n_kv_heads, .pre = norm_pre, .post = norm_post, .ones = ones, .value_scale = value_scale, .sinks = sinks, .rope = rope_sliding, .mask = smask,
                .cache_k = &ring_k, .cache_v = &ring_v, .pos = ring_end - m, .view_first = ring_end - view.key_count, .view_end = ring_end,
                .router_t = router_t, .bias = router_bias, .lut = lut_pair, .lut_host = &lut_host, .fused = &fused_union, .banks = &banks, .predictor = pw, .m = m, .s = s,
            };
            const cs = try measureChain(&bench, if (kind == .full) "layer_full_server" else "layer_sliding_server", &sc_chain, x);
            chains[mi][ki][pi] = cs;
            std.debug.print("MIMO_M4ATTR chain kind={s} M={d} predictor={s} layers={d} gpu_ms_per_layer={d:.4} wall_ms_per_layer={d:.4} buffers_per_layer={d:.2} starved_ms_per_layer={d:.4} queued_ms_per_layer={d:.4} wake_ms_per_layer={d:.4} slot_union_per_layer={d:.2}\n", .{ @tagName(kind), m, @tagName(pw), chain_layers, cs.gpu_ms, cs.wall_ms, cs.buffers, cs.starved_ms, cs.queued_ms, cs.wake_ms, cs.union_per_layer });
            // K6 / K7: the same chain (the installed predictor) under each other route pair, in the same window.
            if (pw == .rows) for (k6_pairs, 0..) |kp, vi| {
                sc_chain.commits = kp.commits;
                sc_chain.issue = kp.issue;
                const ks = try measureChain(&bench, if (kind == .full) "layer_full_server_k6" else "layer_sliding_server_k6", &sc_chain, x);
                k6chains[mi][ki][vi] = ks;
                std.debug.print("MIMO_M4ATTR chain kind={s} M={d} predictor=rows commits={s} issue={s} layers={d} gpu_ms_per_layer={d:.4} wall_ms_per_layer={d:.4} buffers_per_layer={d:.2} starved_ms_per_layer={d:.4} queued_ms_per_layer={d:.4} wake_ms_per_layer={d:.4} slot_union_per_layer={d:.2}\n", .{ @tagName(kind), m, @tagName(kp.commits), @tagName(kp.issue), chain_layers, ks.gpu_ms, ks.wall_ms, ks.buffers, ks.starved_ms, ks.queued_ms, ks.wake_ms, ks.union_per_layer });
            };
        };
        g.reset();
    }

    // (d) the predictor and (e) the server-shaped layers (mimo/barrier-residual): 46 routed layers carry a predictor
    // (the last has no next router); the forward = layer 0 from its components + 8 routed full + 39 sliding layers as
    // the chains ran them + the head.
    const extra = [_]Line{
        .{ .name = "d.predictor_gemm", .layers = routed_layers - 1, .m1 = pred[0][0], .m4 = pred[1][0], .note = "control: F32 [M, 4096] x [4096, 256] + sigmoid + correction" },
        .{ .name = "d.predictor_rows", .layers = routed_layers - 1, .m1 = pred[0][1], .m4 = pred[1][1], .note = "lever: gathered GEMV per row (M = 1: the control's graph)" },
        .{ .name = "e.layer_sliding_server", .layers = sliding, .m1 = chains[0][0][0].gpu_ms, .m4 = chains[1][0][0].gpu_ms, .note = "GPU per layer, chained, the served order (control predictor)" },
        .{ .name = "e.layer_full_server", .layers = full - 1, .m1 = chains[0][1][0].gpu_ms, .m4 = chains[1][1][0].gpu_ms, .note = "GPU per layer, chained, strided slab view (control predictor)" },
        .{ .name = "e.layer_sliding_server_wall", .layers = sliding, .m1 = chains[0][0][0].wall_ms, .m4 = chains[1][0][0].wall_ms, .note = "host wall per layer (the GPU's plus what the host adds)" },
        .{ .name = "e.layer_full_server_wall", .layers = full - 1, .m1 = chains[0][1][0].wall_ms, .m4 = chains[1][1][0].wall_ms, .note = "host wall per layer" },
        .{ .name = "e.layer_sliding_server_rows", .layers = sliding, .m1 = chains[0][0][1].gpu_ms, .m4 = chains[1][0][1].gpu_ms, .note = "lever predictor" },
        .{ .name = "e.layer_full_server_rows", .layers = full - 1, .m1 = chains[0][1][1].gpu_ms, .m4 = chains[1][1][1].gpu_ms, .note = "lever predictor" },
    };
    for (extra) |l| printRow(l);
    // K6 / K7 (mimo/k6-commits): the server-shaped layers under decode_commits "layer" + decode_miss_issue "event", and per
    // route pair the buffers per layer, the GPU's idle at buffer boundaries (queued) and before a commit (starved), the
    // barrier's wake (its set's last GPU end to the host's return), GPU busy and host wall per layer.
    const k6_extra = [_]Line{
        .{ .name = "e.layer_sliding_server_k6", .layers = sliding, .m1 = k6chains[0][0][2].gpu_ms, .m4 = k6chains[1][0][2].gpu_ms, .note = "decode_commits layer + decode_miss_issue event (installed predictor)" },
        .{ .name = "e.layer_full_server_k6", .layers = full - 1, .m1 = k6chains[0][1][2].gpu_ms, .m4 = k6chains[1][1][2].gpu_ms, .note = "decode_commits layer + decode_miss_issue event" },
        .{ .name = "e.layer_sliding_server_k6_wall", .layers = sliding, .m1 = k6chains[0][0][2].wall_ms, .m4 = k6chains[1][0][2].wall_ms, .note = "host wall per layer" },
        .{ .name = "e.layer_full_server_k6_wall", .layers = full - 1, .m1 = k6chains[0][1][2].wall_ms, .m4 = k6chains[1][1][2].wall_ms, .note = "host wall per layer" },
    };
    for (k6_extra) |l| printRow(l);
    for (0..2) |mi| for ([_]mimo.Attention{ .sliding, .full }, 0..) |kind, ki| {
        const q0 = chains[mi][ki][1];
        const q1 = k6chains[mi][ki][0];
        const q2 = k6chains[mi][ki][1];
        const q3 = k6chains[mi][ki][2];
        std.debug.print("MIMO_M4ATTR_K6 M={d} kind={s} pairs=mlx+host/layer+host/mlx+event/layer+event buffers={d:.2}/{d:.2}/{d:.2}/{d:.2} queued_ms={d:.4}/{d:.4}/{d:.4}/{d:.4} starved_ms={d:.4}/{d:.4}/{d:.4}/{d:.4} wake_ms={d:.4}/{d:.4}/{d:.4}/{d:.4} gpu_ms={d:.4}/{d:.4}/{d:.4}/{d:.4} wall_ms={d:.4}/{d:.4}/{d:.4}/{d:.4}\n", .{
            if (mi == 0) @as(u32, 1) else verify_rows, @tagName(kind),
            q0.buffers,    q1.buffers,    q2.buffers,    q3.buffers,
            q0.queued_ms,  q1.queued_ms,  q2.queued_ms,  q3.queued_ms,
            q0.starved_ms, q1.starved_ms, q2.starved_ms, q3.starved_ms,
            q0.wake_ms,    q1.wake_ms,    q2.wake_ms,    q3.wake_ms,
            q0.gpu_ms,     q1.gpu_ms,     q2.gpu_ms,     q3.gpu_ms,
            q0.wall_ms,    q1.wall_ms,    q2.wall_ms,    q3.wall_ms,
        });
    };
    const k6fwd = struct {
        fn f(r: anytype, sl: ChainSample, fu: ChainSample, wall: bool) f64 {
            return (r.proj_full + r.attn_full_stock + r.glue_full + r.dense) + 8 * (if (wall) fu.wall_ms else fu.gpu_ms) + 39 * (if (wall) sl.wall_ms else sl.gpu_ms) + r.head;
        }
    }.f;
    std.debug.print("MIMO_M4ATTR_SUM k6 forward (8 full + 39 sliding chained layers + layer 0 + head) wall_ms: M1 control={d:.3} layer+event={d:.3} delta={d:.3}; M4 control={d:.3} layer+event={d:.3} delta={d:.3}; gpu_ms: M1 delta={d:.3} M4 delta={d:.3}\n", .{
        k6fwd(res[0], chains[0][0][1], chains[0][1][1], true),  k6fwd(res[0], k6chains[0][0][2], k6chains[0][1][2], true),  k6fwd(res[0], k6chains[0][0][2], k6chains[0][1][2], true) - k6fwd(res[0], chains[0][0][1], chains[0][1][1], true),
        k6fwd(res[1], chains[1][0][1], chains[1][1][1], true),  k6fwd(res[1], k6chains[1][0][2], k6chains[1][1][2], true),  k6fwd(res[1], k6chains[1][0][2], k6chains[1][1][2], true) - k6fwd(res[1], chains[1][0][1], chains[1][1][1], true),
        k6fwd(res[0], k6chains[0][0][2], k6chains[0][1][2], false) - k6fwd(res[0], chains[0][0][1], chains[0][1][1], false),
        k6fwd(res[1], k6chains[1][0][2], k6chains[1][1][2], false) - k6fwd(res[1], chains[1][0][1], chains[1][1][1], false),
    });
    // Each server-shaped layer against its components measured alone (the in-window residual per layer kind).
    const Comp = struct {
        fn sliding_(r: anytype, moe_ms: f64, pred_ms: f64, floor: f64) f64 {
            return r.proj_sliding + r.attn_sliding_stock + r.glue_sliding + r.router + pred_ms + moe_ms + r.combine + floor;
        }
        fn full_(r: anytype, moe_ms: f64, pred_ms: f64, floor: f64) f64 {
            return r.proj_full + r.attn_full_stock + r.glue_full + r.router + pred_ms + moe_ms + r.combine + floor;
        }
    };
    for ([_]u32{ 0, 1 }) |mi| {
        const r = res[mi];
        const moe_ms = if (mi == 0) r.moe[1][1][0] else r.moe[1][1][2];
        const cs = Comp.sliding_(r, moe_ms, pred[mi][0], one.gpu_ms);
        const cf = Comp.full_(r, moe_ms, pred[mi][0], one.gpu_ms);
        const ss = chains[mi][0][0];
        const sf = chains[mi][1][0];
        std.debug.print("MIMO_M4ATTR_SERVER M={d} kind=sliding server_gpu_ms={d:.4} components_ms={d:.4} residual_ms={d:.4} wall_ms={d:.4} starved_ms={d:.4} queued_ms={d:.4} wake_ms={d:.4} buffers={d:.2}\n", .{ if (mi == 0) @as(u32, 1) else verify_rows, ss.gpu_ms, cs, ss.gpu_ms - cs, ss.wall_ms, ss.starved_ms, ss.queued_ms, ss.wake_ms, ss.buffers });
        std.debug.print("MIMO_M4ATTR_SERVER M={d} kind=full server_gpu_ms={d:.4} components_ms={d:.4} residual_ms={d:.4} wall_ms={d:.4} starved_ms={d:.4} queued_ms={d:.4} wake_ms={d:.4} buffers={d:.2}\n", .{ if (mi == 0) @as(u32, 1) else verify_rows, sf.gpu_ms, cf, sf.gpu_ms - cf, sf.wall_ms, sf.starved_ms, sf.queued_ms, sf.wake_ms, sf.buffers });
    }
    const fwd = struct {
        fn f(r: anytype, ch: anytype, which: usize, wall: bool) f64 {
            const sl = if (wall) ch[0][which].wall_ms else ch[0][which].gpu_ms;
            const fu = if (wall) ch[1][which].wall_ms else ch[1][which].gpu_ms;
            return (r.proj_full + r.attn_full_stock + r.glue_full + r.dense) + 8 * fu + 39 * sl + r.head;
        }
    }.f;
    std.debug.print("MIMO_M4ATTR_SUM route=server(the chains as served: stock attention, hitpass, union texts, real routing) m1_gpu_ms={d:.3} m1_wall_ms={d:.3} measured={d:.3} residual={d:.3} m4_gpu_ms={d:.3} m4_wall_ms={d:.3} measured={d:.3} residual={d:.3}\n", .{ fwd(r1, chains[0], 0, false), fwd(r1, chains[0], 0, true), measured_serial_barrier_ms, measured_serial_barrier_ms - fwd(r1, chains[0], 0, true), fwd(r4, chains[1], 0, false), fwd(r4, chains[1], 0, true), measured_verify_barrier_ms, measured_verify_barrier_ms - fwd(r4, chains[1], 0, true) });
    std.debug.print("MIMO_M4ATTR_SUM levers server m4_gpu_ms: control={d:.3} predictor_rows={d:.3} delta={d:.3}; m1_gpu_ms: control={d:.3} predictor_rows={d:.3} delta={d:.3}\n", .{ fwd(r4, chains[1], 0, false), fwd(r4, chains[1], 1, false), fwd(r4, chains[1], 1, false) - fwd(r4, chains[1], 0, false), fwd(r1, chains[0], 0, false), fwd(r1, chains[0], 1, false), fwd(r1, chains[0], 1, false) - fwd(r1, chains[0], 0, false) });

    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_M4ATTR peak: mlx={d} bound={d}\n", .{ peak, b_bytes });
    if (peak > b_bytes) return error.MimoM4AttrPeakExceeded;
}

test "mimo m4 attribution CPU: the family's MLX graphs (projections, head, dense MLP, router, glue, stock attention) build and evaluate at the real shapes on the CPU stream" {
    // The device family's first run failed with a bare MlxError (the glue's residual added the heads' flat [M, 8192] to
    // x [M, 4096]). Every component whose ops MLX's CPU backend runs is built here exactly as `graph` builds it, at the
    // real shapes (zero weights), M = 1 and 4, and evaluated; the Metal-only ones (the decode attention kernels, the fused
    // MoE texts) are covered by their own device checks.
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const c: mimo.Config = .{};
    const H: c_int = @intCast(c.hidden_size);
    const z = struct {
        fn of(gg: *ops.MlxOps, shape: []const c_int, dt: mlx.mlx_dtype) !T {
            return gg.zeros(shape, dt);
        }
    }.of;
    const proj = struct {
        fn of(gg: *ops.MlxOps, kv: c_int) !Proj {
            const h: c_int = 4096;
            return .{ .q = try z(gg, &.{ 64 * 192, h }, .bfloat16), .k = try z(gg, &.{ kv * 192, h }, .bfloat16), .v = try z(gg, &.{ kv * 128, h }, .bfloat16), .o = try z(gg, &.{ h, 64 * 128 }, .bfloat16) };
        }
    }.of;
    const pf = try proj(&g, 4);
    const ps = try proj(&g, 8);
    const router_t = try z(&g, &.{ H, 256 }, .float32);
    const bias = try z(&g, &.{256}, .float32);
    const ones = try g.ones(&.{H}, .bfloat16);
    const norm = try g.ones(&.{H}, .bfloat16);
    const sinks = try z(&g, &.{64}, .bfloat16);
    const value_scale = try g.adopt(mlx.mlx_array_new_float(0.707));
    const dense_gate = try z(&g, &.{ 1024, H }, .bfloat16); // the MLP's ops at a narrower intermediate (CPU time)
    const dense_up = try z(&g, &.{ 1024, H }, .bfloat16);
    const dense_down = try z(&g, &.{ H, 1024 }, .bfloat16);
    const head = try z(&g, &.{ 4096, H }, .bfloat16); // the head's op on 4,096 rows (CPU time)
    var slab_k = g.keep(try z(&g, &.{ 1, 4, @intCast(keys), 192 }, .bfloat16));
    var slab_v = g.keep(try z(&g, &.{ 1, 4, @intCast(keys), 128 }, .bfloat16));
    var ring_k = g.keep(try z(&g, &.{ 1, 8, 656, 192 }, .bfloat16));
    var ring_v = g.keep(try z(&g, &.{ 1, 8, 656, 128 }, .bfloat16));
    defer for ([_]*T{ &slab_k, &slab_v, &ring_k, &ring_v }) |p| g.release(p.*);
    try g.evalAll(&.{ slab_k, slab_v, ring_k, ring_v });
    for ([_]u32{ 1, verify_rows }) |m| {
        const mark = g.mark();
        const n: c_int = @intCast(m);
        const x = try z(&g, &.{ n, H }, .bfloat16);
        const xo = try z(&g, &.{ n, 64 * 128 }, .bfloat16);
        var outs: [8]T = undefined;
        var count: usize = 0;
        count = try (Projections{ .w = pf, .x = x, .xo = xo }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        count = try (Projections{ .w = ps, .x = x, .xo = xo }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        count = try (Matmul{ .w = head, .x = x }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        count = try (DenseMlp{ .gate = dense_gate, .up = dense_up, .down = dense_down, .x = x, .s = s }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        count = try (Router{ .x = x, .w_t = router_t, .bias = bias, .s = s }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        try testing.expectEqual(@as(c_int, 8), g.shapeOf(outs[0]).dim(1));
        const qv = try z(&g, &.{ 1, 64, n, 192 }, .bfloat16);
        const kv_k = try g.slice(slab_k, &.{ 0, 0, 0, 0 }, &.{ 1, 4, @intCast(keys), 192 }, &.{ 1, 1, 1, 1 });
        const kv_v = try g.slice(slab_v, &.{ 0, 0, 0, 0 }, &.{ 1, 4, @intCast(keys), 128 }, &.{ 1, 1, 1, 1 });
        const fmask = try g.adopt(try module.causalMask(a, keys - m, m, 0, keys, 0));
        count = try (Attn{ .route = decode_attention.stock, .q = qv, .k = kv_k, .v = kv_v, .mask = fmask, .sinks = .{}, .s = s }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        try testing.expectEqual(@as(c_int, 128), g.shapeOf(outs[0]).dim(3));
        const view = mimo.windowView(keys - m, m, c.window);
        const count_k: c_int = @intCast(view.key_count);
        const sk = try g.slice(ring_k, &.{ 0, 0, 400 - count_k, 0 }, &.{ 1, 8, 400, 192 }, &.{ 1, 1, 1, 1 });
        const sv = try g.slice(ring_v, &.{ 0, 0, 400 - count_k, 0 }, &.{ 1, 8, 400, 128 }, &.{ 1, 1, 1, 1 });
        const smask = try g.adopt(try module.causalMask(a, keys - m, m, view.first_key, view.key_count, c.window));
        count = try (Attn{ .route = decode_attention.stock, .q = qv, .k = sk, .v = sv, .mask = smask, .sinks = sinks, .s = s }).graph(&g, &outs);
        try g.evalAll(outs[0..count]);
        const freq = module.inverseFrequencies(c.rope_theta, c.rope_head_dim);
        defer _ = mlx.mlx_array_free(freq);
        var rope = try module.rotaryTables(freq, keys - m, m, s);
        defer rope.deinit();
        const attn = try z(&g, &.{ 1, 64, n, 128 }, .bfloat16);
        for ([_]struct { kv: u32, k: *T, v: *T, pos: u32 }{ .{ .kv = 4, .k = &slab_k, .v = &slab_v, .pos = keys - m }, .{ .kv = 8, .k = &ring_k, .v = &ring_v, .pos = 400 - m } }) |lt| {
            const kvh: c_int = @intCast(lt.kv);
            const glue: Glue = .{ .x = x, .attn_o = x, .qp = try z(&g, &.{ n, 64 * 192 }, .bfloat16), .kp = try z(&g, &.{ n, kvh * 192 }, .bfloat16), .vp = try z(&g, &.{ n, kvh * 128 }, .bfloat16), .attn = attn, .mlp = x, .pre = norm, .post = norm, .ones = ones, .value_scale = value_scale, .rope = rope, .kv_heads = lt.kv, .slab_k = lt.k, .slab_v = lt.v, .pos = lt.pos, .m = m, .s = s };
            count = try glue.graph(&g, &outs);
            try g.evalAll(outs[0..count]);
            try testing.expectEqual(@as(c_int, H), g.shapeOf(outs[3]).dim(1));
        }
        g.resetTo(mark);
        std.debug.print("MIMO_M4ATTR_CPU M={d} graphs ok\n", .{m});
    }
}

// ── mimo/k2-glue: K2 and K3 rows (`m4attr_k2`, test builds only) ──

const glue_k2 = @import("glue.zig");
const sliding_k3 = @import("sliding.zig");

/// (K2) one routed layer's glue, `Layer.forward`'s ops or the fused kernels (glue.zig): the boundary add and input norm,
/// the attention residual and post norm, the router (its product and top-8). Inputs: the residual h, the previous
/// layer's MLP output, the attention output; outputs: the normed input, the residual, mlp_in, ids, weights.
const GlueK2 = struct {
    fused: bool,
    k: *const glue_k2.Kernels,
    h: T,
    mlp: T,
    attn: T,
    pre: T,
    post: T,
    ones: T,
    gate: T,
    gate_t: T,
    bias: T,
    s: Stream,
    fn graph(c: *const GlueK2, g: *ops.MlxOps, outs: *[8]T) !usize {
        if (!c.fused) {
            const x = try g.add(c.h, c.mlp);
            outs[0] = try g.adopt(try module.rmsNorm(x, c.pre, c.ones, 1e-6, c.s));
            const residual = try g.add(x, c.attn);
            const mlp_in = try g.adopt(try module.rmsNorm(residual, c.post, c.ones, 1e-6, c.s));
            const r = try module.route(mlp_in, c.gate_t, c.bias, 8, c.s);
            outs[1] = residual;
            outs[2] = mlp_in;
            outs[3] = try g.adopt(r.ids);
            outs[4] = try g.adopt(r.scores);
            return 5;
        }
        const b = try glue_k2.addNorm(c.k, c.h, c.mlp, c.pre, c.s);
        const hb = try g.adopt(b.h);
        outs[0] = try g.adopt(b.n);
        const m = g.shapeOf(c.h).dim(0);
        const post = if (m == 1) try glue_k2.router1(c.k, hb, c.attn, c.post, c.gate, c.s) else try glue_k2.addNormF32(c.k, hb, c.attn, c.post, c.s);
        outs[1] = try g.adopt(post.h);
        outs[2] = try g.adopt(post.n);
        const px = try g.adopt(post.x);
        const logits = if (m == 1) px else try g.matmul(px, c.gate_t);
        const t8 = try glue_k2.top8(c.k, logits, c.bias, c.s);
        outs[3] = try g.adopt(t8.ids);
        outs[4] = try g.adopt(t8.scores);
        return 5;
    }
};

/// (K3) a sliding layer's attention block from its input norm to the o projection: the control (`Attention.forwardOps`
/// on the kernels tier: q/k/v GEMVs, rope, value scale, the ring write, the window kernel, the transpose, o) or the fused
/// route (sliding.zig: qkv, the ring write, window_o, o). The ring is the family's 656-row buffer, written at `pos`.
const SlidingK3 = struct {
    fused: bool,
    sk: *const sliding_k3.Kernels,
    route: decode_attention.Route,
    w: Proj,
    x: T,
    rope: module.RotaryTables,
    value_scale: T,
    sinks: T,
    mask: T,
    ring_k: *T,
    ring_v: *T,
    pos: u32,
    view_first: u32,
    view_end: u32,
    m: u32,
    s: Stream,
    fn graph(c: *const SlidingK3, g: *ops.MlxOps, outs: *[8]T) !usize {
        const n: c_int = @intCast(c.m);
        var q: T = undefined;
        var k: T = undefined;
        var v: T = undefined;
        if (c.fused) {
            const p = try sliding_k3.qkv(c.sk, c.x, c.w.q, c.w.k, c.w.v, c.rope.cos, c.rope.sin, c.value_scale, c.s);
            q = try g.adopt(p.q);
            k = try g.adopt(p.k);
            v = try g.adopt(p.v);
        } else {
            const qp = try g.matmul(c.x, try g.transpose(c.w.q));
            const kp = try g.matmul(c.x, try g.transpose(c.w.k));
            const vp = try g.matmul(c.x, try g.transpose(c.w.v));
            q = try g.adopt(try module.partialRope(try g.adopt(try module.headsView(qp, n, 64, 192, c.s)), c.rope, 64, c.s));
            k = try g.adopt(try module.partialRope(try g.adopt(try module.headsView(kp, n, 8, 192, c.s)), c.rope, 64, c.s));
            v = try g.adopt(try module.scaleValues(try g.adopt(try module.headsView(vp, n, 8, 128, c.s)), c.value_scale, c.s));
        }
        var nk: T = .{};
        var nv: T = .{};
        try mlx.check(mlx.mlx_slice_update(&nk, c.ring_k.*, k, &.{ 0, 0, @intCast(c.pos), 0 }, 4, &.{ 1, 8, @intCast(c.pos + c.m), 192 }, 4, &.{ 1, 1, 1, 1 }, 4, c.s));
        try mlx.check(mlx.mlx_slice_update(&nv, c.ring_v.*, v, &.{ 0, 0, @intCast(c.pos), 0 }, 4, &.{ 1, 8, @intCast(c.pos + c.m), 128 }, 4, &.{ 1, 1, 1, 1 }, 4, c.s));
        // the ring drops its previous handle (its update donates the slot), as the module's
        _ = mlx.mlx_array_free(c.ring_k.*);
        _ = mlx.mlx_array_free(c.ring_v.*);
        c.ring_k.* = nk;
        c.ring_v.* = nv;
        const vk = try g.slice(nk, &.{ 0, 0, @intCast(c.view_first), 0 }, &.{ 1, 8, @intCast(c.view_end), 192 }, &.{ 1, 1, 1, 1 });
        const vv = try g.slice(nv, &.{ 0, 0, @intCast(c.view_first), 0 }, &.{ 1, 8, @intCast(c.view_end), 128 }, &.{ 1, 1, 1, 1 });
        const flat = if (c.fused)
            try g.adopt(try sliding_k3.windowO(c.sk, q, vk, vv, c.sinks, c.s))
        else blk: {
            const attn = try g.adopt(try c.route.run(&c.route, q, vk, vv, 1 / @sqrt(@as(f32, 192)), c.mask, c.sinks, c.s));
            break :blk try g.reshape(try g.adopt(try module.transpose4(attn, c.s)), &.{ n, 64 * 128 });
        };
        outs[0] = try g.matmul(flat, try g.transpose(c.w.o));
        outs[1] = nk;
        outs[2] = nv;
        return 3;
    }
};

/// Static bound: layer 1's BF16 projections (189 MB), layer 5's router (BF16 2 MB + F32 4 MB) and norms, two 656-row
/// sliding buffers (3.4 MB), the scrub (268 MB), a graph's transients (< 64 MiB), the BF16 pack's host staging of the
/// largest tensor (q: 100.7 MB), JIT and allocator slack 512 MiB.
pub fn boundK2K3() u64 {
    const c: mimo.Config = .{};
    const h: u64 = c.hidden_size;
    const proj = 2 * (h * c.n_heads * c.head_dim + h * c.swa_n_kv_heads * (c.head_dim + c.value_head_dim) + h * c.n_heads * c.value_head_dim);
    const router = @as(u64, c.n_routed_experts) * h * (2 + 4) + 4 * h * 2;
    const ring = @as(u64, c.swa_n_kv_heads) * 656 * (c.head_dim + c.value_head_dim) * 2;
    const staging = h * c.n_heads * c.head_dim * 2;
    return proj + router + ring + scrub_bytes + (64 << 20) + staging + (512 << 20);
}

test "mimo GPU M4 attribution K2K3: fused glue (d.glue_*) and the fused sliding attention block (e.layer_sliding_*) vs the control at M = 1..4 on the installed weights" {
    const enabled = std.c.getenv("MIMO_M4ATTR_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const pack_dir = std.mem.span(std.c.getenv("MIMO_M4ATTR_PACK") orelse return error.MimoM4AttrPackRequired);
    const ceiling: u64 = 2 << 30;
    const b_bytes = boundK2K3();
    std.debug.print("MIMO_M4ATTR_K2K3 static: bound={d} ceiling={d} pack={s}\n", .{ b_bytes, ceiling, pack_dir });
    if (b_bytes > ceiling) return error.MimoM4AttrAdmission;
    const a = testing.allocator;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    const rc = probe.mimo_gpu_probe_install();
    if (rc != 0) return error.MimoM4AttrProbeInstall;

    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    var kept: std.ArrayList(T) = .empty;
    defer {
        for (kept.items) |x| g.release(x);
        kept.deinit(a);
    }
    const K = struct {
        fn keep(list: *std.ArrayList(T), al: std.mem.Allocator, x: T) !T {
            try list.append(al, x);
            return x;
        }
    };
    var pack = try Pack.open(a, pack_dir);
    defer pack.close();
    const c: mimo.Config = .{};
    const H: c_int = @intCast(c.hidden_size);
    var buf: [128]u8 = undefined;
    var w: Proj = undefined;
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, [_][2]c_int{ .{ 64 * 192, H }, .{ 8 * 192, H }, .{ 8 * 128, H }, .{ H, 64 * 128 } }, 0..) |nm, sh, i| {
        const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.{s}.weight", .{ sliding_layer, nm });
        const t = try K.keep(&kept, a, try pack.tensor(&g, name, &sh, .bfloat16));
        switch (i) {
            0 => w.q = t,
            1 => w.k = t,
            2 => w.v = t,
            else => w.o = t,
        }
    }
    const gate = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.mlp.gate.weight", &.{ @intCast(c.n_routed_experts), H }, .bfloat16));
    const bias = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.mlp.gate.e_score_correction_bias", &.{@intCast(c.n_routed_experts)}, .float32));
    const sinks = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.1.self_attn.attention_sink_bias", &.{@intCast(c.n_heads)}, .bfloat16));
    const norm_pre = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.input_layernorm.weight", &.{H}, .bfloat16));
    const norm_post = try K.keep(&kept, a, try pack.tensor(&g, "model.layers.5.post_attention_layernorm.weight", &.{H}, .bfloat16));
    const gate_t = blk: {
        const m = g.mark();
        const t = g.keep(try g.transpose(try g.astype(gate, .float32)));
        try g.evalAll(&.{t});
        g.resetTo(m);
        break :blk try K.keep(&kept, a, t);
    };
    const scrub = blk: {
        const z = g.keep(try g.zeros(&.{@intCast(scrub_bytes / 2)}, .bfloat16));
        try g.evalAll(&.{z});
        g.reset();
        break :blk try K.keep(&kept, a, z);
    };
    const ones = blk: {
        var o: T = .{};
        try mlx.check(mlx.mlx_ones(&o, &.{H}, 1, .bfloat16, s));
        break :blk try K.keep(&kept, a, o);
    };
    const value_scale = try K.keep(&kept, a, mlx.mlx_array_new_float(c.attention_value_scale));
    var gk = try glue_k2.Kernels.init(c);
    defer gk.deinit();
    var sk = try sliding_k3.Kernels.init(c);
    defer sk.deinit();
    var dk = try decode_attention.Kernels.init(c);
    defer dk.deinit();
    const kr = decode_attention.install(.kernels, .sliding, keys, &dk);
    var ring_k = try randomBf16(a, &g, &.{ 1, 8, 656, 192 }, 1, 0x33);
    var ring_v = try randomBf16(a, &g, &.{ 1, 8, 656, 128 }, 1, 0x44);
    defer g.release(ring_k);
    defer g.release(ring_v);
    g.reset();

    const rows_buf = try a.alloc(Row, probe_cap);
    defer a.free(rows_buf);
    const spans = try a.alloc([2]f64, probe_cap);
    defer a.free(spans);
    var bench: Bench = .{ .a = a, .g = &g, .s = s, .scrub = scrub, .rows = rows_buf, .spans = spans };

    const freq = module.inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
    defer _ = mlx.mlx_array_free(freq);
    var res: [4][4]f64 = undefined; // [M - 1][glue ops, glue fused, sliding ops, sliding fused]
    var exact: [4][2]bool = undefined;
    for (1..5) |mu| {
        const m: u32 = @intCast(mu);
        const n: c_int = @intCast(m);
        const h = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 4, 0xe00 + m));
        const mlp = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 2, 0xe10 + m));
        const attn = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 1, 0xe20 + m));
        const x = try K.keep(&kept, a, try randomBf16(a, &g, &.{ n, H }, 1, 0xe30 + m));
        for ([_]bool{ false, true }, 0..) |fused, fi| {
            res[mu - 1][fi] = (try bench.measure(if (fused) "glue_fused" else "glue_ops", m, &GlueK2{ .fused = fused, .k = &gk, .h = h, .mlp = mlp, .attn = attn, .pre = norm_pre, .post = norm_post, .ones = ones, .gate = gate, .gate_t = gate_t, .bias = bias, .s = s })).gpu_ms;
        }
        var rope = try module.rotaryTables(freq, keys - m, m, s);
        defer rope.deinit();
        const view = mimo.windowView(keys - m, m, c.window);
        const end: u32 = 400;
        const smask = try K.keep(&kept, a, g.keep(try g.adopt(try module.causalMask(a, keys - m, m, view.first_key, view.key_count, c.window))));
        for ([_]bool{ false, true }, 0..) |fused, fi| {
            res[mu - 1][2 + fi] = (try bench.measure(if (fused) "layer_sliding_fused" else "layer_sliding_ops", m, &SlidingK3{ .fused = fused, .sk = &sk, .route = kr, .w = w, .x = x, .rope = rope, .value_scale = value_scale, .sinks = sinks, .mask = smask, .ring_k = &ring_k, .ring_v = &ring_v, .pos = end - m, .view_first = end - view.key_count, .view_end = end, .m = m, .s = s })).gpu_ms;
        }
        // The same graphs once more, compared bit for bit on the real weights (the device self-check's claim, here).
        {
            const mark = g.mark();
            defer g.resetTo(mark);
            var o_ops: [8]T = undefined;
            var o_fu: [8]T = undefined;
            _ = try (GlueK2{ .fused = false, .k = &gk, .h = h, .mlp = mlp, .attn = attn, .pre = norm_pre, .post = norm_post, .ones = ones, .gate = gate, .gate_t = gate_t, .bias = bias, .s = s }).graph(&g, &o_ops);
            _ = try (GlueK2{ .fused = true, .k = &gk, .h = h, .mlp = mlp, .attn = attn, .pre = norm_pre, .post = norm_post, .ones = ones, .gate = gate, .gate_t = gate_t, .bias = bias, .s = s }).graph(&g, &o_fu);
            var glue_ok = true;
            for (0..5) |i| {
                const y = if (i == 3) try g.astype(o_ops[i], .uint32) else o_ops[i];
                const b = try glue_k2.bitwise(a, o_fu[i], y, s);
                glue_ok = glue_ok and b.equal == b.total;
            }
            var s_ops: [8]T = undefined;
            var s_fu: [8]T = undefined;
            _ = try (SlidingK3{ .fused = false, .sk = &sk, .route = kr, .w = w, .x = x, .rope = rope, .value_scale = value_scale, .sinks = sinks, .mask = smask, .ring_k = &ring_k, .ring_v = &ring_v, .pos = end - m, .view_first = end - view.key_count, .view_end = end, .m = m, .s = s }).graph(&g, &s_ops);
            try g.evalAll(s_ops[0..3]);
            _ = try (SlidingK3{ .fused = true, .sk = &sk, .route = kr, .w = w, .x = x, .rope = rope, .value_scale = value_scale, .sinks = sinks, .mask = smask, .ring_k = &ring_k, .ring_v = &ring_v, .pos = end - m, .view_first = end - view.key_count, .view_end = end, .m = m, .s = s }).graph(&g, &s_fu);
            const bs = try glue_k2.bitwise(a, s_fu[0], s_ops[0], s);
            exact[mu - 1] = .{ glue_ok, bs.equal == bs.total };
        }
        std.debug.print("MIMO_M4ATTR_K2K3 M={d} glue_ops_ms={d:.4} glue_fused_ms={d:.4} sliding_ops_ms={d:.4} sliding_fused_ms={d:.4} glue_bitwise={} sliding_bitwise={}\n", .{ m, res[mu - 1][0], res[mu - 1][1], res[mu - 1][2], res[mu - 1][3], exact[mu - 1][0], exact[mu - 1][1] });
        g.reset();
    }
    const routed_layers: u32 = 47;
    const sliding_n: u32 = 39;
    for ([_]struct { name: []const u8, layers: u32, col: usize, note: []const u8 }{
        .{ .name = "d.glue_ops", .layers = routed_layers, .col = 0, .note = "control: boundary add + input norm, residual + post norm, router (product + top-8)" },
        .{ .name = "d.glue_fused", .layers = routed_layers, .col = 1, .note = "K2: addnorm, router1 (M = 1) / addnorm32 + MLX matmul, top8" },
        .{ .name = "e.layer_sliding_ops", .layers = sliding_n, .col = 2, .note = "control: q/k/v + rope + v scale + ring write + window kernel + transpose + o" },
        .{ .name = "e.layer_sliding_fused", .layers = sliding_n, .col = 3, .note = "K3: qkv + ring write + window_o + o" },
    }) |l| {
        printRow(.{ .name = l.name, .layers = l.layers, .m1 = res[0][l.col], .m4 = res[3][l.col], .note = l.note });
        std.debug.print("MIMO_M4ATTR_K2K3 row={s} m1_ms_per_layer={d:.4} m2_ms_per_layer={d:.4} m3_ms_per_layer={d:.4} m4_ms_per_layer={d:.4}\n", .{ l.name, res[0][l.col], res[1][l.col], res[2][l.col], res[3][l.col] });
    }
    for (0..4) |mi| {
        const dg = (res[mi][1] - res[mi][0]) * routed_layers;
        const ds = (res[mi][3] - res[mi][2]) * sliding_n;
        std.debug.print("MIMO_M4ATTR_K2K3_SUM M={d} glue_delta_ms_per_forward={d:.3} sliding_delta_ms_per_forward={d:.3} total_delta_ms={d:.3} exact={}\n", .{ mi + 1, dg, ds, dg + ds, exact[mi][0] and exact[mi][1] });
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_M4ATTR_K2K3 peak: mlx={d} bound={d}\n", .{ peak, b_bytes });
    for (exact) |e| if (!(e[0] and e[1])) return error.MimoM4AttrK2K3NotExact;
    if (peak > b_bytes) return error.MimoM4AttrPeakExceeded;
}
