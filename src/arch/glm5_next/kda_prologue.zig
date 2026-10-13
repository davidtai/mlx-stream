//! GLM-5.3-Flash's KDA prompt prologue in one dispatch (`kernels/glm5n_kda_prologue.metal`): the short causal conv over
//! [q | k | v], SiLU, the bf16 cast, the q / k L2 norms and scales and the next conv tail, read from the three bf16
//! projection outputs and the state's f32 tail and written as contiguous q / k / v [1, R, hv, dk] bf16 and the f32 tail.
//! It replaces the op chain's (kda.zig `prologueOps`) concat, f32 cast, window concat, conv, SiLU, bf16 cast, slice
//! copies, two rms_norms and two scale multiplies: about 1.25 GB of traffic a 1,024-row span down to about 0.1 GB.
//! Each step is MLX's kernel for the op it replaces, in the same operand order and types (the text's header names them),
//! so the outputs are the op chain's bits; the window's parity test proves it bit for bit at Flash's shapes.
//!
//! The route is chosen once, at construction (`install`, from Module.init, by `settings.kda_prologue`): `fused` (the
//! default) on a GPU stream installs the text as `prompt` and the load logs
//! `[kda] fused prompt prologue engaged (conv + silu + q/k rms, 32-thread groups per row x head)`; `ops` (the control),
//! or a CPU stream, keeps the op chain and the load logs why. kda.zig's forward runs `prompt` on every KDA span wider than
//! the decode lane (only the prompt pass has one); the decode lane and the MTP verify's capture run the op chain under
//! either. The routes line names the installed route (`kda_prologue=fused|ops`). No bill term: the text's live arrays
//! are a subset of the op chain's (bill.zig `kdaRowBytes` stays the bound).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const log = sdk.log;
const glm = @import("config.zig");
const kda = @import("kda.zig");
const kda_percore = @import("kda_percore.zig");
const settings = @import("settings.zig");

const G = kda.G;
const T = kda.T;

/// The key head width the text is written for (32 lanes x 4 reads: MLX's rms_single_row at axis 128).
const head_dim: c_int = 128;
/// (row, section, head) simdgroups per threadgroup.
const groups_per_tg: c_int = 8;

/// The prompt prologue a construction installs (the routes line's `kda_prologue=`).
pub const Route = settings.KdaPrologue;
/// Why a construction keeps the op chain: the setting names it, or a CPU stream.
pub const Decline = enum { setting, cpu_stream };

/// The construction's choice: the text on a GPU stream unless the setting names the op chain.
pub fn choose(gpu: bool, setting: Route) ?Decline {
    if (setting == .ops) return .setting;
    if (!gpu) return .cpu_stream;
    return null;
}

/// The prologue of a span of `rows` rows: the installed prompt prologue past the decode lane, the op chain within it.
pub fn spanPrologue(rows: c_int) *const kda.PrologueFn {
    return if (rows > kda_percore.lane_rows) prompt else &kda.prologueOps;
}

/// The prompt prologue, set by `install` at construction, before any forward.
var prompt: *const kda.PrologueFn = &kda.prologueOps;
var kernel_obj: ?mlx.mlx_fast_metal_kernel = null;

const kernel_source = @embedFile("kernels/glm5n_kda_prologue.metal");

comptime {
    // The next tail is the span's last taps - 1 rows: every span the text takes is at least that wide.
    if (kda_percore.lane_rows + 1 < 3) @compileError("glm5_next kda prologue: a prompt span narrower than the conv tail");
}

/// The prompt prologue, once, at construction (no device work: the library compiles at its first launch, the load's
/// prompt-kernel warm-up). Refuses a geometry the text is not written for by name.
pub fn install(g: *G, c: *const glm.Config, setting: Route) !Route {
    if (choose(mlx.streamIsGpu(g.s), setting)) |why| {
        prompt = &kda.prologueOps;
        log.info("[kda] fused prompt prologue declined ({t}); the prompt pass runs the op chain\n", .{why});
        return .ops;
    }
    const taps: c_int = @intCast(c.kda_conv);
    const heads: c_int = @intCast(c.kda_heads);
    if (c.kda_head_dim != 128 or taps < 2 or taps - 1 > kda_percore.lane_rows + 1 or @rem(3 * heads, groups_per_tg) != 0)
        return error.Glm5NextKdaPrologueGeometry;
    _ = try kernel();
    prompt = &prologueFused;
    log.info("[kda] fused prompt prologue engaged (conv + silu + q/k rms, 32-thread groups per row x head)\n", .{});
    return .fused;
}

/// The kernel object, built once.
fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_obj) |k| return k;
    const ins = [_][*:0]const u8{ "qp", "kp", "vp", "tail", "conv_w", "q_scale", "k_scale", "rows" };
    const outs = [_][*:0]const u8{ "q_out", "k_out", "v_out", "tail_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new("glm5_next_kda_prologue", iv, ov, kernel_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_obj = k;
    return k;
}

/// The text over one span (`prompt` once `install` engaged it): the same inputs as the op chain, and its two bf16
/// scalars (`kda.qkScales` through `G.scalar`) as the scales.
fn prologueFused(g: *G, c: *const glm.Config, w: *const kda.Weights, qp: T, kp: T, vp: T, tail: T) anyerror!kda.Prologue {
    const rows = g.shapeOf(qp).dim(1);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim: c_int = hv * dk;
    const cdim: c_int = 3 * dim;
    const taps: c_int = @intCast(c.kda_conv);
    const dt = mlx.mlx_array_dtype(qp);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, hv, dk }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, taps - 1, cdim }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 3 * hv, rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, groups_per_tg, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "DK", dk }, .{ "HV", hv }, .{ "DIM", dim }, .{ "CDIM", cdim }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const scales = kda.qkScales(c);
    const rows_arr = try g.adopt(mlx.mlx_array_new_int(rows));
    const ins = [_]T{ qp, kp, vp, tail, w.conv_w, try g.scalar(scales.q, .bfloat16), try g.scalar(scales.k, .bfloat16), rows_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kernel_obj.?, iv, cfg, g.s));
    var out: [4]T = undefined;
    for (&out, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&x, ov, i));
        o.* = try g.adopt(x);
    }
    return .{ .q = out[0], .k = out[1], .v = out[2], .next_tail = out[3] };
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "glm5_next kda prologue CPU route: the text on a GPU stream unless the setting names ops; the decode lane's spans take the op chain under either" {
    try testing.expectEqual(@as(?Decline, null), choose(true, .fused));
    try testing.expectEqual(@as(?Decline, .setting), choose(true, .ops));
    try testing.expectEqual(@as(?Decline, .cpu_stream), choose(false, .fused));
    try testing.expectEqual(@as(?Decline, .setting), choose(false, .ops));
    const saved = prompt;
    defer prompt = saved;
    for ([_]*const kda.PrologueFn{ &kda.prologueOps, &prologueFused }) |installed| {
        prompt = installed;
        var rows: c_int = 1;
        while (rows <= kda_percore.lane_rows) : (rows += 1) try testing.expect(spanPrologue(rows) == &kda.prologueOps);
        for ([_]c_int{ kda_percore.lane_rows + 1, 37, 1024 }) |r| try testing.expect(spanPrologue(r) == installed);
    }
}

test "glm5_next kda prologue CPU: Flash's geometry is the text's (dk 128, 4 taps, 3 x heads a multiple of the groups)" {
    const c: glm.Config = .{};
    try testing.expectEqual(@as(u32, @intCast(head_dim)), c.kda_head_dim);
    try testing.expect(@as(c_int, @intCast(c.kda_conv)) - 1 <= kda_percore.lane_rows + 1);
    try testing.expectEqual(@as(c_int, 0), @rem(3 * @as(c_int, @intCast(c.kda_heads)), groups_per_tg));
    // The text's lane map: 32 lanes x 4 channels cover a head's 128 channels once.
    var cover: [128]u8 = @splat(0);
    for (0..32) |lane| for (0..4) |j| {
        cover[lane * 4 + j] += 1;
    };
    for (cover) |n| try testing.expectEqual(@as(u8, 1), n);
}

// ── the window's parity test ──────────────────────────────────────────────

/// Static bound (declared before any run): the synthetic layer's residents (q / k / v / o 4 x 64 MiB BF16, the gate
/// stages and conv < 8 MiB, the f32 conv copy 0.4 MiB), one 1,024-row span's two prologues (the op chain's f32 window
/// and conv stages 4 x 101 MB, its bf16 stages 4 x 50 MB; the text's outputs 50 MB), two forwards' arrays (< 0.4 GB),
/// the host read-backs (4 x 33.6 MB f32 per compared pair): < 1.4 GB plus the texts' JIT; declared 1.6 GB.
const parity_bound_bytes: u64 = 1_600_000_000;

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn randBf16(g: *G, a: std.mem.Allocator, r: std.Random, shape: []const c_int, sd: f32, mean: f32) !T {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try a.alloc(u16, n);
    defer a.free(v);
    for (v) |*x| x.* = bf16Bits(mean + sd * r.floatNorm(f32));
    const arr = g.keep(try g.hostArray(std.mem.sliceAsBytes(v), shape, .bfloat16));
    try g.evalAll(&.{arr});
    return arr;
}

/// Elements of `x` and `y` (any float dtype, read as f32: a bf16 widens exactly) whose bits differ.
fn bitDiffs(g: *G, a: std.mem.Allocator, x: T, y: T) !usize {
    const xf = try g.astype(x, .float32);
    const yf = try g.astype(y, .float32);
    try g.evalAll(&.{ xf, yf });
    const n = mlx.mlx_array_size(xf);
    if (n != mlx.mlx_array_size(yf)) return error.ShapeMismatch;
    const xv = try a.alloc(f32, n);
    defer a.free(xv);
    const yv = try a.alloc(f32, n);
    defer a.free(yv);
    _ = try g.hostF32(xf, xv);
    _ = try g.hostF32(yf, yv);
    var d: usize = 0;
    for (xv, yv) |p, q| d += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
    return d;
}

test "glm5_next kda prologue GPU: the fused text against the op chain at Flash's shapes (spans of 1,024, a carried 1,024 and 37 rows): q, k, v, the tail and the whole layer's output and state bit for bit" {
    _ = std.c.getenv("GLM5NEXT_KDA_PROLOGUE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5N_KDA_PROLOGUE_PARITY bound_bytes={d}\n", .{parity_bound_bytes});
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: glm.Config = .{};
    const graph = @import("graph.zig");
    var prng = std.Random.DefaultPrng.init(0x6b6470);
    const r = prng.random();
    const h: c_int = @intCast(c.hidden_size);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim = hv * dk;
    const sd = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
    var kept: std.ArrayList(T) = .empty;
    defer {
        for (kept.items) |x| g.release(x);
        kept.deinit(a);
    }
    const W = struct {
        fn mk(gg: *G, al: std.mem.Allocator, rr: std.Random, list: *std.ArrayList(T), shape: []const c_int, d: f32, m: f32) !T {
            const x = try randBf16(gg, al, rr, shape, d, m);
            try list.append(al, x);
            return x;
        }
    };
    // The layer as the bind builds it (kda_decode_parity.zig's synthetic layer).
    const conv_raw = try W.mk(&g, a, r, &kept, &.{ 3 * dim, 1, @intCast(c.kda_conv) }, 0.5, 0.0);
    const a_log = try W.mk(&g, a, r, &kept, &.{hv}, 0.5, 0.0);
    const dt_raw = try W.mk(&g, a, r, &kept, &.{dim}, 0.5, 0.0);
    const a_exp = blk: {
        const f = try g.astype(a_log, .float32);
        const e = try g.exp(try g.reshape(try g.broadcastTo(try g.reshape(f, &.{ hv, 1 }), &.{ hv, dk }), &.{dim}));
        try g.evalAll(&.{e});
        break :blk g.keep(e);
    };
    try kept.append(a, a_exp);
    const a_log32 = blk: {
        const f = try g.astype(a_log, .float32);
        try g.evalAll(&.{f});
        break :blk g.keep(f);
    };
    try kept.append(a, a_log32);
    const dt_bias = try graph.f32Copy(s, dt_raw, false);
    defer _ = mlx.mlx_array_free(dt_bias);
    const conv_t = try g.transposeAxes(conv_raw, &.{ 0, 2, 1 });
    const conv_w = try graph.f32Copy(s, conv_t, false);
    defer _ = mlx.mlx_array_free(conv_w);
    const w: kda.Weights = .{
        .q = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dim, h }, sd, 0.0) },
        .k = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dim, h }, sd, 0.0) },
        .v = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dim, h }, sd, 0.0) },
        .f_a = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dk, h }, sd, 0.0) },
        .f_b = try W.mk(&g, a, r, &kept, &.{ dim, dk }, 0.1, 0.0),
        .g_a = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ dk, h }, sd, 0.0) },
        .g_b = try W.mk(&g, a, r, &kept, &.{ dim, dk }, 0.1, 0.0),
        .b = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ hv, h }, sd, 0.0) },
        .a_log = a_log32,
        .a_exp = a_exp,
        .dt_bias = dt_bias,
        .conv_w = conv_w,
        .conv_raw = conv_raw,
        .o_norm = try W.mk(&g, a, r, &kept, &.{dk}, 0.2, 1.0),
        .o = .{ .bf16 = try W.mk(&g, a, r, &kept, &.{ h, dim }, 1.0 / @sqrt(@as(f32, @floatFromInt(dim))), 0.0) },
    };
    // The module's installs: the recurrence's kernels, the per-core prompt route, then this route.
    try kda.install(&c);
    const saved = prompt;
    defer prompt = saved;
    _ = try kda_percore.install(&g, &c);
    try testing.expectEqual(Route.fused, try install(&g, &c, .fused));
    // Two layer states carried across the spans: one through the op chain, one through the text.
    var s_ops: kda.State = .{};
    defer s_ops.reset(&g);
    var s_txt: kda.State = .{};
    defer s_txt.reset(&g);
    var total: usize = 0;
    for ([_]c_int{ 1024, 1024, 37 }, 0..) |rows, span| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try randBf16(&g, a, r, &.{ 1, rows, h }, 1.0, 0.0);
        defer g.release(x);
        // The prologues alone, from the same projections and the op chain's carried tail.
        const qp = try g.matmul(x, try g.transpose(w.q.bf16));
        const kp = try g.matmul(x, try g.transpose(w.k.bf16));
        const vp = try g.matmul(x, try g.transpose(w.v.bf16));
        const tail = s_ops.conv orelse try g.zeros(&.{ 1, @as(c_int, @intCast(c.kda_conv)) - 1, 3 * dim }, .float32);
        const po = try kda.prologueOps(&g, &c, &w, qp, kp, vp, tail);
        const pf = try prologueFused(&g, &c, &w, qp, kp, vp, tail);
        const d_q = try bitDiffs(&g, a, pf.q, po.q);
        const d_k = try bitDiffs(&g, a, pf.k, po.k);
        const d_v = try bitDiffs(&g, a, pf.v, po.v);
        const d_t = try bitDiffs(&g, a, pf.next_tail, po.next_tail);
        // The whole layer: kda.forward with the op chain installed, then with the text (each carrying its own state).
        prompt = &kda.prologueOps;
        const y_ops = try kda.forward(&g, &c, &w, x, &s_ops);
        prompt = &prologueFused;
        const y_txt = try kda.forward(&g, &c, &w, x, &s_txt);
        try g.evalAll(&.{ y_ops, y_txt, s_ops.rec.?, s_txt.rec.?, s_ops.conv.?, s_txt.conv.? });
        const d_y = try bitDiffs(&g, a, y_txt, y_ops);
        const d_s = try bitDiffs(&g, a, s_txt.rec.?, s_ops.rec.?);
        const d_c = try bitDiffs(&g, a, s_txt.conv.?, s_ops.conv.?);
        std.debug.print("GLM5N_KDA_PROLOGUE_PARITY span={d} rows={d} bit_diffs q={d} k={d} v={d} tail={d} layer_y={d} state={d} conv_tail={d}\n", .{ span, rows, d_q, d_k, d_v, d_t, d_y, d_s, d_c });
        total += d_q + d_k + d_v + d_t + d_y + d_s + d_c;
    }
    std.debug.print("GLM5N_KDA_PROLOGUE_PARITY {s} total_bit_diffs={d}\n", .{ if (total == 0) "PASS" else "FAIL", total });
    try testing.expectEqual(@as(usize, 0), total);
}
