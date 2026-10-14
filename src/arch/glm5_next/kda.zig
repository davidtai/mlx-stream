//! GLM-5.3-Flash's Kimi Delta Attention (KDA) layers: the q / k / v projections through the short causal conv and SiLU,
//! the per-head L2 norms, the per-channel bounded-sigmoid forget gate from the low-rank `f` projections, beta = sigmoid(b),
//! the delta-rule recurrence over an f32 state, and the gated RMSNorm (sigmoid of the low-rank `g` projections) into the
//! output projection. The per-token route runs mlx-serve's stock gated-delta kernel with the per-channel gate
//! (`transformer.zig` `gdnKernelSource(vectorized = true)`, mlx-lm's `_make_gated_delta_kernel(vectorized=True)`: the exact
//! recurrence, one token at a time), copied here, over the materialized f32 gate. A span wider than the decode lane (the
//! prompt pass's) runs the prompt route installed at construction (kda_percore.zig: the host's per-core kernel where this
//! GPU takes it, else the per-token route); the host's fused decode kernel is not used (it needs quantized gate
//! projections). The MTP lane's verify (`forwardCapture`) runs the host's capture variant of the stock kernel
//! (`gdnKernelSource(vectorized, capture_seq)`): besides the final state it writes the state after every row but the
//! last, so a partial accept restores row a's state (`restore`) by a copy, with no re-forward.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const fp8 = @import("fp8.zig");
const kda_percore = @import("kda_percore.zig");
const kda_prologue = @import("kda_prologue.zig");
const kda_verify = @import("kda_verify.zig");

comptime {
    // kda_percore.zig's, kda_prologue.zig's and kda_verify.zig's tests run with this file's (src/tests.zig lists kda.zig).
    _ = kda_percore;
    _ = kda_prologue;
    _ = kda_verify;
}

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// One KDA layer's residents (the pack's projections, `x @ w^T`: BF16, or under `residents_format: fp8_block128` the
/// FP8 route's, fp8.zig), with the f32 tables its gate reads.
pub const Weights = struct {
    q: fp8.Linear,
    k: fp8.Linear,
    v: fp8.Linear,
    /// The forget gate's low-rank stages [rank, hidden] and [hv * dk, rank]; the output gate's alike (the second stages
    /// BF16 in either format: K = 128).
    f_a: fp8.Linear,
    f_b: T,
    g_a: fp8.Linear,
    g_b: T,
    /// beta's projection [heads, hidden].
    b: fp8.Linear,
    /// f32 [hv]: A_log as its own F32 table, converted once at bind (the pack stores F32 or BF16 by layer); the per-core
    /// kernel takes exp(A_log) of it itself.
    a_log: T,
    /// f32 [hv * dk]: exp(A_log) of each head repeated over its key channels (the gate's constant factor, built once).
    a_exp: T,
    /// f32 [hv * dk].
    dt_bias: T,
    /// f32 [conv dim, taps, 1]: the depthwise conv in MLX's layout.
    conv_w: T,
    /// The pack's BF16 conv weight [conv dim, 1, taps] as stored (the fused decode text reads it: kda_decode.zig).
    conv_raw: T,
    /// [dv]: the output RMSNorm's weight, per head.
    o_norm: T,
    o: fp8.Linear,
};

/// One request's state of a KDA layer: the conv's last `taps - 1` inputs [1, taps - 1, conv dim] (f32, as the conv
/// reads them) and the recurrent state [1, hv, dv, dk] (f32). Kept handles (`G.keep`); null before the first token.
pub const State = struct {
    conv: ?T = null,
    rec: ?T = null,

    pub fn reset(st: *State, g: *G) void {
        if (st.conv) |x| g.release(x);
        if (st.rec) |x| g.release(x);
        st.* = .{};
    }

    pub fn set(st: *State, g: *G, conv: T, rec: T) void {
        st.reset(g);
        st.conv = g.keep(conv);
        st.rec = g.keep(rec);
    }

    /// The bytes one state holds (the bill's).
    pub fn bytes(c: *const glm.Config) u64 {
        return @as(u64, c.kda_conv - 1) * c.kdaConvDim() * 4 + @as(u64, c.kda_heads) * c.kda_head_dim * c.kda_head_dim * 4;
    }
};

/// The MTP verify's capture of one KDA layer (`forwardCapture`): the recurrence's state after each of its rows but the
/// last (`seq`, f32 [T, 1, hv, dv, dk]; row T - 1 is never written: a full accept keeps `State.rec`) and the conv's
/// window (f32 [1, taps - 1 + T, conv dim]: the state's tail, then the rows), so row a's state is `seq[a]` and its conv
/// tail the window's rows [a + 1, a + taps). Kept handles, released after the round.
pub const Capture = struct {
    seq: ?T = null,
    window: ?T = null,
    rows: u32 = 0,

    pub fn release(cap: *Capture, g: *G) void {
        if (cap.seq) |x| g.release(x);
        if (cap.window) |x| g.release(x);
        cap.* = .{};
    }

    /// One verify row's capture of one layer (its recurrent state and its conv tail): the bill's rate.
    pub fn rowBytes(c: *const glm.Config) u64 {
        return State.bytes(c);
    }
};

fn linear(g: *G, x: T, w: T) !T {
    return g.matmul(x, try g.transpose(w));
}

fn rmsNormOnes(g: *G, x: T, ones: T, eps: f32) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, ones, eps, g.s));
    return g.adopt(o);
}

/// x [1, R, hidden] (R rows after `st`'s positions) -> [1, R, hidden]; `st` advanced past them.
pub fn forward(g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State) !T {
    return forwardWith(g, c, w, x, st, false, {});
}

/// The decode lane's KDA layer, bound at construction (`settings.decode_kda`): the op chain (`forward`, the control) or
/// the fused text (kda_decode.zig `run` over its installed kernels, `ctx`). Same contract as `forward`.
pub const Step = struct {
    ctx: ?*const anyopaque = null,
    run: *const fn (ctx: ?*const anyopaque, g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State) anyerror!T = forwardOps,
};

fn forwardOps(_: ?*const anyopaque, g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State) anyerror!T {
    return forward(g, c, w, x, st);
}

/// `forward` with every row's state captured into `cap` (the MTP verify's rows: `restore` rolls back to any of them).
/// Its own comptime instance: the serial forward's code carries no capture branch.
pub fn forwardCapture(g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State, cap: *Capture) !T {
    return forwardWith(g, c, w, x, st, true, cap);
}

/// The MTP verify's KDA layer, bound at construction (`settings.kda_verify`): the op chain (`forwardCapture`, the
/// default and the control) or the fused verify text (kda_verify.zig `run` over its installed kernels, `ctx`). Same
/// contract as `forwardCapture`.
pub const VerifyStep = struct {
    ctx: ?*const anyopaque = null,
    run: *const fn (ctx: ?*const anyopaque, g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State, cap: *Capture) anyerror!T = forwardCaptureOps,
};

fn forwardCaptureOps(_: ?*const anyopaque, g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State, cap: *Capture) anyerror!T {
    return forwardCapture(g, c, w, x, st, cap);
}

/// The state after verify row `row` (a partial accept: `row` < the capture's rows - 1), copied into `st` from `cap`
/// (its own buffers: the capture is released after the round).
pub fn restore(g: *G, c: *const glm.Config, st: *State, cap: *const Capture, row: u32) !void {
    std.debug.assert(row + 1 < cap.rows);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const taps: c_int = @intCast(c.kda_conv);
    const cdim: c_int = 3 * hv * dk;
    const r: c_int = @intCast(row);
    const one = try g.scalar(1.0, .float32);
    const rec_row = try g.slice(cap.seq.?, &.{ r, 0, 0, 0, 0 }, &.{ r + 1, 1, hv, dk, dk }, &.{ 1, 1, 1, 1, 1 });
    const rec = try g.mul(try g.reshape(rec_row, &.{ 1, hv, dk, dk }), one);
    const conv = try g.mul(try g.slice(cap.window.?, &.{ 0, r + 1, 0 }, &.{ 1, r + taps, cdim }, &.{ 1, 1, 1 }), one);
    st.set(g, conv, rec);
}

fn forwardWith(g: *G, c: *const glm.Config, w: *const Weights, x: T, st: *State, comptime capture: bool, cap: if (capture) *Capture else void) !T {
    const sh = g.shapeOf(x);
    const rows = sh.dim(1);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim: c_int = hv * dk;
    const cdim: c_int = 3 * dim;
    const taps: c_int = @intCast(c.kda_conv);
    const head4 = [_]c_int{ 1, rows, hv, dk };
    // The prologue (the short conv, SiLU, the q / k norms, the next tail) from the three projections and the state's
    // tail: past the decode lane the prompt prologue installed at construction (kda_prologue.zig), else the op chain.
    // The MTP verify's capture keeps the op chain (it captures the conv window).
    const qp = try fp8.project(g, w.q, x);
    const kp = try fp8.project(g, w.k, x);
    const vp = try fp8.project(g, w.v, x);
    const tail = st.conv orelse try g.zeros(&.{ 1, taps - 1, cdim }, .float32);
    const pro = if (capture) try prologueOps(g, c, w, qp, kp, vp, tail) else try kda_prologue.spanPrologue(rows)(g, c, w, qp, kp, vp, tail);
    const qs = pro.q;
    const ks = pro.k;
    const v = pro.v;
    const next_tail = pro.next_tail;
    // The forget gate's pre-activation per key channel (each route makes the gate of it), beta, the state.
    const a = try linear(g, try fp8.project(g, w.f_a, x), w.f_b);
    const beta = try g.sigmoid(try fp8.project(g, w.b, x));
    const state = st.rec orelse try g.zeros(&.{ 1, hv, dk, dk }, .float32);
    const r = if (capture) blk: {
        // The MTP verify's rows: the stock kernel's capture variant over the f32 gate.
        const rs = try recurSeq(g, qs, ks, v, try gateOf(g, c, w, a), beta, state, rows, hv, dk);
        cap.release(g);
        cap.* = .{ .seq = g.keep(rs.seq), .window = g.keep(pro.window.?), .rows = @intCast(rows) };
        break :blk Recur{ .y = rs.y, .state = rs.state };
    } else blk: {
        // The span's route: past the decode lane the prompt route installed at construction, else the per-token route.
        const route = kda_percore.spanRoute(rows);
        break :blk try route(g, c, w, qs, ks, v, a, beta, state);
    };
    // The gated RMSNorm: rms_norm(y) per head (o_norm) times sigmoid(g_b(g_a(x))).
    const z = try g.reshape(try linear(g, try fp8.project(g, w.g_a, x), w.g_b), &head4);
    var yn = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&yn, r.y, w.o_norm, c.rms_norm_eps, g.s));
    const gated = try g.mul(try g.adopt(yn), try g.sigmoid(z));
    st.set(g, next_tail, r.state);
    return fp8.project(g, w.o, try g.reshape(gated, &.{ 1, rows, dim }));
}

/// One span's prologue: q and k [1, R, hv, dk] bf16 (normalized and scaled), v [1, R, hv, dk] bf16, the next conv tail
/// f32 [1, taps - 1, conv dim], and (the op chain's only) the conv window the MTP capture keeps.
pub const Prologue = struct { q: T, k: T, v: T, next_tail: T, window: ?T = null };

/// A prologue route over one span: the q / k / v projections [1, R, hv * dk] (the trunk's dtype) and the state's conv
/// tail f32 [1, taps - 1, conv dim] -> `Prologue`.
pub const PrologueFn = fn (g: *G, c: *const glm.Config, w: *const Weights, qp: T, kp: T, vp: T, tail: T) anyerror!Prologue;

/// The op chain (the decode lane's, the capture's, and the prompt pass's control): the short conv over [q | k | v] in
/// f32, its window the state's tail then the rows; SiLU; bf16; q = rms_norm(q) / dk and k = rms_norm(k) / sqrt(dk) (the
/// L2 norms with the reference's scales).
pub fn prologueOps(g: *G, c: *const glm.Config, w: *const Weights, qp: T, kp: T, vp: T, tail: T) anyerror!Prologue {
    const rows = g.shapeOf(qp).dim(1);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim: c_int = hv * dk;
    const cdim: c_int = 3 * dim;
    const taps: c_int = @intCast(c.kda_conv);
    const qkv = try g.astype(try g.concat(&.{ qp, kp, vp }, 2), .float32);
    const window = try g.concat(&.{ tail, qkv }, 1);
    var conv_raw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&conv_raw, window, w.conv_w, 1, 0, 1, cdim, g.s));
    const conv = try g.astype(try g.silu(try g.adopt(conv_raw)), .bfloat16);
    // Its own buffer (times one): a view would keep the whole window alive across tokens.
    const next_tail = try g.mul(try g.slice(window, &.{ 0, rows, 0 }, &.{ 1, rows + taps - 1, cdim }, &.{ 1, 1, 1 }), try g.scalar(1.0, .float32));
    const head4 = [_]c_int{ 1, rows, hv, dk };
    const q = try g.reshape(try g.slice(conv, &.{ 0, 0, 0 }, &.{ 1, rows, dim }, &.{ 1, 1, 1 }), &head4);
    const k = try g.reshape(try g.slice(conv, &.{ 0, 0, dim }, &.{ 1, rows, 2 * dim }, &.{ 1, 1, 1 }), &head4);
    const v = try g.reshape(try g.slice(conv, &.{ 0, 0, 2 * dim }, &.{ 1, rows, cdim }, &.{ 1, 1, 1 }), &head4);
    const ones = try g.ones(&.{dk}, .bfloat16);
    const scales = qkScales(c);
    const qs = try g.mul(try rmsNormOnes(g, q, ones, 1e-6), try g.scalar(scales.q, .bfloat16));
    const ks = try g.mul(try rmsNormOnes(g, k, ones, 1e-6), try g.scalar(scales.k, .bfloat16));
    return .{ .q = qs, .k = ks, .v = v, .next_tail = next_tail, .window = window };
}

/// The q / k scales (1 / dk and 1 / sqrt(dk)), as f64 for `G.scalar` (both prologue routes take the same bf16 scalars).
pub fn qkScales(c: *const glm.Config) struct { q: f64, k: f64 } {
    const inv_dk = 1.0 / @as(f64, @floatFromInt(c.kda_head_dim));
    return .{ .q = inv_dk, .k = @sqrt(inv_dk) };
}

pub const Recur = struct { y: T, state: T };

/// A recurrence route over one span: q, k, v [1, R, hv, dk] bf16 (q / k normalized and scaled), the gate's pre-activation
/// `a` [1, R, hv * dk], beta [1, R, hv], the state f32 [1, hv, dv, dk] -> y [1, R, hv, dv] bf16 and the state after the
/// span's last row.
pub const RecurFn = fn (g: *G, c: *const glm.Config, w: *const Weights, q: T, k: T, v: T, a: T, beta: T, state: T) anyerror!Recur;

/// The forget gate per key channel, f32, exp(lb * sigmoid(exp(A_log) * (a + dt_bias))): the stock kernels' input.
pub fn gateOf(g: *G, c: *const glm.Config, w: *const Weights, a: T) !T {
    const a32 = try g.astype(try g.add(a, w.dt_bias), .float32);
    return g.exp(try g.mul(try g.sigmoid(try g.mul(w.a_exp, a32)), try g.scalar(c.kda_lower_bound, .float32)));
}

/// The per-token route: the f32 gate (`gateOf`), then the stock kernel (`recur`). Every decode-lane span takes it, and
/// the prompt pass's where the construction declined the per-core route.
pub fn recurStock(g: *G, c: *const glm.Config, w: *const Weights, q: T, k: T, v: T, a: T, beta: T, state: T) anyerror!Recur {
    return recur(g, q, k, v, try gateOf(g, c, w, a), beta, state, g.shapeOf(q).dim(1), @intCast(c.kda_heads), @intCast(c.kda_head_dim));
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

var seq_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The recurrence's prepared launches at 1 .. `plan_rows` rows (the decode lane, the MTP verify): each text's config and
/// the row-count scalar `T` (and the capture's stride), built once by `install` for the model's heads and head dim; a
/// launch at those rows indexes them, a wider one (a prompt span on the per-token route) builds its own.
pub const plan_rows: usize = 8;
const Plan = struct {
    hv: c_int = 0,
    dk: c_int = 0,
    cfg: [plan_rows]mlx.mlx_fast_metal_kernel_config = @splat(.{ .ctx = null }),
    cfg_seq: [plan_rows]mlx.mlx_fast_metal_kernel_config = @splat(.{ .ctx = null }),
    t: [plan_rows]mlx.mlx_array = @splat(.{ .ctx = null }),
    stride: mlx.mlx_array = .{ .ctx = null },
};
var plan: Plan = .{};

/// The recurrence's kernel objects (the stock one and the verify's capture variant) and their launches at the decode
/// lane's row counts, built once at construction for `c`'s geometry (no device work: a library compiles at its first
/// dispatch); a forward only reads them. A second model of another geometry is refused by name (the launches are the
/// process's).
pub fn install(c: *const glm.Config) !void {
    _ = try recurKernel();
    _ = try recurSeqKernel();
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    if (plan.hv != 0) {
        if (plan.hv != hv or plan.dk != dk) return error.Glm5NextKdaPlanGeometry;
        return;
    }
    var p: Plan = .{ .hv = hv, .dk = dk };
    errdefer freePlan(&p);
    for (0..plan_rows) |i| {
        const rows: c_int = @intCast(i + 1);
        p.cfg[i] = try recurConfig(rows, hv, dk);
        p.cfg_seq[i] = try recurSeqConfig(rows, hv, dk);
        p.t[i] = mlx.mlx_array_new_int(rows);
    }
    p.stride = mlx.mlx_array_new_int(hv * dk * dk);
    plan = p;
}

fn freePlan(p: *Plan) void {
    for (p.cfg ++ p.cfg_seq) |x| if (x.ctx != null) {
        _ = mlx.mlx_fast_metal_kernel_config_free(x);
    };
    for (p.t ++ [_]mlx.mlx_array{p.stride}) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    p.* = .{};
}

fn recurConfig(rows: c_int, hv: c_int, dk: c_int) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, hv, dk }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, hv, dk, dk }, 4, .float32));
    try recurCommon(cfg, hv, dk);
    return cfg;
}

fn recurSeqConfig(rows: c_int, hv: c_int, dk: c_int) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, hv, dk }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, 1, hv, dk, dk }, 5, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, hv, dk, dk }, 4, .float32));
    try recurCommon(cfg, hv, dk);
    return cfg;
}

fn recurCommon(cfg: mlx.mlx_fast_metal_kernel_config, hv: c_int, dk: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, dk, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dv", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hk", hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hv", hv));
}

fn recurSeqKernel() !mlx.mlx_fast_metal_kernel {
    if (seq_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" };
    const outs = [_][*:0]const u8{ "y", "state_seq", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new("glm5_next_gated_delta_vec_seq", iv, ov, recur_seq_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    seq_kernel = k;
    return k;
}

const RecurSeq = struct { y: T, state: T, seq: T };

/// `recur` with every row's state but the last written to `seq` f32 [rows, 1, hv, dv, dk] (the host's `capture_seq`).
fn recurSeq(g: *G, q: T, k: T, v: T, gate: T, beta: T, state: T, rows: c_int, hv: c_int, dk: c_int) !RecurSeq {
    const kern = seq_kernel.?;
    // The verify's rows (<= `plan_rows`) on the installed plan (its config, its row-count scalar, the per-row stride into
    // `state_seq`: B * Hv * Dv * Dk, B = 1).
    std.debug.assert(rows >= 1 and rows <= plan_rows and hv == plan.hv and dk == plan.dk);
    const ri: usize = @intCast(rows - 1);
    const cfg = plan.cfg_seq[ri];
    const ins = [_]T{ q, k, v, gate, beta, state, plan.t[ri], plan.stride };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var outs: [3]T = undefined;
    for (&outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&x, ov, i));
        o.* = try g.adopt(x);
    }
    return .{ .y = outs[0], .seq = outs[1], .state = outs[2] };
}

fn recurKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const outs = [_][*:0]const u8{ "y", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new("glm5_next_gated_delta_vec", iv, ov, recur_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The delta rule over `rows` tokens: q, k, v [1, R, hv, dk] bf16, the gate f32 [1, R, hv * dk], beta [1, R, hv], the
/// state f32 [1, hv, dv, dk] -> y [1, R, hv, dv] bf16 and the state after the last row.
fn recur(g: *G, q: T, k: T, v: T, gate: T, beta: T, state: T, rows: c_int, hv: c_int, dk: c_int) !Recur {
    const kern = kernel.?;
    // The decode lane's rows on the installed plan; a wider span (a prompt span on the per-token route) its own launch.
    const planned = rows <= plan_rows;
    std.debug.assert(!planned or (hv == plan.hv and dk == plan.dk));
    const own = if (planned) null else try recurConfig(rows, hv, dk);
    defer if (own) |c| {
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    };
    const cfg = own orelse plan.cfg[@intCast(rows - 1)];
    const t_arr = if (planned) plan.t[@intCast(rows - 1)] else try g.adopt(mlx.mlx_array_new_int(rows));
    const ins = [_]T{ q, k, v, gate, beta, state, t_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    const yt = try g.adopt(y);
    var s_out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&s_out, ov, 1));
    return .{ .y = yt, .state = try g.adopt(s_out) };
}

/// mlx-serve `transformer.zig` `gdnKernelSource(vectorized = true, capture_seq = false)` as it reads at ddalcu main
/// 6e15b046: the per-channel gate `g` [B, T, Hv, Dk], one simdgroup per (head, value row), Dk / 32 state elements per
/// lane in f32, one token per step.
const recur_source: [:0]const u8 =
    \\auto n = thread_position_in_grid.z;
    \\auto b_idx = n / Hv;
    \\auto hv_idx = n % Hv;
    \\auto hk_idx = hv_idx / (Hv / Hk);
    \\constexpr int n_per_t = Dk / 32;
    \\
    \\auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\
    \\auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    \\y += b_idx * T * Hv * Dv + hv_idx * Dv;
    \\
    \\auto dk_idx = thread_position_in_threadgroup.x;
    \\auto dv_idx = thread_position_in_grid.y;
    \\
    \\auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    \\auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    \\
    \\float state[n_per_t];
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  state[i] = static_cast<float>(i_state[s_idx]);
    \\}
    \\
    \\auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
    \\auto beta_ = beta + b_idx * T * Hv;
    \\
    \\for (int t = 0; t < T; ++t) {
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] * g_[s_idx];
    \\    kv_mem += state[i] * k_[s_idx];
    \\  }
    \\  kv_mem = simd_sum(kv_mem);
    \\
    \\  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
    \\
    \\  float out = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] + k_[s_idx] * delta;
    \\    out += state[i] * q_[s_idx];
    \\  }
    \\  out = simd_sum(out);
    \\  if (thread_index_in_simdgroup == 0) {
    \\    y[dv_idx] = static_cast<OutT>(out);
    \\  }
    \\  q_ += Hk * Dk;
    \\  k_ += Hk * Dk;
    \\  v_ += Hv * Dv;
    \\  y += Hv * Dv;
    \\  g_ += Hv * Dk;
    \\  beta_ += Hv;
    \\}
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  o_state[s_idx] = static_cast<StT>(state[i]);
    \\}
;

/// mlx-serve `transformer.zig` `gdnKernelSource(vectorized = true, capture_seq = true)` at ddalcu main 6e15b046: the
/// stock text plus `seq_base` and, after each row but the last, the state written to `state_seq` at `t * seq_stride`.
/// The capture text (kda_verify.zig's tests read its statements against the fused verify text's).
pub const recur_seq_source_text = recur_seq_source;

const recur_seq_source: [:0]const u8 =
    \\auto n = thread_position_in_grid.z;
    \\auto b_idx = n / Hv;
    \\auto hv_idx = n % Hv;
    \\auto hk_idx = hv_idx / (Hv / Hk);
    \\constexpr int n_per_t = Dk / 32;
    \\
    \\auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\
    \\auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    \\y += b_idx * T * Hv * Dv + hv_idx * Dv;
    \\
    \\auto dk_idx = thread_position_in_threadgroup.x;
    \\auto dv_idx = thread_position_in_grid.y;
    \\
    \\auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    \\auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    \\auto seq_base = (n * Dv + dv_idx) * Dk;
    \\
    \\float state[n_per_t];
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  state[i] = static_cast<float>(i_state[s_idx]);
    \\}
    \\
    \\auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
    \\auto beta_ = beta + b_idx * T * Hv;
    \\
    \\for (int t = 0; t < T; ++t) {
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] * g_[s_idx];
    \\    kv_mem += state[i] * k_[s_idx];
    \\  }
    \\  kv_mem = simd_sum(kv_mem);
    \\
    \\  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
    \\
    \\  float out = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] + k_[s_idx] * delta;
    \\    out += state[i] * q_[s_idx];
    \\  }
    \\  out = simd_sum(out);
    \\  if (thread_index_in_simdgroup == 0) {
    \\    y[dv_idx] = static_cast<OutT>(out);
    \\  }
    \\  if (t + 1 < T) {
    \\    auto t_state = state_seq + t * seq_stride + seq_base;
    \\    for (int i = 0; i < n_per_t; ++i) {
    \\      auto s_idx = n_per_t * dk_idx + i;
    \\      t_state[s_idx] = static_cast<StT>(state[i]);
    \\    }
    \\  }
    \\  q_ += Hk * Dk;
    \\  k_ += Hk * Dk;
    \\  v_ += Hv * Dv;
    \\  y += Hv * Dv;
    \\  g_ += Hv * Dk;
    \\  beta_ += Hv;
    \\}
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  o_state[s_idx] = static_cast<StT>(state[i]);
    \\}
;

const testing = std.testing;

test "glm5_next kda: the capture text is the stock text plus the per-row state writes (the host's capture_seq arm)" {
    // Every line of the stock text appears in the capture text, in order; the capture adds `seq_base` and the row write.
    var it = std.mem.splitScalar(u8, recur_source, '\n');
    var at: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const i = std.mem.indexOfPos(u8, recur_seq_source, at, line) orelse return error.TestUnexpectedResult;
        at = i + line.len;
    }
    try testing.expect(std.mem.indexOf(u8, recur_seq_source, "auto seq_base = (n * Dv + dv_idx) * Dk;") != null);
    try testing.expect(std.mem.indexOf(u8, recur_seq_source, "auto t_state = state_seq + t * seq_stride + seq_base;") != null);
    try testing.expect(std.mem.indexOf(u8, recur_seq_source, "if (t + 1 < T) {") != null);
    try testing.expect(std.mem.indexOf(u8, recur_source, "state_seq") == null);
    // The bill's per-row capture: one layer's state and conv tail.
    try testing.expectEqual(@as(u64, 4_194_304 + 294_912), Capture.rowBytes(&.{}));
}

test "glm5_next kda: the recurrence text is the host's stock vector-gated kernel" {
    // The per-channel gate's three lines (gdnKernelSource's `vectorized` arm).
    try testing.expect(std.mem.indexOf(u8, recur_source, "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;") != null);
    try testing.expect(std.mem.indexOf(u8, recur_source, "state[i] = state[i] * g_[s_idx];") != null);
    try testing.expect(std.mem.indexOf(u8, recur_source, "  g_ += Hv * Dk;") != null);
    try testing.expectEqual(@as(u64, 3 * 24_576 * 4 + 64 * 128 * 128 * 4), State.bytes(&.{}));
}

test "glm5_next kda: the recurrence's launches at M = 1 .. 8 are the per-call launches; another model's geometry is refused by name" {
    const c: glm.Config = .{};
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    for (1..plan_rows + 1) |m| {
        const cf = try recurConfig(@intCast(m), hv, dk);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cf);
        try testing.expectEqual([9]c_int{ 32, dk, hv, 32, 4, 1, 2, 6, 0 }, readConfig(cf));
        const cs = try recurSeqConfig(@intCast(m), hv, dk);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cs);
        try testing.expectEqual([9]c_int{ 32, dk, hv, 32, 4, 1, 3, 6, 0 }, readConfig(cs));
    }
    // A plan already installed for these heads: the same model again is a no-op, another one refused (no arrays made:
    // the CPU lane creates no device).
    const saved = plan;
    defer plan = saved;
    plan = .{ .hv = hv, .dk = dk };
    try install(&c);
    var other = c;
    other.kda_heads = 32;
    try testing.expectError(error.Glm5NextKdaPlanGeometry, install(&other));
    try @import("sdk").testing.expectNoDevice();
}

extern "c" fn mlxlc_test_config(config: mlx.mlx_fast_metal_kernel_config, out: *[9]c_int) void;

fn readConfig(cf: mlx.mlx_fast_metal_kernel_config) [9]c_int {
    var got: [9]c_int = undefined;
    mlxlc_test_config(cf, &got);
    return got;
}
