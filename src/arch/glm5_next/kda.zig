//! GLM-5.3-Flash's Kimi Delta Attention (KDA) layers: the q / k / v projections through the short causal conv and SiLU,
//! the per-head L2 norms, the per-channel bounded-sigmoid forget gate from the low-rank `f` projections, beta = sigmoid(b),
//! the delta-rule recurrence over an f32 state, and the gated RMSNorm (sigmoid of the low-rank `g` projections) into the
//! output projection. The per-token route runs mlx-serve's stock gated-delta kernel with the per-channel gate
//! (`transformer.zig` `gdnKernelSource(vectorized = true)`, mlx-lm's `_make_gated_delta_kernel(vectorized=True)`: the exact
//! recurrence, one token at a time), copied here, over the materialized f32 gate. A span wider than the decode lane (the
//! prompt pass's) runs the prompt route installed at construction (kda_percore.zig: the host's per-core kernel where this
//! GPU takes it, else the per-token route); the host's fused decode kernel is not used (it needs quantized gate
//! projections).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const kda_percore = @import("kda_percore.zig");

comptime {
    // kda_percore.zig's tests run with this file's (src/tests.zig lists kda.zig).
    _ = kda_percore;
}

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// One KDA layer's residents (the pack's BF16 projections, `x @ w^T`), with the f32 tables its gate reads.
pub const Weights = struct {
    q: T,
    k: T,
    v: T,
    /// The forget gate's low-rank stages [rank, hidden] and [hv * dk, rank]; the output gate's alike.
    f_a: T,
    f_b: T,
    g_a: T,
    g_b: T,
    /// beta's projection [heads, hidden].
    b: T,
    /// f32 [hv]: A_log as its own F32 table, converted once at bind (the pack stores F32 or BF16 by layer); the per-core
    /// kernel takes exp(A_log) of it itself.
    a_log: T,
    /// f32 [hv * dk]: exp(A_log) of each head repeated over its key channels (the gate's constant factor, built once).
    a_exp: T,
    /// f32 [hv * dk].
    dt_bias: T,
    /// f32 [conv dim, taps, 1]: the depthwise conv in MLX's layout.
    conv_w: T,
    /// [dv]: the output RMSNorm's weight, per head.
    o_norm: T,
    o: T,
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

    fn set(st: *State, g: *G, conv: T, rec: T) void {
        st.reset(g);
        st.conv = g.keep(conv);
        st.rec = g.keep(rec);
    }

    /// The bytes one state holds (the bill's).
    pub fn bytes(c: *const glm.Config) u64 {
        return @as(u64, c.kda_conv - 1) * c.kdaConvDim() * 4 + @as(u64, c.kda_heads) * c.kda_head_dim * c.kda_head_dim * 4;
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
    const sh = g.shapeOf(x);
    const rows = sh.dim(1);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim: c_int = hv * dk;
    const cdim: c_int = 3 * dim;
    const taps: c_int = @intCast(c.kda_conv);
    // The short conv over [q | k | v], f32, its window the state's tail then the rows.
    const qkv = try g.astype(try g.concat(&.{ try linear(g, x, w.q), try linear(g, x, w.k), try linear(g, x, w.v) }, 2), .float32);
    const tail = st.conv orelse try g.zeros(&.{ 1, taps - 1, cdim }, .float32);
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
    // q = rms_norm(q) / dk and k = rms_norm(k) / sqrt(dk): the L2 norms with the reference's scales.
    const ones = try g.ones(&.{dk}, .bfloat16);
    const inv_dk = 1.0 / @as(f64, @floatFromInt(dk));
    const qs = try g.mul(try rmsNormOnes(g, q, ones, 1e-6), try g.scalar(inv_dk, .bfloat16));
    const ks = try g.mul(try rmsNormOnes(g, k, ones, 1e-6), try g.scalar(@sqrt(inv_dk), .bfloat16));
    // The forget gate's pre-activation per key channel (each route makes the gate of it), beta, the state.
    const a = try linear(g, try linear(g, x, w.f_a), w.f_b);
    const beta = try g.sigmoid(try linear(g, x, w.b));
    const state = st.rec orelse try g.zeros(&.{ 1, hv, dk, dk }, .float32);
    // The span's route: past the decode lane the prompt route installed at construction, else the per-token route.
    const route = kda_percore.spanRoute(rows);
    const r = try route(g, c, w, qs, ks, v, a, beta, state);
    // The gated RMSNorm: rms_norm(y) per head (o_norm) times sigmoid(g_b(g_a(x))).
    const z = try g.reshape(try linear(g, try linear(g, x, w.g_a), w.g_b), &head4);
    var yn = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&yn, r.y, w.o_norm, c.rms_norm_eps, g.s));
    const gated = try g.mul(try g.adopt(yn), try g.sigmoid(z));
    st.set(g, next_tail, r.state);
    return linear(g, try g.reshape(gated, &.{ 1, rows, dim }), w.o);
}

pub const Recur = struct { y: T, state: T };

/// A recurrence route over one span: q, k, v [1, R, hv, dk] bf16 (q / k normalized and scaled), the gate's pre-activation
/// `a` [1, R, hv * dk], beta [1, R, hv], the state f32 [1, hv, dv, dk] -> y [1, R, hv, dv] bf16 and the state after the
/// span's last row.
pub const RecurFn = fn (g: *G, c: *const glm.Config, w: *const Weights, q: T, k: T, v: T, a: T, beta: T, state: T) anyerror!Recur;

/// The per-token route: the forget gate per key channel, f32, exp(lb * sigmoid(exp(A_log) * (a + dt_bias))), then the
/// stock kernel (`recur`). Every decode-lane span takes it, and the prompt pass's where the construction declined the
/// per-core route.
pub fn recurStock(g: *G, c: *const glm.Config, w: *const Weights, q: T, k: T, v: T, a: T, beta: T, state: T) anyerror!Recur {
    const a32 = try g.astype(try g.add(a, w.dt_bias), .float32);
    const gate = try g.exp(try g.mul(try g.sigmoid(try g.mul(w.a_exp, a32)), try g.scalar(c.kda_lower_bound, .float32)));
    return recur(g, q, k, v, gate, beta, state, g.shapeOf(q).dim(1), @intCast(c.kda_heads), @intCast(c.kda_head_dim));
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The recurrence's kernel object, built once at construction (no device work: its library compiles at the first
/// dispatch); a forward only reads it.
pub fn install() !void {
    _ = try recurKernel();
}

fn recurKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const outs = [_][*:0]const u8{ "y", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("glm5_next_gated_delta_vec", iv, ov, recur_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The delta rule over `rows` tokens: q, k, v [1, R, hv, dk] bf16, the gate f32 [1, R, hv * dk], beta [1, R, hv], the
/// state f32 [1, hv, dv, dk] -> y [1, R, hv, dv] bf16 and the state after the last row.
fn recur(g: *G, q: T, k: T, v: T, gate: T, beta: T, state: T, rows: c_int, hv: c_int, dk: c_int) !Recur {
    const kern = try recurKernel();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, hv, dk }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, hv, dk, dk }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, dk, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dv", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hk", hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hv", hv));
    const t_arr = try g.adopt(mlx.mlx_array_new_int(rows));
    const ins = [_]T{ q, k, v, gate, beta, state, t_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
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

const testing = std.testing;

test "glm5_next kda: the recurrence text is the host's stock vector-gated kernel" {
    // The per-channel gate's three lines (gdnKernelSource's `vectorized` arm).
    try testing.expect(std.mem.indexOf(u8, recur_source, "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;") != null);
    try testing.expect(std.mem.indexOf(u8, recur_source, "state[i] = state[i] * g_[s_idx];") != null);
    try testing.expect(std.mem.indexOf(u8, recur_source, "  g_ += Hv * Dk;") != null);
    try testing.expectEqual(@as(u64, 3 * 24_576 * 4 + 64 * 128 * 128 * 4), State.bytes(&.{}));
}
