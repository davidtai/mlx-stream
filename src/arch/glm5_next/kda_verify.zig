//! GLM-5.3-Flash's KDA layer of the MTP verify in one text (`kda_verify: fused`, the default; `chain`, by name, is the op chain:
//! kda.zig `forwardCapture`, the control). The text (kernels/glm5n_kda_verify.metal, its element functions in
//! kernels/header_glm5n_kda_verify.metal) runs, per KDA layer of every verify (R = 2 .. 4 rows), the prologue (the prompt
//! prologue's exact text: conv, SiLU, bf16, the q / k norms and scales), the forget gate (`gateOf`), beta (bf16 sigmoid),
//! the conv window and next tail the capture keeps, the stock capture recurrence (`recur_seq_source`, per value row, the
//! state after every row but the last) and the gated RMSNorm (rms_single_row with o_norm, times the bf16 sigmoid of the
//! output gate): one dispatch where the op chain dispatches 21 to 29 (the audit's count to MLX's primitives: the three
//! concatenations' per-input copies, two casts, the conv, SiLU, the tail multiply, the `ones` fill, the q / k norms with
//! their contiguous copies and scale multiplies, the v copy, six gate ops, beta, the recurrence, the norm, the output
//! sigmoid and the multiply). The projections stay outside the text: q, k, v, f_a, b, g_a and o on the bound
//! format's GEMV (fp8.zig `project`), f_b and g_b on MLX's matmul, so their accumulation order is the op chain's.
//!
//! Exact against the op chain: every stage is the MLX kernel (or the stock text) of the op it replaces, per element in the
//! same types and operand order, and every intermediate the op chain materializes in bf16 is rounded to bf16 at the same
//! point (the header names each). The rounding boundary the text stops at: the low-rank second stages f_b / g_b (MLX
//! steel matmul, kda.zig `linear`) and the projections. Fusing those (as kda_decode.zig's text does) changes their
//! summation order: rounding-class, kept out.
//!
//! Installed once at construction (`install`, by `settings.kda_verify`, with the MTP lane only): the geometry, the GPU
//! stream and the bound dtypes refused by name, the launches per verify width prepared, one launch over the first KDA
//! layer as the compile check. The verify's call (`run`) reads only the installed launches: no eligibility branch.
//! No bill term: the text's outputs are the op chain's capture, state and gated rows; its intermediates live in
//! threadgroup memory (bill.zig `kdaRowBytes` stays the bound).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const glm = @import("config.zig");
const kda = @import("kda.zig");
const fp8 = @import("fp8.zig");
const ops_be = @import("../deepseek_v41/ops.zig");

const G = kda.G;
const T = kda.T;

/// The widest verify the text takes: the MTP lane's verify rows (depth + 1, <= 4). The text's simdgroup map needs
/// 4R + 1 <= 32 (3R prologue groups, R gate groups, one beta group).
pub const max_rows: u32 = 4;
pub const threads: c_int = 1024;
const head_dim: u32 = 128;

comptime {
    if (@import("mtp.zig").verify_rows > max_rows) @compileError("glm5_next kda_verify: the launches do not cover the MTP verify");
    if (4 * max_rows + 1 > 32) @compileError("glm5_next kda_verify: the text's simdgroup map holds 7 rows at most");
}

pub const texts = struct {
    pub const header = @embedFile("kernels/header_glm5n_kda_verify.metal");
    pub const verify = @embedFile("kernels/glm5n_kda_verify.metal");
};

pub const input_names = [_][*:0]const u8{ "qp", "kp", "vp", "tail", "conv_w", "fa", "dt_bias", "a_exp", "bp", "z", "norm_w", "state_in", "cf", "sc" };
pub const output_names = [_][*:0]const u8{ "y_out", "state_seq", "state_out", "window", "next_tail" };

/// The model the text serves (head width 128: 32 lanes x 4; one threadgroup per head), refused by name.
pub fn requireGeometry(c: *const glm.Config) error{Glm5NextKdaVerifyGeometry}!void {
    if (c.kda_head_dim != head_dim or c.kda_conv < 2 or c.kda_heads == 0 or c.nKda() == 0) return error.Glm5NextKdaVerifyGeometry;
}

/// The dtypes the text reads as the op chain's ops receive them: the bound tables (o_norm, f_b, g_b BF16 as stored; the
/// conv, dt_bias and exp(A_log) the bind's F32 copies), refused by name. The projections' outputs are checked on the
/// compile check's launch (`compileCheck`).
pub fn requireDtypes(w: *const kda.Weights) error{Glm5NextKdaVerifyDtype}!void {
    const bf = [_]mlx.mlx_array{ w.o_norm, w.f_b, w.g_b };
    for (bf) |x| if (mlx.mlx_array_dtype(x) != .bfloat16) return error.Glm5NextKdaVerifyDtype;
    const f = [_]mlx.mlx_array{ w.conv_w, w.dt_bias, w.a_exp };
    for (f) |x| if (mlx.mlx_array_dtype(x) != .float32) return error.Glm5NextKdaVerifyDtype;
}

const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

pub const Kernels = struct {
    verify: Kernel = .{ .ctx = null },
    /// Index rows - 1: the launch of a verify of that many rows.
    cfg: [max_rows]Config = @splat(.{ .ctx = null }),
    /// f32 [8]: the gate's lower bound, the output norm's eps, zeros (eight words: MLX passes a smaller input in the
    /// constant address space).
    cf: T = .{ .ctx = null },
    /// bf16 [8]: the q and k scales as `G.scalar` makes them (kda.qkScales), zeros.
    sc: T = .{ .ctx = null },

    pub fn deinit(k: *Kernels) void {
        if (k.verify.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(k.verify);
        for (k.cfg) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        if (k.cf.ctx != null) _ = mlx.mlx_array_free(k.cf);
        if (k.sc.ctx != null) _ = mlx.mlx_array_free(k.sc);
        k.* = .{};
    }
};

/// One launch's geometry: one threadgroup of `threads` per head.
pub fn grid(c: *const glm.Config) [3]c_int {
    return .{ threads * @as(c_int, @intCast(c.kda_heads)), 1, 1 };
}

/// The constant words: f32 {lower bound, norm eps, 0 ...} and the bf16 q / k scales' bits {q, k, 0 ...}.
pub fn constWords(c: *const glm.Config) struct { cf: [8]f32, sc: [8]u16 } {
    const s = kda.qkScales(c);
    return .{
        .cf = .{ c.kda_lower_bound, c.rms_norm_eps, 0, 0, 0, 0, 0, 0 },
        .sc = .{ ops_be.bf16Bits(@floatCast(s.q)), ops_be.bf16Bits(@floatCast(s.k)), 0, 0, 0, 0, 0, 0 },
    };
}

fn config(c: *const glm.Config, rows: c_int) !Config {
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const dim: c_int = hv * dk;
    const cdim: c_int = 3 * dim;
    const taps: c_int = @intCast(c.kda_conv);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, dim }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, 1, hv, dk, dk }, 5, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, hv, dk, dk }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, taps - 1 + rows, cdim }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, taps - 1, cdim }, 3, .float32));
    const gr = grid(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, gr[0], gr[1], gr[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    inline for (.{ .{ "R", rows }, .{ "DK", dk }, .{ "HV", hv }, .{ "DIM", dim }, .{ "CDIM", cdim }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    return cfg;
}

/// The route, once, at construction (the MTP lane installed, `kda_verify: fused`): the geometry, the GPU stream and
/// every KDA layer's bound dtypes refused by name; the kernel object and its launches at 1 .. `max_rows` rows; one launch
/// at the widest verify over zeros and the first KDA layer's tables, evaluated (Metal compiles the text at its first
/// dispatch and refuses a threadgroup its registers cannot hold).
pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, layers: []const *const kda.Weights, diag: *sdk.Diag) !Kernels {
    requireGeometry(c) catch |e| {
        diag.set("glm5_next kda_verify=fused: the model is not the text's (KDA head {d}, conv {d}; the text takes head 128)", .{ c.kda_head_dim, c.kda_conv });
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next kda_verify=fused: the module's stream is not a GPU stream", .{});
        return error.Glm5NextKdaVerifyNotGpu;
    }
    for (layers, 0..) |w, i| requireDtypes(w) catch |e| {
        diag.set("glm5_next kda_verify=fused: KDA layer {d}'s tables are not the text's (o_norm {t}, f_b {t}, g_b {t} must be bfloat16; conv {t}, dt_bias {t}, a_exp {t} float32)", .{ i, mlx.mlx_array_dtype(w.o_norm), mlx.mlx_array_dtype(w.f_b), mlx.mlx_array_dtype(w.g_b), mlx.mlx_array_dtype(w.conv_w), mlx.mlx_array_dtype(w.dt_bias), mlx.mlx_array_dtype(w.a_exp) });
        return e;
    };
    var k: Kernels = .{};
    errdefer k.deinit();
    {
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        k.verify = mtl.mlx_fast_metal_kernel_new("glm5n_kda_verify", in_vec, out_vec, texts.verify, texts.header, true, false);
        if (k.verify.ctx == null) return error.Glm5NextKdaVerifyKernel;
    }
    for (&k.cfg, 1..) |*cfg, m| cfg.* = try config(c, @intCast(m));
    const cw = constWords(c);
    k.cf = mlx.mlx_array_new_data(&cw.cf, &[_]c_int{8}, 1, .float32);
    if (k.cf.ctx == null) return error.MlxError;
    k.sc = mlx.mlx_array_new_data(&cw.sc, &[_]c_int{8}, 1, .bfloat16);
    if (k.sc.ctx == null) return error.MlxError;
    compileCheck(a, s, c, &k, layers[0]) catch |err| {
        var eb: [400]u8 = undefined;
        diag.set("glm5_next kda_verify=fused: the compile check failed ({s}): {s}", .{ @errorName(err), mlx.takeError(&eb) orelse "" });
        return err;
    };
    return k;
}

/// One launch at the widest verify over zeros and `w`'s tables, evaluated; the projections' dtypes (the text reads q, k,
/// v, the gate's pre-activation, b and z as bf16) and the output's shape checked.
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const glm.Config, k: *const Kernels, w: *const kda.Weights) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    const h: c_int = @intCast(c.hidden_size);
    const rows: c_int = @intCast(max_rows);
    const x = try g.zeros(&.{ 1, rows, h }, .bfloat16);
    const p = try projections(&g, w, x);
    for ([_]T{ p.q, p.k, p.v, p.a, p.b, p.z }) |t| if (mlx.mlx_array_dtype(t) != .bfloat16) return error.Glm5NextKdaVerifyDtype;
    var st: kda.State = .{};
    defer st.reset(&g);
    var cap: kda.Capture = .{};
    defer cap.release(&g);
    const y = try step(&g, k, c, w, x, &st, &cap);
    try g.evalAll(&.{ y, st.conv.?, st.rec.?, cap.seq.?, cap.window.? });
    if (!std.mem.eql(c_int, g.shapeOf(y).slice(), &[_]c_int{ 1, rows, h })) return error.Glm5NextKdaVerifyShape;
}

const Projections = struct { q: T, k: T, v: T, a: T, b: T, z: T };

/// The layer's projections as the op chain makes them (kda.zig `forwardWith`): q, k, v, b on the bound format's GEMV, the
/// gates' low-rank stages f_b(f_a(x)) and g_b(g_a(x)) with the second stage on MLX's matmul (`linear`).
fn projections(g: *G, w: *const kda.Weights, x: T) !Projections {
    return .{
        .q = try fp8.project(g, w.q, x),
        .k = try fp8.project(g, w.k, x),
        .v = try fp8.project(g, w.v, x),
        .a = try g.matmul(try fp8.project(g, w.f_a, x), try g.transpose(w.f_b)),
        .b = try fp8.project(g, w.b, x),
        .z = try g.matmul(try fp8.project(g, w.g_a, x), try g.transpose(w.g_b)),
    };
}

/// `kda.VerifyStep`'s fused route: `ctx` the module's installed `Kernels`.
pub fn run(ctx: ?*const anyopaque, g: *G, c: *const glm.Config, w: *const kda.Weights, x: T, st: *kda.State, cap: *kda.Capture) anyerror!T {
    const k: *const Kernels = @ptrCast(@alignCast(ctx.?));
    return step(g, k, c, w, x, st, cap);
}

/// The verify's KDA layer: x [1, R, hidden] (R <= `max_rows`) -> [1, R, hidden]; `st` advanced past the rows, `cap` the
/// capture (`kda.forwardCapture`'s contract: the state after every row but the last, the conv window).
pub fn step(g: *G, k: *const Kernels, c: *const glm.Config, w: *const kda.Weights, x: T, st: *kda.State, cap: *kda.Capture) !T {
    const rows: u32 = @intCast(g.shapeOf(x).dim(1));
    std.debug.assert(rows >= 1 and rows <= max_rows);
    const hv: c_int = @intCast(c.kda_heads);
    const dk: c_int = @intCast(c.kda_head_dim);
    const taps: c_int = @intCast(c.kda_conv);
    const p = try projections(g, w, x);
    const tail = st.conv orelse try g.zeros(&.{ 1, taps - 1, 3 * hv * dk }, .float32);
    const state = st.rec orelse try g.zeros(&.{ 1, hv, dk, dk }, .float32);
    const ins = [_]T{ p.q, p.k, p.v, tail, w.conv_w, p.a, w.dt_bias, w.a_exp, p.b, p.z, w.o_norm, state, k.cf, k.sc };
    const in = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k.verify, in, k.cfg[rows - 1], g.s));
    var outs: [5]T = undefined;
    for (&outs, 0..) |*o, i| {
        var y = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&y, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(y);
            return e;
        };
        o.* = try g.adopt(y);
    }
    cap.release(g);
    cap.* = .{ .seq = g.keep(outs[1]), .window = g.keep(outs[3]), .rows = rows };
    st.set(g, outs[4], outs[2]);
    return fp8.project(g, w.o, outs[0]);
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "glm5_next kda_verify CPU: Flash's geometry is the text's; one 1,024-thread threadgroup per head; another refused by name; the constants are the op chain's scalars" {
    const c: glm.Config = .{};
    try requireGeometry(&c);
    try testing.expectEqual([3]c_int{ 64 * 1024, 1, 1 }, grid(&c));
    for ([_]glm.Config{ .{ .kda_head_dim = 64 }, .{ .kda_conv = 1 }, .{ .kda_heads = 0 } }) |bad| try testing.expectError(error.Glm5NextKdaVerifyGeometry, requireGeometry(&bad));
    // The value rows: 32 simdgroups x 4 rows cover a head's 128 once; the lanes x 4 channels cover the keys once.
    try testing.expectEqual(@as(u32, 32 * 4), head_dim);
    const cw = constWords(&c);
    try testing.expectEqual(c.kda_lower_bound, cw.cf[0]);
    try testing.expectEqual(c.rms_norm_eps, cw.cf[1]);
    // The scales' bits are `G.scalar(.., .bfloat16)`'s (ops.zig `scalar`): bf16(1 / 128) and bf16(1 / sqrt(128)).
    const s = kda.qkScales(&c);
    try testing.expectEqual(ops_be.bf16Bits(@floatCast(s.q)), cw.sc[0]);
    try testing.expectEqual(ops_be.bf16Bits(@floatCast(s.k)), cw.sc[1]);
    try testing.expectEqual(@as(u16, 0x3C00), cw.sc[0]);
    // The text reads every input by the name the launch binds, writes only its outputs (no input write: MLX's concurrent
    // encoder orders nothing after a kernel writing through an input) and keeps the stock recurrence's statements.
    for (input_names) |n| try testing.expect(std.mem.indexOf(u8, texts.verify, std.mem.span(n)) != null);
    for (output_names) |n| try testing.expect(std.mem.indexOf(u8, texts.verify, std.mem.span(n)) != null);
    for ([_][]const u8{ "qp[", "kp[", "vp[", "tail[", "conv_w[", "fa[", "dt_bias[", "a_exp[", "bp[", "z[", "norm_w[", "state_in[", "cf[", "sc[" }) |n| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, texts.verify, at, n)) |i| : (at = i + n.len) {
            // The name itself, not the tail of another (next_tail[ holds tail[).
            if (i > 0 and (std.ascii.isAlphanumeric(texts.verify[i - 1]) or texts.verify[i - 1] == '_')) continue;
            const rest = std.mem.trimStart(u8, texts.verify[std.mem.indexOfScalarPos(u8, texts.verify, i, ']').? + 1 ..], " ");
            try testing.expect(!(rest.len > 1 and rest[0] == '=' and rest[1] != '='));
        }
    }
    for ([_][]const u8{
        "state[i] = state[i] * gs[t][s_idx];",
        "kv_mem += state[i] * ks[t][s_idx];",
        "auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];",
        "state[i] = state[i] + ks[t][s_idx] * delta;",
        "out += state[i] * qs[t][s_idx];",
        "if (t + 1 < R) {",
    }) |line| try testing.expect(std.mem.indexOf(u8, texts.verify, line) != null);
}

test "glm5_next kda_verify CPU: the stock capture text's statements, in order, are the fused text's recurrence (renamed reads only)" {
    // kda.zig's capture text with its device reads renamed to the fused text's threadgroup rows: q_ / k_ / g_ -> qs / ks /
    // gs [t], v_[dv_idx] -> vs[t][dv_idx], beta_[hv_idx] -> betas[t], y -> ys[t]; the pointer walks dropped.
    const stock = kda.recur_seq_source_text;
    const pairs = [_][2][]const u8{
        .{ "state[i] = state[i] * g_[s_idx];", "state[i] = state[i] * gs[t][s_idx];" },
        .{ "kv_mem += state[i] * k_[s_idx];", "kv_mem += state[i] * ks[t][s_idx];" },
        .{ "kv_mem = simd_sum(kv_mem);", "kv_mem = simd_sum(kv_mem);" },
        .{ "auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];", "auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];" },
        .{ "state[i] = state[i] + k_[s_idx] * delta;", "state[i] = state[i] + ks[t][s_idx] * delta;" },
        .{ "out += state[i] * q_[s_idx];", "out += state[i] * qs[t][s_idx];" },
        .{ "out = simd_sum(out);", "out = simd_sum(out);" },
        .{ "y[dv_idx] = static_cast<OutT>(out);", "ys[t][dv_idx] = static_cast<bfloat16_t>(out);" },
        .{ "if (t + 1 < R) {", "if (t + 1 < R) {" },
        .{ "t_state[s_idx] = static_cast<StT>(state[i]);", "t_state[s_idx] = static_cast<float>(state[i]);" },
        .{ "o_state[s_idx] = static_cast<StT>(state[i]);", "o_state[s_idx] = static_cast<float>(state[i]);" },
    };
    var at_s: usize = 0;
    var at_f: usize = 0;
    for (pairs) |p| {
        const want = if (std.mem.eql(u8, p[0], "if (t + 1 < R) {")) "if (t + 1 < T) {" else p[0];
        at_s = (std.mem.indexOfPos(u8, stock, at_s, want) orelse return error.TestUnexpectedResult) + want.len;
        at_f = (std.mem.indexOfPos(u8, texts.verify, at_f, p[1]) orelse return error.TestUnexpectedResult) + p[1].len;
    }
}

// ── the CPU reference: the op chain's arithmetic against the fused text's, bit for bit ────────────────────────

/// The element model both references share: each MLX kernel's per-element arithmetic on the CPU (bf16 round to nearest
/// even after each bf16 op, f32 ops in f32, `exp` and `rsqrt` as the host's f32 functions, one fixed simd_sum order).
/// The reference proves the fused text's data flow (which value each thread reads, where each bf16 rounding happens,
/// the window, tail, capture and state layouts, the stage order) is the op chain's at Flash's shapes; the device's own
/// element functions are the window parity's (glm5next-kda-fused-parity).
const Ref = struct {
    fn bf(x: f32) f32 {
        return @bitCast(@as(u32, ops_be.bf16Bits(x)) << 16);
    }
    fn sigmoidF(x: f32) f32 {
        const y = 1.0 / (1.0 + @exp(@abs(x)));
        return if (x < 0) y else 1.0 - y;
    }
    /// unary_ops.h Sigmoid on bfloat16_t: every op's result rounded to bf16.
    fn sigmoidB(x: f32) f32 {
        const e = bf(@exp(bf(@abs(x))));
        const y = bf(1.0 / bf(1.0 + e));
        return if (x < 0) y else bf(1.0 - y);
    }
    fn simdSum(v: *const [32]f32) f32 {
        var t = v.*;
        var n: usize = 16;
        while (n > 0) : (n /= 2) for (0..n) |i| {
            t[i] = t[i] + t[i + n];
        };
        return t[0];
    }
    fn rsqrt(x: f32) f32 {
        return 1.0 / @sqrt(x);
    }
    /// rms_single_row at axis 128 over one row's 128 values (each already bf16): 32 lanes x 4, the two simd_sums,
    /// w * bf16(x * inv) in bf16.
    fn rmsRow(x: []const f32, w: []const f32, eps: f32, out: []f32) void {
        var lanes: [32]f32 = undefined;
        for (0..32) |l| {
            var acc: f32 = 0;
            for (0..4) |j| acc += x[l * 4 + j] * x[l * 4 + j];
            lanes[l] = acc;
        }
        var acc = simdSum(&lanes);
        var second: [32]f32 = @splat(0);
        second[0] = acc;
        acc = simdSum(&second);
        const inv = rsqrt(acc / 128.0 + eps);
        for (0..128) |i| out[i] = bf(w[i] * bf(x[i] * inv));
    }
    fn gate(a: f32, dt: f32, ae: f32, lower: f32) f32 {
        const a32 = a + dt;
        const m = ae * a32;
        const s = sigmoidF(m);
        return @exp(s * lower);
    }
};

const RefIn = struct {
    rows: usize,
    hv: usize,
    dk: usize,
    taps: usize,
    qp: []f32,
    kp: []f32,
    vp: []f32,
    tail: []f32,
    conv_w: []f32,
    fa: []f32,
    dt_bias: []f32,
    a_exp: []f32,
    bp: []f32,
    z: []f32,
    norm_w: []f32,
    state: []f32,
    lower: f32,
    eps: f32,
    q_scale: f32,
    k_scale: f32,
};

const RefOut = struct {
    y: []f32,
    seq: []f32,
    state: []f32,
    window: []f32,
    next_tail: []f32,

    fn init(a: std.mem.Allocator, in: *const RefIn) !RefOut {
        const dim = in.hv * in.dk;
        return .{
            .y = try a.alloc(f32, in.rows * dim),
            .seq = try a.alloc(f32, in.rows * in.hv * in.dk * in.dk),
            .state = try a.alloc(f32, in.hv * in.dk * in.dk),
            .window = try a.alloc(f32, (in.taps - 1 + in.rows) * 3 * dim),
            .next_tail = try a.alloc(f32, (in.taps - 1) * 3 * dim),
        };
    }
    fn deinit(o: *RefOut, a: std.mem.Allocator) void {
        for ([_][]f32{ o.y, o.seq, o.state, o.window, o.next_tail }) |x| a.free(x);
    }
};

/// The op chain (kda.zig `forwardCapture` up to the o projection), op by op, each array materialized as MLX's kernel
/// for that op writes it.
fn refOpChain(a: std.mem.Allocator, in: *const RefIn, out: *RefOut) !void {
    const R = in.rows;
    const hv = in.hv;
    const dk = in.dk;
    const dim = hv * dk;
    const cdim = 3 * dim;
    const taps = in.taps;
    // qkv = astype(concat(q, k, v), f32); window = concat(tail, qkv) along rows.
    const nrow = taps - 1 + R;
    for (0..nrow) |r| for (0..cdim) |ch| {
        out.window[r * cdim + ch] = if (r < taps - 1) in.tail[r * cdim + ch] else blk: {
            const t = r - (taps - 1);
            const sec = ch / dim;
            const src = if (sec == 0) in.qp else if (sec == 1) in.kp else in.vp;
            break :blk src[t * dim + ch - sec * dim];
        };
    };
    // conv1d (depthwise_conv_1d: acc from 0, taps in order), SiLU (x * sigmoid(x)), astype bf16.
    const conv = try a.alloc(f32, R * cdim);
    defer a.free(conv);
    for (0..R) |t| for (0..cdim) |ch| {
        var acc: f32 = 0;
        for (0..taps) |i| acc += out.window[(t + i) * cdim + ch] * in.conv_w[ch * taps + i];
        conv[t * cdim + ch] = Ref.bf(acc * Ref.sigmoidF(acc));
    };
    // next tail = window[R .. R + taps - 1] * 1.
    for (0..taps - 1) |m| for (0..cdim) |ch| {
        out.next_tail[m * cdim + ch] = out.window[(R + m) * cdim + ch] * 1.0;
    };
    // q / k = rms_norm(slice, ones, 1e-6) * bf16 scale; v = the slice (copied by the recurrence's contiguity).
    const q = try a.alloc(f32, R * dim);
    defer a.free(q);
    const k = try a.alloc(f32, R * dim);
    defer a.free(k);
    const v = try a.alloc(f32, R * dim);
    defer a.free(v);
    const ones: [128]f32 = @splat(1.0);
    var row: [128]f32 = undefined;
    for (0..R) |t| for (0..hv) |h| {
        Ref.rmsRow(conv[t * cdim + h * dk ..][0..dk], &ones, 1e-6, &row);
        for (0..dk) |i| q[t * dim + h * dk + i] = Ref.bf(row[i] * in.q_scale);
        Ref.rmsRow(conv[t * cdim + dim + h * dk ..][0..dk], &ones, 1e-6, &row);
        for (0..dk) |i| k[t * dim + h * dk + i] = Ref.bf(row[i] * in.k_scale);
        for (0..dk) |i| v[t * dim + h * dk + i] = conv[t * cdim + 2 * dim + h * dk + i];
    };
    // gate = exp(sigmoid(a_exp * (astype(a) + dt_bias)) * lower); beta = sigmoid(b) in bf16.
    const gt = try a.alloc(f32, R * dim);
    defer a.free(gt);
    for (0..R) |t| for (0..dim) |ch| {
        gt[t * dim + ch] = Ref.gate(in.fa[t * dim + ch], in.dt_bias[ch], in.a_exp[ch], in.lower);
    };
    const beta = try a.alloc(f32, R * hv);
    defer a.free(beta);
    for (beta, in.bp) |*b, x| b.* = Ref.sigmoidB(x);
    // The stock capture recurrence (recur_seq_source): per (head, value row), lanes x 4 keys, row by row.
    const y = try a.alloc(f32, R * dim);
    defer a.free(y);
    for (0..hv) |n| for (0..dk) |dv| {
        var st: [32][4]f32 = undefined;
        for (0..32) |l| for (0..4) |i| {
            st[l][i] = in.state[(n * dk + dv) * dk + 4 * l + i];
        };
        for (0..R) |t| {
            var lanes: [32]f32 = undefined;
            for (0..32) |l| {
                var kv: f32 = 0;
                for (0..4) |i| {
                    st[l][i] = st[l][i] * gt[t * dim + n * dk + 4 * l + i];
                    kv += st[l][i] * k[t * dim + n * dk + 4 * l + i];
                }
                lanes[l] = kv;
            }
            const kv_mem = Ref.simdSum(&lanes);
            const delta = (v[t * dim + n * dk + dv] - kv_mem) * beta[t * hv + n];
            for (0..32) |l| {
                var o: f32 = 0;
                for (0..4) |i| {
                    st[l][i] = st[l][i] + k[t * dim + n * dk + 4 * l + i] * delta;
                    o += st[l][i] * q[t * dim + n * dk + 4 * l + i];
                }
                lanes[l] = o;
            }
            y[t * dim + n * dk + dv] = Ref.bf(Ref.simdSum(&lanes));
            if (t + 1 < R) for (0..32) |l| for (0..4) |i| {
                out.seq[t * hv * dk * dk + (n * dk + dv) * dk + 4 * l + i] = st[l][i];
            };
        }
        for (0..32) |l| for (0..4) |i| {
            out.state[(n * dk + dv) * dk + 4 * l + i] = st[l][i];
        };
    };
    // The gated RMSNorm: rms_norm(y, o_norm, eps) * sigmoid(z) in bf16.
    for (0..R) |t| for (0..hv) |h| {
        Ref.rmsRow(y[t * dim + h * dk ..][0..dk], in.norm_w, in.eps, &row);
        for (0..dk) |i| out.y[t * dim + h * dk + i] = Ref.bf(row[i] * Ref.sigmoidB(in.z[t * dim + h * dk + i]));
    };
}

/// The fused text (kernels/glm5n_kda_verify.metal), thread by thread: per head h the threadgroup's stages in its order,
/// each thread's indices as the text computes them.
fn refFused(in: *const RefIn, out: *RefOut) void {
    const R = in.rows;
    const HV = in.hv;
    const DK = in.dk;
    const DIM = HV * DK;
    const CDIM = 3 * DIM;
    const TAPS = in.taps;
    const NROW = TAPS - 1 + R;
    for (0..HV) |h| {
        var qs: [max_rows][128]f32 = undefined;
        var ks: [max_rows][128]f32 = undefined;
        var vs: [max_rows][128]f32 = undefined;
        var gs: [max_rows][128]f32 = undefined;
        var betas: [max_rows]f32 = undefined;
        var ys: [max_rows][128]f32 = undefined;
        // 1. prologue: simdgroups sg < 3R (sec = sg / R, t = sg % R), lanes x 4 channels.
        for (0..3 * R) |sg| {
            const sec = sg / R;
            const t = sg % R;
            var o: [32][4]f32 = undefined;
            for (0..32) |lane| {
                const cl = h * DK + lane * 4;
                const c = sec * DIM + cl;
                const p = if (sec == 0) in.qp else if (sec == 1) in.kp else in.vp;
                for (0..4) |j| {
                    var acc: f32 = 0;
                    for (0..TAPS) |i| {
                        const wr = t + i;
                        const xin = if (wr < TAPS - 1) in.tail[wr * CDIM + c + j] else p[(wr - (TAPS - 1)) * DIM + cl + j];
                        acc += xin * in.conv_w[(c + j) * TAPS + i];
                    }
                    const y = 1.0 / (1.0 + @exp(@abs(acc)));
                    const s = if (acc < 0) y else 1.0 - y;
                    o[lane][j] = Ref.bf(acc * s);
                }
            }
            if (sec == 2) {
                for (0..32) |lane| for (0..4) |j| {
                    vs[t][lane * 4 + j] = o[lane][j];
                };
            } else {
                var lanes: [32]f32 = undefined;
                for (0..32) |lane| {
                    var acc: f32 = 0;
                    for (0..4) |j| acc += o[lane][j] * o[lane][j];
                    lanes[lane] = acc;
                }
                var acc = Ref.simdSum(&lanes);
                var second: [32]f32 = @splat(0);
                second[0] = acc;
                acc = Ref.simdSum(&second);
                const inv = Ref.rsqrt(acc / @as(f32, @floatFromInt(DK)) + 1e-6);
                const scale = if (sec == 0) in.q_scale else in.k_scale;
                for (0..32) |lane| for (0..4) |j| {
                    const r = Ref.bf(1.0 * Ref.bf(o[lane][j] * inv));
                    if (sec == 0) qs[t][lane * 4 + j] = Ref.bf(r * scale) else ks[t][lane * 4 + j] = Ref.bf(r * scale);
                };
            }
        }
        // 2. gate: simdgroups 3R .. 4R - 1; beta: simdgroup 4R, lanes < R.
        for (3 * R..4 * R) |sg| {
            const t = sg - 3 * R;
            for (0..32) |lane| for (0..4) |j| {
                const ch = lane * 4 + j;
                const gc = h * DK + ch;
                gs[t][ch] = Ref.gate(in.fa[t * DIM + gc], in.dt_bias[gc], in.a_exp[gc], in.lower);
            };
        }
        for (0..32) |lane| if (lane < R) {
            betas[lane] = Ref.sigmoidB(in.bp[lane * HV + h]);
        };
        // 3. window and next tail: every thread e = tid, tid + 1024, ...
        var e: usize = 0;
        while (e < NROW * 3 * DK) : (e += 1) {
            const wr = e / (3 * DK);
            const rem = e - wr * (3 * DK);
            const sec = rem / DK;
            const ch = rem - sec * DK;
            const c = sec * DIM + h * DK + ch;
            const p = if (sec == 0) in.qp else if (sec == 1) in.kp else in.vp;
            const val = if (wr < TAPS - 1) in.tail[wr * CDIM + c] else p[(wr - (TAPS - 1)) * DIM + h * DK + ch];
            out.window[wr * CDIM + c] = val;
            if (wr >= R) out.next_tail[(wr - R) * CDIM + c] = val;
        }
        // 4. recurrence: simdgroup sg takes value rows sg + 32 j.
        const seq_stride = HV * DK * DK;
        for (0..32) |sg| for (0..DK / 32) |jj| {
            const dv_idx = sg + 32 * jj;
            var state: [32][4]f32 = undefined;
            for (0..32) |lane| for (0..4) |i| {
                state[lane][i] = in.state[(h * DK + dv_idx) * DK + 4 * lane + i];
            };
            for (0..R) |t| {
                var lanes: [32]f32 = undefined;
                for (0..32) |lane| {
                    var kv_mem: f32 = 0;
                    for (0..4) |i| {
                        const s_idx = 4 * lane + i;
                        state[lane][i] = state[lane][i] * gs[t][s_idx];
                        kv_mem += state[lane][i] * ks[t][s_idx];
                    }
                    lanes[lane] = kv_mem;
                }
                const kv_mem = Ref.simdSum(&lanes);
                const delta = (vs[t][dv_idx] - kv_mem) * betas[t];
                for (0..32) |lane| {
                    var o: f32 = 0;
                    for (0..4) |i| {
                        const s_idx = 4 * lane + i;
                        state[lane][i] = state[lane][i] + ks[t][s_idx] * delta;
                        o += state[lane][i] * qs[t][s_idx];
                    }
                    lanes[lane] = o;
                }
                ys[t][dv_idx] = Ref.bf(Ref.simdSum(&lanes));
                if (t + 1 < R) for (0..32) |lane| for (0..4) |i| {
                    out.seq[t * seq_stride + (h * DK + dv_idx) * DK + 4 * lane + i] = state[lane][i];
                };
            }
            for (0..32) |lane| for (0..4) |i| {
                out.state[(h * DK + dv_idx) * DK + 4 * lane + i] = state[lane][i];
            };
        };
        // 5. gated norm: simdgroup t < R.
        for (0..R) |t| {
            var lanes: [32]f32 = undefined;
            for (0..32) |lane| {
                var acc: f32 = 0;
                for (0..4) |j| acc += ys[t][lane * 4 + j] * ys[t][lane * 4 + j];
                lanes[lane] = acc;
            }
            var acc = Ref.simdSum(&lanes);
            var second: [32]f32 = @splat(0);
            second[0] = acc;
            acc = Ref.simdSum(&second);
            const inv = Ref.rsqrt(acc / @as(f32, @floatFromInt(DK)) + in.eps);
            for (0..32) |lane| for (0..4) |j| {
                const ch = lane * 4 + j;
                const at = t * DIM + h * DK + ch;
                const rn = Ref.bf(in.norm_w[ch] * Ref.bf(ys[t][ch] * inv));
                const zg = Ref.sigmoidB(in.z[at]);
                out.y[at] = Ref.bf(rn * zg);
            };
        }
    }
}

fn refBits(x: []const f32, y: []const f32) usize {
    var d: usize = 0;
    for (x, y) |p, q| d += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
    return d;
}

test "glm5_next kda_verify CPU reference: the fused text's arithmetic against the op chain's at Flash's shapes (verify rows 2, 3, 4; random inputs, a carried state): output, state, capture rows, window and next tail bit for bit" {
    const a = testing.allocator;
    const c: glm.Config = .{};
    const hv: usize = c.kda_heads;
    const dk: usize = c.kda_head_dim;
    const dim = hv * dk;
    const cdim = 3 * dim;
    const taps: usize = c.kda_conv;
    var prng = std.Random.DefaultPrng.init(0x6b6676);
    const r = prng.random();
    const Mk = struct {
        fn buf(al: std.mem.Allocator, rr: std.Random, n: usize, sd: f32, mean: f32, round: bool) ![]f32 {
            const v = try al.alloc(f32, n);
            for (v) |*x| {
                const f = mean + sd * rr.floatNorm(f32);
                x.* = if (round) Ref.bf(f) else f;
            }
            return v;
        }
    };
    const s = kda.qkScales(&c);
    for ([_]usize{ 2, 3, 4 }) |rows| {
        var bufs: std.ArrayList([]f32) = .empty;
        defer {
            for (bufs.items) |b| a.free(b);
            bufs.deinit(a);
        }
        const specs = [_]struct { n: usize, sd: f32, mean: f32, round: bool }{
            .{ .n = rows * dim, .sd = 1.0, .mean = 0, .round = true }, // qp
            .{ .n = rows * dim, .sd = 1.0, .mean = 0, .round = true }, // kp
            .{ .n = rows * dim, .sd = 1.0, .mean = 0, .round = true }, // vp
            .{ .n = (taps - 1) * cdim, .sd = 1.0, .mean = 0, .round = true }, // tail (bf16-exact: the rows widened)
            .{ .n = cdim * taps, .sd = 0.5, .mean = 0, .round = true }, // conv_w (the bind's f32 copy of bf16)
            .{ .n = rows * dim, .sd = 1.0, .mean = 0, .round = true }, // fa
            .{ .n = dim, .sd = 0.5, .mean = 0, .round = true }, // dt_bias
            .{ .n = dim, .sd = 0.5, .mean = 1.0, .round = false }, // a_exp
            .{ .n = rows * hv, .sd = 1.0, .mean = 0, .round = true }, // bp
            .{ .n = rows * dim, .sd = 1.0, .mean = 0, .round = true }, // z
            .{ .n = dk, .sd = 0.2, .mean = 1.0, .round = true }, // norm_w
            .{ .n = hv * dk * dk, .sd = 0.05, .mean = 0, .round = false }, // the carried state
        };
        for (specs) |sp| try bufs.append(a, try Mk.buf(a, r, sp.n, sp.sd, sp.mean, sp.round));
        for (bufs.items[7]) |*x| x.* = @abs(x.*);
        const b = bufs.items;
        const in: RefIn = .{
            .rows = rows,
            .hv = hv,
            .dk = dk,
            .taps = taps,
            .qp = b[0],
            .kp = b[1],
            .vp = b[2],
            .tail = b[3],
            .conv_w = b[4],
            .fa = b[5],
            .dt_bias = b[6],
            .a_exp = b[7],
            .bp = b[8],
            .z = b[9],
            .norm_w = b[10],
            .state = b[11],
            .lower = c.kda_lower_bound,
            .eps = c.rms_norm_eps,
            .q_scale = Ref.bf(@floatCast(s.q)),
            .k_scale = Ref.bf(@floatCast(s.k)),
        };
        var ops_out = try RefOut.init(a, &in);
        defer ops_out.deinit(a);
        var fused_out = try RefOut.init(a, &in);
        defer fused_out.deinit(a);
        // The capture's last row is never written (either route): zero it on both so the compare reads written rows only.
        @memset(ops_out.seq, 0);
        @memset(fused_out.seq, 0);
        try refOpChain(a, &in, &ops_out);
        refFused(&in, &fused_out);
        const d = [_]usize{
            refBits(fused_out.y, ops_out.y),
            refBits(fused_out.state, ops_out.state),
            refBits(fused_out.seq, ops_out.seq),
            refBits(fused_out.window, ops_out.window),
            refBits(fused_out.next_tail, ops_out.next_tail),
        };
        std.debug.print("GLM5NEXT_KDA_FUSED_CPU_REF rows={d} bit_diffs y={d} state={d} capture_seq={d} window={d} next_tail={d}\n", .{ rows, d[0], d[1], d[2], d[3], d[4] });
        for (d) |n| try testing.expectEqual(@as(usize, 0), n);
        // Not vacuous: the outputs move (a non-zero state and output, a capture row that differs from the final state).
        var nz: usize = 0;
        for (ops_out.y) |x| nz += @intFromBool(x != 0);
        try testing.expect(nz > ops_out.y.len / 2);
        try testing.expect(refBits(ops_out.seq[0 .. hv * dk * dk], ops_out.state) > 0);
    }
}

// ── the window's parity test ──────────────────────────────────────────────

/// Static bound (declared before any run): the synthetic layer's residents (q / k / v / o 4 x 64 MiB BF16, the gate
/// stages and conv < 8 MiB): 0.27 GB; one 37-row prompt span through the op chain for the carried state (< 0.1 GB); per
/// verify width (2, 3, 4 rows) two state copies (4.3 MB each), two captures (the state rows f32 [R, 1, 64, 128, 128] and
/// the window: < 17 MB each), the op chain's intermediates (< 3 MB) and the stage probe (< 1 MB); the host read-backs
/// (two f32 copies of the widest compared array, 2 x 16.8 MB): < 0.45 GB plus the texts' JIT; declared 0.6 GB.
const fused_parity_bound_bytes: u64 = 600_000_000;

fn bf16Bits(x: f32) u16 {
    return ops_be.bf16Bits(x);
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

/// The stage probe's text: the header's element functions alone over flat arrays (the gate over `fa` with the
/// per-channel tables, the bf16 sigmoid over `sx`), compared in the parity test with the op chain's own ops.
const probe_source: [:0]const u8 =
    \\const uint i = thread_position_in_grid.x;
    \\if (i < uint(NG)) gate_out[i] = glm5n_kv_gate(fa[i], dt_bias[i % uint(DIM)], a_exp[i % uint(DIM)], cf[0]);
    \\if (i < uint(NS)) sig_out[i] = glm5n_kv_sigmoid<bfloat16_t>(sx[i]);
;

fn probe(g: *G, k: *const Kernels, fa: T, dt_bias: T, a_exp: T, sx: T, dim: c_int) !struct { gate: T, sig: T } {
    const ng: c_int = @intCast(mlx.mlx_array_size(fa));
    const ns: c_int = @intCast(mlx.mlx_array_size(sx));
    const ins_n = [_][*:0]const u8{ "fa", "dt_bias", "a_exp", "sx", "cf" };
    const outs_n = [_][*:0]const u8{ "gate_out", "sig_out" };
    const iv_n = mlx.mlx_vector_string_new_data(&ins_n, ins_n.len);
    defer _ = mlx.mlx_vector_string_free(iv_n);
    const ov_n = mlx.mlx_vector_string_new_data(&outs_n, outs_n.len);
    defer _ = mlx.mlx_vector_string_free(ov_n);
    const kern = mtl.mlx_fast_metal_kernel_new("glm5n_kda_verify_probe", iv_n, ov_n, probe_source, texts.header, true, false);
    if (kern.ctx == null) return error.MetalKernelCompileFailed;
    defer _ = mtl.mlx_fast_metal_kernel_free(kern);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ng}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ns}, 1, .bfloat16));
    const n = @max(ng, ns);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(n + 255, 256) * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    inline for (.{ .{ "NG", ng }, .{ "NS", ns }, .{ "DIM", dim } }) |ta| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    const ins = [_]T{ try g.reshape(fa, &.{ng}), dt_bias, a_exp, try g.reshape(sx, &.{ns}), k.cf };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var o: [2]T = undefined;
    for (&o, 0..) |*x, i| {
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&y, ov, i));
        x.* = try g.adopt(y);
    }
    return .{ .gate = o[0], .sig = o[1] };
}

test "glm5_next kda verify fused GPU: the fused verify text against the op chain at the verify's widths (2, 3, 4 rows from a carried state): the whole capture forward (output, state, conv tail, captured rows and window) bit for bit, and the element functions against the op chain's ops" {
    _ = std.c.getenv("GLM5NEXT_KDA_FUSED_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5NEXT_KDA_FUSED_PARITY bound_bytes={d}\n", .{fused_parity_bound_bytes});
    mlx.installErrorHandler();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: glm.Config = .{};
    const graph = @import("graph.zig");
    const kda_percore = @import("kda_percore.zig");
    var prng = std.Random.DefaultPrng.init(0x6b6670);
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
    // The layer as the bind builds it (the prompt and verify-prologue parities' synthetic layer).
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
    // The module's installs: the recurrence's kernels and plan, the per-core prompt route, then this route (its compile
    // check included).
    try kda.install(&c);
    _ = try kda_percore.install(&g, &c);
    var diag: sdk.Diag = .{};
    var kern = install(a, s, &c, &.{&w}, &diag) catch |e| {
        std.debug.print("GLM5NEXT_KDA_FUSED_PARITY install refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
        return e;
    };
    defer kern.deinit();
    // The carried state: one 37-row prompt span through kda.forward (a non-zero conv tail and recurrent state).
    var base: kda.State = .{};
    defer base.reset(&g);
    {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try randBf16(&g, a, r, &.{ 1, 37, h }, 1.0, 0.0);
        defer g.release(x);
        _ = try kda.forward(&g, &c, &w, x, &base);
        try g.evalAll(&.{ base.conv.?, base.rec.? });
    }
    var total: usize = 0;
    // The element functions alone (the header's gate and bf16 sigmoid) against the op chain's ops on a verify's inputs.
    {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try randBf16(&g, a, r, &.{ 1, 4, h }, 1.0, 0.0);
        defer g.release(x);
        const fa = try g.matmul(try fp8.project(&g, w.f_a, x), try g.transpose(w.f_b));
        const z = try g.matmul(try fp8.project(&g, w.g_a, x), try g.transpose(w.g_b));
        const bp = try fp8.project(&g, w.b, x);
        const pr = try probe(&g, &kern, fa, w.dt_bias, w.a_exp, z, dim);
        const pb = try probe(&g, &kern, fa, w.dt_bias, w.a_exp, bp, dim);
        const d_gate = try bitDiffs(&g, a, try g.reshape(pr.gate, &.{ 1, 4, dim }), try kda.gateOf(&g, &c, &w, fa));
        const d_z = try bitDiffs(&g, a, try g.reshape(pr.sig, &.{ 1, 4, dim }), try g.sigmoid(z));
        const d_b = try bitDiffs(&g, a, try g.reshape(pb.sig, &.{ 1, 4, hv }), try g.sigmoid(bp));
        const sum = d_gate + d_z + d_b;
        std.debug.print("GLM5NEXT_KDA_FUSED_PARITY stage=elements {s} bit_diffs gate={d} zgate={d} beta={d}\n", .{ if (sum == 0) "PASS" else "FAIL", d_gate, d_z, d_b });
        total += sum;
    }
    for ([_]c_int{ 2, 3, 4 }) |rows| {
        const m = g.mark();
        defer g.resetTo(m);
        const x = try randBf16(&g, a, r, &.{ 1, rows, h }, 1.0, 0.0);
        defer g.release(x);
        const one = try g.scalar(1.0, .float32);
        var s_ops: kda.State = .{};
        defer s_ops.reset(&g);
        var s_txt: kda.State = .{};
        defer s_txt.reset(&g);
        s_ops.set(&g, try g.mul(base.conv.?, one), try g.mul(base.rec.?, one));
        s_txt.set(&g, try g.mul(base.conv.?, one), try g.mul(base.rec.?, one));
        var cap_ops: kda.Capture = .{};
        defer cap_ops.release(&g);
        var cap_txt: kda.Capture = .{};
        defer cap_txt.release(&g);
        const y_ops = try kda.forwardCapture(&g, &c, &w, x, &s_ops, &cap_ops);
        const y_txt = try run(&kern, &g, &c, &w, x, &s_txt, &cap_txt);
        try g.evalAll(&.{ y_ops, y_txt, s_ops.rec.?, s_txt.rec.?, s_ops.conv.?, s_txt.conv.?, cap_ops.seq.?, cap_txt.seq.?, cap_ops.window.?, cap_txt.window.? });
        // The captured rows a partial accept restores: 0 .. R - 2 (row R - 1 is never written by either route).
        const seq_rows = [_]c_int{ rows - 1, 1, hv, dk, dk };
        const zero5 = [_]c_int{ 0, 0, 0, 0, 0 };
        const ones5 = [_]c_int{ 1, 1, 1, 1, 1 };
        const d_y = try bitDiffs(&g, a, y_txt, y_ops);
        const d_s = try bitDiffs(&g, a, s_txt.rec.?, s_ops.rec.?);
        const d_c = try bitDiffs(&g, a, s_txt.conv.?, s_ops.conv.?);
        const d_seq = try bitDiffs(&g, a, try g.slice(cap_txt.seq.?, &zero5, &seq_rows, &ones5), try g.slice(cap_ops.seq.?, &zero5, &seq_rows, &ones5));
        const d_win = try bitDiffs(&g, a, cap_txt.window.?, cap_ops.window.?);
        const shape_diffs = d_y + d_s + d_c + d_seq + d_win;
        std.debug.print("GLM5NEXT_KDA_FUSED_PARITY shape=verify_rows{d} {s} bit_diffs layer_y={d} state={d} conv_tail={d} capture_seq={d} capture_window={d}\n", .{ rows, if (shape_diffs == 0) "PASS" else "FAIL", d_y, d_s, d_c, d_seq, d_win });
        total += shape_diffs;
    }
    std.debug.print("GLM5NEXT_KDA_FUSED_PARITY overall {s} total_bit_diffs={d}\n", .{ if (total == 0) "PASS" else "FAIL", total });
    try testing.expectEqual(@as(usize, 0), total);
}
