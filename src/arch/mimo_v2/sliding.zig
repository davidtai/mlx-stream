//! K3 (mimo/k2-glue, `"decode_sliding": "fused"`): a sliding layer's attention block at M <= 8 rows as
//!   `mimo_sliding_qkv` (the q/k/v BF16 projections in the exact per-row accumulation of the MLX kernel the control runs
//!   at that M, the partial rope on q and k and the value scale on v: one dispatch for 3 GEMVs + 22 element-wise ops),
//!   the ring's two slice_update writes (MLX: they donate the ring's slot buffers in place; a custom kernel cannot),
//!   `mimo_sliding_window_o` (the kernels route's `mimo_dec_attn_window` text with its output written as the o
//!   projection's [M, 64 x 128] rows: the transpose copy gone) and the o projection (MLX's matmul, unchanged).
//! Bit-equal per row to the control at the same M (`decode_attention: kernels`, the decode_attention route it requires).
//! The control itself is not M-invariant: its projections run MLX `gemv` at M = 1 and `gemv_wide` with k_lanes 32 at
//! M = 2..5 and 16 at M = 6..8, and its window kernel splits the keys into KS = 4 / 4 / 2 / 2 / 1.. slices by M; the
//! fused route reproduces each M's arithmetic exactly, so a row's bits depend on M exactly as the control's do.
//! Construction refuses the route on anything but BF16 q/k/v residents (the fp8 resident tier's hook is
//! header_mimo_sliding.metal `sliding_w4`), the BF16 KV tier and the kernels decode-attention tier.
const std = @import("std");
const mlx = @import("sdk").mlx;
const mimo = @import("config.zig");
const glue = @import("glue.zig");
const decode_attention = @import("decode_attention.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const Kernel = mlx.mlx_fast_metal_kernel;

pub const Route = enum { ops, fused, invalid };

pub fn parseRoute(text: []const u8) Route {
    return std.meta.stringToEnum(Route, text) orelse .invalid;
}

const hidden: c_int = 4096;
const heads: c_int = 64;
const kv_heads: c_int = 8;
const kd: c_int = 192;
const vd: c_int = 128;
/// 64-row blocks: q 64 heads x 3, k 8 x 3, v 8 x 2.
pub const qkv_groups: c_int = heads * 3 + kv_heads * 3 + kv_heads * 2;

/// The control's projection kernel per M (MLX matmul.cpp at 64ea011cb: M = 1 `gemv`; M >= 2 `gemv_wide` with
/// passes = (M + 4) / 5, k_lanes 32 for one pass, 16 otherwise): 0 gemv, 1 gemv_wide k32, 2 gemv_wide k16.
pub fn mode(rows: c_int) c_int {
    return if (rows == 1) 0 else if (rows <= 5) 1 else 2;
}

pub const texts = struct {
    pub const header = @embedFile("kernels/header_mimo_sliding.metal");
    pub const qkv = @embedFile("kernels/mimo_sliding_qkv.metal");
    pub const window_o = @embedFile("kernels/mimo_sliding_window_o.metal");
    /// The kernels route's text the window_o text is derived from, and its header.
    pub const window = decode_attention.texts.window;
    pub const mma = decode_attention.texts.header;
};
pub const text_pins = [_]struct { file: []const u8, text: []const u8, sha256: []const u8 }{
    .{ .file = "header_mimo_sliding.metal", .text = texts.header, .sha256 = "00842d377563567b8c97e9b15e153f7b6af54bd1fe19aa3da604d2927dd1ecd5" },
    .{ .file = "mimo_sliding_qkv.metal", .text = texts.qkv, .sha256 = "e2775f6a3b7c656b7e932d4caf7d0ff94bfdf2a816f6fa184da01cccdd2915ac" },
    .{ .file = "mimo_sliding_window_o.metal", .text = texts.window_o, .sha256 = "e09916fabab924f1fcad60a926b453222ed67cee66ba29aa95dd76f0a4c9554d" },
};
/// window_o = the window text with exactly these lines replaced (the output's layout; the arithmetic is the window's).
pub const window_o_edits = [_]struct { old: []const u8, new: []const u8 }{
    .{ .old = "// normalised row is written as T to out [1, Hq, S, BDV]. MMA, staging, online\n", .new = "// normalised row is written as T to out [S, HQ x BDV] (K3: the o projection's rows). MMA, staging, online\n" },
    .{ .old = "  const long ob = ((long)hq * S + qs) * BDV;\n", .new = "  const long ob = ((long)qs * HQ + hq) * BDV;\n" },
};

pub fn checkPins() !void {
    try glue.checkPins();
    for (text_pins) |p| if (!std.mem.eql(u8, &glue.hexOf(p.text), p.sha256)) return error.MimoSlidingTextNotPinned;
}

/// The geometry the texts are written for; anything else is refused at construction.
pub fn requireGeometry(c: mimo.Config) !void {
    if (c.hidden_size != hidden or c.n_heads != heads or c.swa_n_kv_heads != kv_heads or c.head_dim != kd or c.value_head_dim != vd or c.rope_head_dim != 64 or c.window != 128)
        return error.MimoSlidingGeometry;
    try decode_attention.requireGeometry(c);
}

pub const Kernels = struct {
    qkv: Kernel = .{ .ctx = null },
    window_o: Kernel = .{ .ctx = null },
    scl: Array = .{},

    pub fn init(c: mimo.Config) !Kernels {
        try requireGeometry(c);
        try checkPins();
        var k: Kernels = .{};
        errdefer k.deinit();
        k.qkv = try makeKernel("mimo_sliding_qkv", &.{ "x", "wq", "wk", "wv", "cosv", "sinv", "vscale" }, &.{ "q", "k", "v" }, texts.qkv, texts.header, true);
        // K/V are the ring's views: read through their strides, never copied (as the window kernel's).
        k.window_o = try makeKernel("mimo_sliding_window_o", &.{ "q", "k", "v", "sinks", "scl" }, &.{"out"}, texts.window_o, texts.mma, false);
        const scale = [_]f32{1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)))};
        const one = [_]c_int{1};
        k.scl = mlx.mlx_array_new_data(&scale, &one, 1, .float32);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.qkv, k.window_o }) |x| if (x.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_free(x);
        };
        drop(k.scl);
        k.* = .{};
    }
};

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, contiguous: bool) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, contiguous, false);
    if (k.ctx == null) return error.MimoSlidingKernel;
    return k;
}

pub const Qkv = struct {
    q: Array,
    k: Array,
    v: Array,
    pub fn deinit(p: *Qkv) void {
        drop(p.q);
        drop(p.k);
        drop(p.v);
        p.* = .{ .q = .{}, .k = .{}, .v = .{} };
    }
};

/// x [M, 4096] (the input norm's output), the BF16 q/k/v weights [out, 4096], the trunk's rope tables [1, 1, M, 64] and
/// the value scale (F32 scalar) -> q [1, 64, M, 192] and k [1, 8, M, 192] roped, v [1, 8, M, 128] scaled.
pub fn qkv(k: *const Kernels, x: Array, wq: Array, wk: Array, wv: Array, cos: Array, sin: Array, vscale: Array, s: Stream) !Qkv {
    const m = mlx.getShape(x)[0];
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 1, heads, m, kd }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 1, kv_heads, m, kd }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 1, kv_heads, m, vd }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 256, qkv_groups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "S", m));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "MODE", mode(m)));
    const ins = [_]Array{ x, wq, wk, wv, cos, sin, vscale };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&res, k.qkv, in_vec, c, s));
    var out: Qkv = .{ .q = .{}, .k = .{}, .v = .{} };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.q, res, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.k, res, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.v, res, 2));
    return out;
}

/// The window attention over the ring's views (q [1, 64, M, 192], K / V [1, 8, keys, D] views, sinks [64]) as the
/// kernels route computes it, written [M, 64 x 128] (the o projection's input).
pub fn windowO(k: *const Kernels, q: Array, kview: Array, vview: Array, sinks: Array, s: Stream) !Array {
    const rows = mlx.getShape(q)[2];
    const l = decode_attention.windowLaunch(kv_heads, 8, rows);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ rows, heads * vd }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    for ([_]struct { [*:0]const u8, c_int }{ .{ "G", 8 }, .{ "S", rows }, .{ "BDK", kd }, .{ "BDV", vd }, .{ "KS", l.ks }, .{ "BK", 32 }, .{ "W", 128 }, .{ "HQ", heads } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const ins = [_]Array{ q, kview, vview, sinks, k.scl };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k.window_o, in_vec, c, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

// ---- host models (CPU proofs): the control's MLX kernels per row from MLX's source, the qkv kernel transliterated ----

const model = glue.model;
const K: usize = 4096;

/// `dot(float4, float4)`: one shared model (the hardware's order is the same on both sides: the same intrinsic).
fn dot4(a: [4]f32, b: [4]f32) f32 {
    return @mulAdd(f32, a[3], b[3], @mulAdd(f32, a[2], b[2], @mulAdd(f32, a[1], b[1], a[0] * b[0])));
}

/// MLX `gemv` (GEMVKernel, SN = 32, TN = 4; TM rows per simdgroup do not touch a row's order): one output row.
pub fn mlxGemvRow(w: []const f32, x: []const f32) f32 {
    var lanes: [32]f32 = undefined;
    for (0..32) |l| {
        var result: f32 = 0;
        var bn = 4 * l;
        for (0..32) |_| {
            for (0..4) |tn| result = @mulAdd(f32, w[bn + tn], x[bn + tn], result);
            bn += 128;
        }
        lanes[l] = result;
    }
    return model.shuffleTree(lanes, 32, 0);
}

/// MLX `gemv_wide` (GemvWide::run, unroll 8, n_main = n_v4): one output row x one vector, `k_lanes` lanes.
pub fn mlxGemvWideRow(w: []const f32, x: []const f32, k_lanes: usize) f32 {
    var lanes: [32]f32 = @splat(0);
    for (0..k_lanes) |kl| {
        var result: f32 = 0;
        var base: usize = 0;
        while (base < K / 4) : (base += k_lanes * 8) {
            var acc: f32 = 0;
            for (0..8) |i| {
                const v4 = base + i * k_lanes + kl;
                acc += dot4(w[4 * v4 ..][0..4].*, x[4 * v4 ..][0..4].*);
            }
            result += acc;
        }
        lanes[kl] = result;
    }
    return model.shuffleTree(lanes, k_lanes, 0);
}

/// mimo_sliding_qkv.metal's product for one 64-row block (rows `row0 ..`): out[row * S + r] (BF16-rounded).
pub fn fusedBlock(W: []const f32, row0: usize, x: []const f32, s_rows: usize, out: []f32) void {
    const md = mode(@intCast(s_rows));
    for (0..8) |sg| {
        if (md == 0) {
            var lanes: [8][32]f32 = undefined;
            for (0..32) |lane| {
                var res: [8]f32 = @splat(0);
                for (0..32) |i| {
                    const col = 128 * i + 4 * lane;
                    for (0..8) |j| {
                        const wr = W[(row0 + 8 * sg + j) * K ..];
                        for (0..4) |t| res[j] = @mulAdd(f32, wr[col + t], x[col + t], res[j]);
                    }
                }
                for (0..8) |j| lanes[j][lane] = res[j];
            }
            for (0..8) |j| out[(8 * sg + j) * s_rows] = model.bf16Round(model.shuffleTree(lanes[j], 32, 0));
        } else {
            const kl_n: usize = if (md == 1) 32 else 16;
            const rps = 32 / kl_n;
            for (0..8 / rps) |p| {
                var lanes: [8][32]f32 = undefined;
                for (&lanes) |*row| row.* = @splat(0);
                var rows_of: [32]usize = undefined;
                for (0..32) |lane| {
                    const rr = 8 * sg + rps * p + lane / kl_n;
                    rows_of[lane] = rr;
                    const kl = lane % kl_n;
                    const wr = W[(row0 + rr) * K ..];
                    var result: [8]f32 = @splat(0);
                    var base: usize = 0;
                    while (base < K / 4) : (base += kl_n * 8) {
                        var wf: [8][4]f32 = undefined;
                        for (0..8) |i| wf[i] = wr[4 * (base + i * kl_n + kl) ..][0..4].*;
                        for (0..s_rows) |r| {
                            var acc: f32 = 0;
                            for (0..8) |i| acc += dot4(wf[i], x[r * K + 4 * (base + i * kl_n + kl) ..][0..4].*);
                            result[r] += acc;
                        }
                    }
                    for (0..s_rows) |r| lanes[r][lane] = result[r];
                }
                for (0..s_rows) |r| {
                    var lane: usize = 0;
                    while (lane < 32) : (lane += kl_n) out[rows_of[lane] * s_rows + r] = model.bf16Round(model.shuffleTree(lanes[r], kl_n, lane));
                }
            }
        }
    }
}

/// The qkv kernel's epilogue for a block (`part`, `kind` 0 q / 1 k / 2 v) on its BF16 tile: rope or value scale.
pub fn fusedEpilogue(tile: []const f32, s_rows: usize, kind: u8, part: usize, cos: []const f32, sin: []const f32, vscale: f32, out: []f32) void {
    for (0..64 * s_rows) |t| {
        const d = t / s_rows;
        const r = t % s_rows;
        var val = tile[d * s_rows + r];
        if (kind == 2) {
            val = model.bf16Round(val * vscale);
        } else if (part == 0) {
            const other = tile[(if (d < 32) d + 32 else d - 32) * s_rows + r];
            const rot = if (d < 32) -other else other;
            const c = model.bf16Round(val * cos[r * 64 + d]);
            const si = model.bf16Round(rot * sin[r * 64 + d]);
            val = model.bf16Round(c + si);
        }
        out[d * s_rows + r] = val;
    }
}

// ---- CPU tests ----

const testing = std.testing;

const Rng = struct {
    st: u32,
    fn next(r: *Rng) u32 {
        r.st ^= r.st << 13;
        r.st ^= r.st >> 17;
        r.st ^= r.st << 5;
        return r.st;
    }
    fn bf(r: *Rng, amp: f32) f32 {
        const u = @as(f32, @floatFromInt(r.next() >> 8)) / @as(f32, 1 << 24);
        return model.bf16Round((2 * u - 1) * amp);
    }
};

test "mimo sliding CPU: K3 texts match their pins; window_o is the kernels route's window text with only its output layout replaced" {
    try checkPins();
    const J = struct {
        headers: []const struct { file: []const u8, sha256: []const u8 },
        kernels: []const struct { source: struct { file: []const u8, sha256: []const u8 } },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, glue.texts.manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    var seen: usize = 0;
    for (text_pins) |p| {
        for (m.value.headers) |h| if (std.mem.eql(u8, h.file, p.file)) {
            try testing.expectEqualStrings(p.sha256, h.sha256);
            seen += 1;
        };
        for (m.value.kernels) |k| if (std.mem.eql(u8, k.source.file, p.file)) {
            try testing.expectEqualStrings(p.sha256, k.source.sha256);
            seen += 1;
        };
    }
    try testing.expectEqual(text_pins.len, seen);
    var derived: []const u8 = texts.window;
    var buf: [2][8192]u8 = undefined;
    for (window_o_edits, 0..) |e, i| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, derived, e.old));
        const n = std.mem.replacementSize(u8, derived, e.old, e.new);
        _ = std.mem.replace(u8, derived, e.old, e.new, buf[i][0..n]);
        derived = buf[i][0..n];
    }
    try testing.expectEqualStrings(texts.window_o, derived);
    // The window text is still the one the decode-attention pins hold (the arithmetic K3 claims to reproduce).
    try decode_attention.checkPins();
    try requireGeometry(.{});
    var odd: mimo.Config = .{};
    odd.swa_n_kv_heads = 4;
    try testing.expectError(error.MimoSlidingGeometry, requireGeometry(odd));
}

test "mimo sliding CPU: launch geometry: 232 qkv blocks cover q/k/v rows once; window_o launches as the kernels route's window; modes by M" {
    // Blocks: every output row of the three projections in exactly one block.
    var covered: [12288 + 1536 + 1024]u32 = @splat(0);
    var blk: c_int = 0;
    while (blk < qkv_groups) : (blk += 1) {
        const b: usize = @intCast(blk);
        const kind: usize = if (b < 192) 0 else if (b < 216) 1 else 2;
        const head: usize = if (kind == 0) b / 3 else if (kind == 1) (b - 192) / 3 else (b - 216) / 2;
        const part: usize = if (kind == 0) b % 3 else if (kind == 1) (b - 192) % 3 else (b - 216) % 2;
        const hd: usize = if (kind == 2) 128 else 192;
        const base: usize = if (kind == 0) 0 else if (kind == 1) 12288 else 12288 + 1536;
        for (0..64) |d| covered[base + head * hd + part * 64 + d] += 1;
    }
    for (covered) |c| try testing.expectEqual(@as(u32, 1), c);
    try testing.expectEqual(@as(c_int, 0), mode(1));
    for ([_]c_int{ 2, 3, 4, 5 }) |m| try testing.expectEqual(@as(c_int, 1), mode(m));
    for ([_]c_int{ 6, 7, 8 }) |m| try testing.expectEqual(@as(c_int, 2), mode(m));
    const J = struct {
        kernels: []const struct {
            name: []const u8,
            launch_samples: []const struct { grid: [3]c_int, threadgroup: [3]c_int, vars: []const struct { name: []const u8, value: c_int }, template: []const struct { name: []const u8, int: ?c_int = null } },
        },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, glue.texts.manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    var checked: usize = 0;
    for (m.value.kernels) |k| {
        const is_qkv = std.mem.eql(u8, k.name, "mimo_sliding_qkv");
        const is_win = std.mem.eql(u8, k.name, "mimo_sliding_window_o");
        if (!is_qkv and !is_win) continue;
        for (k.launch_samples) |sample| {
            var rows: c_int = 0;
            for (sample.vars) |v| if (std.mem.eql(u8, v.name, "S")) {
                rows = v.value;
            };
            if (is_qkv) {
                try testing.expectEqual([3]c_int{ 256, qkv_groups, 1 }, sample.grid);
                try testing.expectEqual([3]c_int{ 256, 1, 1 }, sample.threadgroup);
                for (sample.template) |t| if (std.mem.eql(u8, t.name, "MODE")) try testing.expectEqual(mode(rows), t.int.?);
            } else {
                const l = decode_attention.windowLaunch(8, 8, rows);
                try testing.expectEqual(l.grid, sample.grid);
                try testing.expectEqual(l.tg, sample.threadgroup);
                for (sample.template) |t| if (std.mem.eql(u8, t.name, "KS")) try testing.expectEqual(l.ks, t.int.?);
            }
            checked += 1;
        }
    }
    try testing.expectEqual(@as(usize, 16), checked);
}

test "mimo sliding CPU: the qkv product equals MLX gemv (M = 1) / gemv_wide k32 (M = 2..5) / k16 (M = 6..8) per row bit for bit on real-shaped blocks" {
    const a = testing.allocator;
    var r: Rng = .{ .st = 0x2545f491 };
    // Four 64-row blocks of a real-width [rows, 4096] BF16 weight (q's rope block, a q pass-through block, k, v rows).
    const W = try a.alloc(f32, 256 * K);
    defer a.free(W);
    for (W) |*w| w.* = r.bf(0.03);
    const x = try a.alloc(f32, 8 * K);
    defer a.free(x);
    for (x) |*v| v.* = r.bf(2);
    var got: [64 * 8]f32 = undefined;
    var s_rows: usize = 1;
    while (s_rows <= 8) : (s_rows += 1) {
        for (0..4) |blk| {
            const row0 = 64 * blk;
            fusedBlock(W, row0, x, s_rows, &got);
            for (0..64) |d| for (0..s_rows) |rr| {
                const wr = W[(row0 + d) * K ..][0..K];
                const xr = x[rr * K ..][0..K];
                const want = model.bf16Round(switch (mode(@intCast(s_rows))) {
                    0 => mlxGemvRow(wr, xr),
                    1 => mlxGemvWideRow(wr, xr, 32),
                    else => mlxGemvWideRow(wr, xr, 16),
                });
                try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got[d * s_rows + rr])));
            };
        }
    }
}

test "mimo sliding CPU: the qkv epilogue equals module.partialRope / module.scaleValues (MLX CPU ops) bit for bit at M = 1..8" {
    // Element-wise BF16 and F32 ops: MLX's CPU backend computes exactly what its GPU kernels do, so the stock graph runs.
    const module = @import("module.zig");
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var r: Rng = .{ .st = 0x7f4a7c15 };
    const c: mimo.Config = .{};
    const freq = module.inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
    defer drop(freq);
    var m: u32 = 1;
    while (m <= 8) : (m += 1) {
        const n: c_int = @intCast(m);
        var tables = try module.rotaryTables(freq, 2175 + 37 * m, m, s);
        defer tables.deinit();
        try mlx.check(mlx.mlx_array_eval(tables.cos));
        try mlx.check(mlx.mlx_array_eval(tables.sin));
        const cosv = mlx.mlx_array_data_bfloat16(tables.cos).?;
        const sinv = mlx.mlx_array_data_bfloat16(tables.sin).?;
        var cos: [8 * 64]f32 = undefined;
        var sin: [8 * 64]f32 = undefined;
        for (0..m * 64) |i| {
            cos[i] = model.fromBits(cosv[i]);
            sin[i] = model.fromBits(sinv[i]);
        }
        // a [M, 8 x 192] k projection (heads x dims, the GEMV's BF16 output) and [M, 8 x 128] v
        for ([_]u8{ 1, 2 }) |kind| {
            const dim: usize = if (kind == 2) 128 else 192;
            const bits = try a.alloc(u16, m * 8 * dim);
            defer a.free(bits);
            for (bits) |*b| b.* = model.bf16Bits(r.bf(3));
            const proj = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{ n, @intCast(8 * dim) }, 2, .bfloat16);
            defer drop(proj);
            const view = try module.headsView(proj, n, 8, @intCast(dim), s);
            defer drop(view);
            const vscale = mlx.mlx_array_new_float(c.attention_value_scale);
            defer drop(vscale);
            const stock = if (kind == 2) try module.scaleValues(view, vscale, s) else try module.partialRope(view, tables, 64, s);
            defer drop(stock);
            var cont: Array = .{};
            defer drop(cont);
            try mlx.check(mlx.mlx_contiguous(&cont, stock, false, s));
            try mlx.check(mlx.mlx_array_eval(cont));
            const want = mlx.mlx_array_data_bfloat16(cont).?; // [1, 8, M, dim]
            var tile: [64 * 8]f32 = undefined;
            var out: [64 * 8]f32 = undefined;
            for (0..8) |h| for (0..dim / 64) |part| {
                for (0..64) |d| for (0..m) |rr| {
                    tile[d * m + rr] = model.fromBits(bits[rr * 8 * dim + h * dim + part * 64 + d]);
                };
                fusedEpilogue(&tile, m, kind, part, &cos, &sin, c.attention_value_scale, &out);
                for (0..64) |d| for (0..m) |rr| {
                    try testing.expectEqual(want[(h * m + rr) * dim + part * 64 + d], model.bf16Bits(out[d * m + rr]));
                };
            };
        }
    }
}
