//! L4: MiMo attention for logical M <= 8 (decode, and the MTP verify rows) over the BF16 cache views, read in place by
//! kernels. Replaces MLX SDPA with its host-built "array" mask on that phase; prefill (M > 8) keeps its padded route.
//!
//! - Full layers (9; 64 q / 4 KV heads, K192 / V128): mlx-serve 7a9e392 `src/dec_attn.zig`, `msv_dec_attn_split` +
//!   `msv_dec_attn_merge`, texts verbatim (donor CONTRACT: reports/.../mimo/donors/dec_attn). The kernel reads the
//!   `[0, pos + M)` slice of the `[1, 4, reserve, D]` slab through its strides; causal at the bottom right, no mask.
//!   Upstream keeps one query row under 8,192 keys on MLX's vector kernel (`MIN_KEYS_ONE_ROW`): that cutoff is installed
//!   here as a route sized by the served context (`one_row_min_keys`, to be re-measured on our shapes).
//! - Sliding layers (39; 64 q / 8 KV heads, window 128, per-head sinks): `mimo_dec_attn_window`, the split text with one
//!   split, a window lower bound, the sink folded once after the key-slice merge and a normalised BF16 output: one
//!   dispatch over the L2 `SlidingRing` view (656-row slots, head stride 656 x D, row stride D).
//!
//! Rounding-class against MLX SDPA (f32 MMA accumulation, exp2 online softmax). The tier is a construction-time route
//! (`Tier`): `.kernels` (the served default at every served context) or `.stock` (the control), by settings.
//! Construction proves the geometry once (the texts' pins are build-time facts the CPU suite holds); the installed
//! routes branch only on M and the key count, both runtime values. No key count bounds the texts
//! (`keyRangeInvariants`): keys are `int`, row offsets `long`, the split count is fixed (32 for 4 KV heads) and each
//! split's key slice only lengthens its loop; no buffer is sized by keys.
//! The device self-check (`mimo GPU decode attention ...`, env + lock gated) runs in the pre-ship gate, never at
//! construction.
const std = @import("std");
const mlx = @import("sdk").mlx;
const mimo = @import("config.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const Kernel = mlx.mlx_fast_metal_kernel;
const kset = @import("../../sdk_ext/kernel_set.zig");

pub const Tier = enum { kernels, stock, invalid };

pub fn parseTier(text: []const u8) Tier {
    return std.meta.stringToEnum(Tier, text) orelse .invalid;
}

/// Query positions one dispatch takes (an MTP verify window); M > 8 is prefill.
pub const max_rows: c_int = 8;
/// Upstream `MIN_KEYS_ONE_ROW`: one query row below this many keys stays on MLX's vector kernel.
pub const one_row_min_keys: u32 = 8192;
/// The kernels' key-count limit: the texts index keys as `int` (chunk, k0, k1, lim, c0 + r) and form element offsets as
/// `long` (key x row stride), so a view holds at most `maxInt(i32) / 2` keys; the served context is at most the
/// checkpoint's `max_position_embeddings` (1,048,576; settings `validate`), 1,024x inside it (`keyRangeInvariants`).
pub const max_keys: u64 = std.math.maxInt(i32) / 2;

/// Once, at construction (`Kernels.init`): the served context is inside the texts' index range (a comptime-known
/// checkpoint bound in practice; this refuses a config that would widen it).
pub fn requireKeyRange(served_context: u64) !void {
    if (served_context > max_keys) return error.MimoDecodeAttentionKeyRange;
}
const target_tgs: c_int = 128;
const bk: c_int = 32;
const kd: c_int = 192;
const vd: c_int = 128;
const window: c_int = 128;

// ---- registry: the texts and their pins (kernels/manifest-dec-attn.json) ----

pub const texts = struct {
    pub const manifest = @embedFile("kernels/manifest-dec-attn.json");
    pub const header = @embedFile("kernels/dec_attn_header_mma.metal");
    pub const split = @embedFile("kernels/dec_attn_split.metal");
    pub const merge = @embedFile("kernels/dec_attn_merge.metal");
    pub const window = @embedFile("kernels/dec_attn_window.metal");
};
/// sha256 of kernels/manifest-dec-attn.json, which records every text's sha256 (checked against `text_pins` by a CPU test).
pub const manifest_sha256 = "6d15c8c43eed1ec6765445eb9fbb7b7bc4f3c2e0e50979aaeef58cb46f57e001";
pub const text_pins = [_]struct { file: []const u8, text: []const u8, sha256: []const u8 }{
    .{ .file = "dec_attn_header_mma.metal", .text = texts.header, .sha256 = "5d70530f61a08aead840a7ac7a03ef087b40a16d23b7de5371b0be42426cb821" },
    .{ .file = "dec_attn_split.metal", .text = texts.split, .sha256 = "e8d8eaaa5eff9cf105752e3c894103691c23c17befaf24eceb99c24eb5d025d4" },
    .{ .file = "dec_attn_merge.metal", .text = texts.merge, .sha256 = "1c43462d81fe43b9eb6988346685518d8f812880bcba61ee7f4b42be57aa8a2a" },
    .{ .file = "dec_attn_window.metal", .text = texts.window, .sha256 = "8b83077c17db8a2f0ead905fb547625538d83972e806ee88c570c0d762aa54ba" },
};

pub fn checkPins() !void {
    if (!std.mem.eql(u8, &kset.shaHex(texts.manifest), manifest_sha256)) return error.MimoDecodeAttentionManifestNotPinned;
    for (text_pins) |p| if (!std.mem.eql(u8, &kset.shaHex(p.text), p.sha256)) return error.MimoDecodeAttentionTextNotPinned;
}

/// The geometry the texts are written for; anything else is refused at construction.
pub fn requireGeometry(c: mimo.Config) !void {
    if (c.n_heads != 64 or c.n_kv_heads != 4 or c.swa_n_kv_heads != 8 or c.head_dim != kd or c.value_head_dim != vd or c.window != window)
        return error.MimoDecodeAttentionGeometry;
}

// ---- launch geometry (host arithmetic, CPU-tested against the manifest's launch samples) ----

pub const Launch = struct {
    rg: c_int,
    ks: c_int,
    splits: c_int,
    grid: [3]c_int,
    tg: [3]c_int,
};

/// Donor geometry (`dec_attn.zig` `attention`): rows = G*M/8 row groups; 8 simdgroups while they leave room for key slices.
pub fn splitLaunch(hk: c_int, g: c_int, rows: c_int, keys: c_int) Launch {
    const rg = @divExact(g * rows, 8);
    const ks: c_int = if (rg < 8) @divTrunc(8, rg) else 1;
    const splits: c_int = @max(1, @min(@divTrunc(target_tgs, hk), @divTrunc(keys + bk - 1, bk)));
    return .{ .rg = rg, .ks = ks, .splits = splits, .grid = .{ splits * 32, hk * rg * ks, 1 }, .tg = .{ 32, rg * ks, 1 } };
}

/// One split per KV head. Key slices capped at the tile's 4 sub-tiles (the donor's 8 / RG leaves half idle at RG = 1).
pub fn windowLaunch(hk: c_int, g: c_int, rows: c_int) Launch {
    const rg = @divExact(g * rows, 8);
    const ks: c_int = @min(@divTrunc(bk, 8), if (rg < 8) @divTrunc(8, rg) else 1);
    return .{ .rg = rg, .ks = ks, .splits = 1, .grid = .{ 32, hk * rg * ks, 1 }, .tg = .{ 32, rg * ks, 1 } };
}

/// The window kernel's key bounds for query row `qs` of `rows` over a view of `keys` keys: [lim - (W - 1), lim].
pub fn windowKeys(keys: i64, rows: i64, qs: i64) struct { lo: i64, hi: i64 } {
    const lim = keys - rows + qs;
    return .{ .lo = lim - (window - 1), .hi = lim };
}

/// Transient bytes one call allocates: the split partials (full layers) and the BF16 output.
pub fn workspaceBytes(rows: u64, keys: u64) u64 {
    const r: u64 = 64 * rows;
    const l = splitLaunch(4, 16, @intCast(rows), @intCast(@min(keys, 1 << 30)));
    return r * @as(u64, @intCast(l.splits)) * (vd + 2) * 4 + r * vd * 2;
}

// ---- kernels and routes ----

/// The compiled-on-first-use kernel objects, one set per module; every layer's route borrows them.
pub const Kernels = struct {
    split: Kernel = .{ .ctx = null },
    merge: Kernel = .{ .ctx = null },
    window: Kernel = .{ .ctx = null },
    scl: Array = .{},

    pub fn init(c: mimo.Config) !Kernels {
        try requireGeometry(c);
        var k: Kernels = .{};
        errdefer k.deinit();
        k.split = try makeKernel("msv_dec_attn_split", &.{ "q", "k", "v", "scl" }, &.{ "pacc", "pm", "pl" }, texts.split, texts.header);
        k.merge = try makeKernel("msv_dec_attn_merge", &.{ "pacc", "pm", "pl" }, &.{"out"}, texts.merge, "");
        k.window = try makeKernel("mimo_dec_attn_window", &.{ "q", "k", "v", "sinks", "scl" }, &.{"out"}, texts.window, texts.header);
        const scale = [_]f32{1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)))};
        const one = [_]c_int{1};
        k.scl = mlx.mlx_array_new_data(&scale, &one, 1, .float32);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for ([_]Kernel{ k.split, k.merge, k.window }) |x| if (x.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_free(x);
        };
        drop(k.scl);
        k.* = .{};
    }
};

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !Kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: read through their strides, never copied (ensure_row_contiguous = false).
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, false, false);
    if (k.ctx == null) return error.MimoDecodeAttentionKernel;
    return k;
}

pub const Run = *const fn (r: *const Route, q: Array, k: Array, v: Array, scale: f32, mask: Array, sinks: Array, s: Stream) anyerror!Array;

/// One layer's installed M <= 8 attention. `stock` is the control and the default of a bare `Attention`.
pub const Route = struct {
    run: Run = runStock,
    kernels: ?*const Kernels = null,
    gqa: c_int = 0,
};
pub const stock: Route = .{};

/// Construction: the tier, the layer kind and the served context pick the route once.
pub fn install(tier: Tier, mode: mimo.Attention, served_context: u64, kernels: *const Kernels) Route {
    return switch (tier) {
        .stock => stock,
        .kernels => switch (mode) {
            // A served context under the cutoff never gives one row 8,192 keys: one row is MLX's outright.
            .full => .{ .run = if (served_context < one_row_min_keys) runFullRowsOnly else runFullCutoff, .kernels = kernels, .gqa = 16 },
            .sliding => .{ .run = runWindow, .kernels = kernels, .gqa = 8 },
        },
        .invalid => unreachable, // settings.validate refuses it before construction.
    };
}

/// The control: the unchanged MLX SDPA call with the caller's mask (and the sliding layers' sinks).
fn runStock(_: *const Route, q: Array, k: Array, v: Array, scale: f32, mask: Array, sinks: Array, s: Stream) anyerror!Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "array", mask, sinks, false, s));
    return out;
}

fn runFullCutoff(r: *const Route, q: Array, k: Array, v: Array, scale: f32, mask: Array, sinks: Array, s: Stream) anyerror!Array {
    if (mlx.getShape(q)[2] == 1 and mlx.getShape(k)[2] < one_row_min_keys) return runStock(r, q, k, v, scale, mask, sinks, s);
    return split(r, q, k, v, s);
}

fn runFullRowsOnly(r: *const Route, q: Array, k: Array, v: Array, scale: f32, mask: Array, sinks: Array, s: Stream) anyerror!Array {
    if (mlx.getShape(q)[2] == 1) return runStock(r, q, k, v, scale, mask, sinks, s);
    return split(r, q, k, v, s);
}

fn runWindow(r: *const Route, q: Array, k: Array, v: Array, _: f32, _: Array, sinks: Array, s: Stream) anyerror!Array {
    return windowed(r, q, k, v, sinks, s);
}

/// Full layers: split + merge (donor host code, geometry fixed at install).
pub fn split(r: *const Route, q: Array, k: Array, v: Array, s: Stream) !Array {
    const ks = r.kernels.?;
    const rows = mlx.getShape(q)[2];
    const hk = mlx.getShape(k)[1];
    const hq = hk * r.gqa;
    const parts = try splitPartials(r, q, k, v, s);
    defer _ = mlx.mlx_vector_array_free(parts);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 1, hq, rows, vd }, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, vd, hq * rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, vd, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "BDV", vd));
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, ks.merge, parts, c, s));
    }
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

/// The split dispatch alone: pacc [Hq*M, splits, 128] (unnormalised O), pm / pl [Hq*M, splits] (the slice's running max
/// in log2 units of the scaled score, and its sum). The caller frees the vector.
pub fn splitPartials(r: *const Route, q: Array, k: Array, v: Array, s: Stream) !mlx.mlx_vector_array {
    const ks = r.kernels.?;
    const rows = mlx.getShape(q)[2];
    const hk = mlx.getShape(k)[1];
    const keys = mlx.getShape(k)[2];
    const hq = hk * r.gqa;
    const l = splitLaunch(hk, r.gqa, rows, keys);
    const r_all = hq * rows;
    var parts = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(parts);
    {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ r_all, l.splits, vd }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ r_all, l.splits }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ r_all, l.splits }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
        for ([_]struct { [*:0]const u8, c_int }{ .{ "G", r.gqa }, .{ "S", rows }, .{ "BDK", kd }, .{ "BDV", vd }, .{ "KS", l.ks }, .{ "BK", bk } }) |t|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
        const ins = [_]Array{ q, k, v, ks.scl };
        const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, ks.split, in_vec, c, s));
    }
    return parts;
}

/// The split's key range per slice (the kernel text's `chunk`, `k0`, `k1`): host arithmetic for the diagnosis and the
/// 32K geometry test.
pub fn splitChunk(keys: c_int, splits: c_int) c_int {
    return @divTrunc(@divTrunc(keys + splits - 1, splits) + bk - 1, bk) * bk;
}

/// Sliding layers: one dispatch over the ring view, sinks folded in.
pub fn windowed(r: *const Route, q: Array, k: Array, v: Array, sinks: Array, s: Stream) !Array {
    const ks = r.kernels.?;
    const rows = mlx.getShape(q)[2];
    const hk = mlx.getShape(k)[1];
    const hq = hk * r.gqa;
    const l = windowLaunch(hk, r.gqa, rows);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 1, hq, rows, vd }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    for ([_]struct { [*:0]const u8, c_int }{ .{ "G", r.gqa }, .{ "S", rows }, .{ "BDK", kd }, .{ "BDV", vd }, .{ "KS", l.ks }, .{ "BK", bk }, .{ "W", window } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const ins = [_]Array{ q, k, v, sinks, ks.scl };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, ks.window, in_vec, c, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

// ---- CPU tests (hermetic; no kernel is applied) ----

const testing = std.testing;
const kv = @import("kv.zig");
const settings_mod = @import("settings.zig");

test "mimo decode attention CPU: the manifest and every text match their pins" {
    try checkPins();
    const J = struct {
        headers: []const struct { file: []const u8, sha256: []const u8, bytes: usize },
        kernels: []const struct { name: []const u8, source: struct { file: []const u8, sha256: []const u8, bytes: usize } },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, texts.manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    var seen: usize = 0;
    for (text_pins) |p| {
        for (m.value.headers) |h| if (std.mem.eql(u8, h.file, p.file)) {
            try testing.expectEqualStrings(p.sha256, h.sha256);
            try testing.expectEqual(p.text.len, h.bytes);
            seen += 1;
        };
        for (m.value.kernels) |k| if (std.mem.eql(u8, k.source.file, p.file)) {
            try testing.expectEqualStrings(p.sha256, k.source.sha256);
            try testing.expectEqual(p.text.len, k.source.bytes);
            seen += 1;
        };
    }
    try testing.expectEqual(text_pins.len, seen);
    try testing.expectEqual(@as(usize, 3), m.value.kernels.len);
}

test "mimo decode attention CPU: launch geometry equals the manifest's launch samples and the donor's" {
    const J = struct {
        kernels: []const struct {
            name: []const u8,
            launch_samples: []const struct {
                template: []const struct { name: []const u8, int: ?c_int = null },
                vars: []const struct { name: []const u8, value: c_int },
                grid: [3]c_int,
                threadgroup: [3]c_int,
            },
        },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, texts.manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    var checked: usize = 0;
    for (m.value.kernels) |k| {
        const is_split = std.mem.eql(u8, k.name, "msv_dec_attn_split");
        const is_window = std.mem.eql(u8, k.name, "mimo_dec_attn_window");
        if (!is_split and !is_window) continue;
        for (k.launch_samples) |sample| {
            var rows: c_int = 0;
            var keys: c_int = 0;
            for (sample.vars) |x| {
                if (std.mem.eql(u8, x.name, "S")) rows = x.value;
                if (std.mem.eql(u8, x.name, "N")) keys = x.value;
            }
            const l = if (is_split) splitLaunch(4, 16, rows, keys) else windowLaunch(8, 8, rows);
            try testing.expectEqual(sample.grid, l.grid);
            try testing.expectEqual(sample.threadgroup, l.tg);
            for (sample.template) |t| {
                if (std.mem.eql(u8, t.name, "KS")) try testing.expectEqual(t.int.?, l.ks);
                if (std.mem.eql(u8, t.name, "S")) try testing.expectEqual(t.int.?, rows);
                if (std.mem.eql(u8, t.name, "W")) try testing.expectEqual(t.int.?, window);
            }
            // The threadgroup is whole simdgroups (<= 512 threads), every MMA row is a real (head, position).
            try testing.expect(l.tg[1] * 32 <= 512 and @rem(@as(c_int, if (is_split) 16 else 8) * rows, 8) == 0);
            checked += 1;
        }
    }
    try testing.expectEqual(@as(usize, 5 + 8), checked);
    // The donor's own numbers (CONTRACT.md): 32 splits for 4 KV heads at N >= 1024; S = 8 takes 16 simdgroups.
    try testing.expectEqual(@as(c_int, 32), splitLaunch(4, 16, 1, 1024).splits);
    try testing.expectEqual(@as(c_int, 1), splitLaunch(4, 16, 1, 1).splits);
    try testing.expectEqual(@as(c_int, 5), splitLaunch(4, 16, 1, 129).splits);
    try testing.expectEqual([3]c_int{ 32, 16, 1 }, splitLaunch(4, 16, 8, 16384).tg);
    // Every M of an MTP verify window is a launchable shape on both kinds.
    var m_rows: c_int = 1;
    while (m_rows <= max_rows) : (m_rows += 1) {
        const f = splitLaunch(4, 16, m_rows, 4096);
        const w = windowLaunch(8, 8, m_rows);
        try testing.expect(f.ks >= 1 and f.tg[1] == f.rg * f.ks and f.tg[1] <= 16);
        try testing.expect(w.ks >= 1 and w.ks <= 4 and w.tg[1] == w.rg * w.ks and w.tg[1] <= 8);
    }
}

test "mimo decode attention CPU: window bound arithmetic at positions 127 128 129 656 657 2175" {
    // The ring view at a step of M rows starting at `pos` holds [pos - min(pos, W - 1), pos + M); row qs is the query at
    // pos + qs. The kernel's [lo, hi] (view indices) must be exactly the keys `attends` admits, and nothing else.
    for ([_]u64{ 0, 1, 126, 127, 128, 129, 655, 656, 657, 2175 }) |pos| {
        var rows: u32 = 1;
        while (rows <= max_rows) : (rows += 1) {
            const span = mimo.windowView(pos, rows, window);
            try testing.expectEqual(@as(u64, pos - @min(pos, window - 1)), span.first_key);
            try testing.expectEqual(@as(u32, @intCast(@min(pos, window - 1) + rows)), span.key_count);
            try testing.expect(span.key_count <= window - 1 + max_rows);
            for (0..rows) |qs| {
                const b = windowKeys(span.key_count, rows, @intCast(qs));
                var admitted: u32 = 0;
                for (0..span.key_count) |t| {
                    const kernel = @as(i64, @intCast(t)) >= b.lo and @as(i64, @intCast(t)) <= b.hi;
                    const eager = mimo.attends(pos + qs, span.first_key + t, window);
                    try testing.expectEqual(eager, kernel);
                    admitted += @intFromBool(kernel);
                }
                try testing.expectEqual(@as(u32, @intCast(@min(pos + qs + 1, window))), admitted);
            }
        }
    }
}

test "mimo decode attention CPU: ring views carry head stride 656 x D and 16-byte rows at the named positions" {
    try @import("mtp.zig").requireWindow();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const c: mimo.Config = .{};
    for ([_]u32{ 127, 128, 129, 656, 657, 2175 }) |pos| for ([_]u32{ 1, 8 }) |rows| {
        var ring = try kv.SlidingRing.allocate(c.swa_n_kv_heads, c.head_dim, c.value_head_dim, c.window, s);
        defer ring.deinit();
        var at: u32 = 0;
        while (at < pos + rows) {
            const n = if (at < pos) @min(pos - at, mimo.prefill_chunk) else rows;
            var kz: Array = .{};
            defer drop(kz);
            var vz: Array = .{};
            defer drop(vz);
            try mlx.check(mlx.mlx_zeros(&kz, &.{ 1, @intCast(c.swa_n_kv_heads), @intCast(n), @intCast(c.head_dim) }, 4, .bfloat16, s));
            try mlx.check(mlx.mlx_zeros(&vz, &.{ 1, @intCast(c.swa_n_kv_heads), @intCast(n), @intCast(c.value_head_dim) }, 4, .bfloat16, s));
            var view = try ring.append(kz, vz, n);
            defer view.deinit();
            try mlx.check(mlx.mlx_array_eval(view.k));
            try mlx.check(mlx.mlx_array_eval(view.v));
            at += n;
            if (at != pos + rows) continue;
            const count: c_int = @intCast(@min(pos, c.window - 1) + rows);
            for ([_]Array{ view.k, view.v }, [_]u32{ c.head_dim, c.value_head_dim }) |x, width| {
                try testing.expectEqualSlices(c_int, &.{ 1, @intCast(c.swa_n_kv_heads), count, @intCast(width) }, mlx.getShape(x));
                const st = mlx.mlx_array_strides(x);
                try testing.expectEqual(@as(usize, 656) * width, st[1]);
                try testing.expectEqual(@as(usize, width), st[2]);
                try testing.expectEqual(@as(usize, 1), st[3]);
                // vec<T, 8> row loads: the row stride and the view's first row are 16-byte aligned.
                try testing.expectEqual(@as(usize, 0), (st[2] * 2) % 16);
                const base = @intFromPtr(mlx.mlx_array_data_bfloat16(x).?);
                try testing.expectEqual(@as(usize, 0), base % 16);
            }
        }
    };
}

test "mimo decode attention CPU: split geometry at 16K..32K and 1M keys covers every key once, no 16-bit or 16K limit" {
    // The kernel text's own arithmetic (dec_attn_split.metal): NS = threadgroups_per_grid.x, chunk = ceil(ceil(N/NS)/BK)*BK,
    // slice j reads [j*chunk, min(N, (j+1)*chunk)). Indices are int (keys) and long (key * row stride); nothing is
    // 16-bit but the lane coordinates (< 32), and no buffer is sized by a key count (partials are [rows, splits, 128]).
    const c: mimo.Config = .{};
    for ([_]c_int{ 8192, 16_383, 16_384, 16_385, 20_480, 23_000, 24_576, 32_767, 32_768, 32_776, 49_152, 65_536, 147_456, 278_528, 1 << 20 }) |keys| {
        var m: c_int = 1;
        while (m <= max_rows) : (m += 1) {
            const l = splitLaunch(4, 16, m, keys);
            try testing.expectEqual(@as(c_int, 32), l.splits);
            try testing.expectEqual(l.splits * 32, l.grid[0]);
            const chunk = splitChunk(keys, l.splits);
            try testing.expectEqual(@as(c_int, 0), @rem(chunk, bk));
            // Every key in exactly one slice, every slice inside [0, N).
            var covered: i64 = 0;
            var j: c_int = 0;
            while (j < l.splits) : (j += 1) {
                const k0 = j * chunk;
                const k1 = @min(keys, k0 + chunk);
                if (k1 > k0) covered += k1 - k0;
            }
            try testing.expectEqual(@as(i64, keys), covered);
            // Trailing slices may be empty (16,385 keys: chunk 544, slice 31 starts at 16,864): l = 0, the merge skips them.
            try testing.expect(@as(i64, l.splits) * chunk >= keys);
            // The largest element offset the kernel forms (int keys x long row stride over a 32,768-row slab view).
            const slab_rows: i64 = @max(keys, @as(c_int, 32_768));
            try testing.expect(@as(i64, 3) * slab_rows * c.head_dim < std.math.maxInt(i64));
            try testing.expect(keys <= std.math.maxInt(i32) / 2);
        }
    }
    // The served workspace: partials independent of the key count, to the checkpoint's 1,048,576.
    try testing.expectEqual(workspaceBytes(8, 16_384), workspaceBytes(8, 32_768));
    try testing.expectEqual(workspaceBytes(8, 16_384), workspaceBytes(8, 1 << 20));
    try testing.expect((settings_mod.Config{}).mimo.max_position_embeddings <= max_keys);
}

test "mimo decode attention CPU: routes install once by tier, kind and served context" {
    var k: Kernels = .{};
    try testing.expect(install(.stock, .full, 1 << 20, &k).run == runStock);
    try testing.expect(install(.stock, .sliding, 1 << 20, &k).run == runStock);
    try testing.expect(install(.kernels, .full, one_row_min_keys - 1, &k).run == runFullRowsOnly);
    try testing.expect(install(.kernels, .full, one_row_min_keys, &k).run == runFullCutoff);
    try testing.expect(install(.kernels, .full, 17_408, &k).run == runFullCutoff);
    try testing.expectEqual(@as(c_int, 16), install(.kernels, .full, 17_408, &k).gqa);
    try testing.expect(install(.kernels, .sliding, 17_408, &k).run == runWindow);
    try testing.expectEqual(@as(c_int, 8), install(.kernels, .sliding, 17_408, &k).gqa);
    try testing.expect(stock.run == runStock and stock.kernels == null);
    try testing.expectEqual(Tier.kernels, parseTier("kernels"));
    try testing.expectEqual(Tier.stock, parseTier("stock"));
    try testing.expectEqual(Tier.invalid, parseTier("fast"));
    try requireGeometry(.{});
    var odd: mimo.Config = .{};
    odd.window = 64;
    try testing.expectError(error.MimoDecodeAttentionGeometry, requireGeometry(odd));
}

test "mimo decode attention CPU: the kernels are the default route at every served context to the checkpoint's 1,048,576 keys; stock only when named" {
    // David 2026-10-08 ("typical should be generally usable"): the default tier is the kernels for every request shape
    // the served context admits (greedy or sampled, any M <= 8, any key count up to the context). The one stock M = 1
    // cutoff below `one_row_min_keys` is a route on the runtime key count (bitwise the control there), not a tier.
    const sdk = @import("sdk");
    try testing.expectEqual(Tier.kernels, (settings_mod.Config{}).decode_attention);
    var k: Kernels = .{};
    const max_ctx = (settings_mod.Config{}).mimo.max_position_embeddings;
    try testing.expectEqual(@as(u64, 1_048_576), max_ctx);
    for ([_]u64{ 1, 4_096, one_row_min_keys - 1, one_row_min_keys, 17_408, 24_576, 32_768, 33_792, 49_152, 81_920, 131_072, 147_456, 278_528, 1_047_488, max_ctx }) |context| {
        const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10_000_000_000, .wired_margin_bytes = 2_000_000_000, .effective_context = context };
        const c = (settings_mod.Config{}).withFacts(&facts);
        try c.validate();
        try testing.expectEqual(Tier.kernels, c.decodeAttentionTier());
        const full = install(c.decodeAttentionTier(), .full, c.reserveTokens(), &k);
        const sliding = install(c.decodeAttentionTier(), .sliding, c.reserveTokens(), &k);
        try testing.expect(full.run != runStock and full.kernels != null);
        try testing.expect(sliding.run == runWindow and sliding.kernels != null);
        const want: Run = if (context < one_row_min_keys) runFullRowsOnly else runFullCutoff;
        try testing.expect(full.run == want);
        // The stock control is a named construction-time route at the same contexts.
        var control: settings_mod.Config = .{ .decode_attention = .stock };
        control = control.withFacts(&facts);
        try control.validate();
        try testing.expect(install(control.decodeAttentionTier(), .full, control.reserveTokens(), &k).run == runStock);
    }
    // The host's default reservation (no effective context: 16K prompt + 1K) is on the kernels too.
    try (settings_mod.Config{}).validate();
    try testing.expectEqual(Tier.kernels, (settings_mod.Config{}).decodeAttentionTier());
    // The texts' index range is the one bound, and it refuses at construction past the checkpoint's context only.
    try requireKeyRange(max_ctx);
    try requireKeyRange(max_keys);
    try testing.expectError(error.MimoDecodeAttentionKeyRange, requireKeyRange(max_keys + 1));
}

test "mimo decode attention CPU: the M <= 8 workspace fits the request's transient terms it replaces" {
    // The kernel route allocates partials + output only; the bill's request carries the unfused M <= 8 SDPA scores
    // (3 x 64 x 8 x reserve x 4 B) and the chunk workspace, which this phase no longer reaches.
    const c: mimo.Config = .{};
    for ([_]u64{ 8, 9, 512, 8192, 17_408, 32_768, 1_048_576 }) |reserve| {
        var rows: u64 = 1;
        while (rows <= max_rows) : (rows += 1)
            try testing.expect(workspaceBytes(rows, @min(reserve, 1 << 20)) <= c.requestBytes(reserve) - c.kvBytes(reserve));
    }
    try testing.expectEqual(@as(u64, 64 * 32 * 130 * 4 + 64 * 128 * 2), workspaceBytes(1, 16_384));
}

// ---- device self-check (pre-ship gate; env + lock gated) ----

fn nextBits(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

fn bf16Bits(value: f32) u16 {
    const word: u32 = @bitCast(value);
    return @intCast((word + 0x7fff + ((word >> 16) & 1)) >> 16);
}

fn bf16Value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Host BF16 values in [-amp, amp), and the device array holding the same bits.
const HostArray = struct {
    bits: []u16,
    arr: Array,
    fn init(a: std.mem.Allocator, shape: []const c_int, seed: u32, amp: f32) !HostArray {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        const bits = try a.alloc(u16, n);
        errdefer a.free(bits);
        var st = seed;
        for (bits) |*b| {
            const u = @as(f32, @floatFromInt(nextBits(&st) >> 8)) / @as(f32, 1 << 24);
            b.* = bf16Bits((2 * u - 1) * amp);
        }
        return .{ .bits = bits, .arr = mlx.mlx_array_new_data(bits.ptr, shape.ptr, @intCast(shape.len), .bfloat16) };
    }
    fn deinit(h: HostArray, a: std.mem.Allocator) void {
        a.free(h.bits);
        drop(h.arr);
    }
    fn at(h: HostArray, i: usize) f32 {
        return bf16Value(h.bits[i]);
    }
};

const Case = struct {
    name: []const u8,
    mode: mimo.Attention,
    hk: c_int,
    rows: c_int,
    /// Full: the key count of the [0, keys) slab view. Sliding: the position of the step's first row.
    keys_or_pos: u32,
    sink_kind: enum { none, distinct, dominant } = .none,
    /// Full: call the split kernel directly (below the route's one-row cutoff too).
    direct: bool = false,
    /// Full: a long-range retrieval probe. Key `37 + 4096 j` of KV head j carries 3x the first query row of its group's
    /// first head, so that row's attention sits on one key tens of thousands of positions back (split 0 of the merge).
    needle: bool = false,
};

const cases = [_]Case{
    .{ .name = "full_1row_8191_routed_stock", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 8191 },
    .{ .name = "full_1row_8192", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 8192 },
    .{ .name = "full_1row_16384", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 16384 },
    .{ .name = "full_1row_16385", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 16385 },
    .{ .name = "full_1row_129_direct", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 129, .direct = true },
    .{ .name = "full_1row_1_direct", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 1, .direct = true },
    .{ .name = "full_verify4_3000", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 3000 },
    .{ .name = "full_verify8_16384", .mode = .full, .hk = 4, .rows = 8, .keys_or_pos = 16384 },
    .{ .name = "sliding_pos0", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 0, .sink_kind = .distinct },
    .{ .name = "sliding_pos126", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 126, .sink_kind = .distinct },
    .{ .name = "sliding_pos127", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 127, .sink_kind = .distinct },
    .{ .name = "sliding_pos128", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 128, .sink_kind = .distinct },
    .{ .name = "sliding_pos129_dominant_sink", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 129, .sink_kind = .dominant },
    .{ .name = "sliding_pos2175", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 2175, .sink_kind = .distinct },
    .{ .name = "sliding_verify4_pos125", .mode = .sliding, .hk = 8, .rows = 4, .keys_or_pos = 125, .sink_kind = .distinct },
    .{ .name = "sliding_verify8_pos2175", .mode = .sliding, .hk = 8, .rows = 8, .keys_or_pos = 2175, .sink_kind = .distinct },
};

const Errors = struct { max_abs: f64 = 0, dot: f64 = 0, aa: f64 = 0, bb: f64 = 0 };

fn compare(e: *Errors, got: f64, want: f64) void {
    e.max_abs = @max(e.max_abs, @abs(got - want));
    e.dot += got * want;
    e.aa += got * got;
    e.bb += want * want;
}

fn hostF32(a: std.mem.Allocator, x: Array, s: Stream) ![]f32 {
    var f: Array = .{};
    defer drop(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const out = try a.alloc(f32, n);
    @memcpy(out, (mlx.mlx_array_data_float32(f) orelse return error.MimoDecodeAttentionData)[0..n]);
    return out;
}

fn runCase(a: std.mem.Allocator, kernels: *const Kernels, cs: Case, s: Stream) !void {
    const g: c_int = if (cs.mode == .full) 16 else 8;
    const hq = cs.hk * g;
    const rows: u32 = @intCast(cs.rows);
    // The views as the caches hand them over: a slice of a larger slab (head stride = slab rows x D).
    var slab_rows: c_int = @intCast(cs.keys_or_pos + 64);
    var first: c_int = 0;
    var keys: c_int = @intCast(cs.keys_or_pos);
    if (cs.mode == .sliding) {
        const span = mimo.windowView(cs.keys_or_pos, rows, 128);
        slab_rows = 656;
        first = @intCast(@min(span.first_key, 656 - span.key_count));
        keys = @intCast(span.key_count);
    }
    const qh = try HostArray.init(a, &.{ 1, hq, cs.rows, kd }, 0x732eb179 +% cs.keys_or_pos, 1);
    defer qh.deinit(a);
    var kh = try HostArray.init(a, &.{ 1, cs.hk, slab_rows, kd }, 0x61f953c7 +% cs.keys_or_pos, 1);
    defer kh.deinit(a);
    if (cs.needle) {
        for (0..@intCast(cs.hk)) |j| {
            const t = 37 + 4096 * j;
            const kb = (j * @as(usize, @intCast(slab_rows)) + t) * kd;
            const qb = (j * @as(usize, @intCast(g)) * rows) * kd;
            for (0..kd) |d| kh.bits[kb + d] = bf16Bits(3 * qh.at(qb + d));
        }
        drop(kh.arr);
        kh.arr = mlx.mlx_array_new_data(kh.bits.ptr, &[_]c_int{ 1, cs.hk, slab_rows, kd }, 4, .bfloat16);
    }
    const vh = try HostArray.init(a, &.{ 1, cs.hk, slab_rows, vd }, 0xa18de563 +% cs.keys_or_pos, 0.707);
    defer vh.deinit(a);
    var k: Array = .{};
    defer drop(k);
    var v: Array = .{};
    defer drop(v);
    try mlx.check(mlx.mlx_slice(&k, kh.arr, &.{ 0, 0, first, 0 }, 4, &.{ 1, cs.hk, first + keys, kd }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    try mlx.check(mlx.mlx_slice(&v, vh.arr, &.{ 0, 0, first, 0 }, 4, &.{ 1, cs.hk, first + keys, vd }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    var sink_vals: [64]f32 = @splat(0);
    var sinks: Array = .{};
    defer drop(sinks);
    if (cs.sink_kind != .none) {
        var bits: [64]u16 = undefined;
        for (&bits, 0..) |*b, h| {
            const x: f32 = if (cs.sink_kind == .dominant) 12 else 2 * @as(f32, @floatFromInt(h % 5)) - 2 + 0.25 * @as(f32, @floatFromInt(h % 3));
            b.* = bf16Bits(x);
            sink_vals[h] = bf16Value(b.*);
        }
        sinks = mlx.mlx_array_new_data(&bits, &[_]c_int{hq}, 1, .bfloat16);
    }
    // The stock call the control tier makes, with the caller's exact bool mask.
    const mask_vals = try a.alloc(bool, rows * @as(usize, @intCast(keys)));
    defer a.free(mask_vals);
    for (0..rows) |qs| for (0..@intCast(keys)) |t| {
        const b = windowKeys(keys, rows, @intCast(qs));
        const ti: i64 = @intCast(t);
        mask_vals[qs * @as(usize, @intCast(keys)) + t] = ti <= b.hi and (cs.mode == .full or ti >= b.lo);
    };
    const mask = mlx.mlx_array_new_data(mask_vals.ptr, &[_]c_int{ cs.rows, keys }, 2, .bool_);
    defer drop(mask);
    const scale = 1 / @sqrt(@as(f32, kd));
    const route = install(.kernels, cs.mode, 1 << 20, kernels);
    const ours = if (cs.direct) try split(&route, qh.arr, k, v, s) else try route.run(&route, qh.arr, k, v, scale, mask, sinks, s);
    defer drop(ours);
    var stock_out: Array = .{};
    defer drop(stock_out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&stock_out, qh.arr, k, v, scale, "array", mask, sinks, false, s));
    try testing.expectEqualSlices(c_int, &.{ 1, hq, cs.rows, vd }, mlx.getShape(ours));
    const got = try hostF32(a, ours, s);
    defer a.free(got);
    const ref = try hostF32(a, stock_out, s);
    defer a.free(ref);
    // f64 truth from the BF16 operands, and the Xiaomi eager oracle (checkAttentionMode's arithmetic: BF16 QK, scale,
    // FP32 softmax, BF16 probabilities, BF16 AV).
    var e_ours: Errors = .{};
    var e_stock: Errors = .{};
    var e_eager: Errors = .{};
    var e_sdpa: Errors = .{};
    // Per query head: max |ours - SDPA| (the first head over the bound localizes a failure).
    var head_sdpa: [64]f64 = @splat(0);
    const nk: usize = @intCast(keys);
    const scores = try a.alloc(f64, nk);
    defer a.free(scores);
    const eager_p = try a.alloc(f32, nk);
    defer a.free(eager_p);
    for (0..@intCast(hq)) |h| for (0..rows) |qs| {
        const hk_i = h / @as(usize, @intCast(g));
        const b = windowKeys(keys, rows, @intCast(qs));
        const lo: usize = if (cs.mode == .full) 0 else @intCast(@max(0, b.lo));
        const hi: usize = @intCast(b.hi);
        var peak: f64 = if (cs.sink_kind != .none) sink_vals[h] else -std.math.inf(f64);
        var epeak: f32 = if (cs.sink_kind != .none) sink_vals[h] else -std.math.inf(f32);
        for (lo..hi + 1) |t| {
            var dot: f64 = 0;
            var dotf: f32 = 0;
            const kb = (hk_i * @as(usize, @intCast(slab_rows)) + @as(usize, @intCast(first)) + t) * kd;
            const qb = (h * rows + qs) * kd;
            for (0..kd) |d| {
                dot += @as(f64, qh.at(qb + d)) * kh.at(kb + d);
                dotf += qh.at(qb + d) * kh.at(kb + d);
            }
            scores[t] = dot * scale;
            peak = @max(peak, scores[t]);
            eager_p[t] = bf16Value(bf16Bits(bf16Value(bf16Bits(dotf)) * scale));
            epeak = @max(epeak, eager_p[t]);
        }
        var z: f64 = if (cs.sink_kind != .none) @exp(sink_vals[h] - peak) else 0;
        var ez: f32 = if (cs.sink_kind != .none) @exp(bf16Value(bf16Bits(sink_vals[h] - epeak))) else 0;
        for (lo..hi + 1) |t| {
            scores[t] = @exp(scores[t] - peak);
            z += scores[t];
            eager_p[t] = @exp(bf16Value(bf16Bits(eager_p[t] - epeak)));
            ez += eager_p[t];
        }
        for (lo..hi + 1) |t| eager_p[t] = bf16Value(bf16Bits(eager_p[t] / ez));
        for (0..vd) |d| {
            var acc: f64 = 0;
            var eacc: f32 = 0;
            for (lo..hi + 1) |t| {
                const vv = vh.at((hk_i * @as(usize, @intCast(slab_rows)) + @as(usize, @intCast(first)) + t) * vd + d);
                acc += scores[t] * vv;
                eacc += eager_p[t] * vv;
            }
            const truth = acc / z;
            const o = (h * rows + qs) * vd + d;
            compare(&e_ours, got[o], truth);
            compare(&e_stock, ref[o], truth);
            compare(&e_eager, got[o], bf16Value(bf16Bits(eacc)));
            compare(&e_sdpa, got[o], ref[o]);
            head_sdpa[h] = @max(head_sdpa[h], @abs(@as(f64, got[o]) - ref[o]));
        }
    };
    var first_bad_head: ?usize = null;
    var worst_head: usize = 0;
    for (head_sdpa[0..@intCast(hq)], 0..) |e, h| {
        if (e > head_sdpa[worst_head]) worst_head = h;
        if (first_bad_head == null and !(e <= 0.015625)) first_bad_head = h;
    }
    if (first_bad_head) |h| {
        std.debug.print("MIMO_DEC_ATTN case={s} FIRST HEAD over the SDPA bound: head={d} (kv head {d}) max_abs={d:.7}\n", .{ cs.name, h, h / @as(usize, @intCast(g)), head_sdpa[h] });
    } else std.debug.print("MIMO_DEC_ATTN case={s} per-head vs SDPA: worst head={d} max_abs={d:.7}, no head over 0.015625\n", .{ cs.name, worst_head, head_sdpa[worst_head] });
    const cosine = e_ours.dot / @sqrt(e_ours.aa * e_ours.bb);
    const cos_sdpa = e_sdpa.dot / @sqrt(e_sdpa.aa * e_sdpa.bb);
    const bound = 2 * e_stock.max_abs + 1e-3;
    std.debug.print("MIMO_DEC_ATTN case={s} keys={d} rows={d} ours_vs_f64={d:.7} stock_vs_f64={d:.7} bound={d:.7} cosine={d:.9} vs_eager={d:.7} eager_tol=0.015625 vs_sdpa={d:.7} cos_sdpa={d:.9} out_rms={d:.6}\n", .{ cs.name, keys, rows, e_ours.max_abs, e_stock.max_abs, bound, cosine, e_eager.max_abs, e_sdpa.max_abs, cos_sdpa, @sqrt(e_ours.bb / @as(f64, @floatFromInt(got.len))) });
    const failed = !std.math.isFinite(e_ours.max_abs) or e_ours.max_abs > bound or cosine < 0.99999 or e_eager.max_abs > 0.015625 or
        // MLX's own SDPA on the same BF16 views: the same BF16-output tolerance and cosine.
        !std.math.isFinite(e_sdpa.max_abs) or e_sdpa.max_abs > 0.015625 or cos_sdpa < 0.99999;
    // Full layers: the split's partials slice by slice against f64 (every one-row case, and any failing case), which names
    // the first head / key block (slice) that goes wrong, or clears the slices and points at the merge.
    if (cs.mode == .full and (rows == 1 or failed)) try diagnoseSplits(a, kernels, cs.name, qh, kh, vh, qh.arr, k, v, slab_rows, keys, rows, g, scale, got, s);
    if (failed) return error.MimoDecodeAttentionSelfCheck;
    // Below the cutoff the installed route is MLX's own call: bitwise the control.
    if (cs.mode == .full and !cs.direct and cs.rows == 1 and cs.keys_or_pos < one_row_min_keys)
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref), std.mem.sliceAsBytes(got));
}

/// The split kernel's partials, slice by slice, against f64 on the same BF16 operands: the normalised partial
/// O_j = pacc / pl and the slice's log-sum-exp (pm ln2 + ln pl) vs the host's over the slice's keys [k0, min(k1, lim + 1)),
/// then the host merge of the kernel's partials vs the merged BF16 output. Prints the first (head, row, slice, key range)
/// over the bound, the worst slice, and errors out on any.
fn diagnoseSplits(a: std.mem.Allocator, kernels: *const Kernels, name: []const u8, qh: HostArray, kh: HostArray, vh: HostArray, q: Array, k: Array, v: Array, slab_rows: c_int, keys: c_int, rows: u32, g: c_int, scale: f32, got: []const f32, s: Stream) !void {
    const route = install(.kernels, .full, 1 << 20, kernels);
    const parts = try splitPartials(&route, q, k, v, s);
    defer _ = mlx.mlx_vector_array_free(parts);
    var arrs: [3]Array = @splat(.{});
    defer for (arrs) |x| drop(x);
    for (&arrs, 0..) |*x, i| try mlx.check(mlx.mlx_vector_array_get(x, parts, i));
    const pacc = try hostF32(a, arrs[0], s);
    defer a.free(pacc);
    const pm = try hostF32(a, arrs[1], s);
    defer a.free(pm);
    const pl = try hostF32(a, arrs[2], s);
    defer a.free(pl);
    const hk = mlx.getShape(k)[1];
    const hq: usize = @intCast(hk * g);
    const l = splitLaunch(hk, g, @intCast(rows), keys);
    const ns: usize = @intCast(l.splits);
    const chunk = splitChunk(keys, l.splits);
    const tol: f64 = 1e-3;
    var part_max: f64 = 0;
    var lse_max: f64 = 0;
    var merge_max: f64 = 0;
    var worst: struct { h: usize = 0, j: usize = 0, e: f64 = 0 } = .{};
    var first_bad: ?struct { h: usize, qs: usize, j: usize, k0: c_int, k1: c_int, e_o: f64, e_lse: f64 } = null;
    var empty_bad: usize = 0;
    const sc = try a.alloc(f64, @intCast(chunk));
    defer a.free(sc);
    var o: [vd]f64 = undefined;
    for (0..hq) |h| for (0..rows) |qs| {
        const kvh = h / @as(usize, @intCast(g));
        const lim: c_int = keys - @as(c_int, @intCast(rows)) + @as(c_int, @intCast(qs));
        const row = h * rows + qs;
        for (0..ns) |j| {
            const k0: c_int = @as(c_int, @intCast(j)) * chunk;
            const k1: c_int = @min(keys, k0 + chunk);
            const hi: c_int = @min(k1 - 1, lim);
            const pb = row * ns + j;
            if (hi < k0) {
                if (pl[pb] != 0) empty_bad += 1;
                continue;
            }
            var mx: f64 = -std.math.inf(f64);
            for (@intCast(k0)..@as(usize, @intCast(hi)) + 1) |t| {
                var dot: f64 = 0;
                const kb = (kvh * @as(usize, @intCast(slab_rows)) + t) * kd;
                const qb = row * kd;
                for (0..kd) |d| dot += @as(f64, qh.at(qb + d)) * kh.at(kb + d);
                sc[t - @as(usize, @intCast(k0))] = dot * scale;
                mx = @max(mx, dot * scale);
            }
            var z: f64 = 0;
            @memset(&o, 0);
            for (@intCast(k0)..@as(usize, @intCast(hi)) + 1) |t| {
                const w = @exp(sc[t - @as(usize, @intCast(k0))] - mx);
                z += w;
                const vb = (kvh * @as(usize, @intCast(slab_rows)) + t) * vd;
                for (0..vd) |d| o[d] += w * vh.at(vb + d);
            }
            var e_o: f64 = 0;
            for (0..vd) |d| e_o = @max(e_o, @abs(@as(f64, pacc[pb * vd + d]) / pl[pb] - o[d] / z));
            const e_lse = @abs(@as(f64, pm[pb]) * std.math.ln2 + @log(@as(f64, pl[pb])) - (mx + @log(z)));
            if (!std.math.isFinite(e_o)) part_max = std.math.inf(f64) else part_max = @max(part_max, e_o);
            if (!std.math.isFinite(e_lse)) lse_max = std.math.inf(f64) else lse_max = @max(lse_max, e_lse);
            if (!(e_o <= worst.e)) worst = .{ .h = h, .j = j, .e = e_o };
            if (first_bad == null and !(e_o <= tol and e_lse <= tol)) first_bad = .{ .h = h, .qs = qs, .j = j, .k0 = k0, .k1 = k1, .e_o = e_o, .e_lse = e_lse };
        }
        // The host merge of the kernel's own partials vs the merged BF16 output (the merge kernel alone).
        var m_all: f64 = -std.math.inf(f64);
        for (0..ns) |j| if (pl[row * ns + j] > 0) {
            m_all = @max(m_all, pm[row * ns + j]);
        };
        var zz: f64 = 0;
        @memset(&o, 0);
        for (0..ns) |j| if (pl[row * ns + j] > 0) {
            const e = std.math.pow(f64, 2, pm[row * ns + j] - m_all);
            zz += pl[row * ns + j] * e;
            for (0..vd) |d| o[d] += e * pacc[(row * ns + j) * vd + d];
        };
        for (0..vd) |d| merge_max = @max(merge_max, @abs(got[row * vd + d] - o[d] / zz));
    };
    std.debug.print("MIMO_DEC_ATTN_SPLITS case={s} keys={d} rows={d} splits={d} chunk={d} partial_vs_f64={d:.8} lse_vs_f64={d:.8} merge_vs_host_merge={d:.7} worst=(head {d}, slice {d} keys [{d}, {d})) empty_slices_nonzero={d}", .{ name, keys, rows, ns, chunk, part_max, lse_max, merge_max, worst.h, worst.j, @as(c_int, @intCast(worst.j)) * chunk, @min(keys, @as(c_int, @intCast(worst.j + 1)) * chunk), empty_bad });
    if (first_bad) |b| {
        std.debug.print(" FIRST BAD: head={d} (kv head {d}) row={d} slice={d} keys=[{d}, {d}) partial={d:.7} lse={d:.7} tol={d}\n", .{ b.h, b.h / @as(usize, @intCast(g)), b.qs, b.j, b.k0, b.k1, b.e_o, b.e_lse, tol });
    } else std.debug.print(" no slice over tol={d}\n", .{tol});
    // Merge: the BF16 rounding of |O| < 2 is at most 2^-8; 2^-7 admits one more ulp of f32 order.
    if (first_bad != null or empty_bad != 0 or !(merge_max <= 1.0 / 128.0)) return error.MimoDecodeAttentionSplits;
}

/// Static bound for the largest case (full, 16,448-row slab, 8 rows): host + device operands, f32 copies, partials.
fn selfCheckBound() u64 {
    const slab: u64 = 4 * (16_384 + 64) * (kd + vd);
    const device = 2 * slab + 2 * 64 * 8 * kd + 64 * 8 * 32 * (vd + 2) * 4 + 4 * 64 * 8 * vd * 4 + 8 * 16_384;
    return 2 * device + (256 << 20) + (512 << 20);
}

/// The served range past 16,385 keys (the 16K-prompt x 16K-generation cell reaches 32,768 keys; a retrieval failure
/// showed from ~23,000): full layers at 20,480 / 24,576 / 32,768 keys, M = 1 / 4 / 8, two needle probes, and sliding
/// layers at positions 20,480 and 32,767 (the ring has wrapped ~250 times), M = 1 / 4.
const cases_32k = [_]Case{
    .{ .name = "full_1row_16384", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 16_384 },
    .{ .name = "full_1row_16385", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 16_385 },
    .{ .name = "full_1row_20480", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 20_480 },
    .{ .name = "full_verify4_20480", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 20_480 },
    .{ .name = "full_verify8_20480", .mode = .full, .hk = 4, .rows = 8, .keys_or_pos = 20_480 },
    .{ .name = "full_1row_24576", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 24_576 },
    .{ .name = "full_verify4_24576", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 24_576 },
    .{ .name = "full_verify8_24576", .mode = .full, .hk = 4, .rows = 8, .keys_or_pos = 24_576 },
    .{ .name = "full_1row_32767", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 32_767 },
    .{ .name = "full_1row_32768", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 32_768 },
    .{ .name = "full_verify4_32768", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 32_768 },
    .{ .name = "full_verify8_32768", .mode = .full, .hk = 4, .rows = 8, .keys_or_pos = 32_768 },
    .{ .name = "full_1row_32768_needle", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 32_768, .needle = true },
    .{ .name = "full_verify4_32768_needle", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 32_768, .needle = true },
    .{ .name = "sliding_pos16385", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 16_385, .sink_kind = .distinct },
    .{ .name = "sliding_pos24576", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 24_576, .sink_kind = .distinct },
    .{ .name = "sliding_pos20480", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 20_480, .sink_kind = .distinct },
    .{ .name = "sliding_verify4_pos20480", .mode = .sliding, .hk = 8, .rows = 4, .keys_or_pos = 20_480, .sink_kind = .distinct },
    .{ .name = "sliding_pos32767", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 32_767, .sink_kind = .distinct },
    .{ .name = "sliding_verify4_pos32767", .mode = .sliding, .hk = 8, .rows = 4, .keys_or_pos = 32_767, .sink_kind = .distinct },
};

/// Static bound for the largest 32K case (full, 32,832-row slab, 8 rows): host + device BF16 operands, f32 copies,
/// partials, the bool mask, and MLX SDPA's unfused f32 scores + probabilities if it takes that route (64 x 8 x 32,768 x 4 B
/// each), plus the 256 MiB cache and 512 MiB compiler allowances. One case at a time (each frees before the next).
pub fn selfCheckBound32k() u64 {
    const slab: u64 = 4 * (32_768 + 64) * (kd + vd) * 2;
    const scores: u64 = 2 * 64 * 8 * 32_768 * 4;
    const device = slab + 2 * 64 * 8 * kd + 64 * 8 * 32 * (vd + 2) * 4 + 4 * 64 * 8 * vd * 4 + 8 * 32_768 + scores;
    return 2 * device + (256 << 20) + (512 << 20);
}

test "mimo GPU32K decode attention: split-K full to 32,768 keys and windowed sliding to position 32,767 vs eager, f64 truth and MLX SDPA" {
    const enabled = std.c.getenv("MIMO_DEC_ATTN_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ceiling: u64 = 2 << 30;
    const bound = selfCheckBound32k();
    std.debug.print("MIMO_DEC_ATTN32K static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoDecodeAttentionAdmission;
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
    var kernels = try Kernels.init(.{});
    defer kernels.deinit();
    for (cases_32k) |cs| try runCase(testing.allocator, &kernels, cs, s);
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_DEC_ATTN32K peak: mlx={d} bound={d}\n", .{ peak, bound });
    if (peak > bound) return error.MimoDecodeAttentionPeakExceeded;
}

/// The served range past 32,768 keys to the checkpoint's 1,048,576 (the card ladder's 49,152 / 81,920 / 147,456 /
/// 278,528 contexts and the top): full layers M = 1 / 4 / 8 and far needles (key 37 + 4,096 j, ~1M positions back at
/// the top), sliding layers at the matching positions (the ring has wrapped up to ~8,000 times). One case at a time.
const cases_long = [_]Case{
    .{ .name = "full_1row_49152", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 49_152 },
    .{ .name = "full_verify8_49152", .mode = .full, .hk = 4, .rows = 8, .keys_or_pos = 49_152 },
    .{ .name = "full_1row_147456", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 147_456 },
    .{ .name = "full_verify4_147456", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 147_456 },
    .{ .name = "full_1row_278528", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 278_528 },
    .{ .name = "full_verify8_278528", .mode = .full, .hk = 4, .rows = 8, .keys_or_pos = 278_528 },
    .{ .name = "full_1row_278528_needle", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 278_528, .needle = true },
    .{ .name = "full_1row_1048576", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 1_048_576 },
    .{ .name = "full_verify4_1048576", .mode = .full, .hk = 4, .rows = 4, .keys_or_pos = 1_048_576 },
    .{ .name = "full_1row_1048576_needle", .mode = .full, .hk = 4, .rows = 1, .keys_or_pos = 1_048_576, .needle = true },
    .{ .name = "sliding_pos49151", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 49_151, .sink_kind = .distinct },
    .{ .name = "sliding_verify4_pos278527", .mode = .sliding, .hk = 8, .rows = 4, .keys_or_pos = 278_527, .sink_kind = .distinct },
    .{ .name = "sliding_pos1048575", .mode = .sliding, .hk = 8, .rows = 1, .keys_or_pos = 1_048_575, .sink_kind = .distinct },
    .{ .name = "sliding_verify8_pos1048568", .mode = .sliding, .hk = 8, .rows = 8, .keys_or_pos = 1_048_568, .sink_kind = .distinct },
};

/// Static bound of one long case (selfCheckBound32k's terms at the case's key count and rows): the device BF16 slab, MLX
/// SDPA's unfused f32 scores + probabilities if it takes that route, partials, outputs, mask, each twice, plus the
/// 256 MiB cache and 512 MiB compiler allowances. The test's bound is the widest case's.
pub fn selfCheckBoundCase(cs: Case) u64 {
    const keys: u64 = if (cs.mode == .full) cs.keys_or_pos else 656;
    const rows: u64 = @intCast(cs.rows);
    const slab: u64 = 4 * (keys + 64) * (kd + vd) * 2;
    const scores: u64 = 2 * 64 * rows * keys * 4;
    const device = slab + 2 * 64 * rows * kd + 64 * rows * 32 * (vd + 2) * 4 + 4 * 64 * rows * vd * 4 + rows * keys + scores;
    return 2 * device + (256 << 20) + (512 << 20);
}

pub fn selfCheckBoundLong() u64 {
    var bound: u64 = 0;
    for (cases_long) |cs| bound = @max(bound, selfCheckBoundCase(cs));
    return bound;
}

test "mimo GPULONG decode attention: split-K full to 1,048,576 keys and windowed sliding to position 1,048,575 vs eager, f64 truth and MLX SDPA" {
    const enabled = std.c.getenv("MIMO_DEC_ATTN_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ceiling: u64 = 12 << 30;
    const bound = selfCheckBoundLong();
    std.debug.print("MIMO_DEC_ATTN_LONG static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoDecodeAttentionAdmission;
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
    var kernels = try Kernels.init(.{});
    defer kernels.deinit();
    for (cases_long) |cs| {
        try runCase(testing.allocator, &kernels, cs, s);
        try mlx.check(mlx.mlx_synchronize(s));
        try mlx.check(mlx.mlx_clear_cache());
    }
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_DEC_ATTN_LONG peak: mlx={d} bound={d}\n", .{ peak, bound });
    if (peak > bound) return error.MimoDecodeAttentionPeakExceeded;
}

test "mimo decode attention CPU: the long self-check's cases fit its declared ceiling and reach the checkpoint's context" {
    try testing.expect(selfCheckBoundLong() <= 12 << 30);
    var widest: u32 = 0;
    for (cases_long) |cs| if (cs.mode == .full) {
        widest = @max(widest, cs.keys_or_pos);
    };
    try testing.expectEqual(@as(u32, @intCast((settings_mod.Config{}).mimo.max_position_embeddings)), widest);
}

test "mimo GPU decode attention: split-K full and windowed sliding agree with the eager oracle and f64 truth" {
    const enabled = std.c.getenv("MIMO_DEC_ATTN_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ceiling: u64 = 2 << 30;
    const bound = selfCheckBound();
    std.debug.print("MIMO_DEC_ATTN static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoDecodeAttentionAdmission;
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
    var kernels = try Kernels.init(.{});
    defer kernels.deinit();
    for (cases) |cs| try runCase(testing.allocator, &kernels, cs, s);
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_DEC_ATTN peak: mlx={d} bound={d}\n", .{ peak, bound });
    if (peak > bound) return error.MimoDecodeAttentionPeakExceeded;
}
