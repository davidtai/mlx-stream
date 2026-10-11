//! `+attn` (design-prefill §3b, rounding-class): MiMo's prompt attention through the NAX (tensor unit) 192/128 sink
//! kernel `msv_nax_attn_192_128` (donor: reports/.../mimo/donors/nax_attention, CONTRACT.md; mlx-serve origin/mimo-v2
//! @ 7a9e392f `src/nax_attention.zig`, from oMLX @ 5dcfe243, itself MLX's steel_attention_nax). BF16 views read
//! through their strides, the window and sinks per layer kind (global: causal, no sinks; sliding: band 128 + sinks),
//! no mask array and no pad to 256. Output is physically [1, L, 64, 128], which o_proj reads as [L, 64*128] with no
//! transpose copy. Past 12,256 keys (384 key blocks) a 512-row query tile runs as key-range passes that hand on 17.0 MB of fp32 row
//! state (`passStateBytes`, billed by bill.zig).
//!
//! The route is a construction-time choice (`Route`, settings `prefill_attention`): `.nax` on a NAX device unless the
//! stock route is asked for; the stock route (`+mask`'s padded fused SDPA) is the explicit control. Nothing in the hot
//! path checks eligibility or falls back. The span rule (`attentionSpanRows`, design §4) is read by module and bill.
const std = @import("std");
const mlx = @import("sdk").mlx;
const mimo = @import("config.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

pub const Route = enum { nax, stock };

// ---- kernel texts and their pins (kernels/manifest-prefill-attn.json; byte-identical to the donor) ----
pub const tiles_text = @embedFile("kernels/nax_attn_tiles.metal");
pub const core_text = @embedFile("kernels/nax_attn_core.metal");
pub const source_text = @embedFile("kernels/nax_attn_192_128.metal");
pub const manifest_text = @embedFile("kernels/manifest-prefill-attn.json");
pub const pins = struct {
    pub const tiles = "b510169e09786999553106f8f13ea7ef89212110894a0eda0ae8849bdf3d1a21";
    pub const core = "20241c4eab1c666aa4bf33d4fadb6fd4c88ca1f251b81cfa2e401ad6c8bee17f";
    pub const source = "8d2cb4b4a2fa58f217890727a74599ca3d5113eaf6ccedeb2e17ad35fa7bfc43";
    pub const manifest = "f1e4d53b1ea3cda0ae2c56f56e8d46adc2f89ed19443a01ec6b6030bfaedec35";
};
/// The header MLX compiles: nax_tiles ++ nax_attention (no separator; the tiles text ends in a newline).
const header_text = tiles_text ++ core_text;

// ---- geometry (the kernel is fixed at BD 192 / BDV 128; MiMo's 64 query heads) ----
const heads: u64 = 64;
const head_dim: u64 = 192;
const value_dim: u64 = 128;
const BK: c_int = 32;
const threads: c_int = 128;
/// Keys per pass and the grid from which passes pay (oMLX's measured defaults, unchanged).
const pass_keys: c_int = 8192;
const pass_min_groups: c_int = 512;
pub const max_passes = 64;
/// A dispatch's query rows: the prompt tile. Its work (512 x keys x 64 heads, split into ~8,192-key passes past
/// 12,288 keys) stays far below the IOGPU preemption cliff upstream measured (~1.5e9 units per dispatch).
pub const query_tile_rows: u32 = mimo.prefill_chunk;

/// Once, at construction: MiMo's attention geometry is the kernel's (64 heads, GQA 16 / 8, 192 / 128, window 128).
pub fn requireGeometry(c: mimo.Config) !void {
    if (c.n_heads != heads or c.head_dim != head_dim or c.value_head_dim != value_dim or c.window != 128 or
        c.n_kv_heads == 0 or c.swa_n_kv_heads == 0 or c.n_heads % c.n_kv_heads != 0 or c.n_heads % c.swa_n_kv_heads != 0)
        return error.MimoNaxAttentionGeometry;
}

// ---- construction-time route ----

/// MLX's NAX gate (mirrored from the host's `naxAvailableFrom`): a G17-class GPU (G18 for phone parts) and macOS >= 26.2.
pub fn naxFrom(arch: []const u8, os_version: []const u8) bool {
    var buf: [128]u8 = undefined;
    const n = @min(arch.len, buf.len);
    for (arch[0..n], 0..) |ch, i| buf[i] = std.ascii.toLower(ch);
    const a = buf[0..n];
    const suffix = if (std.mem.startsWith(u8, a, "applegpu_g")) a[10..] else return false;
    var digits: usize = 0;
    while (digits < suffix.len and std.ascii.isDigit(suffix[digits])) digits += 1;
    if (digits == 0 or suffix.len - digits > 1) return false;
    if (digits < suffix.len and !std.ascii.isAlphabetic(suffix[digits])) return false;
    const gen = std.fmt.parseInt(u32, suffix[0..digits], 10) catch return false;
    const phone = digits < suffix.len and suffix[digits] == 'p';
    if (gen < @as(u32, if (phone) 18 else 17)) return false;
    var it = std.mem.splitScalar(u8, os_version, '.');
    const major = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const minor: u32 = if (it.next()) |m| (std.fmt.parseInt(u32, m, 10) catch 0) else 0;
    return major > 26 or (major == 26 and minor >= 2);
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// The stream's device has NAX units (read once, at construction).
pub fn deviceHasNax(s: Stream) bool {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_stream_get_device(&dev, s) != 0) return false;
    defer _ = mlx.mlx_device_free(dev);
    var kind: mlx.mlx_device_type = .cpu;
    if (mlx.mlx_device_get_type(&kind, dev) != 0 or kind != .gpu) return false;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return false;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return false;
    var ver: [64]u8 = undefined;
    var len: usize = ver.len;
    if (sysctlbyname("kern.osproductversion", &ver, &len, null, 0) != 0) return false;
    var n = @min(len, ver.len);
    while (n > 0 and ver[n - 1] == 0) n -= 1;
    return naxFrom(std.mem.span(cstr), ver[0..n]);
}

/// The installed route: the stock control when asked for, else NAX when the device has it. Asking for NAX on a device
/// without it refuses once, here (never a silent stock route).
pub fn resolve(requested: ?Route, has_nax: bool) !Route {
    if (requested) |r| {
        if (r == .nax and !has_nax) return error.MimoNaxAttentionUnavailable;
        return r;
    }
    return if (has_nax) .nax else .stock;
}

/// The widest key count the NAX device checks prove: pre-ship family `attn32k` (`mimo GPU32K +attn`, PASS 1/1,
/// 2026-10-08T13:29Z, stack 1737d62): 24,576 and 32,768 keys in real-tile passes (3 and 4), multi-pass bit-equal to one
/// dispatch, within 0.02 of the eager oracle. The NAX route serves at most this context (`requireCheckedContext`).
pub const checked_keys: u64 = 32_768;

/// Once, at construction (Module.init): `resolve`, then the checked range. The default (no "prefill_attention" in the
/// request's settings) means NAX where it is device-checked: past `checked_keys` it takes the stock control (Module.init
/// logs `settings.prefillAttentionStockLine`; the bill reads `defaultTakesStock`). The served context is a construction
/// parameter: a construction-time route, never a per-call branch. An explicit "nax" past the range refuses by number.
pub fn resolveServed(requested: ?Route, has_nax: bool, served_context: u64) !Route {
    const route = try resolve(requested, has_nax);
    if (route == .nax and defaultTakesStock(requested, served_context)) return .stock;
    try requireCheckedContext(route, served_context);
    return route;
}

/// The default route past the checked keys is the stock control (module and bill read this one function).
pub fn defaultTakesStock(requested: ?Route, served_context: u64) bool {
    return requested == null and served_context > checked_keys;
}

/// Once, at construction (Module.init, inside `resolveServed`): the NAX route serves at most `checked_keys` keys; an
/// explicit "nax" past it is refused by number (never a silent stock route; the default and the stock control serve it).
pub fn requireCheckedContext(route: Route, served_context: u64) !void {
    if (route != .nax or served_context <= checked_keys) return;
    std.log.scoped(.mimo).warn("refused: prefill_attention=nax is device-checked to {d} keys; the served context {d} is past it (serve at most {d}, or set \"prefill_attention\":\"stock\")", .{ checked_keys, served_context, checked_keys });
    return error.MimoNaxAttentionContextUnchecked;
}

// ---- span rule (design §4): module and bill read this one function ----

/// The bytes one row of a global span holds besides the per-tile workspace the bill already charges
/// (config.workspaceBytes at the 512-row tile): for each KV head, the K/V projections (192 + 128 BF16), the roped K and
/// its rotary temporaries (192 + 5 x 64 BF16), V's F32 scale and BF16 result (128 x (4 + 2)), and the Q8 tier's
/// incoming triple (320 + 4 x 5 groups). No per-key term: no mask, no score, no pad.
pub fn spanRowBytes(kv_heads: u32) u64 {
    const per_head: u64 = (head_dim + value_dim) * 2 + (head_dim + 5 * 64) * 2 + value_dim * (4 + 2) + (head_dim + value_dim + 4 * 5);
    return @as(u64, kv_heads) * per_head;
}
/// The span's K/V arrays may hold at most this much (one span per call up to the 16,384-row call: 16,384 x 11,088 B).
pub const span_target_bytes: u64 = 256 << 20;

/// The rows of one prompt attention span on the NAX route: the rows whose K/V are projected, written and viewed once
/// (the Q8 tier dequantizes its dense view once per span, not per tile); queries then run in `query_tile_rows` tiles
/// over that view. Global: min(call, target / spanRowBytes), so every call up to 16,384 rows is one span per layer and
/// the standard 16K cell keeps one call. Sliding: the 512-row tile, the ring's view geometry (kv.zig `ringBase` holds
/// window + 511 keys; its comptime check).
pub fn attentionSpanRows(kv_heads: u32, mode: mimo.Attention, call_rows: u64) u32 {
    if (call_rows == 0) return 0;
    const rows = @min(call_rows, @as(u64, mimo.prefill_max_tokens));
    if (mode == .sliding) return @intCast(@min(rows, @as(u64, mimo.prefill_chunk)));
    return @intCast(@max(1, @min(rows, span_target_bytes / spanRowBytes(kv_heads))));
}

/// Key-block boundaries of the passes over `nk` key blocks (the donor's rule): one dispatch for a band or a grid too
/// small to drift apart, else ~8,192-key passes.
pub fn passEdges(out: *[max_passes + 1]c_int, groups: c_int, nk: c_int, window: c_int) []const c_int {
    const n_pass: c_int = if (window > 0 or groups < pass_min_groups)
        1
    else
        @max(1, @min(@min(nk, max_passes), @divTrunc(nk * BK + pass_keys / 2, pass_keys)));
    for (0..@intCast(n_pass + 1)) |i| out[i] = @divTrunc(@as(c_int, @intCast(i)) * nk, n_pass);
    return out[0..@intCast(n_pass + 1)];
}

/// Passes of one dispatch at `rows` queries over `keys` keys.
pub fn passCount(rows: u64, keys: u64, window: u32) u32 {
    var buf: [max_passes + 1]c_int = undefined;
    const groups: c_int = @intCast(heads * ((rows + 63) / 64));
    const nk: c_int = @intCast((keys + 31) / 32);
    return @intCast(passEdges(&buf, groups, nk, @intCast(window)).len - 1);
}

/// One pass hand-off at `rows` queries (WN = 2: BQ = 32): fp32 O[128] + max + sum per query row and head.
pub fn passStateBytes(rows: u64) u64 {
    return heads * ((rows + 31) / 32) * 32 * (value_dim + 2) * 4;
}

/// The multi-pass state the prompt holds at once: a pass reads the previous state while it writes its own (two live),
/// at the widest tile, once any global tile can reach the pass threshold (keys up to `reserve`).
pub fn passStateBillBytes(reserve: u64) u64 {
    const rows = @min(reserve, @as(u64, query_tile_rows));
    if (rows <= 8 or passCount(rows, reserve, 0) <= 1) return 0;
    return 2 * passStateBytes(rows);
}

// ---- the installed kernel ----

pub const Kernel = struct {
    k: mlx.mlx_fast_metal_kernel = .{},
    /// One-element placeholders: absent sinks (global layers) and the first pass's state.
    no_sinks: Array = .{},
    no_state: Array = .{},

    pub fn init(s: Stream) !Kernel {
        const ins = [_][*:0]const u8{ "q", "k", "v", "sinks", "state", "params" };
        const outs = [_][*:0]const u8{ "out", "state_out" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        // Q/K/V are read through their strides (cache views, transposed projections): ensure_row_contiguous = false.
        var self: Kernel = .{ .k = mlx.mlx_fast_metal_kernel_new("msv_nax_attn_192_128", in_vec, out_vec, source_text, header_text, false, false) };
        if (self.k.ctx == null) return error.MimoNaxAttentionKernel;
        errdefer self.deinit();
        const zero = [_]f32{0};
        const f = mlx.mlx_array_new_data(&zero, &.{1}, 1, .float32);
        defer drop(f);
        try mlx.check(mlx.mlx_astype(&self.no_sinks, f, .bfloat16, s));
        try mlx.check(mlx.mlx_copy(&self.no_state, f, s));
        try mlx.check(mlx.mlx_array_eval(self.no_sinks));
        try mlx.check(mlx.mlx_array_eval(self.no_state));
        return self;
    }

    pub fn deinit(self: *Kernel) void {
        if (self.k.ctx != null) _ = mlx.mlx_fast_metal_kernel_free(self.k);
        drop(self.no_sinks);
        drop(self.no_state);
        self.* = .{};
    }

    /// Causal (bottom-right) attention, banded to the last `window` keys when > 0, for q [1, 64, L, 192] (L > 8) over
    /// k [1, Hk, kL, 192], v [1, Hk, kL, 128] (kL >= L; head dim contiguous, any outer strides) with BF16 `sinks` [64]
    /// or none (`.{}`). Returns the BF16 output physically [1, L, 64, 128]. The shapes are the installed geometry
    /// (`requireGeometry`) and the caller's views; nothing here re-validates them.
    pub fn attention(self: *const Kernel, q: Array, k: Array, v: Array, scale: f32, window: u32, sinks: Array, s: Stream) !Array {
        const qs = mlx.getShape(q);
        const ks = mlx.getShape(k);
        var edges_buf: [max_passes + 1]c_int = undefined;
        const nk = @divTrunc(ks[2] + BK - 1, BK);
        const edges = passEdges(&edges_buf, qs[0] * qs[1] * @divTrunc(qs[2] + 63, 64), nk, @intCast(window));
        return self.dispatch(q, k, v, scale, @intCast(window), sinks, edges, if (edges.len > 2) 2 else 1, s);
    }

    /// The passes between consecutive `edges` (key blocks) at head-dim split `wn` (the donor's dispatch).
    pub fn dispatch(self: *const Kernel, q: Array, k: Array, v: Array, scale: f32, window: c_int, sinks: Array, edges: []const c_int, wn: c_int, s: Stream) !Array {
        const qs = mlx.getShape(q);
        const ks = mlx.getShape(k);
        const b = qs[0];
        const h = qs[1];
        const ql = qs[2];
        const kl = ks[2];
        const wm = @divExact(4, wn);
        const bq = 16 * wm;
        const nq = @divTrunc(ql + bq - 1, bq);
        const nq_aligned = @divTrunc(ql, bq);
        const nk = @divTrunc(kl + BK - 1, BK);
        const nk_aligned = @divTrunc(kl, BK);
        const has_sinks = sinks.ctx != null;
        const one = [_]c_int{1};
        const state_size = b * h * nq * bq * @as(c_int, @intCast(value_dim + 2));
        var state: Array = self.no_state;
        var owned_state: Array = .{};
        defer drop(owned_state);
        var out: Array = .{};
        errdefer drop(out);
        for (edges[0 .. edges.len - 1], edges[1..], 0..) |kb_begin, kb_end, i| {
            const first = i == 0;
            const last = i == edges.len - 2;
            // omlx_nax::AttnParams: B H D qL kL gqa scale NQ NK NQ_aligned NK_aligned qL_rem kL_rem qL_off kb_begin kb_end.
            const pdata = [16]i32{ b, h, 192, ql, kl, @divExact(h, ks[1]), @bitCast(scale), nq, nk, nq_aligned, nk_aligned, ql - nq_aligned * bq, kl - nk_aligned * BK, kl - ql, kb_begin, kb_end };
            const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{16}, 1, .int32);
            defer drop(params);
            const c = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
            if (last) {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ b, ql, h, 128 }, 4, .bfloat16));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &one, 1, .float32));
            } else {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &one, 1, .bfloat16));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{state_size}, 1, .float32));
            }
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, nq * threads, h, b));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, threads, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WM", wm));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WN", wn));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WINDOW", window));
            for ([_]struct { [*:0]const u8, bool }{
                .{ "ALIGN_Q", @rem(ql, bq) == 0 }, .{ "ALIGN_K", @rem(kl, BK) == 0 }, .{ "HAS_SINKS", has_sinks },
                .{ "FIRST", first },               .{ "LAST", last },
            }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, t[0], t[1]));
            const ins = [_]Array{ q, k, v, if (has_sinks) sinks else self.no_sinks, state, params };
            const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var outs = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(outs);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, self.k, in_vec, c, s));
            if (last) {
                try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
            } else {
                var next: Array = .{};
                try mlx.check(mlx.mlx_vector_array_get(&next, outs, 1));
                drop(owned_state);
                owned_state = next;
                state = next;
            }
        }
        return out;
    }
};

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

// ---- CPU tests (no device) ----

test "mimo +attn pins: the embedded kernel texts and manifest are the donor's bytes" {
    const hex = std.fmt.bytesToHex;
    var d: [32]u8 = undefined;
    inline for (.{ .{ tiles_text, pins.tiles }, .{ core_text, pins.core }, .{ source_text, pins.source }, .{ manifest_text, pins.manifest } }) |p| {
        std.crypto.hash.sha2.Sha256.hash(p[0], &d, .{});
        try std.testing.expectEqualStrings(p[1], &hex(d, .lower));
    }
    // The manifest pins each text by the same digest.
    for ([_][]const u8{ pins.tiles, pins.core, pins.source }) |pin| try std.testing.expect(std.mem.indexOf(u8, manifest_text, pin) != null);
    try std.testing.expect(tiles_text[tiles_text.len - 1] == '\n');
}

test "mimo +attn route: NAX on a G17 device at macOS 26.2+, the stock control on request, refusal when NAX is absent" {
    try std.testing.expect(naxFrom("applegpu_g17s", "26.2"));
    try std.testing.expect(naxFrom("applegpu_g17c", "27.0"));
    try std.testing.expect(!naxFrom("applegpu_g16s", "26.4"));
    try std.testing.expect(!naxFrom("applegpu_g17s", "26.1"));
    try std.testing.expect(!naxFrom("applegpu_g17p", "26.4"));
    try std.testing.expect(!naxFrom("air64_v27", "26.4"));
    try std.testing.expectEqual(Route.nax, try resolve(null, true));
    try std.testing.expectEqual(Route.stock, try resolve(null, false));
    try std.testing.expectEqual(Route.stock, try resolve(.stock, true));
    try std.testing.expectError(error.MimoNaxAttentionUnavailable, resolve(.nax, false));
    try requireGeometry(.{});
}

test "mimo +attn passes: a global 512-row tile splits past 12,256 keys, a sliding band never does; 17.0 MB state" {
    // n = (nk x 32 + 4,096) / 8,192 reaches 2 at nk = 384 key blocks: keys > 12,256.
    try std.testing.expectEqual(@as(u32, 1), passCount(512, 12_256, 0));
    try std.testing.expectEqual(@as(u32, 2), passCount(512, 12_257, 0));
    try std.testing.expectEqual(@as(u32, 2), passCount(512, 16_384, 0));
    try std.testing.expectEqual(@as(u32, 1), passCount(512, 639, 128));
    try std.testing.expectEqual(@as(u32, 1), passCount(9, 16_384, 0));
    try std.testing.expectEqual(@as(u32, 64), passCount(512, 1 << 20, 0));
    try std.testing.expectEqual(@as(u64, 17_039_360), passStateBytes(512));
    try std.testing.expectEqual(@as(u64, 0), passStateBillBytes(12_256));
    try std.testing.expectEqual(@as(u64, 2 * 17_039_360), passStateBillBytes(16_384));
    var buf: [max_passes + 1]c_int = undefined;
    try std.testing.expectEqualSlices(c_int, &.{ 0, 1024 }, passEdges(&buf, 8192, 1024, 128));
    try std.testing.expectEqualSlices(c_int, &.{ 0, 256, 512, 768, 1024 }, passEdges(&buf, 8192, 1024, 0));
}

test "mimo +attn span: one global span per call up to 16,384 rows, the ring's tile for sliding" {
    const c: mimo.Config = .{};
    try std.testing.expectEqual(@as(u64, 11_088), spanRowBytes(c.n_kv_heads));
    for ([_]u64{ 16_384, 131_072, 1 << 20 }) |prompt| {
        const call = @min(prompt, @as(u64, mimo.prefill_max_tokens));
        try std.testing.expectEqual(@as(u32, 16_384), attentionSpanRows(c.n_kv_heads, .full, call));
        try std.testing.expectEqual(@as(u32, 512), attentionSpanRows(c.swa_n_kv_heads, .sliding, call));
    }
    try std.testing.expectEqual(@as(u32, 9), attentionSpanRows(c.n_kv_heads, .full, 9));
    try std.testing.expect(spanRowBytes(c.n_kv_heads) * attentionSpanRows(c.n_kv_heads, .full, 16_384) <= span_target_bytes);
}

test "mimo +attn range: NAX at every served context to its checked 32,768 keys, refused past it by name, stock unbounded" {
    for ([_]u64{ 1, 8, 512, 12_256, 16_384, 17_408, 24_576, checked_keys }) |context| try requireCheckedContext(.nax, context);
    for ([_]u64{ checked_keys + 1, 65_536, 1 << 20 }) |context| {
        try std.testing.expectError(error.MimoNaxAttentionContextUnchecked, requireCheckedContext(.nax, context));
        try requireCheckedContext(.stock, context);
    }
    // The served cell's context (32,768) and the host's default (16K + 1K) are inside the range.
    try requireCheckedContext(.nax, @import("settings.zig").fill_reserve_tokens);
}

test "mimo +attn CPU: the default prompt route is NAX within the checked keys and the stock control past them; an explicit nax past them refuses by number" {
    try std.testing.expectEqual(Route.nax, try resolveServed(null, true, checked_keys));
    try std.testing.expectEqual(Route.stock, try resolveServed(null, true, 33_792));
    try std.testing.expectEqual(Route.stock, try resolveServed(null, true, 131_072));
    try std.testing.expectEqual(Route.stock, try resolveServed(null, false, 33_792));
    try std.testing.expectEqual(Route.stock, try resolveServed(.stock, true, 33_792));
    try std.testing.expectEqual(Route.nax, try resolveServed(.nax, true, checked_keys));
    try std.testing.expectError(error.MimoNaxAttentionContextUnchecked, resolveServed(.nax, true, 33_792));
    try std.testing.expectError(error.MimoNaxAttentionUnavailable, resolveServed(.nax, false, 17_408));
    try std.testing.expect(defaultTakesStock(null, 33_792) and !defaultTakesStock(null, checked_keys) and !defaultTakesStock(.nax, 33_792));
}
