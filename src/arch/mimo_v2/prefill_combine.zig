//! Lever 1 of prefill-attr-20261008: the prompt pass's routed combine as ONE text per routed layer.
//!
//! The stock route (`Route.stock`, math.zig `combineParts`, the control): per gather tile of 1,024 tokens, up to 24
//! takes + a concatenate + a permuting take build the tile's [count, 8, H] bf16 routed rows (word copies), evaluated;
//! per reduction tile of 256 tokens (`reduceGraph`) f32(row) * f32 w, MLX's axis-1 sum, bf16 once, evaluated and
//! synchronized; then the tiles' outputs concatenated [nt, H], evaluated. 16 + 64 + 1 evals at 16K tokens.
//!
//! The fused route (`Route.fused`): `mimo_prefill_combine` reads each token's 8 routed rows in place from the wide
//! call's <= 24 sources (`deepseek_v41_experts.mergeJoinless`) through the (source, row) table, and computes the stock
//! reduction tile's Metal arithmetic word for word: per column, each product `f32(row_k) * w_k` rounded to f32 (no FMA
//! contraction), added to the +0 init, folded k = 0..7 in router order as `product_k + total`, rounded to bf16 once.
//! One launch and one eval per routed layer; no gathered rows, no f32 transients, no output concatenate.
//!
//! Exactness: the stock reduction is MLX's `col_reduce_small` (reduce.cpp `strided_reduce_general_dispatch`: reduction
//! size 8 x 1 < 32) with threadgroup y = 8: thread k computes `op(p_k, 0)`, lid.y 0 then folds
//! `op(shared_k, total)` for k = 1..7 (reduce_col.h) -- the order this text and `combineRef` state. The residual add
//! (module.zig `next = residual + mlp`) is its own bf16 op after the combine and is not folded in; MiMo has no shared
//! expert (config.zig refuses `n_shared_experts`), so there is no shared term.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");
const bank_mod = @import("bank.zig");
const moe = @import("moe.zig");

const T = mlx.mlx_array;

/// The prompt pass's routed combine, fixed at construction (`settings.Config.prefill_combine`): `stock` (the control),
/// `fused` (the text over <= 24 bound sources, the call's smallest outputs merged into one by the shared scheduler), or
/// P7 `inplace` (the SAME text reading every source of the call in place through a table of their GPU addresses: no
/// merge concatenate, no merged buffer); any other name is refused (`error.MimoPrefillCombine`).
pub const Route = enum { stock, fused, inplace, invalid };

/// P7 `inplace`: the most sources a call's combine reads in place (the scheduler's source index is a byte); a call with
/// more falls back to the merge at `sources` (the stock rule) -- a MiMo prompt call has at most ~140 tiles.
pub const inplace_sources: usize = 255;

/// P7 `inplace`: the fewest entries the table hands MLX. MLX 64ea011c (backend/common/metal_kernel.cpp
/// `max_constant_array_size`) declares a custom-kernel input of fewer than 8 elements in the `constant` address space,
/// which `mimo_pc_pick`'s table overload (`const device uint64_t*`) does not take: the text failed to compile ("no
/// matching function for call to 'mimo_pc_pick'") for every call of 1..7 sources. The table is padded to this many
/// entries, so every call is the one `device` form (one compiled library for every source count).
pub const table_min_entries: usize = 8;

comptime {
    if (inplace_sources < table_min_entries) @compileError("mimo prefill combine: the table bound must hold the padding");
}

/// The table of `n` sources' addresses (`addr[0..n]`) as MLX is handed it: padded to `table_min_entries` with source
/// 0's address. The padding is never read: every `loc` source index is < `n`.
fn paddedTable(addr: *[inplace_sources]u64, n: usize) []const u64 {
    std.debug.assert(n > 0 and n <= inplace_sources);
    const len = @max(n, table_min_entries);
    for (addr[n..len]) |*x| x.* = addr[0];
    return addr[0..len];
}

pub const hidden: u32 = moe.hidden;
pub const top_k: u32 = moe.top_k;
/// The text's source slots (`deepseek_v41_experts.joinless_sources`); unused slots alias source 0.
pub const sources: usize = @import("../deepseek_v41/experts.zig").joinless_sources;
/// Columns per thread (one 8-byte bf16 load per routed row), threads per threadgroup: 256 x 4 = 1,024 columns of one
/// token per threadgroup, H / 1,024 = 4 threadgroups per token.
pub const cols_per_thread: u32 = 4;
pub const threads: u32 = 256;

comptime {
    if (hidden % (cols_per_thread * threads) != 0) @compileError("mimo prefill combine: H must be whole threadgroups of columns");
    if (sources + 2 + 1 > 31) @compileError("mimo prefill combine: the text's buffers exceed Metal's 31 argument slots");
}

/// The texts (pinned by `kernels/manifest-prefill-combine.json`; `manifest_sha256` pins the manifest).
pub const texts = struct {
    pub const header = @embedFile("kernels/header_mimo_prefill_combine.metal");
    pub const combine = @embedFile("kernels/mimo_prefill_combine.metal");
    pub const manifest = @embedFile("kernels/manifest-prefill-combine.json");
    /// P7 `inplace`: the same text with its 24 bound sources replaced by the table (`mimo_pc_pick`'s table overload).
    pub const combine_inplace = tableText(combine);
};

/// The bound sources' argument list as the text passes it to `mimo_pc_pick`.
const bound_args = blk: {
    var t: []const u8 = "s0";
    for (1..sources) |i| t = t ++ std.fmt.comptimePrint(", s{d}", .{i});
    break :blk t;
};

/// `text` with every `mimo_pc_pick` argument list of the 24 bound sources replaced by `tbl` (comptime).
fn tableText(comptime text: []const u8) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(100_000);
        var out: []const u8 = "";
        var rest: []const u8 = text;
        while (std.mem.indexOf(u8, rest, bound_args)) |i| {
            out = out ++ rest[0..i] ++ "tbl";
            rest = rest[i + bound_args.len ..];
        }
        const full = out ++ rest ++ [_]u8{0};
        return full[0 .. full.len - 1 :0];
    }
}

/// The table form's buffers: the sources' GPU addresses, then the text's own.
const inplace_input_names = [_][*:0]const u8{ "tbl", "loc", "w" };

/// P7 `inplace` (csrc/mimo_gpu_address.inc): each evaluated source's GPU address (its MTLBuffer's plus its offset).
extern fn mimo_gpu_addresses(arrs: [*]const T, n: usize, out: [*]u64) c_int;
extern fn dsv41_alloc_last_error() [*:0]const u8;
pub const manifest_sha256 = "ed3b2e1af2a2803791f337adc73605b0d08f0e733e298da86ef5a90ae323b3f3";

const input_names = blk: {
    var names: [sources + 2][*:0]const u8 = undefined;
    for (0..sources) |i| names[i] = std.fmt.comptimePrint("s{d}", .{i});
    names[sources] = "loc";
    names[sources + 1] = "w";
    break :blk names;
};

/// The fused route's text, built once at construction (no device work: it compiles at its first dispatch; the
/// byte-equality self-check is the pre-ship gate's `prefill_combine` family).
pub const Kernel = struct {
    k: mlx.mlx_fast_metal_kernel = .{},
    /// P7 `inplace`: the table form of the same text (`initInplace`).
    k_inplace: mlx.mlx_fast_metal_kernel = .{},

    /// Construction boundary: the routed geometry the text is instantiated for (any bank's geometry: its `hidden_size`).
    pub fn init(geometry: anytype) !Kernel {
        if (geometry.hidden_size != hidden) return error.MimoPrefillCombineGeometry;
        return .{ .k = try newKernel("mimo_prefill_combine", &input_names, texts.combine) };
    }

    /// P7 `inplace`: the text's table form, built once at construction (no device work: it compiles at its first
    /// dispatch; the byte-equality self-check is the pre-ship family `prefill_combine`).
    pub fn initInplace(geometry: anytype) !Kernel {
        if (geometry.hidden_size != hidden) return error.MimoPrefillCombineGeometry;
        return .{ .k_inplace = try newKernel("mimo_prefill_combine_inplace", &inplace_input_names, texts.combine_inplace) };
    }

    fn newKernel(name: [*:0]const u8, names: []const [*:0]const u8, text: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, text, texts.header, true, false);
        if (k.ctx == null) return error.MimoPrefillCombineKernel;
        return k;
    }

    pub fn deinit(self: *Kernel) void {
        if (self.k.ctx != null) _ = mlx.mlx_fast_metal_kernel_free(self.k);
        if (self.k_inplace.ctx != null) _ = mlx.mlx_fast_metal_kernel_free(self.k_inplace);
        self.* = .{};
    }

    /// P7 `inplace`: `outs` the wide call's evaluated bf16 sources [r_i, H] (any number up to `inplace_sources`, none
    /// merged), `loc` int32 [nt, 8, 2] each routed position's (source, row), `weights` f32 [nt, 8] -> bf16 [nt, H],
    /// unevaluated. The sources are read through `tbl` (their GPU addresses): the caller keeps them alive until the
    /// output is evaluated (`Source.forward`: `releaseParts` after it).
    pub fn callInplace(self: *const Kernel, g: *ops.MlxOps, outs: []const T, loc: T, weights: T, nt: u32) !T {
        std.debug.assert(outs.len > 0 and outs.len <= inplace_sources);
        var addr: [inplace_sources]u64 = undefined;
        if (mimo_gpu_addresses(outs.ptr, outs.len, &addr) != 0) {
            std.log.err("MiMo prefill_combine=inplace: {s}", .{dsv41_alloc_last_error()});
            return error.MimoPrefillCombineSource;
        }
        const table = paddedTable(&addr, outs.len);
        const tbl = try g.hostArray(std.mem.sliceAsBytes(table), &.{@intCast(table.len)}, .uint64);
        const in = [_]T{ tbl, loc, weights };
        return self.apply(g, self.k_inplace, &in, nt);
    }

    fn apply(_: *const Kernel, g: *ops.MlxOps, k: mlx.mlx_fast_metal_kernel, in: []const T, nt: u32) !T {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const shape = [_]c_int{ @intCast(nt), @intCast(hidden) };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(nt * (hidden / cols_per_thread)), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(threads), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", @intCast(top_k)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", @intCast(hidden)));
        const in_vec = mlx.mlx_vector_array_new_data(in.ptr, in.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, k, in_vec, cfg, g.s));
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0)) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        return g.adopt(out);
    }

    /// `outs`: the wide call's evaluated bf16 sources [r_i, H] (1..24), `loc` int32 [nt, 8, 2] each routed position's
    /// (source, row), `weights` f32 [nt, 8] -> bf16 [nt, H], unevaluated. The launch's grid follows the call's logical
    /// M (`nt`, the one runtime-varying value): nt x H / 4 threads.
    pub fn call(self: *const Kernel, g: *ops.MlxOps, outs: []const T, loc: T, weights: T, nt: u32) !T {
        std.debug.assert(outs.len > 0 and outs.len <= sources);
        var in: [sources + 2]T = undefined;
        for (in[0..sources], 0..) |*x, i| x.* = outs[if (i < outs.len) i else 0];
        in[sources] = loc;
        in[sources + 1] = weights;
        return self.apply(g, self.k, &in, nt);
    }
};

// ── Host reference of the text's arithmetic (CPU proofs; the device self-check compares the text) ──

fn bf16Bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast(((u +% (0x7fff + ((u >> 16) & 1))) >> 16) & 0xffff);
}

fn bf16Value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The text, per token t and column c: `p_k = f32(row_k) * w_k + 0` (f32 product, never contracted), `acc = p_0`,
/// `acc = p_k + acc` for k = 1..7, bf16 once. Rows are read in place: `srcs[loc[2p]][loc[2p + 1] * h + c]`, p = t*8 + k.
pub fn combineRef(srcs: []const []const u16, loc: []const i32, w: []const f32, nt: usize, h: usize, out: []u16) void {
    for (0..nt) |t| for (0..h) |c| {
        var acc: f32 = 0;
        for (0..top_k) |k| {
            const p = t * top_k + k;
            const row: usize = @intCast(loc[2 * p + 1]);
            const v = bf16Value(srcs[@intCast(loc[2 * p])][row * h + c]);
            const q: f32 = v * w[p] + 0.0;
            acc = if (k == 0) q else q + acc;
        }
        out[t * h + c] = bf16Bits(acc);
    };
}

// ── Fixtures: real-shaped wide-call sources (CPU proofs and the device self-check) ──

fn nextU32(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

/// A wide call's combine inputs: `outs` (KEPT bf16 [r_i, H] sources), `loc` (host + array), `weights` (host + array);
/// `bits` the sources' host words when `keep_host` (the reference reads them).
const Fixture = struct {
    outs: std.ArrayList(T) = .empty,
    bits: std.ArrayList([]const u16) = .empty,
    loc: []i32 = &.{},
    w: []f32 = &.{},
    loc_arr: T = .{},
    w_arr: T = .{},

    fn deinit(f: *Fixture, a: std.mem.Allocator, g: *ops.MlxOps) void {
        for (f.outs.items) |x| g.release(x);
        f.outs.deinit(a);
        for (f.bits.items) |b| a.free(b);
        f.bits.deinit(a);
        a.free(f.loc);
        a.free(f.w);
        if (f.loc_arr.ctx != null) g.release(f.loc_arr);
        if (f.w_arr.ctx != null) g.release(f.w_arr);
    }
};

/// `n_out` outputs of random sizes (>= 1 row each) holding the call's nt x 8 routed rows, a seeded permutation of
/// (output, row) over routed positions; with `merge`, the shared scheduler's merge (`mergeJoinless`) down to <= 24
/// sources, as `runWideParts` does. bf16 words over wide exponents with signed zeros; weights as module.route makes
/// them (f32 sigmoid picks over their sum).
fn makeFixture(a: std.mem.Allocator, g: *ops.MlxOps, nt: u32, n_out: u32, seed: u32, keep_host: bool, merge: bool) !Fixture {
    const nr: usize = @as(usize, nt) * top_k;
    std.debug.assert(n_out >= 1 and n_out <= nr and (merge or n_out <= inplace_sources));
    var f: Fixture = .{};
    errdefer f.deinit(a, g);
    var state: u32 = seed | 1;
    // Sizes: even split, then random moves between neighbours (every output keeps >= 1 row).
    const sizes = try a.alloc(u32, n_out);
    defer a.free(sizes);
    for (sizes, 0..) |*s, i| s.* = @intCast(nr / n_out + @intFromBool(i < nr % n_out));
    for (0..n_out - 1) |i| {
        const d = nextU32(&state) % sizes[i];
        sizes[i] -= d;
        sizes[i + 1] += d;
    }
    // Join order: output i's rows back to back; pos[j] = the routed position join row j holds.
    const pos = try a.alloc(u32, nr);
    defer a.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i);
    var i: usize = nr;
    while (i > 1) {
        i -= 1;
        std.mem.swap(u32, &pos[i], &pos[nextU32(&state) % (i + 1)]);
    }
    const H: c_int = @intCast(hidden);
    for (sizes) |r| {
        const words = try a.alloc(u16, @as(usize, r) * hidden);
        errdefer a.free(words);
        for (words, 0..) |*v, wi| {
            const bits = nextU32(&state);
            v.* = if (wi % 997 == 0) @as(u16, @intCast((bits & 1) << 15)) else @intCast(((bits >> 16) & 0x8000) | (((112 + (bits >> 3) % 24) & 0xff) << 7) | ((bits >> 9) & 0x7f));
        }
        const mark = g.mark();
        const x = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(r), H }, .bfloat16);
        try g.evalAll(&.{x});
        try f.outs.append(a, g.keep(x));
        g.resetTo(mark);
        if (keep_host) try f.bits.append(a, words) else a.free(words);
    }
    f.loc = try a.alloc(i32, 2 * nr);
    if (merge) {
        var sc: @import("../deepseek_v41/experts.zig").JoinlessScratch(T) = .{};
        defer sc.deinit(a);
        _ = try @import("../deepseek_v41/experts.zig").mergeJoinless(ops.MlxOps, g, a, &f.outs, pos, f.loc, &sc);
        try g.evalAll(f.outs.items);
        if (keep_host) {
            // The merged source's words: the merged outputs' words in join order (the concatenate's copy).
            var rows: [256]u32 = undefined;
            var order: [256]u32 = undefined;
            var src: [256]u8 = undefined;
            var off: [256]u32 = undefined;
            const sz = sizes[0..n_out];
            @memcpy(rows[0..n_out], sz);
            const n_src = @import("../deepseek_v41/experts.zig").planJoinless(rows[0..n_out], order[0..n_out], src[0..n_out], off[0..n_out]);
            if (n_src < n_out) {
                var merged: std.ArrayList(u16) = .empty;
                errdefer merged.deinit(a);
                var kept: usize = 0;
                for (f.bits.items, 0..) |b, oi| {
                    if (src[oi] == n_src - 1) {
                        try merged.appendSlice(a, b);
                        a.free(b);
                    } else {
                        f.bits.items[kept] = b;
                        kept += 1;
                    }
                }
                f.bits.shrinkRetainingCapacity(kept);
                try f.bits.append(a, try merged.toOwnedSlice(a));
            }
        }
    } else {
        var j: usize = 0;
        for (sizes, 0..) |r, s| for (0..r) |row| {
            f.loc[2 * pos[j]] = @intCast(s);
            f.loc[2 * pos[j] + 1] = @intCast(row);
            j += 1;
        };
    }
    f.w = try a.alloc(f32, nr);
    for (0..nt) |t| {
        var sum: f32 = 0;
        for (f.w[t * top_k ..][0..top_k]) |*v| {
            v.* = moe.sigmoidRef(@as(f32, @floatFromInt(@as(i32, @intCast(nextU32(&state) % 2001)) - 1000)) / 250);
            sum += v.*;
        }
        for (f.w[t * top_k ..][0..top_k]) |*v| v.* /= sum + 1e-20;
    }
    const mark = g.mark();
    const la = try g.hostArray(std.mem.sliceAsBytes(f.loc), &.{ @intCast(nt), @intCast(top_k), 2 }, .int32);
    const wa = try g.hostArray(std.mem.sliceAsBytes(f.w), &.{ @intCast(nt), @intCast(top_k) }, .float32);
    try g.evalAll(&.{ la, wa });
    f.loc_arr = g.keep(la);
    f.w_arr = g.keep(wa);
    g.resetTo(mark);
    return f;
}

/// The stock route over a fixture (math.zig `combineParts` with the sync control's reductions): KEPT bf16 [nt, H].
fn stockOf(g: *ops.MlxOps, f: *const Fixture, nt: u32) !T {
    const prefill_math = @import("math.zig");
    return prefill_math.combineParts(g, f.outs.items, f.loc, f.w_arr, nt, @intCast(hidden), (try prefill_math.drains(.sync)).reduce);
}

/// An evaluated bf16 [nt, H]'s words, read 1,024 tokens at a time (bounded host staging).
fn wordsOf(a: std.mem.Allocator, g: *ops.MlxOps, x: T, nt: u32) ![]u16 {
    const out = try a.alloc(u16, @as(usize, nt) * hidden);
    errdefer a.free(out);
    const chunk: u32 = 1024;
    const tmp = try a.alloc(f32, @as(usize, @min(chunk, nt)) * hidden);
    defer a.free(tmp);
    var t: u32 = 0;
    while (t < nt) {
        const n = @min(chunk, nt - t);
        const mark = g.mark();
        const part = try g.astype(try g.slice(x, &.{ @intCast(t), 0 }, &.{ @intCast(t + n), @intCast(hidden) }, &.{ 1, 1 }), .float32);
        const vals = tmp[0 .. @as(usize, n) * hidden];
        _ = try g.hostF32(part, vals);
        for (vals, out[@as(usize, t) * hidden ..][0..vals.len]) |v, *o| o.* = @intCast(@as(u32, @bitCast(v)) >> 16);
        g.resetTo(mark);
        t += n;
    }
    return out;
}

fn expectWords(name: []const u8, want: []const u16, got: []const u16) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, x, i| if (w != x) {
        std.debug.print("MIMO_PREFILL_COMBINE first divergence case={s} token={d} col={d} stock=0x{x:0>4} fused=0x{x:0>4}\n", .{ name, i / hidden, i % hidden, w, x });
        return error.MimoPrefillCombineNotExact;
    };
}

// ── CPU tests ──

const testing = std.testing;

const shaHex = @import("../../sdk_ext/kernel_set.zig").shaHex;

test "mimo prefill combine registry: the manifest pins the header and the text, and the pin is the manifest's sha256" {
    try testing.expectEqualStrings(manifest_sha256, &shaHex(texts.manifest));
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, texts.manifest, .{});
    defer parsed.deinit();
    const m = parsed.value.object;
    const hs = m.get("headers").?.array.items;
    try testing.expectEqual(@as(usize, 1), hs.len);
    try testing.expectEqualStrings("header_mimo_prefill_combine.metal", hs[0].object.get("file").?.string);
    try testing.expectEqualStrings(hs[0].object.get("sha256").?.string, &shaHex(texts.header));
    const ks = m.get("kernels").?.array.items;
    try testing.expectEqual(@as(usize, 2), ks.len);
    // Both kernels are the one text (P7 `inplace`: its table form): one source file, one sha.
    for (ks, [_][]const u8{ "mimo_prefill_combine", "mimo_prefill_combine_inplace" }, [_][]const [*:0]const u8{ &input_names, &inplace_input_names }) |kv, kname, names| {
        const k = kv.object;
        try testing.expectEqualStrings(kname, k.get("name").?.string);
        try testing.expectEqualStrings("mimo_prefill_combine.metal", k.get("source").?.object.get("file").?.string);
        try testing.expectEqualStrings(k.get("source").?.object.get("sha256").?.string, &shaHex(texts.combine));
        try testing.expect(k.get("ensure_row_contiguous").?.bool);
        // The manifest's inputs are the form's buffer names, in order.
        const ins = k.get("inputs").?.array.items;
        try testing.expectEqual(names.len, ins.len);
        for (ins, names) |x, name| try testing.expectEqualStrings(std.mem.span(name), x.object.get("name").?.string);
    }
}

test "mimo prefill combine inplace text: the fused text with its 24 bound sources replaced by the table, nothing else" {
    // The bound list appears exactly at the two picks; the table form differs from the text only there.
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, texts.combine, bound_args));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, texts.combine_inplace, bound_args));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, texts.combine_inplace, "mimo_pc_pick<T>(loc[2 * lb], tbl)") + std.mem.count(u8, texts.combine_inplace, "mimo_pc_pick<T>(loc[2 * (lb + k)], tbl)"));
    try testing.expectEqual(texts.combine.len - 2 * (bound_args.len - "tbl".len), texts.combine_inplace.len);
    // The fold is the fused text's, character for character (everything outside the picks).
    for ([_][]const u8{ "acc[i] = static_cast<float>(p[i]) * wk + 0.0f;", "const float q = static_cast<float>(p[i]) * wk + 0.0f;", "acc[i] = q + acc[i];", "for (uint k = 1; k < uint(TOPK); ++k)" }) |line| {
        try testing.expectEqual(std.mem.count(u8, texts.combine, line), std.mem.count(u8, texts.combine_inplace, line));
    }
    try testing.expect(std.mem.indexOf(u8, texts.header, "inline const device T* mimo_pc_pick(int s, const device uint64_t* tbl)") != null);
    try testing.expectError(error.MimoPrefillCombineGeometry, Kernel.initInplace(try bank_mod.Geometry.init(64, 64, 64, 1, 1)));
}

test "mimo prefill combine inplace table: padded with source 0 to MLX's device-array bound, the sources' entries unchanged" {
    for ([_]usize{ 1, 2, 7, 8, 24, 25, 140, 255 }) |n| {
        var addr: [inplace_sources]u64 = undefined;
        for (&addr, 0..) |*x, i| x.* = 0x1_0000_0000 + 0x1000 * @as(u64, i);
        const table = paddedTable(&addr, n);
        // MLX passes an input of fewer than 8 elements in the `constant` address space (no table overload takes it).
        try testing.expectEqual(@max(n, table_min_entries), table.len);
        try testing.expect(table.len >= table_min_entries);
        for (table, 0..) |x, i| try testing.expectEqual(if (i < n) 0x1_0000_0000 + 0x1000 * @as(u64, i) else table[0], x);
    }
}

test "mimo prefill combine text: the stock reduction's fold, stated in the text (no contraction, +0 init, k order, one rounding)" {
    try testing.expect(std.mem.indexOf(u8, texts.header, "#pragma METAL fp contract(off)") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "acc[i] = static_cast<float>(p[i]) * wk + 0.0f;") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "const float q = static_cast<float>(p[i]) * wk + 0.0f;") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "acc[i] = q + acc[i];") != null);
    try testing.expect(std.mem.indexOf(u8, texts.combine, "for (uint k = 1; k < uint(TOPK); ++k)") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, texts.combine, "static_cast<T>("));
    try testing.expect(std.mem.indexOf(u8, texts.combine, "constexpr uint C = 4;") != null);
    // Construction refuses geometry the text is not instantiated for (no device work before the first dispatch).
    try testing.expectError(error.MimoPrefillCombineGeometry, Kernel.init(try bank_mod.Geometry.init(64, 64, 64, 1, 1)));
}

test "mimo prefill combine CPU: the fused text's host reference is bit-equal to the stock gather + reduction route at real shapes" {
    try @import("mtp.zig").requireWindow();
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const Case = struct { nt: u32, outs: []const u32, merge: bool = false };
    const all24 = comptime blk: {
        var v: [24]u32 = undefined;
        for (&v, 1..) |*x, n| x.* = n;
        break :blk v;
    };
    const cases = [_]Case{
        .{ .nt = 1, .outs = all24[0..8] },
        .{ .nt = 7, .outs = &all24 },
        .{ .nt = 1024, .outs = &.{ 1, 2, 24 } },
        .{ .nt = 1037, .outs = &.{ 1, 24 } },
        .{ .nt = 1037, .outs = &.{30}, .merge = true },
        // P7 `inplace`: more than 24 sources, none merged (the table form reads each in place).
        .{ .nt = 1037, .outs = &.{ 25, 140 } },
    };
    var checked: usize = 0;
    for (cases, 0..) |c, ci| for (c.outs) |n_out| {
        const seed = 0x7c0b1e5d +% @as(u32, @intCast(ci * 131 + n_out));
        var f = try makeFixture(a, &g, c.nt, n_out, seed, true, c.merge);
        defer f.deinit(a, &g);
        // Past 24 sources the stock route reads the merged twin (the same rows; the merge copies words).
        var twin: ?Fixture = if (!c.merge and n_out > sources) try makeFixture(a, &g, c.nt, n_out, seed, false, true) else null;
        defer if (twin) |*t| t.deinit(a, &g);
        const stock = try stockOf(&g, if (twin) |*t| t else &f, c.nt);
        defer g.release(stock);
        const want = try wordsOf(a, &g, stock, c.nt);
        defer a.free(want);
        const got = try a.alloc(u16, want.len);
        defer a.free(got);
        combineRef(f.bits.items, f.loc, f.w, c.nt, hidden, got);
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "t{d}_outs{d}{s}", .{ c.nt, n_out, if (c.merge) "_merged" else "" });
        try expectWords(name, want, got);
        std.debug.print("MIMO_PREFILL_COMBINE case={s} tokens={d} sources={d} words={d} bit_equal=true\n", .{ name, c.nt, f.outs.items.len, want.len });
        checked += 1;
    };
    try testing.expectEqual(@as(usize, 8 + 24 + 3 + 2 + 1 + 2), checked);
}

// ── Device self-check (pre-ship family `prefill_combine`; never constructed by ordinary host checks) ──
//
// Static bound (declared before any run), the largest case (16,384 tokens, 24 sources): sources 131,072 x 4096 x 2 B =
// 1,073,741,824 B; one source's host words + its device copy while built (<= 2 x 11,000 rows x 8 KiB = 180 MB); the
// stock route's live set: one gather tile's pieces + concatenate + permuting take (3 x 64 MiB), one reduction tile's f32
// promotion + product + sum + bf16 (2 x 32 MiB + 4 MiB + 2 MiB), its 64 kept tile outputs + their concatenate (2 x 128
// MiB); the fused output (128 MiB); loc + weights (1.5 MiB); the comparison's 1,024-token f32 slices (2 x 16 MiB device +
// 2 x 16 MiB host) and both [nt, H] host word copies (2 x 128 MiB); MLX JIT (2 texts) + allocator cache slack 256 MiB.
// P7 `inplace` past 24 sources (16,384 tokens, 140 sources): the merged twin the stock route reads (its sources, the
// same 1,073,741,824 B) and its merge's concatenate while it is built (<= the same again), the table (<= 2 KiB).
const selfcheck_ceiling_bytes: u64 = 5 << 30;

fn selfcheckBound(nt: u64) u64 {
    const mib: u64 = 1 << 20;
    const row = @as(u64, hidden) * 2;
    const nr = nt * top_k;
    const twin = 2 * nr * row + 4096;
    return nr * row + 2 * 11_000 * row + 3 * 64 * mib + (2 * 32 + 4 + 2) * mib + 2 * nt * row + nt * row + nt * top_k * 12 + 4 * 16 * mib + 2 * nt * row + 256 * mib + twin;
}

/// A device self-check case: tokens, outputs of the wide call, merged (the stock scheduler's merge) or not, host reference.
const SelfCheckCase = struct { nt: u32, outs: u32, merge: bool = false, host: bool = true };

/// The device self-check's position in its cases (`selfCheckFailed`).
const SelfCheckAt = struct {
    name: []const u8,
    stage: []const u8 = "fixture",
    form: []const u8 = "-",
    tokens: u32,
    sources: usize,
    words: usize = 0,
};

/// A failing case's one line: where it was and MLX's latched error text (newlines folded), then the error goes on.
fn selfCheckFailed(at: SelfCheckAt, e: anyerror) void {
    var buf: [4096]u8 = undefined;
    const msg = mlx.takeError(&buf) orelse "(no MLX error latched)";
    for (buf[0..msg.len]) |*ch| if (ch.* == '\n') {
        ch.* = '|';
    };
    std.debug.print("MIMO_PREFILL_COMBINE_SELFCHECK FAILED case={s} form={s} stage={s} tokens={d} sources={d} words={d} error={s} mlx=\"{s}\"\n", .{ at.name, at.form, at.stage, at.tokens, at.sources, at.words, @errorName(e), msg });
}

/// One device self-check case: the fused form (<= 24 sources) and the table form vs the stock route's words (and the
/// host reference when the case keeps host words); `at` follows the case through its stages for `selfCheckFailed`.
fn selfCheckCase(a: std.mem.Allocator, g: *ops.MlxOps, kernel: *const Kernel, inplace: *const Kernel, c: SelfCheckCase, ci: usize, at: *SelfCheckAt) !void {
    const seed = 0x4d505243 +% @as(u32, @intCast(ci * 977));
    var f = try makeFixture(a, g, c.nt, c.outs, seed, c.host, c.merge);
    defer f.deinit(a, g);
    // Past 24 sources the stock route reads the merged twin (the same rows; the merge copies words).
    var twin: ?Fixture = if (!c.merge and c.outs > sources) try makeFixture(a, g, c.nt, c.outs, seed, false, true) else null;
    defer if (twin) |*t| t.deinit(a, g);
    at.stage = "stock";
    const stock = try stockOf(g, if (twin) |*t| t else &f, c.nt);
    defer g.release(stock);
    const want = try wordsOf(a, g, stock, c.nt);
    defer a.free(want);
    at.words = want.len;
    // The bound form takes <= 24 sources; the table form every case.
    for ([_]bool{ false, true }) |table| {
        if (!table and f.outs.items.len > sources) continue;
        at.form = if (table) "inplace" else "fused";
        at.sources = f.outs.items.len;
        at.stage = "dispatch";
        const mark = g.mark();
        const lazy = if (table) try inplace.callInplace(g, f.outs.items, f.loc_arr, f.w_arr, c.nt) else try kernel.call(g, f.outs.items, f.loc_arr, f.w_arr, c.nt);
        at.stage = "eval";
        try g.evalAll(&.{lazy});
        const fused = g.keep(lazy);
        defer g.release(fused);
        g.resetTo(mark);
        at.stage = "read";
        const got = try wordsOf(a, g, fused, c.nt);
        defer a.free(got);
        at.stage = "compare";
        try expectWords(at.name, want, got);
        if (c.host) {
            at.stage = "host_ref";
            const ref = try a.alloc(u16, want.len);
            defer a.free(ref);
            combineRef(f.bits.items, f.loc, f.w, c.nt, hidden, ref);
            try expectWords(at.name, ref, got);
        }
        std.debug.print("MIMO_PREFILL_COMBINE_SELFCHECK case={s} form={s} tokens={d} sources={d} words={d} byte_equal=true host_ref={s}\n", .{ at.name, at.form, c.nt, f.outs.items.len, want.len, if (c.host) "equal" else "skipped" });
    }
    g.reset();
}

test "mimo prefill combine GPU: the fused text is byte-equal to the stock gather + reduction route on the device" {
    _ = std.c.getenv("MIMO_PREFILL_COMBINE_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const bound = selfcheckBound(16_384);
    std.debug.print("MIMO_PREFILL_COMBINE_SELFCHECK bound_bytes={d} ceiling_bytes={d}\n", .{ bound, selfcheck_ceiling_bytes });
    if (bound > selfcheck_ceiling_bytes) return error.MimoPrefillCombineAdmission;
    mlx.installErrorHandler();
    const a = testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, selfcheck_ceiling_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    // P7 `inplace` reads its sources unbound, through MLX's residency set: the served condition is the host's wired
    // policy `max` (`applyWiredPolicy`: wired limit = the device's max recommended working set, every MLX buffer in the
    // set), so the family runs under it rather than under a test binary's capacity 0 (no buffer in the set).
    const max_rec = mlx.maxRecommendedWorkingSet();
    if (max_rec == 0) return error.MimoPrefillCombineResidency;
    var old_wired: usize = 0;
    try mlx.check(mlx.mlx_set_wired_limit(&old_wired, max_rec));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_wired_limit(&ignored, old_wired);
    }
    std.debug.print("MIMO_PREFILL_COMBINE_SELFCHECK wired_limit_bytes={d} (host policy max)\n", .{max_rec});
    var g = try ops.MlxOps.init(a, stream);
    defer g.deinit();
    var kernel = try Kernel.init(try bank_mod.Geometry.init(hidden, moe.inter, 256, 1, 47));
    defer kernel.deinit();
    // P7 `inplace`: the table form, checked on the same cases (<= 24 sources) and past 24 (none merged).
    var inplace = try Kernel.initInplace(try bank_mod.Geometry.init(hidden, moe.inter, 256, 1, 47));
    defer inplace.deinit();
    const cases = [_]SelfCheckCase{
        .{ .nt = 2, .outs = 1 },
        .{ .nt = 7, .outs = 1 },
        .{ .nt = 7, .outs = 24 },
        .{ .nt = 1037, .outs = 3 },
        .{ .nt = 1037, .outs = 30, .merge = true },
        // The std-16k prompt pass's shape (the merge itself is a word copy, covered at 1,037 tokens).
        .{ .nt = 16_384, .outs = 24, .host = false },
        // P7 `inplace`: past 24 sources, none merged; the last one the std-16k pass's tile count.
        .{ .nt = 1037, .outs = 25 },
        .{ .nt = 1037, .outs = 255 },
        .{ .nt = 16_384, .outs = 140, .host = false },
    };
    for (cases, 0..) |c, ci| {
        var name_buf: [64]u8 = undefined;
        // Where the case is when it fails: one FAILED line with the case, form, stage and MLX's own error text.
        var at: SelfCheckAt = .{
            .name = try std.fmt.bufPrint(&name_buf, "t{d}_outs{d}{s}", .{ c.nt, c.outs, if (c.merge) "_merged" else "" }),
            .tokens = c.nt,
            .sources = c.outs,
        };
        selfCheckCase(a, &g, &kernel, &inplace, c, ci, &at) catch |e| {
            selfCheckFailed(at, e);
            return e;
        };
    }
    std.debug.print("MIMO_PREFILL_COMBINE_SELFCHECK PASS\n", .{});
}
