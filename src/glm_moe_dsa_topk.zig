//! GLM-5.3's indexer top-k on the GPU: each row's `k` largest scores as MLX's `argpartition(scores, n - k)[..., n - k:]`
//! gives them, in one launch. MLX partitions by a stable merge sort of the whole row: the last `k` of the stable
//! ascending order, so the `k` largest values, ties at the threshold taken from the highest indices, in ascending
//! (value, index) order. The kernel finds the `k`-th largest of the composite key (the value's order key, then the
//! index) by a radix select over the row, keeps the `k` elements at or above it, and sorts them ascending: the same ids
//! in the same order. Zeros of either sign tie, as they compare equal; a NaN (the indexer writes none) sorts as its
//! bits do. Not the host's glm5_next TOPK_EXPAND: that selects one decode query's pooled scores and writes its pools
//! in chronological order (ties to the lower index); the prompt needs argpartition's ids, in its order, for every row.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("arch/deepseek_v41/ops.zig");

const G = ops.MlxOps;
const T = G.T;

/// One threadgroup of 1024 threads per row; `K` (a power of two up to 2048) a template argument. `topk`'s rows lie in
/// one buffer (`scores`) at one length (`params[0]`); `topkRows`' row `r` is its own input `s<r>` at length `params[r]`.
const source: [:0]const u8 =
    \\    const uint n = uint(params[0]);
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const device float* srow = scores + ulong(row) * ulong(n);
++ "\n" ++ body;

fn rowsSource(comptime r: usize) [:0]const u8 {
    return comptime blk: {
        @setEvalBranchQuota(200_000);
        var cases: []const u8 = "";
        for (1..r) |i| cases = cases ++ std.fmt.comptimePrint("        case {d}: srow = s{d}; break;\n", .{ i, i });
        break :blk std.fmt.comptimePrint(
            \\    const uint row = threadgroup_position_in_grid.y;
            \\    const uint n = uint(params[row]);
            \\    const device float* srow = s0;
            \\    switch (row) {{
            \\{s}        default: break;
            \\    }}
            \\{s}
        , .{ cases, body });
    };
}

const body =
    \\    const uint tid = thread_position_in_threadgroup.x;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    threadgroup atomic_uint hist[2048];
    \\    threadgroup uint part[32];
    \\    threadgroup uint st[6];
    \\    threadgroup atomic_uint taken;
    \\    threadgroup ulong sel[K];
    \\    // st: the composite prefix (value key, index), its masks, the elements still to take.
    \\    if (tid == 0) {
    \\        st[0] = 0; st[1] = 0; st[2] = 0; st[3] = 0; st[4] = uint(K); st[5] = 0;
    \\        atomic_store_explicit(&taken, 0u, memory_order_relaxed);
    \\    }
    \\    uint ibits = 0;
    \\    while ((1u << ibits) < n) ibits++;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // Passes over the composite key from its top: the value key in 11 + 11 + 10 bits, then the index in 11-bit
    \\    // digits (only while the threshold's ties outnumber the elements still to take).
    \\    const uint passes = 3 + (ibits + 10) / 11;
    \\    for (uint p = 0; p < passes; ++p) {
    \\        const bool on_value = p < 3;
    \\        const uint width = on_value ? (p < 2 ? 11u : 10u) : min(11u, ibits - (p - 3) * 11u);
    \\        const uint shift = on_value ? (p == 0 ? 21u : (p == 1 ? 10u : 0u)) : ibits - (p - 3) * 11u - width;
    \\        const uint nbins = 1u << width;
    \\        if (p >= 3 && st[5] == 1u) break;
    \\        for (uint b = tid; b < 2048u; b += 1024u) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        const uint pv = st[0], mv = st[1], pi = st[2], mi = st[3];
    \\        for (uint i = tid; i < n; i += 1024u) {
    \\            const float f = srow[i];
    \\            uint u = as_type<uint>(f);
    \\            if (f == 0.0f) u = 0u;
    \\            const uint kv = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\            if ((kv & mv) != pv || (i & mi) != pi) continue;
    \\            const uint d = ((on_value ? kv : i) >> shift) & (nbins - 1u);
    \\            atomic_fetch_add_explicit(&hist[d], 1u, memory_order_relaxed);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        // Thread t holds bins nbins - 1 - 2t and nbins - 2 - 2t: its inclusive prefix counts the elements from the
    \\        // top bin down to its lower bin.
    \\        const int bh = int(nbins) - 1 - 2 * int(tid);
    \\        const int bl = bh - 1;
    \\        const uint ch = bh >= 0 ? atomic_load_explicit(&hist[bh], memory_order_relaxed) : 0u;
    \\        const uint cl = bl >= 0 ? atomic_load_explicit(&hist[bl], memory_order_relaxed) : 0u;
    \\        const uint s = ch + cl;
    \\        const uint inc = simd_prefix_inclusive_sum(s);
    \\        if (lane == 31) part[sg] = inc;
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint before = 0;
    \\        for (uint k = 0; k < sg; ++k) before += part[k];
    \\        const uint upto = before + inc;
    \\        const uint above = upto - s;
    \\        const uint need = st[4];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (above < need && upto >= need) {
    \\            uint bin, left, ties;
    \\            if (above + ch >= need) { bin = uint(bh); left = need - above; ties = ch; }
    \\            else { bin = uint(bl); left = need - above - ch; ties = cl; }
    \\            if (on_value) { st[0] = pv | (bin << shift); st[1] = mv | ((nbins - 1u) << shift); }
    \\            else { st[2] = pi | (bin << shift); st[3] = mi | ((nbins - 1u) << shift); }
    \\            st[4] = left;
    \\            // Every element of the bin taken: no lower digit decides among them.
    \\            st[5] = ties == left ? 1u : 0u;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    // The k elements at or above the threshold: the prefix's elements whose remaining digits are all taken (the
    \\    // threshold's bin, kept whole) and every element whose composite key is above the prefix.
    \\    const uint pv = st[0], mv = st[1], pi = st[2], mi = st[3];
    \\    for (uint i = tid; i < n; i += 1024u) {
    \\        const float f = srow[i];
    \\        uint u = as_type<uint>(f);
    \\        if (f == 0.0f) u = 0u;
    \\        const uint kv = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\        const bool hi_v = (kv & mv) > pv;
    \\        const bool eq_v = (kv & mv) == pv;
    \\        const bool keep = hi_v || (eq_v && ((i & mi) >= pi));
    \\        if (keep) {
    \\            const uint at = atomic_fetch_add_explicit(&taken, 1u, memory_order_relaxed);
    \\            if (at < uint(K)) sel[at] = (ulong(kv) << 32) | ulong(i);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // Ascending (value, index): a bitonic sort of the k composite keys.
    \\    for (uint size = 2; size <= uint(K); size <<= 1) {
    \\        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
    \\            for (uint t = tid; t < uint(K) / 2u; t += 1024u) {
    \\                const uint i = (t / stride) * 2u * stride + (t % stride);
    \\                const uint j = i + stride;
    \\                const bool up = (i & size) == 0u;
    \\                const ulong a = sel[i];
    \\                const ulong b = sel[j];
    \\                if ((a > b) == up) { sel[i] = b; sel[j] = a; }
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\    for (uint j = tid; j < uint(K); j += 1024u) out[ulong(row) * uint(K) + j] = int(uint(sel[j] & 0xffffffffu));
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
/// `topkRows`' kernels, one per row count (its inputs `s0 ..`).
var rows_kernels: [max_rows]?mlx.mlx_fast_metal_kernel = @splat(null);

/// The most rows `topkRows` takes in one launch: its lengths ride in `params`' eight slots (a verify has at most
/// `max_route_ids / n_experts_per_tok` = 6 rows).
pub const max_rows = 8;

/// Whether the kernel serves a top `k`: a power of two up to 2048 (the memory bill charges the selection by it).
pub fn serves(k: u32) bool {
    return k > 0 and k <= 2048 and std.math.isPowerOfTwo(k);
}

/// The ids `[1, rows, k]` (int32) of each row's `k` largest scores (`scores` `[1, rows, n]` f32, row-contiguous,
/// `k < n`), as `argpartition(scores, n - k)[..., n - k:]` orders them; null where the kernel does not serve the
/// shape (`k` not a power of two up to 2048, or a row of 2^31 or more) or the stream is not the GPU's.
pub fn topk(g: *G, scores: T, k: u32) !?T {
    if (!mlx.streamIsGpu(g.s)) return null;
    const sh = g.shapeOf(scores);
    if (sh.n != 3 or sh.dim(0) != 1) return null;
    const rows = sh.dim(1);
    const n = sh.dim(2);
    if (!serves(k) or n <= k or g.dtypeOf(scores) != .float32) return null;
    const kern = kernel orelse blk: {
        const ins = [_][*:0]const u8{ "scores", "params" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = mlx.mlx_fast_metal_kernel_new("glm_moe_dsa_topk", iv, ov, source, "", true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        kernel = kn;
        break :blk kn;
    };
    // Eight slots: an input of fewer is bound in the constant address space.
    const pv = [8]i32{ n, 0, 0, 0, 0, 0, 0, 0 };
    const params = try g.hostArray(std.mem.sliceAsBytes(&pv), &.{8}, .int32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, @intCast(k) }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(k)));
    const inputs = [_]mlx.mlx_array{ scores, params };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&out, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return try g.adopt(out);
}

/// The ids `[rows.len, k]` (int32) of each row's `k` largest scores, `rows[r]` one row of `n_r > k` f32 scores
/// (`[..., n_r]`, every leading dim 1; the lengths may differ), as `topk` gives them, every row in one launch (a row's
/// threadgroup reads its own input); null where the kernel does not serve the shape or the stream is not the GPU's.
pub fn topkRows(g: *G, rows: []const T, k: u32) !?T {
    std.debug.assert(rows.len > 0 and rows.len <= max_rows);
    if (!serves(k) or !mlx.streamIsGpu(g.s)) return null;
    var pv: [8]i32 = @splat(0);
    for (rows, 0..) |r, i| {
        const sh = g.shapeOf(r);
        const n = sh.dim(-1);
        if (sh.numel() != n or n <= k or g.dtypeOf(r) != .float32) return null;
        pv[i] = n;
    }
    inline for (1..max_rows + 1) |nr| if (rows.len == nr) return try launchRows(nr, g, rows, &pv, k);
    unreachable;
}

fn launchRows(comptime nr: usize, g: *G, rows: []const T, pv: *const [8]i32, k: u32) !T {
    const kern = rows_kernels[nr - 1] orelse blk: {
        const ins = comptime names: {
            var v: [nr + 1][*:0]const u8 = undefined;
            for (0..nr) |i| v[i] = std.fmt.comptimePrint("s{d}", .{i});
            v[nr] = "params";
            break :names v;
        };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = mlx.mlx_fast_metal_kernel_new(std.fmt.comptimePrint("glm_moe_dsa_topk_rows{d}", .{nr}), iv, ov, comptime rowsSource(nr), "", true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        rows_kernels[nr - 1] = kn;
        break :blk kn;
    };
    const params = try g.hostArray(std.mem.sliceAsBytes(pv), &.{8}, .int32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @intCast(nr), @intCast(k) }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, @intCast(nr), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(k)));
    var inputs: [nr + 1]mlx.mlx_array = undefined;
    @memcpy(inputs[0..nr], rows);
    inputs[nr] = params;
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&out, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return try g.adopt(out);
}

const testing = std.testing;

test "glm topk: each row's k largest scores in argpartition's ids and order, ties, both zeros and masked tails included" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();
    var checked: usize = 0;
    // (rows, n, k, case): distinct values; values on a coarse grid (many ties at the threshold); a masked tail of -1e30
    // past each row's causal keys (fewer than k for some rows); zeros of both signs.
    const cases = [_]struct { rows: u32, n: u32, k: u32, kind: u8 }{
        .{ .rows = 33, .n = 5000, .k = 2048, .kind = 0 },
        .{ .rows = 17, .n = 3000, .k = 2048, .kind = 1 },
        .{ .rows = 40, .n = 4100, .k = 2048, .kind = 2 },
        .{ .rows = 9, .n = 2500, .k = 2048, .kind = 3 },
        .{ .rows = 64, .n = 300, .k = 8, .kind = 1 },
        .{ .rows = 5, .n = 70000, .k = 2048, .kind = 0 },
    };
    for (cases) |cs| {
        const m = g.mark();
        defer g.resetTo(m);
        const v = try a.alloc(f32, cs.rows * cs.n);
        defer a.free(v);
        for (v, 0..) |*x, i| {
            const row = i / cs.n;
            const col = i % cs.n;
            x.* = switch (cs.kind) {
                0 => rand.floatNorm(f32),
                1 => @floor(rand.floatNorm(f32) * 4) / 4,
                2 => if (col > 1800 + row * 37) -1e30 else rand.floatNorm(f32),
                else => switch (rand.uintLessThan(u8, 4)) {
                    0 => 0.0,
                    1 => -0.0,
                    else => rand.floatNorm(f32),
                },
            };
        }
        const sc = try g.hostArray(std.mem.sliceAsBytes(v), &.{ 1, @intCast(cs.rows), @intCast(cs.n) }, .float32);
        const got = (try topk(&g, sc, cs.k)).?;
        const kth: c_int = @intCast(cs.n - cs.k);
        const want = try g.astype(try lastSlice(&g, try g.argpartition(sc, kth, -1), kth, @intCast(cs.n)), .int32);
        const gv = try a.alloc(u32, cs.rows * cs.k);
        defer a.free(gv);
        const wv = try a.alloc(u32, cs.rows * cs.k);
        defer a.free(wv);
        _ = try g.hostU32(try g.astype(got, .uint32), gv);
        _ = try g.hostU32(try g.astype(want, .uint32), wv);
        try testing.expectEqualSlices(u32, wv, gv);
        checked += gv.len;
    }
    std.debug.print("glm topk: {d} ids equal argpartition's, in its order\n", .{checked});
}

test "glm topk: rows of their own lengths in one launch, each its k largest in argpartition's ids and order (ties, both zeros)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(19);
    const rand = prng.random();
    const k: u32 = 2048;
    // A verify's lengths (one key more per row), and rows of other lengths on a coarse grid (ties at the threshold)
    // with zeros of both signs; one row up to max_rows rows.
    const sets = [_][]const u32{ &.{ 16900, 16901, 16902, 16903 }, &.{ 2049, 3000, 5000, 2100, 70000, 4097, 2500, 9000 }, &.{6000} };
    var checked: usize = 0;
    for (sets, 0..) |lens, si| {
        const m = g.mark();
        defer g.resetTo(m);
        var rows: [max_rows]T = undefined;
        var want: [max_rows]T = undefined;
        for (lens, 0..) |n, r| {
            const v = try a.alloc(f32, n);
            defer a.free(v);
            for (v) |*x| x.* = switch (si) {
                0 => rand.floatNorm(f32),
                else => switch (rand.uintLessThan(u8, 4)) {
                    0 => 0.0,
                    1 => -0.0,
                    else => @floor(rand.floatNorm(f32) * 4) / 4,
                },
            };
            rows[r] = try g.hostArray(std.mem.sliceAsBytes(v), &.{ 1, 1, 1, @intCast(n) }, .float32);
            const kth: c_int = @intCast(n - k);
            want[r] = try g.astype(try lastSlice(&g, try g.reshape(try g.argpartition(rows[r], kth, -1), &.{ 1, 1, @intCast(n) }), kth, @intCast(n)), .uint32);
        }
        const got = (try topkRows(&g, rows[0..lens.len], k)).?;
        try testing.expect(g.dtypeOf(got) == .int32);
        const gv = try a.alloc(u32, lens.len * k);
        defer a.free(gv);
        _ = try g.hostU32(try g.astype(got, .uint32), gv);
        for (0..lens.len) |r| {
            var wv: [2048]u32 = undefined;
            _ = try g.hostU32(want[r], &wv);
            try testing.expectEqualSlices(u32, &wv, gv[r * k ..][0..k]);
            checked += k;
        }
    }
    std.debug.print("glm topk rows: {d} ids equal argpartition's, in its order\n", .{checked});
}

fn lastSlice(g: *G, x: T, lo: c_int, hi: c_int) !T {
    const s = g.shapeOf(x);
    return g.slice(x, &.{ 0, 0, lo }, &.{ s.dim(0), s.dim(1), hi }, &.{ 1, 1, 1 });
}
