//! HC post emulation probe (kbench, device only; DSV41_PHASE0B_MLX=1 and DSV41_HCPOST_EMUL_BENCH_V1=1): can one pass
//! reproduce the compiled HcPost region (`_hc_post_impl`: the einsum's NAX f32 GEMM, MLX_ENABLE_TF32 default, then the
//! fused `post * x + mixed` tail) word for word? Both prompt-width sites (hc1.h, out.h) run that region.
//! Per shape: the stock-vs-stock control (the region twice, the eager einsum twice), then each emulation variant's
//! mismatching words against the eager einsum (mixed) and the region's output (two tail forms), then timing of the
//! region against a production-shaped one-pass kernel. The lever is alive only on 0 mismatches for one variant at every
//! shape; no served-path change.

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("deepseek_v41_ops.zig");
const v41 = @import("deepseek_v41.zig");
const graph = @import("deepseek_v41_graph.zig");
const TraceOps = ops.TraceOps;

pub const dim = 5120;
pub const hc = 4;

/// Operand conversion before the products (the GEMM's A = comb, B = residual).
pub const Conv = enum(u8) { none = 0, rne = 1, trunc = 2 };
/// The 4-term sum over j: from a zero accumulator in j order, pairwise, (nearly) single-rounded, reversed, or an fma
/// chain (the rounding-class HCFUSE form, the negative control).
pub const Acc = enum(u8) { seq = 0, pair = 1, exact = 2, rev = 3, fma = 4 };
pub const Variant = struct { name: []const u8, conv: Conv, acc: Acc };

pub const variants = [_]Variant{
    .{ .name = "rne_seq", .conv = .rne, .acc = .seq },
    .{ .name = "rne_pair", .conv = .rne, .acc = .pair },
    .{ .name = "rne_exact", .conv = .rne, .acc = .exact },
    .{ .name = "rne_rev", .conv = .rne, .acc = .rev },
    .{ .name = "trunc_seq", .conv = .trunc, .acc = .seq },
    .{ .name = "trunc_pair", .conv = .trunc, .acc = .pair },
    .{ .name = "trunc_exact", .conv = .trunc, .acc = .exact },
    .{ .name = "f32_seq", .conv = .none, .acc = .seq },
    .{ .name = "f32_pair", .conv = .none, .acc = .pair },
    .{ .name = "f32_fma", .conv = .none, .acc = .fma },
};

/// Operand distributions: per-row scales log-uniform in [2^lo, 2^hi] (`elem`: per element instead).
pub const Dist = struct { lo: f32, hi: f32, elem: bool = false };
pub const ShapeCase = struct { name: []const u8, rows: c_int, l0: bool, dist: Dist, timed: bool };

/// f32_*: layers 1..39 (both sites); l0: layer 0's attention site (bf16 embedding rows broadcast over hc);
/// wide / tiny: exponent range and the denormal edge.
pub const shapes = [_]ShapeCase{
    .{ .name = "f32_953", .rows = 953, .l0 = false, .dist = .{ .lo = -4, .hi = 4 }, .timed = true },
    .{ .name = "f32_183", .rows = 183, .l0 = false, .dist = .{ .lo = -4, .hi = 4 }, .timed = true },
    .{ .name = "l0_953", .rows = 953, .l0 = true, .dist = .{ .lo = -4, .hi = 4 }, .timed = true },
    .{ .name = "wide_953", .rows = 953, .l0 = false, .dist = .{ .lo = -40, .hi = 40, .elem = true }, .timed = false },
    .{ .name = "tiny_953", .rows = 953, .l0 = false, .dist = .{ .lo = -124, .hi = -116 }, .timed = false },
};

/// Served calls per prompt pass of 16,384 tokens (17 chunks of 953 + one of 183, 40 layers, two sites; layer 0's
/// attention site on the l0 shape).
pub const served = struct {
    pub const full_chunks = 17;
    pub const layers = 40;
    pub fn calls953F32() f64 {
        return full_chunks * (2 * layers - 1);
    }
    pub fn calls183() f64 {
        return 2 * layers;
    }
    pub fn calls953L0() f64 {
        return full_chunks;
    }
};

const kernel_header =
    \\#pragma METAL fp contract(off)
    \\inline float hcx_tf32_rne(float v) {
    \\    uint u = as_type<uint>(v);
    \\    u = (u + 0x0FFFu + ((u >> 13) & 1u)) & 0xFFFFE000u;
    \\    return as_type<float>(u);
    \\}
    \\inline float hcx_tf32_trunc(float v) { return as_type<float>(as_type<uint>(v) & 0xFFFFE000u); }
    \\inline float hcx_conv(float v, int c) { return c == 1 ? hcx_tf32_rne(v) : (c == 2 ? hcx_tf32_trunc(v) : v); }
    \\inline float hcx_sum(float p0, float p1, float p2, float p3, int a) {
    \\    if (a == 1) return (p0 + p1) + (p2 + p3);
    \\    if (a == 3) { float m = 0.0f; m = m + p3; m = m + p2; m = m + p1; m = m + p0; return m; }
    \\    if (a == 2) {
    \\        float s = p0; float e = 0.0f; float q[3] = {p1, p2, p3};
    \\        for (int i = 0; i < 3; ++i) { float t = s + q[i]; float bb = t - s; float err = (s - (t - bb)) + (q[i] - bb); s = t; e = e + err; }
    \\        return s + e;
    \\    }
    \\    float m = 0.0f; m = m + p0; m = m + p1; m = m + p2; m = m + p3; return m;
    \\}
    \\
;

/// One thread = 4 consecutive columns of one row, every k. Inputs: x [rows, D] (f32), res [rows, hc, D] (RB 0) or
/// the un-broadcast rows [rows, D] (RB 1), post [rows, hc] f32, comb [rows, hc * hc] f32 (`...jk`: j * hc + k).
fn kernelBody(comptime probe: bool) [:0]const u8 {
    const head =
        \\    constexpr uint D = 5120;
        \\    const uint col = thread_position_in_grid.x * 4u;
        \\    const uint row = thread_position_in_grid.y;
        \\    if (col >= D) return;
        \\    float c[16];
        \\    for (int i = 0; i < 16; ++i) c[i] = hcx_conv(comb[row * 16u + i], CONV);
        \\    float r[4][4];
        \\    for (uint j = 0; j < 4u; ++j) for (uint i = 0; i < 4u; ++i) {
        \\        const ulong ri = RB ? ulong(row) * D + col + i : (ulong(row) * 4u + j) * D + col + i;
        \\        r[j][i] = hcx_conv(static_cast<float>(res[ri]), CONV);
        \\    }
        \\    for (uint k = 0; k < 4u; ++k) {
        \\        const float pk = post[row * 4u + k];
        \\        for (uint i = 0; i < 4u; ++i) {
        \\            const float p0 = c[k] * r[0][i];
        \\            const float p1 = c[4u + k] * r[1][i];
        \\            const float p2 = c[8u + k] * r[2][i];
        \\            const float p3 = c[12u + k] * r[3][i];
        \\            const float m = ACC == 4 ? fma(c[12u + k], r[3][i], fma(c[8u + k], r[2][i], fma(c[4u + k], r[1][i], p0))) : hcx_sum(p0, p1, p2, p3, ACC);
        \\            const ulong o = (ulong(row) * 4u + k) * D + col + i;
        \\            const float xv = static_cast<float>(x[ulong(row) * D + col + i]);
        \\            const float t = pk * xv;
        \\
    ;
    const tail = if (probe)
        \\            mixed[o] = m;
        \\            out_sep[o] = t + m;
        \\            out_fma[o] = fma(pk, xv, m);
        \\        }
        \\    }
        \\
    else
        \\            out[o] = t + m;
        \\        }
        \\    }
        \\
    ;
    return head ++ tail;
}

/// A bench-local MLX kernel launched at an explicit grid, with int template args.
const Custom = struct {
    k: mlx.mlx_fast_metal_kernel,
    n_out: usize,

    fn init(name: [:0]const u8, outputs: []const [*:0]const u8, src: [:0]const u8) !Custom {
        const ins = [_][*:0]const u8{ "x", "res", "post", "comb" };
        const vin = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(vin);
        const vout = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
        defer _ = mlx.mlx_vector_string_free(vout);
        const k = mlx.mlx_fast_metal_kernel_new(name.ptr, vin, vout, src.ptr, kernel_header, true, false);
        if (k.ctx == null) return error.KernelCreateFailed;
        return .{ .k = k, .n_out = outputs.len };
    }

    fn deinit(c: *Custom) void {
        _ = mlx.mlx_fast_metal_kernel_free(c.k);
    }

    fn launch(c: *const Custom, g: *ops.MlxOps, ins: []const mlx.mlx_array, rows: c_int, v: Variant, rb: bool, out: []mlx.mlx_array) !void {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const shape = [_]c_int{ rows, hc, dim };
        for (0..c.n_out) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CONV", @backingInt(v.conv)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ACC", @backingInt(v.acc)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RB", @intFromBool(rb)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, dim / 4, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        const vin = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
        defer _ = mlx.mlx_vector_array_free(vin);
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, c.k, vin, cfg, g.s));
        for (out[0..c.n_out], 0..) |*o, i| {
            var x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&x, outs, i));
            o.* = try g.adopt(x);
        }
    }
};

/// The probe's host data for one shape: x [rows, D] f32, the residual (f32 [rows, hc, D], or bf16 rows [rows, D] on
/// layer 0), post [rows, hc] in (0, 2), comb [rows, hc, hc] column-normalized in (0, 1).
const HostData = struct {
    x: []f32,
    res32: []f32,
    res16: []u16,
    post: []f32,
    comb: []f32,

    fn init(a: std.mem.Allocator, r: std.Random, sc: ShapeCase) !HostData {
        const n: usize = @intCast(sc.rows);
        var d: HostData = .{ .x = try a.alloc(f32, n * dim), .res32 = &.{}, .res16 = &.{}, .post = try a.alloc(f32, n * hc), .comb = try a.alloc(f32, n * hc * hc) };
        errdefer d.deinit(a);
        const scale = struct {
            fn f(rr: std.Random, ds: Dist) f32 {
                return std.math.exp2(ds.lo + (ds.hi - ds.lo) * rr.float(f32));
            }
        }.f;
        for (0..n) |row| {
            const sx = scale(r, sc.dist);
            for (d.x[row * dim ..][0..dim]) |*v| v.* = r.floatNorm(f32) * (if (sc.dist.elem) scale(r, sc.dist) else sx);
            for (d.post[row * hc ..][0..hc]) |*v| v.* = 2.0 / (1.0 + @exp(-r.floatNorm(f32)));
            const cb = d.comb[row * hc * hc ..][0 .. hc * hc];
            for (cb) |*v| v.* = r.float(f32) + 1e-3;
            for (0..hc) |k| {
                var s: f32 = 0;
                for (0..hc) |j| s += cb[j * hc + k];
                for (0..hc) |j| cb[j * hc + k] /= s;
            }
        }
        if (sc.l0) {
            d.res16 = try a.alloc(u16, n * dim);
            for (0..n) |row| {
                const s = scale(r, sc.dist);
                for (d.res16[row * dim ..][0..dim]) |*v| v.* = ops.bf16Bits(r.floatNorm(f32) * s);
            }
        } else {
            d.res32 = try a.alloc(f32, n * hc * dim);
            for (0..n * hc) |rj| {
                const s = scale(r, sc.dist);
                for (d.res32[rj * dim ..][0..dim]) |*v| v.* = r.floatNorm(f32) * (if (sc.dist.elem) scale(r, sc.dist) else s);
            }
        }
        return d;
    }

    fn deinit(d: *HostData, a: std.mem.Allocator) void {
        a.free(d.x);
        a.free(d.res32);
        a.free(d.res16);
        a.free(d.post);
        a.free(d.comb);
    }
};

/// The device leaves of one shape: the stock region's inputs (x [1, S, D], residual [1, S, hc, D] (a broadcast view on
/// layer 0), post [1, S, hc], comb [1, S, hc, hc]) and the one-pass kernel's (x [S, D], res, post [S, hc], comb [S, 16]).
fn Leaves(comptime T: type) type {
    return struct { x: T, res: T, post: T, comb: T, kx: T, kres: T, kpost: T, kcomb: T };
}

fn leaves(comptime G: type, g: *G, sc: ShapeCase, d: ?*const HostData) !Leaves(G.T) {
    const S = sc.rows;
    const leaf = struct {
        fn f(gg: *G, bytes: ?[]const u8, shape: []const c_int, dt: ops.Dtype) !G.T {
            if (comptime G == TraceOps) return gg.input(shape, dt);
            return gg.hostArray(bytes.?, shape, dt);
        }
    }.f;
    const b = std.mem.sliceAsBytes;
    const x = try leaf(g, if (d) |h| b(h.x) else null, &.{ 1, S, dim }, .float32);
    const post = try leaf(g, if (d) |h| b(h.post) else null, &.{ 1, S, hc }, .float32);
    const comb = try leaf(g, if (d) |h| b(h.comb) else null, &.{ 1, S, hc, hc }, .float32);
    var res: G.T = undefined;
    var kres: G.T = undefined;
    if (sc.l0) {
        const rows = try leaf(g, if (d) |h| b(h.res16) else null, &.{ 1, S, dim }, .bfloat16);
        res = try g.broadcastTo(try g.expandDims(rows, 2), &.{ 1, S, hc, dim });
        kres = try g.reshape(rows, &.{ S, dim });
    } else {
        res = try leaf(g, if (d) |h| b(h.res32) else null, &.{ 1, S, hc, dim }, .float32);
        kres = try g.reshape(res, &.{ S, hc, dim });
    }
    const l: Leaves(G.T) = .{ .x = x, .res = res, .post = post, .comb = comb, .kx = try g.reshape(x, &.{ S, dim }), .kres = kres, .kpost = try g.reshape(post, &.{ S, hc }), .kcomb = try g.reshape(comb, &.{ S, hc * hc }) };
    try g.evalAll(&.{ l.x, l.res, l.post, l.comb, l.kx, l.kres, l.kpost, l.kcomb });
    return l;
}

/// Mismatching f32 words of `got` against `want` (both evaluated, row-contiguous, `n` words) and the first index.
const Mm = struct { count: u64 = 0, first: i64 = -1, want_bits: u32 = 0, got_bits: u32 = 0 };

fn mismatches(want: mlx.mlx_array, got: mlx.mlx_array, n: usize) !Mm {
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    const pw: [*]const u32 = @ptrCast(mlx.mlx_array_data_float32(want) orelse return error.MlxError);
    const pg: [*]const u32 = @ptrCast(mlx.mlx_array_data_float32(got) orelse return error.MlxError);
    var count: u64 = 0;
    var first: i64 = -1;
    var wb: u32 = 0;
    var gb: u32 = 0;
    for (0..n) |i| if (pw[i] != pg[i]) {
        if (first < 0) {
            first = @intCast(i);
            wb = pw[i];
            gb = pg[i];
        }
        count += 1;
    };
    return .{ .count = count, .first = first, .want_bits = wb, .got_bits = gb };
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

var region_key: v41.Config = undefined;

pub const batch = 8;
pub const reps = 15;
pub const warm = 3;

fn bench(comptime G: type, g: *G, a: std.mem.Allocator, io: std.Io) !void {
    const Tr = graph.Trunk(G);
    const dry = G == TraceOps;
    const T = G.T;
    const t_all = std.Io.Timestamp.now(io, .boot);
    var rng = std.Random.DefaultPrng.init(0x4c9_0057);
    const r = rng.random();
    try g.prepareTape(Tr.HcPost, &region_key);
    var probe_k: if (dry) void else Custom = undefined;
    var fused_k: if (dry) void else Custom = undefined;
    if (!dry) {
        probe_k = try Custom.init("dsv41_hcx_probe", &.{ "mixed", "out_sep", "out_fma" }, comptime kernelBody(true));
        fused_k = try Custom.init("dsv41_hcx_fused", &.{"out"}, comptime kernelBody(false));
    }
    defer if (!dry) {
        probe_k.deinit();
        fused_k.deinit();
    };
    var stock_ms: [shapes.len]f64 = @splat(0);
    var fused_ms: [shapes.len]f64 = @splat(0);
    for (shapes, 0..) |sc, si| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        var host: ?HostData = if (dry) null else try HostData.init(a, r, sc);
        defer if (host) |*h| h.deinit(a);
        const l = try leaves(G, g, sc, if (host) |*h| h else null);
        const words: usize = @as(usize, @intCast(sc.rows)) * hc * dim;
        const stockOut = struct {
            fn f(gg: *G, ll: Leaves(T)) !T {
                var o: [1]T = undefined;
                try gg.tape(Tr.HcPost, &region_key, &.{ ll.x, ll.res, ll.post, ll.comb }, &o);
                return o[0];
            }
        }.f;
        const einsumOut = struct {
            fn f(gg: *G, ll: Leaves(T)) !T {
                return gg.einsum("...jk,...jd->...kd", &.{ ll.comb, try gg.astype(ll.res, .float32) });
            }
        }.f;
        // (1) The control: the region and the eager einsum, each twice on the same inputs.
        const s1 = try stockOut(g, l);
        const s2 = try stockOut(g, l);
        const e1 = try einsumOut(g, l);
        const e2 = try einsumOut(g, l);
        try g.evalAll(&.{ s1, s2, e1, e2 });
        if (!dry) {
            const cs = try mismatches(s1, s2, words);
            const ce = try mismatches(e1, e2, words);
            if (cs.count != 0 or ce.count != 0) {
                std.debug.print("NATIVE HCPOSTX_CONTROL_MISMATCH {{\"shape\": \"{s}\", \"stock_mismatch\": {d}, \"einsum_mismatch\": {d}}}\n", .{ sc.name, cs.count, ce.count });
                return error.HcPostControlNotExact;
            }
        }
        std.debug.print("NATIVE HCPOSTX_CONTROL {{\"shape\": \"{s}\", \"rows\": {d}, \"words\": {d}, \"stock_mismatch\": 0, \"einsum_mismatch\": 0}}\n", .{ sc.name, sc.rows, words });
        // (2) Each variant: its mixed against the eager einsum, its two tails against the region.
        for (variants) |v| {
            const mv = g.mark();
            defer g.resetTo(mv);
            var o: [3]T = undefined;
            if (dry) {
                for (&o) |*x| x.* = try g.input(&.{ sc.rows, hc, dim }, .float32);
            } else try probe_k.launch(g, &.{ l.kx, l.kres, l.kpost, l.kcomb }, sc.rows, v, sc.l0, &o);
            try g.evalAll(&o);
            var mm: [3]Mm = @splat(.{});
            if (!dry) {
                mm[0] = try mismatches(e1, o[0], words);
                mm[1] = try mismatches(s1, o[1], words);
                mm[2] = try mismatches(s1, o[2], words);
            }
            std.debug.print("NATIVE HCPOSTX_VARIANT {{\"shape\": \"{s}\", \"variant\": \"{s}\", \"words\": {d}, \"mixed_mismatch\": {d}, \"out_sep_mismatch\": {d}, \"out_fma_mismatch\": {d}, \"mixed_first\": {d}, \"mixed_first_want\": \"0x{x:0>8}\", \"mixed_first_got\": \"0x{x:0>8}\", \"out_sep_first\": {d}}}\n", .{ sc.name, v.name, words, mm[0].count, mm[1].count, mm[2].count, mm[0].first, mm[0].want_bits, mm[0].got_bits, mm[1].first });
        }
        // (3) Timing (the served shapes): `batch` calls per evaluation, median of `reps` after `warm`.
        if (sc.timed) {
            for (0..2) |arm| {
                var samples: [reps]f64 = undefined;
                for (0..warm + reps) |rep| {
                    const mt = g.mark();
                    defer g.resetTo(mt);
                    var outs: [batch]T = undefined;
                    const t0 = std.Io.Timestamp.now(io, .boot);
                    for (&outs) |*x| {
                        if (arm == 0) {
                            x.* = try stockOut(g, l);
                        } else if (dry) {
                            x.* = try g.input(&.{ sc.rows, hc, dim }, .float32);
                        } else {
                            var one: [1]T = undefined;
                            try fused_k.launch(g, &.{ l.kx, l.kres, l.kpost, l.kcomb }, sc.rows, variants[0], sc.l0, &one);
                            x.* = one[0];
                        }
                    }
                    try g.evalAll(&outs);
                    const ns: f64 = @floatFromInt(t0.untilNow(io, .boot).nanoseconds);
                    if (rep >= warm) samples[rep - warm] = ns / 1e6 / batch;
                }
                const med = median(&samples);
                if (arm == 0) stock_ms[si] = med else fused_ms[si] = med;
            }
            std.debug.print("NATIVE HCPOSTX_TIME {{\"shape\": \"{s}\", \"rows\": {d}, \"stock_ms\": {d:.5}, \"fused_ms\": {d:.5}, \"delta_ms\": {d:.5}, \"batch\": {d}, \"reps\": {d}}}\n", .{ sc.name, sc.rows, stock_ms[si], fused_ms[si], fused_ms[si] - stock_ms[si], batch, reps });
        }
    }
    try sharedMidPart(G, g, a, r, io);
    const per = struct {
        fn f(v: *const [shapes.len]f64) f64 {
            return (served.calls953F32() * v[0] + served.calls183() * v[1] + served.calls953L0() * v[2]) / 1e3;
        }
    }.f;
    std.debug.print("NATIVE HCPOSTX_SERVED {{\"calls\": {d}, \"stock_s_per_prefill\": {d:.4}, \"fused_s_per_prefill\": {d:.4}, \"delta_s_per_prefill\": {d:.4}}}\n", .{ served.calls953F32() + served.calls183() + served.calls953L0(), per(&stock_ms), per(&fused_ms), per(&fused_ms) - per(&stock_ms) });
    std.debug.print("NATIVE HCPOSTX_WALL {{\"s\": {d:.2}}}\n", .{@as(f64, @floatFromInt(t_all.untilNow(io, .boot).nanoseconds)) / 1e9});
}

/// C22's SharedMid region (the shared expert's clamp / silu / product, compiled) at prompt widths against the eager chain
/// `sharedExpertQ` runs there today (f32 stream): mismatching words and time per call. Its served use would be a
/// prompt-width route; the decode verdict (dispatch count at M <= 8) did not price the bandwidth of [rows, I] f32 passes.
pub const mid_rows = [_]c_int{ 953, 183 };
var mid_key: v41.Config = undefined;

fn sharedMidPart(comptime G: type, g: *G, a: std.mem.Allocator, r: std.Random, io: std.Io) !void {
    const Tr = graph.Trunk(G);
    const T = G.T;
    const dry = G == TraceOps;
    const inter: c_int = 2304;
    mid_key.swiglu_limit = 10.0;
    try g.prepareTape(Tr.SharedMid, &mid_key);
    var ms: [mid_rows.len][2]f64 = undefined;
    for (mid_rows, 0..) |rows, ri| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const n: usize = @as(usize, @intCast(rows)) * @as(usize, @intCast(inter));
        var gate: T = undefined;
        var up: T = undefined;
        if (dry) {
            gate = try g.input(&.{ 1, rows, inter }, .float32);
            up = try g.input(&.{ 1, rows, inter }, .float32);
        } else {
            const buf = try a.alloc(f32, 2 * n);
            defer a.free(buf);
            for (buf) |*v| v.* = r.floatNorm(f32) * 6.0;
            gate = try g.hostArray(std.mem.sliceAsBytes(buf[0..n]), &.{ 1, rows, inter }, .float32);
            up = try g.hostArray(std.mem.sliceAsBytes(buf[n..]), &.{ 1, rows, inter }, .float32);
        }
        const x = try g.zeros(&.{ 1, rows, dim }, .float32);
        try g.evalAll(&.{ gate, up, x });
        const Arm = struct {
            fn eager(gg: *G, gt: T, u: T) !T {
                const lim = mid_key.swiglu_limit;
                var gf = try gg.astype(gt, .float32);
                var uf = try gg.astype(u, .float32);
                uf = try gg.clip(uf, try gg.scalar(-lim, .float32), try gg.scalar(lim, .float32));
                gf = try gg.minimum(gf, try gg.scalar(lim, .float32));
                return gg.astype(try gg.mul(try gg.silu(gf), uf), .float32);
            }
            fn compiled(gg: *G, gt: T, u: T, xx: T) !T {
                var o: [1]T = undefined;
                try gg.tape(Tr.SharedMid, &mid_key, &.{ gt, u, xx }, &o);
                return o[0];
            }
        };
        const e = try Arm.eager(g, gate, up);
        const c = try Arm.compiled(g, gate, up, x);
        try g.evalAll(&.{ e, c });
        var mm: Mm = .{};
        if (!dry) mm = try mismatches(e, c, n);
        for (0..2) |arm| {
            var samples: [reps]f64 = undefined;
            for (0..warm + reps) |rep| {
                const mt = g.mark();
                defer g.resetTo(mt);
                var outs: [batch]T = undefined;
                const t0 = std.Io.Timestamp.now(io, .boot);
                for (&outs) |*o| o.* = if (arm == 0) try Arm.eager(g, gate, up) else try Arm.compiled(g, gate, up, x);
                try g.evalAll(&outs);
                const ns: f64 = @floatFromInt(t0.untilNow(io, .boot).nanoseconds);
                if (rep >= warm) samples[rep - warm] = ns / 1e6 / batch;
            }
            ms[ri][arm] = median(&samples);
        }
        std.debug.print("NATIVE SHAREDMIDX {{\"rows\": {d}, \"words\": {d}, \"mismatch\": {d}, \"first\": {d}, \"eager_ms\": {d:.5}, \"compiled_ms\": {d:.5}, \"delta_ms\": {d:.5}}}\n", .{ rows, n, mm.count, mm.first, ms[ri][0], ms[ri][1], ms[ri][1] - ms[ri][0] });
    }
    const calls953: f64 = served.full_chunks * served.layers;
    const calls183: f64 = served.layers;
    std.debug.print("NATIVE SHAREDMIDX_SERVED {{\"calls\": {d}, \"eager_s_per_prefill\": {d:.4}, \"compiled_s_per_prefill\": {d:.4}, \"delta_s_per_prefill\": {d:.4}}}\n", .{ calls953 + calls183, (calls953 * ms[0][0] + calls183 * ms[1][0]) / 1e3, (calls953 * ms[0][1] + calls183 * ms[1][1]) / 1e3, (calls953 * (ms[0][1] - ms[0][0]) + calls183 * (ms[1][1] - ms[1][0])) / 1e3 });
}

const testing = std.testing;

test "dsv41 hcpost emul bench 0b: one-pass emulations of the compiled HC post against the region, word for word, and its time (MLX, GPU stream)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    _ = std.c.getenv("DSV41_HCPOST_EMUL_BENCH_V1") orelse return error.SkipZigTest;
    const G = ops.MlxOps;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(testing.allocator, s);
    defer g.deinit();
    try bench(G, &g, testing.allocator, testing.io);
}

test "dsv41 hcpost emul bench: the step on the trace backend (shapes, the region's signature, every line)" {
    var g = TraceOps.init(testing.allocator);
    defer g.deinit();
    try bench(TraceOps, &g, testing.allocator, testing.io);
    // One region trace per signature: HcPost f32 at 953 and 183 rows and layer 0's bf16 residual at 953; SharedMid at 953, 183.
    try testing.expectEqual(@as(usize, 5), g.compiles);
}

test "dsv41 hcpost emul bench: the served call count is the prompt pass's (17 x 953 + 183 rows, 40 layers, two sites)" {
    try testing.expectEqual(@as(f64, 2 * 40 * 18), served.calls953F32() + served.calls183() + served.calls953L0());
}
