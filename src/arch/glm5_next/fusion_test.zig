//! The trunk launch fusion's CPU tests (fusion.zig; no device: each ends on `sdk.testing.expectNoDevice`):
//!   - the texts are the pinned ones, and the segment function is mimo_fp8_gemv.metal's code line for line;
//!   - the fused launches' geometry at Flash's sites (from the bound plans at M = 1 .. 8) is the manifest's (the offline
//!     compile's instantiations) and the prepared configs carry it;
//!   - the fused texts' thread maps, transliterated: every output of every segment written once, and each one bit for bit
//!     the stock launch's sum (MiMo's KS-order emulation `hostGemvDotW` at the site's plan), on random codes, scales and
//!     rows at every R a width takes;
//!   - the launches the routes build (the graph's custom-kernel nodes over placeholder inputs): a KDA layer's input
//!     projections 6 stock, 1 fused; an MLP's gate / up 2 and 1; the predictor's product and epilogue 4 and 2 (every
//!     node that dispatches); a routed call's join and combine 2 and 1; the same outputs (shapes, dtype) either way;
//!   - the install's plan refuses a BF16 site, an off-shape site and a plan one launch cannot hold, by name.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const glm = @import("config.zig");
const fp8 = @import("fp8.zig");
const kda = @import("kda.zig");
const graph = @import("graph.zig");
const swiglu = @import("swiglu.zig");
const fusion = @import("fusion.zig");
const experts = @import("experts.zig");
const glue = @import("glue.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const mimo_fp8 = @import("../mimo_v2/fp8_linear.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = fusion.G;
const Gemv = mimo_fp8.Gemv;

extern "c" fn mlxlc_test_config(config: mlx.mlx_fast_metal_kernel_config, out: *[9]c_int) void;
extern "c" fn mlxlc_test_gpu_stream() mlx.mlx_stream;
extern "c" fn mlxlc_test_placeholder(shape: [*]const c_int, ndim: c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream, out: *mlx.mlx_array) c_int;
extern "c" fn mlxlc_test_count_launches(outputs: mlx.mlx_vector_array) c_int;
extern "c" fn mlxlc_test_graph_ops(outputs: mlx.mlx_vector_array, out: [*]u8, cap: c_int) c_int;

test "glm5_next fusion: the embedded manifest and texts are the pinned ones; the segment function is mimo_fp8_gemv.metal's code line for line but its four kernel-scope lines" {
    try fusion.checkPins();
    try kset.expectManifestPins(fusion.texts.manifest, &fusion.text_pins, 4);
    // MiMo's text, its code lines (no comments), in order, against the function's body.
    const swapped = [_][]const u8{
        "    const int sgi = int(simdgroup_index_in_threadgroup);",
        "    const int lane = int(thread_index_in_simdgroup);",
        "    threadgroup float part[KS > 1 ? RG * KS * M * R : 1];",
    };
    const n0_stock = "    const int n0 = (int(threadgroup_position_in_grid.y) * RG + rg) * R;";
    const n0_seg = "    const int n0 = (tg * RG + rg) * R;";
    const header = fusion.texts.header;
    const open = std.mem.indexOf(u8, header, "    threadgroup float* part) {\n") orelse return error.TestUnexpectedResult;
    const body = header[open + "    threadgroup float* part) {\n".len ..];
    var hl = std.mem.splitScalar(u8, body, '\n');
    var stock = std.mem.splitScalar(u8, mimo_fp8.texts.gemv, '\n');
    var matched: usize = 0;
    var skipped: usize = 0;
    while (stock.next()) |line| {
        if (line.len == 0 or std.mem.startsWith(u8, std.mem.trimStart(u8, line, " "), "//")) continue;
        if (for (swapped) |sw| {
            if (std.mem.eql(u8, line, sw)) break true;
        } else false) {
            skipped += 1;
            continue;
        }
        const want = if (std.mem.eql(u8, line, n0_stock)) n0_seg else line;
        const got = hl.next() orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(want, got);
        matched += 1;
    }
    // Then only the function's close.
    try testing.expectEqualStrings("}", hl.next().?);
    try testing.expectEqualStrings("", hl.next().?);
    try testing.expect(hl.next() == null);
    try testing.expectEqual(@as(usize, 3), skipped);
    try testing.expect(matched > 80);
    // The fused texts dispatch the segment function only (no arithmetic of their own).
    for ([_][]const u8{ fusion.texts.kda_in, fusion.texts.pair }) |t| {
        for ([_][]const u8{ "fma(", "simd_shuffle", "static_cast<bfloat16_t>" }) |op| try testing.expect(std.mem.indexOf(u8, t, op) == null);
    }
    try testing.expectEqual(@as(usize, 6), std.mem.count(u8, fusion.texts.kda_in, "glm5n_fp8_gemv_rows<"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, fusion.texts.pair, "glm5n_fp8_gemv_rows<"));
    // The predictor's epilogue: MLX's Sigmoid line for line (mlx/backend/metal/kernels/unary_ops.h at the pinned MLX:
    // `auto y = 1 / (1 + metal::precise::exp(metal::abs(x))); return (x < 0) ? y : 1 - y;`, T = float) between the exact
    // widening and the f32 add of the bias.
    for ([_][]const u8{
        "const float x = static_cast<float>(logits[i]);",
        "const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));",
        "const float s = (x < 0) ? y : 1 - y;",
        "scores[i] = s + correction[i % E];",
    }) |line| try testing.expect(std.mem.indexOf(u8, fusion.texts.pred, line) != null);
    // The indexed combine: glue.zig's combine text (glm5n_moe_combine.metal) code line for line but its row reads, which
    // take the join's order (the rows the take would copy, read in place).
    {
        const swaps = [_][2][]const u8{
            .{ "const device auto* r = rows + size_t(t) * TOPK * H + h;", "const device auto* o = order + size_t(t) * TOPK;" },
            .{ "float acc = float(r[0]) * wt[0] + 0.0f;", "float acc = float(rows[size_t(o[0]) * H + h]) * wt[0] + 0.0f;" },
            .{ "  const float p = float(r[size_t(k) * H]) * wt[k] + 0.0f;", "  const float p = float(rows[size_t(o[k]) * H + h]) * wt[k] + 0.0f;" },
        };
        var stock_lines = std.mem.splitScalar(u8, glue.texts.combine, '\n');
        var idx_lines = std.mem.splitScalar(u8, fusion.texts.combine_idx, '\n');
        var code_lines: usize = 0;
        var n_swapped: usize = 0;
        while (stock_lines.next()) |line| {
            if (line.len == 0 or std.mem.startsWith(u8, line, "//")) continue;
            const want = for (swaps) |sw| {
                if (std.mem.eql(u8, line, sw[0])) {
                    n_swapped += 1;
                    break sw[1];
                }
            } else line;
            const got = while (idx_lines.next()) |l| {
                if (l.len != 0 and !std.mem.startsWith(u8, l, "//")) break l;
            } else return error.TestUnexpectedResult;
            try testing.expectEqualStrings(want, got);
            code_lines += 1;
        }
        try testing.expectEqual(@as(usize, 3), n_swapped);
        try testing.expect(code_lines >= 11);
    }
    try sdk.testing.expectNoDevice();
}

/// Flash's KDA input sites and MLP gate / up as the residents route binds them (fp8.zig `bind`: the plans per width),
/// over a route with no kernels (no MLX object is made).
const Sites = struct {
    route: fp8.Route = .{ .kernels = .{} },

    fn lin(s: *Sites, n: u32, k: u32) !fp8.Linear {
        return .{ .fp8 = try fp8.bind(&s.route, .{ .ctx = null }, .{ .ctx = null }, n, k) };
    }
};

test "glm5_next fusion: the fused launches' geometry at Flash's sites (the bound plans at M = 1 .. 8) is the manifest's instantiations, and the prepared configs carry it" {
    const c: glm.Config = .{};
    var s: Sites = .{};
    const q = (try s.lin(c.kdaDim(), c.hidden_size)).fp8;
    const fa = (try s.lin(c.kda_head_dim, c.hidden_size)).fp8;
    const b = (try s.lin(c.kda_heads, c.hidden_size)).fp8;
    const dense = (try s.lin(c.intermediate_size, c.hidden_size)).fp8;
    const shared = (try s.lin(c.moe_intermediate_size * c.n_shared_experts, c.hidden_size)).fp8;
    // f_a and b share a plan (both one threadgroup of 8 partitions); q / k / v theirs.
    for (1..fusion.max_rows + 1) |m| try testing.expectEqual(fa.plans[m], b.plans[m]);
    const J = struct { kernels: []const struct { name: []const u8, instances: []const std.json.Value } };
    const parsed = try std.json.parseFromSlice(J, testing.allocator, fusion.texts.manifest, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    try testing.expectEqualStrings("glm5n_fp8_gemv_kda_in", ks[0].name);
    try testing.expectEqualStrings("glm5n_fp8_gemv_pair", ks[1].name);
    try testing.expectEqualStrings("glm5n_pred_epilogue", ks[2].name);
    try testing.expectEqual(fusion.max_rows, ks[2].instances.len);
    try testing.expectEqual(fusion.max_rows, ks[0].instances.len);
    try testing.expectEqual(2 * fusion.max_rows, ks[1].instances.len);
    const I = struct {
        fn of(v: std.json.Value, key: []const u8) c_int {
            return @intCast(v.object.get(key).?.integer);
        }
    };
    for (1..fusion.max_rows + 1) |m| {
        const geo = try fusion.kdaInGeometry(q.plans[m], fa.plans[m], c.kdaDim(), c.kda_head_dim, c.kda_heads, c.hidden_size, @intCast(m));
        const inst = ks[0].instances[m - 1];
        for ([_]struct { []const u8, c_int }{
            .{ "M", geo.m },   .{ "K", geo.k },   .{ "NB", geo.nb }, .{ "RB", geo.rb },   .{ "KSB", geo.ksb }, .{ "RGB", geo.rgb },
            .{ "NA", geo.na }, .{ "NC", geo.nc }, .{ "RS", geo.rs }, .{ "KSS", geo.kss }, .{ "RGS", geo.rgs },
        }) |kv| try testing.expectEqual(I.of(inst, kv[0]), kv[1]);
        // One threadgroup shape: 8 simdgroups (the f_a / g_a / b plan's 8 partitions; the q / k / v plan's 1 x 8 row groups).
        try testing.expectEqual(@as(c_int, 8), geo.tgs);
        try testing.expectEqual(geo.tgs, geo.rgb * geo.ksb);
        try testing.expectEqual(geo.tgs, geo.rgs * geo.kss);
        // The stock launches' plans: each segment's K partitions are its site's (the summation order).
        try testing.expectEqual(q.plans[m].ks, geo.ksb);
        try testing.expectEqual(fa.plans[m].ks, geo.kss);
        try testing.expectEqual(q.plans[m].r, geo.rb);
        try testing.expectEqual(fa.plans[m].r, geo.rs);
        const cfg = try fusion.kdaInConfig(geo);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var got: [9]c_int = undefined;
        mlxlc_test_config(cfg, &got);
        try testing.expectEqual([9]c_int{ 32, geo.threadgroups() * 8, 1, 32, 8, 1, 6, 11, 0 }, got);
        for ([_]struct { Gemv, u32, usize }{ .{ dense.plans[m], c.intermediate_size, 0 }, .{ shared.plans[m], c.moe_intermediate_size, 1 } }) |e| {
            const pg = try fusion.pairGeometry(e[0], e[1], c.hidden_size, @intCast(m));
            const pi = ks[1].instances[e[2] * fusion.max_rows + m - 1];
            for ([_]struct { []const u8, c_int }{ .{ "M", pg.m }, .{ "N", pg.n }, .{ "K", pg.k }, .{ "R", pg.p.r }, .{ "KS", pg.p.ks }, .{ "RG", pg.p.rg } }) |kv|
                try testing.expectEqual(I.of(pi, kv[0]), kv[1]);
            const pc = try fusion.pairConfig(pg);
            defer _ = mlx.mlx_fast_metal_kernel_config_free(pc);
            mlxlc_test_config(pc, &got);
            // The stock launch's threadgroup; twice its threadgroups (gate's, then up's).
            const l = mimo_fp8.gemvLaunch(e[0], pg.n);
            try testing.expectEqual([9]c_int{ 32, 2 * l.grid[1], 1, l.tg[0], l.tg[1], 1, 2, 6, 0 }, got);
        }
    }
    // The predictor's epilogue: one threadgroup of E threads a row, one output f32 [M, E].
    for (1..fusion.max_rows + 1) |m| {
        const e: c_int = @intCast(c.n_routed_experts);
        try testing.expectEqual(e, I.of(ks[2].instances[m - 1], "E"));
        try testing.expectEqual(@as(c_int, @intCast(m)), I.of(ks[2].instances[m - 1], "M"));
        const pc = try fusion.predConfig(@intCast(m), e);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(pc);
        var got: [9]c_int = undefined;
        mlxlc_test_config(pc, &got);
        try testing.expectEqual([9]c_int{ @as(c_int, @intCast(m)) * e, 1, 1, e, 1, 1, 1, 1, 0 }, got);
        // The indexed combine: glue.zig's combine launch at the same rows (its three template arguments).
        const ci = try fusion.combineIdxConfig(@intCast(m));
        defer _ = mlx.mlx_fast_metal_kernel_config_free(ci);
        mlxlc_test_config(ci, &got);
        const cg = glue.Geometry.combine(@intCast(m));
        try testing.expectEqual([9]c_int{ cg.grid[0], cg.grid[1], cg.grid[2], cg.group[0], cg.group[1], cg.group[2], 1, 3, 0 }, got);
    }
    // The verify's width (4 rows): q / k / v 2 rows a simdgroup x 8 row groups, f_a / g_a / b 2 rows x 8 partitions.
    const g4 = try fusion.kdaInGeometry(q.plans[4], fa.plans[4], c.kdaDim(), c.kda_head_dim, c.kda_heads, c.hidden_size, 4);
    try testing.expectEqual(@as(c_int, 3 * 512 + 2 * 64 + 32), g4.threadgroups());
    const g1 = try fusion.kdaInGeometry(q.plans[1], fa.plans[1], c.kdaDim(), c.kda_head_dim, c.kda_heads, c.hidden_size, 1);
    try testing.expectEqual(@as(c_int, 3 * 256 + 2 * 32 + 16), g1.threadgroups());
    // Plans one threadgroup cannot hold are refused by name (here: 16 partitions x 4 row groups is 2,048 threads).
    try testing.expectError(error.Glm5NextFusionGeometry, fusion.kdaInGeometry(.{ .r = 4, .ks = 16, .rg = 4 }, fa.plans[1], 8192, 128, 64, 4096, 1));
    try testing.expectError(error.Glm5NextFusionGeometry, fusion.kdaInGeometry(q.plans[1], fa.plans[1], 8192, 128, 64, 4096, 9));
    try sdk.testing.expectNoDevice();
}

/// One threadgroup of `glm5n_fp8_gemv_rows<M, N, K, R, KS, RG>`, transliterated (MiMo's `hostGemvText` per threadgroup):
/// the segment's outputs as f32 sums before the BF16 rounding into `out` [M, N], each write counted in `writes`.
fn rowsTg(sg: fusion.Segment, m: usize, k: usize, x: []const u16, w32: []const f32, out: []f32, writes: []u8, tg: usize) void {
    const r: usize = @intCast(sg.r);
    const ks: usize = @intCast(sg.ks);
    const rg: usize = @intCast(sg.rg);
    const n: usize = @intCast(sg.n);
    const kp = k / ks;
    var part: [8 * 8 * 8 * 4]f32 = undefined;
    var acc: [32 * 8 * 4]f32 = undefined;
    for (0..rg * ks) |sgi| {
        const ksi = sgi % ks;
        const rgi = sgi / ks;
        const n0 = (tg * rg + rgi) * r;
        for (0..32) |lane| {
            const k0 = ksi * kp + mimo_fp8.gemv_kv * lane;
            for (0..m) |mi| for (0..r) |ri| {
                var v: f32 = 0;
                var st: usize = 0;
                while (st < kp / mimo_fp8.gemv_step) : (st += 1) {
                    const kk = k0 + st * mimo_fp8.gemv_step;
                    for (0..mimo_fp8.gemv_kv) |i| v = @mulAdd(f32, mimo_fp8.bf16ToF32(x[mi * k + kk + i]), w32[(n0 + ri) * k + kk + i], v);
                }
                acc[(lane * m + mi) * r + ri] = v;
            };
        }
        for (0..m) |mi| for (0..r) |ri| {
            var lanes: [32]f32 = undefined;
            for (0..32) |l| lanes[l] = acc[(l * m + mi) * r + ri];
            for ([_]usize{ 16, 8, 4, 2, 1 }) |off| {
                var next: [32]f32 = undefined;
                for (0..32) |l| next[l] = lanes[l] + lanes[l ^ off];
                lanes = next;
            }
            if (ks == 1) {
                out[mi * n + n0 + ri] = lanes[0];
                writes[mi * n + n0 + ri] += 1;
            } else part[((rgi * ks + ksi) * m + mi) * r + ri] = lanes[0];
        };
    }
    if (ks > 1) for (0..rg) |rgi| {
        const n0 = (tg * rg + rgi) * r;
        for (0..m) |mi| for (0..r) |ri| {
            var t = part[((rgi * ks) * m + mi) * r + ri];
            for (1..ks) |pi| t += part[((rgi * ks + pi) * m + mi) * r + ri];
            out[mi * n + n0 + ri] = t;
            writes[mi * n + n0 + ri] += 1;
        };
    };
}

/// A fused launch's threadgroups, transliterated from its text: threadgroup `tg` of the launch runs segment s's row
/// block `tg - (the threadgroups of the segments before s)` (the text's subtraction chain, in segment order).
fn fusedText(segs: []const fusion.Segment, m: usize, k: usize, x: []const u16, w32: []const []const f32, outs: []const []f32, writes: []const []u8, arith: bool) void {
    var total: usize = 0;
    for (segs) |sg| total += @intCast(@divExact(sg.n, sg.r * sg.rg));
    for (0..total) |tg_launch| {
        var tg = tg_launch;
        for (segs, 0..) |sg, si| {
            const groups: usize = @intCast(@divExact(sg.n, sg.r * sg.rg));
            if (tg < groups) {
                if (arith) rowsTg(sg, m, k, x, w32[si], outs[si], writes[si], tg) else markTg(sg, m, writes[si], tg);
                break;
            }
            tg -= groups;
        }
    }
}

/// `rowsTg`'s writes only (the real shapes' coverage, no arithmetic).
fn markTg(sg: fusion.Segment, m: usize, writes: []u8, tg: usize) void {
    const r: usize = @intCast(sg.r);
    const rg: usize = @intCast(sg.rg);
    const n: usize = @intCast(sg.n);
    for (0..rg) |rgi| {
        const n0 = (tg * rg + rgi) * r;
        for (0..m) |mi| for (0..r) |ri| {
            writes[mi * n + n0 + ri] += 1;
        };
    }
}

test "glm5_next fusion: the fused texts' thread maps write every output of every segment once at Flash's geometry (M = 1 .. 8), and each output is bit for bit the stock launch's sum (MiMo's KS-order emulation) on random codes, scales and rows at every R" {
    const a = testing.allocator;
    const c: glm.Config = .{};
    var s: Sites = .{};
    const q = (try s.lin(c.kdaDim(), c.hidden_size)).fp8;
    const fa = (try s.lin(c.kda_head_dim, c.hidden_size)).fp8;
    const dense = (try s.lin(c.intermediate_size, c.hidden_size)).fp8;
    const shared = (try s.lin(c.moe_intermediate_size * c.n_shared_experts, c.hidden_size)).fp8;
    // Coverage at the real geometry (writes only).
    var covered: usize = 0;
    for (1..fusion.max_rows + 1) |m| {
        const geo = try fusion.kdaInGeometry(q.plans[m], fa.plans[m], c.kdaDim(), c.kda_head_dim, c.kda_heads, c.hidden_size, @intCast(m));
        const segs = geo.segments();
        var writes: [6][]u8 = undefined;
        for (&writes, segs) |*wr, sg| wr.* = try a.alloc(u8, m * @as(usize, @intCast(sg.n)));
        defer for (writes) |wr| a.free(wr);
        for (writes) |wr| @memset(wr, 0);
        fusedText(&segs, m, c.hidden_size, &.{}, &.{}, &.{}, &writes, false);
        for (writes) |wr| for (wr) |n| {
            try testing.expectEqual(@as(u8, 1), n);
            covered += 1;
        };
        for ([_]struct { Gemv, u32 }{ .{ dense.plans[m], c.intermediate_size }, .{ shared.plans[m], c.moe_intermediate_size } }) |e| {
            const pg = try fusion.pairGeometry(e[0], e[1], c.hidden_size, @intCast(m));
            const one: fusion.Segment = .{ .n = pg.n, .r = pg.p.r, .ks = pg.p.ks, .rg = pg.p.rg };
            var pw: [2][]u8 = undefined;
            for (&pw) |*wr| wr.* = try a.alloc(u8, m * @as(usize, @intCast(pg.n)));
            defer for (pw) |wr| a.free(wr);
            for (pw) |wr| @memset(wr, 0);
            fusedText(&.{ one, one }, m, c.hidden_size, &.{}, &.{}, &.{}, &pw, false);
            for (pw) |wr| for (wr) |n| {
                try testing.expectEqual(@as(u8, 1), n);
                covered += 1;
            };
        }
    }
    // The arithmetic at reduced shapes of each plan class (K 2,048: KS 1 runs 8 steps a lane, KS 8 one), every R a width
    // takes, three segment sizes (so a boundary off by one lands in the wrong segment's rows): bit for bit the stock
    // launch's KS-order sum, row for row.
    var prng: std.Random.DefaultPrng = .init(0x6c66_f8f1_5e00);
    const rnd = prng.random();
    const k: usize = 2048;
    const ns = [6]u32{ 192, 192, 192, 128, 128, 64 };
    var fx: [6]mimo_fp8.Fixture = undefined;
    var w32: [6][]f32 = undefined;
    var made: usize = 0;
    defer for (fx[0..made], w32[0..made]) |f, w| {
        f.deinit(a);
        a.free(w);
    };
    for (ns, 0..) |n, i| {
        fx[i] = try mimo_fp8.Fixture.init(a, n, @intCast(k), fusion.max_rows, rnd.int(u64));
        w32[i] = try a.alloc(f32, n * k);
        made += 1;
        for (0..n) |ni| mimo_fp8.weightRow(fx[i].codes[ni * k ..][0..k], fx[i].srow(ni, k), w32[i][ni * k ..][0..k]);
    }
    const x = fx[0].x;
    var compared: usize = 0;
    for ([_][2]c_int{ .{ 4, 4 }, .{ 2, 2 }, .{ 1, 4 } }) |rr| for ([_]usize{ 1, 3, 4, 5, 8 }) |m| {
        const big: Gemv = .{ .r = rr[0], .ks = 1, .rg = 4 };
        const small: Gemv = .{ .r = rr[1], .ks = 8, .rg = 1 };
        const geo = try fusion.kdaInGeometry(big, small, ns[0], ns[3], ns[5], @intCast(k), @intCast(m));
        const segs = geo.segments();
        var outs: [6][]f32 = undefined;
        var writes: [6][]u8 = undefined;
        for (&outs, &writes, ns) |*o, *wr, n| {
            o.* = try a.alloc(f32, m * n);
            wr.* = try a.alloc(u8, m * n);
            @memset(wr.*, 0);
        }
        defer for (outs, writes) |o, wr| {
            a.free(o);
            a.free(wr);
        };
        const ws = [6][]const f32{ w32[0], w32[1], w32[2], w32[3], w32[4], w32[5] };
        fusedText(&segs, m, k, x, &ws, &outs, &writes, true);
        for (0..6) |si| {
            const site: Gemv = if (si < 3) big else small;
            for (0..m) |mi| for (0..ns[si]) |ni| {
                try testing.expectEqual(@as(u8, 1), writes[si][mi * ns[si] + ni]);
                const want = mimo_fp8.hostGemvDotW(site, x[mi * k ..][0..k], w32[si][ni * k ..][0..k]);
                try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(outs[si][mi * ns[si] + ni])));
                compared += 1;
            };
        }
        // The pair text: gate and up on one plan (the shared expert's class: KS 2, RG 2).
        const pp: Gemv = .{ .r = rr[0], .ks = 2, .rg = 2 };
        const pg = try fusion.pairGeometry(pp, ns[0], @intCast(k), @intCast(m));
        const one: fusion.Segment = .{ .n = pg.n, .r = pp.r, .ks = pp.ks, .rg = pp.rg };
        for (writes[0..2]) |wr| @memset(wr, 0);
        fusedText(&.{ one, one }, m, k, x, &.{ w32[0], w32[1] }, outs[0..2], writes[0..2], true);
        for (0..2) |si| for (0..m) |mi| for (0..ns[0]) |ni| {
            try testing.expectEqual(@as(u8, 1), writes[si][mi * ns[0] + ni]);
            const want = mimo_fp8.hostGemvDotW(pp, x[mi * k ..][0..k], w32[si][ni * k ..][0..k]);
            try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(outs[si][mi * ns[0] + ni])));
            compared += 1;
        };
    };
    std.debug.print("GLM5N_FUSION_TEXT_MAP covered={d} (Flash's geometry, M 1..8, written once) compared={d} bit_equal_to_stock_KS_sum=yes\n", .{ covered, compared });
    try sdk.testing.expectNoDevice();
}

fn placeholder(s: mlx.mlx_stream, shape: []const c_int, dt: mlx.mlx_dtype) !T {
    var out = mlx.mlx_array_new();
    if (mlxlc_test_placeholder(shape.ptr, @intCast(shape.len), dt, s, &out) != 0) {
        _ = mlx.mlx_array_free(out);
        return error.MlxError;
    }
    return out;
}

fn launchesOf(outs: []const T) c_int {
    const v = mlx.mlx_vector_array_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    return mlxlc_test_count_launches(v);
}

/// The graph's nodes that dispatch (every primitive but the views MLX evaluates as no-copy reshapes of their input, and the
/// placeholders), and their names.
fn dispatchesOf(outs: []const T, names: *[512]u8) !struct { n: usize, names: []const u8 } {
    const v = mlx.mlx_vector_array_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    const all = mlxlc_test_graph_ops(v, names, names.len);
    if (all < 0) return error.TestUnexpectedResult;
    const list = std.mem.sliceTo(names, 0);
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |op| {
        const view = for ([_][]const u8{ "MlxlcPlaceholder", "Transpose", "Reshape", "Broadcast", "Squeeze", "ExpandDims" }) |vw| {
            if (std.mem.eql(u8, op, vw)) break true;
        } else false;
        n += @intFromBool(!view);
    }
    return .{ .n = n, .names = list };
}

test "glm5_next fusion: the launches the routes build (the graph's nodes over placeholder inputs): a KDA layer's input projections 6 stock, 1 fused; an MLP's gate / up 2 and 1; the predictor's product and epilogue 4 dispatches and 2; a routed call's take and combine 2 and 1; the outputs' shapes and dtype the stock route's" {
    const a = testing.allocator;
    const c: glm.Config = .{};
    const s = mlxlc_test_gpu_stream();
    defer _ = mlx.mlx_stream_free(s);
    // The backend without its compiled closures (`G.init` compiles them, which makes a device): the routes reshape and
    // adopt only.
    var box = s;
    var g: G = .{ .gpa = a, .s = s, .stream_box = &box, .is_child = true };
    defer g.deinit();
    // The residents route's GEMV and the fused texts as plain MLX kernels (no launch cache: no pipeline lookup).
    var route: fp8.Route = .{ .kernels = .{} };
    {
        const ins = [_][*:0]const u8{ "x", "w", "scales" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        route.kernels.gemv = mlx.mlx_fast_metal_kernel_new("mimo_fp8_gemv", iv, ov, mimo_fp8.texts.gemv, mimo_fp8.texts.header, true, false);
    }
    defer _ = mlx.mlx_fast_metal_kernel_free(route.kernels.gemv);
    var arrays: std.ArrayList(T) = .empty;
    defer {
        for (arrays.items) |x| _ = mlx.mlx_array_free(x);
        arrays.deinit(a);
    }
    const P = struct {
        fn lin(al: std.mem.Allocator, list: *std.ArrayList(T), st: mlx.mlx_stream, r: *fp8.Route, n: u32, k: u32) !fp8.Linear {
            const w = try placeholder(st, &.{ @intCast(n), @intCast(k) }, .uint8);
            try list.append(al, w);
            const sc = try placeholder(st, &fp8.scaleShape(n, k), .float32);
            try list.append(al, sc);
            return .{ .fp8 = try fp8.bind(r, w, sc, n, k) };
        }
    };
    const h = c.hidden_size;
    var w: kda.Weights = undefined;
    inline for (.{ "q", "k", "v" }) |f| @field(w, f) = try P.lin(a, &arrays, s, &route, c.kdaDim(), h);
    w.f_a = try P.lin(a, &arrays, s, &route, c.kda_head_dim, h);
    w.g_a = try P.lin(a, &arrays, s, &route, c.kda_head_dim, h);
    w.b = try P.lin(a, &arrays, s, &route, c.kda_heads, h);
    const mlp: graph.Mlp = .{ .gate = try P.lin(a, &arrays, s, &route, c.moe_intermediate_size, h), .up = try P.lin(a, &arrays, s, &route, c.moe_intermediate_size, h), .down = .{ .bf16 = .{ .ctx = null } } };
    // The fused routes over this geometry with plain kernels (`testInstall`; the stock routes restored after).
    const layer: graph.Layer = .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = w }, .slot = 0 };
    const dense_layer: graph.Layer = .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = w }, .dense = .{ .gate = try P.lin(a, &arrays, s, &route, c.intermediate_size, h), .up = try P.lin(a, &arrays, s, &route, c.intermediate_size, h), .down = .{ .bf16 = .{ .ctx = null } } }, .slot = 1 };
    const shared_layer: graph.Layer = .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = w }, .shared = mlp, .slot = 2 };
    const p = try fusion.plan(&c, &.{ layer, dense_layer, shared_layer }, .{ .kda_inputs = true, .gate_up = true }, null);
    var restore = try fusion.testInstall(p, @intCast(c.n_routed_experts));
    defer restore.done();
    for (1..5) |rows| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const x = try placeholder(s, &.{ 1, @intCast(rows), @intCast(h) }, .bfloat16);
        try arrays.append(a, x);
        const x2 = try placeholder(s, &.{ @intCast(rows), @intCast(h) }, .bfloat16);
        try arrays.append(a, x2);
        var got: [2][2]c_int = undefined;
        var shapes: [2][8][3]c_int = undefined;
        for ([_]bool{ false, true }, 0..) |fused, fi| {
            restore.use(fused);
            const in = try fusion.kdaInputs(&g, &w, x);
            const six = [_]T{ in.q, in.k, in.v, in.f_a, in.g_a, in.b };
            got[fi][0] = launchesOf(&six);
            for (six, 0..) |y, i| {
                const sh = g.shapeOf(y);
                try testing.expectEqual(@as(u8, 3), sh.n);
                try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
                shapes[fi][i] = .{ sh.d[0], sh.d[1], sh.d[2] };
            }
            const gu = try fusion.gateUp(&g, mlp.gate, mlp.up, x2, .shared, @intCast(rows));
            got[fi][1] = launchesOf(&.{ gu.gate, gu.up });
            for ([_]T{ gu.gate, gu.up }, 6..) |y, i| {
                const sh = g.shapeOf(y);
                try testing.expectEqual(@as(u8, 2), sh.n);
                shapes[fi][i] = .{ sh.d[0], sh.d[1], 0 };
            }
        }
        // Stock: each projection its own launch; fused: one launch for the six, one for the pair.
        try testing.expectEqual([2]c_int{ 6, 2 }, got[0]);
        try testing.expectEqual([2]c_int{ 1, 1 }, got[1]);
        // The predictor (experts.zig `scoresBf16` / `scoresBf16Fused`): the product and its epilogue over the next routed
        // layer's BF16 router [E, hidden] and correction f32 [E]: Matmul, AsType, Sigmoid, Add stock; Matmul and the
        // epilogue's one launch fused; the same f32 [rows, E] out.
        {
            const e: c_int = @intCast(c.n_routed_experts);
            const router = try placeholder(s, &.{ e, @intCast(h) }, .bfloat16);
            try arrays.append(a, router);
            const corr = try placeholder(s, &.{e}, .float32);
            try arrays.append(a, corr);
            const pr: mimo_math.Predictor = .{};
            var nb: [512]u8 = undefined;
            const st = try experts.scoresBf16(&pr, &g, x2, router, corr);
            const sd = try dispatchesOf(&.{st}, &nb);
            try testing.expectEqual(@as(usize, 4), sd.n);
            for ([_][]const u8{ "Matmul", "AsType", "Sigmoid", "Add" }) |op| try testing.expect(std.mem.indexOf(u8, sd.names, op) != null);
            const fu = try experts.scoresBf16Fused(&pr, &g, x2, router, corr);
            const fd = try dispatchesOf(&.{fu}, &nb);
            try testing.expectEqual(@as(usize, 2), fd.n);
            try testing.expect(std.mem.indexOf(u8, fd.names, "Matmul") != null and std.mem.indexOf(u8, fd.names, "CustomKernel") != null);
            try testing.expect(std.mem.indexOf(u8, fd.names, "Sigmoid") == null);
            try testing.expect(g.shapeOf(st).eql(g.shapeOf(fu)));
            try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(fu));
        }
        // A routed call's end (deepseek_v41/experts.zig `join` then glue.zig `combineShared`; `fuse_join`: the indexed
        // combine): the joined rows [R, hidden] (the device wave's then a miss part's), the join's order, the weights and
        // the shared expert's rows: the take and the combine stock, the combine alone fused; the same bf16 [rows, hidden].
        {
            const tk: c_int = @intCast(glue.top_k);
            const n_ids: c_int = tk * @as(c_int, @intCast(rows));
            const joined = try placeholder(s, &.{ n_ids + 3, @intCast(h) }, .bfloat16);
            try arrays.append(a, joined);
            const order = try placeholder(s, &.{n_ids}, .uint32);
            try arrays.append(a, order);
            const wts = try placeholder(s, &.{ @intCast(rows), tk }, .float32);
            try arrays.append(a, wts);
            const shared = try placeholder(s, &.{ @intCast(rows), @intCast(h) }, .bfloat16);
            try arrays.append(a, shared);
            var nb: [512]u8 = undefined;
            // The stock end: the join's take into router order, then the combine text (glue.zig's kernel and launch).
            const taken = try g.reshape(try g.take(joined, order, 0), &.{ @intCast(rows), tk, @intCast(h) });
            const flat = try g.reshape(taken, &.{ n_ids, @intCast(h) });
            const ck = blk: {
                const ins = [_][*:0]const u8{ "rows", "w", "shared" };
                const outs = [_][*:0]const u8{"y"};
                const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
                defer _ = mlx.mlx_vector_string_free(iv);
                const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
                defer _ = mlx.mlx_vector_string_free(ov);
                break :blk mlx.mlx_fast_metal_kernel_new("glm5n_moe_combine", iv, ov, glue.texts.combine, glue.texts.header_combine, true, false);
            };
            defer _ = mlx.mlx_fast_metal_kernel_free(ck);
            const cfg = try fusion.combineIdxConfig(@intCast(rows));
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            var so = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(so);
            const sin = mlx.mlx_vector_array_new_data(&[_]T{ flat, wts, shared }, 3);
            defer _ = mlx.mlx_vector_array_free(sin);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&so, ck, sin, cfg, s));
            var stock_y = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&stock_y, so, 0));
            const sy = try g.adopt(stock_y);
            const sd = try dispatchesOf(&.{sy}, &nb);
            try testing.expectEqual(@as(usize, 2), sd.n);
            try testing.expect(std.mem.indexOf(u8, sd.names, "Gather") != null and std.mem.indexOf(u8, sd.names, "CustomKernel") != null);
            const fy = try fusion.combineIndexed(&g, joined, order, wts, shared, @intCast(rows));
            const fd = try dispatchesOf(&.{fy}, &nb);
            try testing.expectEqual(@as(usize, 1), fd.n);
            try testing.expect(std.mem.indexOf(u8, fd.names, "Gather") == null);
            try testing.expect(g.shapeOf(sy).eql(g.shapeOf(fy)));
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(fy));
        }
        // The same outputs either way: [1, rows, N] for the six, [rows, N] for the pair.
        for (0..8) |i| try testing.expectEqual(shapes[0][i], shapes[1][i]);
        try testing.expectEqual([3]c_int{ 1, @intCast(rows), @intCast(c.kda_heads) }, shapes[1][5]);
    }
    // A forward of Flash's layers (the verify, a serial step): 5 fewer launches a KDA layer, 1 fewer an MLP, 2 fewer a routed
    // layer's predictor.
    try testing.expectEqual(@as(u32, 170), 5 * c.nKda());
    try testing.expectEqual(@as(u32, 45), c.n_layers);
    try testing.expectEqual(@as(u32, 84), 2 * c.moeLayers());
    try testing.expectEqual(@as(u32, 42), c.moeLayers());
    try sdk.testing.expectNoDevice();
}

test "glm5_next fusion: the install's plan refuses by name a BF16 site, an off-shape site and an MLP whose gate / up plans differ; off installs nothing" {
    const c: glm.Config = .{};
    var s: Sites = .{};
    const h = c.hidden_size;
    var w: kda.Weights = undefined;
    inline for (.{ "q", "k", "v" }) |f| @field(w, f) = try s.lin(c.kdaDim(), h);
    w.f_a = try s.lin(c.kda_head_dim, h);
    w.g_a = try s.lin(c.kda_head_dim, h);
    w.b = try s.lin(c.kda_heads, h);
    const base: graph.Layer = .{ .input_norm = .{ .ctx = null }, .post_norm = .{ .ctx = null }, .hc_attn = undefined, .hc_ffn = undefined, .attn = .{ .kda = w }, .slot = 0 };
    var dense = base;
    dense.dense = .{ .gate = try s.lin(c.intermediate_size, h), .up = try s.lin(c.intermediate_size, h), .down = .{ .bf16 = .{ .ctx = null } } };
    var shared = base;
    shared.shared = .{ .gate = try s.lin(c.moe_intermediate_size, h), .up = try s.lin(c.moe_intermediate_size, h), .down = .{ .bf16 = .{ .ctx = null } } };
    var diag: sdk.Diag = .{};
    _ = try fusion.plan(&c, &.{ base, dense, shared }, .{ .kda_inputs = true, .gate_up = true }, &diag);
    _ = try fusion.plan(&c, &.{}, .{}, &diag);
    // A BF16 KDA input (the BF16 pack's): refused by name, naming the site.
    var bf = base;
    bf.attn.kda.g_a = .{ .bf16 = .{ .ctx = null } };
    try testing.expectError(error.Glm5NextFuseKdaInputsNotFp8, fusion.plan(&c, &.{ base, bf }, .{ .kda_inputs = true }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "layer 1 self_attn.g_a_proj is BF16") != null);
    // The switch off: the same layers plan nothing (the stock route reads BF16 sites too).
    _ = try fusion.plan(&c, &.{ base, bf }, .{ .gate_up = false }, &diag);
    // An off-shape site.
    var odd = base;
    odd.attn.kda.b = try s.lin(128, h);
    try testing.expectError(error.Glm5NextFusionGeometry, fusion.plan(&c, &.{odd}, .{ .kda_inputs = true }, &diag));
    // An MLP whose gate is BF16, or whose gate and up plans differ.
    var mbf = shared;
    mbf.shared.?.up = .{ .bf16 = .{ .ctx = null } };
    try testing.expectError(error.Glm5NextFuseGateUpNotFp8, fusion.plan(&c, &.{ dense, mbf }, .{ .gate_up = true }, &diag));
    var mix = dense;
    mix.dense.?.up = try s.lin(c.intermediate_size, h);
    mix.dense.?.up.fp8.plans[4] = .{ .r = 4, .ks = 1, .rg = 4 };
    try testing.expectError(error.Glm5NextFusionGeometry, fusion.plan(&c, &.{ mix, shared }, .{ .gate_up = true }, &diag));
    // Without a dense MLP or a shared expert the pair's launches are not all planned: refused by name.
    try testing.expectError(error.Glm5NextFusionGeometry, fusion.plan(&c, &.{shared}, .{ .gate_up = true }, &diag));
    try sdk.testing.expectNoDevice();
}
