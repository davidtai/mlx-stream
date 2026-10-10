//! The kernel-ops GPU gate: the Python lanes' own device outputs (a fixture written in a
//! GPU run by the reference runtime's dump_kernel_ops_fixture.py) replayed through the ported
//! routes on the registry's kernels, every output word compared. Seeded or file-backed
//! inputs are checked by sha256 first; projection rates and physical row strides are explicit.
//! GPU only: DSV41_KERNELS_GPU=1 and DSV41_KERNEL_OPS_FIXTURE=<dir> in a lock-holding window.

const std = @import("std");
const mlx = @import("sdk").mlx;
const xk = @import("exl3_kernels.zig");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const kr = sdk_ext.kernels.Routes(xk);
const tr = @import("dsv41_kernel_routes.zig");
const xq = @import("exl3_quant.zig");

const Allocator = std.mem.Allocator;
const Dtype = mlx.mlx_dtype;

/// The routes' backend on real MLX arrays (the model's own backend plays this role there):
/// arrays created here are freed by `reset`; kept arrays by `release`.
pub const MlxG = struct {
    pub const T = mlx.mlx_array;

    a: Allocator,
    s: mlx.mlx_stream,
    bound: *const xk.Bound,
    live: std.ArrayList(mlx.mlx_array) = .empty,

    pub fn deinit(g: *MlxG) void {
        g.reset();
        g.live.deinit(g.a);
    }

    pub fn reset(g: *MlxG) void {
        g.resetTo(0);
    }

    /// The live list's length (a wave's start).
    pub fn mark(g: *const MlxG) usize {
        return g.live.items.len;
    }

    /// Frees every array tracked since `m`; kept handles survive.
    pub fn resetTo(g: *MlxG, m: usize) void {
        for (g.live.items[m..]) |x| _ = mlx.mlx_array_free(x);
        g.live.shrinkRetainingCapacity(m);
    }

    fn track(g: *MlxG, x: T) !T {
        g.live.append(g.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    pub fn shapeOf(_: *MlxG, x: T) kr.Shape {
        return .of(mlx.getShape(x));
    }

    pub fn dtypeOf(_: *MlxG, x: T) Dtype {
        return mlx.mlx_array_dtype(x);
    }

    pub fn hostArray(g: *MlxG, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        const x = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (x.ctx == null) return error.MlxError;
        return g.track(x);
    }

    pub fn keep(_: *MlxG, x: T) T {
        var k = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&k, x);
        return k;
    }

    pub fn release(_: *MlxG, x: T) void {
        _ = mlx.mlx_array_free(x);
    }

    pub fn reshape(g: *MlxG, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn astype(g: *MlxG, x: T, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&r, x, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn launch(g: *MlxG, k: xk.Kernel, inputs: []const T, cfg: *const xk.LaunchConfig, out: []T) !void {
        try g.bound.apply(k, inputs, cfg, out);
        for (out) |o| _ = try g.track(o);
    }

    pub const Prepared = xk.Prepared;

    pub fn prepareLaunch(g: *MlxG, k: xk.Kernel, cfg: *const xk.LaunchConfig) !Prepared {
        return g.bound.prepare(k, cfg);
    }

    pub fn launchPrepared(g: *MlxG, p: *const Prepared, inputs: []const T, out: []T) !void {
        try g.bound.applyPrepared(p, inputs, out);
        for (out) |o| _ = try g.track(o);
    }

    pub fn releasePrepared(_: *MlxG, p: *Prepared) void {
        p.deinit();
    }

    pub fn evalAll(_: *MlxG, xs: []const T) !void {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_eval(v));
    }

    pub fn asyncEval(_: *MlxG, xs: []const T) !void {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }

    pub fn concat(g: *MlxG, xs: []const T, axis: c_int) !T {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_concatenate_axis(&r, v, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn take(g: *MlxG, x: T, idx: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_take_axis(&r, x, idx, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// Row-contiguous host bytes of `x` (caller frees).
    fn hostBytes(g: *MlxG, x: T) ![]u8 {
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, x, false, g.s));
        try mlx.check(mlx.mlx_array_eval(c));
        if (mlx.errorPending()) return error.MlxError;
        const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
        const p = mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull;
        return g.a.dupe(u8, p[0..n]);
    }
};

// ── The fixture (dump_kernel_ops_fixture.py spec.json) ──

const JGen = struct {
    kind: []const u8,
    seed: u64 = 0,
    lo: f64 = 0,
    hi: f64 = 0,
    rows: []const u32 = &.{},
    slots: []const u32 = &.{},
    tiles: u32 = 0,
    values: []const f64 = &.{},
};
const JArray = struct { name: []const u8, dtype: []const u8, shape: []const i64, gen: ?JGen = null, sha256: []const u8, file: ?[]const u8 = null };
const JVars = struct { rows: u64 = 0, cap: u64 = 0, experts: u64 = 0 };
const JLayouts = struct { gate: xq.ProjectionLayout, up: xq.ProjectionLayout, down: xq.ProjectionLayout };
const JCase = struct { family: []const u8, case: []const u8, vars: JVars = .{}, site: ?[]const u8 = null, proj: ?[]const u8 = null, layout: ?xq.ProjectionLayout = null, layouts: ?JLayouts = null, eps: ?f64 = null, inputs: []const JArray, outputs: []const JArray };
const JSpec = struct { format: []const u8, manifest_sha256: []const u8, cases: []const JCase };

const fixture_format = "mlx-serve-exl3-kernel-ops-fixture-v2";
const draft_fixture_format = "mlx-serve-exl3-kernel-draft-fixture-v1";
const decode2_fixture_format = "mlx-serve-exl3-kernel-decode2-fixture-v1";
const prefill2_fixture_format = "mlx-serve-exl3-kernel-prefill2-fixture-v1";
const golden: u64 = 0x9E3779B97F4A7C15;

fn sm(seed: u64, i: u64) u64 {
    var z = seed +% (i + 1) *% golden;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn dtypeOf(name: []const u8) Dtype {
    const map = [_]struct { []const u8, Dtype }{
        .{ "float32", .float32 }, .{ "float16", .float16 }, .{ "bfloat16", .bfloat16 }, .{ "int16", .int16 },
        .{ "int32", .int32 },     .{ "uint32", .uint32 },   .{ "uint8", .uint8 },       .{ "bool", .bool_ },
    };
    for (map) |m| if (std.mem.eql(u8, m[0], name)) return m[1];
    unreachable;
}

fn size(dt: Dtype) usize {
    return switch (dt) {
        .uint8, .bool_ => 1,
        .int16, .float16, .bfloat16 => 2,
        .int32, .uint32, .float32 => 4,
        else => unreachable,
    };
}

fn putInt(b: []u8, i: usize, dt: Dtype, v: u64) void {
    switch (dt) {
        .uint8, .bool_ => b[i] = @truncate(v),
        .int32, .uint32 => std.mem.writeInt(u32, b[i * 4 ..][0..4], @truncate(v), .little),
        else => unreachable,
    }
}

/// The bytes of one input, as the dump's `generate` builds them.
fn generate(a: Allocator, gen: JGen, dt: Dtype, n: usize) ![]u8 {
    const b = try a.alloc(u8, n * size(dt));
    errdefer a.free(b);
    const kind = std.meta.stringToEnum(enum { bits, bf16bits, uniform, index, range, srange, wave_rhs, wave_table, slots16, values }, gen.kind) orelse return error.FixtureGenerator;
    switch (kind) {
        .bits, .bf16bits => {
            var w: usize = 0;
            while (w * 8 < b.len) : (w += 1) {
                var le: [8]u8 = undefined;
                std.mem.writeInt(u64, &le, sm(gen.seed, w), .little);
                const end = @min(b.len, w * 8 + 8);
                @memcpy(b[w * 8 .. end], le[0 .. end - w * 8]);
            }
            // bf16bits: the stream's u16 words as bf16, exponent pinned to 0x78..0x7B
            if (kind == .bf16bits) for (0..n) |i| {
                const v = std.mem.readInt(u16, b[i * 2 ..][0..2], .little);
                std.mem.writeInt(u16, b[i * 2 ..][0..2], (v & 0x807F) | ((0x78 + ((v >> 7) & 3)) << 7), .little);
            };
        },
        .uniform => for (0..n) |i| {
            const t = @as(f64, @floatFromInt(sm(gen.seed, i) >> 11)) * 0x1p-53;
            const v: f32 = @floatCast(gen.lo + (gen.hi - gen.lo) * t);
            switch (dt) {
                .float32 => std.mem.writeInt(u32, b[i * 4 ..][0..4], @bitCast(v), .little),
                .float16 => std.mem.writeInt(u16, b[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
                .bfloat16 => {
                    const u: u64 = @as(u32, @bitCast(v));
                    std.mem.writeInt(u16, b[i * 2 ..][0..2], @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16), .little);
                },
                else => unreachable,
            }
        },
        .index => for (0..n) |i| putInt(b, i, dt, sm(gen.seed, i) % @as(u64, @intFromFloat(gen.hi))),
        .range => for (0..n) |i| putInt(b, i, dt, @as(u64, @intFromFloat(gen.lo)) + sm(gen.seed, i) % @as(u64, @intFromFloat(gen.hi - gen.lo))),
        // prefill batch 2: a signed range (lo may be negative: the compressed selection's -1 = no key)
        .srange => for (0..n) |i| putInt(b, i, dt, @bitCast(@as(i64, @intFromFloat(gen.lo)) + @as(i64, @intCast(sm(gen.seed, i) % @as(u64, @intFromFloat(gen.hi - gen.lo)))))),
        .wave_rhs => {
            var r: usize = 0;
            for (gen.rows, 0..) |rows, j| for (0..rows) |_| {
                putInt(b, r, dt, j);
                r += 1;
            };
        },
        .wave_table => {
            var ex: [xq.wave_max]xq.WaveExpert = undefined;
            for (gen.slots, gen.rows, 0..) |s, r, j| ex[j] = .{ .slot = s, .rows = r };
            const t = xq.digTable(ex[0..gen.slots.len], gen.tiles);
            @memcpy(b, std.mem.sliceAsBytes(&t.table));
        },
        .slots16 => {
            const t = xq.rebuildSlots(gen.slots);
            @memcpy(b, std.mem.sliceAsBytes(&t));
        },
        .values => for (0..n) |i| std.mem.writeInt(u32, b[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(gen.values[i]))), .little),
    }
    return b;
}

fn hexEql(want: []const u8, bytes: []const u8) bool {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    const got = std.fmt.bytesToHex(d, .lower);
    return std.mem.eql(u8, want, &got);
}

pub const Line = struct {
    family: []const u8,
    case: []const u8,
    output: []const u8,
    words: u64 = 0,
    bad: u64 = 0,
    ok: bool = false,
    err: []const u8 = "",
};

/// One case: its inputs regenerated and checked, the routes run, every output compared.
fn replayCase(a: Allocator, g: *MlxG, reg: *const xk.Registry, dir: []const u8, c: *const JCase, lines: *std.ArrayList(Line)) !void {
    var ins: std.StringHashMapUnmanaged(mlx.mlx_array) = .empty;
    defer ins.deinit(a);
    for (c.inputs) |*i| {
        const x = try regen(a, g, dir, i) orelse {
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = i.name, .err = "input bytes differ from the dump's" });
            return;
        };
        try ins.put(a, i.name, x);
    }
    var outs: [16]mlx.mlx_array = undefined;
    const n_out = try runFamily(g, reg, c, &ins, &outs);
    if (n_out != c.outputs.len) return error.FixtureOutputs;
    const v = mlx.mlx_vector_array_new_data(&outs, n_out);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    for (c.outputs, outs[0..n_out]) |*o, got_arr| {
        var line: Line = .{ .family = c.family, .case = c.case, .output = o.name };
        const got = try g.hostBytes(got_arr);
        defer a.free(got);
        const path = try std.fs.path.join(a, &.{ dir, o.file.? });
        defer a.free(path);
        const want = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 30));
        defer a.free(want);
        const w = size(dtypeOf(o.dtype));
        if (g.dtypeOf(got_arr) != dtypeOf(o.dtype) or got.len != want.len) {
            line.err = "dtype or size differs";
        } else {
            line.words = got.len / w;
            var k: usize = 0;
            while (k < got.len) : (k += w) line.bad += @intFromBool(!std.mem.eql(u8, got[k..][0..w], want[k..][0..w]));
            line.ok = line.bad == 0 and hexEql(o.sha256, want);
        }
        try lines.append(a, line);
    }
}

fn in(ins: *std.StringHashMapUnmanaged(mlx.mlx_array), name: []const u8) mlx.mlx_array {
    return ins.get(name).?;
}

/// The ported calls of one family, outputs in the dump's order.
fn runFamily(g: *MlxG, reg: *const xk.Registry, c: *const JCase, ins: *std.StringHashMapUnmanaged(mlx.mlx_array), outs: *[16]mlx.mlx_array) !usize {
    const f = c.family;
    const eq = std.mem.eql;
    // decode batch 2 (dump_kernel_decode2_fixture.py): the members the RC tiers still run,
    // ATTN_FUSE softmax, INDEX_TOPK and the wo_a ring transpose
    if (eq(u8, f, "woa_transpose")) {
        var r = try tr.WoaRingTranspose(MlxG).init(g, reg);
        defer r.deinit(g);
        try r.checkLayer(g, in(ins, "packed"), in(ins, "scales"), null);
        outs[0] = try r.call(g, in(ins, "packed"), in(ins, "scales"));
        return 1;
    }
    if (eq(u8, f, "index_topk")) {
        var r = try tr.IndexTopk(MlxG).init(g, reg, &.derived, null);
        defer r.deinit(g);
        outs[0..2].* = try r.select(g, in(ins, "score"), in(ins, "clen"));
        return 2;
    }
    if (eq(u8, f, "attn_fuse")) {
        var r = try tr.AttnSoftmax(MlxG).init(g, reg, null);
        defer r.deinit(g);
        outs[0..2].* = try r.call(g, in(ins, "qk"), in(ins, "valid"), in(ins, "sink"));
        return 2;
    }
    // x1..x8: one call per M on the case's one weight
    const x_names = [_][]const u8{ "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8" };
    if (eq(u8, f, "mxfp8_rows")) {
        const site = std.meta.stringToEnum(tr.M1Site, c.site.?) orelse return error.FixtureSite;
        var r = try tr.Mxfp8Rows(MlxG).init(g, reg, site, in(ins, "w"), in(ins, "scales"), null);
        defer r.deinit(g);
        for (0..8) |i| outs[i] = try r.call(g, in(ins, x_names[i]));
        return 8;
    }
    if (eq(u8, f, "head_rows")) {
        var r = try tr.HeadRows(MlxG).init(g, reg, in(ins, "w"), null);
        defer r.deinit(g);
        for (0..8) |i| outs[i] = try r.call(g, in(ins, x_names[i]));
        return 8;
    }
    if (eq(u8, f, "smallm")) {
        const site = std.meta.stringToEnum(tr.SmallMSite, c.site.?) orelse return error.FixtureSite;
        var r = try tr.SmallM(MlxG).init(g, reg, site, in(ins, "w"), null);
        defer r.deinit(g);
        for (0..8) |i| outs[i] = try r.call(g, in(ins, x_names[i]));
        return 8;
    }
    // prefill batch 2 (dump_kernel_prefill2_fixture.py): the P line's prefill-rows texts; the
    // prefill-rows index top-k replays through the index_topk branch above
    if (eq(u8, f, "idxscore")) {
        const r = try tr.IdxScore(MlxG).init(reg, &.derived, null);
        outs[0] = try r.call(g, in(ins, "q"), in(ins, "k"), in(ins, "w"), in(ins, "clen"));
        return 1;
    }
    if (eq(u8, f, "core_vec") or eq(u8, f, "core_rope")) {
        const kind: tr.CoreKind = if (eq(u8, f, "core_vec")) .vec else .rope;
        const ckv = ins.get("ckv");
        var r = try tr.PrefillAttn(MlxG).init(g, reg, &.derived, kind, g.dtypeOf(in(ins, "q")), g.dtypeOf(in(ins, "win")), ckv != null, null);
        defer r.deinit(g);
        const cmp: ?[2]mlx.mlx_array = if (ckv) |store| .{ store, in(ins, "cidx") } else null;
        const rope: ?[2]mlx.mlx_array = if (kind == .rope) .{ in(ins, "qcos"), in(ins, "qsin") } else null;
        outs[0] = try r.attend(g, in(ins, "q"), in(ins, "win"), in(ins, "widx"), in(ins, "wval"), cmp, in(ins, "sink"), rope);
        return 1;
    }
    if (eq(u8, f, "hcnorm")) {
        const x = in(ins, "x");
        var r = try tr.HcNorm(MlxG).init(g, reg, &.derived, g.dtypeOf(x), @floatCast(c.eps orelse return error.FixtureEps), null);
        defer r.deinit(g);
        outs[0] = try r.rsqrt(g, x);
        outs[1] = try r.preNorm(g, x, in(ins, "pre"), in(ins, "w"));
        return 2;
    }
    if (eq(u8, f, "smallk")) {
        const r = try tr.SmallKCombine(MlxG).init(reg, &.derived, null);
        outs[0] = try r.call(g, in(ins, "routed"), in(ins, "weights"), in(ins, "shared"));
        return 1;
    }
    // DRAFTRC (dump_draftrc_fixture.py): the draft routes, outputs in the dump's order
    if (eq(u8, f, "draft_proj")) {
        const site = std.meta.stringToEnum(tr.DraftSite, c.site.?) orelse return error.FixtureSite;
        const x1 = in(ins, "x1");
        var r = try tr.DraftProj(MlxG).init(g, reg, site, g.dtypeOf(x1), in(ins, "w"), in(ins, "scales"), null);
        defer r.deinit(g);
        outs[0] = try r.call(g, x1);
        outs[1] = try r.call(g, in(ins, "x6"));
        outs[2] = try r.call(g, in(ins, "x8"));
        return 3;
    }
    if (eq(u8, f, "draft_router")) {
        var r = try tr.Router(MlxG).init(g, reg, in(ins, "w"), in(ins, "bias"), null);
        defer r.deinit(g);
        outs[0..2].* = try r.call(g, in(ins, "x1"));
        outs[2..4].* = try r.call(g, in(ins, "x6"));
        outs[4..6].* = try r.call(g, in(ins, "x8"));
        return 6;
    }
    if (eq(u8, f, "draft_tape")) {
        var r = try tr.HcTape(MlxG).init(g, reg, .float32, null);
        defer r.deinit(g);
        var mixed = try tr.HcTapeMixed(MlxG).init(g, reg, null);
        defer mixed.deinit(g);
        const x, const rr, const rb, const post, const comb, const pre, const w = .{ in(ins, "x"), in(ins, "r"), in(ins, "rb"), in(ins, "post"), in(ins, "comb"), in(ins, "pre"), in(ins, "w") };
        outs[0] = try r.combine(g, x, rr, post, comb);
        outs[1..4].* = try r.collapseNorm(g, rr, pre, w);
        outs[4..8].* = try r.combineCollapseNorm(g, x, rr, post, comb, pre, w);
        outs[8..12].* = try mixed.call(g, x, rb, post, comb, pre, w);
        return 12;
    }
    if (eq(u8, f, "router")) {
        var r = try tr.Router(MlxG).init(g, reg, in(ins, "w"), in(ins, "bias"), null);
        defer r.deinit(g);
        outs[0..2].* = try r.call(g, in(ins, "x"));
        return 2;
    }
    if (eq(u8, f, "premix")) {
        var r = try tr.Premix(MlxG).init(g, reg, in(ins, "w"), null);
        defer r.deinit(g);
        outs[0] = try r.call(g, in(ins, "x"));
        return 1;
    }
    if (eq(u8, f, "sinkhorn")) {
        var r = try tr.Sinkhorn(MlxG).init(g, reg);
        defer r.deinit(g);
        outs[0] = try r.call(g, in(ins, "comb"));
        return 1;
    }
    if (eq(u8, f, "rcproj")) {
        const site = std.meta.stringToEnum(tr.RcSite, c.site.?).?;
        var r = try tr.RcProj(MlxG).init(g, reg, site, in(ins, "w"), in(ins, "scales"), null);
        defer r.deinit(g);
        outs[0] = try r.call(g, in(ins, "x"));
        return 1;
    }
    if (eq(u8, f, "hctape")) {
        var r = try tr.HcTape(MlxG).init(g, reg, .bfloat16, null);
        defer r.deinit(g);
        const x, const rr, const post, const comb, const pre, const w = .{ in(ins, "x"), in(ins, "r"), in(ins, "post"), in(ins, "comb"), in(ins, "pre"), in(ins, "w") };
        outs[0] = try r.combine(g, x, rr, post, comb);
        outs[1..4].* = try r.collapseNorm(g, rr, pre, w);
        outs[4..8].* = try r.combineCollapseNorm(g, x, rr, post, comb, pre, w);
        outs[8..11].* = try r.mixfin(g, in(ins, "mm"), in(ins, "ssq"), in(ins, "scale"), in(ins, "base"));
        return 11;
    }
    if (eq(u8, f, "fused_proj")) {
        // the fixture's eps: dump_kernel_ops_fixture.py reads the bank config's rms_norm_eps (1e-20)
        var r = try tr.FusedProj(MlxG).init(g, reg, in(ins, "q_norm"), in(ins, "kv_norm"), 1e-20, null);
        defer r.deinit(g);
        const cos, const sin = .{ in(ins, "cos"), in(ins, "sin") };
        outs[0] = try r.qNorm(g, in(ins, "x_q"));
        outs[1] = try r.kvNormRope(g, in(ins, "x_kv"), cos, sin);
        outs[2] = try r.ropeHeads(g, in(ins, "x_rope"), cos, sin, .fwd);
        outs[3] = try r.ropeHeads(g, in(ins, "x_o"), cos, sin, .inv);
        return 4;
    }
    if (eq(u8, f, "gemv")) {
        const layout = c.layout orelse return error.FixtureLayoutMissing;
        if (layout.k < 2 or layout.k > 4) return error.FixtureRateUnsupported;
        var r = try xq.Gemv(MlxG).initRates(g.a, g, reg, .{}, @as(u3, 1) << @intCast(layout.k - 2));
        defer r.deinit(g);
        const proj: xq.Proj = if (eq(u8, c.proj.?, "down")) .down else .gate;
        outs[0] = try r.project(g, proj, in(ins, "xh"), in(ins, "ids"), in(ins, "code"), .{ .fixed = layout });
        return 1;
    }
    if (eq(u8, f, "prep")) {
        var r = try xq.RinPrep(MlxG).init(g, reg);
        defer r.deinit(g);
        const ids = in(ins, "ids");
        outs[0..2].* = try r.inRin(g, in(ins, "x"), in(ins, "tok"), in(ins, "rin_g"), in(ins, "rin_u"), ids);
        outs[2] = try r.guEpi(g, in(ins, "zg"), in(ins, "zu"), in(ins, "rout_g"), in(ins, "rout_u"), ids);
        outs[3] = try r.dinRin(g, in(ins, "hid"), in(ins, "rin_d"), ids);
        outs[4] = try r.dpost(g, in(ins, "zd"), in(ins, "rout_d"), ids);
        return 5;
    }
    const layouts = c.layouts orelse return error.FixtureLayoutMissing;
    const gate: xq.ProjArrays(mlx.mlx_array) = .{ .code = in(ins, "code_g"), .rout = in(ins, "rout_g"), .rin = in(ins, "rin_g"), .layout = .{ .fixed = layouts.gate } };
    const up: xq.ProjArrays(mlx.mlx_array) = .{ .code = in(ins, "code_u"), .rout = in(ins, "rout_u"), .rin = in(ins, "rin_u"), .layout = .{ .fixed = layouts.up } };
    const down: xq.ProjArrays(mlx.mlx_array) = .{ .code = in(ins, "code_d"), .rout = in(ins, "rout_d"), .rin = in(ins, "rin_d"), .layout = .{ .fixed = layouts.down } };
    if (eq(u8, f, "rebuild")) {
        const r = xq.Rebuild(MlxG).init(reg);
        outs[0..3].* = try r.call(g, gate, up, down, in(ins, "slots"), @intCast(c.vars.experts));
        return 3;
    }
    if (eq(u8, f, "digx")) {
        const r = xq.DigX(MlxG).init(reg);
        const rhs = in(ins, "rhs");
        const tbl_gu = in(ins, "tbl_gu");
        const tbl_dn = in(ins, "tbl_dn");
        // the GEMMs over the fixture's tables at the routed M tile; the rot and onepass stages read their slot
        // column only
        const gu = try routedTable(g, tbl_gu);
        const dn = try routedTable(g, tbl_dn);
        const x = try r.take2(g, in(ins, "act"), in(ins, "ridx"), rhs, tbl_gu, gate.rin, up.rin);
        const z = try r.gemmGateUp(g, x[0], x[1], gate.code, up.code, gu.tbl, gu.tgs, gate.layout, up.layout);
        const hd = try r.onePass(g, z[0], z[1], rhs, tbl_gu, gate.rout, up.rout, down.rin);
        const zd = try r.gemmDown(g, hd, down.code, dn.tbl, dn.tgs, down.layout);
        const o = try r.widen1(g, zd, rhs, tbl_dn, down.rout);
        const rx = try r.roundx(g, in(ins, "act_r"), rhs, tbl_gu, down.rin);
        const w2 = try r.widen2(g, in(ins, "act_g"), in(ins, "act_u"), rhs, tbl_gu, gate.rout, up.rout);
        outs[0..10].* = .{ x[0], x[1], z[0], z[1], hd, zd, o, rx, w2[0], w2[1] };
        return 10;
    }
    return error.FixtureFamily;
}

/// A fixture's 64-row wave table at the routed GEMMs' M tile (`DigX.m_tile`): the same slot, first-row and row
/// columns, the first-threadgroup column counting ceil(rows / m_tile) tiles; the tiles per M tile from its own
/// threadgroup count (entry 65).
fn routedTable(g: *MlxG, tbl: mlx.mlx_array) !xq.DigTableArray(MlxG) {
    const b = try g.hostBytes(tbl);
    defer g.a.free(b);
    var t: [80]i32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&t), b[0 .. 80 * 4]);
    const n: usize = @intCast(t[64]);
    var ex: [xq.wave_max]xq.WaveExpert = undefined;
    var t64: u32 = 0;
    for (0..n) |j| {
        ex[j] = .{ .slot = @intCast(t[j]), .rows = @intCast(t[32 + j]) };
        t64 += (ex[j].rows + 63) / 64;
    }
    const tiles = @divExact(@as(u32, @intCast(t[65])), t64);
    const r = xq.digTableBm(ex[0..n], tiles, xq.DigX(MlxG).m_tile);
    return .{ .tbl = try g.hostArray(std.mem.sliceAsBytes(&r.table), &.{80}, .int32), .tgs = r.tgs };
}

fn wanted(filter: ?[]const u8, family: []const u8) bool {
    const f = filter orelse return true;
    var it = std.mem.splitScalar(u8, f, ',');
    while (it.next()) |name| if (std.mem.eql(u8, name, family) or std.mem.eql(u8, name, "all")) return true;
    return false;
}

// ── The prefill wave fixture (dump_prefill_waves.py spec.json, window PF) ──

const PShape = struct { wave: u32, inflight: u32, row_budget: u32, carry_rows: u32 };
const PCall = struct { name: []const u8, a_rows: u32, slots: []const u32, act: JArray, output: JArray };
const PCase = struct { family: []const u8, case: []const u8, shape: PShape, cap: u32, layouts: JLayouts, inputs: []const JArray, calls: []const PCall };
const PSpec = struct { format: []const u8, manifest_sha256: []const u8, cases: []const PCase };
const prefill_format = "mlx-serve-exl3-prefill-waves-fixture-v2";

/// A generated or file-backed input; null when its bytes differ from the fixture's sha256.
fn regen(a: Allocator, g: *MlxG, dir: []const u8, i: *const JArray) !?mlx.mlx_array {
    const dt = dtypeOf(i.dtype);
    var shape: [8]c_int = undefined;
    var n: usize = 1;
    for (i.shape, 0..) |d, k| {
        shape[k] = @intCast(d);
        n *= @intCast(d);
    }
    if ((i.gen == null) == (i.file == null)) return error.FixtureInputSource;
    const bytes = if (i.gen) |gen| try generate(a, gen, dt, n) else blk: {
        const path = try std.fs.path.join(a, &.{ dir, i.file.? });
        defer a.free(path);
        break :blk try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 31));
    };
    defer a.free(bytes);
    if (bytes.len != n * size(dt) or !hexEql(i.sha256, bytes)) return null;
    return try g.hostArray(bytes, shape[0..i.shape.len], dt);
}

/// One case: the bank and each call's act regenerated and checked, the route's calls in order on
/// one route (then its prefill boundary), every call's result compared with the lane's, word for word.
fn replayPrefill(a: Allocator, g: *MlxG, reg: *const xk.Registry, dir: []const u8, c: *const PCase, lines: *std.ArrayList(Line), diag: *xk.Diag) !void {
    var ins: std.StringHashMapUnmanaged(mlx.mlx_array) = .empty;
    defer ins.deinit(a);
    for (c.inputs) |*i| {
        const x = try regen(a, g, dir, i) orelse {
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = i.name, .err = "input bytes differ from the dump's" });
            return;
        };
        try ins.put(a, i.name, x);
    }
    const P = xq.ProjArrays(mlx.mlx_array);
    const bank: xq.BankArrays(mlx.mlx_array) = .{
        .gate = P{ .code = in(&ins, "gate_proj.code"), .rout = in(&ins, "gate_proj.rout"), .rin = in(&ins, "gate_proj.rin"), .layout = .{ .fixed = c.layouts.gate } },
        .up = P{ .code = in(&ins, "up_proj.code"), .rout = in(&ins, "up_proj.rout"), .rin = in(&ins, "up_proj.rin"), .layout = .{ .fixed = c.layouts.up } },
        .down = P{ .code = in(&ins, "down_proj.code"), .rout = in(&ins, "down_proj.rout"), .rin = in(&ins, "down_proj.rin"), .layout = .{ .fixed = c.layouts.down } },
    };
    try xq.checkBank(MlxG, g, reg, .gate, bank.gate, diag);
    try xq.checkBank(MlxG, g, reg, .up, bank.up, diag);
    try xq.checkBank(MlxG, g, reg, .down, bank.down, diag);
    const shape: xq.PrefillShape = .{ .wave = c.shape.wave, .inflight = c.shape.inflight, .row_budget = c.shape.row_budget, .carry_rows = c.shape.carry_rows };
    var r = try xq.DigXPrefill(MlxG).init(a, reg, shape, diag);
    defer r.deinit(g);
    var results: std.ArrayList(mlx.mlx_array) = .empty;
    defer results.deinit(a);
    for (c.calls) |*cl| {
        const act = try regen(a, g, dir, &cl.act) orelse {
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = cl.name, .err = "act bytes differ from the dump's" });
            return;
        };
        try results.append(a, try r.call(g, act, .{ .slot = cl.slots }, bank));
    }
    defer for (results.items) |x| g.release(x);
    try r.finish(g);
    try g.evalAll(results.items);
    for (c.calls, results.items) |*cl, got_arr| {
        const o = &cl.output;
        var line: Line = .{ .family = c.family, .case = c.case, .output = cl.name };
        const got = try g.hostBytes(got_arr);
        defer a.free(got);
        const path = try std.fs.path.join(a, &.{ dir, o.file.? });
        defer a.free(path);
        const want = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 31));
        defer a.free(want);
        if (g.dtypeOf(got_arr) != dtypeOf(o.dtype) or got.len != want.len) {
            line.err = "dtype or size differs";
        } else {
            line.words = got.len / 4;
            var k: usize = 0;
            while (k < got.len) : (k += 4) line.bad += @intFromBool(!std.mem.eql(u8, got[k..][0..4], want[k..][0..4]));
            line.ok = line.bad == 0 and hexEql(o.sha256, want);
        }
        try lines.append(a, line);
    }
}

const testing = std.testing;

/// One fixture replayed through the routes (the ops gate and the draft gate): the spec's format and
/// manifest pin checked, every case's inputs regenerated and checked, every output word compared;
/// one JSON line per output (the receipt), then `[<label>] <cases> cases, <outputs> outputs, <failed> failed`.
fn replayFixture(dir: []const u8, format: []const u8, filter: ?[]const u8, receipt: ?[*:0]const u8, comptime label: []const u8) !void {
    const a = testing.allocator;
    mlx.installErrorHandler();
    const spec_path = try std.fs.path.join(a, &.{ dir, "spec.json" });
    defer a.free(spec_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, spec_path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(JSpec, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const spec = parsed.value;
    try testing.expectEqualStrings(format, spec.format);
    var diag: xk.Diag = .{};
    var reg = xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
    defer reg.deinit();
    // this manifest or a predecessor whose kernels are unchanged here (the exporter's check)
    if (!reg.acceptsManifest(spec.manifest_sha256)) std.debug.print("fixture manifest {s} is neither {s} nor a predecessor\n", .{ spec.manifest_sha256, xk.manifest_sha256 });
    try testing.expect(reg.acceptsManifest(spec.manifest_sha256));
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var lines: std.ArrayList(Line) = .empty;
    defer lines.deinit(a);
    var cases: usize = 0;
    for (spec.cases) |*c| {
        if (!wanted(filter, c.family)) continue;
        cases += 1;
        replayCase(a, &g, &reg, dir, c, &lines) catch |e| {
            var buf: [256]u8 = undefined;
            const msg = mlx.takeError(&buf) orelse "";
            std.debug.print("[" ++ label ++ "] {s} {s}: {t} {s}\n", .{ c.family, c.case, e, msg });
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = "", .err = @errorName(e) });
        };
        g.reset();
    }
    var failed: usize = 0;
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    for (lines.items) |l| {
        failed += @intFromBool(!l.ok);
        try j.print(a, "{{\"family\":\"{s}\",\"case\":\"{s}\",\"output\":\"{s}\",\"words\":{d},\"bad\":{d},\"ok\":{},\"err\":\"{s}\"}}\n", .{ l.family, l.case, l.output, l.words, l.bad, l.ok, l.err });
    }
    std.debug.print("{s}", .{j.items});
    if (receipt) |path| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = std.mem.span(path), .data = j.items });
    std.debug.print("[" ++ label ++ "] {d} cases, {d} outputs, {d} failed\n", .{ cases, lines.items.len, failed });
    try testing.expect(cases > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

// GPU only: DSV41_KERNELS_GPU=1,
// DSV41_KERNEL_OPS_FIXTURE=<fixture dir>; DSV41_KERNEL_OPS_FAMILIES=<a,b|all> narrows it,
// DSV41_KERNEL_OPS_RECEIPT=<path> keeps the per-output JSON lines.
test "dsv41 kernels ops gpu: every route reproduces its lane's own device output (fixture), bitwise" {
    _ = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_KERNEL_OPS_FIXTURE") orelse return error.SkipZigTest);
    const filter: ?[]const u8 = if (std.c.getenv("DSV41_KERNEL_OPS_FAMILIES")) |f| std.mem.span(f) else null;
    try replayFixture(dir, fixture_format, filter, std.c.getenv("DSV41_KERNEL_OPS_RECEIPT"), "kernel ops gate");
}

// GPU only (block (t)): DSV41_KERNELS_GPU=1, DSV41_KERNEL_DRAFT_FIXTURE=<dir> (the
// dump_draftrc_fixture.py fixture); DSV41_KERNEL_DRAFT_RECEIPT=<path> keeps the per-output JSON lines.
test "dsv41 kernels ops gpu: the DRAFTRC routes reproduce the lane's own draft kernels (fixture), bitwise" {
    _ = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_KERNEL_DRAFT_FIXTURE") orelse return error.SkipZigTest);
    try replayFixture(dir, draft_fixture_format, null, std.c.getenv("DSV41_KERNEL_DRAFT_RECEIPT"), "kernel draft gate");
}

// GPU only (decode batch 2): DSV41_KERNELS_GPU=1, DSV41_KERNEL_DECODE2_FIXTURE=<dir>
// (the dump_kernel_decode2_fixture.py fixture); DSV41_KERNEL_DECODE2_RECEIPT=<path> keeps the lines.
test "dsv41 kernels ops gpu: the decode batch 2 routes reproduce their lanes' own device outputs (fixture), bitwise" {
    _ = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_KERNEL_DECODE2_FIXTURE") orelse return error.SkipZigTest);
    try replayFixture(dir, decode2_fixture_format, null, std.c.getenv("DSV41_KERNEL_DECODE2_RECEIPT"), "kernel decode2 gate");
}

// GPU only (prefill batch 2): DSV41_KERNELS_GPU=1, DSV41_KERNEL_PREFILL2_FIXTURE=<dir>
// (the dump_kernel_prefill2_fixture.py fixture); DSV41_KERNEL_PREFILL2_RECEIPT=<path> keeps the lines.
test "dsv41 kernels ops gpu: the prefill batch 2 routes reproduce their lanes' own device outputs (fixture), bitwise" {
    _ = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_KERNEL_PREFILL2_FIXTURE") orelse return error.SkipZigTest);
    try replayFixture(dir, prefill2_fixture_format, null, std.c.getenv("DSV41_KERNEL_PREFILL2_RECEIPT"), "kernel prefill2 gate");
}

// GPU only (window PG): DSV41_KERNELS_GPU=1, DSV41_KERNEL_PREFILL_FIXTURE=<dir>;
// DSV41_KERNEL_PREFILL_RECEIPT=<path> keeps the per-call JSON lines.
test "dsv41 kernels ops gpu: the prefill wave route reproduces the lane's own dispatch output (fixture), bitwise" {
    _ = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_KERNEL_PREFILL_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    mlx.installErrorHandler();
    const spec_path = try std.fs.path.join(a, &.{ dir, "spec.json" });
    defer a.free(spec_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, spec_path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(PSpec, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const spec = parsed.value;
    try testing.expectEqualStrings(prefill_format, spec.format);
    var diag: xk.Diag = .{};
    var reg = xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
    defer reg.deinit();
    try testing.expect(reg.acceptsManifest(spec.manifest_sha256));
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var lines: std.ArrayList(Line) = .empty;
    defer lines.deinit(a);
    for (spec.cases) |*c| {
        replayPrefill(a, &g, &reg, dir, c, &lines, &diag) catch |e| {
            var buf: [256]u8 = undefined;
            const msg = mlx.takeError(&buf) orelse "";
            std.debug.print("[kernel prefill gate] {s}: {t} {s} {s}\n", .{ c.case, e, msg, diag.message() });
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = "", .err = @errorName(e) });
        };
        g.reset();
    }
    var failed: usize = 0;
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    for (lines.items) |l| {
        failed += @intFromBool(!l.ok);
        try j.print(a, "{{\"family\":\"{s}\",\"case\":\"{s}\",\"call\":\"{s}\",\"words\":{d},\"bad\":{d},\"ok\":{},\"err\":\"{s}\"}}\n", .{ l.family, l.case, l.output, l.words, l.bad, l.ok, l.err });
    }
    std.debug.print("{s}", .{j.items});
    if (std.c.getenv("DSV41_KERNEL_PREFILL_RECEIPT")) |path| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = std.mem.span(path), .data = j.items });
    std.debug.print("[kernel prefill gate] {d} cases, {d} calls, {d} failed\n", .{ spec.cases.len, lines.items.len, failed });
    try testing.expect(spec.cases.len > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "dsv41 kernels ops: the fixture generators are the dump's (sha256 of its --golden cases)" {
    const a = testing.allocator;
    const cases = [_]struct { JGen, Dtype, usize, []const u8 }{
        .{ .{ .kind = "bits", .seed = 11 }, .uint8, 37, "83e82e581b5636875ba0a57d454acc2de295c61eb916838277834128e144a6dd" },
        .{ .{ .kind = "bits", .seed = 12 }, .int16, 15, "a6e7271b92ce8247a1e92589fe17653b23a7a8bb7ea59ab63b86facf00561c15" },
        .{ .{ .kind = "uniform", .seed = 13, .lo = -1, .hi = 1 }, .float32, 9, "c0cf7502358cba5ade2fa5177a4208979444cc8d6ea99c8d65acbecae8321582" },
        .{ .{ .kind = "uniform", .seed = 14, .lo = -0.1, .hi = 0.1 }, .float16, 9, "9e7c8144bd6900bfdc81408800a4c77b9361857ec3976ae04210637dd1f77cc7" },
        .{ .{ .kind = "uniform", .seed = 15, .lo = -2, .hi = 2 }, .bfloat16, 9, "6b080ede7166f2356114342d02f7f4dc2748a99f638ce8e42edc1970ee6906c7" },
        .{ .{ .kind = "uniform", .seed = 16, .lo = 1e3, .hi = 1e5 }, .float32, 4, "8d14c13205546afac9e4fe76dd954a3eb0aea86a5b49fdba335ee50c9785a48e" },
        .{ .{ .kind = "index", .seed = 17, .hi = 4 }, .uint32, 7, "99b96e18a705b3317b58dd3060a7b821e2ccf09e2e0b20a115c9b892a7ef46eb" },
        .{ .{ .kind = "index", .seed = 18, .hi = 160 }, .int32, 7, "9e5c1c1d0e3e5585bc0d757bac84819debfa61f1169c2d3b9d8dfc0eea209113" },
        .{ .{ .kind = "range", .seed = 19, .lo = 112, .hi = 125 }, .uint8, 11, "c262ec63d66f8749a41f817211f3d5e28baaf7e71bd12b9b9cc27aa6d9fdc887" },
        .{ .{ .kind = "wave_rhs", .rows = &.{ 70, 37, 20 } }, .uint32, 127, "72ae2751e9e0066acd9e415b3ac7fa451dbf145597204f4f95a52ed16b0d40c3" },
        .{ .{ .kind = "wave_table", .slots = &.{ 3, 0, 2 }, .rows = &.{ 70, 37, 20 }, .tiles = 72 }, .int32, 80, "e7c9e15972414d0b85f223fc695913b94650858ba19f877f499ff124961b2968" },
        .{ .{ .kind = "slots16", .slots = &.{ 3, 1 } }, .int32, 16, "631b74d731b6e3635ad3bf3bb463aba6f2c678e26f9aa0fb5f5ca4b6d1d29f2f" },
        .{ .{ .kind = "values", .values = &.{ 0.3, 0.7, 1.1 } }, .float32, 3, "bcadc79e95123af764e6813ecf10b40cab57dafc8ef17f8f60118dfcf23ccbfd" },
        // decode batch 2 (dump_kernel_decode2_fixture.py --golden): bool bytes and the bf16 head weight words
        .{ .{ .kind = "range", .seed = 21, .lo = 0, .hi = 2 }, .bool_, 13, "61e3fd288705a89f08816c151a557cdd31d7a9ef95ba607c437d6bd4f66095e3" },
        .{ .{ .kind = "bf16bits", .seed = 22 }, .bfloat16, 11, "7b6c6bc9a8d38b9b98954f6abd0aee09c1693bbbe62c9bac446bde806cf9d0f1" },
        // prefill batch 2 (dump_kernel_prefill2_fixture.py --golden): the signed range (-1 .. -64 = no key)
        .{ .{ .kind = "srange", .seed = 23, .lo = -64, .hi = 2048 }, .int32, 13, "967d23bfcc0bc9205fd577bcd469258765f0c31131017d7fb326be0c5a4a325e" },
    };
    for (cases) |c| {
        const b = try generate(a, c[0], c[1], c[2]);
        defer a.free(b);
        if (!hexEql(c[3], b)) std.debug.print("generator {s} {t} differs from the dump's\n", .{ c[0].kind, c[1] });
        try testing.expect(hexEql(c[3], b));
    }
    // the registry's golden-plane stream is the same splitmix64 (state += golden before each output)
    var st: u64 = 0x5EED0001;
    for (0..4) |i| try testing.expectEqual(xk.splitmix64(&st), sm(0x5EED0001, i));
}

const PublicBlock = struct {
    k: u32,
    code: []u16,
    rin: [128]f32,
    rout: [128]f32,
    weights: []f64,
    inner: []f64,
};

fn publicTensor(header: std.json.Value, data: []const u8, name: []const u8, dtype: []const u8, shape: []const u64) ![]const u8 {
    const t = header.object.get(name) orelse return error.FixtureTensorMissing;
    try testing.expectEqualStrings(dtype, t.object.get("dtype").?.string);
    const dims = t.object.get("shape").?.array.items;
    try testing.expectEqual(shape.len, dims.len);
    var count: u64 = 1;
    for (shape, dims) |want, got| {
        try testing.expectEqual(want, @as(u64, @intCast(got.integer)));
        count *= want;
    }
    const offsets = t.object.get("data_offsets").?.array.items;
    const lo: usize = @intCast(offsets[0].integer);
    const hi: usize = @intCast(offsets[1].integer);
    try testing.expect(lo <= hi and hi <= data.len);
    try testing.expectEqual(count * 2, hi - lo);
    return data[lo..hi];
}

fn publicHalf(bytes: []const u8, i: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little))));
}

// These are the existing library-produced public weights, not this engine's decoder.
// A configured GPU test must find every asset; a missing asset is not a successful skip.
fn loadPublicBlock(a: Allocator, comptime rate: u32) !PublicBlock {
    const env = std.c.getenv(std.fmt.comptimePrint("SUSHI_EXL3_K{d}_FIXTURE", .{rate}));
    const path = if (env) |p| std.mem.span(p) else std.fmt.comptimePrint("lib/sushi/src/exl3/fixtures/exl3_k{d}_linear.safetensors", .{rate});
    const bytes = try sdk.io_util.readAllNoCache(a, path, 1 << 20);
    defer a.free(bytes);
    try testing.expect(bytes.len >= 8);
    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    try testing.expect(n <= bytes.len - 8);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    defer parsed.deinit();
    const data = bytes[8 + n ..];
    const code = try publicTensor(parsed.value, data, "trellis", "U16", &.{ 8, 8, rate * 16 });
    const rin = try publicTensor(parsed.value, data, "suh", "F16", &.{128});
    const rout = try publicTensor(parsed.value, data, "svh", "F16", &.{128});
    const weights = try publicTensor(parsed.value, data, "public", "F16", &.{ 128, 128 });
    const inner = try publicTensor(parsed.value, data, "inner", "F16", &.{ 128, 128 });
    var block: PublicBlock = .{ .k = rate, .code = try a.alloc(u16, code.len / 2), .rin = undefined, .rout = undefined, .weights = undefined, .inner = undefined };
    errdefer a.free(block.code);
    block.weights = try a.alloc(f64, 128 * 128);
    errdefer a.free(block.weights);
    block.inner = try a.alloc(f64, 128 * 128);
    for (block.code, 0..) |*word, i| word.* = std.mem.readInt(u16, code[i * 2 ..][0..2], .little);
    for (0..128) |i| {
        block.rin[i] = publicHalf(rin, i);
        block.rout[i] = publicHalf(rout, i);
    }
    for (block.weights, 0..) |*w, i| w.* = publicHalf(weights, i);
    for (block.inner, 0..) |*w, i| w.* = publicHalf(inner, i);
    return block;
}

fn publicProjection(g: *MlxG, block: *const PublicBlock, down: bool, bank: usize, second_slot: bool) !xq.ProjArrays(mlx.mlx_array) {
    const ni: usize = if (down) 144 else 320;
    const nj: usize = if (down) 320 else 144;
    const cap = bank + 2;
    const width = @max(16 * block.k, 32 + 16 * bank);
    const row_words = ni * nj * width;
    const code = try g.a.alloc(u16, cap * row_words);
    defer g.a.free(code);
    @memset(code, 0xa55a);
    const target = code[(cap - 1) * row_words ..];
    const tile_words = 16 * block.k;
    for (0..ni) |i| {
        for (0..nj) |j| {
            const dst = (i * nj + j) * tile_words;
            const src = ((i % 8) * 8 + j % 8) * tile_words;
            @memcpy(target[dst..][0..tile_words], block.code[src..][0..tile_words]);
        }
    }
    const rin = try g.a.alloc(f16, cap * ni * 16);
    defer g.a.free(rin);
    const rout = try g.a.alloc(f16, cap * nj * 16);
    defer g.a.free(rout);
    @memset(rin, 0);
    @memset(rout, 0);
    const factor: f32 = @floatFromInt(@as(u32, 1) << @intCast(bank));
    for (rin[(cap - 1) * ni * 16 ..], 0..) |*r, i| r.* = @floatCast(block.rin[i % 128] * factor);
    for (rout[(cap - 1) * nj * 16 ..], 0..) |*r, i| r.* = @floatCast(block.rout[i % 128] / 32);
    if (second_slot) {
        std.debug.assert(bank == 2);
        @memcpy(code[row_words..][0 .. ni * nj * tile_words], target[0 .. ni * nj * tile_words]);
        for (rin[ni * 16 ..][0 .. ni * 16], 0..) |*r, i| r.* = @floatCast(block.rin[i % 128]);
        for (rout[nj * 16 ..][0 .. nj * 16], 0..) |*r, i| r.* = @floatCast(block.rout[i % 128] / 32);
    }
    return .{
        .code = try g.hostArray(std.mem.sliceAsBytes(code), &.{ @intCast(cap), @intCast(ni), @intCast(nj), @intCast(width) }, .int16),
        .rin = try g.hostArray(std.mem.sliceAsBytes(rin), &.{ @intCast(cap), @intCast(ni * 16) }, .float16),
        .rout = try g.hostArray(std.mem.sliceAsBytes(rout), &.{ @intCast(cap), @intCast(nj * 16) }, .float16),
        .layout = .{ .fixed = .{ .k = block.k, .code_row_words = row_words } },
    };
}

fn publicInput(row: usize, col: usize) f32 {
    const value: i32 = @as(i32, @intCast((row * 3 + col * 5) % 17)) - 8;
    return @as(f32, @floatFromInt(value)) / 256;
}

fn publicActs(g: *MlxG, rows: usize) !mlx.mlx_array {
    const bits = try g.a.alloc(u16, rows * 5120);
    defer g.a.free(bits);
    for (bits, 0..) |*b, i| b.* = sdk.io_util.bf16Rne(publicInput(i / 5120, i % 5120));
    return g.hostArray(std.mem.sliceAsBytes(bits), &.{ @intCast(rows), 5120 }, .bfloat16);
}

// Expanding the independent public 128x128 matrix gives identical block columns.
// Sum input blocks first, then use dense f64 matmul; no trellis or kernel oracle enters here.
fn publicExpected(blocks: [3]*const PublicBlock, row: usize, bank: usize, hidden: *[128]f64, out: *[128]f64) void {
    var x: [128]f64 = @splat(0);
    for (0..5120) |i| x[i % 128] += publicInput(row, i);
    const factor = @as(f64, @floatFromInt(@as(u32, 1) << @intCast(bank))) / 32;
    for (0..128) |j| {
        var gate: f64 = 0;
        var up: f64 = 0;
        for (x, 0..) |v, i| {
            gate += v * blocks[0].weights[i * 128 + j];
            up += v * blocks[1].weights[i * 128 + j];
        }
        gate = @min(gate * factor, 10);
        up = std.math.clamp(up * factor, -10, 10);
        hidden[j] = gate / (1 + @exp(-gate)) * up;
    }
    for (0..128) |j| {
        var sum: f64 = 0;
        for (hidden, 0..) |v, i| sum += (v * 18) * blocks[2].weights[i * 128 + j];
        out[j] = sum * factor;
    }
}

fn prefillHalfTransform(values: *[128]f64) void {
    referenceH128(values);
    for (values) |*value| value.* = @floatCast(@as(f16, @floatCast(value.*)));
}

// Library inner matrices remain unchanged. Only the documented activation stores round.
fn publicPrefillExpected(blocks: [3]*const PublicBlock, row: usize, bank: usize, hidden: *[128]f64, out: *[128]f64) void {
    const factor: f32 = @floatFromInt(@as(u32, 1) << @intCast(bank));
    var projected: [2][128]f64 = undefined;
    for (0..2) |p| {
        var summed: [128]f64 = @splat(0);
        for (0..40) |block| {
            var transformed: [128]f64 = undefined;
            for (&transformed, 0..) |*value, i| {
                const input: f32 = @bitCast(@as(u32, sdk.io_util.bf16Rne(publicInput(row, block * 128 + i))) << 16);
                const scale: f32 = @floatCast(@as(f16, @floatCast(blocks[p].rin[i] * factor)));
                value.* = input * scale;
            }
            prefillHalfTransform(&transformed);
            for (&summed, transformed) |*sum, value| sum.* += value;
        }
        for (&projected[p], 0..) |*value, j| {
            var sum: f64 = 0;
            for (summed, 0..) |x, i| sum += x * blocks[p].inner[i * 128 + j];
            value.* = @floatCast(@as(f32, @floatCast(sum)));
        }
        referenceH128(&projected[p]);
        for (&projected[p], 0..) |*value, i| {
            const rotated: f32 = @floatCast(value.*);
            const scale: f32 = @floatCast(@as(f16, @floatCast(blocks[p].rout[i] / 32)));
            value.* = rotated * scale;
        }
    }
    for (hidden, projected[0], projected[1]) |*value, gate, up| {
        const g: f32 = @floatCast(@min(gate, 10));
        const u: f32 = @floatCast(std.math.clamp(up, -10, 10));
        const y: f32 = 1 / (1 + @exp(@abs(g)));
        const sigmoid: f32 = if (g < 0) y else 1 - y;
        value.* = (g * sigmoid) * u;
    }
    var transformed: [128]f64 = undefined;
    for (&transformed, hidden, 0..) |*value, h, i| {
        const scale: f32 = @floatCast(@as(f16, @floatCast(blocks[2].rin[i] * factor)));
        value.* = @as(f32, @floatCast(h)) * scale;
    }
    prefillHalfTransform(&transformed);
    for (out, 0..) |*value, j| {
        var sum: f64 = 0;
        for (transformed, 0..) |x, i| sum += (x * 18) * blocks[2].inner[i * 128 + j];
        value.* = @floatCast(@as(f32, @floatCast(sum)));
    }
    referenceH128(out);
    for (out, 0..) |*value, i| {
        const rotated: f32 = @floatCast(value.*);
        const scale: f32 = @floatCast(@as(f16, @floatCast(blocks[2].rout[i] / 32)));
        const scaled = rotated * scale;
        value.* = @as(f32, @bitCast(@as(u32, sdk.io_util.bf16Rne(scaled)) << 16));
    }
}

test "exl3 mixed oracle: FP16 activation stores precede repeated-block accumulation" {
    var block: [128]f64 = @splat(0x1p-30);
    prefillHalfTransform(&block);
    for (block) |value| try testing.expectEqual(@as(f64, 0), value);
    var combined: [128]f64 = @splat(40 * 0x1p-30);
    prefillHalfTransform(&combined);
    try testing.expect(combined[0] > 0);
    // Each block's nonzero mathematical DC component is lost at its own F16 store.
    try testing.expect(combined[0] != block[0] * 40);
}

fn expectPublicRows(g: *MlxG, value: mlx.mlx_array, blocks: [3]*const PublicBlock, rows: usize, banked: bool, bank: usize, hidden_output: bool, slots: ?[]const u32) !void {
    const bytes = try g.hostBytes(value);
    defer g.a.free(bytes);
    const dtype = g.dtypeOf(value);
    const cols: usize = if (hidden_output) 2304 else 5120;
    try testing.expectEqual(rows * cols * size(dtype), bytes.len);
    var worst: f64 = 0;
    for (0..rows) |r| {
        var hidden: [128]f64 = undefined;
        var out: [128]f64 = undefined;
        const expected_bank = if (slots) |s| (if (s[r] == 1) @as(usize, 0) else 2) else if (banked) r % 3 else bank;
        publicExpected(blocks, r, expected_bank, &hidden, &out);
        const expected = if (hidden_output) &hidden else &out;
        var square: f64 = 0;
        for (expected) |v| square += v * v;
        const rms = @sqrt(square / 128);
        try testing.expect(rms > 1e-12);
        for (0..cols) |c| {
            const got = getFloat(bytes, r * cols + c, dtype);
            try testing.expect(std.math.isFinite(got));
            worst = @max(worst, @abs(got - expected[c % 128]) / rms);
        }
    }
    // Public weights are rounded to f16; tiled stages additionally round to half.
    // This is a full-projection error bar, not a bit-equality claim across arithmetic families.
    if (worst > 1.0 / 64.0) std.debug.print("public EXL3 projection error/RMS {d}, rows {d}, banked {}, hidden {}\n", .{ worst, rows, banked, hidden_output });
    try testing.expect(worst <= 1.0 / 64.0);
}

test "dsv41 kernels ops gpu: independent public H128 blocks cover full rate projections and prepared routes" {
    const enabled = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.span(enabled), "0")) return error.SkipZigTest;
    const a = testing.allocator;
    var fixtures_arena = std.heap.ArenaAllocator.init(a);
    defer fixtures_arena.deinit();
    const fa = fixtures_arena.allocator();
    const fixtures = [_]PublicBlock{ try loadPublicBlock(fa, 2), try loadPublicBlock(fa, 3), try loadPublicBlock(fa, 4) };
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var gemv = try xq.Gemv(MlxG).initRates(a, &g, &reg, .{}, 7);
    defer gemv.deinit(&g);
    var prep = try xq.RinPrep(MlxG).init(&g, &reg);
    defer prep.deinit(&g);
    var banked = try xq.Banked(MlxG).init(&g, &reg, .{}, &gemv);
    defer banked.deinit(&g);
    for ([_][3]usize{ .{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, 2, 2 }, .{ 1, 1, 2 }, .{ 0, 2, 1 }, .{ 2, 0, 2 } }) |rates| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const blocks: [3]*const PublicBlock = .{ &fixtures[rates[0]], &fixtures[rates[1]], &fixtures[rates[2]] };
        var banks: [3]xq.BankArrays(mlx.mlx_array) = undefined;
        for (&banks, 0..) |*bank, b| bank.* = .{
            .gate = try publicProjection(&g, blocks[0], false, b, false),
            .up = try publicProjection(&g, blocks[1], false, b, false),
            .down = try publicProjection(&g, blocks[2], true, b, false),
        };
        for (1..49) |rows| {
            const batch = g.mark();
            defer g.resetTo(batch);
            const act = try publicActs(&g, rows);
            var tokens: [48]i32 = undefined;
            var single_ids: [48]u32 = undefined;
            var packed_ids: [48]u32 = undefined;
            for (0..rows) |r| {
                tokens[r] = @intCast(r);
                single_ids[r] = 3;
                packed_ids[r] = @intCast(((r % 3) << 24) | (r % 3 + 1));
            }
            const tok = try g.hostArray(std.mem.sliceAsBytes(tokens[0..rows]), &.{@intCast(rows)}, .int32);
            const ids = try g.hostArray(std.mem.sliceAsBytes(single_ids[0..rows]), &.{@intCast(rows)}, .uint32);
            const bank_ids = try g.hostArray(std.mem.sliceAsBytes(packed_ids[0..rows]), &.{@intCast(rows)}, .uint32);
            const bank = banks[2];
            const xs = try prep.inRin(&g, act, tok, bank.gate.rin, bank.up.rin, ids);
            const zs = try gemv.projectGu(&g, xs[0], xs[1], ids, bank.gate.code, bank.up.code, bank.gate.layout, bank.up.layout);
            const hidden = try prep.guEpi(&g, zs[0], zs[1], bank.gate.rout, bank.up.rout, ids);
            try expectPublicRows(&g, hidden, blocks, rows, false, 2, true, null);
            const hd = try prep.dinRin(&g, hidden, bank.down.rin, ids);
            const zd = try gemv.project(&g, .down, hd, ids, bank.down.code, bank.down.layout);
            const out = try prep.dpost(&g, zd, bank.down.rout, ids);
            try expectPublicRows(&g, out, blocks, rows, false, 2, false, null);
            const bh = try banked.gateUp(&g, act, tok, bank_ids, &banks);
            try expectPublicRows(&g, bh, blocks, rows, true, 0, true, null);
            const bo = try banked.down(&g, bh, bank_ids, &banks);
            try expectPublicRows(&g, bo, blocks, rows, true, 0, false, null);
        }
        var prefill = try xq.DigXPrefill(MlxG).init(a, &reg, .{ .wave = 1, .inflight = 2, .row_budget = 128, .carry_rows = 128 }, &diag);
        defer prefill.deinit(&g);
        for ([_]usize{ 1, 127, 128, 129 }) |rows| {
            const batch = g.mark();
            defer g.resetTo(batch);
            const act = try publicActs(&g, rows);
            const slots: [129]u32 = @splat(3);
            const out = try prefill.call(&g, act, .{ .slot = slots[0..rows] }, banks[2]);
            defer g.release(out);
            try prefill.finish(&g);
            try expectPublicRows(&g, out, blocks, rows, false, 2, false, null);
        }
    }
}

fn getFloat(bytes: []const u8, i: usize, dtype: Dtype) f64 {
    return switch (dtype) {
        .float32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little))),
        .float16 => publicHalf(bytes, i),
        .bfloat16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16)),
        else => unreachable,
    };
}

fn referenceH128(values: []f64) void {
    var block: usize = 0;
    while (block < values.len) : (block += 128) {
        var out: [128]f64 = @splat(0);
        for (&out, 0..) |*sum, r| {
            for (values[block..][0..128], 0..) |v, c| {
                sum.* += (if (@popCount(r & c) % 2 == 0) @as(f64, 1) else -1) * v;
            }
            sum.* *= 0.08838834764831845;
        }
        @memcpy(values[block..][0..128], &out);
    }
}

// MSB-first circular bit reader and dense H128, independent of the engine's funnel
// decoder/butterflies. This oracle is calibrated against library-produced public assets.
fn referenceProjection(a: Allocator, code: []const u16, rate: usize, rin: []const f32, rout: []const f32, x: []const f32, rows: usize) ![]f64 {
    const ni = rin.len / 16;
    const nj = rout.len / 16;
    const transformed = try a.alloc(f64, x.len);
    defer a.free(transformed);
    for (transformed, x, 0..) |*v, xv, i| v.* = @as(f64, xv) * rin[i % rin.len];
    for (0..rows) |r| referenceH128(transformed[r * rin.len ..][0..rin.len]);
    const output = try a.alloc(f64, rows * rout.len);
    errdefer a.free(output);
    @memset(output, 0);
    for (0..ni) |ti| {
        for (0..nj) |tj| {
            const tile = code[(ti * nj + tj) * 16 * rate ..][0 .. 16 * rate];
            for (0..256) |p| {
                var state: u16 = 0;
                for (0..16) |i| {
                    const bit = ((p + 1) * rate + 256 * rate - 16 + i) % (256 * rate);
                    const word = @as(u32, tile[2 * (bit / 32)]) | (@as(u32, tile[2 * (bit / 32) + 1]) << 16);
                    state = (state << 1) | @as(u16, @intCast((word >> @intCast(31 - bit % 32)) & 1));
                }
                const mixed = @as(u32, state) *% 0x83DCD12D;
                var byte_sum: u32 = 0;
                inline for (0..4) |b| byte_sum += (mixed >> (8 * b)) & 255;
                const inverse: f64 = @as(f16, @bitCast(@as(u16, 0x1eee)));
                const bias: f64 = @as(f16, @bitCast(@as(u16, 0xc931)));
                const weight: f64 = @as(f16, @floatCast((1024 + @as(f64, @floatFromInt(byte_sum))) * inverse + bias));
                const lane = p / 8;
                const local_row = (lane % 4) * 2 + p % 2 + ((p % 4) / 2) * 8;
                const local_col = lane / 4 + (p % 8) / 4 * 8;
                for (0..rows) |r| {
                    output[r * rout.len + tj * 16 + local_col] += transformed[r * rin.len + ti * 16 + local_row] * weight;
                }
            }
        }
    }
    for (0..rows) |r| {
        const row = output[r * rout.len ..][0..rout.len];
        referenceH128(row);
        for (row, rout) |*v, scale| v.* *= scale;
    }
    return output;
}

test "exl3 sushi parity: independent scalar full projection oracle matches public assets" {
    const a = testing.allocator;
    inline for (.{ 2, 3, 4 }) |rate| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const fa = arena.allocator();
        const block = loadPublicBlock(fa, rate) catch |err| switch (err) {
            error.FileNotFound => {
                // An explicit fixture path or GPU gate must never turn missing evidence into a skip.
                if (std.c.getenv(std.fmt.comptimePrint("SUSHI_EXL3_K{d}_FIXTURE", .{rate})) != null or std.c.getenv("DSV41_KERNELS_GPU") != null) return err;
                return error.SkipZigTest;
            },
            else => return err,
        };
        var x: [4 * 128]f32 = undefined;
        for (&x, 0..) |*v, i| v.* = publicInput(i / 128, i % 128);
        const reference = try referenceProjection(fa, block.code, rate, &block.rin, &block.rout, &x, 4);
        for (0..4) |r| {
            var square: f64 = 0;
            var error_max: f64 = 0;
            for (0..128) |j| {
                var want: f64 = 0;
                for (x[r * 128 ..][0..128], 0..) |v, i| want += v * block.weights[i * 128 + j];
                square += want * want;
                error_max = @max(error_max, @abs(reference[r * 128 + j] - want));
            }
            try testing.expect(square > 0);
            try testing.expect(error_max <= @sqrt(square / 128) / 512);
        }
    }
}

test "dsv41 kernels ops gpu: captured Pollard K4 down full H128 scaled projection" {
    const enabled = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.span(enabled), "0")) return error.SkipZigTest;
    const fixture = std.c.getenv("DSV41_PUBLIC_DOWN_FIXTURE") orelse return error.PublicDownFixtureRequired;
    const a = testing.allocator;
    // bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard, f129e31a81e1337aa33e129e2d847fc7e37c8733.
    const bytes = try sdk.io_util.readAllNoCache(a, std.mem.span(fixture), 8 << 20);
    defer a.free(bytes);
    try testing.expect(hexEql("53fa4386b151bd4b7adc58299b45a9abfeb46fd537121d174c09688e3c11fc9e", bytes));
    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    defer parsed.deinit();
    const data = bytes[8 + n ..];
    const prefix = "layers.0.ffn.experts.0.w2.";
    const code_bytes = try publicTensor(parsed.value, data, prefix ++ "trellis", "I16", &.{ 144, 320, 64 });
    const rin_bytes = try publicTensor(parsed.value, data, prefix ++ "suh", "F16", &.{2304});
    const rout_bytes = try publicTensor(parsed.value, data, prefix ++ "svh", "F16", &.{5120});
    const code = try a.alloc(u16, code_bytes.len / 2);
    defer a.free(code);
    for (code, 0..) |*v, i| v.* = std.mem.readInt(u16, code_bytes[i * 2 ..][0..2], .little);
    var rin: [2304]f32 = undefined;
    var rout: [5120]f32 = undefined;
    for (&rin, 0..) |*v, i| v.* = publicHalf(rin_bytes, i);
    for (&rout, 0..) |*v, i| v.* = publicHalf(rout_bytes, i);
    var x: [3 * 2304]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = publicInput(i / 2304, i % 2304);
    const want = try referenceProjection(a, code, 4, &rin, &rout, &x, 3);
    defer a.free(want);
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    const bank_code = try a.alloc(u16, 3 * code.len);
    defer a.free(bank_code);
    @memset(bank_code, 0xa55a);
    @memcpy(bank_code[2 * code.len ..], code);
    const bank_rin = try a.alloc(u8, 3 * rin_bytes.len);
    defer a.free(bank_rin);
    const bank_rout = try a.alloc(u8, 3 * rout_bytes.len);
    defer a.free(bank_rout);
    @memset(bank_rin, 0);
    @memset(bank_rout, 0);
    @memcpy(bank_rin[2 * rin_bytes.len ..], rin_bytes);
    @memcpy(bank_rout[2 * rout_bytes.len ..], rout_bytes);
    const projection: xq.ProjArrays(mlx.mlx_array) = .{
        .code = try g.hostArray(std.mem.sliceAsBytes(bank_code), &.{ 3, 144, 320, 64 }, .int16),
        .rin = try g.hostArray(bank_rin, &.{ 3, 2304 }, .float16),
        .rout = try g.hostArray(bank_rout, &.{ 3, 5120 }, .float16),
        .layout = .{ .fixed = .{ .k = 4, .code_row_words = code.len } },
    };
    const input = try g.hostArray(std.mem.sliceAsBytes(&x), &.{ 3, 2304 }, .float32);
    const ids = try g.hostArray(std.mem.sliceAsBytes(&[_]u32{ 2, 2, 2 }), &.{3}, .uint32);
    var gemv = try xq.Gemv(MlxG).initRates(a, &g, &reg, .{}, 4);
    defer gemv.deinit(&g);
    var prep = try xq.RinPrep(MlxG).init(&g, &reg);
    defer prep.deinit(&g);
    const transformed = try prep.dinRin(&g, input, projection.rin, ids);
    const inner = try gemv.project(&g, .down, transformed, ids, projection.code, projection.layout);
    const output = try prep.dpost(&g, inner, projection.rout, ids);
    const got = try g.hostBytes(output);
    defer a.free(got);
    try testing.expectEqual(want.len * 4, got.len);
    for (0..3) |r| {
        var square: f64 = 0;
        var error_max: f64 = 0;
        for (want[r * 5120 ..][0..5120], 0..) |v, c| {
            const actual = getFloat(got, r * 5120 + c, .float32);
            try testing.expect(std.math.isFinite(actual));
            square += v * v;
            error_max = @max(error_max, @abs(actual - v));
        }
        try testing.expect(square > 0);
        try testing.expect(error_max <= @sqrt(square / 5120) / 512);
    }
}

fn publicAccepted(g: *MlxG, reg: *const xk.Registry, rates: [3]u32, diag: *xk.Diag) !*xq.Accepted(MlxG) {
    const a = g.a;
    var gemv = try xq.Gemv(MlxG).initRates(a, g, reg, .{}, 7);
    errdefer gemv.deinit(g);
    var prep = try xq.RinPrep(MlxG).init(g, reg);
    errdefer prep.deinit(g);
    var tok: [48]mlx.mlx_array = @splat(.{});
    var n_tok: usize = 0;
    errdefer for (tok[0..n_tok]) |x| g.release(x);
    var indices: [48]i32 = undefined;
    for (&indices, 0..) |*index, i| index.* = @intCast(i);
    for (1..49) |rows| {
        tok[rows - 1] = g.keep(try g.hostArray(std.mem.sliceAsBytes(indices[0..rows]), &.{@intCast(rows)}, .int32));
        n_tok = rows;
    }
    const waves = try a.alloc(xq.DigXPrefill(MlxG), 1);
    errdefer a.free(waves);
    waves[0] = try xq.DigXPrefill(MlxG).init(a, reg, .{ .wave = 2, .inflight = 2, .row_budget = 256, .carry_rows = 512 }, diag);
    errdefer waves[0].deinit(g);
    const layer_rates = try a.alloc(xq.LayerRates, 1);
    errdefer a.free(layer_rates);
    layer_rates[0] = .{ .fixed = rates };
    const acc = try a.create(xq.Accepted(MlxG));
    acc.* = .{ .a = a, .reg = reg, .layer_rates = layer_rates, .gemv = gemv, .prep = prep, .tok = tok, .n_tok = n_tok, .waves = waves };
    return acc;
}

test "dsv41 kernels ops gpu: accepted routed forms and carried BF16 mixed expert waves match public weights" {
    const enabled = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.span(enabled), "0")) return error.SkipZigTest;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fa = arena.allocator();
    const fixtures = [_]PublicBlock{ try loadPublicBlock(fa, 2), try loadPublicBlock(fa, 3), try loadPublicBlock(fa, 4) };
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    const forms_cases = [_]xq.Forms{
        .{ .gu_one = true },
        .{ .down_pair = true },
        .{ .gu_one = true, .down_pair = true },
    };
    for ([_][3]u32{ .{ 3, 3, 3 }, .{ 3, 3, 4 }, .{ 2, 4, 3 }, .{ 4, 2, 4 } }) |rates| {
        const bank_mark = g.mark();
        defer g.resetTo(bank_mark);
        const blocks: [3]*const PublicBlock = .{ &fixtures[rates[0] - 2], &fixtures[rates[1] - 2], &fixtures[rates[2] - 2] };
        var banks: [3]xq.BankArrays(mlx.mlx_array) = undefined;
        for (&banks, 0..) |*bank, b| bank.* = .{
            .gate = try publicProjection(&g, blocks[0], false, b, b == 2),
            .up = try publicProjection(&g, blocks[1], false, b, b == 2),
            .down = try publicProjection(&g, blocks[2], true, b, b == 2),
        };
        for (forms_cases) |forms| {
            const route_mark = g.mark();
            defer g.resetTo(route_mark);
            const acc = try publicAccepted(&g, &reg, rates, &diag);
            defer acc.deinit(&g);
            try acc.routeForms(&g, forms);
            try acc.routeBanked(&g);
            for (banks) |bank| try acc.checkLayerBank(&g, 0, bank, &diag);
            for ([_]usize{ 1, 3, 7, 48 }) |rows| {
                const batch = g.mark();
                defer g.resetTo(batch);
                const act = try publicActs(&g, rows);
                var ids_host: [48]u32 = undefined;
                for ([_]usize{ 0, 2 }) |b| {
                    @memset(ids_host[0..rows], @intCast(b + 1));
                    const ids = try g.hostArray(std.mem.sliceAsBytes(ids_host[0..rows]), &.{@intCast(rows)}, .uint32);
                    const hidden = try acc.gateUp(&g, act, ids, banks[b].gate, banks[b].up);
                    try expectPublicRows(&g, hidden, blocks, rows, false, b, true, null);
                    const out = try acc.down(&g, hidden, ids, banks[b].down);
                    try expectPublicRows(&g, out, blocks, rows, false, b, false, null);
                }
                for (0..rows) |r| ids_host[r] = @intCast(((r % 3) << 24) | (r % 3 + 1));
                const ids = try g.hostArray(std.mem.sliceAsBytes(ids_host[0..rows]), &.{@intCast(rows)}, .uint32);
                const hidden = try acc.gateUpBanked(&g, act, ids, &banks);
                try expectPublicRows(&g, hidden, blocks, rows, true, 0, true, null);
                const out = try acc.downBanked(&g, hidden, ids, &banks);
                try expectPublicRows(&g, out, blocks, rows, true, 0, false, null);
                if (std.mem.eql(u32, &rates, &.{ 3, 3, 3 })) {
                    // All-tight aliases engage the banked form texts, not the stride-aware fallback.
                    const tight_banks = [_]xq.BankArrays(mlx.mlx_array){ banks[0], banks[0], banks[0] };
                    for (0..rows) |r| ids_host[r] = @intCast(((r % 3) << 24) | 1);
                    const tight_ids = try g.hostArray(std.mem.sliceAsBytes(ids_host[0..rows]), &.{@intCast(rows)}, .uint32);
                    const th = try acc.gateUpBanked(&g, act, tight_ids, &tight_banks);
                    try expectPublicRows(&g, th, blocks, rows, false, 0, true, null);
                    const to = try acc.downBanked(&g, th, tight_ids, &tight_banks);
                    try expectPublicRows(&g, to, blocks, rows, false, 0, false, null);
                }
            }
            try acc.routeExpertBf16();
            var slot_rows: [3][257]u32 = undefined;
            const counts = [_][2]usize{ .{ 127, 128 }, .{ 128, 129 }, .{ 1, 129 } };
            var outputs: [3]mlx.mlx_array = @splat(.{});
            var output_count: usize = 0;
            defer for (outputs[0..output_count]) |out| g.release(out);
            for (counts, 0..) |count, call| {
                var remaining = count;
                const rows = count[0] + count[1];
                for (0..rows) |r| {
                    var expert = (r + call) % 2;
                    if (remaining[expert] == 0) expert = 1 - expert;
                    remaining[expert] -= 1;
                    slot_rows[call][r] = if (expert == 0) 1 else 3;
                }
                const act = try publicActs(&g, rows);
                outputs[call] = try acc.prefill(&g, 0, act, .{ .slot = slot_rows[call][0..rows] }, banks[2]);
                output_count += 1;
                try testing.expect(acc.waves[0].flight.items.len > 0);
            }
            try acc.finishPrefill(&g);
            try testing.expectEqual(@as(usize, 0), acc.waves[0].flight.items.len);
            for (outputs, counts, 0..) |out, count, call| {
                const rows = count[0] + count[1];
                try testing.expectEqual(Dtype.bfloat16, g.dtypeOf(out));
                try expectPublicRows(&g, out, blocks, rows, false, 0, false, slot_rows[call][0..rows]);
            }
        }
    }
}

const mixed_triples = [3][2][3]usize{
    .{ .{ 2, 0, 3 }, .{ 3, 2, 0 } },
    .{ .{ 1, 3, 0 }, .{ 2, 1, 2 } },
    .{ .{ 0, 1, 2 }, .{ 1, 0, 3 } },
};

fn compactPublicProjection(g: *MlxG, fixtures: *const [4]PublicBlock, bank: usize, projection: usize) !xq.ProjArrays(mlx.mlx_array) {
    const ni: usize = if (projection == 2) 144 else 320;
    const nj: usize = if (projection == 2) 320 else 144;
    var descriptors: [2][2]u64 = undefined;
    var words: usize = 256;
    for (&descriptors, 0..) |*descriptor, slot| {
        const k = fixtures[mixed_triples[bank][slot][projection]].k;
        descriptor.* = .{ words, k };
        words += ni * nj * 16 * k + (slot + 1) * 256;
    }
    const code = try g.a.alloc(u16, words);
    defer g.a.free(code);
    @memset(code, 0xa55a);
    const rin = try g.a.alloc(f16, 2 * ni * 16);
    defer g.a.free(rin);
    const rout = try g.a.alloc(f16, 2 * nj * 16);
    defer g.a.free(rout);
    for (descriptors, 0..) |descriptor, slot| {
        const block = &fixtures[mixed_triples[bank][slot][projection]];
        const tw: usize = 16 * block.k;
        for (0..ni) |i| for (0..nj) |j| {
            const dst = descriptor[0] + (i * nj + j) * tw;
            const src = ((i % 8) * 8 + j % 8) * tw;
            @memcpy(code[dst..][0..tw], block.code[src..][0..tw]);
        };
        const factor: f32 = @floatFromInt(@as(u32, 1) << @intCast(bank + slot));
        for (rin[slot * ni * 16 ..][0 .. ni * 16], 0..) |*v, i| v.* = @floatCast(block.rin[i % 128] * factor);
        for (rout[slot * nj * 16 ..][0 .. nj * 16], 0..) |*v, i| v.* = @floatCast(block.rout[i % 128] / 32);
    }
    return .{
        .code = try g.hostArray(std.mem.sliceAsBytes(code), &.{ @intCast(words / 256), 256 }, .int16),
        .rin = try g.hostArray(std.mem.sliceAsBytes(rin), &.{ 2, @intCast(ni * 16) }, .float16),
        .rout = try g.hostArray(std.mem.sliceAsBytes(rout), &.{ 2, @intCast(nj * 16) }, .float16),
        .layout = .{ .compact = try g.hostArray(std.mem.sliceAsBytes(&descriptors), &.{ 2, 2 }, .uint64) },
    };
}

const MixedReference = enum { dense_public, prefill_f16_bf16 };

fn mixedRowMismatch(bytes: []const u8, dtype: Dtype, expected: []const f64) !?usize {
    const rms = try sparkReferenceRms(expected);
    for (0..bytes.len / size(dtype)) |column| {
        const actual = getFloat(bytes, column, dtype);
        if (!std.math.isFinite(actual) or @abs(actual - expected[column % expected.len]) > rms / 64)
            return column;
    }
    return null;
}

test "exl3 mixed oracle: finite actual rejects nonfinite mixed reference" {
    const actual = [_]f32{ 1, 1 };
    for ([_]f64{ std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) }) |invalid| {
        try testing.expectError(error.InvalidSparkReference, mixedRowMismatch(std.mem.sliceAsBytes(&actual), .float32, &.{ 1, invalid }));
    }
    try testing.expectEqual(@as(?usize, null), try mixedRowMismatch(std.mem.sliceAsBytes(&actual), .float32, &.{ 1, 1 }));
    const outside = [_]f32{ 1, 1 + 1.0 / 32.0 };
    try testing.expectEqual(@as(?usize, 1), try mixedRowMismatch(std.mem.sliceAsBytes(&outside), .float32, &.{ 1, 1 }));
}

fn expectMixedRows(g: *MlxG, value: mlx.mlx_array, fixtures: *const [4]PublicBlock, ids: []const u32, hidden_output: bool, reference: MixedReference) !void {
    const bytes = try g.hostBytes(value);
    defer g.a.free(bytes);
    const dtype = g.dtypeOf(value);
    const cols: usize = if (hidden_output) 2304 else 5120;
    try testing.expectEqual(ids.len * cols * size(dtype), bytes.len);
    const Expected = struct { hidden: [128]f64, out: [128]f64 };
    var cache: [6][17]Expected = undefined;
    var ready: [6][17]bool = @splat(@splat(false));
    for (ids, 0..) |id, row| {
        if (id == 0xffffffff) {
            for (bytes[row * cols * size(dtype) ..][0 .. cols * size(dtype)]) |byte| try testing.expectEqual(@as(u8, 0), byte);
            continue;
        }
        const bank = id >> 24;
        const slot = id & 0xffffff;
        const triple = mixed_triples[bank][slot];
        const blocks: [3]*const PublicBlock = .{ &fixtures[triple[0]], &fixtures[triple[1]], &fixtures[triple[2]] };
        const cached = &cache[bank * 2 + slot][row % 17];
        if (!ready[bank * 2 + slot][row % 17]) {
            switch (reference) {
                .dense_public => publicExpected(blocks, row % 17, bank + slot, &cached.hidden, &cached.out),
                .prefill_f16_bf16 => publicPrefillExpected(blocks, row % 17, bank + slot, &cached.hidden, &cached.out),
            }
            ready[bank * 2 + slot][row % 17] = true;
        }
        const expected = if (hidden_output) &cached.hidden else &cached.out;
        const row_bytes = bytes[row * cols * size(dtype) ..][0 .. cols * size(dtype)];
        if (try mixedRowMismatch(row_bytes, dtype, expected)) |column| {
            std.debug.print("mixed projection mismatch: rows={d} row={d} bank={d} slot={d} hidden={} dtype={s} col={d} actual={d} expected={d}\n", .{
                ids.len, row, bank, slot, hidden_output, @tagName(dtype), column, getFloat(row_bytes, column, dtype), expected[column % 128],
            });
            return error.TestUnexpectedResult;
        }
    }
}

fn compactAccepted(g: *MlxG, reg: *const xk.Registry, diag: *xk.Diag) !*xq.Accepted(MlxG) {
    // Numerical construction is deliberately separate from the public rate-admission gate.
    const acc = try publicAccepted(g, reg, .{ 3, 3, 3 }, diag);
    errdefer acc.deinit(g);
    var gemv = try xq.Gemv(MlxG).initStorage(g.a, g, reg, .{}, 0, true);
    errdefer gemv.deinit(g);
    const classes = try g.a.alloc([3]u32, 2);
    classes[0] = .{ 5, 2, 4 };
    classes[1] = .{ 4, 5, 2 };
    acc.gemv.deinit(g);
    acc.gemv = gemv;
    acc.compact = true;
    acc.layer_rates[0] = .{ .compact = classes };
    return acc;
}

test "exl3 mixed gpu: heterogeneous descriptor triples decode verify and carried BF16 prefill" {
    const enabled = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.span(enabled), "0")) return error.SkipZigTest;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fixtures = [4]PublicBlock{ try loadPublicBlock(arena.allocator(), 2), try loadPublicBlock(arena.allocator(), 3), try loadPublicBlock(arena.allocator(), 5), try loadPublicBlock(arena.allocator(), 4) };
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var banks: [3]xq.BankArrays(mlx.mlx_array) = undefined;
    for (&banks, 0..) |*bank, b| bank.* = .{
        .gate = try compactPublicProjection(&g, &fixtures, b, 0),
        .up = try compactPublicProjection(&g, &fixtures, b, 1),
        .down = try compactPublicProjection(&g, &fixtures, b, 2),
    };
    const acc = try compactAccepted(&g, &reg, &diag);
    defer acc.deinit(&g);
    try acc.routeForms(&g, .{ .gu_one = true, .down_pair = true });
    try acc.routeBanked(&g);
    try testing.expectEqual(@as(u32, 0xffffffff), acc.devrouteMissingId());
    for (banks) |bank| try acc.checkLayerBank(&g, 0, bank, &diag);
    for (1..49) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const act = try publicActs(&g, rows);
        var routed_ids: [48]u32 = undefined;
        var local: [48]u32 = undefined;
        for (0..rows) |r| {
            routed_ids[r] = if (r % 5 == 4) 0xffffffff else @intCast((((r / 2) % 3) << 24) | (r % 2));
            local[r] = @intCast(r % 2);
        }
        const ids = try g.hostArray(std.mem.sliceAsBytes(routed_ids[0..rows]), &.{@intCast(rows)}, .uint32);
        const hidden = try acc.gateUpBanked(&g, act, ids, &banks);
        try expectMixedRows(&g, hidden, &fixtures, routed_ids[0..rows], true, .dense_public);
        const out = try acc.downBanked(&g, hidden, ids, &banks);
        try expectMixedRows(&g, out, &fixtures, routed_ids[0..rows], false, .dense_public);
        if (rows == 1 or rows == 8 or rows == 48) {
            const local_ids = try g.hostArray(std.mem.sliceAsBytes(local[0..rows]), &.{@intCast(rows)}, .uint32);
            for (banks, 0..) |bank, b| {
                var expected_ids: [48]u32 = undefined;
                for (local[0..rows], 0..) |slot, r| expected_ids[r] = (@as(u32, @intCast(b)) << 24) | slot;
                const direct_hidden = try acc.gateUp(&g, act, local_ids, bank.gate, bank.up);
                try expectMixedRows(&g, direct_hidden, &fixtures, expected_ids[0..rows], true, .dense_public);
                const direct_out = try acc.down(&g, direct_hidden, local_ids, bank.down);
                try expectMixedRows(&g, direct_out, &fixtures, expected_ids[0..rows], false, .dense_public);
            }
        }
    }
    try acc.routeExpertBf16();
    const counts = [_][2]usize{ .{ 1, 129 }, .{ 127, 128 }, .{ 128, 129 }, .{ 129, 1 } };
    for (banks, 0..) |bank, b| {
        const mark = g.mark();
        defer g.resetTo(mark);
        var slot_rows: [4][257]u32 = undefined;
        var outputs: [4]mlx.mlx_array = undefined;
        var built: usize = 0;
        defer for (outputs[0..built]) |out| g.release(out);
        for (counts, 0..) |count, call| {
            const rows = count[0] + count[1];
            for (slot_rows[call][0..rows], 0..) |*slot, r| slot.* = @intFromBool(r >= count[0]);
            const act = try publicActs(&g, rows);
            outputs[call] = try acc.prefill(&g, 0, act, .{ .slot = slot_rows[call][0..rows] }, bank);
            built += 1;
        }
        try acc.finishPrefill(&g);
        for (counts, outputs, 0..) |count, out, call| {
            const rows = count[0] + count[1];
            var expected_ids: [257]u32 = undefined;
            for (slot_rows[call][0..rows], 0..) |slot, r| expected_ids[r] = (@as(u32, @intCast(b)) << 24) | slot;
            try testing.expectEqual(Dtype.bfloat16, g.dtypeOf(out));
            try expectMixedRows(&g, out, &fixtures, expected_ids[0..rows], false, .prefill_f16_bf16);
        }
    }
}

fn poisonArray(g: *MlxG, rows: usize, cols: usize, dtype: Dtype) !mlx.mlx_array {
    const bytes = try g.a.alloc(u8, rows * cols * size(dtype));
    defer g.a.free(bytes);
    @memset(bytes, 0xff);
    return g.hostArray(bytes, &.{ @intCast(rows), @intCast(cols) }, dtype);
}

fn expectExactZero(g: *MlxG, array: mlx.mlx_array) !void {
    const bytes = try g.hostBytes(array);
    defer g.a.free(bytes);
    for (bytes) |byte| try testing.expectEqual(@as(u8, 0), byte);
}

test "exl3 mixed gpu: all six miss stages ignore poisoned unpublished storage and NaNs" {
    const enabled = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.span(enabled), "0")) return error.SkipZigTest;
    const a = testing.allocator;
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var gemv = try xq.Gemv(MlxG).initStorage(a, &g, &reg, .{}, 0, true);
    defer gemv.deinit(&g);
    var banked = try xq.Banked(MlxG).init(&g, &reg, .{}, &gemv);
    defer banked.deinit(&g);
    const code = try poisonArray(&g, 1, 256, .int16);
    const desc = try g.hostArray(std.mem.sliceAsBytes(&[_]u64{ std.math.maxInt(u64), std.math.maxInt(u64) }), &.{ 1, 2 }, .uint64);
    const rin = try poisonArray(&g, 1, 5120, .float16);
    const rout = try poisonArray(&g, 1, 2304, .float16);
    for (1..49) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const miss: [48]u32 = @splat(0xffffffff);
        const tok_host: [48]i32 = @splat(std.math.maxInt(i32));
        const ids = try g.hostArray(std.mem.sliceAsBytes(miss[0..rows]), &.{@intCast(rows)}, .uint32);
        const tok = try g.hostArray(std.mem.sliceAsBytes(tok_host[0..rows]), &.{@intCast(rows)}, .int32);
        const act = try poisonArray(&g, rows, 5120, .bfloat16);
        const zg = try poisonArray(&g, rows, 2304, .float32);
        const zd = try poisonArray(&g, rows, 5120, .float32);
        var pair: [2]mlx.mlx_array = undefined;
        var out: [1]mlx.mlx_array = undefined;
        try banked.in_rin_p.launch(&g, rows, &.{ act, tok, rin, rin, rin, rin, rin, rin, ids }, &pair);
        for (pair) |value| try expectExactZero(&g, value);
        try banked.gu_epi_p.launch(&g, rows, &.{ zg, zg, rout, rout, rout, rout, rout, rout, ids }, &out);
        try expectExactZero(&g, out[0]);
        try banked.din_rin_p.launch(&g, rows, &.{ zg, rout, rout, rout, ids }, &out);
        try expectExactZero(&g, out[0]);
        try banked.dpost_p.launch(&g, rows, &.{ zd, rin, rin, rin, ids }, &out);
        try expectExactZero(&g, out[0]);
        try banked.gu_p.launch(&g, rows, &.{ zd, ids, code, code, code, desc, desc, desc }, &out);
        try expectExactZero(&g, out[0]);
        try banked.dn_p.launch(&g, rows, &.{ zg, ids, code, code, code, desc, desc, desc }, &out);
        try expectExactZero(&g, out[0]);
    }
}

const SparkProjection = struct {
    k: u8,
    input: usize,
    output: usize,
    code: []const u8,
    rin: []const u8,
    rout: []const u8,
    weights: []f32,
    inner: []const f32,
};

fn sparkProjection(a: Allocator, header: std.json.Value, data: []const u8, comptime expert_prefix: []const u8, comptime k: u8, comptime name: []const u8, comptime input: usize, comptime output: usize) !SparkProjection {
    const prefix = expert_prefix ++ name ++ ".";
    errdefer std.debug.print("Spark projection failed: K{d} tensor={s}\n", .{ k, prefix });
    const code = try publicTensor(header, data, prefix ++ "trellis", "I16", &.{ input / 16, output / 16, 16 * k });
    const rin = try publicTensor(header, data, prefix ++ "suh", "F16", &.{input});
    const rout = try publicTensor(header, data, prefix ++ "svh", "F16", &.{output});
    const inner = try publicTensor(header, data, prefix ++ "inner", "F16", &.{ input, output });
    const public = try publicTensor(header, data, prefix ++ "public", "F16", &.{ input, output });
    try testing.expect(!std.mem.eql(u8, code[0 .. code.len - 32 * k], code[32 * k ..]));
    const words = try a.alloc(i16, code.len / 2);
    defer a.free(words);
    for (words, 0..) |*word, i| word.* = std.mem.readInt(i16, code[i * 2 ..][0..2], .little);
    const actual = try a.alloc(u16, input * output);
    defer a.free(actual);
    var table: [65536]u16 = undefined;
    xk.mul1Table(&table);
    xk.reconstruct(words, input / 16, output / 16, k, &table, actual, null);
    // The expected bytes came from the external pinned CPU library, never this reconstruction.
    for (actual, 0..) |word, i| {
        const want = std.mem.readInt(u16, inner[i * 2 ..][0..2], .little);
        if (want != word) {
            std.debug.print("Spark inner mismatch: index={d} expected=0x{x} actual=0x{x}\n", .{ i, want, word });
            return error.TestUnexpectedResult;
        }
    }
    const weights = try a.alloc(f32, input * output);
    errdefer a.free(weights);
    const inner_weights = try a.alloc(f32, input * output);
    errdefer a.free(inner_weights);
    for (inner_weights, 0..) |*weight, i| {
        weight.* = publicHalf(inner, i);
        if (!std.math.isFinite(weight.*)) return error.InvalidSparkReference;
    }
    for (weights, 0..) |*weight, i| {
        weight.* = publicHalf(public, i);
        if (!std.math.isFinite(weight.*)) return error.InvalidSparkReference;
    }
    return .{ .k = k, .input = input, .output = output, .code = code, .rin = rin, .rout = rout, .weights = weights, .inner = inner_weights };
}

fn sparkBankProjection(g: *MlxG, projection: SparkProjection, offset: u64) !xq.ProjArrays(mlx.mlx_array) {
    const words = offset + projection.code.len / 2;
    try testing.expectEqual(@as(u64, 0), words % 256);
    const arena = try g.a.alloc(u8, @intCast(words * 2));
    defer g.a.free(arena);
    @memset(arena, 0);
    @memcpy(arena[@intCast(offset * 2)..], projection.code);
    const descriptor = [_]u64{ offset, projection.k };
    return .{
        .code = try g.hostArray(arena, &.{ @intCast(words / 256), 256 }, .int16),
        .rin = try g.hostArray(projection.rin, &.{ 1, @intCast(projection.input) }, .float16),
        .rout = try g.hostArray(projection.rout, &.{ 1, @intCast(projection.output) }, .float16),
        .layout = .{ .compact = try g.hostArray(std.mem.sliceAsBytes(&descriptor), &.{ 1, 2 }, .uint64) },
    };
}

fn sparkExpected(a: Allocator, projections: *const [3]SparkProjection) !struct { hidden: []f64, out: []f64 } {
    // publicInput repeats every 17 rows; compute each dense reference once, then reuse it across wave sizes.
    const hidden = try a.alloc(f64, 17 * 2304);
    errdefer a.free(hidden);
    const out = try a.alloc(f64, 17 * 5120);
    errdefer a.free(out);
    var gate: [2304]f64 = undefined;
    var up: [2304]f64 = undefined;
    for (0..17) |row| {
        @memset(&gate, 0);
        @memset(&up, 0);
        for (0..5120) |i| {
            const value: f64 = publicInput(row, i);
            for (0..2304) |j| {
                gate[j] += value * projections[0].weights[i * 2304 + j];
                up[j] += value * projections[1].weights[i * 2304 + j];
            }
        }
        const h = hidden[row * 2304 ..][0..2304];
        for (h, gate, up) |*value, gv, uv| {
            const clamped = @min(gv, 10);
            value.* = clamped / (1 + @exp(-clamped)) * std.math.clamp(uv, -10, 10);
        }
        const d = out[row * 5120 ..][0..5120];
        @memset(d, 0);
        for (h, 0..) |value, i| {
            for (d, projections[2].weights[i * 5120 ..][0..5120]) |*sum, weight| sum.* += value * weight;
        }
    }
    return .{ .hidden = hidden, .out = out };
}

fn sparkPrefillProjection(projection: SparkProjection, input: []const f64, out: []f64) !void {
    try testing.expectEqual(projection.input, input.len);
    try testing.expectEqual(projection.output, out.len);
    var scratch: [5120]f64 = undefined;
    const transformed = scratch[0..input.len];
    for (transformed, input, 0..) |*value, x, i| {
        const product = @as(f32, @floatCast(x)) * publicHalf(projection.rin, i);
        if (!std.math.isFinite(x) or !std.math.isFinite(product)) return error.InvalidSparkReference;
        value.* = product;
    }
    var block: usize = 0;
    while (block < transformed.len) : (block += 128) {
        prefillHalfTransform(transformed[block..][0..128]);
    }
    @memset(out, 0);
    for (transformed, 0..) |x, i| {
        if (!std.math.isFinite(x)) return error.InvalidSparkReference;
        for (out, projection.inner[i * out.len ..][0..out.len]) |*sum, weight| sum.* += x * weight;
    }
    for (out) |*value| {
        value.* = @as(f32, @floatCast(value.*));
        if (!std.math.isFinite(value.*)) return error.InvalidSparkReference;
    }
    referenceH128(out);
    for (out, 0..) |*value, i| {
        value.* = @as(f32, @floatCast(value.*)) * publicHalf(projection.rout, i);
        if (!std.math.isFinite(value.*)) return error.InvalidSparkReference;
    }
}

fn sparkPrefillRow(projections: *const [3]SparkProjection, input: []const f64, hidden: []f64, out: []f64) !void {
    var gate: [2304]f64 = undefined;
    var up: [2304]f64 = undefined;
    try sparkPrefillProjection(projections[0], input, gate[0..hidden.len]);
    try sparkPrefillProjection(projections[1], input, up[0..hidden.len]);
    for (hidden, gate[0..hidden.len], up[0..hidden.len]) |*value, gv, uv| {
        const g: f32 = @floatCast(@min(gv, 10));
        const u: f32 = @floatCast(std.math.clamp(uv, -10, 10));
        const y: f32 = 1 / (1 + @exp(@abs(g)));
        const sigmoid: f32 = if (g < 0) y else 1 - y;
        value.* = (g * sigmoid) * u;
        if (!std.math.isFinite(value.*)) return error.InvalidSparkReference;
    }
    try sparkPrefillProjection(projections[2], hidden, out);
}

fn sparkPrefillExpected(a: Allocator, projections: *const [3]SparkProjection) ![]f64 {
    const out = try a.alloc(f64, 17 * 5120);
    errdefer a.free(out);
    var input: [5120]f64 = undefined;
    var hidden: [2304]f64 = undefined;
    for (0..17) |row| {
        for (&input, 0..) |*value, i| value.* = @as(f32, @bitCast(@as(u32, sdk.io_util.bf16Rne(publicInput(row, i))) << 16));
        try sparkPrefillRow(projections, &input, &hidden, out[row * 5120 ..][0..5120]);
    }
    return out;
}

test "exl3 mixed oracle: full prefill projection preserves nonrepeated blocks and asymmetric scales" {
    const inner = try testing.allocator.alloc(f32, 256 * 256);
    defer testing.allocator.free(inner);
    @memset(inner, 0);
    for (0..128) |i| {
        inner[i * 256 + 128 + i] = 1;
        inner[(128 + i) * 256 + i] = 2;
    }
    var rin: [256]f16 = undefined;
    var rout: [256]f16 = undefined;
    var input: [256]f64 = undefined;
    for (&rin, &rout, &input, 0..) |*ri, *ro, *x, i| {
        ri.* = if (i < 128) 2 else 3;
        ro.* = (if (i < 128) @as(f16, 4) else 5) * (if (i % 2 == 0) @as(f16, 1) else -2);
        x.* = if (i < 128) 1.0 / 32.0 else 3.0 / 32.0;
    }
    const projection: SparkProjection = .{ .k = 2, .input = 256, .output = 256, .code = &.{}, .rin = std.mem.sliceAsBytes(&rin), .rout = std.mem.sliceAsBytes(&rout), .weights = &.{}, .inner = inner };
    var out: [256]f64 = undefined;
    try sparkPrefillProjection(projection, &input, &out);
    for (out, 0..) |value, i| {
        // The two independently rounded F16 DC values are 0.70703125 and 3.181640625.
        const dc: f64 = if (i < 128) 6.36328125 else 0.70703125;
        const expected = @as(f32, @floatCast(dc * 0.08838834764831845)) * @as(f32, rout[i]);
        try testing.expectEqual(@as(f64, expected), value);
    }
    for ([_]f64{ std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) }) |invalid| {
        const saved = input[0];
        input[0] = invalid;
        try testing.expectError(error.InvalidSparkReference, sparkPrefillProjection(projection, &input, &out));
        input[0] = saved;
    }
    rin[0] = std.math.inf(f16);
    try testing.expectError(error.InvalidSparkReference, sparkPrefillProjection(projection, &input, &out));
    rin[0] = 2;
    rout[0] = std.math.nan(f16);
    try testing.expectError(error.InvalidSparkReference, sparkPrefillProjection(projection, &input, &out));
    rout[0] = 4;
    inner[0] = std.math.inf(f32);
    try testing.expectError(error.InvalidSparkReference, sparkPrefillProjection(projection, &input, &out));
}

test "exl3 mixed oracle: full prefill stores every input block before accumulation" {
    const inner = try testing.allocator.alloc(f32, 5120 * 128);
    defer testing.allocator.free(inner);
    @memset(inner, 1);
    const rin: [5120]f16 = @splat(1);
    const rout: [128]f16 = @splat(1);
    var input: [5120]f64 = undefined;
    for (&input, 0..) |*value, i| value.* = if (i / 128 % 2 == 0) 0x1p-30 else 0x1p-29;
    const projection: SparkProjection = .{ .k = 2, .input = 5120, .output = 128, .code = &.{}, .rin = std.mem.sliceAsBytes(&rin), .rout = std.mem.sliceAsBytes(&rout), .weights = &.{}, .inner = inner };
    var out: [128]f64 = undefined;
    try sparkPrefillProjection(projection, &input, &out);
    for (out) |value| try testing.expectEqual(@as(f64, 0), value);
    var collapsed: [128]f64 = @splat(60 * 0x1p-30);
    prefillHalfTransform(&collapsed);
    try testing.expect(collapsed[0] > 0);
}

test "exl3 mixed oracle: full prefill keeps gate up distinct and stores the down input" {
    const inner = try testing.allocator.alloc(f32, 128 * 128);
    defer testing.allocator.free(inner);
    @memset(inner, 0);
    for (0..128) |i| inner[i * 128 + i] = 1;
    const ones: [128]f16 = @splat(1);
    const up_scale: [128]f16 = @splat(4);
    const down_scale: [128]f16 = @splat(0x1p-20);
    const base: SparkProjection = .{ .k = 2, .input = 128, .output = 128, .code = &.{}, .rin = std.mem.sliceAsBytes(&ones), .rout = std.mem.sliceAsBytes(&ones), .weights = &.{}, .inner = inner };
    var projections: [3]SparkProjection = @splat(base);
    projections[1].rout = std.mem.sliceAsBytes(&up_scale);
    projections[2].rin = std.mem.sliceAsBytes(&down_scale);
    const input: [128]f64 = @splat(1.0 / 32.0);
    var hidden: [128]f64 = undefined;
    var out: [128]f64 = undefined;
    try sparkPrefillRow(&projections, &input, &hidden, &out);
    try testing.expect(hidden[0] > 0);
    for (out) |value| try testing.expectEqual(@as(f64, 0), value);
    const forward = hidden[0];
    std.mem.swap(SparkProjection, &projections[0], &projections[1]);
    try sparkPrefillRow(&projections, &input, &hidden, &out);
    try testing.expect(hidden[0] != forward);
    projections[2].rin = std.mem.sliceAsBytes(&ones);
    try sparkPrefillRow(&projections, &input, &hidden, &out);
    try testing.expect(out[0] > 0);
    const actual: f32 = @floatCast(out[0]);
    const rms = try sparkReferenceRms(&out);
    const corrupted = sdk.io_util.bf16Rne(actual) ^ 1;
    try testing.expect(!try sparkPrefillValueMatches(actual, corrupted, out[0], rms / 64));
}

fn sparkReferenceRms(want: []const f64) !f64 {
    var square: f64 = 0;
    for (want) |value| {
        if (!std.math.isFinite(value)) return error.InvalidSparkReference;
        square += value * value;
    }
    const rms = @sqrt(square / @as(f64, @floatFromInt(want.len)));
    if (!std.math.isFinite(rms) or rms <= 1e-12) return error.InvalidSparkReference;
    return rms;
}

test "exl3 mixed oracle: nonfinite reference cannot make tolerance unbounded" {
    try testing.expectEqual(@as(f64, 1), try sparkReferenceRms(&.{ 1, -1 }));
    try testing.expectError(error.InvalidSparkReference, sparkReferenceRms(&.{ 1, std.math.inf(f64) }));
    try testing.expectError(error.InvalidSparkReference, sparkReferenceRms(&.{ 1, -std.math.inf(f64) }));
    try testing.expectError(error.InvalidSparkReference, sparkReferenceRms(&.{ 1, std.math.nan(f64) }));
}

fn sparkValueMatches(actual: f64, reference: f64, tolerance: f64) !bool {
    if (!std.math.isFinite(reference)) return error.InvalidSparkReference;
    return std.math.isFinite(actual) and @abs(actual - reference) <= tolerance;
}

fn sparkPrefillValueMatches(actual: f32, stored: u16, reference: f64, tolerance: f64) !bool {
    if (!try sparkValueMatches(actual, reference, tolerance)) return false;
    if (stored & 0x7f80 == 0x7f80) return false;
    return stored == sdk.io_util.bf16Rne(actual);
}

test "exl3 mixed oracle: paired prefill arithmetic and exact BF16 store straddle a midpoint" {
    const reference: f64 = -0.002143818186596036;
    const tolerance: f64 = 8.31e-6;
    const midpoint: f32 = -0.00214385986328125;
    const above = midpoint + 0x1p-30;
    const below = midpoint - 0x1p-30;
    try testing.expect(try sparkPrefillValueMatches(above, 0xbb0c, reference, tolerance));
    try testing.expect(try sparkPrefillValueMatches(below, 0xbb0d, reference, tolerance));
    try testing.expect(!try sparkPrefillValueMatches(above, 0xbb0d, reference, tolerance));
    try testing.expect(!try sparkPrefillValueMatches(below, 0xbb0c, reference, tolerance));
    const over_budget: f32 = @floatCast(reference + 2 * tolerance);
    try testing.expect(!try sparkPrefillValueMatches(over_budget, sdk.io_util.bf16Rne(over_budget), reference, tolerance));
    for ([_]f32{ std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |invalid| {
        try testing.expect(!try sparkPrefillValueMatches(invalid, 0xbb0c, reference, tolerance));
        try testing.expectError(error.InvalidSparkReference, sparkPrefillValueMatches(above, 0xbb0c, invalid, tolerance));
    }
    for ([_]u16{ 0x7f80, 0xff80, 0x7fc0 }) |invalid| {
        try testing.expect(!try sparkPrefillValueMatches(above, invalid, reference, tolerance));
    }
}

test "exl3 mixed oracle: paired BF16 store ties and binades use nearest even" {
    const cases = [_]struct { value: f32, word: u16 }{
        .{ .value = 0x1.01p0, .word = 0x3f80 },
        .{ .value = 0x1.03p0, .word = 0x3f82 },
        .{ .value = -0x1.01p0, .word = 0xbf80 },
        .{ .value = -0x1.03p0, .word = 0xbf82 },
        .{ .value = 0x1.ffp0, .word = 0x4000 },
        .{ .value = 0x1.fefp0, .word = 0x3fff },
        .{ .value = -0x1.ffp0, .word = 0xc000 },
        .{ .value = -0x1.fefp0, .word = 0xbfff },
    };
    for (cases) |case| {
        try testing.expect(try sparkPrefillValueMatches(case.value, case.word, case.value, 0));
        try testing.expect(!try sparkPrefillValueMatches(case.value, case.word ^ 1, case.value, 0));
    }
}

test "exl3 mixed oracle: F32 endpoint comparison remains unrounded" {
    const reference: f64 = 1 + 0x1p-30;
    try testing.expect(!try sparkValueMatches(1, reference, 0));
    try testing.expect(try sparkValueMatches(1, reference, 0x1p-29));
}

fn expectSparkRows(g: *MlxG, array: mlx.mlx_array, expected: []const f64, normalization: []const f64, cols: usize, stage: []const u8) !void {
    const shape = g.shapeOf(array);
    const rows: usize = @intCast(shape.d[0]);
    errdefer std.debug.print("Spark rows failed: stage={s} rows={d} cols={d}\n", .{ stage, rows, cols });
    const bytes = try g.hostBytes(array);
    defer g.a.free(bytes);
    const dtype = g.dtypeOf(array);
    try testing.expectEqual(rows * cols * size(dtype), bytes.len);
    for (0..rows) |row| {
        const want = expected[(row % 17) * cols ..][0..cols];
        const rms = try sparkReferenceRms(normalization[(row % 17) * cols ..][0..cols]);
        for (want, 0..) |value, col| {
            const got = getFloat(bytes, row * cols + col, dtype);
            if (!try sparkValueMatches(got, value, rms / 64)) {
                std.debug.print("Spark value mismatch: row={d} col={d} dtype={s} reference={d} actual={d} abs_error={d} rms={d} tolerance={d}\n", .{
                    row, col, @tagName(dtype), value, got, @abs(got - value), rms, rms / 64,
                });
                return error.TestUnexpectedResult;
            }
        }
    }
}

fn expectSparkPrefillRows(g: *MlxG, arithmetic: mlx.mlx_array, stored: mlx.mlx_array, expected: []const f64, normalization: []const f64, stage: []const u8) !void {
    const rows: usize = @intCast(g.shapeOf(arithmetic).d[0]);
    errdefer std.debug.print("Spark paired prefill failed: stage={s} rows={d}\n", .{ stage, rows });
    try testing.expectEqual(Dtype.float32, g.dtypeOf(arithmetic));
    try testing.expectEqual(Dtype.bfloat16, g.dtypeOf(stored));
    try testing.expectEqual(@as(usize, 5120), @as(usize, @intCast(g.shapeOf(arithmetic).d[1])));
    try testing.expectEqual(rows, @as(usize, @intCast(g.shapeOf(stored).d[0])));
    try testing.expectEqual(@as(usize, 5120), @as(usize, @intCast(g.shapeOf(stored).d[1])));
    const values = try g.hostBytes(arithmetic);
    defer g.a.free(values);
    const words = try g.hostBytes(stored);
    defer g.a.free(words);
    try testing.expectEqual(rows * 5120 * 4, values.len);
    try testing.expectEqual(rows * 5120 * 2, words.len);
    for (0..rows) |row| {
        const rms = try sparkReferenceRms(normalization[(row % 17) * 5120 ..][0..5120]);
        for (0..5120) |col| {
            const index = row * 5120 + col;
            const actual: f32 = @floatCast(getFloat(values, index, .float32));
            const word = std.mem.readInt(u16, words[index * 2 ..][0..2], .little);
            const reference = expected[(row % 17) * 5120 + col];
            if (!try sparkPrefillValueMatches(actual, word, reference, rms / 64)) {
                std.debug.print("Spark paired mismatch: row={d} col={d} reference={d} actual_f32={d} abs_error={d} tolerance={d} expected_bf16=0x{x} actual_bf16=0x{x}\n", .{
                    row, col, reference, actual, @abs(actual - reference), rms / 64, sdk.io_util.bf16Rne(actual), word,
                });
                return error.TestUnexpectedResult;
            }
        }
    }
}

fn expectSparkExpert(comptime k: u8, comptime expert_prefix: []const u8, comptime sample_sha256: []const u8, comptime reference_env: [:0]const u8) !void {
    errdefer std.debug.print("Spark expert failed: K{d} expert={s}\n", .{ k, expert_prefix });
    const fixture = std.c.getenv(reference_env) orelse return error.SparkReferenceRequired;
    const a = testing.allocator;
    const bytes = try sdk.io_util.readAllNoCache(a, std.mem.span(fixture), 256 << 20);
    defer a.free(bytes);
    try testing.expect(bytes.len >= 8);
    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    try testing.expect(n <= bytes.len - 8);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    defer parsed.deinit();
    const metadata = parsed.value.object.get("__metadata__") orelse return error.ReferenceProvenanceMissing;
    const comparison_policy = metadata.object.get("comparison_policy") orelse return error.ReferenceProvenanceMissing;
    try testing.expectEqualStrings("pony_numpy_f16_v1", comparison_policy.string);
    try testing.expectEqualStrings("8e7fa6b1556f59fc669e25087903b279b9b0346f", metadata.object.get("reference_revision").?.string);
    try testing.expectEqualStrings(sample_sha256, metadata.object.get("sample_sha256").?.string);
    try testing.expectEqualStrings("bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8", metadata.object.get("codebook_sha256").?.string);
    const data = bytes[8 + n ..];
    var oracle_arena = std.heap.ArenaAllocator.init(a);
    defer oracle_arena.deinit();
    const oa = oracle_arena.allocator();
    const projections = [3]SparkProjection{
        try sparkProjection(oa, parsed.value, data, expert_prefix, k, "w1", 5120, 2304),
        try sparkProjection(oa, parsed.value, data, expert_prefix, k, "w3", 5120, 2304),
        try sparkProjection(oa, parsed.value, data, expert_prefix, k, "w2", 2304, 5120),
    };
    const expected = try sparkExpected(oa, &projections);
    const prefill_expected = try sparkPrefillExpected(oa, &projections);
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    const bank: xq.BankArrays(mlx.mlx_array) = .{
        .gate = try sparkBankProjection(&g, projections[0], 256),
        .up = try sparkBankProjection(&g, projections[1], 512),
        .down = try sparkBankProjection(&g, projections[2], 768),
    };
    const acc = try compactAccepted(&g, &reg, &diag);
    defer acc.deinit(&g);
    acc.layer_rates[0].compact[0] = .{ k, k, k };
    try acc.routeBanked(&g);
    try acc.checkLayerBank(&g, 0, bank, &diag);
    const bf16_acc = try compactAccepted(&g, &reg, &diag);
    defer bf16_acc.deinit(&g);
    bf16_acc.layer_rates[0].compact[0] = .{ k, k, k };
    try bf16_acc.checkLayerBank(&g, 0, bank, &diag);
    try bf16_acc.routeExpertBf16();
    const zero_slots: [129]u32 = @splat(0);
    for ([_]usize{ 1, 8, 48 }) |rows| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const act = try publicActs(&g, rows);
        const ids = try g.hostArray(std.mem.sliceAsBytes(zero_slots[0..rows]), &.{@intCast(rows)}, .uint32);
        const hidden = try acc.gateUp(&g, act, ids, bank.gate, bank.up);
        try expectSparkRows(&g, hidden, expected.hidden, expected.hidden, 2304, "direct gate/up");
        const out = try acc.down(&g, hidden, ids, bank.down);
        try expectSparkRows(&g, out, expected.out, expected.out, 5120, "direct down");
        const banks = [_]xq.BankArrays(mlx.mlx_array){ bank, bank, bank };
        const bh = try acc.gateUpBanked(&g, act, ids, &banks);
        try expectSparkRows(&g, bh, expected.hidden, expected.hidden, 2304, "banked gate/up");
        const bo = try acc.downBanked(&g, bh, ids, &banks);
        try expectSparkRows(&g, bo, expected.out, expected.out, 5120, "banked down");
    }
    var arithmetic: [4]mlx.mlx_array = undefined;
    var outputs: [4]mlx.mlx_array = undefined;
    var built_arithmetic: usize = 0;
    var built_outputs: usize = 0;
    defer for (arithmetic[0..built_arithmetic]) |out| g.release(out);
    defer for (outputs[0..built_outputs]) |out| g.release(out);
    for ([_]usize{ 1, 127, 128, 129 }, 0..) |rows, i| {
        const act = try publicActs(&g, rows);
        arithmetic[i] = try acc.prefill(&g, 0, act, .{ .slot = zero_slots[0..rows] }, bank);
        built_arithmetic += 1;
        outputs[i] = try bf16_acc.prefill(&g, 0, act, .{ .slot = zero_slots[0..rows] }, bank);
        built_outputs += 1;
    }
    try acc.finishPrefill(&g);
    try bf16_acc.finishPrefill(&g);
    for (arithmetic, outputs) |f32_out, bf16_out| {
        try expectSparkPrefillRows(&g, f32_out, bf16_out, prefill_expected, expected.out, "prefill");
    }
    // The payload begins beyond 4 GiB; truncating its byte offset reads zeros.
    const high = try sparkBankProjection(&g, projections[2], (@as(u64, 1) << 31) + 256);
    const act = try publicActs(&g, 1);
    const ids = try g.hostArray(std.mem.sliceAsBytes(zero_slots[0..1]), &.{1}, .uint32);
    const hidden = try acc.gateUp(&g, act, ids, bank.gate, bank.up);
    const high_out = try acc.down(&g, hidden, ids, high);
    try expectSparkRows(&g, high_out, expected.out, expected.out, 5120, "high-offset direct down");
    var high_bank = bank;
    high_bank.down = high;
    const high_banks = [_]xq.BankArrays(mlx.mlx_array){ high_bank, high_bank, high_bank };
    const high_banked = try acc.downBanked(&g, hidden, ids, &high_banks);
    try expectSparkRows(&g, high_banked, expected.out, expected.out, 5120, "high-offset banked down");
    const prefill_act = try publicActs(&g, 129);
    const high_arithmetic = try acc.prefill(&g, 0, prefill_act, .{ .slot = &zero_slots }, high_bank);
    defer g.release(high_arithmetic);
    const high_prefill = try bf16_acc.prefill(&g, 0, prefill_act, .{ .slot = &zero_slots }, high_bank);
    defer g.release(high_prefill);
    try acc.finishPrefill(&g);
    try bf16_acc.finishPrefill(&g);
    try expectSparkPrefillRows(&g, high_arithmetic, high_prefill, prefill_expected, expected.out, "high-offset prefill");
}

test "exl3 mixed gpu: full nonrepeated Spark K2 K3 K5 oracles and arena offsets beyond 4 GiB" {
    const enabled = std.c.getenv("DSV41_KERNELS_GPU") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.span(enabled), "0")) return error.SkipZigTest;
    try expectSparkExpert(2, "layers.0.ffn.experts.0.", "108fe9acee54485f014d32fc770ae5673cb92582ed25d3a990d8bc29d30ea526", "DSV41_SPARK_K2_REFERENCE");
    try expectSparkExpert(3, "layers.10.ffn.experts.1.", "aba3c9f13766c9bf7bbae904cde140ae760a2f569943e03b7ecbc23bb6ab2438", "DSV41_SPARK_K3_REFERENCE");
    try expectSparkExpert(5, "layers.0.ffn.experts.2.", "d7db53973044421de435e62901243205ba0e1868fb0056bd5da19673db0af014", "DSV41_SPARK_K5_REFERENCE");
}
