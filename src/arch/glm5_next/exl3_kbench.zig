//! The GLM-5.3-Flash EXL3 K=3 decode kernel bench (window only): GLM5NEXT_EXL3_KBENCH=1 (else skipped) and
//! _GPU_WINDOW_LOCKED=1 (else refused: the window orchestrator's child). One process, no model, no pack: synthetic
//! records in the pack's layout (`exl3_forms_ref.synthRecord`, R/mimo/glm5next-exl3-pack-contract.md) scattered into
//! slot banks (base 215 slots, ext 8, transient 30) and an NVFP4 control bank (base 145, ext 8, transient 30).
//!
//! 1. Parity (before any timing): waves P = 1, 3 (0 hits: transient only), 16 (0 misses: base only), 24 (M = 3, three
//!    banks, a slot named 3 and 5 times), 64 (8 tokens: the widest wave). Every arm's h (f32 [P, inter]) and down
//!    output (bf16 [P, hidden]; the fused down forms also f32) bit for bit against the seam (exl3_kernels.zig gateUp /
//!    down; the forms' seam instance at P > 48), and the seam against the f64 truth (exl3_forms_ref: exl3_ref.py's
//!    forward, pinned by the CPU test) per pair: h and the f32 down output max |err| / rms <= 1e-4 (DeepSeek-V4.1's
//!    tolerance), each bf16 output word within 2^-8 |ref| + 1e-4 rms (its rounding on top of that tolerance). The mcg arm against the mcg truth. An arm failing parity is not timed.
//! 2. Timing: per case, 42 chained layers (layer l + 1's x = layer l's output; each layer its own wave of fresh slots),
//!    one eval per round, rounds interleaved over the arms (start rotated), 2 warm rounds, `rounds` timed; per arm the
//!    median / p10 / p90 us per layer and the effective GB/s (the wave's distinct records' bytes over the time).
//!    Cases: hit waves M = 1, 2, 3 tokens of top-8 (rows share 30 % of the first row's slots: 8 / 13 / 18 distinct)
//!    over 145 or 215 resident slots; miss parts of 1 and 3 pairs over a transient bank of 10 or 30 records. The NVFP4
//!    arms take the same pattern modulo their bank (145 base).
//! Lines: GLM5NEXT_EXL3_KPARITY / KREF per (wave, arm), GLM5NEXT_EXL3_KBENCH per (case, arm), a `pick` line per case,
//! `overall PASS|FAIL`; GLM5NEXT_EXL3_KSTAGE before each stage (install, banks, parity_<codebook>_<wave>,
//! timing_<case>), and a watchdog thread that ends the process (status 124, `TIMEOUT stage=<name>`, `overall FAIL`) when a
//! stage outlives its budget. Static bound (declared, printed first): EXL3 slot arrays 253 x 9,474,048 B = 2.40 GB; NVFP4
//! 183 x 14,155,788 B = 2.59 GB + line-isolated scales 0.01 GB; a round's chain <= 42 x 32 x 82 KiB = 0.11 GB live,
//! MLX's cache cleared per case (limit 0.5 GB); host truth <= 8 decoded experts x 50 MB = 0.40 GB + the f64 work;
//! JIT and the test binary ~0.4 GB. Bound 6.5 GB.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const quant = @import("../../sdk_ext.zig").quant;
const xk = @import("exl3_kernels.zig");
const xf = @import("exl3_forms.zig");
const ref = @import("exl3_forms_ref.zig");
const nk = @import("decode_kernels.zig");
const nvfp4 = @import("nvfp4.zig");
const nbank = @import("bank.zig");

const testing = std.testing;
const T = mlx.mlx_array;
const G = ops.MlxOps;

pub const bound_bytes: u64 = 6_500_000_000;
const H = ref.hidden;
const I = ref.inter;
const layers: u32 = 42;
const rounds: u32 = 9;
const warm: u32 = 2;
const caps_x = [3]u32{ 215, 8, 30 };
const caps_n = [3]u32{ 145, 8, 30 };
const nv_record: u64 = 14_155_788;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ci(x: u32) c_int {
    return @intCast(x);
}

/// A zero array evaluated (MLX's Full: one contiguous buffer) and its bytes, written in place from the host.
fn hostZeros(g: *G, shape: []const c_int, dt: mlx.mlx_dtype) !struct { arr: T, bytes: []u8 } {
    const x = g.keep(try g.zeros(shape, dt));
    try g.evalAll(&.{x});
    return .{ .arr = x, .bytes = try g.hostBytes(x) };
}

// ── The EXL3 banks: records in the pack's layout scattered by the segment table ──

const ExlBanks = struct {
    banks: [3]xk.BankArrays,
    keep: [27]T,

    fn deinit(self: *ExlBanks, g: *G) void {
        for (self.keep) |x| g.release(x);
    }
};

fn recordSeed(bank: u32, slot: u32) u64 {
    return 0x6b62_0000_0000 + (@as(u64, bank) << 20) + @as(u64, slot) * 16;
}

fn buildExl3(a: std.mem.Allocator, g: *G) !ExlBanks {
    var out: ExlBanks = undefined;
    const rec = try a.alignedAlloc(u8, .@"16", ref.record_bytes);
    defer a.free(rec);
    var k: usize = 0;
    for (caps_x, 0..) |cap, b| {
        const gu = xk.projDims(.gate_up);
        const dn = xk.projDims(.down);
        // per component: the array shape (rows = cap) and its bytes
        var arrs: [9]struct { arr: T, bytes: []u8 } = undefined;
        for (0..9) |c| {
            const comp: ref.Component = @fromBackingInt(@intCast(c));
            const is_dn = c >= 6;
            const d = if (is_dn) dn else gu;
            arrs[c] = switch (c % 3) {
                0 => blk: {
                    const z = try hostZeros(g, &.{ ci(cap), ci(d.code[0]), ci(d.code[1]), ci(d.code[2]) }, .int16);
                    break :blk .{ .arr = z.arr, .bytes = z.bytes };
                },
                1 => blk: {
                    const z = try hostZeros(g, &.{ ci(cap), ci(d.rout) }, .float16);
                    break :blk .{ .arr = z.arr, .bytes = z.bytes };
                },
                else => blk: {
                    const z = try hostZeros(g, &.{ ci(cap), ci(d.rin) }, .float16);
                    break :blk .{ .arr = z.arr, .bytes = z.bytes };
                },
            };
            if (arrs[c].bytes.len != @as(usize, cap) * ref.segment(comp).len) return error.Glm5nKbenchBankShape;
            out.keep[k] = arrs[c].arr;
            k += 1;
        }
        for (0..cap) |s| {
            ref.synthRecord(recordSeed(@intCast(b), @intCast(s)), rec);
            for (0..9) |c| {
                const sg = ref.segment(@fromBackingInt(@intCast(c)));
                @memcpy(arrs[c].bytes[s * sg.len ..][0..sg.len], rec[sg.off..][0..sg.len]);
            }
        }
        out.banks[b] = .{
            .gate = .{ .code = arrs[0].arr, .rout = arrs[1].arr, .rin = arrs[2].arr },
            .up = .{ .code = arrs[3].arr, .rout = arrs[4].arr, .rin = arrs[5].arr },
            .down = .{ .code = arrs[6].arr, .rout = arrs[7].arr, .rin = arrs[8].arr },
        };
    }
    return out;
}

// ── The NVFP4 control banks (random codes, E4M3 scales 0x28 .. 0x57, line-isolated global scales) ──

const NvBanks = struct {
    banks: [3]nk.BankArrays,
    keep: [27]T,

    fn deinit(self: *NvBanks, g: *G) void {
        for (self.keep) |x| g.release(x);
    }
};

fn buildNv(g: *G) !NvBanks {
    var out: NvBanks = undefined;
    var k: usize = 0;
    var s: u64 = 0x6e76;
    const lines: u32 = @intCast(nbank.scale_array_min_bytes / nbank.scale_line_bytes);
    for (caps_n, 0..) |cap, b| {
        var p3: [3]nk.ProjArrays = undefined;
        for (0..3) |pi| {
            const rows: u32 = if (pi == 2) H else I;
            const cols: u32 = if (pi == 2) I else H;
            const w = try hostZeros(g, &.{ ci(cap), ci(rows), ci(cols / 8) }, .uint32);
            var i: usize = 0;
            while (i + 8 <= w.bytes.len) : (i += 8) std.mem.writeInt(u64, w.bytes[i..][0..8], @import("../../exl3/kernels.zig").splitmix64(&s), .little);
            const sc = try hostZeros(g, &.{ ci(cap), ci(rows), ci(cols / 16) }, .uint8);
            for (sc.bytes, 0..) |*v, j| v.* = @intCast(0x28 + (j *% 2654435761 >> 7) % 0x30);
            const nl = @max(cap, lines);
            const gs = try hostZeros(g, &.{ ci(nl), ci(nbank.scale_line_floats) }, .float32);
            const gf: []f32 = @alignCast(std.mem.bytesAsSlice(f32, gs.bytes));
            for (0..cap) |r| gf[r * nbank.scale_line_floats] = 0.002 + 0.0001 * @as(f32, @floatFromInt((r + b + pi) % 7));
            p3[pi] = .{ .w = w.arr, .scales = sc.arr, .global_scale = gs.arr };
            out.keep[k] = w.arr;
            out.keep[k + 1] = sc.arr;
            out.keep[k + 2] = gs.arr;
            k += 3;
        }
        out.banks[b] = .{ .gate = p3[0], .up = p3[1], .down = p3[2] };
    }
    return out;
}

// ── Arms ──

const ArmKind = enum { seam, forms, nv_fused, nv_stock };
const Arm = struct {
    name: []const u8,
    kind: ArmKind,
    forms: xf.Forms = .{},
    inst: ?*xf.Installed = null,
    inst32: ?*xf.Installed = null,
    launches: u32,
    ok: bool = true,
    times: [rounds]f64 = undefined,
};

fn launchCount(f0: xf.Forms) u32 {
    const f = f0.at(f0.min_p);
    const gu: u32 = switch (f.gu) {
        .seam, .mcg => 4,
        .guone => 3,
        .fused16, .fused8 => 2,
    };
    const dn: u32 = switch (f.dn) {
        .seam, .pair, .mcg => 3,
        .fused1, .fused2 => 2,
        .pro1, .pro2 => 1,
    };
    return gu + dn;
}

const NamedArm = struct { name: []const u8, forms: xf.Forms };
const named_arms = [_]NamedArm{
    .{ .name = "exl3_seam_forms", .forms = .{} },
    .{ .name = "exl3_dsv41_forms", .forms = .{ .gu = .guone, .dn = .pair } },
    .{ .name = "exl3_mcg", .forms = .{ .gu = .mcg, .dn = .mcg } },
    .{ .name = "exl3_f16_seam", .forms = .{ .gu = .fused16 } },
    .{ .name = "exl3_seam_f1", .forms = .{ .dn = .fused1 } },
    .{ .name = "exl3_f16_f1_m1", .forms = .{ .gu = .fused16, .dn = .fused1 } },
    .{ .name = "exl3_f16_f2_m1", .forms = .{ .gu = .fused16, .dn = .fused2 } },
    .{ .name = "exl3_f8_f1_m1", .forms = .{ .gu = .fused8, .dn = .fused1 } },
    .{ .name = "exl3_f8_f2_m1", .forms = .{ .gu = .fused8, .dn = .fused2 } },
    .{ .name = "exl3_f16_f1_m4", .forms = .{ .gu = .fused16, .dn = .fused1, .maxm = 4 } },
    .{ .name = "exl3_f16_f2_m4", .forms = .{ .gu = .fused16, .dn = .fused2, .maxm = 4 } },
    .{ .name = "exl3_f8_f1_m4", .forms = .{ .gu = .fused8, .dn = .fused1, .maxm = 4 } },
    .{ .name = "exl3_f8_f2_m4", .forms = .{ .gu = .fused8, .dn = .fused2, .maxm = 4 } },
    .{ .name = "exl3_f16_p1", .forms = .{ .gu = .fused16, .dn = .pro1 } },
    .{ .name = "exl3_f16_p2", .forms = .{ .gu = .fused16, .dn = .pro2 } },
    .{ .name = "exl3_f8_p1", .forms = .{ .gu = .fused8, .dn = .pro1 } },
    .{ .name = "exl3_f8_p2", .forms = .{ .gu = .fused8, .dn = .pro2 } },
    .{ .name = "exl3_hyb_f16_f1_m1_p4", .forms = .{ .gu = .fused16, .dn = .fused1, .min_p = 4 } },
    .{ .name = "exl3_hyb_f8_f1_m4_p4", .forms = .{ .gu = .fused8, .dn = .fused1, .maxm = 4, .min_p = 4 } },
};
/// The seam, the named form arms, the two NVFP4 controls. The table's length comes from the list: the 9105bbf binary
/// sized it by hand (24 for 23 arms), its `assert(n == out.len)` was UB in ReleaseFast and the test compiled to a
/// one-instruction self-loop right after G.init (the window hang of 2026-10-14T04:48Z).
pub const arm_count = 1 + named_arms.len + 2;
/// The arm table, built and checked at compile time: names distinct, every form installable (`Forms.check`), exactly
/// one mcg arm (the mcg codebook's reference chain).
const arms_table: [arm_count]Arm = blk: {
    var out: [arm_count]Arm = undefined;
    out[0] = .{ .name = "exl3_seam", .kind = .seam, .launches = 7 };
    for (named_arms, 1..) |x, i| {
        x.forms.check() catch @compileError("glm5next exl3 kbench: arm " ++ x.name ++ " names forms no build installs");
        out[i] = .{ .name = x.name, .kind = .forms, .forms = x.forms, .launches = launchCount(x.forms) };
    }
    out[arm_count - 2] = .{ .name = "nvfp4_fused", .kind = .nv_fused, .launches = 2 };
    out[arm_count - 1] = .{ .name = "nvfp4_stock_gather", .kind = .nv_stock, .launches = 4 };
    for (out, 0..) |p, i| for (out[0..i]) |q| if (std.mem.eql(u8, p.name, q.name)) @compileError("glm5next exl3 kbench: arm " ++ p.name ++ " named twice");
    break :blk out;
};
/// The mcg arm's index (the mcg codebook's reference chain), exactly one.
const mcg_arm_index: usize = blk: {
    var found: ?usize = null;
    for (arms_table, 0..) |arm, i| if (arm.kind == .forms and arm.forms.gu == .mcg) {
        if (found != null) @compileError("glm5next exl3 kbench: two mcg arms");
        found = i;
    };
    break :blk found orelse @compileError("glm5next exl3 kbench: no mcg arm");
};

fn armsList() [arm_count]Arm {
    return arms_table;
}

const Ctx = struct {
    a: std.mem.Allocator,
    g: *G,
    seam: *const xk.Kernels,
    nv: *const nk.Installed,
    xb: *const ExlBanks,
    nb: *const NvBanks,
};

/// One layer of an arm: x -> y (EXL3: bf16 [P, H]; NVFP4 fused bf16 [P, H]; NVFP4 stock [P, 1, H]).
fn layer(c: *const Ctx, arm: *const Arm, x: T, ids_x: T, ids_n: T, rows_n: T, nv_bank: u32) !T {
    const g = c.g;
    switch (arm.kind) {
        .seam => {
            const h = try xk.gateUp(c.seam, g, x, ids_x, &c.xb.banks);
            return xk.down(c.seam, g, h, ids_x, &c.xb.banks);
        },
        .forms => {
            const h = try xf.gateUp(arm.inst.?, g, x, ids_x, &c.xb.banks);
            return xf.down(arm.inst.?, g, h, ids_x, &c.xb.banks);
        },
        .nv_fused => {
            const h = try g.adopt(try c.nv.fused.gateUp(x, ids_n, &c.nb.banks));
            return g.adopt(try c.nv.fused.down(h, ids_n, &c.nb.banks));
        },
        .nv_stock => {
            const bk = c.nb.banks[nv_bank];
            const pg: nbank.Arrays(T) = .{ .w = bk.gate.w, .scales = bk.gate.scales, .global_scale = bk.gate.global_scale };
            const pu: nbank.Arrays(T) = .{ .w = bk.up.w, .scales = bk.up.scales, .global_scale = bk.up.global_scale };
            const pd: nbank.Arrays(T) = .{ .w = bk.down.w, .scales = bk.down.scales, .global_scale = bk.down.global_scale };
            const gt = try nvfp4.Format.gather(g, x, pg, rows_n);
            const ut = try nvfp4.Format.gather(g, x, pu, rows_n);
            const act = try quant.activation(G, g, .{ .swiglu_clamped = 10.0 }, gt, ut);
            return nvfp4.Format.gather(g, act, pd, rows_n);
        },
    }
}

// ── Waves ──

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        return @import("../../exl3/kernels.zig").splitmix64(&r.s);
    }
    fn below(r: *Rng, n: u32) u32 {
        return @intCast(r.next() % n);
    }
};

/// Draws per wave are bounded: a rejection loop that cannot fill (fewer distinct values than it needs) fails by name
/// instead of spinning. The cases' geometry is checked at compile time below, so the bound is never reached.
const max_draws: u32 = 1 << 16;

/// A hit wave: M tokens of top-8 over `res` slots, rows after the first share 30 % of the first row's slots.
fn hitWave(r: *Rng, m: u32, res: u32, out: []u32) error{Glm5nKbenchWaveDraw}!u32 {
    if (res < 8 or out.len < @as(usize, m) * 8) return error.Glm5nKbenchWaveDraw;
    var first: [8]u32 = undefined;
    var p: u32 = 0;
    for (0..m) |t| {
        var row: [8]u32 = undefined;
        var n: usize = 0;
        var draws: u32 = 0;
        while (n < 8) : (draws += 1) {
            if (draws >= max_draws) return error.Glm5nKbenchWaveDraw;
            const v = if (t > 0 and r.below(10) < 3) first[r.below(8)] else r.below(res);
            if (std.mem.indexOfScalar(u32, row[0..n], v) != null) continue;
            row[n] = v;
            n += 1;
        }
        if (t == 0) first = row;
        for (row) |v| {
            out[p] = v;
            p += 1;
        }
    }
    return p;
}

/// A miss part: `p` distinct records of a transient bank of `res`.
fn missPart(r: *Rng, p: u32, res: u32, out: []u32) error{Glm5nKbenchWaveDraw}!u32 {
    if (p > res or out.len < p) return error.Glm5nKbenchWaveDraw;
    var n: u32 = 0;
    var draws: u32 = 0;
    while (n < p) : (draws += 1) {
        if (draws >= max_draws) return error.Glm5nKbenchWaveDraw;
        const v = r.below(res);
        if (std.mem.indexOfScalar(u32, out[0..n], v) != null) continue;
        out[n] = v;
        n += 1;
    }
    return n;
}

fn distinct(ids: []const u32) u32 {
    var n: u32 = 0;
    for (ids, 0..) |v, i| n += @intFromBool(std.mem.indexOfScalar(u32, ids[0..i], v) == null);
    return n;
}

const Case = struct { name: []const u8, hit: bool, m: u32, res: u32, p: u32 };
const cases = [_]Case{
    .{ .name = "hit_m1_r145", .hit = true, .m = 1, .res = 145, .p = 8 },
    .{ .name = "hit_m2_r145", .hit = true, .m = 2, .res = 145, .p = 16 },
    .{ .name = "hit_m3_r145", .hit = true, .m = 3, .res = 145, .p = 24 },
    .{ .name = "hit_m1_r215", .hit = true, .m = 1, .res = 215, .p = 8 },
    .{ .name = "hit_m2_r215", .hit = true, .m = 2, .res = 215, .p = 16 },
    .{ .name = "hit_m3_r215", .hit = true, .m = 3, .res = 215, .p = 24 },
    .{ .name = "miss_p1_n10", .hit = false, .m = 1, .res = 10, .p = 1 },
    .{ .name = "miss_p3_n10", .hit = false, .m = 1, .res = 10, .p = 3 },
    .{ .name = "miss_p1_n30", .hit = false, .m = 1, .res = 30, .p = 1 },
    .{ .name = "miss_p3_n30", .hit = false, .m = 1, .res = 30, .p = 3 },
};

comptime {
    for (cases) |cs| {
        if (cs.hit and (cs.res < 8 or cs.m * 8 != cs.p or cs.p > 32 or cs.res > caps_x[0])) @compileError("glm5next exl3 kbench: hit case " ++ cs.name);
        if (!cs.hit and (cs.p > cs.res or cs.res > caps_x[2] or cs.res > caps_n[2])) @compileError("glm5next exl3 kbench: miss case " ++ cs.name);
    }
}

// ── Parity ──

fn hostF32(a: std.mem.Allocator, g: *G, x: T) ![]f32 {
    const n: usize = @intCast(g.shapeOf(x).numel());
    const out = try a.alloc(f32, n);
    errdefer a.free(out);
    _ = try g.hostF32(try g.astype(x, .float32), out);
    return out;
}

fn bitEq(p: []const f32, q: []const f32) usize {
    var n: usize = 0;
    for (p, q) |x, y| n += @intFromBool(@as(u32, @bitCast(x)) == @as(u32, @bitCast(y)));
    return n;
}

/// The host truth's experts, decoded on demand (at most `cap` at once: the bound's 8).
const Truth = struct {
    a: std.mem.Allocator,
    tbl: *[65536]u16,
    keys: [8]u64 = @splat(std.math.maxInt(u64)),
    recs: [8][]align(16) u8 = undefined,
    exps: [8]ref.Expert = undefined,
    n: usize = 0,
    next: usize = 0,

    fn get(t: *Truth, seed: u64) !*const ref.Expert {
        for (t.keys[0..t.n], 0..) |k, i| if (k == seed) return &t.exps[i];
        const i = if (t.n < 8) t.n else t.next;
        if (t.n < 8) {
            t.recs[i] = try t.a.alignedAlloc(u8, .@"16", ref.record_bytes);
            t.n += 1;
        } else {
            t.exps[i].deinit(t.a);
            t.next = (t.next + 1) % 8;
        }
        ref.synthRecord(seed, t.recs[i]);
        t.exps[i] = try ref.Expert.init(t.a, t.recs[i], t.tbl);
        t.keys[i] = seed;
        return &t.exps[i];
    }

    fn deinit(t: *Truth) void {
        for (0..t.n) |i| {
            t.exps[i].deinit(t.a);
            t.a.free(t.recs[i]);
        }
    }
};

// ── Stages and the watchdog ──

const PWave = struct { name: []const u8, p: u32 };
const pwaves = [_]PWave{ .{ .name = "p1", .p = 1 }, .{ .name = "p3_misses_only", .p = 3 }, .{ .name = "p16_hits_only", .p = 16 }, .{ .name = "p24_m3_banks_repeats", .p = 24 }, .{ .name = "p64_widest", .p = 64 } };
const codebooks = [_]ref.Codebook{ .mul1, .mcg };

/// A stage of the bench: a progress line names it before it starts, and the watchdog fails the process by its name
/// when it outlives its budget (a hang is attributable from the guard log alone).
pub const StageKind = enum(u8) { install, banks, parity, timing };
/// Budgets, seconds: the whole bench is about 2-4 min; each budget is several times its stage's expected time (the first
/// parity wave also JIT-compiles every arm's pipelines). A fully stalled run fails in at most 300 + 180 + 10 x 120 +
/// 10 x 120 = 2,880 s, one stage's budget after the stall began.
pub fn stageBudgetS(kind: StageKind) u64 {
    return switch (kind) {
        .install => 300,
        .banks => 180,
        .parity => 120,
        .timing => 120,
    };
}

/// The stage's name: install, banks, parity_<codebook>_<wave> (sub = codebook x waves + wave), timing_<case>.
pub fn stageName(kind: StageKind, sub: u32, buf: []u8) []const u8 {
    return switch (kind) {
        .install, .banks => @tagName(kind),
        .parity => if (sub < codebooks.len * pwaves.len)
            std.fmt.bufPrint(buf, "parity_{t}_{s}", .{ codebooks[sub / pwaves.len], pwaves[sub % pwaves.len].name }) catch "parity"
        else
            "parity_out_of_range",
        .timing => if (sub < cases.len) std.fmt.bufPrint(buf, "timing_{s}", .{cases[sub].name}) catch "timing" else "timing_out_of_range",
    };
}

const Watchdog = struct {
    /// kind << 32 | sub
    stage: std.atomic.Value(u64) = .init(0),
    /// ns (CLOCK_MONOTONIC); max while a stage is entered
    deadline: std.atomic.Value(u64) = .init(std.math.maxInt(u64)),
    quit: std.atomic.Value(bool) = .init(false),
    t0: u64,
    /// the poll period, ns (the tests shorten it)
    poll_ns: u64 = 250 * std.time.ns_per_ms,
    /// what a timeout does: the bench exits the process (status 124); the CPU test records it
    on_timeout: *const fn (w: *Watchdog, name: []const u8) void = exitOnTimeout,
    fired: std.atomic.Value(bool) = .init(false),

    fn enter(w: *Watchdog, kind: StageKind, sub: u32) void {
        w.enterBudget(kind, sub, stageBudgetS(kind) * std.time.ns_per_s);
    }

    fn enterBudget(w: *Watchdog, kind: StageKind, sub: u32, budget_ns: u64) void {
        w.deadline.store(std.math.maxInt(u64), .seq_cst);
        w.stage.store(@as(u64, @backingInt(kind)) << 32 | sub, .seq_cst);
        var buf: [64]u8 = undefined;
        std.debug.print("GLM5NEXT_EXL3_KSTAGE stage={s} budget_s={d} t_ms={d}\n", .{ stageName(kind, sub, &buf), budget_ns / std.time.ns_per_s, (nowNs() - w.t0) / std.time.ns_per_ms });
        w.deadline.store(nowNs() + budget_ns, .seq_cst);
    }

    fn run(w: *Watchdog) void {
        while (!w.quit.load(.seq_cst)) {
            const req: std.c.timespec = .{ .sec = @intCast(w.poll_ns / std.time.ns_per_s), .nsec = @intCast(w.poll_ns % std.time.ns_per_s) };
            _ = std.c.nanosleep(&req, null);
            if (nowNs() <= w.deadline.load(.seq_cst)) continue;
            const st = w.stage.load(.seq_cst);
            var buf: [64]u8 = undefined;
            const name = stageName(@fromBackingInt(@intCast(@as(u8, @truncate(st >> 32)))), @truncate(st), &buf);
            w.fired.store(true, .seq_cst);
            w.on_timeout(w, name);
            return;
        }
    }

    fn exitOnTimeout(w: *Watchdog, name: []const u8) void {
        std.debug.print("GLM5NEXT_EXL3_KBENCH TIMEOUT stage={s} t_ms={d}\nGLM5NEXT_EXL3_KBENCH overall FAIL timeout stage={s}\n", .{ name, (nowNs() - w.t0) / std.time.ns_per_ms, name });
        std.c._exit(124);
    }

    fn stop(w: *Watchdog, th: std.Thread) void {
        w.quit.store(true, .seq_cst);
        th.join();
    }
};

test "glm5next exl3 kbench GPU: the K=3 decode forms at GLM's expert shape: parity against the seam and the exl3_ref truth, then us per layer (hit waves M 1..3 over 145 / 215 slots, miss parts over 10 / 30 records) against the NVFP4 control" {
    _ = std.c.getenv("GLM5NEXT_EXL3_KBENCH") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5NEXT_EXL3_KBENCH bound_bytes={d} layers={d} rounds={d} warm={d} arms={d}\n", .{ bound_bytes, layers, rounds, warm, arm_count });
    var wd: Watchdog = .{ .t0 = nowNs() };
    const wd_thread = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, Watchdog.run, .{&wd});
    defer wd.stop(wd_thread);
    wd.enter(.install, 0);
    const a = std.heap.c_allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    if (!mlx.streamIsGpu(stream)) return error.SkipZigTest;
    var old_limit: usize = 0;
    _ = mlx.mlx_set_cache_limit(&old_limit, 512 << 20);
    var diag: sdk.Diag = .{};
    const seam = xk.install(a, stream, &diag) catch |e| {
        std.debug.print("GLM5NEXT_EXL3_KBENCH seam install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer {
        seam.deinit();
        a.destroy(seam);
    }
    try nk.checkModel(H, I, 8, 10.0, &diag);
    var nv = nk.install(a, stream, &diag) catch |e| {
        std.debug.print("GLM5NEXT_EXL3_KBENCH nvfp4 install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer nv.deinit();
    var g = try G.init(a, stream);
    defer g.deinit();

    var arms = armsList();
    for (&arms) |*arm| if (arm.kind == .forms) {
        arm.inst = xf.install(a, seam, arm.forms, .bfloat16, &diag) catch |e| {
            std.debug.print("GLM5NEXT_EXL3_KBENCH arm={s} install refused: {s}\n", .{ arm.name, diag.message() });
            return e;
        };
        if (arm.forms.fusedDn() or arm.forms.gu == .mcg) arm.inst32 = try xf.install(a, seam, arm.forms, .float32, &diag);
    };
    defer for (&arms) |*arm| {
        if (arm.inst) |p| {
            p.deinit();
            a.destroy(p);
        }
        if (arm.inst32) |p| {
            p.deinit();
            a.destroy(p);
        }
    };
    // the seam chain with an f32 down output (the reference check's words; at P > 48 the seam itself)
    const seam32 = try xf.install(a, seam, .{}, .float32, &diag);
    defer {
        seam32.deinit();
        a.destroy(seam32);
    }
    const seam16 = try xf.install(a, seam, .{}, .bfloat16, &diag);
    defer {
        seam16.deinit();
        a.destroy(seam16);
    }

    wd.enter(.banks, 0);
    const t_build = nowNs();
    var xb = try buildExl3(a, &g);
    defer xb.deinit(&g);
    var nb = try buildNv(&g);
    defer nb.deinit(&g);
    g.reset();
    std.debug.print("GLM5NEXT_EXL3_KBENCH banks built exl3_slots={d} nvfp4_slots={d} ms={d}\n", .{ caps_x[0] + caps_x[1] + caps_x[2], caps_n[0] + caps_n[1] + caps_n[2], (nowNs() - t_build) / 1_000_000 });
    const ctx: Ctx = .{ .a = a, .g = &g, .seam = seam, .nv = &nv, .xb = &xb, .nb = &nb };

    // ── 1. Parity ──
    const tbl = try a.create([65536]u16);
    defer a.destroy(tbl);
    var fails: u32 = 0;
    var rng: Rng = .{ .s = 0x70617269 };
    for (codebooks, 0..) |cb, cbi| {
        ref.table(cb, tbl);
        var truth: Truth = .{ .a = a, .tbl = tbl };
        defer truth.deinit();
        for (pwaves, 0..) |pw, pwi| {
            wd.enter(.parity, @intCast(cbi * pwaves.len + pwi));
            var ids: [64]u32 = undefined;
            for (ids[0..pw.p], 0..) |*v, i| {
                const bank: u32, const slot: u32 = if (std.mem.eql(u8, pw.name, "p1")) .{ 0, 5 } else if (std.mem.startsWith(u8, pw.name, "p3")) .{ 2, rng.below(30) } else if (std.mem.startsWith(u8, pw.name, "p16")) .{ 0, rng.below(215) } else if (std.mem.startsWith(u8, pw.name, "p24"))
                    (if (i % 8 == 0) .{ 0, 7 } else if (i < 6) .{ 1, 3 } else .{ @as(u32, @intCast(i % 3)), rng.below(8) })
                else
                    .{ @as(u32, @intCast(i % 3)), rng.below(8) };
                v.* = (bank << 24) | slot;
            }
            const xin = try a.alloc(f32, pw.p * H);
            defer a.free(xin);
            for (0..pw.p) |r| ref.fillX(0x7800 + r, xin[r * H ..][0..H]);
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xin), &.{ ci(pw.p), ci(H) }, .float32), .bfloat16);
            const ida = try g.hostArray(std.mem.sliceAsBytes(ids[0..pw.p]), &.{ci(pw.p)}, .uint32);
            // the reference chain of this codebook: the seam (mul1) or the mcg arm's
            const mcg_arm = &arms[mcg_arm_index];
            const r32 = if (cb == .mul1) seam32 else mcg_arm.inst32.?;
            const r16 = if (cb == .mul1) seam16 else mcg_arm.inst.?;
            const h_ref = try xf.gateUp(r32, &g, x, ida, &xb.banks);
            const y32 = try xf.down(r32, &g, h_ref, ida, &xb.banks);
            const y16 = try xf.down(r16, &g, h_ref, ida, &xb.banks);
            try g.evalAll(&.{ h_ref, y32, y16 });
            const hh = try hostF32(a, &g, h_ref);
            defer a.free(hh);
            const yy32 = try hostF32(a, &g, y32);
            defer a.free(yy32);
            const yy16 = try hostF32(a, &g, y16);
            defer a.free(yy16);
            // the truth per pair
            var w_h: f64 = 0;
            var w_y32: f64 = 0;
            var w_y16: f64 = 0;
            const th = try a.alloc(f64, I);
            defer a.free(th);
            const ty = try a.alloc(f64, H);
            defer a.free(ty);
            const gv = try a.alloc(f64, H);
            defer a.free(gv);
            for (0..pw.p) |p| {
                const e = try truth.get(recordSeed(ids[p] >> 24, ids[p] & 0xFFFFFF));
                var xr: [H]f64 = undefined;
                for (&xr, xin[p * H ..][0..H]) |*o, v| o.* = v;
                try e.run(a, &xr, th, ty);
                for (gv[0..I], hh[p * I ..][0..I]) |*o, v| o.* = v;
                w_h = @max(w_h, ref.maxErrOverRms(gv[0..I], th));
                for (gv, yy32[p * H ..][0..H]) |*o, v| o.* = v;
                w_y32 = @max(w_y32, ref.maxErrOverRms(gv, ty));
                for (gv, yy16[p * H ..][0..H]) |*o, v| o.* = v;
                w_y16 = @max(w_y16, ref.bf16ErrRatio(gv, ty));
            }
            const ref_ok = w_h <= ref.tol_f32 and w_y32 <= ref.tol_f32 and w_y16 <= 1.0;
            if (!ref_ok) fails += 1;
            std.debug.print("GLM5NEXT_EXL3_KREF wave={s} codebook={t} pairs={d} h_err={e:.3} y32_err={e:.3} ybf16_ratio={d:.3} {s}\n", .{ pw.name, cb, pw.p, w_h, w_y32, w_y16, if (ref_ok) "PASS" else "FAIL" });
            // every arm of this codebook against the reference chain, bit for bit
            for (&arms) |*arm| {
                const is_mcg = arm.kind == .forms and (arm.forms.gu == .mcg);
                if (arm.kind == .nv_fused or arm.kind == .nv_stock) continue;
                if (is_mcg != (cb == .mcg)) continue;
                if (arm.kind == .seam and pw.p > xk.max_pairs) continue;
                const mark = g.mark();
                const h = if (arm.kind == .seam) try xk.gateUp(seam, &g, x, ida, &xb.banks) else try xf.gateUp(arm.inst.?, &g, x, ida, &xb.banks);
                const y = if (arm.kind == .seam) try xk.down(seam, &g, h, ida, &xb.banks) else try xf.down(arm.inst.?, &g, h, ida, &xb.banks);
                const y3 = if (arm.inst32) |p32| try xf.down(p32, &g, h, ida, &xb.banks) else y;
                try g.evalAll(&.{ h, y, y3 });
                const ah = try hostF32(a, &g, h);
                defer a.free(ah);
                const ay = try hostF32(a, &g, y);
                defer a.free(ay);
                const ay3 = try hostF32(a, &g, y3);
                defer a.free(ay3);
                const eh = bitEq(ah, hh);
                const ey = bitEq(ay, yy16);
                const ey3 = if (arm.inst32 != null) bitEq(ay3, yy32) else ay3.len;
                const ok = eh == hh.len and ey == yy16.len and ey3 == yy32.len;
                if (!ok) {
                    arm.ok = false;
                    fails += 1;
                }
                std.debug.print("GLM5NEXT_EXL3_KPARITY wave={s} arm={s} h_bitwise={d}/{d} y_bitwise={d}/{d} y32_bitwise={d}/{d} {s}\n", .{ pw.name, arm.name, eh, hh.len, ey, yy16.len, ey3, yy32.len, if (ok) "PASS" else "FAIL" });
                g.resetTo(mark);
            }
            g.reset();
        }
    }

    // ── 2. Timing ──
    var xinit: [32 * H]f32 = undefined;
    for (0..32) |r| ref.fillX(0x7900 + r, xinit[r * H ..][0..H]);
    for (cases, 0..) |cs, csi| {
        wd.enter(.timing, @intCast(csi));
        _ = mlx.mlx_clear_cache();
        // the case's waves: one per layer (EXL3 packed ids, NVFP4 packed ids modulo its bank, NVFP4 stock rows)
        var ids_x: [layers]T = undefined;
        var ids_n: [layers]T = undefined;
        var rows_n: [layers]T = undefined;
        var bytes_x: f64 = 0;
        var bytes_n: f64 = 0;
        var r: Rng = .{ .s = 0x77617665 ^ (@as(u64, cs.m) << 8) ^ cs.res ^ (@as(u64, cs.p) << 16) };
        const bank: u32 = if (cs.hit) 0 else 2;
        for (0..layers) |l| {
            var w: [32]u32 = undefined;
            const p = if (cs.hit) try hitWave(&r, cs.m, cs.res, &w) else try missPart(&r, cs.p, cs.res, &w);
            var px: [32]u32 = undefined;
            var pn: [32]u32 = undefined;
            var rn: [32]u32 = undefined;
            for (0..p) |i| {
                px[i] = (bank << 24) | w[i];
                rn[i] = w[i] % caps_n[bank];
                pn[i] = (bank << 24) | rn[i];
            }
            ids_x[l] = g.keep(try g.hostArray(std.mem.sliceAsBytes(px[0..p]), &.{ci(p)}, .uint32));
            ids_n[l] = g.keep(try g.hostArray(std.mem.sliceAsBytes(pn[0..p]), &.{ci(p)}, .uint32));
            rows_n[l] = g.keep(try g.hostArray(std.mem.sliceAsBytes(rn[0..p]), &.{ci(p)}, .uint32));
            bytes_x += @as(f64, @floatFromInt(distinct(px[0..p]))) * ref.record_bytes;
            bytes_n += @as(f64, @floatFromInt(distinct(pn[0..p]))) * @as(f64, @floatFromInt(nv_record));
        }
        defer for (0..layers) |l| {
            g.release(ids_x[l]);
            g.release(ids_n[l]);
            g.release(rows_n[l]);
        };
        try g.evalAll(&ids_x);
        try g.evalAll(&ids_n);
        try g.evalAll(&rows_n);
        const P: u32 = @intCast(g.shapeOf(ids_x[0]).d[0]);
        const x0 = g.keep(try g.astype(try g.hostArray(std.mem.sliceAsBytes(xinit[0 .. P * H]), &.{ ci(P), ci(H) }, .float32), .bfloat16));
        defer g.release(x0);
        const x0s = g.keep(try g.reshape(x0, &.{ ci(P), 1, ci(H) }));
        defer g.release(x0s);
        try g.evalAll(&.{ x0, x0s });
        for (0..warm + rounds) |ri| {
            for (0..arms.len) |ai0| {
                const arm = &arms[(ai0 + ri) % arms.len];
                if (!arm.ok) continue;
                var x = if (arm.kind == .nv_stock) x0s else x0;
                const mark = g.mark();
                for (0..layers) |l| x = try layer(&ctx, arm, x, ids_x[l], ids_n[l], rows_n[l], bank);
                const t0 = nowNs();
                try g.evalAll(&.{x});
                const dt = @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0 / @as(f64, @floatFromInt(layers));
                if (ri >= warm) arm.times[ri - warm] = dt;
                g.resetTo(mark);
            }
        }
        var best: ?*const Arm = null;
        var best_t: f64 = 0;
        var seam_t: f64 = 0;
        var nv_t: f64 = 0;
        for (&arms) |*arm| {
            if (!arm.ok) {
                std.debug.print("GLM5NEXT_EXL3_KBENCH case={s} arm={s} SKIPPED (parity failed)\n", .{ cs.name, arm.name });
                continue;
            }
            var t = arm.times;
            std.mem.sort(f64, &t, {}, std.sort.asc(f64));
            const med = t[rounds / 2];
            const is_nv = arm.kind == .nv_fused or arm.kind == .nv_stock;
            const bytes = (if (is_nv) bytes_n else bytes_x) / layers;
            std.debug.print("GLM5NEXT_EXL3_KBENCH case={s} pairs={d} arm={s} launches={d} us_per_layer={d:.1} p10={d:.1} p90={d:.1} gbps={d:.1}\n", .{ cs.name, P, arm.name, arm.launches, med, t[1], t[rounds - 2], bytes / (med * 1000.0) });
            if (arm.kind == .seam) seam_t = med;
            if (arm.kind == .nv_fused) nv_t = med;
            if (!is_nv and !(arm.kind == .forms and arm.forms.gu == .mcg) and (best == null or med < best_t)) {
                best = arm;
                best_t = med;
            }
        }
        if (best) |b| std.debug.print("GLM5NEXT_EXL3_KBENCH pick case={s} best={s} us_per_layer={d:.1} vs_seam_pct={d:.2} vs_nvfp4_fused_pct={d:.2}\n", .{ cs.name, b.name, best_t, 100.0 * (best_t - seam_t) / seam_t, 100.0 * (best_t - nv_t) / nv_t });
        g.reset();
    }
    std.debug.print("GLM5NEXT_EXL3_KBENCH overall {s} parity_fails={d}\n", .{ if (fails == 0) "PASS" else "FAIL", fails });
    try testing.expectEqual(@as(u32, 0), fails);
}

test "glm5_next exl3 kbench: the arm table is the list's length (22: seam, 19 forms, 2 NVFP4), every entry named and installable, one mcg arm" {
    const arms = armsList();
    try testing.expectEqual(@as(usize, 22), arms.len);
    try testing.expectEqual(named_arms.len + 3, arms.len);
    try testing.expectEqual(ArmKind.seam, arms[0].kind);
    try testing.expectEqual(ArmKind.nv_fused, arms[arms.len - 2].kind);
    try testing.expectEqual(ArmKind.nv_stock, arms[arms.len - 1].kind);
    for (arms, 0..) |arm, i| {
        try testing.expect(arm.name.len > 0);
        try testing.expect(arm.launches >= 1 and arm.launches <= 7);
        try testing.expect(arm.ok and arm.inst == null and arm.inst32 == null);
        if (arm.kind == .forms) {
            try arm.forms.check();
            try testing.expectEqualStrings(named_arms[i - 1].name, arm.name);
        }
        for (arms[0..i]) |q| try testing.expect(!std.mem.eql(u8, arm.name, q.name));
    }
    try testing.expectEqualStrings("exl3_mcg", arms[mcg_arm_index].name);
}

test "glm5_next exl3 kbench: every case's waves draw within the bound (distinct per row / part), an impossible draw fails by name" {
    for (cases) |cs| {
        var r: Rng = .{ .s = 0x77617665 ^ (@as(u64, cs.m) << 8) ^ cs.res ^ (@as(u64, cs.p) << 16) };
        for (0..layers) |_| {
            var w: [32]u32 = undefined;
            const p = if (cs.hit) try hitWave(&r, cs.m, cs.res, &w) else try missPart(&r, cs.p, cs.res, &w);
            try testing.expectEqual(cs.p, p);
            var row: u32 = 0;
            while (row * 8 < p) : (row += 1) {
                const part = w[row * 8 .. @min(p, row * 8 + 8)];
                try testing.expectEqual(@as(u32, @intCast(part.len)), distinct(part));
                for (part) |v| try testing.expect(v < cs.res);
            }
        }
    }
    var r: Rng = .{ .s = 1 };
    var w: [32]u32 = undefined;
    try testing.expectError(error.Glm5nKbenchWaveDraw, hitWave(&r, 1, 7, &w));
    try testing.expectError(error.Glm5nKbenchWaveDraw, hitWave(&r, 5, 145, &w));
    try testing.expectError(error.Glm5nKbenchWaveDraw, missPart(&r, 11, 10, &w));
}

test "glm5_next exl3 kbench: every stage has a name; the watchdog fires on an overrun stage by its name and stays quiet on one that ends in budget" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("install", stageName(.install, 0, &buf));
    try testing.expectEqualStrings("banks", stageName(.banks, 0, &buf));
    try testing.expectEqualStrings("parity_mul1_p1", stageName(.parity, 0, &buf));
    try testing.expectEqualStrings("parity_mcg_p64_widest", stageName(.parity, codebooks.len * pwaves.len - 1, &buf));
    try testing.expectEqualStrings("timing_miss_p3_n30", stageName(.timing, cases.len - 1, &buf));
    try testing.expectEqualStrings("timing_out_of_range", stageName(.timing, cases.len, &buf));
    var total: u64 = 0;
    for (std.enums.values(StageKind)) |k| total += stageBudgetS(k) * @as(u64, switch (k) {
        .install, .banks => 1,
        .parity => codebooks.len * pwaves.len,
        .timing => cases.len,
    });
    // the budgets' sum: 2,880 s (under an hour)
    try testing.expectEqual(@as(u64, 2880), total);

    const Rec = struct {
        var name: [64]u8 = undefined;
        var len: usize = 0;
        fn record(_: *Watchdog, n: []const u8) void {
            @memcpy(name[0..n.len], n);
            len = n.len;
        }
    };
    // in budget: no fire
    {
        var wd: Watchdog = .{ .t0 = nowNs(), .poll_ns = 2 * std.time.ns_per_ms, .on_timeout = Rec.record };
        const th = try std.Thread.spawn(.{}, Watchdog.run, .{&wd});
        wd.enterBudget(.banks, 0, 10 * std.time.ns_per_s);
        const req: std.c.timespec = .{ .sec = 0, .nsec = 30 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&req, null);
        wd.stop(th);
        try testing.expect(!wd.fired.load(.seq_cst));
    }
    // overrun: fires with the stage's name, bounded
    {
        var wd: Watchdog = .{ .t0 = nowNs(), .poll_ns = 2 * std.time.ns_per_ms, .on_timeout = Rec.record };
        const th = try std.Thread.spawn(.{}, Watchdog.run, .{&wd});
        wd.enterBudget(.timing, 3, 20 * std.time.ns_per_ms);
        var waited: u32 = 0;
        while (!wd.fired.load(.seq_cst) and waited < 500) : (waited += 1) {
            const req: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&req, null);
        }
        wd.stop(th);
        try testing.expect(wd.fired.load(.seq_cst));
        try testing.expectEqualStrings("timing_hit_m1_r215", Rec.name[0..Rec.len]);
    }
}
