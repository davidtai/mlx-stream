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
//! `overall PASS|FAIL`. Static bound (declared, printed first): EXL3 slot arrays 253 x 9,474,048 B = 2.40 GB; NVFP4
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
            const comp: ref.Component = @enumFromInt(c);
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
                const sg = ref.segment(@enumFromInt(c));
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
        .seam, .u8, .mcg => 4,
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

fn armsList() [24]Arm {
    var out: [24]Arm = undefined;
    var n: usize = 0;
    out[n] = .{ .name = "exl3_seam", .kind = .seam, .launches = 7 };
    n += 1;
    const named = [_]struct { []const u8, xf.Forms }{
        .{ "exl3_seam_forms", .{} },
        .{ "exl3_u8", .{ .gu = .u8 } },
        .{ "exl3_dsv41_forms", .{ .gu = .guone, .dn = .pair } },
        .{ "exl3_mcg", .{ .gu = .mcg, .dn = .mcg } },
        .{ "exl3_f16_seam", .{ .gu = .fused16 } },
        .{ "exl3_seam_f1", .{ .dn = .fused1 } },
        .{ "exl3_f16_f1_m1", .{ .gu = .fused16, .dn = .fused1 } },
        .{ "exl3_f16_f2_m1", .{ .gu = .fused16, .dn = .fused2 } },
        .{ "exl3_f8_f1_m1", .{ .gu = .fused8, .dn = .fused1 } },
        .{ "exl3_f8_f2_m1", .{ .gu = .fused8, .dn = .fused2 } },
        .{ "exl3_f16_f1_m4", .{ .gu = .fused16, .dn = .fused1, .maxm = 4 } },
        .{ "exl3_f16_f2_m4", .{ .gu = .fused16, .dn = .fused2, .maxm = 4 } },
        .{ "exl3_f8_f1_m4", .{ .gu = .fused8, .dn = .fused1, .maxm = 4 } },
        .{ "exl3_f8_f2_m4", .{ .gu = .fused8, .dn = .fused2, .maxm = 4 } },
        .{ "exl3_f16_p1", .{ .gu = .fused16, .dn = .pro1 } },
        .{ "exl3_f16_p2", .{ .gu = .fused16, .dn = .pro2 } },
        .{ "exl3_f8_p1", .{ .gu = .fused8, .dn = .pro1 } },
        .{ "exl3_f8_p2", .{ .gu = .fused8, .dn = .pro2 } },
        .{ "exl3_hyb_f16_f1_m1_p4", .{ .gu = .fused16, .dn = .fused1, .min_p = 4 } },
        .{ "exl3_hyb_f8_f1_m4_p4", .{ .gu = .fused8, .dn = .fused1, .maxm = 4, .min_p = 4 } },
    };
    for (named) |x| {
        out[n] = .{ .name = x[0], .kind = .forms, .forms = x[1], .launches = launchCount(x[1]) };
        n += 1;
    }
    out[n] = .{ .name = "nvfp4_fused", .kind = .nv_fused, .launches = 2 };
    n += 1;
    out[n] = .{ .name = "nvfp4_stock_gather", .kind = .nv_stock, .launches = 4 };
    n += 1;
    std.debug.assert(n == out.len);
    return out;
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

/// A hit wave: M tokens of top-8 over `res` slots, rows after the first share 30 % of the first row's slots.
fn hitWave(r: *Rng, m: u32, res: u32, out: []u32) u32 {
    var first: [8]u32 = undefined;
    var p: u32 = 0;
    for (0..m) |t| {
        var row: [8]u32 = undefined;
        var n: usize = 0;
        while (n < 8) {
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

test "glm5next exl3 kbench GPU: the K=3 decode forms at GLM's expert shape: parity against the seam and the exl3_ref truth, then us per layer (hit waves M 1..3 over 145 / 215 slots, miss parts over 10 / 30 records) against the NVFP4 control" {
    _ = std.c.getenv("GLM5NEXT_EXL3_KBENCH") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("GLM5NEXT_EXL3_KBENCH bound_bytes={d} layers={d} rounds={d} warm={d}\n", .{ bound_bytes, layers, rounds, warm });
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
    const PWave = struct { name: []const u8, p: u32 };
    const pwaves = [_]PWave{ .{ .name = "p1", .p = 1 }, .{ .name = "p3_misses_only", .p = 3 }, .{ .name = "p16_hits_only", .p = 16 }, .{ .name = "p24_m3_banks_repeats", .p = 24 }, .{ .name = "p64_widest", .p = 64 } };
    for ([_]ref.Codebook{ .mul1, .mcg }) |cb| {
        ref.table(cb, tbl);
        var truth: Truth = .{ .a = a, .tbl = tbl };
        defer truth.deinit();
        for (pwaves) |pw| {
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
            const mcg_arm = for (&arms) |*arm| {
                if (arm.kind == .forms and arm.forms.gu == .mcg) break arm;
            } else unreachable;
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
    for (cases) |cs| {
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
            const p = if (cs.hit) hitWave(&r, cs.m, cs.res, &w) else blk: {
                var n: u32 = 0;
                while (n < cs.p) {
                    const v = r.below(cs.res);
                    if (std.mem.indexOfScalar(u32, w[0..n], v) != null) continue;
                    w[n] = v;
                    n += 1;
                }
                break :blk n;
            };
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
