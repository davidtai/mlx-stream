//! Tests across the kernel set's two consumers (the EXL3 quant and the V4.1 trunk routes) and
//! the C2 interface: the partition, each consumer's accept on its subset, every route at the
//! lanes' own launches, the prepared per-M tables, the profiled backend, and a second quant on the
//! same interface. Test code only.

const std = @import("std");
const mlx = @import("sdk").mlx;
const xk = @import("exl3_kernels.zig");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const selfcheck = @import("exl3_selfcheck.zig");
const kr = sdk_ext.kernels.Routes(xk);
const ks = sdk_ext.kernels.KernelSet(xk);
const kt = sdk_ext.kernels.Trace(xk);
const quant = @import("sdk_ext.zig").quant;
const eq = @import("exl3_quant.zig");
const tr = @import("dsv41_kernel_routes.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Entry = xk.Entry;
const Vars = xk.Vars;
const LaunchConfig = xk.LaunchConfig;
const Dtype = mlx.mlx_dtype;
const no_vars = kr.no_vars;
const rowsVars = kr.rowsVars;
const Statics = kr.Statics;
const Trace = kt.Trace;
const testRegistry = kt.testRegistry;
const expectLaunch = kt.expectLaunch;
const expectPreparedDecode = kt.expectPreparedDecode;
const sampleAt = kt.sampleAt;
const templateInt = kt.templateInt;
// the trunk's routes
const Router = tr.Router;
const Premix = tr.Premix;
const Sinkhorn = tr.Sinkhorn;
const RcProj = tr.RcProj;
const RcSite = tr.RcSite;
const HcTape = tr.HcTape;
const FusedProj = tr.FusedProj;
const RopeDir = tr.RopeDir;
// the EXL3 quant's routes
const Proj = eq.Proj;
const ProjArrays = eq.ProjArrays;
const checkBank = eq.checkBank;
const Gemv = eq.Gemv;
const RinPrep = eq.RinPrep;
const Rebuild = eq.Rebuild;
const DigX = eq.DigX;

/// DeepSeek-V4.1's routed experts, as the arch asks the quant.
const v41_spec: quant.Spec = .{ .hidden = 5120, .inter = 2304, .top_k = 6, .n_layers = 40, .act = .{ .swiglu_clamped = 10.0 }, .input = .bfloat16 };

// ── 1. The partition ──

comptime {
    ks.checkPartition(&.{ &eq.kernels, &tr.kernels });
    quant.checkAccepted(eq, Trace);
    quant.checkAccepted(quant.FromGatherMatmul(quant.GatherQmm), Trace);
    quant.checkAccepted(quant.FromGatherMatmul(FakeInt8), Trace);
}

test "dsv41 kernels c2: the EXL3 quant and the trunk partition the kernel set (kernels and headers)" {
    var reg = try testRegistry();
    defer reg.deinit();
    var buf: [256]u8 = undefined;
    if (ks.partitionError(&reg, &.{ &eq.kernels, &tr.kernels }, &buf)) |m| {
        std.debug.print("partition: {s}\n", .{m});
        return error.TestUnexpectedResult;
    }
    try testing.expectEqual(xk.n_kernels, eq.kernels.len + tr.kernels.len);
    // the EXL3 subset is exactly the EXL3 families; its headers are the DIG ones
    const exl3_families = [_][]const u8{ "exl3_decode_gemv", "exl3_rin_stage", "prefill_rebuild", "prefill_digx", "prefill_digx_check" };
    const ex = ks.subsetOf(&eq.kernels);
    for (&reg.entries) |*e| {
        var fam = false;
        for (exl3_families) |f| fam = fam or std.mem.eql(u8, e.family, f);
        try testing.expectEqual(fam, ex.contains(e.kernel));
        if (e.header) |h| try testing.expectEqual(ex.contains(e.kernel), h == .dig2_x or h == .dig_mul1_k3 or h == .dig_mul1h_k3 or h == .dig_mul1h_k3_lut or h == .mul1h_rate);
    }
    // the EXL3-only check kinds stay on the EXL3 side
    for (&reg.entries) |*e| {
        const exl3_kind = e.checks.contains(.decode_table) or e.checks.contains(.golden_tiles) or e.checks.contains(.composition) or e.checks.contains(.mlx_chain);
        if (exl3_kind) try testing.expect(ex.contains(e.kernel));
    }
    // the executor's checks that launch a second kernel launch one of their own side
    const Pair = struct { Kernel, Kernel };
    for ([_]Pair{
        .{ .q3rc_router_tail, .q3rc_gate_part },                                             .{ .q3rc_router_tail__n128_top3, .q3rc_gate_part__n128 },
        .{ .q3rc_premix_fin, .q3rc_premix_part },                                            .{ .q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3, .q3_exl3_dig_decmat_5120x2304_mul1hk3 },
        .{ .q3_prefill_dig_gemm_2304x5120_xmul1hk3, .q3_exl3_dig_decmat_2304x5120_mul1hk3 }, .{ .q3ht_combine_collapse_norm, .q3ht_combine },
    }) |p| try testing.expectEqual(ex.contains(p[0]), ex.contains(p[1]));
    // a violation is named: a kernel in both lists, one in none
    try testing.expect(std.mem.indexOf(u8, ks.partitionError(&reg, &.{ &eq.kernels, &eq.kernels }, &buf).?, "subsets 0 and 1") != null);
    try testing.expect(std.mem.indexOf(u8, ks.partitionError(&reg, &.{&eq.kernels}, &buf).?, "in no subset") != null);
}

// ── 5. Each consumer's accept on its own subset (stub device) ──

test "dsv41 kernels c2: each consumer's accept (stub device) runs exactly its subset's plan; a failure refuses only its owner" {
    const a = testing.allocator;
    var peek = try @import("expert_bank.zig").peekText(a, @embedFile("fixtures/dsv41_bank_peek.json"), null);
    defer peek.deinit();
    var t: Trace = .{ .a = a };
    defer t.deinit();
    var diag: xk.Diag = .{};
    const set = try ks.Set.init(a, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    set.install(Trace, &t);
    try testing.expectEqual(@as(*const xk.Bound, &set.bound), t.launcher.?);
    // the construction plan of a subset: every kernel's compile, no numeric probe
    const Plan = struct {
        fn count(reg: *const xk.Registry, subset: []const Kernel) usize {
            const p = selfcheck.plan(reg, ks.subsetOf(subset), .compile);
            var n: usize = 0;
            for (subset) |k| n += p.get(k).count();
            return n;
        }
        /// The first check of `subset`'s construction plan (a compile: a scripted failure must be one that runs).
        fn probe(reg: *const xk.Registry, subset: []const Kernel) struct { k: Kernel, c: xk.Check } {
            const p = selfcheck.plan(reg, ks.subsetOf(subset), .compile);
            for (&reg.entries) |*e| {
                var it = p.get(e.kernel).iterator();
                while (it.next()) |c| return .{ .k = e.kernel, .c = c };
            }
            unreachable;
        }
    };
    // the EXL3 quant: exactly its construction plan, the decode routes prepared (GEMV 2 x 48, rin 4 x 48)
    const acc = try eq.accept(Trace, a, &t, .{ .kernels = set.ref(), .peek = &peek.view }, v41_spec, &diag);
    const exl3_plan = Plan.count(&set.reg, &eq.checked_at_accept);
    try testing.expectEqual(exl3_plan, acc.report.results.items.len);
    // every kernel compiles at construction, and nothing else runs
    try testing.expectEqual(eq.checked_at_accept.len, exl3_plan);
    const ex = ks.subsetOf(&eq.kernels);
    for (acc.report.results.items) |r| try testing.expect(ex.contains(r.kernel) and r.ok);
    try testing.expectEqual(@as(isize, 288), t.prepared_live);
    try testing.expectEqual(@as(usize, 40), acc.waves.len);
    // the fused down GEMM's arm: its compiles join the report, and every layer's waves launch it
    const fused_checks = Plan.count(&set.reg, &eq.w1_texts);
    try testing.expectEqual(eq.w1_texts.len, fused_checks);
    try testing.expectEqual(@as(u32, 5), acc.waves[0].wave_launches);
    try acc.routeFusedDown(set, &diag);
    try testing.expectEqual(exl3_plan + fused_checks, acc.report.results.items.len);
    for (acc.waves) |*w| try testing.expectEqual(@as(u32, 4), w.wave_launches);
    // the routed forms: the GEMVs rebuilt on their texts at construction (no check joins the report: the registry's
    // twins run in a check window), the prepared tables swap (gate|up's one-launch table joins: + 48)
    try acc.routeForms(&t, .{ .down_pair = true, .gu_one = true });
    try testing.expectEqual(Kernel.dsv41_exl3_pair_k3_5120, acc.gemv.dn.kernel);
    try testing.expect(acc.gemv.gu1_p != null);
    try testing.expectEqual(exl3_plan + fused_checks, acc.report.results.items.len);
    try testing.expectEqual(@as(isize, 288 + 48), t.prepared_live);
    // the banked route over the forms: six prepared tables (in_rin, gu_epi, din_rin, dpost, gu_one's, the pair's: + 288),
    // no kept array (its GEMV statics are the forms' own, aliased), no check joins the report; the forms are its
    // input, so a later forms route is refused
    const keeps0 = t.keeps;
    try acc.routeBanked(&t);
    try testing.expectEqual(@as(isize, 288 + 48 + 288), t.prepared_live);
    try testing.expectEqual(keeps0, t.keeps);
    try testing.expectEqual(Kernel.dsv41_exl3_b3_pair_k3_5120, acc.banked.?.dn_e.kernel);
    try testing.expectEqual(Kernel.dsv41_exl3_b3_guone_k3_2304, acc.banked.?.gu_e.kernel);
    for (5..acc.banked.?.dn_e.inputs.len) |i| try testing.expectEqual(acc.gemv.dn_statics.arrays[i - 2], acc.banked.?.dn_st[i]);
    try testing.expectEqual(exl3_plan + fused_checks, acc.report.results.items.len);
    try testing.expectError(error.FormsAfterBanked, acc.routeForms(&t, .{}));
    try testing.expectError(error.BankedRoutedTwice, acc.routeBanked(&t));
    acc.banked.?.deinit(&t);
    acc.banked = null;
    try testing.expectEqual(@as(isize, 288 + 48), t.prepared_live);
    try acc.routeForms(&t, .{});
    try testing.expectEqual(@as(isize, 288), t.prepared_live);
    // the trunk: its own construction plan, none of the EXL3 kernels
    var rep: selfcheck.Report = .{};
    defer rep.deinit(a);
    try tr.accept(a, set, &rep, &diag);
    try testing.expectEqual(Plan.count(&set.reg, &tr.kernels), rep.results.items.len);
    const full_exl3 = selfcheck.plan(&set.reg, ks.subsetOf(&eq.checked_at_accept), .full);
    const full_trunk = selfcheck.plan(&set.reg, ks.subsetOf(&tr.kernels), .full);
    var nf: [2]usize = .{ 0, 0 };
    for (eq.checked_at_accept) |k| nf[0] += full_exl3.get(k).count();
    for (tr.kernels) |k| nf[1] += full_trunk.get(k).count();
    std.debug.print("\nconstruction self-checks (checks, before sites): EXL3 accept {d} -> {d}, trunk {d} -> {d}\n", .{ nf[0], exl3_plan, nf[1], rep.results.items.len });
    for (rep.results.items) |r| try testing.expect(!ex.contains(r.kernel) and r.ok);
    // a scripted failure of a planned compile refuses its owner's accept by name; the other consumer's passes
    const pe = Plan.probe(&set.reg, &eq.checked_at_accept);
    const pt = Plan.probe(&set.reg, &tr.kernels);
    const Fail = struct { k: Kernel, c: xk.Check, exl3: bool };
    for ([_]Fail{ .{ .k = pe.k, .c = pe.c, .exl3 = true }, .{ .k = pt.k, .c = pt.c, .exl3 = false } }) |f| {
        const bad = try ks.Set.init(a, .{ .device = .{ .stub = .{ .fail = .{ .kernel = f.k, .check = f.c } } } }, &diag);
        defer bad.deinit();
        var r2: selfcheck.Report = .{};
        defer r2.deinit(a);
        var want: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&want, "{t} {t}", .{ f.k, f.c });
        if (f.exl3) {
            try testing.expectError(error.SelfCheckFailed, eq.accept(Trace, a, &t, .{ .kernels = bad.ref(), .peek = &peek.view }, v41_spec, &diag));
            try testing.expect(std.mem.indexOf(u8, diag.message(), name) != null);
            try tr.accept(a, bad, &r2, &diag);
        } else {
            try testing.expectError(error.SelfCheckFailed, tr.accept(a, bad, &r2, &diag));
            try testing.expect(std.mem.indexOf(u8, diag.message(), name) != null);
            const ok = try eq.accept(Trace, a, &t, .{ .kernels = bad.ref(), .peek = &peek.view }, v41_spec, &diag);
            ok.deinit(&t);
        }
    }
    // without the load context's kernel set: refused, by name
    try testing.expectError(error.NoKernelSet, eq.accept(Trace, a, &t, .{}, v41_spec, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "kernel set") != null);
    acc.deinit(&t);
    ks.Set.uninstall(Trace, &t);
    try testing.expect(t.launcher == null);
    try testing.expectEqual(@as(isize, 0), t.keeps);
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
}

test "dsv41 kernels c2: the kernel set refuses by name (text, pin) and a backend without a route method" {
    const a = testing.allocator;
    var diag: xk.Diag = .{};
    // a text that is not the pinned manifest's
    var texts = xk.embedded;
    const k = Kernel.q3drc_mxfp8_fma_f32x;
    const bad = try a.dupeSentinel(u8, xk.embedded.sources[@backingInt(k)], 0);
    defer a.free(bad);
    bad[bad.len / 3] ^= 0x04;
    texts.sources[@backingInt(k)] = bad;
    try testing.expectError(error.TextSha256Mismatch, ks.Set.init(a, .{ .device = .{ .stub = .{} }, .texts = &texts }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), @tagName(k)) != null);
    // another pin
    try testing.expectError(error.ManifestNotPinned, ks.Set.init(a, .{ .device = .{ .stub = .{} }, .pin = "0000000000000000000000000000000000000000000000000000000000000000" }, &diag));
    // the route methods: the first one a backend lacks (`Set.install` names it at compile time)
    try testing.expect(ks.missingBackendMethod(Trace) == null);
    try testing.expectEqualStrings("launch", ks.missingBackendMethod(struct {}).?);
    const NoRelease = struct {
        pub const T = u32;
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    try testing.expectEqualStrings("release", ks.missingBackendMethod(NoRelease).?);
}

// ── 7. A second quant on the same interface ──

/// A made-up gather quant (int8 rows, one scale per expert row) on the host trace: the C2
/// conformance through `FromGatherMatmul`, to show the interface is not EXL3-shaped.
const FakeInt8 = struct {
    pub const name = "fake-int8";

    pub fn Arrays(comptime T: type) type {
        return struct { w: T };
    }

    pub const Params = struct { rows_scale: bool };

    pub fn claims(peek: *const quant.BankPeek, why: ?*quant.Diag) ?quant.Priority {
        const mode = quant.str(peek.quantization, "mode") orelse "";
        if (!std.mem.eql(u8, mode, "fake-int8")) return quant.decline(why, "fake-int8: quantization.mode \"{s}\"", .{mode});
        return .native;
    }

    pub fn params(peek: *const quant.BankPeek, spec: quant.Spec, diag: *quant.Diag) !Params {
        _ = spec;
        if (claims(peek, diag) == null) return error.NotClaimed;
        return .{ .rows_scale = true };
    }

    pub fn gatherMatmul(comptime G: type, g: *G, p: *const Params, x: G.T, w: Arrays(G.T), rhs: G.T, sorted: bool) !G.T {
        _ = p;
        const sx = g.shapeOf(x);
        const y = try g.gatherMatmul(try g.reshape(x, &.{ sx.d[0], 1, sx.d[1] }), w.w, null, null, rhs, 8, 0, null, sorted);
        return g.reshape(y, &.{ sx.d[0], g.shapeOf(w.w).d[1] });
    }

    pub fn checkArrays(comptime G: type, g: *G, p: *const Params, which: quant.Proj, w: Arrays(G.T), in: u32, out: u32, diag: *quant.Diag) !void {
        _ = p;
        const s = g.shapeOf(w.w);
        if (g.dtypeOf(w.w) != .int8 or s.n != 3 or s.d[1] != @as(c_int, @intCast(out)) or s.d[2] != @as(c_int, @intCast(in)))
            return quant.refuse(diag, error.BankArrays, "fake-int8: {t} w is {t} {any}", .{ which, g.dtypeOf(w.w), s.slice() });
    }
};

fn peekOf(quantization: std.json.Value, layers: []const quant.LayerPeek) quant.BankPeek {
    return .{ .quantization = quantization, .hidden = 5120, .inter = 2304, .n_experts = 128, .n_layers = layers.len, .layers = layers };
}

/// One call's log lines (from `from` on): the graph ops and joins as `kt.traceEvent` renders them.
fn opLines(t: *const Trace, from: usize, out: *std.ArrayList(u8)) !void {
    for (t.log.items[from..]) |e| {
        try kt.traceEvent(t, e, out);
        try out.append(t.a, '\n');
    }
}

test "dsv41 kernels c2: a second quant through FromGatherMatmul passes the same conformance (claims, gateUp, down, prefill, checkBank)" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const fake_q = try std.json.parseFromSliceLeaky(std.json.Value, aa, "{\"mode\":\"fake-int8\"}", .{});
    const mx4_q = try std.json.parseFromSliceLeaky(std.json.Value, aa, "{\"mode\":\"mxfp4\",\"bits\":4,\"group_size\":32}", .{});
    const layers = [_]quant.LayerPeek{.{ .bits = 8, .segments = &.{} }};
    const fake_peek = peekOf(fake_q, &layers);
    const mx4_peek = peekOf(mx4_q, &layers);
    const F = quant.FromGatherMatmul(FakeInt8);
    const S = quant.FromGatherMatmul(quant.GatherQmm);
    var why: quant.Diag = .{};
    // claims: each quant claims its own description and declines the others', naming the field
    try testing.expectEqual(@as(?quant.Priority, .native), F.claims(&fake_peek, &why));
    try testing.expectEqual(@as(?quant.Priority, .generic), S.claims(&mx4_peek, &why));
    try testing.expectEqual(@as(?quant.Priority, null), S.claims(&fake_peek, &why));
    try testing.expect(std.mem.indexOf(u8, why.message(), "quantization.mode \"fake-int8\"") != null);
    try testing.expectEqual(@as(?quant.Priority, null), F.claims(&mx4_peek, &why));
    try testing.expectEqual(@as(?quant.Priority, null), eq.claims(&fake_peek, &why));
    try testing.expect(std.mem.indexOf(u8, why.message(), "quantization.mode") != null);
    const bad_q = try std.json.parseFromSliceLeaky(std.json.Value, aa, "{\"mode\":\"mxfp4\",\"bits\":8,\"group_size\":32}", .{});
    try testing.expectEqual(@as(?quant.Priority, null), S.claims(&peekOf(bad_q, &layers), &why));
    try testing.expect(std.mem.indexOf(u8, why.message(), "mxfp4 at 8 bits") != null);

    var t: Trace = .{ .a = a };
    defer t.deinit();
    var diag: quant.Diag = .{};
    try testing.expectError(error.NoWeightDescription, F.accept(Trace, a, &t, .{}, v41_spec, &diag));
    const fa = try F.accept(Trace, a, &t, .{ .peek = &fake_peek }, v41_spec, &diag);
    defer fa.deinit(&t);
    const sa = try S.accept(Trace, a, &t, .{ .peek = &mx4_peek }, v41_spec, &diag);
    defer sa.deinit(&t);
    try testing.expectEqual(quant.GatherQmm.Params{ .mode = .mxfp4, .bits = 4, .group_size = 32 }, sa.params);
    var lines: std.ArrayList(u8) = .empty;
    defer lines.deinit(a);

    // gateUp = two gathers + the clamped SwiGLU (mlx-lm's order); down = one gather; no kernel launch
    const x = try t.ext("x", &.{ 6, 5120 }, .bfloat16);
    const ids = try t.ext("ids", &.{6}, .uint32);
    const fw = struct {
        fn bank(tt: *Trace, dt: Dtype, packed_in: c_int, packed_inter: c_int) !quant.BankArrays(FakeInt8.Arrays(u32)) {
            return .{ .gate = .{ .w = try tt.ext("gate.w", &.{ 128, 2304, packed_in }, dt) }, .up = .{ .w = try tt.ext("up.w", &.{ 128, 2304, packed_in }, dt) }, .down = .{ .w = try tt.ext("down.w", &.{ 128, 5120, packed_inter }, dt) } };
        }
    };
    const fb = try fw.bank(&t, .int8, 5120, 2304);
    try fa.checkBank(&t, fb, &diag);
    const log0 = t.log.items.len;
    const h = try fa.gateUp(&t, x, ids, fb.gate, fb.up);
    const y = try fa.down(&t, h, ids, fb.down);
    try opLines(&t, log0, &lines);
    try testing.expectEqualStrings(
        \\op gather_mm sorted=false in=ext:x@[6,1,5120];ext:gate.w;ext:ids out=[6,1,2304]
        \\op gather_mm sorted=false in=ext:x@[6,1,5120];ext:up.w;ext:ids out=[6,1,2304]
        \\op scalar -10 in= out=[]
        \\op scalar 10 in= out=[]
        \\op clip in=O1@[6,2304];O2;O3 out=[6,2304]
        \\op scalar 10 in= out=[]
        \\op minimum in=O0@[6,2304];O5 out=[6,2304]
        \\op silu in=O6 out=[6,2304]
        \\op mul in=O7;O4 out=[6,2304]
        \\op gather_mm sorted=false in=O8@[6,1,2304];ext:down.w;ext:ids out=[6,1,5120]
        \\
    , lines.items);
    try testing.expectEqualSlices(c_int, &.{ 6, 5120 }, t.shapeOf(y).slice());
    try testing.expectEqual(@as(usize, 0), t.launches.items.len);
    // prefill: the rows sorted by slot on the host (stable), the gathers sorted, the order restored
    lines.clearRetainingCapacity();
    const log1 = t.log.items.len;
    const act = try t.ext("act", &.{ 3, 5120 }, .bfloat16);
    const yp = try fa.prefill(&t, 0, act, .{ .slot = &.{ 7, 2, 7, 2, 5, 2 }, .act_row = &.{ 0, 0, 1, 1, 2, 2 } }, fb);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 5, 4, 0, 2 }, fa.order.items);
    try testing.expectEqualSlices(u32, &.{ 2, 2, 2, 5, 7, 7 }, fa.ids.items);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 2, 0, 1 }, fa.src.items);
    try testing.expectEqualSlices(u32, &.{ 4, 0, 5, 1, 3, 2 }, fa.inv.items);
    try opLines(&t, log1, &lines);
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, lines.items, "take "));
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, lines.items, "sorted=true"));
    try testing.expectEqualSlices(c_int, &.{ 6, 5120 }, t.shapeOf(yp).slice());
    try fa.finishPrefill(&t);
    // checkBank refuses another dtype or shape, by projection
    const wrong = try fw.bank(&t, .int8, 4096, 2304);
    try testing.expectError(error.BankArrays, fa.checkBank(&t, wrong, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "gate w") != null);

    // the stock quant on the DSpark shape (mxfp4, group 32): gather_qmm with its params
    const sb: quant.BankArrays(quant.GatherQmm.Arrays(u32)) = .{
        .gate = .{ .w = try t.ext("w1", &.{ 128, 2304, 640 }, .uint32), .scales = try t.ext("s1", &.{ 128, 2304, 160 }, .uint8) },
        .up = .{ .w = try t.ext("w3", &.{ 128, 2304, 640 }, .uint32), .scales = try t.ext("s3", &.{ 128, 2304, 160 }, .uint8) },
        .down = .{ .w = try t.ext("w2", &.{ 128, 5120, 288 }, .uint32), .scales = try t.ext("s2", &.{ 128, 5120, 72 }, .uint8) },
    };
    try sa.checkBank(&t, sb, &diag);
    lines.clearRetainingCapacity();
    const log2 = t.log.items.len;
    _ = try sa.down(&t, try sa.gateUp(&t, x, ids, sb.gate, sb.up), ids, sb.down);
    try opLines(&t, log2, &lines);
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, lines.items, "op gather_qmm mxfp4 bits=4 group=32 sorted=false"));
    var no_scales = sb;
    no_scales.down.scales = null;
    try testing.expectError(error.BankArrays, sa.checkBank(&t, no_scales, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "down has no scales") != null);
}

// ── 4. The existing suite, through the moved routes ──

test "dsv41 kernels ops: a bound array of another dtype or shape is refused, by name" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    var diag: xk.Diag = .{};
    const f32w = try t.node(&.{ 384, 5120 }, .float32, &.{});
    const bias = try t.node(&.{384}, .float32, &.{});
    try testing.expectError(error.RouteInput, Router(Trace).init(&t, &reg, f32w, bias, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3rc_gate_part input w") != null);
    const short_w = try t.node(&.{ 24, 10240 }, .float32, &.{});
    try testing.expectError(error.RouteInput, Premix(Trace).init(&t, &reg, short_w, &diag));
    const w = try t.node(&.{ 1280, 1280 }, .uint32, &.{});
    const sc = try t.node(&.{ 1280, 40 }, .uint8, &.{});
    try testing.expectError(error.RouteInput, RcProj(Trace).init(&t, &reg, .wq_a, w, sc, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "q3rc_mxfp8_fma input scales") != null);
    try testing.expectError(error.TemplateNotRegistered, HcTape(Trace).init(&t, &reg, .float16, &diag));
    const qn32 = try t.node(&.{1280}, .float32, &.{});
    try testing.expectError(error.RouteInput, FusedProj(Trace).init(&t, &reg, qn32, try t.node(&.{512}, .bfloat16, &.{}), 1e-20, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "mtplx_dsv41_fp_rmsnorm_tg128_d1280 input weight") != null);
    // K36's RMSNorm eps is bound from the model config at install: another rms_norm_eps (or 0) is
    // refused before anything is built; the registered static is the f32 1e-20 the lane passes
    const qn = try t.node(&.{1280}, .bfloat16, &.{});
    const kn = try t.node(&.{512}, .bfloat16, &.{});
    for ([_]f32{ 0, 1e-6, 1e-19 }) |bad_eps| {
        try testing.expectError(error.RouteInput, FusedProj(Trace).init(&t, &reg, qn, kn, bad_eps, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "mtplx_dsv41_fp_rmsnorm_tg128_d1280 is registered at eps") != null);
    }
    inline for (.{ Kernel.mtplx_dsv41_fp_rmsnorm_tg128_d1280, Kernel.mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64 }) |k| {
        var sbuf: [1024]u8 = undefined;
        const shape, const bytes = kr.staticBytes(kr.argOf(reg.get(k), "eps"), &sbuf);
        try testing.expectEqual(@as(usize, 0), shape.slice().len);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&@as(f32, 1e-20)), bytes);
    }
    const bank: ProjArrays(Trace.T) = .{
        .code = try t.node(&.{ 4, 144, 320, 48 }, .int16, &.{}),
        .rout = try t.node(&.{ 4, 5120 }, .float32, &.{}),
        .rin = try t.node(&.{ 4, 2304 }, .float16, &.{}),
        .layout = .{ .fixed = eq.tightLayout(3) },
    };
    try testing.expectError(error.RouteInput, checkBank(Trace, &t, &reg, .down, bank, &diag));
    const ok: ProjArrays(Trace.T) = .{ .code = bank.code, .rout = try t.node(&.{ 4, 5120 }, .float16, &.{}), .rin = bank.rin, .layout = bank.layout };
    try checkBank(Trace, &t, &reg, .down, ok, &diag);
    try testing.expectError(error.RouteInput, checkBank(Trace, &t, &reg, .gate, ok, &diag));
    try testing.expectEqual(@as(isize, 0), t.keeps);
}

test "dsv41 kernels rates: padded K3 bank accepts contiguous payload capacity" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    var diag: xk.Diag = .{};
    const bank: ProjArrays(Trace.T) = .{
        .code = try t.node(&.{ 4, 144, 320, 64 }, .int16, &.{}),
        .rout = try t.node(&.{ 4, 5120 }, .float16, &.{}),
        .rin = try t.node(&.{ 4, 2304 }, .float16, &.{}),
        .layout = .{ .fixed = .{ .k = 3, .code_row_words = 144 * 320 * 64 } },
    };
    try checkBank(Trace, &t, &reg, .down, bank, &diag);
    const gate: ProjArrays(Trace.T) = .{
        .code = try t.node(&.{ 4, 320, 144, 48 }, .int16, &.{}),
        .rout = try t.node(&.{ 4, 2304 }, .float16, &.{}),
        .rin = try t.node(&.{ 4, 5120 }, .float16, &.{}),
        .layout = .{ .fixed = eq.tightLayout(3) },
    };
    const diagnostic = Rebuild(Trace).init(&reg);
    const slots = try t.node(&.{16}, .int32, &.{});
    try testing.expectError(error.RebuildNeedsTightK3, diagnostic.call(&t, gate, gate, bank, slots, 1));
    try testing.expectEqual(@as(usize, 0), t.launches.items.len);
}

test "dsv41 kernels ops: the routes carry the lanes' installed configuration" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    // Q3_DECODE_RCPROJ_INSTALL geometry: (R, KS, RG, KV) per site, R 1 at M 7 / 8; the head = wq_b's form
    const geo = [_]struct { RcSite, [4]i32 }{
        .{ .wq_a, .{ 2, 2, 1, 4 } }, .{ .wkv, .{ 2, 4, 1, 4 } }, .{ .wq_b, .{ 2, 1, 2, 4 } },
        .{ .wo_b, .{ 2, 2, 1, 4 } }, .{ .woa, .{ 2, 1, 2, 4 } }, .{ .head, .{ 2, 1, 2, 4 } },
    };
    const e = reg.get(.q3rc_mxfp8_fma);
    for (geo) |row| {
        var vars: Vars = .initFill(0);
        xk.siteVars(e.site(@tagName(row[0])).?, &vars);
        var r = try RcProj(Trace).init(&t, &reg, row[0], try t.arg(e, "w", &vars), try t.arg(e, "scales", &vars), null);
        defer r.deinit(&t);
        for (&r.plans.cfg, 1..) |*p, m| {
            const want = [4]i32{ if (m >= 7) 1 else row[1][0], row[1][1], row[1][2], row[1][3] };
            for ([_][]const u8{ "R", "KS", "RG", "KV" }, want) |name, v| try testing.expectEqual(v, templateInt(p.template, name));
            try testing.expectEqual(@as(i32, @intCast(m)), templateInt(p.template, "M"));
        }
    }
    // Q3_DECODE_RCTAIL_INSTALL router plan: N 384, K 5120, KP 512, P 10, top-6
    const part = reg.get(.q3rc_gate_part).template;
    const tail = reg.get(.q3rc_router_tail).template;
    try testing.expectEqual(@as(i32, 384), templateInt(part, "N"));
    try testing.expectEqual(@as(i32, 5120), templateInt(part, "K"));
    try testing.expectEqual(@as(i32, 512), templateInt(part, "KP"));
    try testing.expectEqual(@as(i32, 10), templateInt(tail, "P"));
    try testing.expectEqual(@as(i32, 6), templateInt(tail, "TOPK"));
    // q3_exl3_decode_kernels: the lane_uint launch's constants (MCG_MULT, 0, LOP3_MASK, LOP3_XOR)
    var gv = try Gemv(Trace).init(&t, &reg);
    defer gv.deinit(&t);
    const cb = t.nodes.items[gv.gu_statics.arrays[3]].bytes;
    var cbv: [4]u32 = undefined;
    for (&cbv, 0..) |*v, i| v.* = std.mem.readInt(u32, cb[i * 4 ..][0..4], .little);
    try testing.expectEqualSlices(u32, &.{ 0xCBAC1FED, 0, 0x8FFF8FFF, 0x3B603B60 }, &cbv);
    for (gv.dn_statics.arrays[4..9]) |z| try testing.expect(std.mem.allEqual(u8, t.nodes.items[z].bytes, 0));
    // config.json rms_norm_eps 1e-20 (a 0-d f32 input); every K36 kernel stores bf16
    var fp = try FusedProj(Trace).init(&t, &reg, try t.node(&.{1280}, .bfloat16, &.{}), try t.node(&.{512}, .bfloat16, &.{}), 1e-20, null);
    defer fp.deinit(&t);
    const eps = std.mem.bytesToValue(f32, t.nodes.items[fp.rms_statics.arrays[2]].bytes);
    try testing.expectEqual(@as(f32, 1e-20), eps);
    for ([_]*const Entry{ fp.rms, fp.rms_rope, fp.fwd, fp.inv }) |k36| try testing.expectEqual(Dtype.bfloat16, k36.template[0].value.dtype);
}

test "dsv41 kernels ops: the routed forms launch their texts with the stock arguments (down pair; gate + up in one launch)" {
    const xq = @import("exl3_quant.zig");
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    const ge = reg.get(.dsv41_exl3_mul1h_k3_2304);
    const de = reg.get(.dsv41_exl3_mul1h_k3_5120);
    const s = &ge.samples[0];
    const code_g = try t.arg(ge, "code", &s.vars);
    const code_u = try t.arg(ge, "code", &s.vars);
    const code_d = try t.arg(de, "code", &s.vars);
    for ([_]xq.Forms{ .{}, .{ .down_pair = true }, .{ .gu_one = true }, .{ .down_pair = true, .gu_one = true } }) |f| {
        var gv = try xq.Gemv(Trace).initForms(&t, &reg, f);
        defer gv.deinit(&t);
        for ([_]c_int{ 1, 6, 8 }) |m| {
            const xg = try t.node(&.{ m, 5120 }, .float32, &.{});
            const xu = try t.node(&.{ m, 5120 }, .float32, &.{});
            const ids = try t.node(&.{m}, .uint32, &.{});
            const n0 = t.launches.items.len;
            const z = try gv.projectGu(&t, xg, xu, ids, code_g, code_u, .{ .fixed = eq.tightLayout(3) }, .{ .fixed = eq.tightLayout(3) });
            if (f.gu_one) {
                try testing.expectEqual(n0 + 1, t.launches.items.len);
                const l = t.back(1);
                try testing.expectEqual(xk.Kernel.dsv41_exl3_guone_k3_2304, l.k);
                try testing.expectEqualSlices(Trace.T, &.{ xg, xu, ids, code_g, code_u }, l.inputs[0..l.n_in]);
                try testing.expectEqual([3]u32{ 4608, @intCast(2 * m), 1 }, l.cfg.grid);
                try testing.expectEqual(@as(usize, 2), l.cfg.n_out);
            } else {
                try testing.expectEqual(n0 + 2, t.launches.items.len);
                try testing.expectEqual(xk.Kernel.dsv41_exl3_mul1h_k3_2304, t.back(1).k);
            }
            try testing.expectEqualSlices(c_int, t.shapeOf(z[0]).slice(), t.shapeOf(z[1]).slice());
            const xd = try t.node(&.{ m, 2304 }, .float32, &.{});
            _ = try gv.project(&t, .down, xd, ids, code_d, .{ .fixed = eq.tightLayout(3) });
            const ld = t.back(1);
            try testing.expectEqual(if (f.down_pair) xk.Kernel.dsv41_exl3_pair_k3_5120 else xk.Kernel.dsv41_exl3_mul1h_k3_5120, ld.k);
            // the pair text takes mul1h's signature and grid
            try testing.expectEqual([3]u32{ 10240, @intCast(m), 1 }, ld.cfg.grid);
        }
    }
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
}

test "dsv41 kernels ops: the banked route launches its texts with every bank's arrays in bank order and the packed ids, one launch per stage" {
    const xq = @import("exl3_quant.zig");
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    const ge = reg.get(.dsv41_exl3_mul1h_k3_2304);
    const de = reg.get(.dsv41_exl3_mul1h_k3_5120);
    const ie = reg.get(.q3_exl3_prep_in_rin);
    const pe = reg.get(.q3_moeprep_dpost);
    const ne = reg.get(.q3_exl3_prep_din_rin);
    const s = &ge.samples[0];
    const P = xq.ProjArrays(Trace.T);
    var banks: [3]@import("sdk_ext.zig").quant.BankArrays(P) = undefined;
    for (&banks) |*b| b.* = .{
        .gate = .{ .code = try t.arg(ge, "code", &s.vars), .rout = try t.arg(reg.get(.q3_exl3_prep_gu_epi), "rg", &s.vars), .rin = try t.arg(ie, "rg", &s.vars), .layout = .{ .fixed = eq.tightLayout(3) } },
        .up = .{ .code = try t.arg(ge, "code", &s.vars), .rout = try t.arg(reg.get(.q3_exl3_prep_gu_epi), "ru", &s.vars), .rin = try t.arg(ie, "ru", &s.vars), .layout = .{ .fixed = eq.tightLayout(3) } },
        .down = .{ .code = try t.arg(de, "code", &s.vars), .rout = try t.arg(pe, "rd", &s.vars), .rin = try t.arg(ne, "rn", &s.vars), .layout = .{ .fixed = eq.tightLayout(3) } },
    };
    for ([_]xq.Forms{ .{}, .{ .down_pair = true }, .{ .gu_one = true }, .{ .down_pair = true, .gu_one = true } }) |f| {
        var gv = try xq.Gemv(Trace).initForms(&t, &reg, f);
        defer gv.deinit(&t);
        var bk = try xq.Banked(Trace).init(&t, &reg, f, &gv);
        defer bk.deinit(&t);
        for ([_]c_int{ 1, 6, 8 }) |m| {
            const x = try t.node(&.{ m, 5120 }, .bfloat16, &.{});
            const tok = try t.node(&.{m}, .int32, &.{});
            const ids = try t.node(&.{m}, .uint32, &.{});
            const n0 = t.launches.items.len;
            const h = try bk.gateUp(&t, x, tok, ids, &banks);
            try testing.expectEqual(n0 + @as(usize, if (f.gu_one) 3 else 4), t.launches.items.len);
            const li = t.launches.items[n0];
            try testing.expectEqual(xk.Kernel.dsv41_exl3_b3_prep_in_rin, li.k);
            try testing.expectEqualSlices(Trace.T, &.{ x, tok, banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, li.inputs[0..li.n_in]);
            const lg = t.launches.items[n0 + 1];
            if (f.gu_one) {
                try testing.expectEqual(xk.Kernel.dsv41_exl3_b3_guone_k3_2304, lg.k);
                try testing.expectEqualSlices(Trace.T, &.{ lg.inputs[0], lg.inputs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, lg.inputs[0..lg.n_in]);
                try testing.expectEqual([3]u32{ 4608, @intCast(2 * m), 1 }, lg.cfg.grid);
            } else {
                try testing.expectEqual(xk.Kernel.dsv41_exl3_b3_mul1h_k3_2304, lg.k);
                try testing.expectEqualSlices(Trace.T, &.{ banks[0].gate.code, banks[1].gate.code, banks[2].gate.code }, lg.inputs[2..5]);
                try testing.expectEqualSlices(Trace.T, gv.gu_statics.arrays[3..9], lg.inputs[5..11]);
                try testing.expectEqualSlices(Trace.T, &.{ banks[0].up.code, banks[1].up.code, banks[2].up.code }, t.launches.items[n0 + 2].inputs[2..5]);
            }
            try testing.expectEqual(xk.Kernel.dsv41_exl3_b3_prep_gu_epi, t.back(1).k);
            const n1 = t.launches.items.len;
            _ = try bk.down(&t, h, ids, &banks);
            try testing.expectEqual(n1 + 3, t.launches.items.len);
            const ld = t.launches.items[n1 + 1];
            try testing.expectEqual(if (f.down_pair) xk.Kernel.dsv41_exl3_b3_pair_k3_5120 else xk.Kernel.dsv41_exl3_b3_mul1h_k3_5120, ld.k);
            try testing.expectEqualSlices(Trace.T, &.{ ids, banks[0].down.code, banks[1].down.code, banks[2].down.code }, ld.inputs[1..5]);
            try testing.expectEqualSlices(Trace.T, gv.dn_statics.arrays[3..9], ld.inputs[5..11]);
            try testing.expectEqual([3]u32{ 10240, @intCast(m), 1 }, ld.cfg.grid);
            try testing.expectEqual(xk.Kernel.dsv41_exl3_b3_moeprep_dpost, t.back(1).k);
        }
    }
    try testing.expectEqual(@as(isize, 0), t.prepared_live);
}

test "dsv41 kernels ops: the routes launch the same through the profiling backend (dsv41_profile.Profiled over the host trace)" {
    const prof = @import("dsv41_profile.zig");
    const P = prof.Profiled(Trace);
    const a = testing.allocator;
    var peek = try @import("expert_bank.zig").peekText(a, @embedFile("fixtures/dsv41_bank_peek.json"), null);
    defer peek.deinit();
    var reg = try testRegistry();
    defer reg.deinit();
    var clock: u64 = 0;
    var pt = try P.init(a, .{ .a = a }, .{ .manual = &clock }, 2);
    // the wrapper deinits the wrapped trace (Trace.deinit is public in kernel_trace)
    defer pt.deinit();
    var t: Trace = .{ .a = a };
    defer t.deinit();
    // startup through the wrapper: the set's launcher lands on the wrapped backend; the EXL3
    // quant's decode routes prepare their 288 configs through the wrapper (the GEMV 2 x 48, the
    // rin stage 4 x 48: RowPlans sees the wrapped capability)
    var diag: xk.Diag = .{};
    const set = try ks.Set.init(a, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    set.install(P, &pt);
    const acc = try eq.accept(P, a, &pt, .{ .kernels = set.ref(), .peek = &peek.view }, v41_spec, &diag);
    try testing.expectEqual(@as(*const xk.Bound, &set.bound), pt.inner.launcher.?);
    try testing.expectEqual(@as(isize, 288), pt.inner.prepared_live);
    // the router at the lane's samples, wrapped (inside a phase) and not
    const pe, const te = .{ reg.get(.q3rc_gate_part), reg.get(.q3rc_router_tail) };
    var rp = try Router(P).init(&pt, &reg, try pt.inner.arg(pe, "w", &no_vars), try pt.inner.arg(te, "bias", &no_vars), null);
    var rt = try Router(Trace).init(&t, &reg, try t.arg(pe, "w", &no_vars), try t.arg(te, "bias", &no_vars), null);
    try testing.expectEqual(@as(isize, 288 + 16), pt.inner.prepared_live);
    prof.cycleBegin(&pt);
    prof.beginPhase(&pt, .barrier);
    for (pe.samples) |*s| {
        _ = try rp.call(&pt, try pt.inner.arg(pe, "x", &s.vars));
        _ = try rt.call(&t, try t.arg(pe, "x", &s.vars));
    }
    prof.endPhase(&pt);
    prof.cycleEnd(&pt);
    // the same launches in the same order (kernel, inputs, config, prepared), none added
    try testing.expect(t.launches.items.len >= 2);
    try testing.expectEqual(t.launches.items.len, pt.inner.launches.items.len);
    for (t.launches.items, pt.inner.launches.items) |x, y| {
        try testing.expect(x.k == y.k and x.n_in == y.n_in and x.prepared and y.prepared);
        try testing.expect(std.meta.eql(x.cfg, y.cfg));
    }
    // each counted once, in the phase that was open
    const c = pt.prof.stored[0].tags[@backingInt(prof.Tag.barrier)];
    try testing.expectEqual(@as(u32, @intCast(t.launches.items.len)), c.launches);
    try testing.expectEqual(@as(u32, 1), c.calls);
    rp.deinit(&pt);
    rt.deinit(&t);
    acc.deinit(&pt);
    ks.Set.uninstall(P, &pt);
    try testing.expect(pt.inner.launcher == null);
    try testing.expectEqual(@as(isize, 0), pt.inner.prepared_live);
}

test "dsv41 kernels rates: verify widths reuse prepared routes across rates and padded banks" {
    var reg = try testRegistry();
    defer reg.deinit();
    var t: Trace = .{ .a = testing.allocator };
    defer t.deinit();
    for ([_]eq.Forms{ .{}, .{ .down_pair = true, .gu_one = true } }) |forms| {
        var gv = try eq.Gemv(Trace).initRates(testing.allocator, &t, &reg, forms, 7);
        defer gv.deinit(&t);
        var b3 = try eq.Banked(Trace).init(&t, &reg, forms, &gv);
        defer b3.deinit(&t);
        const prepared = t.prepared_live;
        for ([_]u32{ 2, 3, 4, 3, 2 }) |k| {
            var banks: [3]eq.BankArrays(Trace.T) = undefined;
            for (&banks, 0..) |*b, bi| {
                const cap: c_int = @intCast(bi + 2);
                const width: c_int = if (bi == 2) 64 else @intCast(16 * k);
                const layout: eq.ProjectionStorage(Trace.T) = .{ .fixed = .{ .k = k, .code_row_words = 320 * 144 * @as(u64, @intCast(width)) } };
                b.* = .{
                    .gate = .{ .code = try t.node(&.{ cap, 320, 144, width }, .int16, &.{}), .rout = try t.node(&.{ cap, 2304 }, .float16, &.{}), .rin = try t.node(&.{ cap, 5120 }, .float16, &.{}), .layout = layout },
                    .up = .{ .code = try t.node(&.{ cap, 320, 144, width }, .int16, &.{}), .rout = try t.node(&.{ cap, 2304 }, .float16, &.{}), .rin = try t.node(&.{ cap, 5120 }, .float16, &.{}), .layout = layout },
                    .down = .{ .code = try t.node(&.{ cap, 144, 320, width }, .int16, &.{}), .rout = try t.node(&.{ cap, 5120 }, .float16, &.{}), .rin = try t.node(&.{ cap, 2304 }, .float16, &.{}), .layout = layout },
                };
            }
            for (1..49) |n| {
                const m: c_int = @intCast(n);
                const xg = try t.node(&.{ m, 5120 }, .float32, &.{});
                const ids = try t.node(&.{m}, .uint32, &.{});
                const tok = try t.node(&.{m}, .int32, &.{});
                const p = banks[2].gate;
                const z = try gv.projectGu(&t, xg, xg, ids, p.code, banks[2].up.code, p.layout, banks[2].up.layout);
                try testing.expectEqualSlices(c_int, &.{ m, 2304 }, t.shapeOf(z[0]).slice());
                const x = try t.node(&.{ m, 5120 }, .bfloat16, &.{});
                const h = try b3.gateUp(&t, x, tok, ids, &banks);
                const y = try b3.down(&t, h, ids, &banks);
                try testing.expectEqualSlices(c_int, &.{ m, 5120 }, t.shapeOf(y).slice());
                try testing.expectEqual(prepared, t.prepared_live);
            }
        }
    }
}
