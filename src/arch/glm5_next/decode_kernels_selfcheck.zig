//! The self-check of GLM-5.3-Flash's NVFP4 decode kernel registry (`decode_kernels`), as `sdk_ext.kernels.KernelSet` runs
//! it and the module at its construction (`decode_kernels.install`): `compile`, the kernel builds, binds and completes one
//! launch over zeros at every instantiation the manifest registers, at the wave widths that reach each of MLX's two
//! passings of `ids` (a one-token instantiation at 1 pair, `constant`, and at 8, `device`; a union one at 9 and at the
//! widest wave, 64), with the banks' global scales at 8 slots (`device`, as the module's banks pass them). No numeric
//! probe: the numerics are the window's parity test's (`decode_kernels_parity.zig`) and the CPU reference's
//! (`decode_kernels_ref.zig`).

const std = @import("std");
const mlx = @import("sdk").mlx;
const log = @import("sdk").log;
const xk = @import("decode_kernels.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Check = xk.Check;

const scaffold = kset.Checks(Kernel, Check);
pub const Result = scaffold.Result;
pub const Report = scaffold.Report;
pub const Depth = kset.CheckDepth;
const Scope = kset.Scope;

/// The checks of `want`'s entries at `depth`, per kernel.
pub fn plan(reg: *const xk.Registry, want: std.EnumSet(Kernel), depth: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
    return scaffold.plan(&reg.entries, want, depth);
}

/// The wave widths an instantiation's compile check launches: the `constant` and the `device` passing of `ids`.
pub fn checkWidths(inst: xk.Instance) []const u32 {
    return if (inst.maxm == 1) &.{ 1, xk.top_k } else &.{ xk.union_min_pairs, xk.max_pairs };
}

/// The plan over `subset` at `depth` on the bound kernels: each kernel at every instantiation the manifest registers and
/// each of its check widths, one result per launch.
pub fn runSubset(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, subset: []const Kernel, depth: Depth, report: *Report) !void {
    var want: std.EnumSet(Kernel) = .empty;
    for (subset) |k| want.insert(k);
    const checks = plan(reg, want, depth);
    for (&reg.entries) |*e| {
        if (!want.contains(e.kernel) or !checks.get(e.kernel).contains(.compile)) continue;
        for (e.instances) |inst| for (checkWidths(inst)) |pairs| try checkOne(a, bound, e.kernel, inst, pairs, report);
    }
}

/// The verdict over a finished plan: refused (error.SelfCheckFailed, `diag` naming the first failure) unless every
/// result passed and the plan ran.
pub fn judge(report: *const Report, diag: ?*xk.Diag) error{SelfCheckFailed}!void {
    return scaffold.judge(report, "glm5next nvfp4 kernels", diag);
}

/// One `compile` result of `k` at `inst` over `pairs` pairs: a raised launch is recorded with its error and the latched
/// MLX message.
fn checkOne(a: Allocator, bound: *const xk.Bound, k: Kernel, inst: xk.Instance, pairs: u32, report: *Report) !void {
    launchZeros(a, bound, k, inst, pairs) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        var buf: [512]u8 = undefined;
        const m = mlx.takeError(&buf) orelse "";
        log.err("[glm5next nvfp4 selfcheck] {t} at ROWS {d}, MAXM {d}, {d} pairs: {s} {s}\n", .{ k, inst.rows, inst.maxm, pairs, @errorName(e), m });
        return report.appendRaised(a, k, .compile, e, m);
    };
    try report.results.append(a, .{ .kernel = k, .check = .compile, .words = pairs, .ok = true });
}

/// Slots of a check bank's global scales: 8 lines of 32 f32 (bank.zig `scale_line_floats`; MLX's `device` passing, as
/// the module's line-isolated banks).
const check_scale_slots = 8;
const check_scale_floats = check_scale_slots * @import("bank.zig").scale_line_floats;

/// One launch of `k` at `inst` over `pairs` pairs of zeros, evaluated: each bank one slot of zero words and scales (the
/// same arrays in all three banks), its global scales `check_scale_slots` zeros, every id slot 0 of bank 0.
fn launchZeros(a: Allocator, bound: *const xk.Bound, k: Kernel, inst: xk.Instance, pairs: u32) !void {
    var sc: Scope = .{ .a = a };
    defer sc.deinit();
    const rows: usize = inst.n;
    const cols: usize = inst.k;
    const xz = try a.alloc(u16, pairs * cols);
    defer a.free(xz);
    @memset(xz, 0);
    const wz = try a.alloc(u32, rows * cols / 8);
    defer a.free(wz);
    @memset(wz, 0);
    const sz = try a.alloc(u8, rows * cols / inst.gs);
    defer a.free(sz);
    @memset(sz, 0);
    const gz: [check_scale_floats]f32 = @splat(0);
    const idz = try a.alloc(u32, pairs);
    defer a.free(idz);
    @memset(idz, 0);
    const x = try sc.host(std.mem.sliceAsBytes(xz), &.{ @intCast(pairs), @intCast(cols) }, .bfloat16);
    const w = try sc.host(std.mem.sliceAsBytes(wz), &.{ 1, @intCast(rows), @intCast(cols / 8) }, .uint32);
    const s = try sc.host(sz, &.{ 1, @intCast(rows), @intCast(cols / inst.gs) }, .uint8);
    const g = try sc.host(std.mem.sliceAsBytes(&gz), &.{ check_scale_slots, @import("bank.zig").scale_line_floats }, .float32);
    const ids = try sc.host(std.mem.sliceAsBytes(idz), &.{@intCast(pairs)}, .uint32);
    var inputs: [xk.max_inputs]mlx.mlx_array = undefined;
    var n: usize = 0;
    inputs[n] = x;
    n += 1;
    const per: usize = if (xk.familyOf(k) == .gate_up) 2 else 1;
    for (0..xk.n_banks * per) |_| {
        inputs[n..][0..3].* = .{ w, s, g };
        n += 3;
    }
    inputs[n] = ids;
    n += 1;
    const cfg = xk.waveLaunch(k, pairs, inst, .bfloat16);
    var outs: [xk.max_outputs]mlx.mlx_array = undefined;
    try bound.apply(k, inputs[0..n], &cfg, &outs);
    _ = try sc.keep(outs[0]);
    try mlx.check(mlx.mlx_array_eval(outs[0]));
}

const testing = std.testing;

test "glm5next nvfp4 selfcheck: the stub device's plan gives one compile result per kernel; a scripted failure refuses by name" {
    const KS = kset.KernelSet(xk);
    var diag: xk.Diag = .{};
    const set = try KS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    var report: Report = .{};
    defer report.deinit(testing.allocator);
    try set.selfCheck(testing.allocator, std.enums.values(Kernel), .compile, &report, &diag);
    try testing.expectEqual(@as(usize, xk.n_kernels), report.results.items.len);
    for (report.results.items) |r| try testing.expect(r.ok and r.check == .compile);
    const bad = try KS.Set.init(testing.allocator, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .glm5n_nvfp4_down, .check = .compile } } } }, &diag);
    defer bad.deinit();
    var r2: Report = .{};
    defer r2.deinit(testing.allocator);
    try testing.expectError(error.SelfCheckFailed, bad.selfCheck(testing.allocator, &.{.glm5n_nvfp4_down}, .compile, &r2, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4_down") != null);
    // the check widths reach both of MLX's passings of ids, within the widest wave
    try testing.expectEqualSlices(u32, &.{ 1, 8 }, checkWidths(xk.gate_up_instances[0]));
    try testing.expectEqualSlices(u32, &.{ 9, 64 }, checkWidths(xk.down_instances[1]));
}

test "glm5next nvfp4 selfcheck: every kernel builds and completes a launch at every registered instantiation and check width on the GPU; the module's install" {
    _ = std.c.getenv("GLM5NEXT_NVFP4_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const set = try xk.openForTest(a, s);
    defer set.deinit();
    var report: Report = .{};
    defer report.deinit(a);
    try runSubset(a, &set.reg, &set.bound, std.enums.values(Kernel), .full, &report);
    try judge(&report, null);
    var launches: usize = 0;
    for (std.enums.values(Kernel)) |k| for (xk.instancesOf(k)) |inst| {
        launches += checkWidths(inst).len;
    };
    try testing.expectEqual(launches, report.results.items.len);
    var diag: xk.Diag = .{};
    var inst = xk.install(a, s, &diag) catch |e| {
        std.debug.print("glm5next nvfp4 install refused: {s}\n", .{diag.message()});
        return e;
    };
    defer inst.deinit();
    try testing.expectEqual(xk.max_pairs, inst.fused.n_prepared);
    std.debug.print("GLM5N_NVFP4_SELFCHECK PASS launches={d}\n", .{report.results.items.len});
}
