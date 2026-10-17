//! The self-check of GLM-5.3-Flash's NVFP4 decode kernel registry (`decode_kernels`): the shared set's
//! (src/expert/nvfp4_decode_kernels.zig `SelfcheckOf`), as `sdk_ext.kernels.KernelSet` runs it and the module at its
//! construction (`decode_kernels.install`): `compile`, the kernel builds, binds and completes one launch over zeros at
//! every instantiation the manifest registers, at the wave widths that reach each of MLX's two passings of `ids` (a
//! one-token instantiation at 1 pair, `constant`, and at 8, `device`; a union one at 9 and at the widest wave, 64), with
//! the banks' global scales at 8 slots (`device`, as the module's banks pass them). No numeric probe: the numerics are
//! the window's parity test's (`decode_kernels_parity.zig`) and the CPU reference's (`decode_kernels_ref.zig`).

const std = @import("std");
const mlx = @import("sdk").mlx;
const xk = @import("decode_kernels.zig");
const kset = @import("../../sdk_ext/kernel_set.zig");

const Kernel = xk.Kernel;
const S = xk.selfcheck;

pub const Result = S.Result;
pub const Report = S.Report;
pub const Depth = S.Depth;
pub const plan = S.plan;
pub const checkWidths = S.checkWidths;
pub const runSubset = S.runSubset;
pub const judge = S.judge;

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
