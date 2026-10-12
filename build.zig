const std = @import("std");

/// Standalone build: runs an mlx-serve checkout's build (`-Dmlx-serve=<path>`, default ../mlx-serve) with
/// `-Dmlx-stream-dir=<this checkout>`, so the plugin is always compiled exactly as the host compiles it.
pub fn build(b: *std.Build) void {
    const host = b.option([]const u8, "mlx-serve", "Path to an mlx-serve checkout (default: ../mlx-serve)") orelse "../mlx-serve";
    const filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const optimize_name = b.option([]const u8, "optimize", "Optimization mode (default ReleaseSafe; tests retain safety checks)") orelse "ReleaseSafe";
    const optimize: std.builtin.Optimize = if (std.mem.eql(u8, optimize_name, "ReleaseFast"))
        .fast
    else if (std.mem.eql(u8, optimize_name, "ReleaseSafe"))
        .safe
    else if (std.mem.eql(u8, optimize_name, "Debug"))
        .debug
    else
        std.meta.stringToEnum(std.builtin.Optimize, optimize_name) orelse @panic("invalid optimization mode");
    const test_optimize: std.builtin.Optimize = switch (optimize) {
        .debug, .safe => optimize,
        else => .safe,
    };
    const here = b.root.root_dir.path orelse ".";
    addExl3Bench(b, host, optimize, test_optimize);

    const bill_tests = b.addTest(.{
        .name = "memory-bill-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("sdk/memory_bill.zig"),
            .target = b.graph.host,
            .optimize = test_optimize,
            .link_libc = true,
        }),
        .filters = if (filter) |f| &.{f} else &.{},
    });
    const run_bill_tests = b.addRunArtifact(bill_tests);
    b.step("memory-bill-test", "Run pure SDK memory admission tests").dependOn(&run_bill_tests.step);

    const Spec = struct { name: []const u8, desc: []const u8, args: []const []const u8 };
    const steps = [_]Spec{
        .{ .name = "test", .desc = "Run the plugin's tests (the host's mlx-stream-test step)", .args = &.{"mlx-stream-test"} },
        .{ .name = "conformance", .desc = "Run the plugin's conformance suite (CPU lane, no device)", .args = &.{"mlx-stream-conformance"} },
        .{ .name = "serve", .desc = "Build the host's ReleaseFast server with this plugin (zig-out/bin/mlx-serve in the host checkout)", .args = &.{"-Doptimize=fast"} },
    };
    inline for (steps) |s| {
        const run = b.addSystemCommand(&.{ b.graph.zig_exe, "build" });
        run.setCwd(.{ .cwd_relative = host });
        run.addArgs(s.args);
        if (!std.mem.eql(u8, s.name, "serve"))
            run.addArg(b.fmt("-Doptimize={s}", .{@tagName(test_optimize)}));
        run.addArg(b.fmt("-Dmlx-stream-dir={s}", .{here}));
        if (filter) |f| run.addArg(b.fmt("-Dtest-filter={s}", .{f}));
        run.has_side_effects = true;
        const step = b.step(s.name, s.desc);
        step.dependOn(&run.step);
        if (std.mem.eql(u8, s.name, "test")) step.dependOn(&run_bill_tests.step);
    }

    // The compile-time refusals of the contracts in src/sdk_ext.zig: each case compiles src/refusals.zig with one bad
    // declaration against this repo's sdk over the host's `mlx_host` (compile only: no MLX stage or link) and passes
    // only on the error that names it.
    const refusals = b.step("refusals", "Compile each refusal case and expect the compile error that names it");
    const target = b.graph.host;
    const mlx_host = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ host, "src/plugin_host.zig" }) }, .target = target, .link_libc = true });
    const sdk = b.createModule(.{ .root_source_file = b.path("sdk/root.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "mlx_host", .module = mlx_host }} });
    for (refusal_cases) |c| {
        const opts = b.addOptions();
        opts.addOption([]const u8, "name", c.case);
        const m = b.createModule(.{ .root_source_file = b.path("src/refusals.zig"), .target = target, .link_libc = true, .imports = &.{
            .{ .name = "sdk", .module = sdk },
            .{ .name = "refusal_case", .module = opts.createModule() },
        } });
        const obj = b.addObject(.{ .name = b.fmt("refusal-{s}", .{c.case}), .root_module = m });
        obj.expect_errors = .{ .contains = c.err };
        refusals.dependOn(&obj.step);
    }
}

/// Only the host's MLX ABI and log surface: no server, model, reader pool or C++ shims.
fn addExl3Bench(b: *std.Build, host_dir: []const u8, optimize: std.builtin.Optimize, test_optimize: std.builtin.Optimize) void {
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .macos,
        .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
    });
    const root = benchModule(b, host_dir, target, optimize);
    const exe = b.addExecutable(.{ .name = "exl3-bench", .root_module = root });
    b.step("exl3-bench", "Build the standalone selected-record EXL3 benchmark").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const test_root = benchModule(b, host_dir, target, test_optimize);
    const tests = b.addTest(.{ .name = "exl3-bench-test", .root_module = test_root, .filters = &.{"exl3 bench:"} });
    b.step("exl3-bench-test", "Run deterministic native benchmark tests (no GPU)").dependOn(&b.addRunArtifact(tests).step);
}

fn benchModule(b: *std.Build, host_dir: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize) *std.Build.Module {
    const host = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ host_dir, "src/plugin_host.zig" }) },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const sdk = b.createModule(.{
        .root_source_file = b.path("sdk/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
    const root = b.createModule(.{
        .root_source_file = b.path("src/exl3_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "sdk", .module = sdk }},
    });
    const lib_dir = b.pathResolve(&.{ host_dir, "lib/mlx/lib" });
    root.addLibraryPath(.{ .cwd_relative = lib_dir });
    root.addRPath(.{ .cwd_relative = lib_dir });
    root.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    return root;
}

/// src/refusals.zig's cases and the compile error line each must end with.
const refusal_cases = [_]struct { case: []const u8, err: []const u8 }{
    .{ .case = "expert_source_wrong_caps", .err = "WrongCaps.caps: a u32 where the SDK has sdk_ext.expert.Caps" },
    .{ .case = "quant_half_contract", .err = "HalfQuant: no Accepted" },
    .{ .case = "quant_claims_mistyped", .err = "QuantWrongClaims.claims: takes a different parameter count than the SDK's" },
    .{ .case = "quant_check_direct", .err = "QuantWrongClaims.claims: takes a different parameter count than the SDK's" },
    .{ .case = "profile_hook_missing_probe", .err = "the prefill probes lack now" },
};
