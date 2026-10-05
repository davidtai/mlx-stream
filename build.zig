const std = @import("std");

/// Standalone build: builds and tests this plugin against an mlx-serve checkout (`-Dmlx-serve=<path>`, default
/// ../mlx-serve). mlx-serve does not use this file: its build.zig roots a module at src/root.zig (one import, `sdk`)
/// and compiles csrc/ against its own staged MLX. Every step here runs the host's build with
/// `-Dmlx-stream-dir=<this checkout>`, so the plugin is always compiled exactly as the host compiles it.
pub fn build(b: *std.Build) void {
    const host = b.option([]const u8, "mlx-serve", "Path to an mlx-serve checkout (default: ../mlx-serve)") orelse "../mlx-serve";
    const filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const optimize = b.option([]const u8, "optimize", "Optimize mode passed to the host build (default: Debug)");
    const here = b.root.root_dir.path orelse ".";

    const Spec = struct { name: []const u8, desc: []const u8, args: []const []const u8, env: Env = .keep, filter: bool = false };
    const steps = [_]Spec{
        .{ .name = "test-hermetic", .desc = "Run the plugin's tests without a bank (DSV41_BANK unset; CPU, MLX_DEFAULT_DEVICE=cpu)", .args = &.{"mlx-stream-test"}, .env = .hermetic, .filter = true },
        .{ .name = "test-bank", .desc = "Run the plugin's tests with the bank in DSV41_BANK (CPU, MLX_DEFAULT_DEVICE=cpu)", .args = &.{"mlx-stream-test"}, .env = .bank, .filter = true },
        .{ .name = "conformance", .desc = "Run the plugin's conformance suite against the host (CPU lane, no device)", .args = &.{"mlx-stream-conformance"}, .env = .hermetic },
        .{ .name = "host-conformance", .desc = "Run the host SDK's tests, the host registry's conformance and this plugin's", .args = &.{"conformance"}, .env = .hermetic },
        .{ .name = "check", .desc = "Check that the host's server graph compiles with this plugin (no codegen)", .args = &.{"check"} },
        .{ .name = "check-slim", .desc = "Check the slim host with this plugin (no codegen)", .args = &.{ "check", "-Dslim=true" } },
        .{ .name = "profile", .desc = "Check the host graph with the plugin's profile probes compiled in", .args = &.{ "check", "-Dplugin-profile=true" } },
        .{ .name = "cell", .desc = "Build the served AR cell test binary (ReleaseFast) into the host's zig-out/tests/mlx-stream-test", .args = &.{ "mlx-stream-test-build", "-Doptimize=ReleaseFast", "-Dtest-filter=dsv41 served cell" } },
        .{ .name = "serve", .desc = "Build the host's ReleaseFast server with this plugin (zig-out/bin/mlx-serve in the host checkout)", .args = &.{"-Doptimize=ReleaseFast"} },
    };
    inline for (steps) |s| {
        const run = b.addSystemCommand(&.{ b.graph.zig_exe, "build" });
        run.setCwd(.{ .cwd_relative = host });
        run.addArgs(s.args);
        run.addArg(b.fmt("-Dmlx-stream-dir={s}", .{here}));
        if (s.filter) if (filter) |f| run.addArg(b.fmt("-Dtest-filter={s}", .{f}));
        if (optimize) |o| if (!std.mem.eql(u8, s.name, "cell") and !std.mem.eql(u8, s.name, "serve")) run.addArg(b.fmt("-Doptimize={s}", .{o}));
        switch (s.env) {
            .keep => {},
            .hermetic => {
                run.setEnvironmentVariable("MLX_DEFAULT_DEVICE", "cpu");
                run.removeEnvironmentVariable("DSV41_BANK");
            },
            .bank => run.setEnvironmentVariable("MLX_DEFAULT_DEVICE", "cpu"),
        }
        run.has_side_effects = true;
        b.step(s.name, s.desc).dependOn(&run.step);
    }

    // The compile-time refusals of the contracts in src/sdk_ext.zig: each case compiles src/refusals.zig with one bad
    // declaration against the host's sdk (compile only: no MLX stage or link) and passes only on the error that
    // names it.
    const refusals = b.step("refusals", "Compile each refusal case and expect the compile error that names it");
    const target = b.graph.host;
    const hostFile = struct {
        fn f(bb: *std.Build, h: []const u8, rel: []const u8) std.Build.LazyPath {
            return .{ .cwd_relative = bb.pathJoin(&.{ h, rel }) };
        }
    }.f;
    const log = b.createModule(.{ .root_source_file = hostFile(b, host, "src/log.zig"), .target = target, .link_libc = true });
    const io_util = b.createModule(.{ .root_source_file = hostFile(b, host, "src/io_util.zig"), .target = target, .link_libc = true });
    const mtp_acceptance = b.createModule(.{ .root_source_file = hostFile(b, host, "src/mtp_acceptance.zig"), .target = target });
    const mlx = b.createModule(.{ .root_source_file = hostFile(b, host, "src/mlx.zig"), .target = target, .link_libc = true, .imports = &.{.{ .name = "log", .module = log }} });
    const sdk_build = b.addOptions();
    sdk_build.addOption(bool, "plugin_profile", false);
    const sdk = b.createModule(.{ .root_source_file = hostFile(b, host, "src/sdk.zig"), .target = target, .link_libc = true, .imports = &.{
        .{ .name = "mlx", .module = mlx },
        .{ .name = "log", .module = log },
        .{ .name = "io_util", .module = io_util },
        .{ .name = "mtp_acceptance", .module = mtp_acceptance },
        .{ .name = "sdk_build", .module = sdk_build.createModule() },
    } });
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

    // HOST_PIN: the mlx-serve commit this repo is tested against.
    const pin = b.addSystemCommand(&.{ "sh", "scripts/check_host_pin.sh", host });
    pin.setCwd(b.path("."));
    pin.has_side_effects = true;
    b.step("host-pin", "Compare HOST_PIN with the host checkout's HEAD").dependOn(&pin.step);
}

const Env = enum { keep, hermetic, bank };

/// src/refusals.zig's cases and the compile error line each must end with.
const refusal_cases = [_]struct { case: []const u8, err: []const u8 }{
    .{ .case = "expert_source_wrong_caps", .err = "WrongCaps.caps: a u32 where the SDK has sdk_ext.expert.Caps" },
    .{ .case = "quant_half_contract", .err = "HalfQuant: no Accepted" },
    .{ .case = "quant_claims_mistyped", .err = "QuantWrongClaims.claims: takes a different parameter count than the SDK's" },
    .{ .case = "quant_check_direct", .err = "QuantWrongClaims.claims: takes a different parameter count than the SDK's" },
    .{ .case = "profile_hook_missing_probe", .err = "the prefill probes lack now" },
};
