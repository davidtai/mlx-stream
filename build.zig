const std = @import("std");

/// Standalone build: runs an mlx-serve checkout's build (`-Dmlx-serve=<path>`, default ../mlx-serve) with
/// `-Dmlx-stream-dir=<this checkout>`, so the plugin is always compiled exactly as the host compiles it.
pub fn build(b: *std.Build) void {
    const host = b.option([]const u8, "mlx-serve", "Path to an mlx-serve checkout (default: ../mlx-serve)") orelse "../mlx-serve";
    const filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const here = b.root.root_dir.path orelse ".";

    const Spec = struct { name: []const u8, desc: []const u8, args: []const []const u8 };
    const steps = [_]Spec{
        .{ .name = "test", .desc = "Run the plugin's tests (the host's mlx-stream-test step)", .args = &.{"mlx-stream-test"} },
        .{ .name = "conformance", .desc = "Run the plugin's conformance suite (CPU lane, no device)", .args = &.{"mlx-stream-conformance"} },
        .{ .name = "serve", .desc = "Build the host's ReleaseFast server with this plugin (zig-out/bin/mlx-serve in the host checkout)", .args = &.{"-Doptimize=ReleaseFast"} },
    };
    inline for (steps) |s| {
        const run = b.addSystemCommand(&.{ b.graph.zig_exe, "build" });
        run.setCwd(.{ .cwd_relative = host });
        run.addArgs(s.args);
        run.addArg(b.fmt("-Dmlx-stream-dir={s}", .{here}));
        if (filter) |f| run.addArg(b.fmt("-Dtest-filter={s}", .{f}));
        run.has_side_effects = true;
        b.step(s.name, s.desc).dependOn(&run.step);
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

/// src/refusals.zig's cases and the compile error line each must end with.
const refusal_cases = [_]struct { case: []const u8, err: []const u8 }{
    .{ .case = "expert_source_wrong_caps", .err = "WrongCaps.caps: a u32 where the SDK has sdk_ext.expert.Caps" },
    .{ .case = "quant_half_contract", .err = "HalfQuant: no Accepted" },
    .{ .case = "quant_claims_mistyped", .err = "QuantWrongClaims.claims: takes a different parameter count than the SDK's" },
    .{ .case = "quant_check_direct", .err = "QuantWrongClaims.claims: takes a different parameter count than the SDK's" },
    .{ .case = "profile_hook_missing_probe", .err = "the prefill probes lack now" },
};
