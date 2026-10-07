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
}
