//! mlx-stream: the native streaming stack as one plugin: the DeepSeek-V4.1 arch, the EXL3 quant it binds (the routed
//! experts' kernels over the package's pinned registry) and the EXL3 expert source that streams them. Built only where
//! the macOS-only sources are. The module's one import is the host's `sdk`.

const sdk = @import("sdk");

pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 1, .minor = 0 },
    // The MLX this plugin is tested on; the host refuses it at compile time unless its own MLX is the same.
    .mlx = "v0.32.3",
    .macos_only = true,
    // The host registers the arch; the EXL3 quant and the EXL3 expert source are its internals, bound at comptime
    // (sdk_ext.zig).
    .provides = .{ .arch = @import("deepseek_v41_plugin.zig") },
};

/// What the host's tests read of the package, reached through the registry (`plugins.mlx_stream_testing`, null when a
/// build leaves the plugin out): the arch's config fixture and settings type, the modules the host's reader proof
/// drives, and the pins the conformance tests compare.
pub const testing = struct {
    pub const v41 = @import("deepseek_v41.zig");
    pub const engram = @import("deepseek_v41_engram.zig");
    pub const Settings = @import("deepseek_v41_settings.zig").Config;
};
