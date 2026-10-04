//! mlx-stream: the native streaming stack as one plugin: the DeepSeek-V4.1 arch, the EXL3 quant it binds (the routed
//! experts' kernels over the package's pinned registry) and the EXL3 expert source that streams them. Built only where
//! the macOS-only sources are.

const sdk = @import("sdk");

pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 1, .minor = 0 },
    .mlx = "v0.32.2",
    .macos_only = true,
    .provides = .{
        .arch = @import("deepseek_v41_plugin.zig"),
        .quant = @import("exl3_quant.zig"),
        .expert_source = @import("exl3_source.zig"),
    },
};

/// What the host's tests read of the package, reached through the registry (`plugins.mlx_stream_testing`, null when a
/// build leaves the package out): the arch's config fixture and settings type, the modules the host's reader proof
/// drives, and the pins the conformance tests compare.
pub const testing = struct {
    pub const v41 = @import("deepseek_v41.zig");
    pub const engram = @import("deepseek_v41_engram.zig");
    pub const Settings = @import("deepseek_v41_settings.zig").Config;
    pub const kernel_manifest_sha256 = @import("exl3_kernels.zig").manifest_sha256;
    pub const stream_uses_reader = @import("expert_stream.zig").uses_reader;
};
