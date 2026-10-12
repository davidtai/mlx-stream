//! mlx-stream: the native streaming stack as one plugin: the DeepSeek-V4.1 arch, the EXL3 quant it binds (the routed
//! experts' kernels over the package's pinned registry) and the EXL3 expert source that streams them; MiMo-V2.6-Flash
//! and GLM-5.3 over the same expert reader. Built only where the macOS-only sources are. The module imports its own
//! `sdk` (sdk/) and the host's `mlx_host`.


pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 2, .minor = 1 },
    // The MLX this plugin is tested on; the conformance suite fails when the host links another.
    .mlx = "v0.32.3",
    .macos_only = true,
    // The host registers the arch; the EXL3 quant and the EXL3 expert source are its internals, bound at comptime
    // (sdk_ext.zig).
    .provides = .{ .arch = @import("arch/deepseek_v41/plugin.zig") },
};

/// MiMo-V2.6-Flash's prepared MXFP4 streaming pack: the package's second arch, over the same expert reader.
pub const plugin_mimo = sdk.Plugin{
    .name = "mlx-stream-mimo",
    .api = .{ .major = 2, .minor = 0 },
    .mlx = "v0.32.3",
    .macos_only = true,
    .provides = .{ .arch = @import("arch/mimo_v2/root.zig") },
};

/// What the host's glue calls for the second arch: its entry points, the context a load bills when the model sets
/// none, and the positions a request may generate past that context (none: MiMo reserves generation inside it).
pub const mimo = struct {
    pub const arch = @import("arch/mimo_v2/root.zig");
    pub const default_context: u64 = @import("arch/mimo_v2/settings.zig").fill_reserve_tokens;
    pub const generation_headroom: u64 = 0;
};

/// What the host's glue calls for the third arch: GLM-5.3's entry points, the context a load bills when the model sets
/// none (the standard request) and the generation a request may take past it until a parse has read the model's
/// `max_output` (its default, 128K: the glue reads the arch's `maxOutput` once it has the model's settings).
pub const glm = struct {
    pub const arch = @import("arch/glm_moe_dsa/plugin.zig");
    pub const default_context: u64 = @import("phase_memory.zig").fill_prompt_tokens;
    pub const generation_headroom: u64 = @import("arch/glm_moe_dsa/settings.zig").Config.default_max_output;
};

/// What the host's tests read of the package, reached through the registry (`plugins.mlx_stream_testing`, null when a
/// build leaves the plugin out): the arch's config fixture and settings type, the modules the host's reader proof
/// drives, and the pins the conformance tests compare.
pub const testing = struct {
    pub const v41 = @import("arch/deepseek_v41/deepseek_v41.zig");
    pub const engram = @import("arch/deepseek_v41/engram.zig");
    pub const Settings = @import("arch/deepseek_v41/settings.zig").Config;
};

/// What the host's glue (mlx-serve `src/arch/mlx_stream.zig`) calls: the arch's entry points and their contract types.
pub const arch = @import("arch/deepseek_v41/plugin.zig");
/// Every arch the package serves through the SDK's table (`sdk.Arch.of`), in claim order: DeepSeek-V4.1 and GLM-5.3
/// (the conformance suite's list; the host's glue binds each arch by its own entry: the root's, `mimo` and `glm`).
pub const archs = .{ @import("arch/deepseek_v41/plugin.zig"), @import("arch/glm_moe_dsa/plugin.zig") };
pub const sdk = @import("sdk");
/// The host surface beyond the SDK's this build sees (comptime, from the host's pub declarations): GLM-5.3's.
pub const host_caps = @import("host_caps.zig");
/// The context a construction bills when the model sets none (`ctx_size`): the standard request.
pub const default_context: u64 = @import("arch/deepseek_v41/bill.zig").fill_prompt_tokens;
/// Positions a request may generate past its prompt (the KV lanes' bound) for an arch without its own `maxOutput`
/// (DeepSeek-V4.1's).
pub const generation_headroom: u64 = @import("arch/deepseek_v41/module.zig").generation_headroom;
