//! mlx-stream's one bridge to host files, for its harnesses (the AR cell, parity) and its bank tests: the host's
//! config parse through the registry, its loaders and its memory knobs. Served code reaches the host only through
//! `sdk`; this file refuses to compile outside a test build, so no served path can import it.

const std = @import("std");
const builtin = @import("builtin");
const settings = @import("deepseek_v41_settings.zig");

comptime {
    if (!builtin.is_test) @compileError("deepseek_v41_host.zig is the harness and test bridge to the host; served code reaches the host only through sdk");
}

pub const model = @import("../model.zig");
pub const gpu_ceiling = @import("../gpu_ceiling.zig");
pub const transformer = @import("../transformer.zig");

/// The host's loaders, as the served load hands them over (`sdk.LoadCtx.loader`).
pub const loader: *const @import("sdk").WeightLoader = &model.weight_loader;

/// The host's parsed config of this arch's model (its registry entry's config), with the host's load facts.
pub fn configOf(host: *const model.ModelConfig) error{NotDeepseekV41}!settings.Config {
    const vt = host.arch orelse return error.NotDeepseekV41;
    if (!std.mem.eql(u8, vt.name, "deepseek_v41")) return error.NotDeepseekV41;
    const c: *const settings.Config = @ptrCast(@alignCast(host.arch_cfg.?));
    return c.withFacts(&host.loadFacts());
}

/// A model directory's config as the host parses it (config.json through the registry), as this arch's config;
/// its strings live in `a`.
pub fn loadConfig(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !settings.Config {
    return configOf(&try model.parseConfig(io, a, model_dir));
}
