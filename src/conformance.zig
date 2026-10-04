//! mlx-stream's conformance suite (docs/plugins.md, CPU lane): `sdk.testing` over this plugin's declaration, its arch's
//! table, and the EXL3 quant and expert source the arch binds; then the host's registry routing deepseek_v41 here.
//! It is its own test build's root (`zig build conformance` here, `zig build mlx-stream-conformance` in the host), so the
//! last check sees only this suite's device use; like src/tests.zig it re-exports the plugin for the host's registry.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
pub const plugin = @import("root.zig").plugin;
pub const testing = @import("root.zig").testing;



comptime {
    _ = @import("sdk_ext/kinds.zig");
}

test "mlx-stream conformance: the plugin negotiates with this host and registers one macOS-only arch" {
    try sdk.negotiate(plugin, sdk.host);
    try std.testing.expectEqualStrings("mlx-stream", plugin.name);
    try std.testing.expect(plugin.macos_only);
    try std.testing.expect(plugin.provides.arch != null and plugin.provides.source == null and plugin.provides.engine == null);
}

test "mlx-stream conformance: the arch and the kinds it binds decline what is not theirs" {
    const near_misses = [_]sdk.testing.ClaimCase{
        .{ .config = "{}", .want = null },
        .{ .config = "{\"model_type\":\"__no_such_model__\"}", .want = null },
    };
    const arch = comptime sdk.Arch.of(plugin.provides.arch.?);
    try sdk.testing.expectClaims(arch.claims, &near_misses);
    const source = comptime sdk_ext.ExpertSource.of(@import("exl3_source.zig"));
    try sdk.testing.expectClaims(source.claims, &near_misses);
    // a weight group no quant's format describes
    const group_near_misses = [_]sdk.testing.GroupClaimCase{
        .{ .quantization = "{}", .hidden = 5120, .inter = 2304, .want = null },
        .{ .quantization = "{\"mode\":\"__no_such_format__\",\"bits\":4,\"group_size\":32}", .hidden = 5120, .inter = 2304, .want = null },
    };
    const quant = comptime sdk_ext.Quant.of(@import("exl3_quant.zig"));
    try sdk.testing.expectGroupClaims(quant.claims, &group_near_misses);
}

test "mlx-stream conformance: the EXL3 quant is pinned by the kernel registry's manifest" {
    const quant = comptime sdk_ext.Quant.of(@import("exl3_quant.zig"));
    try std.testing.expectEqualStrings("exl3-mul1-k3", quant.name);
    try std.testing.expectEqualStrings(@import("exl3_kernels.zig").manifest_sha256, quant.kernels.?.manifest_sha256);
}

test "mlx-stream conformance: the EXL3 source's capabilities, and the arch claims the one reader for the process" {
    const k = comptime sdk_ext.ExpertSource.of(@import("exl3_source.zig"));
    try std.testing.expectEqualStrings("exl3-stream", k.name);
    try std.testing.expect(k.caps.two_phase and k.caps.transient_release and k.caps.event_gates and !k.caps.construction_reset);
    try std.testing.expect(@import("expert_stream.zig").uses_reader);
    const arch = comptime sdk.Arch.of(plugin.provides.arch.?);
    try std.testing.expect(arch.claim_process != null and arch.release_process != null);
    try arch.claim_process.?();
    try std.testing.expectError(error.ExpertReaderInUse, arch.claim_process.?());
    arch.release_process.?();
    try arch.claim_process.?();
    arch.release_process.?();
}

test "mlx-stream conformance: the host's registry serves deepseek_v41 with this arch; one arch, no tie to break" {
    const plugins = @import("mlx_serve_host").plugins;
    const registry = plugins.registry;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"deepseek_v41\"}");
    const e = registry.arch(&peek, null).?;
    try std.testing.expectEqualStrings("mlx-stream", e.plugin);
    try std.testing.expectEqualStrings(",\"plugins\":[{\"plugin\":\"mlx-stream\",\"kind\":\"arch\",\"name\":\"deepseek_v41\"}]", registry.servedJson(&e.kind));
    try std.testing.expect(!registry.arch_ties_possible and registry.registered("mlx-stream"));
    try std.testing.expect(registry.arch(&peek, "fake-a").? == e);
}

// Declared last so it runs after every other conformance test (the CPU lane's bar).
test "mlx-stream conformance: the CPU lane created no Metal device" {
    try sdk.testing.expectNoDevice();
}
