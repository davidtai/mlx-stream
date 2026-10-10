//! mlx-stream's conformance suite (CPU lane): `sdk.testing` over this plugin's declaration, its arch's table, and the
//! EXL3 quant and expert source the arch binds. It is its own test build's root (`zig build conformance` here,
//! `zig build mlx-stream-conformance` in the host), so the last check sees only this suite's device use.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const plugin = @import("root.zig").plugin;
const plugin_mimo = @import("root.zig").plugin_mimo;



comptime {
    _ = @import("sdk_ext/kinds.zig");
}

test "mlx-stream conformance: the plugin negotiates with this host and registers one macOS-only arch" {
    var buf: [64]u8 = undefined;
    try sdk.negotiate(plugin, try sdk.host(&buf));
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
    const source = comptime sdk_ext.ExpertSource.of(@import("exl3/source.zig"));
    try sdk.testing.expectClaims(source.claims, &near_misses);
    // a weight group no quant's format describes
    const group_near_misses = [_]sdk.testing.GroupClaimCase{
        .{ .quantization = "{}", .hidden = 5120, .inter = 2304, .want = null },
        .{ .quantization = "{\"mode\":\"__no_such_format__\",\"bits\":4,\"group_size\":32}", .hidden = 5120, .inter = 2304, .want = null },
    };
    const quant = comptime sdk_ext.Quant.of(@import("exl3/quant.zig"));
    try sdk.testing.expectGroupClaims(quant.claims, &group_near_misses);
}

test "mlx-stream conformance: the EXL3 quant is pinned by the kernel registry's manifest" {
    const quant = comptime sdk_ext.Quant.of(@import("exl3/quant.zig"));
    try std.testing.expectEqualStrings("exl3-mul1-k3", quant.name);
    try std.testing.expectEqualStrings(@import("exl3/kernels.zig").manifest_sha256, quant.kernels.?.manifest_sha256);
}

test "mlx-stream conformance: the EXL3 source's capabilities, and the arch claims the one reader for the process" {
    const k = comptime sdk_ext.ExpertSource.of(@import("exl3/source.zig"));
    try std.testing.expectEqualStrings("exl3-stream", k.name);
    try std.testing.expect(k.caps.two_phase and k.caps.transient_release and k.caps.event_gates and !k.caps.construction_reset);
    try std.testing.expect(@import("expert/stream.zig").uses_reader);
    const arch = comptime sdk.Arch.of(plugin.provides.arch.?);
    try std.testing.expect(arch.claim_process != null and arch.release_process != null);
    try arch.claim_process.?();
    try std.testing.expectError(error.ExpertReaderInUse, arch.claim_process.?());
    arch.release_process.?();
    try arch.claim_process.?();
    arch.release_process.?();
}

test "mlx-stream conformance: MiMo is a second macOS-only arch from the same package" {
    var buf: [64]u8 = undefined;
    try sdk.negotiate(plugin_mimo, try sdk.host(&buf));
    try std.testing.expectEqualStrings("mlx-stream-mimo", plugin_mimo.name);
    try std.testing.expect(plugin_mimo.macos_only);
    try std.testing.expect(plugin_mimo.provides.arch != null and plugin_mimo.provides.source == null and plugin_mimo.provides.engine == null);
    try std.testing.expectEqualStrings("mimo_v2_flash", plugin_mimo.provides.arch.?.name);
}

test "mlx-stream conformance: each arch claims only its own pack" {
    const dsv = comptime sdk.Arch.of(plugin.provides.arch.?);
    const mimo = comptime sdk.Arch.of(plugin_mimo.provides.arch.?);
    const mimo_pack = @embedFile("fixtures/mimo_v2_flash_q4_config.json");
    const dsv_pack = @embedFile("fixtures/dsv41_dspark_mini_config.json");
    // MiMo declines DeepSeek-V4.1 and an upstream (resident) mimo_v2 config with no expert_streaming marker.
    try sdk.testing.expectClaims(mimo.claims, &.{
        .{ .config = mimo_pack, .want = .native },
        .{ .config = dsv_pack, .want = null },
        .{ .config = "{\"model_type\":\"deepseek_v41\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-mxfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"}}", .want = null },
        .{ .config = "{\"model_type\":\"mimo_v2_flash\"}", .want = null },
        .{ .config = "{\"model_type\":\"mimo_v2\"}", .want = null },
    });
    // DeepSeek-V4.1 declines the MiMo pack.
    try sdk.testing.expectClaims(dsv.claims, &.{ .{ .config = mimo_pack, .want = null }, .{ .config = dsv_pack, .want = .native } });
}

test "mlx-stream conformance: both arches share one expert reader claim" {
    const dsv = comptime sdk.Arch.of(plugin.provides.arch.?);
    const mimo = comptime sdk.Arch.of(plugin_mimo.provides.arch.?);
    try dsv.claim_process.?();
    try std.testing.expectError(error.ExpertReaderInUse, mimo.claim_process.?());
    dsv.release_process.?();
    try mimo.claim_process.?();
    try std.testing.expectError(error.ExpertReaderInUse, dsv.claim_process.?());
    mimo.release_process.?();
}

// Declared last so it runs after every other conformance test (the CPU lane's bar).
test "mlx-stream conformance: the CPU lane created no Metal device" {
    try sdk.testing.expectNoDevice();
}
