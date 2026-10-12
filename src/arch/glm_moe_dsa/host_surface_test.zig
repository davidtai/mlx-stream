//! GLM-5.3 on a host without its surface (`host_caps.glm` false: tests.zig compiles this file only there, so a host
//! with the surface counts none of it): the arch still claims its model type, and the parse and the construction refuse
//! by name before any work.

const std = @import("std");
const sdk = @import("sdk");
const plugin = @import("plugin.zig");
const testing = std.testing;

test "glm plugin: on a host without the GLM-5.3 surface the parse and the construction refuse by name (GlmHostSurfaceMissing)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"glm_moe_dsa\"}");
    try testing.expectEqual(@as(?sdk.Priority, .native), plugin.claims(&peek));
    var diag: sdk.Diag = .{};
    try testing.expectError(error.GlmHostSurfaceMissing, plugin.parse(testing.allocator, &peek, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "mlx_host.sushi_exl3") != null);
    try testing.expect(std.mem.indexOf(u8, diag.message(), "GlmHostSurfaceMissing") != null);
    // The construction refuses before it reads its arguments (no config of this arch exists here); the module type is
    // never formatted (that would resolve it, and with it the host surface this host lacks).
    const load: sdk.LoadCtx = undefined;
    const c: plugin.Config = .{};
    if (plugin.init(&load, &c)) |_| return error.TestUnexpectedResult else |e| try testing.expectEqual(error.GlmHostSurfaceMissing, e);
}
