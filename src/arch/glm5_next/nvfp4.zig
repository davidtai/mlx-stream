//! glm5_next's routed-expert projection format for MiMo's expert math (`mimo_v2/math.zig` `MathOf`): the shared NVFP4
//! format (src/expert/nvfp4_format.zig: the NVFP4 bank's slot arrays and MLX's gather_qmm over them in nvfp4 mode with
//! each slot's global scale) with glm5_next's fused decode texts (`FusedDecode`, decode_kernels.zig: registered at Flash's
//! geometry only).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const bank = @import("bank.zig");
const decode_kernels = @import("decode_kernels.zig");
const shared = @import("../../expert/nvfp4_format.zig");

const T = mlx.mlx_array;

/// `MathOf`'s format: the shared NVFP4 format with the fused decode texts, prompt calls of at most 16,384 rows.
pub const Format = shared.FormatOf(.{ .Fused = FusedDecode, .max_call_tokens = 16_384 });

/// The decode lane's fused NVFP4 texts (decode_kernels.zig, MiMo's k1 family over the NVFP4 bank) as MiMo's math binds
/// a format's fused route (`MathOf`'s `F.Fused`; the shared format's `FusedDecodeOf`): installed once at construction
/// (`install`: the set built on the module's GPU stream, its compile self-check, the launches prepared per wave width),
/// each wave one launch of each text over the three banks (`ids` packed `bank << 24 | slot row`).
pub const FusedDecode = shared.FusedDecodeOf(decode_kernels.Kernels);

/// A projection's slot arrays as the gathers take them (the shared format's check), refused as `Glm5NextSlotArrays`.
pub fn checkArrays(g: *ops.MlxOps, which: []const u8, w: bank.Arrays(T), in: u32, out: u32) !void {
    shared.checkArrays("glm5_next", g, which, w, in, out) catch return error.Glm5NextSlotArrays;
}

pub const scaleShapeOk = shared.scaleShapeOk;

const testing = std.testing;

test "glm5_next nvfp4 format: the fused route's banks are the kernels' (gate, up, down; w, scales, global scale), field for field" {
    const banks: [decode_kernels.n_banks]bank.BankArrays = @splat(.{ .gate = .{ .w = .{ .ctx = @ptrFromInt(0x10) }, .scales = .{ .ctx = @ptrFromInt(0x20) }, .global_scale = .{ .ctx = @ptrFromInt(0x30) } }, .up = .{ .w = .{ .ctx = @ptrFromInt(0x40) }, .scales = .{ .ctx = @ptrFromInt(0x50) }, .global_scale = .{ .ctx = @ptrFromInt(0x60) } }, .down = .{ .w = .{ .ctx = @ptrFromInt(0x70) }, .scales = .{ .ctx = @ptrFromInt(0x80) }, .global_scale = .{ .ctx = @ptrFromInt(0x90) } } });
    const kb = FusedDecode.kernelBanks(&banks);
    for (kb, banks) |k, b| {
        try testing.expectEqual(b.gate.w.ctx, k.gate.w.ctx);
        try testing.expectEqual(b.up.global_scale.ctx, k.up.global_scale.ctx);
        try testing.expectEqual(b.down.scales.ctx, k.down.scales.ctx);
    }
}


