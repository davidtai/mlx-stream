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
/// a format's fused route (`MathOf`'s `F.Fused`): installed once at construction (`install`: the set built on the
/// module's GPU stream, its compile self-check, the launches prepared per wave width), each wave one launch of each
/// text over the three banks (`ids` packed `bank << 24 | slot row`).
pub const FusedDecode = struct {
    installed: decode_kernels.Installed,

    pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, hidden: u32, inter: u32, top_k: u32, limit: f64, diag: *sdk.Diag) !FusedDecode {
        try decode_kernels.checkModel(hidden, inter, top_k, limit, diag);
        return .{ .installed = try decode_kernels.install(a, s, diag) };
    }

    pub fn deinit(f: *FusedDecode) void {
        f.installed.deinit();
    }

    pub fn kernelBanks(banks: *const [decode_kernels.n_banks]bank.BankArrays) [decode_kernels.n_banks]decode_kernels.BankArrays {
        var out: [decode_kernels.n_banks]decode_kernels.BankArrays = undefined;
        for (&out, banks) |*o, b| o.* = .{
            .gate = .{ .w = b.gate.w, .scales = b.gate.scales, .global_scale = b.gate.global_scale },
            .up = .{ .w = b.up.w, .scales = b.up.scales, .global_scale = b.up.global_scale },
            .down = .{ .w = b.down.w, .scales = b.down.scales, .global_scale = b.down.global_scale },
        };
        return out;
    }

    /// x bf16 [P, hidden] (each pair's token row) -> the clamped SwiGLU rows bf16 [P, inter].
    pub fn gateUp(f: *const FusedDecode, g: *ops.MlxOps, x: T, ids: T, banks: *const [decode_kernels.n_banks]bank.BankArrays) !T {
        const kb = kernelBanks(banks);
        return g.adopt(try f.installed.fused.gateUp(x, ids, &kb));
    }

    /// h bf16 [P, inter] -> each pair's down row bf16 [P, hidden].
    pub fn down(f: *const FusedDecode, g: *ops.MlxOps, h: T, ids: T, banks: *const [decode_kernels.n_banks]bank.BankArrays) !T {
        const kb = kernelBanks(banks);
        return g.adopt(try f.installed.fused.down(h, ids, &kb));
    }
};

comptime {
    // Every decode wave the shared scheduler forms (at most one route's ids) has a prepared launch.
    if (decode_kernels.max_pairs < @import("../../sdk_ext.zig").expert.policy.max_route_ids) @compileError("glm5_next: the fused decode texts' widest wave is narrower than the scheduler's route");
}

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


