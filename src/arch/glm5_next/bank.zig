//! GLM-5.3-Flash's routed experts as an NVFP4 expert bank: the shared NVFP4 bank (src/expert/nvfp4_bank.zig: the
//! `nvfp4-gud-9part` layout, its manifest, the opener and the host decoder) at Flash's topology, plus the MTP layer's
//! draft bank (the same layout and manifest form in its own two files: `mtp.zig` holds it resident).

const shared = @import("../../expert/nvfp4_bank.zig");

pub const format = shared.format;
pub const n_components = shared.n_components;
pub const gu_components = shared.gu_components;
pub const routed_top_k = shared.routed_top_k;
pub const group_size = shared.group_size;
pub const bits = shared.bits;
pub const page = shared.page;
pub const scale_line_bytes = shared.scale_line_bytes;
pub const scale_line_floats = shared.scale_line_floats;
pub const scale_array_min_bytes = shared.scale_array_min_bytes;
pub const Records = shared.Records;
pub const Dtype = shared.Dtype;
pub const layout = shared.layout;
pub const Files = shared.Files;
pub const routed_files = shared.routed_files;
pub const mtp_files: Files = .{ .manifest = "mtp-expert-manifest-nvfp4.json", .data = "mtp-experts-nvfp4.bin" };
pub const Component = shared.Component;
pub const Segment = shared.Segment;
pub const mlxDtype = shared.mlxDtype;
pub const Arrays = shared.Arrays;
pub const BankArraysOf = shared.BankArraysOf;
pub const BankArrays = shared.BankArrays;
pub const bankArraysOf = shared.bankArraysOf;
pub const Layer = shared.Layer;
pub const layerOf = shared.layerOf;
pub const SDKBank = shared.SDKBank;
pub const SDK = shared.SDK;
pub const Topology = shared.Topology;
pub const Geometry = shared.Geometry;
pub const Bank = shared.Bank;
pub const inspect = shared.inspect;
pub const e2m1 = shared.e2m1;
pub const e4m3 = shared.e4m3;
pub const invScaleEnc = shared.invScaleEnc;
pub const dequantRow = shared.dequantRow;
pub const toHalf = shared.toHalf;
pub const projections = shared.projections;
pub const Synth = shared.Synth;

/// GLM-5.3-Flash (glm5_next): hidden 4096, moe_intermediate 2048, 288 routed experts, MoE layers 3..44.
pub const flash: Topology = .{ .hidden_size = 4096, .intermediate_size = 2048, .num_experts = 288, .first_moe_layer = 3, .num_moe_layers = 42 };

pub fn requireFlash(g: *const Geometry) !void {
    return g.require(flash);
}

const testing = @import("std").testing;

test "glm5_next nvfp4 bank: Flash's topology is the one the arch takes; another bank's is refused by name" {
    const g = try Geometry.init(4096, 2048, 288, 3, 42);
    try requireFlash(&g);
    try testing.expectEqual(@as(u64, 14_159_872), g.record_bytes);
    const tiny = try Geometry.init(64, 32, 2, 3, 1);
    try testing.expectError(error.UnsupportedTopology, requireFlash(&tiny));
}
