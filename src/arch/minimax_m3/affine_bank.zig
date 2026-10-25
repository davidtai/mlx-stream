const std = @import("std");
const shared = @import("../glm_moe_dsa/bank.zig");
const sdk = @import("sdk");
const bank_file = @import("../../sdk_ext.zig").expert.bank_file;
pub const format = "mlx-stream-affine-expert-bank-v1";
pub const layout = "affine-gud-9part";
pub const n_components = shared.n_components;
pub const gu_components = shared.gu_components;
pub const routed_top_k = 4;
pub const bits = 3;
pub const group_size = 64;
pub const Component = shared.Component;
pub const Records = shared.Records;
pub const Layer = shared.Layer;
pub const Arrays = shared.Arrays;
pub const BankArrays = shared.BankArrays;
pub const bankArraysOf = shared.bankArraysOf;
pub const mlxDtype = shared.mlxDtype;
pub const slot_array_min_bytes = std.heap.page_size_max;
pub const requireFlash = @import("bank.zig").requireFlash;
const part_names = [_][]const u8{ "gate_proj.weight", "gate_proj.scales", "gate_proj.biases", "up_proj.weight", "up_proj.scales", "up_proj.biases", "down_proj.weight", "down_proj.scales", "down_proj.biases" };

pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    logical_bytes: u64,
    record_bytes: u64,
    gate_up_bytes: u64,
    total_bytes: u64,
    segments: [n_components]shared.Segment,
    pub fn init(hidden: u64, inter: u64, experts: u64, first: u64, count: u64) !Geometry {
        if (hidden == 0 or inter == 0 or hidden > 65536 or inter > 65536 or experts == 0 or experts > 65535 or count == 0 or count > 4096 or first > 4096 or first + count > 4096) return error.MiniMaxAffineGeometry;
        const segments = shared.layerSegments(bits, hidden, inter) orelse return error.MiniMaxAffineGeometry;
        const logical = shared.logicalBytes(&segments);
        const stride = shared.recordBytes(logical);
        return .{ .hidden_size = @intCast(hidden), .intermediate_size = @intCast(inter), .num_experts = @intCast(experts), .first_moe_layer = @intCast(first), .num_moe_layers = @intCast(count), .logical_bytes = logical, .record_bytes = stride, .gate_up_bytes = segments[gu_components].offset, .total_bytes = try std.math.mul(u64, try std.math.mul(u64, experts, count), stride), .segments = segments };
    }
    pub fn slotRowBytes(self: *const Geometry) u64 {
        return self.logical_bytes;
    }
};

const Manifest = struct {
    format: []const u8,
    data_file: []const u8,
    layout: []const u8,
    alignment: u64,
    hidden_size: u64,
    intermediate_size: u64,
    num_experts: u64,
    first_moe_layer: u64,
    num_moe_layers: u64,
    record_bytes: u64,
    record_stride: u64,
    gate_up_bytes: u64,
    down_bytes: u64,
    quantization: struct { mode: []const u8, bits: u32, group_size: u32 },
    source: struct { repository: []const u8, revision: []const u8 },
    parts: []const struct { name: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 },
    records: []const struct { layer: u64, expert: u64, offset: u64, length: u64, sha256: []const u8 },
    output_files: std.json.Value,
};

pub const Bank = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    digests: [][32]u8 = &.{},
    sidecar_fd: std.c.fd_t = -1,
    sidecar_file_size: u64 = 0,
    pub fn fromText(a: std.mem.Allocator, text: []const u8) !Bank {
        const parsed = try std.json.parseFromSlice(Manifest, a, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        return adopt(a, parsed.value);
    }
    fn adopt(a: std.mem.Allocator, manifest: Manifest) !Bank {
        const m = manifest;
        if (!std.mem.eql(u8, m.format, format) or !std.mem.eql(u8, m.layout, layout) or !std.mem.eql(u8, m.data_file, "experts.bin") or m.alignment != 4096) return error.MiniMaxAffineFormat;
        if (!std.mem.eql(u8, m.quantization.mode, "affine") or m.quantization.bits != bits or m.quantization.group_size != group_size) return error.MiniMaxAffineQuantization;
        const g = try Geometry.init(m.hidden_size, m.intermediate_size, m.num_experts, m.first_moe_layer, m.num_moe_layers);
        if (m.record_bytes != g.logical_bytes or m.record_stride != g.record_bytes or m.gate_up_bytes != g.gate_up_bytes or m.down_bytes != g.logical_bytes - g.gate_up_bytes) return error.MiniMaxAffineGeometry;
        if (m.parts.len != n_components) return error.MiniMaxAffineParts;
        for (m.parts, g.segments, part_names) |part, expected, name| {
            if (!std.mem.eql(u8, part.name, name) or !std.mem.eql(u8, part.dtype, @tagName(expected.dtype)) or !std.mem.eql(u64, part.shape, expected.shape[0..expected.rank]) or part.offset != expected.offset or part.length != expected.length) return error.MiniMaxAffineParts;
        }
        if (m.source.repository.len == 0 or m.source.revision.len != 40) return error.MiniMaxAffineSource;
        var revision: [20]u8 = undefined;
        _ = std.fmt.hexToBytes(&revision, m.source.revision) catch return error.MiniMaxAffineSource;
        if (m.records.len != @as(u64, g.num_moe_layers) * g.num_experts) return error.MiniMaxAffineRecords;
        var out: Bank = .{ .allocator = a, .geometry = g };
        errdefer out.deinit();
        out.digests = try a.alloc([32]u8, m.records.len);
        for (m.records, out.digests, 0..) |record, *digest, i| {
            if (record.layer != g.first_moe_layer + i / g.num_experts or record.expert != i % g.num_experts or record.offset != @as(u64, i) * g.record_bytes or record.length != g.logical_bytes or record.sha256.len != 64) return error.MiniMaxAffineRecords;
            _ = std.fmt.hexToBytes(digest, record.sha256) catch return error.MiniMaxAffineRecords;
        }
        if (m.output_files != .object) return error.MiniMaxAffineOutput;
        const file = m.output_files.object.get("experts.bin") orelse return error.MiniMaxAffineOutput;
        if (file != .object) return error.MiniMaxAffineOutput;
        const size = file.object.get("size") orelse return error.MiniMaxAffineOutput;
        const sha = file.object.get("sha256") orelse return error.MiniMaxAffineOutput;
        if (size != .integer or size.integer < 0 or size.integer != g.total_bytes or sha != .string or sha.string.len != 64) return error.MiniMaxAffineOutput;
        var digest: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&digest, sha.string) catch return error.MiniMaxAffineOutput;
        return out;
    }
    fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
        if (!std.fs.path.isAbsolute(dir)) return error.BankDirNotAbsolute;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const manifest = try bank_file.readManifest(Manifest, arena.allocator(), io, dir, "expert-manifest.json", null);
        return adopt(a, manifest);
    }
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
        var out = try read(a, io, dir);
        errdefer out.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
        defer a.free(path);
        const sidecar = try bank_file.openSidecar(path, out.geometry.total_bytes, .exact, null);
        out.sidecar_fd = sidecar.fd;
        out.sidecar_file_size = sidecar.size;
        return out;
    }
    pub fn deinit(self: *Bank) void {
        if (self.sidecar_fd >= 0) _ = std.c.close(self.sidecar_fd);
        self.allocator.free(self.digests);
        self.* = undefined;
    }
};
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    var bank = try Bank.read(a, io, dir);
    defer bank.deinit();
    return bank.geometry;
}
pub const SDKBank = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    n_experts: u32,
    first_moe_layer: u32,
    record_bytes: u64,
    sidecar: @import("../../sdk_ext.zig").expert.UncachedFd,
    pub fn init(a: std.mem.Allocator, owner: *const Bank) !SDKBank {
        const g = &owner.geometry;
        const entries = try a.alloc(Layer, g.num_moe_layers);
        for (entries, 0..) |*entry, index| entry.* = .{ .model_layer = @intCast(g.first_moe_layer + index), .record_bytes = g.record_bytes, .logical_bytes = g.logical_bytes, .base_offset = index * @as(u64, g.num_experts) * g.record_bytes, .segments = g.segments };
        return .{ .allocator = a, .layers = entries, .n_experts = g.num_experts, .first_moe_layer = g.first_moe_layer, .record_bytes = g.record_bytes, .sidecar = .{ .fd = owner.sidecar_fd, .size = owner.sidecar_file_size } };
    }
    pub fn deinit(self: *SDKBank) void {
        self.allocator.free(self.layers);
        self.* = undefined;
    }
    pub fn spans(self: *const SDKBank, layer: u32, expert: u32) struct { gu_offset: u64, down_offset: u64 } {
        const offset = self.recordOffset(layer, expert);
        return .{ .gu_offset = offset, .down_offset = offset + self.layers[layer].segments[gu_components].offset };
    }
    pub fn recordOffset(self: *const SDKBank, layer: u32, expert: u32) u64 {
        return self.layers[layer].base_offset + @as(u64, expert) * self.record_bytes;
    }
};
pub const SDK = struct {
    pub const n_components = shared.n_components;
    pub const gu_components = shared.gu_components;
    pub const Component = shared.Component;
    pub const Layer = shared.Layer;
    pub const Records = shared.Records;
    pub const Bank = SDKBank;
    pub const BankArrays = shared.BankArrays;
    pub const bankArraysOf = shared.bankArraysOf;
    pub const mlxDtype = shared.mlxDtype;
    pub const routed_top_k = 4;
    pub const slot_array_min_bytes = std.heap.page_size_max;
};

test "MiniMax affine bank geometry preserves source three-bit weights scales and biases" {
    const geometry = try Geometry.init(6144, 3072, 128, 3, 57);
    try std.testing.expectEqual(@as(u64, 24_772_608), geometry.logical_bytes);
    try std.testing.expectEqual(@as(u64, 16_515_072), geometry.gate_up_bytes);
    try std.testing.expectEqual(@as(u64, 180_740_947_968), geometry.total_bytes);
    try std.testing.expectEqual(@as(u64, 576), geometry.segments[0].shape[1]);
    try std.testing.expectEqual(shared.Dtype.BF16, geometry.segments[2].dtype);
    try std.testing.expectEqual(geometry.logical_bytes, geometry.slotRowBytes());
    try std.testing.expectError(error.MiniMaxAffineGeometry, Geometry.init(96, 64, 2, 3, 1));
}

test "MiniMax affine bank validates parts record ordering and aligned sidecar size" {
    const a = std.testing.allocator;
    var bank = try Bank.fromText(a, fixture);
    defer bank.deinit();
    try std.testing.expectEqual(@as(usize, 2), bank.digests.len);
    try std.testing.expectEqual(@as(u64, 16384), bank.geometry.total_bytes);
    const bad_stride = try std.mem.replaceOwned(u8, a, fixture, "\"record_stride\":8192", "\"record_stride\":4096");
    defer a.free(bad_stride);
    try std.testing.expectError(error.MiniMaxAffineGeometry, Bank.fromText(a, bad_stride));
    const bad_part = try std.mem.replaceOwned(u8, a, fixture, "\"BF16\"", "\"U8\"");
    defer a.free(bad_part);
    try std.testing.expectError(error.MiniMaxAffineParts, Bank.fromText(a, bad_part));
    const duplicate = try std.mem.replaceOwned(u8, a, fixture, "\"expert\":1", "\"expert\":0");
    defer a.free(duplicate);
    try std.testing.expectError(error.MiniMaxAffineRecords, Bank.fromText(a, duplicate));
}

const fixture =
    \\{"format":"mlx-stream-affine-expert-bank-v1","layout":"affine-gud-9part","data_file":"experts.bin","alignment":4096,
    \\"hidden_size":64,"intermediate_size":64,"num_experts":2,"first_moe_layer":3,"num_moe_layers":1,
    \\"record_bytes":5376,"record_stride":8192,"gate_up_bytes":3584,"down_bytes":1792,
    \\"quantization":{"mode":"affine","bits":3,"group_size":64},"source":{"repository":"fixture","revision":"0000000000000000000000000000000000000000"},
    \\"parts":[{"name":"gate_proj.weight","dtype":"U32","shape":[64,6],"offset":0,"length":1536},
    \\{"name":"gate_proj.scales","dtype":"BF16","shape":[64,1],"offset":1536,"length":128},
    \\{"name":"gate_proj.biases","dtype":"BF16","shape":[64,1],"offset":1664,"length":128},
    \\{"name":"up_proj.weight","dtype":"U32","shape":[64,6],"offset":1792,"length":1536},
    \\{"name":"up_proj.scales","dtype":"BF16","shape":[64,1],"offset":3328,"length":128},
    \\{"name":"up_proj.biases","dtype":"BF16","shape":[64,1],"offset":3456,"length":128},
    \\{"name":"down_proj.weight","dtype":"U32","shape":[64,6],"offset":3584,"length":1536},
    \\{"name":"down_proj.scales","dtype":"BF16","shape":[64,1],"offset":5120,"length":128},
    \\{"name":"down_proj.biases","dtype":"BF16","shape":[64,1],"offset":5248,"length":128}],
    \\"records":[{"layer":3,"expert":0,"offset":0,"length":5376,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"},
    \\{"layer":3,"expert":1,"offset":8192,"length":5376,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}],
    \\"output_files":{"experts.bin":{"size":16384,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}}
;
