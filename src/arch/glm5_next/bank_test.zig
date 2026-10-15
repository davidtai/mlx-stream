//! The NVFP4 bank's parity: records written as the converter lays them out, read back through the expert reader into
//! slot rows, and decoded against the values the writer chose. The CPU lane decodes host rows (`bank.dequantRow`); a
//! guarded window (`_GPU_WINDOW_LOCKED`) runs MLX's nvfp4 dequantize over the stream's MLX slot arrays on both streams.
//! `GLM5_NEXT_BANK_VECTORS` names the converter's test vectors (`vectorsOf`).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const bank = @import("bank.zig");
const expert_io = sdk_ext.expert.io;
const S = sdk_ext.expert.stream.StreamOf(bank.SDK, false);
const testing = std.testing;
const Sha256 = std.crypto.hash.sha2.Sha256;

const Component = bank.Component;
const n_components = bank.n_components;

/// The projections' components: (weight, block scales, global scale).
const projections = [3][3]Component{
    .{ .gate_weight, .gate_scales, .gate_global_scale },
    .{ .up_weight, .up_scales, .up_global_scale },
    .{ .down_weight, .down_scales, .down_global_scale },
};

/// A bank of `layers` x `experts` records written into `dir` with its manifest: every E2M1 code and every E4M3 byte
/// but the two NaNs from a seeded generator, positive global scales, the padding zero. `data` is the whole experts.bin.
const Synth = struct {
    geom: bank.Geometry,
    data: []u8,

    fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, hidden: u32, inter: u32, experts: u32, first: u32, layers: u32, seed: u64) !Synth {
        const g = try bank.Geometry.init(hidden, inter, experts, first, layers);
        const data = try a.alloc(u8, @intCast(g.total_bytes));
        errdefer a.free(data);
        @memset(data, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const n_records = @as(usize, experts) * layers;
        for (0..n_records) |i| {
            const rec = data[i * g.record_bytes ..][0..@intCast(g.logical_bytes)];
            for (projections) |p| {
                const w = g.segments[@backingInt(p[0])];
                rand.bytes(rec[@intCast(w.offset)..][0..@intCast(w.length)]);
                const s = g.segments[@backingInt(p[1])];
                for (rec[@intCast(s.offset)..][0..@intCast(s.length)]) |*b| {
                    b.* = rand.int(u8);
                    if (b.* & 0x7f == 0x7f) b.* -= 1;
                }
                const gs = g.segments[@backingInt(p[2])];
                const scale: f32 = 0.01 + rand.float(f32) * 4.0;
                std.mem.writeInt(u32, rec[@intCast(gs.offset)..][0..4], @bitCast(scale), .little);
            }
        }
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(a);
        try text.print(a, "{{\"format\":\"{s}\",\"data_file\":\"experts.bin\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_experts\":{d},\"first_moe_layer\":{d},\"num_moe_layers\":{d},", .{ bank.format, hidden, inter, experts, first, layers });
        try text.print(a, "\"layout\":\"{s}\",\"alignment\":{d},\"record_bytes\":{d},\"record_stride\":{d},\"gate_up_bytes\":{d},\"down_bytes\":{d},\"quantization\":{{\"mode\":\"nvfp4\",\"bits\":4,\"group_size\":16}},", .{ bank.layout, bank.page, g.logical_bytes, g.record_bytes, g.gate_up_bytes, g.logical_bytes - g.gate_up_bytes });
        try text.appendSlice(a, "\"source\":{\"repository\":\"synthetic\",\"revision\":\"0123456789abcdef0123456789abcdef01234567\"},\"parts\":[");
        for (g.segments, 0..) |s, c| {
            try text.print(a, "{s}{{\"name\":\"{s}\",\"dtype\":\"{t}\",\"shape\":[", .{ if (c == 0) "" else ",", @as(Component, @fromBackingInt(@intCast(c))).name(), s.dtype });
            for (s.shape[0..s.rank], 0..) |d, k| try text.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
            try text.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ s.offset, s.length });
        }
        try text.appendSlice(a, "],\"records\":[");
        for (0..n_records) |i| {
            var d: [32]u8 = undefined;
            Sha256.hash(data[i * g.record_bytes ..][0..@intCast(g.logical_bytes)], &d, .{});
            try text.print(a, "{s}{{\"layer\":{d},\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{x}\"}}", .{ if (i == 0) "" else ",", first + i / experts, i % experts, i * g.record_bytes, g.logical_bytes, d });
        }
        var whole: [32]u8 = undefined;
        Sha256.hash(data, &whole, .{});
        try text.print(a, "],\"output_files\":{{\"experts.bin\":{{\"size\":{d},\"sha256\":\"{x}\"}}}}}}", .{ g.total_bytes, whole });
        try dir.writeFile(io, .{ .sub_path = "experts.bin", .data = data });
        try dir.writeFile(io, .{ .sub_path = "expert-manifest.json", .data = text.items });
        return .{ .geom = g, .data = data };
    }

    fn deinit(self: *Synth, a: std.mem.Allocator) void {
        a.free(self.data);
        self.* = undefined;
    }

    fn record(self: *const Synth, layer: u32, expert: u32) []const u8 {
        return self.data[@intCast(self.geom.recordOffset(layer, expert))..][0..@intCast(self.geom.logical_bytes)];
    }
};

/// `refs` (SDK layer, expert) read through one pool job into rows 0.. of `rows`; every ticket settled ok.
fn readRecords(b: *const bank.SDKBank, refs: []const [2]u32, rows: anytype) !void {
    std.debug.assert(refs.len <= expert_io.max_items);
    var gu: [expert_io.max_items]u64 = undefined;
    var down: [expert_io.max_items]u64 = undefined;
    var dests: [expert_io.max_items][n_components]u64 = undefined;
    for (refs, 0..) |r, i| {
        const sp = b.spans(r[0], r[1]);
        gu[i] = sp.gu_offset;
        down[i] = sp.down_offset;
        dests[i] = rows.rowDest(@intCast(i));
    }
    var lens: [n_components]u64 = undefined;
    for (&lens, b.layers[refs[0][0]].segments) |*l, s| l.* = s.length;
    var pool = try expert_io.Pool.start(testing.allocator, .{});
    defer pool.stop();
    const n = refs.len;
    const first = try bank.Records.submit(pool, b.sidecar, gu[0..n], down[0..n], dests[0..n], &lens);
    try pool.wait(first, @intCast(2 * n), 60 * std.time.ns_per_s);
    for (0..2 * n) |t| try testing.expectEqual(expert_io.Status.ok, pool.result(first + @as(u32, @intCast(t))).status);
}

/// An E4M3 byte as f32 from its bits (an f32 exponent built directly; subnormals as m x 2^-9).
fn refE4m3(b: u8) f32 {
    const sign: u32 = @as(u32, b >> 7) << 31;
    const e: u32 = (b >> 3) & 0xf;
    const m: u32 = b & 7;
    if (e == 0) {
        const mag: f32 = @as(f32, @floatFromInt(m)) / 512.0;
        return @bitCast(sign | @as(u32, @bitCast(mag)));
    }
    return @bitCast(sign | ((e + 120) << 23) | (m << 20));
}

/// An E2M1 code as f32 from its fields: exponent 0 holds 0 and 0.5, exponent e > 0 is 2^(e-1) x (1 + m / 2).
fn refE2m1(code: u4) f32 {
    const e: u32 = (code >> 1) & 3;
    const m: f32 = @floatFromInt(code & 1);
    const mag: f32 = if (e == 0) m * 0.5 else std.math.ldexp(1.0 + m * 0.5, @as(i32, @intCast(e)) - 1);
    return if (code & 8 != 0) -mag else mag;
}

/// One projection of a record as rows of the reference's values: e2m1 x (e4m3 x (g / 2688)), in that order.
fn refProjection(a: std.mem.Allocator, rec: []const u8, g: *const bank.Geometry, p: [3]Component) ![]f32 {
    const w = g.segments[@backingInt(p[0])];
    const s = g.segments[@backingInt(p[1])];
    const gs = g.segments[@backingInt(p[2])];
    const rows: usize = @intCast(w.shape[0]);
    const in: usize = @intCast(w.shape[1] * 8);
    const scale_g: f32 = @bitCast(std.mem.readInt(u32, rec[@intCast(gs.offset)..][0..4], .little));
    const out = try a.alloc(f32, rows * in);
    for (0..rows) |r| for (0..in) |j| {
        const byte = rec[@intCast(w.offset)..][r * in / 2 + j / 2];
        const code: u4 = @truncate(if (j % 2 == 0) byte else byte >> 4);
        const sb = rec[@intCast(s.offset)..][r * (in / 16) + j / 16];
        out[r * in + j] = refE2m1(code) * (refE4m3(sb) * (scale_g / 2688.0));
    };
    return out;
}

/// One projection decoded from slot rows (`row(c, i)` of record i) by the plugin's decoder.
fn decodeProjection(a: std.mem.Allocator, rows: anytype, i: u32, g: *const bank.Geometry, p: [3]Component) ![]f32 {
    const w = g.segments[@backingInt(p[0])];
    const out_rows: usize = @intCast(w.shape[0]);
    const words_per_row: usize = @intCast(w.shape[1]);
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(rows.row(p[0], i))));
    const scales = rows.row(p[1], i);
    const scale_g: f32 = @bitCast(std.mem.readInt(u32, rows.row(p[2], i)[0..4], .little));
    const out = try a.alloc(f32, out_rows * words_per_row * 8);
    for (0..out_rows) |r| bank.dequantRow(words[r * words_per_row ..][0..words_per_row], scales[r * words_per_row / 2 ..][0 .. words_per_row / 2], scale_g, out[r * words_per_row * 8 ..][0 .. words_per_row * 8]);
    return out;
}

fn expectBitsEqual(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |x, y, k| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("value {d}: want {e} ({x}) got {e} ({x})\n", .{ k, x, @as(u32, @bitCast(x)), y, @as(u32, @bitCast(y)) });
            return error.TestExpectedEqual;
        }
    }
}

/// sha256 of `values` as little-endian f16 (MLX's astype to float16) and as f32.
fn checksums(values: []const f32) struct { f16: [32]u8, f32: [32]u8 } {
    var h16 = Sha256.init(.{});
    var h32 = Sha256.init(.{});
    for (values) |v| {
        var b16: [2]u8 = undefined;
        std.mem.writeInt(u16, &b16, @bitCast(bank.toHalf(v)), .little);
        h16.update(&b16);
        var b32: [4]u8 = undefined;
        std.mem.writeInt(u32, &b32, @bitCast(v), .little);
        h32.update(&b32);
    }
    return .{ .f16 = h16.finalResult(), .f32 = h32.finalResult() };
}

test "glm5_next nvfp4 bank: records through the reader into host slot rows, byte for byte, and decoded as the writer's values" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var synth = try Synth.write(a, io, tmp.dir, 64, 32, 3, 3, 2, 0x5eed);
    defer synth.deinit(a);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var owner = try bank.Bank.open(a, io, dir);
    defer owner.deinit();
    try testing.expectEqual(synth.geom.total_bytes, owner.sidecar_file_size);
    var b = try bank.SDKBank.init(a, &owner);
    defer b.deinit();
    // Every record of both layers, one job: the second layer's records start past the first's padding.
    const refs = [_][2]u32{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 1, 0 }, .{ 1, 1 }, .{ 1, 2 } };
    var rows = try S.HostSlotRows.init(&b.layers[0], refs.len);
    defer rows.deinit();
    try readRecords(&b, &refs, &rows);
    var decoded: usize = 0;
    for (refs, 0..) |r, i| {
        const rec = synth.record(synth.geom.first_moe_layer + r[0], r[1]);
        for (synth.geom.segments, 0..) |s, c| {
            // The record's bytes at the row's start (a global scale's row is its own 128-byte line; the rest is never
            // written or read).
            const row = rows.row(@fromBackingInt(@intCast(c)), @intCast(i));
            try testing.expectEqual(S.rowBytes(&b.layers[0], c), row.len);
            try testing.expectEqualSlices(u8, rec[@intCast(s.offset)..][0..@intCast(s.length)], row[0..@intCast(s.length)]);
        }
        for (projections) |p| {
            const want = try refProjection(a, rec, &synth.geom, p);
            defer a.free(want);
            const got = try decodeProjection(a, &rows, @intCast(i), &synth.geom, p);
            defer a.free(got);
            try expectBitsEqual(want, got);
            decoded += got.len;
        }
    }
    try testing.expectEqual(@as(usize, 6 * 3 * 64 * 32), decoded);
    // With a window the MLX tests may have made the Metal device in this process.
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) try sdk.testing.expectNoDevice();
}

test "glm5_next nvfp4 bank: Flash's spans read whole through the reader's default staging, and the opener refuses a short bank" {
    const g = try bank.Geometry.init(4096, 2048, 288, 3, 42);
    // The reader stages a gate/up span or a down span whole, page-aligned with a page to spare.
    const staging = (expert_io.Options{}).staging_bytes;
    try testing.expect(std.mem.alignForward(u64, g.gate_up_bytes, bank.page) + bank.page > staging);
    try testing.expect(std.mem.alignForward(u64, g.logical_bytes - g.gate_up_bytes, bank.page) + bank.page <= staging);
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var synth = try Synth.write(a, io, tmp.dir, 64, 32, 2, 3, 1, 7);
    defer synth.deinit(a);
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.bin", .data = synth.data[0 .. synth.data.len - 1] });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try testing.expectError(error.SidecarGeometry, bank.Bank.open(a, io, dir));
    const geom = try bank.inspect(a, io, dir);
    try testing.expectEqual(@as(u32, 64), geom.hidden_size);
}

/// The converter's test vectors (`GLM5_NEXT_BANK_VECTORS`): a bank directory (expert-manifest.json, experts.bin) and
/// `reference.json`, `{"records": [{"layer", "expert", "gate" | "up" | "down": {"f32_sha256", "fp16_sha256"}}]}`: the
/// reference reader's values of each projection, row-major [out, in], little-endian.
const Vectors = struct {
    records: []const struct {
        layer: u32,
        expert: u32,
        gate: Sums,
        up: Sums,
        down: Sums,
    },
    const Sums = struct { f32_sha256: []const u8, fp16_sha256: []const u8 };
};

fn hexEq(hex: []const u8, d: *const [32]u8) bool {
    var got: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&got, hex) catch return false;
    return hex.len == 64 and std.mem.eql(u8, &got, d);
}

test "glm5_next nvfp4 bank: the converter's test vectors decode to the reference reader's values (f32 and fp16 sha256)" {
    const dir = std.mem.span(std.c.getenv("GLM5_NEXT_BANK_VECTORS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var owner = try bank.Bank.open(a, io, dir);
    defer owner.deinit();
    var b = try bank.SDKBank.init(a, &owner);
    defer b.deinit();
    const path = try std.fs.path.join(a, &.{ dir, "reference.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Vectors, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const g = &owner.geometry;
    for (parsed.value.records) |r| {
        var rows = try S.HostSlotRows.init(&b.layers[r.layer - g.first_moe_layer], 1);
        defer rows.deinit();
        try readRecords(&b, &.{.{ r.layer - g.first_moe_layer, r.expert }}, &rows);
        for (projections, [_]Vectors.Sums{ r.gate, r.up, r.down }, [_][]const u8{ "gate", "up", "down" }) |p, want, which| {
            const got = try decodeProjection(a, &rows, 0, g, p);
            defer a.free(got);
            const sums = checksums(got);
            testing.expect(hexEq(want.f32_sha256, &sums.f32) and hexEq(want.fp16_sha256, &sums.f16)) catch |e| {
                std.debug.print("layer {d} expert {d} {s}: f32 {x} fp16 {x}\n", .{ r.layer, r.expert, which, sums.f32, sums.f16 });
                return e;
            };
        }
    }
    std.debug.print("glm5_next nvfp4 vectors: {d} records match the reference reader\n", .{parsed.value.records.len});
}

test "glm5_next nvfp4 bank (window): the stream's MLX slot arrays dequantize in MLX as the host decoder, bit for bit, on both streams" {
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var synth = try Synth.write(a, io, tmp.dir, 64, 32, 3, 3, 1, 0x9e3);
    defer synth.deinit(a);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var owner = try bank.Bank.open(a, io, dir);
    defer owner.deinit();
    var b = try bank.SDKBank.init(a, &owner);
    defer b.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const gpu = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(gpu);
    var slots = try S.LayerSlotBank.init(&b.layers[0], 3, cpu);
    defer slots.deinit();
    const refs = [_][2]u32{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 } };
    try readRecords(&b, &refs, &slots);
    const arrays = bank.bankArraysOf(slots.arrays);
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(arrays.gate.global_scale));
    // Line-isolated: [32, 32] f32 (3 slots floored at one page), each slot's word at column 0 of its row.
    try testing.expectEqual(@as(usize, 2), mlx.mlx_array_ndim(arrays.gate.global_scale));
    try testing.expectEqual(@as(c_int, bank.scale_line_floats), mlx.mlx_array_shape(arrays.gate.global_scale)[1]);
    try testing.expectEqual(@as(c_int, bank.scale_array_min_bytes / bank.scale_line_bytes), mlx.mlx_array_shape(arrays.gate.global_scale)[0]);
    for ([_]mlx.mlx_stream{ cpu, gpu }) |s| for (refs, 0..) |r, i| {
        const rec = synth.record(synth.geom.first_moe_layer + r[0], r[1]);
        for (projections, [_]bank.Arrays(mlx.mlx_array){ arrays.gate, arrays.up, arrays.down }) |p, pa| {
            const want = try refProjection(a, rec, &synth.geom, p);
            defer a.free(want);
            const got = try mlxDequant(a, pa, @intCast(i), s);
            defer a.free(got);
            try expectBitsEqual(want, got);
        }
    };
}

/// Row `i` of a projection's slot arrays through MLX's nvfp4 dequantize on `s` (its global scale a scalar), as f32.
fn mlxDequant(a: std.mem.Allocator, pa: bank.Arrays(mlx.mlx_array), i: i32, s: mlx.mlx_stream) ![]f32 {
    var arrs: [6]mlx.mlx_array = @splat(mlx.mlx_array_new());
    defer for (arrs) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const sw = mlx.mlx_array_shape(pa.w);
    const ss = mlx.mlx_array_shape(pa.scales);
    try mlx.check(mlx.mlx_slice(&arrs[0], pa.w, &[_]c_int{ i, 0, 0 }, 3, &[_]c_int{ i + 1, sw[1], sw[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_reshape(&arrs[1], arrs[0], &[_]c_int{ sw[1], sw[2] }, 2, s));
    try mlx.check(mlx.mlx_slice(&arrs[2], pa.scales, &[_]c_int{ i, 0, 0 }, 3, &[_]c_int{ i + 1, ss[1], ss[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_reshape(&arrs[3], arrs[2], &[_]c_int{ ss[1], ss[2] }, 2, s));
    try mlx.check(mlx.mlx_slice(&arrs[4], pa.global_scale, &[_]c_int{ i, 0 }, 2, &[_]c_int{ i + 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var gs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gs);
    try mlx.check(mlx.mlx_reshape(&gs, arrs[4], &[_]c_int{}, 0, s));
    try mlx.check(mlx.mlx_dequantize(&arrs[5], arrs[1], arrs[3], .{}, .{ .value = bank.group_size, .has_value = true }, .{ .value = bank.bits, .has_value = true }, "nvfp4", gs, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(arrs[5]));
    const n: usize = @intCast(sw[1] * sw[2] * 8);
    const data = mlx.mlx_array_data_float32(arrs[5]) orelse return error.MlxNoData;
    return a.dupe(f32, data[0..n]);
}

// ── Scale-line isolation (bank.zig `scale_line_bytes`): construction geometry, CPU only ──

/// Flash's SDK layer (one routed layer's segments).
fn flashLayer() !bank.Layer {
    const g = try bank.Geometry.init(4096, 2048, 288, 3, 42);
    return .{ .logical_bytes = g.logical_bytes, .base_offset = 0, .segments = g.segments };
}

test "glm5_next nvfp4 bank: scale lines isolated: each slot's global scale owns a 128-byte line, every other row is whole lines" {
    const l = try flashLayer();
    try testing.expectEqual(@as(u64, 128), S.slot_line_bytes);
    try testing.expectEqual(@as(u64, bank.page), S.slot_min_bytes);
    for (l.segments, 0..) |seg, c| {
        const comp: Component = @fromBackingInt(@intCast(c));
        const scalar = comp == .gate_global_scale or comp == .up_global_scale or comp == .down_global_scale;
        try testing.expectEqual(scalar, S.padded(&l, c));
        try testing.expectEqual(@as(u64, 0), S.rowBytes(&l, c) % bank.scale_line_bytes);
        try testing.expectEqual(if (scalar) @as(u64, bank.scale_line_bytes) else seg.length, S.rowBytes(&l, c));
    }
    // The slot row (every component's row) is the bank's `slotRowBytes`, within the record stride (the bill's per_row).
    var row: u64 = 0;
    for (0..n_components) |c| row += S.rowBytes(&l, c);
    const fg = try bank.Geometry.init(4096, 2048, 288, 3, 42);
    try testing.expectEqual(fg.slotRowBytes(), row);
    try testing.expectEqual(@as(u64, 14_155_776 + 3 * 128), row);
    try testing.expect(row <= 14_159_872);
}

test "glm5_next nvfp4 bank: scale lines isolated: the arrays' shapes, a page at least, no line shared by two slots" {
    const l = try flashLayer();
    const g_comps = [_]usize{ @backingInt(Component.gate_global_scale), @backingInt(Component.up_global_scale), @backingInt(Component.down_global_scale) };
    // ext banks of 1..7 rows (lr2), window 0, the base rows at 115/131, the MTP bank's 288.
    for ([_]u32{ 1, 2, 3, 7, 8, 31, 32, 33, 64, 115, 131, 288 }) |rows| {
        for (g_comps) |c| {
            var shape: [4]c_int = undefined;
            try testing.expectEqual(@as(usize, 2), try S.arrayShape(&l, c, rows, &shape));
            try testing.expectEqual(@as(c_int, @intCast(@max(rows, 32))), shape[0]);
            try testing.expectEqual(@as(c_int, bank.scale_line_floats), shape[1]);
            const bytes = S.arrayBytes(&l, c, rows);
            try testing.expectEqual(@as(u64, @intCast(shape[0])) * bank.scale_line_bytes, bytes);
            // Never MLX's small-buffer heap (< 256 B) and never `constant`-class (< 8 elements): a page at least.
            try testing.expect(bytes >= bank.scale_array_min_bytes and bytes >= 256 and @as(u64, @intCast(shape[0] * shape[1])) >= 8);
        }
        var shape: [4]c_int = undefined;
        try testing.expectEqual(@as(usize, 3), try S.arrayShape(&l, @backingInt(Component.gate_weight), rows, &shape));
        try testing.expectEqualSlices(c_int, &.{ @intCast(rows), 2048, 512 }, shape[0..3]);
        try testing.expectEqual(@as(u64, rows) * 4_194_304, S.arrayBytes(&l, @backingInt(Component.gate_weight), rows));
    }
    // Host rows: every slot's global-scale row starts on its own 128-byte line, and the pool's destinations (rowDest)
    // of two slots never fall in one line.
    var rows = try S.HostSlotRows.init(&l, 40);
    defer rows.deinit();
    for (g_comps) |c| {
        var lines = std.AutoHashMap(u64, u32).init(testing.allocator);
        defer lines.deinit();
        for (0..40) |r| {
            const d = rows.rowDest(@intCast(r))[c];
            try testing.expectEqual(@as(u64, 0), d % bank.scale_line_bytes);
            const gop = try lines.getOrPut(d / bank.scale_line_bytes);
            try testing.expect(!gop.found_existing);
            gop.value_ptr.* = @intCast(r);
            // The word's 4 bytes stay inside the line.
            try testing.expectEqual(d / bank.scale_line_bytes, (d + 3) / bank.scale_line_bytes);
        }
    }
}

test "glm5_next nvfp4 bank: scale lines isolated: a padded component that is not one scalar per id is refused at construction" {
    var l = try flashLayer();
    // A block-scale row of 100 bytes would need padding inside a rank-2 row: no layout keeps it whole, refused by name.
    l.segments[@backingInt(Component.gate_scales)].length = 100;
    var shape: [4]c_int = undefined;
    try testing.expect(S.padded(&l, @backingInt(Component.gate_scales)));
    try testing.expectError(error.SlotLineGeometry, S.arrayShape(&l, @backingInt(Component.gate_scales), 4, &shape));
}
