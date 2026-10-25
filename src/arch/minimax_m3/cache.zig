//! Head-major capacity KV/index storage. Three dense or six Q8 owning handles
//! survive a layer wave. Valid-row views must be released before the next write
//! so MLX can donate the buffer rather than copy its full capacity.
const std = @import("std");
const common = @import("../deepseek_v41/ops.zig");
pub const quantum: u32 = 256;

pub fn paddedCapacity(limit: u32) u32 {
    return @intCast(@min(std.mem.alignForward(u64, limit, quantum), std.math.maxInt(c_int)));
}

pub fn nextCapacity(old: u32, need: u32, limit: u32) !u32 {
    if (need == 0 or need > limit or limit > std.math.maxInt(c_int)) return error.MiniMaxCacheLimit;
    if (old >= need) return old;
    const geometric = @as(u64, old) + old / 4;
    const wanted = std.mem.alignForward(u64, @max(need, geometric), quantum);
    return @intCast(@min(wanted, paddedCapacity(limit)));
}

pub fn CacheOf(comptime G: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        pub const View = struct { k: T, v: T, index_k: T, k_scales: ?T = null, v_scales: ?T = null, index_scales: ?T = null, bits: u8 = 0 };
        k: ?T = null,
        v: ?T = null,
        index_k: ?T = null,
        k_scales: ?T = null,
        v_scales: ?T = null,
        index_scales: ?T = null,
        bits: u8 = 0,
        len: u32 = 0,
        capacity: u32 = 0,
        limit: u32 = 1048576,

        pub fn clear(self: *Self, g: *G) void {
            if (self.k) |x| g.release(x);
            if (self.v) |x| g.release(x);
            if (self.index_k) |x| g.release(x);
            if (self.k_scales) |x| g.release(x);
            if (self.v_scales) |x| g.release(x);
            if (self.index_scales) |x| g.release(x);
            self.* = .{ .limit = self.limit, .bits = self.bits };
        }

        /// Views are wave-tracked, never retained by the cache. A shorter prefix
        /// lets each query of a prefill chunk attend its own causal history.
        pub fn view(self: *const Self, g: *G, rows: u32) !View {
            if (rows == 0 or rows > self.len) return error.MiniMaxCacheRows;
            return .{ .k = try prefix(g, self.k.?, rows), .v = try prefix(g, self.v.?, rows), .index_k = try prefix(g, self.index_k.?, rows), .bits = self.bits, .k_scales = if (self.k_scales) |x| try prefix(g, x, rows) else null, .v_scales = if (self.v_scales) |x| try prefix(g, x, rows) else null, .index_scales = if (self.index_scales) |x| try prefix(g, x, rows) else null };
        }

        pub fn append(self: *Self, g: *G, k: T, v: T, index_k: T) !View {
            try self.write(g, k, v, index_k);
            return self.view(g, self.len);
        }

        fn prefix(g: *G, x: T, rows: u32) !T {
            const sh = g.shapeOf(x);
            return g.slice(x, &.{ 0, 0, 0, 0 }, &.{ 1, sh.d[1], @intCast(rows), sh.d[3] }, &.{ 1, 1, 1, 1 });
        }

        fn update(g: *G, old: ?T, x: T, used: u32, old_capacity: u32, capacity: u32, start: T, zero_start: T) !T {
            const sh = g.shapeOf(x);
            var source: T = undefined;
            const grows = capacity != old_capacity;
            if (grows) {
                source = try g.zeros(&.{ 1, sh.d[1], @intCast(capacity), sh.d[3] }, g.dtypeOf(x));
                if (old) |previous| {
                    const retained = try prefix(g, previous, used);
                    const copied = try g.sliceUpdateDyn(source, retained, zero_start);
                    g.drop(source);
                    g.drop(retained);
                    source = copied;
                }
            } else source = old.?;
            const result = try g.sliceUpdateDyn(source, x, start);
            if (grows) g.drop(source);
            return result;
        }

        fn write(self: *Self, g: *G, k: T, v: T, index_k: T) !void {
            const ks = g.shapeOf(k);
            const vs = g.shapeOf(v);
            const is = g.shapeOf(index_k);
            if (ks.n != 4 or is.n != 4 or ks.d[0] != 1 or is.d[0] != 1 or is.d[1] != 1 or ks.d[2] < 1 or !std.mem.eql(c_int, ks.slice(), vs.slice()) or is.d[2] != ks.d[2] or g.dtypeOf(k) != g.dtypeOf(v) or g.dtypeOf(k) != g.dtypeOf(index_k)) return error.MiniMaxCacheShape;
            if (self.bits != 0 and self.bits != 8) return error.MiniMaxKvPrecisionUnsupported;
            if (self.bits == 8 and (ks.d[3] != 128 or is.d[3] != 128)) return error.MiniMaxCacheShape;
            if (self.k) |old| {
                const sh = g.shapeOf(old);
                const ih = g.shapeOf(self.index_k.?);
                if (sh.d[1] != ks.d[1] or sh.d[3] != ks.d[3] or ih.d[3] != is.d[3] or g.dtypeOf(old) != (if (self.bits == 8) .int8 else g.dtypeOf(k))) return error.MiniMaxCacheShape;
            }
            const count: u32 = @intCast(ks.d[2]);
            const need = std.math.add(u32, self.len, count) catch return error.MiniMaxCacheLimit;
            const capacity = try nextCapacity(self.capacity, need, self.limit);
            const mark = g.mark();
            defer g.resetTo(mark);
            const offset: [4]i32 = .{ 0, 0, @intCast(self.len), 0 };
            const zero: [4]i32 = @splat(0);
            const start = try g.hostArray(std.mem.sliceAsBytes(&offset), &.{4}, .int32);
            const zero_start = try g.hostArray(std.mem.sliceAsBytes(&zero), &.{4}, .int32);
            var incoming: [6]T = undefined;
            const previous: [6]?T = .{ self.k, self.v, self.index_k, self.k_scales, self.v_scales, self.index_scales };
            const lanes: usize = if (self.bits == 8) 6 else 3;
            if (self.bits == 8) {
                if (comptime G != common.MlxOps) {
                    return error.MiniMaxQ8BackendUnsupported;
                } else {
                    const q8 = @import("q8_cache.zig");
                    const qk = try q8.quantize(g, k);
                    const qv = try q8.quantize(g, v);
                    const qi = try q8.quantize(g, index_k);
                    incoming = .{ qk.codes, qv.codes, qi.codes, qk.scales, qv.scales, qi.scales };
                }
            } else {
                incoming[0] = k;
                incoming[1] = v;
                incoming[2] = index_k;
            }
            var next: [6]T = undefined;
            for (0..lanes) |i| next[i] = try update(g, previous[i], incoming[i], self.len, self.capacity, capacity, start, zero_start);
            // Drop all owning handles before evaluation to allow donation.
            self.clear(g);
            errdefer self.clear(g);
            try g.evalAll(next[0..lanes]);
            self.k = g.keep(next[0]);
            self.v = g.keep(next[1]);
            self.index_k = g.keep(next[2]);
            if (self.bits == 8) {
                self.k_scales = g.keep(next[3]);
                self.v_scales = g.keep(next[4]);
                self.index_scales = g.keep(next[5]);
            }
            self.len = need;
            self.capacity = capacity;
        }
    };
}

pub const Cache = CacheOf(common.MlxOps);

test "MiniMax capacity rounds its bill and bounds geometric growth" {
    try std.testing.expectEqual(@as(u32, 256), try nextCapacity(0, 8, 513));
    try std.testing.expectEqual(@as(u32, 512), try nextCapacity(256, 257, 513));
    try std.testing.expectEqual(@as(u32, 768), try nextCapacity(512, 513, 513));
    try std.testing.expectEqual(@as(u32, 131072), paddedCapacity(131072));
    try std.testing.expectEqual(@as(u32, 131328), paddedCapacity(131073));
    try std.testing.expectError(error.MiniMaxCacheLimit, nextCapacity(512, 514, 513));
}

test "MiniMax capacity updates donate after views expire and retain only three buffers" {
    const G = common.TraceOps;
    var g = G.init(std.testing.allocator);
    defer g.deinit();
    g.track_live = true;
    var cache: CacheOf(G) = .{ .limit = 513 };
    defer cache.clear(&g);
    for ([_]u32{ 1, 7, 249, 256 }) |count| {
        const mark = g.mark();
        const k = try g.zeros(&.{ 1, 4, @intCast(count), 128 }, .bfloat16);
        const index = try g.zeros(&.{ 1, 1, @intCast(count), 128 }, .bfloat16);
        const view = try cache.append(&g, k, k, index);
        try std.testing.expectEqual(@as(c_int, @intCast(cache.len)), g.shapeOf(view.k).d[2]);
        g.resetTo(mark);
    }
    try std.testing.expectEqual(@as(u32, 513), cache.len);
    try std.testing.expectEqual(@as(u32, 768), cache.capacity);
    for (g.lane_writes.items) |write| try std.testing.expect(write.pinned_by == null);
    var owners: u32 = 0;
    var it = g.live_kept.valueIterator();
    while (it.next()) |count| if (count.* > 0) {
        owners += @intCast(count.*);
    };
    try std.testing.expectEqual(@as(u32, 3), owners);
    try std.testing.expectEqual(@as(u32, 0), g.use_after_drop);
    cache.clear(&g);
    try std.testing.expectEqual(@as(u32, 513), cache.limit);
    try std.testing.expectEqual(@as(u32, 0), cache.len);
}

test "MiniMax capacity donation detector sees a retained prefix view" {
    const G = common.TraceOps;
    var g = G.init(std.testing.allocator);
    defer g.deinit();
    g.track_live = true;
    var cache: CacheOf(G) = .{ .limit = 256 };
    defer cache.clear(&g);
    const k = try g.zeros(&.{ 1, 4, 1, 128 }, .bfloat16);
    const index = try g.zeros(&.{ 1, 1, 1, 128 }, .bfloat16);
    const first = try cache.append(&g, k, k, index);
    const held = g.keep(first.k);
    defer g.release(held);
    _ = try cache.append(&g, k, k, index);
    var detected = false;
    for (g.lane_writes.items) |write| if (write.pinned_by != null) {
        detected = true;
    };
    try std.testing.expect(detected);
}

test "MiniMax GPU capacity appends grow preserve prefixes and donate steady buffers" {
    const enabled = std.c.getenv("MINIMAX_GPU_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const mlx = @import("sdk").mlx;
    const a = std.testing.allocator;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try common.MlxOps.init(a, stream);
    defer g.deinit();
    var cache: Cache = .{ .limit = 513 };
    defer cache.clear(&g);
    var previous_pointer: usize = 0;
    var previous_capacity: u32 = 0;
    for ([_]u32{ 1, 7, 249, 256 }) |count| {
        const before = cache.len;
        const mark = g.mark();
        const values = try a.alloc(f32, 2 * count * 4);
        defer a.free(values);
        for (0..2) |h| for (0..count) |t| for (0..4) |d| {
            values[(h * count + t) * 4 + d] = @floatFromInt(h * 10000 + (before + t) * 8 + d);
        };
        const k = try g.hostArray(std.mem.sliceAsBytes(values), &.{ 1, 2, @intCast(count), 4 }, .float32);
        const index = try g.slice(k, &.{ 0, 0, 0, 0 }, &.{ 1, 1, @intCast(count), 4 }, &.{ 1, 1, 1, 1 });
        const view = try cache.append(&g, k, k, index);
        const pointer = @intFromPtr(mlx.mlx_array_data_float32(cache.k.?).?);
        if (cache.capacity == previous_capacity) try std.testing.expectEqual(previous_pointer, pointer);
        previous_capacity = cache.capacity;
        previous_pointer = pointer;
        const got = try a.alloc(f32, 2 * cache.len * 4);
        defer a.free(got);
        _ = try g.hostF32(view.k, got);
        for (0..2) |h| for (0..cache.len) |t| for (0..4) |d| {
            try std.testing.expectEqual(@as(f32, @floatFromInt(h * 10000 + t * 8 + d)), got[(h * cache.len + t) * 4 + d]);
        };
        g.resetTo(mark);
    }
    try std.testing.expectEqual(@as(u32, 513), cache.len);
    try std.testing.expectEqual(@as(u32, 768), cache.capacity);
    cache.clear(&g);
    try std.testing.expectEqual(@as(u32, 513), cache.limit);
    try std.testing.expectEqual(@as(u32, 0), cache.capacity);
}
