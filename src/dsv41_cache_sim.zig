//! MLX 0.32.2's Metal buffer cache policy on the host (what a cache limit, `mlx_set_cache_limit`, bounds). From
//! lib/mlx-src (metal/allocator.cpp, common/buffer_cache.h):
//!   - malloc rounds a size above one page up to whole pages, then takes the smallest cached buffer of at least that
//!     size and below min(2 x size, size + 2 pages); else it allocates fresh;
//!   - free recycles the buffer into the cache while the cache holds less than the limit (so it may end one buffer
//!     over), else releases it to the driver.
//! Pure host code; no MLX call.
const std = @import("std");

/// The Metal page on Apple silicon (vm_page_size).
pub const page: u64 = 16_384;

pub fn rounded(size: u64) u64 {
    return if (size > page) std.mem.alignForward(u64, size, page) else size;
}

pub const Sim = struct {
    limit: u64,
    /// Cached buffer sizes (order is irrelevant to reuse: the smallest fitting size wins).
    pool: std.ArrayList(u64) = .empty,
    pool_bytes: u64 = 0,
    fresh: u64 = 0,
    fresh_bytes: u64 = 0,
    reused: u64 = 0,

    pub fn deinit(s: *Sim, a: std.mem.Allocator) void {
        s.pool.deinit(a);
    }

    /// One malloc of `size` bytes; returns the buffer's bytes (a reused buffer may be larger).
    pub fn alloc(s: *Sim, size: u64) u64 {
        const want = rounded(size);
        const bound = @min(2 * want, want + 2 * page);
        var best: ?usize = null;
        for (s.pool.items, 0..) |b, i| {
            if (b < want or b >= bound) continue;
            if (best == null or b < s.pool.items[best.?]) best = i;
        }
        if (best) |i| {
            const b = s.pool.swapRemove(i);
            s.pool_bytes -= b;
            s.reused += 1;
            return b;
        }
        s.fresh += 1;
        s.fresh_bytes += want;
        return want;
    }

    /// One free of a buffer of `bytes` (as `alloc` returned it).
    pub fn free(s: *Sim, a: std.mem.Allocator, bytes: u64) !void {
        if (s.pool_bytes < s.limit) {
            try s.pool.append(a, bytes);
            s.pool_bytes += bytes;
        }
    }
};

const testing = std.testing;

test "dsv41 cache sim: MLX's reuse window and limit, on known sequences" {
    const a = testing.allocator;
    // Sizes round to pages above one page; a 2-page buffer serves a 1.5-page request, a 4-page one does not.
    try testing.expectEqual(@as(u64, 32_768), rounded(20_000));
    try testing.expectEqual(@as(u64, 100), rounded(100));
    var s: Sim = .{ .limit = 1 << 20 };
    defer s.deinit(a);
    try s.free(a, 32_768);
    try testing.expectEqual(@as(u64, 32_768), s.alloc(24_000));
    try s.free(a, 65_536);
    _ = s.alloc(24_000);
    try testing.expectEqual(@as(u64, 1), s.fresh);
    // Small buffers reuse below twice their size only.
    try s.free(a, 100);
    try testing.expectEqual(@as(u64, 100), s.alloc(60));
    // At the limit a free releases: the pool may end one buffer over, never more.
    var t: Sim = .{ .limit = 50_000 };
    defer t.deinit(a);
    try t.free(a, 32_768);
    try t.free(a, 32_768);
    try t.free(a, 32_768);
    try testing.expectEqual(@as(u64, 65_536), t.pool_bytes);
}
