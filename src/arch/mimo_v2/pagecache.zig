//! The page cache the pack's own files hold, and the fill's baseline built on it. The window guard stops a step on
//! the box's physical used (vm_stat wired + active + inactive + compressor: file cache included), so the rows fill on
//! top of that, less only the bytes the pack's own files hold in the page cache: construction evicts those (the
//! residents and experts then load past the page cache, `caps.residents_past_page_cache`), every other file's cache
//! (another server's mapping, a download, build caches) stays billed against the rows instead of against the guard.
const std = @import("std");
const sdk = @import("sdk");

const log = std.log.scoped(.mimo);

const MS_SYNC: c_int = 0x10;
const MS_INVALIDATE: c_int = 0x2;
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;
extern "c" fn msync(addr: *anyopaque, len: usize, flags: c_int) c_int;

/// The box's file-backed pages (vm_stat "File-backed pages"), bytes.
pub fn fileBackedBytes() u64 {
    return sdk.memory.vmBytes().external;
}

/// `path`'s bytes resident in the page cache (mincore over a read-only shared mapping); `evict` first invalidates them
/// (msync MS_INVALIDATE), so the count is what the eviction left.
pub fn fileCachedBytes(path: [:0]const u8, evict: bool) !u64 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end <= 0) return 0;
    const size: usize = @intCast(end);
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(map);
    if (evict and msync(map.ptr, size, MS_SYNC | MS_INVALIDATE) != 0) return error.MsyncFailed;
    const page = std.heap.pageSize();
    var vec: [1 << 16]u8 = undefined;
    var n: u64 = 0;
    var off: usize = 0;
    while (off < size) {
        const len = @min(size - off, vec.len * page);
        if (mincore(map.ptr + off, len, &vec) != 0) return error.MincoreFailed;
        for (vec[0 .. (len + page - 1) / page]) |v| n += v & 1;
        off += len;
    }
    return n * page;
}

/// The bytes every regular file of the pack directory `dir` holds in the page cache, counted before `evict`
/// invalidates them.
pub fn packCachedBytes(io: std.Io, dir: []const u8, evict: bool) !u64 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    var total: u64 = 0;
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    while (try it.next(io)) |e| {
        if (e.kind != .file) continue;
        const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, e.name }, 0);
        const cached = try fileCachedBytes(path, false);
        total += cached;
        if (evict and cached > 0) _ = try fileCachedBytes(path, true);
    }
    return total;
}

/// The fill's baseline: the box's physical used less the pack's own cached bytes, never below the host's non-file
/// baseline (`nonfile`: wired + compressor + internal - purgeable; a pack page vm_stat counts as speculative is in the
/// census but not in physical used).
pub fn baselineOf(physical_used: u64, pack_cached: u64, nonfile: u64) u64 {
    return @max(nonfile, physical_used -| pack_cached);
}

/// Once, at construction: the box's physical used, the pack's cached bytes (then evicted, so the subtraction holds),
/// and the baseline the rows fill on. Off Darwin physical used reads 0 and the baseline is the host's `nonfile`.
pub fn admissionBaseline(io: std.Io, dir: []const u8, nonfile: u64) !u64 {
    const physical_used = sdk.memory.physicalUsedBytes(sdk.memory.vmBytes());
    const pack_cached = try packCachedBytes(io, dir, true);
    const baseline = baselineOf(physical_used, pack_cached, nonfile);
    log.info("MiMo admission baseline: physical_used={d} pack_cached={d} nonfile={d} -> baseline={d}", .{ physical_used, pack_cached, nonfile, baseline });
    return baseline;
}

const testing = std.testing;

test "mimo filepart: the baseline is physical used less the pack's cached bytes, floored at the host's non-file baseline" {
    // loop2-bf16-20261008 std-16k census (vm_stat at cell start, after the pack eviction): physical 11.668 GB,
    // non-file 8.899 GB, the pack evicted (0 cached): the rows fill on 11.668 GB, not 8.899 GB.
    try testing.expectEqual(@as(u64, 11_668_000_000), baselineOf(11_668_000_000, 0, 8_899_000_000));
    // loop1-bf16-20261008 (guard kill): 24.4 GB physical used on an 8.4 GB non-file sample; 16 GB of it foreign
    // file cache, counted against the rows.
    try testing.expectEqual(@as(u64, 24_400_000_000), baselineOf(24_400_000_000, 0, 8_400_000_000));
    // The pack's own cached bytes come off (construction evicts them).
    try testing.expectEqual(@as(u64, 12_400_000_000), baselineOf(24_400_000_000, 12_000_000_000, 8_400_000_000));
    // Never below the host's non-file baseline, and no underflow.
    try testing.expectEqual(@as(u64, 8_400_000_000), baselineOf(9_000_000_000, 4_000_000_000, 8_400_000_000));
    try testing.expectEqual(@as(u64, 8_400_000_000), baselineOf(1, 4_000_000_000, 8_400_000_000));
    try testing.expectEqual(@as(u64, 0), baselineOf(0, 0, 0));
}

test "mimo filepart: the pack census counts every file's cached bytes, evicts them, and skips subdirectories" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const page = std.heap.pageSize();
    const a_bytes = try testing.allocator.alloc(u8, 3 * page + 17);
    defer testing.allocator.free(a_bytes);
    @memset(a_bytes, 0x5a);
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.bin", .data = a_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.json", .data = "" });
    try tmp.dir.createDirPath(io, "sub");
    const dir = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(dir);
    // Read through the cache: both files' pages resident.
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    const experts = try std.fmt.bufPrintSentinel(&buf, "{s}/experts.bin", .{dir}, 0);
    testing.allocator.free(try tmp.dir.readFileAlloc(io, "experts.bin", testing.allocator, .unlimited));
    const before = try fileCachedBytes(experts, false);
    const census = try packCachedBytes(io, dir, false);
    try testing.expect(census >= before);
    try testing.expect(census <= 5 * page);
    // The evicting census reports what it found, and leaves the pack uncached.
    const found = try packCachedBytes(io, dir, true);
    try testing.expect(found >= before);
    try testing.expectEqual(@as(u64, 0), try packCachedBytes(io, dir, false));
    try testing.expectEqual(@as(u64, 0), try fileCachedBytes(experts, false));
    std.debug.print("\nmimo filepart census: experts.bin cached {d} B before, pack census {d} B, evicted to 0\n", .{ before, found });
    // The admission baseline on this box: never below the host's non-file baseline; equal to it when it is the larger.
    const huge: u64 = 1 << 60;
    try testing.expectEqual(huge, try admissionBaseline(io, dir, huge));
    try testing.expect(try admissionBaseline(io, dir, 0) > 0 or !@import("builtin").os.tag.isDarwin());
}
