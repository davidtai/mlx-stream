const std = @import("std");
pub const tokens_per_chunk = 8;

pub fn run(comptime Tensor: type, executor: anytype, ids: []const u32) !Tensor {
    if (ids.len == 0) return error.MiniMaxEmptyPrompt;
    var last: ?Tensor = null;
    errdefer if (last) |value| executor.releaseStep(value);
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + tokens_per_chunk);
        if (last) |value| executor.releaseStep(value);
        last = null;
        last = try executor.chunk(ids[at..end], end == ids.len);
        at = end;
    }
    return last.?;
}

test "MiniMax prefill driver bounds chunks and requests only final logits" {
    const Executor = struct {
        lengths: [3]usize = undefined,
        calls: usize = 0,
        heads: usize = 0,
        forwarded: usize = 0,
        released: usize = 0,
        pub fn chunk(self: *@This(), ids: []const u32, head: bool) !u32 {
            self.lengths[self.calls] = ids.len;
            self.calls += 1;
            self.forwarded += ids.len;
            if (head) self.heads += 1;
            return ids[ids.len - 1];
        }
        pub fn releaseStep(self: *@This(), _: u32) void {
            self.released += 1;
        }
    };
    var executor: Executor = .{};
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    try std.testing.expectEqual(@as(u32, 16), try run(u32, &executor, &ids));
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 1 }, &executor.lengths);
    try std.testing.expectEqual(@as(usize, 17), executor.forwarded);
    try std.testing.expectEqual(@as(usize, 1), executor.heads);
    try std.testing.expectEqual(@as(usize, 2), executor.released);
}

test "MiniMax prefill driver releases the completed tile before a later failure" {
    const Executor = struct {
        calls: usize = 0,
        heads: usize = 0,
        released: usize = 0,
        pub fn chunk(self: *@This(), _: []const u32, head: bool) !u32 {
            self.calls += 1;
            if (self.calls == 2) return error.FixtureChunkFailure;
            if (head) self.heads += 1;
            return 1;
        }
        pub fn releaseStep(self: *@This(), _: u32) void {
            self.released += 1;
        }
    };
    var executor: Executor = .{};
    const ids: [17]u32 = @splat(1);
    try std.testing.expectError(error.FixtureChunkFailure, run(u32, &executor, &ids));
    try std.testing.expectEqual(@as(usize, 2), executor.calls);
    try std.testing.expectEqual(@as(usize, 0), executor.heads);
    try std.testing.expectEqual(@as(usize, 1), executor.released);
}
