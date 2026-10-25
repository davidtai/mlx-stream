const std = @import("std");

pub fn run(comptime Tensor: type, executor: anytype, ids: []const u32) !Tensor {
    if (ids.len == 0) return error.MiniMaxEmptyStep;
    var last: ?Tensor = null;
    errdefer if (last) |value| executor.releaseStep(value);
    for (ids, 0..) |id, index| {
        if (last) |value| executor.releaseStep(value);
        last = null;
        last = try executor.token(id, index + 1 == ids.len);
    }
    return last.?;
}

test "MiniMax serial step runs every trunk but only the final vocabulary head" {
    const Executor = struct {
        state: u32 = 0,
        trunk_calls: usize = 0,
        head_calls: usize = 0,
        released: usize = 0,
        pub fn token(self: *@This(), id: u32, head: bool) !u32 {
            self.trunk_calls += 1;
            self.state = self.state * 10 + id;
            if (head) self.head_calls += 1;
            return if (head) self.state + 1000 else self.state;
        }
        pub fn releaseStep(self: *@This(), _: u32) void {
            self.released += 1;
        }
    };
    var executor: Executor = .{};
    const logits = try run(u32, &executor, &.{ 2, 3, 5 });
    try std.testing.expectEqual(@as(u32, 1235), logits);
    try std.testing.expectEqual(@as(u32, 235), executor.state);
    try std.testing.expectEqual(@as(usize, 3), executor.trunk_calls);
    try std.testing.expectEqual(@as(usize, 1), executor.head_calls);
    try std.testing.expectEqual(@as(usize, 2), executor.released);
    _ = try run(u32, &executor, &.{7});
    try std.testing.expectEqual(@as(usize, 2), executor.head_calls);
}

test "MiniMax serial step stops on failed forward and never evaluates a pending head" {
    const Executor = struct {
        trunk_calls: usize = 0,
        head_calls: usize = 0,
        released: usize = 0,
        pub fn token(self: *@This(), id: u32, head: bool) !u32 {
            self.trunk_calls += 1;
            if (id == 9) return error.FixtureForwardFailure;
            if (head) self.head_calls += 1;
            return id;
        }
        pub fn releaseStep(self: *@This(), _: u32) void {
            self.released += 1;
        }
    };
    var executor: Executor = .{};
    try std.testing.expectError(error.FixtureForwardFailure, run(u32, &executor, &.{ 1, 9, 2 }));
    try std.testing.expectEqual(@as(usize, 2), executor.trunk_calls);
    try std.testing.expectEqual(@as(usize, 0), executor.head_calls);
    try std.testing.expectEqual(@as(usize, 1), executor.released);
}
