const std = @import("std");

pub fn Grouping(comptime Ref: type) type {
    return GroupingCapacity(Ref, 4);
}
pub fn GroupingCapacity(comptime Ref: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();
        const Group = struct {
            bank: @FieldType(Ref, "bank"),
            rows: [capacity]u32 = undefined,
            positions: [capacity]usize = undefined,
            n: usize = 0,
        };
        groups: [3]Group = undefined,
        n: usize = 0,

        pub fn init(refs: []const Ref) !Self {
            if (refs.len == 0 or refs.len > capacity or refs.len % 4 != 0) return error.MiniMaxRouteWidth;
            var out: Self = .{};
            for (refs, 0..) |ref, position| {
                var index: usize = 0;
                while (index < out.n and out.groups[index].bank != ref.bank) : (index += 1) {}
                if (index == out.n) {
                    if (out.n == out.groups.len) return error.MiniMaxSlotBankMissing;
                    out.groups[index] = .{ .bank = ref.bank };
                    out.n += 1;
                }
                const group = &out.groups[index];
                group.rows[group.n] = ref.row;
                group.positions[group.n] = position;
                group.n += 1;
            }
            return out;
        }
    };
}

test "MiniMax gather groups preserve bank rows and original router positions" {
    const Ref = struct { bank: enum { base, ext, transient }, row: u32 };
    const refs = [_]Ref{ .{ .bank = .transient, .row = 1 }, .{ .bank = .base, .row = 0 }, .{ .bank = .transient, .row = 0 }, .{ .bank = .base, .row = 1 } };
    const groups = try Grouping(Ref).init(&refs);
    try std.testing.expectEqual(@as(usize, 2), groups.n);
    try std.testing.expectEqual(refs[0].bank, groups.groups[0].bank);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, groups.groups[0].rows[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, groups.groups[0].positions[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, groups.groups[1].rows[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, groups.groups[1].positions[0..2]);
    var reconstructed: [4]Ref = undefined;
    for (groups.groups[0..groups.n]) |group| for (group.rows[0..group.n], group.positions[0..group.n]) |row, position| {
        reconstructed[position] = .{ .bank = group.bank, .row = row };
    };
    try std.testing.expectEqualSlices(Ref, &refs, &reconstructed);
}

test "MiniMax gather groups support one bank and all three slot banks" {
    const Ref = struct { bank: enum { base, ext, transient }, row: u32 };
    const refs = [_]Ref{ .{ .bank = .base, .row = 9 }, .{ .bank = .ext, .row = 0 }, .{ .bank = .base, .row = 4 }, .{ .bank = .transient, .row = 0 } };
    const groups = try Grouping(Ref).init(&refs);
    try std.testing.expectEqual(@as(usize, 3), groups.n);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, groups.groups[0].positions[0..2]);
    var same = refs;
    for (&same) |*ref| ref.bank = .base;
    const single = try Grouping(Ref).init(&same);
    try std.testing.expectEqual(@as(usize, 1), single.n);
    try std.testing.expectEqualSlices(u32, &.{ 9, 0, 4, 0 }, single.groups[0].rows[0..4]);
    try std.testing.expectError(error.MiniMaxRouteWidth, Grouping(Ref).init(refs[0..3]));
}

test "MiniMax gather groups retain all thirty-two routed positions in a prefill tile" {
    const Ref = struct { bank: enum { base, ext, transient }, row: u32 };
    var refs: [32]Ref = undefined;
    for (&refs, 0..) |*ref, index| ref.* = .{ .bank = if (index % 2 == 0) .base else .transient, .row = @intCast(index % 4) };
    const groups = try GroupingCapacity(Ref, 32).init(&refs);
    try std.testing.expectEqual(@as(usize, 2), groups.n);
    var rebuilt: [32]Ref = undefined;
    for (groups.groups[0..groups.n]) |group| for (group.positions[0..group.n], group.rows[0..group.n]) |position, row| {
        rebuilt[position] = .{ .bank = group.bank, .row = row };
    };
    try std.testing.expectEqualSlices(Ref, &refs, &rebuilt);
}
