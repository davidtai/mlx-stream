const std = @import("std");

/// Bounded leases for module-owned contexts. A live request is never a prefix-cache victim.
pub const Slots = struct {
    capacity: u32 = 1,
    entries: [2]Entry = @splat(.{}),
    clock: u64 = 0,

    const Entry = struct { handle: ?u64 = null, released: u64 = 0 };
    pub const Error = error{ RequestSlotsFull, InvalidRequestSlot, RequestSlotClockExhausted };

    pub fn acquire(self: *Slots, prefix_lengths: *const [2]u64) Error!u64 {
        var best: ?usize = null;
        for (self.entries[0..self.capacity], 0..) |entry, i| {
            if (entry.handle != null) continue;
            if (best == null or prefix_lengths[i] > prefix_lengths[best.?] or
                (prefix_lengths[i] == prefix_lengths[best.?] and entry.released < self.entries[best.?].released)) best = i;
        }
        const i = best orelse return error.RequestSlotsFull;
        if (self.clock >= (std.math.maxInt(u64) - 2) / 2) return error.RequestSlotClockExhausted;
        self.clock += 1;
        const handle = self.clock * 2 + i;
        self.entries[i].handle = handle;
        return handle;
    }

    pub fn index(self: *const Slots, handle: u64) Error!usize {
        const i: usize = @intCast(handle % 2);
        if (i >= self.capacity or self.entries[i].handle != handle) return error.InvalidRequestSlot;
        return i;
    }

    pub fn release(self: *Slots, handle: u64) Error!void {
        const i = try self.index(handle);
        if (self.clock >= (std.math.maxInt(u64) - 2) / 2) return error.RequestSlotClockExhausted;
        self.clock += 1;
        self.entries[i].released = self.clock;
        self.entries[i].handle = null;
    }
};

test "dsv41 request slots: two live leases cannot be evicted and released handles cannot select a reused slot" {
    var slots: Slots = .{ .capacity = 2 };
    const a = try slots.acquire(&.{ 0, 0 });
    const b = try slots.acquire(&.{ 99, 0 });
    try std.testing.expect(a != b);
    try std.testing.expectError(error.RequestSlotsFull, slots.acquire(&.{ 100, 100 }));
    try slots.release(a);
    const c = try slots.acquire(&.{ 10, 100 });
    try std.testing.expectEqual(try slots.index(c), @as(usize, 0));
    try std.testing.expectError(error.InvalidRequestSlot, slots.index(a));
    try std.testing.expectError(error.InvalidRequestSlot, slots.release(a));
    try std.testing.expectEqual(@as(usize, 1), try slots.index(b));
}

test "dsv41 request slots: best idle prefix wins, otherwise least recently released; default capacity remains one" {
    var slots: Slots = .{ .capacity = 2 };
    const a = try slots.acquire(&.{ 0, 0 });
    const b = try slots.acquire(&.{ 0, 0 });
    try slots.release(b);
    try slots.release(a);
    const reuse = try slots.acquire(&.{ 5, 7 });
    try std.testing.expectEqual(@as(usize, 1), try slots.index(reuse));
    try slots.release(reuse);
    const cold = try slots.acquire(&.{ 0, 0 });
    try std.testing.expectEqual(@as(usize, 0), try slots.index(cold));
    var one: Slots = .{};
    const only = try one.acquire(&.{ 0, 999 });
    try std.testing.expectEqual(@as(usize, 0), try one.index(only));
    try std.testing.expectError(error.RequestSlotsFull, one.acquire(&.{ 0, 999 }));
}
