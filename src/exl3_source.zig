//! The EXL3 expert stream as the expert_source kind's registry entry (G6): discovery and `/props` ask it; the arch
//! that streams with it binds `expert_stream.Stream` at comptime, so nothing per layer crosses this table.

const std = @import("std");
const sdk = @import("sdk");
const expert_stream = @import("expert_stream.zig");

pub const name = "exl3-stream";
pub const caps = expert_stream.source_caps;
pub const uses_reader = expert_stream.uses_reader;

/// Discovery by manifest presence; opening and the quant's claim validate the complete selected geometry.
pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    if (p.model_dir.len == 0) return null;
    var buf: [4096]u8 = undefined;
    for ([_][]const u8{ "expert-manifest-v3.json", "expert-manifest-v2.json" }) |name_| {
        const path = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ p.model_dir, name_ }, 0) catch return null;
        if (std.c.access(path.ptr, 0) == 0) return .native;
    }
    return null;
}

const testing = std.testing;

test "dsv41 plugin: the EXL3 source claims a model directory that holds an expert bank, and nothing else" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(std.testing.io, &rbuf)];
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), root, "{\"model_type\":\"deepseek_v41\"}");
    try testing.expectEqual(@as(?sdk.Priority, null), claims(&peek));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "expert-manifest-v2.json", .data = "{}" });
    try testing.expectEqual(@as(?sdk.Priority, .native), claims(&peek));
}

test "dsv41 plugin: variable selected bank participates in source discovery" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = try @import("expert_bank.zig").writeVariableSynth(testing.allocator, &tmp);
    defer testing.allocator.free(image);
    var rbuf: [1024]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try sdk.ConfigPeek.parse(arena.allocator(), root, "{\"model_type\":\"deepseek_v41\"}");
    try testing.expectEqual(@as(?sdk.Priority, .native), claims(&p));
}
