//! `expert_rows_by_layer`: the decode rows' split between a streamed arch's routed layers at the plan's total (glm5_next,
//! beam). `uniform` puts every layer at the plan's rows; `offsets` adds one integer per routed layer (summing to 0) to the
//! decode rows. The split moves rows, never bytes: the one row term is linear in the rows, the decode total is the
//! plan's, and the prompt total can only fall. The setting is `"uniform"` or an array of integers (each within i16)
//! summing to 0, of the model's routed layers' length; anything else refuses by name (`check`).

const std = @import("std");

/// The routed layers an offsets array may name at most.
pub const max_layers = 64;

pub const LayerRows = union(enum) {
    uniform,
    offsets: Offsets,
    /// Not `"uniform"` nor an array of integers within i16 (`setting`), an array longer than `max_layers` (`length`), or
    /// one whose entries do not sum to 0 (`sum`).
    invalid: enum { setting, length, sum },

    pub const Offsets = struct {
        len: u8,
        d: [max_layers]i16 = @splat(0),

        pub fn slice(o: *const Offsets) []const i16 {
            return o.d[0..o.len];
        }
    };

    /// The setting's value: "uniform", or integers (each within i16, at most `max_layers`) summing to 0.
    pub fn parse(v: std.json.Value) LayerRows {
        switch (v) {
            .string => |s| return if (std.mem.eql(u8, s, "uniform")) .uniform else .{ .invalid = .setting },
            .array => |arr| {
                for (arr.items) |item| if (item != .integer or item.integer < std.math.minInt(i16) or item.integer > std.math.maxInt(i16)) return .{ .invalid = .setting };
                if (arr.items.len == 0 or arr.items.len > max_layers) return .{ .invalid = .length };
                var out: Offsets = .{ .len = @intCast(arr.items.len) };
                var sum: i64 = 0;
                for (arr.items, out.d[0..arr.items.len]) |item, *o| {
                    o.* = @intCast(item.integer);
                    sum += item.integer;
                }
                return if (sum == 0) .{ .offsets = out } else .{ .invalid = .sum };
            },
            else => return .{ .invalid = .setting },
        }
    }

    /// The setting against a model of `routed_layers`: refused by name (`ExpertRowsByLayer`, `...Length`, `...Sum`).
    pub fn check(r: LayerRows, routed_layers: u32) error{ ExpertRowsByLayer, ExpertRowsByLayerLength, ExpertRowsByLayerSum }!void {
        switch (r) {
            .uniform => {},
            .offsets => |o| if (o.len != routed_layers) return error.ExpertRowsByLayerLength,
            .invalid => |why| return switch (why) {
                .setting => error.ExpertRowsByLayer,
                .length => error.ExpertRowsByLayerLength,
                .sum => error.ExpertRowsByLayerSum,
            },
        }
    }
};

/// Slot rows per routed layer, per phase (`sdk.Rows`' fields).
pub const Rows = struct { prompt: u32, decode: u32 };

/// The per-layer rows of both phases (routed-layer order) into `prompt` / `decode` (each of the model's routed layers'
/// length), from the plan's uniform `rows`, once at construction. `uniform`: every layer at `rows`. `offsets`: the prompt
/// rows stay uniform at `rows.prompt` in every layer (equal prompt bytes; the split applies at the phase change);
/// decode = rows.decode + offset, held at rows.prompt at least (the stream grows a layer at the phase change, never
/// shrinks it), and the rows that floor adds are taken back from the widest decode layers (the first widest first, one
/// row at a time, never under rows.prompt) so the decode rows sum to the plan's: both phases bill exactly as the uniform
/// rows. A layer is also held at `n_experts` at most, and the rows that ceiling removes go to the narrowest decode layers
/// (the first narrowest first, one row at a time, never over `n_experts`). Refused by name when the plan's rows are over
/// `n_experts` or the floor cannot be paid back (`ExpertRowsByLayerRange`).
pub fn layerRows(split: LayerRows, rows: anytype, n_experts: u32, prompt: []u32, decode: []u32) error{ ExpertRowsByLayer, ExpertRowsByLayerLength, ExpertRowsByLayerRange }!void {
    if (prompt.len != decode.len) return error.ExpertRowsByLayerLength;
    switch (split) {
        .uniform => {
            @memset(prompt, rows.prompt);
            @memset(decode, rows.decode);
        },
        .offsets => |*o| {
            if (o.len != decode.len) return error.ExpertRowsByLayerLength;
            if (rows.decode > n_experts or rows.prompt > rows.decode) return error.ExpertRowsByLayerRange;
            var surplus: i64 = 0;
            for (prompt, decode, o.slice()) |*p, *q, off| {
                const dd = @as(i64, rows.decode) + off;
                const held = std.math.clamp(dd, @as(i64, rows.prompt), @as(i64, n_experts));
                surplus += held - dd;
                q.* = @intCast(held);
                p.* = rows.prompt;
            }
            while (surplus < 0) : (surplus += 1) {
                var ni: usize = 0;
                for (decode, 0..) |d, i| {
                    if (d < decode[ni]) ni = i;
                }
                if (decode[ni] >= n_experts) return error.ExpertRowsByLayerRange;
                decode[ni] += 1;
            }
            while (surplus > 0) : (surplus -= 1) {
                var wi: usize = 0;
                for (decode, 0..) |d, i| {
                    if (d > decode[wi]) wi = i;
                }
                if (decode[wi] <= rows.prompt) return error.ExpertRowsByLayerRange;
                decode[wi] -= 1;
            }
        },
        .invalid => return error.ExpertRowsByLayer,
    }
}

const testing = std.testing;

test "expert layer rows: uniform fills both phases; offsets keep the decode total and the prompt rows; bad settings refuse by name" {
    var p: [4]u32 = undefined;
    var d: [4]u32 = undefined;
    try layerRows(.uniform, Rows{ .prompt = 10, .decode = 12 }, 16, &p, &d);
    try testing.expectEqualSlices(u32, &.{ 12, 12, 12, 12 }, &d);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "[6, -1, -1, -4]", .{});
    defer parsed.deinit();
    const split = LayerRows.parse(parsed.value);
    try split.check(4);
    try layerRows(split, Rows{ .prompt = 10, .decode = 12 }, 16, &p, &d);
    var sum: u32 = 0;
    for (p, d) |a, b| {
        try testing.expectEqual(@as(u32, 10), a);
        try testing.expect(b >= a and b <= 16);
        sum += b;
    }
    try testing.expectEqual(@as(u32, 48), sum);
    try testing.expectError(error.ExpertRowsByLayerLength, split.check(5));
    const bad = try std.json.parseFromSlice(std.json.Value, testing.allocator, "[1, 2]", .{});
    defer bad.deinit();
    try testing.expectError(error.ExpertRowsByLayerSum, LayerRows.parse(bad.value).check(2));
    try testing.expectError(error.ExpertRowsByLayer, LayerRows.parse(.{ .string = "derived" }).check(2));
}
