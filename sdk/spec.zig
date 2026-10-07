//! G3, speculative decode as an opt-in `arch` capability: `none`, or the arch's own `draft_lane` over its
//! module-owned state.

const std = @import("std");
const check = @import("check.zig");

pub const Spec = union(enum) {
    none,
    draft_lane: DraftLane,
};

/// What the host knows of a request when it arms a lane: the lane decides which requests it serves.
pub const ArmRequest = struct {
    /// temperature 0 (or top_k 1)
    greedy: bool,
    /// no logprobs, grammar or penalties
    clean: bool,
};

pub const DraftArm = enum { off, greedy, typical, stochastic };

/// One round's result: [t1, <= accepted_cap accepted drafts] (owned by the round's allocator) and the next token,
/// which is not in the state.
pub const DraftRound = struct {
    tokens: []u32,
    accepted: u32,
    next_token: u32,

    pub fn deinit(self: *DraftRound, a: std.mem.Allocator) void {
        a.free(self.tokens);
    }
};

/// A lane's counters over the request, for `[spec-stats]`, `/props` and the receipts.
pub const DraftStats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    generated: u64 = 0,
};

/// The arch drives its own rounds over its module-owned state; the host dispatches, stops and emits. Every round
/// keeps the state at prompt + emitted with t1 not in it, and never stops: EOS, stop strings and the budget are
/// the host's. One indirect call per round.
pub const DraftLane = struct {
    /// Rows verified per round; 0 = serial (the lane's readiness signal).
    block_size: *const fn (m: *const anyopaque) u32,
    /// The lane as installed, for the server log and the receipts.
    lane_name: *const fn (m: *const anyopaque) []const u8,
    arm: *const fn (m: *const anyopaque, req: ArmRequest) DraftArm,
    round: *const fn (m: *anyopaque, a: std.mem.Allocator, t1: u32, accepted_cap: u32) anyerror!DraftRound,
    stats: *const fn (m: *const anyopaque) DraftStats,

    /// The table of `L` (an arch's `draft_lane` namespace) over modules of type `M`. A missing or mistyped
    /// declaration is a compile error naming it.
    pub fn of(comptime M: type, comptime L: type) DraftLane {
        comptime {
            const w = "draft_lane " ++ @typeName(L);
            check.fnDecl(w, L, "blockSize", &.{*const M}, u32);
            check.fnDecl(w, L, "laneName", &.{*const M}, []const u8);
            check.fnDecl(w, L, "arm", &.{ *const M, ArmRequest }, DraftArm);
            check.fnDecl(w, L, "round", &.{ *M, std.mem.Allocator, u32, u32 }, DraftRound);
            check.fnDecl(w, L, "stats", &.{*const M}, DraftStats);
        }
        const W = struct {
            fn blockSize(m: *const anyopaque) u32 {
                return L.blockSize(@ptrCast(@alignCast(m)));
            }
            fn laneName(m: *const anyopaque) []const u8 {
                return L.laneName(@ptrCast(@alignCast(m)));
            }
            fn arm(m: *const anyopaque, req: ArmRequest) DraftArm {
                return L.arm(@ptrCast(@alignCast(m)), req);
            }
            fn round(m: *anyopaque, a: std.mem.Allocator, t1: u32, accepted_cap: u32) anyerror!DraftRound {
                return L.round(@ptrCast(@alignCast(m)), a, t1, accepted_cap);
            }
            fn stats(m: *const anyopaque) DraftStats {
                return L.stats(@ptrCast(@alignCast(m)));
            }
        };
        return .{ .block_size = W.blockSize, .lane_name = W.laneName, .arm = W.arm, .round = W.round, .stats = W.stats };
    }
};

const testing = std.testing;

test "sdk spec: a draft lane's table calls its namespace on the erased module" {
    const M = struct { block: u32, rounds: u64 = 0 };
    const L = struct {
        pub fn blockSize(m: *const M) u32 {
            return m.block;
        }
        pub fn laneName(_: *const M) []const u8 {
            return "fake";
        }
        pub fn arm(_: *const M, req: ArmRequest) DraftArm {
            return if (req.greedy and req.clean) .typical else .off;
        }
        pub fn round(m: *M, a: std.mem.Allocator, t1: u32, accepted_cap: u32) !DraftRound {
            m.rounds += 1;
            const toks = try a.alloc(u32, 1 + accepted_cap);
            for (toks, 0..) |*t, i| t.* = t1 + @as(u32, @intCast(i));
            return .{ .tokens = toks, .accepted = accepted_cap, .next_token = t1 + accepted_cap + 1 };
        }
        pub fn stats(m: *const M) DraftStats {
            return .{ .rounds = m.rounds };
        }
    };
    const lane = DraftLane.of(M, L);
    var m: M = .{ .block = 5 };
    try testing.expectEqual(@as(u32, 5), lane.block_size(&m));
    try testing.expectEqualStrings("fake", lane.lane_name(&m));
    try testing.expectEqual(DraftArm.typical, lane.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expectEqual(DraftArm.off, lane.arm(&m, .{ .greedy = true, .clean = false }));
    var r = try lane.round(&m, testing.allocator, 7, 2);
    defer r.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, r.tokens);
    try testing.expectEqual(@as(u64, 1), lane.stats(&m).rounds);
}
