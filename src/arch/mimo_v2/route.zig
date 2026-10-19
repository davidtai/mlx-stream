//! MiMo's expert route as the NVFP4-streamed arches install it (glm5_next, beam): the construction routes of record
//! (the read schedule, the drains, the bank view, the decode window, the read-ahead relocation, MLX's cache limits per
//! phase), the stream's construction options over a bank's bridge (`streamOptionsOf`), a request's start on the stream,
//! the decode lane's rule, the prompt combine the host's wired policy allows, the decode lane's stock combine and the
//! lookahead's scores on a router's stored weight. Every route here is exact against its control (mimo.md, glm5-next.md).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const expert_event = sdk_ext.expert.event;
const expert_tally = sdk_ext.expert.tally;
const ops = @import("../deepseek_v41/ops.zig");
const mimo_stream = @import("stream.zig");
const mimo_math = @import("math.zig");
const prefill_combine = @import("prefill_combine.zig");

const T = mlx.mlx_array;

/// MiMo's construction routes, fixed (its stack of record; each exact against its control).
pub const read_schedule: mimo_stream.ReadSchedule = .early_whole_trim;
pub const drain: mimo_math.Drain = .pipelined;
pub const bank_view: mimo_math.BankView = .window;
pub const decode_window: mimo_stream.DecodeWindow = .verify;
/// The overlap schedule's relocation (`Options.prompt_relocate`, glm5_next's prompt lever 1): a read-ahead record the
/// call's plans evict before any group hits it and load again later in the call into a transient row no earlier group
/// reads is copied at the plans from its read-ahead row into that row (read once, not twice). Exact by construction.
pub const prompt_relocate = true;
/// P7 `pooled`: MLX's cache limit through the prompt pass, purged at a prompt layer's end once half full; decode keeps
/// its own limit.
pub const prompt_cache_bytes: usize = 8 << 30; // diag (perf/beam-misspath): the prompt-phase buffer cache probe, 2 GiB in the served build
pub const decode_cache_bytes: usize = 256 << 20;

/// The stream's construction options over a bank's bridge `Bp` (pure: the CPU tests read them): MiMo's event gates and
/// lookahead over `rows`, the phase rows' transient release, the read schedule's overlap route with the relocation,
/// decode's window 0, and the rate tally (`tally`: every layer pins at most floor(ratio x experts); none at ratio 0).
/// `direct` sets the read pool's direct demand reads and turns the decode pre-read off (a pre-range stages); the stream
/// proves the geometry at its construction.
pub fn streamOptionsOf(comptime Bp: type, rows: []const u32, memory: Bp.SlotMemory, event: expert_event.Event, tally: expert_tally.Settings, direct: bool) !Bp.Options {
    var opts = Bp.eventOptions(rows, memory, event);
    opts.transient_release = true;
    try Bp.applyReadSchedule(&opts, read_schedule);
    opts.prompt_relocate = prompt_relocate;
    opts.decode_route_ids = mimo_stream.decodeRouteIds(decode_window);
    opts.tally = if (tally.enabled()) .{ .ratio = tally.ratio, .window = tally.window } else null;
    if (direct) {
        opts.pool.direct = true;
        opts.lookahead.?.preread = false;
    }
    return opts;
}

/// What came before a request's start (`requestStart`): the construction (the stream still in its prompt phase, its
/// warm-up's residents), the usage seed (the residents the load read), or a reverse change.
pub const Start = enum { construction, seeded, reversed };

/// A request's start on the stream (or its source), after the source's resume of the prompt phase (nothing live). After
/// the construction: every resident and the prompt state forgotten. After the usage seed: the seeded residents stay
/// (whatever `prompt_residents` says). After a reverse change: `forget` (the control) forgets every resident; `keep`
/// leaves the residents `shrink(.keep)` left in the prompt rows. Then the prompt counts are zeroed and the pins
/// refreshed (`resetPromptCounts`).
pub fn requestStart(stream: anytype, residents: sdk_ext.expert.usage.PromptResidents, start: Start) !void {
    const forget = switch (start) {
        .construction => true,
        .seeded => false,
        .reversed => residents == .forget,
    };
    if (forget) _ = try stream.forgetResidents();
    stream.resetPromptCounts();
}

/// The phase route of a routed call (MiMo's `decodeLane`): one row in either phase, up to `lane_rows` in decode.
pub fn decodeLane(nt: u32, decoding: bool, lane_rows: u32) bool {
    return nt == 1 or (decoding and nt <= lane_rows);
}

/// The prompt combine the host's wired policy allows (`MLX_SERVE_WIRED`, read once at construction): `inplace` under
/// `max` (the host's default), `fused` under any other.
pub fn combineRoute() prefill_combine.Route {
    const wired = std.c.getenv("MLX_SERVE_WIRED") orelse return .inplace;
    return if (std.mem.eql(u8, std.mem.span(wired), "max")) .inplace else .fused;
}

/// The decode lane's combine (MiMo's stock): `routed` [n, k, H] bf16 in router order, f32 weights [n, k]; f32 products,
/// the sum over the k routes, bf16 once.
pub fn combineStock(g: *ops.MlxOps, routed: T, route_weights: T, top_k: u32) !T {
    const n = g.shapeOf(routed).d[0];
    const expanded = try g.reshape(route_weights, &.{ n, @intCast(top_k), 1 });
    return g.astype(try g.sum(try g.mul(try g.astype(routed, .float32), expanded), 1, false), .bfloat16);
}

/// The lookahead's scores on the next routed layer's router weight as stored ([experts, hidden], MLX's matmul over
/// x [n, hidden]; any float dtype), then the sigmoid and the correction as MiMo's F32 route: f32 [n, experts]. No copy of
/// the weight; the scores pick the lookahead's reads only.
pub fn scoresStored(_: *const mimo_math.Predictor, g: *ops.MlxOps, x: T, router: T, correction: T) anyerror!T {
    const logits = try g.matmul(x, try g.transpose(router));
    return g.add(try g.sigmoid(try g.astype(logits, .float32)), correction);
}

/// Once, at construction (every command retired, MLX's cache cleared): the footprint beside the bill's construction
/// terms at the prompt rows, one line. A report, as MiMo's: an over-footprint past `tolerance` is the bill's bug for its
/// tests and the window guard, logged as an under-bill; the load proceeds.
pub fn constructionCheck(comptime arch: []const u8, s: mlx.mlx_stream, b: sdk.MemoryBill, rows: u32, tolerance: u64) void {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    const footprint = sdk.memory.footprint().now;
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    const billed = b.constructionBytes(rows);
    sdk.log.info(arch ++ " construction check: footprint {d} B (MLX active {d} B, cache {d} B, host side {d} B), billed construction terms {d} B, residual {d} B (billed less footprint)\n", .{ footprint, active, cache, footprint -| active -| cache, billed, @as(i64, @intCast(billed)) - @as(i64, @intCast(footprint)) });
    if (footprint > billed + tolerance)
        sdk.log.warn(arch ++ " construction check: the footprint {d} B exceeds the billed construction terms {d} B by more than {d} B (an under-bill: proceeding)\n", .{ footprint, billed, tolerance });
}

/// The host's monotonic clock (ns): a phase change's one timing, never per call.
pub fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const testing = std.testing;

test "mimo route: the decode lane is one row in either phase and up to the lane's rows in decode" {
    try testing.expect(decodeLane(1, false, 4));
    try testing.expect(!decodeLane(2, false, 4));
    try testing.expect(decodeLane(4, true, 4));
    try testing.expect(!decodeLane(5, true, 4));
    try testing.expect(!decodeLane(2, true, 1));
}
