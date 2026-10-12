//! MiMo MXFP4 math for the shared DeepSeek expert prefill scheduler.
//! The scheduler owns expert admission, reads, groups, and routed positions.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");
const quant = @import("../../sdk_ext.zig").quant;
const bank_mod = @import("bank.zig");
const profile = @import("profile.zig");
const moe = @import("moe.zig");
const timeline = @import("prefill_timeline.zig");
const glue = @import("glue.zig");

pub const tile_rows: u32 = 256;
pub const projection_rows: u32 = 1024;
const route_group_rows: u32 = @import("../../sdk_ext.zig").expert.policy.max_route_ids;
const max_rows: u32 = @import("config.zig").prefill_max_tokens * 8;
const T = mlx.mlx_array;
const Proj = quant.GatherQmm.Arrays(T);
const Arrays = quant.BankArrays(Proj);
const Activation = *const fn (*ops.MlxOps, T, T) anyerror!T;
const projection_params: quant.GatherQmm.Params = .{ .mode = .mxfp4, .bits = 4, .group_size = 32 };

/// SDK projection backend over the existing native operation. MlxOps' own
/// gatherQmm fixes sorted=false; this adapter exposes the SDK's sorted flag.
const ProjectionOps = struct {
    pub const T = mlx.mlx_array;
    g: *ops.MlxOps,
    pub fn shapeOf(p: *ProjectionOps, x: mlx.mlx_array) ops.Shape {
        return p.g.shapeOf(x);
    }
    pub fn reshape(p: *ProjectionOps, x: mlx.mlx_array, shape: []const c_int) !mlx.mlx_array {
        return p.g.reshape(x, shape);
    }
    pub fn gatherMatmul(p: *ProjectionOps, x: mlx.mlx_array, w: mlx.mlx_array, scales: ?mlx.mlx_array, biases: ?mlx.mlx_array, rhs: mlx.mlx_array, bits: u32, group: u32, mode: ?@import("sdk").QuantMode, sorted: bool) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        mlx.check(mlx.mlx_gather_qmm(&result, x, w, scales.?, biases orelse .{}, .{}, rhs, true, .some(@intCast(group)), .some(@intCast(bits)), mode.?.cstr(), sorted, p.g.s)) catch |err| {
            _ = mlx.mlx_array_free(result);
            return err;
        };
        return p.g.adopt(result);
    }
};

fn sortedProjection(g: *ops.MlxOps, x: T, p: Proj, rhs: T) !T {
    var projection: ProjectionOps = .{ .g = g };
    return quant.GatherQmm.gatherMatmul(ProjectionOps, &projection, &projection_params, x, p, rhs, true);
}

/// The shared scheduler's profiling surface, using MiMo's existing buckets.
/// Deferred-base modes and its wide-call counter are unused by this route.
pub const Profile = struct {
    pub const enabled = profile.enabled;
    pub const now = profile.now;
    pub const Mode = @import("../deepseek_v41/prefill_timers.zig").Mode;
    pub const charge = profile.charge;
    pub fn setMode(_: Mode) void {}
    pub fn countWideCall() void {}
};

/// The prompt pass's tile drains (`settings.prefill_drain`, bound once at construction: `drains`). `sync` (the
/// control): each expert projection tile and each weighted reduction tile is evaluated and the stream synchronized
/// before the next tile's graph is built, so the GPU idles while the host wakes, builds, allocates and encodes the next
/// one. `pipelined`: each tile is committed (async eval) and then the tile BEFORE it is waited, so the GPU holds the
/// next tile while the host builds the one after: at most two tiles in flight (two tiles' arrays live:
/// `pipelinedExtraBytes`), and a call's last tile is waited, and the stream synchronized, before the call returns (its
/// slots may be refilled after it). The same graphs over the same inputs in the same order: EXACT by construction.
pub const Drain = enum { sync, pipelined, invalid };

const TilesFn = fn (self: *Math, g: *ops.MlxOps, x: T, rows: quant.PrefillRows, bank: Arrays, kept: *std.ArrayList(T)) anyerror!void;
const ReduceFn = fn (g: *ops.MlxOps, rows: T, rows_token: u32, route_weights: T, token: u32, stop: u32, hidden: c_int, out: *CombineOutputs) anyerror!void;
/// The installed drain route: the projection tiles' and the reduction tiles' drivers.
pub const Drains = struct { tiles: *const TilesFn, reduce: *const ReduceFn };

pub fn drains(d: Drain) !Drains {
    return switch (d) {
        .sync => .{ .tiles = Math.projectTilesSync, .reduce = reduceTilesSync },
        .pipelined => .{ .tiles = Math.projectTilesPipelined, .reduce = reduceTilesPipelined },
        .invalid => error.MimoPrefillDrain,
    };
}

/// The expert tile's bank view (`settings.prefill_bank_view`, bound once at construction: `bankViews`). MLX picks a
/// sorted gather_qmm's kernel from the tile's rows B against the bank rows E it is handed (`GatherQMM::eval_gpu`,
/// `gather_qmm_rhs_nax`): the sorted NAX GEMM when B >= 16 and B / E >= 4, its row tile bm = 64 when B / E >= 64 (else
/// 32), and a grid of ceil(B / bm) + E - 1 row threadgroups, the empty ones returning at once; else per-row gather_qmv.
/// `full` (the control): every tile is handed its whole bank (E = 128 / 140 / 146 persistent, 96 transient), so a
/// 1,024-row tile runs bm 32 over ~E - 1 empty threadgroups. `window`: a tile whose full-bank path is the sorted GEMM
/// is handed the slice [first slot, last slot] of its bank (a view: no copy; the packed rows are sorted by slot) and
/// its slots rebased to it, so E = its slot span (prompt pass: ~1-9), bm 64 where B / span >= 64, and no empty
/// threadgroups beyond span - 1. A tile on gather_qmv under the full bank keeps the full bank (the window would move
/// it onto the GEMM: another kernel family), so the window changes only bm and the grid, never the kernel family.
/// Rows, their order, the bank and its layout are the control's. Class: expected exact. Per 16-row fragment the NAX rhs
/// kernel runs the same `tile_matmad_nax` MMA sequence over K for bm 32 (TM = 1) and bm 64 (TM = 2) (BN 64, WN 2:
/// TN = 2, the `TN % 2 == 0` branch either way), and a row's fragment (its expert's start + 16 x floor(i / 16)) is the
/// same at both heights, so each output element accumulates the same products in the same order. Proven byte-equal on
/// the CPU (`mimo prefill bank view CPU`); the device proof is the pre-ship family `bankview` (`mimo bankview GPU`).
pub const BankView = enum { full, window, invalid };

/// One tile's bank as the GEMM sees it: the arrays handed to MLX and the slots rebased in place (`slots` sorted).
const ViewFn = fn (g: *ops.MlxOps, bank: Arrays, slots: []u32) anyerror!Arrays;

pub fn bankViews(v: BankView) !*const ViewFn {
    return switch (v) {
        .full => fullView,
        .window => windowView,
        .invalid => error.MimoPrefillBankView,
    };
}

/// MLX's rule (`GatherQMM::eval_gpu`): a sorted M = 1 gather_qmm of B rows over E bank rows takes the sorted GEMM.
pub fn sortedGemm(rows: usize, bank_rows: usize) bool {
    return rows >= 16 and rows / bank_rows >= 4;
}

fn fullView(_: *ops.MlxOps, bank: Arrays, _: []u32) anyerror!Arrays {
    return bank;
}

fn windowView(g: *ops.MlxOps, bank: Arrays, slots: []u32) anyerror!Arrays {
    if (!sortedGemm(slots.len, @intCast(g.shapeOf(bank.gate.w).d[0]))) return bank;
    const lo = slots[0];
    const hi = slots[slots.len - 1] + 1;
    for (slots) |*s| s.* -= lo;
    return .{ .gate = try windowRows(g, bank.gate, lo, hi), .up = try windowRows(g, bank.up, lo, hi), .down = try windowRows(g, bank.down, lo, hi) };
}

/// Bank rows [lo, hi) of one projection: a leading-axis slice of a row-contiguous array (MLX `slice`: a shared-buffer
/// view at an offset, still row-contiguous, so `ensure_row_contiguous` passes it through uncopied).
fn windowRows(g: *ops.MlxOps, p: Proj, lo: u32, hi: u32) !Proj {
    const ws = g.shapeOf(p.w);
    const ss = g.shapeOf(p.scales.?);
    const l: c_int = @intCast(lo);
    const h: c_int = @intCast(hi);
    return .{
        .w = try g.slice(p.w, &.{ l, 0, 0 }, &.{ h, ws.d[1], ws.d[2] }, &.{ 1, 1, 1 }),
        .scales = try g.slice(p.scales.?, &.{ l, 0, 0 }, &.{ h, ss.d[1], ss.d[2] }, &.{ 1, 1, 1 }),
    };
}

/// The pipelined route's second tile in flight: one more projection tile and one more reduction tile, at
/// `source.workspaceBytes`'s per-row rates (`matrix_tile`, `reduction_tile`), for a call of `tokens` tokens.
pub fn pipelinedExtraBytes(hidden: u64, inter: u64, tokens: u32) u64 {
    const nr = @as(u64, tokens) * top_k;
    return @min(nr, projection_rows) * (20 * hidden + 24 * inter + 32) +
        @as(u64, @min(tokens, tile_rows)) * (top_k * hidden * 8 + hidden * 8 + top_k * 8);
}

/// X19 (`predict_router`): which router weight the decode lookahead predictor reads. `f32` (the control): the transposed
/// F32 copy through `Math.predictorScores` (MLX's gemv at M = 1, gather_mv per row at M > 1). `bf16`: the router's BF16
/// weight through `glue.RouterRows` (each row the M = 1 product: bit-equal to the control at M = 1; at M > 1 the M = 1
/// numerics instead of gather_mv's order), then the same sigmoid and correction. Token-exact: the scores choose only the
/// lookahead's pre-reads (`Stream.route`'s hint), never an expert, a weight or a slot of a forward.
pub const PredictRoute = enum { f32, bf16, invalid };

pub const Predictor = struct {
    rows: glue.RouterRows = .{},
    scores: *const fn (*const Predictor, *ops.MlxOps, T, T, T) anyerror!T = scoresF32,

    /// Construction: `.bf16` builds the kernel object and its per-M configs (hidden 4096, 256 experts only).
    pub fn install(route: PredictRoute, geometry: bank_mod.Geometry) !Predictor {
        return switch (route) {
            .f32 => .{},
            .bf16 => .{ .rows = try glue.RouterRows.init(geometry.hidden_size, geometry.num_experts), .scores = scoresBf16 },
            .invalid => error.MimoPredictRouter,
        };
    }

    pub fn deinit(p: *Predictor) void {
        p.rows.deinit();
    }

    fn scoresF32(_: *const Predictor, g: *ops.MlxOps, x: T, router: T, correction: T) anyerror!T {
        return Math.predictorScores(g, x, router, correction);
    }

    /// `router`: the next layer's BF16 [256, 4096] weight (the gate table `Source.validateGates` checked for this route).
    fn scoresBf16(p: *const Predictor, g: *ops.MlxOps, x: T, router: T, correction: T) anyerror!T {
        const logits = try g.adopt(try p.rows.apply(x, router, g.s));
        return g.add(try g.sigmoid(logits), correction);
    }
};

pub const Math = struct {
    /// The decode lane's widest call: the MTP lane's verify (t1 + 3 drafts, `draft.verify_rows`), 4 x 8 = 32 routed
    /// ids, one route (`max_route_ids` 48; construction refuses a wider bound). `Source.decodeLane` sends a decode-phase
    /// call of at most these rows here: the fused texts (P <= 48 pairs, combine n <= 4), the hit pass, the decode
    /// policy's admissions, the lookahead and its pre-read, each miss read once for the rows' union.
    pub const decode_forward_rows: u32 = 4;
    /// The scheduler's `routed` takes the wide lane from here (M >= 5). The prompt's M = 2..4 chunks take the wide
    /// lane through `routedParts` (`Source.forward`'s phase route), so prompt calls are unchanged.
    pub const prefill_min_tokens: u32 = decode_forward_rows + 1;
    pub const wide_event_gates = true;
    pub const predictor_decode_only = true;
    pub const Profile = @import("math.zig").Profile;
    /// Speculative hints only. Source installs borrowed F32 transposed routers
    /// and correction vectors once; authoritative routing stays in Module.
    /// BARRIER (mimo/barrier-residual): on MLX, a call of M > 1 rows (the MTP verify's) takes `predictorRows`, MLX's
    /// gathered GEMV, in place of the [M, 4096] x [4096, 256] F32 GEMM; M = 1 (the serial token) keeps this graph op
    /// for op. The scores only choose the lookahead's pre-reads (never an expert, a weight or a slot of the decode), so
    /// the tokens are unchanged by construction; a verify row's hint may differ from the GEMM's in its last bits.
    pub fn predictorScores(g: anytype, x: @TypeOf(g.*).T, router: @TypeOf(g.*).T, correction: @TypeOf(g.*).T) !@TypeOf(g.*).T {
        if (comptime @TypeOf(g.*) == ops.MlxOps) {
            const m = g.shapeOf(x).dim(0);
            if (m > 1) return g.add(try g.sigmoid(try predictorRows(g, try g.astype(x, .float32), router, m)), correction);
        }
        return g.add(try g.sigmoid(try g.matmul(try g.astype(x, .float32), router)), correction);
    }

    /// `xf [M, K] @ router [K, N]` as M one-row products in one dispatch: `gather_mm` with a batch of M single rows
    /// against the one router (index 0) routes to MLX's `gather_mv` (`gemv_t_gather`, the M = 1 product's `gemv_t`
    /// blocking per row: 32 threadgroups at N = 256) where the M x 256 GEMM runs 8 output tiles down K = 4,096. The
    /// m4lever1 cells measured that GEMM's predictor at ~62 us of each routed layer's barrier set at M = 4 (layer 47,
    /// which has no next router, ends its barrier set 62 us sooner than every other full layer).
    pub fn predictorRows(g: *ops.MlxOps, xf: T, router: T, m: c_int) !T {
        const k = g.shapeOf(xf).dim(1);
        const n = g.shapeOf(router).dim(1);
        const rows = try g.reshape(xf, &.{ m, 1, k });
        const one = try g.reshape(router, &.{ 1, k, n });
        const zero: [1]u32 = .{0};
        const rhs = try g.hostArray(std.mem.sliceAsBytes(&zero), &.{1}, .uint32);
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_gather_mm(&out, rows, one, .{}, rhs, false, g.s)) catch |err| {
            _ = mlx.mlx_array_free(out);
            return err;
        };
        return g.reshape(try g.adopt(out), &.{ m, n });
    }
    /// X19: the scheduler's decode predictor (`ExpertsWith.scoresOf`) through the installed route.
    pub fn predictorScoresBound(self: *const Math, g: *ops.MlxOps, x: T, router: T, correction: T) !T {
        return self.predictor.scores(&self.predictor, g, x, router, correction);
    }

    pub fn Projection(comptime A: type) type {
        return quant.GatherQmm.Arrays(A);
    }

    /// Native MXFP4 bindings are validated once by Source. The event aliases
    /// retain exactly the packed weights and proven-present U8 exponents.
    pub fn EventProjection(comptime A: type) type {
        return struct {
            pub const count: usize = 2;
            pub fn inputs(p: Projection(A)) [count]A {
                return .{ p.w, p.scales.? };
            }
            pub fn fromWaited(out: [count]A) Projection(A) {
                return .{ .w = out[0], .scales = out[1], .biases = null };
            }
        };
    }

    allocator: std.mem.Allocator,
    hidden: c_int,
    inter: c_int,
    activation: Activation,
    /// Only the projection's row packing lives here. The shared scheduler
    /// supplies the expert group and restores the full call's routed order.
    order: []u32,
    inverse: []u32,
    kept: std.ArrayList(T),
    zero_rhs: T,
    /// L5: the fused decode texts, installed at construction with `Options.banked` (moe.Route.fused); null on the
    /// stock route, whose scheduler never binds the banked waves.
    fused: ?moe.Fused = null,
    /// The prompt pass's tile drains (`Drain`), bound at construction (`Source.Options.prefill_drain`); sync = the control.
    drain: Drains = .{ .tiles = projectTilesSync, .reduce = reduceTilesSync },
    /// The expert tiles' bank view (`BankView`), bound at construction (`Source.Options.prefill_bank_view`); full = the control.
    bank_view: *const ViewFn = fullView,
    /// X19 (`predict_router`): the decode lookahead predictor's route, bound at construction (`Source.Options.predict_router`);
    /// the control (`.f32`) is `predictorScores` on the transposed F32 router copy.
    predictor: Predictor = .{},

    pub fn init(a: std.mem.Allocator, geometry: bank_mod.Geometry, activation: Activation) !Math {
        const order = try a.alloc(u32, max_rows);
        errdefer a.free(order);
        const inverse = try a.alloc(u32, max_rows);
        errdefer a.free(inverse);
        var kept = try std.ArrayList(T).initCapacity(a, (max_rows + tile_rows - 1) / tile_rows + route_group_rows);
        errdefer kept.deinit(a);
        const zero = [_]u32{0};
        const shape = [_]c_int{1};
        const zero_rhs = mlx.mlx_array_new_data(std.mem.sliceAsBytes(&zero).ptr, &shape, shape.len, .uint32);
        if (zero_rhs.ctx == null) return error.MlxError;
        return .{ .allocator = a, .hidden = @intCast(geometry.hidden_size), .inter = @intCast(geometry.intermediate_size), .activation = activation, .order = order, .inverse = inverse, .kept = kept, .zero_rhs = zero_rhs };
    }

    pub fn deinit(self: *Math, g: *ops.MlxOps) void {
        if (self.fused) |*f| f.deinit();
        self.predictor.deinit();
        for (self.kept.items) |x| g.release(x);
        g.release(self.zero_rhs);
        self.kept.deinit(self.allocator);
        self.allocator.free(self.order);
        self.allocator.free(self.inverse);
    }

    /// Required decode surface of the shared wrapper. Source.forward keeps
    /// its existing one-token route; the installed prefill has no cold lane.
    pub fn gateUp(self: *const Math, g: *ops.MlxOps, x: T, ids: T, gate: Proj, up: Proj) !T {
        const n = g.shapeOf(x).d[0];
        const repeated = try g.reshape(x, &.{ n, 1, self.hidden });
        const gg = try g.reshape(try g.gatherQmm(repeated, gate.w, gate.scales.?, ids, .mxfp4), &.{ n, self.inter });
        const uu = try g.reshape(try g.gatherQmm(repeated, up.w, up.scales.?, ids, .mxfp4), &.{ n, self.inter });
        return self.activation(g, gg, uu);
    }

    pub fn down(self: *const Math, g: *ops.MlxOps, h: T, ids: T, d: Proj) !T {
        const n = g.shapeOf(h).d[0];
        return g.reshape(try g.gatherQmm(try g.reshape(h, &.{ n, 1, self.inter }), d.w, d.scales.?, ids, .mxfp4), &.{ n, self.hidden });
    }

    /// L5: the shared scheduler's banked waves (bound by `Options.banked`, only with `fused` installed): every pair of
    /// a wave in one text over the three banks, ids packed `bank << 24 | row`, x [P, H] one row per pair.
    pub const has_banked = true;
    /// Stock HITPASS misses read base row zero; the scheduler never joins those device rows.
    pub const devroute_missing_id: u32 = 0;
    pub fn devrouteMissingId(_: *const Math) u32 {
        return devroute_missing_id;
    }

    pub fn gateUpBanked(self: *const Math, g: *ops.MlxOps, x: T, ids: T, banks: *const [moe.n_banks]Arrays) !T {
        return self.fused.?.gateUp(g, x, ids, banks);
    }

    /// The per-pair down rows [P, H] bf16 (the scheduler's join and the arch's combine follow).
    pub fn downBanked(self: *const Math, g: *ops.MlxOps, h: T, ids: T, banks: *const [moe.n_banks]Arrays) !T {
        return self.fused.?.down(g, h, ids, banks);
    }

    /// x is [1,count,K], so the existing gather operation sees the actual
    /// token count as matrix M and a single RHS expert bank row.
    fn matrix(self: *const Math, g: *ops.MlxOps, x: T, p: Proj, count: c_int, _: c_int, out: c_int) !T {
        return g.reshape(try g.gatherQmm(x, p.w, p.scales.?, self.zero_rhs, .mxfp4), &.{ count, out });
    }

    /// Returns evaluated BF16 rows in the supplied group order, kept across
    /// backend resets. Sorted projection graphs span experts and are bounded
    /// by projection_rows, separately from the source's weighted reductions.
    pub fn prefill(self: *Math, g: *ops.MlxOps, _: u32, x: T, rows: quant.PrefillRows, bank: Arrays) !T {
        const n = rows.slot.len;
        defer {
            for (self.kept.items) |y| g.release(y);
            self.kept.clearRetainingCapacity();
        }
        try self.drain.tiles(self, g, x, rows, bank, &self.kept);
        for (self.order[0..n], 0..) |o, i| self.inverse[o] = @intCast(i);
        const join_start = profile.now();
        const mark = g.mark();
        const joined = try g.concat(self.kept.items, 0);
        try g.evalAll(&.{joined});
        const held_joined = g.keep(joined);
        defer g.release(held_joined);
        for (self.kept.items) |y| g.release(y);
        self.kept.clearRetainingCapacity();
        g.resetTo(mark);
        const inverse = try g.hostArray(std.mem.sliceAsBytes(self.inverse[0..n]), &.{@intCast(n)}, .uint32);
        const result = try g.take(held_joined, inverse, 0);
        try g.evalAll(&.{result});
        const held = g.keep(result);
        g.resetTo(mark);
        profile.charge(.join, join_start);
        return held;
    }

    /// +join: the shared scheduler's unjoined route (`routedParts`) takes this.
    pub const has_parts = true;

    /// +join: `prefill`'s tiles as they are (no concatenate, no inverse take),
    /// each tile's rows' indices into `rows` appended to `pos` in tile order.
    /// The tiles are the same graphs over the same rows as `prefill`'s.
    pub fn prefillParts(self: *Math, g: *ops.MlxOps, _: u32, x: T, rows: quant.PrefillRows, bank: Arrays, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
        const n = rows.slot.len;
        try outs.ensureUnusedCapacity(alloc, (n + projection_rows - 1) / projection_rows);
        try pos.ensureUnusedCapacity(alloc, n);
        try self.drain.tiles(self, g, x, rows, bank, outs);
        pos.appendSliceAssumeCapacity(self.order[0..n]);
    }

    /// The stable slot packing (`self.order`): the call's rows sorted by slot (the same packing as the SDK's
    /// gather-matmul prefill), the bounded sorted gathers then span the group's physical slots.
    fn packRows(self: *Math, rows: quant.PrefillRows) []u32 {
        const n = rows.slot.len;
        const order = self.order[0..n];
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.sort.block(u32, order, rows.slot, struct {
            fn lt(slots: []const u32, a: u32, b: u32) bool {
                return slots[a] < slots[b];
            }
        }.lt);
        return order;
    }

    /// One projection tile's graph over packed rows [start, stop): take, gate, up, activation, down (unevaluated).
    fn tileGraph(self: *Math, g: *ops.MlxOps, x: T, rows: quant.PrefillRows, bank: Arrays, order: []const u32, start: usize, stop: usize) !T {
        const encode_start = profile.now();
        var source_rows: [projection_rows]u32 = undefined;
        var slot_rows: [projection_rows]u32 = undefined;
        for (order[start..stop], source_rows[0 .. stop - start], slot_rows[0 .. stop - start]) |o, *src, *slot| {
            src.* = rows.act_row.?[o];
            slot.* = rows.slot[o];
        }
        const view = try self.bank_view(g, bank, slot_rows[0 .. stop - start]);
        if (comptime timeline.enabled) timeline.tile(.moe, @intCast(stop - start), @intCast(g.shapeOf(view.gate.w).d[0]), slot_rows[0 .. stop - start]);
        const count: c_int = @intCast(stop - start);
        const src = try g.hostArray(std.mem.sliceAsBytes(source_rows[0 .. stop - start]), &.{count}, .uint32);
        const rhs = try g.hostArray(std.mem.sliceAsBytes(slot_rows[0 .. stop - start]), &.{count}, .uint32);
        const act = try g.take(x, src, 0);
        const gg = try sortedProjection(g, act, view.gate, rhs);
        const uu = try sortedProjection(g, act, view.up, rhs);
        const h = try self.activation(g, gg, uu);
        const y = try sortedProjection(g, h, view.down, rhs);
        profile.charge(.encode, encode_start);
        return y;
    }

    /// `Drain.sync` (the control): each bounded sorted projection tile evaluated, synchronized and appended KEPT to
    /// `kept` (capacity reserved by the caller) before the next tile's graph is built.
    fn projectTilesSync(self: *Math, g: *ops.MlxOps, x: T, rows: quant.PrefillRows, bank: Arrays, kept: *std.ArrayList(T)) !void {
        const n = rows.slot.len;
        const order = self.packRows(rows);
        var start: usize = 0;
        while (start < n) {
            const stop = @min(start + projection_rows, n);
            const mark = g.mark();
            const y = try self.tileGraph(g, x, rows, bank, order, start, stop);
            const drain_start = profile.now();
            try g.evalAll(&.{y});
            try mlx.check(mlx.mlx_synchronize(g.s));
            profile.charge(.drain, drain_start);
            kept.appendAssumeCapacity(g.keep(y));
            g.resetTo(mark);
            start = stop;
        }
    }

    /// `Drain.pipelined`: the same tiles, each committed (async eval) and appended KEPT, then the tile before it waited:
    /// the GPU holds tile i while the host builds tile i + 1 (at most two in flight). The last tile is waited and the
    /// stream synchronized before the call returns, as the control leaves it.
    fn projectTilesPipelined(self: *Math, g: *ops.MlxOps, x: T, rows: quant.PrefillRows, bank: Arrays, kept: *std.ArrayList(T)) !void {
        const n = rows.slot.len;
        const order = self.packRows(rows);
        var start: usize = 0;
        var prev: ?T = null;
        while (start < n) {
            const stop = @min(start + projection_rows, n);
            const mark = g.mark();
            const y = try self.tileGraph(g, x, rows, bank, order, start, stop);
            try g.asyncEval(&.{y});
            const held = g.keep(y);
            kept.appendAssumeCapacity(held);
            g.resetTo(mark);
            if (prev) |p| {
                const drain_start = profile.now();
                try g.evalAll(&.{p});
                profile.charge(.drain, drain_start);
            }
            prev = held;
            start = stop;
        }
        if (prev) |p| {
            const drain_start = profile.now();
            try g.evalAll(&.{p});
            try mlx.check(mlx.mlx_synchronize(g.s));
            profile.charge(.drain, drain_start);
        }
    }

    /// Every matrix tile was evaluated and synchronized before its graph
    /// was released. The scheduler still performs its existing group drain.
    pub fn finishPrefill(_: *Math, _: *ops.MlxOps) !void {}
};

/// The prompt pass's per-token routed combine (Source.forward, logical M >= 2): the unjoined tile outputs
/// (`routedParts`); per gather tile of `combine_tokens` tokens, each source's rows taken in place, concatenated and
/// permuted into [count, 8, H] (no [n, 8, H] array is built), then per tile of `tile_rows` tokens f32(row), times its f32
/// route weight (a separate multiply), MLX's sum over the 8 routes (axis 1, fold order k = 0..7), BF16 once.

pub const top_k: u32 = 8;
/// Tokens per joinless gather tile (a multiple of `tile_rows`).
pub const combine_tokens: u32 = 4 * tile_rows;
const max_sources = @import("../deepseek_v41/experts.zig").joinless_sources;
const max_combine_tiles = (@import("config.zig").prefill_max_tokens + tile_rows - 1) / tile_rows;
comptime {
    if (combine_tokens % tile_rows != 0) @compileError("mimo_v2: a gather tile must hold whole reduction tiles");
}

const CombineOutputs = struct {
    items: [max_combine_tiles]T = undefined,
    n: usize = 0,

    fn release(o: *CombineOutputs, g: *ops.MlxOps) void {
        for (o.items[0..o.n]) |x| g.release(x);
        o.n = 0;
    }
};

/// One reduction tile's graph: tokens [t, t + count) of `rows` (first row token `rows_token`), unevaluated.
fn reduceGraph(g: *ops.MlxOps, rows: T, rows_token: u32, route_weights: T, t: u32, count: u32, hidden: c_int) !T {
    if (comptime timeline.enabled) timeline.tile(.reduce, count, 0, &.{});
    const local = t - rows_token;
    const encode_start = profile.now();
    const weights = try g.slice(route_weights, &.{ @intCast(t), 0 }, &.{ @intCast(t + count), top_k }, &.{ 1, 1 });
    const expanded = try g.reshape(weights, &.{ @intCast(count), top_k, 1 });
    const down = try g.slice(rows, &.{ @intCast(local), 0, 0 }, &.{ @intCast(local + count), top_k, hidden }, &.{ 1, 1, 1 });
    const expert_out = try g.astype(down, .float32);
    const output = try g.astype(try g.sum(try g.mul(expert_out, expanded), 1, false), .bfloat16);
    profile.charge(.encode, encode_start);
    return output;
}

/// `Drain.pipelined`'s reduction tiles: each committed and kept, then the one before it waited; the last waited and
/// the stream synchronized before returning (the same outputs as `reduceTilesSync`).
fn reduceTilesPipelined(g: *ops.MlxOps, rows: T, rows_token: u32, route_weights: T, token: u32, stop: u32, hidden: c_int, out: *CombineOutputs) !void {
    var t = token;
    var prev: ?T = null;
    while (t < stop) {
        const count = @min(tile_rows, stop - t);
        const tile_mark = g.mark();
        const output = try reduceGraph(g, rows, rows_token, route_weights, t, count, hidden);
        try g.asyncEval(&.{output});
        const held = g.keep(output);
        out.items[out.n] = held;
        out.n += 1;
        g.resetTo(tile_mark);
        if (prev) |p| {
            const drain_start = profile.now();
            try g.evalAll(&.{p});
            profile.charge(.drain, drain_start);
        }
        prev = held;
        t += count;
    }
    if (prev) |p| {
        const drain_start = profile.now();
        try g.evalAll(&.{p});
        try mlx.check(mlx.mlx_synchronize(g.s));
        profile.charge(.drain, drain_start);
    }
}

/// `Drain.sync` (the control): tokens [token, stop) of `rows` ([.., 8, H] evaluated BF16 whose first row is
/// token `rows_token`), one evaluated, synchronized BF16 [count, H] per tile.
fn reduceTilesSync(g: *ops.MlxOps, rows: T, rows_token: u32, route_weights: T, token: u32, stop: u32, hidden: c_int, out: *CombineOutputs) !void {
    var t = token;
    while (t < stop) {
        const count = @min(tile_rows, stop - t);
        const tile_mark = g.mark();
        const output = try reduceGraph(g, rows, rows_token, route_weights, t, count, hidden);
        const drain_start = profile.now();
        try g.evalAll(&.{output});
        try mlx.check(mlx.mlx_synchronize(g.s));
        profile.charge(.drain, drain_start);
        out.items[out.n] = g.keep(output);
        out.n += 1;
        g.resetTo(tile_mark);
        t += count;
    }
}

/// The tiles' outputs joined [nt, H] (evaluated, KEPT).
fn joinOutputs(g: *ops.MlxOps, out: *CombineOutputs) !T {
    if (comptime timeline.enabled) timeline.stage(.output);
    const output_join_start = profile.now();
    const joined = try g.concat(out.items[0..out.n], 0);
    try g.evalAll(&.{joined});
    profile.charge(.join, output_join_start);
    return g.keep(joined);
}

/// The joined reference (the CPU tests' control): `ordered` is the evaluated [nt, 8, H] BF16 routed rows.
/// `reduce`: the installed drain's reduction driver (`Math.drain.reduce`).
pub fn combineJoined(g: *ops.MlxOps, ordered: T, route_weights: T, nt: u32, hidden: c_int, reduce: *const ReduceFn) !T {
    var out: CombineOutputs = .{};
    defer out.release(g);
    try reduce(g, ordered, 0, route_weights, 0, nt, hidden, &out);
    return joinOutputs(g, &out);
}

/// The joinless combine: `outs` the evaluated sources (`routedParts`),
/// `loc` each routed position's (source, row) as int32 pairs, position
/// p = token * 8 + k. Each gather tile builds exactly the control's
/// [count, 8, H] rows (copies of the same words), then the same reductions.
pub fn combineParts(g: *ops.MlxOps, outs: []const T, loc: []const i32, route_weights: T, nt: u32, hidden: c_int, reduce: *const ReduceFn) !T {
    std.debug.assert(outs.len > 0 and outs.len <= max_sources and loc.len == 2 * @as(usize, nt) * top_k);
    var out: CombineOutputs = .{};
    defer out.release(g);
    var idx: [combine_tokens * top_k]u32 = undefined;
    var perm: [combine_tokens * top_k]u32 = undefined;
    var token: u32 = 0;
    while (token < nt) {
        const stop = @min(token + combine_tokens, nt);
        const m: usize = (stop - token) * top_k;
        const tile_loc = loc[2 * @as(usize, token) * top_k ..][0 .. 2 * m];
        if (comptime timeline.enabled) timeline.stage(.gather);
        const join_start = profile.now();
        const mark = g.mark();
        // Source-major gather lists (assignment order within a source) and
        // each assignment's row in their concatenate.
        var counts: [max_sources]u32 = @splat(0);
        for (0..m) |i| counts[@intCast(tile_loc[2 * i])] += 1;
        var fill: [max_sources]u32 = undefined;
        var at: u32 = 0;
        var used: usize = 0;
        var only: usize = 0;
        for (counts[0..outs.len], fill[0..outs.len], 0..) |c, *f, s| {
            f.* = at;
            at += c;
            if (c > 0) {
                used += 1;
                only = s;
            }
        }
        for (0..m) |i| {
            const s: usize = @intCast(tile_loc[2 * i]);
            idx[fill[s]] = @intCast(tile_loc[2 * i + 1]);
            perm[i] = fill[s];
            fill[s] += 1;
        }
        const rows = if (used == 1) blk: {
            // One source: its rows already in assignment order.
            const ia = try g.hostArray(std.mem.sliceAsBytes(idx[0..m]), &.{@intCast(m)}, .uint32);
            break :blk try g.take(outs[only], ia, 0);
        } else blk: {
            var pieces: [max_sources]T = undefined;
            var np: usize = 0;
            var start: u32 = 0;
            for (counts[0..outs.len], 0..) |c, s| {
                if (c == 0) continue;
                const ia = try g.hostArray(std.mem.sliceAsBytes(idx[start..][0..c]), &.{@intCast(c)}, .uint32);
                pieces[np] = try g.take(outs[s], ia, 0);
                np += 1;
                start += c;
            }
            const pa = try g.hostArray(std.mem.sliceAsBytes(perm[0..m]), &.{@intCast(m)}, .uint32);
            break :blk try g.take(try g.concat(pieces[0..np], 0), pa, 0);
        };
        const gathered = try g.reshape(rows, &.{ @intCast(stop - token), top_k, hidden });
        try g.evalAll(&.{gathered});
        const held = g.keep(gathered);
        defer g.release(held);
        g.resetTo(mark);
        profile.charge(.join, join_start);
        try reduce(g, held, token, route_weights, token, stop, hidden, &out);
        token = stop;
    }
    return joinOutputs(g, &out);
}

test "mimo event gates: actual native projection adapter preserves optional scale contract" {
    const Adapter = Math.EventProjection(u32);
    const p: Math.Projection(u32) = .{ .w = 7, .scales = 11, .biases = null };
    try std.testing.expectEqual(@as(usize, 2), Adapter.count);
    try std.testing.expectEqualSlices(u32, &.{ 7, 11 }, &Adapter.inputs(p));
    const waited = Adapter.fromWaited(.{ 17, 19 });
    try std.testing.expectEqual(@as(u32, 17), waited.w);
    try std.testing.expectEqual(@as(u32, 19), waited.scales.?);
    try std.testing.expect(waited.biases == null);
    try std.testing.expect(Math.wide_event_gates);
}

// One standalone dense numerical gate; no model files or reader pool.
fn matrixProbeNext(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}

fn matrixProbeProjection(a: std.mem.Allocator, g: *ops.MlxOps, in: u32, out: u32, seed: u32) !Proj {
    const words = try a.alloc(u32, out * (in / 8));
    defer a.free(words);
    const scales = try a.alloc(u8, out * (in / 32));
    defer a.free(scales);
    var state = seed;
    // Every word contains eight native signed FP4 codes. Scale exponents
    // vary per group of 32, rather than repeating sparse fixture rows.
    for (words) |*word| word.* = matrixProbeNext(&state);
    for (scales) |*scale| scale.* = @intCast(117 + matrixProbeNext(&state) % 5);
    return .{
        .w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ 1, @intCast(out), @intCast(in / 8) }, .uint32),
        .scales = try g.hostArray(scales, &.{ 1, @intCast(out), @intCast(in / 32) }, .uint8),
    };
}

fn matrixProbeInput(a: std.mem.Allocator, g: *ops.MlxOps, rows: u32, width: u32, seed: u32) !T {
    const values = try a.alloc(f32, rows * width);
    defer a.free(values);
    var state = seed;
    for (values) |*value| {
        const bits = matrixProbeNext(&state);
        const exponent = 125 + ((bits >> 21) & 3);
        value.* = @bitCast((bits & 0x80000000) | (exponent << 23) | (((bits >> 10) & 127) << 16));
    }
    return g.astype(try g.hostArray(std.mem.sliceAsBytes(values), &.{ @intCast(rows), @intCast(width) }, .float32), .bfloat16);
}

fn matrixProbeActivation(g: *ops.MlxOps, gate: T, up: T) !T {
    const f = try g.astype(gate, .float32);
    const silu = try g.astype(try g.silu(f), .bfloat16);
    return g.mul(silu, up);
}

fn matrixProbeGemv(g: *ops.MlxOps, x: T, p: Proj, rhs: T, rows: c_int, in: c_int, out: c_int) !T {
    const vectors = try g.reshape(x, &.{ rows, 1, in });
    return g.reshape(try g.gatherQmm(vectors, p.w, p.scales.?, rhs, .mxfp4), &.{ rows, out });
}

fn matrixProbeDigest(g: *ops.MlxOps, bank: Arrays) ![32]u8 {
    try g.evalAll(&.{ bank.gate.w, bank.gate.scales.?, bank.up.w, bank.up.scales.?, bank.down.w, bank.down.scales.? });
    var hash = std.crypto.hash.Blake3.init(.{});
    for ([_]Proj{ bank.gate, bank.up, bank.down }) |p| {
        const words = mlx.mlx_array_data_uint32(p.w) orelse return error.MimoMatrixData;
        const scales = mlx.mlx_array_data_uint8(p.scales.?) orelse return error.MimoMatrixData;
        hash.update(std.mem.sliceAsBytes(words[0..@intCast(mlx.mlx_array_size(p.w))]));
        hash.update(scales[0..@intCast(mlx.mlx_array_size(p.scales.?))]);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}

fn matrixProbeAgreement(a: std.mem.Allocator, g: *ops.MlxOps, name: []const u8, actual: T, reference: T) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(reference), mlx.getShape(actual));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(actual));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference));
    const n: usize = @intCast(mlx.mlx_array_size(actual));
    const got = try a.alloc(f32, n);
    defer a.free(got);
    const want = try a.alloc(f32, n);
    defer a.free(want);
    _ = try g.hostF32(try g.astype(actual, .float32), got);
    _ = try g.hostF32(try g.astype(reference, .float32), want);
    var max_abs: f64 = 0;
    var diff_squared: f64 = 0;
    var reference_squared: f64 = 0;
    var actual_squared: f64 = 0;
    var dot: f64 = 0;
    for (got, want) |left, right| {
        if (!std.math.isFinite(left) or !std.math.isFinite(right)) return error.MimoMatrixNonFinite;
        const l: f64 = left;
        const r: f64 = right;
        const diff = l - r;
        max_abs = @max(max_abs, @abs(diff));
        diff_squared += diff * diff;
        reference_squared += r * r;
        actual_squared += l * l;
        dot += l * r;
    }
    if (reference_squared == 0 or actual_squared == 0) return error.MimoMatrixDegenerateProbe;
    const relative_l2 = @sqrt(diff_squared / reference_squared);
    const cosine = dot / @sqrt(actual_squared * reference_squared);
    std.debug.print("MIMO_MATRIX_NUMERICS stage={s} values={d} max_abs={d:.9} relative_l2={d:.9} cosine={d:.9}\n", .{ name, n, max_abs, relative_l2, cosine });
    if (relative_l2 > 0.01) return error.MimoMatrixAgreement;
}

test "mimo sdk2 GPU: dense MXFP4 matrix projections agree with gathered GEMV" {
    _ = std.c.getenv("MIMO_Q4_MATRIX") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const rows: u32 = 64;
    const H: u32 = 4096;
    const I: u32 = 2048;
    const geometry = try bank_mod.Geometry.init(H, I, 1, 1, 1);
    const ceiling: u64 = 2 << 30;
    const cache_bytes: u64 = 256 << 20;
    const compiler_bytes: u64 = 512 << 20;
    const host_bytes: u64 = 128 << 20;
    const array_bytes: u64 = 96 << 20;
    // Four record payloads cover host construction plus MLX storage/copies;
    // explicit arrays include every kept output, cast, metric and host index.
    const bound = 4 * geometry.record_bytes + array_bytes + cache_bytes + compiler_bytes + host_bytes;
    std.debug.print("MIMO_MATRIX static: H={d} I={d} rows={d} record={d} arrays={d} cache={d} compiler={d} host={d} bound={d} ceiling={d}\n", .{ H, I, rows, geometry.record_bytes, array_bytes, cache_bytes, compiler_bytes, host_bytes, bound, ceiling });
    if (bound > ceiling) return error.MimoMatrixAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, cache_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    @import("sdk").memory.startFootprintInterval();
    {
        var g = try ops.MlxOps.init(a, s);
        defer g.deinit();
        var math = try Math.init(a, geometry, matrixProbeActivation);
        defer {
            _ = mlx.mlx_synchronize(s);
            math.deinit(&g);
        }
        const bank: Arrays = .{
            .gate = try matrixProbeProjection(a, &g, H, I, 0x7a02d195),
            .up = try matrixProbeProjection(a, &g, H, I, 0xc58143af),
            .down = try matrixProbeProjection(a, &g, I, H, 0x190bd6e3),
        };
        const params: quant.GatherQmm.Params = .{ .mode = .mxfp4, .bits = 4, .group_size = 32 };
        var diag: quant.Diag = .{};
        try quant.GatherQmm.checkArrays(ops.MlxOps, &g, &params, .gate, bank.gate, H, I, &diag);
        try quant.GatherQmm.checkArrays(ops.MlxOps, &g, &params, .up, bank.up, H, I, &diag);
        try quant.GatherQmm.checkArrays(ops.MlxOps, &g, &params, .down, bank.down, I, H, &diag);
        for ([_]Proj{ bank.gate, bank.up, bank.down }) |p| {
            try std.testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(p.w));
            try std.testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(p.scales.?));
        }
        const digest_before = try matrixProbeDigest(&g, bank);
        const x = try matrixProbeInput(a, &g, rows, H, 0x38abd247);
        const h = try matrixProbeInput(a, &g, rows, I, 0xe970531b);
        const zeros: [rows]u32 = @splat(0);
        const rhs = try g.hostArray(std.mem.sliceAsBytes(&zeros), &.{rows}, .uint32);
        const repeated_x = try g.reshape(x, &.{ 1, rows, H });
        const gate = try math.matrix(&g, repeated_x, bank.gate, rows, H, I);
        const gate_ref = try matrixProbeGemv(&g, x, bank.gate, rhs, rows, H, I);
        try matrixProbeAgreement(a, &g, "gate", gate, gate_ref);
        const up = try math.matrix(&g, repeated_x, bank.up, rows, H, I);
        const up_ref = try matrixProbeGemv(&g, x, bank.up, rhs, rows, H, I);
        try matrixProbeAgreement(a, &g, "up", up, up_ref);
        const down = try math.matrix(&g, try g.reshape(h, &.{ 1, rows, I }), bank.down, rows, I, H);
        const down_ref = try matrixProbeGemv(&g, h, bank.down, rhs, rows, I, H);
        try matrixProbeAgreement(a, &g, "down", down, down_ref);
        const activation = try matrixProbeActivation(&g, gate, up);
        const activation_ref = try matrixProbeActivation(&g, gate_ref, up_ref);
        const composed = try math.matrix(&g, try g.reshape(activation, &.{ 1, rows, I }), bank.down, rows, I, H);
        const composed_ref = try matrixProbeGemv(&g, activation_ref, bank.down, rhs, rows, I, H);
        try matrixProbeAgreement(a, &g, "silu_gate_times_up_down", composed, composed_ref);
        const digest_after = try matrixProbeDigest(&g, bank);
        try std.testing.expectEqualSlices(u8, &digest_before, &digest_after);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = @import("sdk").memory.processMemory();
    std.debug.print("MIMO_MATRIX peak: mlx={d} footprint_interval={d} bound={d} ceiling={d}\n", .{ peak, process.footprint_interval_peak, bound, ceiling });
    if (peak > ceiling or process.footprint_interval_peak > ceiling) return error.MimoMatrixPeakExceeded;
}

fn sortedProbeProjection(g: *ops.MlxOps, capacity: u32, in: u32, out: u32) !Proj {
    return .{
        .w = try g.zeros(&.{ @intCast(capacity), @intCast(out), @intCast(in / 8) }, .uint32),
        .scales = try g.zeros(&.{ @intCast(capacity), @intCast(out), @intCast(in / 32) }, .uint8),
    };
}

/// Populate one already evaluated, synchronized bank row. This uses the
/// same shared-memory write contract as the SDK reader, without a host bank.
fn sortedProbeFill(p: Proj, slot: u32, in: u32, out: u32, seed: u32) !void {
    const row_words: usize = out * (in / 8);
    const row_scales: usize = out * (in / 32);
    const words = @constCast(mlx.mlx_array_data_uint32(p.w) orelse return error.MimoMatrixData);
    const scales = @constCast(mlx.mlx_array_data_uint8(p.scales.?) orelse return error.MimoMatrixData);
    var state = seed;
    for (words[@as(usize, slot) * row_words ..][0..row_words]) |*word| word.* = matrixProbeNext(&state);
    for (scales[@as(usize, slot) * row_scales ..][0..row_scales]) |*scale| scale.* = @intCast(117 + matrixProbeNext(&state) % 5);
}

fn sortedProbeDigest(bank: Arrays) ![32]u8 {
    var hash = std.crypto.hash.Blake3.init(.{});
    for ([_]Proj{ bank.gate, bank.up, bank.down }) |p| {
        const row_words: usize = @intCast(mlx.getShape(p.w)[1] * mlx.getShape(p.w)[2]);
        const row_scales: usize = @intCast(mlx.getShape(p.scales.?)[1] * mlx.getShape(p.scales.?)[2]);
        const words = mlx.mlx_array_data_uint32(p.w) orelse return error.MimoMatrixData;
        const scales = mlx.mlx_array_data_uint8(p.scales.?) orelse return error.MimoMatrixData;
        for ([_]usize{ 0, 63, 127 }) |slot| {
            hash.update(std.mem.sliceAsBytes(words[slot * row_words ..][0..row_words]));
            hash.update(scales[slot * row_scales ..][0..row_scales]);
        }
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}

fn sortedProbeExact(a: std.mem.Allocator, g: *ops.MlxOps, actual: T, expected: T) !void {
    const mark = g.mark();
    defer g.resetTo(mark);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(actual));
    const n: usize = @intCast(mlx.mlx_array_size(actual));
    const left = try a.alloc(f32, n);
    defer a.free(left);
    const right = try a.alloc(f32, n);
    defer a.free(right);
    _ = try g.hostF32(try g.astype(actual, .float32), left);
    _ = try g.hostF32(try g.astype(expected, .float32), right);
    try std.testing.expectEqualSlices(f32, right, left);
}

fn sortedProbeOutputDigest(a: std.mem.Allocator, g: *ops.MlxOps, x: T) ![32]u8 {
    const mark = g.mark();
    defer g.resetTo(mark);
    const values = try a.alloc(f32, @intCast(mlx.mlx_array_size(x)));
    defer a.free(values);
    _ = try g.hostF32(try g.astype(x, .float32), values);
    var digest: [32]u8 = undefined;
    std.crypto.hash.Blake3.hash(std.mem.sliceAsBytes(values), &digest, .{});
    return digest;
}

test "mimo sdk2 GPU: sorted MXFP4 batches span sparse slots and a short tail" {
    _ = std.c.getenv("MIMO_Q4_SORTED") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const capacity: u32 = 128;
    const rows: u32 = projection_rows + 13;
    const H: u32 = 4096;
    const I: u32 = 2048;
    const geometry = try bank_mod.Geometry.init(H, I, capacity, 1, 1);
    const ceiling: u64 = 3 << 30;
    const cache_bytes: u64 = 256 << 20;
    const compiler_bytes: u64 = 512 << 20;
    const host_bytes: u64 = 128 << 20;
    const array_bytes: u64 = 256 << 20;
    // One bank, no duplicate host bank. Scoped probes retain only inputs and
    // two evaluated output pieces; the largest graph is one 1024-row batch.
    const bound = capacity * geometry.record_bytes + 4 * geometry.record_bytes + array_bytes + cache_bytes + compiler_bytes + host_bytes;
    std.debug.print("MIMO_SORTED static: capacity={d} H={d} I={d} rows={d} bank={d} arrays={d} cache={d} compiler={d} host={d} bound={d} ceiling={d}\n", .{ capacity, H, I, rows, @as(u64, capacity) * geometry.record_bytes, array_bytes, cache_bytes, compiler_bytes, host_bytes, bound, ceiling });
    if (bound > ceiling) return error.MimoMatrixAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, cache_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    @import("sdk").memory.startFootprintInterval();
    {
        var g = try ops.MlxOps.init(a, s);
        defer g.deinit();
        var math = try Math.init(a, geometry, matrixProbeActivation);
        defer {
            _ = mlx.mlx_synchronize(s);
            math.deinit(&g);
        }
        const bank: Arrays = .{
            .gate = try sortedProbeProjection(&g, capacity, H, I),
            .up = try sortedProbeProjection(&g, capacity, H, I),
            .down = try sortedProbeProjection(&g, capacity, I, H),
        };
        try g.evalAll(&.{ bank.gate.w, bank.gate.scales.?, bank.up.w, bank.up.scales.?, bank.down.w, bank.down.scales.? });
        try mlx.check(mlx.mlx_synchronize(s));
        for ([_]u32{ 0, 63, 127 }, 0..) |slot, i| {
            try sortedProbeFill(bank.gate, slot, H, I, 0x7a02d195 + @as(u32, @intCast(i)) * 101);
            try sortedProbeFill(bank.up, slot, H, I, 0xc58143af + @as(u32, @intCast(i)) * 103);
            try sortedProbeFill(bank.down, slot, I, H, 0x190bd6e3 + @as(u32, @intCast(i)) * 107);
        }
        var diag: quant.Diag = .{};
        try quant.GatherQmm.checkArrays(ops.MlxOps, &g, &projection_params, .gate, bank.gate, H, I, &diag);
        try quant.GatherQmm.checkArrays(ops.MlxOps, &g, &projection_params, .up, bank.up, H, I, &diag);
        try quant.GatherQmm.checkArrays(ops.MlxOps, &g, &projection_params, .down, bank.down, I, H, &diag);
        const bytes_before = try sortedProbeDigest(bank);
        const input_mark = g.mark();
        const input = try matrixProbeInput(a, &g, rows, H, 0x38abd247);
        const intermediate = try matrixProbeInput(a, &g, rows, I, 0xe970531b);
        try g.evalAll(&.{ input, intermediate });
        const x = g.keep(input);
        defer g.release(x);
        const h = g.keep(intermediate);
        defer g.release(h);
        g.resetTo(input_mark);
        var slots: [rows]u32 = undefined;
        var acts: [rows]u32 = undefined;
        for (&slots, &acts, 0..) |*slot, *act, i| {
            const rank = (i * 31) % rows;
            slot.* = if (rank < 351) 0 else if (rank < 684) 63 else 127;
            act.* = @intCast(i);
        }
        // Independent stable bucketing supplies the original-row inverse.
        // Boundaries 351 and 684 cut through native 32-row matrix tiles.
        var order: [rows]u32 = undefined;
        var sorted_slots: [rows]u32 = undefined;
        var inverse: [rows]u32 = undefined;
        var position: usize = 0;
        for ([_]u32{ 0, 63, 127 }) |slot| for (slots, 0..) |id, i| {
            if (id != slot) continue;
            order[position] = @intCast(i);
            sorted_slots[position] = slot;
            inverse[i] = @intCast(position);
            position += 1;
        };
        try std.testing.expectEqual(@as(usize, rows), position);
        var pieces: [2]T = undefined;
        var used: usize = 0;
        defer for (pieces[0..used]) |piece| g.release(piece);
        var start: usize = 0;
        while (start < rows) {
            const end = @min(start + projection_rows, rows);
            const count: c_int = @intCast(end - start);
            const mark = g.mark();
            const src = try g.hostArray(std.mem.sliceAsBytes(order[start..end]), &.{count}, .uint32);
            const rhs = try g.hostArray(std.mem.sliceAsBytes(sorted_slots[start..end]), &.{count}, .uint32);
            const xs = try g.take(x, src, 0);
            const hs = try g.take(h, src, 0);
            const gate = try sortedProjection(&g, xs, bank.gate, rhs);
            const gate_ref = try matrixProbeGemv(&g, xs, bank.gate, rhs, count, H, I);
            try matrixProbeAgreement(a, &g, if (start == 0) "sorted_gate_1024" else "tail_gate_13", gate, gate_ref);
            const up = try sortedProjection(&g, xs, bank.up, rhs);
            const up_ref = try matrixProbeGemv(&g, xs, bank.up, rhs, count, H, I);
            try matrixProbeAgreement(a, &g, if (start == 0) "sorted_up_1024" else "tail_up_13", up, up_ref);
            const down = try sortedProjection(&g, hs, bank.down, rhs);
            const down_ref = try matrixProbeGemv(&g, hs, bank.down, rhs, count, I, H);
            try matrixProbeAgreement(a, &g, if (start == 0) "sorted_down_1024" else "tail_down_13", down, down_ref);
            const mid = try matrixProbeActivation(&g, gate, up);
            const mid_ref = try matrixProbeActivation(&g, gate_ref, up_ref);
            const composed = try sortedProjection(&g, mid, bank.down, rhs);
            const composed_ref = try matrixProbeGemv(&g, mid_ref, bank.down, rhs, count, I, H);
            try matrixProbeAgreement(a, &g, if (start == 0) "sorted_composed_1024" else "tail_composed_13", composed, composed_ref);
            try mlx.check(mlx.mlx_synchronize(s));
            pieces[used] = g.keep(composed);
            used += 1;
            g.resetTo(mark);
            start = end;
        }
        const compare_mark = g.mark();
        const joined = try g.concat(pieces[0..used], 0);
        const inv = try g.hostArray(std.mem.sliceAsBytes(&inverse), &.{rows}, .uint32);
        const expected = try g.take(joined, inv, 0);
        const actual = try math.prefill(&g, 0, x, .{ .slot = &slots, .act_row = &acts }, bank);
        defer g.release(actual);
        try sortedProbeExact(a, &g, actual, expected);
        const held_digest = try sortedProbeOutputDigest(a, &g, actual);
        g.resetTo(compare_mark);
        for (&acts, 0..) |*act, i| act.* = @intCast(rows - 1 - i);
        const subsequent = try math.prefill(&g, 0, x, .{ .slot = &slots, .act_row = &acts }, bank);
        defer g.release(subsequent);
        const held_after = try sortedProbeOutputDigest(a, &g, actual);
        try std.testing.expectEqualSlices(u8, &held_digest, &held_after);
        const bytes_after = try sortedProbeDigest(bank);
        try std.testing.expectEqualSlices(u8, &bytes_before, &bytes_after);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = @import("sdk").memory.processMemory();
    std.debug.print("MIMO_SORTED peak: mlx={d} footprint_interval={d} bound={d} ceiling={d}\n", .{ peak, process.footprint_interval_peak, bound, ceiling });
    if (peak > ceiling or process.footprint_interval_peak > ceiling) return error.MimoMatrixPeakExceeded;
}

// ── +join CPU proofs (synthetic shapes; MLX on the CPU stream) ──

fn joinProbeBits(a: std.mem.Allocator, g: *ops.MlxOps, x: T) ![]u32 {
    const mark = g.mark();
    defer g.resetTo(mark);
    const values = try a.alloc(f32, @intCast(mlx.mlx_array_size(x)));
    defer a.free(values);
    _ = try g.hostF32(try g.astype(x, .float32), values);
    const bits = try a.alloc(u32, values.len);
    for (values, bits) |v, *b| b.* = @bitCast(v);
    return bits;
}

/// Exact-or-name: the first differing element (row, column, both words) is printed before failing.
fn joinProbeExpectBits(name: []const u8, hidden: u32, want: []const u32, got: []const u32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, x, i| if (w != x) {
        std.debug.print("MIMO_JOIN first divergence case={s} token={d} col={d} control=0x{x:0>8} candidate=0x{x:0>8}\n", .{ name, i / hidden, i % hidden, w, x });
        return error.MimoJoinNotExact;
    };
}

fn bf16RoundBits(v: f32) u32 {
    const u: u32 = @bitCast(v);
    const r = u +% (0x7fff + ((u >> 16) & 1));
    return r & 0xffff0000;
}

/// The control's join and combine against the joinless gather and the same
/// combine, over `parts` (KEPT, join order; consumed) whose join-ordered row j
/// holds routed position pos[j]. Returns the number of host-reference words
/// (k = 0..7 sequential f32 fold, one BF16 rounding) equal to the control's.
fn joinProbeCase(a: std.mem.Allocator, g: *ops.MlxOps, name: []const u8, nt: u32, hidden: u32, parts: *std.ArrayList(T), pos: []const u32, weights_host: []const f32) !usize {
    const H: c_int = @intCast(hidden);
    const nr: usize = @as(usize, nt) * top_k;
    const mark = g.mark();
    defer g.resetTo(mark);
    const weights = try g.hostArray(std.mem.sliceAsBytes(weights_host), &.{ @intCast(nt), top_k }, .float32);
    // Control: concatenate + take(argsort(pos)) as runWide does, then the control combine.
    const inv = try a.alloc(u32, nr);
    defer a.free(inv);
    for (pos, 0..) |p, j| inv[p] = @intCast(j);
    const ordered_lazy = try g.reshape(try g.take(try g.concat(parts.items, 0), try g.hostArray(std.mem.sliceAsBytes(inv), &.{@intCast(nr)}, .uint32), 0), &.{ @intCast(nt), top_k, H });
    try g.evalAll(&.{ordered_lazy});
    const ordered = g.keep(ordered_lazy);
    defer g.release(ordered);
    const control = try combineJoined(g, ordered, weights, nt, H, reduceTilesSync);
    defer g.release(control);
    // Candidate: the shared merge to at most 24 sources, then the joinless gather.
    const loc = try a.alloc(i32, 2 * nr);
    defer a.free(loc);
    var sc: @import("../deepseek_v41/experts.zig").JoinlessScratch(T) = .{};
    defer sc.deinit(a);
    _ = try @import("../deepseek_v41/experts.zig").mergeJoinless(ops.MlxOps, g, a, parts, pos, loc, &sc);
    try g.evalAll(parts.items);
    const candidate = try combineParts(g, parts.items, loc, weights, nt, H, reduceTilesSync);
    defer g.release(candidate);
    const want = try joinProbeBits(a, g, control);
    defer a.free(want);
    const got = try joinProbeBits(a, g, candidate);
    defer a.free(got);
    try joinProbeExpectBits(name, hidden, want, got);
    // The fold order, stated against a host loop over the same words.
    const rows = try a.alloc(f32, nr * hidden);
    defer a.free(rows);
    _ = try g.hostF32(try g.astype(ordered, .float32), rows);
    var sequential: usize = 0;
    for (0..nt) |t| for (0..hidden) |c| {
        var acc: f32 = 0;
        for (0..top_k) |k| acc += rows[(t * top_k + k) * hidden + c] * weights_host[t * top_k + k];
        if (bf16RoundBits(acc) == want[t * hidden + c]) sequential += 1;
    };
    std.debug.print("MIMO_JOIN case={s} tokens={d} hidden={d} sources_after_merge={d} words={d} bit_equal=true sequential_fold_matches={d}/{d}\n", .{ name, nt, hidden, parts.items.len, want.len, sequential, want.len });
    return sequential;
}

test "mimo join CPU: joinless gather + combine is bit-equal to the joined control" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const Case = struct { name: []const u8, nt: u32, hidden: u32, sizes: []const u32 };
    const cases = [_]Case{
        .{ .name = "t1_one_source", .nt = 1, .hidden = 256, .sizes = &.{8} },
        .{ .name = "t7", .nt = 7, .hidden = 256, .sizes = &.{ 3, 1, 20 } },
        .{ .name = "t128_rows1024_one_source", .nt = 128, .hidden = 256, .sizes = &.{1024} },
        .{ .name = "t256_real_hidden", .nt = 256, .hidden = 4096, .sizes = &.{ 1024, 13, 1, 97 } },
        .{ .name = "t1024_gather_boundary_merged", .nt = 1024, .hidden = 256, .sizes = &.{ 1024, 13, 1, 97 } },
        .{ .name = "t1025_past_boundary_merged", .nt = 1025, .hidden = 256, .sizes = &.{ 512, 1, 33, 1024, 5 } },
    };
    for (cases, 0..) |c, ci| {
        const nr: u32 = c.nt * top_k;
        var parts: std.ArrayList(T) = .empty;
        defer {
            for (parts.items) |x| g.release(x);
            parts.deinit(a);
        }
        var done: u32 = 0;
        var si: usize = 0;
        while (done < nr) : (si += 1) {
            const r = @min(c.sizes[si % c.sizes.len], nr - done);
            const mark = g.mark();
            const x = try matrixProbeInput(a, &g, r, c.hidden, 0x51ed270b +% @as(u32, @intCast(ci * 977 + si * 131)));
            try g.evalAll(&.{x});
            try parts.append(a, g.keep(x));
            g.resetTo(mark);
            done += r;
        }
        // A seeded permutation: join-ordered row j holds routed position pos[j].
        const pos = try a.alloc(u32, nr);
        defer a.free(pos);
        var state: u32 = 0x2545f491 +% @as(u32, @intCast(ci));
        for (pos, 0..) |*p, i| p.* = @intCast(i);
        var i: usize = nr;
        while (i > 1) {
            i -= 1;
            std.mem.swap(u32, &pos[i], &pos[matrixProbeNext(&state) % (i + 1)]);
        }
        const w = try a.alloc(f32, nr);
        defer a.free(w);
        for (w) |*v| v.* = @as(f32, @floatFromInt(matrixProbeNext(&state) % 100_003)) / 77_003.0 - 0.125;
        _ = try joinProbeCase(a, &g, c.name, c.nt, c.hidden, &parts, pos, w);
    }
}

test "mimo join CPU: prefillParts tiles and positions rebuild prefill's rows exactly" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const H: u32 = 64;
    const I: u32 = 64;
    const capacity: u32 = 5;
    const geometry = try bank_mod.Geometry.init(H, I, capacity, 1, 1);
    var math = try Math.init(a, geometry, matrixProbeActivation);
    defer math.deinit(&g);
    var bank: Arrays = undefined;
    inline for (.{ "gate", "up", "down" }, 0..) |field, fi| {
        const in: u32 = if (fi == 2) I else H;
        const out: u32 = if (fi == 2) H else I;
        const words = try a.alloc(u32, capacity * out * (in / 8));
        defer a.free(words);
        const scales = try a.alloc(u8, capacity * out * (in / 32));
        defer a.free(scales);
        var state: u32 = 0x7a02d195 +% fi;
        for (words) |*word| word.* = matrixProbeNext(&state);
        for (scales) |*scale| scale.* = @intCast(117 + matrixProbeNext(&state) % 5);
        @field(bank, field) = .{
            .w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(capacity), @intCast(out), @intCast(in / 8) }, .uint32),
            .scales = try g.hostArray(scales, &.{ @intCast(capacity), @intCast(out), @intCast(in / 32) }, .uint8),
        };
    }
    const tokens: u32 = 300;
    const x_lazy = try matrixProbeInput(a, &g, tokens, H, 0x38abd247);
    try g.evalAll(&.{x_lazy});
    const x = g.keep(x_lazy);
    defer g.release(x);
    // 1037 rows: one full 1024-row projection tile and a 13-row tail, slots unsorted.
    for ([_]u32{ 1, 7, 1024, 1037 }) |n| {
        const slots = try a.alloc(u32, n);
        defer a.free(slots);
        const acts = try a.alloc(u32, n);
        defer a.free(acts);
        var state: u32 = 0x190bd6e3 +% n;
        for (slots, acts) |*sl, *ac| {
            sl.* = matrixProbeNext(&state) % capacity;
            ac.* = matrixProbeNext(&state) % tokens;
        }
        const rows: quant.PrefillRows = .{ .slot = slots, .act_row = acts };
        const control = try math.prefill(&g, 0, x, rows, bank);
        defer g.release(control);
        var outs: std.ArrayList(T) = .empty;
        defer {
            for (outs.items) |y| g.release(y);
            outs.deinit(a);
        }
        var pos: std.ArrayList(u32) = .empty;
        defer pos.deinit(a);
        try math.prefillParts(&g, 0, x, rows, bank, a, &outs, &pos);
        try std.testing.expectEqual(@as(usize, (n + projection_rows - 1) / projection_rows), outs.items.len);
        try std.testing.expectEqual(@as(usize, n), pos.items.len);
        // Rebuild rows order from the tiles: rows index pos[j] is joined row j.
        const inv = try a.alloc(u32, n);
        defer a.free(inv);
        for (pos.items, 0..) |p, j| inv[p] = @intCast(j);
        const mark = g.mark();
        const rebuilt = try g.take(try g.concat(outs.items, 0), try g.hostArray(std.mem.sliceAsBytes(inv), &.{@intCast(n)}, .uint32), 0);
        const want = try joinProbeBits(a, &g, control);
        defer a.free(want);
        const got = try joinProbeBits(a, &g, rebuilt);
        defer a.free(got);
        g.resetTo(mark);
        try joinProbeExpectBits("prefill_parts", H, want, got);
        std.debug.print("MIMO_JOIN prefill_parts rows={d} tiles={d} bit_equal=true\n", .{ n, outs.items.len });
    }
}

// ---- prefill bank view (`BankView`): fixtures and proofs ----

/// The real expert geometry (MiMo-V2.6: hidden 4096, routed intermediate 2048, MXFP4 group 32).
const bank_view_hidden: u32 = 4096;
const bank_view_inter: u32 = 2048;
/// The fixture bank's rows: enough that a 1,024-row tile over the whole bank is bm 32 (1,024 / 32 = 32 < 64) and the
/// sorted GEMM (>= 4), as the prompt pass's 128 / 140 / 146 / 96-row banks are; 32 x 12.85 MB = 411 MB.
const bank_view_slots: u32 = 32;
/// One case: B rows over `experts` distinct slots, the first at `lo`, the last at `lo + span - 1`.
const BankViewCase = struct { rows: u32, experts: u32, lo: u32, span: u32 };
const bank_view_cases = [_]BankViewCase{
    .{ .rows = 1, .experts = 1, .lo = 5, .span = 1 },
    .{ .rows = 7, .experts = 3, .lo = 0, .span = 4 },
    .{ .rows = 32, .experts = 2, .lo = 30, .span = 2 },
    .{ .rows = 64, .experts = 5, .lo = 11, .span = 9 },
    .{ .rows = 300, .experts = 9, .lo = 7, .span = 12 },
    .{ .rows = 600, .experts = 6, .lo = 20, .span = 9 },
    .{ .rows = 1024, .experts = 1, .lo = 17, .span = 1 },
    .{ .rows = 1024, .experts = 3, .lo = 2, .span = 5 },
    .{ .rows = 1024, .experts = 9, .lo = 23, .span = 9 },
};

/// A seeded MXFP4 bank of `slots` rows at the real geometry (E8M0 scales near 2^-10 .. 2^-6: the outputs stay finite).
fn bankViewBank(a: std.mem.Allocator, g: *ops.MlxOps, slots: u32, seed: u32) !Arrays {
    var bank: Arrays = undefined;
    inline for (.{ "gate", "up", "down" }, 0..) |field, fi| {
        const in: u32 = if (fi == 2) bank_view_inter else bank_view_hidden;
        const out: u32 = if (fi == 2) bank_view_hidden else bank_view_inter;
        const words = try a.alloc(u32, slots * out * (in / 8));
        defer a.free(words);
        const scales = try a.alloc(u8, slots * out * (in / 32));
        defer a.free(scales);
        var state: u32 = seed +% @as(u32, fi) *% 0x9e3779b9;
        for (words) |*word| word.* = matrixProbeNext(&state);
        for (scales) |*scale| scale.* = @intCast(117 + matrixProbeNext(&state) % 5);
        const w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(slots), @intCast(out), @intCast(in / 8) }, .uint32);
        const sc = try g.hostArray(scales, &.{ @intCast(slots), @intCast(out), @intCast(in / 32) }, .uint8);
        try g.evalAll(&.{ w, sc });
        @field(bank, field) = .{ .w = g.keep(w), .scales = g.keep(sc) };
    }
    return bank;
}

fn bankViewRelease(g: *ops.MlxOps, bank: Arrays) void {
    for ([_]Proj{ bank.gate, bank.up, bank.down }) |p| {
        g.release(p.w);
        g.release(p.scales.?);
    }
}

/// The case's rows, unsorted: expert j (slot lo, the evenly spaced middles, lo + span - 1) holds ~(j + 1) shares of B.
fn bankViewRows(c: BankViewCase, tokens: u32, slots: []u32, acts: []u32, seed: u32) void {
    var experts: [16]u32 = undefined;
    for (experts[0..c.experts], 0..) |*e, j| e.* = if (c.experts == 1) c.lo else c.lo + @as(u32, @intCast(j)) * (c.span - 1) / (c.experts - 1);
    const shares = c.experts * (c.experts + 1) / 2;
    var state = seed;
    var r: u32 = 0;
    for (0..c.experts) |j| {
        const n = if (j + 1 == c.experts) c.rows - r else @min(c.rows - r, @max(1, c.rows * @as(u32, @intCast(j + 1)) / shares));
        for (slots[r..][0..n]) |*sl| sl.* = experts[j];
        r += n;
    }
    for (acts[0..c.rows]) |*ac| ac.* = matrixProbeNext(&state) % tokens;
    // Shuffle the rows (packRows sorts them by slot, stably).
    var i: usize = c.rows;
    while (i > 1) {
        i -= 1;
        const k = matrixProbeNext(&state) % (i + 1);
        std.mem.swap(u32, &slots[i], &slots[k]);
        std.mem.swap(u32, &acts[i], &acts[k]);
    }
}

/// One case on `g`'s stream: the control (`full`) vs the installed `window` route through `prefillParts` (the bits the
/// served tile path produces), and the unconditional window (every tile sliced, the qmv rule off: lever 2b's view) vs the
/// full bank on one tile. Returns the words compared and how many were bit-equal for both comparisons.
const BankViewResult = struct { route_words: usize, route_equal: usize, slice_words: usize, slice_equal: usize, full_path: u2, view_path: u2, slice_path: u2 };

fn bankViewCompare(a: std.mem.Allocator, g: *ops.MlxOps, full_math: *Math, window_math: *Math, x: T, bank: Arrays, c: BankViewCase, tokens: u32) !BankViewResult {
    const slots = try a.alloc(u32, c.rows);
    defer a.free(slots);
    const acts = try a.alloc(u32, c.rows);
    defer a.free(acts);
    bankViewRows(c, tokens, slots, acts, 0x2545f491 +% c.rows *% 31 +% c.lo);
    const rows: quant.PrefillRows = .{ .slot = slots, .act_row = acts };
    var outs: [2]std.ArrayList(T) = .{ .empty, .empty };
    var pos: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&outs, &pos) |*o, *p| {
        for (o.items) |y| g.release(y);
        o.deinit(a);
        p.deinit(a);
    };
    try full_math.prefillParts(g, 0, x, rows, bank, a, &outs[0], &pos[0]);
    try window_math.prefillParts(g, 0, x, rows, bank, a, &outs[1], &pos[1]);
    try std.testing.expectEqualSlices(u32, pos[0].items, pos[1].items);
    try std.testing.expectEqual(outs[0].items.len, outs[1].items.len);
    var res: BankViewResult = .{ .route_words = 0, .route_equal = 0, .slice_words = 0, .slice_equal = 0, .full_path = 0, .view_path = 0, .slice_path = 0 };
    for (outs[0].items, outs[1].items) |want_t, got_t| {
        const want = try joinProbeBits(a, g, want_t);
        defer a.free(want);
        const got = try joinProbeBits(a, g, got_t);
        defer a.free(got);
        res.route_words += want.len;
        for (want, got) |w, v| res.route_equal += @intFromBool(w == v);
    }
    // The unconditional window on one tile (the case's rows fit one tile): sorted slots, rebased, every projection sliced.
    std.debug.assert(c.rows <= projection_rows);
    const order = full_math.packRows(rows);
    var sorted: [projection_rows]u32 = undefined;
    var src: [projection_rows]u32 = undefined;
    for (order, sorted[0..c.rows], src[0..c.rows]) |o, *sl, *sr| {
        sl.* = slots[o];
        sr.* = acts[o];
    }
    const mark = g.mark();
    defer g.resetTo(mark);
    const n: c_int = @intCast(c.rows);
    const act = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(src[0..c.rows]), &.{n}, .uint32), 0);
    const rhs_full = try g.hostArray(std.mem.sliceAsBytes(sorted[0..c.rows]), &.{n}, .uint32);
    const lo = sorted[0];
    const hi = sorted[c.rows - 1] + 1;
    for (sorted[0..c.rows]) |*sl| sl.* -= lo;
    const rhs_win = try g.hostArray(std.mem.sliceAsBytes(sorted[0..c.rows]), &.{n}, .uint32);
    const win: Arrays = .{ .gate = try windowRows(g, bank.gate, lo, hi), .up = try windowRows(g, bank.up, lo, hi), .down = try windowRows(g, bank.down, lo, hi) };
    const y_full = try sortedProjection(g, try matrixProbeActivation(g, try sortedProjection(g, act, bank.gate, rhs_full), try sortedProjection(g, act, bank.up, rhs_full)), bank.down, rhs_full);
    const y_win = try sortedProjection(g, try matrixProbeActivation(g, try sortedProjection(g, act, win.gate, rhs_win), try sortedProjection(g, act, win.up, rhs_win)), win.down, rhs_win);
    const want = try joinProbeBits(a, g, y_full);
    defer a.free(want);
    const got = try joinProbeBits(a, g, y_win);
    defer a.free(got);
    res.slice_words = want.len;
    for (want, got) |w, v| res.slice_equal += @intFromBool(w == v);
    res.full_path = timeline.pathOf(c.rows, bank_view_slots);
    res.view_path = if (sortedGemm(c.rows, bank_view_slots)) timeline.pathOf(c.rows, hi - lo) else res.full_path;
    res.slice_path = timeline.pathOf(c.rows, hi - lo);
    return res;
}

const bank_view_path_names = [_][]const u8{ "rhs_bm32", "rhs_bm64", "qmv" };

test "mimo prefill bank view CPU: windowed tiles are bit-equal to the full bank at real shapes (1 to 1,024 rows over 1-9 experts)" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    try std.testing.expectError(error.MimoPrefillBankView, bankViews(.invalid));
    try std.testing.expect((try bankViews(.full)) == &fullView);
    try std.testing.expect((try bankViews(.window)) == &windowView);
    // MLX's rule, as `GatherQMM::eval_gpu` states it.
    try std.testing.expect(!sortedGemm(15, 1) and sortedGemm(16, 4) and !sortedGemm(511, 128) and sortedGemm(512, 128));
    const geometry = try bank_mod.Geometry.init(bank_view_hidden, bank_view_inter, bank_view_slots, 1, 1);
    var full_math = try Math.init(a, geometry, matrixProbeActivation);
    defer full_math.deinit(&g);
    var window_math = try Math.init(a, geometry, matrixProbeActivation);
    defer window_math.deinit(&g);
    window_math.bank_view = try bankViews(.window);
    const bank = try bankViewBank(a, &g, bank_view_slots, 0x62616e6b);
    defer bankViewRelease(&g, bank);
    const tokens: u32 = 1100;
    const x_lazy = try matrixProbeInput(a, &g, tokens, bank_view_hidden, 0x77696e64);
    try g.evalAll(&.{x_lazy});
    const x = g.keep(x_lazy);
    defer g.release(x);
    for (bank_view_cases) |c| {
        const r = try bankViewCompare(a, &g, &full_math, &window_math, x, bank, c, tokens);
        std.debug.print("MIMO_BANKVIEW device=cpu rows={d} experts={d} span={d} E_full={d} path_full={s} path_window={s} path_slice={s} route_bit_equal={d}/{d} slice_bit_equal={d}/{d}\n", .{ c.rows, c.experts, c.span, bank_view_slots, bank_view_path_names[r.full_path], bank_view_path_names[r.view_path], bank_view_path_names[r.slice_path], r.route_equal, r.route_words, r.slice_equal, r.slice_words });
        try std.testing.expectEqual(r.route_words, r.route_equal);
        try std.testing.expectEqual(r.slice_words, r.slice_equal);
    }
}

/// The device fixture's bound: the 32-row bank (411 MB) + x (1,100 x 4096 bf16, 9 MB) + one case's tiles and
/// intermediates (1,024 x 2048 x 3 bf16 + 1,024 x 4096 x 4 x 4 B host/f32 copies, < 90 MB), the MLX buffer cache
/// (limited below to 256 MiB) and the 896 MiB JIT / compiler / host allowances of the pre-ship families: < 1.7 GB.
pub const bank_view_selfcheck_bound_bytes: u64 = 1_700_000_000;

test "mimo bankview GPU: windowed sorted-GEMM tiles are byte-equal to the full bank on the device (bm 32 -> bm 64, MLX's NAX rhs kernel)" {
    _ = std.c.getenv("MIMO_BANKVIEW_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    std.debug.print("MIMO_BANKVIEW_SELFCHECK bound_bytes={d}\n", .{bank_view_selfcheck_bound_bytes});
    mlx.installErrorHandler();
    const a = std.testing.allocator;
    var previous: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&previous, 256 << 20));
    defer _ = mlx.mlx_set_cache_limit(&previous, previous);
    var previous_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&previous_limit, bank_view_selfcheck_bound_bytes));
    defer _ = mlx.mlx_set_memory_limit(&previous_limit, previous_limit);
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const geometry = try bank_mod.Geometry.init(bank_view_hidden, bank_view_inter, bank_view_slots, 1, 1);
    var full_math = try Math.init(a, geometry, matrixProbeActivation);
    defer full_math.deinit(&g);
    var window_math = try Math.init(a, geometry, matrixProbeActivation);
    defer window_math.deinit(&g);
    window_math.bank_view = try bankViews(.window);
    const bank = try bankViewBank(a, &g, bank_view_slots, 0x62616e6b);
    defer bankViewRelease(&g, bank);
    const tokens: u32 = 1100;
    const x_lazy = try matrixProbeInput(a, &g, tokens, bank_view_hidden, 0x77696e64);
    try g.evalAll(&.{x_lazy});
    const x = g.keep(x_lazy);
    defer g.release(x);
    var bm_flips: u32 = 0;
    for (bank_view_cases) |c| {
        const r = try bankViewCompare(a, &g, &full_math, &window_math, x, bank, c, tokens);
        std.debug.print("MIMO_BANKVIEW device=gpu rows={d} experts={d} span={d} E_full={d} path_full={s} path_window={s} path_slice={s} route_bit_equal={d}/{d} slice_bit_equal={d}/{d}\n", .{ c.rows, c.experts, c.span, bank_view_slots, bank_view_path_names[r.full_path], bank_view_path_names[r.view_path], bank_view_path_names[r.slice_path], r.route_equal, r.route_words, r.slice_equal, r.slice_words });
        // The route's gate: byte-equal on every case (the window changes only bm and the grid). The unconditional
        // slice (lever 2b: qmv tiles onto the GEMM) is reported, not gated.
        try std.testing.expectEqual(r.route_words, r.route_equal);
        bm_flips += @intFromBool(r.full_path == 0 and r.view_path == 1);
        g.reset();
    }
    // The fixture must exercise the bm 32 -> bm 64 flip (else the gate proves nothing about it).
    try std.testing.expect(bm_flips >= 3);
    std.debug.print("MIMO_BANKVIEW_SELFCHECK PASS bm32_to_bm64_cases={d}\n", .{bm_flips});
}

test "mimo prefill drain CPU: pipelined projection and reduction tiles are bit-equal to the sync control" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const H: u32 = 64;
    const I: u32 = 64;
    const capacity: u32 = 6;
    const geometry = try bank_mod.Geometry.init(H, I, capacity, 1, 1);
    var sync_math = try Math.init(a, geometry, matrixProbeActivation);
    defer sync_math.deinit(&g);
    var pipe_math = try Math.init(a, geometry, matrixProbeActivation);
    defer pipe_math.deinit(&g);
    pipe_math.drain = try drains(.pipelined);
    try std.testing.expectError(error.MimoPrefillDrain, drains(.invalid));
    try std.testing.expect(sync_math.drain.tiles == (try drains(.sync)).tiles);
    var bank: Arrays = undefined;
    inline for (.{ "gate", "up", "down" }, 0..) |field, fi| {
        const in: u32 = if (fi == 2) I else H;
        const out: u32 = if (fi == 2) H else I;
        const words = try a.alloc(u32, capacity * out * (in / 8));
        defer a.free(words);
        const scales = try a.alloc(u8, capacity * out * (in / 32));
        defer a.free(scales);
        var state: u32 = 0x5d1e0b93 +% fi;
        for (words) |*word| word.* = matrixProbeNext(&state);
        for (scales) |*scale| scale.* = @intCast(117 + matrixProbeNext(&state) % 5);
        @field(bank, field) = .{
            .w = try g.hostArray(std.mem.sliceAsBytes(words), &.{ @intCast(capacity), @intCast(out), @intCast(in / 8) }, .uint32),
            .scales = try g.hostArray(scales, &.{ @intCast(capacity), @intCast(out), @intCast(in / 32) }, .uint8),
        };
    }
    const tokens: u32 = 700;
    const x_lazy = try matrixProbeInput(a, &g, tokens, H, 0x0b5e11a7);
    try g.evalAll(&.{x_lazy});
    const x = g.keep(x_lazy);
    defer g.release(x);
    // One tile, a full tile, a tail, three tiles (two in flight at once, then the last waited).
    for ([_]u32{ 1, 7, 1024, 1037, 2600 }) |n| {
        const slots = try a.alloc(u32, n);
        defer a.free(slots);
        const acts = try a.alloc(u32, n);
        defer a.free(acts);
        var state: u32 = 0x44c1a3f9 +% n;
        for (slots, acts) |*sl, *ac| {
            sl.* = matrixProbeNext(&state) % capacity;
            ac.* = matrixProbeNext(&state) % tokens;
        }
        const rows: quant.PrefillRows = .{ .slot = slots, .act_row = acts };
        var outs: [2]std.ArrayList(T) = .{ .empty, .empty };
        var pos: [2]std.ArrayList(u32) = .{ .empty, .empty };
        defer for (&outs, &pos) |*o, *p| {
            for (o.items) |y| g.release(y);
            o.deinit(a);
            p.deinit(a);
        };
        try sync_math.prefillParts(&g, 0, x, rows, bank, a, &outs[0], &pos[0]);
        try pipe_math.prefillParts(&g, 0, x, rows, bank, a, &outs[1], &pos[1]);
        try std.testing.expectEqual(outs[0].items.len, outs[1].items.len);
        try std.testing.expectEqualSlices(u32, pos[0].items, pos[1].items);
        for (outs[0].items, outs[1].items) |want_t, got_t| {
            const want = try joinProbeBits(a, &g, want_t);
            defer a.free(want);
            const got = try joinProbeBits(a, &g, got_t);
            defer a.free(got);
            try joinProbeExpectBits("drain_tiles", H, want, got);
        }
        std.debug.print("MIMO_DRAIN case=tiles rows={d} tiles={d} bit_equal=true\n", .{ n, outs[0].items.len });
    }
    // The reductions: 600 tokens (one gather tile, three reduction tiles) and 1,300 (two gather tiles), H = 4096.
    for ([_]u32{ 600, 1300 }) |nt| {
        const hidden: u32 = 4096;
        const nr = nt * top_k;
        const mark = g.mark();
        defer g.resetTo(mark);
        var parts: std.ArrayList(T) = .empty;
        defer {
            for (parts.items) |p| g.release(p);
            parts.deinit(a);
        }
        var done: u32 = 0;
        var si: u32 = 0;
        while (done < nr) : (si += 1) {
            const r = @min(@as(u32, 1024) - si * 61 % 300, nr - done);
            const pm = g.mark();
            const xr = try matrixProbeInput(a, &g, r, hidden, 0x6a09e667 +% si);
            try g.evalAll(&.{xr});
            try parts.append(a, g.keep(xr));
            g.resetTo(pm);
            done += r;
        }
        const loc = try a.alloc(i32, 2 * nr);
        defer a.free(loc);
        var state: u32 = 0x3c6ef372 +% nt;
        // Each routed position's (source, row): a seeded permutation of the join-ordered rows.
        const perm = try a.alloc(u32, nr);
        defer a.free(perm);
        for (perm, 0..) |*p, i| p.* = @intCast(i);
        var i: usize = nr;
        while (i > 1) {
            i -= 1;
            std.mem.swap(u32, &perm[i], &perm[matrixProbeNext(&state) % (i + 1)]);
        }
        var src: usize = 0;
        var base: usize = 0;
        for (0..nr) |j| {
            while (j - base >= @as(usize, @intCast(mlx.getShape(parts.items[src])[0]))) {
                base += @as(usize, @intCast(mlx.getShape(parts.items[src])[0]));
                src += 1;
            }
            loc[2 * perm[j]] = @intCast(src);
            loc[2 * perm[j] + 1] = @intCast(j - base);
        }
        const w = try a.alloc(f32, nr);
        defer a.free(w);
        for (w) |*v| v.* = @as(f32, @floatFromInt(matrixProbeNext(&state) % 100_003)) / 77_003.0 - 0.125;
        const weights = try g.hostArray(std.mem.sliceAsBytes(w), &.{ @intCast(nt), top_k }, .float32);
        if (parts.items.len > max_sources) return error.SkipZigTest;
        const control = try combineParts(&g, parts.items, loc, weights, nt, @intCast(hidden), reduceTilesSync);
        defer g.release(control);
        const candidate = try combineParts(&g, parts.items, loc, weights, nt, @intCast(hidden), reduceTilesPipelined);
        defer g.release(candidate);
        const want = try joinProbeBits(a, &g, control);
        defer a.free(want);
        const got = try joinProbeBits(a, &g, candidate);
        defer a.free(got);
        try joinProbeExpectBits("drain_reduce", hidden, want, got);
        std.debug.print("MIMO_DRAIN case=reduce tokens={d} sources={d} words={d} bit_equal=true\n", .{ nt, parts.items.len, want.len });
    }
}

test "mimo prefill drain: the pipelined route bills one more projection tile and reduction tile at 16K" {
    // 1,024 x (20 x 4096 + 24 x 2048 + 32) + 256 x (8 x 4096 x 8 + 4096 x 8 + 64)
    try std.testing.expectEqual(@as(u64, 134_250_496 + 75_513_856), pipelinedExtraBytes(4096, 2048, 16_384));
    try std.testing.expectEqual(@as(u64, 8 * (20 * 4096 + 24 * 2048 + 32) + 1 * (8 * 4096 * 8 + 4096 * 8 + 64)), pipelinedExtraBytes(4096, 2048, 1));
}

test "mimo barrier predictor CPU: a verify's rows take the gathered GEMV within F32 rounding of the GEMM and with its top-8; M = 1 keeps the GEMM graph bit for bit" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    const K: c_int = 4096;
    const N: c_int = 256;
    const M: c_int = 4;
    const xv = try a.alloc(f32, @intCast(M * K));
    defer a.free(xv);
    const wv = try a.alloc(f32, @intCast(K * N));
    defer a.free(wv);
    var bv: [256]f32 = undefined;
    var st: u32 = 0x9e3779b9;
    const R = struct {
        fn next(state: *u32) f32 {
            state.* ^= state.* << 13;
            state.* ^= state.* >> 17;
            state.* ^= state.* << 5;
            return (@as(f32, @floatFromInt(state.* % 20001)) - 10000) / 10000;
        }
    };
    for (xv) |*v| v.* = R.next(&st);
    for (wv) |*v| v.* = R.next(&st) * 0.05;
    for (&bv) |*v| v.* = R.next(&st) * 0.01;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xv), &.{ M, K }, .float32), .bfloat16);
    const router = try g.hostArray(std.mem.sliceAsBytes(wv), &.{ K, N }, .float32);
    const bias = try g.hostArray(std.mem.sliceAsBytes(&bv), &.{N}, .float32);
    // the control's graph, op for op
    const stock = try g.add(try g.sigmoid(try g.matmul(try g.astype(x, .float32), router)), bias);
    const rows = try Math.predictorScores(&g, x, router, bias);
    try std.testing.expectEqualSlices(c_int, &.{ M, N }, g.shapeOf(rows).slice());
    var want: [4 * 256]f32 = undefined;
    var got: [4 * 256]f32 = undefined;
    _ = try g.hostF32(stock, &want);
    _ = try g.hostF32(rows, &got);
    var max_abs: f32 = 0;
    for (want, got) |w, v| max_abs = @max(max_abs, @abs(w - v));
    try std.testing.expect(max_abs <= 1e-5);
    // each row's top 8 hints (the lookahead's k) are the GEMM's
    const Top = struct {
        fn of(v: []const f32, out: *[8]u16) void {
            var used: [256]bool = @splat(false);
            for (out) |*o| {
                var best: u16 = 0;
                var bval: f32 = -std.math.inf(f32);
                for (v, 0..) |e, i| if (!used[i] and e > bval) {
                    bval = e;
                    best = @intCast(i);
                };
                used[best] = true;
                o.* = best;
            }
        }
    };
    for (0..@intCast(M)) |r| {
        var tw: [8]u16 = undefined;
        var tg: [8]u16 = undefined;
        Top.of(want[r * 256 ..][0..256], &tw);
        Top.of(got[r * 256 ..][0..256], &tg);
        try std.testing.expectEqualSlices(u16, &tw, &tg);
        // and within rounding of the serial token's own (M = 1) predictor on that row
        const one = try Math.predictorScores(&g, try g.slice(x, &.{ @intCast(r), 0 }, &.{ @intCast(r + 1), K }, &.{ 1, 1 }), router, bias);
        var serial: [256]f32 = undefined;
        _ = try g.hostF32(one, &serial);
        for (serial, got[r * 256 ..][0..256]) |w, v| try std.testing.expectApproxEqAbs(w, v, 1e-5);
    }
    // M = 1: the GEMM graph itself (the served serial token), bit for bit
    const x1 = try g.slice(x, &.{ 0, 0 }, &.{ 1, K }, &.{ 1, 1 });
    const stock1 = try g.add(try g.sigmoid(try g.matmul(try g.astype(x1, .float32), router)), bias);
    const lever1 = try Math.predictorScores(&g, x1, router, bias);
    var w1: [256]f32 = undefined;
    var l1: [256]f32 = undefined;
    _ = try g.hostF32(stock1, &w1);
    _ = try g.hostF32(lever1, &l1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&w1), std.mem.sliceAsBytes(&l1));
}
