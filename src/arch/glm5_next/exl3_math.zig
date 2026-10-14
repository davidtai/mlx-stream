//! glm5_next's routed-expert math over the EXL3 K=3 bank (the `exl3` family): the shared DeepSeek scheduler's math seam
//! (`deepseek_v41_experts.ExpertsWith`'s `M`) on the GLM-shape EXL3 texts (exl3_kernels.zig). The scheduler, the stream,
//! the hit wave / miss parts, the gates, the lookahead and the combine are the NVFP4 family's; only the projection math
//! differs:
//!   - decode: DeepSeek-V4.1's banked PREP=rin chain (in_rin, the gate and up mul1h GEMVs, gu_epi, din_rin, the down GEMV,
//!     dpost), one launch of each per wave over the three banks (`gateUpBanked` / `downBanked`, ids `bank << 24 | row`);
//!     the unbanked route (`decode_moe: stock`) runs the same texts with every bank the call's one bank;
//!   - prompt: DeepSeek-V4.1's DIG-X waves (take2, the gate|up GEMM decoding the trellis in its B loader, the onepass
//!     SwiGLU epilogue, the down GEMM, rot_widen1 into bf16) over the call's rows sorted by slot, a wave at most
//!     `wave_rows` rows of at most `xq.wave_max` experts, pipelined as MiMo's drain (one wave in flight behind the next).
//! The decode lookahead's predictor is MiMo's (`mimo_math.Predictor`), bound at construction.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../deepseek_v41/ops.zig");
const quant = @import("../../sdk_ext.zig").quant;
const xq = @import("../../exl3/quant.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const profile = @import("../mimo_v2/profile.zig");
const kernels = @import("exl3_kernels.zig");
const bank = @import("exl3_bank.zig");

const T = mlx.mlx_array;
const G = ops.MlxOps;
const Proj = bank.Arrays(T);
const Arrays = bank.BankArraysOf(T);

/// A prompt wave's rows at most: MiMo's projection tile (the bill's `expert_workspace` prices a tile of these rows at
/// NVFP4's per-row bytes, above a DIG-X wave's: take2 16 KiB + z 16 KiB + hd 4 KiB + zd 16 KiB + y 8 KiB a row).
pub const wave_rows: u32 = mimo_math.projection_rows;
/// A prompt call's routed rows at most (the module's calls: `graph.prompt_call_rows` tokens x top-8; MiMo's math's bound).
pub const max_call_rows: u32 = @import("nvfp4.zig").Format.max_call_tokens * 8;
pub const wave_bytes_per_row: u64 = 2 * kernels.hidden * 2 + 2 * kernels.inter * 4 + kernels.inter * 2 + kernels.hidden * 4 + kernels.hidden * 2 + 8;

comptime {
    // One wave table serves both GEMMs (their threadgroups per M tile agree at GLM's dims).
    if (kernels.gu_tiles != kernels.dn_tiles) @compileError("glm5_next exl3: the gate|up and down GEMMs need their own wave tables");
    if (wave_bytes_per_row > 20 * kernels.hidden + 24 * kernels.inter + 32) @compileError("glm5_next exl3: a wave row outgrows the billed tile row");
}

pub const Math = struct {
    const Self = @This();

    /// MiMo's decode-lane bounds and phase routes (mimo_math `MathOf`): the decode lane takes at most 4 rows, the wide
    /// lane from 5; the wide waves are event-gated; the predictor runs in decode only.
    pub const decode_forward_rows: u32 = mimo_math.Math.decode_forward_rows;
    pub const prefill_min_tokens: u32 = decode_forward_rows + 1;
    pub const wide_event_gates = true;
    pub const predictor_decode_only = true;
    pub const Profile = mimo_math.Profile;
    pub const has_banked = true;
    pub const has_parts = true;

    pub fn predictorScores(g: anytype, x: @TypeOf(g.*).T, router: @TypeOf(g.*).T, correction: @TypeOf(g.*).T) !@TypeOf(g.*).T {
        return mimo_math.Math.predictorScores(g, x, router, correction);
    }

    allocator: std.mem.Allocator,
    k: *kernels.Kernels,
    predictor: mimo_math.Predictor = .{},
    order: []u32,
    /// A wave's host tables: act rows, entries, the wave table.
    ridx: []i32,
    rhs: []u32,

    /// Construction (the module's GPU stream): the texts installed (`exl3_kernels.install`: pins, kernels, the decode
    /// launches, the compile check), refused by name.
    pub fn init(a: std.mem.Allocator, s: mlx.mlx_stream, call_rows: u32, diag: *sdk.Diag) !Self {
        const k = try kernels.install(a, s, diag);
        errdefer {
            k.deinit();
            a.destroy(k);
        }
        const order = try a.alloc(u32, call_rows);
        errdefer a.free(order);
        const ridx = try a.alloc(i32, wave_rows);
        errdefer a.free(ridx);
        return .{ .allocator = a, .k = k, .order = order, .ridx = ridx, .rhs = try a.alloc(u32, wave_rows) };
    }

    pub fn deinit(self: *Self, _: *G) void {
        self.predictor.deinit();
        self.k.deinit();
        self.allocator.destroy(self.k);
        self.allocator.free(self.order);
        self.allocator.free(self.ridx);
        self.allocator.free(self.rhs);
    }

    pub fn predictorScoresBound(self: *const Self, g: *G, x: T, router: T, correction: T) !T {
        return self.predictor.scores(&self.predictor, g, x, router, correction);
    }

    // ── Decode ──

    /// The unbanked route: one bank, its slot rows (bank 0 in the packing), every bank its arrays.
    pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
        const one: Arrays = .{ .gate = gate, .up = up, .down = gate };
        const banks: [kernels.n_banks]Arrays = @splat(one);
        return kernels.gateUp(self.k, g, x, ids, &banks);
    }

    pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
        const one: Arrays = .{ .gate = d, .up = d, .down = d };
        const banks: [kernels.n_banks]Arrays = @splat(one);
        return kernels.down(self.k, g, h, ids, &banks);
    }

    /// x bf16 [P, hidden] (each pair's token row), ids packed `bank << 24 | row` -> the clamped SwiGLU f32 [P, inter].
    pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [kernels.n_banks]Arrays) !T {
        return kernels.gateUp(self.k, g, x, ids, banks);
    }

    /// h f32 [P, inter] -> each pair's down row bf16 [P, hidden] (the scheduler's join and the arch's combine follow).
    pub fn downBanked(self: *const Self, g: *G, h: T, ids: T, banks: *const [kernels.n_banks]Arrays) !T {
        return kernels.down(self.k, g, h, ids, banks);
    }

    // ── Prompt: DIG-X waves ──

    /// The call's rows sorted by slot (stable): each wave's experts are contiguous rows of it.
    fn packRows(self: *Self, rows: quant.PrefillRows) []u32 {
        const order = self.order[0..rows.slot.len];
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.sort.block(u32, order, rows.slot, struct {
            fn lt(slots: []const u32, a: u32, b: u32) bool {
                return slots[a] < slots[b];
            }
        }.lt);
        return order;
    }

    /// The wave from packed row `start`: at most `wave_rows` rows and `xq.wave_max` experts (an expert's rows may span
    /// two waves: each wave reads its own rows of the slot). Returns the wave's end.
    pub fn waveEnd(order: []const u32, slots: []const u32, start: usize) usize {
        var end = start;
        var experts: u32 = 0;
        var last: ?u32 = null;
        while (end < order.len and end - start < wave_rows) : (end += 1) {
            const s = slots[order[end]];
            if (last == null or s != last.?) {
                if (experts == xq.wave_max) break;
                experts += 1;
                last = s;
            }
        }
        return end;
    }

    /// The wave's experts (slot, rows) in packed order and each row's entry.
    pub fn waveExperts(order: []const u32, slots: []const u32, out: *[xq.wave_max]xq.WaveExpert, rhs: []u32) []const xq.WaveExpert {
        var n: usize = 0;
        for (order, rhs) |o, *r| {
            const s = slots[o];
            if (n == 0 or out[n - 1].slot != s) {
                out[n] = .{ .slot = s, .rows = 0 };
                n += 1;
            }
            out[n - 1].rows += 1;
            r.* = @intCast(n - 1);
        }
        return out[0..n];
    }

    /// One wave's graph over packed rows [start, stop) (unevaluated): bf16 [rows, hidden].
    fn waveGraph(self: *Self, g: *G, x: T, rows: quant.PrefillRows, b: Arrays, order: []const u32) !T {
        const encode_start = profile.now();
        const n = order.len;
        for (order, self.ridx[0..n]) |o, *r| r.* = @intCast(if (rows.act_row) |ar| ar[o] else o);
        var ex: [xq.wave_max]xq.WaveExpert = undefined;
        const experts = waveExperts(order, rows.slot, &ex, self.rhs[0..n]);
        const t = xq.digTableBm(experts, kernels.gu_tiles, kernels.m_tile);
        const cn: c_int = @intCast(n);
        const ridx = try g.hostArray(std.mem.sliceAsBytes(self.ridx[0..n]), &.{cn}, .int32);
        const rhs = try g.hostArray(std.mem.sliceAsBytes(self.rhs[0..n]), &.{cn}, .uint32);
        const tbl = try g.hostArray(std.mem.sliceAsBytes(&t.table), &.{t.table.len}, .int32);
        const x2 = try kernels.take2(self.k, g, x, ridx, rhs, tbl, b.gate.rin, b.up.rin);
        const z = try kernels.gemmGateUp(self.k, g, x2[0], x2[1], b.gate.code, b.up.code, tbl, t.tgs);
        const hd = try kernels.onePass(self.k, g, z[0], z[1], rhs, tbl, b.gate.rout, b.up.rout, b.down.rin);
        const zd = try kernels.gemmDown(self.k, g, hd, b.down.code, tbl, t.tgs);
        const y = try kernels.widen1(self.k, g, zd, rhs, tbl, b.down.rout, .bfloat16);
        profile.charge(.encode, encode_start);
        return g.reshape(y, &.{ cn, @intCast(kernels.hidden) });
    }

    /// The call's waves, each committed and appended KEPT to `kept`, the wave before it then waited (MiMo's pipelined
    /// drain: at most two in flight); the last waited and the stream synchronized before the call returns.
    fn waves(self: *Self, g: *G, x: T, rows: quant.PrefillRows, b: Arrays, kept: *std.ArrayList(T), alloc: std.mem.Allocator) ![]const u32 {
        const order = self.packRows(rows);
        var start: usize = 0;
        var prev: ?T = null;
        while (start < order.len) {
            const stop = waveEnd(order, rows.slot, start);
            const mark = g.mark();
            const y = try self.waveGraph(g, x, rows, b, order[start..stop]);
            try g.asyncEval(&.{y});
            try kept.append(alloc, g.keep(y));
            g.resetTo(mark);
            if (prev) |p| {
                const drain_start = profile.now();
                try g.evalAll(&.{p});
                profile.charge(.drain, drain_start);
            }
            prev = kept.items[kept.items.len - 1];
            start = stop;
        }
        if (prev) |p| {
            const drain_start = profile.now();
            try g.evalAll(&.{p});
            try mlx.check(mlx.mlx_synchronize(g.s));
            profile.charge(.drain, drain_start);
        }
        return order;
    }

    /// The call's routed rows' outputs bf16 [rows, hidden] in `rows` order, evaluated and KEPT.
    pub fn prefill(self: *Self, g: *G, _: u32, x: T, rows: quant.PrefillRows, b: Arrays) !T {
        var kept: std.ArrayList(T) = .empty;
        defer {
            for (kept.items) |y| g.release(y);
            kept.deinit(self.allocator);
        }
        const order = try self.waves(g, x, rows, b, &kept, self.allocator);
        const inverse = try self.allocator.alloc(u32, order.len);
        defer self.allocator.free(inverse);
        for (order, 0..) |o, i| inverse[o] = @intCast(i);
        const mark = g.mark();
        const joined = try g.concat(kept.items, 0);
        const result = try g.take(joined, try g.hostArray(std.mem.sliceAsBytes(inverse), &.{@intCast(order.len)}, .uint32), 0);
        try g.evalAll(&.{result});
        const held = g.keep(result);
        g.resetTo(mark);
        return held;
    }

    /// `prefill`'s waves as they are: each wave's KEPT output appended to `outs`, its rows' indices into `rows` to `pos`.
    pub fn prefillParts(self: *Self, g: *G, _: u32, x: T, rows: quant.PrefillRows, b: Arrays, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
        const order = try self.waves(g, x, rows, b, outs, alloc);
        try pos.appendSlice(alloc, order);
    }

    /// Every wave was evaluated and synchronized before its call returned.
    pub fn finishPrefill(_: *Self, _: *G) !void {}
};

const testing = std.testing;

test "glm5_next exl3 math: a call's waves hold at most 16 experts and wave_rows rows, cover every row once, entries in slot order" {
    // Slots of 40 rows over 20 experts (2 rows each), sorted.
    var slots: [40]u32 = undefined;
    for (&slots, 0..) |*s, i| s.* = @intCast(i / 2 * 3);
    var order: [40]u32 = undefined;
    for (&order, 0..) |*o, i| o.* = @intCast(i);
    var start: usize = 0;
    var waves_n: u32 = 0;
    while (start < order.len) {
        const end = Math.waveEnd(&order, &slots, start);
        var ex: [xq.wave_max]xq.WaveExpert = undefined;
        var rhs: [40]u32 = undefined;
        const e = Math.waveExperts(order[start..end], &slots, &ex, rhs[0 .. end - start]);
        try testing.expect(e.len <= xq.wave_max);
        var total: u32 = 0;
        for (e, 0..) |w, j| {
            total += w.rows;
            if (j > 0) try testing.expect(w.slot > e[j - 1].slot);
        }
        try testing.expectEqual(@as(u32, @intCast(end - start)), total);
        try testing.expectEqual(@as(u32, @intCast(e.len - 1)), rhs[end - start - 1]);
        start = end;
        waves_n += 1;
    }
    try testing.expectEqual(@as(u32, 2), waves_n);
    // One expert of more rows than a wave takes splits over waves of wave_rows.
    const big_n = wave_rows + 7;
    const big_slots = try testing.allocator.alloc(u32, big_n);
    defer testing.allocator.free(big_slots);
    @memset(big_slots, 5);
    const big_order = try testing.allocator.alloc(u32, big_n);
    defer testing.allocator.free(big_order);
    for (big_order, 0..) |*o, i| o.* = @intCast(i);
    try testing.expectEqual(@as(usize, wave_rows), Math.waveEnd(big_order, big_slots, 0));
    try testing.expectEqual(@as(usize, big_n), Math.waveEnd(big_order, big_slots, wave_rows));
}
