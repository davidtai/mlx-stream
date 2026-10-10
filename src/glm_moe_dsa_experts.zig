//! GLM-5.3's routed experts over the expert stream. Per routed-layer call the trunk hands the driver the MoE input
//! `x [n, hidden]`, the router's top-k `indices [n, k]` (int32) and their weights `scores [n, k]` (f32, normalized and
//! scaled), and gets back the weighted sum `[n, hidden]` in x's dtype (the reference's `(y * scores[..., None]).sum(-2)`).
//! The bank module `Bk` (its stream, its slot arrays) and the routed-expert math `M` (the C2 quant contract:
//! `gateUp`, `down`, `prefill`, `finishPrefill`) are comptime parameters: the affine bank binds MLX's `gather_qmm`
//! (`sdk_ext.quant.FromGatherMatmul(GatherQmm)`), another bank binds its own quant with no change here or in the trunk.
//!
//! A call of at most `max_route_ids` routed ids (decode, short prompts) is the decode lane: the routing barrier (the ids
//! and, with the lookahead, the next routed layer's scores read on the host), the route, the residents' wave at once,
//! then each miss part's gate/up after its gate/up bytes landed and its down after the rest (or every wave at once over
//! event-gate aliases), the release; the outputs joined in routed order. A wider call is the wide lane: the call's
//! experts hottest first, the layer's residency seeded from its ids, groups of `max_route_ids` experts routed up to
//! `wide_depth` ahead, each group's rows per bank through the math's `prefill` in slices, drained before the group's
//! slots go back; the join and the combine in token slices.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("sdk_ext.zig");
const expert_policy = sdk_ext.expert.policy;
const expert_event = sdk_ext.expert.event;
const expert_stream = @import("expert_stream.zig");
const ops = @import("deepseek_v41_ops.zig");
const glm = @import("glm_moe_dsa.zig");

pub const BankKind = sdk_ext.expert.BankKind;
pub const SlotRef = sdk_ext.expert.SlotRef;
pub const max_route_ids = expert_policy.max_route_ids;
const n_banks = std.meta.fieldNames(BankKind).len;

/// The driver's construction-time routes.
pub const Options = struct {
    /// The waves wait on the reads' event gates (the stream built with `event`); off: the host waits.
    gated: bool = false,
    /// The MLX event the stream signals (gated MLX backends).
    event: ?expert_event.Event = null,
    /// The wide lane's groups in flight (at most the stream's `wide_depth`).
    wide_depth: u8 = 1,
};

pub fn Experts(comptime G: type, comptime Bk: type, comptime M: type) type {
    comptime sdk_ext.expert.assertBank(Bk);
    return struct {
        const Self = @This();
        const T = G.T;
        const S = Bk.Stream;
        pub const Stream = S.Stream;
        pub const Arrays = Bk.BankArrays;

        a: std.mem.Allocator,
        stream: *Stream,
        math: *M,
        hidden: c_int,
        n_experts: u32,
        /// Per routed layer, per bank kind: what the math binds (null: no rows, or released).
        banks: [][n_banks]?Arrays,
        opt: Options,
        transient_released: bool = false,
        wide: Wide = .{},

        /// The wide lane's host scratch, reused across calls.
        const Wide = struct {
            ids: std.ArrayList(u16) = .empty,
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            count: std.ArrayList(u32) = .empty,
            slot: std.ArrayList(u32) = .empty,
            act: std.ArrayList(u32) = .empty,
            pos: std.ArrayList(u32) = .empty,
            inv: std.ArrayList(u32) = .empty,
            kept: std.ArrayList(T) = .empty,

            fn deinit(w: *Wide, a: std.mem.Allocator, g: *G) void {
                for (w.kept.items) |x| g.release(x);
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.count, &w.slot, &w.act, &w.pos, &w.inv, &w.kept }) |l| l.deinit(a);
            }
        };

        pub fn init(a: std.mem.Allocator, g: *G, stream: *Stream, math: *M, hidden: u32, opt: Options) !Self {
            if (opt.gated and !stream.gated) return error.GatedNeedsStreamEvent;
            if (opt.gated and G == ops.MlxOps and opt.event == null) return error.GatedNeedsEvent;
            if (opt.wide_depth < 1 or opt.wide_depth > stream.wide_depth) return error.WideDepthExceedsStream;
            const banks = try a.alloc([n_banks]?Arrays, stream.layers.len);
            errdefer a.free(banks);
            for (banks, 0..) |*b, l| {
                b.* = @splat(null);
                for ([_]BankKind{ .base, .transient }) |kind| b[@backingInt(kind)] = try bind(stream, @intCast(l), kind);
            }
            _ = g;
            return .{ .a = a, .stream = stream, .math = math, .hidden = @intCast(hidden), .n_experts = stream.bank.n_experts, .banks = banks, .opt = opt };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.wide.deinit(self.a, g);
            self.a.free(self.banks);
            self.* = undefined;
        }

        /// A bank with rows must have arrays the math binds (MLX slot memory).
        fn bind(stream: *Stream, layer: u32, kind: BankKind) !?Arrays {
            const arrays = stream.bankArrays(layer, kind);
            if (arrays == null and rowsOf(stream, layer, kind) > 0) return error.SlotArraysUnbound;
            return arrays;
        }

        fn rowsOf(stream: *Stream, layer: u32, kind: BankKind) u32 {
            const ls = &stream.layers[layer];
            return switch (kind) {
                .base => ls.base.rows,
                .ext => if (ls.ext) |e| e.rows else 0,
                .transient => stream.transient.rows,
            };
        }

        /// The phase change's first free: every transient binding nulled, then the stream frees the scratch.
        pub fn releaseTransient(self: *Self) !u64 {
            for (self.banks) |*b| b[@backingInt(BankKind.transient)] = null;
            self.transient_released = true;
            return self.stream.releaseTransient();
        }

        /// The one phase change: the stream grows; the grown rows and decode's window 0 are bound.
        pub fn grow(self: *Self, decode_rows: []const u32) !void {
            try self.stream.grow(decode_rows);
            for (self.banks, 0..) |*b, l| {
                b[@backingInt(BankKind.ext)] = try bind(self.stream, @intCast(l), .ext);
                b[@backingInt(BankKind.transient)] = try bind(self.stream, @intCast(l), .transient);
            }
            self.transient_released = false;
        }

        /// The reverse phase change's free: the grown rows' and window 0's bindings nulled, then the stream frees them.
        pub fn shrink(self: *Self, prompt_rows: []const u32) !u64 {
            for (self.banks) |*b| {
                b[@backingInt(BankKind.ext)] = null;
                b[@backingInt(BankKind.transient)] = null;
            }
            self.transient_released = true;
            return self.stream.shrink(prompt_rows);
        }

        /// The reverse phase change's allocation: the prompt's scratch re-created and bound.
        pub fn regrowTransient(self: *Self) !u64 {
            const bytes = try self.stream.regrowTransient();
            for (self.banks, 0..) |*b, l| b[@backingInt(BankKind.transient)] = try bind(self.stream, @intCast(l), .transient);
            self.transient_released = false;
            return bytes;
        }

        /// After the forward's last eval: settles and unpins released calls.
        pub fn flush(self: *Self) !void {
            try self.stream.flush();
        }

        /// P1: layer `layer`'s predicted experts (hottest first) read ahead of its routed call (prompt phase).
        pub fn readAheadSeed(self: *Self, layer: u32, experts: []const u16) !void {
            return self.stream.readAheadSeed(layer, experts);
        }

        /// One routed-layer call: x [n, hidden], indices [n, k] int32, scores [n, k] f32 -> the weighted sum [n, hidden]
        /// in x's dtype. `next_scores` [n, n_experts] f32: the next routed layer's routing scores on this call's rows (the
        /// decode lookahead's predictor), evaluated with the ids at the routing barrier; null for none. `hoist`: arrays
        /// that do not wait on this call (the shared expert), committed behind the hit wave once the reads are issued, so
        /// the GPU runs them during the read wait (the decode lane; the wide lane leaves them to the caller).
        pub fn call(self: *Self, g: *G, layer: u32, x: T, indices: T, scores: T, next_scores: ?T, hoist: []const T) !T {
            const n: u32 = @intCast(g.shapeOf(x).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            if (n * k > self.stream.max_route_ids) return self.callWide(g, layer, x, indices, scores, n, k);
            const y = try self.callDecode(g, layer, x, indices, next_scores, n, k, hoist);
            return combine(g, y, scores, g.dtypeOf(x));
        }

        /// `(y * scores[..., None]).sum(-2)` in f32, cast to `dt` (the reference's combine).
        fn combine(g: *G, y: T, scores: T, dt: ops.Dtype) !T {
            const w = try g.expandDims(scores, -1);
            return g.astype(try g.sum(try g.mul(y, w), -2, false), dt);
        }

        const Group = struct { bank: BankKind, n: u32 = 0, pos: [max_route_ids]u8 = undefined };
        const Wave = struct {
            groups: [n_banks]Group = undefined,
            n: usize = 0,
            ids: [n_banks]T = undefined,
            h: [n_banks]T = undefined,
        };
        const Acc = struct {
            outs: [2 * max_route_ids]T = undefined,
            n_outs: usize = 0,
            pos: [max_route_ids]u32 = undefined,
            n_pos: usize = 0,
        };

        /// Wave `w`'s positions grouped by bank in first appearance, each group's gate/up and activation.
        fn gateUpWave(self: *Self, g: *G, layer: u32, x: T, k: u32, sv: sdk_ext.expert.Served, w: u8, over: ?*const [n_banks]?Arrays) !Wave {
            var wave: Wave = .{};
            for (sv.waves, sv.refs, 0..) |wv, ref, pos| {
                if (wv != w) continue;
                const gi = for (wave.groups[0..wave.n], 0..) |gr, i| {
                    if (gr.bank == ref.bank) break i;
                } else blk: {
                    wave.groups[wave.n] = .{ .bank = ref.bank };
                    wave.n += 1;
                    break :blk wave.n - 1;
                };
                const gr = &wave.groups[gi];
                gr.pos[gr.n] = @intCast(pos);
                gr.n += 1;
            }
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]) orelse return error.SlotArraysUnbound;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |pos, *t, *r| {
                    t.* = @intCast(pos / k);
                    r.* = sv.refs[pos].row;
                }
                const nn: c_int = @intCast(gr.n);
                const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{nn}, .int32), 0);
                wave.ids[i] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{nn}, .uint32);
                wave.h[i] = try self.math.gateUp(g, xs, wave.ids[i], arrays.gate, arrays.up);
            }
            return wave;
        }

        fn downWave(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) ![]const T {
            const first = acc.n_outs;
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]) orelse return error.SlotArraysUnbound;
                acc.outs[acc.n_outs] = try self.math.down(g, wave.h[i], wave.ids[i], arrays.down);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |pos| {
                    acc.pos[acc.n_pos] = pos;
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

        /// One event wait for every array of the projections in `ps` (a struct of projections, or of optional ones):
        /// their aliases, read by the GPU after `value` and `deps` (one encoder break for the set, not one per array).
        fn waitAll(self: *Self, g: *G, ps: anytype, value: u64, deps: []const T) !@TypeOf(ps) {
            const P = @TypeOf(ps);
            var out = ps;
            if (G != ops.MlxOps) {
                inline for (comptime std.meta.fieldNames(P)) |pf| {
                    const p = @field(ps, pf);
                    if (@typeInfo(@TypeOf(p)) == .optional) {
                        if (p) |q| @field(out, pf) = try self.waitArrays(g, q, value, deps);
                    } else @field(out, pf) = try self.waitArrays(g, p, value, deps);
                }
                return out;
            }
            var xs: [32]T = undefined;
            var n: usize = 0;
            inline for (comptime std.meta.fieldNames(P)) |pf| {
                const po = @field(ps, pf);
                const pv = if (@typeInfo(@TypeOf(po)) == .optional) po else @as(?@TypeOf(po), po);
                if (pv) |p| inline for (comptime std.meta.fieldNames(@TypeOf(p))) |f| {
                    const v = @field(p, f);
                    const x: ?T = if (@TypeOf(v) == T) v else v;
                    if (x) |arr| {
                        xs[n] = arr;
                        n += 1;
                    }
                };
            }
            if (n == 0) return out;
            var os: [32]T = undefined;
            for (os[0..n]) |*o| o.* = mlx.mlx_array_new();
            expert_event.wait(xs[0..n], self.opt.event.?, value, deps, false, g.s, os[0..n]) catch |e| {
                for (os[0..n]) |o| _ = mlx.mlx_array_free(o);
                return e;
            };
            var i: usize = 0;
            inline for (comptime std.meta.fieldNames(P)) |pf| {
                const po = @field(ps, pf);
                const is_opt = @typeInfo(@TypeOf(po)) == .optional;
                const pv = if (is_opt) po else @as(?@TypeOf(po), po);
                if (pv) |p| {
                    var q = p;
                    inline for (comptime std.meta.fieldNames(@TypeOf(p))) |f| {
                        const v = @field(p, f);
                        const x: ?T = if (@TypeOf(v) == T) v else v;
                        if (x != null) {
                            @field(q, f) = try g.adopt(os[i]);
                            i += 1;
                        }
                    }
                    @field(out, pf) = q;
                }
            }
            return out;
        }

        /// An event wait's aliases of every array of one projection (`outs` fresh), read by the GPU after `value`.
        fn waitArrays(self: *Self, g: *G, p: anytype, value: u64, deps: []const T) !@TypeOf(p) {
            var out = p;
            inline for (comptime std.meta.fieldNames(@TypeOf(p))) |f| {
                const v = @field(p, f);
                const x: ?T = if (@TypeOf(v) == T) v else v;
                if (x) |arr| {
                    var o = [1]T{undefined};
                    if (G == ops.MlxOps) {
                        o[0] = mlx.mlx_array_new();
                        try expert_event.wait(&.{arr}, self.opt.event.?, value, deps, false, g.s, &o);
                        o[0] = try g.adopt(o[0]);
                    } else o[0] = try g.eventAlias(arr, value, deps.len);
                    @field(out, f) = o[0];
                }
            }
            return out;
        }

        /// The gated waves: every part's gate/up over bank arrays waited at `gu`, then part p's down over arrays
        /// waited at `down_first + p`, after every gate/up and the previous part's down. Nothing waits on the host.
        fn gatedParts(self: *Self, g: *G, layer: u32, x: T, k: u32, sv: sdk_ext.expert.Served, gates: sdk_ext.expert.Gates, acc: *Acc) !void {
            var gu: [n_banks]?Arrays = @splat(null);
            for (sv.waves, sv.refs) |w, ref| {
                const b = @backingInt(ref.bank);
                if (w == 0 or gu[b] != null) continue;
                const arrays = self.banks[layer][b] orelse return error.SlotArraysUnbound;
                const w2 = try self.waitAll(g, .{ .gate = arrays.gate, .up = arrays.up }, gates.gu, &.{});
                gu[b] = .{ .gate = w2.gate, .up = w2.up, .down = arrays.down };
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gateUpWave(g, layer, x, k, sv, @intCast(p + 1), &gu);
                for (waves[p].h[0..waves[p].n]) |h| {
                    hs[n_hs] = h;
                    n_hs += 1;
                }
            }
            var prev: []const T = &.{};
            for (0..sv.n_parts) |p| {
                var deps: [2 * max_route_ids]T = undefined;
                @memcpy(deps[0..n_hs], hs[0..n_hs]);
                @memcpy(deps[n_hs..][0..prev.len], prev);
                var dn = gu;
                const d0 = if (dn[0]) |arr| arr.down else null;
                const d1 = if (dn[1]) |arr| arr.down else null;
                const d2 = if (dn[2]) |arr| arr.down else null;
                const wd = try self.waitAll(g, .{ .b0 = d0, .b1 = d1, .b2 = d2 }, gates.down_first + p, deps[0 .. n_hs + prev.len]);
                if (dn[0]) |*arr| arr.down = wd.b0.?;
                if (dn[1]) |*arr| arr.down = wd.b1.?;
                if (dn[2]) |*arr| arr.down = wd.b2.?;
                prev = try self.downWave(g, layer, &waves[p], acc, &dn);
            }
        }

        /// The decode lane: the routed outputs `[n, k, hidden]` in routed order.
        fn callDecode(self: *Self, g: *G, layer: u32, x: T, indices: T, next_scores: ?T, n: u32, k: u32, hoist: []const T) !T {
            const n_ids = n * k;
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [sdk_ext.expert.lookahead.max_rows * 512]f32 = undefined;
            var sc: []const f32 = &.{};
            const read_next = next_scores != null and self.stream.route_lookahead and n * self.n_experts <= score_buf.len;
            if (read_next) {
                try g.evalAll(&.{ indices, next_scores.? });
                sc = try g.hostF32(next_scores.?, score_buf[0 .. n * self.n_experts]);
            }
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const r = try self.stream.route(layer, ids, sc);
            var released = false;
            errdefer if (!released) self.stream.release(r);
            var refs: [max_route_ids]SlotRef = undefined;
            var waves: [max_route_ids]u8 = undefined;
            _ = self.stream.refsOf(r, &refs);
            var bufs: [max_route_ids][max_route_ids]expert_policy.Load = undefined;
            var parts: [max_route_ids][]const expert_policy.Load = undefined;
            for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
            expert_stream.wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], waves[0..n_ids]);
            const sv: sdk_ext.expert.Served = .{ .refs = refs[0..n_ids], .waves = waves[0..n_ids], .n_parts = r.n_parts };
            var acc: Acc = .{};
            const hits = try self.gateUpWave(g, layer, x, k, sv, 0, null);
            // The hit wave and the hoisted arrays in one commit, ahead of every wait on a read.
            var early: [n_banks + 4]T = undefined;
            var n_early: usize = 0;
            if (hits.n > 0) for (try self.downWave(g, layer, &hits, &acc, null)) |o| {
                early[n_early] = o;
                n_early += 1;
            };
            for (hoist) |h| {
                if (n_early == early.len) break;
                early[n_early] = h;
                n_early += 1;
            }
            if (n_early > 0) try g.asyncEval(early[0..n_early]);
            if (self.opt.gated) {
                if (try self.stream.gate(r)) |gates| try self.gatedParts(g, layer, x, k, sv, gates, &acc);
            } else for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                try self.stream.waitGu(r, part);
                const wave = try self.gateUpWave(g, layer, x, k, sv, @intCast(p + 1), null);
                try g.asyncEval(wave.h[0..wave.n]);
                try self.stream.waitDown(r, part);
                try g.asyncEval(try self.downWave(g, layer, &wave, &acc, null));
            }
            self.stream.release(r);
            released = true;
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            for (acc.pos[0..acc.n_pos], 0..) |p, j| order[p] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// The wide lane: the weighted sum `[n, hidden]`.
        fn callWide(self: *Self, g: *G, layer: u32, x: T, indices: T, scores: T, n: u32, k: u32) !T {
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            const group_n = self.stream.max_route_ids;
            try w.ids.resize(a, n_ids);
            _ = try g.hostIds(indices, w.ids.items);
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            try w.count.resize(a, self.n_experts);
            @memset(w.count.items, 0);
            w.distinct.clearRetainingCapacity();
            for (w.ids.items) |e| {
                w.count.items[e] += 1;
                if (w.first.items[e] < 0) {
                    w.first.items[e] = 0;
                    try w.distinct.append(a, e);
                }
            }
            // Hottest first (rows descending, ties by id): the first groups' slots are the seed's.
            std.sort.pdq(u16, w.distinct.items, @as([]const u32, w.count.items), struct {
                fn lt(cnt: []const u32, p: u16, q: u16) bool {
                    return if (cnt[p] != cnt[q]) cnt[p] > cnt[q] else p < q;
                }
            }.lt);
            for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            try self.stream.awaitReadAhead(layer);
            try self.stream.seedPrefill(layer, w.ids.items);
            w.pos.clearRetainingCapacity();
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            const n_distinct = w.distinct.items.len;
            const n_groups = (n_distinct + group_n - 1) / group_n;
            const depth: usize = self.opt.wide_depth;
            var routes: [expert_stream.max_wide_depth]?*S.Route = @splat(null);
            errdefer for (&routes) |*rt| if (rt.*) |r| {
                self.stream.release(r);
                rt.* = null;
            };
            for (0..@min(depth, n_groups)) |gi| routes[gi % depth] = try self.stream.route(layer, groupOf(w.distinct.items, gi, group_n), &.{});
            for (0..n_groups) |gi| {
                const start = gi * group_n;
                const group = groupOf(w.distinct.items, gi, group_n);
                const r = routes[gi % depth].?;
                for (0..r.n_parts) |p| {
                    try self.stream.waitGu(r, @intCast(p));
                    try self.stream.waitDown(r, @intCast(p));
                }
                var refs: [max_route_ids]SlotRef = undefined;
                _ = self.stream.refsOf(r, &refs);
                const k0 = w.kept.items.len;
                const m = g.mark();
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    w.slot.clearRetainingCapacity();
                    w.act.clearRetainingCapacity();
                    const p0 = w.pos.items.len;
                    for (w.ids.items, 0..) |e, row| {
                        const fi: usize = @intCast(w.first.items[e]);
                        if (fi < start or fi >= start + group.len) continue;
                        const ref = refs[fi - start];
                        if (ref.bank != kind) continue;
                        try w.slot.append(a, ref.row);
                        try w.act.append(a, @intCast(row / k));
                        try w.pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0) continue;
                    const arrays = self.banks[layer][@backingInt(kind)] orelse return error.SlotArraysUnbound;
                    // The group's rows of this bank in slices (each slice's transient bounded, `group_slice_rows`).
                    var s0: usize = 0;
                    while (s0 < w.slot.items.len) : (s0 += glm.group_slice_rows) {
                        const s1 = @min(s0 + glm.group_slice_rows, w.slot.items.len);
                        const y = try self.math.prefill(g, layer, x, .{ .slot = w.slot.items[s0..s1], .act_row = w.act.items[s0..s1] }, arrays);
                        try w.kept.append(a, g.keep(y));
                    }
                    std.debug.assert(w.pos.items.len - p0 == w.slot.items.len);
                }
                try self.math.finishPrefill(g);
                // The group's waves drained before its slots go back (the next route may refill them).
                try g.evalAll(w.kept.items[k0..]);
                g.resetTo(m);
                self.stream.release(r);
                routes[gi % depth] = null;
                if (gi + depth < n_groups) routes[gi % depth] = try self.stream.route(layer, groupOf(w.distinct.items, gi + depth, group_n), &.{});
            }
            // The join: every routed row's output in routed order, then the combine in token slices.
            const joined = try g.concat(w.kept.items, 0);
            try g.evalAll(&.{joined});
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            try w.inv.resize(a, n_ids);
            for (w.pos.items, 0..) |p, j| w.inv.items[p] = @intCast(j);
            var outs: std.ArrayList(T) = .empty;
            defer outs.deinit(a);
            const ts: u32 = @intCast(glm.combine_slice_tokens);
            var t0: u32 = 0;
            while (t0 < n) : (t0 += ts) {
                const t1 = @min(t0 + ts, n);
                const rows: c_int = @intCast((t1 - t0) * k);
                const ord = try g.hostArray(std.mem.sliceAsBytes(w.inv.items[t0 * k .. t1 * k]), &.{rows}, .uint32);
                const y = try g.reshape(try g.take(joined, ord, 0), &.{ @intCast(t1 - t0), @intCast(k), self.hidden });
                const s = try g.slice(scores, &.{ @intCast(t0), 0 }, &.{ @intCast(t1), @intCast(k) }, &.{ 1, 1 });
                try outs.append(a, try combine(g, y, s, g.dtypeOf(x)));
            }
            return if (outs.items.len == 1) outs.items[0] else g.concat(outs.items, 0);
        }

        fn groupOf(d: []const u16, gi: usize, group_n: usize) []const u16 {
            const s0 = gi * group_n;
            return d[s0..@min(s0 + group_n, d.len)];
        }
    };
}
