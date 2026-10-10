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
/// PROFILE builds (`-Dplugin-profile=true`): each decode-lane call's host stamps on the verify's GPU timeline.
const timeline = @import("dsv41_verify_timeline.zig");

pub const BankKind = sdk_ext.expert.BankKind;
pub const SlotRef = sdk_ext.expert.SlotRef;
pub const max_route_ids = expert_policy.max_route_ids;
const n_banks = std.meta.fieldNames(BankKind).len;

/// The driver's construction-time routes.
/// The decode lane's host time per call, summed: blocked in the routing barrier (the GPU up to the router, with event
/// gates also its wait on the previous layer's bytes), the routes (the ids read, the plans, the reads submitted, the
/// lookahead's step), the waves' encode and commit.
pub const HostSplit = struct { calls: u64 = 0, barrier_ns: u64 = 0, route_ns: u64 = 0, wave_ns: u64 = 0 };

/// A monotonic clock for the host split (ns).
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

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
        /// Stream layers per routed layer (GLM-5.3's EXL3 bank: its K3 and K4 bank layers): a call's ids split by the
        /// bank layer that holds each expert, one route each.
        pub const bpl: u32 = if (@hasDecl(Bk, "banks_per_layer")) Bk.banks_per_layer else 1;
        /// The quant fuses gate, up and down (`M.fused`): each wave runs once all its segments landed.
        const fused = @hasDecl(M, "fused_waves") and M.fused_waves;
        comptime {
            if (bpl > 1 and !fused) @compileError("a bank of several bank layers per routed layer streams through a fused quant");
        }

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
        /// The decode lane's read bytes by the first row of the call that routes the expert (row i of `x`), summed over
        /// calls until the caller zeroes it: a draft round's verify attributes its reads to the rows it keeps or rejects.
        row_bytes: [max_route_ids]u64 = @splat(0),
        /// The decode lane's host time, summed over calls until the caller zeroes it (`HostSplit`).
        host: HostSplit = .{},

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
            side_ids: std.ArrayList(u16) = .empty,

            fn deinit(w: *Wide, a: std.mem.Allocator, g: *G) void {
                for (w.kept.items) |x| g.release(x);
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.count, &w.slot, &w.act, &w.pos, &w.inv, &w.kept, &w.side_ids }) |l| l.deinit(a);
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
            if (bpl == 1) return self.stream.readAheadSeed(layer, experts) else {
                // One read-ahead is live at a time: the first bank layer's share of the seed, hottest first.
                var buf: [512]u16 = undefined;
                var n: usize = 0;
                for (experts) |e| if (n < buf.len and self.stream.bank.streamLayer(layer, e) == bpl * layer) {
                    buf[n] = e;
                    n += 1;
                };
                return self.stream.readAheadSeed(bpl * layer, buf[0..n]);
            }
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
            const y = if (bpl > 1) try self.callDecodeSplit(g, layer, x, indices, next_scores, n, k, hoist) else try self.callDecode(g, layer, x, indices, next_scores, n, k, hoist);
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
            const t0 = nowNs();
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [sdk_ext.expert.lookahead.max_rows * 512]f32 = undefined;
            var sc: []const f32 = &.{};
            const read_next = next_scores != null and self.stream.route_lookahead and n * self.n_experts <= score_buf.len;
            if (read_next) {
                try g.evalAll(&.{ indices, next_scores.? });
                sc = try g.hostF32(next_scores.?, score_buf[0 .. n * self.n_experts]);
            }
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const t1 = nowNs();
            const r = try self.stream.route(layer, ids, sc);
            const t2 = nowNs();
            defer self.noteHost(t0, t1, t2);
            var released = false;
            errdefer if (!released) self.stream.release(r);
            const record = self.stream.bank.layers[layer].logical_bytes;
            for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, read| if (read) {
                const at = std.mem.indexOfScalar(u16, ids, l.expert) orelse continue;
                self.row_bytes[at / k] += record;
            };
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

        /// One route of a split call: its stream layer, its ids and their positions in the call (routed order), its
        /// slot refs and each position's wave.
        const Side = struct {
            sl: u32 = 0,
            n: u32 = 0,
            ids: [max_route_ids]u16 = undefined,
            pos: [max_route_ids]u8 = undefined,
            r: ?*S.Route = null,
            refs: [max_route_ids]SlotRef = undefined,
            waves: [max_route_ids]u8 = undefined,
        };

        /// The decode lane over a routed layer's `bpl` bank layers: the ids split by the bank layer that holds each
        /// expert and every route made before any wave (their reads in flight together; the last route settles the
        /// lookahead and reads ahead the next routed layer's experts over all its bank layers), every route's hit wave
        /// and the hoisted arrays in one commit, then each route's parts as fused waves, each once all its segments
        /// landed; the outputs `[n, k, hidden]` in routed order.
        fn callDecodeSplit(self: *Self, g: *G, layer: u32, x: T, indices: T, next_scores: ?T, n: u32, k: u32, hoist: []const T) !T {
            const n_ids = n * k;
            const bank = self.stream.bank;
            const t0 = nowNs();
            if (comptime timeline.enabled) timeline.point(layer, .call, self.readGauge(), false);
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [sdk_ext.expert.lookahead.max_rows * 512]f32 = undefined;
            var sc: []const f32 = &.{};
            const read_next = next_scores != null and self.stream.route_lookahead and n * self.n_experts <= score_buf.len;
            if (read_next) {
                try g.evalAll(&.{ indices, next_scores.? });
                sc = try g.hostF32(next_scores.?, score_buf[0 .. n * self.n_experts]);
            }
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const t1 = nowNs();
            if (comptime timeline.enabled) timeline.point(layer, .barrier, 0, false);
            var sides: [bpl]Side = @splat(.{});
            for (&sides, 0..) |*sd, i| sd.sl = bpl * layer + @as(u32, @intCast(i));
            for (ids, 0..) |e, p| {
                const sd = &sides[bank.streamLayer(layer, e) - bpl * layer];
                sd.ids[sd.n] = e;
                sd.pos[sd.n] = @intCast(p);
                sd.n += 1;
            }
            errdefer for (&sides) |*sd| if (sd.r) |r| {
                self.stream.release(r);
                sd.r = null;
            };
            var last: usize = 0;
            for (sides, 0..) |sd, i| if (sd.n > 0) {
                last = i;
            };
            // The route recorder (a diagnostic): one route of the call, each bank layer's route flags its own ids.
            if (self.stream.recorder) |rec| if (self.stream.phase == .decode) rec.call(layer, ids);
            defer if (self.stream.recorder) |rec| rec.endCall();
            for (&sides, 0..) |*sd, i| {
                if (sd.n == 0) continue;
                const r = if (i == last) try self.stream.route(sd.sl, sd.ids[0..sd.n], sc) else try self.stream.routeHeld(sd.sl, sd.ids[0..sd.n]);
                sd.r = r;
                const geom = &bank.layers[sd.sl];
                const record = @as(u64, S.minisOf(geom)) * geom.logical_bytes;
                for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, read| if (read) {
                    const at = std.mem.indexOfScalar(u16, sd.ids[0..sd.n], l.expert) orelse continue;
                    self.row_bytes[sd.pos[at] / k] += record;
                };
                _ = self.stream.refsOf(r, &sd.refs);
                var bufs: [max_route_ids][max_route_ids]expert_policy.Load = undefined;
                var parts: [max_route_ids][]const expert_policy.Load = undefined;
                for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
                expert_stream.wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], sd.waves[0..sd.n]);
            }
            if (comptime timeline.enabled) timeline.point(layer, .route, 0, false);
            const t2 = nowNs();
            defer self.noteHost(t0, t1, t2);
            var acc: Acc = .{};
            // Every route's hit wave and the hoisted arrays in one commit, ahead of every wait on a read.
            var early: [bpl * n_banks + 4]T = undefined;
            var n_early: usize = 0;
            for (&sides) |*sd| if (sd.r != null) for (try self.fusedWave(g, sd, x, k, 0, &acc, null)) |o| {
                early[n_early] = o;
                n_early += 1;
            };
            for (hoist) |h| {
                if (n_early == early.len) break;
                early[n_early] = h;
                n_early += 1;
            }
            if (n_early > 0) try g.asyncEval(early[0..n_early]);
            if (comptime timeline.enabled) timeline.point(layer, .hit, 0, false);
            var prev: []const T = &.{};
            var missed = false;
            var gu_first: u64 = 0;
            for (&sides) |*sd| if (sd.r) |r| {
                missed = missed or r.n_parts > 0;
                if (self.opt.gated) {
                    const gates = (try self.stream.gate(r)) orelse continue;
                    if (gu_first == 0) gu_first = gates.gu;
                    // The layer's gate/up value is its first route's, its last down value its last route's.
                    if (comptime timeline.enabled) timeline.gate(layer, gu_first, gates.down_first, gates.n_parts);
                    for (0..r.n_parts) |p| {
                        // Part p's banks waited at its down gate: every gate/up of the route and its own down landed.
                        var over: [n_banks]?Arrays = @splat(null);
                        for (sd.waves[0..sd.n], sd.refs[0..sd.n]) |w, ref| {
                            const b = @backingInt(ref.bank);
                            if (w != p + 1 or over[b] != null) continue;
                            const arrays = self.banks[sd.sl][b] orelse return error.SlotArraysUnbound;
                            over[b] = try self.waitAll(g, arrays, gates.down_first + p, prev);
                        }
                        prev = try self.fusedWave(g, sd, x, k, @intCast(p + 1), &acc, &over);
                    }
                } else for (0..r.n_parts) |p| {
                    try self.stream.waitGu(r, @intCast(p));
                    try self.stream.waitDown(r, @intCast(p));
                    try g.asyncEval(try self.fusedWave(g, sd, x, k, @intCast(p + 1), &acc, null));
                }
            };
            if (comptime timeline.enabled) timeline.point(layer, .end, self.readGauge(), missed);
            for (&sides) |*sd| if (sd.r) |r| {
                self.stream.release(r);
                sd.r = null;
            };
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            for (acc.pos[0..acc.n_pos], 0..) |p, j| order[p] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// Side `sd`'s positions in wave `w`, grouped by bank in first appearance: each group's rows of x through the
        /// quant's fused wave (over `over`'s waited arrays when given), their positions in the call recorded in `acc`.
        fn fusedWave(self: *Self, g: *G, sd: *const Side, x: T, k: u32, w: u8, acc: *Acc, over: ?*const [n_banks]?Arrays) ![]const T {
            const first = acc.n_outs;
            var groups: [n_banks]Group = undefined;
            var ng: usize = 0;
            for (sd.waves[0..sd.n], sd.refs[0..sd.n], 0..) |wv, ref, i| {
                if (wv != w) continue;
                const gi = for (groups[0..ng], 0..) |gr, j| {
                    if (gr.bank == ref.bank) break j;
                } else blk: {
                    groups[ng] = .{ .bank = ref.bank };
                    ng += 1;
                    break :blk ng - 1;
                };
                groups[gi].pos[groups[gi].n] = @intCast(i);
                groups[gi].n += 1;
            }
            for (groups[0..ng]) |*gr| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[sd.sl][@backingInt(gr.bank)]) orelse return error.SlotArraysUnbound;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |i, *t, *r| {
                    t.* = @intCast(sd.pos[i] / k);
                    r.* = sd.refs[i].row;
                }
                const nn: c_int = @intCast(gr.n);
                const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{nn}, .int32), 0);
                acc.outs[acc.n_outs] = try self.math.fused(g, xs, rows[0..gr.n], arrays);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |i| {
                    acc.pos[acc.n_pos] = sd.pos[i];
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

        /// A decode-lane call's host split: barrier [t0, t1), routes [t1, t2), waves [t2, now).
        fn noteHost(self: *Self, t0: u64, t1: u64, t2: u64) void {
            const t3 = nowNs();
            self.host.calls += 1;
            self.host.barrier_ns += t1 -| t0;
            self.host.route_ns += t2 -| t1;
            self.host.wave_ns += t3 -| t2;
        }

        /// The read pool's wall time with a read in flight (ns): the timeline's read gauge.
        fn readGauge(self: *const Self) u64 {
            return @intCast(@max(self.stream.pool.readGauge()[4], 0));
        }

        /// The wide lane: the weighted sum `[n, hidden]`.
        fn callWide(self: *Self, g: *G, layer: u32, x: T, indices: T, scores: T, n: u32, k: u32) !T {
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            try w.ids.resize(a, n_ids);
            _ = try g.hostIds(indices, w.ids.items);
            w.pos.clearRetainingCapacity();
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            for (0..bpl) |side| try self.wideBank(g, layer, @intCast(side), x, k);
            return self.wideJoin(g, scores, x, n, k);
        }

        /// The wide lane's groups on stream layer `bpl x layer + side`: the routed rows whose expert it holds, their
        /// outputs appended to `wide.kept` and their positions to `wide.pos`.
        fn wideBank(self: *Self, g: *G, layer: u32, side: u32, x: T, k: u32) !void {
            const a = self.a;
            const w = &self.wide;
            const sl = bpl * layer + side;
            const group_n = self.stream.max_route_ids;
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            try w.count.resize(a, self.n_experts);
            @memset(w.count.items, 0);
            w.distinct.clearRetainingCapacity();
            w.side_ids.clearRetainingCapacity();
            for (w.ids.items) |e| {
                if (bpl > 1 and self.stream.bank.streamLayer(layer, e) != sl) continue;
                try w.side_ids.append(a, e);
                w.count.items[e] += 1;
                if (w.first.items[e] < 0) {
                    w.first.items[e] = 0;
                    try w.distinct.append(a, e);
                }
            }
            if (w.distinct.items.len == 0) return;
            // Hottest first (rows descending, ties by id): the first groups' slots are the seed's.
            std.sort.pdq(u16, w.distinct.items, @as([]const u32, w.count.items), struct {
                fn lt(cnt: []const u32, p: u16, q: u16) bool {
                    return if (cnt[p] != cnt[q]) cnt[p] > cnt[q] else p < q;
                }
            }.lt);
            for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            try self.stream.awaitReadAhead(sl);
            try self.stream.seedPrefill(sl, w.side_ids.items);
            const n_distinct = w.distinct.items.len;
            const n_groups = (n_distinct + group_n - 1) / group_n;
            const depth: usize = self.opt.wide_depth;
            var routes: [expert_stream.max_wide_depth]?*S.Route = @splat(null);
            errdefer for (&routes) |*rt| if (rt.*) |r| {
                self.stream.release(r);
                rt.* = null;
            };
            for (0..@min(depth, n_groups)) |gi| routes[gi % depth] = try self.stream.route(sl, groupOf(w.distinct.items, gi, group_n), &.{});
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
                        if (bpl > 1 and self.stream.bank.streamLayer(layer, e) != sl) continue;
                        const fi: usize = @intCast(w.first.items[e]);
                        if (fi < start or fi >= start + group.len) continue;
                        const ref = refs[fi - start];
                        if (ref.bank != kind) continue;
                        try w.slot.append(a, ref.row);
                        try w.act.append(a, @intCast(row / k));
                        try w.pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0) continue;
                    const arrays = self.banks[sl][@backingInt(kind)] orelse return error.SlotArraysUnbound;
                    // The group's rows of this bank in slices (each slice's transient bounded, `group_slice_rows`).
                    var s0: usize = 0;
                    while (s0 < w.slot.items.len) : (s0 += glm.group_slice_rows) {
                        const s1 = @min(s0 + glm.group_slice_rows, w.slot.items.len);
                        const y = try self.math.prefill(g, sl, x, .{ .slot = w.slot.items[s0..s1], .act_row = w.act.items[s0..s1] }, arrays);
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
                if (gi + depth < n_groups) routes[gi % depth] = try self.stream.route(sl, groupOf(w.distinct.items, gi + depth, group_n), &.{});
            }
        }

        /// The wide lane's join: every routed row's output in routed order, then the combine in token slices.
        fn wideJoin(self: *Self, g: *G, scores: T, x: T, n: u32, k: u32) !T {
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
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
