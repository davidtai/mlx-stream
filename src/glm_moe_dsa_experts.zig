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
//! event-gate aliases), the release; the outputs joined in routed order. A wider call is the wide lane: the layer's
//! residency seeded from the call's ids, the call's experts in groups of `max_route_ids` (its residents first, then
//! the rest, each hottest first) routed up to `wide_depth` ahead, each group's rows per bank through the math's
//! `prefill` in slices, drained before the group's slots go back; the join and the combine in token slices.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("sdk_ext.zig");
const expert_policy = sdk_ext.expert.policy;
const expert_event = sdk_ext.expert.event;
const expert_stream = @import("expert_stream.zig");
const ops = @import("deepseek_v41_ops.zig");
const glm = @import("glm_moe_dsa.zig");
const pt = @import("glm_moe_dsa_prefill_timers.zig");

/// The wide lane's combine (`combineOnGpu`): one thread per token and 4 hidden values, the token's `K` routed rows in
/// routed order, each read from its part (`loc`: the part in the top 8 bits, the row in the low 24).
const max_combine_parts = 24;
const combine_source = blk: {
    var cases: []const u8 = "";
    for (0..max_combine_parts) |i| cases = cases ++ std.fmt.comptimePrint("        case {d}: part = p{d}; break;\n", .{ i, i });
    break :blk std.fmt.comptimePrint("{s}{s}{s}", .{ combine_head, cases, combine_tail });
};
const combine_head =
    \\    const uint t = thread_position_in_grid.y;
    \\    const uint h = thread_position_in_grid.x * 4;
    \\    if (h >= H) return;
    \\    float4 acc = float4(0.0f);
    \\    for (int k = 0; k < K; ++k) {
    \\        const uint lc = loc[t * K + k];
    \\        const device T* part;
    \\        switch (lc >> 24) {
    \\
;
const combine_tail =
    \\        default: part = p0; break;
    \\        }
    \\        const vec<T, 4> y = *(const device vec<T, 4>*)(part + ulong(lc & 0xFFFFFFu) * H + h);
    \\        acc = acc + float4(y) * w[t * K + k];
    \\    }
    \\    *(device vec<T, 4>*)(out + ulong(t) * H + h) = vec<T, 4>(acc);
    \\
;

/// `(y * scores[..., None]).sum(-2)` of the routed rows in `parts` (at most `max_combine_parts`; token t's j-th routed
/// row is row `loc[t * k + j] & 0xFFFFFF` of part `loc[t * k + j] >> 24`, u32 `[n * k]`) as one launch: each token's
/// rows weighted and summed in f32 in routed order, products and sums rounded apart (no contraction), cast to `dt`;
/// the op chain's bits.
fn combineOnGpu(slot: *?mlx.mlx_fast_metal_kernel, g: *ops.MlxOps, parts: []const ops.MlxOps.T, loc: ops.MlxOps.T, scores: ops.MlxOps.T, n: u32, k: u32, hidden: c_int, dt: ops.Dtype) !ops.MlxOps.T {
    std.debug.assert(parts.len > 0 and parts.len <= max_combine_parts);
    const names = comptime blk: {
        var v: [max_combine_parts + 2][*:0]const u8 = undefined;
        for (0..max_combine_parts) |i| v[i] = std.fmt.comptimePrint("p{d}", .{i});
        v[max_combine_parts] = "loc";
        v[max_combine_parts + 1] = "w";
        break :blk v;
    };
    const kern = slot.* orelse blk: {
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = mlx.mlx_fast_metal_kernel_new("glm_moe_dsa_combine", iv, ov, combine_source, "#pragma METAL fp contract(off)\n", true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = kn;
        break :blk kn;
    };
    const h4: c_int = @divExact(hidden, 4);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @intCast(n), hidden }, 2, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, h4, @intCast(n), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(h4, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(k)));
    // The slots past the parts hold the first part (never read).
    var inputs: [max_combine_parts + 2]mlx.mlx_array = undefined;
    for (inputs[0..max_combine_parts], 0..) |*in, i| in.* = if (i < parts.len) parts[i] else parts[0];
    inputs[max_combine_parts] = loc;
    inputs[max_combine_parts + 1] = scores;
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&out, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}

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
        /// The wide lane's combine on the GPU (`combine_source`), built at its first call.
        combine_kernel: ?mlx.mlx_fast_metal_kernel = null,

        /// The wide lane's host scratch, reused across calls.
        const Wide = struct {
            ids: std.ArrayList(u16) = .empty,
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            order: std.ArrayList(u16) = .empty,
            count: std.ArrayList(u32) = .empty,
            /// The call's routed rows by group (each group's in row order) and each group's first index there.
            by_group: std.ArrayList(u32) = .empty,
            group_at: std.ArrayList(u32) = .empty,
            slot: std.ArrayList(u32) = .empty,
            act: std.ArrayList(u32) = .empty,
            pos: std.ArrayList(u32) = .empty,
            inv: std.ArrayList(u32) = .empty,
            kept: std.ArrayList(T) = .empty,

            fn deinit(w: *Wide, a: std.mem.Allocator, g: *G) void {
                for (w.kept.items) |x| g.release(x);
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.order, &w.count, &w.by_group, &w.group_at, &w.slot, &w.act, &w.pos, &w.inv, &w.kept }) |l| l.deinit(a);
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
            if (self.combine_kernel) |kern| _ = mlx.mlx_fast_metal_kernel_free(kern);
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
        /// decode lookahead's predictor), evaluated with the ids at the routing barrier; null for none.
        pub fn call(self: *Self, g: *G, layer: u32, x: T, indices: T, scores: T, next_scores: ?T) !T {
            const n: u32 = @intCast(g.shapeOf(x).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            if (n * k > self.stream.max_route_ids) return self.callWide(g, layer, x, indices, scores, n, k);
            const y = try self.callDecode(g, layer, x, indices, next_scores, n, k);
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
                gu[b] = .{ .gate = try self.waitArrays(g, arrays.gate, gates.gu, &.{}), .up = try self.waitArrays(g, arrays.up, gates.gu, &.{}), .down = arrays.down };
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
                for (&dn) |*d| if (d.*) |*arr| {
                    arr.down = try self.waitArrays(g, arr.down, gates.down_first + p, deps[0 .. n_hs + prev.len]);
                };
                prev = try self.downWave(g, layer, &waves[p], acc, &dn);
            }
        }

        /// The decode lane: the routed outputs `[n, k, hidden]` in routed order.
        fn callDecode(self: *Self, g: *G, layer: u32, x: T, indices: T, next_scores: ?T, n: u32, k: u32) !T {
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
            if (hits.n > 0) try g.asyncEval(try self.downWave(g, layer, &hits, &acc, null));
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
            var tp = pt.now();
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
            pt.chargeRouted(.barrier, tp);
            tp = pt.now();
            try self.stream.awaitReadAhead(layer);
            pt.chargeRouted(.ahead, tp);
            tp = pt.now();
            try self.stream.seedPrefill(layer, w.ids.items);
            // The call's residents first (hottest first among them), then the rest: every resident is routed before any
            // miss is planned, so no miss evicts a resident (one the read-ahead landed) whose group is still to come and
            // each routed expert is read at most once in the call.
            const policy = &self.stream.layers[layer].policy;
            var n_res: usize = 0;
            for (w.distinct.items) |e| n_res += @intFromBool(policy.slotOf(e) != null);
            try w.order.resize(a, w.distinct.items.len);
            var at_res: usize = 0;
            var at_miss: usize = n_res;
            for (w.distinct.items) |e| {
                if (policy.slotOf(e) != null) {
                    w.order.items[at_res] = e;
                    at_res += 1;
                } else {
                    w.order.items[at_miss] = e;
                    at_miss += 1;
                }
            }
            @memcpy(w.distinct.items, w.order.items);
            for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            w.pos.clearRetainingCapacity();
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            const n_distinct = w.distinct.items.len;
            const n_groups = (n_distinct + group_n - 1) / group_n;
            // The routed rows bucketed by group once (a counting sort, each group's rows in row order).
            try w.group_at.resize(a, n_groups + 1);
            @memset(w.group_at.items, 0);
            for (w.ids.items) |e| w.group_at.items[@as(usize, @intCast(w.first.items[e])) / group_n + 1] += 1;
            for (1..n_groups + 1) |gi| w.group_at.items[gi] += w.group_at.items[gi - 1];
            try w.by_group.resize(a, n_ids);
            try w.slot.resize(a, n_groups);
            @memcpy(w.slot.items, w.group_at.items[0..n_groups]);
            for (w.ids.items, 0..) |e, row| {
                const gi = @as(usize, @intCast(w.first.items[e])) / group_n;
                w.by_group.items[w.slot.items[gi]] = @intCast(row);
                w.slot.items[gi] += 1;
            }
            const depth: usize = self.opt.wide_depth;
            var routes: [expert_stream.max_wide_depth]?*S.Route = @splat(null);
            errdefer for (&routes) |*rt| if (rt.*) |r| {
                self.stream.release(r);
                rt.* = null;
            };
            for (0..@min(depth, n_groups)) |gi| routes[gi % depth] = try self.stream.route(layer, groupOf(w.distinct.items, gi, group_n), &.{});
            pt.chargeRouted(.route, tp);
            for (0..n_groups) |gi| {
                const start = gi * group_n;
                const r = routes[gi % depth].?;
                const tw = pt.now();
                for (0..r.n_parts) |p| {
                    try self.stream.waitGu(r, @intCast(p));
                    try self.stream.waitDown(r, @intCast(p));
                }
                pt.chargeRouted(.wait, tw);
                const tc = pt.now();
                var refs: [max_route_ids]SlotRef = undefined;
                _ = self.stream.refsOf(r, &refs);
                const k0 = w.kept.items.len;
                const m = g.mark();
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    w.slot.clearRetainingCapacity();
                    w.act.clearRetainingCapacity();
                    const p0 = w.pos.items.len;
                    for (w.by_group.items[w.group_at.items[gi]..w.group_at.items[gi + 1]]) |row| {
                        const fi: usize = @intCast(w.first.items[w.ids.items[row]]);
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
                pt.chargeRouted(.encode, tc);
                const td = pt.now();
                // The group's waves drained before its slots go back (the next route may refill them).
                try g.evalAll(w.kept.items[k0..]);
                pt.chargeRouted(.compute, td);
                g.resetTo(m);
                const tr = pt.now();
                self.stream.release(r);
                routes[gi % depth] = null;
                if (gi + depth < n_groups) routes[gi % depth] = try self.stream.route(layer, groupOf(w.distinct.items, gi + depth, group_n), &.{});
                pt.chargeRouted(.route, tr);
            }
            try w.inv.resize(a, n_ids);
            // On the GPU the combine reads each routed row where its group's call left it (no join).
            if (G == ops.MlxOps and mlx.streamIsGpu(g.s) and @rem(self.hidden, 4) == 0 and w.kept.items.len <= max_combine_parts) {
                const tb = pt.now();
                defer if (pt.enabled) pt.chargeRouted(.combine, tb);
                var j: usize = 0;
                for (w.kept.items, 0..) |part, pi| for (0..@intCast(g.shapeOf(part).dim(0))) |row| {
                    w.inv.items[w.pos.items[j]] = @as(u32, @intCast(pi)) << 24 | @as(u32, @intCast(row));
                    j += 1;
                };
                const loc = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
                const out = try combineOnGpu(&self.combine_kernel, g, w.kept.items, loc, scores, n, k, self.hidden, g.dtypeOf(x));
                // The combine holds the parts until it is evaluated.
                for (w.kept.items) |o| g.release(o);
                w.kept.clearRetainingCapacity();
                return out;
            }
            // The join: every routed row's output in routed order, then the combine in token slices.
            const tj = pt.now();
            const joined = try g.concat(w.kept.items, 0);
            try g.evalAll(&.{joined});
            pt.chargeRouted(.join, tj);
            const tb = pt.now();
            defer if (pt.enabled) pt.chargeRouted(.combine, tb);
            for (w.kept.items) |o| g.release(o);
            w.kept.clearRetainingCapacity();
            for (w.pos.items, 0..) |p, j| w.inv.items[p] = @intCast(j);
            if (G == ops.MlxOps and mlx.streamIsGpu(g.s) and @rem(self.hidden, 4) == 0) {
                const inv = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
                return combineOnGpu(&self.combine_kernel, g, &.{joined}, inv, scores, n, k, self.hidden, g.dtypeOf(x));
            }
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

test "glm experts: the wide lane's GPU combine over its parts equals the op chain's bits at GLM-5.3's hidden size" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    var slot: ?mlx.mlx_fast_metal_kernel = null;
    defer if (slot) |kern| {
        _ = mlx.mlx_fast_metal_kernel_free(kern);
    };
    const n: u32 = 300;
    const k: u32 = glm.routed_top_k;
    const hidden: c_int = 6144;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    const yv = try a.alloc(f32, n * k * @as(usize, @intCast(hidden)));
    defer a.free(yv);
    for (yv) |*v| v.* = rand.floatNorm(f32);
    const wv = try a.alloc(f32, n * k);
    defer a.free(wv);
    for (wv) |*v| v.* = rand.float(f32);
    // The joined rows in a shuffled order and its inverse, as the wide lane's groups leave them.
    const inv = try a.alloc(u32, n * k);
    defer a.free(inv);
    for (inv, 0..) |*v, i| v.* = @intCast(i);
    rand.shuffle(u32, inv);
    const joined = try g.astype(try g.hostArray(std.mem.sliceAsBytes(yv), &.{ @intCast(n * k), hidden }, .float32), .bfloat16);
    const scores = try g.hostArray(std.mem.sliceAsBytes(wv), &.{ @intCast(n), @intCast(k) }, .float32);
    const ord = try g.hostArray(std.mem.sliceAsBytes(inv), &.{@intCast(n * k)}, .uint32);
    // The rows in 3 parts, each row's place as (part, row).
    const cuts = [_]c_int{ 0, 700, 1900, @intCast(n * k) };
    var parts: [3]ops.MlxOps.T = undefined;
    for (&parts, 0..) |*p, i| p.* = try g.slice(joined, &.{ cuts[i], 0 }, &.{ cuts[i + 1], hidden }, &.{ 1, 1 });
    const locv = try a.alloc(u32, n * k);
    defer a.free(locv);
    for (inv, locv) |r, *l| {
        const pi: u32 = if (r < cuts[1]) 0 else if (r < cuts[2]) 1 else 2;
        l.* = pi << 24 | (r - @as(u32, @intCast(cuts[pi])));
    }
    const loc = try g.hostArray(std.mem.sliceAsBytes(locv), &.{@intCast(n * k)}, .uint32);
    const got = try combineOnGpu(&slot, &g, &parts, loc, scores, n, k, hidden, .bfloat16);
    const y = try g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), hidden });
    const want = try g.astype(try g.sum(try g.mul(y, try g.expandDims(scores, -1)), -2, false), .bfloat16);
    const same = try g.astype(try g.equal(got, want), .float32);
    const all = try g.sum(try g.sum(same, -1, false), -1, false);
    var out: [1]f32 = undefined;
    _ = try g.hostF32(all, &out);
    try std.testing.expectEqual(@as(f32, @floatFromInt(n * @as(u32, @intCast(hidden)))), out[0]);
}
