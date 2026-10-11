//! GLM-5.3's EXL3 routed-expert quant (C2, `sdk_ext.quant`): sushi's EXL3 MoE (`mlx_host.sushi_exl3`: the mcg
//! codebook, SwiGLU unclamped, GLM has no swiglu_limit) over the EXL3 bank's slot arrays (`glm_moe_dsa_exl3_bank`).
//! A routed (token, expert) pair is one sushi row: the expert's `tp` minis (slot x tp + rank) at score 1, the
//! router's score being the arch's combine. Sushi's chain fuses gate, up and down, so the decode lane hands a wave
//! over once all its segments landed (`fused`, in place of `gateUp` / `down`, which refuse); a wave of any width runs
//! the decode chain. The prompt's slices run sushi's MoE (its prefill GEMM past its decode rows: `prefill`). The served
//! EXL3 module binds GLM-5.3's own kernels (`glm53_exl3_quant`), which take this binding's decode chain; the binding is
//! also their reference (`Exl3SushiModule`).

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("sdk_ext.zig");
const quant = sdk_ext.quant;
const bank_mod = @import("glm_moe_dsa_exl3_bank.zig");
const sushi = @import("mlx_host").sushi_exl3;

const Allocator = std.mem.Allocator;
const Diag = quant.Diag;

pub const name = "glm-exl3-sushi";
pub const Arrays = bank_mod.Proj;

/// The bank's `quantization`: exl3, the mcg codebook at sushi's multiplier, `tp_ranks` minis per expert.
fn parse(peek: *const quant.BankPeek, why: ?*Diag) ?u32 {
    const q = peek.quantization;
    const mode = quant.str(q, "mode") orelse "";
    const cb = quant.str(q, "codebook") orelse "";
    const mult = quant.int(q, "codebook_multiplier") orelse -1;
    const tp = quant.int(q, "tp_ranks") orelse 0;
    if (!std.mem.eql(u8, mode, "exl3") or !std.mem.eql(u8, cb, "mcg") or mult != sushi.format.MCG_MULT or tp < 1) {
        _ = quant.decline(why, "quant " ++ name ++ ": quantization {s} / {s} / multiplier {d} / {d} ranks (serves exl3 mcg at {d})", .{ mode, cb, mult, tp, sushi.format.MCG_MULT });
        return null;
    }
    return @intCast(tp);
}

pub fn claims(peek: *const quant.BankPeek, why: ?*Diag) ?quant.Priority {
    _ = parse(peek, why) orelse return null;
    return .native;
}

pub fn Accepted(comptime G: type) type {
    return struct {
        const Self = @This();
        const A = Arrays(G.T);
        const Bank = quant.BankArrays(A);
        /// Any width: the decode lane's waves run the decode chain (`fused`).
        pub const max_decode_rows: u32 = sdk_ext.expert.policy.max_route_ids;
        /// The decode lane calls `fused` per wave once the wave's every segment landed.
        pub const fused_waves = true;
        a: Allocator,
        tp: u32,
        spec: quant.Spec,
        /// Host scratch: each pair's minis, its x row, the scores.
        minis: std.ArrayList(u32) = .empty,
        src: std.ArrayList(u32) = .empty,
        ones: std.ArrayList(f32) = .empty,

        /// Each projection: trellis uint16 `[rows x tp, in/16, out/16, 16K]` at a K sushi decodes, suh float16
        /// `[rows x tp, in]`, svh float16 `[rows x tp, out]`.
        pub fn checkBank(self: *const Self, g: *G, bank: Bank, diag: *Diag) !void {
            _ = g;
            const h = self.spec.hidden;
            const mini = self.spec.inter / self.tp;
            const rows = mlx.getShape(bank.gate.trellis)[0];
            if (rows <= 0 or @mod(rows, @as(c_int, @intCast(self.tp))) != 0) return quant.refuse(diag, error.BankArrays, "quant " ++ name ++ ": {d} slot rows, not a multiple of {d} minis", .{ rows, self.tp });
            inline for (.{ .{ "gate", bank.gate, h, mini }, .{ "up", bank.up, h, mini }, .{ "down", bank.down, mini, h } }) |p| {
                const proj = sushi.Proj{ .trellis = p[1].trellis, .suh = p[1].suh, .svh = p[1].svh };
                sushi.validateClampedProjection(proj, rows, @intCast(p[2]), @intCast(p[3])) catch |e|
                    return quant.refuse(diag, error.BankArrays, "quant " ++ name ++ ": the " ++ p[0] ++ " arrays are not sushi's [{d}, {d}/16, {d}/16, 16K] / [{d}, {d}] / [{d}, {d}] ({s})", .{ rows, p[2], p[3], rows, p[2], rows, p[3], @errorName(e) });
            }
        }

        pub fn gateUp(self: *const Self, g: *G, x: G.T, slot_ids: G.T, gate: A, up: A) !G.T {
            _ = .{ self, g, x, slot_ids, gate, up };
            return error.FusedWaves;
        }

        pub fn down(self: *const Self, g: *G, h: G.T, slot_ids: G.T, d: A) !G.T {
            _ = .{ self, g, h, slot_ids, d };
            return error.FusedWaves;
        }

        /// One decode wave: `x [m, hidden]` (each routed pair's row), `slots` its experts' slot rows -> each pair's
        /// expert output `[m, hidden]` (its minis summed, unweighted), sushi's decode chain.
        pub fn fused(self: *Self, g: *G, x: G.T, slots: []const u32, bank: Bank) !G.T {
            return self.run(g, x, slots, bank, true);
        }

        /// x [tokens or rows, hidden], the call's routed rows -> [rows, hidden] in routed-row order: sushi's prefill
        /// GEMM past its decode rows.
        pub fn prefill(self: *Self, g: *G, layer: u32, x: G.T, rows: quant.PrefillRows, bank: Bank) !G.T {
            _ = layer;
            const n = rows.slot.len;
            try self.src.resize(self.a, n);
            for (self.src.items, 0..) |*s, i| s.* = if (rows.act_row) |ar| ar[i] else @intCast(i);
            const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(self.src.items), &.{@intCast(n)}, .uint32), 0);
            return self.run(g, xs, rows.slot, bank, false);
        }

        fn run(self: *Self, g: *G, x: G.T, slots: []const u32, bank: Bank, decode_chain: bool) !G.T {
            const m = slots.len;
            const tp: usize = self.tp;
            try self.minis.resize(self.a, m * tp);
            for (slots, 0..) |s, i| for (0..tp) |r| {
                self.minis.items[i * tp + r] = s * self.tp + @as(u32, @intCast(r));
            };
            if (self.ones.items.len < m * tp) {
                try self.ones.resize(self.a, m * tp);
                @memset(self.ones.items, 1);
            }
            const mc: c_int = @intCast(m);
            const tc: c_int = @intCast(tp);
            const h: c_int = @intCast(self.spec.hidden);
            const inds = try g.hostArray(std.mem.sliceAsBytes(self.minis.items), &.{ 1, mc, tc }, .uint32);
            const sc = try g.hostArray(std.mem.sliceAsBytes(self.ones.items[0 .. m * tp]), &.{ 1, mc, tc }, .float32);
            const sb: sushi.Bank = .{
                .gate = .{ .trellis = bank.gate.trellis, .suh = bank.gate.suh, .svh = bank.gate.svh },
                .up = .{ .trellis = bank.up.trellis, .suh = bank.up.suh, .svh = bank.up.svh },
                .down = .{ .trellis = bank.down.trellis, .suh = bank.down.suh, .svh = bank.down.svh },
            };
            const y = try g.adopt(try sushi.moe(g.s, try g.reshape(x, &.{ 1, mc, h }), sb, inds, sc, .mcg, decode_chain));
            return g.reshape(y, &.{ mc, h });
        }

        /// Nothing is left in flight (every call's graph is the caller's).
        pub fn finishPrefill(self: *Self, g: *G) !void {
            _ = .{ self, g };
        }

        pub fn deinit(self: *Self, g: *G) void {
            _ = g;
            inline for (.{ &self.minis, &self.src, &self.ones }) |l| l.deinit(self.a);
            self.a.destroy(self);
        }
    };
}

/// The bank's ranks from its description; the arch's SwiGLU unclamped, bf16 inputs, hidden and the minis multiples of
/// 128 (the kernels' Hadamard blocks).
pub fn accept(comptime G: type, a: Allocator, g: *G, ctx: quant.Context, spec: quant.Spec, diag: *Diag) !*Accepted(G) {
    _ = g;
    const peek = ctx.peek orelse return quant.refuse(diag, error.NoWeightDescription, "quant " ++ name ++ ": accepted without the bank's description", .{});
    const tp = parse(peek, diag) orelse return error.NotClaimed;
    if (spec.act != .swiglu) return quant.refuse(diag, error.ActivationNotFused, "quant " ++ name ++ ": sushi's chain runs the unclamped SwiGLU", .{});
    if (spec.input != .bfloat16) return quant.refuse(diag, error.InputDtype, "quant " ++ name ++ ": the MoE input is not bfloat16", .{});
    if (spec.hidden % 128 != 0 or spec.inter % tp != 0 or (spec.inter / tp) % 128 != 0)
        return quant.refuse(diag, error.DimsNotImplemented, "quant " ++ name ++ ": hidden {d} and minis of {d} / {d} (the kernels take multiples of 128)", .{ spec.hidden, spec.inter, tp });
    if (spec.top_k > sushi.kernels.REDUCE_MAX_TOPK) return quant.refuse(diag, error.TopKTooWide, "quant " ++ name ++ ": top-{d}", .{spec.top_k});
    const acc = try a.create(Accepted(G));
    acc.* = .{ .a = a, .tp = tp, .spec = spec };
    return acc;
}

comptime {
    quant.check(@This());
}

const testing = std.testing;

test "glm exl3 quant: claims the EXL3 bank's description and declines the affine one; accept refuses what sushi's chain does not run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exl3 = try std.json.parseFromSliceLeaky(std.json.Value, a, try std.fmt.allocPrint(a, "{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"tp_ranks\":4}}", .{sushi.format.MCG_MULT}), .{});
    const p: quant.BankPeek = .{ .quantization = exl3, .hidden = 6144, .inter = 2048, .n_experts = 256, .n_layers = 150, .layers = &.{} };
    try testing.expectEqual(@as(?quant.Priority, .native), claims(&p, null));
    const aff = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"mode\":\"affine\",\"bits\":4,\"group_size\":64}", .{});
    var why: Diag = .{};
    try testing.expectEqual(@as(?quant.Priority, null), claims(&.{ .quantization = aff, .hidden = 6144, .inter = 2048, .n_experts = 256, .n_layers = 75, .layers = &.{} }, &why));
    try testing.expect(std.mem.indexOf(u8, why.message(), "affine") != null);
    const mul1 = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"mode\":\"exl3\",\"codebook\":\"mul1\",\"codebook_multiplier\":1,\"tp_ranks\":4}", .{});
    try testing.expectEqual(@as(?quant.Priority, null), claims(&.{ .quantization = mul1, .hidden = 6144, .inter = 2048, .n_experts = 256, .n_layers = 150, .layers = &.{} }, null));
    const G = @import("glm_moe_dsa_graph.zig").G;
    var diag: Diag = .{};
    const spec: quant.Spec = .{ .hidden = 6144, .inter = 2048, .top_k = 8, .n_layers = 150, .act = .swiglu, .input = .bfloat16 };
    const ok = try accept(G, testing.allocator, undefined, .{ .peek = &p }, spec, &diag);
    try testing.expectEqual(@as(u32, 4), ok.tp);
    ok.deinit(undefined);
    var clamped = spec;
    clamped.act = .{ .swiglu_clamped = 10 };
    try testing.expectError(error.ActivationNotFused, accept(G, testing.allocator, undefined, .{ .peek = &p }, clamped, &diag));
    var narrow = spec;
    narrow.inter = 256;
    try testing.expectError(error.DimsNotImplemented, accept(G, testing.allocator, undefined, .{ .peek = &p }, narrow, &diag));
    try testing.expectError(error.NoWeightDescription, accept(G, testing.allocator, undefined, .{}, spec, &diag));
}

test "glm exl3 quant: a wide prompt call with its misses staged at the layer's start (both bank layers) equals sushi's host decode, reads what the unstaged call reads and leaves its residents" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const graph = @import("glm_moe_dsa_graph.zig");
    const G = graph.G;
    const experts_mod = @import("glm_moe_dsa_experts.zig");
    const Ex = experts_mod.Experts(G, bank_mod, Accepted(G));
    var c = try bank_mod.tinyConfigInter(a, 512);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try bank_mod.writeSynth(a, testing.io, tmp.dir, &c, .{ .signs = true });
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try bank_mod.Bank.open(a, testing.io, try bank_mod.tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const h: usize = c.hidden_size;
    const k = 8;
    const n = 16;
    var worst: f32 = 0;
    var scale: f32 = 0;
    var read: [2]u64 = undefined;
    // Each arm's residents per bank layer after its call (expert ids, slot order).
    var residents: [2][8][2]u16 = undefined;
    var promoted: u64 = 0;
    for ([_]bool{ false, true }, 0..) |staged, arm| {
        var g = try G.init(a, s);
        defer g.deinit();
        // Two expert slots per bank layer: most of each layer's experts are misses; five windows (three staged).
        const prompt_rows: [8]u32 = @splat(2);
        const st = try bank_mod.Stream.Stream.init(a, &b, .{ .rows = &prompt_rows, .max_route_ids = 48, .transient_rows = 5 * 48, .wide_depth = 5, .transient_release = true, .records_per_part = 2, .slot_memory = .{ .mlx = s }, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 4096 } });
        defer st.deinit();
        var diag: Diag = .{};
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const p = try b.peek(arena.allocator());
        const math = try accept(G, a, &g, .{ .peek = &p }, .{ .hidden = @intCast(h), .inter = c.moe_intermediate_size, .top_k = k, .n_layers = 4, .act = .swiglu, .input = .bfloat16 }, &diag);
        defer math.deinit(&g);
        var ex = try Ex.init(a, &g, st, math, @intCast(h), .{ .wide_depth = 5 });
        defer ex.deinit(&g);
        // The same calls in both arms: row r routes expert r and seven others, so every expert is routed.
        var rng = std.Random.DefaultPrng.init(23);
        for (0..4) |layer| {
            const m = g.mark();
            defer g.resetTo(m);
            var xs: [n * 128]f32 = undefined;
            for (xs[0 .. n * h]) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
            var ids: [n * k]u16 = undefined;
            var ids32: [n * k]i32 = undefined;
            var sc: [n * k]f32 = undefined;
            for (0..n) |row| {
                var j: usize = 0;
                while (j < k) {
                    const e: u16 = if (j == 0) @intCast(row) else rng.random().uintLessThan(u16, 16);
                    if (std.mem.indexOfScalar(u16, ids[row * k ..][0..j], e) != null) continue;
                    ids[row * k + j] = e;
                    ids32[row * k + j] = e;
                    sc[row * k + j] = 0.05 + rng.random().float(f32) * 0.2;
                    j += 1;
                }
            }
            const nc: c_int = n;
            const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. n * h]), &.{ nc, @intCast(h) }, .float32), .bfloat16);
            var xin: [n * 128]f32 = undefined;
            _ = try g.hostF32(try g.astype(xb, .float32), xin[0 .. n * h]);
            if (staged) {
                try ex.stageMisses(@intCast(layer));
                // Each bank layer's experts (none resident yet) staged, one route per bank layer.
                try testing.expectEqual(@as(usize, 2), ex.staged.n_routes);
            }
            const y = try ex.call(&g, @intCast(layer), xb, try g.hostArray(std.mem.sliceAsBytes(ids32[0 .. n * k]), &.{ nc, k }, .int32), try g.hostArray(std.mem.sliceAsBytes(sc[0 .. n * k]), &.{ nc, k }, .float32), null, &.{});
            var got: [n * 128]f32 = undefined;
            _ = try g.hostF32(try g.astype(y, .float32), got[0 .. n * h]);
            try ex.flush();
            try testing.expectEqual(@as(?u32, null), ex.staged.layer);
            // Every seed expert resident (a staged one copied from its transient row), each resident row its record.
            for (0..2) |side| {
                const sl = 2 * layer + side;
                const pol = &st.layers[sl].policy;
                try testing.expectEqual(@as(usize, 0), pol.seed.count());
                for (0..2) |slot| {
                    const e = pol.slot_to_expert[slot];
                    residents[arm][sl][slot] = e;
                    if (slot == 1) std.mem.sort(u16, &residents[arm][sl], {}, std.sort.asc(u16));
                    const geom = &b.layers[sl];
                    const off = b.recordOffset(@intCast(sl), e);
                    for (geom.segments, 0..) |sg, ci| {
                        const row = st.slotRow(@intCast(sl), @intCast(slot), @enumFromInt(ci));
                        for (0..geom.minis) |mi| try testing.expectEqualSlices(u8, img[off + mi * geom.record_bytes + sg.offset ..][0..sg.length], row[mi * sg.length ..][0..sg.length]);
                    }
                }
            }
            var want: [n * 128]f32 = undefined;
            hostMoe(img, &b, @intCast(layer), xin[0 .. n * h], ids[0 .. n * k], sc[0 .. n * k], k, want[0 .. n * h]);
            for (want[0 .. n * h], got[0 .. n * h]) |wv, gv| {
                scale = @max(scale, @abs(wv));
                worst = @max(worst, @abs(wv - gv));
            }
        }
        read[arm] = st.stats().expert_bytes_read;
        if (staged) promoted = st.stats().promoted;
    }
    std.debug.print("glm exl3 quant staged: max |delta| {d:.5} at outputs up to {d:.3}; {d} B read unstaged, {d} B staged; {d} records made resident from transient rows\n", .{ worst, scale, read[0], read[1], promoted });
    try testing.expect(scale > 0.1);
    try testing.expect(worst <= 0.03 * scale);
    // Every expert of every layer read once in both arms; the staged call leaves the residents the unstaged one does
    // (its two hottest per bank layer), those its staged route had not put in a prompt row copied from transient rows.
    try testing.expectEqual(read[0], read[1]);
    try testing.expectEqual(residents[0], residents[1]);
    try testing.expect(promoted > 0 and promoted <= 4 * 2 * 2);
}

/// The host reference of one routed layer's call: each row's experts' minis decoded by sushi's own host path
/// (`format.project`) from the records, SwiGLU between, the minis summed at the expert's score (tests; the MTP lane's
/// bank too, its one routed layer 0).
pub fn hostMoe(img: []const u8, b: *const bank_mod.Bank, layer: u32, xs: []const f32, ids: []const u16, sc: []const f32, k: usize, out: []f32) void {
    const h: usize = @intCast(b.hidden);
    const mini: usize = @intCast(b.mini_inter);
    var t_in: [512]f32 = undefined;
    var inner: [512]f32 = undefined;
    var gate: [512]f32 = undefined;
    var up: [512]f32 = undefined;
    var down: [512]f32 = undefined;
    var act: [512]f32 = undefined;
    @memset(out, 0);
    for (ids, 0..) |e, p| {
        const row = p / k;
        const sl = b.streamLayer(layer, e);
        const l = &b.layers[sl];
        const rate = sushi.format.kFromPackedDim(@intCast(16 * l.k)).?;
        for (0..l.minis) |r| {
            const rec = img[b.recordOffset(sl, e) + r * l.record_bytes ..];
            const seg = struct {
                fn f(ll: *const bank_mod.Layer, bytes: []const u8, ci: usize) []const u16 {
                    const sg = ll.segments[ci];
                    return @alignCast(std.mem.bytesAsSlice(u16, bytes[sg.offset..][0..sg.length]));
                }
            }.f;
            const x = xs[row * h ..][0..h];
            sushi.format.project(x, seg(l, rec, 0), seg(l, rec, 2), seg(l, rec, 1), h, mini, rate, .mcg, t_in[0..h], inner[0..mini], gate[0..mini]);
            sushi.format.project(x, seg(l, rec, 3), seg(l, rec, 5), seg(l, rec, 4), h, mini, rate, .mcg, t_in[0..h], inner[0..mini], up[0..mini]);
            for (act[0..mini], gate[0..mini], up[0..mini]) |*o, gv, uv| o.* = gv / (1 + @exp(-gv)) * uv;
            sushi.format.project(act[0..mini], seg(l, rec, 6), seg(l, rec, 8), seg(l, rec, 7), mini, h, rate, .mcg, t_in[0..mini], inner[0..h], down[0..h]);
            for (out[row * h ..][0..h], down[0..h]) |*o, d| o.* += sc[p] * d;
        }
    }
}

/// The streamed EXL3 bank's MoE through the experts driver against sushi's host decode of the records (`hostMoe`): both
/// bank layers, every lane (the prompt's wide lane, the decode lane), host waits and event gates, on the GPU stream `s`
/// (tests). `Q` is the quant (its `Accepted` and `accept`; `ctx` its accept context, the bank's peek added), `label`
/// the lines' name, `widths` the calls' rows (0: the phase change), `bound(scale)` the largest |delta| allowed at
/// outputs up to `scale`. One body for every EXL3 quant over this bank.
pub fn expectStreamedMoe(comptime Q: type, comptime label: []const u8, s: mlx.mlx_stream, ctx: quant.Context, widths: []const u32, comptime bound: fn (f32) f32) !void {
    const a = testing.allocator;
    const graph = @import("glm_moe_dsa_graph.zig");
    const G = graph.G;
    const experts_mod = @import("glm_moe_dsa_experts.zig");
    const expert_event = sdk_ext.expert.event;
    const Ex = experts_mod.Experts(G, bank_mod, Q.Accepted(G));
    var c = try bank_mod.tinyConfigInter(a, 512);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try bank_mod.writeSynth(a, testing.io, tmp.dir, &c, .{ .signs = true });
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try bank_mod.Bank.open(a, testing.io, try bank_mod.tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    const h: usize = c.hidden_size;
    const k = 8;
    var rng = std.Random.DefaultPrng.init(11);
    var worst: f32 = 0;
    var scale: f32 = 0;
    for ([_]bool{ false, true }) |gated| {
        var g = try G.init(a, s);
        defer g.deinit();
        var ev: ?expert_event.Event = null;
        if (gated) ev = try expert_event.createMetal();
        // Two expert slots per bank layer (8 held): misses, evictions and transient rows in every call.
        const prompt_rows: [8]u32 = @splat(2);
        const decode_rows: [8]u32 = @splat(3);
        const st = try bank_mod.Stream.Stream.init(a, &b, .{ .rows = &prompt_rows, .max_route_ids = 48, .transient_rows = 96, .wide_depth = 2, .transient_release = true, .records_per_part = 2, .slot_memory = .{ .mlx = s }, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 4096 }, .lookahead = .{ .k = 8, .budget = 2 }, .event = if (ev) |e| .{ .backend = .{ .metal = e.object }, .watchdog_ms = 2000 } else null });
        defer st.deinit();
        var diag: Diag = .{};
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const p = try b.peek(arena.allocator());
        var qctx = ctx;
        qctx.peek = &p;
        const math = try Q.accept(G, a, &g, qctx, .{ .hidden = @intCast(h), .inter = c.moe_intermediate_size, .top_k = k, .n_layers = 4, .act = .swiglu, .input = .bfloat16 }, &diag);
        defer math.deinit(&g);
        var ex = try Ex.init(a, &g, st, math, @intCast(h), .{ .gated = gated, .event = ev, .wide_depth = 2 });
        defer ex.deinit(&g);
        // Every decode-lane call's GPU combine against the op chain's join and combine, bit for bit.
        ex.combine_check.on = true;
        for (ex.banks) |bk| if (bk[0]) |arr| try math.checkBank(&g, arr, &diag);
        for (widths) |n| {
            if (n == 0) {
                _ = try ex.releaseTransient();
                try ex.grow(&decode_rows);
                continue;
            }
            for (0..4) |layer| {
                const m = g.mark();
                defer g.resetTo(m);
                var xs: [9 * 128]f32 = undefined;
                for (xs[0 .. n * h]) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
                var ids: [9 * k]u16 = undefined;
                var ids32: [9 * k]i32 = undefined;
                var sc: [9 * k]f32 = undefined;
                for (0..n) |row| {
                    var j: usize = 0;
                    while (j < k) {
                        const e = rng.random().uintLessThan(u16, 16);
                        if (std.mem.indexOfScalar(u16, ids[row * k ..][0..j], e) != null) continue;
                        ids[row * k + j] = e;
                        ids32[row * k + j] = e;
                        sc[row * k + j] = 0.05 + rng.random().float(f32) * 0.2;
                        j += 1;
                    }
                }
                const nc: c_int = @intCast(n);
                const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. n * h]), &.{ nc, @intCast(h) }, .float32), .bfloat16);
                var xin: [9 * 128]f32 = undefined;
                _ = try g.hostF32(try g.astype(xb, .float32), xin[0 .. n * h]);
                var next: [9 * 16]f32 = undefined;
                for (next[0 .. n * 16]) |*v| v.* = rng.random().float(f32);
                const nx = try g.hostArray(std.mem.sliceAsBytes(next[0 .. n * 16]), &.{ nc, 16 }, .float32);
                const y = try ex.call(&g, @intCast(layer), xb, try g.hostArray(std.mem.sliceAsBytes(ids32[0 .. n * k]), &.{ nc, k }, .int32), try g.hostArray(std.mem.sliceAsBytes(sc[0 .. n * k]), &.{ nc, k }, .float32), nx, &.{});
                var got: [9 * 128]f32 = undefined;
                _ = try g.hostF32(try g.astype(y, .float32), got[0 .. n * h]);
                try ex.flush();
                var want: [9 * 128]f32 = undefined;
                hostMoe(img, &b, @intCast(layer), xin[0 .. n * h], ids[0 .. n * k], sc[0 .. n * k], k, want[0 .. n * h]);
                for (want[0 .. n * h], got[0 .. n * h]) |wv, gv| {
                    scale = @max(scale, @abs(wv));
                    worst = @max(worst, @abs(wv - gv));
                }
            }
        }
        const stats = st.stats();
        std.debug.print(label ++ " [{s}]: {d} routes, {d} hits, {d} misses, {d} B read, lookahead {d} issued / {d} used; GPU combine {d} decode calls (bits as the op chain's: {d} differ), {d} joined\n", .{ if (gated) "event gates" else "host waits", stats.route_calls, stats.expert_cache_hits, stats.expert_cache_misses, stats.expert_bytes_read, stats.spec_issued, stats.claimed, ex.combine_check.checked, ex.combine_check.differed, ex.host.joined });
        try testing.expect(stats.expert_cache_misses > 0 and stats.transient_loads > 0);
        try testing.expect(ex.combine_check.checked > 0);
        try testing.expectEqual(@as(u32, 0), ex.combine_check.differed);
    }
    std.debug.print(label ++ ": streamed MoE against the host decode, max |delta| {d:.5} at outputs up to {d:.3}\n", .{ worst, scale });
    try testing.expect(scale > 0.1);
    try testing.expect(worst <= bound(scale));
}

test "glm exl3 quant: the streamed EXL3 bank's MoE on the GPU equals sushi's host decode of the records: both bank layers, every lane, host waits and event gates" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    // The prompt phase's widths (9 rows: the wide lane; 3 and 1: the decode lane), then the decode phase's.
    try expectStreamedMoe(@This(), "glm exl3 quant", s, .{}, &.{ 9, 3, 1, 0, 1, 3, 6, 1 }, struct {
        fn f(scale: f32) f32 {
            return 0.03 * scale;
        }
    }.f);
}

/// One case of the test below: `n_pairs` pairs over `n_experts` experts at rate K (`k`), through sushi's chain.
fn minisBytesMatch(a: Allocator, s: mlx.mlx_stream, k: u32, n_pairs: usize, n_experts: usize, seed: u64) !void {
    const kern = sushi.kernels;
    const tp = 4;
    const h = 6144;
    const mini = 512;
    const rows = n_experts * tp;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    // the bank: random trellis words, the scales at the pack's magnitudes (|suh| ~0.01, |svh| ~1)
    const tile_words = (h / 16) * (mini / 16) * 16 * k;
    var tr: [3][]u16 = undefined;
    for (&tr) |*t| {
        t.* = try ar.alloc(u16, rows * tile_words);
        for (t.*) |*v| v.* = r.int(u16);
    }
    const mags = [4]f32{ 0.0126, 1.03, 0.0083, 1.009 };
    const lens = [4]usize{ h, mini, mini, h };
    var sc: [4][]f16 = undefined;
    for (&sc, mags, lens) |*b, mag, len| {
        b.* = try ar.alloc(f16, rows * len);
        for (b.*) |*v| v.* = @floatCast(if (r.boolean()) mag else -mag);
    }
    const xs = try ar.alloc(f32, n_pairs * h);
    for (xs) |*v| v.* = r.floatNorm(f32) * 0.5;
    // each pair its expert's minis: slot = expert row x 4 + rank
    const slots_h = try ar.alloc(u32, n_pairs * tp);
    for (0..n_pairs) |p| {
        const e = r.uintLessThan(u32, @intCast(n_experts));
        for (0..tp) |q| slots_h[p * tp + q] = e * tp + @as(u32, @intCast(q));
    }
    const ones = try ar.alloc(f32, n_pairs * tp);
    @memset(ones, 1);
    const ci = struct {
        fn f(v: usize) c_int {
            return @intCast(v);
        }
    }.f;
    const arr: [10]mlx.mlx_array = .{
        mlx.mlx_array_new_data(tr[0].ptr, &[_]c_int{ ci(rows), h / 16, mini / 16, ci(16 * k) }, 4, .uint16),
        mlx.mlx_array_new_data(tr[1].ptr, &[_]c_int{ ci(rows), h / 16, mini / 16, ci(16 * k) }, 4, .uint16),
        mlx.mlx_array_new_data(tr[2].ptr, &[_]c_int{ ci(rows), mini / 16, h / 16, ci(16 * k) }, 4, .uint16),
        mlx.mlx_array_new_data(sc[0].ptr, &[_]c_int{ ci(rows), h }, 2, .float16),
        mlx.mlx_array_new_data(sc[1].ptr, &[_]c_int{ ci(rows), mini }, 2, .float16),
        mlx.mlx_array_new_data(sc[2].ptr, &[_]c_int{ ci(rows), mini }, 2, .float16),
        mlx.mlx_array_new_data(sc[3].ptr, &[_]c_int{ ci(rows), h }, 2, .float16),
        mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{ci(n_pairs * tp)}, 1, .uint32),
        mlx.mlx_array_new_data(ones.ptr, &[_]c_int{ci(n_pairs * tp)}, 1, .float32),
        mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ ci(n_pairs), h }, 2, .float32),
    };
    defer for (arr) |v| {
        _ = mlx.mlx_array_free(v);
    };
    const tg, const tu, const td, const suh, const svh, const suhd, const svhd, const slots, const scores = .{ arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8] };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, arr[9], .bfloat16, s));
    const ns: c_int = ci(n_pairs * tp);
    const bytes = struct {
        fn of(al: Allocator, st: mlx.mlx_stream, v: mlx.mlx_array) ![]const u8 {
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_astype(&c, v, .float32, st));
            try mlx.check(mlx.mlx_array_eval(c));
            const p = mlx.mlx_array_data_float32(c) orelse return error.Unreadable;
            return al.dupe(u8, std.mem.sliceAsBytes(p[0..mlx.mlx_array_size(c)]));
        }
    }.of;
    errdefer std.debug.print("minis case K{d}: {d} pairs over {d} experts\n", .{ k, n_pairs, n_experts });
    // the pair planes
    const want = try kern.pairGemv(s, x, suh, suh, tg, tu, slots, h, mini, ns, tp, 0);
    defer _ = mlx.mlx_array_free(want[0]);
    defer _ = mlx.mlx_array_free(want[1]);
    const got = try kern.pairGemvMinis(s, x, suh, suh, tg, tu, slots, h, mini, ns, tp);
    defer _ = mlx.mlx_array_free(got[0]);
    defer _ = mlx.mlx_array_free(got[1]);
    for ([2]mlx.mlx_array{ want[0], want[1] }, [2]mlx.mlx_array{ got[0], got[1] }) |w, g| try testing.expectEqualSlices(u8, try bytes(ar, s, w), try bytes(ar, s, g));
    // the fused-mid down over the same planes
    const dw = try kern.downGemvFusedMid(s, want[0], want[1], td, svh, svh, suhd, slots, mini, h, ns);
    defer _ = mlx.mlx_array_free(dw);
    const dg = try kern.downGemvFusedMidMinis(s, want[0], want[1], td, svh, svh, suhd, slots, mini, h, ns, tp);
    defer _ = mlx.mlx_array_free(dg);
    try testing.expectEqualSlices(u8, try bytes(ar, s, dw), try bytes(ar, s, dg));
    // the chain as the binding calls it, through the single-slot kernels and then the minis kernels
    var chain: [2][]const u8 = undefined;
    for ([2]bool{ true, false }, &chain) |off, *out| {
        kern.setGroupedOffForTest(off);
        const y = try kern.moeSwigluFused(s, x, tg, suh, svh, tu, suh, svh, td, suhd, svhd, slots, scores, .bfloat16);
        defer _ = mlx.mlx_array_free(y);
        out.* = try bytes(ar, s, y);
        try testing.expectEqual(!off and n_pairs >= 2, kern.groupedEngagedForTest());
    }
    try testing.expectEqualSlices(u8, chain[0], chain[1]);
}

// GLM-5.3's routed pairs through sushi's decode chain as `fused` calls it: each pair its expert's four 512-wide minis
// over hidden 6144. Sushi's minis kernels decode each mini once for the pairs that route it; against its single-slot
// kernels the pair planes, the fused-mid down over the same planes and the chain's bf16 output are the same bits, at K3
// and K4: 48 ids over 12 experts (most repeat, past two members), the hit wave's 29 over 22, two pairs of one expert,
// 33 pairs of 33 experts (none repeat) and a lone pair.
test "glm exl3 quant: sushi's chain decodes each GLM-5.3 mini once for the pairs that route it, to the single-slot kernels' bits (4 minis of 512, hidden 6144, 48 ids, K3 and K4)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    sushi.kernels.setDecodeParams(.mcg);
    defer sushi.kernels.setGroupedOffForTest(false);
    for ([_]u32{ 3, 4 }) |k| {
        try minisBytesMatch(testing.allocator, s, k, 48, 12, 900 + k);
        try minisBytesMatch(testing.allocator, s, k, 29, 22, 930 + k);
        try minisBytesMatch(testing.allocator, s, k, 2, 1, 960 + k);
        try minisBytesMatch(testing.allocator, s, k, 33, 33, 990 + k);
        try minisBytesMatch(testing.allocator, s, k, 1, 1, 1020 + k);
    }
}
