//! GLM-5.3's EXL3 routed-expert quant (C2, `sdk_ext.quant`): sushi's EXL3 MoE (`mlx_host.sushi_exl3`: the mcg
//! codebook, SwiGLU unclamped, GLM has no swiglu_limit) over the EXL3 bank's slot arrays (`glm_moe_dsa_exl3_bank`).
//! A routed (token, expert) pair is one sushi row: the expert's `tp` minis (slot x tp + rank) at score 1, the
//! router's score being the arch's combine. Sushi's chain fuses gate, up and down, so the decode lane hands a wave
//! over once all its segments landed (`fused`, in place of `gateUp` / `down`, which refuse); a wave of any width runs
//! the decode chain. The prompt's slices run sushi's prefill GEMM past its decode rows (`prefill`), on the tensor units
//! with its routing table built from the slots the host holds (`prefillTabled`).

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
        /// `prefillTabled`'s scratch: the rows by mini, each row's x row (as the prepare reads it), the inverse, the
        /// counting sort's counts, the windows.
        sorted: std.ArrayList(u32) = .empty,
        xrow: std.ArrayList(i32) = .empty,
        inverse: std.ArrayList(u32) = .empty,
        counts: std.ArrayList(u32) = .empty,
        starts: std.ArrayList(u32) = .empty,
        lives: std.ArrayList(u32) = .empty,
        /// Sushi's NAX sorted GEMM (`PrefillGridSupport`), built at the first prompt slice on the GPU; null where the
        /// device has no tensor units (sushi's `moe` serves the slice).
        gemm: ?mlx.mlx_fast_metal_kernel = null,
        gemm_probed: bool = false,

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
            if (n > sushi.kernels.DECODE_ROWS_MAX and self.tabledKernel(g) != null) return self.prefillTabled(g, x, rows, bank);
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

        /// The window rows of sushi's sorted GEMM (`GEMM_WINDOW_ROWS`).
        const gemm_window: u32 = 32;

        /// Sushi's NAX sorted GEMM where this stream runs it (a GPU with tensor units), else null.
        fn tabledKernel(self: *Self, g: *G) ?mlx.mlx_fast_metal_kernel {
            if (self.gemm_probed) return self.gemm;
            self.gemm_probed = true;
            if (comptime !@hasField(G, "s")) return null;
            if (!mlx.streamIsGpu(g.s) or !sushi.kernels.PrefillGridSupport.available()) return null;
            const sup = sushi.kernels.PrefillGridSupport;
            self.gemm = sup.makeKernel(sup.source, sushi.format.Decode.mcg.window) catch null;
            return self.gemm;
        }

        /// One prompt slice through sushi's prefill GEMM (`moePrefill`'s chain: the pairs' rows through suh, the gate and
        /// up GEMMs over each mini's run in windows of 32 rows, the middle, the down GEMM, each pair's minis reduced) with
        /// its routing table built here from the slots: sushi's `moePrefill` reads its sorted slots back from the GPU to
        /// build it (a sync per slice). The rows ordered by mini (a stable counting sort, as MLX's sort orders them),
        /// each run of one mini cut in windows from its start, each row's x row read where it is (no gather), the
        /// finish reading each pair's minis through the inverse order. The same kernels over the same rows: sushi's
        /// output.
        fn prefillTabled(self: *Self, g: *G, x: G.T, rows: quant.PrefillRows, bank: Bank) !G.T {
            const a = self.a;
            const n = rows.slot.len;
            const tp: usize = self.tp;
            const ns = n * tp;
            const n_minis: usize = @intCast(mlx.getShape(bank.gate.trellis)[0]);
            try self.minis.resize(a, ns);
            try self.src.resize(a, ns);
            for (rows.slot, 0..) |s, i| {
                const xr: u32 = if (rows.act_row) |ar| ar[i] else @intCast(i);
                for (0..tp) |r| {
                    self.minis.items[i * tp + r] = s * self.tp + @as(u32, @intCast(r));
                    // The prepare reads row `order / tp` of x.
                    self.src.items[i * tp + r] = xr * self.tp + @as(u32, @intCast(r));
                }
            }
            try self.counts.resize(a, n_minis + 1);
            @memset(self.counts.items, 0);
            for (self.minis.items) |m| self.counts.items[m + 1] += 1;
            for (1..n_minis + 1) |i| self.counts.items[i] += self.counts.items[i - 1];
            try self.sorted.resize(a, ns);
            try self.xrow.resize(a, ns);
            try self.inverse.resize(a, ns);
            for (self.minis.items, self.src.items, 0..) |m, xr, j| {
                const at = self.counts.items[m];
                self.counts.items[m] += 1;
                self.sorted.items[at] = m;
                self.xrow.items[at] = @intCast(xr);
                self.inverse.items[j] = at;
            }
            self.starts.clearRetainingCapacity();
            self.lives.clearRetainingCapacity();
            var p: usize = 0;
            while (p < ns) {
                var end = p + 1;
                while (end < ns and self.sorted.items[end] == self.sorted.items[p]) end += 1;
                var off = p;
                while (off < end) : (off += gemm_window) {
                    try self.starts.append(a, @intCast(off));
                    try self.lives.append(a, @intCast(@min(gemm_window, end - off)));
                }
                p = end;
            }
            if (self.ones.items.len < ns) {
                try self.ones.resize(a, ns);
                @memset(self.ones.items, 1);
            }
            sushi.kernels.setDecodeParams(sushi.format.Decode.mcg);
            const sup = sushi.kernels.PrefillGridSupport;
            const nsc: c_int = @intCast(ns);
            const nwin: c_int = @intCast(self.starts.items.len);
            const h: c_int = @intCast(self.spec.hidden);
            const mini: c_int = @intCast(self.spec.inter / self.tp);
            const tc: c_int = @intCast(self.tp);
            const sorted = try g.hostArray(std.mem.sliceAsBytes(self.sorted.items), &.{nsc}, .uint32);
            const order = try g.hostArray(std.mem.sliceAsBytes(self.xrow.items), &.{nsc}, .int32);
            const starts = try g.hostArray(std.mem.sliceAsBytes(self.starts.items), &.{nwin}, .uint32);
            const lives = try g.hostArray(std.mem.sliceAsBytes(self.lives.items), &.{nwin}, .uint32);
            const inverse = try g.hostArray(std.mem.sliceAsBytes(self.inverse.items), &.{nsc}, .uint32);
            const minis = try g.hostArray(std.mem.sliceAsBytes(self.minis.items), &.{nsc}, .uint32);
            const ones = try g.hostArray(std.mem.sliceAsBytes(self.ones.items[0..ns]), &.{nsc}, .float32);
            const prep = try sup.prepare(g.s, x, bank.gate.suh, bank.up.suh, sorted, order, h, nsc, tc);
            const pg = try g.adopt(prep[0]);
            const pu = try g.adopt(prep[1]);
            const gi = try self.gemmTabled(g, pg, bank.gate.trellis, sorted, starts, lives, nwin);
            const ui = try self.gemmTabled(g, pu, bank.up.trellis, sorted, starts, lives, nwin);
            const mid = try g.adopt(try sup.middle(g.s, gi, ui, bank.gate.svh, bank.up.svh, bank.down.suh, sorted, mini, nsc, 0));
            const di = try self.gemmTabled(g, mid, bank.down.trellis, sorted, starts, lives, nwin);
            return g.adopt(try sup.finish(g.s, di, inverse, bank.down.svh, minis, ones, h, @intCast(n), tc, .bfloat16));
        }

        /// `x [rows, in] @` each window's mini of `trellis` (`[minis, in/16, out/16, n]`): sushi's NAX sorted GEMM as its
        /// prefill dispatches it (128 threads per 128 outputs of one window, f16 out).
        fn gemmTabled(self: *Self, g: *G, x: G.T, trellis: G.T, eids: G.T, starts: G.T, lives: G.T, nwin: c_int) !G.T {
            const xs = mlx.getShape(x);
            const ts = mlx.getShape(trellis);
            const out_dim: c_int = ts[2] * 16;
            const rate = sushi.format.kFromPackedDim(@intCast(ts[3])) orelse return error.BadExl3Shape;
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ xs[0], out_dim }, 2, .float16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", xs[1]));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIN", @intCast(gemm_window)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", @intCast(rate.n)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, nwin, 1));
            const inputs = [_]mlx.mlx_array{ x, trellis, eids, starts, lives };
            const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
            defer _ = mlx.mlx_vector_array_free(iv);
            var ov = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ov);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, self.gemm.?, iv, cfg, g.s));
            var out = mlx.mlx_array_new();
            mlx.check(mlx.mlx_vector_array_get(&out, ov, 0)) catch |e| {
                _ = mlx.mlx_array_free(out);
                return e;
            };
            return g.adopt(out);
        }

        /// Nothing is left in flight (every call's graph is the caller's).
        pub fn finishPrefill(self: *Self, g: *G) !void {
            _ = .{ self, g };
        }

        pub fn deinit(self: *Self, g: *G) void {
            _ = g;
            if (self.gemm) |kern| _ = mlx.mlx_fast_metal_kernel_free(kern);
            inline for (.{ &self.minis, &self.src, &self.ones, &self.sorted, &self.xrow, &self.inverse, &self.counts, &self.starts, &self.lives }) |l| l.deinit(self.a);
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

test "glm exl3 quant: a prompt slice over the host-built routing table equals sushi's moe bit for bit (K3 and K4, runs past one window)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const G = @import("glm_moe_dsa_graph.zig").G;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const q = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"tp_ranks\":4}}", .{sushi.format.MCG_MULT}), .{});
    const hidden: u32 = 256;
    const inter: u32 = 512;
    const p: quant.BankPeek = .{ .quantization = q, .hidden = hidden, .inter = inter, .n_experts = 8, .n_layers = 2, .layers = &.{} };
    var diag: Diag = .{};
    const ok = try accept(G, a, &g, .{ .peek = &p }, .{ .hidden = hidden, .inter = inter, .top_k = 8, .n_layers = 2, .act = .swiglu, .input = .bfloat16 }, &diag);
    defer ok.deinit(&g);
    if (ok.tabledKernel(&g) == null) return error.SkipZigTest;
    const tp = ok.tp;
    const mini = inter / tp;
    // Six slot rows (24 minis); 40 pairs (past sushi's 16 decode rows) and 300 (runs of one mini past 32 rows).
    const rows_n: u32 = 6;
    var prng = std.Random.DefaultPrng.init(5);
    const rand = prng.random();
    const Gen = struct {
        fn proj(gg: *G, ra: std.Random, al: std.mem.Allocator, n_minis: u32, in: u32, out: u32, nhw: u32) !Arrays(G.T) {
            const t = try al.alloc(u16, n_minis * (in / 16) * (out / 16) * nhw);
            defer al.free(t);
            for (t) |*v| v.* = ra.int(u16);
            const suh = try al.alloc(u16, n_minis * in);
            defer al.free(suh);
            for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(if (ra.boolean()) 1.0 + 0.5 * ra.float(f32) else -1.0 - 0.5 * ra.float(f32))));
            const svh = try al.alloc(u16, n_minis * out);
            defer al.free(svh);
            for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(0.01 + 0.02 * ra.float(f32))));
            return .{
                .trellis = try gg.hostArray(std.mem.sliceAsBytes(t), &.{ @intCast(n_minis), @intCast(in / 16), @intCast(out / 16), @intCast(nhw) }, .uint16),
                .suh = try gg.hostArray(std.mem.sliceAsBytes(suh), &.{ @intCast(n_minis), @intCast(in) }, .float16),
                .svh = try gg.hostArray(std.mem.sliceAsBytes(svh), &.{ @intCast(n_minis), @intCast(out) }, .float16),
            };
        }
    };
    var checked: usize = 0;
    for ([_]u32{ 48, 64 }) |nhw| {
        const m0 = g.mark();
        defer g.resetTo(m0);
        const bank: quant.BankArrays(Arrays(G.T)) = .{
            .gate = try Gen.proj(&g, rand, a, rows_n * tp, hidden, mini, nhw),
            .up = try Gen.proj(&g, rand, a, rows_n * tp, hidden, mini, nhw),
            .down = try Gen.proj(&g, rand, a, rows_n * tp, mini, hidden, nhw),
        };
        for ([_]u32{ 40, 300 }) |n| {
            const m = g.mark();
            defer g.resetTo(m);
            const tokens = n + 10;
            const xv = try a.alloc(f32, tokens * hidden);
            defer a.free(xv);
            for (xv) |*v| v.* = rand.floatNorm(f32);
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xv), &.{ @intCast(tokens), @intCast(hidden) }, .float32), .bfloat16);
            const slot = try a.alloc(u32, n);
            defer a.free(slot);
            const act = try a.alloc(u32, n);
            defer a.free(act);
            for (slot, act) |*sl, *ar| {
                sl.* = rand.uintLessThan(u32, rows_n);
                ar.* = rand.uintLessThan(u32, tokens);
            }
            // Sushi's moe over the gathered rows (the slice's path before the host-built table).
            const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(act), &.{@intCast(n)}, .uint32), 0);
            const want = try ok.run(&g, xs, slot, bank, false);
            const got = try ok.prefillTabled(&g, x, .{ .slot = slot, .act_row = act }, bank);
            const wv = try a.alloc(f32, n * hidden);
            defer a.free(wv);
            const gv = try a.alloc(f32, n * hidden);
            defer a.free(gv);
            _ = try g.hostF32(try g.astype(want, .float32), wv);
            _ = try g.hostF32(try g.astype(got, .float32), gv);
            var big: f32 = 0;
            for (wv, gv) |w, v| {
                try testing.expect(std.math.isFinite(w));
                try testing.expectEqual(@as(u32, @bitCast(w)), @as(u32, @bitCast(v)));
                big = @max(big, @abs(w));
            }
            try testing.expect(big > 0);
            checked += wv.len;
        }
    }
    std.debug.print("glm exl3 quant tabled: {d} outputs equal sushi's moe bit for bit\n", .{checked});
}

/// The host reference of one routed layer's call: each row's experts' minis decoded by sushi's own host path
/// (`format.project`) from the records, SwiGLU between, the minis summed at the expert's score.
fn hostMoe(img: []const u8, b: *const bank_mod.Bank, layer: u32, xs: []const f32, ids: []const u16, sc: []const f32, k: usize, out: []f32) void {
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

test "glm exl3 quant: the streamed EXL3 bank's MoE on the GPU equals sushi's host decode of the records: both bank layers, every lane, host waits and event gates" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const graph = @import("glm_moe_dsa_graph.zig");
    const G = graph.G;
    const experts_mod = @import("glm_moe_dsa_experts.zig");
    const expert_event = sdk_ext.expert.event;
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
        const math = try accept(G, a, &g, .{ .peek = &p }, .{ .hidden = @intCast(h), .inter = c.moe_intermediate_size, .top_k = k, .n_layers = 4, .act = .swiglu, .input = .bfloat16 }, &diag);
        defer math.deinit(&g);
        var ex = try Ex.init(a, &g, st, math, @intCast(h), .{ .gated = gated, .event = ev, .wide_depth = 2 });
        defer ex.deinit(&g);
        for (ex.banks) |bk| if (bk[0]) |arr| try math.checkBank(&g, arr, &diag);
        // The prompt phase's widths (9 rows: the wide lane; 3 and 1: the decode lane), then the decode phase's.
        for ([_]u32{ 9, 3, 1, 0, 1, 3, 6, 1 }) |n| {
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
        std.debug.print("glm exl3 quant [{s}]: {d} routes, {d} hits, {d} misses, {d} B read, lookahead {d} issued / {d} used\n", .{ if (gated) "event gates" else "host waits", stats.route_calls, stats.expert_cache_hits, stats.expert_cache_misses, stats.expert_bytes_read, stats.spec_issued, stats.claimed });
        try testing.expect(stats.expert_cache_misses > 0 and stats.transient_loads > 0);
    }
    std.debug.print("glm exl3 quant: streamed MoE against the host decode, max |delta| {d:.5} at outputs up to {d:.3}\n", .{ worst, scale });
    try testing.expect(scale > 0.1);
    try testing.expect(worst <= 0.03 * scale);
}
