//! GLM-5.3's EXL3 routed-expert quant `glm53_exl3` (C2, `sdk_ext.quant`): the pinned kernels of
//! `glm53_exl3_kernels` over the EXL3 bank's slot arrays (`glm_moe_dsa_exl3_bank`), specialized to GLM-5.3 (hidden 6144,
//! a mini's intermediate 512, four minis per expert, K 3 and 4, the mcg codebook, SwiGLU unclamped, top-8).
//!   - Decode (`decode_layer`): per wave of a routed layer's pairs over one bank layer, the three bank kinds in one
//!     launch per stage (`decodeWave`: the inputs per mini, the gate and up GEMVs per distinct slot, the down with the
//!     SwiGLU middle, the minis' sum and the pair's score fused in); per call, every row's pairs summed in routed order
//!     (`decodeCombine`). x is read by row, the slot arrays by slot; nothing syncs with the host.
//!   - Prompt (`prefill`): the call's rows sorted by slot on the host, waves of at most 16 experts and `wave_rows` rows,
//!     each through the inputs, the 128-row gate and up GEMMs (the trellis decoded per stage), the middle and the down
//!     GEMM (the minis summed in its epilogue), then the waves joined in the call's row order.
//! Its kernels are self-checked at accept on the load context's kernel set (compile, one launch each at the accepted
//! dims, both Ks).

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("sdk_ext.zig");
const quant = sdk_ext.quant;
const bank_mod = @import("glm_moe_dsa_exl3_bank.zig");
const xk = @import("glm53_exl3_kernels.zig");
const routes = @import("glm53_exl3_routes.zig");
const selfcheck = xk.selfcheck;
const ks = sdk_ext.kernels.KernelSet(xk);

const Allocator = std.mem.Allocator;
const Diag = quant.Diag;

pub const name = "glm53-exl3";
/// The registry the load context builds this quant's kernel set over.
pub const kernel_registry = xk;
pub const kernel_pin: sdk_ext.kernels.Pin = .{ .manifest_sha256 = xk.manifest_sha256 };
pub const kernels = std.enums.values(xk.Kernel);
pub const Arrays = bank_mod.Proj;

/// Rows per prompt wave (its inputs per mini are rows x 4 x 2 x hidden f16) and the waves left in flight.
pub const wave_rows: u32 = 4096;
pub const waves_in_flight: u32 = 2;

/// The bank's `quantization`: exl3, the mcg codebook at the texts' multiplier, four minis per expert.
fn parse(peek: *const quant.BankPeek, why: ?*Diag) ?void {
    const q = peek.quantization;
    const mode = quant.str(q, "mode") orelse "";
    const cb = quant.str(q, "codebook") orelse "";
    const mult = quant.int(q, "codebook_multiplier") orelse -1;
    const tp = quant.int(q, "tp_ranks") orelse 0;
    if (!std.mem.eql(u8, mode, "exl3") or !std.mem.eql(u8, cb, xk.bank_codebook) or mult != xk.bank_multiplier or tp != xk.tp_ranks) {
        _ = quant.decline(why, "quant " ++ name ++ ": quantization {s} / {s} / multiplier {d} / {d} ranks (the texts decode exl3 mcg at {d}, {d} ranks)", .{ mode, cb, mult, tp, xk.bank_multiplier, xk.tp_ranks });
        return null;
    }
}

pub fn claims(peek: *const quant.BankPeek, why: ?*Diag) ?quant.Priority {
    parse(peek, why) orelse return null;
    return .native;
}

pub fn Accepted(comptime G: type) type {
    comptime {
        if (G.T != mlx.mlx_array) @compileError("quant " ++ name ++ ": the kernels launch on MLX arrays");
    }
    return struct {
        const Self = @This();
        const A = Arrays(G.T);
        const Bank = quant.BankArrays(A);
        /// A decode wave's widest pair count (6 verify rows x top-8).
        pub const max_decode_rows: u32 = routes.max_pairs;
        /// The decode lane hands each wave over once its segments landed, and the call's combine to `decodeCombine`.
        pub const fused_waves = true;
        pub const decode_layer = true;
        /// A decode slot's member pairs per GEMV (`decodeWave`'s gt).
        pub const slot_members = routes.max_members;
        pub const top_k = routes.top_k;
        pub const combine_sources = routes.combine_sources;
        a: Allocator,
        set: *const ks.Set,
        hidden: u32,
        mini: u32,
        report: selfcheck.Report = .{},
        /// per K (3, 4): the decode and prompt chains
        dec: [2]routes.Decode,
        pf: [2]routes.Prompt,
        cmb: routes.Combine,
        // the prompt's host scratch
        counts: std.ArrayList(u32) = .empty,
        order: std.ArrayList(u32) = .empty,
        inv: std.ArrayList(u32) = .empty,
        rt: std.ArrayList(u32) = .empty,
        waves: std.ArrayList(mlx.mlx_array) = .empty,
        flight: std.ArrayList(mlx.mlx_array) = .empty,

        fn kOf(trellis: mlx.mlx_array) ?u32 {
            const sh = mlx.getShape(trellis);
            if (sh.len != 4 or @mod(sh[3], 16) != 0) return null;
            const k: u32 = @intCast(@divExact(sh[3], 16));
            return if (std.mem.indexOfScalar(u32, &xk.bank_ks, k) != null) k else null;
        }

        fn ki(k: u32) usize {
            return k - 3;
        }

        /// Each projection: trellis uint16 [minis, in/16, out/16, 16 K] at K 3 or 4, suh f16 [minis, in], svh f16
        /// [minis, out]; the minis a multiple of four.
        pub fn checkBank(self: *const Self, g: *G, bank: Bank, diag: *Diag) !void {
            _ = g;
            const rows = mlx.getShape(bank.gate.trellis)[0];
            const k = kOf(bank.gate.trellis) orelse return quant.refuse(diag, error.BankArrays, "quant " ++ name ++ ": the gate trellis {any} is not at K 3 or 4", .{mlx.getShape(bank.gate.trellis)});
            if (rows <= 0 or @mod(rows, 4) != 0) return quant.refuse(diag, error.BankArrays, "quant " ++ name ++ ": {d} slot rows, not a multiple of 4 minis", .{rows});
            inline for (.{ .{ "gate", bank.gate, self.hidden, self.mini }, .{ "up", bank.up, self.hidden, self.mini }, .{ "down", bank.down, self.mini, self.hidden } }) |p| {
                const in_d: c_int = @intCast(p[2]);
                const out_d: c_int = @intCast(p[3]);
                const ok = std.mem.eql(c_int, mlx.getShape(p[1].trellis), &.{ rows, @divExact(in_d, 16), @divExact(out_d, 16), @intCast(16 * k) }) and
                    std.mem.eql(c_int, mlx.getShape(p[1].suh), &.{ rows, in_d }) and std.mem.eql(c_int, mlx.getShape(p[1].svh), &.{ rows, out_d }) and
                    mlx.mlx_array_dtype(p[1].trellis) == .uint16 and mlx.mlx_array_dtype(p[1].suh) == .float16 and mlx.mlx_array_dtype(p[1].svh) == .float16;
                if (!ok) return quant.refuse(diag, error.BankArrays, "quant " ++ name ++ ": the " ++ p[0] ++ " arrays are not [{d}, {d}/16, {d}/16, {d}] u16 / [{d}, {d}] f16 / [{d}, {d}] f16", .{ rows, in_d, out_d, 16 * k, rows, in_d, rows, out_d });
            }
        }

        pub fn gateUp(self: *const Self, g: *G, x: G.T, slot_ids: G.T, gate: A, up: A) !G.T {
            _ = .{ self, g, x, slot_ids, gate, up };
            return error.LayeredDecode;
        }

        pub fn down(self: *const Self, g: *G, h: G.T, slot_ids: G.T, d: A) !G.T {
            _ = .{ self, g, h, slot_ids, d };
            return error.LayeredDecode;
        }

        pub fn fused(self: *Self, g: *G, x: G.T, slots: []const u32, bank: Bank) !G.T {
            _ = .{ self, g, x, slots, bank };
            return error.LayeredDecode;
        }

        /// One decode wave over a bank layer: x [rows, hidden] bf16 (the call's rows), `pairs` per pair (x row, packed
        /// slot bank << 24 | row, routed position), `banks` the bank kinds' arrays (a kind no pair names may repeat
        /// another's), sc the router's weights [rows, top-k] f32 -> f32 [pairs, hidden], each pair's output times its
        /// score. `scratch`: the pair and slot tables' host words (`decodeTables`).
        pub fn decodeWave(self: *Self, g: *G, x: G.T, pairs: []const [3]u32, banks: *const [3]Bank, sc: G.T) !G.T {
            const k = kOf(banks[0].gate.trellis) orelse return error.BankArrays;
            const tabs = try selfcheck.decodeTables(self.a, pairs);
            defer self.a.free(tabs.pt);
            defer self.a.free(tabs.gt);
            const pt = try g.hostArray(std.mem.sliceAsBytes(tabs.pt), &.{@intCast(tabs.pt.len)}, .uint32);
            const gt = try g.hostArray(std.mem.sliceAsBytes(tabs.gt), &.{@intCast(tabs.gt.len)}, .int32);
            var rb: [3]routes.Bank = undefined;
            for (banks, &rb) |b, *o| o.* = .{ .gate = proj(b.gate), .up = proj(b.up), .down = proj(b.down) };
            return g.adopt(try self.dec[ki(k)].wave(x, pt, gt, pairs.len, tabs.n_slots, &rb, try g.reshape(sc, &.{-1})));
        }

        fn proj(p: A) routes.Proj {
            return .{ .trellis = p.trellis, .suh = p.suh, .svh = p.svh };
        }

        /// The call's combine: `sources` the waves' outputs (`decodeWave`), `loc` per routed position (row x top-k + j)
        /// its source << 24 | its pair there -> [rows, hidden] bf16, each row's pairs summed in routed order.
        pub fn decodeCombine(self: *Self, g: *G, sources: []const G.T, loc: []const u32, n_rows: u32) !G.T {
            const n_ids = n_rows * routes.top_k;
            std.debug.assert(loc.len == n_ids and sources.len >= 1);
            var ct: [routes.max_rows * routes.top_k]u32 = undefined;
            var acc: ?G.T = null;
            var s0: usize = 0;
            while (true) {
                const s1 = @min(s0 + routes.combine_sources, sources.len);
                for (ct[0..n_ids], loc) |*c, l| {
                    const src = l >> 24;
                    c.* = if (src >= s0 and src < s1) (@as(u32, @intCast(src - s0)) << 24) | (l & 0xffffff) else 0xffffffff;
                }
                const ctd = try g.hostArray(std.mem.sliceAsBytes(ct[0..@max(8, n_ids)]), &.{@intCast(@max(8, n_ids))}, .uint32);
                const last = s1 == sources.len;
                acc = try g.adopt(try self.cmb.launch(sources[s0..s1], ctd, acc, n_rows, last));
                if (last) return acc.?;
                s0 = s1;
            }
        }

        /// x [tokens, hidden] bf16, the call's routed rows (one bank kind) -> [rows, hidden] bf16 in routed-row order.
        pub fn prefill(self: *Self, g: *G, layer: u32, x: G.T, rows: quant.PrefillRows, bank: Bank) !G.T {
            _ = layer;
            const a = self.a;
            const n = rows.slot.len;
            const k = kOf(bank.gate.trellis) orelse return error.BankArrays;
            const n_slots: usize = @intCast(@divExact(mlx.getShape(bank.gate.trellis)[0], 4));
            // the rows by slot (a stable counting sort), each expert's rows in their call order
            try self.counts.resize(a, n_slots + 1);
            @memset(self.counts.items, 0);
            for (rows.slot) |s| self.counts.items[s + 1] += 1;
            for (1..n_slots + 1) |i| self.counts.items[i] += self.counts.items[i - 1];
            try self.order.resize(a, n);
            try self.inv.resize(a, n);
            for (rows.slot, 0..) |s, i| {
                const at = self.counts.items[s];
                self.order.items[at] = @intCast(i);
                self.inv.items[i] = at;
                self.counts.items[s] += 1;
            }
            // a wave's table is at least 8 words (an MLX input below 8 elements binds as `constant`): 8 past the end
            try self.rt.resize(a, 2 * n + 8);
            @memset(self.rt.items, 0);
            for (self.order.items, 0..) |o, i| {
                self.rt.items[2 * i] = if (rows.act_row) |ar| ar[o] else o;
                self.rt.items[2 * i + 1] = rows.slot[o];
            }
            for (self.waves.items) |w| _ = mlx.mlx_array_free(w);
            self.waves.clearRetainingCapacity();
            const bk: routes.Bank = .{ .gate = proj(bank.gate), .up = proj(bank.up), .down = proj(bank.down) };
            const pf = &self.pf[ki(k)];
            const tiles_gu = routes.tilesGateUp(self.mini);
            const tiles_dn = routes.tilesDown(self.hidden);
            // waves: experts in slot order, at most 16 and `wave_rows` rows (a larger expert alone)
            var i: usize = 0;
            while (i < n) {
                var slots: [routes.wave_experts]u32 = undefined;
                var cnts: [routes.wave_experts]u32 = undefined;
                var ne: usize = 0;
                var w_rows: usize = 0;
                var j = i;
                while (j < n and ne < routes.wave_experts) {
                    const s = rows.slot[self.order.items[j]];
                    var e = j;
                    while (e < n and rows.slot[self.order.items[e]] == s) e += 1;
                    if (ne > 0 and w_rows + (e - j) > wave_rows) break;
                    slots[ne] = s;
                    cnts[ne] = @intCast(e - j);
                    ne += 1;
                    w_rows += e - j;
                    j = e;
                }
                const tgu = routes.waveTable(slots[0..ne], cnts[0..ne], tiles_gu);
                const tdn = routes.waveTable(slots[0..ne], cnts[0..ne], tiles_dn);
                const m = g.mark();
                const rtd = try g.hostArray(std.mem.sliceAsBytes(self.rt.items[2 * i ..][0..@max(8, 2 * w_rows)]), &.{@intCast(@max(8, 2 * w_rows))}, .uint32);
                const tgud = try g.hostArray(std.mem.sliceAsBytes(&tgu.t), &.{80}, .int32);
                const tdnd = try g.hostArray(std.mem.sliceAsBytes(&tdn.t), &.{80}, .int32);
                try self.waves.ensureUnusedCapacity(a, 1);
                try self.flight.ensureUnusedCapacity(a, 1);
                const y = try pf.wave(x, rtd, w_rows, tgud, tgu.tgs, tdnd, tdn.tgs, bk);
                self.waves.appendAssumeCapacity(y);
                // a wave waits for all but `waves_in_flight - 1` older ones, so their inputs per mini go back
                while (self.flight.items.len >= waves_in_flight) try self.drainOne(g);
                try g.asyncEval(&.{y});
                self.flight.appendAssumeCapacity(g.keep(y));
                g.resetTo(m);
                i = j;
            }
            const joined = try g.adopt(if (self.waves.items.len == 1) g.keep(self.waves.items[0]) else try concat(self.waves.items, g.s));
            for (self.waves.items) |w| _ = mlx.mlx_array_free(w);
            self.waves.clearRetainingCapacity();
            const ord = try g.hostArray(std.mem.sliceAsBytes(self.inv.items), &.{@intCast(n)}, .uint32);
            return g.take(joined, ord, 0);
        }

        fn concat(xs: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
            const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
            defer _ = mlx.mlx_vector_array_free(v);
            var out = mlx.mlx_array_new();
            mlx.check(mlx.mlx_concatenate_axis(&out, v, 0, s)) catch |e| {
                _ = mlx.mlx_array_free(out);
                return e;
            };
            return out;
        }

        fn drainOne(self: *Self, g: *G) !void {
            const y = self.flight.orderedRemove(0);
            defer g.release(y);
            try g.evalAll(&.{y});
        }

        /// The prompt's boundary: every wave still in flight evaluated.
        pub fn finishPrefill(self: *Self, g: *G) !void {
            while (self.flight.items.len > 0) try self.drainOne(g);
        }

        pub fn deinit(self: *Self, g: *G) void {
            for (self.flight.items) |y| g.release(y);
            for (self.waves.items) |w| _ = mlx.mlx_array_free(w);
            for (&self.dec) |*d| d.deinit();
            self.cmb.deinit();
            inline for (.{ &self.counts, &self.order, &self.inv, &self.rt, &self.waves, &self.flight }) |l| l.deinit(self.a);
            self.report.deinit(self.a);
            self.a.destroy(self);
        }
    };
}

/// The load context's kernel set over this registry, the bank's description, the arch's Spec (SwiGLU unclamped, bf16
/// inputs, top-8, dims the manifest registers), then the self-check (compile) of every kernel at those dims.
pub fn accept(comptime G: type, a: Allocator, g: *G, ctx: quant.Context, spec: quant.Spec, diag: *Diag) !*Accepted(G) {
    _ = g;
    const ref = ctx.kernels orelse return quant.refuse(diag, error.NoKernelSet, "quant " ++ name ++ ": accepted without the load context's kernel set", .{});
    const set = ks.Set.of(ref) orelse return quant.refuse(diag, error.NoKernelSet, "quant " ++ name ++ ": accepted with a kernel set over another registry (manifest {s})", .{ref.manifest_sha256});
    const peek = ctx.peek orelse return quant.refuse(diag, error.NoWeightDescription, "quant " ++ name ++ ": accepted without the bank's description", .{});
    parse(peek, diag) orelse return error.NotClaimed;
    if (spec.act != .swiglu) return quant.refuse(diag, error.ActivationNotFused, "quant " ++ name ++ ": the texts fuse the unclamped SwiGLU", .{});
    if (spec.input != .bfloat16) return quant.refuse(diag, error.InputDtype, "quant " ++ name ++ ": the MoE input is not bfloat16", .{});
    if (spec.top_k != routes.top_k) return quant.refuse(diag, error.TopKTooWide, "quant " ++ name ++ ": top-{d} (the decode tables are top-{d})", .{ spec.top_k, routes.top_k });
    const mini = spec.inter / xk.tp_ranks;
    const inst: xk.Instance = .{ .k = 3, .hidden = spec.hidden, .mini = mini };
    var registered = spec.inter % xk.tp_ranks == 0;
    if (registered) for (std.enums.values(xk.Kernel)) |k| for (xk.bank_ks) |kk| {
        var at = inst;
        at.k = kk;
        _ = set.reg.instance(k, at, null) catch {
            registered = false;
        };
    };
    if (!registered) return quant.refuse(diag, error.DimsNotImplemented, "quant " ++ name ++ ": hidden {d}, inter {d} (the manifest registers the texts at other dims)", .{ spec.hidden, spec.inter });
    const acc = try a.create(Accepted(G));
    errdefer a.destroy(acc);
    acc.* = .{ .a = a, .set = set, .hidden = spec.hidden, .mini = mini, .dec = undefined, .pf = undefined, .cmb = .init(&set.bound, inst) };
    errdefer acc.report.deinit(a);
    switch (set.device) {
        .stream => {
            try selfcheck.runSubsetAt(a, &set.reg, &set.bound, kernels, .compile, &.{.{ .hidden = spec.hidden, .mini = mini }}, &acc.report);
            try selfcheck.judge(&acc.report, diag);
        },
        .stub => try set.selfCheck(a, kernels, .compile, &acc.report, diag),
    }
    for (xk.bank_ks, 0..) |kk, i| {
        const ik: xk.Instance = .{ .k = kk, .hidden = spec.hidden, .mini = mini };
        acc.dec[i] = .init(&set.bound, ik);
        acc.pf[i] = .init(&set.bound, ik);
    }
    return acc;
}

comptime {
    quant.check(@This());
}

const testing = std.testing;
const sushi = @import("mlx_host").sushi_exl3;

fn peekOf(a: Allocator, q: []const u8) !quant.BankPeek {
    return .{ .quantization = try std.json.parseFromSliceLeaky(std.json.Value, a, q, .{}), .hidden = 6144, .inter = 2048, .n_experts = 256, .n_layers = 150, .layers = &.{} };
}

test "glm53 exl3 quant: claims GLM-5.3's EXL3 bank only; accept refuses what the texts do not implement, by name, before any launch" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ar = arena.allocator();
    const exl3 = try peekOf(ar, try std.fmt.allocPrint(ar, "{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"tp_ranks\":4}}", .{xk.bank_multiplier}));
    try testing.expectEqual(@as(?quant.Priority, .native), claims(&exl3, null));
    var why: Diag = .{};
    for ([_][]const u8{
        "{\"mode\":\"affine\",\"bits\":4,\"group_size\":64}",
        "{\"mode\":\"exl3\",\"codebook\":\"mul1\",\"codebook_multiplier\":2212286765,\"tp_ranks\":4}",
        "{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":3417055213,\"tp_ranks\":2}",
    }) |q| {
        try testing.expectEqual(@as(?quant.Priority, null), claims(&try peekOf(ar, q), &why));
        try testing.expect(std.mem.indexOf(u8, why.message(), "quant " ++ name) != null);
    }
    const G = @import("glm_moe_dsa_graph.zig").G;
    var diag: Diag = .{};
    const spec: quant.Spec = .{ .hidden = 6144, .inter = 2048, .top_k = 8, .n_layers = 150, .act = .swiglu, .input = .bfloat16 };
    const set = try ks.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    const ok = try accept(G, testing.allocator, undefined, .{ .kernels = set.ref(), .peek = &exl3 }, spec, &diag);
    try testing.expectEqual(@as(u32, 512), ok.mini);
    try testing.expectEqual(@as(usize, xk.n_kernels), ok.report.results.items.len);
    ok.deinit(undefined);
    const Case = struct { spec: quant.Spec, ctx: quant.Context, err: anyerror, names: []const u8 };
    var clamped = spec;
    clamped.act = .{ .swiglu_clamped = 10 };
    var f32in = spec;
    f32in.input = .float32;
    var top6 = spec;
    top6.top_k = 6;
    var dsv41 = spec;
    dsv41.hidden = 5120;
    dsv41.inter = 2304;
    for ([_]Case{
        .{ .spec = spec, .ctx = .{ .peek = &exl3 }, .err = error.NoKernelSet, .names = "kernel set" },
        .{ .spec = spec, .ctx = .{ .kernels = .{ .set = set, .manifest_sha256 = "another" }, .peek = &exl3 }, .err = error.NoKernelSet, .names = "another registry" },
        .{ .spec = spec, .ctx = .{ .kernels = set.ref() }, .err = error.NoWeightDescription, .names = "description" },
        .{ .spec = clamped, .ctx = .{ .kernels = set.ref(), .peek = &exl3 }, .err = error.ActivationNotFused, .names = "unclamped" },
        .{ .spec = f32in, .ctx = .{ .kernels = set.ref(), .peek = &exl3 }, .err = error.InputDtype, .names = "bfloat16" },
        .{ .spec = top6, .ctx = .{ .kernels = set.ref(), .peek = &exl3 }, .err = error.TopKTooWide, .names = "top-6" },
        .{ .spec = dsv41, .ctx = .{ .kernels = set.ref(), .peek = &exl3 }, .err = error.DimsNotImplemented, .names = "hidden 5120, inter 2304" },
    }) |c| {
        try testing.expectError(c.err, accept(G, testing.allocator, undefined, c.ctx, c.spec, &diag));
        testing.expect(std.mem.indexOf(u8, diag.message(), c.names) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ c.names, diag.message() });
            return e;
        };
    }
    // a scripted compile failure refuses the accept by name
    const bad = try ks.Set.init(testing.allocator, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .glm53_pf_down, .check = .compile } } } }, &diag);
    defer bad.deinit();
    try testing.expectError(error.SelfCheckFailed, accept(G, testing.allocator, undefined, .{ .kernels = bad.ref(), .peek = &exl3 }, spec, &diag));
}

const old_quant = @import("glm_moe_dsa_exl3_quant.zig");

test "glm53 exl3 quant: the streamed EXL3 bank's MoE through the experts driver equals sushi's host decode of the records: both bank layers, every lane, host waits and event gates" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const G = @import("glm_moe_dsa_graph.zig").G;
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
    var kd: Diag = .{};
    const set = try ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd);
    defer set.deinit();
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
        const math = try accept(G, a, &g, .{ .kernels = set.ref(), .peek = &p }, .{ .hidden = @intCast(h), .inter = c.moe_intermediate_size, .top_k = k, .n_layers = 4, .act = .swiglu, .input = .bfloat16 }, &diag);
        defer math.deinit(&g);
        var ex = try Ex.init(a, &g, st, math, @intCast(h), .{ .gated = gated, .event = ev, .wide_depth = 2 });
        defer ex.deinit(&g);
        for (ex.banks) |bk| if (bk[0]) |arr| try math.checkBank(&g, arr, &diag);
        // The prompt phase's widths (9 rows: the wide lane; 3 and 1: the decode lane), then the decode phase's 1 to 6.
        for ([_]u32{ 9, 3, 1, 0, 1, 3, 6, 2, 4, 5, 1 }) |n| {
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
                old_quant.hostMoe(img, &b, @intCast(layer), xin[0 .. n * h], ids[0 .. n * k], sc[0 .. n * k], k, want[0 .. n * h]);
                for (want[0 .. n * h], got[0 .. n * h]) |wv, gv| {
                    scale = @max(scale, @abs(wv));
                    worst = @max(worst, @abs(wv - gv));
                }
            }
        }
        const stats = st.stats();
        std.debug.print("glm53 exl3 quant [{s}]: {d} routes, {d} hits, {d} misses, lookahead {d} issued / {d} used\n", .{ if (gated) "event gates" else "host waits", stats.route_calls, stats.expert_cache_hits, stats.expert_cache_misses, stats.spec_issued, stats.claimed });
        try testing.expect(stats.expert_cache_misses > 0 and stats.transient_loads > 0);
    }
    std.debug.print("glm53 exl3 quant: streamed MoE against the host decode, max |delta| {d:.5} at outputs up to {d:.3} ({d:.2} bf16 ulp of the scale)\n", .{ worst, scale, worst / selfcheck.bf16Ulp(scale) });
    try testing.expect(scale > 0.1);
    try testing.expect(worst <= 2 * selfcheck.bf16Ulp(scale));
}

test "glm53 exl3 quant: on the same slot banks the GLM-5.3 kernels and sushi's EXL3 chain agree within bf16 rounding, decode widths 1 to 6 and prompt widths" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const G = @import("glm_moe_dsa_graph.zig").G;
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
    var g = try G.init(a, s);
    defer g.deinit();
    var kd: Diag = .{};
    const set = try ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd);
    defer set.deinit();
    // Routed layer 0's 16 experts resident: its K3 bank layer (even experts) and K4 one (odd), 8 slots each.
    const rows_all: [8]u32 = @splat(8);
    const st = try bank_mod.Stream.Stream.init(a, &b, .{ .rows = &rows_all, .max_route_ids = 16, .transient_rows = 16, .records_per_part = 2, .slot_memory = .{ .mlx = s }, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 4096 } });
    defer st.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const p = try b.peek(arena.allocator());
    const h: usize = c.hidden_size;
    const spec: quant.Spec = .{ .hidden = @intCast(h), .inter = c.moe_intermediate_size, .top_k = 8, .n_layers = 4, .act = .swiglu, .input = .bfloat16 };
    var diag: Diag = .{};
    const ours = try accept(G, a, &g, .{ .kernels = set.ref(), .peek = &p }, spec, &diag);
    defer ours.deinit(&g);
    const theirs = try old_quant.accept(G, a, &g, .{ .peek = &p }, spec, &diag);
    defer theirs.deinit(&g);
    var slot_of: [16]u32 = undefined;
    var banks: [2]quant.BankArrays(Arrays(G.T)) = undefined;
    var routes_: [2]*bank_mod.Stream.Route = undefined;
    for (0..2) |side| {
        var ids: [8]u16 = undefined;
        for (&ids, 0..) |*e, i| e.* = @intCast(2 * i + side);
        const r = try st.route(@intCast(side), &ids, &.{});
        for (0..r.n_parts) |q| {
            try st.waitGu(r, @intCast(q));
            try st.waitDown(r, @intCast(q));
        }
        var refs: [sdk_ext.expert.policy.max_route_ids]sdk_ext.expert.SlotRef = undefined;
        _ = st.refsOf(r, &refs);
        for (ids, refs[0..8]) |e, ref| {
            try testing.expectEqual(sdk_ext.expert.BankKind.base, ref.bank);
            slot_of[e] = ref.row;
        }
        banks[side] = st.bankArrays(@intCast(side), .base).?;
        routes_[side] = r;
    }
    defer for (routes_) |r| st.release(r);
    var rng = std.Random.DefaultPrng.init(0x5a3e);
    var worst: f32 = 0;
    var scale: f32 = 0;
    // decode widths: each row 8 distinct experts of the 16, the two bank layers' waves, the call's combine
    for (1..7) |n| {
        const m = g.mark();
        defer g.resetTo(m);
        var xs: [6 * 128]f32 = undefined;
        for (xs[0 .. n * h]) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
        var ids: [48]u16 = undefined;
        var sc: [48]f32 = undefined;
        for (0..n) |row| {
            var j: usize = 0;
            while (j < 8) {
                const e = rng.random().uintLessThan(u16, 16);
                if (std.mem.indexOfScalar(u16, ids[row * 8 ..][0..j], e) != null) continue;
                ids[row * 8 + j] = e;
                sc[row * 8 + j] = 0.05 + rng.random().float(f32) * 0.2;
                j += 1;
            }
        }
        const nc: c_int = @intCast(n);
        const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. n * h]), &.{ nc, @intCast(h) }, .float32), .bfloat16);
        const scd = try g.hostArray(std.mem.sliceAsBytes(sc[0 .. n * 8]), &.{ nc, 8 }, .float32);
        var loc: [48]u32 = undefined;
        var outs: [2]G.T = undefined;
        var ref_rows: [48]f32 = undefined;
        _ = &ref_rows;
        var want_y: [6 * 128]f32 = @splat(0);
        for (0..2) |side| {
            var pairs: [48][3]u32 = undefined;
            var np: u32 = 0;
            var tok: [48]i32 = undefined;
            var slots: [48]u32 = undefined;
            var pos_of: [48]usize = undefined;
            for (ids[0 .. n * 8], 0..) |e, pos| {
                if (e % 2 != side) continue;
                pairs[np] = .{ @intCast(pos / 8), slot_of[e], @intCast(pos) };
                loc[pos] = (@as(u32, @intCast(side)) << 24) | np;
                tok[np] = @intCast(pos / 8);
                slots[np] = slot_of[e];
                pos_of[np] = pos;
                np += 1;
            }
            const bk3 = [3]quant.BankArrays(Arrays(G.T)){ banks[side], banks[side], banks[side] };
            outs[side] = try ours.decodeWave(&g, xb, pairs[0..np], &bk3, scd);
            // sushi's chain on the same rows of the same slots: each pair's minis summed, then the score
            const xt = try g.take(xb, try g.hostArray(std.mem.sliceAsBytes(tok[0..np]), &.{@intCast(np)}, .int32), 0);
            const yt = try theirs.fused(&g, xt, slots[0..np], banks[side]);
            var yh: [48 * 128]f32 = undefined;
            _ = try g.hostF32(try g.astype(yt, .float32), yh[0 .. np * h]);
            for (0..np) |i| for (0..h) |q| {
                want_y[(pos_of[i] / 8) * h + q] += sc[pos_of[i]] * yh[i * h + q];
            };
        }
        const y = try ours.decodeCombine(&g, &outs, loc[0 .. n * 8], @intCast(n));
        var got: [6 * 128]f32 = undefined;
        _ = try g.hostF32(try g.astype(y, .float32), got[0 .. n * h]);
        for (want_y[0 .. n * h], got[0 .. n * h]) |wv, gv| {
            scale = @max(scale, @abs(wv));
            worst = @max(worst, @abs(wv - gv));
        }
    }
    const dec_ulp = worst / selfcheck.bf16Ulp(scale);
    std.debug.print("glm53 exl3 quant vs sushi on the same slots: decode widths 1..6 max |d| {d:.5} at {d:.3} ({d:.2} bf16 ulp)\n", .{ worst, scale, dec_ulp });
    try testing.expect(scale > 0.1 and worst <= 2 * selfcheck.bf16Ulp(scale));
    // prompt widths: one call per bank layer over 40 and 300 routed rows of its experts (act rows of 64 tokens)
    var pw: f32 = 0;
    var ps: f32 = 0;
    for ([_]usize{ 40, 300 }) |nr| for (0..2) |side| {
        const m = g.mark();
        defer g.resetTo(m);
        var xs: [64 * 128]f32 = undefined;
        for (&xs) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
        const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&xs), &.{ 64, @intCast(h) }, .float32), .bfloat16);
        var slot: [300]u32 = undefined;
        var act: [300]u32 = undefined;
        for (0..nr) |i| {
            slot[i] = slot_of[2 * rng.random().uintLessThan(u32, 8) + side];
            act[i] = rng.random().uintLessThan(u32, 64);
        }
        const rows: quant.PrefillRows = .{ .slot = slot[0..nr], .act_row = act[0..nr] };
        const yo = try ours.prefill(&g, @intCast(side), xb, rows, banks[side]);
        const yt = try theirs.prefill(&g, @intCast(side), xb, rows, banks[side]);
        try ours.finishPrefill(&g);
        var go: [300 * 128]f32 = undefined;
        var gt_: [300 * 128]f32 = undefined;
        _ = try g.hostF32(try g.astype(yo, .float32), go[0 .. nr * h]);
        _ = try g.hostF32(try g.astype(yt, .float32), gt_[0 .. nr * h]);
        for (go[0 .. nr * h], gt_[0 .. nr * h]) |o, t| {
            ps = @max(ps, @abs(t));
            pw = @max(pw, @abs(o - t));
        }
    };
    std.debug.print("glm53 exl3 quant vs sushi on the same slots: prompt widths 40 and 300 max |d| {d:.5} at {d:.3} ({d:.2} bf16 ulp)\n", .{ pw, ps, pw / selfcheck.bf16Ulp(ps) });
    try testing.expect(ps > 0.1 and pw <= 2 * selfcheck.bf16Ulp(ps));
}
