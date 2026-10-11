//! The launches of GLM-5.3's EXL3 kernels (`glm53_exl3_kernels`) over MLX arrays, shared by the quant and the
//! self-check. A route holds its prepared launch configs (one per count of pairs, slots or rows the call has, built at
//! the first such call) and launches one stage chain per call; every array it returns is new (the caller frees it), the
//! intermediates are freed here once launched (the pending graph keeps them).
//!   - Decode, per wave of a routed layer's pairs over one bank layer (its three bank kinds through packed slots, bank
//!     << 24 | row): prep (each pair's gate and up inputs per mini, f16, lane order), gu (the GEMVs per distinct slot,
//!     each weight decoded once for the slot's member pairs), down (the SwiGLU middle, the down GEMV, each output block
//!     rotated, scaled by its mini's rout, the minis summed, times the pair's score: f32 per pair); per call, combine
//!     (each row's pairs in routed order: x's dtype).
//!   - Prompt, per wave of a call's rows grouped by expert (one bank kind): prep (each row's inputs per mini, f16), the
//!     gate and up GEMMs (128-row M tiles, the trellis decoded per stage), the middle, the down GEMM (the minis summed in
//!     its epilogue): bf16 per row in the wave's order.

const std = @import("std");
const mlx = @import("sdk").mlx;
const xk = @import("glm53_exl3_kernels.zig");

const Kernel = xk.Kernel;

/// One projection's slot arrays: trellis uint16 [minis, in/16, out/16, 16 K], suh f16 [minis, in], svh f16 [minis, out].
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// The widest decode wave (pairs) and call (rows) the routes prepare: 6 verify rows x top-8.
pub const max_pairs = 48;
pub const max_rows = 6;
pub const top_k = 8;
/// A decode slot's members (pairs sharing its weight decode): more spill the GEMV's accumulators.
pub const max_members = 2;
/// The combine's sources per launch (more chain through its accumulator).
pub const combine_sources = 16;
/// The prompt wave table: at most 16 experts.
pub const wave_experts = 16;
/// Threadgroups per 128-row M tile: the gate|up GEMM's (2 projections x 4 minis x MINI / 64 N tiles), the down's (HID /
/// 128 N tiles).
pub fn tilesGateUp(mini: u32) u32 {
    return 8 * mini / 64;
}
pub fn tilesDown(hidden: u32) u32 {
    return hidden / 128;
}

fn free(xs: []const mlx.mlx_array) void {
    for (xs) |x| _ = mlx.mlx_array_free(x);
}

fn cfgOf(e: *const xk.Entry, inst: xk.Instance, grid: [3]u32, outs: []const struct { []const c_int, mlx.mlx_dtype }) xk.LaunchConfig {
    var c: xk.LaunchConfig = .{ .grid = grid, .threadgroup = e.threadgroup, .n_out = outs.len };
    for (e.template, 0..) |name, i| c.ints[i] = if (std.mem.eql(u8, name, "K")) @intCast(inst.k) else if (std.mem.eql(u8, name, "HID")) @intCast(inst.hidden) else if (std.mem.eql(u8, name, "MINI")) @intCast(inst.mini) else if (std.mem.eql(u8, name, "MAXM")) max_members else if (std.mem.eql(u8, name, "TOPK")) top_k else 0;
    for (outs, 0..) |o, i| {
        c.out_ranks[i] = o[0].len;
        @memcpy(c.out_shapes[i][0..o[0].len], o[0]);
        c.out_dtypes[i] = o[1];
    }
    return c;
}

/// A table of prepared configs indexed by a count 1..N, built at first use.
fn Plans(comptime N: usize) type {
    return struct {
        const Self = @This();
        p: [N]?xk.Prepared = @splat(null),

        fn get(self: *Self, b: *const xk.Bound, k: Kernel, n: usize, cfg: *const xk.LaunchConfig) !*const xk.Prepared {
            std.debug.assert(n >= 1 and n <= N);
            if (self.p[n - 1] == null) self.p[n - 1] = try b.prepare(k, cfg);
            return &self.p[n - 1].?;
        }

        fn deinit(self: *Self) void {
            for (&self.p) |*x| if (x.*) |*p| p.deinit();
        }
    };
}

/// The decode chain of one bank layer's geometry (K, dims).
pub const Decode = struct {
    b: *const xk.Bound,
    inst: xk.Instance,
    prep: Plans(max_pairs) = .{},
    down: Plans(max_pairs) = .{},
    /// the gu GEMV's: its grid counts the slots, its outputs the pairs
    gu: [max_pairs]Plans(max_pairs) = @splat(.{}),

    pub fn init(b: *const xk.Bound, inst: xk.Instance) Decode {
        return .{ .b = b, .inst = inst };
    }

    pub fn deinit(self: *Decode) void {
        self.prep.deinit();
        self.down.deinit();
        for (&self.gu) |*p| p.deinit();
    }

    /// One wave: x [rows, HID] bf16, pt u32 [>= 8] (per pair: x row, packed slot, routed position, 0), gt i32 [>= 8]
    /// (per distinct slot: packed slot, members, member pairs), `banks` the bank kinds' arrays (packed bank 0..2), sc f32
    /// [rows x top-k] -> y f32 [pairs, HID] (each pair's output times its score).
    pub fn wave(self: *Decode, x: mlx.mlx_array, pt: mlx.mlx_array, gt: mlx.mlx_array, n_pairs: usize, n_slots: usize, banks: *const [3]Bank, sc: mlx.mlx_array) !mlx.mlx_array {
        std.debug.assert(n_slots >= 1 and n_slots <= n_pairs and n_pairs <= max_pairs);
        const reg = self.b.reg;
        const h: u32 = self.inst.hidden;
        const mi: u32 = self.inst.mini;
        const np: c_int = @intCast(n_pairs);
        var u: [2]mlx.mlx_array = .{ .{}, .{} };
        var z: [2]mlx.mlx_array = .{ .{}, .{} };
        defer free(&u);
        defer free(&z);
        {
            const sh = [_]c_int{np * 4 * @as(c_int, @intCast(h))};
            const cfg = cfgOf(reg.get(.glm53_dec_prep), self.inst, .{ 32 * h / 128, @intCast(n_pairs * 4), 2 }, &.{ .{ &sh, .float16 }, .{ &sh, .float16 } });
            const p = try self.prep.get(self.b, .glm53_dec_prep, n_pairs, &cfg);
            try self.b.applyPrepared(p, &.{ x, pt, banks[0].gate.suh, banks[1].gate.suh, banks[2].gate.suh, banks[0].up.suh, banks[1].up.suh, banks[2].up.suh }, &u);
        }
        {
            const sh = [_]c_int{np * 4 * @as(c_int, @intCast(mi))};
            const cfg = cfgOf(reg.get(.glm53_dec_gu), self.inst, .{ 32 * mi / 16, @intCast(n_slots * 4), 2 }, &.{ .{ &sh, .float32 }, .{ &sh, .float32 } });
            const p = try self.gu[n_pairs - 1].get(self.b, .glm53_dec_gu, n_slots, &cfg);
            try self.b.applyPrepared(p, &.{ u[0], u[1], gt, banks[0].gate.trellis, banks[1].gate.trellis, banks[2].gate.trellis, banks[0].up.trellis, banks[1].up.trellis, banks[2].up.trellis }, &z);
        }
        const sh = [_]c_int{ np, @intCast(h) };
        const cfg = cfgOf(reg.get(.glm53_dec_down), self.inst, .{ 256 * h / 128, @intCast(n_pairs), 1 }, &.{.{ &sh, .float32 }});
        const p = try self.down.get(self.b, .glm53_dec_down, n_pairs, &cfg);
        var y: [1]mlx.mlx_array = undefined;
        try self.b.applyPrepared(p, &.{ z[0], z[1], pt, sc, banks[0].gate.svh, banks[1].gate.svh, banks[2].gate.svh, banks[0].up.svh, banks[1].up.svh, banks[2].up.svh, banks[0].down.suh, banks[1].down.suh, banks[2].down.suh, banks[0].down.trellis, banks[1].down.trellis, banks[2].down.trellis, banks[0].down.svh, banks[1].down.svh, banks[2].down.svh }, &y);
        return y[0];
    }
};

/// The decode call's combine: each row's top-k pairs summed in routed order.
pub const Combine = struct {
    b: *const xk.Bound,
    inst: xk.Instance,
    /// per (out bf16 / f32, with an accumulator or not), per rows
    plans: [2][2]Plans(max_rows) = @splat(@splat(.{})),

    pub fn init(b: *const xk.Bound, inst: xk.Instance) Combine {
        return .{ .b = b, .inst = .{ .hidden = inst.hidden, .mini = inst.mini } };
    }

    pub fn deinit(self: *Combine) void {
        for (&self.plans) |*pp| for (pp) |*p| p.deinit();
    }

    /// `sources` (1..16 f32 [pairs, HID]), ct u32 [rows x top-k] (source << 24 | pair, 0xFFFFFFFF: not this launch's),
    /// acc (f32 [rows, HID] summed so far, or null) -> [rows, HID] bf16 (`last`) or f32.
    pub fn launch(self: *Combine, sources: []const mlx.mlx_array, ct: mlx.mlx_array, acc: ?mlx.mlx_array, n_rows: usize, last: bool) !mlx.mlx_array {
        std.debug.assert(sources.len >= 1 and sources.len <= combine_sources and n_rows >= 1 and n_rows <= max_rows);
        const e = self.b.reg.get(.glm53_dec_combine);
        const h: u32 = self.inst.hidden;
        const dt: mlx.mlx_dtype = if (last) .bfloat16 else .float32;
        const sh = [_]c_int{ @intCast(n_rows), @intCast(h) };
        var cfg = cfgOf(e, self.inst, .{ h / 4, @intCast(n_rows), 1 }, &.{.{ &sh, dt }});
        cfg.threadgroup = .{ @min(h / 4, 256), 1, 1 };
        cfg.dtype = dt;
        for (e.template, 0..) |name, i| if (std.mem.eql(u8, name, "ACC")) {
            cfg.ints[i] = @intFromBool(acc != null);
        };
        const p = try self.plans[@intFromBool(last)][@intFromBool(acc != null)].get(self.b, .glm53_dec_combine, n_rows, &cfg);
        var ins: [combine_sources + 2]mlx.mlx_array = undefined;
        for (ins[0..combine_sources], 0..) |*v, i| v.* = if (i < sources.len) sources[i] else sources[0];
        ins[combine_sources] = ct;
        ins[combine_sources + 1] = acc orelse sources[0];
        var out: [1]mlx.mlx_array = undefined;
        try self.b.applyPrepared(p, &ins, &out);
        return out[0];
    }
};

/// The prompt chain of one bank layer's geometry.
pub const Prompt = struct {
    b: *const xk.Bound,
    inst: xk.Instance,

    pub fn init(b: *const xk.Bound, inst: xk.Instance) Prompt {
        return .{ .b = b, .inst = inst };
    }

    /// One wave of `rows` rows grouped by expert: x [tokens, HID] bf16, rt u32 [>= 8] (per row: x row, slot row), the
    /// gate|up and down wave tables (i32 [80]) and their threadgroups, the bank kind's arrays -> y bf16 [rows, HID].
    pub fn wave(self: *const Prompt, x: mlx.mlx_array, rt: mlx.mlx_array, rows: usize, tgu: mlx.mlx_array, tgs_gu: u32, tdn: mlx.mlx_array, tgs_dn: u32, bank: Bank) !mlx.mlx_array {
        const reg = self.b.reg;
        const h: u32 = self.inst.hidden;
        const mi: u32 = self.inst.mini;
        const rc: c_int = @intCast(rows);
        var a: [2]mlx.mlx_array = .{ .{}, .{} };
        var z: [2]mlx.mlx_array = .{ .{}, .{} };
        var hm: [1]mlx.mlx_array = .{.{}};
        defer free(&a);
        defer free(&z);
        defer free(&hm);
        {
            const sh = [_]c_int{rc * 4 * @as(c_int, @intCast(h))};
            const cfg = cfgOf(reg.get(.glm53_pf_prep), self.inst, .{ 32 * h / 128, @intCast(rows * 4), 2 }, &.{ .{ &sh, .float16 }, .{ &sh, .float16 } });
            try self.b.apply(.glm53_pf_prep, &.{ x, rt, bank.gate.suh, bank.up.suh }, &cfg, &a);
        }
        {
            const sh = [_]c_int{rc * 4 * @as(c_int, @intCast(mi))};
            const cfg = cfgOf(reg.get(.glm53_pf_gu), self.inst, .{ 256 * tgs_gu, 1, 1 }, &.{ .{ &sh, .float32 }, .{ &sh, .float32 } });
            try self.b.apply(.glm53_pf_gu, &.{ a[0], a[1], tgu, bank.gate.trellis, bank.up.trellis }, &cfg, &z);
        }
        {
            const sh = [_]c_int{rc * 4 * @as(c_int, @intCast(mi))};
            const cfg = cfgOf(reg.get(.glm53_pf_mid), self.inst, .{ 32 * mi / 128, @intCast(rows * 4), 1 }, &.{.{ &sh, .float16 }});
            try self.b.apply(.glm53_pf_mid, &.{ z[0], z[1], rt, bank.gate.svh, bank.up.svh, bank.down.suh }, &cfg, &hm);
        }
        const sh = [_]c_int{ rc, @intCast(h) };
        const cfg = cfgOf(reg.get(.glm53_pf_down), self.inst, .{ 512 * tgs_dn, 1, 1 }, &.{.{ &sh, .bfloat16 }});
        var y: [1]mlx.mlx_array = undefined;
        try self.b.apply(.glm53_pf_down, &.{ hm[0], tdn, bank.down.trellis, bank.down.svh }, &cfg, &y);
        return y[0];
    }
};

/// A prompt wave table: per expert j (<= 16) slot, first row, rows, first threadgroup (INT_MAX unused); [64] experts,
/// [65] threadgroups; `tiles` threadgroups per 128-row M tile.
pub const WaveTable = struct { t: [80]i32, tgs: u32 };

pub fn waveTable(slots: []const u32, counts: []const u32, tiles: u32) WaveTable {
    std.debug.assert(slots.len == counts.len and slots.len <= wave_experts);
    var t: [80]i32 = @splat(0);
    var row0: i32 = 0;
    var tg: i32 = 0;
    for (0..wave_experts) |j| {
        if (j >= slots.len) {
            t[48 + j] = std.math.maxInt(i32);
            continue;
        }
        t[j] = @intCast(slots[j]);
        t[16 + j] = row0;
        t[32 + j] = @intCast(counts[j]);
        t[48 + j] = tg;
        row0 += @intCast(counts[j]);
        tg += @intCast((counts[j] + 127) / 128 * tiles);
    }
    t[64] = @intCast(slots.len);
    t[65] = tg;
    return .{ .t = t, .tgs = @intCast(tg) };
}
