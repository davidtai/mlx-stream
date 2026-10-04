//! C2, the routed-expert `quant`: the routed-expert half of PLG's `quant` kind (weight load + the
//! MoE matmul). Given the slot ids and the bank, an MoE layer asks the quant that claimed the
//! weight group for its call's expert matmuls.
//!
//! The arch keeps the routing (gate, top-k, weights), the combine, the shared expert and the
//! choice between decode and prefill width (made on the logical M, a genuine runtime value). C1
//! `expert_source` keeps which slot holds which expert, the bank arrays, the wait gates and the
//! growth.
//!
//! The load path asks every quant `claims(peek)` once per weight group and takes the highest
//! claim. The peek is the group's description: C1's manifest reader for a streamed bank
//! (`expert_bank.peek`), the checkpoint's quantization_config for resident experts. The claimed
//! quant is `accept`ed once per backend (built, and self-checked where it brings kernels), and
//! every entry point of its `Accepted(G)` is prebound there: nothing is looked up or revalidated
//! per call. Clients:
//!   - `exl3_quant`: the streamed EXL3 bank's pinned Metal texts;
//!   - `FromGatherMatmul(Q)`: any gather-matmul quant, e.g. `GatherQmm` (mlx-serve's gather_qmm /
//!     gather_mm path; the V4.1 DSpark head's mxfp4 switch).
//!
//! A quant `Q` declares (`check(Q)` names a missing or mistyped declaration at comptime):
//!   name: []const u8
//!   Arrays(T): type                                   one projection's per-slot arrays
//!   claims(*const BankPeek, ?*Diag) ?Priority         at load, once per weight group
//!   Accepted(G): type                                 the quant accepted on backend G
//!   accept(G, Allocator, *G, Context, Spec, *Diag) !*Accepted(G)   once, before the weights bind
//! and its `Accepted(G)` (`checkAccepted(Q, G)`):
//!   max_decode_rows: u32                              the widest call gateUp / down serve
//!   checkBank(*const, *G, BankArrays(Arrays(G.T)), *Diag) !void    per bank bind and per grow
//!   gateUp(*const, *G, x, slot_ids, gate, up) !G.T   the activation of the gate and up rows
//!   down(*const, *G, h, slot_ids, down) !G.T
//!   prefill(*, *G, layer, x, PrefillRows, BankArrays(..)) !G.T     any width: the call's down rows
//!   finishPrefill(*, *G) !void                        the prefill boundary: drains what is in flight
//!   deinit(*, *G) void

const std = @import("std");
const mlx = @import("sdk").mlx;
const pk = @import("sdk");
const kernel_reg = @import("kernels.zig");
const QuantMode = @import("sdk").QuantMode;

const Allocator = std.mem.Allocator;
const Dtype = mlx.mlx_dtype;

pub const Diag = pk.Diag;

/// How strongly a quant claims a weight group: the load path asks every quant and takes the highest.
pub const Priority = pk.Priority;

/// A weight group's per-expert tensor, its layers, and its description at load (the SDK's `GroupPeek`).
pub const Segment = pk.Segment;
pub const LayerPeek = pk.LayerPeek;
pub const BankPeek = pk.GroupPeek;

/// The routed experts' activation between gate / up and down. It is the arch's: a quant whose
/// texts fuse it declares the one they fuse and refuses the rest at accept.
pub const Activation = union(enum) {
    /// silu(gate) * up
    swiglu,
    /// silu(min(gate, limit)) * clip(up, -limit, limit) (DeepSeek-V4.1's swiglu_limit)
    swiglu_clamped: f64,
};

/// What the arch asks of the quant at accept.
pub const Spec = struct {
    hidden: u32,
    inter: u32,
    top_k: u32,
    n_layers: u32,
    act: Activation,
    /// the MoE input's dtype
    input: Dtype,
};

/// What the load context hands a quant's accept: the kernel set it owns (a quant with pinned
/// texts takes it back as its own registry's set and self-checks its subset there) and the
/// claimed group's description.
pub const Context = struct {
    kernels: ?kernel_reg.SetRef = null,
    peek: ?*const BankPeek = null,
};

/// One prefill call's routed rows: `slot[i]` is routed row i's bank slot; row i reads x row i,
/// or x row `act_row[i]` when given (the chunk's tokens: routed row / top_k).
pub const PrefillRows = struct { slot: []const u32, act_row: ?[]const u32 = null };

/// A layer bank's three projections, each one `Q.Arrays(T)`.
pub fn BankArrays(comptime A: type) type {
    return struct { gate: A, up: A, down: A };
}

pub const Proj = enum { gate, up, down };

/// Refusals of accept (construction time; never a per-call branch).
pub const Refusal = error{
    /// the arch's activation is not the one the quant's texts fuse
    ActivationNotFused,
    /// dims the quant's kernels are not compiled for
    DimsNotImplemented,
    /// an MoE input dtype the quant's kernels do not read
    InputDtype,
    /// decode rows (8 tokens x top_k) wider than the quant's decode tables
    TopKTooWide,
    /// a quant with pinned texts accepted without the load context's kernel set
    NoKernelSet,
    /// a quant whose parameters come from the group's description accepted without it
    NoWeightDescription,
    /// a description the quant does not serve (its claims would have declined it)
    NotClaimed,
    /// bank arrays that are not the kernels' signature
    BankArrays,
};

pub fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

/// A declined claim, the first mismatch in `why`.
pub fn decline(why: ?*Diag, comptime fmt: []const u8, args: anytype) ?Priority {
    if (why) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return null;
}

/// The string field `name` of a JSON object (null when absent or not a string).
pub fn str(v: std.json.Value, name: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const f = v.object.get(name) orelse return null;
    return if (f == .string) f.string else null;
}

/// The integer field `name` of a JSON object (null when absent or not an integer).
pub fn int(v: std.json.Value, name: []const u8) ?i64 {
    if (v != .object) return null;
    const f = v.object.get(name) orelse return null;
    return if (f == .integer) f.integer else null;
}

/// The object field `name` of a JSON object (`.null` when absent).
pub fn obj(v: std.json.Value, name: []const u8) std.json.Value {
    if (v != .object) return .null;
    return v.object.get(name) orelse .null;
}

// ── The interface checks (comptime) ──

fn expectFn(comptime where: []const u8, comptime F: type, comptime params: []const type, comptime Payload: ?type) void {
    const info = @typeInfo(F).@"fn";
    if (info.param_types.len != params.len) @compileError(where ++ ": takes a different parameter count than the C2 contract");
    for (info.param_types, params) |p, t| {
        if (p.? != t) @compileError(where ++ ": parameter " ++ @typeName(p.?) ++ " where the C2 contract has " ++ @typeName(t));
    }
    if (Payload) |want| {
        const R = info.return_type.?;
        const P = switch (@typeInfo(R)) {
            .error_union => |eu| eu.payload,
            else => R,
        };
        if (P != want) @compileError(where ++ ": returns " ++ @typeName(P) ++ " where the C2 contract has " ++ @typeName(want));
    }
}

/// Comptime: `Q` declares the quant half of the contract; a compile error names the first
/// declaration missing or mistyped.
pub fn check(comptime Q: type) void {
    comptime {
        for ([_][]const u8{ "name", "Arrays", "claims", "Accepted", "accept" }) |d| {
            if (!@hasDecl(Q, d)) @compileError("quant " ++ @typeName(Q) ++ ": no " ++ d);
        }
        if (@TypeOf(Q.name) != []const u8 and @TypeOf(Q.name) != *const [Q.name.len:0]u8) @compileError("quant " ++ @typeName(Q) ++ ": name is not a string");
        expectFn("quant " ++ @typeName(Q) ++ ".claims", @TypeOf(Q.claims), &.{ *const BankPeek, ?*Diag }, ?Priority);
    }
}

/// Comptime: `Q.Accepted(G)` declares the call half of the contract on backend G.
pub fn checkAccepted(comptime Q: type, comptime G: type) void {
    comptime {
        check(Q);
        const A = Q.Accepted(G);
        const Ar = Q.Arrays(G.T);
        const Bank = BankArrays(Ar);
        const where = "quant " ++ @typeName(Q) ++ ".Accepted";
        if (!@hasDecl(A, "max_decode_rows") or @TypeOf(A.max_decode_rows) != u32) @compileError(where ++ ": no max_decode_rows: u32");
        expectFn(where ++ ".checkBank", @TypeOf(A.checkBank), &.{ *const A, *G, Bank, *Diag }, void);
        expectFn(where ++ ".gateUp", @TypeOf(A.gateUp), &.{ *const A, *G, G.T, G.T, Ar, Ar }, G.T);
        expectFn(where ++ ".down", @TypeOf(A.down), &.{ *const A, *G, G.T, G.T, Ar }, G.T);
        expectFn(where ++ ".prefill", @TypeOf(A.prefill), &.{ *A, *G, u32, G.T, PrefillRows, Bank }, G.T);
        expectFn(where ++ ".finishPrefill", @TypeOf(A.finishPrefill), &.{ *A, *G }, void);
        expectFn(where ++ ".deinit", @TypeOf(A.deinit), &.{ *A, *G }, void);
        // accept(G, a, g, ctx, spec, diag) !*A: instantiated through a wrapper of the contract's
        // signature, so a mistyped accept fails here, named
        const Wrap = struct {
            fn call(a: Allocator, g: *G, ctx: Context, spec: Spec, diag: *Diag) anyerror!*A {
                return Q.accept(G, a, g, ctx, spec, diag);
            }
        };
        _ = &Wrap.call;
    }
}

// ── The activation on the backend's graph (the adapter's; a fused quant runs its own) ──

/// The backend methods `activation` uses.
pub const activation_methods = [_][]const u8{ "mul", "silu", "clip", "minimum", "scalar", "dtypeOf" };

/// The arch's activation of the gate and up rows as backend ops, in mlx-lm's `SwitchGLU` +
/// `ClampedSwiGLU` order (m1 `deepseek_v41_dspark_head.zig` Resident.routed): up clipped to
/// +-limit and gate capped at limit (the scalars in the operand's dtype), then silu(gate) * up.
pub fn activation(comptime G: type, g: *G, act: Activation, gate: G.T, up: G.T) !G.T {
    switch (act) {
        .swiglu => return g.mul(try g.silu(gate), up),
        .swiglu_clamped => |lim| {
            const du = g.dtypeOf(up);
            const uc = try g.clip(up, try g.scalar(-lim, du), try g.scalar(lim, du));
            const gc = try g.minimum(gate, try g.scalar(lim, g.dtypeOf(gate)));
            return g.mul(try g.silu(gc), uc);
        },
    }
}

// ── FromGatherMatmul: the stock path as a C2 client ──

/// A C2 quant from a gather-matmul quant `Q` (PLG's `gatherMatmul`). This makes the stock path
/// an explicit construction-time client, not a fallback. `Q` declares:
///   name, Arrays(T), claims (as a C2 quant);
///   Params: type, and params(*const BankPeek, Spec, *Diag) !Params (what the description fixes);
///   gatherMatmul(G, *G, *const Params, x [rows, in], w: Arrays(G.T), rhs [rows] u32, sorted: bool)
///       !G.T -> [rows, out];
///   checkArrays(G, *G, *const Params, Proj, Arrays(G.T), in: u32, out: u32, *Diag) !void.
/// The adapter derives the rest:
///   - gateUp: two gathers, then the arch's activation on the backend's graph;
///   - down: one gather;
///   - prefill: the routed rows sorted by slot on the host, the three gathers with sorted indices,
///     the rows put back (mlx-lm's SwitchGLU at 64 routed rows and above);
///   - finishPrefill: nothing is in flight;
///   - checkBank: Q's check per projection;
///   - accept: MLX's own kernels, nothing to self-check.
pub fn FromGatherMatmul(comptime Q: type) type {
    return struct {
        pub const name = Q.name;
        pub const Arrays = Q.Arrays;
        pub const claims = Q.claims;
        /// mlx-lm's SwitchGLU sorts the routed rows from 64 on (`do_sort = indices.size >= 64`)
        pub const sort_rows = 64;

        pub fn Accepted(comptime G: type) type {
            comptime {
                for (activation_methods ++ [_][]const u8{ "take", "hostArray", "reshape", "shapeOf" }) |m| {
                    if (!@hasDecl(G, m)) @compileError("quant " ++ Q.name ++ ": the backend " ++ @typeName(G) ++ " lacks " ++ m);
                }
            }
            return struct {
                const Self = @This();
                const A = Q.Arrays(G.T);
                pub const max_decode_rows: u32 = sort_rows - 1;
                a: Allocator,
                params: Q.Params,
                spec: Spec,
                // host scratch of prefill, reused across calls
                order: std.ArrayList(u32) = .empty,
                ids: std.ArrayList(u32) = .empty,
                src: std.ArrayList(u32) = .empty,
                inv: std.ArrayList(u32) = .empty,

                pub fn checkBank(self: *const Self, g: *G, bank: BankArrays(A), diag: *Diag) !void {
                    const s = &self.spec;
                    try Q.checkArrays(G, g, &self.params, .gate, bank.gate, s.hidden, s.inter, diag);
                    try Q.checkArrays(G, g, &self.params, .up, bank.up, s.hidden, s.inter, diag);
                    try Q.checkArrays(G, g, &self.params, .down, bank.down, s.inter, s.hidden, diag);
                }

                /// x [rows, hidden], slot_ids u32 [rows] -> the activation [rows, inter].
                pub fn gateUp(self: *const Self, g: *G, x: G.T, slot_ids: G.T, gate: A, up: A) !G.T {
                    const gg = try Q.gatherMatmul(G, g, &self.params, x, gate, slot_ids, false);
                    const uu = try Q.gatherMatmul(G, g, &self.params, x, up, slot_ids, false);
                    return activation(G, g, self.spec.act, gg, uu);
                }

                /// h [rows, inter], slot_ids u32 [rows] -> [rows, hidden].
                pub fn down(self: *const Self, g: *G, h: G.T, slot_ids: G.T, d: A) !G.T {
                    return Q.gatherMatmul(G, g, &self.params, h, d, slot_ids, false);
                }

                /// x [tokens or rows, hidden], the call's routed rows -> [rows, hidden] in routed-row order.
                pub fn prefill(self: *Self, g: *G, layer: u32, x: G.T, rows: PrefillRows, bank: BankArrays(A)) !G.T {
                    _ = layer;
                    const n = rows.slot.len;
                    try self.order.resize(self.a, n);
                    try self.ids.resize(self.a, n);
                    try self.src.resize(self.a, n);
                    try self.inv.resize(self.a, n);
                    for (self.order.items, 0..) |*o, i| o.* = @intCast(i);
                    // stable: equal slots keep their routed order (mlx-lm's argsort of the flat indices)
                    std.sort.block(u32, self.order.items, rows.slot, struct {
                        fn lt(slot: []const u32, x0: u32, x1: u32) bool {
                            return slot[x0] < slot[x1];
                        }
                    }.lt);
                    for (self.order.items, 0..) |o, i| {
                        self.ids.items[i] = rows.slot[o];
                        self.src.items[i] = if (rows.act_row) |ar| ar[o] else o;
                        self.inv.items[o] = @intCast(i);
                    }
                    const n_c: c_int = @intCast(n);
                    const src = try g.hostArray(std.mem.sliceAsBytes(self.src.items), &.{n_c}, .uint32);
                    const rhs = try g.hostArray(std.mem.sliceAsBytes(self.ids.items), &.{n_c}, .uint32);
                    const xs = try g.take(x, src, 0);
                    const gg = try Q.gatherMatmul(G, g, &self.params, xs, bank.gate, rhs, true);
                    const uu = try Q.gatherMatmul(G, g, &self.params, xs, bank.up, rhs, true);
                    const h = try activation(G, g, self.spec.act, gg, uu);
                    const y = try Q.gatherMatmul(G, g, &self.params, h, bank.down, rhs, true);
                    return g.take(y, try g.hostArray(std.mem.sliceAsBytes(self.inv.items), &.{n_c}, .uint32), 0);
                }

                /// Nothing is left in flight (every call's graph is the caller's).
                pub fn finishPrefill(self: *Self, g: *G) !void {
                    _ = self;
                    _ = g;
                }

                pub fn deinit(self: *Self, g: *G) void {
                    _ = g;
                    inline for (.{ &self.order, &self.ids, &self.src, &self.inv }) |l| l.deinit(self.a);
                    self.a.destroy(self);
                }
            };
        }

        /// Q's parameters from the claimed group's description (refused: none given, or one Q
        /// does not serve). No kernels of its own: nothing to self-check.
        pub fn accept(comptime G: type, a: Allocator, g: *G, ctx: Context, spec: Spec, diag: *Diag) !*Accepted(G) {
            _ = g;
            const peek = ctx.peek orelse return refuse(diag, error.NoWeightDescription, "quant {s}: accepted without the group's description", .{Q.name});
            const p = try Q.params(peek, spec, diag);
            const acc = try a.create(Accepted(G));
            acc.* = .{ .a = a, .params = p, .spec = spec };
            return acc;
        }
    };
}

// ── GatherQmm: mlx-serve's built-in routed-expert quant ──

/// mlx-serve's built-in routed-expert quant as a C2 client: the body of `transformer.zig`
/// gatherExpertMm.
///   - MLX's gather_qmm for affine / nvfp4 / mxfp4 / mxfp8 experts: bits and group size from the
///     group's quantization_config; affine carries biases.
///   - MLX's gather_mm for dense experts, with sorted indices forced off (mlx's dense gather_mm is
///     wrong with them: transformer.zig:40641).
/// The backend runs it:
///     gatherMatmul(x [rows, 1, in], w, scales: ?T, biases: ?T, rhs [rows] u32, bits, group_size,
///                  mode: ?QuantMode, sorted) !T -> [rows, 1, out]
/// (the model lane adds it to MlxOps over mlx_gather_qmm / mlx_gather_mm). V4.1's DSpark head
/// (`mtp.0.ffn.experts.*`: mxfp4, group 32) is its client.
pub const GatherQmm = struct {
    pub const name = "mlx-gather-qmm";

    /// w: u32 [slots, out, in * bits / 32] (dense: [slots, out, in]); scales [slots, out, in /
    /// group_size]; biases (affine) as scales.
    pub fn Arrays(comptime T: type) type {
        return struct { w: T, scales: ?T = null, biases: ?T = null };
    }

    /// mode null = dense experts (gather_mm).
    pub const Params = struct { mode: ?QuantMode, bits: u32, group_size: u32 };

    /// The mode's (bits, group sizes) MLX 0.32.2 quantizes.
    fn valid(mode: QuantMode, bits: u32, group: u32) bool {
        return switch (mode) {
            .affine => (bits >= 2 and bits <= 8 and bits != 7) and (group == 32 or group == 64 or group == 128),
            .mxfp4 => bits == 4 and group == 32,
            .nvfp4 => bits == 4 and group == 16,
            .mxfp8 => bits == 8 and group == 32,
            // Raw ggml blocks are no MLX quantization.
            .gguf => false,
        };
    }

    fn parse(peek: *const BankPeek, why: ?*Diag) ?Params {
        const q = peek.quantization;
        if (q == .null) return .{ .mode = null, .bits = 16, .group_size = 0 };
        if (q != .object) {
            _ = decline(why, "quant mlx-gather-qmm: quantization is not an object", .{});
            return null;
        }
        const mode_name = str(q, "mode") orelse "affine";
        const mode = QuantMode.fromString(mode_name) orelse {
            _ = decline(why, "quant mlx-gather-qmm: quantization.mode \"{s}\" is not an MLX quantization mode", .{mode_name});
            return null;
        };
        const bits = int(q, "bits") orelse {
            _ = decline(why, "quant mlx-gather-qmm: quantization.bits missing", .{});
            return null;
        };
        const group = int(q, "group_size") orelse {
            _ = decline(why, "quant mlx-gather-qmm: quantization.group_size missing", .{});
            return null;
        };
        if (bits < 0 or group < 0 or !valid(mode, @intCast(bits), @intCast(group))) {
            _ = decline(why, "quant mlx-gather-qmm: {t} at {d} bits, group {d} is not an MLX quantization", .{ mode, bits, group });
            return null;
        }
        return .{ .mode = mode, .bits = @intCast(bits), .group_size = @intCast(group) };
    }

    pub fn claims(peek: *const BankPeek, why: ?*Diag) ?Priority {
        _ = parse(peek, why) orelse return null;
        return .generic;
    }

    pub fn params(peek: *const BankPeek, spec: Spec, diag: *Diag) !Params {
        _ = spec;
        return parse(peek, diag) orelse error.NotClaimed;
    }

    /// x [rows, in], rhs u32 [rows] (each row's slot) -> [rows, out].
    pub fn gatherMatmul(comptime G: type, g: *G, p: *const Params, x: G.T, w: Arrays(G.T), rhs: G.T, sorted: bool) !G.T {
        const sx = g.shapeOf(x);
        const rows = sx.d[0];
        const x3 = try g.reshape(x, &.{ rows, 1, sx.d[1] });
        const y = try g.gatherMatmul(x3, w.w, w.scales, w.biases, rhs, p.bits, p.group_size, p.mode, sorted and p.mode != null);
        const sy = g.shapeOf(y);
        return g.reshape(y, &.{ rows, sy.d[sy.n - 1] });
    }

    /// A projection's per-slot arrays against the params and the projection's (in, out).
    pub fn checkArrays(comptime G: type, g: *G, p: *const Params, which: Proj, w: Arrays(G.T), in: u32, out: u32, diag: *Diag) !void {
        const sw = g.shapeOf(w.w);
        const packed_in: c_int = if (p.mode != null) @intCast(in * p.bits / 32) else @intCast(in);
        if (sw.n != 3 or sw.d[1] != @as(c_int, @intCast(out)) or sw.d[2] != packed_in)
            return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} w is {any}, the params read [slots, {d}, {d}]", .{ which, sw.slice(), out, packed_in });
        if (p.mode) |mode| {
            const sc = w.scales orelse return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} has no scales", .{which});
            const ss = g.shapeOf(sc);
            if (ss.n != 3 or ss.d[0] != sw.d[0] or ss.d[1] != sw.d[1] or ss.d[2] != @as(c_int, @intCast(in / p.group_size)))
                return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} scales are {any}, want [slots, {d}, {d}]", .{ which, ss.slice(), out, in / p.group_size });
            if (mode.hasBiases() != (w.biases != null))
                return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} biases {s} for {t}", .{ which, if (w.biases != null) "given" else "missing", mode });
        }
    }
};
