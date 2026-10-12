//! GLM-5.3-Flash's fused NVFP4 decode experts (glm5next/fused-nvfp4): MiMo's k1 decode MoE texts (src/arch/mimo_v2/moe.zig,
//! `KernelFamily.k1`) at the same tiling and dispatch geometry, over an NVFP4 bank (E2M1 codes 8 per U32 word, one E4M3
//! scale per 16 inputs, one f32 global scale g per slot and projection) instead of MiMo's MXFP4 one (E8M0 per 32):
//!   - `glm5n_nvfp4_gateup`: gate + up of every routed pair, then the clamped SwiGLU (silu(min(gate, limit)) *
//!     clip(up, -limit, limit), GLM-5.3-Flash's swiglu_limit 10; MiMo's f32 SiLU of the bf16 gate, rounded once), bf16
//!     [P, inter];
//!   - `glm5n_nvfp4_down`: each pair's down row, bf16 [P, hidden] (no score fold: the arch's combine follows).
//! A weight is MLX's fp_dequantize value in MLX's order (inv = g / 2688, s = e4m3 * inv, w = s * e2m1, all f32): the
//! texts decode each code as e2m1 x 2^-14 (a half, exact) and carry 2^14 in the lane's scale (exact), so each product
//! rounds as s * e2m1 does (`decode_kernels_ref.zig` checks it on the test vectors and on every scale byte and code).
//! The dot (each weight times its activation, two partial sums, the fold, simd_sum) is k1's: rounding-class against MLX's
//! gather_qmm nvfp4, which multiplies the unscaled codes and applies g / 2688 last.
//! Each wave of P routed pairs (P <= 64: up to 8 tokens of top-8) takes one launch of each text; ids are packed
//! `bank << 24 | slot row` over the expert stream's three banks (base, ext, transient), as MiMo's. A one-token wave
//! (P <= 8, every slot distinct) runs 4 rows per simdgroup and one member per leader; a wider wave the union rule (a
//! slot's pairs read it once per leader of up to 4 members), per member row the one-token wave's arithmetic.
//! Registered the way GLM-5.3's trunk kernels are (`trunk_kernels.zig`): the texts under kernels/glm5_next_nvfp4/ with a
//! manifest of their signatures, template parameters, threadgroup, registered instantiations and checks
//! (`scripts/glm5next_nvfp4_kernels_manifest.py` writes it and prints its pin). `Registry.init` checks the manifest
//! against `manifest_sha256` and every text against the manifest, once, and refuses by name; `Registry.bind` builds the
//! mlx fast kernels on a GPU stream, once. `install` is the module's construction: the set on its stream, the compile
//! self-check at every registered instantiation, and the launches prepared per wave width (`Fused`). Not routed here:
//! the arch's module installs it. docs/glm5-next-decode-kernels.md is the dispatch contract.

const std = @import("std");
const mlx = @import("sdk").mlx;
const kset = @import("../../sdk_ext/kernel_set.zig");
/// The registry's self-check plan (`sdk_ext.kernels.KernelSet(R)` runs it; the module at its construction).
pub const selfcheck = @import("decode_kernels_selfcheck.zig");

const Allocator = std.mem.Allocator;

/// sha256 of kernels/glm5_next_nvfp4/manifest.json (scripts/glm5next_nvfp4_kernels_manifest.py prints it): pins every text.
pub const manifest_sha256 = "38e703b97dc1d827d74c8104a39e337163d4d547c6d97668906e7c38d6969571";
pub const format = "mlx-stream-glm5next-nvfp4-kernels-v1";
const dir = "../../kernels/glm5_next_nvfp4/";
const prefix = "glm5next nvfp4 kernels";

/// Every kernel of the registry; the tag is the kernel's MLX name and its file name.
pub const Kernel = enum { glm5n_nvfp4_gateup, glm5n_nvfp4_down };
pub const Header = enum { glm5n_nvfp4 };
/// `compile`: the kernel builds, binds and completes one launch over zeros at each registered instantiation (the
/// construction's acceptance). Its numerics are the window's parity test's (`decode_kernels_parity.zig`).
pub const Check = enum { compile };
pub const Family = enum { gate_up, down };

pub fn headerFile(h: Header) []const u8 {
    return switch (h) {
        inline else => |t| "header_" ++ @tagName(t) ++ ".metal",
    };
}

pub fn familyOf(k: Kernel) Family {
    return switch (k) {
        .glm5n_nvfp4_gateup => .gate_up,
        .glm5n_nvfp4_down => .down,
    };
}

pub const n_kernels = std.meta.fieldNames(Kernel).len;
pub const n_headers = std.meta.fieldNames(Header).len;
/// The expert stream's slot banks (`sdk_ext.expert.BankKind`: base, ext, transient), each a text input group.
pub const n_banks: u32 = std.meta.fieldNames(@import("../../sdk_ext.zig").expert.BankKind).len;
/// x, three arrays (words, scales, global scales) per projection and bank, ids: gate/up's 20, down's 11.
pub const max_inputs = 2 + 6 * n_banks;
pub const max_outputs = 1;
pub const max_template = 7;
pub const max_headers = 1;

// ── GLM-5.3-Flash's routed geometry and the waves' plan ──

/// The routed experts the texts are registered at (text_config: hidden_size, moe_intermediate_size,
/// num_experts_per_tok, swiglu_limit; the pack's nvfp4 group).
pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
pub const top_k: u32 = 8;
pub const group_size: u32 = 16;
pub const swiglu_limit: u32 = 10;
/// Simdgroups per threadgroup (the texts' threadgroup [32, 2, 1]).
pub const simdgroups: u32 = 2;
/// The widest wave: 8 tokens of top-8 (the union vote covers a wave in two 32-lane ballots).
pub const max_tokens: u32 = 8;
pub const max_pairs: u32 = max_tokens * top_k;
/// A wave from this many pairs takes the union geometry (one token's pairs name distinct slots).
pub const union_min_pairs: u32 = top_k + 1;
/// A packed id's row field (`bank << 24 | row`).
pub const row_bits: u5 = 24;

comptime {
    if (hidden % 512 != 0 or inter % 512 != 0) @compileError("glm5next nvfp4: K and I must be multiples of 512 (16 values x 32 lanes)");
    if (512 % group_size != 0 or group_size != 16) @compileError("glm5next nvfp4: a lane's 16 codes are one E4M3 group");
    if (max_pairs > 64) @compileError("glm5next nvfp4: the union vote takes at most 64 pairs per wave");
}

/// A text's launch geometry: output rows per simdgroup (`ROWS`), and the widest union one leader serves (`MAXM`).
pub const Geometry = struct { rows: u32, maxm: u32 };

/// MiMo's k1 plan (src/arch/mimo_v2/moe.zig `k1_plan`): a one-token wave keeps 4 rows per simdgroup and one member per
/// leader; a wider wave the union rule, 4 members per leader, gate/up at 2 rows (its decoded codes of two projections)
/// and down at 4.
pub const Plan = struct {
    serial_gate_up: Geometry = .{ .rows = 4, .maxm = 1 },
    serial_down: Geometry = .{ .rows = 4, .maxm = 1 },
    union_gate_up: Geometry = .{ .rows = 2, .maxm = 4 },
    union_down: Geometry = .{ .rows = 4, .maxm = 4 },

    pub fn gateUp(p: Plan, pairs: u32) Geometry {
        return if (pairs >= union_min_pairs) p.union_gate_up else p.serial_gate_up;
    }

    pub fn down(p: Plan, pairs: u32) Geometry {
        return if (pairs >= union_min_pairs) p.union_down else p.serial_down;
    }
};
pub const plan: Plan = .{};

/// One registered instantiation: `ROWS`, `MAXM`, the reduction width `k` (K or I), the output rows `n` (N or H), `GS` and
/// the gate/up's `LIMIT` (0 for down).
pub const Instance = struct { rows: u32, maxm: u32, k: u32, n: u32, gs: u32, limit: u32 };

/// The instantiations this build serves, which the manifest must register as they are: `plan`'s geometries at
/// GLM-5.3-Flash's shapes.
pub const gate_up_instances = [_]Instance{
    .{ .rows = plan.serial_gate_up.rows, .maxm = plan.serial_gate_up.maxm, .k = hidden, .n = inter, .gs = group_size, .limit = swiglu_limit },
    .{ .rows = plan.union_gate_up.rows, .maxm = plan.union_gate_up.maxm, .k = hidden, .n = inter, .gs = group_size, .limit = swiglu_limit },
};
pub const down_instances = [_]Instance{
    .{ .rows = plan.serial_down.rows, .maxm = plan.serial_down.maxm, .k = inter, .n = hidden, .gs = group_size, .limit = 0 },
    .{ .rows = plan.union_down.rows, .maxm = plan.union_down.maxm, .k = inter, .n = hidden, .gs = group_size, .limit = 0 },
};

pub fn instancesOf(k: Kernel) []const Instance {
    return switch (familyOf(k)) {
        .gate_up => &gate_up_instances,
        .down => &down_instances,
    };
}

/// The instantiation a wave of `pairs` launches of `k` (`plan`).
pub fn instanceFor(k: Kernel, pairs: u32) Instance {
    const set = instancesOf(k);
    const g = switch (familyOf(k)) {
        .gate_up => plan.gateUp(pairs),
        .down => plan.down(pairs),
    };
    for (set) |i| if (i.rows == g.rows and i.maxm == g.maxm) return i;
    unreachable;
}

/// The model geometry the set serves, refused by name otherwise (the module does not install the set then).
pub fn checkModel(hidden_size: u32, moe_intermediate_size: u32, experts_per_token: u32, limit: f64, diag: ?*Diag) Refusal!void {
    if (hidden_size != hidden or moe_intermediate_size != inter or experts_per_token != top_k or limit != @as(f64, @floatFromInt(swiglu_limit)))
        return refuse(diag, error.InstanceNotRegistered, prefix ++ ": hidden {d} / inter {d} / top-{d} / swiglu_limit {d} is not the registered {d} / {d} / top-{d} / {d}", .{ hidden_size, moe_intermediate_size, experts_per_token, limit, hidden, inter, top_k, swiglu_limit });
}

// ── The texts and the manifest ──

pub const Texts = struct {
    manifest: []const u8,
    sources: [n_kernels][:0]const u8,
    headers: [n_headers][:0]const u8,
};

pub const embedded: Texts = .{
    .manifest = @embedFile(dir ++ "manifest.json"),
    .sources = embedSources(),
    .headers = embedHeaders(),
};

fn embedSources() [n_kernels][:0]const u8 {
    var out: [n_kernels][:0]const u8 = undefined;
    inline for (std.meta.fieldNames(Kernel), 0..) |name, i| out[i] = @embedFile(dir ++ name ++ ".metal");
    return out;
}

fn embedHeaders() [n_headers][:0]const u8 {
    var out: [n_headers][:0]const u8 = undefined;
    inline for (comptime std.enums.values(Header), 0..) |h, i| out[i] = @embedFile(dir ++ comptime headerFile(h));
    return out;
}

pub const Refusal = error{
    ManifestNotPinned,
    ManifestSyntax,
    ManifestFormat,
    UnknownKernel,
    MissingKernel,
    DuplicateKernel,
    UnknownHeader,
    MissingHeader,
    TextSha256Mismatch,
    SchemaInvalid,
    SignatureMismatch,
    InstancesNotServed,
    InstanceNotRegistered,
    NotGpuStream,
    KernelCreateFailed,
};

pub const Diag = @import("sdk").Diag;

const refuse = @import("../../sdk_ext/expert/bank_file.zig").refuse;

/// An argument's name and dtype; an argument of dtype "T" takes the launch's template dtype T.
pub const Arg = struct { name: [:0]const u8, dtype: ?mlx.mlx_dtype };

/// One argument of the dispatch's signature: its name and dtype (null: the launch's template dtype T).
pub const SigArg = struct { name: []const u8, dtype: ?mlx.mlx_dtype };

fn signatureLen(f: Family) usize {
    return if (f == .gate_up) 2 + 6 * n_banks else 2 + 3 * n_banks;
}

/// The signature `Fused` binds by position, which the manifest must list as it is: gate/up's x, then per bank the gate's
/// words, scales and global scales and the up's, then ids; down's act, then per bank the down's three, then ids.
pub fn signature(comptime f: Family) [signatureLen(f)]SigArg {
    const projs: []const []const u8 = if (f == .gate_up) &.{ "gate", "up" } else &.{"down"};
    var out: [signatureLen(f)]SigArg = undefined;
    out[0] = .{ .name = if (f == .gate_up) "x" else "act", .dtype = null };
    var i: usize = 1;
    for (0..n_banks) |b| for (projs) |p| {
        out[i] = .{ .name = std.fmt.comptimePrint("{s}_w{d}", .{ p, b }), .dtype = .uint32 };
        out[i + 1] = .{ .name = std.fmt.comptimePrint("{s}_s{d}", .{ p, b }), .dtype = .uint8 };
        out[i + 2] = .{ .name = std.fmt.comptimePrint("{s}_g{d}", .{ p, b }), .dtype = .float32 };
        i += 3;
    };
    out[i] = .{ .name = "ids", .dtype = .uint32 };
    return out;
}

/// The template parameters in the texts' order (after T: the ints `LaunchConfig.ints` carries at the same positions).
pub fn templateNames(f: Family) []const []const u8 {
    return switch (f) {
        .gate_up => &.{ "T", "ROWS", "K", "N", "GS", "MAXM", "LIMIT" },
        .down => &.{ "T", "ROWS", "I", "H", "GS", "MAXM" },
    };
}

pub const Entry = struct {
    kernel: Kernel,
    family: Family,
    source: [:0]const u8,
    headers: []const Header,
    /// the headers' texts in order, concatenated: the one header mlx compiles before `source`
    header_text: [:0]const u8,
    text_sha256: [32]u8,
    inputs: []const Arg,
    outputs: []const Arg,
    template: []const [:0]const u8,
    threadgroup: [3]u32,
    instances: []const Instance,
    checks: std.EnumSet(Check),

    pub fn registered(e: *const Entry, inst: Instance) bool {
        for (e.instances) |i| if (std.meta.eql(i, inst)) return true;
        return false;
    }
};

pub const Registry = struct {
    arena: std.heap.ArenaAllocator,
    entries: [n_kernels]Entry,
    headers: [n_headers][:0]const u8,
    manifest_hex: [64]u8,

    /// Parses the manifest, checks it against `pin` (hex sha256) and every text against it, once.
    pub fn init(gpa: Allocator, texts: *const Texts, pin: []const u8, diag: ?*Diag) (Refusal || Allocator.Error)!Registry {
        const got = try kset.checkPin(texts.manifest, pin, prefix, diag);
        var reg: Registry = .{ .arena = .init(gpa), .entries = undefined, .headers = texts.headers, .manifest_hex = got };
        errdefer reg.arena.deinit();
        const a = reg.arena.allocator();
        const m = std.json.parseFromSliceLeaky(JManifest, a, texts.manifest, .{ .allocate = .alloc_always }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ManifestSyntax, prefix ++ ": manifest does not parse ({t})", .{e}),
        };
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, prefix ++ ": format \"{s}\" is not " ++ format, .{m.format});
        try kset.adoptHeaders(Header, &texts.headers, m.headers, prefix, diag);
        for (m.headers) |h| {
            const id = std.meta.stringToEnum(Header, h.id).?;
            if (!std.mem.eql(u8, h.file, headerFile(id))) return refuse(diag, error.SchemaInvalid, prefix ++ ": header {t} is file \"{s}\", not {s}", .{ id, h.file, headerFile(id) });
        }
        var tally: kset.Tally(Kernel) = .{};
        for (m.kernels) |jk| {
            const k = try tally.admit(jk.name, prefix, diag);
            reg.entries[@backingInt(k)] = try adoptKernel(a, texts, k, jk, diag);
        }
        try tally.done(prefix, diag);
        return reg;
    }

    pub fn deinit(self: *Registry) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const Registry, k: Kernel) *const Entry {
        return &self.entries[@backingInt(k)];
    }

    /// `inst` of `k`, refused by name when the manifest does not register it.
    pub fn instance(self: *const Registry, k: Kernel, inst: Instance, diag: ?*Diag) Refusal!Instance {
        if (!self.get(k).registered(inst)) return refuse(diag, error.InstanceNotRegistered, prefix ++ ": {t} is not registered at ROWS {d}, MAXM {d}, {d} x {d}, GS {d}, LIMIT {d}", .{ k, inst.rows, inst.maxm, inst.n, inst.k, inst.gs, inst.limit });
        return inst;
    }

    /// Builds every kernel's mlx object on `stream`, once; the objects reach Metal at their first launch.
    pub fn bind(self: *const Registry, stream: mlx.mlx_stream, diag: ?*Diag) Refusal!Bound {
        if (!mlx.streamIsGpu(stream)) return refuse(diag, error.NotGpuStream, prefix ++ ": bind needs a GPU stream", .{});
        var b: Bound = .{ .reg = self, .stream = stream, .kernels = @splat(.{}) };
        errdefer b.deinit();
        for (&self.entries, 0..) |*e, i| {
            var in_names: [max_inputs][*:0]const u8 = undefined;
            var out_names: [max_outputs][*:0]const u8 = undefined;
            for (e.inputs, 0..) |arg, n| in_names[n] = arg.name.ptr;
            for (e.outputs, 0..) |arg, n| out_names[n] = arg.name.ptr;
            b.kernels[i] = kset.newKernel(@tagName(e.kernel), in_names[0..e.inputs.len], out_names[0..e.outputs.len], e.source.ptr, e.header_text.ptr, true) orelse
                return refuse(diag, error.KernelCreateFailed, prefix ++ ": mlx_fast_metal_kernel_new({t}) failed", .{e.kernel});
        }
        return b;
    }
};

// ── Launches ──

/// One launch's geometry: the grid (threads), the threadgroup, the template values in the entry's order (ints; `T` a
/// dtype), the output's shape and dtype.
pub const LaunchConfig = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    ints: [max_template]i32 = @splat(0),
    dtype: mlx.mlx_dtype = .bfloat16,
    n_out: usize = 1,
    out_ranks: [max_outputs]usize = @splat(0),
    out_shapes: [max_outputs][4]c_int = @splat(@splat(0)),
    out_dtypes: [max_outputs]mlx.mlx_dtype = @splat(.bfloat16),
};

/// The launch of `k` over a wave of `pairs` routed pairs at instantiation `inst`: grid (32, n / ROWS, P), threadgroup
/// (32, simdgroups, 1), the template `T` = `dt`, then ROWS, K|I, N|H, GS, MAXM (and gate/up's LIMIT); the output
/// [P, n] in `dt`.
pub fn waveLaunch(k: Kernel, pairs: u32, inst: Instance, dt: mlx.mlx_dtype) LaunchConfig {
    std.debug.assert(pairs >= 1 and pairs <= max_pairs and inst.n % (inst.rows * simdgroups) == 0);
    var c: LaunchConfig = .{ .grid = .{ 32, inst.n / inst.rows, pairs }, .threadgroup = .{ 32, simdgroups, 1 }, .dtype = dt };
    c.ints[1] = @intCast(inst.rows);
    c.ints[2] = @intCast(inst.k);
    c.ints[3] = @intCast(inst.n);
    c.ints[4] = @intCast(inst.gs);
    c.ints[5] = @intCast(inst.maxm);
    if (familyOf(k) == .gate_up) c.ints[6] = @intCast(inst.limit);
    c.out_dtypes[0] = dt;
    c.out_ranks[0] = 2;
    c.out_shapes[0] = .{ @intCast(pairs), @intCast(inst.n), 0, 0 };
    return c;
}

pub const Prepared = kset.Prepared(Kernel);

/// The kernels built on one GPU stream (`Registry.bind`); freed after that stream has drained.
pub const Bound = kset.BoundOf(@This());

/// `cfg` as the mlx launch (`Bound.prepare`): the entry's template parameters in its order, `T` the config's dtype.
pub fn launchOf(reg: *const Registry, k: Kernel, cfg: *const LaunchConfig, shapes: *[max_outputs][]const c_int, template: *[16]kset.TemplateArg) kset.Launch {
    const e = reg.get(k);
    for (e.template, template[0..e.template.len], 0..) |name, *t, i| t.* = .{ .name = name, .value = if (std.mem.eql(u8, name, "T")) .{ .dtype = cfg.dtype } else .{ .int = cfg.ints[i] } };
    for (0..cfg.n_out) |i| shapes[i] = cfg.out_shapes[i][0..cfg.out_ranks[i]];
    return .{ .grid = cfg.grid, .threadgroup = cfg.threadgroup, .out_shapes = shapes[0..cfg.n_out], .out_dtypes = cfg.out_dtypes[0..cfg.n_out], .template = template[0..e.template.len] };
}

/// The load's kernel set over this registry (`sdk_ext.kernels.KernelSet`).
pub const Set = kset.KernelSet(@This()).Set;

/// Device tests: the set on GPU stream `s` (a refusal printed by name).
pub fn openForTest(a: Allocator, s: mlx.mlx_stream) !*Set {
    var diag: Diag = .{};
    return Set.init(a, .{ .device = .{ .stream = s } }, &diag) catch |e| {
        std.debug.print(prefix ++ " refused: {s}\n", .{diag.message()});
        return e;
    };
}

// ── The module's install and dispatch ──

/// One projection's slot arrays as the bank scatters them (the arch's `bank.Arrays`): the E2M1 words [slots, out, in / 8]
/// U32, the E4M3 scales [slots, out, in / 16] U8 and the f32 global scales [slots], whole (the texts' row-contiguous
/// inputs: a slice would be copied per call).
pub const ProjArrays = struct { w: mlx.mlx_array, scales: mlx.mlx_array, global_scale: mlx.mlx_array };
/// One bank's three projections.
pub const BankArrays = struct { gate: ProjArrays, up: ProjArrays, down: ProjArrays };

/// The launches of both texts prepared per wave width (P = 1 .. `max_pairs`), on a bound set: the module's decode calls
/// them with no per-call config work.
pub const Fused = struct {
    bound: *const Bound,
    gate_up_launch: [max_pairs]Prepared = undefined,
    down_launch: [max_pairs]Prepared = undefined,
    n_prepared: u32 = 0,

    /// Every wave width's two launches (bf16 rows), each at the instantiation `plan` names for it, which the registry must
    /// register (refused by name otherwise).
    pub fn init(bound: *const Bound, diag: ?*Diag) !Fused {
        var f: Fused = .{ .bound = bound };
        errdefer f.deinit();
        for (1..max_pairs + 1) |p| {
            const pairs: u32 = @intCast(p);
            const gi = try bound.reg.instance(.glm5n_nvfp4_gateup, instanceFor(.glm5n_nvfp4_gateup, pairs), diag);
            const di = try bound.reg.instance(.glm5n_nvfp4_down, instanceFor(.glm5n_nvfp4_down, pairs), diag);
            const gc = waveLaunch(.glm5n_nvfp4_gateup, pairs, gi, .bfloat16);
            const dc = waveLaunch(.glm5n_nvfp4_down, pairs, di, .bfloat16);
            f.gate_up_launch[p - 1] = try bound.prepare(.glm5n_nvfp4_gateup, &gc);
            f.down_launch[p - 1] = bound.prepare(.glm5n_nvfp4_down, &dc) catch |e| {
                f.gate_up_launch[p - 1].deinit();
                return e;
            };
            f.n_prepared = pairs;
        }
        return f;
    }

    pub fn deinit(f: *Fused) void {
        for (f.gate_up_launch[0..f.n_prepared], f.down_launch[0..f.n_prepared]) |*g, *d| {
            g.deinit();
            d.deinit();
        }
        f.n_prepared = 0;
    }

    /// `x` bf16 [P, hidden] (each pair's token row), `ids` u32 [P] packed `bank << 24 | slot row`; returns bf16 [P, inter]
    /// (the clamped SwiGLU rows; a new array, the caller frees it).
    pub fn gateUp(f: *const Fused, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [n_banks]BankArrays) !mlx.mlx_array {
        const p = pairsOf(ids) orelse return error.Glm5nWaveWidth;
        var in: [2 + 6 * n_banks]mlx.mlx_array = undefined;
        in[0] = x;
        for (banks, 0..) |b, i| {
            in[1 + 6 * i ..][0..6].* = .{ b.gate.w, b.gate.scales, b.gate.global_scale, b.up.w, b.up.scales, b.up.global_scale };
        }
        in[in.len - 1] = ids;
        var out: [1]mlx.mlx_array = undefined;
        try f.bound.applyPrepared(&f.gate_up_launch[p - 1], &in, &out);
        return out[0];
    }

    /// `h` bf16 [P, inter], `ids` as `gateUp`; returns bf16 [P, hidden]: each pair's down row (a new array, the caller
    /// frees it).
    pub fn down(f: *const Fused, h: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [n_banks]BankArrays) !mlx.mlx_array {
        const p = pairsOf(ids) orelse return error.Glm5nWaveWidth;
        var in: [2 + 3 * n_banks]mlx.mlx_array = undefined;
        in[0] = h;
        for (banks, 0..) |b, i| {
            in[1 + 3 * i ..][0..3].* = .{ b.down.w, b.down.scales, b.down.global_scale };
        }
        in[in.len - 1] = ids;
        var out: [1]mlx.mlx_array = undefined;
        try f.bound.applyPrepared(&f.down_launch[p - 1], &in, &out);
        return out[0];
    }

    fn pairsOf(ids: mlx.mlx_array) ?u32 {
        if (mlx.mlx_array_ndim(ids) != 1) return null;
        const p = mlx.mlx_array_size(ids);
        return if (p >= 1 and p <= max_pairs) @intCast(p) else null;
    }
};

/// What the module holds once installed: the set (pinned, both kernels built on its stream, the compile check passed at
/// every registered instantiation) and the prepared launches. `deinit` after the stream has drained.
pub const Installed = struct {
    set: *Set,
    fused: Fused,

    pub fn deinit(self: *Installed) void {
        self.fused.deinit();
        self.set.deinit();
        self.* = undefined;
    }
};

/// The module's construction (its GPU stream `s`, the model geometry already through `checkModel`): the set, the
/// construction's self-check (`.compile`: one launch over zeros of each kernel at each registered instantiation, its
/// ids in both of MLX's passings), and `Fused`. Refused by name (`diag`): a registry refusal, NotGpuStream /
/// KernelCreateFailed at bind, error.SelfCheckFailed naming the first failing kernel.
pub fn install(a: Allocator, s: mlx.mlx_stream, diag: *Diag) !Installed {
    const set = try Set.init(a, .{ .device = .{ .stream = s } }, diag);
    errdefer set.deinit();
    var report: selfcheck.Report = .{};
    defer report.deinit(a);
    try set.selfCheck(a, std.enums.values(Kernel), .compile, &report, diag);
    return .{ .set = set, .fused = try Fused.init(&set.bound, diag) };
}

// ── Manifest adoption (once, at init) ──

const JText = struct { id: []const u8 = "", file: []const u8, sha256: []const u8, bytes: u64 };
const JArg = struct { name: []const u8, dtype: []const u8 };
const JInstance = struct { ROWS: u32, MAXM: u32, K: u32, N: u32, GS: u32, LIMIT: u32 };
const JKernel = struct {
    name: []const u8,
    family: []const u8,
    source: JText,
    headers: []const []const u8,
    inputs: []const JArg,
    outputs: []const JArg,
    template: []const []const u8,
    threadgroup: [3]u32,
    instances: []const JInstance,
    checks: []const []const u8,
};
const JManifest = struct { format: []const u8, headers: []const JText, kernels: []const JKernel };

fn parseDtype(s: []const u8, k: Kernel, diag: ?*Diag) Refusal!?mlx.mlx_dtype {
    if (std.mem.eql(u8, s, "T")) return null;
    return std.meta.stringToEnum(mlx.mlx_dtype, s) orelse refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: unknown dtype \"{s}\"", .{ k, s });
}

fn adoptArgs(a: Allocator, js: []const JArg, max: usize, k: Kernel, diag: ?*Diag) (Refusal || Allocator.Error)![]const Arg {
    if (js.len == 0 or js.len > max) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: {d} arguments", .{ k, js.len });
    const out = try a.alloc(Arg, js.len);
    for (js, out, 0..) |j, *o, i| {
        for (js[0..i]) |prev| if (std.mem.eql(u8, prev.name, j.name)) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: argument {s} twice", .{ k, j.name });
        o.* = .{ .name = try a.dupeSentinel(u8, j.name, 0), .dtype = try parseDtype(j.dtype, k, diag) };
    }
    return out;
}

/// The manifest's inputs, outputs and template against the signature `Fused` binds (by position, so a renamed or
/// reordered input is refused by name here, never bound to the wrong buffer).
fn checkSignature(k: Kernel, inputs: []const Arg, outputs: []const Arg, template: []const [:0]const u8, diag: ?*Diag) Refusal!void {
    switch (familyOf(k)) {
        inline else => |f| {
            const want = comptime signature(f);
            if (inputs.len != want.len) return refuse(diag, error.SignatureMismatch, prefix ++ ": {t}: {d} inputs, the dispatch binds {d}", .{ k, inputs.len, want.len });
            for (inputs, want) |got, w| if (!std.mem.eql(u8, got.name, w.name) or got.dtype != w.dtype)
                return refuse(diag, error.SignatureMismatch, prefix ++ ": {t}: input {s} where the dispatch binds {s}", .{ k, got.name, w.name });
        },
    }
    if (outputs.len != 1 or !std.mem.eql(u8, outputs[0].name, "y") or outputs[0].dtype != null)
        return refuse(diag, error.SignatureMismatch, prefix ++ ": {t}: the output is not y of dtype T", .{k});
    const names = templateNames(familyOf(k));
    if (template.len != names.len) return refuse(diag, error.SignatureMismatch, prefix ++ ": {t}: {d} template parameters, not {d}", .{ k, template.len, names.len });
    for (template, names) |got, w| if (!std.mem.eql(u8, got, w))
        return refuse(diag, error.SignatureMismatch, prefix ++ ": {t}: template parameter {s} where the launch sets {s}", .{ k, got, w });
}

fn adoptKernel(a: Allocator, texts: *const Texts, k: Kernel, j: JKernel, diag: ?*Diag) (Refusal || Allocator.Error)!Entry {
    const source = texts.sources[@backingInt(k)];
    try kset.checkText(source, j.source.sha256, j.source.bytes, prefix, @tagName(k), diag);
    if (j.headers.len > max_headers) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: {d} headers", .{ k, j.headers.len });
    const headers = try a.alloc(Header, j.headers.len);
    var hbytes: usize = 0;
    for (j.headers, headers) |name, *o| {
        o.* = std.meta.stringToEnum(Header, name) orelse return refuse(diag, error.UnknownHeader, prefix ++ ": {t}: header \"{s}\"", .{ k, name });
        hbytes += texts.headers[@backingInt(o.*)].len;
    }
    const htext = try a.allocSentinel(u8, hbytes, 0);
    var at: usize = 0;
    for (headers) |h| {
        const t = texts.headers[@backingInt(h)];
        @memcpy(htext[at..][0..t.len], t);
        at += t.len;
    }
    const family = std.meta.stringToEnum(Family, j.family) orelse return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: family \"{s}\"", .{ k, j.family });
    if (family != familyOf(k)) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: family {t}, not {t}", .{ k, family, familyOf(k) });
    const tg = j.threadgroup;
    if (tg[0] != 32 or tg[1] != simdgroups or tg[2] != 1) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: threadgroup {d}x{d}x{d}, not 32x{d}x1", .{ k, tg[0], tg[1], tg[2], simdgroups });
    if (j.template.len == 0 or j.template.len > max_template) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: {d} template parameters", .{ k, j.template.len });
    const template = try a.alloc([:0]const u8, j.template.len);
    for (j.template, template) |t, *o| o.* = try a.dupeSentinel(u8, t, 0);
    const inputs = try adoptArgs(a, j.inputs, max_inputs, k, diag);
    const outputs = try adoptArgs(a, j.outputs, max_outputs, k, diag);
    try checkSignature(k, inputs, outputs, template, diag);
    // The instantiations are the ones this build serves, as they are (`gate_up_instances`, `down_instances`).
    const served = instancesOf(k);
    const inst = try a.alloc(Instance, j.instances.len);
    for (j.instances, inst) |ji, *o| o.* = .{ .rows = ji.ROWS, .maxm = ji.MAXM, .k = ji.K, .n = ji.N, .gs = ji.GS, .limit = ji.LIMIT };
    if (inst.len != served.len or !for (inst, served) |x, y| {
        if (!std.meta.eql(x, y)) break false;
    } else true) return refuse(diag, error.InstancesNotServed, prefix ++ ": {t}: the manifest registers {d} instantiations, not the {d} this build serves", .{ k, inst.len, served.len });
    var checks: std.EnumSet(Check) = .empty;
    for (j.checks) |c| checks.insert(std.meta.stringToEnum(Check, c) orelse return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: check \"{s}\"", .{ k, c }));
    if (!checks.contains(.compile)) return refuse(diag, error.SchemaInvalid, prefix ++ ": {t}: no compile check", .{k});
    return .{
        .kernel = k,
        .family = family,
        .source = source,
        .headers = headers,
        .header_text = htext,
        .text_sha256 = kset.hexSha(j.source.sha256).?,
        .inputs = inputs,
        .outputs = outputs,
        .template = template,
        .threadgroup = tg,
        .instances = inst,
        .checks = checks,
    };
}

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;
const shaHex = kset.shaHex;

fn initOrPrint(texts: *const Texts, pin: []const u8) !Registry {
    var diag: Diag = .{};
    return Registry.init(testing.allocator, texts, pin, &diag) catch |e| {
        std.debug.print(prefix ++ " refused: {s}\n", .{diag.message()});
        return e;
    };
}

test "glm5next nvfp4 kernels: the embedded manifest is the pinned one; both texts at GLM-5.3-Flash's instantiations, the signature the dispatch binds" {
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    for (reg.entries, 0..) |e, i| {
        try testing.expectEqual(@as(Kernel, @fromBackingInt(@intCast(i))), e.kernel);
        try testing.expect(e.checks.contains(.compile));
        try testing.expectEqualStrings(embedded.headers[0], e.header_text);
        try testing.expectEqual([3]u32{ 32, simdgroups, 1 }, e.threadgroup);
        for (instancesOf(e.kernel)) |inst| _ = try reg.instance(e.kernel, inst, null);
    }
    const gu = reg.get(.glm5n_nvfp4_gateup);
    try testing.expectEqual(@as(usize, 2 + 6 * n_banks), gu.inputs.len);
    try testing.expectEqualStrings("x", gu.inputs[0].name);
    try testing.expectEqualStrings("gate_g0", gu.inputs[3].name);
    try testing.expectEqualStrings("up_w2", gu.inputs[16].name);
    try testing.expectEqualStrings("ids", gu.inputs[gu.inputs.len - 1].name);
    try testing.expect(gu.inputs[0].dtype == null and gu.inputs[3].dtype.? == .float32 and gu.outputs[0].dtype == null);
    const dn = reg.get(.glm5n_nvfp4_down);
    try testing.expectEqual(@as(usize, 2 + 3 * n_banks), dn.inputs.len);
    try testing.expectEqualStrings("down_s1", dn.inputs[5].name);
    try testing.expectEqual(@as(usize, 6), dn.template.len);
    var diag: Diag = .{};
    try testing.expectError(error.InstanceNotRegistered, reg.instance(.glm5n_nvfp4_gateup, .{ .rows = 4, .maxm = 1, .k = 4096, .n = 2048, .gs = 16, .limit = 7 }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4_gateup is not registered at ROWS 4, MAXM 1, 2048 x 4096, GS 16, LIMIT 7") != null);
    try checkModel(4096, 2048, 8, 10.0, null);
    try testing.expectError(error.InstanceNotRegistered, checkModel(4096, 2048, 8, 7.5, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "swiglu_limit 7.5 is not the registered") != null);
    try testing.expectError(error.InstanceNotRegistered, checkModel(4096, 1536, 8, 10.0, null));
}

test "glm5next nvfp4 kernels: a tampered text or header, another pin, an unknown kernel, a reordered input or other instantiations are refused by name" {
    const a = testing.allocator;
    var diag: Diag = .{};
    {
        const bad = try a.dupeSentinel(u8, embedded.sources[@backingInt(Kernel.glm5n_nvfp4_down)], 0);
        defer a.free(bad);
        bad[bad.len / 2] ^= 0x20;
        var texts = embedded;
        texts.sources[@backingInt(Kernel.glm5n_nvfp4_down)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4_down") != null);
    }
    {
        const bad = try a.dupeSentinel(u8, embedded.headers[@backingInt(Header.glm5n_nvfp4)], 0);
        defer a.free(bad);
        bad[0] ^= 0x01;
        var texts = embedded;
        texts.headers[@backingInt(Header.glm5n_nvfp4)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4") != null);
    }
    try testing.expectError(error.ManifestNotPinned, Registry.init(a, &embedded, &(@as([64]u8, @splat('0'))), &diag));
    const Case = struct { needle: []const u8, repl: []const u8, err: Refusal, names: []const u8 };
    for ([_]Case{
        .{ .needle = "\"name\":\"glm5n_nvfp4_down\"", .repl = "\"name\":\"glm5n_nvfp4_down2\"", .err = error.UnknownKernel, .names = "glm5n_nvfp4_down2" },
        .{ .needle = "\"name\":\"glm5n_nvfp4_down\"", .repl = "\"name\":\"glm5n_nvfp4_gateup\"", .err = error.DuplicateKernel, .names = "glm5n_nvfp4_gateup listed twice" },
        .{ .needle = "\"format\":\"mlx-stream-glm5next-nvfp4-kernels-v1\"", .repl = "\"format\":\"mlx-stream-glm5next-nvfp4-kernels-v2\"", .err = error.ManifestFormat, .names = "v2" },
        .{ .needle = "{\"dtype\":\"uint32\",\"name\":\"down_w1\"},{\"dtype\":\"uint8\",\"name\":\"down_s1\"}", .repl = "{\"dtype\":\"uint8\",\"name\":\"down_s1\"},{\"dtype\":\"uint32\",\"name\":\"down_w1\"}", .err = error.SignatureMismatch, .names = "input down_s1 where the dispatch binds down_w1" },
        .{ .needle = "{\"GS\":16,\"K\":4096,\"LIMIT\":10,\"MAXM\":4,\"N\":2048,\"ROWS\":2}", .repl = "{\"GS\":16,\"K\":4096,\"LIMIT\":10,\"MAXM\":4,\"N\":2048,\"ROWS\":4}", .err = error.InstancesNotServed, .names = "glm5n_nvfp4_gateup: the manifest registers 2" },
        .{ .needle = "\"template\":[\"T\",\"ROWS\",\"I\",\"H\",\"GS\",\"MAXM\"],\"threadgroup\":[32,2,1]", .repl = "\"template\":[\"T\",\"ROWS\",\"I\",\"H\",\"GS\",\"MAXM\"],\"threadgroup\":[32,4,1]", .err = error.SchemaInvalid, .names = "glm5n_nvfp4_down: threadgroup 32x4x1" },
        .{ .needle = "\"template\":[\"T\",\"ROWS\",\"I\",\"H\",\"GS\",\"MAXM\"]", .repl = "\"template\":[\"T\",\"ROWS\",\"H\",\"I\",\"GS\",\"MAXM\"]", .err = error.SignatureMismatch, .names = "template parameter H where the launch sets I" },
    }) |c| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, embedded.manifest, c.needle));
        const m = try std.mem.replaceOwned(u8, a, embedded.manifest, c.needle, c.repl);
        defer a.free(m);
        var texts = embedded;
        texts.manifest = m;
        try testing.expectError(c.err, Registry.init(a, &texts, &shaHex(m), &diag));
        testing.expect(std.mem.indexOf(u8, diag.message(), c.names) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ c.names, diag.message() });
            return e;
        };
    }
}

test "glm5next nvfp4 kernels: every wave width launches at the plan's instantiation; the template in the texts' order" {
    try testing.expectEqual(Instance{ .rows = 4, .maxm = 1, .k = 4096, .n = 2048, .gs = 16, .limit = 10 }, instanceFor(.glm5n_nvfp4_gateup, 8));
    try testing.expectEqual(Instance{ .rows = 2, .maxm = 4, .k = 4096, .n = 2048, .gs = 16, .limit = 10 }, instanceFor(.glm5n_nvfp4_gateup, 9));
    try testing.expectEqual(Instance{ .rows = 4, .maxm = 4, .k = 2048, .n = 4096, .gs = 16, .limit = 0 }, instanceFor(.glm5n_nvfp4_down, max_pairs));
    const g = waveLaunch(.glm5n_nvfp4_gateup, 3, instanceFor(.glm5n_nvfp4_gateup, 3), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 512, 3 }, g.grid);
    try testing.expectEqual([3]u32{ 32, 2, 1 }, g.threadgroup);
    try testing.expectEqualSlices(c_int, &.{ 3, 2048 }, g.out_shapes[0][0..g.out_ranks[0]]);
    const u = waveLaunch(.glm5n_nvfp4_gateup, 17, instanceFor(.glm5n_nvfp4_gateup, 17), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 1024, 17 }, u.grid);
    const d = waveLaunch(.glm5n_nvfp4_down, 64, instanceFor(.glm5n_nvfp4_down, 64), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 1024, 64 }, d.grid);
    try testing.expectEqualSlices(c_int, &.{ 64, 4096 }, d.out_shapes[0][0..d.out_ranks[0]]);
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    var shapes: [max_outputs][]const c_int = undefined;
    var template: [16]kset.TemplateArg = undefined;
    const l = launchOf(&reg, .glm5n_nvfp4_gateup, &u, &shapes, &template);
    try testing.expectEqual(@as(usize, 7), l.template.len);
    try testing.expectEqualStrings("T", l.template[0].name);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, l.template[0].value.dtype);
    const want = [_]struct { []const u8, i32 }{ .{ "ROWS", 2 }, .{ "K", 4096 }, .{ "N", 2048 }, .{ "GS", 16 }, .{ "MAXM", 4 }, .{ "LIMIT", 10 } };
    for (want, l.template[1..]) |w, t| {
        try testing.expectEqualStrings(w[0], t.name);
        try testing.expectEqual(w[1], t.value.int);
    }
    const ld = launchOf(&reg, .glm5n_nvfp4_down, &d, &shapes, &template);
    try testing.expectEqualStrings("I", ld.template[2].name);
    try testing.expectEqual(@as(i32, 2048), ld.template[2].value.int);
    try testing.expectEqualStrings("H", ld.template[3].name);
    try testing.expectEqual(@as(i32, 4096), ld.template[3].value.int);
}
