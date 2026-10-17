//! The fused NVFP4 decode experts every NVFP4-streamed arch can install (glm5_next, beam): a pinned kernel set of two
//! texts over the expert stream's three slot banks (base, ext, transient), one launch of each per wave of P routed pairs
//! (P <= 64: up to 8 tokens of top-8), ids packed `bank << 24 | slot row`:
//!   - gate/up: gate + up of every routed pair, then the arch's SwiGLU, bf16 [P, inter];
//!   - down: each pair's down row, bf16 [P, hidden] (no score fold: the arch's combine follows).
//! The texts live under src/kernels/nvfp4/ in two families on one helper header (header_glm5n_nvfp4.metal: the slot
//! ids, the global scale's line, the E4M3 decode, the activation load, the f32 sigmoid, the union vote):
//!   - k1 (glm5n_nvfp4_*: GLM-5.3-Flash's): MiMo's k1 dot over MLX's dequantized weights, rounding-class against MLX's
//!     gather_qmm nvfp4; the clamped SwiGLU (template LIMIT);
//!   - qmv (nvfp4_qmv_*, header_nvfp4_qmv.metal: beam's): MLX's gather_qmv_fast arithmetic in its order, so each pair's
//!     row is the stock route's bit for bit; the f32 SwiGLU.
//! `KernelsOf(C)` is one arch's set. The case `C` names its kernels and headers (enums whose tags are the texts' names),
//! its manifest file and pin, the manifest's format, its routed geometry and its activation; the registry, the plan,
//! the launches, `Fused` and `install` are the same for every case. Each case has its own manifest
//! (`scripts/nvfp4_kernels_manifest.py` writes every case's and prints its pin), so one arch's instances never move
//! another's pin. `Registry.init` checks the manifest against the pin and every text against the manifest, once, and
//! refuses by name; `Registry.bind` builds the mlx fast kernels on a GPU stream, once. `install` is a module's
//! construction: the set on its stream, the compile self-check at every registered instantiation, and the launches
//! prepared per wave width (`Fused`). A wave of P <= 8 pairs (one token, every slot distinct) runs 4 rows per simdgroup
//! and one member per leader; a wider wave the union rule (a slot's pairs read it once per leader of up to 4 members),
//! per member row the one-token wave's arithmetic (MiMo's k1 plan). docs/nvfp4-decode-kernels.md is the dispatch
//! contract.
//!
//! The case `C` declares:
//!   tag: []const u8                 the log and refusal prefix ("<tag> kernels: ...", "[<tag> selfcheck]")
//!   manifest: []const u8            the manifest's file under src/kernels/nvfp4/
//!   manifest_sha256: []const u8     its pin
//!   format: []const u8              the manifest's `format`
//!   Kernel: enum                    two tags: the gate/up text's name, then the down text's (each `<tag>.metal`)
//!   Header: enum                    the headers' names (each `header_<tag>.metal`), in the order the texts include them
//!   hidden, inter, top_k: u32       the routed experts the texts are registered at
//!   activation: quant.Activation    `.swiglu_clamped` (the LIMIT template parameter) or `.swiglu_f32`

const std = @import("std");
const mlx = @import("sdk").mlx;
const log = @import("sdk").log;
const kset = @import("../sdk_ext/kernel_set.zig");
const sdk_ext = @import("../sdk_ext.zig");
const bank = @import("nvfp4_bank.zig");

const nvfp4_decode = @This();
const Allocator = std.mem.Allocator;
const Activation = sdk_ext.quant.Activation;

pub const Diag = @import("sdk").Diag;
const refuse = @import("../sdk_ext/expert/bank_file.zig").refuse;

/// `compile`: the kernel builds, binds and completes one launch over zeros at each registered instantiation (the
/// construction's acceptance). Its numerics are each case's parity tests'.
pub const Check = enum { compile };
pub const Family = enum { gate_up, down };

/// The expert stream's slot banks (`sdk_ext.expert.BankKind`: base, ext, transient), each a text input group.
pub const n_banks: u32 = std.meta.fieldNames(sdk_ext.expert.BankKind).len;
/// x, three arrays (words, scales, global scales) per projection and bank, ids: gate/up's 20, down's 11.
pub const max_inputs = 2 + 6 * n_banks;
pub const max_outputs = 1;
pub const max_template = 7;
pub const max_headers = 2;
/// The NVFP4 group: a lane's 16 codes are one E4M3 group.
pub const group_size: u32 = bank.group_size;
/// Simdgroups per threadgroup (the texts' threadgroup [32, 2, 1]).
pub const simdgroups: u32 = 2;
/// The widest wave: 8 tokens (the union vote covers a wave in two 32-lane ballots).
pub const max_tokens: u32 = 8;
/// A packed id's row field (`bank << 24 | row`).
pub const row_bits: u5 = 24;

/// A text's launch geometry: output rows per simdgroup (`ROWS`), and the widest union one leader serves (`MAXM`).
pub const Geometry = struct { rows: u32, maxm: u32 };

/// One registered instantiation: `ROWS`, `MAXM`, the reduction width `k` (K or I), the output rows `n` (N or H), `GS` and
/// the gate/up's `LIMIT` (0 for down, and for an activation without a limit).
pub const Instance = struct { rows: u32, maxm: u32, k: u32, n: u32, gs: u32, limit: u32 };

/// An argument's name and dtype; an argument of dtype "T" takes the launch's template dtype T.
pub const Arg = struct { name: [:0]const u8, dtype: ?mlx.mlx_dtype };

/// One argument of the dispatch's signature: its name and dtype (null: the launch's template dtype T).
pub const SigArg = struct { name: []const u8, dtype: ?mlx.mlx_dtype };

fn signatureLen(f: Family) usize {
    return if (f == .gate_up) 2 + 6 * n_banks else 2 + 3 * n_banks;
}

/// The signature `Fused` binds by position, which a manifest must list as it is: gate/up's x, then per bank the gate's
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

/// One projection's slot arrays as the bank scatters them (`nvfp4_bank.Arrays`): the E2M1 words [slots, out, in / 8]
/// U32, the E4M3 scales [slots, out, in / 16] U8 and the f32 global scales [>= slots, 32] (one 128-byte line per slot,
/// its word at column 0: the texts read `g[slot * 32]`, header `glm5n_g_at`), whole (the texts' row-contiguous inputs: a
/// slice would be copied per call).
pub const ProjArrays = struct { w: mlx.mlx_array, scales: mlx.mlx_array, global_scale: mlx.mlx_array };
/// One bank's three projections.
pub const BankArrays = struct { gate: ProjArrays, up: ProjArrays, down: ProjArrays };

/// The texts' directory, from this file.
const dir = "../kernels/nvfp4/";

/// An activation as the refusals print it: `swiglu_limit <l>`, `swiglu_f32` or `swiglu`.
fn activationText(act: Activation, buf: []u8) []const u8 {
    return switch (act) {
        .swiglu_clamped => |l| std.fmt.bufPrint(buf, "swiglu_limit {d}", .{l}) catch "swiglu_limit",
        else => @tagName(act),
    };
}

pub fn KernelsOf(comptime C: type) type {
    return struct {
        const Self = @This();

        /// The registry's self-check plan (`sdk_ext.kernels.KernelSet(R)` runs it; the module at its construction).
        pub const selfcheck = SelfcheckOf(Self);
        pub const manifest_sha256: []const u8 = C.manifest_sha256;
        pub const format: []const u8 = C.format;
        pub const tag: []const u8 = C.tag;
        pub const prefix = C.tag ++ " kernels";
        pub const Kernel = C.Kernel;
        pub const Header = C.Header;
        pub const Check = nvfp4_decode.Check;
        pub const Family = nvfp4_decode.Family;
        pub const Diag = nvfp4_decode.Diag;
        pub const Refusal = nvfp4_decode.Refusal;
        pub const Instance = nvfp4_decode.Instance;
        pub const LaunchConfig = nvfp4_decode.LaunchConfig;
        pub const ProjArrays = nvfp4_decode.ProjArrays;
        pub const BankArrays = nvfp4_decode.BankArrays;
        pub const n_banks = nvfp4_decode.n_banks;
        pub const max_inputs = nvfp4_decode.max_inputs;
        pub const max_outputs = nvfp4_decode.max_outputs;
        pub const max_template = nvfp4_decode.max_template;
        pub const simdgroups = nvfp4_decode.simdgroups;
        pub const group_size = nvfp4_decode.group_size;
        pub const max_tokens = nvfp4_decode.max_tokens;
        pub const row_bits = nvfp4_decode.row_bits;

        pub const n_kernels = std.meta.fieldNames(Kernel).len;
        pub const n_headers = std.meta.fieldNames(Header).len;

        /// The routed experts the texts are registered at, and their activation.
        pub const hidden: u32 = C.hidden;
        pub const inter: u32 = C.inter;
        pub const top_k: u32 = C.top_k;
        pub const activation: Activation = C.activation;
        /// The gate/up's `LIMIT` (the clamped SwiGLU's limit; 0 without one, and no LIMIT template parameter).
        pub const limit: u32 = switch (activation) {
            .swiglu_clamped => |l| @intFromFloat(l),
            else => 0,
        };
        const has_limit = activation == .swiglu_clamped;
        pub const max_pairs: u32 = max_tokens * top_k;
        /// A wave from this many pairs takes the union geometry (one token's pairs name distinct slots).
        pub const union_min_pairs: u32 = top_k + 1;

        /// The gate/up kernel and the down kernel (the two tags in order).
        pub const gate_up_kernel: Kernel = @fromBackingInt(0);
        pub const down_kernel: Kernel = @fromBackingInt(1);

        comptime {
            if (n_kernels != 2) @compileError(prefix ++ ": a set is one gate/up and one down text");
            if (n_headers > max_headers) @compileError(prefix ++ ": more headers than a kernel takes");
            if (hidden % 512 != 0 or inter % 512 != 0) @compileError(prefix ++ ": K and I must be multiples of 512 (16 values x 32 lanes)");
            if (512 % group_size != 0 or group_size != 16) @compileError(prefix ++ ": a lane's 16 codes are one E4M3 group");
            if (max_pairs > 64) @compileError(prefix ++ ": the union vote takes at most 64 pairs per wave");
            switch (activation) {
                .swiglu_clamped => |l| if (l != @as(f64, @floatFromInt(@as(u32, @intFromFloat(l))))) @compileError(prefix ++ ": LIMIT is an integer template parameter"),
                .swiglu_f32 => {},
                .swiglu => @compileError(prefix ++ ": no text implements the plain SwiGLU"),
            }
        }

        pub fn headerFile(h: Header) []const u8 {
            return switch (h) {
                inline else => |t| "header_" ++ @tagName(t) ++ ".metal",
            };
        }

        pub fn familyOf(k: Kernel) Family {
            return if (k == gate_up_kernel) .gate_up else .down;
        }

        /// MiMo's k1 plan (src/arch/mimo_v2/moe.zig `k1_plan`): a one-token wave keeps 4 rows per simdgroup and one
        /// member per leader; a wider wave the union rule, 4 members per leader, gate/up at 2 rows (its decoded codes of
        /// two projections) and down at 4.
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

        /// The instantiations this build serves, which the manifest must register as they are: `plan`'s geometries at
        /// the case's shapes.
        pub const gate_up_instances = [_]Instance{
            .{ .rows = plan.serial_gate_up.rows, .maxm = plan.serial_gate_up.maxm, .k = hidden, .n = inter, .gs = group_size, .limit = limit },
            .{ .rows = plan.union_gate_up.rows, .maxm = plan.union_gate_up.maxm, .k = hidden, .n = inter, .gs = group_size, .limit = limit },
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
        pub fn checkModel(hidden_size: u32, moe_intermediate_size: u32, experts_per_token: u32, act: Activation, diag: ?*Diag) Refusal!void {
            if (hidden_size == hidden and moe_intermediate_size == inter and experts_per_token == top_k and std.meta.eql(act, activation)) return;
            var b0: [48]u8 = undefined;
            var b1: [48]u8 = undefined;
            return refuse(diag, error.InstanceNotRegistered, prefix ++ ": hidden {d} / inter {d} / top-{d} / {s} is not the registered {d} / {d} / top-{d} / {s}", .{ hidden_size, moe_intermediate_size, experts_per_token, activationText(act, &b0), hidden, inter, top_k, activationText(activation, &b1) });
        }

        /// The template parameters in the texts' order (after T: the ints `LaunchConfig.ints` carries at the same
        /// positions); gate/up's LIMIT only under a clamped SwiGLU.
        pub fn templateNames(f: Family) []const []const u8 {
            return switch (f) {
                .gate_up => if (has_limit) &.{ "T", "ROWS", "K", "N", "GS", "MAXM", "LIMIT" } else &.{ "T", "ROWS", "K", "N", "GS", "MAXM" },
                .down => &.{ "T", "ROWS", "I", "H", "GS", "MAXM" },
            };
        }

        // ── The texts and the manifest ──

        pub const Texts = struct {
            manifest: []const u8,
            sources: [n_kernels][:0]const u8,
            headers: [n_headers][:0]const u8,
        };

        pub const embedded: Texts = .{
            .manifest = @embedFile(dir ++ C.manifest),
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
                if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, prefix ++ ": format \"{s}\" is not {s}", .{ m.format, format });
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

        /// The launch of `k` over a wave of `pairs` routed pairs at instantiation `inst`: grid (32, n / ROWS, P),
        /// threadgroup (32, simdgroups, 1), the template `T` = `dt`, then ROWS, K|I, N|H, GS, MAXM (and gate/up's LIMIT
        /// under a clamped SwiGLU); the output [P, n] in `dt`.
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
        pub const Bound = kset.BoundOf(Self);

        /// `cfg` as the mlx launch (`Bound.prepare`): the entry's template parameters in its order, `T` the config's dtype.
        pub fn launchOf(reg: *const Registry, k: Kernel, cfg: *const LaunchConfig, shapes: *[max_outputs][]const c_int, template: *[16]kset.TemplateArg) kset.Launch {
            const e = reg.get(k);
            for (e.template, template[0..e.template.len], 0..) |name, *t, i| t.* = .{ .name = name, .value = if (std.mem.eql(u8, name, "T")) .{ .dtype = cfg.dtype } else .{ .int = cfg.ints[i] } };
            for (0..cfg.n_out) |i| shapes[i] = cfg.out_shapes[i][0..cfg.out_ranks[i]];
            return .{ .grid = cfg.grid, .threadgroup = cfg.threadgroup, .out_shapes = shapes[0..cfg.n_out], .out_dtypes = cfg.out_dtypes[0..cfg.n_out], .template = template[0..e.template.len] };
        }

        /// The load's kernel set over this registry (`sdk_ext.kernels.KernelSet`).
        pub const Set = kset.KernelSet(Self).Set;

        /// Device tests: the set on GPU stream `s` (a refusal printed by name).
        pub fn openForTest(a: Allocator, s: mlx.mlx_stream) !*Set {
            var diag: Diag = .{};
            return Set.init(a, .{ .device = .{ .stream = s } }, &diag) catch |e| {
                std.debug.print(prefix ++ " refused: {s}\n", .{diag.message()});
                return e;
            };
        }

        // ── The module's install and dispatch ──

        /// The launches of both texts prepared per wave width (P = 1 .. `max_pairs`), on a bound set: the module's
        /// decode calls them with no per-call config work.
        pub const Fused = struct {
            bound: *const Bound,
            gate_up_launch: [max_pairs]Prepared = undefined,
            down_launch: [max_pairs]Prepared = undefined,
            n_prepared: u32 = 0,

            /// Every wave width's two launches (bf16 rows), each at the instantiation `plan` names for it, which the
            /// registry must register (refused by name otherwise).
            pub fn init(bound: *const Bound, diag: ?*Diag) !Fused {
                var f: Fused = .{ .bound = bound };
                errdefer f.deinit();
                for (1..max_pairs + 1) |p| {
                    const pairs: u32 = @intCast(p);
                    const gi = try bound.reg.instance(gate_up_kernel, instanceFor(gate_up_kernel, pairs), diag);
                    const di = try bound.reg.instance(down_kernel, instanceFor(down_kernel, pairs), diag);
                    const gc = waveLaunch(gate_up_kernel, pairs, gi, .bfloat16);
                    const dc = waveLaunch(down_kernel, pairs, di, .bfloat16);
                    f.gate_up_launch[p - 1] = try bound.prepare(gate_up_kernel, &gc);
                    f.down_launch[p - 1] = bound.prepare(down_kernel, &dc) catch |e| {
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

            /// `x` bf16 [P, hidden] (each pair's token row), `ids` u32 [P] packed `bank << 24 | slot row`; returns bf16
            /// [P, inter] (the SwiGLU rows; a new array, the caller frees it).
            pub fn gateUp(f: *const Fused, x: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [n_banks]BankArrays) !mlx.mlx_array {
                const p = pairsOf(ids) orelse return error.Nvfp4WaveWidth;
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

            /// `h` bf16 [P, inter], `ids` as `gateUp`; returns bf16 [P, hidden]: each pair's down row (a new array, the
            /// caller frees it).
            pub fn down(f: *const Fused, h: mlx.mlx_array, ids: mlx.mlx_array, banks: *const [n_banks]BankArrays) !mlx.mlx_array {
                const p = pairsOf(ids) orelse return error.Nvfp4WaveWidth;
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

        /// What the module holds once installed: the set (pinned, both kernels built on its stream, the compile check
        /// passed at every registered instantiation) and the prepared launches. `deinit` after the stream has drained.
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
        /// construction's self-check (`.compile`: one launch over zeros of each kernel at each registered instantiation,
        /// its ids in both of MLX's passings), and `Fused`. Refused by name (`diag`): a registry refusal, NotGpuStream /
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
    };
}

/// The self-check of a set `xk` (`KernelsOf`), as `sdk_ext.kernels.KernelSet` runs it and a module at its construction
/// (`install`): `compile`, the kernel builds, binds and completes one launch over zeros at every instantiation the
/// manifest registers, at the wave widths that reach each of MLX's two passings of `ids` (a one-token instantiation at 1
/// pair, `constant`, and at top-k, `device`; a union one at top-k + 1 and at the widest wave), with the banks' global
/// scales at 8 slots (`device`, as a module's banks pass them). No numeric probe: the numerics are each case's parity
/// tests'.
pub fn SelfcheckOf(comptime xk: type) type {
    return struct {
        const Kernel = xk.Kernel;
        const scaffold = kset.Checks(Kernel, Check);
        pub const Result = scaffold.Result;
        pub const Report = scaffold.Report;
        pub const Depth = kset.CheckDepth;
        const Scope = kset.Scope;
        const log_tag = "[" ++ xk.tag ++ " selfcheck]";

        /// The checks of `want`'s entries at `depth`, per kernel.
        pub fn plan(reg: *const xk.Registry, want: std.EnumSet(Kernel), depth: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
            return scaffold.plan(&reg.entries, want, depth);
        }

        /// The wave widths an instantiation's compile check launches: the `constant` and the `device` passing of `ids`.
        pub fn checkWidths(inst: Instance) []const u32 {
            return if (inst.maxm == 1) &.{ 1, xk.top_k } else &.{ xk.union_min_pairs, xk.max_pairs };
        }

        /// The plan over `subset` at `depth` on the bound kernels: each kernel at every instantiation the manifest
        /// registers and each of its check widths, one result per launch.
        pub fn runSubset(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, subset: []const Kernel, depth: Depth, report: *Report) !void {
            var want: std.EnumSet(Kernel) = .empty;
            for (subset) |k| want.insert(k);
            const checks = plan(reg, want, depth);
            for (&reg.entries) |*e| {
                if (!want.contains(e.kernel) or !checks.get(e.kernel).contains(.compile)) continue;
                for (e.instances) |inst| for (checkWidths(inst)) |pairs| try checkOne(a, bound, e.kernel, inst, pairs, report);
            }
        }

        /// The verdict over a finished plan: refused (error.SelfCheckFailed, `diag` naming the first failure) unless
        /// every result passed and the plan ran.
        pub fn judge(report: *const Report, diag: ?*Diag) error{SelfCheckFailed}!void {
            return scaffold.judge(report, xk.prefix, diag);
        }

        /// One `compile` result of `k` at `inst` over `pairs` pairs: a raised launch is recorded with its error and the
        /// latched MLX message.
        fn checkOne(a: Allocator, bound: *const xk.Bound, k: Kernel, inst: Instance, pairs: u32, report: *Report) !void {
            launchZeros(a, bound, k, inst, pairs) catch |e| {
                if (e == error.OutOfMemory) return error.OutOfMemory;
                var buf: [512]u8 = undefined;
                const m = mlx.takeError(&buf) orelse "";
                log.err(log_tag ++ " {t} at ROWS {d}, MAXM {d}, {d} pairs: {s} {s}\n", .{ k, inst.rows, inst.maxm, pairs, @errorName(e), m });
                return report.appendRaised(a, k, .compile, e, m);
            };
            try report.results.append(a, .{ .kernel = k, .check = .compile, .words = pairs, .ok = true });
        }

        /// Slots of a check bank's global scales: 8 lines of 32 f32 (`nvfp4_bank.scale_line_floats`; MLX's `device`
        /// passing, as a module's line-isolated banks).
        const check_scale_slots = 8;
        const check_scale_floats = check_scale_slots * bank.scale_line_floats;

        /// One launch of `k` at `inst` over `pairs` pairs of zeros, evaluated: each bank one slot of zero words and
        /// scales (the same arrays in all three banks), its global scales `check_scale_slots` zeros, every id slot 0 of
        /// bank 0.
        fn launchZeros(a: Allocator, bound: *const xk.Bound, k: Kernel, inst: Instance, pairs: u32) !void {
            var sc: Scope = .{ .a = a };
            defer sc.deinit();
            const rows: usize = inst.n;
            const cols: usize = inst.k;
            const xz = try a.alloc(u16, pairs * cols);
            defer a.free(xz);
            @memset(xz, 0);
            const wz = try a.alloc(u32, rows * cols / 8);
            defer a.free(wz);
            @memset(wz, 0);
            const sz = try a.alloc(u8, rows * cols / inst.gs);
            defer a.free(sz);
            @memset(sz, 0);
            const gz: [check_scale_floats]f32 = @splat(0);
            const idz = try a.alloc(u32, pairs);
            defer a.free(idz);
            @memset(idz, 0);
            const x = try sc.host(std.mem.sliceAsBytes(xz), &.{ @intCast(pairs), @intCast(cols) }, .bfloat16);
            const w = try sc.host(std.mem.sliceAsBytes(wz), &.{ 1, @intCast(rows), @intCast(cols / 8) }, .uint32);
            const s = try sc.host(sz, &.{ 1, @intCast(rows), @intCast(cols / inst.gs) }, .uint8);
            const g = try sc.host(std.mem.sliceAsBytes(&gz), &.{ check_scale_slots, bank.scale_line_floats }, .float32);
            const ids = try sc.host(std.mem.sliceAsBytes(idz), &.{@intCast(pairs)}, .uint32);
            var inputs: [max_inputs]mlx.mlx_array = undefined;
            var n: usize = 0;
            inputs[n] = x;
            n += 1;
            const per: usize = if (xk.familyOf(k) == .gate_up) 2 else 1;
            for (0..n_banks * per) |_| {
                inputs[n..][0..3].* = .{ w, s, g };
                n += 3;
            }
            inputs[n] = ids;
            n += 1;
            const cfg = xk.waveLaunch(k, pairs, inst, .bfloat16);
            var outs: [max_outputs]mlx.mlx_array = undefined;
            try bound.apply(k, inputs[0..n], &cfg, &outs);
            _ = try sc.keep(outs[0]);
            try mlx.check(mlx.mlx_array_eval(outs[0]));
        }
    };
}
