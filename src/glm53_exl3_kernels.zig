//! The pinned Metal kernel registry of GLM-5.3's EXL3 routed experts (the `glm53_exl3` quant's): the texts under
//! kernels/glm53_exl3/ with a manifest of their signatures, template parameters, threadgroups and registered
//! instantiations (K 3 and 4 at GLM-5.3's dims, hidden 6144 and a mini's intermediate 512, and the tiny fixture's).
//! `Registry.init` checks the manifest against `manifest_sha256` and every text against the manifest, once, and refuses
//! by name; `Registry.bind` builds the mlx fast kernels on a GPU stream, once. The set is `sdk_ext.kernels.KernelSet`'s
//! (the load context owns it; the quant self-checks its kernels at accept).

const std = @import("std");
const mlx = @import("sdk").mlx;
/// The registry's self-check plan (`sdk_ext.kernels.KernelSet(R)` runs it at the quant's accept).
pub const selfcheck = @import("glm53_exl3_selfcheck.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;

/// sha256 of kernels/glm53_exl3/manifest.json (scripts/glm53_exl3_manifest.py prints it): pins every text.
pub const manifest_sha256 = "74bde426a9baa188bb66a375cf05963e6d60cf4ce72f9ac74e5ec39bd469f152";
pub const format = "mlx-stream-glm53-exl3-kernels-v1";
const dir = "kernels/glm53_exl3/";

/// The bank these texts decode: EXL3 codebook mcg at sushi's multiplier, window 16, K 3 and 4, four minis per expert.
pub const bank_codebook = "mcg";
pub const bank_multiplier: u64 = 0xCBAC1FED;
pub const bank_window: u64 = 16;
pub const bank_ks = [_]u32{ 3, 4 };
pub const tp_ranks: u32 = 4;

/// Every kernel of the registry; the tag is the kernel's MLX name and its file name.
pub const Kernel = enum {
    glm53_dec_prep,
    glm53_dec_gu,
    glm53_dec_down,
    glm53_dec_combine,
    glm53_pf_gu,
    glm53_pf_mid,
    glm53_pf_down,
};

pub const Header = enum { glm53_dec, glm53_nax };
/// One text of the registry: a kernel's or a header's.
pub const TextRef = union(enum) { kernel: Kernel, header: Header };
/// `compile`: the kernel builds, binds and completes one launch (the construction's acceptance); `chain`: its stage
/// chain's words against sushi's host decode of the same records (the device test).
pub const Check = enum { compile, chain };
pub const Family = enum { decode, prompt };

pub const n_kernels = std.meta.fieldNames(Kernel).len;
pub const n_headers = std.meta.fieldNames(Header).len;
pub const max_inputs = 20;
pub const max_outputs = 2;
pub const max_template = 5;
pub const max_rank = 4;

pub const Texts = struct {
    manifest: []const u8,
    sources: [n_kernels][:0]const u8,
    headers: [n_headers][:0]const u8,
};

pub const embedded: Texts = .{
    .manifest = @embedFile(dir ++ "manifest.json"),
    .sources = embedAll(Kernel, ""),
    .headers = embedAll(Header, "header_"),
};

fn embedAll(comptime E: type, comptime prefix: []const u8) [std.meta.fieldNames(E).len][:0]const u8 {
    const names = std.meta.fieldNames(E);
    var out: [names.len][:0]const u8 = undefined;
    inline for (names, 0..) |name, i| out[i] = @embedFile(dir ++ prefix ++ name ++ ".metal");
    return out;
}

pub const Refusal = error{
    ManifestNotPinned,
    ManifestSyntax,
    ManifestFormat,
    BankNotImplemented,
    UnknownKernel,
    MissingKernel,
    DuplicateKernel,
    UnknownHeader,
    MissingHeader,
    TextSha256Mismatch,
    SchemaInvalid,
    InstanceNotRegistered,
    NotGpuStream,
    KernelCreateFailed,
};

pub const Diag = @import("sdk").Diag;

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// An argument's name and dtype; an output of dtype "T" takes the launch's template dtype T.
pub const Arg = struct { name: [:0]const u8, dtype: ?mlx.mlx_dtype };

/// One registered instantiation: K (0 for a kernel without K) and the dims.
pub const Instance = struct { k: u32 = 0, hidden: u32, mini: u32 };

pub const Entry = struct {
    kernel: Kernel,
    family: Family,
    source: [:0]const u8,
    header: Header,
    text_sha256: [32]u8,
    inputs: []const Arg,
    outputs: []const Arg,
    template: []const [:0]const u8,
    threadgroup: [3]u32,
    instances: []const Instance,
    checks: std.EnumSet(Check),
    /// whether the kernel is instantiated per K
    per_k: bool,

    pub fn registered(e: *const Entry, inst: Instance) bool {
        for (e.instances) |i| if (std.meta.eql(i, inst)) return true;
        return false;
    }
};

pub const Registry = struct {
    arena: std.heap.ArenaAllocator,
    entries: [n_kernels]Entry,
    headers: [n_headers][:0]const u8,
    /// the dims the texts are registered at (the manifest's `dims`, in order: the served dims first)
    dims: []const Instance,
    maxm: u32,
    manifest_hex: [64]u8,

    /// Parses the manifest, checks it against `pin` (hex sha256) and every text against it, once.
    pub fn init(gpa: Allocator, texts: *const Texts, pin: []const u8, diag: ?*Diag) (Refusal || Allocator.Error)!Registry {
        var d: [32]u8 = undefined;
        Sha256.hash(texts.manifest, &d, .{});
        const got = std.fmt.bytesToHex(d, .lower);
        if (!std.mem.eql(u8, &got, pin)) return refuse(diag, error.ManifestNotPinned, "glm53 exl3 kernels: manifest sha256 {s} is not the pinned {s}", .{ &got, pin });
        var reg: Registry = .{ .arena = .init(gpa), .entries = undefined, .headers = texts.headers, .dims = &.{}, .maxm = 0, .manifest_hex = got };
        errdefer reg.arena.deinit();
        const a = reg.arena.allocator();
        const m = std.json.parseFromSliceLeaky(JManifest, a, texts.manifest, .{ .allocate = .alloc_always }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ManifestSyntax, "glm53 exl3 kernels: manifest does not parse ({t})", .{e}),
        };
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, "glm53 exl3 kernels: format \"{s}\" is not " ++ format, .{m.format});
        const b = m.bank;
        if (!std.mem.eql(u8, b.codebook, bank_codebook) or b.multiplier != bank_multiplier or b.window != bank_window or !std.mem.eql(u32, b.K, &bank_ks) or b.tp_ranks != tp_ranks)
            return refuse(diag, error.BankNotImplemented, "glm53 exl3 kernels: the manifest's bank ({s} at {d}, window {d}, K {any}, {d} ranks) is not the mcg K 3 / 4 bank these texts decode", .{ b.codebook, b.multiplier, b.window, b.K, b.tp_ranks });
        if (m.dims.len == 0 or m.maxm < 1) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {d} dims, maxm {d}", .{ m.dims.len, m.maxm });
        const dims = try a.alloc(Instance, m.dims.len);
        for (m.dims, dims) |jd, *o| {
            if (jd.hidden == 0 or jd.hidden % 128 != 0 or jd.mini == 0 or jd.mini % 128 != 0) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: dims {d} / {d} (multiples of 128)", .{ jd.hidden, jd.mini });
            o.* = .{ .hidden = jd.hidden, .mini = jd.mini };
        }
        reg.dims = dims;
        reg.maxm = m.maxm;
        var seen_h: std.EnumSet(Header) = .empty;
        for (m.headers) |h| {
            const id = std.meta.stringToEnum(Header, h.id) orelse return refuse(diag, error.UnknownHeader, "glm53 exl3 kernels: manifest header \"{s}\" is not one this build embeds", .{h.id});
            if (seen_h.contains(id)) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: header {t} listed twice", .{id});
            seen_h.insert(id);
            try checkText(texts.headers[@backingInt(id)], h.file, h.sha256, h.bytes, @tagName(id), diag);
        }
        if (seen_h.count() != n_headers) return refuse(diag, error.MissingHeader, "glm53 exl3 kernels: the manifest lists {d} of {d} headers", .{ seen_h.count(), n_headers });
        var seen: std.EnumSet(Kernel) = .empty;
        for (m.kernels) |jk| {
            const k = std.meta.stringToEnum(Kernel, jk.name) orelse return refuse(diag, error.UnknownKernel, "glm53 exl3 kernels: manifest kernel \"{s}\" is not one this build implements", .{jk.name});
            if (seen.contains(k)) return refuse(diag, error.DuplicateKernel, "glm53 exl3 kernels: {t} listed twice", .{k});
            seen.insert(k);
            reg.entries[@backingInt(k)] = try adoptKernel(a, texts, k, jk, dims, diag);
        }
        if (seen.count() != n_kernels) {
            var it = seen.complement().iterator();
            return refuse(diag, error.MissingKernel, "glm53 exl3 kernels: the manifest does not list {t} ({d} of {d})", .{ it.next().?, seen.count(), n_kernels });
        }
        return reg;
    }

    pub fn deinit(self: *Registry) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const Registry, k: Kernel) *const Entry {
        return &self.entries[@backingInt(k)];
    }

    /// The instantiation of `k` at `inst` (K ignored for a kernel without K), refused by name when the manifest does
    /// not register it.
    pub fn instance(self: *const Registry, k: Kernel, inst: Instance, diag: ?*Diag) Refusal!Instance {
        const e = self.get(k);
        const want: Instance = .{ .k = if (e.per_k) inst.k else 0, .hidden = inst.hidden, .mini = inst.mini };
        if (!e.registered(want)) return refuse(diag, error.InstanceNotRegistered, "glm53 exl3 kernels: {t} is not registered at K {d}, hidden {d}, mini {d}", .{ k, want.k, want.hidden, want.mini });
        return want;
    }

    /// Builds every kernel's mlx object on `stream`, once; the objects reach Metal at their first launch.
    pub fn bind(self: *const Registry, stream: mlx.mlx_stream, diag: ?*Diag) Refusal!Bound {
        if (!mlx.streamIsGpu(stream)) return refuse(diag, error.NotGpuStream, "glm53 exl3 kernels: bind needs a GPU stream", .{});
        var b: Bound = .{ .reg = self, .stream = stream, .kernels = @splat(.{}) };
        errdefer b.deinit();
        for (&self.entries, 0..) |*e, i| {
            var in_names: [max_inputs][*:0]const u8 = undefined;
            var out_names: [max_outputs][*:0]const u8 = undefined;
            for (e.inputs, 0..) |arg, n| in_names[n] = arg.name.ptr;
            for (e.outputs, 0..) |arg, n| out_names[n] = arg.name.ptr;
            const vin = mlx.mlx_vector_string_new_data(&in_names, e.inputs.len);
            defer _ = mlx.mlx_vector_string_free(vin);
            const vout = mlx.mlx_vector_string_new_data(&out_names, e.outputs.len);
            defer _ = mlx.mlx_vector_string_free(vout);
            b.kernels[i] = mlx.mlx_fast_metal_kernel_new(@tagName(e.kernel), vin, vout, e.source.ptr, self.headers[@backingInt(e.header)].ptr, true, false);
            if (b.kernels[i].ctx == null) return refuse(diag, error.KernelCreateFailed, "glm53 exl3 kernels: mlx_fast_metal_kernel_new({t}) failed", .{e.kernel});
        }
        return b;
    }
};

/// One launch's geometry: the grid (threads), the entry's threadgroup unless overridden, the template values in the
/// entry's order (ints; `T` a dtype), the outputs' shapes and dtypes.
pub const LaunchConfig = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    ints: [max_template]i32 = @splat(0),
    dtype: mlx.mlx_dtype = .float32,
    n_out: usize,
    out_ranks: [max_outputs]usize = @splat(0),
    out_shapes: [max_outputs][max_rank]c_int = @splat(@splat(0)),
    out_dtypes: [max_outputs]mlx.mlx_dtype = @splat(.float32),
};

/// A launch's mlx config built once (`Bound.prepare`) and handed to every launch at that geometry.
pub const Prepared = struct {
    kernel: Kernel,
    config: mlx.mlx_fast_metal_kernel_config,
    n_out: usize,

    pub fn deinit(self: *Prepared) void {
        _ = mlx.mlx_fast_metal_kernel_config_free(self.config);
        self.* = undefined;
    }
};

/// The kernels built on one GPU stream; freed after that stream has drained.
pub const Bound = struct {
    reg: *const Registry,
    stream: mlx.mlx_stream,
    kernels: [n_kernels]mlx.mlx_fast_metal_kernel,

    pub fn deinit(self: *Bound) void {
        for (self.kernels) |k| if (k.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_free(k);
        };
        self.* = undefined;
    }

    /// The profile hook's launch probe: these kernels report no launch keys.
    pub fn observe(self: *Bound, comptime L: type) void {
        _ = .{ self, L };
    }

    pub fn prepare(self: *const Bound, k: Kernel, cfg: *const LaunchConfig) error{MlxError}!Prepared {
        const e = self.reg.get(k);
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        for (0..cfg.n_out) |i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &cfg.out_shapes[i], cfg.out_ranks[i], cfg.out_dtypes[i]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(cfg.grid[0]), @intCast(cfg.grid[1]), @intCast(cfg.grid[2])));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(cfg.threadgroup[0]), @intCast(cfg.threadgroup[1]), @intCast(cfg.threadgroup[2])));
        for (e.template, 0..) |name, i| {
            if (std.mem.eql(u8, name, "T")) {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, name.ptr, cfg.dtype));
            } else try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name.ptr, cfg.ints[i]));
        }
        return .{ .kernel = k, .config = c, .n_out = cfg.n_out };
    }

    /// One launch of a prepared config; `outs` receives `p.n_out` new arrays (caller frees).
    pub fn applyPrepared(self: *const Bound, p: *const Prepared, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) error{MlxError}!void {
        std.debug.assert(inputs.len == self.reg.get(p.kernel).inputs.len);
        const vin = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = mlx.mlx_vector_array_free(vin);
        var vout = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vout);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&vout, self.kernels[@backingInt(p.kernel)], vin, p.config, self.stream));
        for (outs[0..p.n_out], 0..) |*o, i| {
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(o, vout, i));
        }
    }

    /// One launch with a config built for it alone (the prompt's per-wave grids).
    pub fn apply(self: *const Bound, k: Kernel, inputs: []const mlx.mlx_array, cfg: *const LaunchConfig, outs: []mlx.mlx_array) error{MlxError}!void {
        var p = try self.prepare(k, cfg);
        defer p.deinit();
        try self.applyPrepared(&p, inputs, outs);
    }
};

// ── Manifest adoption (once, at init) ──

const JText = struct { id: []const u8 = "", file: []const u8, sha256: []const u8, bytes: u64 };
const JArg = struct { name: []const u8, dtype: []const u8 };
const JInstance = struct { K: u32 = 0, hidden: u32, mini: u32 };
const JKernel = struct {
    name: []const u8,
    family: []const u8,
    source: JText,
    header: []const u8,
    inputs: []const JArg,
    outputs: []const JArg,
    template: []const []const u8,
    threadgroup: [3]u32,
    instances: []const JInstance,
    checks: []const []const u8,
};
const JBank = struct { codebook: []const u8, multiplier: u64, window: u64, K: []const u32, tp_ranks: u32 };
const JDims = struct { hidden: u32, mini: u32 };
const JManifest = struct { format: []const u8, bank: JBank, dims: []const JDims, maxm: u32, headers: []const JText, kernels: []const JKernel };

fn hexSha(s: []const u8) ?[32]u8 {
    if (s.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch return null;
    return out;
}

fn checkText(text: []const u8, file: []const u8, sha: []const u8, bytes: u64, what: []const u8, diag: ?*Diag) Refusal!void {
    const want = hexSha(sha) orelse return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {s}: sha256 \"{s}\" is not 64 hex", .{ what, sha });
    var d: [32]u8 = undefined;
    Sha256.hash(text, &d, .{});
    if (!std.mem.eql(u8, &d, &want) or text.len != bytes)
        return refuse(diag, error.TextSha256Mismatch, "glm53 exl3 kernels: {s} ({s}, {d} B) differs from the manifest ({d} B, sha256 {s})", .{ what, file, text.len, bytes, sha });
}

fn parseDtype(s: []const u8, k: Kernel, diag: ?*Diag) Refusal!?mlx.mlx_dtype {
    if (std.mem.eql(u8, s, "T")) return null;
    return std.meta.stringToEnum(mlx.mlx_dtype, s) orelse refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: unknown dtype \"{s}\"", .{ k, s });
}

fn adoptArgs(a: Allocator, js: []const JArg, max: usize, k: Kernel, diag: ?*Diag) (Refusal || Allocator.Error)![]const Arg {
    if (js.len == 0 or js.len > max) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: {d} arguments", .{ k, js.len });
    const out = try a.alloc(Arg, js.len);
    for (js, out, 0..) |j, *o, i| {
        for (js[0..i]) |prev| if (std.mem.eql(u8, prev.name, j.name)) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: argument {s} twice", .{ k, j.name });
        o.* = .{ .name = try a.dupeSentinel(u8, j.name, 0), .dtype = try parseDtype(j.dtype, k, diag) };
    }
    return out;
}

fn adoptKernel(a: Allocator, texts: *const Texts, k: Kernel, j: JKernel, dims: []const Instance, diag: ?*Diag) (Refusal || Allocator.Error)!Entry {
    const source = texts.sources[@backingInt(k)];
    try checkText(source, j.source.file, j.source.sha256, j.source.bytes, @tagName(k), diag);
    const header = std.meta.stringToEnum(Header, j.header) orelse return refuse(diag, error.UnknownHeader, "glm53 exl3 kernels: {t}: header \"{s}\"", .{ k, j.header });
    const family = std.meta.stringToEnum(Family, j.family) orelse return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: family \"{s}\"", .{ k, j.family });
    const tg = j.threadgroup;
    if (tg[0] == 0 or tg[1] == 0 or tg[2] == 0 or @as(u64, tg[0]) * tg[1] * tg[2] > 1024) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: threadgroup {d}x{d}x{d}", .{ k, tg[0], tg[1], tg[2] });
    if (j.template.len == 0 or j.template.len > max_template) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: {d} template parameters", .{ k, j.template.len });
    const template = try a.alloc([:0]const u8, j.template.len);
    var per_k = false;
    for (j.template, template) |t, *o| {
        o.* = try a.dupeSentinel(u8, t, 0);
        per_k = per_k or std.mem.eql(u8, t, "K");
    }
    const inst = try a.alloc(Instance, j.instances.len);
    for (j.instances, inst) |ji, *o| {
        const k_ok = if (per_k) std.mem.indexOfScalar(u32, &bank_ks, ji.K) != null else ji.K == 0;
        var dims_ok = false;
        for (dims) |d| dims_ok = dims_ok or (d.hidden == ji.hidden and d.mini == ji.mini);
        if (!k_ok or !dims_ok) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: instance K {d} at {d} / {d}", .{ k, ji.K, ji.hidden, ji.mini });
        o.* = .{ .k = ji.K, .hidden = ji.hidden, .mini = ji.mini };
    }
    if (inst.len == 0) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: no instance", .{k});
    var checks: std.EnumSet(Check) = .empty;
    for (j.checks) |c| checks.insert(std.meta.stringToEnum(Check, c) orelse return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: check \"{s}\"", .{ k, c }));
    if (!checks.contains(.compile)) return refuse(diag, error.SchemaInvalid, "glm53 exl3 kernels: {t}: no compile check", .{k});
    return .{
        .kernel = k,
        .family = family,
        .source = source,
        .header = header,
        .text_sha256 = hexSha(j.source.sha256).?,
        .inputs = try adoptArgs(a, j.inputs, max_inputs, k, diag),
        .outputs = try adoptArgs(a, j.outputs, max_outputs, k, diag),
        .template = template,
        .threadgroup = tg,
        .instances = inst,
        .checks = checks,
        .per_k = per_k,
    };
}

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;

fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// `texts` with `which`'s text replaced by `text` and the manifest re-pinned to it (caller frees the returned manifest):
/// a registry built over a mutated text (the self-check's mutation tests).
pub fn retexted(a: Allocator, base: *const Texts, which: TextRef, text: [:0]const u8, out: *Texts) ![]u8 {
    out.* = base.*;
    const old = switch (which) {
        .kernel => |k| base.sources[@backingInt(k)],
        .header => |h| base.headers[@backingInt(h)],
    };
    switch (which) {
        .kernel => |k| out.sources[@backingInt(k)] = text,
        .header => |h| out.headers[@backingInt(h)] = text,
    }
    var ob: [32]u8 = undefined;
    const old_rec = try std.fmt.bufPrint(&ob, "\"bytes\":{d}", .{old.len});
    var nb: [32]u8 = undefined;
    const new_rec = try std.fmt.bufPrint(&nb, "\"bytes\":{d}", .{text.len});
    const old_hex = shaHex(old);
    const new_hex = shaHex(text);
    // every record of the text: its bytes then its file, then its sha256
    const m1 = try std.mem.replaceOwned(u8, a, base.manifest, &old_hex, &new_hex);
    defer a.free(m1);
    // the bytes field sits right before the record's "file": replace it only next to the old text's file name
    const file = switch (which) {
        .kernel => |k| @tagName(k),
        .header => |h| @tagName(h),
    };
    var fb: [96]u8 = undefined;
    const pat_old = try std.fmt.bufPrint(&fb, "{s},\"file\":\"{s}{s}.metal\"", .{ old_rec, if (which == .header) "header_" else "", file });
    var gb: [96]u8 = undefined;
    const pat_new = try std.fmt.bufPrint(&gb, "{s},\"file\":\"{s}{s}.metal\"", .{ new_rec, if (which == .header) "header_" else "", file });
    return std.mem.replaceOwned(u8, a, m1, pat_old, pat_new);
}

fn initOrPrint(texts: *const Texts, pin: []const u8) !Registry {
    var diag: Diag = .{};
    return Registry.init(testing.allocator, texts, pin, &diag) catch |e| {
        std.debug.print("glm53 exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
}

test "glm53 exl3 kernels: the embedded manifest is the pinned one; every kernel is registered at GLM-5.3's dims and the fixture's, both Ks" {
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 7), n_kernels);
    try testing.expectEqual(@as(u32, 6144), reg.dims[0].hidden);
    try testing.expectEqual(@as(u32, 512), reg.dims[0].mini);
    for (reg.entries, 0..) |e, i| {
        try testing.expectEqual(@as(Kernel, @fromBackingInt(@intCast(i))), e.kernel);
        for (reg.dims) |d| for (bank_ks) |k| {
            _ = try reg.instance(e.kernel, .{ .k = k, .hidden = d.hidden, .mini = d.mini }, null);
        };
        try testing.expect(e.checks.contains(.compile) and e.checks.contains(.chain));
    }
    try testing.expectEqual(Header.glm53_nax, reg.get(.glm53_pf_gu).header);
    try testing.expectEqual(Family.prompt, reg.get(.glm53_pf_down).family);
    try testing.expect(reg.get(.glm53_dec_gu).per_k and !reg.get(.glm53_dec_prep).per_k);
    // the down stage binds every bank kind's five arrays: 19 inputs
    try testing.expectEqual(@as(usize, 19), reg.get(.glm53_dec_down).inputs.len);
    var diag: Diag = .{};
    try testing.expectError(error.InstanceNotRegistered, reg.instance(.glm53_dec_gu, .{ .k = 3, .hidden = 5120, .mini = 576 }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "glm53_dec_gu is not registered at K 3, hidden 5120, mini 576") != null);
    try testing.expectError(error.InstanceNotRegistered, reg.instance(.glm53_pf_gu, .{ .k = 2, .hidden = 6144, .mini = 512 }, &diag));
}

test "glm53 exl3 kernels: a tampered text or header, another pin, another bank or an unknown kernel is refused by name" {
    const a = testing.allocator;
    var diag: Diag = .{};
    {
        const bad = try a.dupeSentinel(u8, embedded.sources[@backingInt(Kernel.glm53_dec_down)], 0);
        defer a.free(bad);
        bad[bad.len / 2] ^= 0x20;
        var texts = embedded;
        texts.sources[@backingInt(Kernel.glm53_dec_down)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm53_dec_down") != null);
    }
    {
        const bad = try a.dupeSentinel(u8, embedded.headers[@backingInt(Header.glm53_nax)], 0);
        defer a.free(bad);
        bad[0] ^= 0x01;
        var texts = embedded;
        texts.headers[@backingInt(Header.glm53_nax)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm53_nax") != null);
    }
    try testing.expectError(error.ManifestNotPinned, Registry.init(a, &embedded, &(@as([64]u8, @splat('0'))), &diag));
    const Case = struct { needle: []const u8, repl: []const u8, err: Refusal, names: []const u8 };
    for ([_]Case{
        .{ .needle = "\"multiplier\":3417055213", .repl = "\"multiplier\":2212286765", .err = error.BankNotImplemented, .names = "2212286765" },
        .{ .needle = "\"codebook\":\"mcg\"", .repl = "\"codebook\":\"mul1\"", .err = error.BankNotImplemented, .names = "mul1" },
        .{ .needle = "\"K\":[3,4]", .repl = "\"K\":[3]", .err = error.BankNotImplemented, .names = "K { 3 }" },
        .{ .needle = "\"name\":\"glm53_pf_mid\"", .repl = "\"name\":\"glm53_pf_mid2\"", .err = error.UnknownKernel, .names = "glm53_pf_mid2" },
        .{ .needle = "\"name\":\"glm53_pf_mid\"", .repl = "\"name\":\"glm53_pf_gu\"", .err = error.DuplicateKernel, .names = "glm53_pf_gu listed twice" },
        .{ .needle = "\"format\":\"mlx-stream-glm53-exl3-kernels-v1\"", .repl = "\"format\":\"mlx-stream-glm53-exl3-kernels-v2\"", .err = error.ManifestFormat, .names = "v2" },
        .{ .needle = "\"threadgroup\":[128,1,1]", .repl = "\"threadgroup\":[2048,1,1]", .err = error.SchemaInvalid, .names = "threadgroup 2048x1x1" },
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

test "glm53 exl3 kernels: a re-pinned mutated text builds a registry (the self-check's mutations)" {
    const a = testing.allocator;
    const src = embedded.sources[@backingInt(Kernel.glm53_dec_down)];
    const mutated = try std.mem.replaceOwned(u8, a, src, "for (uint r = 0u; r < 4u; ++r) {\n            const threadgroup float* zr", "for (uint r = 0u; r < 3u; ++r) {\n            const threadgroup float* zr");
    defer a.free(mutated);
    try testing.expect(!std.mem.eql(u8, mutated, src));
    const z = try a.dupeSentinel(u8, mutated, 0);
    defer a.free(z);
    var texts: Texts = undefined;
    const m = try retexted(a, &embedded, .{ .kernel = .glm53_dec_down }, z, &texts);
    defer a.free(m);
    texts.manifest = m;
    var reg = try initOrPrint(&texts, &shaHex(m));
    defer reg.deinit();
    try testing.expectEqualStrings(z, reg.get(.glm53_dec_down).source);
}
