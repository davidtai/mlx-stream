//! Word-for-word parity of the Zig trunk against our Python oracle's dump
//! (the reference runtime's dump_dsv41_layer_parity.py: stock eager path, MLX
//! CPU device, real layers, routed experts as a stand-in whose output is
//! injected here). The comparator is host code; the runner builds MLX graphs,
//! so its test runs only on a GPU machine (on a GPU machine).

const std = @import("std");
const mlx = @import("sdk").mlx;
const model = @import("deepseek_v41_host.zig").model;
/// Bank shards open past the page cache (F_NOCACHE): a test's reads leave no credited cache behind
/// for the next window (served run 3 found 4.2 GB).
const bank_load: model.LoadOpts = .{ .nocache = true };
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const graph = @import("deepseek_v41_graph.zig");
const engram = @import("deepseek_v41_engram.zig");
const routes = @import("deepseek_v41_routes.zig");

/// How two same-shape tensors differ, in raw words.
pub const Diff = struct {
    n: u64 = 0,
    mismatches: u64 = 0,
    /// Largest distance in units in the last place (floats; ints: |a - b|).
    max_ulp: u64 = 0,
    max_abs: f64 = 0,
    first: ?u64 = null,
    first_a: f64 = 0,
    first_b: f64 = 0,
};

/// Order-preserving map of a float's bits onto an unsigned line, so the
/// distance between two words is their ulp distance (sign-magnitude folded).
fn ordered(bits: u64, width: u6) u64 {
    const sign = @as(u64, 1) << (width - 1);
    const mask = (sign << 1) -% 1;
    return if (bits & sign != 0) (~bits) & mask else bits | sign;
}

fn wordValue(dtype: v41.StDtype, bits: u64) f64 {
    return switch (dtype) {
        .F32 => @floatCast(@as(f32, @bitCast(@as(u32, @intCast(bits))))),
        .BF16 => @floatCast(@as(f32, @bitCast(@as(u32, @intCast(bits)) << 16))),
        .F16 => @floatCast(@as(f16, @bitCast(@as(u16, @intCast(bits))))),
        .I32 => @floatFromInt(@as(i32, @bitCast(@as(u32, @intCast(bits))))),
        .I64 => @floatFromInt(@as(i64, @bitCast(bits))),
        .I16 => @floatFromInt(@as(i16, @bitCast(@as(u16, @intCast(bits))))),
        .I8 => @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(bits))))),
        else => @floatFromInt(bits),
    };
}

fn wordAt(bytes: []const u8, size: u64, i: u64) u64 {
    const at: usize = @intCast(i * size);
    return switch (size) {
        1 => bytes[at],
        2 => std.mem.readInt(u16, bytes[at..][0..2], .little),
        4 => std.mem.readInt(u32, bytes[at..][0..4], .little),
        8 => std.mem.readInt(u64, bytes[at..][0..8], .little),
        else => unreachable,
    };
}

pub fn compareWords(dtype: v41.StDtype, a: []const u8, b: []const u8) !Diff {
    const size = dtype.size();
    if (a.len != b.len or a.len % size != 0) return error.LengthMismatch;
    var d: Diff = .{ .n = a.len / size };
    const float = dtype == .F32 or dtype == .BF16 or dtype == .F16;
    const width: u6 = if (float) @intCast(size * 8) else 0;
    for (0..d.n) |i| {
        const wa = wordAt(a, size, i);
        const wb = wordAt(b, size, i);
        if (wa == wb) continue;
        d.mismatches += 1;
        const va = wordValue(dtype, wa);
        const vb = wordValue(dtype, wb);
        if (d.first == null) {
            d.first = i;
            d.first_a = va;
            d.first_b = vb;
        }
        const ulp = if (float) blk: {
            const oa = ordered(wa, width);
            const ob = ordered(wb, width);
            break :blk if (oa > ob) oa - ob else ob - oa;
        } else if (va > vb) @as(u64, @intFromFloat(va - vb)) else @as(u64, @intFromFloat(vb - va));
        d.max_ulp = @max(d.max_ulp, ulp);
        const abs = @abs(va - vb);
        if (abs > d.max_abs or std.math.isNan(abs)) d.max_abs = abs;
    }
    return d;
}

// ── the window runner (MLX: GPU only) ──

pub fn mlxDtype(d: v41.StDtype) !mlx.mlx_dtype {
    return switch (d) {
        .BOOL => .bool_,
        .U8 => .uint8,
        .U32 => .uint32,
        .I32 => .int32,
        .I64 => .int64,
        .F16 => .float16,
        .BF16 => .bfloat16,
        .F32 => .float32,
        else => error.UnsupportedDtype,
    };
}

pub fn stDtype(d: mlx.mlx_dtype) !v41.StDtype {
    return switch (d) {
        .bool_ => .BOOL,
        .uint8 => .U8,
        .uint32 => .U32,
        .int32 => .I32,
        .int64 => .I64,
        .float16 => .F16,
        .bfloat16 => .BF16,
        .float32 => .F32,
        else => error.UnsupportedDtype,
    };
}

/// An MLX array holding a copy of host bytes.
fn arrayFrom(bytes: []const u8, t: v41.Checkpoint.Tensor) !mlx.mlx_array {
    var shape: [v41.max_rank]c_int = undefined;
    for (t.shape[0..t.rank], 0..) |d, i| shape[i] = @intCast(d);
    const a = mlx.mlx_array_new_data(bytes.ptr, &shape, @intCast(t.rank), try mlxDtype(t.dtype));
    if (a.ctx == null) return error.MlxError;
    return a;
}

/// An evaluated, row-contiguous copy's bytes; valid until the backend resets.
fn hostBytes(g: *ops.MlxOps, x: mlx.mlx_array) ![]const u8 {
    var c = mlx.mlx_array_new();
    mlx.check(mlx.mlx_contiguous(&c, x, false, g.s)) catch |e| {
        _ = mlx.mlx_array_free(c);
        return e;
    };
    _ = try g.adopt(c);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.MlxError;
    return p[0..n];
}

fn getReq(w: *const model.Weights, buf: []u8, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return error.NameTooLong;
    return w.get(name) orelse error.MissingWeight;
}

fn getQ(w: *const model.Weights, buf: []u8, comptime base: []const u8, args: anytype) !graph.Q(mlx.mlx_array) {
    return .{ .w = try getReq(w, buf, base ++ ".weight", args), .s = try getReq(w, buf, base ++ ".scales", args), .mode = .mxfp8 };
}

/// One trunk layer's residents from a weights map `checkLoaded` already validated.
pub fn bindLayer(w: *const model.Weights, li: v41.LayerInfo, l: u32) !graph.LayerW(mlx.mlx_array) {
    var b: [192]u8 = undefined;
    var lw: graph.LayerW(mlx.mlx_array) = .{
        .attn_norm = try getReq(w, &b, "layers.{d}.attn_norm.weight", .{l}),
        .ffn_norm = try getReq(w, &b, "layers.{d}.ffn_norm.weight", .{l}),
        .hc_attn_fn = try getReq(w, &b, "layers.{d}.hc_attn_fn", .{l}),
        .hc_attn_base = try getReq(w, &b, "layers.{d}.hc_attn_base", .{l}),
        .hc_attn_scale = try getReq(w, &b, "layers.{d}.hc_attn_scale", .{l}),
        .hc_ffn_fn = try getReq(w, &b, "layers.{d}.hc_ffn_fn", .{l}),
        .hc_ffn_base = try getReq(w, &b, "layers.{d}.hc_ffn_base", .{l}),
        .hc_ffn_scale = try getReq(w, &b, "layers.{d}.hc_ffn_scale", .{l}),
        .attn_sink = try getReq(w, &b, "layers.{d}.attn.attn_sink", .{l}),
        .q_norm = try getReq(w, &b, "layers.{d}.attn.q_norm.weight", .{l}),
        .kv_norm = try getReq(w, &b, "layers.{d}.attn.kv_norm.weight", .{l}),
        .wq_a = try getQ(w, &b, "layers.{d}.attn.wq_a", .{l}),
        .wq_b = try getQ(w, &b, "layers.{d}.attn.wq_b", .{l}),
        .wkv = try getQ(w, &b, "layers.{d}.attn.wkv", .{l}),
        .wo_a = try getQ(w, &b, "layers.{d}.attn.wo_a", .{l}),
        .wo_b = try getQ(w, &b, "layers.{d}.attn.wo_b", .{l}),
        .gate_w = try getReq(w, &b, "layers.{d}.ffn.gate.weight", .{l}),
        .gate_bias = try getReq(w, &b, "layers.{d}.ffn.gate.bias", .{l}),
        .sh_w1 = try getQ(w, &b, "layers.{d}.ffn.shared_experts.w1", .{l}),
        .sh_w2 = try getQ(w, &b, "layers.{d}.ffn.shared_experts.w2", .{l}),
        .sh_w3 = try getQ(w, &b, "layers.{d}.ffn.shared_experts.w3", .{l}),
    };
    if (li.kv_source) {
        lw.comp = .{
            .wkv = try getReq(w, &b, "layers.{d}.attn.compressor.wkv.weight", .{l}),
            .wgate = if (li.ratio > 1) try getReq(w, &b, "layers.{d}.attn.compressor.wgate.weight", .{l}) else null,
            .norm = try getReq(w, &b, "layers.{d}.attn.compressor.norm.weight", .{l}),
        };
        lw.idx_k = .{ .wk = try getReq(w, &b, "layers.{d}.attn.indexer.wk.weight", .{l}), .k_norm = try getReq(w, &b, "layers.{d}.attn.indexer.k_norm.weight", .{l}) };
    }
    if (li.index_source) {
        lw.idx_q = .{ .wq_b = try getQ(w, &b, "layers.{d}.attn.indexer.wq_b", .{l}), .weights_proj = try getReq(w, &b, "layers.{d}.attn.indexer.weights_proj.weight", .{l}) };
    }
    return lw;
}

/// A shared-runtime mask outlives the per-layer reset: keep the latest one
/// (a later index source may publish a new array) and point the slot at it.
fn persist(g: *ops.MlxOps, slot: *?mlx.mlx_array, kept: *?mlx.mlx_array) void {
    const m = slot.* orelse return;
    if (kept.*) |k| if (k.ctx == m.ctx) return;
    if (kept.*) |old| g.release(old);
    kept.* = g.keep(m);
    slot.* = kept.*;
}

/// Keeps every stage a trunk function reports, by name, for this pass.
const KeepProbe = struct {
    g: *ops.MlxOps,
    gpa: std.mem.Allocator,
    names: std.ArrayList([]const u8) = .empty,
    arrays: std.ArrayList(mlx.mlx_array) = .empty,

    pub fn put(self: *KeepProbe, name: []const u8, x: mlx.mlx_array) !void {
        try self.names.append(self.gpa, name);
        try self.arrays.append(self.gpa, self.g.keep(x));
    }

    fn clear(self: *KeepProbe) void {
        for (self.arrays.items) |a| self.g.release(a);
        self.names.clearRetainingCapacity();
        self.arrays.clearRetainingCapacity();
    }

    fn deinit(self: *KeepProbe) void {
        self.clear();
        self.names.deinit(self.gpa);
        self.arrays.deinit(self.gpa);
    }
};

/// The routed experts of this layer and pass: the Python stand-in's output.
const DumpRouted = struct {
    arr: mlx.mlx_array,

    pub fn routed(self: DumpRouted, _: *ops.MlxOps, _: mlx.mlx_array, _: mlx.mlx_array) !mlx.mlx_array {
        return self.arr;
    }
};

/// The Engram half of a parity run: the host hash over the exported token
/// map, the bank rows, and the MLX residents (wkv + gates) of each layer.
const EngramSide = struct {
    hashing: engram.Hashing,
    bank: engram.Bank,
    map: []u32,
    state: engram.HashState = .{},
    fds: [engram.max_layers]std.c.fd_t = @splat(-1),
    weights: model.Weights,

    fn init(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, bank_dir: []const u8, c: *const v41.Config, map_path: []const u8, s: mlx.mlx_stream) !EngramSide {
        var diag: v41.Diag = .{};
        errdefer std.debug.print("dsv41 parity engram: {s}\n", .{diag.message()});
        const mtext = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/engram/engram-manifest.json", .{bank_dir}), a, .limited(1 << 20));
        const m = try engram.parseManifest(a, mtext, c, &diag);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, map_path, a, .limited(16 << 20));
        const map = try a.alloc(u32, raw.len / 4);
        for (map, 0..) |*v, i| v.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        var self: EngramSide = .{ .hashing = m.hashing, .bank = m.bank, .map = map, .weights = model.Weights.init(gpa) };
        errdefer self.deinit(gpa);
        for (0..m.hashing.n_layers) |i| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}/engram/{s}", .{ bank_dir, m.bank.files[i] }, 0);
            self.fds[i] = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (self.fds[i] < 0) return error.FileNotFound;
        }
        const res = try std.fmt.allocPrintSentinel(a, "{s}/engram/engram-residents.safetensors", .{bank_dir}, 0);
        try model.loadSafetensorsFile(gpa, &self.weights, res.ptr, s, bank_load);
        return self;
    }

    fn deinit(self: *EngramSide, gpa: std.mem.Allocator) void {
        for (self.fds) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        self.state.deinit(gpa);
        self.weights.deinit();
    }

    fn slot(self: *const EngramSide, layer: u32) ?usize {
        return std.mem.indexOfScalar(u32, self.hashing.layer_ids[0..self.hashing.n_layers], layer);
    }

    fn resident(self: *const EngramSide, layer: u32) !graph.EngramW(mlx.mlx_array) {
        var b: [96]u8 = undefined;
        return .{
            .wkv = .{ .w = try getReq(&self.weights, &b, "layers.{d}.engram.wkv.weight", .{layer}), .s = try getReq(&self.weights, &b, "layers.{d}.engram.wkv.scales", .{layer}), .mode = .mxfp8 },
            .q_weight = try getReq(&self.weights, &b, "layers.{d}.engram.q_weight", .{layer}),
            .k_weight = try getReq(&self.weights, &b, "layers.{d}.engram.k_weight", .{layer}),
        };
    }

    /// The dequantized `[1, s, cols, head_dim]` rows of layer slot `li` for
    /// this pass (`rows` from `state.advance`), as `NGramRowCache.dequantize`.
    fn rowsFor(self: *const EngramSide, g: *ops.MlxOps, a: std.mem.Allocator, rows: []const i64, li: usize, s_len: usize) !mlx.mlx_array {
        const cols = self.hashing.cols();
        const per = self.hashing.n_layers * cols;
        const ids = try a.alloc(i64, s_len * cols);
        for (0..s_len) |t| @memcpy(ids[t * cols ..][0..cols], rows[t * per + li * cols ..][0..cols]);
        const hd: usize = self.bank.head_dim;
        const codes = try a.alloc(u8, ids.len * hd);
        const scales = try a.alloc(u8, ids.len * (hd / 32));
        try engram.readRows(self.fds[li], &self.bank, ids, codes, scales);
        const n: c_int = @intCast(ids.len);
        const wshape = [_]c_int{ n, @intCast(hd / 4) };
        const sshape = [_]c_int{ n, @intCast(hd / 32) };
        const w = try g.adopt(mlx.mlx_array_new_data(codes.ptr, &wshape, 2, .uint32));
        const sc = try g.adopt(mlx.mlx_array_new_data(scales.ptr, &sshape, 2, .uint8));
        const dq = try g.dequantize(w, sc, .mxfp8);
        const shape = [_]c_int{ 1, @intCast(s_len), @intCast(cols), @intCast(hd) };
        return g.reshape(dq, &shape);
    }
};

/// JSON has no NaN / infinity: clamp them to strings' stand-in values.
const JsonNum = struct {
    v: f64,
    pub fn format(self: JsonNum, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (std.math.isNan(self.v)) return w.writeAll("\"nan\"");
        if (std.math.isInf(self.v)) return w.writeAll(if (self.v > 0) "\"inf\"" else "\"-inf\"");
        return w.print("{e}", .{self.v});
    }
};

fn jsonNum(v: f64) JsonNum {
    return .{ .v = v };
}

/// The window-2 dump's `__metadata__` strings (safetensors header).
pub fn readMetadata(a: std.mem.Allocator, path: []const u8) !std.json.ObjectMap {
    const pz = try a.dupeSentinel(u8, path, 0);
    const fd = std.c.open(pz.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var lenb: [8]u8 = undefined;
    if (std.c.pread(fd, &lenb, 8, 0) != 8) return error.ShortRead;
    const n = std.mem.readInt(u64, &lenb, .little);
    if (n > 64 << 20) return error.ShardHeader;
    const json = try a.alloc(u8, @intCast(n));
    if (std.c.pread(fd, json.ptr, json.len, 8) != @as(isize, @intCast(json.len))) return error.ShortRead;
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    const md = v.object.get("__metadata__") orelse return error.MetadataMissing;
    if (md != .object) return error.MetadataMissing;
    return md.object;
}

/// The window-2 schedule: tokens per pass and the trims after given passes.
pub const Schedule = struct {
    pass_tokens: []u32,
    /// `trim_after[p]` tokens dropped from every lane after pass p.
    trim_after: []u32,

    pub fn parse(a: std.mem.Allocator, pass_tokens: []const u8, trims: []const u8) !Schedule {
        var toks: std.ArrayList(u32) = .empty;
        var it = std.mem.tokenizeScalar(u8, pass_tokens, ',');
        while (it.next()) |t| try toks.append(a, try std.fmt.parseInt(u32, t, 10));
        const trim_after = try a.alloc(u32, toks.items.len);
        @memset(trim_after, 0);
        var tt = std.mem.tokenizeScalar(u8, trims, ',');
        while (tt.next()) |t| {
            const colon = std.mem.indexOfScalar(u8, t, ':') orelse return error.ScheduleSyntax;
            const p = try std.fmt.parseInt(usize, t[0..colon], 10);
            if (p >= trim_after.len) return error.ScheduleSyntax;
            trim_after[p] = try std.fmt.parseInt(u32, t[colon + 1 ..], 10);
        }
        return .{ .pass_tokens = toks.items, .trim_after = trim_after };
    }
};

pub const StageResult = struct { key: []const u8, dtype: v41.StDtype, shape: [v41.max_rank]u64, rank: u8, diff: Diff, note: []const u8 = "" };

/// Runs the dump's passes through the Zig trunk and compares every stage the
/// dump holds. `dump` and `bank` are absolute paths.
pub const Runner = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    results: std.ArrayList(StageResult) = .empty,

    pub fn deinit(self: *Runner) void {
        self.arena.deinit();
    }

    pub fn firstMismatch(self: *const Runner) ?*const StageResult {
        for (self.results.items) |*r| if (r.diff.mismatches > 0 or r.note.len > 0) return r;
        return null;
    }

    fn record(self: *Runner, key: []const u8, t: v41.Checkpoint.Tensor, diff: Diff, note: []const u8) !void {
        const a = self.arena.allocator();
        try self.results.append(a, .{ .key = try a.dupe(u8, key), .dtype = t.dtype, .shape = t.shape, .rank = t.rank, .diff = diff, .note = note });
    }

    /// Compare one evaluated Zig array with the dump tensor `key`.
    fn check(self: *Runner, g: *ops.MlxOps, dump: *const v41.Checkpoint, key: []const u8, x: mlx.mlx_array) !void {
        const a = self.arena.allocator();
        const t = dump.tensors.get(key) orelse return;
        const got_dt = try stDtype(mlx.mlx_array_dtype(x));
        const got_shape = mlx.getShape(x);
        var same_shape = got_shape.len == t.rank;
        if (same_shape) for (got_shape, t.shape[0..t.rank]) |gd, td| {
            if (@as(u64, @intCast(gd)) != td) same_shape = false;
        };
        if (got_dt != t.dtype or !same_shape) return self.record(key, t, .{}, try std.fmt.allocPrint(a, "zig {s} {any} vs dump {s} {any}", .{ @tagName(got_dt), got_shape, @tagName(t.dtype), t.shape[0..t.rank] }));
        const want = try v41.readTensor(self.gpa, dump, key);
        defer self.gpa.free(want);
        const got = try hostBytes(g, x);
        try self.record(key, t, try compareWords(t.dtype, got, want), "");
    }

    /// `gpu` runs the trunk on the default GPU stream (a dump made with
    /// MLX_DEFAULT_DEVICE=gpu); otherwise the CPU stream, as the dump's default.
    pub fn run(gpa: std.mem.Allocator, io: std.Io, dump_path: []const u8, bank: []const u8, gpu: bool) !Runner {
        var self: Runner = .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.deinit();
        const a = self.arena.allocator();
        var diag: v41.Diag = .{};
        errdefer std.debug.print("dsv41 parity: {s}\n", .{diag.message()});

        // Host checks first: config, spec and headers refuse a bad bank before any MLX array.
        const c = try v41.Config.load(gpa, io, bank, &diag);
        var ck = try v41.Checkpoint.openIndexed(gpa, io, bank, &diag);
        defer ck.deinit();
        _ = try v41.WeightMap.build(gpa, try v41.residentSpec(a, &c), &ck, &diag);
        var dump = try v41.Checkpoint.openFile(gpa, dump_path, &diag);
        defer dump.deinit();

        // Layers and passes from the dump's keys.
        var layers: std.ArrayList(u32) = .empty;
        var n_pass: u32 = 0;
        for (dump.tensors.keys()) |k| {
            if (std.mem.startsWith(u8, k, "p0.L") and std.mem.endsWith(u8, k, ".in.h")) {
                const l = try std.fmt.parseInt(u32, k[4 .. k.len - 5], 10);
                try layers.append(a, l);
            }
            if (std.mem.startsWith(u8, k, "p") and std.mem.endsWith(u8, k, ".positions")) n_pass += 1;
        }
        std.mem.sort(u32, layers.items, {}, std.sort.asc(u32));
        if (layers.items.len == 0 or n_pass == 0) return error.EmptyDump;

        const s = if (gpu) mlx.mlx_default_gpu_stream_new() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        // Safetensors loads run on the CPU stream whatever the trunk's device (as mlx-serve loads).
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var g = try ops.MlxOps.init(gpa, s);
        defer g.deinit();
        const Tr = graph.Trunk(ops.MlxOps);
        const rt: graph.Routes = .{};

        // The layer shards through mlx-serve's loader (lazy arrays, CPU stream).
        var weights = model.Weights.init(gpa);
        defer weights.deinit();
        var loaded: std.ArrayList(u16) = .empty;
        for (layers.items) |l| {
            var nb: [64]u8 = undefined;
            const t = ck.tensors.get(try std.fmt.bufPrint(&nb, "layers.{d}.attn.wq_a.weight", .{l})) orelse return error.TensorMissing;
            if (std.mem.indexOfScalar(u16, loaded.items, t.shard) != null) continue;
            try loaded.append(a, t.shard);
            const path = try ck.shardPath(a, t.shard);
            try model.loadSafetensorsFile(gpa, &weights, path.ptr, cpu, bank_load);
        }
        const nl = layers.items.len;
        const lws = try a.alloc(graph.LayerW(mlx.mlx_array), nl);
        const inv = try a.alloc(mlx.mlx_array, nl);
        const caches = try a.alloc(Tr.Cache, nl);
        for (caches, layers.items) |*cc, l| cc.* = Tr.Cache.init(c.layers[l], c.window, .{});
        defer for (caches) |*cc| cc.deinit(&g);
        var kb: [96]u8 = undefined;
        for (layers.items, 0..) |l, i| {
            const li = c.layers[l];
            lws[i] = try bindLayer(&weights, li, l);
            const f = if (li.ratio > 0) try Tr.yarnInvFreq(&g, &c) else try Tr.swaInvFreq(&g, &c);
            inv[i] = g.keep(f);
            try self.check(&g, &dump, try std.fmt.bufPrint(&kb, "const.L{d}.inv_freq", .{l}), inv[i]);
        }
        defer for (inv) |x| g.release(x);

        var probe: KeepProbe = .{ .g = &g, .gpa = gpa };
        defer probe.deinit();

        // Engram layers of the chain, when the dump ran their hooks (--engram).
        var eng: ?EngramSide = null;
        defer if (eng) |*e| e.deinit(gpa);
        for (layers.items) |l| {
            if (dump.tensors.get(try std.fmt.bufPrint(&kb, "p0.L{d}.engram.in", .{l})) == null) continue;
            const map_path = std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.EngramTokenMapRequired;
            eng = try EngramSide.init(gpa, a, io, bank, &c, std.mem.span(map_path), cpu);
            break;
        }
        const ids_bytes = try v41.readTensor(a, &dump, "inputs.ids");
        var tok: u64 = 0;
        for (0..n_pass) |p| {
            defer g.reset();
            const pos_key = try std.fmt.allocPrint(a, "p{d}.positions", .{p});
            const pos_t = dump.tensors.get(pos_key) orelse return error.TensorMissing;
            const positions = try g.adopt(try arrayFrom(try v41.readTensor(a, &dump, pos_key), pos_t));
            const s_len = pos_t.shape[0];

            // Embedding rows straight from the shard (the Python dump reads them the same way).
            const emb_key = try std.fmt.allocPrint(a, "p{d}.embed", .{p});
            if (dump.tensors.get(emb_key)) |et| {
                const row_bytes: u64 = @as(u64, c.hidden_size) * 2;
                const rows = try a.alloc(u8, @intCast(s_len * row_bytes));
                const et_embed = ck.tensors.get("embed.weight") orelse return error.TensorMissing;
                const epath = try ck.shardPath(a, et_embed.shard);
                const fd = std.c.open(epath.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
                if (fd < 0) return error.ShardMissing;
                defer _ = std.c.close(fd);
                for (0..@intCast(s_len)) |j| {
                    const id: u64 = @intCast(std.mem.readInt(i32, ids_bytes[@intCast((tok + j) * 4)..][0..4], .little));
                    const dst = rows[@intCast(j * row_bytes)..][0..@intCast(row_bytes)];
                    if (std.c.pread(fd, dst.ptr, dst.len, @intCast(et_embed.begin + id * row_bytes)) != @as(isize, @intCast(dst.len))) return error.ShortRead;
                }
                const want = try v41.readTensor(a, &dump, emb_key);
                try self.record(emb_key, et, try compareWords(.BF16, rows, want), "");
                const emb = try g.adopt(try arrayFrom(rows, et));
                const e = try Tr.expandEmbedding(&g, &c, emb);
                try self.check(&g, &dump, try std.fmt.bufPrint(&kb, "p{d}.L{d}.in.h", .{ p, layers.items[0] }), e.h);
                try self.check(&g, &dump, try std.fmt.bufPrint(&kb, "p{d}.L{d}.in.pre_mix", .{ p, layers.items[0] }), e.pre_mix);
            }
            tok += s_len;

            var shared: Tr.Share = .{};
            var eng_rows: []i64 = &.{};
            if (eng) |*e| {
                const per = e.hashing.n_layers * e.hashing.cols();
                eng_rows = try a.alloc(i64, @as(usize, @intCast(s_len)) * per);
                const span = try a.alloc(u32, @intCast(s_len));
                for (span, 0..) |*v, j| v.* = @intCast(std.mem.readInt(i32, ids_bytes[@intCast((tok - s_len + j) * 4)..][0..4], .little));
                try e.state.advance(gpa, &e.hashing, e.map, span, null, eng_rows);
            }
            for (layers.items, 0..) |l, i| {
                probe.clear();
                const pre = try std.fmt.allocPrint(a, "p{d}.L{d}.", .{ p, l });
                if (eng) |*e| if (e.slot(l)) |li| {
                    if (dump.tensors.get(try std.fmt.allocPrint(a, "{s}engram.in", .{pre})) != null) {
                        const rows = try e.rowsFor(&g, a, eng_rows, li, @intCast(s_len));
                        const out = try Tr.engramApply(&g, &c, try e.resident(l), try self.dumpArray(&g, &dump, pre, "engram.in"), rows);
                        try self.check(&g, &dump, try std.fmt.allocPrint(a, "{s}engram.out", .{pre}), out);
                    }
                };
                const h = try self.dumpArray(&g, &dump, pre, "in.h");
                const pm = try self.dumpArray(&g, &dump, pre, "in.pre_mix");
                const routed: DumpRouted = .{ .arr = try self.dumpArray(&g, &dump, pre, "moe.routed") };
                _ = try Tr.layer(&g, &probe, &c, &rt, .{}, c.layers[l], &lws[i], inv[i], h, pm, positions, &caches[i], &shared, routed);
                const vec = mlx.mlx_vector_array_new_data(probe.arrays.items.ptr, probe.arrays.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                for (probe.names.items, probe.arrays.items) |name, arr| {
                    const key = try std.fmt.allocPrint(a, "{s}{s}", .{ pre, name });
                    try self.check(&g, &dump, key, arr);
                }
            }

            for (caches) |*cc| cc.advance(@intCast(s_len));
            // Final collapse + norm and the head slice, each from the dump's own input.
            const last = layers.items[nl - 1];
            const lpre = try std.fmt.allocPrint(a, "p{d}.L{d}.", .{ p, last });
            const fin_key = try std.fmt.allocPrint(a, "p{d}.final.h", .{p});
            const norm_t = ck.tensors.get("norm.weight") orelse return error.TensorMissing;
            const norm_w = try g.adopt(try arrayFrom(try v41.readTensor(a, &ck, "norm.weight"), norm_t));
            const fin = try Tr.finalNorm(&g, &c, try self.dumpArray(&g, &dump, lpre, "out.h"), try self.dumpArray(&g, &dump, lpre, "out.pre_mix"), norm_w);
            try self.check(&g, &dump, fin_key, fin);
            const head_key = try std.fmt.allocPrint(a, "p{d}.head.logits", .{p});
            if (dump.tensors.get(head_key)) |ht| {
                const rows = ht.shape[ht.rank - 1];
                const head_t = ck.tensors.get("head.weight") orelse return error.TensorMissing;
                var sub = head_t;
                sub.shape[0] = rows;
                sub.end = sub.begin + rows * @as(u64, c.hidden_size) * 2;
                const hbytes = blk: {
                    const buf = try a.alloc(u8, @intCast(sub.end - sub.begin));
                    const hpath = try ck.shardPath(a, head_t.shard);
                    const fd = std.c.open(hpath.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
                    if (fd < 0) return error.ShardMissing;
                    defer _ = std.c.close(fd);
                    if (std.c.pread(fd, buf.ptr, buf.len, @intCast(sub.begin)) != @as(isize, @intCast(buf.len))) return error.ShortRead;
                    break :blk buf;
                };
                const head_w = try g.adopt(try arrayFrom(hbytes, sub));
                try self.check(&g, &dump, head_key, try Tr.head(&g, &rt, try self.dumpArray(&g, &dump, "", fin_key), .{ .dense = head_w }));
            }
        }
        // Stages the dump holds that the Zig trunk never reported (inputs excepted).
        for (dump.tensors.keys(), dump.tensors.values()) |k, t| {
            if (self.reported(k)) continue;
            const input = std.mem.eql(u8, k, "inputs.ids") or std.mem.endsWith(u8, k, ".positions") or
                std.mem.endsWith(u8, k, ".moe.routed") or std.mem.endsWith(u8, k, ".in.h") or std.mem.endsWith(u8, k, ".in.pre_mix") or
                std.mem.endsWith(u8, k, ".engram.in");
            if (input) continue;
            try self.record(k, t, .{}, "in the dump, not produced by the Zig trunk");
        }
        return self;
    }

    /// Window 2: the dump's lever set as the Zig tier (routes + KV backing),
    /// its passes and trims replayed layer by layer from the dump's inputs, the
    /// Engram rows through the row source and the converter's token map, and
    /// every lane view compared after each pass.
    pub fn runRoutes(gpa: std.mem.Allocator, io: std.Io, dump_path: []const u8, bank: []const u8, map_path: []const u8, gpu: bool) !Runner {
        var self: Runner = .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.deinit();
        const a = self.arena.allocator();
        var diag: v41.Diag = .{};
        errdefer std.debug.print("dsv41 parity: {s}\n", .{diag.message()});

        const c = try v41.Config.load(gpa, io, bank, &diag);
        var ck = try v41.Checkpoint.openIndexed(gpa, io, bank, &diag);
        defer ck.deinit();
        _ = try v41.WeightMap.build(gpa, try v41.residentSpec(a, &c), &ck, &diag);
        var dump = try v41.Checkpoint.openFile(gpa, dump_path, &diag);
        defer dump.deinit();
        const md = try readMetadata(a, dump_path);
        const levers = (md.get("levers") orelse return error.MetadataMissing).string;
        const tier = try routes.parse(try routes.splitPairs(a, levers), &diag);
        const rt = tier.routes;
        const sched = try Schedule.parse(a, (md.get("pass_tokens") orelse return error.MetadataMissing).string, (md.get("trims") orelse return error.MetadataMissing).string);
        var layers: std.ArrayList(u32) = .empty;
        var lit = std.mem.tokenizeScalar(u8, (md.get("layers") orelse return error.MetadataMissing).string, ',');
        while (lit.next()) |t| try layers.append(a, try std.fmt.parseInt(u32, t, 10));
        const drops = try std.json.parseFromSliceLeaky([]const std.json.ArrayHashMap(u32), a, (md.get("window_drop") orelse return error.MetadataMissing).string, .{});

        // Compile availability follows the default device, as Python's mx.set_default_device.
        var prev = mlx.mlx_device{ .ctx = null };
        _ = mlx.mlx_get_default_device(&prev);
        defer {
            _ = mlx.mlx_set_default_device(prev);
            _ = mlx.mlx_device_free(prev);
        }
        const dev = mlx.mlx_device_new_type(if (gpu) .gpu else .cpu, 0);
        defer _ = mlx.mlx_device_free(dev);
        try mlx.check(mlx.mlx_set_default_device(dev));
        const s = if (gpu) mlx.mlx_default_gpu_stream_new() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var g = try ops.MlxOps.init(gpa, s);
        defer g.deinit();
        const Tr = graph.Trunk(ops.MlxOps);
        try Tr.prepareRegions(&g, &c, &rt, tier.layer_major);

        var weights = model.Weights.init(gpa);
        defer weights.deinit();
        var loaded: std.ArrayList(u16) = .empty;
        for (layers.items) |l| {
            var nb: [64]u8 = undefined;
            const t = ck.tensors.get(try std.fmt.bufPrint(&nb, "layers.{d}.attn.wq_a.weight", .{l})) orelse return error.TensorMissing;
            if (std.mem.indexOfScalar(u16, loaded.items, t.shard) != null) continue;
            try loaded.append(a, t.shard);
            try model.loadSafetensorsFile(gpa, &weights, (try ck.shardPath(a, t.shard)).ptr, cpu, bank_load);
        }
        const nl = layers.items.len;
        const lws = try a.alloc(graph.LayerW(mlx.mlx_array), nl);
        const inv = try a.alloc(mlx.mlx_array, nl);
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer for (owned.items) |x| g.release(x);
        const caches = try a.alloc(Tr.Cache, nl);
        for (caches, layers.items) |*cc, l| cc.* = Tr.Cache.init(c.layers[l], c.window, tier.kv);
        defer for (caches) |*cc| cc.deinit(&g);
        var kb: [96]u8 = undefined;
        for (layers.items, 0..) |l, i| {
            const li = c.layers[l];
            lws[i] = try bindLayer(&weights, li, l);
            if (rt.wo_a_f32) {
                lws[i].wo_a_dense = g.keep(try Tr.woaDenseF32(&g, &c, lws[i].wo_a));
                try owned.append(a, lws[i].wo_a_dense.?);
            }
            inv[i] = g.keep(if (li.ratio > 0) try Tr.yarnInvFreq(&g, &c) else try Tr.swaInvFreq(&g, &c));
            try owned.append(a, inv[i]);
            try self.check(&g, &dump, try std.fmt.bufPrint(&kb, "const.L{d}.inv_freq", .{l}), inv[i]);
        }
        try g.evalAll(owned.items);

        // The head slice the dump used, under the lever's codec.
        const head_t = ck.tensors.get("head.weight") orelse return error.TensorMissing;
        const head_key0 = "p0.head.logits";
        const head_rows: u64 = if (dump.tensors.get(head_key0)) |ht| ht.shape[ht.rank - 1] else 0;
        var head_w: Tr.HeadW = undefined;
        if (head_rows > 0) {
            var sub = head_t;
            sub.shape[0] = head_rows;
            sub.end = sub.begin + head_rows * @as(u64, c.hidden_size) * 2;
            const buf = try a.alloc(u8, @intCast(sub.end - sub.begin));
            const hfd = std.c.open((try ck.shardPath(a, head_t.shard)).ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (hfd < 0) return error.ShardMissing;
            defer _ = std.c.close(hfd);
            if (std.c.pread(hfd, buf.ptr, buf.len, @intCast(sub.begin)) != @as(isize, @intCast(buf.len))) return error.ShortRead;
            const dense = g.keep(try arrayFrom(buf, sub));
            try owned.append(a, dense);
            head_w = switch (rt.head) {
                .f32, .bf16 => .{ .dense = dense },
                .mxfp8 => blk: {
                    const q = try Tr.quantizeHead(&g, dense);
                    const qw = g.keep(q.w);
                    const qs = g.keep(q.s);
                    try owned.appendSlice(a, &.{ qw, qs });
                    try g.evalAll(&.{ qw, qs });
                    break :blk .{ .mxfp8 = .{ .w = qw, .s = qs, .mode = .mxfp8 } };
                },
            };
        }

        // Engram through the production row source (the converter's map + sidecar).
        var src: ?engram.RowSource = null;
        defer if (src) |*x| x.deinit();
        var hash: engram.HashState = .{};
        defer hash.deinit(gpa);
        var eweights = model.Weights.init(gpa);
        defer eweights.deinit();
        const eng_on = (md.get("engram") orelse return error.MetadataMissing).string.len > 0;
        if (eng_on) {
            src = try engram.RowSource.open(gpa, io, bank, map_path, &c, &diag);
            const res = try std.fmt.allocPrintSentinel(a, "{s}/engram/engram-residents.safetensors", .{bank}, 0);
            try model.loadSafetensorsFile(gpa, &eweights, res.ptr, cpu, bank_load);
        }

        const norm_t = ck.tensors.get("norm.weight") orelse return error.TensorMissing;
        const norm_w = g.keep(try arrayFrom(try v41.readTensor(a, &ck, "norm.weight"), norm_t));
        try owned.append(a, norm_w);
        const ids_bytes = try v41.readTensor(a, &dump, "inputs.ids");
        var probe: KeepProbe = .{ .g = &g, .gpa = gpa };
        defer probe.deinit();
        for (sched.pass_tokens, 0..) |s_len, p| {
            defer g.reset();
            const pos_key = try std.fmt.allocPrint(a, "p{d}.positions", .{p});
            const pos_t = dump.tensors.get(pos_key) orelse return error.TensorMissing;
            const positions = try g.adopt(try arrayFrom(try v41.readTensor(a, &dump, pos_key), pos_t));
            var eng_rows: []i64 = &.{};
            if (src) |*es| {
                const span = try a.alloc(u32, s_len);
                const base = try self.passStart(a, &dump, p);
                for (span, 0..) |*v, j| v.* = @intCast(std.mem.readInt(i32, ids_bytes[(base + j) * 4 ..][0..4], .little));
                eng_rows = try a.alloc(i64, s_len * es.perToken());
                try es.advance(gpa, &hash, span, eng_rows);
            }
            var shared: Tr.Share = .{};
            for (layers.items, 0..) |l, i| {
                probe.clear();
                const pre = try std.fmt.allocPrint(a, "p{d}.L{d}.", .{ p, l });
                if (src) |*es| if (c.layers[l].engram_slot) |slot| {
                    const nr: c_int = @intCast(s_len * es.hashing.cols());
                    const hd = es.bank.head_dim;
                    const ids = try a.alloc(i64, s_len * es.hashing.cols());
                    const codes = try a.alloc(u8, ids.len * hd);
                    const scales = try a.alloc(u8, ids.len * (hd / 32));
                    try es.read(slot, eng_rows, s_len, ids, codes, scales);
                    const er = try Tr.engramRows(&g, try g.hostArray(codes, &.{ nr, @intCast(hd / 4) }, .uint32), try g.hostArray(scales, &.{ nr, @intCast(hd / 32) }, .uint8), 1, @intCast(s_len), @intCast(es.hashing.cols()));
                    var b: [96]u8 = undefined;
                    const ew: graph.EngramW(mlx.mlx_array) = .{
                        .wkv = .{ .w = try getReq(&eweights, &b, "layers.{d}.engram.wkv.weight", .{l}), .s = try getReq(&eweights, &b, "layers.{d}.engram.wkv.scales", .{l}), .mode = .mxfp8 },
                        .q_weight = try getReq(&eweights, &b, "layers.{d}.engram.q_weight", .{l}),
                        .k_weight = try getReq(&eweights, &b, "layers.{d}.engram.k_weight", .{l}),
                    };
                    const out = try Tr.engramApply(&g, &c, ew, try self.dumpArray(&g, &dump, pre, "engram.in"), er);
                    try self.check(&g, &dump, try std.fmt.allocPrint(a, "{s}engram.out", .{pre}), out);
                };
                const h = try self.dumpArray(&g, &dump, pre, "in.h");
                const pm = try self.dumpArray(&g, &dump, pre, "in.pre_mix");
                const routed: DumpRouted = .{ .arr = try self.dumpArray(&g, &dump, pre, "moe.routed") };
                _ = try Tr.layer(&g, &probe, &c, &rt, .{}, c.layers[l], &lws[i], inv[i], h, pm, positions, &caches[i], &shared, routed);
                const vec = mlx.mlx_vector_array_new_data(probe.arrays.items.ptr, probe.arrays.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                for (probe.names.items, probe.arrays.items) |name, arr| try self.check(&g, &dump, try std.fmt.allocPrint(a, "{s}{s}", .{ pre, name }), arr);
            }
            for (caches) |*cc| cc.advance(s_len);
            // The lanes after the pass: the same rows and the same drop offsets.
            for (layers.items, caches) |l, *cc| {
                const pre = try std.fmt.allocPrint(a, "p{d}.L{d}.cache.", .{ p, l });
                if (try cc.window.view(&g)) |x| try self.check(&g, &dump, try std.fmt.allocPrint(a, "{s}window", .{pre}), x);
                if (try cc.compress.view(&g)) |x| try self.check(&g, &dump, try std.fmt.allocPrint(a, "{s}compress", .{pre}), x);
                if (try cc.index.view(&g)) |x| try self.check(&g, &dump, try std.fmt.allocPrint(a, "{s}index", .{pre}), x);
                var nb: [16]u8 = undefined;
                const want_drop = drops[p].map.get(try std.fmt.bufPrint(&nb, "{d}", .{l})) orelse return error.MetadataMissing;
                if (cc.window.dropOffset() != want_drop) {
                    const t: v41.Checkpoint.Tensor = .{ .shard = 0, .dtype = .I32, .shape = @splat(0), .rank = 0, .begin = 0, .end = 0 };
                    try self.record(try std.fmt.allocPrint(a, "{s}window_drop", .{pre}), t, .{ .n = 1, .mismatches = 1 }, try std.fmt.allocPrint(a, "zig {d} vs python {d}", .{ cc.window.dropOffset(), want_drop }));
                }
            }
            if (sched.trim_after[p] > 0) {
                const n = sched.trim_after[p];
                for (caches) |*cc| if (try cc.trim(&g, n) != n) return error.TrimRefused;
                if (src != null) hash.trim(n);
            }
            // Final collapse + norm and the head slice, each from the dump's own input.
            const last = layers.items[nl - 1];
            const lpre = try std.fmt.allocPrint(a, "p{d}.L{d}.", .{ p, last });
            const fin_key = try std.fmt.allocPrint(a, "p{d}.final.h", .{p});
            const fin = try Tr.finalNorm(&g, &c, try self.dumpArray(&g, &dump, lpre, "out.h"), try self.dumpArray(&g, &dump, lpre, "out.pre_mix"), norm_w);
            try self.check(&g, &dump, fin_key, fin);
            if (head_rows > 0) {
                const head_key = try std.fmt.allocPrint(a, "p{d}.head.logits", .{p});
                try self.check(&g, &dump, head_key, try Tr.head(&g, &rt, try self.dumpArray(&g, &dump, "", fin_key), head_w));
            }
        }
        for (dump.tensors.keys(), dump.tensors.values()) |k, t| {
            if (self.reported(k)) continue;
            const input = std.mem.eql(u8, k, "inputs.ids") or std.mem.endsWith(u8, k, ".positions") or std.mem.endsWith(u8, k, ".embed") or
                std.mem.endsWith(u8, k, ".moe.routed") or std.mem.endsWith(u8, k, ".in.h") or std.mem.endsWith(u8, k, ".in.pre_mix") or
                std.mem.endsWith(u8, k, ".engram.in");
            if (input) continue;
            try self.record(k, t, .{}, "in the dump, not produced by the Zig trunk");
        }
        return self;
    }

    /// The first id index of pass `p` (its first position).
    fn passStart(self: *Runner, a: std.mem.Allocator, dump: *const v41.Checkpoint, p: usize) !usize {
        _ = self;
        const pos = try v41.readTensor(a, dump, try std.fmt.allocPrint(a, "p{d}.positions", .{p}));
        return @intCast(std.mem.readInt(i32, pos[0..4], .little));
    }

    const Digest = struct { sha256: []const u8, dtype: []const u8, shape: []const u64 };
    const Digests = struct {
        layers: []const u32,
        pass_tokens: []const u32,
        ids: []const u32,
        head_rows: u32,
        engram: []const u32 = &.{},
        digests: std.json.ArrayHashMap(Digest),
    };

    /// Compare one evaluated array with a digest entry by sha256 of its bytes.
    fn checkDigest(self: *Runner, g: *ops.MlxOps, dg: *const Digests, key: []const u8, x: mlx.mlx_array) !void {
        const a = self.arena.allocator();
        const d = dg.digests.map.get(key) orelse return;
        var t: v41.Checkpoint.Tensor = .{ .shard = 0, .dtype = std.meta.stringToEnum(v41.StDtype, d.dtype) orelse return error.UnsupportedDtype, .shape = @splat(0), .rank = @intCast(d.shape.len), .begin = 0, .end = 0 };
        @memcpy(t.shape[0..t.rank], d.shape);
        const got_dt = try stDtype(mlx.mlx_array_dtype(x));
        const got_shape = mlx.getShape(x);
        var same = got_shape.len == t.rank and got_dt == t.dtype;
        if (same) for (got_shape, t.shape[0..t.rank]) |gd, td| {
            if (@as(u64, @intCast(gd)) != td) same = false;
        };
        if (!same) return self.record(key, t, .{}, try std.fmt.allocPrint(a, "zig {s} {any} vs dump {s} {any}", .{ @tagName(got_dt), got_shape, d.dtype, d.shape }));
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(try hostBytes(g, x), &h, .{});
        const hex = std.fmt.bytesToHex(h, .lower);
        var numel: u64 = 1;
        for (t.shape[0..t.rank]) |v| numel *= v;
        const differs = !std.mem.eql(u8, &hex, d.sha256);
        try self.record(key, t, .{ .n = numel, .mismatches = @intFromBool(differs) }, if (differs) "sha256 differs" else "");
    }

    /// The chained run: the Zig trunk feeds each layer its own
    /// previous output (routed experts = the Zig stand-in), and every stage's
    /// sha256 is compared with the Python dump's `--digest` file.
    pub fn runChained(gpa: std.mem.Allocator, io: std.Io, digests_path: []const u8, bank: []const u8, gpu: bool) !Runner {
        var self: Runner = .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.deinit();
        const a = self.arena.allocator();
        var diag: v41.Diag = .{};
        errdefer std.debug.print("dsv41 parity: {s}\n", .{diag.message()});

        const c = try v41.Config.load(gpa, io, bank, &diag);
        var ck = try v41.Checkpoint.openIndexed(gpa, io, bank, &diag);
        defer ck.deinit();
        _ = try v41.WeightMap.build(gpa, try v41.residentSpec(a, &c), &ck, &diag);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, digests_path, a, .limited(256 << 20));
        const dg = try std.json.parseFromSliceLeaky(Digests, a, text, .{ .ignore_unknown_fields = true });
        if (dg.layers.len == 0 or dg.pass_tokens.len == 0) return error.EmptyDump;

        const s = if (gpu) mlx.mlx_default_gpu_stream_new() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var g = try ops.MlxOps.init(gpa, s);
        defer g.deinit();
        const Tr = graph.Trunk(ops.MlxOps);
        const rt: graph.Routes = .{};
        // Bound: MlxOps holds every op output of a layer until its reset (a
        // 2,048-token ratio-1 layer holds 16.0 GiB, the host trace in graph.zig),
        // and the MLX buffer cache would keep each layer type's freed set on top.
        // No cache: a layer's buffers go back at its reset.
        var prev_cache_limit: usize = 0;
        _ = mlx.mlx_set_cache_limit(&prev_cache_limit, 0);
        defer _ = mlx.mlx_set_cache_limit(&prev_cache_limit, prev_cache_limit);

        var weights = model.Weights.init(gpa);
        defer weights.deinit();
        var loaded: std.ArrayList(u16) = .empty;
        for (dg.layers) |l| {
            var nb: [64]u8 = undefined;
            const t = ck.tensors.get(try std.fmt.bufPrint(&nb, "layers.{d}.attn.wq_a.weight", .{l})) orelse return error.TensorMissing;
            if (std.mem.indexOfScalar(u16, loaded.items, t.shard) != null) continue;
            try loaded.append(a, t.shard);
            try model.loadSafetensorsFile(gpa, &weights, (try ck.shardPath(a, t.shard)).ptr, cpu, bank_load);
        }
        const nl = dg.layers.len;
        const lws = try a.alloc(graph.LayerW(mlx.mlx_array), nl);
        const inv = try a.alloc(mlx.mlx_array, nl);
        const caches = try a.alloc(Tr.Cache, nl);
        for (caches, dg.layers) |*cc, l| cc.* = Tr.Cache.init(c.layers[l], c.window, .{});
        defer for (caches) |*cc| cc.deinit(&g);
        for (dg.layers, 0..) |l, i| {
            lws[i] = try bindLayer(&weights, c.layers[l], l);
            inv[i] = g.keep(if (c.layers[l].ratio > 0) try Tr.yarnInvFreq(&g, &c) else try Tr.swaInvFreq(&g, &c));
        }
        defer for (inv) |x| g.release(x);

        const table = try a.alloc(f32, c.n_routed_experts);
        graph.StandIn(ops.MlxOps).table(table);
        const tshape = [_]c_int{@intCast(table.len)};
        const scale = mlx.mlx_array_new_data(table.ptr, &tshape, 1, .float32);
        defer _ = mlx.mlx_array_free(scale);
        const stand_in: graph.StandIn(ops.MlxOps) = .{ .scale = scale };

        var eng: ?EngramSide = null;
        defer if (eng) |*e| e.deinit(gpa);
        if (dg.engram.len > 0) {
            const map_path = std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.EngramTokenMapRequired;
            eng = try EngramSide.init(gpa, a, io, bank, &c, std.mem.span(map_path), cpu);
        }

        const norm_t = ck.tensors.get("norm.weight") orelse return error.TensorMissing;
        const norm_bytes = try v41.readTensor(a, &ck, "norm.weight");
        const embed_t = ck.tensors.get("embed.weight") orelse return error.TensorMissing;
        const head_t = ck.tensors.get("head.weight") orelse return error.TensorMissing;
        const row_bytes: u64 = @as(u64, c.hidden_size) * 2;
        const efd = std.c.open((try ck.shardPath(a, embed_t.shard)).ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (efd < 0) return error.ShardMissing;
        defer _ = std.c.close(efd);
        const head_bytes = try a.alloc(u8, @intCast(dg.head_rows * row_bytes));
        {
            const hfd = std.c.open((try ck.shardPath(a, head_t.shard)).ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (hfd < 0) return error.ShardMissing;
            defer _ = std.c.close(hfd);
            if (std.c.pread(hfd, head_bytes.ptr, head_bytes.len, @intCast(head_t.begin)) != @as(isize, @intCast(head_bytes.len))) return error.ShortRead;
        }
        var head_sub = head_t;
        head_sub.shape[0] = dg.head_rows;

        var probe: KeepProbe = .{ .g = &g, .gpa = gpa };
        defer probe.deinit();
        var tok: usize = 0;
        for (dg.pass_tokens, 0..) |s_len, p| {
            defer g.reset();
            const pos = g.keep(try g.arange(@floatFromInt(tok), @floatFromInt(tok + s_len), 1, .int32));
            defer g.release(pos);
            try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "p{d}.positions", .{p}), pos);
            const rows = try a.alloc(u8, s_len * row_bytes);
            for (0..s_len) |j| {
                const off = embed_t.begin + @as(u64, dg.ids[tok + j]) * row_bytes;
                const dst = rows[j * row_bytes ..][0..row_bytes];
                if (std.c.pread(efd, dst.ptr, dst.len, @intCast(off)) != @as(isize, @intCast(dst.len))) return error.ShortRead;
            }
            var et = embed_t;
            et.rank = 3;
            et.shape = @splat(0);
            et.shape[0] = 1;
            et.shape[1] = s_len;
            et.shape[2] = c.hidden_size;
            const emb = try g.adopt(try arrayFrom(rows, et));
            try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "p{d}.embed", .{p}), emb);
            const e0 = try Tr.expandEmbedding(&g, &c, emb);
            // What the next layer reads survives the per-layer reset: the stream, the
            // shared selection masks (the caches keep their own arrays).
            var cur: Tr.Out = .{ .h = g.keep(e0.h), .pre_mix = g.keep(e0.pre_mix) };
            defer {
                g.release(cur.h);
                g.release(cur.pre_mix);
            }
            var shared: Tr.Share = .{};
            var masks: [3]?mlx.mlx_array = .{ null, null, null };
            defer for (masks) |m| if (m) |x| g.release(x);
            var eng_rows: []i64 = &.{};
            if (eng) |*e| {
                eng_rows = try a.alloc(i64, s_len * e.hashing.n_layers * e.hashing.cols());
                try e.state.advance(gpa, &e.hashing, e.map, dg.ids[tok .. tok + s_len], null, eng_rows);
            }
            for (dg.layers, 0..) |l, i| {
                probe.clear();
                const pre = try std.fmt.allocPrint(a, "p{d}.L{d}.", .{ p, l });
                if (eng) |*e| if (e.slot(l)) |li| {
                    const erows = try e.rowsFor(&g, a, eng_rows, li, s_len);
                    const eh = g.keep(try Tr.engramApply(&g, &c, try e.resident(l), cur.h, erows));
                    g.release(cur.h);
                    cur.h = eh;
                    try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "{s}engram.out", .{pre}), cur.h);
                };
                try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "{s}in.h", .{pre}), cur.h);
                try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "{s}in.pre_mix", .{pre}), cur.pre_mix);
                const out = try Tr.layer(&g, &probe, &c, &rt, .{}, c.layers[l], &lws[i], inv[i], cur.h, cur.pre_mix, pos, &caches[i], &shared, stand_in);
                const vec = mlx.mlx_vector_array_new_data(probe.arrays.items.ptr, probe.arrays.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                for (probe.names.items, probe.arrays.items) |name, arr| {
                    try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "{s}{s}", .{ pre, name }), arr);
                }
                const next: Tr.Out = .{ .h = g.keep(out.h), .pre_mix = g.keep(out.pre_mix) };
                g.release(cur.h);
                g.release(cur.pre_mix);
                cur = next;
                persist(&g, &shared.topk_mask, &masks[0]);
                persist(&g, &shared.candidates, &masks[1]);
                persist(&g, &shared.win_mask, &masks[2]);
                g.reset();
                g.clearCache();
                if (p == 0) {
                    var act: usize = 0;
                    var peak: usize = 0;
                    _ = mlx.mlx_get_active_memory(&act);
                    _ = mlx.mlx_get_peak_memory(&peak);
                    std.debug.print("dsv41 chain: p0 L{d} done; MLX active {d:.2} GiB, peak {d:.2} GiB\n", .{ l, @as(f64, @floatFromInt(act)) / (1 << 30), @as(f64, @floatFromInt(peak)) / (1 << 30) });
                }
            }
            const norm_w = try g.adopt(try arrayFrom(norm_bytes, norm_t));
            const fin = try Tr.finalNorm(&g, &c, cur.h, cur.pre_mix, norm_w);
            try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "p{d}.final.h", .{p}), fin);
            const head_w = try g.adopt(try arrayFrom(head_bytes, head_sub));
            try self.checkDigest(&g, &dg, try std.fmt.allocPrint(a, "p{d}.head.logits", .{p}), try Tr.head(&g, &rt, fin, .{ .dense = head_w }));
            for (caches) |*cc| cc.advance(@intCast(s_len));
            tok += s_len;
        }
        return self;
    }

    fn reported(self: *const Runner, key: []const u8) bool {
        for (self.results.items) |r| {
            if (std.mem.eql(u8, r.key, key)) return true;
        }
        return false;
    }

    fn dumpArray(self: *Runner, g: *ops.MlxOps, dump: *const v41.Checkpoint, prefix: []const u8, stage: []const u8) !mlx.mlx_array {
        const a = self.arena.allocator();
        const key = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, stage });
        const t = dump.tensors.get(key) orelse return error.TensorMissing;
        return g.adopt(try arrayFrom(try v41.readTensor(a, dump, key), t));
    }

    /// One JSON object per stage, then a verdict line.
    pub fn writeReport(self: *const Runner, io: std.Io, path: []const u8, dump_path: []const u8) !void {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(self.gpa);
        try out.print(self.gpa, "{{\"format\":\"mlx-serve-dsv41-parity-report-v1\",\"dump\":\"{s}\",\"stages\":[\n", .{dump_path});
        for (self.results.items, 0..) |r, i| {
            try out.print(self.gpa, "{s}{{\"key\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",\n", r.key, @tagName(r.dtype) });
            for (r.shape[0..r.rank], 0..) |d, k| try out.print(self.gpa, "{s}{d}", .{ if (k == 0) "" else ",", d });
            try out.print(self.gpa, "],\"n\":{d},\"mismatch\":{d},\"max_ulp\":{d},\"max_abs\":{f},\"first\":", .{ r.diff.n, r.diff.mismatches, r.diff.max_ulp, jsonNum(r.diff.max_abs) });
            if (r.diff.first) |f| try out.print(self.gpa, "{d},\"first_zig\":{f},\"first_py\":{f}", .{ f, jsonNum(r.diff.first_a), jsonNum(r.diff.first_b) }) else try out.appendSlice(self.gpa, "null");
            try out.print(self.gpa, ",\"note\":\"{s}\"}}", .{r.note});
        }
        const fm = self.firstMismatch();
        try out.print(self.gpa, "\n],\"verdict\":\"{s}\",\"first_mismatch\":\"{s}\"}}\n", .{ if (fm == null) "IDENTICAL" else "DIFF", if (fm) |r| r.key else "" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
    }
};

const testing = std.testing;

test "dsv41 parity: word comparison counts mismatches and ulps per dtype" {
    const f = [_]f32{ 1.0, -2.0, 0.0, 3.5 };
    var g2 = f;
    g2[1] = @bitCast(@as(u32, @bitCast(g2[1])) + 3); // 3 ulp further from zero
    g2[2] = -0.0; // +0 vs -0: one word apart on the ordered line
    const d = try compareWords(.F32, std.mem.sliceAsBytes(&f), std.mem.sliceAsBytes(&g2));
    try testing.expectEqual(@as(u64, 4), d.n);
    try testing.expectEqual(@as(u64, 2), d.mismatches);
    try testing.expectEqual(@as(u64, 3), d.max_ulp);
    try testing.expectEqual(@as(?u64, 1), d.first);
    try testing.expectEqual(@as(f64, -2.0), d.first_a);
    const same = try compareWords(.F32, std.mem.sliceAsBytes(&f), std.mem.sliceAsBytes(&f));
    try testing.expectEqual(@as(u64, 0), same.mismatches);
    // bf16 words: 0x3F80 (1.0) vs 0x3F81 is one ulp; sign flips cross zero.
    const a16 = [_]u16{ 0x3F80, 0xBF80, 0x0001 };
    const b16 = [_]u16{ 0x3F81, 0xBF80, 0x8001 };
    const d16 = try compareWords(.BF16, std.mem.sliceAsBytes(&a16), std.mem.sliceAsBytes(&b16));
    try testing.expectEqual(@as(u64, 2), d16.mismatches);
    try testing.expectEqual(@as(u64, 3), d16.max_ulp); // -min .. -0 .. +0 .. +min
    const ia = [_]i32{ 5, -1, 7 };
    const ib = [_]i32{ 5, 2, 7 };
    const di = try compareWords(.I32, std.mem.sliceAsBytes(&ia), std.mem.sliceAsBytes(&ib));
    try testing.expectEqual(@as(u64, 1), di.mismatches);
    try testing.expectEqual(@as(u64, 3), di.max_ulp);
    const ba = [_]u8{ 1, 0, 1 };
    const bb = [_]u8{ 1, 1, 1 };
    try testing.expectEqual(@as(u64, 1), (try compareWords(.BOOL, &ba, &bb)).mismatches);
    try testing.expectError(error.LengthMismatch, compareWords(.F32, std.mem.sliceAsBytes(&f), std.mem.sliceAsBytes(f[0..3])));
}

test "dsv41 parity: the report is JSON naming the first differing stage" {
    var r: Runner = .{ .gpa = testing.allocator, .arena = std.heap.ArenaAllocator.init(testing.allocator) };
    defer r.deinit();
    const a = r.arena.allocator();
    var t: v41.Checkpoint.Tensor = .{ .shard = 0, .dtype = .F32, .shape = @splat(0), .rank = 2, .begin = 0, .end = 12 };
    t.shape[0] = 1;
    t.shape[1] = 3;
    try r.record("p0.L0.attn.q", t, .{ .n = 3 }, "");
    try r.record("p0.L0.attn.o", t, .{ .n = 3, .mismatches = 1, .max_ulp = 2, .max_abs = std.math.nan(f64), .first = 1, .first_a = 1.5, .first_b = std.math.inf(f64) }, "");
    try r.record("p0.L2.gate.indices", t, .{}, "zig I32 { 1, 3 } vs dump F32 { 1, 3 }");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try std.fmt.allocPrint(a, "{s}/report.json", .{buf[0..try tmp.dir.realPath(testing.io, &buf)]});
    try r.writeReport(testing.io, path, "/tmp/dump.safetensors");
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 20));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    try testing.expectEqualStrings("DIFF", parsed.object.get("verdict").?.string);
    try testing.expectEqualStrings("p0.L0.attn.o", parsed.object.get("first_mismatch").?.string);
    const stages = parsed.object.get("stages").?.array.items;
    try testing.expectEqual(@as(usize, 3), stages.len);
    try testing.expectEqualStrings("nan", stages[1].object.get("max_abs").?.string);
    try testing.expectEqual(@as(i64, 1), stages[1].object.get("first").?.integer);
}

// GPU only: DSV41_PARITY_DUMP=<dump.safetensors> DSV41_BANK=<bank> _GPU_WINDOW_LOCKED=1
// [DSV41_PARITY_DEVICE=gpu (the dump's device)] [DSV41_ENGRAM_TOKEN_MAP=<u32 map> (dumps made with --engram)]
// [DSV41_PARITY_REPORT=<json>] [DSV41_PARITY_REPORT_ONLY=1]
// GPU only: DSV41_PARITY_DIGESTS=<dump>.digests.json DSV41_BANK=<bank> _GPU_WINDOW_LOCKED=1
// [DSV41_PARITY_DEVICE=gpu] [DSV41_ENGRAM_TOKEN_MAP=<u32 map>] [DSV41_PARITY_REPORT=<json>] [DSV41_PARITY_REPORT_ONLY=1]
test "dsv41 parity: the chained Zig trunk hashes every stage as the Python stock path does" {
    const digests = std.mem.span(std.c.getenv("DSV41_PARITY_DIGESTS") orelse return error.SkipZigTest);
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpu = if (std.c.getenv("DSV41_PARITY_DEVICE")) |d| std.mem.eql(u8, std.mem.span(d), "gpu") else false;
    var r = try Runner.runChained(testing.allocator, testing.io, digests, bank, gpu);
    defer r.deinit();
    try finish(&r, digests);
}

/// Report, summary lines, and the verdict (a difference fails unless REPORT_ONLY).
fn finish(r: *Runner, source: []const u8) !void {
    const report = if (std.c.getenv("DSV41_PARITY_REPORT")) |p| std.mem.span(p) else try std.fmt.allocPrint(r.arena.allocator(), "{s}.zig-report.json", .{source});
    try r.writeReport(testing.io, report, source);
    var mism: usize = 0;
    for (r.results.items) |x| {
        if (x.diff.mismatches > 0 or x.note.len > 0) mism += 1;
    }
    std.debug.print("dsv41 parity: {d} stages compared, {d} differ; report {s}\n", .{ r.results.items.len, mism, report });
    if (r.firstMismatch()) |f| {
        std.debug.print("dsv41 parity: FIRST DIFF {s}: {d}/{d} words, max {d} ulp, max |d| {e} {s}\n", .{ f.key, f.diff.mismatches, f.diff.n, f.diff.max_ulp, f.diff.max_abs, f.note });
        if (std.c.getenv("DSV41_PARITY_REPORT_ONLY") == null) return error.ParityMismatch;
    } else std.debug.print("dsv41 parity: IDENTICAL\n", .{});
}

test "dsv41 parity: the Zig trunk equals the Python stock path word for word" {
    const dump_path = std.mem.span(std.c.getenv("DSV41_PARITY_DUMP") orelse return error.SkipZigTest);
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    // Any MLX array constructs the Metal device: never outside the lock.
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpu = if (std.c.getenv("DSV41_PARITY_DEVICE")) |d| std.mem.eql(u8, std.mem.span(d), "gpu") else false;
    var r = try Runner.run(testing.allocator, testing.io, dump_path, bank, gpu);
    defer r.deinit();
    try finish(&r, dump_path);
}

// GPU only: DSV41_PARITY2_DUMP=<dump_dsv41_parity_w2.py output> DSV41_BANK=<bank>
// DSV41_ENGRAM_TOKEN_MAP=<converter output, with its .json> _GPU_WINDOW_LOCKED=1
// [DSV41_PARITY_DEVICE=gpu] [DSV41_PARITY_REPORT=<json>] [DSV41_PARITY_REPORT_ONLY=1]
test "dsv41 parity: the tier routes, window ring, chunked passes and trims equal the Python run" {
    const dump_path = std.mem.span(std.c.getenv("DSV41_PARITY2_DUMP") orelse return error.SkipZigTest);
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.EngramTokenMapRequired);
    const gpu = if (std.c.getenv("DSV41_PARITY_DEVICE")) |d| std.mem.eql(u8, std.mem.span(d), "gpu") else false;
    var r = try Runner.runRoutes(testing.allocator, testing.io, dump_path, bank, map_path, gpu);
    defer r.deinit();
    try finish(&r, dump_path);
}

test "dsv41 parity: the window-2 schedule metadata parses to passes and trims" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sch = try Schedule.parse(arena.allocator(), "64,64,64,8,1,1,1,6,1,1", "7:3");
    try testing.expectEqual(@as(usize, 10), sch.pass_tokens.len);
    try testing.expectEqual(@as(u32, 3), sch.trim_after[7]);
    try testing.expectEqual(@as(u32, 0), sch.trim_after[6]);
    try testing.expectError(error.ScheduleSyntax, Schedule.parse(arena.allocator(), "1,1", "5:3"));
}

// GPU only: DSV41_BANK=<bank> _GPU_WINDOW_LOCKED=1. mlx-serve's own
// loader maps all 49 shards lazily (no tensor data is read); every resident of
// the resident spec must bind with the declared dtype and shape.
test "dsv41 weights: every resident binds through model.loadWeights with its spec dtype and shape" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try v41.Config.load(testing.allocator, testing.io, bank, &diag);
    var w = try model.loadWeightsOpt(testing.io, testing.allocator, bank, bank_load);
    defer w.deinit();
    try v41.checkLoaded(&w, try v41.residentSpec(arena.allocator(), &c), &diag);
    for (0..c.n_layers) |l| _ = try bindLayer(&w, c.layers[l], @intCast(l));
    std.debug.print("dsv41 weights: {d} arrays loaded, every resident bound\n", .{w.count()});
}

// Declared last so it runs after the other dsv41 tests: only creating a Metal
// device maps a GPU driver bundle (AGXMetal*), so none may be mapped here.
test "dsv41 host: the host-only tests created no Metal device" {
    if (std.c.getenv("_GPU_WINDOW_LOCKED") != null) return error.SkipZigTest; // window runs build MLX arrays
    try @import("sdk").testing.expectNoDevice();
}

// GPU only (_GPU_WINDOW_LOCKED; MLX on the CPU stream): DSV41_BANK=<bank> DSV41_HEAD_FIXTURE_DUMP=<a parity dump with p*.final.h and
// p*.head.logits over the head's first `head_rows` rows>. HEAD_MODE mxfp8 is LOSSY: this reports, against the dump's
// own logits (the f32 head of the stock levers), the max / rms error and the top-1 agreement of the bf16 head, (a) MLX's mxfp8 quantized matmul (today's path) and (b)
// the RCPROJ route's numerics on the host (the dequantized codes, x cast to bf16, an f32 product, the bf16 output); the
// bar is the harness (the bf16 head reproduces the dump) and finite values, not identity.
test "dsv41 parity: the mxfp8 head against the bf16 head on the fixture's logits (max, rms, top-1)" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const dump_path = std.mem.span(std.c.getenv("DSV41_HEAD_FIXTURE_DUMP") orelse return error.SkipZigTest);
    // Any MLX array constructs the Metal device, the CPU stream's too: never outside the lock.
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 parity head: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, testing.io, bank, &diag);
    var ck = try v41.Checkpoint.openIndexed(gpa, testing.io, bank, &diag);
    defer ck.deinit();
    var dump = try v41.Checkpoint.openFile(gpa, dump_path, &diag);
    defer dump.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var g = try ops.MlxOps.init(gpa, cpu);
    defer g.deinit();
    const Tr = graph.Trunk(ops.MlxOps);
    // The head's first `rows` rows, read past the page cache.
    const lt0 = dump.tensors.get("p0.head.logits") orelse return error.TensorMissing;
    const rows: u64 = lt0.shape[lt0.rank - 1];
    const head_t = ck.tensors.get("head.weight") orelse return error.TensorMissing;
    var sub = head_t;
    sub.shape[0] = rows;
    sub.end = sub.begin + rows * @as(u64, c.hidden_size) * 2;
    const buf = try a.alloc(u8, @intCast(sub.end - sub.begin));
    const hfd = std.c.open((try ck.shardPath(a, head_t.shard)).ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (hfd < 0) return error.ShardMissing;
    defer _ = std.c.close(hfd);
    _ = std.c.fcntl(hfd, 48, @as(c_int, 1)); // F_NOCACHE
    if (std.c.pread(hfd, buf.ptr, buf.len, @intCast(sub.begin)) != @as(isize, @intCast(buf.len))) return error.ShortRead;
    const dense = try g.adopt(try arrayFrom(buf, sub));
    const q = try Tr.quantizeHead(&g, dense);
    const deq = try g.dequantize(q.w, q.s, .mxfp8);
    const deq_t = try g.transpose(try g.astype(deq, .float32));
    try g.evalAll(&.{ q.w, q.s, deq_t });
    const r32: graph.Routes = .{ .head = .f32 };
    const r16: graph.Routes = .{ .head = .bf16 };
    const r8: graph.Routes = .{ .head = .mxfp8 };
    const Acc = struct {
        max_abs: f64 = 0,
        se: f64 = 0,
        sr: f64 = 0,
        top1: u64 = 0,
        fn add(s: *@This(), want: []const f32, got: []const f32, n_cols: usize) void {
            var r: usize = 0;
            while (r * n_cols < want.len) : (r += 1) {
                const w = want[r * n_cols ..][0..n_cols];
                const o = got[r * n_cols ..][0..n_cols];
                for (w, o) |x, y| {
                    s.max_abs = @max(s.max_abs, @abs(@as(f64, x) - y));
                    s.se += (@as(f64, x) - y) * (@as(f64, x) - y);
                    s.sr += @as(f64, x) * x;
                }
                s.top1 += @intFromBool(std.mem.indexOfMax(f32, w) == std.mem.indexOfMax(f32, o));
            }
        }
    };
    var acc: [4]Acc = .{ .{}, .{}, .{}, .{} };
    var n_rows: u64 = 0;
    var p: usize = 0;
    while (true) : (p += 1) {
        const fk = try std.fmt.allocPrint(a, "p{d}.final.h", .{p});
        const lk = try std.fmt.allocPrint(a, "p{d}.head.logits", .{p});
        const ft = dump.tensors.get(fk) orelse break;
        if (dump.tensors.get(lk) == null) return error.TensorMissing;
        const m = g.mark();
        defer g.resetTo(m);
        const fin = try g.adopt(try arrayFrom(try v41.readTensor(a, &dump, fk), ft));
        const want_bytes = try v41.readTensor(a, &dump, lk);
        const want = try a.alloc(f32, want_bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(want), want_bytes);
        const outs = [4]ops.MlxOps.T{
            try Tr.head(&g, &r32, fin, .{ .dense = dense }),
            try Tr.head(&g, &r16, fin, .{ .dense = dense }),
            try Tr.head(&g, &r8, fin, .{ .mxfp8 = q }),
            try g.astype(try g.astype(try g.matmul(try g.astype(try g.astype(fin, .bfloat16), .float32), deq_t), .bfloat16), .float32),
        };
        for (outs, &acc) |o, *s| {
            const got = try a.alloc(f32, want.len);
            _ = try g.hostF32(o, got);
            for (got) |v| try testing.expect(std.math.isFinite(v));
            s.add(want, got, @intCast(rows));
        }
        n_rows += want.len / rows;
    }
    try testing.expect(n_rows > 0);
    const names = [_][]const u8{ "f32_head", "bf16_head", "mxfp8_mlx_qmm", "mxfp8_rcproj_host" };
    for (names, acc) |nm, s| std.debug.print("NATIVE HEAD_MXFP8_FIXTURE {{\"path\": \"{s}\", \"rows\": {d}, \"cols\": {d}, \"max_abs\": {e:.4}, \"rms_rel\": {e:.4}, \"top1_agree\": {d}}}\n", .{ nm, n_rows, rows, s.max_abs, @sqrt(s.se / s.sr), s.top1 });
    // The harness: the dump's own head (levers none: f32 x over the bf16 weight) reproduced on the same rows.
    try testing.expectEqual(n_rows, acc[0].top1);
    try testing.expect(@sqrt(acc[0].se / acc[0].sr) < 1e-4);
}
