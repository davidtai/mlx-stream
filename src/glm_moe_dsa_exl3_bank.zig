//! GLM-5.3's EXL3 expert bank: the pack's `experts.bin` and its `expert-manifest-exl3-v1.json` (the contract the
//! converter `scripts/convert_glm_exl3_bank.py` writes, docs/glm53-exl3-pack-format.md) as a bank module for the
//! expert stream (`sdk_ext.expert.assertBank`).
//!   - A stream layer is one bank layer: a routed layer's experts at one K, its K3 bank layer at `2 r`, its K4 bank
//!     layer at `2 r + 1`. The MTP layer's bank layers are the draft lane's (`mtp/`), outside the stream.
//!   - A routed id is the model's expert id; a bank layer routes only the experts it holds. Its record is the expert's
//!     `tp` minis (one tensor-parallel rank each: nine segments, code I16 = the trellis, rout F16 = svh, rin F16 =
//!     suh per projection), adjacent in the file and read into adjacent rows of the slot arrays (`Layer.minis`).
//!   - The K3 and K4 bank layers' code rows differ: the stream keeps a transient scratch per geometry.
//! Every rule runs once in `Bank.open`, which also opens the sidecar the read pool reads through; a mismatch refuses
//! the whole bank by name. The slot arrays bind to sushi's EXL3 MoE (`glm_moe_dsa_exl3_quant`).

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("sdk_ext.zig");
const expert_io = sdk_ext.expert.io;
const quant = sdk_ext.quant;
const glm = @import("glm_moe_dsa.zig");
const affine = @import("glm_moe_dsa_bank.zig");
const sushi = @import("mlx_host").sushi_exl3;

pub const format = "mlx-stream-expert-manifest-exl3-v1";
pub const manifest_file = "expert-manifest-exl3-v1.json";
pub const record_alignment: u64 = 4096;
const max_manifest_bytes = 64 << 20;

// ── the bank contract (`sdk_ext.expert.assertBank`) ──

pub const Component = enum(u8) {
    gate_code,
    gate_rout,
    gate_rin,
    up_code,
    up_rout,
    up_rin,
    down_code,
    down_rout,
    down_rin,

    pub fn name(c: Component) []const u8 {
        return names[@intFromEnum(c)];
    }

    pub const names = [_][]const u8{ "gate_proj.code", "gate_proj.rout", "gate_proj.rin", "up_proj.code", "up_proj.rout", "up_proj.rin", "down_proj.code", "down_proj.rout", "down_proj.rin" };
};
pub const n_components = 9;
pub const gu_components = 6;
pub const Records = expert_io.Records(n_components, gu_components);
pub const routed_top_k = glm.routed_top_k;
/// The K3 and the K4 bank layers keep their own transient scratch (`sdk_ext.expert.stream`).
pub const per_geometry_transients = true;
/// Stream layers per routed layer: its K3 bank layer, then its K4 one.
pub const banks_per_layer: u32 = 2;

/// The trellis is I16 in the source and read as uint16 (sushi's dtype for the same bytes); rout and rin are F16.
pub const Dtype = enum { I16, F16 };

pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) {
        .I16 => .uint16,
        .F16 => .float16,
    };
}

pub const Segment = struct {
    /// Relative to the mini's record start.
    offset: u64,
    length: u64,
    dtype: Dtype,
    shape: [3]u64 = @splat(0),
    rank: u8,
};

pub const Layer = struct {
    /// The model layer whose experts at `k` this bank layer holds.
    model_layer: u32,
    k: u32,
    /// One mini's record: its stride in the file and its logical bytes (the nine segments).
    record_bytes: u64,
    logical_bytes: u64,
    base_offset: u64,
    /// A routed id's records: the expert's tensor-parallel minis.
    minis: u32,
    /// The experts this bank layer holds: the most slot rows it takes.
    n_held: u32,
    segments: [n_components]Segment,
};

/// One projection's slot arrays as sushi's EXL3 MoE binds them (`sushi.Proj`): trellis uint16 `[minis, in/16,
/// out/16, 16K]`, suh float16 `[minis, in]`, svh float16 `[minis, out]`.
pub fn Proj(comptime T: type) type {
    return struct { trellis: T, suh: T, svh: T };
}
pub const Arrays = Proj(mlx.mlx_array);
pub const BankArrays = quant.BankArrays(Arrays);

/// The slot arrays in component order: code = trellis, rout = svh (the output side), rin = suh (the input side).
pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .trellis = x[0], .svh = x[1], .suh = x[2] },
        .up = .{ .trellis = x[3], .svh = x[4], .suh = x[5] },
        .down = .{ .trellis = x[6], .svh = x[7], .suh = x[8] },
    };
}

pub const Spans = affine.Spans;
pub const Geometry = affine.Geometry;
pub const Diag = @import("sdk").Diag;

/// Segment `ci`'s shape (gate / up: in = hidden, out = mini; down: in = mini, out = hidden; code = trellis
/// `[in/16, out/16, 16K]`, rout = svh `[out]`, rin = suh `[in]`) and its rank.
pub fn segmentShape(ci: usize, hidden: u64, mini: u64, k: u64) struct { shape: [3]u64, rank: u8 } {
    const down = ci >= 6;
    const in_d = if (down) mini else hidden;
    const out_d = if (down) hidden else mini;
    return switch (ci % 3) {
        0 => .{ .shape = .{ in_d / 16, out_d / 16, 16 * k }, .rank = 3 },
        1 => .{ .shape = .{ out_d, 0, 0 }, .rank = 1 },
        else => .{ .shape = .{ in_d, 0, 0 }, .rank = 1 },
    };
}

/// One mini's nine segments at K `k`, back to back, and their bytes.
pub fn miniSegments(hidden: u64, mini: u64, k: u64) struct { segs: [n_components]Segment, logical: u64 } {
    var segs: [n_components]Segment = undefined;
    var off: u64 = 0;
    for (&segs, 0..) |*sg, ci| {
        const sh = segmentShape(ci, hidden, mini, k);
        var n: u64 = 2;
        for (sh.shape[0..sh.rank]) |x| n *= x;
        sg.* = .{ .offset = off, .length = n, .dtype = if (ci % 3 == 0) .I16 else .F16, .shape = sh.shape, .rank = sh.rank };
        off += n;
    }
    return .{ .segs = segs, .logical = off };
}

pub const Refusal = error{
    BankDirNotAbsolute,
    ManifestMissing,
    ManifestSyntax,
    ManifestFormat,
    ModelType,
    QuantNotImplemented,
    DimsMismatch,
    LayerGeometry,
    ExpertMap,
    SidecarGeometry,
    SidecarMissing,
    SidecarOpen,
    RecordCount,
    RecordGeometry,
    RecordDuplicate,
    RecordSha256,
    ParityNotPassed,
};

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

const SegJson = struct { component: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 };
const LayerJson = struct { bank_layer: u64, layer: u64, k: u64, mtp: bool, n_minis: u64, record_bytes: u64, logical_bytes: u64, base_offset: u64, experts: []const u64, segments: []const SegJson };
const RecordJson = struct { bank_layer: u64, mini: u64, expert: u64, rank: u64, sidecar_offset: u64, sha256: []const u8 };
const ManifestJson = struct {
    format: []const u8,
    model_type: []const u8,
    quantization: struct { mode: []const u8, codebook: []const u8, codebook_multiplier: u64, tp_ranks: u64 },
    dims: struct { hidden: u64, inter: u64, mini_inter: u64, n_experts: u64 },
    components: []const []const u8,
    layers: []const LayerJson,
    experts: std.json.ArrayHashMap([]const [2]u64),
    sidecar: struct { file: []const u8, alignment: u64, size: u64 },
    records: []const RecordJson,
    parity: struct { all_pass: bool },
};

/// No local index: the expert is the layer's other bank layer's.
pub const no_local: u16 = std.math.maxInt(u16);

pub const Bank = struct {
    allocator: std.mem.Allocator,
    hidden: u64 = 0,
    inter: u64 = 0,
    mini_inter: u64 = 0,
    tp: u32 = 0,
    /// The routed ids' range: the model's experts.
    n_experts: u32 = 0,
    /// The stream layers: routed layer r's K3 bank layer at 2 r, its K4 bank layer at 2 r + 1.
    layers: []Layer = &.{},
    /// [stream layer * n_experts + expert]: the expert's index among its bank layer's experts (`no_local`: not there).
    local: []u16 = &.{},
    /// [routed layer * n_experts + expert]: the expert's bank layer of its routed layer (0: the first, 1: the second).
    side: []u1 = &.{},
    sidecar_path: [:0]const u8 = "",
    sidecar: expert_io.UncachedFd = .{ .fd = -1, .size = 0 },

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag) !Bank {
        return openWith(allocator, io, dir, c, diag, true);
    }

    /// The bank's geometry from its manifest alone (no sidecar descriptor), for the bill.
    pub fn geometry(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag) !Geometry {
        var b = try openWith(allocator, io, dir, c, diag, false);
        defer b.deinit();
        return b.geometryOf();
    }

    /// What the bill reads: the routed layers; the fill's units (`maxUnits`); one slot of every geometry as the
    /// widest record (each geometry's transient scratch holds `max_route_ids` of its own), so `n_layers x
    /// widest_record` bounds a unit's bytes (`rowsAt` rounds each bank layer's rows down); the widest mini range.
    pub fn geometryOf(self: *const Bank) Geometry {
        var g: Geometry = .{ .n_layers = @intCast(self.layers.len / banks_per_layer), .n_experts = self.maxUnits() };
        if (self.layers.len == 0) return g;
        for (self.layers[0..banks_per_layer]) |*l| g.widest_record += @as(u64, l.minis) * l.logical_bytes;
        for (self.layers) |*l| {
            const gu = l.segments[gu_components].offset;
            g.widest_span = @max(g.widest_span, @max(gu, l.logical_bytes - gu));
        }
        return g;
    }

    /// The fill's units: a unit is 1/`maxUnits` of every bank layer's experts.
    pub fn maxUnits(self: *const Bank) u32 {
        return self.n_experts / banks_per_layer;
    }

    /// Each stream layer's slot rows at `units`: its experts' share, rounded down (never above what the unit bills).
    pub fn rowsAt(self: *const Bank, units: u32, out: []u32) void {
        const d = self.maxUnits();
        for (self.layers, out) |*l, *r| r.* = @intCast(@min(@as(u64, units) * l.n_held / d, l.n_held));
    }

    pub fn deinit(self: *Bank) void {
        if (self.sidecar.fd >= 0) self.sidecar.close();
        self.allocator.free(self.layers);
        self.allocator.free(self.local);
        self.allocator.free(self.side);
        if (self.sidecar_path.len > 0) self.allocator.free(self.sidecar_path);
        self.* = undefined;
    }

    /// The stream layer that holds expert `e` of routed layer `routed`.
    pub fn streamLayer(self: *const Bank, routed: u32, e: u16) u32 {
        return banks_per_layer * routed + self.side[@as(usize, routed) * self.n_experts + e];
    }

    /// The expert's first mini in the sidecar (its minis follow at `record_bytes`).
    pub fn recordOffset(self: *const Bank, layer: u32, expert: u32) u64 {
        const l = &self.layers[layer];
        const loc = self.local[@as(usize, layer) * self.n_experts + expert];
        std.debug.assert(loc != no_local);
        return l.base_offset + @as(u64, loc) * l.minis * l.record_bytes;
    }

    /// The first mini's read ranges (the stream adds the stride per mini).
    pub fn spans(self: *const Bank, layer: u32, expert: u32) Spans {
        const off = self.recordOffset(layer, expert);
        return .{ .gu_offset = off, .down_offset = off + self.layers[layer].segments[gu_components].offset };
    }

    /// The bank's description for the EXL3 quant's claim (`sdk_ext.quant.BankPeek`), in `a`.
    pub fn peek(self: *const Bank, a: std.mem.Allocator) !quant.BankPeek {
        const q = try std.fmt.allocPrint(a, "{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"tp_ranks\":{d}}}", .{ sushi.format.MCG_MULT, self.tp });
        return .{ .quantization = try std.json.parseFromSliceLeaky(std.json.Value, a, q, .{}), .hidden = self.hidden, .inter = self.inter, .n_experts = self.n_experts, .n_layers = self.layers.len, .layers = &.{} };
    }

    fn openWith(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag, sidecar: bool) !Bank {
        if (!std.fs.path.isAbsolute(dir)) return refuse(diag, error.BankDirNotAbsolute, "bank dir \"{s}\" is not an absolute path", .{dir});
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const path = try std.fmt.allocPrint(a, "{s}/" ++ manifest_file, .{dir});
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(max_manifest_bytes)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.FileNotFound => return refuse(diag, error.ManifestMissing, manifest_file ++ ": not found in {s}", .{dir}),
            else => return refuse(diag, error.ManifestMissing, manifest_file ++ ": {s}", .{@errorName(e)}),
        };
        const m = std.json.parseFromSliceLeaky(ManifestJson, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ManifestSyntax, manifest_file ++ ": {s}", .{@errorName(e)}),
        };
        var bank: Bank = .{ .allocator = allocator };
        errdefer bank.deinit();
        const size = try bank.adopt(a, &m, c, diag);
        if (!sidecar) return bank;
        bank.sidecar_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir, m.sidecar.file }, 0);
        try bank.openSidecar(size, diag);
        return bank;
    }

    /// The manifest's rules against the model's config; returns the sidecar's bytes.
    fn adopt(self: *Bank, a: std.mem.Allocator, m: *const ManifestJson, c: *const glm.Config, diag: ?*Diag) !u64 {
        const gpa = self.allocator;
        // 1. The format.
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, manifest_file ++ ": format \"{s}\" is not " ++ format, .{m.format});
        if (!std.mem.eql(u8, m.model_type, glm.model_type)) return refuse(diag, error.ModelType, manifest_file ++ ": model_type \"{s}\" is not " ++ glm.model_type, .{m.model_type});
        // 2. The quantization and the dims against config.json.
        const q = m.quantization;
        if (!std.mem.eql(u8, q.mode, "exl3") or !std.mem.eql(u8, q.codebook, "mcg") or q.codebook_multiplier != sushi.format.MCG_MULT or q.tp_ranks == 0)
            return refuse(diag, error.QuantNotImplemented, manifest_file ++ ": quantization {s} / {s} / multiplier {d} / {d} ranks (this build reads exl3 mcg at multiplier {d})", .{ q.mode, q.codebook, q.codebook_multiplier, q.tp_ranks, sushi.format.MCG_MULT });
        const d = m.dims;
        if (d.hidden != c.hidden_size or d.inter != c.moe_intermediate_size or d.n_experts != c.n_routed_experts or d.mini_inter * q.tp_ranks != d.inter or d.n_experts % banks_per_layer != 0)
            return refuse(diag, error.DimsMismatch, manifest_file ++ ": dims hidden {d} / inter {d} = {d} x {d} / {d} experts, config.json {d} / {d} / {d}", .{ d.hidden, d.inter, q.tp_ranks, d.mini_inter, d.n_experts, c.hidden_size, c.moe_intermediate_size, c.n_routed_experts });
        if (m.components.len != n_components) return refuse(diag, error.LayerGeometry, manifest_file ++ ": {d} components, the record has {d}", .{ m.components.len, n_components });
        for (m.components, Component.names) |x, y| if (!std.mem.eql(u8, x, y)) return refuse(diag, error.LayerGeometry, manifest_file ++ ": component \"{s}\" where the format has \"{s}\"", .{ x, y });
        self.hidden = d.hidden;
        self.inter = d.inter;
        self.mini_inter = d.mini_inter;
        self.tp = @intCast(q.tp_ranks);
        self.n_experts = @intCast(d.n_experts);
        const ne: usize = self.n_experts;
        // 3. The bank layers: every one's segments as the format derives them, back to back, the bases contiguous;
        // the routed layers' first (two per layer, K ascending, their experts each layer's once), then the MTP layer's.
        var sparse: [glm.max_layers]u32 = undefined;
        const routed = c.sparseLayers(&sparse);
        const n_stream = routed.len * banks_per_layer;
        if (m.layers.len < n_stream) return refuse(diag, error.LayerGeometry, manifest_file ++ ": {d} bank layers, the {d} routed layers need {d}", .{ m.layers.len, routed.len, n_stream });
        self.layers = try gpa.alloc(Layer, n_stream);
        self.local = try gpa.alloc(u16, n_stream * ne);
        @memset(self.local, no_local);
        self.side = try gpa.alloc(u1, routed.len * ne);
        var base: u64 = 0;
        var n_minis: u64 = 0;
        for (m.layers, 0..) |lj, i| {
            const trunk = i < n_stream;
            const want_layer: u64 = if (trunk) routed[i / banks_per_layer] else c.n_layers;
            if (lj.bank_layer != i or lj.layer != want_layer or lj.mtp == trunk)
                return refuse(diag, error.LayerGeometry, manifest_file ++ ": bank layer entry {d} is bank layer {d} of layer {d} (mtp {}), want layer {d} (mtp {})", .{ i, lj.bank_layer, lj.layer, lj.mtp, want_layer, !trunk });
            if (lj.k == 0 or lj.k > 8 or sushi.format.kFromPackedDim(@intCast(16 * lj.k)) == null) return refuse(diag, error.QuantNotImplemented, manifest_file ++ ": bank layer {d} at K {d}", .{ i, lj.k });
            if (lj.n_minis != q.tp_ranks * lj.experts.len or lj.base_offset != base)
                return refuse(diag, error.LayerGeometry, manifest_file ++ ": bank layer {d}: {d} minis for {d} experts, base {d} (want {d})", .{ i, lj.n_minis, lj.experts.len, lj.base_offset, base });
            const want = miniSegments(d.hidden, d.mini_inter, lj.k);
            if (lj.segments.len != n_components) return refuse(diag, error.LayerGeometry, manifest_file ++ ": bank layer {d} has {d} segments", .{ i, lj.segments.len });
            for (lj.segments, want.segs, 0..) |sj, w, ci| {
                const dt = std.meta.stringToEnum(Dtype, sj.dtype);
                if (!std.mem.eql(u8, sj.component, Component.names[ci]) or dt == null or dt.? != w.dtype or sj.offset != w.offset or sj.length != w.length or !std.mem.eql(u64, sj.shape, w.shape[0..w.rank]))
                    return refuse(diag, error.LayerGeometry, manifest_file ++ ": bank layer {d} segment {s}: {s} {any} at {d}, {d} B (want {s} {any} at {d}, {d} B)", .{ i, sj.component, sj.dtype, sj.shape, sj.offset, sj.length, @tagName(w.dtype), w.shape[0..w.rank], w.offset, w.length });
            }
            if (lj.logical_bytes != want.logical or lj.record_bytes != std.mem.alignForward(u64, want.logical, record_alignment))
                return refuse(diag, error.LayerGeometry, manifest_file ++ ": bank layer {d}: logical {d} / record {d} B, want {d} / {d}", .{ i, lj.logical_bytes, lj.record_bytes, want.logical, std.mem.alignForward(u64, want.logical, record_alignment) });
            for (lj.experts, 0..) |e, j| if (e >= ne or (j > 0 and e <= lj.experts[j - 1]))
                return refuse(diag, error.ExpertMap, manifest_file ++ ": bank layer {d}: experts not ascending ids below {d} at {d}", .{ i, ne, j });
            if (trunk) {
                const r = i / banks_per_layer;
                const sd: u1 = @intCast(i % banks_per_layer);
                if (sd == 1 and lj.k <= m.layers[i - 1].k) return refuse(diag, error.LayerGeometry, manifest_file ++ ": layer {d}'s bank layers at K {d} then {d} (K ascending)", .{ lj.layer, m.layers[i - 1].k, lj.k });
                for (lj.experts, 0..) |e, j| {
                    if (sd == 1 and self.local[(i - 1) * ne + e] != no_local) return refuse(diag, error.ExpertMap, manifest_file ++ ": layer {d}: expert {d} in both bank layers", .{ lj.layer, e });
                    self.local[i * ne + e] = @intCast(j);
                    self.side[r * ne + e] = sd;
                }
                if (sd == 1) for (0..ne) |e| if (self.local[(i - 1) * ne + e] == no_local and self.local[i * ne + e] == no_local)
                    return refuse(diag, error.ExpertMap, manifest_file ++ ": layer {d}: expert {d} in neither bank layer", .{ lj.layer, e });
                self.layers[i] = .{ .model_layer = @intCast(lj.layer), .k = @intCast(lj.k), .record_bytes = lj.record_bytes, .logical_bytes = lj.logical_bytes, .base_offset = lj.base_offset, .minis = self.tp, .n_held = @intCast(lj.experts.len), .segments = want.segs };
            }
            base += lj.n_minis * lj.record_bytes;
            n_minis += lj.n_minis;
        }
        // 4. The experts map: each routed layer's experts at their bank layer's K and local index.
        for (routed, 0..) |ml, r| {
            var kb: [16]u8 = undefined;
            const key = std.fmt.bufPrint(&kb, "{d}", .{ml}) catch unreachable;
            const ex = m.experts.map.get(key) orelse return refuse(diag, error.ExpertMap, manifest_file ++ ": experts has no layer {s}", .{key});
            if (ex.len != ne) return refuse(diag, error.ExpertMap, manifest_file ++ ": experts[{s}] lists {d} experts, want {d}", .{ key, ex.len, ne });
            for (ex, 0..) |kl, e| {
                const sl = banks_per_layer * r + self.side[r * ne + e];
                if (kl[0] != self.layers[sl].k or kl[1] != self.local[sl * ne + e])
                    return refuse(diag, error.ExpertMap, manifest_file ++ ": experts[{s}][{d}] = [{d}, {d}], its bank layer has it at [{d}, {d}]", .{ key, e, kl[0], kl[1], self.layers[sl].k, self.local[sl * ne + e] });
            }
        }
        // 5. The sidecar.
        const sc = m.sidecar;
        if (!plainName(sc.file)) return refuse(diag, error.SidecarGeometry, manifest_file ++ ": sidecar file \"{s}\" is not a plain file name", .{sc.file});
        if (sc.alignment != record_alignment or sc.size != base) return refuse(diag, error.SidecarGeometry, manifest_file ++ ": sidecar alignment {d}, size {d} (want {d} and the bank layers' {d})", .{ sc.alignment, sc.size, record_alignment, base });
        // 6. The records: every (bank layer, mini) once, at its offset, its expert and rank the layout's.
        if (m.records.len != n_minis) return refuse(diag, error.RecordCount, manifest_file ++ ": {d} records, the bank layers have {d} minis", .{ m.records.len, n_minis });
        const seen = try a.alloc(bool, @intCast(n_minis));
        @memset(seen, false);
        const first = try a.alloc(u64, m.layers.len);
        var at: u64 = 0;
        for (m.layers, first) |lj, *f| {
            f.* = at;
            at += lj.n_minis;
        }
        for (m.records) |rj| {
            if (rj.bank_layer >= m.layers.len or rj.mini >= m.layers[rj.bank_layer].n_minis) return refuse(diag, error.RecordGeometry, manifest_file ++ ": record ({d}, {d}) out of range", .{ rj.bank_layer, rj.mini });
            const lj = m.layers[rj.bank_layer];
            const k = first[rj.bank_layer] + rj.mini;
            if (seen[k]) return refuse(diag, error.RecordDuplicate, manifest_file ++ ": record ({d}, {d}) duplicated", .{ rj.bank_layer, rj.mini });
            seen[k] = true;
            if (rj.sidecar_offset != lj.base_offset + rj.mini * lj.record_bytes or rj.expert != lj.experts[rj.mini / q.tp_ranks] or rj.rank != rj.mini % q.tp_ranks)
                return refuse(diag, error.RecordGeometry, manifest_file ++ ": record ({d}, {d}) is expert {d} rank {d} at {d}, want expert {d} rank {d} at {d}", .{ rj.bank_layer, rj.mini, rj.expert, rj.rank, rj.sidecar_offset, lj.experts[rj.mini / q.tp_ranks], rj.mini % q.tp_ranks, lj.base_offset + rj.mini * lj.record_bytes });
            var dg: [32]u8 = undefined;
            if (rj.sha256.len != 64) return refuse(diag, error.RecordSha256, manifest_file ++ ": record ({d}, {d}) sha256 is not 64 hex digits", .{ rj.bank_layer, rj.mini });
            _ = std.fmt.hexToBytes(&dg, rj.sha256) catch return refuse(diag, error.RecordSha256, manifest_file ++ ": record ({d}, {d}) sha256 is not hex", .{ rj.bank_layer, rj.mini });
        }
        // 7. The converter's parity.
        if (!m.parity.all_pass) return refuse(diag, error.ParityNotPassed, manifest_file ++ ": parity.all_pass is not true", .{});
        return base;
    }

    /// The pool's descriptor: no symlink, a regular file of exactly the bank layers' bytes, the page cache bypassed.
    fn openSidecar(self: *Bank, size: u64, diag: ?*Diag) !void {
        var errno: c_int = 0;
        const f = expert_io.openUncached(self.sidecar_path.ptr, &errno) catch |e| return switch (e) {
            error.NotFound => refuse(diag, error.SidecarMissing, "{s}: not found", .{self.sidecar_path}),
            error.OpenFailed => refuse(diag, error.SidecarOpen, "{s}: open(O_RDONLY | O_NOFOLLOW) failed, errno {d}", .{ self.sidecar_path, errno }),
            error.StatFailed => refuse(diag, error.SidecarOpen, "{s}: fstat failed", .{self.sidecar_path}),
            error.NotRegularFile => refuse(diag, error.SidecarGeometry, "{s} is not a regular file", .{self.sidecar_path}),
            error.NoCacheRefused => refuse(diag, error.SidecarOpen, "{s}: F_NOCACHE / F_RDAHEAD refused", .{self.sidecar_path}),
        };
        self.sidecar = f;
        if (f.size != size) return refuse(diag, error.SidecarGeometry, "{s} is {d} B, the manifest's sidecar {d} B", .{ self.sidecar_path, f.size, size });
    }
};

fn plainName(s: []const u8) bool {
    return s.len > 0 and std.mem.indexOfScalar(u8, s, '/') == null and !std.mem.eql(u8, s, ".") and !std.mem.eql(u8, s, "..");
}

/// Whether `dir` holds an EXL3 bank (its manifest): the plugin's choice of bank at load.
pub fn present(dir: []const u8) bool {
    var pb: [std.fs.max_path_bytes + 1]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&pb, "{s}/" ++ manifest_file, .{dir}, 0) catch return false;
    var st: std.c.Stat = undefined;
    return std.c.stat(p.ptr, &st) == 0;
}

/// The stream over this bank (`sdk_ext.expert.stream.StreamOf`).
pub const Stream = sdk_ext.expert.stream.StreamOf(@This(), false);

comptime {
    sdk_ext.expert.assertBank(@This());
    sdk_ext.expert.checkTopology(n_components, gu_components);
}

// ── synthetic packs (tests) ──

const testing = std.testing;

/// A synthetic EXL3 bank for `c`: per routed layer a K3 bank layer of its even experts and a K4 one of its odd
/// experts, then the MTP layer's two, `tp` minis per expert of seeded bytes (with `signs`: the suh / svh segments
/// +/-1 in f16), the records, the experts map and the sidecar. Each field below perturbs one rule.
pub const Synth = struct {
    tp: u32 = 4,
    signs: bool = false,
    format: []const u8 = format,
    multiplier: u64 = 0,
    hidden_delta: u64 = 0,
    seg_len_delta: u64 = 0,
    base_delta: u64 = 0,
    k_swap: bool = false,
    expert_both: bool = false,
    map_delta: u64 = 0,
    sidecar_size_delta: u64 = 0,
    sidecar_file: []const u8 = "experts.bin",
    drop_record: bool = false,
    dup_record: bool = false,
    record_rank_delta: u64 = 0,
    bad_sha: bool = false,
    all_pass: bool = true,
    truncate_file: bool = false,
    no_manifest: bool = false,
};

/// Writes `s`'s bank for `c` into `dir`; returns the experts.bin image (caller frees).
pub fn writeSynth(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, s: Synth) ![]u8 {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    const h: u64 = c.hidden_size;
    const mini: u64 = c.moe_intermediate_size / s.tp;
    const ne: u64 = c.n_routed_experts;
    var sparse: [glm.max_layers]u32 = undefined;
    const routed = c.sparseLayers(&sparse);
    const n_bank = (routed.len + 1) * banks_per_layer;
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    var recs: std.ArrayList(u8) = .empty;
    defer recs.deinit(a);
    var image: std.ArrayList(u8) = .empty;
    errdefer image.deinit(a);
    var rng = std.Random.DefaultPrng.init(53);
    const mult = if (s.multiplier != 0) s.multiplier else sushi.format.MCG_MULT;
    try j.print(a, "{{\"format\":\"{s}\",\"model_type\":\"glm_moe_dsa\",\"source\":{{\"repo\":\"synthetic\",\"revision\":null}},\"quantization\":{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"mcg_scalar\":0,\"k_values\":[3,4],\"tp_ranks\":{d}}},", .{ s.format, mult, s.tp });
    try j.print(a, "\"dims\":{{\"hidden\":{d},\"inter\":{d},\"mini_inter\":{d},\"n_experts\":{d},\"n_model_layers\":{d},\"n_bank_layers\":{d}}},\"components\":[", .{ h + s.hidden_delta, c.moe_intermediate_size, mini, ne, routed.len + 1, n_bank });
    for (Component.names, 0..) |n, i| try j.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", n });
    try j.appendSlice(a, "],\"layers\":[");
    var base: u64 = 0;
    var first_rec = true;
    for (0..n_bank) |bl| {
        const r = bl / banks_per_layer;
        const ml: u64 = if (r < routed.len) routed[r] else c.n_layers;
        var sd = bl % banks_per_layer;
        if (s.k_swap and bl == 2) sd = 1;
        if (s.k_swap and bl == 3) sd = 0;
        const k: u64 = if (sd == 0) 3 else 4;
        const held = (bl % banks_per_layer) == 0;
        var members: std.ArrayList(u64) = .empty;
        defer members.deinit(a);
        for (0..ne) |e| if ((e % 2 == 0) == held or (s.expert_both and bl == 1 and e == 0)) try members.append(a, e);
        const ms = miniSegments(h, mini, k);
        const rb = std.mem.alignForward(u64, ms.logical, record_alignment);
        const n_minis = s.tp * members.items.len;
        try j.print(a, "{s}{{\"bank_layer\":{d},\"layer\":{d},\"k\":{d},\"mtp\":{},\"n_minis\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"base_offset\":{d},\"experts\":[", .{ if (bl == 0) "" else ",", bl, ml, k, r == routed.len, n_minis, rb, ms.logical, base + (if (bl == 3) s.base_delta else 0) });
        for (members.items, 0..) |e, i| try j.print(a, "{s}{d}", .{ if (i == 0) "" else ",", e });
        try j.appendSlice(a, "],\"segments\":[");
        for (ms.segs, 0..) |sg, ci| {
            try j.print(a, "{s}{{\"component\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ if (ci == 0) "" else ",", Component.names[ci], @tagName(sg.dtype) });
            for (sg.shape[0..sg.rank], 0..) |x, i| try j.print(a, "{s}{d}", .{ if (i == 0) "" else ",", x });
            try j.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ sg.offset, sg.length + (if (bl == 2 and ci == 4) s.seg_len_delta else 0) });
        }
        try j.appendSlice(a, "]}");
        for (0..n_minis) |mi| {
            const at = image.items.len;
            try image.appendNTimes(a, 0, @intCast(rb));
            const rec = image.items[at..][0..@intCast(ms.logical)];
            rng.random().bytes(rec);
            if (s.signs) for (ms.segs, 0..) |sg, ci| if (ci % 3 != 0) for (0..sg.length / 2) |i| {
                std.mem.writeInt(u16, rec[sg.offset + 2 * i ..][0..2], if (rng.random().boolean()) 0x3C00 else 0xBC00, .little);
            };
            var dg: [32]u8 = undefined;
            Sha256.hash(rec, &dg, .{});
            const hex = std.fmt.bytesToHex(dg, .lower);
            if (s.drop_record and bl == 1 and mi == 3) continue;
            const mm = if (s.dup_record and bl == 1 and mi == 3) mi - 1 else mi;
            const rank = mm % s.tp + (if (bl == 2 and mi == 5) s.record_rank_delta else 0);
            try recs.print(a, "{s}{{\"bank_layer\":{d},\"mini\":{d},\"expert\":{d},\"rank\":{d},\"sidecar_offset\":{d},\"sha256\":\"{s}\"}}", .{ if (first_rec) "" else ",", bl, mm, members.items[mm / s.tp], rank, base + mm * rb, if (s.bad_sha and bl == 0 and mi == 1) "abc" else &hex });
            first_rec = false;
        }
        base += n_minis * rb;
    }
    try j.appendSlice(a, "],\"experts\":{");
    for (routed, 0..) |ml, r| {
        try j.print(a, "{s}\"{d}\":[", .{ if (r == 0) "" else ",", ml });
        for (0..ne) |e| try j.print(a, "{s}[{d},{d}]", .{ if (e == 0) "" else ",", if (e % 2 == 0) @as(u64, 3) else 4, e / 2 + (if (r == 1 and e == 5) s.map_delta else 0) });
        try j.appendSlice(a, "]");
    }
    try j.print(a, "}},\"sidecar\":{{\"file\":\"{s}\",\"alignment\":4096,\"size\":{d}}},\"records\":[{s}],\"parity\":{{\"all_pass\":{},\"checked\":0,\"total\":0,\"method\":\"bytes-equal-source\"}}}}", .{ s.sidecar_file, base + s.sidecar_size_delta, recs.items, s.all_pass });
    if (!s.no_manifest) try dir.writeFile(io, .{ .sub_path = manifest_file, .data = j.items });
    try dir.writeFile(io, .{ .sub_path = "experts.bin", .data = if (s.truncate_file) image.items[0 .. image.items.len - 4096] else image.items });
    return image.toOwnedSlice(a);
}

/// The tiny model's config with `inter` routed intermediate (the kernels' Hadamard blocks need minis of 128).
pub fn tinyConfigInter(a: std.mem.Allocator, inter: u32) !glm.Config {
    const text = try glm.tinyConfigJson(a, glm.tiny_quant);
    defer a.free(text);
    var buf: [64]u8 = undefined;
    const t2 = try std.mem.replaceOwned(u8, a, text, "\"moe_intermediate_size\":64,", try std.fmt.bufPrint(&buf, "\"moe_intermediate_size\":{d},", .{inter}));
    defer a.free(t2);
    return glm.Config.parse(a, t2, null, null);
}

pub const tmpRoot = affine.tmpRoot;

test "glm exl3 bank: GLM-5.3's minis: K3 3,578,880 B in 3,579,904 B records, K4 4,758,528 in 4,759,552, the gate/up span first" {
    const k3 = miniSegments(6144, 512, 3);
    const k4 = miniSegments(6144, 512, 4);
    try testing.expectEqual(@as(u64, 3_578_880), k3.logical);
    try testing.expectEqual(@as(u64, 4_758_528), k4.logical);
    try testing.expectEqual(@as(u64, 3_579_904), std.mem.alignForward(u64, k3.logical, record_alignment));
    try testing.expectEqual(@as(u64, 4_759_552), std.mem.alignForward(u64, k4.logical, record_alignment));
    try testing.expectEqualSlices(u64, &.{ 384, 32, 48 }, k3.segs[0].shape[0..3]);
    try testing.expectEqualSlices(u64, &.{ 32, 384, 64 }, k4.segs[6].shape[0..3]);
    try testing.expectEqual(@as(u64, 2_385_920), k3.segs[gu_components].offset);
    try testing.expectEqualSlices(u64, &.{6144}, k3.segs[7].shape[0..1]);
    try testing.expectEqualSlices(u64, &.{512}, k3.segs[8].shape[0..1]);
    // The box's layer: 148 K3 and 108 K4 experts, 4 minis each: 4,175,429,632 B.
    try testing.expectEqual(@as(u64, 4_175_429_632), 148 * 4 * 3_579_904 + 108 * 4 * 4_759_552);
}

test "glm exl3 bank: a clean synthetic bank opens: the stream layers, each expert's bank layer and its minis' offsets" {
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var diag: Diag = .{};
    var b = Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer b.deinit();
    // 4 routed layers, 2 bank layers each; hidden 128, minis of 16.
    try testing.expectEqual(@as(usize, 8), b.layers.len);
    try testing.expectEqual(@as(u32, 3), b.layers[2].k);
    try testing.expectEqual(@as(u32, 4), b.layers[3].k);
    try testing.expectEqual(@as(u32, 2), b.layers[3].model_layer);
    try testing.expectEqual(@as(u32, 8), b.layers[3].n_held);
    try testing.expectEqual(@as(u32, 4), b.layers[0].minis);
    try testing.expectEqual(@as(u32, 2), b.streamLayer(1, 6));
    try testing.expectEqual(@as(u32, 3), b.streamLayer(1, 7));
    const ms3 = miniSegments(128, 16, 3);
    const ms4 = miniSegments(128, 16, 4);
    const rb3 = std.mem.alignForward(u64, ms3.logical, 4096);
    const rb4 = std.mem.alignForward(u64, ms4.logical, 4096);
    // Expert 7 of routed layer 1 (stream layer 3, K4): local 3, after layer 0's two bank layers and layer 1's K3.
    const base3 = 8 * 4 * rb3 + 8 * 4 * rb4 + 8 * 4 * rb3;
    try testing.expectEqual(base3 + 3 * 4 * rb4, b.recordOffset(3, 7));
    try testing.expectEqual(Spans{ .gu_offset = base3 + 3 * 4 * rb4, .down_offset = base3 + 3 * 4 * rb4 + ms4.segs[gu_components].offset }, b.spans(3, 7));
    // The sidecar holds the MTP layer's bank layers too.
    try testing.expectEqual(@as(u64, img.len), b.sidecar.size);
    try testing.expectEqual(@as(u64, 5 * 8 * 4 * (rb3 + rb4)), img.len);
    // The fill: units of 1/8 of each bank layer's 8 experts; a unit's bound covers both geometries' slots.
    try testing.expectEqual(@as(u32, 8), b.maxUnits());
    var rows: [8]u32 = undefined;
    b.rowsAt(5, &rows);
    for (rows) |r| try testing.expectEqual(@as(u32, 5), r);
    const g = b.geometryOf();
    try testing.expectEqual(@as(u32, 4), g.n_layers);
    try testing.expectEqual(4 * (ms3.logical + ms4.logical), g.widest_record);
    try testing.expect(present(try tmpRoot(&tmp, &rbuf)));
    // GLM-5.3's split: 148 K3 and 108 K4 experts, units of 1/128 of each, rounded down.
    var two = [2]Layer{ b.layers[0], b.layers[1] };
    two[0].n_held = 148;
    two[1].n_held = 108;
    const glm53: Bank = .{ .allocator = a, .n_experts = 256, .layers = &two };
    var r2: [2]u32 = undefined;
    glm53.rowsAt(100, &r2);
    try testing.expectEqualSlices(u32, &.{ 115, 84 }, &r2);
    glm53.rowsAt(128, &r2);
    try testing.expectEqualSlices(u32, &.{ 148, 108 }, &r2);
    glm53.rowsAt(16, &r2);
    try testing.expectEqualSlices(u32, &.{ 18, 13 }, &r2);
}

test "glm exl3 bank: every rule of the pack's contract refuses by name" {
    const Case = struct { s: Synth, err: anyerror, why: []const u8 };
    const cases = [_]Case{
        .{ .s = .{ .no_manifest = true }, .err = error.ManifestMissing, .why = "not found" },
        .{ .s = .{ .format = "mlx-stream-expert-manifest-exl3-v2" }, .err = error.ManifestFormat, .why = "format" },
        .{ .s = .{ .multiplier = 0x83DCD12D }, .err = error.QuantNotImplemented, .why = "multiplier" },
        .{ .s = .{ .hidden_delta = 128 }, .err = error.DimsMismatch, .why = "hidden 256" },
        .{ .s = .{ .seg_len_delta = 2 }, .err = error.LayerGeometry, .why = "segment up_proj.rout" },
        .{ .s = .{ .base_delta = 4096 }, .err = error.LayerGeometry, .why = "base" },
        .{ .s = .{ .k_swap = true }, .err = error.LayerGeometry, .why = "K ascending" },
        .{ .s = .{ .expert_both = true }, .err = error.ExpertMap, .why = "both bank layers" },
        .{ .s = .{ .map_delta = 1 }, .err = error.ExpertMap, .why = "experts[2][5]" },
        .{ .s = .{ .sidecar_size_delta = 4096 }, .err = error.SidecarGeometry, .why = "sidecar alignment" },
        .{ .s = .{ .sidecar_file = "../experts.bin" }, .err = error.SidecarGeometry, .why = "plain file name" },
        .{ .s = .{ .drop_record = true }, .err = error.RecordCount, .why = "records" },
        .{ .s = .{ .dup_record = true }, .err = error.RecordDuplicate, .why = "duplicated" },
        .{ .s = .{ .record_rank_delta = 1 }, .err = error.RecordGeometry, .why = "record (2, 5)" },
        .{ .s = .{ .bad_sha = true }, .err = error.RecordSha256, .why = "sha256" },
        .{ .s = .{ .all_pass = false }, .err = error.ParityNotPassed, .why = "all_pass" },
        .{ .s = .{ .truncate_file = true }, .err = error.SidecarGeometry, .why = "experts.bin is" },
    };
    var c = try affine.tinyConfig(testing.allocator);
    defer c.deinit(testing.allocator);
    for (cases) |cs| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bin = try writeSynth(testing.allocator, testing.io, tmp.dir, &c, cs.s);
        defer testing.allocator.free(bin);
        var rbuf: [512]u8 = undefined;
        var diag: Diag = .{};
        const r = Bank.open(testing.allocator, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag);
        if (r) |ok| {
            var b = ok;
            b.deinit();
            std.debug.print("case {any}: opened\n", .{cs.s});
            return error.TestExpectedError;
        } else |e| {
            testing.expectEqual(cs.err, e) catch |x| {
                std.debug.print("case {any}: {s}\n", .{ cs.s, diag.message() });
                return x;
            };
            testing.expect(std.mem.indexOf(u8, diag.message(), cs.why) != null) catch |x| {
                std.debug.print("want \"{s}\" in \"{s}\"\n", .{ cs.why, diag.message() });
                return x;
            };
        }
    }
    var diag: Diag = .{};
    try testing.expectError(error.BankDirNotAbsolute, Bank.open(testing.allocator, testing.io, "packs/glm", &c, &diag));
}

test "glm exl3 bank: the stream reads each routed expert's minis into adjacent rows of its bank layer's slots, persistent and transient, both geometries" {
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    // Two persistent experts per bank layer: most loads go to the transient scratch of their geometry.
    var rows: [8]u32 = undefined;
    @memset(&rows, 2);
    const s = try Stream.Stream.init(a, &b, .{ .rows = &rows, .max_route_ids = 8, .transient_rows = 16, .wide_depth = 2, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    // The K4 bank layers are the widest (the first scratch), the K3 ones the second.
    try testing.expectEqual(@as(u32, 1), s.transient_layer);
    try testing.expectEqual(@as(u32, 0), s.alt.?.layer);
    try testing.expectEqual(@as(u1, 0), s.geom_of[3]);
    try testing.expectEqual(@as(u1, 1), s.geom_of[2]);
    var rng = std.Random.DefaultPrng.init(7);
    var ids: [8]u16 = undefined;
    for (0..40) |step| {
        const l: u32 = @intCast(step % 8);
        const n = 1 + rng.random().uintLessThan(usize, 4);
        var k: usize = 0;
        while (k < n) {
            const e: u16 = @intCast(2 * rng.random().uintLessThan(u16, 8) + l % 2);
            if (std.mem.indexOfScalar(u16, ids[0..k], e) == null) {
                ids[k] = e;
                k += 1;
            }
        }
        const r = try s.route(l, ids[0..n], &.{});
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        for (ids[0..n], r.plan.slotsOf()) |e, slot| {
            const off = b.recordOffset(l, e);
            const geom = &b.layers[l];
            for (geom.segments, 0..) |sg, ci| {
                const row = s.slotRow(l, slot, @enumFromInt(ci));
                try testing.expectEqual(@as(usize, geom.minis * sg.length), row.len);
                for (0..geom.minis) |mi| try testing.expectEqualSlices(u8, img[off + mi * geom.record_bytes + sg.offset ..][0..sg.length], row[mi * sg.length ..][0..sg.length]);
            }
        }
        s.release(r);
    }
    try s.flush();
}

test "glm exl3 bank: the lookahead reads ahead the next routed layer's experts over both its bank layers, each its own span, claimable until the call's last route" {
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    var rows: [8]u32 = undefined;
    @memset(&rows, 2);
    const s = try Stream.Stream.init(a, &b, .{ .rows = &rows, .max_route_ids = 8, .transient_rows = 16, .wide_depth = 2, .transient_release = true, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 }, .lookahead = .{ .k = 8, .budget = 4, .preread = false } });
    defer s.deinit();
    _ = try s.releaseTransient();
    try s.grow(&rows);
    // Routed layer 0's call: its K3 route held, its K4 route steps with the next layer's scores, in order for experts
    // 5 (K4), 2 (K3), 6 (K3), 7 (K4) of routed layer 1, none resident. Each speculative chunk held 100-200 ms on the
    // two speculative threads: 5 and 2 in flight, 6 and 7 queued when the next call routes.
    var scores: [16]f32 = @splat(0);
    scores[5] = 9;
    scores[2] = 8.5;
    scores[6] = 8;
    scores[7] = 7.5;
    const q3raw = struct {
        extern fn q3ld_test_spec_delay(max_ns: i64) void;
    };
    q3raw.q3ld_test_spec_delay(200 * std.time.ns_per_ms);
    defer q3raw.q3ld_test_spec_delay(0);
    const r0 = try s.routeHeld(0, &.{ 0, 4 });
    const r1 = try s.route(1, &.{ 1, 3 }, &scores);
    for ([_]*Stream.Route{ r0, r1 }) |r| for (0..r.n_parts) |p| try s.waitDown(r, @intCast(p));
    s.release(r0);
    s.release(r1);
    try testing.expectEqual(@as(u64, 4), s.stats().spec_issued);
    // Routed layer 1's call: its held K3 route claims 2 (in flight) and reads 6 itself (queued); nothing is settled
    // before its K4 route, which claims 5 and reads 7 itself. Every mini's two ranges of 2 and 5 are served from the
    // records (5's read at its K4 span).
    const q0 = try s.routeHeld(2, &.{ 2, 6 });
    const q1 = try s.route(3, &.{ 5, 7 }, &.{});
    for ([_]*Stream.Route{ q0, q1 }) |r| for (0..r.n_parts) |p| try s.waitDown(r, @intCast(p));
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2), st.claimed);
    try testing.expectEqual(@as(u64, 2), st.spec_cancelled);
    try testing.expectEqual(@as(u64, 0), st.spec_expired);
    try testing.expectEqual(@as(u64, 2 * 4 * 2), st.adopt_ranges);
    for ([_]struct { r: *Stream.Route, l: u32 }{ .{ .r = q0, .l = 2 }, .{ .r = q1, .l = 3 } }) |x| {
        const geom = &b.layers[x.l];
        for (x.r.plan.slotsOf(), [_]u16{ if (x.l == 2) 2 else 5, if (x.l == 2) 6 else 7 }) |slot, e| {
            const off = b.recordOffset(x.l, e);
            for (geom.segments, 0..) |sg, ci| {
                const row = s.slotRow(x.l, slot, @enumFromInt(ci));
                for (0..geom.minis) |mi| try testing.expectEqualSlices(u8, img[off + mi * geom.record_bytes + sg.offset ..][0..sg.length], row[mi * sg.length ..][0..sg.length]);
            }
        }
    }
    s.release(q0);
    s.release(q1);
    try s.flush();
}
