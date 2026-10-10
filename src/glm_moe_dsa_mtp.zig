//! GLM-5.3's multi-token prediction head (the release's layer `num_hidden_layers`, `num_nextn_predict_layers` = 1) and
//! the draft lane's state over it. Pair p is (the token at p + 1, the target's final-normed hidden at p); the head
//! computes `h = layer(eh_proj(cat[enorm(embed(token)), hnorm(hidden)]))` and its logits `lm_head(shared_head.norm(h))`
//! with the target's embedding and head (transformers `MtpLayer` / `MtpModel`, modeling_layers.py; vLLM
//! `DeepSeekMultiTokenPredictorLayer`). The layer is the trunk's decoder layer (MLA with a full indexer, a routed MLP
//! with its shared expert) over its own cache: pair p's latent, rope key and indexer key at position p.
//!
//! A round's first step appends every pending pair (the pairs the target verified since the last round) and runs the
//! layer for the last one, which drafts the first token. Depth k > 1 runs the same layer on (the draft k - 1,
//! shared_head.norm(h) of the step before) at position p + k - 1, the last pair's p. `index_share_for_mtp_iteration`
//! (vLLM's `set_skip_topk`): such a step runs no indexer and attends the keys the first step's row selected, every
//! committed pair's key while they fit `index_topk`; it appends nothing, so a rejected draft leaves the cache as it was.
//!
//! The residents are the published BF16 tensors of `mtp/mtp-residents.safetensors`, bound as they are: `kv_b_proj`
//! split into the per-head `embed_q` / `unembed_out` by a reshape and slices (views of the same bytes, the reference's
//! unquantized sanitize). The routed experts are a bank kind bound at comptime: `.exl3` (the served lane: the EXL3
//! mini-expert records of `mtp/mtp-experts.bin`, resident, through sushi's EXL3 MoE on the host's `mlx_host`) or
//! `.dense` (the CPU lane's fixture: BF16 `mlp.switch_mlp.*` tensors in the residents file).

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const log = sdk.log;
const glm = @import("glm_moe_dsa.zig");
const graph = @import("glm_moe_dsa_graph.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const ds = @import("deepseek_v41_dspark.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");

const G = graph.G;
const T = G.T;
const Diag = glm.Diag;

pub const dir_name = "mtp";
pub const residents_file = "mtp-residents.safetensors";
pub const manifest_file = "mtp-manifest-exl3-v1.json";
pub const bank_file = "mtp-experts.bin";
pub const manifest_format = "mlx-stream-expert-manifest-exl3-v1";
/// The deepest draft a round verifies (depth + 1 rows of top-8 routes in one decode-lane call).
pub const max_depth = settings.Config.max_mtp_depth;
/// Prompt pairs whose keys one append covers (the prompt pass's catch-up, in chunks).
pub const append_rows: u32 = 2048;
/// Serial steps whose hidden the lane keeps for a later round; past them the lane stops tracking the request.
pub const max_pending: u32 = 64;

pub const BankKind = enum { exl3, dense };

pub const Refusal = error{ MtpPackMissing, MtpConfig, MtpTensorMissing, MtpTensorDtype, MtpTensorShape, MtpTensorUnexpected, MtpManifest, MtpBankSize };

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// The MTP layer as a one-layer model of the trunk's config: index 0, a full indexer, a routed MLP. Never deinit'd
/// (it borrows nothing it frees).
pub fn layerConfig(c: *const glm.Config) glm.Config {
    var m = c.*;
    m.n_layers = 1;
    m.indexer_types = @splat(.shared);
    m.indexer_types[0] = .full;
    m.mlp_types = @splat(.dense);
    m.mlp_types[0] = .sparse;
    m.owned_overrides = &.{};
    m.owned_paths = &.{};
    return m;
}

/// The release's config as the lane needs it: an MTP layer, and a selection shared across a round's steps when the
/// lane drafts more than one token (the lane implements `index_share_for_mtp_iteration` only).
pub fn checkConfig(c: *const glm.Config, depth: u32, diag: ?*Diag) Refusal!void {
    if (c.n_nextn < 1) return refuse(diag, error.MtpConfig, "config: num_nextn_predict_layers is {d} (the draft lane needs the release's MTP layer)", .{c.n_nextn});
    if (depth > 1 and !c.index_share_mtp) return refuse(diag, error.MtpConfig, "config: index_share_for_mtp_iteration is not true (a draft past the first reuses the first step's selection; mtp_depth {d})", .{depth});
}

// ── the pack's MTP directory, host side ──

/// The MTP layer's residents as published (BF16; the indexer, the router and its bias as stored), names in `a`.
/// `.dense` adds the routed experts' `mlp.switch_mlp.{gate,up,down}_proj.weight` `[E, out, in]` (the CPU lane's).
pub fn residentSpec(a: std.mem.Allocator, c: *const glm.Config, kind: BankKind) ![]glm.Param {
    var out: std.ArrayList(glm.Param) = .empty;
    const bf16 = &[_]glm.StDtype{.BF16};
    const S = struct {
        fn add(o: *std.ArrayList(glm.Param), al: std.mem.Allocator, l: u32, rest: []const u8, dtypes: []const glm.StDtype, shape: []const u64) !void {
            var p: glm.Param = .{ .name = try std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ l, rest }), .dtypes = dtypes, .rank = @intCast(shape.len) };
            @memcpy(p.shape[0..shape.len], shape);
            try o.append(al, p);
        }
    };
    const l = c.n_layers;
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
    const e: u64 = c.n_routed_experts;
    for ([_][]const u8{ "enorm.weight", "hnorm.weight", "shared_head.norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight" }) |n| try S.add(&out, a, l, n, bf16, &.{h});
    try S.add(&out, a, l, "eh_proj.weight", bf16, &.{ h, 2 * h });
    try S.add(&out, a, l, "self_attn.q_a_proj.weight", bf16, &.{ c.q_lora_rank, h });
    try S.add(&out, a, l, "self_attn.q_a_layernorm.weight", bf16, &.{c.q_lora_rank});
    try S.add(&out, a, l, "self_attn.q_b_proj.weight", bf16, &.{ heads * c.qHeadDim(), c.q_lora_rank });
    try S.add(&out, a, l, "self_attn.kv_a_proj_with_mqa.weight", bf16, &.{ c.kv_lora_rank + c.qk_rope_head_dim, h });
    try S.add(&out, a, l, "self_attn.kv_a_layernorm.weight", bf16, &.{c.kv_lora_rank});
    try S.add(&out, a, l, "self_attn.kv_b_proj.weight", bf16, &.{ heads * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank });
    try S.add(&out, a, l, "self_attn.o_proj.weight", bf16, &.{ h, heads * c.v_head_dim });
    try S.add(&out, a, l, "self_attn.indexer.wq_b.weight", &glm.float_dtypes, &.{ @as(u64, c.index_n_heads) * c.index_head_dim, c.q_lora_rank });
    try S.add(&out, a, l, "self_attn.indexer.wk.weight", &glm.float_dtypes, &.{ c.index_head_dim, h });
    try S.add(&out, a, l, "self_attn.indexer.k_norm.weight", &glm.float_dtypes, &.{c.index_head_dim});
    try S.add(&out, a, l, "self_attn.indexer.k_norm.bias", &glm.float_dtypes, &.{c.index_head_dim});
    try S.add(&out, a, l, "self_attn.indexer.weights_proj.weight", &glm.float_dtypes, &.{ c.index_n_heads, h });
    try S.add(&out, a, l, "mlp.gate.weight", &glm.float_dtypes, &.{ e, h });
    try S.add(&out, a, l, "mlp.gate.e_score_correction_bias", &glm.float_dtypes, &.{e});
    try S.add(&out, a, l, "mlp.shared_experts.gate_proj.weight", bf16, &.{ si, h });
    try S.add(&out, a, l, "mlp.shared_experts.up_proj.weight", bf16, &.{ si, h });
    try S.add(&out, a, l, "mlp.shared_experts.down_proj.weight", bf16, &.{ h, si });
    if (kind == .dense) {
        const inter: u64 = c.moe_intermediate_size;
        try S.add(&out, a, l, "mlp.switch_mlp.gate_proj.weight", bf16, &.{ e, inter, h });
        try S.add(&out, a, l, "mlp.switch_mlp.up_proj.weight", bf16, &.{ e, inter, h });
        try S.add(&out, a, l, "mlp.switch_mlp.down_proj.weight", bf16, &.{ e, h, inter });
    }
    return out.toOwnedSlice(a);
}

/// The residents file against the spec from its header alone: every spec tensor present in one of its dtypes at its
/// exact shape, nothing else. Returns its tensor bytes.
pub fn checkResidents(spec: []const glm.Param, ck: *const glm.Checkpoint, diag: ?*Diag) Refusal!u64 {
    for (spec) |p| {
        const t = ck.tensors.get(p.name) orelse return refuse(diag, error.MtpTensorMissing, "{s}/{s}: {s} missing", .{ dir_name, residents_file, p.name });
        if (std.mem.indexOfScalar(glm.StDtype, p.dtypes, t.dtype) == null) return refuse(diag, error.MtpTensorDtype, "{s}: dtype {t}, want {any}", .{ p.name, t.dtype, p.dtypes });
        if (t.rank != p.rank or !std.mem.eql(u64, t.shape[0..t.rank], p.shapeOf())) return refuse(diag, error.MtpTensorShape, "{s}: shape {any}, want {any}", .{ p.name, t.shape[0..t.rank], p.shapeOf() });
    }
    var bytes: u64 = 0;
    for (ck.tensors.keys(), ck.tensors.values()) |name, t| {
        for (spec) |p| {
            if (std.mem.eql(u8, p.name, name)) break;
        } else return refuse(diag, error.MtpTensorUnexpected, "{s}/{s}: {s} is not a tensor of the MTP layer this build binds", .{ dir_name, residents_file, name });
        bytes += t.end - t.begin;
    }
    return bytes;
}

/// What the lane holds besides its KV, from the pack's headers and manifest (the bill's terms).
pub const Facts = struct { resident_bytes: u64, expert_bytes: u64 };

/// The MTP directory of the pack at `dir` checked from its headers (the residents against the spec, the EXL3
/// manifest against the format): refused by name when it is missing.
pub fn facts(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, kind: BankKind, diag: ?*Diag) !Facts {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}/{s}", .{ dir, dir_name, residents_file }, 0);
    if (std.c.access(path.ptr, std.c.F_OK) != 0) return refuse(diag, error.MtpPackMissing, "{s}: no MTP layer in this pack (mtp_depth > 0 needs {s}/ beside the shards: convert_glm_exl3_bank.py --mtp-only)", .{ path, dir_name });
    var ck = glm.Checkpoint.openFile(gpa, path, diag) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(diag, error.MtpPackMissing, "{s}: {s}", .{ path, @errorName(e) }),
    };
    defer ck.deinit();
    const resident_bytes = try checkResidents(try residentSpec(a, c, kind), &ck, diag);
    const expert_bytes = switch (kind) {
        .dense => 0,
        .exl3 => blk: {
            const mdir = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, dir_name });
            const m = try Manifest.load(a, io, mdir, c, diag);
            var n: u64 = 0;
            for (m.layers) |l| n += @as(u64, l.n_minis) * l.logical_bytes;
            break :blk n;
        },
    };
    return .{ .resident_bytes = resident_bytes, .expert_bytes = expert_bytes };
}

/// `mtp-manifest-exl3-v1.json` (docs/glm53-exl3-pack-format.md, the MTP directory): the MTP layer's bank layers.
pub const Manifest = struct {
    layers: []const Layer,
    /// Per expert: [K, local].
    experts: []const [2]u32,
    tp: u32,
    mini_inter: u32,

    pub const Segment = struct { component: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 };
    pub const Layer = struct { bank_layer: u32, layer: u32, k: u32, mtp: bool, n_minis: u32, record_bytes: u64, logical_bytes: u64, base_offset: u64, experts: []const u32, segments: []const Segment };

    const Json = struct {
        format: []const u8,
        model_type: []const u8,
        quantization: struct { mode: []const u8, codebook: []const u8, codebook_multiplier: u64, tp_ranks: u32 },
        dims: struct { hidden: u32, inter: u32, mini_inter: u32, n_experts: u32 },
        components: []const []const u8,
        layers: []const Layer,
        experts: std.json.ArrayHashMap([]const [2]u32),
        sidecar: struct { file: []const u8, size: u64 },
    };

    pub const components = [_][]const u8{ "gate_proj.code", "gate_proj.rout", "gate_proj.rin", "up_proj.code", "up_proj.rout", "up_proj.rin", "down_proj.code", "down_proj.rout", "down_proj.rin" };

    /// The manifest in `mdir` (the pack's `mtp/`) checked against the format and `c`, the sidecar's size against it.
    pub fn load(a: std.mem.Allocator, io: std.Io, mdir: []const u8, c: *const glm.Config, diag: ?*Diag) !Manifest {
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ mdir, manifest_file });
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.MtpPackMissing, "{s}: {s}", .{ path, @errorName(e) }),
        };
        const j = std.json.parseFromSliceLeaky(Json, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.MtpManifest, "{s}: {s}", .{ path, @errorName(e) }),
        };
        const q = j.quantization;
        const d = j.dims;
        if (!std.mem.eql(u8, j.format, manifest_format) or !std.mem.eql(u8, j.model_type, glm.model_type)) return refuse(diag, error.MtpManifest, "{s}: format {s} / model_type {s} (want {s} / {s})", .{ path, j.format, j.model_type, manifest_format, glm.model_type });
        if (!std.mem.eql(u8, q.mode, "exl3") or !std.mem.eql(u8, q.codebook, "mcg") or q.codebook_multiplier != sushi.format.MCG_MULT) return refuse(diag, error.MtpManifest, "{s}: quantization {s} / {s} / multiplier {d} (the lane binds exl3 mcg at {d})", .{ path, q.mode, q.codebook, q.codebook_multiplier, sushi.format.MCG_MULT });
        if (d.hidden != c.hidden_size or d.inter != c.moe_intermediate_size or d.n_experts != c.n_routed_experts or q.tp_ranks == 0 or d.mini_inter * q.tp_ranks != d.inter)
            return refuse(diag, error.MtpManifest, "{s}: dims hidden {d} inter {d} = {d} x {d} experts {d} (the config's {d} / {d} / {d})", .{ path, d.hidden, d.inter, q.tp_ranks, d.mini_inter, d.n_experts, c.hidden_size, c.moe_intermediate_size, c.n_routed_experts });
        if (j.components.len != components.len) return refuse(diag, error.MtpManifest, "{s}: {d} components (want 9)", .{ path, j.components.len });
        for (j.components, components) |x, y| if (!std.mem.eql(u8, x, y)) return refuse(diag, error.MtpManifest, "{s}: component {s} where the format has {s}", .{ path, x, y });
        if (j.layers.len == 0 or j.layers.len > 2) return refuse(diag, error.MtpManifest, "{s}: {d} bank layers (one per K of the MTP layer)", .{ path, j.layers.len });
        var size: u64 = 0;
        for (j.layers, 0..) |l, i| {
            if (l.bank_layer != i or l.layer != c.n_layers or !l.mtp) return refuse(diag, error.MtpManifest, "{s}: bank layer {d} is layer {d} (mtp {}), want the MTP layer {d}", .{ path, l.bank_layer, l.layer, l.mtp, c.n_layers });
            if (l.k < 2 or l.k > 4 or l.n_minis != q.tp_ranks * l.experts.len or l.base_offset != size) return refuse(diag, error.MtpManifest, "{s}: bank layer {d}: K {d}, {d} minis for {d} experts, base offset {d}", .{ path, i, l.k, l.n_minis, l.experts.len, l.base_offset });
            if (l.segments.len != components.len) return refuse(diag, error.MtpManifest, "{s}: bank layer {d} has {d} segments", .{ path, i, l.segments.len });
            var off: u64 = 0;
            for (l.segments, 0..) |sg, ci| {
                const want = segmentShape(ci, d.hidden, d.mini_inter, l.k);
                const dtype_ok = std.mem.eql(u8, sg.dtype, if (ci % 3 == 0) "I16" else "F16");
                var n: u64 = 2;
                for (sg.shape) |x| n *= x;
                if (!std.mem.eql(u8, sg.component, components[ci]) or !dtype_ok or !std.mem.eql(u64, sg.shape, want.slice()) or sg.offset != off or sg.length != n)
                    return refuse(diag, error.MtpManifest, "{s}: bank layer {d} segment {s}: {s} {any} at {d} (want {s} {any} at {d})", .{ path, i, sg.component, sg.dtype, sg.shape, sg.offset, components[ci], want.slice(), off });
                off += n;
            }
            if (off != l.logical_bytes or l.record_bytes < off) return refuse(diag, error.MtpManifest, "{s}: bank layer {d}: logical {d} / record {d} B for {d} B of segments", .{ path, i, l.logical_bytes, l.record_bytes, off });
            size += @as(u64, l.n_minis) * l.record_bytes;
        }
        const key = try std.fmt.allocPrint(a, "{d}", .{c.n_layers});
        const ex = j.experts.map.get(key) orelse return refuse(diag, error.MtpManifest, "{s}: experts has no layer {s}", .{ path, key });
        if (ex.len != c.n_routed_experts) return refuse(diag, error.MtpManifest, "{s}: experts[{s}] lists {d} experts", .{ path, key, ex.len });
        for (ex, 0..) |kl, e| {
            const l = for (j.layers) |l| {
                if (l.k == kl[0]) break l;
            } else return refuse(diag, error.MtpManifest, "{s}: expert {d} at K {d}, no bank layer has it", .{ path, e, kl[0] });
            if (kl[1] >= l.experts.len or l.experts[kl[1]] != e) return refuse(diag, error.MtpManifest, "{s}: expert {d} at [{d}, {d}] is not that bank layer's", .{ path, e, kl[0], kl[1] });
        }
        if (!std.mem.eql(u8, j.sidecar.file, bank_file) or j.sidecar.size != size) return refuse(diag, error.MtpManifest, "{s}: sidecar {s} of {d} B (want {s} of {d} B)", .{ path, j.sidecar.file, j.sidecar.size, bank_file, size });
        const bin = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ mdir, bank_file }, 0);
        var st: std.c.Stat = undefined;
        if (std.c.stat(bin.ptr, &st) != 0) return refuse(diag, error.MtpPackMissing, "{s}: cannot stat", .{bin});
        if (@as(u64, @intCast(st.size)) != size) return refuse(diag, error.MtpBankSize, "{s}: {d} B, the manifest gives {d}", .{ bin, st.size, size });
        return .{ .layers = j.layers, .experts = ex, .tp = q.tp_ranks, .mini_inter = d.mini_inter };
    }

    const Shape = struct {
        d: [3]u64,
        n: u8,
        fn slice(s: *const Shape) []const u64 {
            return s.d[0..s.n];
        }
    };

    /// Segment `ci`'s shape (gate / up: in = hidden, out = mini; down: in = mini, out = hidden; code = trellis
    /// `[in/16, out/16, 16K]`, rout = svh `[out]`, rin = suh `[in]`).
    fn segmentShape(ci: usize, hidden: u64, mini: u64, k: u64) Shape {
        const down = ci >= 6;
        const in_d = if (down) mini else hidden;
        const out_d = if (down) hidden else mini;
        return switch (ci % 3) {
            0 => .{ .d = .{ in_d / 16, out_d / 16, 16 * k }, .n = 3 },
            1 => .{ .d = .{ out_d, 0, 0 }, .n = 1 },
            else => .{ .d = .{ in_d, 0, 0 }, .n = 1 },
        };
    }
};

const sushi = @import("mlx_host").sushi_exl3;

// ── the routed experts ──

/// The MTP layer's routed experts of bank kind `kind`: `moe(x [n, hidden], indices [n, k] int32, scores [n, k] f32)`
/// = the weighted sum `[n, hidden]` in x's dtype.
pub fn Experts(comptime kind: BankKind) type {
    return switch (kind) {
        .dense => Dense,
        .exl3 => Exl3,
    };
}

/// BF16 experts `[E, out, in]` (the fixture's), each routed expert's matrices taken by id and multiplied in a batch.
pub const Dense = struct {
    gate: T,
    up: T,
    down: T,

    pub fn open(_: std.mem.Allocator, _: std.Io, _: []const u8, c: *const glm.Config, w: *const sdk.Weights, diag: ?*Diag) !Dense {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.mlp.switch_mlp", .{c.n_layers});
        return .{ .gate = try get(w, p, "gate_proj.weight", diag), .up = try get(w, p, "up_proj.weight", diag), .down = try get(w, p, "down_proj.weight", diag) };
    }

    pub fn deinit(_: *Dense) void {}

    pub fn moe(self: *const Dense, g: *G, x: T, indices: T, scores: T) !T {
        const k: c_int = g.shapeOf(indices).dim(1);
        const n: c_int = g.shapeOf(x).dim(0);
        const h: c_int = g.shapeOf(x).dim(1);
        const flat = try g.reshape(indices, &.{-1});
        const x4 = try g.reshape(x, &.{ n, 1, 1, h });
        const P = struct {
            fn proj(gg: *G, w: T, idx: T, v: T, kk: c_int) !T {
                const ws = try gg.take(w, idx, 0);
                const s = gg.shapeOf(ws);
                return gg.matmul(v, try graph.swapLast(gg, try gg.reshape(ws, &.{ -1, kk, s.dim(1), s.dim(2) })));
            }
        };
        const act = try g.mul(try g.silu(try P.proj(g, self.gate, flat, x4, k)), try P.proj(g, self.up, flat, x4, k));
        const y = try g.reshape(try P.proj(g, self.down, flat, act, k), &.{ n, k, h });
        return g.astype(try g.sum(try g.mul(y, try g.expandDims(scores, -1)), -2, false), g.dtypeOf(y));
    }
};

/// The EXL3 bank: each bank layer (one K) its nine components as resident arrays `[minis, ...]` (the records'
/// segments, bytes unchanged; the trellis read as uint16), and per expert its first mini's slot across the bank
/// layers. A routed expert is its `tp` minis at the expert's score (docs/glm53-exl3-pack-format.md, Mini-expert).
pub const Exl3 = struct {
    banks: [2]sushi.Bank = undefined,
    n_banks: usize = 0,
    /// int32 [E]: the expert's first mini slot (its bank layer's offset + local x tp).
    base: T,
    tp: c_int,
    /// Every array this bank made.
    owned: std.ArrayList(T) = .empty,
    gpa: std.mem.Allocator,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, _: *const sdk.Weights, diag: ?*Diag) !Exl3 {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const mdir = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, dir_name });
        const m = try Manifest.load(a, io, mdir, c, diag);
        const bin = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ mdir, bank_file }, 0);
        const fd = std.c.open(bin.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return refuse(diag, error.MtpPackMissing, "{s}: cannot open", .{bin});
        defer _ = std.c.close(fd);
        var self: Exl3 = .{ .base = undefined, .tp = @intCast(m.tp), .gpa = gpa };
        errdefer self.deinit();
        var slot_base: [2]u32 = .{ 0, 0 };
        var next: u32 = 0;
        for (m.layers, 0..) |l, b| {
            slot_base[b] = next;
            next += l.n_minis;
            var arrs: [9]T = undefined;
            for (l.segments, 0..) |sg, ci| {
                // On mapped pages, unmapped at the copy's end: a large block freed to libc's malloc can stay in the
                // process's footprint (its large cache), outside every term of the bill.
                const host = try std.heap.page_allocator.alloc(u8, @intCast(@as(u64, l.n_minis) * sg.length));
                defer std.heap.page_allocator.free(host);
                for (0..l.n_minis) |mi| {
                    const off = l.base_offset + mi * l.record_bytes + sg.offset;
                    if (!preadAll(fd, host[mi * sg.length ..][0..@intCast(sg.length)], off)) return refuse(diag, error.MtpBankSize, "{s}: short read at {d}", .{ bin, off });
                }
                var shape: [4]c_int = undefined;
                shape[0] = @intCast(l.n_minis);
                for (sg.shape, 0..) |x, i| shape[i + 1] = @intCast(x);
                const arr = mlx.mlx_array_new_data(host.ptr, &shape, @intCast(1 + sg.shape.len), if (ci % 3 == 0) .uint16 else .float16);
                if (arr.ctx == null) return error.MlxError;
                try self.owned.append(gpa, arr);
                arrs[ci] = arr;
            }
            // code = trellis, rout = svh (the output side), rin = suh (the input side).
            self.banks[b] = .{
                .gate = .{ .trellis = arrs[0], .svh = arrs[1], .suh = arrs[2] },
                .up = .{ .trellis = arrs[3], .svh = arrs[4], .suh = arrs[5] },
                .down = .{ .trellis = arrs[6], .svh = arrs[7], .suh = arrs[8] },
            };
            self.n_banks = b + 1;
        }
        const base = try gpa.alloc(i32, m.experts.len);
        defer gpa.free(base);
        for (m.experts, base) |kl, *x| {
            const b = for (m.layers, 0..) |l, i| {
                if (l.k == kl[0]) break i;
            } else unreachable;
            x.* = @intCast(slot_base[b] + kl[1] * m.tp);
        }
        const shape = [_]c_int{@intCast(base.len)};
        self.base = mlx.mlx_array_new_data(base.ptr, &shape, 1, .int32);
        if (self.base.ctx == null) return error.MlxError;
        try self.owned.append(gpa, self.base);
        var n: u64 = 0;
        for (m.layers) |l| n += @as(u64, l.n_minis) * l.logical_bytes;
        log.info("glm_moe_dsa: MTP experts {d} bank layers, {d} minis, {d} B resident ({s})\n", .{ m.layers.len, next, n, bin });
        return self;
    }

    pub fn deinit(self: *Exl3) void {
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.gpa);
        self.n_banks = 0;
    }

    /// The minis' slots and scores of the routed experts `[1, n, k * tp]`, then sushi's EXL3 MoE over the bank layers.
    pub fn moe(self: *const Exl3, g: *G, x: T, indices: T, scores: T) !T {
        const sh = g.shapeOf(indices);
        const n = sh.dim(0);
        const k = sh.dim(1);
        const h = g.shapeOf(x).dim(1);
        const first = try g.reshape(try g.take(self.base, try g.reshape(indices, &.{-1}), 0), &.{ n, k, 1 });
        const ranks = try g.reshape(try g.arange(0, @floatFromInt(self.tp), 1, .int32), &.{ 1, 1, self.tp });
        const slots = try g.reshape(try g.add(first, ranks), &.{ 1, n, k * self.tp });
        const sc = try g.reshape(try g.broadcastTo(try g.reshape(scores, &.{ n, k, 1 }), &.{ n, k, self.tp }), &.{ 1, n, k * self.tp });
        const y = try g.adopt(try sushi.moeGroups(g.s, try g.reshape(x, &.{ 1, n, h }), self.banks[0..self.n_banks], slots, sc, .mcg, false));
        return g.reshape(y, &.{ n, h });
    }
};

fn preadAll(fd: std.c.fd_t, buf: []u8, offset: u64) bool {
    var done: usize = 0;
    while (done < buf.len) {
        const r = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(offset + done));
        if (r < 0) {
            if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
            return false;
        }
        if (r == 0) return false;
        done += @intCast(r);
    }
    return true;
}

fn get(w: *const sdk.Weights, prefix: []const u8, rest: []const u8, diag: ?*Diag) !T {
    var buf: [192]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, rest });
    return w.get(name) orelse refuse(diag, error.MtpTensorMissing, "{s}: not in the MTP residents", .{name});
}

// ── the head ──

/// The MTP layer's residents bound (the decoder layer's projections dense, `embed_q` / `unembed_out` as views of
/// `kv_b_proj`), the views kept by the head.
pub const Head = struct {
    layer: graph.Layer,
    eh_proj: T,
    enorm: T,
    hnorm: T,
    head_norm: T,
    /// The views of `kv_b_proj` (embed_q, unembed_out), released with the head.
    views: [2]T,

    pub fn bind(g: *G, w: *const sdk.Weights, c: *const glm.Config, diag: ?*Diag) !Head {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}", .{c.n_layers});
        const d = graph.QLinear.dense;
        const m0 = g.mark();
        defer g.resetTo(m0);
        // kv_b_proj [heads * (nope + v), kv_lora] -> [heads, nope + v, kv_lora]; embed_q's weight is the nope rows
        // swapped (mlx-lm's `MultiLinear` [heads, kv_lora, nope]), unembed_out's the v rows [heads, v, kv_lora].
        const heads: c_int = @intCast(c.n_heads);
        const nope: c_int = @intCast(c.qk_nope_head_dim);
        const vd: c_int = @intCast(c.v_head_dim);
        const kvr: c_int = @intCast(c.kv_lora_rank);
        const v3 = try g.reshape(try get(w, p, "self_attn.kv_b_proj.weight", diag), &.{ heads, nope + vd, kvr });
        const eq = g.keep(try graph.swapLast(g, try g.slice(v3, &.{ 0, 0, 0 }, &.{ heads, nope, kvr }, &.{ 1, 1, 1 })));
        errdefer g.release(eq);
        const uo = g.keep(try g.slice(v3, &.{ 0, nope, 0 }, &.{ heads, nope + vd, kvr }, &.{ 1, 1, 1 }));
        errdefer g.release(uo);
        const at = struct {
            fn f(ww: *const sdk.Weights, pre: []const u8, rest: []const u8, dg: ?*Diag) !T {
                return get(ww, pre, rest, dg);
            }
        }.f;
        return .{
            .layer = .{
                .input_norm = try at(w, p, "input_layernorm.weight", diag),
                .post_norm = try at(w, p, "post_attention_layernorm.weight", diag),
                .q_a = d(try at(w, p, "self_attn.q_a_proj.weight", diag)),
                .q_a_norm = try at(w, p, "self_attn.q_a_layernorm.weight", diag),
                .q_b = d(try at(w, p, "self_attn.q_b_proj.weight", diag)),
                .kv_a = d(try at(w, p, "self_attn.kv_a_proj_with_mqa.weight", diag)),
                .kv_a_norm = try at(w, p, "self_attn.kv_a_layernorm.weight", diag),
                .embed_q = d(eq),
                .unembed_out = d(uo),
                .o = d(try at(w, p, "self_attn.o_proj.weight", diag)),
                .indexer = .{
                    .wq_b = try at(w, p, "self_attn.indexer.wq_b.weight", diag),
                    .wk = try at(w, p, "self_attn.indexer.wk.weight", diag),
                    .k_norm_w = try at(w, p, "self_attn.indexer.k_norm.weight", diag),
                    .k_norm_b = try at(w, p, "self_attn.indexer.k_norm.bias", diag),
                    .weights_proj = try at(w, p, "self_attn.indexer.weights_proj.weight", diag),
                },
                .router = .{ .w = try at(w, p, "mlp.gate.weight", diag), .bias = try at(w, p, "mlp.gate.e_score_correction_bias", diag) },
                .shared = .{ .gate = d(try at(w, p, "mlp.shared_experts.gate_proj.weight", diag)), .up = d(try at(w, p, "mlp.shared_experts.up_proj.weight", diag)), .down = d(try at(w, p, "mlp.shared_experts.down_proj.weight", diag)) },
            },
            .eh_proj = try at(w, p, "eh_proj.weight", diag),
            .enorm = try at(w, p, "enorm.weight", diag),
            .hnorm = try at(w, p, "hnorm.weight", diag),
            .head_norm = try at(w, p, "shared_head.norm.weight", diag),
            .views = .{ eq, uo },
        };
    }

    pub fn deinit(self: *Head, g: *G) void {
        for (self.views) |v| g.release(v);
    }
};

// ── acceptance ──

/// A round's decision on the verify rows' logits `[w, vocab]` (row i scores the token after row i's): how many
/// drafts the target accepts and the token after them (the correction, or the bonus when all are accepted).
pub const Decision = struct { accepted: u32, next: u32 };

/// `drafts` against `logits` under `mode` and the request's sampling (`base`: the absolute position of row 0, the
/// sampled draws' key). Greedy: exact accepts a draft equal to its row's argmax; typical accepts one whose probability
/// at temperature 1 exceeds min(eps, delta * exp(-entropy)) (the host's `sdk.acceptance.typicalThreshold`); the
/// correction and the bonus are the argmax. Sampled: exact is DeepSeek-V4.1's lane rule (`ds.Sampling`: point-mass
/// drafts accepted with probability p(draft) of the tempered, filtered row, the residual's draw at the first
/// rejection); typical tests the same p against its own entropy and draws the correction and the bonus from p.
pub fn decide(g: *G, logits: T, drafts: []const u32, mode: sdk.acceptance.Mode, sampling: sdk.SamplingParams, base: u64) !Decision {
    const w: usize = @intCast(g.shapeOf(logits).dim(0));
    std.debug.assert(w == drafts.len + 1 and w <= max_depth + 1);
    const nd: c_int = @intCast(drafts.len);
    const vocab = g.shapeOf(logits).dim(1);
    var tok: [max_depth + 1]u32 = undefined;
    var flags: [max_depth]bool = undefined;
    var idb: [max_depth]i32 = undefined;
    for (idb[0..drafts.len], drafts) |*d, v| d.* = @intCast(v);
    const typ: ?@FieldType(sdk.acceptance.Mode, "typical") = switch (mode) {
        .typical => |t| t,
        .exact => null,
        else => return error.MtpAcceptanceNotImplemented,
    };
    var have_flags = false;
    if (sampling.greedy()) {
        _ = try g.hostU32(try g.argmax(logits, -1), tok[0..w]);
        if (typ) |t| if (drafts.len > 0) {
            const lg = try g.astype(try g.slice(logits, &.{ 0, 0 }, &.{ nd, vocab }, &.{ 1, 1 }), .float32);
            const lp = try g.sub(lg, try g.logsumexp(lg, -1, true));
            _ = try g.hostBool(try typicalFlags(g, lp, try g.exp(lp), idb[0..drafts.len], t), flags[0..drafts.len]);
            have_flags = true;
        };
    } else {
        const sm: ds.Sampling = .{ .temperature = sampling.temperature, .top_p = sampling.top_p, .top_k = sampling.top_k, .min_p = sampling.min_p, .seed = sampling.seed };
        const Loop = dsl.Loop(G);
        if (typ) |t| {
            const sr = try Loop.sampledRows(g, logits, sm, base, &.{});
            _ = try g.hostU32(sr.tok, tok[0..w]);
            if (drafts.len > 0) {
                const p = try g.slice(sr.p, &.{ 0, 0 }, &.{ nd, vocab }, &.{ 1, 1 });
                const lp = try g.log(p);
                _ = try g.hostBool(try typicalFlags(g, lp, p, idb[0..drafts.len], t), flags[0..drafts.len]);
                have_flags = true;
            }
        } else {
            const sr = try Loop.sampledRows(g, logits, sm, base, drafts);
            _ = try g.hostU32(sr.tok, tok[0..w]);
            if (sr.accept) |acc| {
                _ = try g.hostBool(acc, flags[0..drafts.len]);
                have_flags = true;
            }
        }
    }
    var a: u32 = 0;
    while (a < drafts.len) : (a += 1) {
        const ok = if (have_flags) flags[a] else drafts[a] == tok[a];
        if (!ok) break;
    }
    return .{ .accepted = a, .next = tok[a] };
}

/// p(draft) > min(eps, delta * exp(-H(p))) per drafted row (`lp` = log p, f32 `[d, vocab]`; zero-mass entries add
/// nothing to the entropy).
fn typicalFlags(g: *G, lp: T, p: T, ids: []const i32, t: @FieldType(sdk.acceptance.Mode, "typical")) !T {
    const zero = try g.scalar(0, .float32);
    const plogp = try g.where(try g.greater(p, zero), try g.mul(p, lp), zero);
    const entropy = try g.neg(try g.sum(plogp, -1, false));
    const floor = try g.minimum(try g.scalar(t.eps, .float32), try g.mul(try g.scalar(t.delta, .float32), try g.exp(try g.neg(entropy))));
    const idx = try g.hostArray(std.mem.sliceAsBytes(ids), &.{ @intCast(ids.len), 1 }, .int32);
    return g.greater(try g.reshape(try g.takeAlongAxis(p, idx, -1), &.{-1}), floor);
}

// ── the lane ──

/// The lane's counters over one request (the host's `sdk.DraftStats` and the request's line).
pub const Counts = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    generated: u64 = 0,
    /// Rounds the lane ran without drafts (the budget's last tokens, or a request it did not track).
    serial: u64 = 0,
    wall_ns: u64 = 0,
    /// The rounds' time in the drafts (the MTP steps) and in the verify forward (the decision and the commit are
    /// the rest of `wall_ns`).
    draft_ns: u64 = 0,
    verify_ns: u64 = 0,
    /// The largest rise of MLX's high-water mark over a round's start (the round's transient, measured).
    peak_rise: u64 = 0,
    /// The verify's demand reads in bytes, and of them the reads whose expert no kept row routes (the first row of
    /// the layer's call that routes it is past the accepted ones): the rejected rows' own reads.
    verify_bytes: u64 = 0,
    rejected_bytes: u64 = 0,
    /// Per tenth of the head's top probability at a draft step: the steps the decision reached (the accepted ones
    /// and the first rejected one) and the accepted ones.
    decided: [10]u64 = @splat(0),
    accepted_by_p: [10]u64 = @splat(0),
};

/// The tenth of `p` (a probability): 0 .. 9.
pub fn tenth(p: f32) usize {
    return @min(@as(usize, @intFromFloat(@max(p, 0) * 10)), 9);
}

/// The lane's line at the request's end.
pub const Line = struct {
    depth: u32,
    mode: []const u8,
    delta: f32,
    c: Counts,

    pub fn format(p: Line, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const rounds: f64 = @floatFromInt(@max(p.c.rounds, 1));
        const rate: f64 = if (p.c.drafted == 0) 0 else 100 * @as(f64, @floatFromInt(p.c.accepted)) / @as(f64, @floatFromInt(p.c.drafted));
        const s = @as(f64, @floatFromInt(p.c.wall_ns)) / 1e9;
        try w.print("glm_moe_dsa: mtp depth {d} acceptance {s}", .{ p.depth, p.mode });
        if (std.mem.eql(u8, p.mode, "typical")) try w.print(" (delta {d})", .{p.delta});
        try w.print(": {d} rounds ({d} without drafts), {d} drafted, {d} accepted ({d:.1}%), {d:.2} accepted / {d:.2} tokens per round, {d} tokens in {d:.2} s ({d:.1} tok/s; drafts {d:.2} s, verify {d:.2} s; round peak {d:.0} MB)", .{
            p.c.rounds,                                           p.c.serial,                                            p.c.drafted, p.c.accepted, rate,
            @as(f64, @floatFromInt(p.c.accepted)) / rounds,       @as(f64, @floatFromInt(p.c.generated)) / rounds,       p.c.generated, s,
            if (s == 0) 0 else @as(f64, @floatFromInt(p.c.generated)) / s, @as(f64, @floatFromInt(p.c.draft_ns)) / 1e9, @as(f64, @floatFromInt(p.c.verify_ns)) / 1e9,
            @as(f64, @floatFromInt(p.c.peak_rise)) / 1e6,
        });
        const gb = @as(f64, @floatFromInt(p.c.verify_bytes)) / 1e9;
        try w.print(", verify reads {d:.2} GB ({d:.3} GB per emitted token), {d:.2} GB of it for rejected rows only", .{
            gb, if (p.c.generated == 0) 0 else gb / @as(f64, @floatFromInt(p.c.generated)), @as(f64, @floatFromInt(p.c.rejected_bytes)) / 1e9,
        });
    }
};

/// The lane's acceptance by the head's top probability, at the request's end.
pub const ProbLine = struct {
    c: Counts,

    pub fn format(p: ProbLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("glm_moe_dsa: mtp acceptance by the head's top probability (accepted / decided):");
        for (p.c.decided, p.c.accepted_by_p, 0..) |d, acc, i| try w.print("{s} {d}.{d} {d}/{d}", .{ if (i == 0) "" else ",", i / 10, i % 10, acc, d });
    }
};

/// The MTP head over its cache, with bank kind `kind`. The cache holds the pairs 0 .. `cache.len - 1`; `pending` the
/// target's final-normed hidden of the positions `cache.len ..` whose pairs are not appended yet (their next tokens
/// arrive later): `cache.len + n_pending` is the target's length while the lane tracks the request.
pub fn Lane(comptime kind: BankKind) type {
    return struct {
        const Self = @This();
        pub const Bank = Experts(kind);

        gpa: std.mem.Allocator,
        /// The MTP layer as a one-layer config (its cache's and its attention's).
        cfg1: glm.Config,
        weights: sdk.Weights,
        head: Head,
        bank: Bank,
        cache: graph.Cache,
        pending: ?T = null,
        n_pending: u32 = 0,
        depth: u32,
        mode: sdk.acceptance.Mode,
        counts: Counts = .{},
        /// The head's top probability at each step of the last round's drafts.
        top_p: [max_depth]f32 = @splat(0),
        /// Logged once per request that the lane stopped tracking.
        untracked_logged: bool = false,

        /// The lane over the pack at `dir` (its `mtp/`), `depth` drafts a round, `cap` positions in its cache.
        pub fn open(gpa: std.mem.Allocator, io: std.Io, g: *G, dir: []const u8, c: *const glm.Config, depth: u32, mode: sdk.acceptance.Mode, cap: u32, nocache: bool, diag: ?*Diag) !*Self {
            try checkConfig(c, depth, diag);
            _ = try facts(gpa, io, dir, c, kind, diag);
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}/{s}", .{ dir, dir_name, residents_file }, 0);
            defer gpa.free(path);
            var w = sdk.Weights.init(gpa);
            errdefer w.deinit();
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try sdk.loader.file(gpa, &w, path.ptr, cpu, .{ .nocache = nocache });
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .cfg1 = layerConfig(c), .weights = w, .head = undefined, .bank = undefined, .cache = undefined, .depth = depth, .mode = mode };
            var active0: usize = 0;
            _ = mlx.mlx_get_active_memory(&active0);
            self.head = try Head.bind(g, &self.weights, c, diag);
            errdefer self.head.deinit(g);
            self.bank = try Bank.open(gpa, io, dir, c, &self.weights, diag);
            errdefer self.bank.deinit();
            self.cache = try graph.Cache.init(gpa, &self.cfg1, cap);
            // The residents resident now (a lazy load would land in the first round), and their bytes as MLX counts them.
            var it = self.weights.map.valueIterator();
            while (it.next()) |v| try mlx.check(mlx.mlx_array_eval(v.*));
            var active1: usize = 0;
            _ = mlx.mlx_get_active_memory(&active1);
            log.info("glm_moe_dsa: MTP lane depth {d}, acceptance {s}, {d} residents ({s}), {d} B resident with its experts (measured)\n", .{ depth, sdk.acceptance.name(mode), self.weights.count(), path, active1 -| active0 });
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            if (self.pending) |p| g.release(p);
            self.cache.deinit(g);
            self.bank.deinit();
            self.head.deinit(g);
            self.weights.deinit();
            self.gpa.destroy(self);
        }

        /// The pairs appended plus the positions pending: the target length the lane tracks.
        pub fn tracked(self: *const Self) u64 {
            return @as(u64, self.cache.len) + self.n_pending;
        }

        /// The final-normed hidden `[rows, hidden]` of the target positions after the tracked ones, pending. Past
        /// `max_pending` serial positions (a request the lane does not draft) the lane stops tracking: the rows are
        /// dropped and `tracked` stays behind the target.
        pub fn pend(self: *Self, g: *G, target_len: u64, hidden: T, rows: u32, serial: bool) !void {
            if (self.tracked() + rows != target_len) return;
            if (serial and self.n_pending + rows > max_pending) {
                if (!self.untracked_logged) log.info("glm_moe_dsa: MTP lane: {d} serial steps past its last round, the lane stops tracking this request\n", .{self.n_pending});
                self.untracked_logged = true;
                return;
            }
            const m = g.mark();
            defer g.resetTo(m);
            const all = if (self.pending) |p| try g.concat(&.{ p, hidden }, 0) else hidden;
            const kept = g.keep(all);
            try g.evalAll(&.{kept});
            if (self.pending) |p| g.release(p);
            self.pending = kept;
            self.n_pending += rows;
        }

        /// The pairs input `eh_proj(cat[enorm(embed(tokens)), hnorm(hidden)])` `[n, hidden]`.
        fn input(self: *const Self, g: *G, tw: *const graph.Weights, tokens: []const u32, hidden: T) !T {
            const c = &self.cfg1;
            const e = try graph.rmsNorm(g, try graph.embed(g, tw, tokens), self.head.enorm, c.rms_norm_eps);
            const hn = try graph.rmsNorm(g, try g.astype(hidden, g.dtypeOf(e)), self.head.hnorm, c.rms_norm_eps);
            return graph.linear(g, try g.concat(&.{ e, hn }, 1), self.head.eh_proj);
        }

        /// The pending pairs whose next token `tokens` gives (`tokens[i]` follows pending row i), at most
        /// `tokens.len`, appended in chunks of `append_rows`; the rest stay pending.
        pub fn append(self: *Self, g: *G, tw: *const graph.Weights, tokens: []const u32) !void {
            const n: u32 = @intCast(@min(tokens.len, self.n_pending));
            if (n == 0) return;
            const lw = &self.head.layer;
            var at: u32 = 0;
            while (at < n) {
                const end = @min(at + append_rows, n);
                const m = g.mark();
                defer g.resetTo(m);
                const x = try self.input(g, tw, tokens[at..end], try graph.rowSlice(g, self.pending.?, @intCast(at), @intCast(end)));
                const a_in = try graph.rmsNorm(g, x, lw.input_norm, self.cfg1.rms_norm_eps);
                const k = try graph.keysOf(g, &self.cfg1, lw, a_in, self.cache.len, end - at);
                try self.cache.append(g, 0, k.latent, k.k_pe, k.index);
                self.cache.commit(end - at);
                var bufs: [3]T = undefined;
                try g.evalAll(self.cache.buffers(0, &bufs));
                at = end;
            }
            const m = g.mark();
            defer g.resetTo(m);
            const old = self.pending.?;
            self.pending = if (n == self.n_pending) null else g.keep(try graph.rowSlice(g, old, @intCast(n), @intCast(self.n_pending)));
            if (self.pending) |p| try g.evalAll(&.{p});
            g.release(old);
            self.n_pending -= n;
        }

        /// The layer's output for one row `x [1, hidden]` at position `pos` over the cache as it stands, with the
        /// selection `topk` (null: every key).
        fn row(self: *const Self, g: *G, x: T, pos: u32, topk: ?T) !T {
            const c = &self.cfg1;
            const lw = &self.head.layer;
            const a_in = try graph.rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
            const q = try graph.queryOf(g, c, lw, a_in, pos, 1);
            const o = try graph.absorbedRow(g, c, lw, q.q_nope, q.q_pe, try self.cache.latentView(g, 0), try self.cache.ropeView(g, 0), topk);
            const attn = try graph.qlinear(g, try g.reshape(try g.transposeAxes(o, &.{ 0, 2, 1, 3 }), &.{ 1, @as(c_int, @intCast(c.n_heads * c.v_head_dim)) }), lw.o);
            const h1 = try g.add(x, attn);
            const x2 = try graph.rmsNorm(g, h1, lw.post_norm, c.rms_norm_eps);
            const r = try graph.route(g, c, lw.router.?, x2);
            const f = try g.add(try self.bank.moe(g, x2, r.indices, r.weights), try graph.mlp(g, lw.shared.?, x2));
            return g.add(h1, f);
        }

        /// The selection of the cache's last pair's row `x` (its own indexer row over every index key), null while
        /// the keys fit `index_topk`; kept, the caller releases it.
        fn ownSelection(self: *Self, g: *G, a: std.mem.Allocator, x: T, pos: u32) !?T {
            const c = &self.cfg1;
            const lw = &self.head.layer;
            const a_in = try graph.rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
            const q = try graph.queryOf(g, c, lw, a_in, pos, 1);
            var carry = try graph.Carry.init(a, c, 1, pos + 1);
            defer carry.deinit(g);
            try graph.select(g, c, lw.indexer.?, a_in, q.iq.?, try self.cache.indexView(g, 0), pos, 1, &carry);
            return if (carry.topk[0]) |t| g.keep(t) else null;
        }

        /// A round's drafts after `t1` (the token after the target's last position): the pending pairs appended (their
        /// tokens `after` then `t1`), the first step on the last of them, then `depth - 1` steps on the drafts. `force`
        /// (a test's) replaces the first drafts. `logits_out` (a test's, `depth` x vocab) gets each step's logits.
        pub fn draft(self: *Self, g: *G, a: std.mem.Allocator, tw: *const graph.Weights, after: []const u32, t1: u32, depth: u32, out: []u32, force: []const u32, logits_out: ?[]f32) !void {
            std.debug.assert(after.len + 1 == self.n_pending and out.len >= depth and depth >= 1);
            const tokens = try a.alloc(u32, self.n_pending);
            defer a.free(tokens);
            @memcpy(tokens[0..after.len], after);
            tokens[after.len] = t1;
            // Every pending pair but the last leaves only its keys; the last one's row is the first step's.
            try self.append(g, tw, tokens[0 .. tokens.len - 1]);
            const c = &self.cfg1;
            const m0 = g.mark();
            defer g.resetTo(m0);
            var x = try self.input(g, tw, tokens[tokens.len - 1 ..], self.pending.?);
            const lw = &self.head.layer;
            const a_in = try graph.rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
            const k = try graph.keysOf(g, c, lw, a_in, self.cache.len, 1);
            try self.cache.append(g, 0, k.latent, k.k_pe, k.index);
            self.cache.commit(1);
            g.release(self.pending.?);
            self.pending = null;
            self.n_pending = 0;
            const pos = self.cache.len - 1;
            const topk = try self.ownSelection(g, a, x, pos);
            defer if (topk) |t| g.release(t);
            const vocab: usize = c.vocab_size;
            for (0..depth) |step| {
                const s: u32 = @intCast(step);
                const h = try self.row(g, x, pos + s, topk);
                const normed = g.keep(try graph.rmsNorm(g, h, self.head.head_norm, c.rms_norm_eps));
                defer g.release(normed);
                const lg = try graph.qlinear(g, normed, tw.lm_head);
                if (logits_out) |lo| _ = try g.hostF32(try g.astype(lg, .float32), lo[step * vocab ..][0..vocab]);
                // The argmax and the head's top probability `exp(max - logsumexp)`, read together.
                const lf = try g.reshape(try g.astype(lg, .float32), &.{-1});
                const am = try g.argmax(lf, 0);
                const top = try g.exp(try g.sub(try g.max(lf, 0, false), try g.logsumexp(lf, 0, false)));
                try g.evalAll(&.{ am, top });
                var tok: [1]u32 = undefined;
                var tp: [1]f32 = undefined;
                _ = try g.hostU32(am, &tok);
                _ = try g.hostF32(top, &tp);
                self.top_p[step] = tp[0];
                out[step] = if (step < force.len) force[step] else tok[0];
                if (step + 1 < depth) {
                    x = try self.input(g, tw, out[step..][0..1], normed);
                }
            }
            var bufs: [3]T = undefined;
            try g.evalAll(self.cache.buffers(0, &bufs));
        }

        /// The lane's part of a prefix restore: `n` target positions match the new prompt. Returns the positions the
        /// target keeps (at most `n`) so that the lane tracks exactly them: its pairs past them dropped (and a pair
        /// formed with the token after the match), its pending rows past them dropped.
        pub fn keepFor(self: *Self, g: *G, n: u64) u64 {
            const p: u64 = self.cache.len;
            var k: u64 = @min(n, p + self.n_pending);
            if (k > 0 and p >= k and k == n) k -= 1;
            const keep_pairs = @min(p, k);
            self.cache.truncateTo(@intCast(keep_pairs));
            const rows: u32 = @intCast(k - keep_pairs);
            if (rows == 0) {
                if (self.pending) |x| g.release(x);
                self.pending = null;
                self.n_pending = 0;
            } else if (rows < self.n_pending) {
                const m = g.mark();
                defer g.resetTo(m);
                const old = self.pending.?;
                self.pending = g.keep(graph.rowSlice(g, old, 0, @intCast(rows)) catch return k);
                g.release(old);
                self.n_pending = rows;
            }
            self.untracked_logged = false;
            return k;
        }

        pub fn line(self: *const Self) Line {
            return .{ .depth = self.depth, .mode = sdk.acceptance.name(self.mode), .delta = switch (self.mode) {
                .typical => |t| t.delta,
                else => 0,
            }, .c = self.counts };
        }
    };
}

const testing = std.testing;

test "glm mtp: the route limit gives depth 5; the MTP layer config is one full, routed layer" {
    try testing.expectEqual(@as(u32, 5), max_depth);
    var c = try glm.Config.parse(testing.allocator, try tinyText(), null, null);
    defer c.deinit(testing.allocator);
    const m = layerConfig(&c);
    try testing.expectEqual(@as(u32, 1), m.n_layers);
    try testing.expect(m.isFull(0) and m.isSparse(0));
    try testing.expectEqual(@as(u32, 1), m.nSparse());
}

test "glm mtp: the release's MTP residents (23 BF16 tensors, the box's file) pass the spec; the config's flags gate the lane" {
    const text = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    var c = try glm.Config.parse(testing.allocator, text, &glm.glm53, null);
    defer c.deinit(testing.allocator);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), &c, .exl3);
    try testing.expectEqual(@as(usize, 23), spec.len);
    var bytes: u64 = 0;
    for (spec) |p| {
        var n: u64 = 2;
        for (p.shapeOf()) |d| n *= d;
        bytes += n;
    }
    // The box's mtp-residents.safetensors: 578,488,832 tensor bytes (the bias F32: 256 x 4 for 256 x 2 here).
    try testing.expectEqual(@as(u64, 578_488_832 - 512), bytes);
    try testing.expectEqualStrings("model.layers.78.enorm.weight", spec[0].name);
    try checkConfig(&c, 5, null);
    var no = c;
    no.index_share_mtp = false;
    try checkConfig(&no, 1, null);
    var diag: Diag = .{};
    try testing.expectError(error.MtpConfig, checkConfig(&no, 2, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "index_share_for_mtp_iteration") != null);
    no.n_nextn = 0;
    try testing.expectError(error.MtpConfig, checkConfig(&no, 1, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "num_nextn_predict_layers") != null);
}

var tiny_text_buf: [8192]u8 = undefined;

fn tinyText() ![]const u8 {
    var fba = std.heap.FixedBufferAllocator.init(&tiny_text_buf);
    return glm.tinyConfigJson(fba.allocator(), glm.tiny_quant);
}

test "glm mtp: the request's line (the ABBA harness reads it): depth, acceptance, rounds, drafted, accepted, rate, per round, tok/s" {
    const l: Line = .{ .depth = 3, .mode = "exact", .delta = 0, .c = .{ .rounds = 348, .drafted = 1042, .accepted = 676, .generated = 1024, .wall_ns = 207_560_000_000, .draft_ns = 9_000_000_000, .verify_ns = 190_000_000_000, .peak_rise = 312_400_000, .verify_bytes = 2_100_000_000_000, .rejected_bytes = 230_000_000_000, .decided = .{ 9, 0, 0, 0, 0, 0, 0, 0, 30, 990 }, .accepted_by_p = .{ 1, 0, 0, 0, 0, 0, 0, 0, 20, 655 } } };
    const s = try std.fmt.allocPrint(testing.allocator, "{f}", .{l});
    defer testing.allocator.free(s);
    try testing.expectEqualStrings("glm_moe_dsa: mtp depth 3 acceptance exact: 348 rounds (0 without drafts), 1042 drafted, 676 accepted (64.9%), 1.94 accepted / 2.94 tokens per round, 1024 tokens in 207.56 s (4.9 tok/s; drafts 9.00 s, verify 190.00 s; round peak 312 MB), verify reads 2100.00 GB (2.051 GB per emitted token), 230.00 GB of it for rejected rows only", s);
    const ps = try std.fmt.allocPrint(testing.allocator, "{f}", .{ProbLine{ .c = l.c }});
    defer testing.allocator.free(ps);
    try testing.expectEqualStrings("glm_moe_dsa: mtp acceptance by the head's top probability (accepted / decided): 0.0 1/9, 0.1 0/0, 0.2 0/0, 0.3 0/0, 0.4 0/0, 0.5 0/0, 0.6 0/0, 0.7 0/0, 0.8 20/30, 0.9 655/990", ps);
    try testing.expectEqual(@as(usize, 0), tenth(0.0999));
    try testing.expectEqual(@as(usize, 1), tenth(0.1));
    try testing.expectEqual(@as(usize, 9), tenth(1.0));
    const t: Line = .{ .depth = 2, .mode = "typical", .delta = 0.3, .c = .{ .rounds = 2, .drafted = 4, .accepted = 3, .generated = 5, .wall_ns = 1_000_000_000 } };
    const u = try std.fmt.allocPrint(testing.allocator, "{f}", .{t});
    defer testing.allocator.free(u);
    try testing.expect(std.mem.startsWith(u8, u, "glm_moe_dsa: mtp depth 2 acceptance typical (delta 0.3): 2 rounds"));
}

/// A synthetic MTP directory for config `c` (`tp` ranks, experts alternating K3 / K4): the manifest and `mtp-experts.bin`
/// of seeded bytes (each record's logical bytes, zero padding to 4096; with `signs` the suh / svh segments are +/-1 in
/// f16, the trellis seeded codes). Returns the bin's bytes.
pub fn writeSynthMtp(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, tp: u32, multiplier: u64) ![]u8 {
    return writeSynthMtpWith(a, io, dir, c, tp, multiplier, false);
}

pub fn writeSynthMtpWith(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, tp: u32, multiplier: u64, signs: bool) ![]u8 {
    const h: u64 = c.hidden_size;
    const mini: u64 = c.moe_intermediate_size / tp;
    const e = c.n_routed_experts;
    try dir.createDirPath(io, dir_name);
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    try j.print(a, "{{\"format\":\"{s}\",\"model_type\":\"glm_moe_dsa\",\"source\":{{\"repo\":null,\"revision\":null}},\"quantization\":{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"mcg_scalar\":0,\"k_values\":[3,4],\"tp_ranks\":{d}}},", .{ manifest_format, multiplier, tp });
    try j.print(a, "\"dims\":{{\"hidden\":{d},\"inter\":{d},\"mini_inter\":{d},\"n_experts\":{d},\"n_model_layers\":1,\"n_bank_layers\":2}},\"components\":[", .{ h, c.moe_intermediate_size, mini, e });
    for (Manifest.components, 0..) |n, i| try j.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", n });
    try j.appendSlice(a, "],\"layers\":[");
    var base: u64 = 0;
    var image: std.ArrayList(u8) = .empty;
    errdefer image.deinit(a);
    var rng = std.Random.DefaultPrng.init(78);
    for ([_]u32{ 3, 4 }, 0..) |k, b| {
        var members: std.ArrayList(u32) = .empty;
        defer members.deinit(a);
        for (0..e) |x| if ((x % 2 == 0) == (k == 3)) try members.append(a, @intCast(x));
        var logical: u64 = 0;
        var segs: std.ArrayList(u8) = .empty;
        defer segs.deinit(a);
        for (0..9) |ci| {
            const sh = Manifest.segmentShape(ci, h, mini, k);
            var n: u64 = 2;
            for (sh.slice()) |x| n *= x;
            try segs.print(a, "{s}{{\"component\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ if (ci == 0) "" else ",", Manifest.components[ci], if (ci % 3 == 0) "I16" else "F16" });
            for (sh.slice(), 0..) |x, i| try segs.print(a, "{s}{d}", .{ if (i == 0) "" else ",", x });
            try segs.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ logical, n });
            logical += n;
        }
        const record = std.mem.alignForward(u64, logical, 4096);
        const n_minis = tp * members.items.len;
        try j.print(a, "{s}{{\"bank_layer\":{d},\"layer\":{d},\"k\":{d},\"mtp\":true,\"n_minis\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"base_offset\":{d},\"experts\":[", .{ if (b == 0) "" else ",", b, c.n_layers, k, n_minis, record, logical, base });
        for (members.items, 0..) |x, i| try j.print(a, "{s}{d}", .{ if (i == 0) "" else ",", x });
        try j.print(a, "],\"segments\":[{s}]}}", .{segs.items});
        for (0..n_minis) |_| {
            const at = image.items.len;
            try image.appendNTimes(a, 0, @intCast(record));
            rng.random().bytes(image.items[at..][0..@intCast(logical)]);
            if (signs) {
                var off: u64 = 0;
                for (0..9) |ci| {
                    const sh = Manifest.segmentShape(ci, h, mini, k);
                    var n: u64 = 1;
                    for (sh.slice()) |x| n *= x;
                    if (ci % 3 != 0) for (0..n) |i| {
                        const v: u16 = if (rng.random().boolean()) 0x3C00 else 0xBC00;
                        std.mem.writeInt(u16, image.items[at + off + 2 * i ..][0..2], v, .little);
                    };
                    off += 2 * n;
                }
            }
        }
        base += n_minis * record;
    }
    try j.print(a, "],\"experts\":{{\"{d}\":[", .{c.n_layers});
    for (0..e) |x| try j.print(a, "{s}[{d},{d}]", .{ if (x == 0) "" else ",", if (x % 2 == 0) @as(u32, 3) else 4, x / 2 });
    try j.print(a, "]}},\"sidecar\":{{\"file\":\"{s}\",\"alignment\":4096,\"size\":{d}}},\"records\":[],\"parity\":{{\"all_pass\":true,\"checked\":0,\"total\":0,\"method\":\"bytes-equal-source\"}}}}", .{ bank_file, base });
    var sub = try dir.openDir(io, dir_name, .{});
    defer sub.close(io);
    try sub.writeFile(io, .{ .sub_path = manifest_file, .data = j.items });
    try sub.writeFile(io, .{ .sub_path = bank_file, .data = image.items });
    return image.toOwnedSlice(a);
}

test "glm mtp: the EXL3 manifest loads its bank layers and expert map, and every departure from the format is refused by name" {
    const a = testing.allocator;
    var c = try glm.Config.parse(a, try tinyText(), null, null);
    defer c.deinit(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const img = try writeSynthMtp(a, testing.io, tmp.dir, &c, 4, sushi.format.MCG_MULT);
    defer a.free(img);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const mdir = try std.fmt.allocPrint(arena.allocator(), "{s}/{s}", .{ root, dir_name });
    var diag: Diag = .{};
    const m = Manifest.load(arena.allocator(), testing.io, mdir, &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expectEqual(@as(usize, 2), m.layers.len);
    try testing.expectEqual(@as(u32, 3), m.layers[0].k);
    try testing.expectEqual(@as(u32, 32), m.layers[1].n_minis);
    try testing.expectEqual([2]u32{ 4, 3 }, m.experts[7]);
    try testing.expectEqual(@as(u32, 16), m.mini_inter);
    // A wrong codebook multiplier, a short sidecar, a config of other dims.
    const bad = try writeSynthMtp(a, testing.io, tmp.dir, &c, 4, 0x83DCD12D);
    a.free(bad);
    try testing.expectError(error.MtpManifest, Manifest.load(arena.allocator(), testing.io, mdir, &c, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "multiplier") != null);
    a.free(try writeSynthMtp(a, testing.io, tmp.dir, &c, 4, sushi.format.MCG_MULT));
    var sub = try tmp.dir.openDir(testing.io, dir_name, .{});
    defer sub.close(testing.io);
    try sub.writeFile(testing.io, .{ .sub_path = bank_file, .data = img[0 .. img.len - 4096] });
    try testing.expectError(error.MtpBankSize, Manifest.load(arena.allocator(), testing.io, mdir, &c, &diag));
    try sub.writeFile(testing.io, .{ .sub_path = bank_file, .data = img });
    var other = c;
    other.hidden_size = 256;
    try testing.expectError(error.MtpManifest, Manifest.load(arena.allocator(), testing.io, mdir, &other, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "dims") != null);
    // Two ranks: the same experts in minis of twice the width.
    a.free(try writeSynthMtp(a, testing.io, tmp.dir, &c, 2, sushi.format.MCG_MULT));
    const m2 = try Manifest.load(arena.allocator(), testing.io, mdir, &c, &diag);
    try testing.expectEqual(@as(u32, 16), m2.layers[0].n_minis);
    try testing.expectEqual(@as(u32, 32), m2.mini_inter);
}

test "glm mtp: the EXL3 bank's arrays are the records' segments byte for byte, [minis, ...], and each expert's first slot is its bank offset + local x tp" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    var c = try glm.Config.parse(a, try tinyText(), null, null);
    defer c.deinit(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const img = try writeSynthMtp(a, testing.io, tmp.dir, &c, 4, sushi.format.MCG_MULT);
    defer a.free(img);
    var w = sdk.Weights.init(a);
    defer w.deinit();
    var b = try Exl3.open(a, testing.io, root, &c, &w, null);
    defer b.deinit();
    try testing.expectEqual(@as(usize, 2), b.n_banks);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const m = try Manifest.load(arena.allocator(), testing.io, try std.fmt.allocPrint(arena.allocator(), "{s}/{s}", .{ root, dir_name }), &c, null);
    for (m.layers, 0..) |l, bi| {
        const bank = b.banks[bi];
        const arrs = [9]mlx.mlx_array{ bank.gate.trellis, bank.gate.svh, bank.gate.suh, bank.up.trellis, bank.up.svh, bank.up.suh, bank.down.trellis, bank.down.svh, bank.down.suh };
        for (l.segments, arrs, 0..) |sg, arr, ci| {
            try mlx.check(mlx.mlx_array_eval(arr));
            try testing.expectEqual(if (ci % 3 == 0) mlx.mlx_dtype.uint16 else mlx.mlx_dtype.float16, mlx.mlx_array_dtype(arr));
            try testing.expectEqual(@as(usize, l.n_minis) * sg.length, (mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr)));
            const bytes = mlx.mlx_array_data_uint8(arr).?[0..(mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr))];
            for (0..l.n_minis) |mi| {
                const off = l.base_offset + mi * l.record_bytes + sg.offset;
                try testing.expectEqualSlices(u8, img[off..][0..sg.length], bytes[mi * sg.length ..][0..sg.length]);
            }
        }
    }
    try mlx.check(mlx.mlx_array_eval(b.base));
    const base = mlx.mlx_array_data_int32(b.base).?;
    for (0..c.n_routed_experts) |e| try testing.expectEqual(@as(i32, @intCast(if (e % 2 == 0) (e / 2) * 4 else 32 + (e / 2) * 4)), base[e]);
}

test "glm mtp: the EXL3 bank's MoE on the GPU equals sushi's host decode of the same records (each routed expert = its 4 minis at its score)" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    // hidden 128 and minis of 128 (inter 512 over 4 ranks): the kernels' Hadamard blocks.
    const base_text = try tinyText();
    const text = try std.mem.replaceOwned(u8, a, base_text, "\"moe_intermediate_size\":64,", "\"moe_intermediate_size\":512,");
    defer a.free(text);
    var c = try glm.Config.parse(a, text, null, null);
    defer c.deinit(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    const img = try writeSynthMtpWith(a, testing.io, tmp.dir, &c, 4, sushi.format.MCG_MULT, true);
    defer a.free(img);
    var w = sdk.Weights.init(a);
    defer w.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var b = try Exl3.open(a, testing.io, root, &c, &w, null);
    defer b.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const m = try Manifest.load(arena.allocator(), testing.io, try std.fmt.allocPrint(arena.allocator(), "{s}/{s}", .{ root, dir_name }), &c, null);
    const h = c.hidden_size;
    const n = 2;
    const k = 8;
    var rng = std.Random.DefaultPrng.init(5);
    var xs: [n * 128]f32 = undefined;
    for (&xs) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
    const ids = [n * k]i32{ 0, 3, 5, 6, 9, 10, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14 };
    var sc: [n * k]f32 = undefined;
    for (&sc) |*v| v.* = 0.05 + rng.random().float(f32) * 0.2;
    const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&xs), &.{ n, @intCast(h) }, .float32), .bfloat16);
    var xin: [n * 128]f32 = undefined;
    _ = try g.hostF32(try g.astype(xb, .float32), &xin);
    const y = try b.moe(&g, xb, try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ n, k }, .int32), try g.hostArray(std.mem.sliceAsBytes(&sc), &.{ n, k }, .float32));
    var got: [n * 128]f32 = undefined;
    _ = try g.hostF32(try g.astype(y, .float32), &got);
    // The host reference: sushi's own decode (`format.project`) of each mini's three projections from the records.
    var want: [n * 128]f32 = @splat(0);
    const mini: usize = m.mini_inter;
    var t_in: [512]f32 = undefined;
    var inner: [512]f32 = undefined;
    var gate: [512]f32 = undefined;
    var up: [512]f32 = undefined;
    var down: [512]f32 = undefined;
    for (0..n) |row| for (0..k) |j| {
        const e: usize = @intCast(ids[row * k + j]);
        const kl = m.experts[e];
        const l = for (m.layers) |l| {
            if (l.k == kl[0]) break l;
        } else unreachable;
        const rate = sushi.format.kFromPackedDim(16 * l.k).?;
        for (0..m.tp) |r| {
            const rec = img[l.base_offset + (kl[1] * m.tp + r) * l.record_bytes ..];
            const seg = struct {
                fn f(ll: Manifest.Layer, bytes: []const u8, ci: usize) []const u16 {
                    const sg = ll.segments[ci];
                    return @alignCast(std.mem.bytesAsSlice(u16, bytes[sg.offset..][0..sg.length]));
                }
            }.f;
            const x = xin[row * h ..][0..h];
            sushi.format.project(x, seg(l, rec, 0), seg(l, rec, 2), seg(l, rec, 1), h, mini, rate, .mcg, t_in[0..h], inner[0..mini], gate[0..mini]);
            sushi.format.project(x, seg(l, rec, 3), seg(l, rec, 5), seg(l, rec, 4), h, mini, rate, .mcg, t_in[0..h], inner[0..mini], up[0..mini]);
            var act: [512]f32 = undefined;
            for (act[0..mini], gate[0..mini], up[0..mini]) |*o, gv, uv| o.* = gv / (1 + @exp(-gv)) * uv;
            sushi.format.project(act[0..mini], seg(l, rec, 6), seg(l, rec, 8), seg(l, rec, 7), mini, h, rate, .mcg, t_in[0..mini], inner[0..h], down[0..h]);
            for (want[row * h ..][0..h], down[0..h]) |*o, d| o.* += sc[row * k + j] * d;
        }
    };
    var scale: f32 = 0;
    var worst: f32 = 0;
    for (want, got) |x, y_| {
        scale = @max(scale, @abs(x));
        worst = @max(worst, @abs(x - y_));
    }
    std.debug.print("glm mtp exl3: GPU MoE against the host decode, max |delta| {d:.5} at outputs up to {d:.3}\n", .{ worst, scale });
    try testing.expect(scale > 0.1);
    try testing.expect(worst <= 0.03 * scale);
}
