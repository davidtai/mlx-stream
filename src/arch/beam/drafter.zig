//! Beam's DFlash2 drafter (reflection/Beam-1.0-501B-A23B-DFlash2; the reference is z-lab/dflash `DFlash2DraftModel`,
//! model_mlx.py at 07ebd93d): a block drafter conditioned on the trunk's hidden states, from the pack's sidecar
//! (`drafter.safetensors` beside `drafter-manifest.json`, format `mlx-stream-dflash2-drafter-v1`: the release's file byte
//! for byte and every tensor's sha256, written by the converter's `drafter` step). Construction checks the manifest
//! against the trunk and verifies every tensor (sdk_ext/sidecar.zig) before its bytes become arrays.
//!
//! The context: the trunk's residual stream at the end of each target layer (`target_layer_ids`; 4, 16, 28, 40 and 46
//! on Beam 1.0), the targets concatenated, `fc`, `hidden_norm`; then each drafter layer's keys (k_proj, k_norm, RoPE at
//! the row's position) and values (v_proj) go to that layer's ring (MiMo's `SlidingRing` at the drafter's window): the
//! newest window - 1 positions, all a block can attend.
//!
//! A draft at trunk length p (t1 at p, not in the trunk's KV): the block [t1, mask x d] at positions p .. p + d, the
//! trunk's embedding rows (no scale), through every layer:
//!
//!   (h, k2) = attention_conv.prepare(input_layernorm(x));    x = x + attention_conv.finish(attn(h), k2)
//!   (h, k2) = mlp_conv.prepare(post_attention_layernorm(x));  x = x + mlp_conv.finish(mlp(h), k2)
//!
//! `prepare` is the grouped two-tap dynamic causal conv of h with base_kernel[0] and the first half of
//! kernel_projection(h); `finish` the conv of its input with base_kernel[1] and the second half. The conv mixes block row
//! t with row t - 1 (zero before the anchor), per group of `conv_group_size` columns: out[t] = sum_o (base[o] +
//! dyn[t, o, group]) x in[t - o], in bf16 in the reference's order. The attention: the queries and the block's own keys
//! from h (q_norm, k_norm, RoPE at the block's positions), the ring's context keys before them; block row i attends
//! context position j when p + i - j < window, and every block row (no causal mask). The block's keys go into the ring's
//! headroom for the attention and leave it after (`trimBy`). The MLP is Qwen3's bf16 SwiGLU. Then `norm`, the anchor row
//! dropped, the trunk's lm_head: the [d, vocab] bf16 logits.
//!
//! The selector: each position's `selector_top_k` candidates by the raw logits (MLX's argpartition, as the reference),
//! their unary scores the soft-capped logits in f32 (SGLang's `DFlash2DraftModel`, the release's serving path: the cap on
//! the k values, f32; z-lab's MLX reference caps the whole row in bf16), and for every (previous candidate, candidate)
//! pair the bigram edge sum_r (pred[prev][r] x hp[t][r]) x succ[cand][r] (hp = hidden_projection of the row; the
//! codebooks), each product rounded to bf16 as the reference rounds it. One eval brings the scores and the candidates to
//! the host, which walks the greedy path from t1 (the first largest score, as the references' argmax).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const sidecar = @import("../../sdk_ext/sidecar.zig");
const kv = @import("../mimo_v2/kv.zig");
const beam = @import("config.zig");
const graph = @import("graph.zig");

const G = graph.G;
const T = graph.T;

pub const manifest_file = "drafter-manifest.json";
pub const data_file = "drafter.safetensors";
pub const family = "mlx-stream";
pub const format = "mlx-stream-dflash2-drafter-v1";
pub const max_layers = 8;
pub const max_targets = 8;
/// The selector's candidates a position at most (the host's score buffers).
pub const max_top_k = 32;
/// A block's rows at most: the trained block (16) is the most a served depth (`settings.max_draft_depth` 15) drafts.
pub const max_block = 16;
/// A ring append's rows at most: the ring's headroom (MiMo's prompt tile), so the ring's two slots keep their size.
pub const append_rows: u32 = kv.ring_geometry.headroom;
const manifest_limit = 4 << 20;
const header_limit = 1 << 20;

pub const Error = error{ BeamDrafterMissing, BeamDrafterManifestSyntax, BeamDrafterManifestField, BeamDrafterManifestFormat, BeamDrafterGeometry, BeamDrafterUnsupported, BeamDrafterTensors };

/// The drafter's geometry, from the manifest's `config` (the release's config.json and its `dflash_config`).
pub const Config = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    inter: u32,
    vocab: u32,
    eps: f32,
    rope_theta: f32,
    /// The context keys a block row attends: positions p + i - j < window.
    window: u32,
    num_target_layers: u32,
    block: u32,
    mask_id: u32,
    targets: [max_targets]u32 = @splat(0),
    n_targets: u32,
    conv_k: u32,
    conv_g: u32,
    rank: u32,
    top_k: u32,
    softcap: f32,

    pub fn targetSlice(c: *const Config) []const u32 {
        return c.targets[0..c.n_targets];
    }

    /// Layer `l`'s index among the targets, or null.
    pub fn targetIndex(c: *const Config, l: u32) ?usize {
        return std.mem.indexOfScalar(u32, c.targetSlice(), l);
    }

    pub fn groups(c: *const Config) u32 {
        return c.hidden / c.conv_g;
    }
};

/// The tensors of the drafter (the release's names, all BF16), by role.
const Role = enum { fc, hidden_norm, norm, hproj, pred, succ, in_norm, post_norm, q, k, v, o, q_norm, k_norm, gate, up, down, attn_base, attn_kproj, mlp_base, mlp_kproj };
const RoleName = struct { Role, []const u8 };
const global_roles = [_]RoleName{
    .{ .fc, "fc.weight" },                                 .{ .hidden_norm, "hidden_norm.weight" },
    .{ .norm, "norm.weight" },                             .{ .hproj, "candidate_selector.hidden_projection.weight" },
    .{ .pred, "candidate_selector.predecessor_codebook" }, .{ .succ, "candidate_selector.successor_codebook" },
};
const layer_roles = [_]RoleName{
    .{ .in_norm, "input_layernorm.weight" },                     .{ .post_norm, "post_attention_layernorm.weight" },
    .{ .q, "self_attn.q_proj.weight" },                          .{ .k, "self_attn.k_proj.weight" },
    .{ .v, "self_attn.v_proj.weight" },                          .{ .o, "self_attn.o_proj.weight" },
    .{ .q_norm, "self_attn.q_norm.weight" },                     .{ .k_norm, "self_attn.k_norm.weight" },
    .{ .gate, "mlp.gate_proj.weight" },                          .{ .up, "mlp.up_proj.weight" },
    .{ .down, "mlp.down_proj.weight" },                          .{ .attn_base, "attention_conv.base_kernel" },
    .{ .attn_kproj, "attention_conv.kernel_projection.weight" }, .{ .mlp_base, "mlp_conv.base_kernel" },
    .{ .mlp_kproj, "mlp_conv.kernel_projection.weight" },
};

/// A tensor's shape under `c` (rank and dims).
fn shapeOf(c: *const Config, r: Role) struct { rank: u8, dims: [3]u64 } {
    const H: u64 = c.hidden;
    const q: u64 = @as(u64, c.heads) * c.head_dim;
    const kvd: u64 = @as(u64, c.kv_heads) * c.head_dim;
    const kp: u64 = 2 * @as(u64, c.conv_k) * c.groups();
    return switch (r) {
        .fc => .{ .rank = 2, .dims = .{ H, c.n_targets * H, 0 } },
        .hidden_norm, .norm, .in_norm, .post_norm => .{ .rank = 1, .dims = .{ H, 0, 0 } },
        .hproj => .{ .rank = 2, .dims = .{ c.rank, H, 0 } },
        .pred, .succ => .{ .rank = 2, .dims = .{ c.vocab, c.rank, 0 } },
        .q => .{ .rank = 2, .dims = .{ q, H, 0 } },
        .k, .v => .{ .rank = 2, .dims = .{ kvd, H, 0 } },
        .o => .{ .rank = 2, .dims = .{ H, q, 0 } },
        .q_norm, .k_norm => .{ .rank = 1, .dims = .{ c.head_dim, 0, 0 } },
        .gate, .up => .{ .rank = 2, .dims = .{ c.inter, H, 0 } },
        .down => .{ .rank = 2, .dims = .{ H, c.inter, 0 } },
        .attn_base, .mlp_base => .{ .rank = 3, .dims = .{ 2, c.conv_k, H } },
        .attn_kproj, .mlp_kproj => .{ .rank = 2, .dims = .{ kp, H, 0 } },
    };
}

/// The manifest, checked against the trunk (no tensor bytes read): the geometry, and every tensor where `c` puts it.
pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    config: Config,
    entries: []sidecar.Entry,
    /// Each entry's role and layer (globals: layer 0).
    roles: []Role,
    layers: []u8,
    data_size: u64,
    data_sha256: [32]u8,
    payload_bytes: u64,
    header_bytes: u64,

    pub fn deinit(m: *Manifest) void {
        m.arena.deinit();
    }

    /// `text` (drafter-manifest.json) checked against the trunk `trunk`.
    pub fn parse(a: std.mem.Allocator, text: []const u8, trunk: *const beam.Config) Error!Manifest {
        var out: Manifest = .{ .arena = std.heap.ArenaAllocator.init(a), .config = undefined, .entries = &.{}, .roles = &.{}, .layers = &.{}, .data_size = 0, .data_sha256 = undefined, .payload_bytes = 0, .header_bytes = 0 };
        errdefer out.arena.deinit();
        const aa = out.arena.allocator();
        const root = std.json.parseFromSliceLeaky(std.json.Value, aa, text, .{}) catch return error.BeamDrafterManifestSyntax;
        if (root != .object) return error.BeamDrafterManifestSyntax;
        const o = root.object;
        try expectString(o, "family", family);
        try expectString(o, "format", format);
        try expectString(o, "data_file", data_file);
        out.data_size = try uint(o, "data_size");
        out.payload_bytes = try uint(o, "payload_bytes");
        out.header_bytes = try uint(o, "header_bytes");
        out.data_sha256 = sidecar.hex32(o.get("data_sha256")) orelse return error.BeamDrafterManifestField;
        if (out.header_bytes + out.payload_bytes != out.data_size) return error.BeamDrafterManifestField;
        const cv = o.get("config") orelse return error.BeamDrafterManifestField;
        if (cv != .object) return error.BeamDrafterManifestField;
        out.config = try parseConfig(cv.object, trunk);
        const c = &out.config;
        const tensors = o.get("tensors") orelse return error.BeamDrafterManifestField;
        if (tensors != .object) return error.BeamDrafterManifestField;
        const n = global_roles.len + @as(usize, c.layers) * layer_roles.len;
        if (tensors.object.count() != n) return error.BeamDrafterTensors;
        out.entries = aa.alloc(sidecar.Entry, n) catch return error.BeamDrafterManifestSyntax;
        out.roles = aa.alloc(Role, n) catch return error.BeamDrafterManifestSyntax;
        out.layers = aa.alloc(u8, n) catch return error.BeamDrafterManifestSyntax;
        var i: usize = 0;
        var total: u64 = 0;
        for (0..c.layers + 1) |li| {
            const roles: []const RoleName = if (li == 0) &global_roles else &layer_roles;
            for (roles) |r| {
                const name = if (li == 0) r[1] else std.fmt.allocPrint(aa, "layers.{d}.{s}", .{ li - 1, r[1] }) catch return error.BeamDrafterManifestSyntax;
                const t = tensors.object.get(name) orelse return error.BeamDrafterTensors;
                if (t != .object) return error.BeamDrafterTensors;
                const sh = shapeOf(c, r[0]);
                const dt = t.object.get("dtype") orelse return error.BeamDrafterTensors;
                if (dt != .string or !std.mem.eql(u8, dt.string, "BF16")) return error.BeamDrafterTensors;
                const dims = t.object.get("shape") orelse return error.BeamDrafterTensors;
                if (dims != .array or dims.array.items.len != sh.rank) return error.BeamDrafterTensors;
                var elems: u64 = 1;
                for (dims.array.items, sh.dims[0..sh.rank]) |d, want| {
                    if (d != .integer or d.integer != want) return error.BeamDrafterTensors;
                    elems *= want;
                }
                const len = try uint(t.object, "length");
                if (len != 2 * elems) return error.BeamDrafterTensors;
                out.entries[i] = .{ .name = name, .dtype = "BF16", .rank = sh.rank, .offset = try uint(t.object, "offset"), .length = len, .sha256 = sidecar.hex32(t.object.get("sha256")) orelse return error.BeamDrafterManifestField };
                for (sh.dims[0..sh.rank], 0..) |d, k| out.entries[i].dims[k] = d;
                out.roles[i] = r[0];
                out.layers[i] = @intCast(if (li == 0) 0 else li - 1);
                total += len;
                i += 1;
            }
        }
        if (total != out.payload_bytes) return error.BeamDrafterManifestField;
        return out;
    }
};

/// The manifest's `config` against the trunk and the served form, refused by name.
fn parseConfig(o: std.json.ObjectMap, trunk: *const beam.Config) Error!Config {
    var c: Config = .{
        .hidden = try u32Of(o, "hidden_size"),
        .layers = try u32Of(o, "num_hidden_layers"),
        .heads = try u32Of(o, "num_attention_heads"),
        .kv_heads = try u32Of(o, "num_key_value_heads"),
        .head_dim = try u32Of(o, "head_dim"),
        .inter = try u32Of(o, "intermediate_size"),
        .vocab = try u32Of(o, "vocab_size"),
        .eps = @floatCast(try float(o, "rms_norm_eps")),
        .rope_theta = @floatCast(try float(o, "rope_theta")),
        .window = try u32Of(o, "sliding_window"),
        .num_target_layers = try u32Of(o, "num_target_layers"),
        .block = try u32Of(o, "block_size"),
        .mask_id = try u32Of(o, "mask_token_id"),
        .n_targets = 0,
        .conv_k = try u32Of(o, "conv_kernel_size"),
        .conv_g = try u32Of(o, "conv_group_size"),
        .rank = try u32Of(o, "selector_rank"),
        .top_k = try u32Of(o, "selector_top_k"),
        .softcap = @floatCast(try float(o, "final_logit_softcapping")),
    };
    const tl = o.get("target_layer_ids") orelse return error.BeamDrafterManifestField;
    if (tl != .array or tl.array.items.len == 0 or tl.array.items.len > max_targets) return error.BeamDrafterManifestField;
    for (tl.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0) return error.BeamDrafterManifestField;
        c.targets[i] = @intCast(v.integer);
    }
    c.n_targets = @intCast(tl.array.items.len);
    // The trunk the drafter reads: its embedding and head (hidden, vocab), its layers.
    if (c.hidden != trunk.hidden_size or c.vocab != trunk.vocab_size or c.num_target_layers != trunk.n_layers) return error.BeamDrafterGeometry;
    for (c.targetSlice()) |t| if (t >= trunk.n_layers) return error.BeamDrafterGeometry;
    // The served form: every layer sliding over a window, a non-causal block, no input or output scale.
    const types = o.get("layer_types") orelse return error.BeamDrafterManifestField;
    if (types != .array or types.array.items.len != c.layers) return error.BeamDrafterManifestField;
    for (types.array.items) |v| if (v != .string or !std.mem.eql(u8, v.string, "sliding_attention")) return error.BeamDrafterUnsupported;
    const causal = o.get("is_causal") orelse return error.BeamDrafterManifestField;
    if (causal != .bool or causal.bool) return error.BeamDrafterUnsupported;
    if (try float(o, "input_embedding_scale") != 1 or try float(o, "output_multiplier") != 1) return error.BeamDrafterUnsupported;
    if (c.layers == 0 or c.layers > max_layers or c.kv_heads == 0 or c.heads % c.kv_heads != 0) return error.BeamDrafterUnsupported;
    // MLX's fused attention; the conv's groups; the selector's host buffers; a block within the served depths.
    if (c.head_dim != 64 and c.head_dim != 128) return error.BeamDrafterUnsupported;
    if (c.conv_k == 0 or c.conv_k > 4 or c.conv_g == 0 or c.hidden % c.conv_g != 0) return error.BeamDrafterUnsupported;
    if (c.top_k == 0 or c.top_k > max_top_k or c.top_k > c.vocab or c.rank == 0) return error.BeamDrafterUnsupported;
    if (c.block < 2 or c.block > max_block or c.mask_id >= c.vocab or c.window < c.block or !(c.softcap > 0)) return error.BeamDrafterUnsupported;
    return c;
}

fn expectString(o: std.json.ObjectMap, k: []const u8, want: []const u8) Error!void {
    const v = o.get(k) orelse return error.BeamDrafterManifestField;
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.BeamDrafterManifestFormat;
}
fn uint(o: std.json.ObjectMap, k: []const u8) Error!u64 {
    const v = o.get(k) orelse return error.BeamDrafterManifestField;
    if (v != .integer or v.integer < 0) return error.BeamDrafterManifestField;
    return @intCast(v.integer);
}
fn u32Of(o: std.json.ObjectMap, k: []const u8) Error!u32 {
    const v = try uint(o, k);
    if (v > std.math.maxInt(u32)) return error.BeamDrafterManifestField;
    return @intCast(v);
}
fn float(o: std.json.ObjectMap, k: []const u8) Error!f64 {
    const v = o.get(k) orelse return error.BeamDrafterManifestField;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BeamDrafterManifestField,
    };
}

/// The pack's drafter manifest (pure host: no tensor read): `BeamDrafterMissing` when the pack has none.
pub fn readManifest(a: std.mem.Allocator, io: std.Io, dir: []const u8, trunk: *const beam.Config) !Manifest {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = d.readFileAlloc(io, manifest_file, a, .limited(manifest_limit)) catch |e| return switch (e) {
        error.FileNotFound => error.BeamDrafterMissing,
        else => e,
    };
    defer a.free(text);
    return Manifest.parse(a, text, trunk);
}

/// The drafter's resident bytes as MLX allocates them (each tensor in its page-rounded buffer, the base kernels' halves
/// copied out), and the arrays (the bill's census).
pub fn residentBytes(m: *const Manifest, page: u64) struct { bytes: u64, arrays: u64 } {
    var b: u64 = 0;
    for (m.entries) |e| b += graph.bufferBytes(e.length, page);
    const c = &m.config;
    const halves: u64 = 2 * @as(u64, c.layers) * 2 * c.conv_k;
    b += halves * graph.bufferBytes(@as(u64, c.hidden) * 2, page);
    return .{ .bytes = b, .arrays = m.entries.len + halves };
}

/// Every context ring (one per layer, K and V, both slots at the ring base: MiMo's `SlidingRing.allocate`), bf16.
pub fn contextBytes(c: *const Config, page: u64) u64 {
    const row: u64 = @as(u64, c.kv_heads) * c.head_dim * 2;
    return @as(u64, c.layers) * 2 * 2 * graph.bufferBytes(kv.ringBase(c.window) * row, page);
}

/// A draft's arrays at a block of `n` rows over `ctx` context keys (every array the graph builds, live until its eval):
/// the block's stream (the embedding, each layer's norms, the convs' projections and products, the residual adds), the
/// q / k / v / o projections and their head views, the attention's scores (f32, materialized beside a mask), the MLP
/// (gate, up, SiLU, product, down), the head's logits (bf16, the soft cap's stages, the gate's f32 log-sum-exp) and the
/// selector (the candidates, their unary scores and codebook rows, the pair products and edges, the f32 scores).
pub fn roundBytes(c: *const Config, n: u64, ctx: u64) u64 {
    const H: u64 = c.hidden;
    const q: u64 = @as(u64, c.heads) * c.head_dim;
    const kvd: u64 = @as(u64, c.kv_heads) * c.head_dim;
    const kp: u64 = 2 * @as(u64, c.conv_k) * c.groups();
    const k: u64 = c.top_k;
    const stream = n * H * 2 * 16 + n * kp * 2 * 4;
    const attn = n * (3 * q + 4 * kvd) * 2 + @as(u64, c.heads) * n * (ctx + n) * 4 + n * (ctx + n);
    const mlp = n * @as(u64, c.inter) * 2 * 4;
    const head = n * @as(u64, c.vocab) * (2 * 4 + 4 * 2);
    const sel = n * k * (2 * 3 + 4) + n * @as(u64, c.rank) * 2 + n * k * @as(u64, c.rank) * 2 * 3 + n * k * k * (@as(u64, c.rank) * 2 + 2 * 2 + 4);
    return stream + @max(attn, mlp) + head + sel;
}

/// The context's arrays for one append of `n` rows (a prompt tile or a round's committed rows): the targets' rows
/// concatenated, `fc`'s output and its norm, each layer's keys and values (one layer's graph at a time is billed whole:
/// the appends are lazy until the round's eval).
pub fn appendBytes(c: *const Config, n: u64) u64 {
    const H: u64 = c.hidden;
    const kvd: u64 = @as(u64, c.kv_heads) * c.head_dim;
    return n * (@as(u64, c.n_targets) * H + 2 * H) * 2 + @as(u64, c.layers) * n * kvd * 2 * 4;
}

/// The prompt pass's captures: each target layer's last window - 1 rows (bf16, contiguous), held until the call's end.
pub fn promptCaptureBytes(c: *const Config, rows: u64, page: u64) u64 {
    const keep = @min(rows, @as(u64, c.window) - 1);
    return @as(u64, c.n_targets) * graph.bufferBytes(keep * c.hidden * 2, page);
}

// ── The weights ──

pub const Layer = struct {
    in_norm: T = .{},
    post_norm: T = .{},
    q: T = .{},
    k: T = .{},
    v: T = .{},
    o: T = .{},
    q_norm: T = .{},
    k_norm: T = .{},
    gate: T = .{},
    up: T = .{},
    down: T = .{},
    attn_base: T = .{},
    attn_kproj: T = .{},
    mlp_base: T = .{},
    mlp_kproj: T = .{},
    /// base_kernel[half][offset] as [1, groups, group size] (bf16, contiguous), for the attention's conv and the MLP's.
    attn_taps: [2][4]T = @splat(@splat(.{})),
    mlp_taps: [2][4]T = @splat(@splat(.{})),
};

pub const Weights = struct {
    fc: T = .{},
    hidden_norm: T = .{},
    norm: T = .{},
    hproj: T = .{},
    pred: T = .{},
    succ: T = .{},
    layers: [max_layers]Layer = @splat(.{}),

    fn slot(w: *Weights, r: Role, l: usize) *T {
        const lw = &w.layers[l];
        return switch (r) {
            .fc => &w.fc,
            .hidden_norm => &w.hidden_norm,
            .norm => &w.norm,
            .hproj => &w.hproj,
            .pred => &w.pred,
            .succ => &w.succ,
            .in_norm => &lw.in_norm,
            .post_norm => &lw.post_norm,
            .q => &lw.q,
            .k => &lw.k,
            .v => &lw.v,
            .o => &lw.o,
            .q_norm => &lw.q_norm,
            .k_norm => &lw.k_norm,
            .gate => &lw.gate,
            .up => &lw.up,
            .down => &lw.down,
            .attn_base => &lw.attn_base,
            .attn_kproj => &lw.attn_kproj,
            .mlp_base => &lw.mlp_base,
            .mlp_kproj => &lw.mlp_kproj,
        };
    }

    fn deinit(w: *Weights) void {
        for ([_]*T{ &w.fc, &w.hidden_norm, &w.norm, &w.hproj, &w.pred, &w.succ }) |x| free(x);
        for (&w.layers) |*lw| {
            inline for (@typeInfo(Layer).@"struct".field_names) |name| {
                if (comptime @TypeOf(@field(lw.*, name)) == T) free(&@field(lw.*, name));
            }
            for ([_]*[2][4]T{ &lw.attn_taps, &lw.mlp_taps }) |taps| for (taps) |*half| for (half) |*x| free(x);
        }
    }
};

fn free(x: *T) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
    x.* = .{};
}

/// base_kernel [2, K, H] -> tap [half][offset] as [1, groups, group size], contiguous and evaluated.
fn splitTaps(s: mlx.mlx_stream, base: T, c: *const Config, out: *[2][4]T) !void {
    const H: c_int = @intCast(c.hidden);
    for (0..2) |h| for (0..c.conv_k) |o| {
        var sl: T = .{};
        defer free(&sl);
        try mlx.check(mlx.mlx_slice(&sl, base, &[_]c_int{ @intCast(h), @intCast(o), 0 }, 3, &[_]c_int{ @intCast(h + 1), @intCast(o + 1), H }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var r: T = .{};
        defer free(&r);
        try mlx.check(mlx.mlx_reshape(&r, sl, &[_]c_int{ 1, @intCast(c.groups()), @intCast(c.conv_g) }, 3, s));
        var out_x: T = .{};
        errdefer free(&out_x);
        try mlx.check(mlx.mlx_contiguous(&out_x, r, false, s));
        try mlx.check(mlx.mlx_array_eval(out_x));
        out[h][o] = out_x;
    };
}

/// The drafter as installed: its geometry, its weights, a context ring per layer and the context's end.
pub const Drafter = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    c: Config,
    w: Weights = .{},
    rings: [max_layers]kv.SlidingRing = @splat(.{}),
    /// The rings hold the context positions [ctx_end - their rows, ctx_end); null before the request's first append.
    ctx_end: ?u64 = null,
    payload_bytes: u64 = 0,
    data_sha256: [32]u8 = undefined,
    read_ns: u64 = 0,

    /// Construction: the manifest checked against the trunk, every tensor verified (`sidecar.verify`) before its bytes
    /// become arrays, the base kernels' taps split out, a ring per layer at the drafter's window.
    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, trunk: *const beam.Config, s: mlx.mlx_stream) !*Drafter {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var m = try readManifest(a, io, dir, trunk);
        defer m.deinit();
        const self = try a.create(Drafter);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .s = s, .c = m.config, .payload_bytes = m.payload_bytes, .data_sha256 = m.data_sha256 };
        errdefer self.w.deinit();
        const Take = struct {
            m: *const Manifest,
            w: *Weights,
            pub fn take(t: @This(), i: usize, bytes: []const u8) !void {
                const e = t.m.entries[i];
                var dims: [3]c_int = undefined;
                for (dims[0..e.rank], e.dims[0..e.rank]) |*d, v| d.* = @intCast(v);
                const arr = mlx.mlx_array_new_data(bytes.ptr, &dims, e.rank, .bfloat16);
                if (arr.ctx == null) return error.BeamDrafterArray;
                t.w.slot(t.m.roles[i], t.m.layers[i]).* = arr;
            }
        };
        sidecar.verify(a, dir, data_file, m.entries, m.data_size, m.data_sha256, m.payload_bytes, header_limit, Take{ .m = &m, .w = &self.w }) catch |e| return switch (e) {
            error.SidecarMissing => error.BeamDrafterMissing,
            error.SidecarSize, error.SidecarHeader, error.SidecarLayout => error.BeamDrafterSidecar,
            error.SidecarTensorSha => error.BeamDrafterTensorSha,
            error.SidecarSha => error.BeamDrafterSidecarSha,
            else => e,
        };
        const c = &self.c;
        for (self.w.layers[0..c.layers]) |*lw| {
            try splitTaps(s, lw.attn_base, c, &lw.attn_taps);
            try splitTaps(s, lw.mlp_base, c, &lw.mlp_taps);
        }
        var n_rings: usize = 0;
        errdefer for (self.rings[0..n_rings]) |*r| r.deinit();
        for (self.rings[0..c.layers]) |*r| {
            r.* = try kv.SlidingRing.allocate(c.kv_heads, c.head_dim, c.head_dim, c.window, s);
            n_rings += 1;
        }
        self.read_ns = @intCast(t0.untilNow(io, .awake).nanoseconds);
        return self;
    }

    pub fn deinit(self: *Drafter) void {
        for (self.rings[0..self.c.layers]) |*r| r.deinit();
        self.w.deinit();
        self.a.destroy(self);
    }

    /// A new request: the rings rewound (their slots kept), no context.
    pub fn restart(self: *Drafter) void {
        for (self.rings[0..self.c.layers]) |*r| r.rewind();
        self.ctx_end = null;
    }

    /// Whether a draft at trunk length `p` has its context: the rings end at p.
    pub fn ready(self: *const Drafter, p: u64) bool {
        return self.ctx_end != null and self.ctx_end.? == p;
    }

    /// Context positions [p0, p0 + n): `caps` the target layers' residual rows (target order, each [n, hidden] bf16, or
    /// a view of more rows whose rows [off, off + n) are these), `fc` and `hidden_norm`, then each layer's keys (k_norm,
    /// RoPE at the positions) and values appended to its ring, in appends of at most `append_rows`. A run of at least
    /// window - 1 positions, or a gap, restarts the rings at p0. Lazy (`evalRings` or the next draft's eval runs it).
    pub fn appendContext(self: *Drafter, g: *G, caps: []const T, off: u32, p0: u64, n: u32) !void {
        const c = &self.c;
        std.debug.assert(caps.len == c.n_targets and n > 0);
        if (self.ctx_end == null or self.ctx_end.? != p0 or n >= c.window - 1) {
            for (self.rings[0..c.layers]) |*r| r.rewind();
        }
        const H: c_int = @intCast(c.hidden);
        var at: u32 = 0;
        while (at < n) {
            const m = @min(append_rows, n - at);
            const mark = g.mark();
            defer g.resetTo(mark);
            var parts: [max_targets]T = undefined;
            for (caps, parts[0..caps.len]) |x, *pt| pt.* = try g.slice(x, &.{ @intCast(off + at), 0 }, &.{ @intCast(off + at + m), H }, &.{ 1, 1 });
            const cat = if (caps.len == 1) parts[0] else try g.concat(parts[0..caps.len], 1);
            const hc = try graph.rmsNorm(g, try graph.linear(g, cat, self.w.fc), self.w.hidden_norm, c.eps);
            const pos: u32 = @intCast(p0 + at);
            for (self.w.layers[0..c.layers], self.rings[0..c.layers]) |*lw, *ring| {
                const k = try graph.rope(g, try headsOf(g, try graph.rmsNorm(g, try g.reshape(try graph.linear(g, hc, lw.k), &.{ @intCast(m), @intCast(c.kv_heads), @intCast(c.head_dim) }), lw.k_norm, c.eps), m, c.kv_heads, c.head_dim), c.head_dim, c.rope_theta, pos);
                const v = try headsOf(g, try graph.linear(g, hc, lw.v), m, c.kv_heads, c.head_dim);
                var view = try ring.append(k, v, m);
                view.deinit();
            }
            at += m;
        }
        self.ctx_end = p0 + n;
    }

    /// Every ring's current slots, evaluated (the prompt pass's context, before its captures go).
    pub fn evalRings(self: *Drafter, g: *G) !void {
        var xs: [2 * max_layers]T = undefined;
        var n: usize = 0;
        for (self.rings[0..self.c.layers]) |*r| for ([_]*kv.SlidingRing.Lane{ &r.k, &r.v }) |lane| if (lane.bufs[lane.cur]) |b| {
            xs[n] = b;
            n += 1;
        };
        if (n > 0) try g.evalAll(xs[0..n]);
    }

    /// The draft at trunk length `p` (t1 there): the block [t1, mask x d] (d + 1 rows), the selector's scores for every
    /// (previous candidate, candidate) pair and the candidates; with `conf`, each candidate's drafter probability (the
    /// gate's). Lazy, in `g`'s scope; `pathOf` walks them on the host.
    pub fn draft(self: *Drafter, g: *G, a: std.mem.Allocator, trunk: *const graph.Weights, t1: u32, p: u64, d: u32, conf: bool) !Draft {
        const c = &self.c;
        std.debug.assert(d >= 1 and d + 1 <= c.block and self.ready(p));
        const n = d + 1;
        var ids: [max_block]u32 = undefined;
        ids[0] = t1;
        @memset(ids[1..n], c.mask_id);
        const x = try g.take(trunk.embed, try g.hostArray(std.mem.sliceAsBytes(ids[0..n]), &.{@intCast(n)}, .uint32), 0);
        const hidden = try self.blockHidden(g, a, x, p);
        const H: c_int = @intCast(c.hidden);
        const rows = try g.slice(hidden, &.{ 1, 0 }, &.{ @intCast(n), H }, &.{ 1, 1 });
        // The trunk's head on the drafter's rows (bf16); the selector caps its candidates' values.
        return self.selector(g, rows, try graph.linear(g, rows, trunk.head), c.softcap, t1, d, conf);
    }

    /// The block's rows after every layer and `norm` (the anchor row kept): `x0` [n, hidden] bf16, the block's input rows
    /// (the embedding rows of [t1, mask ...]; the parity test's noise rows), at positions p .. p + n - 1 over the context.
    pub fn blockHidden(self: *Drafter, g: *G, a: std.mem.Allocator, x0: T, p: u64) !T {
        const c = &self.c;
        const n: u32 = @intCast(g.shapeOf(x0).dim(0));
        std.debug.assert(n >= 1 and n <= max_block);
        var x = x0;
        // Every ring holds the same context rows: the block's keys come after the newest window - 1 of them.
        const ctx: u32 = @min(self.rings[0].position(), c.window - 1);
        const mask = try blockMask(g, a, n, ctx, c.window);
        const pos: u32 = @intCast(p);
        for (self.w.layers[0..c.layers], self.rings[0..c.layers]) |*lw, *ring| {
            const xn = try graph.rmsNorm(g, x, lw.in_norm, c.eps);
            const kp1 = try kernelHalves(g, c, try graph.linear(g, xn, lw.attn_kproj), n);
            const h1 = try gconv(g, c, xn, kp1.first, &lw.attn_taps[0], n);
            const att = try self.attention(g, lw, ring, h1, n, pos, mask);
            x = try g.add(x, try gconv(g, c, att, kp1.second, &lw.attn_taps[1], n));
            const xm = try graph.rmsNorm(g, x, lw.post_norm, c.eps);
            const kp2 = try kernelHalves(g, c, try graph.linear(g, xm, lw.mlp_kproj), n);
            const h2 = try gconv(g, c, xm, kp2.first, &lw.mlp_taps[0], n);
            const mlp = try graph.linear(g, try g.mul(try g.silu(try graph.linear(g, h2, lw.gate)), try graph.linear(g, h2, lw.up)), lw.down);
            x = try g.add(x, try gconv(g, c, mlp, kp2.second, &lw.mlp_taps[1], n));
        }
        return graph.rmsNorm(g, x, self.w.norm, c.eps);
    }

    fn attention(self: *Drafter, g: *G, lw: *const Layer, ring: *kv.SlidingRing, h: T, n: u32, pos: u32, mask: ?T) !T {
        const c = &self.c;
        const ni: c_int = @intCast(n);
        const q = try graph.rope(g, try headsOf(g, try graph.rmsNorm(g, try g.reshape(try graph.linear(g, h, lw.q), &.{ ni, @intCast(c.heads), @intCast(c.head_dim) }), lw.q_norm, c.eps), n, c.heads, c.head_dim), c.head_dim, c.rope_theta, pos);
        const k = try graph.rope(g, try headsOf(g, try graph.rmsNorm(g, try g.reshape(try graph.linear(g, h, lw.k), &.{ ni, @intCast(c.kv_heads), @intCast(c.head_dim) }), lw.k_norm, c.eps), n, c.kv_heads, c.head_dim), c.head_dim, c.rope_theta, pos);
        const v = try headsOf(g, try graph.linear(g, h, lw.v), n, c.kv_heads, c.head_dim);
        // The block's keys after the context's in the ring's headroom; the ring returns to the context after the graph.
        const view = try ring.append(k, v, n);
        const vk = try g.adopt(view.k);
        const vv = try g.adopt(view.v);
        try ring.trimBy(n);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        const out = if (mask) |mk| try graph.sdpa(g, q, vk, vv, scale, "array", mk) else try graph.sdpa(g, q, vk, vv, scale, "", .{});
        const o = try g.reshape(try g.transposeAxes(out, &.{ 0, 2, 1, 3 }), &.{ ni, @intCast(c.heads * c.head_dim) });
        return graph.linear(g, o, lw.o);
    }

    /// The selector's graph over the drafter's rows (`rows` [d, hidden] after `norm`; `logits` [d, vocab], the head's raw
    /// bf16 rows, or the parity test's synthetic f32 rows with `softcap` null), the path from `t1`.
    pub fn selector(self: *Drafter, g: *G, rows: T, logits: T, softcap: ?f32, t1: u32, d: u32, conf: bool) !Draft {
        const c = &self.c;
        const k: c_int = @intCast(c.top_k);
        const di: c_int = @intCast(d);
        const r: c_int = @intCast(c.rank);
        const vocab: c_int = @intCast(c.vocab);
        // The reference's `argpartition(logits, -top_k)[..., -top_k:]`: the same op, so the same candidate order (the cap
        // is monotonic: the raw and the capped rows rank alike).
        const cand = try g.slice(try g.argpartition(logits, vocab - k, -1), &.{ 0, vocab - k }, &.{ di, vocab }, &.{ 1, 1 });
        const capf = if (softcap) |s| try g.scalar(s, .float32) else null;
        const raw = try g.astype(try g.takeAlongAxis(logits, cand, -1), .float32);
        const unary = if (capf) |s| try g.mul(try graph.tanh(g, try g.div(raw, s)), s) else raw;
        const hp = try graph.linear(g, rows, self.w.hproj);
        const flat = try g.reshape(cand, &.{di * k});
        const succ = try g.reshape(try g.take(self.w.succ, flat, 0), &.{ di, k, r });
        // Each position's predecessors: t1 at the first (repeated over the k rows), the previous position's candidates.
        var anchor: [max_top_k]u32 = undefined;
        @memset(anchor[0..c.top_k], t1);
        const first = try g.hostArray(std.mem.sliceAsBytes(anchor[0..c.top_k]), &.{k}, .uint32);
        const prev_ids = if (d == 1) first else try g.concat(&.{ first, try g.slice(flat, &.{0}, &.{(di - 1) * k}, &.{1}) }, 0);
        const pred = try g.reshape(try g.take(self.w.pred, prev_ids, 0), &.{ di, k, r });
        // (pred x hp) x succ, each product bf16, summed over the rank: edges [d, k (previous), k (candidate)].
        const ph = try g.mul(pred, try g.reshape(hp, &.{ di, 1, r }));
        const edges = try g.sum(try g.mul(try g.reshape(ph, &.{ di, k, 1, r }), try g.reshape(succ, &.{ di, 1, k, r })), -1, false);
        const scores = try g.add(try g.reshape(unary, &.{ di, 1, k }), try g.astype(edges, .float32));
        var probs: ?T = null;
        if (conf) {
            // Each candidate's probability under the drafter's capped row (the gate's q).
            const lf = try g.astype(logits, .float32);
            const capped = if (capf) |s| try g.mul(try graph.tanh(g, try g.div(lf, s)), s) else lf;
            probs = try g.exp(try g.sub(unary, try g.logsumexp(capped, -1, true)));
        }
        return .{ .scores = scores, .cand = cand, .probs = probs, .d = d, .k = c.top_k };
    }
};

/// A draft's arrays (lazy, in the backend's scope): `scores` f32 [d, k, k], `cand` u32 [d, k], `probs` f32 [d, k] (each
/// candidate's drafter probability; the gate only).
pub const Draft = struct {
    scores: T,
    cand: T,
    probs: ?T,
    d: u32,
    k: u32,

    /// The arrays one eval needs.
    pub fn roots(dr: *const Draft, out: *[3]T) []const T {
        out[0] = dr.scores;
        out[1] = dr.cand;
        if (dr.probs) |p| {
            out[2] = p;
            return out[0..3];
        }
        return out[0..2];
    }
};

/// The greedy path through the evaluated scores (`scores` [d][k][k] row-major, `cand` [d][k]): from the anchor's row,
/// each position's first largest score over the candidates, then that candidate's row (the reference's argmax chain).
/// `path` gets the d draft ids; `pick` (when given) each chosen candidate's index.
pub fn pathOf(scores: []const f32, cand: []const u32, d: u32, k: u32, path: []u32, pick: ?[]u32) void {
    var j: usize = 0;
    for (0..d) |t| {
        const row = scores[(t * k + j) * k ..][0..k];
        var best: usize = 0;
        for (row, 0..) |v, i| if (v > row[best]) {
            best = i;
        };
        path[t] = cand[t * k + best];
        if (pick) |p| p[t] = @intCast(best);
        j = best;
    }
}

/// [n, heads * dim] (or [n, heads, dim]) -> [1, heads, n, dim].
fn headsOf(g: *G, x: T, n: u32, heads: u32, dim: u32) !T {
    return g.transposeAxes(try g.reshape(x, &.{ 1, @intCast(n), @intCast(heads), @intCast(dim) }), &.{ 0, 2, 1, 3 });
}

/// kernel_projection's output [n, 2 K groups] as its two halves [n, K, groups] (the reference's [..., 2, K, groups]).
fn kernelHalves(g: *G, c: *const Config, kp: T, n: u32) !struct { first: T, second: T } {
    const ni: c_int = @intCast(n);
    const kk: c_int = @intCast(c.conv_k);
    const gr: c_int = @intCast(c.groups());
    const r = try g.reshape(kp, &.{ ni, 2, kk, gr });
    return .{
        .first = try g.reshape(try g.slice(r, &.{ 0, 0, 0, 0 }, &.{ ni, 1, kk, gr }, &.{ 1, 1, 1, 1 }), &.{ ni, kk, gr }),
        .second = try g.reshape(try g.slice(r, &.{ 0, 1, 0, 0 }, &.{ ni, 2, kk, gr }, &.{ 1, 1, 1, 1 }), &.{ ni, kk, gr }),
    };
}

/// The grouped dynamic causal conv (`_grouped_dynamic_convolve`): x [n, H] bf16, dyn [n, K, groups] bf16, taps[o]
/// [1, groups, group size]. For each offset o: values = x shifted down o rows (zero rows before the block),
/// out = out + tap[o] x values, then out = out + dyn[:, o] x values (the reference's two adds, each rounded to bf16).
fn gconv(g: *G, c: *const Config, x: T, dyn: T, taps: *const [4]T, n: u32) !T {
    const ni: c_int = @intCast(n);
    const gr: c_int = @intCast(c.groups());
    const gs: c_int = @intCast(c.conv_g);
    const blocks = try g.reshape(x, &.{ ni, gr, gs });
    var out: ?T = null;
    for (0..c.conv_k) |o| {
        const oi: c_int = @intCast(o);
        const values = if (o == 0) blocks else if (oi >= ni) try g.zeros(&.{ ni, gr, gs }, .bfloat16) else try g.concat(&.{ try g.zeros(&.{ oi, gr, gs }, .bfloat16), try g.slice(blocks, &.{ 0, 0, 0 }, &.{ ni - oi, gr, gs }, &.{ 1, 1, 1 }) }, 0);
        const tap = try g.mul(taps[o], values);
        const base = if (out) |prev| try g.add(prev, tap) else tap;
        const dy = try g.reshape(try g.slice(dyn, &.{ 0, oi, 0 }, &.{ ni, oi + 1, gr }, &.{ 1, 1, 1 }), &.{ ni, gr, 1 });
        out = try g.add(base, try g.mul(dy, values));
    }
    return g.reshape(out.?, &.{ ni, @intCast(c.hidden) });
}

/// The block's mask over its keys (the newest `ctx` context positions, then the n block rows): row i sees the context
/// row at index j (position p - ctx + j) when ctx + i - j < window, and every block row. Null when every row sees every
/// key (ctx + n <= window).
fn blockMask(g: *G, a: std.mem.Allocator, n: u32, ctx: u32, window: u32) !?T {
    if (ctx + n <= window) return null;
    const keys = ctx + n;
    const buf = try a.alloc(bool, @as(usize, n) * keys);
    defer a.free(buf);
    for (0..n) |i| for (0..keys) |j| {
        buf[i * keys + j] = j >= ctx or ctx + i < window + j;
    };
    return try g.hostArray(std.mem.sliceAsBytes(buf), &.{ @intCast(n), @intCast(keys) }, .bool_);
}

/// The prompt pass's and the verify's captures (graph.zig `forward`'s `capture`): the residual stream at the end of each
/// target layer. `tail` null keeps a reference to the layer's output (a verify chunk, a serial step: lazy, no copy);
/// `tail` = t keeps the call's last min(rows, t) rows as an evaluated contiguous copy (a prompt call: the layer's output
/// is evaluated at its end, and the copy lets its buffer go with the layer).
pub const Captures = struct {
    targets: []const u32,
    tail: ?u32 = null,
    arrays: [max_targets]?T = @splat(null),
    /// The rows each kept array holds (the same for every target of a call).
    rows: u32 = 0,

    pub fn take(self: *Captures, g: *G, layer: u32, x: T, rows: u32) !void {
        const i = std.mem.indexOfScalar(u32, self.targets, layer) orelse return;
        std.debug.assert(self.arrays[i] == null);
        if (self.tail) |t| {
            const keep = @min(rows, t);
            const m = g.mark();
            defer g.resetTo(m);
            const sl = try g.slice(x, &.{ @intCast(rows - keep), 0 }, &.{ @intCast(rows), g.shapeOf(x).dim(1) }, &.{ 1, 1 });
            var out: T = .{};
            try mlx.check(mlx.mlx_contiguous(&out, sl, false, g.s));
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
            self.arrays[i] = out;
            self.rows = keep;
        } else {
            self.arrays[i] = g.keep(x);
            self.rows = rows;
        }
    }

    /// Every target taken (a forward through every layer).
    pub fn complete(self: *const Captures) bool {
        for (self.arrays[0..self.targets.len]) |x| if (x == null) return false;
        return true;
    }

    pub fn slice(self: *const Captures, out: *[max_targets]T) []const T {
        for (self.arrays[0..self.targets.len], out[0..self.targets.len]) |x, *o| o.* = x.?;
        return out[0..self.targets.len];
    }

    pub fn release(self: *Captures) void {
        for (&self.arrays) |*x| if (x.*) |v| {
            _ = mlx.mlx_array_free(v);
            x.* = null;
        };
        self.rows = 0;
    }
};

const testing = std.testing;

const zero_sha = "0000000000000000000000000000000000000000000000000000000000000000";

/// A manifest for a drafter of geometry `c` (zero sha256s; the tensors in role order from offset 0; a 1,000-byte header).
pub fn manifestText(a: std.mem.Allocator, c: *const Config) ![]u8 {
    var tensors: std.ArrayList(u8) = .empty;
    defer tensors.deinit(a);
    var off: u64 = 0;
    for (0..c.layers + 1) |li| {
        const roles: []const RoleName = if (li == 0) &global_roles else &layer_roles;
        for (roles) |r| {
            const sh = shapeOf(c, r[0]);
            var n: u64 = 2;
            for (sh.dims[0..sh.rank]) |d| n *= d;
            const name = if (li == 0) try a.dupe(u8, r[1]) else try std.fmt.allocPrint(a, "layers.{d}.{s}", .{ li - 1, r[1] });
            defer a.free(name);
            const dims = switch (sh.rank) {
                1 => try std.fmt.allocPrint(a, "[{d}]", .{sh.dims[0]}),
                2 => try std.fmt.allocPrint(a, "[{d},{d}]", .{ sh.dims[0], sh.dims[1] }),
                else => try std.fmt.allocPrint(a, "[{d},{d},{d}]", .{ sh.dims[0], sh.dims[1], sh.dims[2] }),
            };
            defer a.free(dims);
            try tensors.print(a, "{s}\"{s}\":{{\"dtype\":\"BF16\",\"shape\":{s},\"offset\":{d},\"length\":{d},\"sha256\":\"{s}\"}}", .{ if (tensors.items.len == 0) "" else ",", name, dims, off, n, zero_sha });
            off += n;
        }
    }
    var targets: std.ArrayList(u8) = .empty;
    defer targets.deinit(a);
    for (c.targetSlice(), 0..) |t, i| try targets.print(a, "{s}{d}", .{ if (i == 0) "" else ",", t });
    var types: std.ArrayList(u8) = .empty;
    defer types.deinit(a);
    for (0..c.layers) |i| try types.print(a, "{s}\"sliding_attention\"", .{if (i == 0) "" else ","});
    return std.fmt.allocPrint(a,
        \\{{"family":"mlx-stream","format":"mlx-stream-dflash2-drafter-v1","data_file":"drafter.safetensors","data_size":{d},"data_sha256":"{s}","header_bytes":1000,"payload_bytes":{d},
        \\"source":{{"repo":"fixture","revision":"0","file":"model.safetensors","size":{d},"sha256":"{s}"}},
        \\"config":{{"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"intermediate_size":{d},"vocab_size":{d},"rms_norm_eps":1e-06,"rope_theta":10000.0,"sliding_window":{d},"layer_types":[{s}],"is_causal":false,"num_target_layers":{d},"block_size":{d},"mask_token_id":{d},"target_layer_ids":[{s}],"conv_kernel_size":{d},"conv_group_size":{d},"selector_rank":{d},"selector_top_k":{d},"final_logit_softcapping":30,"input_embedding_scale":1.0,"output_multiplier":1.0}},
        \\"tensors":{{{s}}}}}
    , .{ off + 1000, zero_sha, off, off + 1000, zero_sha, c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter, c.vocab, c.window, types.items, c.num_target_layers, c.block, c.mask_id, targets.items, c.conv_k, c.conv_g, c.rank, c.top_k, tensors.items });
}

/// Beam 1.0's drafter (the release's config).
pub const beam_drafter: Config = .{ .hidden = 5120, .layers = 5, .heads = 40, .kv_heads = 8, .head_dim = 128, .inter = 17408, .vocab = 151_936, .eps = 1e-6, .rope_theta = 10_000, .window = 4096, .num_target_layers = 52, .block = 16, .mask_id = 151_669, .targets = .{ 4, 16, 28, 40, 46, 0, 0, 0 }, .n_targets = 5, .conv_k = 2, .conv_g = 16, .rank = 256, .top_k = 16, .softcap = 30 };

test "beam drafter: the release's manifest parses to its 81 tensors and 3,854,969,344 payload bytes; the bill's terms" {
    const a = testing.allocator;
    const text = try manifestText(a, &beam_drafter);
    defer a.free(text);
    const trunk: beam.Config = .{};
    var m = try Manifest.parse(a, text, &trunk);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 81), m.entries.len);
    try testing.expectEqual(@as(u64, 3_854_969_344), m.payload_bytes);
    try testing.expectEqualSlices(u32, &.{ 4, 16, 28, 40, 46 }, m.config.targetSlice());
    try testing.expectEqual(@as(?usize, 4), m.config.targetIndex(46));
    try testing.expectEqual(@as(?usize, null), m.config.targetIndex(5));
    try testing.expectEqual(@as(u32, 320), m.config.groups());
    // fc [5120, 25600] first; the codebooks [151936, 256] with no ".weight".
    try testing.expectEqualStrings("fc.weight", m.entries[0].name);
    try testing.expectEqual(@as(u64, 262_144_000), m.entries[0].length);
    try testing.expectEqualStrings("candidate_selector.successor_codebook", m.entries[5].name);
    try testing.expectEqualStrings("layers.4.mlp_conv.kernel_projection.weight", m.entries[80].name);
    try testing.expectEqualSlices(u64, &.{ 1280, 5120 }, m.entries[80].dims[0..2]);
    // The rings: 5 layers x K, V x 2 slots x 4,624 rows x 8 x 128 x 2 B.
    try testing.expectEqual(@as(u64, 4624), kv.ringBase(4096));
    try testing.expectEqual(@as(u64, 5 * 2 * 2 * 4624 * 2048), contextBytes(&m.config, 16_384));
    const res = residentBytes(&m, 16_384);
    try testing.expect(res.bytes >= m.payload_bytes and res.bytes < m.payload_bytes + (2 << 20));
    try testing.expectEqual(@as(u64, 81 + 20), res.arrays);
    std.debug.print("\nBEAM DRAFTER bill: residents {d} B ({d} arrays), context {d} B, round(16 rows, 4095 keys) {d} B, prompt captures {d} B\n", .{ res.bytes, res.arrays, contextBytes(&m.config, 16_384), roundBytes(&m.config, 16, 4095), promptCaptureBytes(&m.config, 16_384, 16_384) });
}

test "beam drafter: a manifest off the trunk or the served form refuses by name" {
    const a = testing.allocator;
    const trunk: beam.Config = .{};
    const Case = struct { c: Config, err: Error };
    var cases: [7]Case = undefined;
    for (&cases) |*cs| cs.* = .{ .c = beam_drafter, .err = error.BeamDrafterGeometry };
    cases[0].c.hidden = 4096;
    cases[1].c.vocab = 151_000;
    cases[2].c.num_target_layers = 51;
    cases[3].c.targets[4] = 52;
    cases[4].c.block = 17;
    cases[4].err = error.BeamDrafterUnsupported;
    cases[5].c.top_k = 64;
    cases[5].err = error.BeamDrafterUnsupported;
    cases[6].c.conv_g = 3;
    cases[6].err = error.BeamDrafterUnsupported;
    for (cases) |cs| {
        const text = try manifestText(a, &cs.c);
        defer a.free(text);
        try testing.expectError(cs.err, Manifest.parse(a, text, &trunk));
    }
    // The served form and the manifest's own fields.
    const base = try manifestText(a, &beam_drafter);
    defer a.free(base);
    const Swap = struct { from: []const u8, to: []const u8, err: Error };
    for ([_]Swap{
        .{ .from = "\"is_causal\":false", .to = "\"is_causal\":true", .err = error.BeamDrafterUnsupported },
        .{ .from = "\"output_multiplier\":1.0", .to = "\"output_multiplier\":2.0", .err = error.BeamDrafterUnsupported },
        .{ .from = "mlx-stream-dflash2-drafter-v1", .to = "mlx-stream-dflash-drafter-v0", .err = error.BeamDrafterManifestFormat },
        .{ .from = "\"layers.2.self_attn.q_proj.weight\"", .to = "\"layers.2.self_attn.qproj.weight\"", .err = error.BeamDrafterTensors },
        .{ .from = "\"shape\":[5120,25600]", .to = "\"shape\":[5120,20480]", .err = error.BeamDrafterTensors },
        .{ .from = "\"header_bytes\":1000", .to = "\"header_bytes\":999", .err = error.BeamDrafterManifestField },
    }) |sw| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, base, sw.from));
        const text = try std.mem.replaceOwned(u8, a, base, sw.from, sw.to);
        defer a.free(text);
        try testing.expectError(sw.err, Manifest.parse(a, text, &trunk));
    }
    // A sliding layer list with a full layer.
    const one = try std.mem.replaceOwned(u8, a, base, "[\"sliding_attention\",", "[\"full_attention\",");
    defer a.free(one);
    try testing.expectError(error.BeamDrafterUnsupported, Manifest.parse(a, one, &trunk));
}

test "beam drafter: the greedy path walks each position's first largest score from the previous choice's row; the mask cuts only the oldest context keys" {
    // d = 3, k = 2: scores[t][j][i]; t = 0 reads row 0 (the anchor's), then the chosen candidate's row.
    const scores = [_]f32{
        1, 2, 1, 2, // t0: row 0 -> candidate 1
        5, 4, 0, 3, // t1: row 1 (cand 1 of t0) -> [0, 3] -> candidate 1
        7, 7, 2, 2, // t2: row 1 -> [2, 2] -> the first, candidate 0
    };
    const cand = [_]u32{ 10, 11, 20, 21, 30, 31 };
    var path: [3]u32 = undefined;
    var pick: [3]u32 = undefined;
    pathOf(&scores, &cand, 3, 2, &path, &pick);
    try testing.expectEqualSlices(u32, &.{ 11, 21, 30 }, &path);
    try testing.expectEqualSlices(u32, &.{ 1, 1, 0 }, &pick);
}
